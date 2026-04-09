//! Block DSL → Tile SSA translation pass.
//!
//! Translates a `BlockKernel` (BNode DAG) into a `TileFunc` (SSA IR),
//! providing an alternative compilation path via `tile_ssa_lower`.
//!
//! ## Supported BNode subset (Phase 4a + Sub-phase 1-3)
//!
//! - ArgPtr / ArgU32 / ArgF32
//! - ProgramId / Arange / ThreadId
//! - ConstU32 / ConstF32
//! - All F32 arithmetic (Add/Sub/Mul/Neg/Exp/Log/Sqrt/Rcp/Rsqrt/Abs/Max/Min/Fma)
//! - All U32 arithmetic (Add/Mul/Shr/And)
//! - Comparisons (Lt/Ge → bool vector)
//! - Masked Load/Store (f32 + u32), AtomicAddF32
//! - Type conversions (CvtF32U32 / CvtU32F32)
//! - Wave reductions (WaveReduceAdd / WaveReduceMax)
//! - LDS (LdsAlloc / LdsLoad / LdsStore / Barrier)
//! - For loops (ForBegin / ForEnd with runtime bounds)
//! - WMMA (ZeroAcc / CvtPkBf16F32 / Wmma / ExtractF32 / SplatFragment)
//! - WG-level reductions (WgReduceAdd / WgReduceMax)
//!
//! ## Not yet supported
//!
//! (All Phase 4 BNode types are now supported)

use super::block_dsl::{BlockKernel, BNode, BVal, BType};
use super::tile_ssa::{TileFunc, Value, ScalarDType, BinOpKind, UnaryOpKind, CmpOpKind, ForLoop};
use super::tile_ssa_lower;
use super::dsl::{CompiledKernel, KernArgMeta, KernArgType};
use super::ir::{Target, ArgKind};

use std::collections::HashMap;

/// Translate a BlockKernel into TileFunc SSA IR.
///
/// Each BVal in the block_dsl maps to a Value in the TileFunc.
/// The translation preserves the DAG structure as straight-line SSA code.
pub fn block_to_ssa(kb: &BlockKernel) -> Result<TileFunc, String> {
    let _block_size = kb.get_block_size();
    let mut f = TileFunc::new(kb.kernel_name());
    let nodes = kb.get_nodes();

    // BVal → Value mapping
    let mut val_map: HashMap<usize, Value> = HashMap::new();
    // Active ForLoop handles keyed by ForBegin node index
    let mut loop_map: HashMap<usize, ForLoop> = HashMap::new();
    // Active ForLoopAcc handles keyed by ForAccBegin node index
    let mut acc_loop_map: HashMap<usize, super::tile_ssa::ForLoopAcc> = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
        // ForBegin/ForEnd/ForAccBegin/ForAccEnd/ForAccResult are handled inline
        match node {
            BNode::ForBegin { start, end, step } => {
                let start_v = val_map.get(&start.0).copied()
                    .ok_or_else(|| format!("ForBegin: start BVal {} not mapped", start.0))?;
                let end_v = val_map.get(&end.0).copied()
                    .ok_or_else(|| format!("ForBegin: end BVal {} not mapped", end.0))?;
                let lp = f.for_range_runtime(start_v, end_v, *step);
                // Map ForBegin's BVal to the loop induction variable
                val_map.insert(i, lp.iv);
                loop_map.insert(i, lp);
                continue;
            }
            BNode::ForEnd { begin_node } => {
                let lp = loop_map.remove(begin_node)
                    .ok_or_else(|| format!("ForEnd: no matching ForBegin for node {}", begin_node))?;
                f.end_for(&lp);
                continue;
            }
            BNode::ForAccBegin { start, end, step, init_acc } => {
                let start_v = val_map.get(&start.0).copied()
                    .ok_or_else(|| format!("ForAccBegin: start BVal {} not mapped", start.0))?;
                let end_v = val_map.get(&end.0).copied()
                    .ok_or_else(|| format!("ForAccBegin: end BVal {} not mapped", end.0))?;
                let init_v = val_map.get(&init_acc.0).copied()
                    .ok_or_else(|| format!("ForAccBegin: init_acc BVal {} not mapped", init_acc.0))?;
                let acc_ty = f.value_type(init_v).clone();
                let lp = f.for_range_with_acc_runtime(start_v, end_v, *step, init_v, acc_ty);
                // Map ForAccBegin's BVal to the loop induction variable
                val_map.insert(i, lp.iv);
                acc_loop_map.insert(i, lp);
                continue;
            }
            BNode::ForAccPhi { begin_node } => {
                // Map ForAccPhi to the accumulator block param from the matching loop
                let lp = acc_loop_map.get(begin_node)
                    .ok_or_else(|| format!("ForAccPhi: no matching ForAccBegin for node {}", begin_node))?;
                val_map.insert(i, lp.acc);
                continue;
            }
            BNode::ForAccEnd { begin_node, new_acc } => {
                let new_acc_v = val_map.get(&new_acc.0).copied()
                    .ok_or_else(|| format!("ForAccEnd: new_acc BVal {} not mapped", new_acc.0))?;
                let lp = acc_loop_map.get(begin_node)
                    .ok_or_else(|| format!("ForAccEnd: no matching ForAccBegin for node {}", begin_node))?
                    .clone();
                f.end_for_acc(&lp, new_acc_v);
                continue;
            }
            BNode::ForAccResult { begin_node } => {
                let lp = acc_loop_map.remove(begin_node)
                    .ok_or_else(|| format!("ForAccResult: no matching ForAccBegin for node {}", begin_node))?;
                val_map.insert(i, lp.result);
                continue;
            }
            // IfMask/ElseMask/EndIf → ExecMaskPush/Flip/Pop TileOps
            BNode::IfMask(mask) => {
                let mask_v = val_map.get(&mask.0).copied()
                    .ok_or_else(|| format!("IfMask: mask BVal {} not mapped", mask.0))?;
                f.push_op(super::tile_ssa::TileOp::ExecMaskPush { mask: mask_v });
                continue;
            }
            BNode::ElseMask => {
                f.push_op(super::tile_ssa::TileOp::ExecMaskFlip);
                continue;
            }
            BNode::EndIf => {
                f.push_op(super::tile_ssa::TileOp::ExecMaskPop);
                continue;
            }
            _ => {}
        }

        let ssa_val = translate_node(&mut f, kb, node, i, &val_map)?;
        if let Some(v) = ssa_val {
            val_map.insert(i, v);
        }
    }

    // Emit return (s_endpgm)
    f.return_();

    Ok(f)
}

/// Translate a single BNode into TileFunc operations.
/// Returns Some(Value) for nodes that produce a result, None for void ops.
fn translate_node(
    f: &mut TileFunc,
    kb: &BlockKernel,
    node: &BNode,
    _idx: usize,
    val_map: &HashMap<usize, Value>,
) -> Result<Option<Value>, String> {
    let get = |bval: &BVal| -> Result<Value, String> {
        val_map.get(&bval.0)
            .copied()
            .ok_or_else(|| format!("BVal {} not found in val_map", bval.0))
    };

    match node {
        // ── Arguments ──
        BNode::ArgPtr(name) => Ok(Some(f.arg_ptr(name))),
        BNode::ArgU32(name) => Ok(Some(f.arg_u32(name))),
        BNode::ArgF32(name) => {
            Ok(Some(f.arg_f32(name)))
        }

        // ── Indexing ──
        BNode::ProgramId(axis) => Ok(Some(f.program_id(*axis))),
        BNode::ThreadId => {
            let [bx, by] = kb.get_block_size_2d();
            if by > 1 {
                // 2D mode: thread_id_x = flat_tid & (block_size_x - 1)
                let total = bx * by;
                let v = f.alloc_value(
                    super::tile_ssa::TileType::vector(total, super::tile_ssa::ScalarDType::U32),
                    Some("tid_x"),
                );
                f.push_op(super::tile_ssa::TileOp::ThreadIdX2D { result: v, block_x: bx });
                Ok(Some(v))
            } else {
                Ok(Some(f.arange(0, kb.get_block_size())))
            }
        }
        BNode::ThreadIdY { block_x } => {
            // thread_id_y = flat_tid >> log2(block_size_x)
            let total = kb.get_block_size();
            let v = f.alloc_value(
                super::tile_ssa::TileType::vector(total, super::tile_ssa::ScalarDType::U32),
                Some("tid_y"),
            );
            f.push_op(super::tile_ssa::TileOp::ThreadIdY2D { result: v, block_x: *block_x });
            Ok(Some(v))
        }
        BNode::Arange { start, end } => {
            let len = end - start;
            if *start == 0 {
                Ok(Some(f.arange(0, len)))
            } else {
                let base = f.arange(0, len);
                let offset = f.const_u32(*start);
                Ok(Some(f.add(base, offset)))
            }
        }

        // ── Constants ──
        BNode::ConstU32(v) => Ok(Some(f.const_u32(*v))),
        BNode::ConstF32(v) => Ok(Some(f.const_f32(*v))),

        // ── F32 arithmetic ──
        BNode::AddF32(a, b) => Ok(Some(f.add(get(a)?, get(b)?))),
        BNode::MulF32(a, b) => Ok(Some(f.mul(get(a)?, get(b)?))),
        BNode::SubF32(a, b) => Ok(Some(f.sub(get(a)?, get(b)?))),
        BNode::MaxF32(a, b) => Ok(Some(f.max(get(a)?, get(b)?))),
        BNode::MinF32(a, b) => Ok(Some(f.min(get(a)?, get(b)?))),
        BNode::FmaF32(a, b, c) => Ok(Some(f.fma(get(a)?, get(b)?, get(c)?))),

        // BNode::ExpF32 = v_exp_f32 = 2^x (NOT e^x)
        // The BVal DSL's exp() method pre-scales by log2(e) before creating ExpF32.
        // So we must use exp2() here (raw hardware) to avoid double scaling.
        BNode::ExpF32(a)  => Ok(Some(f.exp2(get(a)?))),
        // BNode::Log2F32 = v_log_f32 = log₂(x) (NOT ln(x))
        BNode::Log2F32(a) => Ok(Some(f.log2(get(a)?))),
        BNode::SqrtF32(a) => Ok(Some(f.sqrt(get(a)?))),
        BNode::RcpF32(a)  => Ok(Some(f.rcp(get(a)?))),
        BNode::RsqrtF32(a) => Ok(Some(f.rsqrt(get(a)?))),
        BNode::AbsF32(a)  => Ok(Some(f.abs(get(a)?))),
        BNode::NegF32(a)  => Ok(Some(f.neg(get(a)?))),
        BNode::SinF32(a)  => Ok(Some(f.sin(get(a)?))),
        BNode::CosF32(a)  => Ok(Some(f.cos(get(a)?))),
        BNode::DivF32(a, b) => Ok(Some(f.div(get(a)?, get(b)?))),

        // ── U32 arithmetic ──
        BNode::AddU32(a, b) => Ok(Some(f.add(get(a)?, get(b)?))),
        BNode::SubU32(a, b) => Ok(Some(f.sub(get(a)?, get(b)?))),
        BNode::MulU32(a, b) => Ok(Some(f.mul(get(a)?, get(b)?))),
        BNode::ShrConstU32(a, shift) => {
            let src = get(a)?;
            let shift_val = f.const_u32(*shift as u32);
            Ok(Some(f.binop_raw(BinOpKind::Shr, src, shift_val)))
        }
        BNode::AndConstU32(a, mask) => {
            let src = get(a)?;
            let mask_val = f.const_u32(*mask);
            Ok(Some(f.binop_raw(BinOpKind::And, src, mask_val)))
        }
        BNode::ShlConstU32(a, shift) => {
            let src = get(a)?;
            let shift_val = f.const_u32(*shift as u32);
            Ok(Some(f.binop_raw(BinOpKind::Shl, src, shift_val)))
        }
        BNode::OrConstU32(a, val) => {
            let src = get(a)?;
            let or_val = f.const_u32(*val);
            Ok(Some(f.binop_raw(BinOpKind::Or, src, or_val)))
        }
        BNode::XorConstU32(a, val) => {
            let src = get(a)?;
            let xor_val = f.const_u32(*val);
            Ok(Some(f.binop_raw(BinOpKind::Xor, src, xor_val)))
        }

        // ── Comparisons ──
        BNode::LtU32(a, b) => Ok(Some(f.cmp_lt(get(a)?, get(b)?))),
        BNode::GeU32(a, b) => Ok(Some(f.cmp_ge(get(a)?, get(b)?))),
        BNode::CmpLtF32(a, b) => Ok(Some(f.cmp_lt(get(a)?, get(b)?))),
        BNode::CmpGtF32(a, b) => {
            // a > b  is equivalent to  b < a
            Ok(Some(f.cmp_lt(get(b)?, get(a)?)))
        }
        BNode::AndBool(a, b) => {
            Ok(Some(f.binop_raw(super::tile_ssa::BinOpKind::And, get(a)?, get(b)?)))
        }
        BNode::Select { mask, true_val, false_val } => {
            let m = get(mask)?;
            let t = get(true_val)?;
            let fv = get(false_val)?;
            Ok(Some(f.select(m, t, fv)))
        }

        // ── Memory ──
        BNode::Load { ptr, offsets, mask } => {
            let p = get(ptr)?;
            let idx = get(offsets)?;
            let m = get(mask)?;
            let zero = f.const_f32(0.0);
            Ok(Some(f.load_masked(p, idx, m, zero, ScalarDType::F32)))
        }
        BNode::LoadU32 { ptr, offsets, mask } => {
            let p = get(ptr)?;
            let idx = get(offsets)?;
            let m = get(mask)?;
            let zero = f.const_u32(0);
            Ok(Some(f.load_masked(p, idx, m, zero, ScalarDType::U32)))
        }
        BNode::Store { ptr, offsets, val, mask } => {
            let p = get(ptr)?;
            let idx = get(offsets)?;
            let v = get(val)?;
            let m = get(mask)?;
            f.store_masked(p, idx, v, m);
            Ok(None)
        }
        BNode::AtomicAddF32 { ptr, offsets, val, mask } => {
            let p = get(ptr)?;
            let idx = get(offsets)?;
            let v = get(val)?;
            let m = get(mask)?;
            f.atomic_add_f32(p, idx, v, Some(m));
            Ok(None)
        }

        BNode::AtomicAddU32Rtn { ptr: _ptr, val: _val } => {
            // TODO: Needs TileOp::AtomicAddU32Rtn in tile_ssa.rs
            // For now, emit_printf() uses T0Kernel directly (bypasses block_dsl).
            return Err("AtomicAddU32Rtn not yet supported in block_dsl lowering (use T0Kernel directly)".into());
        }

        // ── BF16 memory ops ──
        BNode::LoadBf16 { ptr, offsets, mask } => {
            let p = get(ptr)?;
            let idx = get(offsets)?;
            let m = get(mask)?;
            let zero = f.const_u32(0);
            // Load as BF16 (16-bit), result is BF16-typed vector
            let bf16_val = f.load_masked(p, idx, m, zero, ScalarDType::BF16);
            // Convert BF16→F32: shift left 16 bits
            let sixteen = f.const_u32(16);
            Ok(Some(f.binop_raw(BinOpKind::Shl, bf16_val, sixteen)))
        }
        BNode::StoreBf16 { ptr, offsets, val, mask } => {
            let p = get(ptr)?;
            let idx = get(offsets)?;
            let v = get(val)?;
            let m = get(mask)?;
            // Convert F32→BF16 via SSA cast (truncation: upper 16 bits of f32)
            let bf16_val = f.cast(v, ScalarDType::BF16);
            // Store with BF16 dtype — tile_ssa_lower selects Width::B16 automatically
            f.store_masked(p, idx, bf16_val, m);
            Ok(None)
        }

        // ── Type conversions ──
        BNode::CvtF32U32(a) => Ok(Some(f.cast(get(a)?, ScalarDType::F32))),
        BNode::CvtU32F32(a) => Ok(Some(f.cast(get(a)?, ScalarDType::U32))),
        BNode::CvtF32BF16(a) => {
            // bf16 → f32: bf16 value stored in lower 16 bits of u32
            // Simply shift left 16 bits to get the f32 bit pattern
            let src = get(a)?;
            let sixteen = f.const_u32(16);
            Ok(Some(f.binop_raw(BinOpKind::Shl, src, sixteen)))
        }

        // ── Wave reductions ──
        BNode::WaveReduceAddF32(a) => Ok(Some(f.reduce_add(get(a)?))),
        BNode::WaveReduceMaxF32(a) => Ok(Some(f.reduce_max(get(a)?, 0))),

        // ── For loops are handled in block_to_ssa() main loop ──
        BNode::ForBegin { .. } | BNode::ForEnd { .. }
        | BNode::ForAccBegin { .. } | BNode::ForAccPhi { .. }
        | BNode::ForAccEnd { .. } | BNode::ForAccResult { .. } => {
            unreachable!("For*/ForAcc* handled inline in block_to_ssa()")
        }

        // ── Conditional branching (handled inline in block_to_ssa) ──
        BNode::IfMask(_) | BNode::ElseMask | BNode::EndIf => {
            unreachable!("IfMask/ElseMask/EndIf handled inline in block_to_ssa()")
        }

        // ── LDS ──
        BNode::LdsAlloc { size_bytes } => {
            Ok(Some(f.lds_alloc(*size_bytes)))
        }
        BNode::LdsLoad { base, offset } => {
            Ok(Some(f.lds_load(get(base)?, get(offset)?)))
        }
        BNode::LdsStore { base, offset, val } => {
            f.lds_store(get(base)?, get(offset)?, get(val)?);
            Ok(None)
        }
        BNode::Barrier => {
            f.barrier();
            Ok(None)
        }
        // ── WG-level reductions ──
        BNode::WgReduceAddF32(a) => {
            let block_size = kb.get_block_size();
            Ok(Some(f.wg_reduce_add(get(a)?, block_size)))
        }
        BNode::WgReduceMaxF32(a) => {
            let block_size = kb.get_block_size();
            Ok(Some(f.wg_reduce_max(get(a)?, block_size)))
        }

        // ── WMMA / BF16 ──
        BNode::ZeroAcc => Ok(Some(f.zero_acc())),
        BNode::CvtPkBf16F32 { lo, hi } => {
            Ok(Some(f.cvt_pk_bf16_f32(get(lo)?, get(hi)?)))
        }
        BNode::Wmma { a, b, c } => {
            Ok(Some(f.wmma_f32(get(a)?, get(b)?, get(c)?)))
        }
        BNode::ExtractF32 { src, idx } => {
            Ok(Some(f.extract_f32(get(src)?, *idx)))
        }
        BNode::SplatFragment(val) => {
            Ok(Some(f.splat_fragment(get(val)?)))
        }

        // ── TileGemm → Tile SSA expansion ──
        // Emit 2D tile ops so analyze_tiled_gemm() can extract M/N/K.
        // Actual ISA generation is handled by lower_tiled_gemm() → tile_ir::lower_gemm().
        BNode::TileGemm { a_ptr, b_ptr, y_ptr, k_dim, n_dim, config } => {
            let a = get(a_ptr)?;
            let b = get(b_ptr)?;
            let y = get(y_ptr)?;
            let k = get(k_dim)?;
            let n = get(n_dim)?;

            // Emit Triton-style tile program (for SSA analysis only)
            let pid_m = f.program_id(0);
            let pid_n = f.program_id(1);
            let cm = f.const_u32(config.tile_m);
            let cn = f.const_u32(config.tile_n);
            let row_off = f.mul(pid_m, cm);
            let col_off = f.mul(pid_n, cn);

            let acc = f.tile_zeros(config.tile_m, config.tile_n);
            let k0 = f.const_u32(0);
            // A[M, K] load
            let tile_a = f.tile_load_2d(a, row_off, k0, k,
                config.tile_m, config.tile_k, ScalarDType::BF16);
            // B[K, N] load (SSA semantics: b[K,N]; NT transpose handled at lowering)
            let tile_b = f.tile_load_2d(b, col_off, k0, k,
                config.tile_k, config.tile_n, ScalarDType::BF16);
            let acc2 = f.tile_dot(tile_a, tile_b, acc);
            f.tile_store_2d(y, row_off, col_off, n, acc2);
            Ok(None) // void
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// compile_via_ssa — alternative compilation path for BlockKernel
// ════════════════════════════════════════════════════════════════════

impl BlockKernel {
    /// Compile this BlockKernel via the Tile SSA IR path.
    ///
    /// Translation: BlockKernel → TileFunc SSA → tile_ssa_lower → T0Kernel → ELF
    ///
    /// This is an alternative to `compile()` which goes BNode → T0Kernel directly.
    /// The SSA path enables future optimizations (DCE, scheduling, etc.) at the IR level.
    ///
    /// For TileGemm mega-ops, routes through `lower_tiled_gemm()` → `tile_ir::lower_gemm()`.
    pub fn compile_via_ssa(&self, target: Target) -> Result<CompiledKernel, String> {
        // Step 1: Translate to TileFunc SSA
        let func = block_to_ssa(self)?;

        // Step 2: Check if this is a GEMM kernel (contains TileLoad2D/TileDot)
        if tile_ssa_lower::analyze_tiled_gemm(&func).is_ok() {
            // GEMM path: lower via tile_ir
            let lowered = tile_ssa_lower::lower_tiled_gemm(&func)?;
            // CRITICAL: use compile_with_info to get final LDS size including
            // SSA regalloc spill regions. Without this, KFD under-allocates LDS → GPU hang.
            let (elf, final_lds) = lowered.kernel.compile_with_info(target)?;

            let args: Vec<KernArgMeta> = lowered.kernel.args().iter().map(|a| {
                KernArgMeta {
                    name: a.name.clone(),
                    kind: match a.kind {
                        ArgKind::Ptr => KernArgType::Ptr,
                        ArgKind::U32 => KernArgType::U32,
                        ArgKind::F32 => KernArgType::F32,
                    },
                    offset: a.offset as usize,
                }
            }).collect();

            Ok(CompiledKernel {
                elf,
                kernarg_size: lowered.kernel.kernarg_size() as usize,
                workgroup_size: [lowered.wg_size(), 1, 1],
                lds_size: final_lds,
                name: func.name.clone(),
                args,
            })
        } else {
            // Elementwise path
            // SSA regalloc is enabled globally (compile.rs default),
            // validated for all kernel types including wg_reduce and GEMM.
            let wg_size = self.get_block_size();
            let epl = 1u32;
            let lowered = tile_ssa_lower::lower_elementwise_1d(&func, wg_size, epl)?;
            let elf = lowered.kernel.compile(target)?;

            let args: Vec<KernArgMeta> = lowered.kernel.args().iter().map(|a| {
                KernArgMeta {
                    name: a.name.clone(),
                    kind: match a.kind {
                        ArgKind::Ptr => KernArgType::Ptr,
                        ArgKind::U32 => KernArgType::U32,
                        ArgKind::F32 => KernArgType::F32,
                    },
                    offset: a.offset as usize,
                }
            }).collect();

            let [bsx, bsy] = self.get_block_size_2d();
            Ok(CompiledKernel {
                elf,
                kernarg_size: lowered.kernel.kernarg_size() as usize,
                workgroup_size: [bsx, bsy, 1],
                lds_size: lowered.kernel.lds_size(),
                name: func.name.clone(),
                args,
            })
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: BlockKernel vector_add → TileFunc SSA translation (no GPU).
    #[test]
    fn test_block_to_ssa_basic() {
        let mut kb = BlockKernel::new("vadd_test", 64);
        let x_ptr = kb.arg_ptr("x");
        let y_ptr = kb.arg_ptr("y");
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let pid = kb.program_id(0);
        let c64 = kb.const_u32(64);
        let base = pid.mul(&mut kb, c64);
        let offsets = kb.arange(0, 64).add(&mut kb, base);
        let mask = offsets.lt(&mut kb, n);
        let a = kb.load(x_ptr, offsets, mask);
        let b = kb.load(y_ptr, offsets, mask);
        let c = a.add(&mut kb, b);
        kb.store(out_ptr, offsets, c, mask);

        // Translate to SSA
        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== block_to_ssa IR ===\n{}", ir_text);

        // Verify structure (match IR dump format)
        assert!(ir_text.contains("ptr /*x*/"));
        assert!(ir_text.contains("ptr /*y*/"));
        assert!(ir_text.contains("ptr /*out*/"));
        assert!(ir_text.contains("Add"));
        assert!(ir_text.contains("load"));
        assert!(ir_text.contains("store"));
    }

    /// Test: BlockKernel SiLU → TileFunc SSA translation (no GPU).
    #[test]
    fn test_block_to_ssa_silu() {
        let mut kb = BlockKernel::new("silu_test", 64);
        let x_ptr = kb.arg_ptr("x");
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n);
        let x = kb.load(x_ptr, offsets, mask);
        let y = x.silu(&mut kb); // silu = x * sigmoid(x)
        kb.store(out_ptr, offsets, y, mask);

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== SiLU SSA IR ===\n{}", ir_text);
        assert!(ir_text.contains("Neg"));
        assert!(ir_text.contains("Exp"));
        assert!(ir_text.contains("Rcp"));
    }

    /// Test: compile_via_ssa produces valid ELF binary (no GPU dispatch).
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_vadd() {
        let mut kb = BlockKernel::new("vadd_ssa", 64);
        let x_ptr = kb.arg_ptr("x");
        let y_ptr = kb.arg_ptr("y");
        let out_ptr = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n_arg);
        let a = kb.load(x_ptr, offsets, mask);
        let b = kb.load(y_ptr, offsets, mask);
        let c = a.add(&mut kb, b);
        kb.store(out_ptr, offsets, c, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa failed");
        eprintln!("✓ vadd compile_via_ssa: {} bytes ELF, kernarg_size={}, wg={}",
            compiled.elf.len(), compiled.kernarg_size, compiled.workgroup_size[0]);
        assert!(compiled.elf.len() > 100, "ELF too small");
        assert_eq!(compiled.workgroup_size[0], 64);
        assert_eq!(compiled.args.len(), 4); // x, y, out, n
    }

    /// Test: compile_via_ssa for SiLU produces valid ELF.
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_silu() {
        let mut kb = BlockKernel::new("silu_ssa", 64);
        let x_ptr = kb.arg_ptr("x");
        let out_ptr = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n_arg);
        let x = kb.load(x_ptr, offsets, mask);
        let y = x.silu(&mut kb);
        kb.store(out_ptr, offsets, y, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa failed for SiLU");
        eprintln!("✓ silu compile_via_ssa: {} bytes ELF, kernarg_size={}, wg={}",
            compiled.elf.len(), compiled.kernarg_size, compiled.workgroup_size[0]);
        assert!(compiled.elf.len() > 100, "ELF too small");
        assert_eq!(compiled.args.len(), 3); // x, out, n
    }

    /// GPU E2E: vector_add via compile_via_ssa() + KFD dispatch
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_vadd_via_ssa_gpu_e2e() {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let n: usize = 64;
        let mut kb = BlockKernel::new("vadd_ssa_e2e", 64);
        let x_ptr = kb.arg_ptr("x");
        let y_ptr = kb.arg_ptr("y");
        let out_ptr = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n_arg);
        let a = kb.load(x_ptr, offsets, mask);
        let b = kb.load(y_ptr, offsets, mask);
        let c = a.add(&mut kb, b);
        kb.store(out_ptr, offsets, c, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa failed");
        eprintln!("✓ compile_via_ssa: {} bytes ELF (ka_size={})", compiled.elf.len(), compiled.kernarg_size);

        // GPU dispatch using crate::kfd API
        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let x_buf = device.alloc_vram(n * 4).unwrap();
        let y_buf = device.alloc_vram(n * 4).unwrap();
        let out_buf = device.alloc_vram(n * 4).unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let y_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        x_buf.write(unsafe { std::slice::from_raw_parts(x_data.as_ptr() as *const u8, n * 4) });
        y_buf.write(unsafe { std::slice::from_raw_parts(y_data.as_ptr() as *const u8, n * 4) });
        out_buf.write(&vec![0u8; n * 4]);

        let gk = GpuKernel::load(&device, &compiled.elf, &KernelLoadConfig {
            workgroup_size: [compiled.workgroup_size[0], 1, 1],
            lds_size: compiled.lds_size,
        }).unwrap();

        // Build kernargs manually (4 args: x_ptr=8, y_ptr=8, out_ptr=8, n=4 = 28 bytes)
        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        ka[24..28].copy_from_slice(&(n as u32).to_le_bytes());

        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [compiled.workgroup_size[0], 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        // Read back and verify
        let mut result = vec![0f32; n];
        unsafe {
            out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n * 4));
        }

        for i in 0..n {
            let expected = x_data[i] + y_data[i];
            let err = (result[i] - expected).abs();
            assert!(err < 1e-6, "vadd[{}]: expected {}, got {}, err={}", i, expected, result[i], err);
        }
        eprintln!("✓ vadd_via_ssa GPU E2E PASSED ({} elements)", n);
    }

    /// Test: LDS alloc + store + barrier + load → SSA translation (unit test, no GPU)
    #[test]
    fn test_block_to_ssa_lds() {
        let mut kb = BlockKernel::new("lds_test", 64);
        let x_ptr = kb.arg_ptr("x");
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n);
        let x = kb.load(x_ptr, offsets, mask);

        // Store to LDS, barrier, load back
        let lds = kb.lds_alloc(64 * 4);
        let tid = kb.thread_id();
        kb.lds_store(lds, tid, x);
        kb.barrier();
        let y = kb.lds_load(lds, tid);

        kb.store(out_ptr, offsets, y, mask);

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== LDS SSA IR ===\n{}", ir_text);
        assert!(ir_text.contains("lds_alloc"));
        assert!(ir_text.contains("lds_store"));
        assert!(ir_text.contains("lds_load"));
        assert!(ir_text.contains("barrier"));
    }

    /// Test: LDS path via compile_via_ssa (compilation only)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_lds() {
        let mut kb = BlockKernel::new("lds_ssa", 64);
        let x_ptr = kb.arg_ptr("x");
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n);
        let x = kb.load(x_ptr, offsets, mask);

        let lds = kb.lds_alloc(64 * 4);
        let tid = kb.thread_id();
        kb.lds_store(lds, tid, x);
        kb.barrier();
        let y = kb.lds_load(lds, tid);

        kb.store(out_ptr, offsets, y, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with LDS failed");
        eprintln!("✓ LDS compile_via_ssa: {} bytes ELF, lds_size={}",
            compiled.elf.len(), compiled.lds_size);
        assert!(compiled.elf.len() > 100);
        assert!(compiled.lds_size > 0, "LDS size should be > 0");
    }

    /// Test: AtomicAddF32 compile_via_ssa (compilation only)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_atomic() {
        let mut kb = BlockKernel::new("atomic_test", 64);
        let x_ptr = kb.arg_ptr("x");
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n);
        let x = kb.load(x_ptr, offsets, mask);

        kb.atomic_add_f32(out_ptr, offsets, x, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with atomic failed");
        eprintln!("✓ atomic compile_via_ssa: {} bytes ELF", compiled.elf.len());
        assert!(compiled.elf.len() > 100);
    }

    /// GPU E2E: LDS round-trip via compile_via_ssa()
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_lds_via_ssa_gpu_e2e() {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let n: usize = 64;
        let mut kb = BlockKernel::new("lds_e2e", 64);
        let x_ptr = kb.arg_ptr("x");
        let out_ptr = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");

        let offsets = kb.arange(0, 64);
        let mask = offsets.lt(&mut kb, n_arg);
        let x = kb.load(x_ptr, offsets, mask);

        // LDS round-trip: store to LDS, barrier, load back
        let lds = kb.lds_alloc(64 * 4);
        let tid = kb.thread_id();
        kb.lds_store(lds, tid, x);
        kb.barrier();
        let y = kb.lds_load(lds, tid);

        kb.store(out_ptr, offsets, y, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with LDS failed");

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let x_buf = device.alloc_vram(n * 4).unwrap();
        let out_buf = device.alloc_vram(n * 4).unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 3.14).collect();
        x_buf.write(unsafe { std::slice::from_raw_parts(x_data.as_ptr() as *const u8, n * 4) });
        out_buf.write(&vec![0u8; n * 4]);

        let gk = GpuKernel::load(&device, &compiled.elf, &KernelLoadConfig {
            workgroup_size: [compiled.workgroup_size[0], 1, 1],
            lds_size: compiled.lds_size,
        }).unwrap();

        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        ka[16..20].copy_from_slice(&(n as u32).to_le_bytes());

        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [compiled.workgroup_size[0], 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n];
        unsafe {
            out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n * 4));
        }

        for i in 0..n {
            let err = (result[i] - x_data[i]).abs();
            assert!(err < 1e-6, "lds[{}]: expected {}, got {}, err={}", i, x_data[i], result[i], err);
        }
        eprintln!("✓ LDS round-trip via SSA GPU E2E PASSED ({} elements)", n);
    }

    /// Test: ForBegin/ForEnd → SSA translation (unit test, no GPU)
    #[test]
    fn test_block_to_ssa_for_loop() {
        // Simple loop: for i in 0..4: out[tid] += 1.0
        let mut kb = BlockKernel::new("loop_test", 32);
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let start = kb.const_u32(0);
        let end = kb.const_u32(4);
        let iter_var = kb.for_range(start, end, 1);

        // Body: just use iter_var to prove the loop variable works
        let _iter_f32 = iter_var.to_f32(&mut kb);

        kb.end_for(iter_var);

        // After loop: store something
        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let one = kb.const_f32(1.0);
        kb.store(out_ptr, offsets, one, mask);

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== ForLoop SSA IR ===\n{}", ir_text);
        // Should have multiple basic blocks (entry, header, body, exit)
        assert!(ir_text.contains("bb1"), "Should have bb1 (header)");
        assert!(ir_text.contains("bb2"), "Should have bb2 (body)");
        assert!(ir_text.contains("bb3"), "Should have bb3 (exit)");
    }

    /// Test: ForLoop compile_via_ssa (compilation only)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_for_loop() {
        let mut kb = BlockKernel::new("loop_compile", 32);
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let start = kb.const_u32(0);
        let end = kb.const_u32(4);
        let _iter = kb.for_range(start, end, 1);
        kb.end_for(_iter);

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let one = kb.const_f32(42.0);
        kb.store(out_ptr, offsets, one, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with for loop failed");
        eprintln!("✓ ForLoop compile_via_ssa: {} bytes ELF", compiled.elf.len());
        assert!(compiled.elf.len() > 100);
    }

    /// GPU E2E: ForLoop accumulation via SSA path
    /// Kernel: out[tid] = iter_count (loop runs N times, stores final iter)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_for_loop_via_ssa_gpu_e2e() {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let n: usize = 32;
        let loop_iters: u32 = 5;

        let mut kb = BlockKernel::new("forloop_e2e", 32);
        let out_ptr = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");
        let iters_arg = kb.arg_u32("iters");

        // for i in 0..iters: (nop body)
        let start = kb.const_u32(0);
        let _iter = kb.for_range(start, iters_arg, 1);
        kb.end_for(_iter);

        // After loop: store a constant to prove loop didn't crash
        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n_arg);
        let result_val = kb.const_f32(99.5);
        kb.store(out_ptr, offsets, result_val, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with for loop failed");

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let out_buf = device.alloc_vram(n * 4).unwrap();
        out_buf.write(&vec![0u8; n * 4]);

        let gk = GpuKernel::load(&device, &compiled.elf, &KernelLoadConfig {
            workgroup_size: [compiled.workgroup_size[0], 1, 1],
            lds_size: compiled.lds_size,
        }).unwrap();

        // kernargs: out_ptr(8) + n(4) + iters(4) = 16 bytes
        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        ka[8..12].copy_from_slice(&(n as u32).to_le_bytes());
        ka[12..16].copy_from_slice(&loop_iters.to_le_bytes());

        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [compiled.workgroup_size[0], 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n];
        unsafe {
            out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n * 4));
        }

        for i in 0..n {
            assert!((result[i] - 99.5).abs() < 1e-6,
                "forloop[{}]: expected 99.5, got {}", i, result[i]);
        }
        eprintln!("✓ ForLoop via SSA GPU E2E PASSED ({} elements, {} iters)", n, loop_iters);
    }

    /// Test: WMMA operations → SSA translation (unit test, no GPU)
    #[test]
    fn test_block_to_ssa_wmma() {
        let mut kb = BlockKernel::new("wmma_test", 32);
        let _out_ptr = kb.arg_ptr("out");

        // ZeroAcc → 8×f32 accumulator
        let acc = kb.zero_acc();
        // SplatFragment → 8×u32 fragment (method on BVal)
        let one_f32 = kb.const_f32(1.0);
        let one_u32 = one_f32.to_u32(&mut kb);
        let frag = one_u32.splat_fragment(&mut kb);
        // CvtPkBf16F32 → bf16x2
        let a_f32 = kb.const_f32(2.0);
        let b_f32 = kb.const_f32(3.0);
        let _pk = kb.cvt_pk_bf16(a_f32, b_f32);
        // Wmma: acc = frag * frag + acc (method on BVal: a.wmma(kb, b, acc))
        let acc2 = frag.wmma(&mut kb, frag, acc);
        // ExtractF32: get [0] from accumulator (method on BVal: acc.extract(kb, idx))
        let _elem = acc2.extract(&mut kb, 0);

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== WMMA SSA IR ===\n{}", ir_text);
        assert!(ir_text.contains("zero_acc"), "Should have zero_acc");
        assert!(ir_text.contains("wmma"), "Should have wmma");
        assert!(ir_text.contains("extract"), "Should have extract");
        assert!(ir_text.contains("splat_fragment"), "Should have splat_fragment");
        assert!(ir_text.contains("cvt_pk_bf16_f32"), "Should have cvt_pk_bf16_f32");
    }

    /// Test: WMMA operations compile_via_ssa
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_wmma() {
        let mut kb = BlockKernel::new("wmma_compile", 32);
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let acc = kb.zero_acc();
        let one_f32 = kb.const_f32(1.0);
        let pk = kb.cvt_pk_bf16(one_f32, one_f32);
        let frag_a = pk.splat_fragment(&mut kb);
        let frag_b = pk.splat_fragment(&mut kb);
        let acc2 = frag_a.wmma(&mut kb, frag_b, acc);
        let elem = acc2.extract(&mut kb, 0);

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        kb.store(out_ptr, offsets, elem, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with WMMA failed");
        eprintln!("✓ WMMA compile_via_ssa: {} bytes ELF", compiled.elf.len());
        assert!(compiled.elf.len() > 100);
    }

    #[test]
    #[cfg(feature = "wsl_dxg")]
    fn test_compile_via_ssa_wmma_gfx1201_operand_widths() {
        let mut kb = BlockKernel::new("wmma_compile_gfx1201", 32);
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let acc = kb.zero_acc();
        let one_f32 = kb.const_f32(1.0);
        let pk = kb.cvt_pk_bf16(one_f32, one_f32);
        let frag_a = pk.splat_fragment(&mut kb);
        let frag_b = pk.splat_fragment(&mut kb);
        let acc2 = frag_a.wmma(&mut kb, frag_b, acc);
        let elem = acc2.extract(&mut kb, 0);

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        kb.store(out_ptr, offsets, elem, mask);

        let func = block_to_ssa(&kb).expect("block_to_ssa failed");
        let lowered = tile_ssa_lower::lower_elementwise_1d(&func, kb.get_block_size(), 1)
            .expect("lower_elementwise_1d failed");
        let asm = lowered.kernel.to_assembly(Target::GFX1201)
            .expect("GFX1201 WMMA assembly failed");
        let wmma_line = asm.lines()
            .find(|line| line.contains("v_wmma_f32_16x16x16_bf16"))
            .expect("missing WMMA instruction in GFX1201 assembly");
        let spans: Vec<u32> = wmma_line
            .split("v[")
            .skip(1)
            .map(|part| {
                let regs = part.split(']').next().expect("unterminated register range");
                let (lo, hi) = regs.split_once(':').expect("expected register range");
                hi.parse::<u32>().unwrap() - lo.parse::<u32>().unwrap() + 1
            })
            .collect();
        eprintln!("GFX1201 WMMA: {}", wmma_line);
        assert_eq!(spans, vec![8, 4, 4, 8], "unexpected GFX1201 WMMA operand widths");

        let compiled = kb.compile_via_ssa(Target::GFX1201)
            .expect("compile_via_ssa with GFX1201 WMMA failed");
        assert!(compiled.elf.len() > 100);
    }

    /// Test: WgReduceAdd SSA compilation (no GPU)
    #[test]
    fn test_block_to_ssa_wg_reduce() {
        let mut kb = BlockKernel::new("wg_reduce_test", 64);
        let _out_ptr = kb.arg_ptr("out");
        let _n = kb.arg_u32("n");

        let val = kb.const_f32(1.0);
        let _sum = kb.wg_reduce_sum(val);
        let _max = kb.wg_reduce_max(val);

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== WgReduce SSA IR ===\n{}", ir_text);
        assert!(ir_text.contains("wg_reduce_add"), "Should have wg_reduce_add");
        assert!(ir_text.contains("wg_reduce_max"), "Should have wg_reduce_max");
    }

    /// Test: WgReduceAdd compile_via_ssa
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_wg_reduce() {
        let mut kb = BlockKernel::new("wg_reduce_compile", 32);
        let out_ptr = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let tid = kb.thread_id();
        let tid_f32 = tid.to_f32(&mut kb);
        let sum = kb.wg_reduce_sum(tid_f32);

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        kb.store(out_ptr, offsets, sum, mask);

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with WgReduce failed");
        eprintln!("✓ WgReduceAdd compile_via_ssa: {} bytes ELF", compiled.elf.len());
        assert!(compiled.elf.len() > 100);
    }

    /// Test: TileGemm → TileFunc SSA translation (unit test, no GPU)
    #[test]
    fn test_block_to_ssa_tile_gemm() {
        use super::super::block_dsl::TileGemmConfig;

        let mut kb = BlockKernel::new("gemm_ssa_test", 128);
        let x = kb.arg_ptr("X");
        let w = kb.arg_ptr("W");
        let y = kb.arg_ptr("Y");
        let k = kb.arg_u32("K");
        let n = kb.arg_u32("N");

        let config = TileGemmConfig {
            tile_m: 32,
            tile_n: 64,
            tile_k: 16,
            wgp_mode: false,
            split_k: 1,
            swap_grid: false,
        };
        kb.tile_gemm(x, w, y, k, n, config);

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== TileGemm SSA IR ===\n{}", ir_text);

        // Verify tile ops were emitted
        assert!(ir_text.contains("tile_load_2d"), "Should have tile_load_2d");
        assert!(ir_text.contains("tile_dot"), "Should have tile_dot");
        assert!(ir_text.contains("tile_store_2d"), "Should have tile_store_2d");
        // tile_zeros() emits splat(0.0) in IR
        assert!(ir_text.contains("splat"), "Should have splat (from tile_zeros)");

        // ── Verify TensorLayout annotations ──
        // tile_load_2d results should have Shared layout (#shared)
        assert!(ir_text.contains("#shared"), 
            "tile_load_2d should produce #shared layout in IR dump. Got:\n{}", ir_text);
        // tile_dot / tile_zeros results should have MmaAccumulator layout (#mma)
        assert!(ir_text.contains("#mma"),
            "tile_dot/tile_zeros should produce #mma layout in IR dump. Got:\n{}", ir_text);

        // Structural: verify layout types on specific values
        use super::super::tile_ssa::TileOp;
        let all_ops = func.all_ops();
        for op in all_ops {
            match op {
                TileOp::TileLoad2D { result, .. } => {
                    assert!(func.value_type(*result).layout().is_shared(),
                        "TileLoad2D result should be Shared, got: {}", func.value_type(*result));
                }
                TileOp::TileDot { result, .. } => {
                    assert!(func.value_type(*result).layout().is_mma(),
                        "TileDot result should be MmaAccumulator, got: {}", func.value_type(*result));
                }
                _ => {}
            }
        }
    }

    /// Test: TileGemm compile_via_ssa → ELF (full pipeline, no GPU dispatch)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_tile_gemm() {
        use super::super::block_dsl::TileGemmConfig;

        let mut kb = BlockKernel::new("gemm_ssa_compile", 128);
        let x = kb.arg_ptr("X");
        let w = kb.arg_ptr("W");
        let y = kb.arg_ptr("Y");
        let k = kb.arg_u32("K");
        let n = kb.arg_u32("N");

        let config = TileGemmConfig {
            tile_m: 32,
            tile_n: 64,
            tile_k: 16,
            wgp_mode: false,
            split_k: 1,
            swap_grid: false,
        };
        kb.tile_gemm(x, w, y, k, n, config);

        // This exercises: block_to_ssa → analyze_tiled_gemm → lower_tiled_gemm → ELF
        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa for TileGemm failed");
        eprintln!("✓ TileGemm compile_via_ssa: {} bytes ELF, kernarg_size={}, wg=[{},{},{}], lds={}",
            compiled.elf.len(), compiled.kernarg_size,
            compiled.workgroup_size[0], compiled.workgroup_size[1], compiled.workgroup_size[2],
            compiled.lds_size);
        assert!(compiled.elf.len() > 100, "ELF too small");
        assert!(compiled.lds_size > 0, "GEMM should use LDS");
    }

    // ═══════════════════════════════════════════════════════════════
    // L2: A/B GPU 对比测试
    // 同一 BlockKernel，分别走 compile() 和 compile_via_ssa()，
    // 在 GPU 上运行相同输入，对比输出必须 bit-accurate。
    // ═══════════════════════════════════════════════════════════════

    /// Helper: dispatch a CompiledKernel on GPU with given inputs, return output buffer.
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn gpu_dispatch_1d(
        compiled: &crate::t0::dsl::CompiledKernel,
        inputs: &[&[f32]],  // multiple input buffers
        n: usize,
    ) -> Vec<f32> {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        // Allocate input buffers
        let mut gpu_bufs = Vec::new();
        for input in inputs {
            let buf = device.alloc_vram(n * 4).unwrap();
            buf.write(unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, n * 4) });
            gpu_bufs.push(buf);
        }

        // Allocate output buffer
        let out_buf = device.alloc_vram(n * 4).unwrap();
        out_buf.write(&vec![0u8; n * 4]);

        let gk = GpuKernel::load(&device, &compiled.elf, &KernelLoadConfig {
            workgroup_size: [compiled.workgroup_size[0], 1, 1],
            lds_size: compiled.lds_size,
        }).unwrap();

        // Build kernargs: [input_ptrs..., out_ptr, n]
        let mut ka = vec![0u8; compiled.kernarg_size];
        let mut offset = 0usize;
        for buf in &gpu_bufs {
            ka[offset..offset+8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
            offset += 8;
        }
        ka[offset..offset+8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        offset += 8;
        ka[offset..offset+4].copy_from_slice(&(n as u32).to_le_bytes());

        let ka_buf = pool.write_kernargs(0, &ka);
        let grid_x = ((n as u32 + compiled.workgroup_size[0] - 1) / compiled.workgroup_size[0])
            * compiled.workgroup_size[0];
        queue.submit(&gk, [grid_x, 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n];
        unsafe {
            out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n * 4));
        }
        result
    }

    /// L2 test: vadd — compile() vs compile_via_ssa()
    /// Helper: compile with SSA regalloc enabled
    /// BlockKernel doesn't expose set_ssa_regalloc, so we use a lower-level path:
    /// Build T0Kernel from compile() internals, enable SSA regalloc, then compile.
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn compile_with_ssa_regalloc(kb: &BlockKernel) -> Result<crate::t0::dsl::CompiledKernel, String> {
        use crate::t0::dsl::{CompiledKernel, KernArgMeta, KernArgType};
        use crate::t0::ir::ArgKind;

        let func = block_to_ssa(kb)?;
        let wg_size = kb.get_block_size();
        let epl = 1u32;
        let mut lowered = crate::t0::tile_ssa_lower::lower_elementwise_1d(&func, wg_size, epl)?;
        lowered.kernel.set_ssa_regalloc(true);
        let elf = lowered.kernel.compile(Target::GFX1100)?;

        let args: Vec<KernArgMeta> = lowered.kernel.args().iter().map(|a| {
            KernArgMeta {
                name: a.name.clone(),
                kind: match a.kind {
                    ArgKind::Ptr => KernArgType::Ptr,
                    ArgKind::U32 => KernArgType::U32,
                    ArgKind::F32 => KernArgType::F32,
                },
                offset: a.offset as usize,
            }
        }).collect();

        Ok(CompiledKernel {
            elf,
            kernarg_size: lowered.kernel.kernarg_size() as usize,
            workgroup_size: [lowered.wg_size, 1, 1],
            lds_size: lowered.kernel.lds_size(),
            name: func.name.clone(),
            args,
        })
    }

    /// L3 test: vadd with SSA regalloc
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_l3_ssa_regalloc_vadd() {
        let n = 64usize;

        let build_vadd = || {
            let mut kb = BlockKernel::new("l3_vadd", 64);
            let x = kb.arg_ptr("x");
            let y = kb.arg_ptr("y");
            let out = kb.arg_ptr("out");
            let nn = kb.arg_u32("n");
            let offs = kb.arange(0, 64);
            let mask = offs.lt(&mut kb, nn);
            let a = kb.load(x, offs, mask);
            let b = kb.load(y, offs, mask);
            let c = a.add(&mut kb, b);
            kb.store(out, offs, c, mask);
            kb
        };

        let compiled_legacy = build_vadd().compile(Target::GFX1100).expect("compile failed");
        let compiled_ssa = compile_with_ssa_regalloc(&build_vadd()).expect("SSA regalloc compile failed");

        let x_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let y_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();

        let result_legacy = gpu_dispatch_1d(&compiled_legacy, &[&x_data, &y_data], n);
        let result_ssa = gpu_dispatch_1d(&compiled_ssa, &[&x_data, &y_data], n);

        for i in 0..n {
            assert!(
                (result_legacy[i] - result_ssa[i]).abs() < 1e-6,
                "L3 vadd regalloc mismatch at [{}]: legacy={}, ssa={}",
                i, result_legacy[i], result_ssa[i]
            );
        }
        eprintln!("✓ L3 SSA regalloc vadd PASSED ({} elements)", n);
    }

    /// L3 test: silu with SSA regalloc (higher register pressure)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_l3_ssa_regalloc_silu() {
        let n = 64usize;

        let build_silu = || {
            let mut kb = BlockKernel::new("l3_silu", 64);
            let x_ptr = kb.arg_ptr("x");
            let out_ptr = kb.arg_ptr("out");
            let nn = kb.arg_u32("n");
            let offs = kb.arange(0, 64);
            let mask = offs.lt(&mut kb, nn);
            let x = kb.load(x_ptr, offs, mask);
            let silu = x.silu(&mut kb);
            kb.store(out_ptr, offs, silu, mask);
            kb
        };

        let compiled_legacy = build_silu().compile(Target::GFX1100).expect("compile failed");
        let compiled_ssa = compile_with_ssa_regalloc(&build_silu()).expect("SSA regalloc compile failed");

        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.2).collect();

        let result_legacy = gpu_dispatch_1d(&compiled_legacy, &[&x_data], n);
        let result_ssa = gpu_dispatch_1d(&compiled_ssa, &[&x_data], n);

        let max_err = (0..n).map(|i| (result_legacy[i] - result_ssa[i]).abs()).fold(0.0f32, f32::max);
        for i in 0..n {
            assert!(
                (result_legacy[i] - result_ssa[i]).abs() < 1e-5,
                "L3 silu regalloc mismatch at [{}]: legacy={}, ssa={}, input={}",
                i, result_legacy[i], result_ssa[i], x_data[i]
            );
        }
        eprintln!("✓ L3 SSA regalloc silu PASSED ({} elements, max_err={:.2e})", n, max_err);
    }

    /// L3 test: fma with SSA regalloc (3 inputs = more register pressure)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_l3_ssa_regalloc_fma() {
        let n = 64usize;

        let build_fma = || {
            let mut kb = BlockKernel::new("l3_fma", 64);
            let a_ptr = kb.arg_ptr("a");
            let b_ptr = kb.arg_ptr("b");
            let c_ptr = kb.arg_ptr("c");
            let out_ptr = kb.arg_ptr("out");
            let nn = kb.arg_u32("n");
            let offs = kb.arange(0, 64);
            let mask = offs.lt(&mut kb, nn);
            let a = kb.load(a_ptr, offs, mask);
            let b = kb.load(b_ptr, offs, mask);
            let c = kb.load(c_ptr, offs, mask);
            let r = a.fma(&mut kb, b, c);
            kb.store(out_ptr, offs, r, mask);
            kb
        };

        let compiled_legacy = build_fma().compile(Target::GFX1100).expect("compile failed");
        let compiled_ssa = compile_with_ssa_regalloc(&build_fma()).expect("SSA regalloc compile failed");

        let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let c_data: Vec<f32> = (0..n).map(|i| (i as f32) * -0.02).collect();

        let result_legacy = gpu_dispatch_1d(&compiled_legacy, &[&a_data, &b_data, &c_data], n);
        let result_ssa = gpu_dispatch_1d(&compiled_ssa, &[&a_data, &b_data, &c_data], n);

        let max_err = (0..n).map(|i| (result_legacy[i] - result_ssa[i]).abs()).fold(0.0f32, f32::max);
        for i in 0..n {
            assert!(
                (result_legacy[i] - result_ssa[i]).abs() < 1e-6,
                "L3 fma regalloc mismatch at [{}]: legacy={}, ssa={}",
                i, result_legacy[i], result_ssa[i]
            );
        }
        eprintln!("✓ L3 SSA regalloc fma PASSED ({} elements, max_err={:.2e})", n, max_err);
    }

    // ════════════════════════════════════════════
    // 2D Tile Lowering Tests
    // ════════════════════════════════════════════

    /// T1: 2D grid dispatch — basic 2D matrix add using program_id(0) and program_id(1)
    /// Y[i,j] = A[i,j] + B[i,j] where each workgroup processes a BLOCK_M × BLOCK_N tile
    #[test]
    fn test_compile_2d_matrix_add() {
        let block_m: u32 = 32;
        let block_n: u32 = 4;
        let total = block_m * block_n;

        let mut kb = BlockKernel::new("mat_add_2d", total);
        kb.set_block_size_2d(block_m, block_n);

        let a_ptr = kb.arg_ptr("A");
        let b_ptr = kb.arg_ptr("B");
        let y_ptr = kb.arg_ptr("Y");
        let m = kb.arg_u32("M");
        let n = kb.arg_u32("N");
        let stride = kb.arg_u32("stride");

        let pid_m = kb.program_id(0);
        let pid_n = kb.program_id(1);
        let tid_x = kb.thread_id();
        let tid_y = kb.thread_id_y();

        // row = pid_m * BLOCK_M + tid_x
        let bm = kb.const_u32(block_m);
        let bn = kb.const_u32(block_n);
        let row_base = pid_m.mul(&mut kb, bm);
        let col_base = pid_n.mul(&mut kb, bn);
        let row = row_base.add(&mut kb, tid_x);
        let col = col_base.add(&mut kb, tid_y);

        // offset = row * stride + col
        let row_off = row.mul(&mut kb, stride);
        let offset = row_off.add(&mut kb, col);

        // Bounds check: mask = (row < M) & (col < N)
        let in_rows = row.lt(&mut kb, m);
        let in_cols = col.lt(&mut kb, n);
        let mask = in_rows.and_bool(&mut kb, in_cols);

        // Load, add, store
        let a_val = kb.load(a_ptr, offset, mask);
        let b_val = kb.load(b_ptr, offset, mask);
        let y_val = a_val.add(&mut kb, b_val);
        kb.store(y_ptr, offset, y_val, mask);

        let compiled = kb.compile(Target::GFX1100).expect("2D matrix add compile failed");
        assert!(compiled.elf.len() > 100, "ELF too small");
        assert_eq!(compiled.workgroup_size, [block_m, block_n, 1],
            "Expected 2D WG [32, 4, 1], got {:?}", compiled.workgroup_size);
        eprintln!("✓ T1 2D matrix add compiled: {} bytes, WG={:?}",
            compiled.elf.len(), compiled.workgroup_size);
    }

    /// T2: 2D transpose — Y[j,i] = X[i,j]
    #[test]
    fn test_compile_2d_transpose() {
        let block_m: u32 = 32;
        let block_n: u32 = 4;

        let mut kb = BlockKernel::new("transpose_2d", block_m * block_n);
        kb.set_block_size_2d(block_m, block_n);

        let x_ptr = kb.arg_ptr("X");
        let y_ptr = kb.arg_ptr("Y");
        let m = kb.arg_u32("M");
        let n = kb.arg_u32("N");
        let stride_x = kb.arg_u32("stride_x");
        let stride_y = kb.arg_u32("stride_y");

        let pid_m = kb.program_id(0);
        let pid_n = kb.program_id(1);
        let tid_x = kb.thread_id();
        let tid_y = kb.thread_id_y();

        let bm = kb.const_u32(block_m);
        let bn = kb.const_u32(block_n);
        let row = pid_m.mul(&mut kb, bm).add(&mut kb, tid_x);
        let col = pid_n.mul(&mut kb, bn).add(&mut kb, tid_y);

        let in_rows = row.lt(&mut kb, m);
        let in_cols = col.lt(&mut kb, n);
        let mask = in_rows.and_bool(&mut kb, in_cols);

        // src_off = row * stride_x + col
        let src_off = row.mul(&mut kb, stride_x).add(&mut kb, col);
        let val = kb.load(x_ptr, src_off, mask);

        // dst_off = col * stride_y + row (transposed)
        let dst_off = col.mul(&mut kb, stride_y).add(&mut kb, row);
        kb.store(y_ptr, dst_off, val, mask);

        let compiled = kb.compile(Target::GFX1100).expect("2D transpose compile failed");
        assert!(compiled.elf.len() > 100);
        assert_eq!(compiled.workgroup_size, [block_m, block_n, 1]);
        eprintln!("✓ T2 2D transpose compiled: {} bytes, WG={:?}",
            compiled.elf.len(), compiled.workgroup_size);
    }

    /// T3: 2D scalar multiply — Y[i,j] = X[i,j] * scalar (mixed scalar/vector 2D)
    #[test]
    fn test_compile_2d_scalar_mul() {
        let block_m: u32 = 64;
        let block_n: u32 = 4;

        let mut kb = BlockKernel::new("scalar_mul_2d", block_m * block_n);
        kb.set_block_size_2d(block_m, block_n);

        let x_ptr = kb.arg_ptr("X");
        let y_ptr = kb.arg_ptr("Y");
        let s = kb.arg_f32("scalar");
        let n_arg = kb.arg_u32("N");
        let stride = kb.arg_u32("stride");

        let pid_m = kb.program_id(0);
        let pid_n = kb.program_id(1);
        let tid_x = kb.thread_id();
        let tid_y = kb.thread_id_y();

        let bm = kb.const_u32(block_m);
        let bn = kb.const_u32(block_n);
        let row = pid_m.mul(&mut kb, bm).add(&mut kb, tid_x);
        let col = pid_n.mul(&mut kb, bn).add(&mut kb, tid_y);

        let off = row.mul(&mut kb, stride).add(&mut kb, col);
        let mask = off.lt(&mut kb, n_arg);
        let x_val = kb.load(x_ptr, off, mask);
        let y_val = x_val.mul(&mut kb, s);
        kb.store(y_ptr, off, y_val, mask);

        let compiled = kb.compile(Target::GFX1100).expect("2D scalar_mul compile failed");
        assert!(compiled.elf.len() > 100);
        assert_eq!(compiled.workgroup_size, [block_m, block_n, 1]);
        eprintln!("✓ T3 2D scalar_mul compiled: {} bytes, WG={:?}",
            compiled.elf.len(), compiled.workgroup_size);
    }

    /// T4: Non-square 2D tile [64×2] with SiLU activation
    #[test]
    fn test_compile_2d_non_square_tile() {
        let block_m: u32 = 64;
        let block_n: u32 = 2;

        let mut kb = BlockKernel::new("non_square_2d", block_m * block_n);
        kb.set_block_size_2d(block_m, block_n);

        let x_ptr = kb.arg_ptr("X");
        let y_ptr = kb.arg_ptr("Y");
        let n_arg = kb.arg_u32("N");
        let stride = kb.arg_u32("stride");

        let pid = kb.program_id(0);
        let tid_x = kb.thread_id();
        let tid_y = kb.thread_id_y();

        let bm = kb.const_u32(block_m);
        let row = pid.mul(&mut kb, bm).add(&mut kb, tid_x);
        let off = row.mul(&mut kb, stride).add(&mut kb, tid_y);
        let mask = off.lt(&mut kb, n_arg);

        let x_val = kb.load(x_ptr, off, mask);
        let y_val = x_val.silu(&mut kb);
        kb.store(y_ptr, off, y_val, mask);

        let compiled = kb.compile(Target::GFX1100).expect("Non-square 2D compile failed");
        assert!(compiled.elf.len() > 100);
        assert_eq!(compiled.workgroup_size, [block_m, block_n, 1]);
        eprintln!("✓ T4 Non-square 2D [64×2] compiled: {} bytes, WG={:?}",
            compiled.elf.len(), compiled.workgroup_size);
    }

    /// T5: 2D with wave reduction — row-wise sum
    #[test]
    fn test_compile_2d_row_sum() {
        let block_m: u32 = 32;
        let block_n: u32 = 32;

        let mut kb = BlockKernel::new("row_sum_2d", block_m * block_n);
        kb.set_block_size_2d(block_m, block_n);

        let x_ptr = kb.arg_ptr("X");
        let y_ptr = kb.arg_ptr("Y");
        let n_arg = kb.arg_u32("N");
        let stride = kb.arg_u32("stride");

        let pid = kb.program_id(0);
        let tid_x = kb.thread_id();
        let tid_y = kb.thread_id_y();

        let bm = kb.const_u32(block_m);
        let row = pid.mul(&mut kb, bm).add(&mut kb, tid_x);
        let off = row.mul(&mut kb, stride).add(&mut kb, tid_y);
        let mask = off.lt(&mut kb, n_arg);

        let x_val = kb.load(x_ptr, off, mask);
        let sum = kb.wave_reduce_sum(x_val);

        // Only tid_y == 0 writes
        let one = kb.const_u32(1);
        let first_mask = tid_y.lt(&mut kb, one); // tid_y < 1 → tid_y == 0
        kb.store(y_ptr, row, sum, first_mask);

        let compiled = kb.compile(Target::GFX1100).expect("2D row_sum compile failed");
        assert!(compiled.elf.len() > 100);
        assert_eq!(compiled.workgroup_size, [block_m, block_n, 1]);
        eprintln!("✓ T5 2D row_sum [32×32] compiled: {} bytes, WG={:?}",
            compiled.elf.len(), compiled.workgroup_size);
    }

    /// T6: 1D kernel still works after 2D changes (regression test)
    #[test]
    fn test_1d_regression_after_2d() {
        let mut kb = BlockKernel::new("vadd_1d", 256);
        let x = kb.arg_ptr("X");
        let y = kb.arg_ptr("Y");
        let z = kb.arg_ptr("Z");
        let n = kb.arg_u32("N");

        let pid = kb.program_id(0);
        let bs = kb.const_u32(256);
        let base = pid.mul(&mut kb, bs);
        let tid = kb.thread_id();
        let idx = base.add(&mut kb, tid);

        let mask = idx.lt(&mut kb, n);
        let xv = kb.load(x, idx, mask);
        let yv = kb.load(y, idx, mask);
        let zv = xv.add(&mut kb, yv);
        kb.store(z, idx, zv, mask);

        let compiled = kb.compile(Target::GFX1100).expect("1D regression compile failed");
        assert!(compiled.elf.len() > 100);
        assert_eq!(compiled.workgroup_size, [256, 1, 1],
            "1D kernel should have workgroup_size [256, 1, 1], got {:?}", compiled.workgroup_size);
        eprintln!("✓ T6 1D regression PASSED: {} bytes, WG={:?}",
            compiled.elf.len(), compiled.workgroup_size);
    }

    /// Test: ForAccBegin/ForAccEnd → SSA for_range_with_acc_runtime translation
    #[test]
    fn test_block_to_ssa_for_acc() {
        use super::super::block_dsl::BlockKernel;

        let mut kb = BlockKernel::new("sum_test", 256);
        let _ptr = kb.arg_ptr("data");
        let out = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        // Create accumulator loop: sum = 0; for i in [0, n) { sum += 1.0 }
        let zero = kb.const_f32(0.0);
        let start = kb.const_u32(0);
        let one = kb.const_f32(1.0);

        let (iter, acc) = kb.for_range_acc(start, n, 1, zero);
        let _ = iter; // iter var available but unused here
        let new_acc = acc.add(&mut kb, one);
        let result = kb.end_for_acc(iter, new_acc);

        // Store result
        let pid = kb.program_id(0);
        let bs = kb.const_u32(256);
        let pid_mul = pid.mul(&mut kb, bs);
        let arange = kb.arange(0, 256);
        let offsets = arange.add(&mut kb, pid_mul);
        let mask = offsets.lt(&mut kb, n);
        kb.store(out, offsets, result, mask);

        // Translate to SSA
        let func = super::block_to_ssa(&kb).expect("block_to_ssa should succeed for ForAcc kernel");
        let ir = func.dump();
        eprintln!("=== ForAcc block_to_ssa IR ===\n{}", ir);

        // Verify SSA structure
        // Should have ≥ 4 blocks: entry, header, body, exit
        let blocks = func.all_blocks();
        assert!(blocks.len() >= 4,
            "Should have ≥4 blocks (entry, header, body, exit), got {}", blocks.len());

        // Header block (bb1) should have 2 params: iv (U32) and acc (F32)
        let header = &blocks[1];
        assert_eq!(header.params.len(), 2,
            "Header block should have 2 params (iv, acc), got {}", header.params.len());

        eprintln!("✓ test_block_to_ssa_for_acc PASSED");
    }

    /// Test: IfMask (if-only, no else) → SSA translation
    #[test]
    fn test_block_to_ssa_if_only() {
        let mut kb = BlockKernel::new("if_only_ssa", 32);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let a = kb.load(x, offsets, mask);
        let two = kb.const_f32(2.0);
        let doubled = a.mul(&mut kb, two);

        kb.if_mask(mask);
        kb.store(y, offsets, doubled, mask);
        kb.end_if();

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== if_only SSA IR ===\n{}", ir_text);
        assert!(ir_text.contains("exec_mask_push"), "Should contain exec_mask_push");
        assert!(ir_text.contains("exec_mask_pop"), "Should contain exec_mask_pop");
        assert!(!ir_text.contains("exec_mask_flip"), "Should NOT contain exec_mask_flip (no else)");
        eprintln!("✓ test_block_to_ssa_if_only PASSED");
    }

    /// Test: IfMask/ElseMask/EndIf → SSA translation
    #[test]
    fn test_block_to_ssa_if_else() {
        let mut kb = BlockKernel::new("if_else_ssa", 32);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let a = kb.load(x, offsets, mask);

        kb.if_mask(mask);
        kb.store(y, offsets, a, mask);
        kb.else_mask();
        let zero = kb.const_f32(0.0);
        kb.store(y, offsets, zero, mask);
        kb.end_if();

        let func = block_to_ssa(&kb).unwrap();
        let ir_text = func.dump();
        eprintln!("=== if_else SSA IR ===\n{}", ir_text);
        assert!(ir_text.contains("exec_mask_push"), "Should contain exec_mask_push");
        assert!(ir_text.contains("exec_mask_flip"), "Should contain exec_mask_flip");
        assert!(ir_text.contains("exec_mask_pop"), "Should contain exec_mask_pop");
        eprintln!("✓ test_block_to_ssa_if_else PASSED");
    }

    /// Test: IfMask compile_via_ssa produces valid ELF (no GPU)
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_via_ssa_if_else() {
        let mut kb = BlockKernel::new("if_else_elf", 32);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let a = kb.load(x, offsets, mask);

        kb.if_mask(mask);
        kb.store(y, offsets, a, mask);
        kb.else_mask();
        let zero = kb.const_f32(0.0);
        kb.store(y, offsets, zero, mask);
        kb.end_if();

        let compiled = kb.compile_via_ssa(Target::GFX1100)
            .expect("compile_via_ssa with if/else failed");
        eprintln!("✓ if_else compile_via_ssa: {} bytes ELF, ka_size={}, wg={}",
            compiled.elf.len(), compiled.kernarg_size, compiled.workgroup_size[0]);
        assert!(compiled.elf.len() > 100, "ELF too small");
        assert_eq!(compiled.args.len(), 3); // x, y, n
    }

    /// COMPILE-ONLY: gemm_tn_naive kernel (same as test_gpu_gemm_tn but no GPU dispatch).
    /// Run with T0_DUMP_ASM=1 to inspect generated ISA safely.
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_compile_gemm_tn_isa_dump() {
        const BM: u32 = 16;
        const BN: u32 = 16;
        const BLOCK_SIZE: u32 = BM * BN;

        let mut kb = BlockKernel::new("gemm_tn_naive", BLOCK_SIZE);
        let a_ptr = kb.arg_ptr("A");
        let b_ptr = kb.arg_ptr("B");
        let c_ptr = kb.arg_ptr("C");
        let m_arg = kb.arg_u32("M");
        let n_arg = kb.arg_u32("N");
        let k_arg = kb.arg_u32("K");

        let pid_m = kb.program_id(0);
        let pid_n = kb.program_id(1);

        let tid = kb.arange(0, BLOCK_SIZE);
        let local_n = tid.bitand(&mut kb, (BN - 1) as u32);
        let local_m = tid.shr(&mut kb, 4);

        let bm = kb.const_u32(BM);
        let bn = kb.const_u32(BN);
        let gm = pid_m.mul(&mut kb, bm).add(&mut kb, local_m);
        let gn = pid_n.mul(&mut kb, bn).add(&mut kb, local_n);

        let mask_m = gm.lt(&mut kb, m_arg);
        let mask_n = gn.lt(&mut kb, n_arg);

        let lds = kb.lds_alloc(BLOCK_SIZE * 4);
        let lds_tid = kb.arange(0, BLOCK_SIZE);

        let zero_f = kb.const_f32(0.0);
        kb.lds_store(lds, lds_tid, zero_f);
        kb.barrier();

        let zero = kb.const_u32(0);
        let iter_k = kb.for_range(zero, k_arg, 1);
        {
            let a_off = iter_k.mul(&mut kb, m_arg).add(&mut kb, gm);
            let a_val = kb.load(a_ptr, a_off, mask_m);

            let b_off = iter_k.mul(&mut kb, n_arg).add(&mut kb, gn);
            let b_val = kb.load(b_ptr, b_off, mask_n);

            let prod = a_val.mul(&mut kb, b_val);

            let cur = kb.lds_load(lds, lds_tid);
            let new_acc = cur.add(&mut kb, prod);
            kb.lds_store(lds, lds_tid, new_acc);
            kb.barrier();
        }
        kb.end_for(iter_k);

        let result = kb.lds_load(lds, lds_tid);
        let c_off = gm.mul(&mut kb, n_arg).add(&mut kb, gn);
        kb.store(c_ptr, c_off, result, mask_m);

        // COMPILE ONLY — no GPU dispatch!
        let compiled = kb.compile(Target::GFX1100).unwrap();
        eprintln!("✓ gemm_tn_naive COMPILE-ONLY: elf={} bytes, lds={}, ka={}, wg=[{},{}]",
            compiled.elf.len(), compiled.lds_size, compiled.kernarg_size,
            compiled.workgroup_size[0], compiled.workgroup_size[1]);
        assert!(compiled.elf.len() > 100, "ELF too small");
        assert_eq!(compiled.args.len(), 6); // A, B, C, M, N, K
    }
}
