//! Tile SSA → T0Kernel Lowering Pass
//!
//! 将 tile_ssa IR 编译为可执行的 T0Kernel（Vec<Op>）。
//!
//! # 支持的模式
//!
//! - **1D Elementwise**: 向量/标量操作，每线程处理 N 个元素
//!   - load, store, add, sub, mul, div, neg, exp, log, sqrt, rcp, relu, silu
//!   - 自动边界检查（EXEC mask）
//!
//! - **Tiled GEMM**: 2D tile 操作（Triton 语义）
//!   - TileLoad2D, TileDot, TileStore2D
//!   - 自动处理 LDS 双缓冲、协作加载、WMMA 调度
//!   - load, store, add, sub, mul, div, neg, exp, log, sqrt, rcp, relu, silu
//!   - 自动边界检查（EXEC mask）
//!
//! # 架构
//!
//! ```text
//! TileFunc (SSA IR)
//!    │
//!    ├─ analyze: 识别模式（elementwise? reduce? dot?）
//!    │
//!    └─ lower_elementwise_1d:
//!       ├─ 声明 kernargs（与 SSA 参数一一对应）
//!       ├─ 计算 global_id = TGID.x * WG + TID
//!       ├─ 为每个 SSA Value 分配 VGPR/SGPR
//!       ├─ 按拓扑序翻译每个 TileOp → Op 序列
//!       └─ 生成 endpgm
//! ```

use super::tile_ssa::*;
use super::ir::*;
use super::compile::T0Kernel;
use super::gemm_gen;
use super::tile_ir;
use std::collections::HashMap;

// ============================================================================
// Lowering Result
// ============================================================================

/// Lowering 结果：编译好的内核 + 部署元数据
pub struct LoweredKernel {
    /// 编译好的 T0Kernel
    pub kernel: T0Kernel,
    /// 建议的 workgroup size
    pub wg_size: u32,
    /// 每线程处理的元素数
    pub elements_per_thread: u32,
}

impl LoweredKernel {
    /// 计算 grid X 维度（线程数）给定总元素数
    pub fn compute_grid(&self, n_elems: u32) -> [u32; 3] {
        let elems_per_wg = self.wg_size * self.elements_per_thread;
        let n_wg = (n_elems + elems_per_wg - 1) / elems_per_wg;
        [n_wg * self.wg_size, 1, 1]
    }
}

// ============================================================================
// Dot (GEMM) Lowering Result
// ============================================================================

/// Dot lowering 结果：GEMM 内核 + Grid 计算
///
/// Dot 是 mega-op：委托给 `gemm_gen::generate()` 生成整个内核。
/// 内核参数布局: [X:u64, WT:u64, Y:u64, K:u32, N:u32, split_k_shift:u32, y_split_stride:u32]
pub struct LoweredDotKernel {
    /// 编译好的 T0Kernel（含 LDS、WGP 配置）
    pub kernel: T0Kernel,
    /// GEMM 配置（tile sizes, split-K, etc.）
    pub config: gemm_gen::GemmConfig,
}

impl LoweredDotKernel {
    /// 计算 grid 维度 for dispatch
    pub fn compute_grid(&self, m: u32, n: u32) -> [u32; 3] {
        let (gx, gy) = gemm_gen::compute_grid_auto(&self.config, m, n);
        [gx, gy, 1]
    }

    /// 构建 40-byte kernarg buffer
    pub fn build_kernargs(
        &self, x_addr: u64, wt_addr: u64, y_addr: u64,
        m: u32, k: u32, n: u32,
    ) -> [u8; 40] {
        gemm_gen::build_kernargs(x_addr, wt_addr, y_addr, k, n, m, &self.config)
    }

    /// Workgroup size
    pub fn wg_size(&self) -> u32 { self.config.wg_size }

    /// LDS size
    pub fn lds_size(&self) -> u32 { self.kernel.lds_size() }
}

/// Lower a TileOp::Dot to a GEMM kernel via gemm_gen.
///
/// The resulting kernel uses NT layout: Y[M,N] = A[M,K] @ B[N,K]^T.
/// Caller must ensure B is in [N,K] (row-major, each row is a K-vector).
///
/// # Arguments
/// * `m` - Rows of A / rows of Y
/// * `k` - Cols of A / cols of B (contraction dimension)
/// * `n` - Rows of B (= cols of Y in NT mode)
pub fn lower_dot(m: u32, k: u32, n: u32) -> Result<LoweredDotKernel, String> {
    lower_dot_for_target(current_target(), m, k, n)
}

pub fn lower_dot_for_target(target: Target, m: u32, k: u32, n: u32) -> Result<LoweredDotKernel, String> {
    // Auto-select optimal GEMM config based on matrix dimensions
    let mut config = with_target_context(target, || gemm_gen::auto_select(m, k, n));

    // CRITICAL: Disable split-K for standalone Dot lowering.
    // Split-K writes to multiple output planes and requires a separate
    // reduction kernel to sum them. lower_dot produces a single kernel,
    // so split-K would cause Y buffer overflow and GPU hang.
    config.split_k = None;

    // Generate the GEMM kernel (cooperative loading, LDS double-buffer, WMMA)
    let kernel = with_target_context(target, || gemm_gen::generate(&config));

    Ok(LoweredDotKernel { kernel, config })
}

// ============================================================================
// Value → Register 映射
// ============================================================================

/// SSA Value 在机器码中的表示形式
#[derive(Clone, Debug)]
enum MachineVal {
    /// Value 存在 VGPR 中（常见情况：tile 元素）
    VReg(VReg),
    /// Value 存在 SGPR 中（kernel 参数、标量常量）
    SReg(SReg),
    /// Value 是 64 位指针（SGPR pair）
    SRegPair(SRegPair),
    /// 内联常量（不需要寄存器）
    InlineInt(i32),
    /// 内联浮点常量
    InlineFloat(f32),
}

// ============================================================================
// Lowering: Elementwise 1D
// ============================================================================

/// Lower 一个 1D elementwise TileFunc 到 T0Kernel。
///
/// 假设：
/// - 所有 tile 操作都是同一长度的 1D vector
/// - 每个线程处理 `epl`（elements per lane）个连续元素
/// - Grid = ceil(n_elems / (WG * epl)) 个 workgroup
pub fn lower_elementwise_1d(func: &TileFunc, wg_size: u32, epl: u32) -> Result<LoweredKernel, String> {
    let mut k = T0Kernel::new(&func.name);
    k.set_wg_size(wg_size);

    // ── Step 1: 声明 kernargs（与 SSA args 一一对应）──
    let mut val_map: HashMap<Value, MachineVal> = HashMap::new();

    for &arg_val in &func.args {
        let def = &func.all_values()[arg_val.0 as usize];
        let name = def.name.as_deref().unwrap_or("arg");
        match &def.ty {
            TileType::Ptr => {
                let pair = k.arg_ptr(name);
                val_map.insert(arg_val, MachineVal::SRegPair(pair));
            }
            TileType::Scalar(ScalarDType::U32) | TileType::Scalar(ScalarDType::I32) => {
                let sr = k.arg_u32(name);
                val_map.insert(arg_val, MachineVal::SReg(sr));
            }
            TileType::Scalar(ScalarDType::F32) => {
                let sr = k.arg_f32(name);
                val_map.insert(arg_val, MachineVal::SReg(sr));
            }
            other => return Err(format!("Unsupported arg type: {}", other)),
        }
    }

    k.emit_arg_loads();

    // Note: 不调用 compute_global_id_x!
    // SSA 程序通过 program_id + arange 显式处理索引。

    // ── Step 2: 为 block params 预分配寄存器 ──
    // CRITICAL: Save the original allocation for each block param in a separate map.
    // This is needed because get_vreg() caches SReg→VReg promotions into val_map,
    // which corrupts the param's identity. When a Branch copies args to params,
    // we must use the ORIGINAL SReg target (not the VReg cache) to ensure the
    // loop header's s_cmp reads the updated value.
    let mut block_param_original: HashMap<Value, MachineVal> = HashMap::new();
    let blocks = func.all_blocks();
    for block in blocks {
        for &param in &block.params {
            let ty = &func.all_values()[param.0 as usize].ty;
            match ty {
                TileType::Scalar(ScalarDType::U32) | TileType::Scalar(ScalarDType::I32)
                | TileType::Scalar(ScalarDType::F32) => {
                    let sr = k.alloc_sreg();
                    val_map.insert(param, MachineVal::SReg(sr));
                    block_param_original.insert(param, MachineVal::SReg(sr));
                }
                TileType::Vector { .. } => {
                    let vr = k.alloc_vreg();
                    val_map.insert(param, MachineVal::VReg(vr));
                    block_param_original.insert(param, MachineVal::VReg(vr));
                }
                _ => return Err(format!("Unsupported block param type: {}", ty)),
            }
        }
    }

    // ── Step 3: 遍历所有 block，按块编号顺序 ──
    let blocks = func.all_blocks();
    for (block_idx, block) in blocks.iter().enumerate() {
        // 非 entry block 发射 label
        if block_idx > 0 {
            k.push(Op::Label(format!("bb{}", block_idx)));
        }

        // EXEC mask save stack for nested if/else (shared across ops in a block)
        let mut exec_save_stack: Vec<SReg> = Vec::new();

        // 翻译块内所有 op
        for &op_idx in &block.ops {
            let op = &func.all_ops()[op_idx];
            lower_tile_op(&mut k, op, func, &mut val_map, &mut exec_save_stack)?;
        }

        // ── Validate ExecMask pairing ──
        // Unmatched ExecMaskPush/Pop would leave EXEC mask corrupted → GPU hang.
        if !exec_save_stack.is_empty() {
            return Err(format!(
                "ExecMask mismatch in block {}: {} unmatched ExecMaskPush without Pop. \
                 This would corrupt EXEC mask and cause GPU hang.",
                block_idx, exec_save_stack.len()
            ));
        }

        // 翻译 terminator
        match &block.terminator {
            Some(Terminator::Return) => {
                k.wait_vmcnt(0);
                k.wait_vscnt(0);
                k.endpgm();
            }
            Some(Terminator::Branch { target, args }) => {
                // 将 args 复制到 target block 的 params
                // CRITICAL: Use block_param_original for the DESTINATION to avoid
                // the val_map pollution from get_vreg() SReg→VReg caching.
                // Without this, loop IV (s15) never gets updated because
                // copy targets VReg(v9) instead of SReg(s15) → infinite loop.
                let target_params: Vec<Value> = func.all_blocks()[target.0 as usize].params.clone();
                for (arg, param) in args.iter().zip(target_params.iter()) {
                    copy_val_to_param(&mut k, &val_map, &block_param_original, *arg, *param)?;
                }
                k.push(Op::Branch(format!("bb{}", target.0)));
            }
            Some(Terminator::CondBranch { cond, true_bb, true_args, false_bb, false_args }) => {
                // 条件分支：先处理 cmp 结果设置 VCC/SCC
                // cond 应该是 Cmp 的结果，VCC 已经被设置
                // s_cbranch_vccz → false_bb (如果 VCC 全 0 则跳转)

                // 先 emit false_args 到 false_bb params（直接 fall through 时使用）
                // 但实际上我们需要：
                // if VCC != 0: branch to true_bb, else fall through to false_bb
                // 但 branch args 需要在跳转前 copy...

                // 简化方案：先复制 true_args, 如果 VCC 跳 true_bb,
                // 否则复制 false_args, 跳(fall-through) false_bb
                // 但这会有 args clobber 问题...

                // 安全方案：emit cond_branch as:
                //   s_cbranch_scc0 → false_label  (SCC=0 跳 false)
                //   <copy true_args>
                //   s_branch → true_label
                // false_entry:
                //   <copy false_args>
                //   s_branch → false_label

                // 更简单：for loop pattern 中 true = body, false = exit
                // body 和 exit 通常没有 args（body args 为空，exit args=loop acc）

                // 对于 for loop: cond_branch(iv < end, body=[], exit=[acc])
                // 使用 s_cmp + s_cbranch_scc1

                // Emit: iv < end 已经在 Cmp 中设置了 VCC
                // 但 cond 是通过 TileOp::Cmp 设置 VCC 的…
                // 我们需要 VCC→SCC 转换：
                // s_cmp_lg_u32 vcc_lo, 0  → SCC = (VCC != 0) ← 不对
                // 更简单：用 s_cbranch_vccnz 跳到 body

                // 实际方案：
                // 1. 如果 cond 来自标量 cmp: 使用 SCC
                // 2. 如果 cond 来自向量 cmp: 使用 VCC
                // for loop 的 iv < end 是标量 cmp → SCC

                let cond_val = val_map.get(cond).cloned();
                let use_scc = matches!(cond_val, Some(MachineVal::SReg(_)) | None);

                if true_args.is_empty() && false_args.is_empty() {
                    // 简单情况：无 args
                    if use_scc {
                        k.push(Op::BranchScc1(format!("bb{}", true_bb.0)));
                        k.push(Op::Branch(format!("bb{}", false_bb.0)));
                    } else {
                        // VCC-based: s_cbranch_vccnz → true_bb
                        k.push(Op::RawAsm(format!("s_cbranch_vccnz .Lbb{}", true_bb.0)));
                        k.push(Op::Branch(format!("bb{}", false_bb.0)));
                    }
                } else {
                    // 有 args 的情况：需要在跳转前复制
                    // Pattern: if SCC=1 goto true_bb else goto false_bb
                    // emit:
                    //   s_cbranch_scc0 .Lfalse_copy  (SCC=0 → goto false path)
                    //   <copy true_args to true_bb params>
                    //   s_branch .Ltrue_bb
                    // .Lfalse_copy:
                    //   <copy false_args to false_bb params>
                    //   s_branch .Lfalse_bb

                    let false_copy_label = format!("bb{}_false", block_idx);

                    if use_scc {
                        k.push(Op::BranchScc0(format!("{}", false_copy_label)));
                    } else {
                        k.push(Op::RawAsm(format!("s_cbranch_vccz .L{}", false_copy_label)));
                    }

                    // True path: copy true_args
                    let true_params: Vec<Value> = func.all_blocks()[true_bb.0 as usize].params.clone();
                    for (arg, param) in true_args.iter().zip(true_params.iter()) {
                        copy_val_to_param(&mut k, &val_map, &block_param_original, *arg, *param)?;
                    }
                    k.push(Op::Branch(format!("bb{}", true_bb.0)));

                    // False path label
                    k.push(Op::Label(false_copy_label));

                    // False path: copy false_args
                    let false_params: Vec<Value> = func.all_blocks()[false_bb.0 as usize].params.clone();
                    for (arg, param) in false_args.iter().zip(false_params.iter()) {
                        copy_val_to_param(&mut k, &val_map, &block_param_original, *arg, *param)?;
                    }
                    k.push(Op::Branch(format!("bb{}", false_bb.0)));
                }
            }
            None => {
                // No terminator: fall through (shouldn't happen in valid SSA)
            }
        }
    }

    Ok(LoweredKernel {
        kernel: k,
        wg_size,
        elements_per_thread: epl,
    })
}

pub fn lower_elementwise_1d_for_target(
    target: Target,
    func: &TileFunc,
    wg_size: u32,
    epl: u32,
) -> Result<LoweredKernel, String> {
    with_target_context(target, || lower_elementwise_1d(func, wg_size, epl))
}

// ============================================================================
// 翻译单个 TileOp
// ============================================================================

fn lower_tile_op(
    k: &mut T0Kernel,
    op: &TileOp,
    func: &TileFunc,
    val_map: &mut HashMap<Value, MachineVal>,
    exec_save_stack: &mut Vec<SReg>,
) -> Result<(), String> {
    match op {
        // ── 常量 ──
        TileOp::ConstU32 { result, value } => {
            if *value <= 64 || (*value as i32) >= -16 && (*value as i32) <= -1 {
                val_map.insert(*result, MachineVal::InlineInt(*value as i32));
            } else {
                let v = k.alloc_vreg();
                k.v_mov_imm(v, *value as i32);
                val_map.insert(*result, MachineVal::VReg(v));
            }
        }
        TileOp::ConstF32 { result, value } => {
            // Check for common inline constants
            let bits = value.to_bits();
            if *value == 0.0 || *value == 0.5 || *value == 1.0 || *value == 2.0 || *value == 4.0
                || *value == -0.5 || *value == -1.0 || *value == -2.0 || *value == -4.0 {
                val_map.insert(*result, MachineVal::InlineFloat(*value));
            } else {
                let v = k.alloc_vreg();
                // Use literal constant via VMov
                k.push(Op::VMov { dst: v, src: Operand::Literal(bits) });
                val_map.insert(*result, MachineVal::VReg(v));
            }
        }

        // ── 索引 ──
        TileOp::ProgramId { result, axis } => {
            let s = k.alloc_sreg();
            match axis {
                0 => k.capture_tgid_x(s),
                1 => k.capture_tgid_y(s),
                2 => k.capture_tgid_z(s),
                _ => return Err(format!("Invalid program_id axis: {}", axis)),
            }
            val_map.insert(*result, MachineVal::SReg(s));
        }

        TileOp::Arange { result, start, len } => {
            // arange(start, len) 生成 per-lane 索引: [start, start+1, ..., start+len-1]
            // 在 wave32 中, 每个 lane 的 index = thread_id_x + start
            // thread_id_x 来自硬件 v0 (WORKITEM_ID_X)
            // 注意：arange 只提供 WG 内的 lane 偏移，workgroup 偏移由 SSA 程序
            //       通过 program_id 显式计算（如 pid * 128 + arange(0, 128)）
            let tid = k.thread_id_x();  // v0 = WORKITEM_ID_X
            let v = k.alloc_vreg();
            if *start == 0 {
                k.v_mov(v, tid);
            } else {
                k.push(Op::VAddU32 {
                    dst: v, src0: Operand::VReg(tid),
                    src1: Operand::InlineInt(*start as i32),
                });
            }
            val_map.insert(*result, MachineVal::VReg(v));
        }

        // ── 2D ThreadId ──
        TileOp::ThreadIdX2D { result, block_x } => {
            // thread_id_x = flat_tid & (block_x - 1)
            assert!(block_x.is_power_of_two(), "block_x must be power of 2");
            let tid = k.thread_id_x();  // v0 = WORKITEM_ID_X (flat)
            let v = k.alloc_vreg();
            k.v_and_b32_imm(v, tid, *block_x - 1);
            val_map.insert(*result, MachineVal::VReg(v));
        }
        TileOp::ThreadIdY2D { result, block_x } => {
            // thread_id_y = flat_tid >> log2(block_x)
            assert!(block_x.is_power_of_two(), "block_x must be power of 2");
            let shift = block_x.trailing_zeros() as u8;
            let tid = k.thread_id_x();  // v0 = WORKITEM_ID_X (flat)
            let v = k.alloc_vreg();
            k.v_lshrrev_b32(v, shift, tid);
            val_map.insert(*result, MachineVal::VReg(v));
        }
        // ── Splat ──
        TileOp::Splat { result, src, .. } => {
            // splat(scalar) → VGPR (broadcast scalar to all lanes)
            let dst_v = k.alloc_vreg();
            match get_val(val_map, *src)? {
                MachineVal::SReg(s) => k.v_mov_from_sgpr(dst_v, s),
                MachineVal::VReg(v) => k.v_mov(dst_v, v),
                MachineVal::InlineInt(i) => k.v_mov_imm(dst_v, i),
                MachineVal::InlineFloat(f) => {
                    k.push(Op::VMov { dst: dst_v, src: Operand::InlineFloat(f) });
                }
                MachineVal::SRegPair(_) => return Err("Cannot splat a pointer".into()),
            }
            val_map.insert(*result, MachineVal::VReg(dst_v));
        }

        // ── Load ──
        // Layout-aware lowering: dispatch based on result's TensorLayout
        TileOp::Load { result, ptr, indices, mask, dtype, .. } => {
            let _result_layout = func.value_type(*result).layout().clone();
            // Currently all 1D loads use Blocked layout (per-lane address calculation)
            // Future: Shared layout → cooperative LDS load, MmaAccumulator → register mapping
            debug_assert!(_result_layout.is_blocked() || matches!(_result_layout, TensorLayout::Scalar),
                "Load: expected Blocked layout, got {:?}", _result_layout);

            let width = match dtype {
                ScalarDType::F32 | ScalarDType::U32 | ScalarDType::I32 => Width::B32,
                ScalarDType::BF16 | ScalarDType::F16 => Width::B16,
                _ => return Err(format!("Unsupported load dtype: {:?}", dtype)),
            };
            let bytes_per_elem = dtype.bytes();

            let dst_v = k.alloc_vreg();
            k.v_mov_imm(dst_v, 0); // zero-init for masked-out lanes

            // 计算 byte address: ptr + indices * bytes_per_elem
            let idx_v = get_vreg(k, val_map, *indices)?;
            let byte_off = k.alloc_vreg();
            match bytes_per_elem {
                4 => k.v_lshlrev_b32(byte_off, 2, idx_v),
                2 => k.v_lshlrev_b32(byte_off, 1, idx_v),
                1 => k.v_mov(byte_off, idx_v),
                _ => return Err(format!("Unsupported element size: {}", bytes_per_elem)),
            }

            // Construct 64-bit address
            let addr = k.alloc_vreg_array(2, Alignment::Align2);
            let ptr_pair = get_ptr(val_map, *ptr)?;
            k.v_mov_from_sgpr(addr, SReg(ptr_pair.0));
            k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(ptr_pair.0 + 1));
            // CRITICAL: clear VCC before 64-bit address add.
            // Without this, VCC carry residual from loop cmp (s_cmp_lt_u32)
            // corrupts the high 32 bits → wrong GPU address → page fault → hard hang.
            k.clear_vcc();
            k.v_add_co(addr, addr, byte_off);
            k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

            // 如果有 mask，用 EXEC mask 保护
            let saved_exec = if let Some(mask_val) = mask {
                // mask 是 bool vector，需要设置到 VCC 再应用到 EXEC
                let mask_v = get_vreg(k, val_map, *mask_val)?;
                // v_cmp_ne_u32 mask, 0 → VCC
                k.v_cmp_gt_u32_imm(mask_v, 0);
                let saved = k.alloc_sreg();
                k.save_exec(saved);
                Some(saved)
            } else {
                None
            };

            k.global_load(dst_v, addr, width, 0);
            k.wait_vmcnt(0);

            if let Some(saved) = saved_exec {
                k.restore_exec(saved);
            }

            val_map.insert(*result, MachineVal::VReg(dst_v));
        }

        // ── Store ──
        // Layout-aware: check source value's layout to determine store strategy
        TileOp::Store { ptr, indices, val, mask } => {
            let _val_layout = func.value_type(*val).layout().clone();
            // Blocked layout: each thread stores its own element via global_store
            // Future: MmaAccumulator → register-to-global with layout conversion
            let val_v = get_vreg(k, val_map, *val)?;
            let val_ty = func.value_type(*val);
            let width = match val_ty.dtype() {
                Some(ScalarDType::F32) | Some(ScalarDType::U32) | Some(ScalarDType::I32) => Width::B32,
                Some(ScalarDType::BF16) | Some(ScalarDType::F16) => Width::B16,
                _ => return Err(format!("Unsupported store dtype: {:?}", val_ty)),
            };
            let bytes_per_elem = val_ty.dtype().map(|d| d.bytes()).unwrap_or(4);

            let idx_v = get_vreg(k, val_map, *indices)?;
            let byte_off = k.alloc_vreg();
            match bytes_per_elem {
                4 => k.v_lshlrev_b32(byte_off, 2, idx_v),
                2 => k.v_lshlrev_b32(byte_off, 1, idx_v),
                _ => k.v_mov(byte_off, idx_v),
            }

            let addr = k.alloc_vreg_array(2, Alignment::Align2);
            let ptr_pair = get_ptr(val_map, *ptr)?;
            k.v_mov_from_sgpr(addr, SReg(ptr_pair.0));
            k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(ptr_pair.0 + 1));
            // CRITICAL: clear VCC before 64-bit address add (same as Load path)
            k.clear_vcc();
            k.v_add_co(addr, addr, byte_off);
            k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

            // 如果有 mask，用 EXEC mask 保护
            let saved_exec = if let Some(mask_val) = mask {
                let mask_v = get_vreg(k, val_map, *mask_val)?;
                k.v_cmp_gt_u32_imm(mask_v, 0);
                let saved = k.alloc_sreg();
                k.save_exec(saved);
                Some(saved)
            } else {
                None
            };

            k.wait_vmcnt(0);
            k.global_store(addr, val_v, width, 0);

            if let Some(saved) = saved_exec {
                k.restore_exec(saved);
            }
        }

        // ── BinOp ──
        TileOp::BinOp { result, op: bin_op, lhs, rhs } => {
            // 检查是否两个操作数都是标量（可以在 SGPR 中操作）
            let lhs_scalar = matches!(val_map.get(lhs), Some(MachineVal::SReg(_)) | Some(MachineVal::InlineInt(_)));
            let rhs_scalar = matches!(val_map.get(rhs), Some(MachineVal::SReg(_)) | Some(MachineVal::InlineInt(_)));
            let result_ty = func.value_type(*result);
            let is_scalar_result = result_ty.is_scalar();

            if lhs_scalar && rhs_scalar && is_scalar_result {
                // 标量路径：s_add_u32, s_mul_i32 等
                let ls = get_sreg_or_inline(val_map, *lhs)?;
                let rs = get_sreg_or_inline(val_map, *rhs)?;
                let dst = k.alloc_sreg();

                // 需要至少一个 SReg 操作数作为 src0
                let (s0, src1) = match (&ls, &rs) {
                    (SOperand::SReg(a), _) => (*a, rs.clone()),
                    (_, SOperand::SReg(b)) => (*b, ls.clone()),
                    (SOperand::InlineInt(a), SOperand::InlineInt(b)) => {
                        // 两个 inline int: 先 mov 一个到 sreg
                        let tmp = k.alloc_sreg();
                        k.push(Op::SMov { dst: tmp, src: SOperand::InlineInt(*a) });
                        (tmp, SOperand::InlineInt(*b))
                    }
                    _ => return Err("Scalar binop: no SReg operand".into()),
                };

                match bin_op {
                    BinOpKind::Add => k.push(Op::SAddU32 { dst, src0: s0, src1 }),
                    BinOpKind::Mul => {
                        // s_mul_i32 需要两个 sreg
                        let s1 = match src1 {
                            SOperand::SReg(r) => r,
                            SOperand::InlineInt(v) => {
                                let tmp = k.alloc_sreg();
                                k.push(Op::SMov { dst: tmp, src: SOperand::InlineInt(v) });
                                tmp
                            }
                            _ => return Err("s_mul_i32 needs SReg".into()),
                        };
                        k.push(Op::SMulI32 { dst, src0: s0, src1: s1 });
                    }
                    _ => {
                        // 其他标量 binop: 回退到 VGPR
                        let lv = get_vreg(k, val_map, *lhs)?;
                        let rv = get_vreg(k, val_map, *rhs)?;
                        let vdst = k.alloc_vreg();
                        match bin_op {
                            BinOpKind::Sub => k.v_sub_u32(vdst, lv, rv),
                            _ => return Err(format!("Unimplemented scalar binop: {:?}", bin_op)),
                        }
                        val_map.insert(*result, MachineVal::VReg(vdst));
                        return Ok(());
                    }
                }
                val_map.insert(*result, MachineVal::SReg(dst));
            } else {
                // 向量路径
                // IMPORTANT: Cache rhs type BEFORE get_vreg, because get_vreg may
                // overwrite InlineInt→VReg in val_map (side effect that caused
                // Shr/Shl to take the wrong raw_asm path → GPU hang).
                let rhs_val_precheck = get_val(val_map, *rhs).ok();
                let lv = get_vreg(k, val_map, *lhs)?;
                let rv = get_vreg(k, val_map, *rhs)?;
                let dst = k.alloc_vreg();
                match bin_op {
                    BinOpKind::Add => {
                        let lty = func.value_type(*lhs);
                        if lty.dtype().map(|d| d.is_float()).unwrap_or(false) {
                            k.v_add_f32(dst, lv, rv);
                        } else {
                            k.v_add_u32(dst, lv, rv);
                        }
                    }
                    BinOpKind::Sub => {
                        let lty = func.value_type(*lhs);
                        if lty.dtype().map(|d| d.is_float()).unwrap_or(false) {
                            k.v_sub_f32(dst, lv, rv);
                        } else {
                            k.v_sub_u32(dst, lv, rv);
                        }
                    }
                    BinOpKind::Mul => {
                        let lty = func.value_type(*lhs);
                        if lty.dtype().map(|d| d.is_float()).unwrap_or(false) {
                            k.v_mul_f32(dst, lv, rv);
                        } else {
                            k.v_mul_lo_u32(dst, lv, rv);
                        }
                    }
                    BinOpKind::Div => {
                        let tmp = k.alloc_vreg();
                        k.v_rcp_f32(tmp, rv);
                        k.v_mul_f32(dst, lv, tmp);
                    }
                    BinOpKind::Max => k.v_max_f32(dst, lv, rv),
                    BinOpKind::Min => k.v_min_f32(dst, lv, rv),
                    BinOpKind::And => k.v_and_b32(dst, lv, rv),
                    BinOpKind::Or  => k.v_or_b32(dst, Operand::VReg(lv), Operand::VReg(rv)),
                    BinOpKind::Xor => k.v_xor_b32(dst, Operand::VReg(lv), Operand::VReg(rv)),
                    BinOpKind::Shl => {
                        // Use pre-cached rhs type (before get_vreg overwrote val_map)
                        match rhs_val_precheck {
                            Some(MachineVal::InlineInt(shift)) => {
                                k.v_lshlrev_b32(dst, shift as u8, lv);
                            }
                            _ => {
                                // TODO: add VLshlrevB32VV Op for truly runtime VReg-VReg shifts.
                                // Current DSL only generates const shifts, so this is unreachable.
                                return Err(format!(
                                    "VReg-VReg Shl not yet supported (rhs type: {:?}). \
                                     All Shl ops should use const shift amounts.", rhs_val_precheck
                                ));
                            }
                        }
                    }
                    BinOpKind::Shr => {
                        // Use pre-cached rhs type (before get_vreg overwrote val_map)
                        match rhs_val_precheck {
                            Some(MachineVal::InlineInt(shift)) => {
                                k.v_lshrrev_b32(dst, shift as u8, lv);
                            }
                            _ => {
                                unreachable!("VReg-VReg Shr: should not reach here for const shifts");
                            }
                        }
                    }
                    _ => return Err(format!("Unimplemented binop: {:?}", bin_op)),
                }
                val_map.insert(*result, MachineVal::VReg(dst));
            }
        }

        // ── UnaryOp ──
        TileOp::UnaryOp { result, op: unary_op, src } => {
            let sv = get_vreg(k, val_map, *src)?;
            let dst = k.alloc_vreg();
            match unary_op {
                UnaryOpKind::Neg => {
                    // neg = xor with sign bit
                    k.v_xor_b32(dst, Operand::VReg(sv), Operand::Literal(0x80000000));
                }
                UnaryOpKind::Exp => {
                    // e^x = 2^(x * log2(e))
                    let log2e = k.alloc_vreg();
                    k.push(Op::VMov { dst: log2e, src: Operand::Literal(0x3FB8AA3B) }); // log2(e) ≈ 1.4427
                    let t = k.alloc_vreg();
                    k.v_mul_f32(t, sv, log2e);
                    k.v_exp_f32(dst, t);
                }
                UnaryOpKind::Log => {
                    // ln(x) = log2(x) * ln(2)
                    let t = k.alloc_vreg();
                    k.v_log_f32(t, sv);
                    let ln2 = k.alloc_vreg();
                    k.push(Op::VMov { dst: ln2, src: Operand::Literal(0x3F317218) }); // ln(2) ≈ 0.6931
                    k.v_mul_f32(dst, t, ln2);
                }
                UnaryOpKind::Sqrt => k.v_sqrt_f32(dst, sv),
                UnaryOpKind::Rcp => k.v_rcp_f32(dst, sv),
                UnaryOpKind::Rsqrt => k.v_rsq_f32(dst, sv),
                UnaryOpKind::Abs => {
                    // abs = clear sign bit
                    k.v_and_b32(dst, sv, VReg(0)); // need literal 0x7FFFFFFF
                    // Actually use proper approach:
                    k.push(Op::VAndB32 {
                        dst, src0: Operand::VReg(sv), src1: Operand::Literal(0x7FFFFFFF)
                    });
                }
                UnaryOpKind::Relu => {
                    // relu = max(0, x)
                    k.push(Op::VMaxF32 {
                        dst, src0: Operand::VReg(sv), src1: Operand::InlineFloat(0.0)
                    });
                }
                UnaryOpKind::Sigmoid => {
                    // sigmoid(x) = 1 / (1 + exp(-x))
                    let neg_x = k.alloc_vreg();
                    k.v_xor_b32(neg_x, Operand::VReg(sv), Operand::Literal(0x80000000));
                    let log2e = k.alloc_vreg();
                    k.push(Op::VMov { dst: log2e, src: Operand::Literal(0x3FB8AA3B) });
                    let t = k.alloc_vreg();
                    k.v_mul_f32(t, neg_x, log2e);
                    let exp_neg = k.alloc_vreg();
                    k.v_exp_f32(exp_neg, t);
                    let one_plus = k.alloc_vreg();
                    k.push(Op::VAddF32 {
                        dst: one_plus, src0: Operand::InlineFloat(1.0), src1: Operand::VReg(exp_neg)
                    });
                    k.v_rcp_f32(dst, one_plus);
                }
                UnaryOpKind::Silu => {
                    // silu(x) = x * sigmoid(x)
                    // First compute sigmoid
                    let neg_x = k.alloc_vreg();
                    k.v_xor_b32(neg_x, Operand::VReg(sv), Operand::Literal(0x80000000));
                    let log2e = k.alloc_vreg();
                    k.push(Op::VMov { dst: log2e, src: Operand::Literal(0x3FB8AA3B) });
                    let t = k.alloc_vreg();
                    k.v_mul_f32(t, neg_x, log2e);
                    let exp_neg = k.alloc_vreg();
                    k.v_exp_f32(exp_neg, t);
                    let one_plus = k.alloc_vreg();
                    k.push(Op::VAddF32 {
                        dst: one_plus, src0: Operand::InlineFloat(1.0), src1: Operand::VReg(exp_neg)
                    });
                    let sig = k.alloc_vreg();
                    k.v_rcp_f32(sig, one_plus);
                    k.v_mul_f32(dst, sv, sig);
                }
                UnaryOpKind::Sin => {
                    // v_sin_f32 computes sin(2π·x), so pre-multiply by 1/(2π)
                    let inv_2pi = k.alloc_vreg();
                    k.push(Op::VMov { dst: inv_2pi, src: Operand::Literal(0x3E22F983) }); // 1/(2π)
                    let scaled = k.alloc_vreg();
                    k.v_mul_f32(scaled, sv, inv_2pi);
                    k.v_sin_f32(dst, scaled);
                }
                UnaryOpKind::Cos => {
                    // v_cos_f32 computes cos(2π·x), so pre-multiply by 1/(2π)
                    let inv_2pi = k.alloc_vreg();
                    k.push(Op::VMov { dst: inv_2pi, src: Operand::Literal(0x3E22F983) }); // 1/(2π)
                    let scaled = k.alloc_vreg();
                    k.v_mul_f32(scaled, sv, inv_2pi);
                    k.v_cos_f32(dst, scaled);
                }
                UnaryOpKind::Exp2 => {
                    // Raw hardware 2^x — direct v_exp_f32 without log2(e) scaling
                    k.v_exp_f32(dst, sv);
                }
                UnaryOpKind::Log2 => {
                    // Raw hardware log₂(x) — direct v_log_f32 without ln(2) scaling
                    k.v_log_f32(dst, sv);
                }
                _ => return Err(format!("Unimplemented unary op: {:?}", unary_op)),
            }
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        // ── Fma ──
        TileOp::Fma { result, a, b, c } => {
            let va = get_vreg(k, val_map, *a)?;
            let vb = get_vreg(k, val_map, *b)?;
            let vc = get_vreg(k, val_map, *c)?;
            let dst = k.alloc_vreg();
            k.v_fma_f32(dst, va, vb, vc);
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        // ── Reduce (wave-level) ──
        TileOp::Reduce { result, src, axis: _, op: reduce_op } => {
            let sv = get_vreg(k, val_map, *src)?;
            let dst = k.alloc_vreg();
            let tmp = k.alloc_vreg();
            k.v_mov(dst, sv);
            match reduce_op {
                ReduceKind::Sum => k.wave_reduce_add_f32(dst, tmp),
                ReduceKind::Max => k.wave_reduce_max_f32(dst, tmp),
                _ => return Err(format!("Unimplemented reduce: {:?}", reduce_op)),
            }
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        // ── Reshape / ExpandDims — zero-cost aliases ──
        TileOp::Reshape { result, src, .. } |
        TileOp::ExpandDims { result, src, .. } => {
            // These don't change physical layout, just the type system.
            // Reuse the same machine register.
            let mv = val_map.get(src)
                .ok_or_else(|| format!("Reshape/ExpandDims: src {:?} not found", src))?
                .clone();
            val_map.insert(*result, mv);
        }

        // ── Cast ──
        TileOp::Cast { result, src, to } => {
            let sv = get_vreg(k, val_map, *src)?;
            let src_dtype = func.value_dtype(*src);
            let dst = k.alloc_vreg();
            match (src_dtype, to) {
                (Some(ScalarDType::U32), ScalarDType::F32) => k.v_cvt_f32_u32(dst, sv),
                (Some(ScalarDType::F32), ScalarDType::U32) => k.v_cvt_u32_f32(dst, sv),
                // F32 → BF16: truncate lower 16 bits (keep upper 16 as bf16)
                (Some(ScalarDType::F32), ScalarDType::BF16) => {
                    k.v_lshrrev_b32(dst, 16, sv);
                }
                // BF16 → F32: shift left 16 to restore f32 format
                (Some(ScalarDType::BF16), ScalarDType::F32) => {
                    k.v_lshlrev_b32(dst, 16, sv);
                }
                _ => {
                    // Fallback: just copy
                    k.v_mov(dst, sv);
                }
            }
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        // ── Barrier ──
        TileOp::Barrier => {
            k.barrier();
        }

        // ── 比较操作 ──
        TileOp::Cmp { result, op: cmp_op, lhs, rhs } => {
            // 判断操作数是标量还是向量
            let lhs_scalar = matches!(val_map.get(lhs), Some(MachineVal::SReg(_)) | Some(MachineVal::InlineInt(_)));
            let rhs_scalar = matches!(val_map.get(rhs), Some(MachineVal::SReg(_)) | Some(MachineVal::InlineInt(_)));

            if lhs_scalar && rhs_scalar {
                // 标量比较 → SCC
                let ls = get_sreg_or_inline(val_map, *lhs)?;
                let rs = get_sreg_or_inline(val_map, *rhs)?;
                match cmp_op {
                    CmpOpKind::Lt => {
                        match (ls, rs) {
                            (SOperand::SReg(a), SOperand::SReg(b)) =>
                                k.push(Op::SCmpLtU32 { src0: a, src1: b }),
                            (SOperand::SReg(a), SOperand::InlineInt(v)) => {
                                // s_cmp_lt_u32 doesn't support inline imm as src1
                                // Use: s_cmpk_lt_u32 or put imm in sreg
                                let tmp = k.alloc_sreg();
                                k.push(Op::SMov { dst: tmp, src: SOperand::InlineInt(v) });
                                k.push(Op::SCmpLtU32 { src0: a, src1: tmp });
                            }
                            _ => return Err("Scalar Cmp Lt: unsupported operand combo".into()),
                        }
                    }
                    _ => return Err(format!("Unimplemented scalar cmp: {:?}", cmp_op)),
                }
                // 标量比较结果存为 SReg（dummy），实际用 SCC
                let dummy = k.alloc_sreg();
                val_map.insert(*result, MachineVal::SReg(dummy));
            } else {
                // 向量比较 → VCC
                let lv = get_vreg(k, val_map, *lhs)?;
                let rv = get_vreg(k, val_map, *rhs)?;
                match cmp_op {
                    CmpOpKind::Lt => k.v_cmp_lt_u32(Operand::VReg(lv), Operand::VReg(rv)),
                    CmpOpKind::Ge => k.v_cmp_ge_u32(Operand::VReg(lv), Operand::VReg(rv)),
                    _ => return Err(format!("Unimplemented cmp: {:?}", cmp_op)),
                }
                // 物化 VCC boolean 到 VGPR: dst = VCC ? 1 : 0
                let dst = k.alloc_vreg();
                k.v_cndmask_b32(dst, Operand::InlineInt(0), Operand::InlineInt(1));
                val_map.insert(*result, MachineVal::VReg(dst));
            }
        }

        TileOp::Select { result, cond, true_val, false_val } => {
            let cond_v = get_vreg(k, val_map, *cond)?;
            let tv = get_vreg(k, val_map, *true_val)?;
            let fv = get_vreg(k, val_map, *false_val)?;
            let dst = k.alloc_vreg();
            // CRITICAL: Re-establish VCC from the materialized cond VGPR.
            // Cmp lowering stores cond as 1/0 in VGPR, but VCC may have been
            // clobbered by intervening v_add_co (64-bit address calculation).
            // Without this, Select reads stale VCC → wrong results.
            k.v_cmp_gt_u32_imm(cond_v, 0); // VCC = (cond > 0) per-lane
            k.v_cndmask_b32(dst, Operand::VReg(fv), Operand::VReg(tv));
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        // ── LDS（共享内存）──
        TileOp::LdsAlloc { result, size_bytes } => {
            // LDS alloc: 分配 LDS 空间，返回基地址 (byte offset constant)
            // 每次调用 lds_alloc 会递增 LDS usage
            let current_lds = k.lds_size();
            k.set_lds_size(current_lds + size_bytes);
            // 将基偏移存入 VGPR
            let dst = k.alloc_vreg();
            k.v_mov_imm(dst, current_lds as i32);
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        TileOp::LdsStore { base, offset, val } => {
            let base_v = get_vreg(k, val_map, *base)?;
            let off_v = get_vreg(k, val_map, *offset)?;
            let val_v = get_vreg(k, val_map, *val)?;
            // addr = base + offset * 4
            let addr = k.alloc_vreg();
            k.v_lshlrev_b32(addr, 2, off_v);
            k.v_add_u32(addr, addr, base_v);
            k.ds_store_b32(addr, val_v, 0);
            k.wait_lgkmcnt(0);
        }

        TileOp::LdsLoad { result, base, offset } => {
            let base_v = get_vreg(k, val_map, *base)?;
            let off_v = get_vreg(k, val_map, *offset)?;
            // addr = base + offset * 4
            let addr = k.alloc_vreg();
            k.v_lshlrev_b32(addr, 2, off_v);
            k.v_add_u32(addr, addr, base_v);
            let dst = k.alloc_vreg();
            k.ds_load_b32(dst, addr, 0);
            k.wait_lgkmcnt(0);
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        // ── Atomic ──
        TileOp::AtomicAddF32 { ptr, indices, val, mask } => {
            let val_v = get_vreg(k, val_map, *val)?;
            let idx_v = get_vreg(k, val_map, *indices)?;
            // byte offset = indices * 4
            let byte_off = k.alloc_vreg();
            k.v_lshlrev_b32(byte_off, 2, idx_v);
            // Build 64-bit address
            let addr = k.alloc_vreg_array(2, Alignment::Align2);
            let ptr_pair = get_ptr(val_map, *ptr)?;
            k.v_mov_from_sgpr(addr, SReg(ptr_pair.0));
            k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(ptr_pair.0 + 1));
            k.v_add_co(addr, addr, byte_off);
            k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

            // 如果有 mask，用 EXEC mask 保护
            let saved_exec = if let Some(mask_val) = mask {
                let mask_v = get_vreg(k, val_map, *mask_val)?;
                k.v_cmp_gt_u32_imm(mask_v, 0);
                let saved = k.alloc_sreg();
                k.save_exec(saved);
                Some(saved)
            } else {
                None
            };

            k.global_atomic_add_f32(addr, val_v, 0);
            k.wait_vmcnt(0);

            if let Some(saved) = saved_exec {
                k.restore_exec(saved);
            }
        }

        // ── WMMA / BF16 ──
        TileOp::ZeroAcc { result } => {
            let target = current_target();
            let format = WmmaFormat::BF16_F32;
            let acc_regs = format.dst_vreg_count(target);
            let acc = k.alloc_vreg_array(acc_regs, format.dst_alignment(target));
            for j in 0..acc_regs {
                k.v_mov_imm(VReg(acc.0 + j), 0);
            }
            // Register coalesced group: opt passes will preserve contiguity
            k.mark_coalesced_group(acc, acc_regs);
            val_map.insert(*result, MachineVal::VReg(acc));
        }

        TileOp::CvtPkBf16F32 { result, lo, hi } => {
            let v_lo = get_vreg(k, val_map, *lo)?;
            let v_hi = get_vreg(k, val_map, *hi)?;
            let dst = k.alloc_vreg();
            k.push(Op::CvtPkBf16F32 { dst, src0: v_lo, src1: v_hi });
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        TileOp::WmmaF32 { result, a, b, c } => {
            let va = get_vreg(k, val_map, *a)?;
            let vb = get_vreg(k, val_map, *b)?;
            let vc = get_vreg(k, val_map, *c)?;
            let target = current_target();
            let format = WmmaFormat::BF16_F32;
            let dst_regs = format.dst_vreg_count(target);
            let c_regs = format.c_vreg_count(target);
            // Allocate new destination with target-specific WMMA constraints,
            // then seed it from the incoming accumulator fragment.
            let dst = k.alloc_vreg_array(dst_regs, format.dst_alignment(target));
            for j in 0..c_regs {
                k.v_mov(VReg(dst.0 + j), VReg(vc.0 + j));
            }
            k.wmma_bf16_f32(dst, va, vb, dst);
            // Register coalesced group: opt passes will preserve contiguity
            k.mark_coalesced_group(dst, dst_regs);
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        TileOp::ExtractF32 { result, src, idx } => {
            let va = get_vreg(k, val_map, *src)?;
            let dst = k.alloc_vreg();
            k.v_mov(dst, VReg(va.0 + idx));
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        TileOp::SplatFragment { result, src } => {
            let v_val = get_vreg(k, val_map, *src)?;
            let target = current_target();
            let format = WmmaFormat::BF16_F32;
            let frag_regs = format.a_vreg_count(target);
            let dst = k.alloc_vreg_array(frag_regs, format.a_alignment(target));
            for j in 0..frag_regs {
                k.v_mov(VReg(dst.0 + j), v_val);
            }
            // Register coalesced group: opt passes will preserve contiguity
            k.mark_coalesced_group(dst, frag_regs);
            val_map.insert(*result, MachineVal::VReg(dst));
        }

        // ── WG 级归约 ──
        TileOp::WgReduceAdd { result, src, block_size } => {
            let va = get_vreg(k, val_map, *src)?;
            let n_waves = (block_size + 31) / 32;

            // Step 1: wave-level reduce
            let wave_sum = k.alloc_vreg();
            let tmp = k.alloc_vreg();
            k.v_mov(wave_sum, va);
            k.wave_reduce_add_f32(wave_sum, tmp);

            // Step 2: allocate LDS for partial sums
            let lds_base = k.lds_size();
            k.set_lds_size(lds_base + n_waves * 4);

            // Step 3: wave leader (lane 0) writes to LDS
            let wave_id = k.alloc_vreg();
            let tid = k.thread_id_x();
            k.v_lshrrev_b32(wave_id, 5, tid);

            let lds_addr = k.alloc_vreg();
            k.v_lshlrev_b32(lds_addr, 2, wave_id);
            if lds_base > 0 {
                let base_v = k.alloc_vreg();
                k.v_mov_imm(base_v, lds_base as i32);
                k.v_add_u32(lds_addr, lds_addr, base_v);
            }

            let lane_id = k.alloc_vreg();
            k.v_and_b32_imm(lane_id, tid, 31);
            k.v_cmp_eq_u32_imm(lane_id, 0);
            let saved_exec = k.alloc_sreg();
            k.save_exec(saved_exec);
            k.ds_store_b32(lds_addr, wave_sum, 0);
            k.restore_exec(saved_exec);

            // Step 4: barrier
            k.wait_lgkmcnt(0);
            k.s_barrier();

            // Step 5: load partial sums, reduce
            let partial = k.alloc_vreg();
            k.v_mov_imm(partial, 0);
            k.v_cmp_lt_u32(Operand::VReg(lane_id), Operand::InlineInt(n_waves as i32));
            let saved2 = k.alloc_sreg();
            k.save_exec(saved2);
            let load_addr = k.alloc_vreg();
            k.v_lshlrev_b32(load_addr, 2, lane_id);
            if lds_base > 0 {
                let base_v2 = k.alloc_vreg();
                k.v_mov_imm(base_v2, lds_base as i32);
                k.v_add_u32(load_addr, load_addr, base_v2);
            }
            k.ds_load_b32(partial, load_addr, 0);
            k.wait_lgkmcnt(0);
            k.restore_exec(saved2);

            // Step 6: wave reduce partial sums
            let tmp2 = k.alloc_vreg();
            k.wave_reduce_add_f32(partial, tmp2);
            val_map.insert(*result, MachineVal::VReg(partial));
        }

        TileOp::WgReduceMax { result, src, block_size } => {
            let va = get_vreg(k, val_map, *src)?;
            let n_waves = (block_size + 31) / 32;

            // Step 1: wave-level reduce max
            let wave_max = k.alloc_vreg();
            let tmp = k.alloc_vreg();
            k.v_mov(wave_max, va);
            k.wave_reduce_max_f32(wave_max, tmp);

            // Step 2: allocate LDS
            let lds_base = k.lds_size();
            k.set_lds_size(lds_base + n_waves * 4);

            // Step 3: wave leader writes to LDS
            let wave_id = k.alloc_vreg();
            let tid = k.thread_id_x();
            k.v_lshrrev_b32(wave_id, 5, tid);
            let lds_addr = k.alloc_vreg();
            k.v_lshlrev_b32(lds_addr, 2, wave_id);
            if lds_base > 0 {
                let base_v = k.alloc_vreg();
                k.v_mov_imm(base_v, lds_base as i32);
                k.v_add_u32(lds_addr, lds_addr, base_v);
            }

            let lane_id = k.alloc_vreg();
            k.v_and_b32_imm(lane_id, tid, 31);
            k.v_cmp_eq_u32_imm(lane_id, 0);
            let saved_exec = k.alloc_sreg();
            k.save_exec(saved_exec);
            k.ds_store_b32(lds_addr, wave_max, 0);
            k.restore_exec(saved_exec);

            // Step 4: barrier
            k.wait_lgkmcnt(0);
            k.s_barrier();

            // Step 5: load partial maxes, init with -inf
            let partial = k.alloc_vreg();
            k.v_mov_imm(partial, f32::NEG_INFINITY.to_bits() as i32);
            k.v_cmp_lt_u32(Operand::VReg(lane_id), Operand::InlineInt(n_waves as i32));
            let saved2 = k.alloc_sreg();
            k.save_exec(saved2);
            let load_addr = k.alloc_vreg();
            k.v_lshlrev_b32(load_addr, 2, lane_id);
            if lds_base > 0 {
                let base_v2 = k.alloc_vreg();
                k.v_mov_imm(base_v2, lds_base as i32);
                k.v_add_u32(load_addr, load_addr, base_v2);
            }
            k.ds_load_b32(partial, load_addr, 0);
            k.wait_lgkmcnt(0);
            k.restore_exec(saved2);

            // Step 6: wave reduce max on partial
            let tmp2 = k.alloc_vreg();
            k.wave_reduce_max_f32(partial, tmp2);
            val_map.insert(*result, MachineVal::VReg(partial));
        }

        // ── EXEC Mask（条件执行）──
        TileOp::ExecMaskPush { mask } => {
            // mask 是 bool vector (materialized as VGPR 0/1 by Cmp lowering)
            // Re-compare to set VCC, then save_exec: saved = EXEC; EXEC &= VCC
            let mask_v = get_vreg(k, val_map, *mask)?;
            k.v_cmp_gt_u32_imm(mask_v, 0);
            let saved = k.alloc_sreg();
            k.save_exec(saved);
            exec_save_stack.push(saved);
        }
        TileOp::ExecMaskFlip => {
            let saved = *exec_save_stack.last()
                .ok_or("ExecMaskFlip: no matching ExecMaskPush")?;
            k.push(Op::XorExec { saved });
        }
        TileOp::ExecMaskPop => {
            let saved = exec_save_stack.pop()
                .ok_or("ExecMaskPop: no matching ExecMaskPush")?;
            k.restore_exec(saved);
        }

        _ => return Err(format!("Unimplemented tile op: {:?}", op)),
    }

    Ok(())
}

/// 复制 SSA Value 到另一个 Value 的已分配寄存器（用于 block param 传递）
fn copy_val(
    k: &mut T0Kernel,
    val_map: &HashMap<Value, MachineVal>,
    src: Value,
    dst: Value,
) -> Result<(), String> {
    let src_val = val_map.get(&src)
        .ok_or_else(|| format!("copy_val: src {:?} not mapped", src))?.clone();
    let dst_val = val_map.get(&dst)
        .ok_or_else(|| format!("copy_val: dst {:?} not mapped", dst))?.clone();

    copy_machine_val(k, &src_val, &dst_val)
}

/// Copy arg to a block param, using the ORIGINAL param allocation from
/// block_param_original (not the potentially-polluted val_map).
/// This ensures loop IV SRegs are correctly written back.
fn copy_val_to_param(
    k: &mut T0Kernel,
    val_map: &HashMap<Value, MachineVal>,
    block_param_original: &HashMap<Value, MachineVal>,
    src: Value,
    dst_param: Value,
) -> Result<(), String> {
    let src_val = val_map.get(&src)
        .ok_or_else(|| format!("copy_val_to_param: src {:?} not mapped", src))?.clone();
    // Use the ORIGINAL block param allocation, not the possibly-polluted val_map entry
    let dst_val = block_param_original.get(&dst_param)
        .or_else(|| val_map.get(&dst_param))
        .ok_or_else(|| format!("copy_val_to_param: dst {:?} not mapped", dst_param))?.clone();

    copy_machine_val(k, &src_val, &dst_val)
}

/// Low-level copy between MachineVal registers.
fn copy_machine_val(
    k: &mut T0Kernel,
    src_val: &MachineVal,
    dst_val: &MachineVal,
) -> Result<(), String> {
    match (src_val, dst_val) {
        (MachineVal::SReg(s), MachineVal::SReg(d)) => {
            if s.0 != d.0 {
                k.push(Op::SMov { dst: *d, src: SOperand::SReg(*s) });
            }
        }
        (MachineVal::InlineInt(v), MachineVal::SReg(d)) => {
            k.push(Op::SMov { dst: *d, src: SOperand::InlineInt(*v) });
        }
        (MachineVal::VReg(s), MachineVal::VReg(d)) => {
            if s.0 != d.0 {
                k.push(Op::VMov { dst: *d, src: Operand::VReg(*s) });
            }
        }
        (MachineVal::SReg(s), MachineVal::VReg(d)) => {
            k.push(Op::VMovFromSgpr { dst: *d, src: *s });
        }
        (MachineVal::VReg(s), MachineVal::SReg(d)) => {
            // VReg→SReg: use v_readfirstlane to move lane 0 value to SGPR.
            // This is correct for scalar loop IVs that were promoted to VReg
            // by get_vreg() caching — all lanes have the same value.
            k.push(Op::RawAsm(format!("v_readfirstlane_b32 s{}, v{}", d.0, s.0)));
        }
        (MachineVal::InlineInt(v), MachineVal::VReg(d)) => {
            k.v_mov_imm(*d, *v);
        }
        _ => return Err(format!("copy_val: unsupported {:?} → {:?}", src_val, dst_val)),
    }
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

fn get_val(val_map: &HashMap<Value, MachineVal>, v: Value) -> Result<MachineVal, String> {
    val_map.get(&v).cloned().ok_or_else(|| format!("Value {} not yet lowered", v))
}

/// Get SOperand for a value that is known to be scalar
fn get_sreg_or_inline(val_map: &HashMap<Value, MachineVal>, v: Value) -> Result<SOperand, String> {
    match get_val(val_map, v)? {
        MachineVal::SReg(sr) => Ok(SOperand::SReg(sr)),
        MachineVal::InlineInt(i) => Ok(SOperand::InlineInt(i)),
        other => Err(format!("Expected scalar, got {:?}", other)),
    }
}

/// Get a VReg for a value, moving from SGPR to VGPR if needed
fn get_vreg(k: &mut T0Kernel, val_map: &mut HashMap<Value, MachineVal>, v: Value) -> Result<VReg, String> {
    match get_val(val_map, v)? {
        MachineVal::VReg(vr) => Ok(vr),
        MachineVal::SReg(sr) => {
            let vr = k.alloc_vreg();
            k.v_mov_from_sgpr(vr, sr);
            // IMPORTANT: Do NOT cache this promotion into val_map!
            // Caching overwrites the SReg identity of block params (e.g. loop IV),
            // causing subsequent scalar operations (like iv += step) to use the
            // VReg path instead of the SReg path, which breaks back-edge copies
            // (the copy targets the VReg instead of the SReg → IV never updates
            // → infinite loop → GPU hang).
            // val_map.insert(v, MachineVal::VReg(vr));  // REMOVED: caused GPU hang
            Ok(vr)
        }
        MachineVal::InlineInt(i) => {
            let vr = k.alloc_vreg();
            k.v_mov_imm(vr, i);
            val_map.insert(v, MachineVal::VReg(vr));
            Ok(vr)
        }
        MachineVal::InlineFloat(f) => {
            let vr = k.alloc_vreg();
            k.push(Op::VMov { dst: vr, src: Operand::InlineFloat(f) });
            val_map.insert(v, MachineVal::VReg(vr));
            Ok(vr)
        }
        MachineVal::SRegPair(_) => Err(format!("Cannot use pointer {} as VGPR operand", v)),
    }
}

fn get_ptr(val_map: &HashMap<Value, MachineVal>, v: Value) -> Result<SRegPair, String> {
    match get_val(val_map, v)? {
        MachineVal::SRegPair(p) => Ok(p),
        _ => Err(format!("Value {} is not a pointer", v)),
    }
}

// ============================================================================
// Tiled GEMM Lowering
//
// TileLoad2D + TileDot + TileStore2D → TileGemm spec → tile_ir::lower_gemm
// ============================================================================

/// SSA 分析结果：从 TileFunc 中提取的 GEMM 参数
#[derive(Debug)]
pub struct TiledGemmAnalysis {
    /// Tile M: 来自 TileLoad2D(A) 的 rows
    pub tile_m: u32,
    /// Tile N: 来自 TileLoad2D(B) 的 cols 或 rows  
    pub tile_n: u32,
    /// Tile K: 来自 TileLoad2D 的 cols (contraction dim)
    pub tile_k: u32,
    /// SSA arg index: X ptr
    pub x_ptr_arg: usize,
    /// SSA arg index: W ptr  
    pub w_ptr_arg: usize,
    /// SSA arg index: Y ptr
    pub y_ptr_arg: usize,
    /// SSA arg index: K dim (stride for X and W)
    pub k_dim_arg: usize,
    /// SSA arg index: N dim (stride for Y)
    pub n_dim_arg: usize,
}

/// 分析 TileFunc 中的 GEMM 模式：提取 tile 维度和参数映射
pub fn analyze_tiled_gemm(func: &TileFunc) -> Result<TiledGemmAnalysis, String> {
    let ops = func.all_ops();
    
    // 扫描所有 op 找到 TileLoad2D 和 TileDot
    let mut loads: Vec<(u32, u32, ScalarDType)> = Vec::new();  // (rows, cols, dtype)
    let mut has_dot = false;
    let mut has_store = false;
    
    for op in ops {
        match op {
            TileOp::TileLoad2D { rows, cols, dtype, .. } => {
                loads.push((*rows, *cols, *dtype));
            }
            TileOp::TileDot { .. } => { has_dot = true; }
            TileOp::TileStore2D { .. } => { has_store = true; }
            _ => {}
        }
    }
    
    if !has_dot {
        return Err("analyze_tiled_gemm: no TileDot found".into());
    }
    if !has_store {
        return Err("analyze_tiled_gemm: no TileStore2D found".into());
    }
    if loads.len() < 2 {
        return Err(format!("analyze_tiled_gemm: expected ≥2 TileLoad2D, found {}", loads.len()));
    }
    
    // 第一个 TileLoad2D = A [M, K], 第二个 = B [K, N] 或 [N, K]
    let (a_rows, a_cols, _) = loads[0];
    let (b_rows, b_cols, _) = loads[1];
    
    // Determine layout: check if K dims match
    let (tile_m, tile_k, tile_n) = if a_cols == b_rows {
        // NN layout: A[M,K] @ B[K,N]
        (a_rows, a_cols, b_cols)
    } else if a_cols == b_cols {
        // NT layout: A[M,K] @ B[N,K]^T  
        (a_rows, a_cols, b_rows)
    } else {
        return Err(format!(
            "analyze_tiled_gemm: K dim mismatch. A=[{},{}], B=[{},{}]",
            a_rows, a_cols, b_rows, b_cols
        ));
    };
    
    // 提取 arg 索引（假定标准布局：X, W, Y, K, N）
    let n_args = func.args.len();
    if n_args < 5 {
        return Err(format!("analyze_tiled_gemm: expected ≥5 args, got {}", n_args));
    }
    
    Ok(TiledGemmAnalysis {
        tile_m, tile_n, tile_k,
        x_ptr_arg: 0,
        w_ptr_arg: 1,
        y_ptr_arg: 2,
        k_dim_arg: 3,
        n_dim_arg: 4,
    })
}

/// Tiled GEMM lowering 结果
pub struct LoweredTiledGemm {
    /// 编译好的 T0Kernel
    pub kernel: T0Kernel,
    /// tile_ir 配置（含 tile sizes, LDS, WG 等）
    pub spec: tile_ir::TileGemm,
}

impl LoweredTiledGemm {
    /// 计算 grid 维度
    pub fn compute_grid(&self, m: u32, n: u32) -> [u32; 3] {
        let n_tiles_m = (m + self.spec.tile_m - 1) / self.spec.tile_m;
        let n_tiles_n = (n + self.spec.tile_n - 1) / self.spec.tile_n;
        if self.spec.swap_grid {
            [n_tiles_n * self.spec.wg_size(), n_tiles_m * self.spec.split_k, 1]
        } else {
            [n_tiles_m * self.spec.wg_size(), n_tiles_n * self.spec.split_k, 1]
        }
    }
    
    /// 构建 40-byte kernarg buffer
    /// Layout: [X:u64, WT:u64, Y:u64, K:u32, N:u32, split_k_shift:u32, y_split_stride:u32]
    pub fn build_kernargs(
        &self, x_addr: u64, wt_addr: u64, y_addr: u64,
        _m: u32, k: u32, n: u32,
    ) -> [u8; 40] {
        // split_k always 1 for tiled GEMM (no y_stride offset)
        let mut ka = [0u8; 40];
        ka[0..8].copy_from_slice(&x_addr.to_le_bytes());
        ka[8..16].copy_from_slice(&wt_addr.to_le_bytes());
        ka[16..24].copy_from_slice(&y_addr.to_le_bytes());
        ka[24..28].copy_from_slice(&k.to_le_bytes());
        ka[28..32].copy_from_slice(&n.to_le_bytes());
        ka[32..36].copy_from_slice(&0u32.to_le_bytes());  // split_k_shift = 0
        ka[36..40].copy_from_slice(&0u32.to_le_bytes());  // y_split_stride = 0
        ka
    }
    
    /// WG 大小
    pub fn wg_size(&self) -> u32 { self.spec.wg_size() }
    
    /// LDS 大小
    pub fn lds_size(&self) -> u32 { self.spec.lds_total() }
}

/// Lower a TileFunc containing TileLoad2D/TileDot/TileStore2D into a GEMM kernel.
///
/// 工作流程：
/// 1. 分析 SSA 图，提取 tile 维度 (M, N, K)
/// 2. 构建 tile_ir::TileGemm 规格
/// 3. 委托 tile_ir::lower_gemm() 生成 ISA 级内核
///
/// # Example
/// ```ignore
/// let mut f = TileFunc::new("my_gemm");
/// // ... build with tile_load_2d, tile_dot, tile_store_2d ...
/// let result = lower_tiled_gemm(&f)?;
/// let elf = result.kernel.compile(Target::GFX1100)?;
/// ```
pub fn lower_tiled_gemm(func: &TileFunc) -> Result<LoweredTiledGemm, String> {
    lower_tiled_gemm_for_target(current_target(), func)
}

pub fn lower_tiled_gemm_for_target(target: Target, func: &TileFunc) -> Result<LoweredTiledGemm, String> {
    // Step 1: 分析 SSA 图
    let analysis = analyze_tiled_gemm(func)?;
    
    eprintln!("[lower_tiled_gemm] analyzed: tile_m={}, tile_n={}, tile_k={}",
        analysis.tile_m, analysis.tile_n, analysis.tile_k);
    
    // Step 2: 构建 TileGemm spec directly from analyzed dimensions.
    // We do NOT use tile_auto_select() because it ignores user-specified
    // tile sizes (e.g., mapping tile_m=64 to 32x64 which is wrong).
    let spec = tile_ir::TileGemm {
        tile_m: analysis.tile_m,
        tile_n: analysis.tile_n,
        tile_k: analysis.tile_k,
        double_buffer: true,
        wgp_mode: true,
        split_k: 1,
        swap_grid: true,
        transpose: tile_ir::TileTranspose::NT,
        acc_swap: false,
        epilogue: vec![],
    };
    
    eprintln!("[lower_tiled_gemm] spec: {} (wg={}, lds={}, db={})",
        spec.name(), spec.wg_size(), spec.lds_total(), spec.double_buffer);
    
    // Step 3: 生成内核
    let kernel = with_target_context(target, || tile_ir::lower_gemm(&spec));
    
    Ok(LoweredTiledGemm { kernel, spec })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_vector_add() {
        // Build vector_add in SSA
        let mut f = TileFunc::new("vector_add");
        let x_ptr = f.arg_ptr("x_ptr");
        let y_ptr = f.arg_ptr("y_ptr");
        let out_ptr = f.arg_ptr("out_ptr");
        let _n = f.arg_u32("n");

        let pid = f.program_id(0);
        let c128 = f.const_u32(128);
        let base = f.mul(pid, c128);
        let offs = f.arange(0, 128);
        let base_v = f.splat(base, 128);
        let idx = f.add(base_v, offs);

        let x = f.load(x_ptr, idx, ScalarDType::F32);
        let y = f.load(y_ptr, idx, ScalarDType::F32);
        let out = f.add(x, y);
        f.store(out_ptr, idx, out);
        f.return_();

        // Lower to T0Kernel
        let result = lower_elementwise_1d(&f, 128, 1);
        assert!(result.is_ok(), "Lowering failed: {:?}", result.err());

        let lowered = result.unwrap();
        assert_eq!(lowered.wg_size, 128);

        // Verify it compiles to ELF
        let elf = lowered.kernel.compile(Target::GFX1100);
        assert!(elf.is_ok(), "Compilation failed: {:?}", elf.err());
        let elf_bytes = elf.unwrap();
        assert!(elf_bytes.len() > 100, "ELF too small: {} bytes", elf_bytes.len());

        eprintln!("✓ vector_add: SSA → T0Kernel → ELF ({} bytes)", elf_bytes.len());
    }

    #[test]
    fn test_dump_vadd_asm() {
        // Dump the same kernel as the GPU test for analysis
        let mut f = TileFunc::new("vadd_e2e");
        let x_ptr = f.arg_ptr("x");
        let y_ptr = f.arg_ptr("y");
        let out_ptr = f.arg_ptr("out");
        let offs = f.arange(0, 128);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let y = f.load(y_ptr, offs, ScalarDType::F32);
        let out = f.add(x, y);
        f.store(out_ptr, offs, out);
        f.return_();

        let lowered = lower_elementwise_1d(&f, 128, 1).unwrap();
        let asm = lowered.kernel.to_assembly(Target::GFX1100).unwrap();
        std::fs::write("/tmp/vadd_asm.txt", &asm).unwrap();
        eprintln!("✓ ASM written to /tmp/vadd_asm.txt ({} bytes, kernarg={})",
                  asm.len(), lowered.kernel.kernarg_size());
    }

    #[test]
    fn test_lower_scale_kernel() {
        // y = x * 2.0
        let mut f = TileFunc::new("scale");
        let x_ptr = f.arg_ptr("x_ptr");
        let y_ptr = f.arg_ptr("y_ptr");
        let _n = f.arg_u32("n");

        let offs = f.arange(0, 64);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let two = f.const_f32(2.0);
        let two_v = f.splat(two, 64);
        let y = f.mul(x, two_v);
        f.store(y_ptr, offs, y);
        f.return_();

        let result = lower_elementwise_1d(&f, 64, 1);
        assert!(result.is_ok(), "Lowering failed: {:?}", result.err());

        let elf = result.unwrap().kernel.compile(Target::GFX1100);
        assert!(elf.is_ok(), "Compilation failed: {:?}", elf.err());
        eprintln!("✓ scale: SSA → ELF ({} bytes)", elf.unwrap().len());
    }

    #[test]
    fn test_lower_silu_kernel() {
        // y = silu(x) = x * sigmoid(x)
        let mut f = TileFunc::new("silu_kernel");
        let x_ptr = f.arg_ptr("x");
        let y_ptr = f.arg_ptr("y");
        let _n = f.arg_u32("n");

        let offs = f.arange(0, 128);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let y = f.silu(x);
        f.store(y_ptr, offs, y);
        f.return_();

        let result = lower_elementwise_1d(&f, 128, 1);
        assert!(result.is_ok(), "Lowering failed: {:?}", result.err());

        let elf = result.unwrap().kernel.compile(Target::GFX1100);
        assert!(elf.is_ok(), "Compilation failed: {:?}", elf.err());
        eprintln!("✓ silu: SSA → ELF ({} bytes)", elf.unwrap().len());
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_vector_add_gpu() {
        // End-to-end: out[i] = x[i] + y[i]
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let mut f = TileFunc::new("vadd_e2e");
        let x_ptr = f.arg_ptr("x");
        let y_ptr = f.arg_ptr("y");
        let out_ptr = f.arg_ptr("out");

        let offs = f.arange(0, 128);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let y = f.load(y_ptr, offs, ScalarDType::F32);
        let out = f.add(x, y);
        f.store(out_ptr, offs, out);
        f.return_();

        let lowered = lower_elementwise_1d(&f, 128, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let n = 128u32;
        let x_buf = device.alloc_vram(n as usize * 4).unwrap();
        let y_buf = device.alloc_vram(n as usize * 4).unwrap();
        let out_buf = device.alloc_vram(n as usize * 4).unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
        x_buf.write(unsafe { std::slice::from_raw_parts(x_data.as_ptr() as *const u8, n as usize * 4) });
        y_buf.write(unsafe { std::slice::from_raw_parts(y_data.as_ptr() as *const u8, n as usize * 4) });
        out_buf.write(&vec![0u8; n as usize * 4]);

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [lowered.wg_size, 1, 1],
            lds_size: 0,
        }).unwrap();

        let mut ka = [0u8; 24];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());

        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [128, 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n as usize];
        unsafe {
            out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n as usize * 4));
        }

        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            let expected = x_data[i] + y_data[i];
            let err = (result[i] - expected).abs();
            max_err = max_err.max(err);
            if err > 1e-5 {
                panic!("Mismatch at [{}]: got {} expected {} err {}", i, result[i], expected, err);
            }
        }
        eprintln!("✓ vector_add GPU PASSED (max_err={:.2e}, {} elements)", max_err, n);
    }

    /// GPU 端到端测试辅助: y = f(x)
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn run_unary_gpu_test(
        name: &str,
        build_fn: fn(&mut TileFunc, Value) -> Value,
        cpu_fn: fn(f32) -> f32,
        input: &[f32],
        tol: f32,
    ) {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let n = input.len() as u32;
        let mut f = TileFunc::new(name);
        let x_ptr = f.arg_ptr("x");
        let y_ptr = f.arg_ptr("y");
        let offs = f.arange(0, n);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let y = build_fn(&mut f, x);
        f.store(y_ptr, offs, y);
        f.return_();

        let lowered = lower_elementwise_1d(&f, n, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();
        let x_buf = device.alloc_vram(n as usize * 4).unwrap();
        let y_buf = device.alloc_vram(n as usize * 4).unwrap();
        x_buf.write(unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, n as usize * 4) });
        y_buf.write(&vec![0u8; n as usize * 4]);

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [n, 1, 1], lds_size: 0,
        }).unwrap();
        let mut ka = [0u8; 16];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [n, 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n as usize];
        unsafe { y_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n as usize * 4)); }

        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            let expected = cpu_fn(input[i]);
            let err = (result[i] - expected).abs();
            let rel = if expected.abs() > 1e-6 { err / expected.abs() } else { err };
            max_err = max_err.max(rel);
            if rel > tol {
                panic!("{} [{}]: gpu={:.6} cpu={:.6} rel_err={:.2e}", name, i, result[i], expected, rel);
            }
        }
        eprintln!("✓ {} GPU PASSED (max_rel_err={:.2e}, {} elems)", name, max_err, n);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_silu_gpu() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        run_unary_gpu_test("silu", |f, x| f.silu(x), |x| x / (1.0 + (-x).exp()), &input, 1e-3);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_exp_gpu() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        run_unary_gpu_test("exp", |f, x| f.exp(x), |x| x.exp(), &input, 1e-3);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_relu_gpu() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        run_unary_gpu_test("relu", |f, x| f.relu(x), |x| if x > 0.0 { x } else { 0.0 }, &input, 1e-6);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_multi_wg_vadd_gpu() {
        // 1024 elements, WG_SIZE=128, 8 WGs → tests program_id lowering
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let wg_size = 128u32;
        let n_total = 1024u32;

        let mut f = TileFunc::new("vadd_mwg");
        let x_ptr = f.arg_ptr("x");
        let y_ptr = f.arg_ptr("y");
        let out_ptr = f.arg_ptr("out");

        // idx = program_id(0) * WG_SIZE + arange(0, WG_SIZE)
        let pid = f.program_id(0);
        let block_sz = f.const_u32(wg_size);
        let block_off = f.mul(pid, block_sz);  // SGPR * SGPR → SGPR
        let lane_off = f.arange(0, wg_size);
        let block_off_v = f.splat(block_off, wg_size);
        let idx = f.add(block_off_v, lane_off);

        let x = f.load(x_ptr, idx, ScalarDType::F32);
        let y = f.load(y_ptr, idx, ScalarDType::F32);
        let out = f.add(x, y);
        f.store(out_ptr, idx, out);
        f.return_();

        let lowered = lower_elementwise_1d(&f, wg_size, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let x_buf = device.alloc_vram(n_total as usize * 4).unwrap();
        let y_buf = device.alloc_vram(n_total as usize * 4).unwrap();
        let out_buf = device.alloc_vram(n_total as usize * 4).unwrap();

        let x_data: Vec<f32> = (0..n_total).map(|i| i as f32 * 0.01).collect();
        let y_data: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32 * 0.01).collect();
        x_buf.write(unsafe { std::slice::from_raw_parts(x_data.as_ptr() as *const u8, n_total as usize * 4) });
        y_buf.write(unsafe { std::slice::from_raw_parts(y_data.as_ptr() as *const u8, n_total as usize * 4) });
        out_buf.write(&vec![0u8; n_total as usize * 4]);

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [wg_size, 1, 1],
            lds_size: 0,
        }).unwrap();

        // kernargs: [x_ptr:u64, y_ptr:u64, out_ptr:u64]
        let mut ka = [0u8; 24];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());

        let ka_buf = pool.write_kernargs(0, &ka);
        // Grid = n_total threads (8 WGs × 128 threads)
        queue.submit(&gk, [n_total, 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n_total as usize];
        unsafe {
            out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n_total as usize * 4));
        }

        let mut max_err: f32 = 0.0;
        for i in 0..n_total as usize {
            let expected = x_data[i] + y_data[i];
            let err = (result[i] - expected).abs();
            max_err = max_err.max(err);
            if err > 1e-4 {
                panic!("Multi-WG mismatch at [{}]: got {} expected {} err {}", i, result[i], expected, err);
            }
        }
        eprintln!("✓ multi_wg_vadd GPU PASSED (max_err={:.2e}, {} elements, {} WGs)",
            max_err, n_total, n_total / wg_size);
    }

    /// GPU 端到端测试：for_range 循环
    /// 每个线程计算 sum = 0 + 1 + 2 + 3 = 6, 然后 out[tid] = float(sum)
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_for_range_gpu() {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let n = 64u32;
        let loop_count = 4u32;

        let mut f = TileFunc::new("forloop_e2e");
        let out_ptr = f.arg_ptr("out");

        // acc = 0
        let zero = f.const_u32(0);

        // for i in [0, loop_count):
        //   acc = acc + i
        let lp = f.for_range_with_acc(0, loop_count, zero,
            TileType::Scalar(ScalarDType::U32));
        let new_acc = f.add(lp.acc, lp.iv);
        f.end_for_acc(&lp, new_acc);

        // 在 exit block: lp.result = 最终 acc (应该 = 0+1+2+3 = 6)
        // out[tid] = float(lp.result)
        let offs = f.arange(0, n);
        let result_v = f.splat(lp.result, n);
        // Cast u32 → f32
        let result_f = f.cast(result_v, ScalarDType::F32);
        f.store(out_ptr, offs, result_f);
        f.return_();

        let lowered = lower_elementwise_1d(&f, n, 1).unwrap();

        // Dump asm for debugging
        let asm = lowered.kernel.to_assembly(Target::GFX1100).unwrap();
        eprintln!("=== forloop asm ===\n{}\n===", asm);

        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let out_buf = device.alloc_vram(n as usize * 4).unwrap();
        out_buf.write(&vec![0u8; n as usize * 4]);

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [n, 1, 1],
            lds_size: 0,
        }).unwrap();

        let mut ka = [0u8; 8];
        ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [n, 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n as usize];
        unsafe {
            out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n as usize * 4));
        }

        let expected = (0..loop_count).sum::<u32>() as f32; // 0+1+2+3 = 6.0
        eprintln!("Output[0..8]: {:?}", &result[..8]);
        for i in 0..n as usize {
            if (result[i] - expected).abs() > 1e-6 {
                panic!("for_range [{}]: got {} expected {}", i, result[i], expected);
            }
        }
        eprintln!("✓ for_range GPU PASSED ({} loops × {} threads, all = {})", loop_count, n, expected);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_sin_gpu() {
        // sin(x) for x in [-π..π]
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        run_unary_gpu_test("sin", |f, x| f.sin(x), |x| x.sin(), &input, 2e-3);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_cos_gpu() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        run_unary_gpu_test("cos", |f, x| f.cos(x), |x| x.cos(), &input, 2e-3);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_lower_cast_bf16_gpu() {
        // Test F32 → BF16 → F32 roundtrip
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let n = 64u32;
        let mut f = TileFunc::new("cast_bf16_rt");
        let x_ptr = f.arg_ptr("x");
        let y_ptr = f.arg_ptr("y");
        let offs = f.arange(0, n);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let bf = f.cast(x, ScalarDType::BF16);
        let y = f.cast(bf, ScalarDType::F32);
        f.store(y_ptr, offs, y);
        f.return_();

        let lowered = lower_elementwise_1d(&f, n, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();
        let x_buf = device.alloc_vram(n as usize * 4).unwrap();
        let y_buf = device.alloc_vram(n as usize * 4).unwrap();

        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.1 + 1.0).collect();
        x_buf.write(unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, n as usize * 4) });
        y_buf.write(&vec![0u8; n as usize * 4]);

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [n, 1, 1], lds_size: 0,
        }).unwrap();
        let mut ka = [0u8; 16];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [n, 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        let mut result = vec![0f32; n as usize];
        unsafe { y_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n as usize * 4)); }

        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            // BF16 roundtrip: truncate lower 16 mantissa bits
            let bits = input[i].to_bits();
            let bf16_bits = bits & 0xFFFF0000;
            let expected = f32::from_bits(bf16_bits);
            let err = (result[i] - expected).abs();
            max_err = max_err.max(err);
            if err > 1e-6 {
                panic!("cast_bf16 [{}]: gpu={:.6} expected={:.6} (input={:.6}) err={:.2e}",
                    i, result[i], expected, input[i], err);
            }
        }
        eprintln!("✓ cast_bf16 GPU PASSED (max_err={:.2e}, {} elems)", max_err, n);
    }

    // ════════════════════════════════════════════
    //  Dot (GEMM) tests
    // ════════════════════════════════════════════

    #[test]
    fn test_lower_dot_compile() {
        // Verify lower_dot produces a valid T0Kernel that compiles to ELF
        let m = 128u32;
        let k = 128u32;
        let n = 128u32;

        let dot = lower_dot(m, k, n).expect("lower_dot failed");

        // Check config sanity
        assert!(dot.wg_size() >= 32, "wg_size too small: {}", dot.wg_size());
        assert!(dot.lds_size() > 0, "LDS should be non-zero for double-buffered GEMM");

        // Check grid computation
        let grid = dot.compute_grid(m, n);
        assert!(grid[0] > 0, "grid_x must be > 0");
        assert!(grid[1] > 0, "grid_y must be > 0");

        // Check kernarg builder
        let ka = dot.build_kernargs(0x1000, 0x2000, 0x3000, m, k, n);
        assert_eq!(ka.len(), 40);
        // X addr at offset 0
        assert_eq!(u64::from_le_bytes(ka[0..8].try_into().unwrap()), 0x1000);
        // WT addr at offset 8
        assert_eq!(u64::from_le_bytes(ka[8..16].try_into().unwrap()), 0x2000);

        // Compile to ELF
        let elf = dot.kernel.compile(Target::GFX1100);
        assert!(elf.is_ok(), "GEMM ELF compilation failed: {:?}", elf.err());
        let elf_bytes = elf.unwrap();
        assert!(elf_bytes.len() > 500, "GEMM ELF too small: {} bytes", elf_bytes.len());

        eprintln!("✓ lower_dot compile PASSED: config={}, ELF={} bytes, grid={:?}, wg={}, lds={}",
                  dot.config.name(), elf_bytes.len(), grid, dot.wg_size(), dot.lds_size());
    }

    /// GPU correctness test for lower_dot GEMM.
    ///
    /// Run with: timeout 15 cargo test --release --features rocm --lib -- test_lower_dot_gpu --ignored --nocapture --test-threads=1
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    #[ignore] // GPU GEMM may hang on misconfigured kernels — run manually with timeout
    fn test_lower_dot_gpu() {
        // GEMM correctness: Y[M,N] = A[M,K] @ B[N,K]^T  (NT layout, bf16 in, f32 out)
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::gpu_backend::{GpuKernel, KernelLoadConfig};

        let m = 128u32;
        let k_dim = 128u32;
        let n = 128u32;

        // ── Generate GEMM kernel ──
        let dot = lower_dot(m, k_dim, n).expect("lower_dot failed");
        eprintln!("[dot_gpu] config={}, wg={}, lds={}", dot.config.name(), dot.wg_size(), dot.lds_size());
        let elf = dot.kernel.compile(Target::GFX1100).expect("compile failed");
        eprintln!("[dot_gpu] ELF compiled: {} bytes", elf.len());

        // ── Prepare bf16 input data ──
        let a_f32: Vec<f32> = (0..m * k_dim)
            .map(|i| ((i as f32) * 0.01 - 0.5))
            .collect();
        let b_f32: Vec<f32> = (0..n * k_dim)
            .map(|i| ((i as f32) * 0.005 + 0.1))
            .collect();
        let a_bf16: Vec<u16> = a_f32.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let b_bf16: Vec<u16> = b_f32.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

        // CPU reference: Y[i,j] = Σ_k A_bf16[i,k] * B_bf16[j,k] (NT layout)
        let mut y_ref = vec![0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0f64;
                for kk in 0..k_dim as usize {
                    let a_val = f32::from_bits((a_bf16[i * k_dim as usize + kk] as u32) << 16);
                    let b_val = f32::from_bits((b_bf16[j * k_dim as usize + kk] as u32) << 16);
                    sum += (a_val as f64) * (b_val as f64);
                }
                y_ref[i * n as usize + j] = sum as f32;
            }
        }

        // ── GPU dispatch via GpuRuntime ──
        let rt = GpuRuntime::new().expect("GpuRuntime::new failed");
        eprintln!("[dot_gpu] GpuRuntime created");

        // Allocate bf16 buffers with 512-byte padding for safety
        let a_bytes = ((m * k_dim * 2) as usize + 511) & !511;
        let b_bytes = ((n * k_dim * 2) as usize + 511) & !511;
        let y_bytes = ((m * n * 4) as usize + 511) & !511;

        let a_buf = rt.alloc_zero(a_bytes).unwrap();
        let b_buf = rt.alloc_zero(b_bytes).unwrap();
        let y_buf = rt.alloc_zero(y_bytes).unwrap();

        // Upload bf16 data
        let a_data_bytes = (m * k_dim * 2) as usize;
        let b_data_bytes = (n * k_dim * 2) as usize;
        a_buf.write(unsafe { std::slice::from_raw_parts(a_bf16.as_ptr() as *const u8, a_data_bytes) });
        b_buf.write(unsafe { std::slice::from_raw_parts(b_bf16.as_ptr() as *const u8, b_data_bytes) });

        // Load kernel
        let gk = GpuKernel::load(&rt.device, &elf, &KernelLoadConfig {
            workgroup_size: [dot.wg_size(), 1, 1],
            lds_size: dot.lds_size(),
        }).unwrap();

        // Build kernargs and compute grid
        let grid = dot.compute_grid(m, n);
        let ka = dot.build_kernargs(
            a_buf.gpu_addr(), b_buf.gpu_addr(), y_buf.gpu_addr(),
            m, k_dim, n,
        );
        eprintln!("[dot_gpu] grid={:?}, ka={} bytes, addrs: A={:#x} B={:#x} Y={:#x}",
            grid, ka.len(), a_buf.gpu_addr(), b_buf.gpu_addr(), y_buf.gpu_addr());
        eprintln!("[dot_gpu] dispatching...");

        // Synchronous dispatch
        rt.dispatch(&gk, grid, &ka).unwrap();
        eprintln!("[dot_gpu] dispatch complete, reading results...");

        // ── Read back and verify ──
        let y_gpu = rt.read_f32(&y_buf, (m * n) as usize);

        let mut max_err: f32 = 0.0;
        let mut max_rel_err: f32 = 0.0;
        for i in 0..(m * n) as usize {
            let err = (y_gpu[i] - y_ref[i]).abs();
            let rel = if y_ref[i].abs() > 1e-6 { err / y_ref[i].abs() } else { err };
            max_err = max_err.max(err);
            max_rel_err = max_rel_err.max(rel);
        }

        eprintln!("GEMM {}×{}×{}: max_abs_err={:.2e}, max_rel_err={:.2e}", m, k_dim, n, max_err, max_rel_err);
        eprintln!("  Y_ref[0..4] = {:?}", &y_ref[..4]);
        eprintln!("  Y_gpu[0..4] = {:?}", &y_gpu[..4]);

        assert!(max_rel_err < 0.05,
            "GEMM too inaccurate: max_rel_err={:.4e} (threshold 5e-2)", max_rel_err);
        eprintln!("✓ dot GPU PASSED ({}×{}×{}, max_rel_err={:.2e})", m, k_dim, n, max_rel_err);
    }

    #[test]
    fn test_lower_tiled_gemm() {
        // Build a GEMM kernel using Triton-style tile ops
        let mut f = TileFunc::new("tiled_gemm_128x64");
        let x_ptr = f.arg_ptr("X");   // [M, K] bf16
        let w_ptr = f.arg_ptr("W");   // [N, K] bf16 (NT layout)
        let y_ptr = f.arg_ptr("Y");   // [M, N] f32
        let k_dim = f.arg_u32("K");
        let n_dim = f.arg_u32("N");

        let pid_m = f.program_id(0);
        let pid_n = f.program_id(1);
        let c128 = f.const_u32(128);
        let c64 = f.const_u32(64);
        let row_off = f.mul(pid_m, c128);
        let col_off = f.mul(pid_n, c64);

        // acc = zeros(128, 64)
        let acc = f.tile_zeros(128, 64);

        // Single K-step (for analysis only — lowering generates the full loop)
        let k0 = f.const_u32(0);
        let a = f.tile_load_2d(x_ptr, row_off, k0, k_dim, 128, 16, ScalarDType::BF16);
        let b = f.tile_load_2d(w_ptr, col_off, k0, k_dim, 16, 64, ScalarDType::BF16);
        let acc2 = f.tile_dot(a, b, acc);

        f.tile_store_2d(y_ptr, row_off, col_off, n_dim, acc2);
        f.return_();

        // Lower to kernel
        let result = lower_tiled_gemm(&f);
        assert!(result.is_ok(), "lower_tiled_gemm failed: {:?}", result.err());

        let lowered = result.unwrap();
        eprintln!("[test] spec: {} (wg={}, lds={})",
            lowered.spec.name(), lowered.wg_size(), lowered.lds_size());

        // Compile to ELF
        let elf = lowered.kernel.compile(Target::GFX1100);
        assert!(elf.is_ok(), "ELF compilation failed: {:?}", elf.err());
        let elf_bytes = elf.unwrap();
        assert!(elf_bytes.len() > 100, "ELF too small: {} bytes", elf_bytes.len());

        eprintln!("✓ tiled_gemm: SSA → TileGemm → T0Kernel → ELF ({} bytes)", elf_bytes.len());
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    #[ignore]  // run with: cargo test -- test_lower_tiled_gemm_gpu --ignored --nocapture --test-threads=1
    fn test_lower_tiled_gemm_gpu() {
        // End-to-end GPU test: 128×128×128 GEMM via tiled SSA path
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let m = 128u32;
        let k_dim = 128u32;
        let n = 64u32;

        // Build GEMM SSA
        let mut f = TileFunc::new("tiled_gemm_e2e");
        let x_ptr = f.arg_ptr("X");
        let w_ptr = f.arg_ptr("W");
        let y_ptr = f.arg_ptr("Y");
        let _k_arg = f.arg_u32("K");
        let _n_arg = f.arg_u32("N");

        let pid_m = f.program_id(0);
        let pid_n = f.program_id(1);
        let c128 = f.const_u32(128);
        let c64 = f.const_u32(64);
        let row_off = f.mul(pid_m, c128);
        let col_off = f.mul(pid_n, c64);

        let acc = f.tile_zeros(128, 64);
        let k0 = f.const_u32(0);
        let a = f.tile_load_2d(x_ptr, row_off, k0, _k_arg, 128, 16, ScalarDType::BF16);
        let b = f.tile_load_2d(w_ptr, col_off, k0, _k_arg, 16, 64, ScalarDType::BF16);
        let acc2 = f.tile_dot(a, b, acc);
        f.tile_store_2d(y_ptr, row_off, col_off, _n_arg, acc2);
        f.return_();

        // Lower
        let lowered = lower_tiled_gemm(&f).expect("lower_tiled_gemm failed");
        let elf = lowered.kernel.compile(Target::GFX1100).expect("compile failed");
        eprintln!("[tiled_gemm_gpu] spec={}, wg={}, lds={}, ELF={}B",
            lowered.spec.name(), lowered.wg_size(), lowered.lds_size(), elf.len());

        // Prepare bf16 test data
        let a_f32: Vec<f32> = (0..m * k_dim)
            .map(|i| ((i as f32) * 0.01 - 0.5))
            .collect();
        let b_f32: Vec<f32> = (0..n * k_dim)
            .map(|i| ((i as f32) * 0.005 + 0.1))
            .collect();
        let a_bf16: Vec<u16> = a_f32.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let b_bf16: Vec<u16> = b_f32.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

        // CPU reference: Y[i,j] = Σ_k A_bf16[i,k] * B_bf16[j,k] (NT layout)
        let mut y_ref = vec![0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0f64;
                for kk in 0..k_dim as usize {
                    let a_val = f32::from_bits((a_bf16[i * k_dim as usize + kk] as u32) << 16);
                    let b_val = f32::from_bits((b_bf16[j * k_dim as usize + kk] as u32) << 16);
                    sum += (a_val as f64) * (b_val as f64);
                }
                y_ref[i * n as usize + j] = sum as f32;
            }
        }

        // GPU dispatch
        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let a_bytes = ((m * k_dim * 2) as usize + 511) & !511;
        let b_bytes = ((n * k_dim * 2) as usize + 511) & !511;
        let y_bytes = ((m * n * 4) as usize + 511) & !511;

        let a_buf = device.alloc_vram(a_bytes).unwrap();
        let b_buf = device.alloc_vram(b_bytes).unwrap();
        let y_buf = device.alloc_vram(y_bytes).unwrap();

        a_buf.write(unsafe { std::slice::from_raw_parts(a_bf16.as_ptr() as *const u8, (m * k_dim * 2) as usize) });
        b_buf.write(unsafe { std::slice::from_raw_parts(b_bf16.as_ptr() as *const u8, (n * k_dim * 2) as usize) });

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [lowered.wg_size(), 1, 1],
            lds_size: lowered.lds_size(),
        }).unwrap();

        let grid = lowered.compute_grid(m, n);
        let ka = lowered.build_kernargs(
            a_buf.gpu_addr(), b_buf.gpu_addr(), y_buf.gpu_addr(),
            m, k_dim, n,
        );

        eprintln!("[tiled_gemm_gpu] grid={:?}, dispatching...", grid);
        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, grid, ka_buf);
        queue.wait_idle().unwrap();

        // Read back and verify
        let mut y_gpu = vec![0f32; (m * n) as usize];
        unsafe { y_buf.read(std::slice::from_raw_parts_mut(y_gpu.as_mut_ptr() as *mut u8, (m * n * 4) as usize)); }

        let mut max_err: f32 = 0.0;
        let mut max_rel_err: f32 = 0.0;
        for i in 0..(m * n) as usize {
            let err = (y_gpu[i] - y_ref[i]).abs();
            let rel = if y_ref[i].abs() > 1e-6 { err / y_ref[i].abs() } else { err };
            max_err = max_err.max(err);
            max_rel_err = max_rel_err.max(rel);
        }

        eprintln!("tiled GEMM {}×{}×{}: max_abs_err={:.2e}, max_rel_err={:.2e}", m, k_dim, n, max_err, max_rel_err);
        eprintln!("  Y_ref[0..4] = {:?}", &y_ref[..4]);
        eprintln!("  Y_gpu[0..4] = {:?}", &y_gpu[..4]);

        assert!(max_rel_err < 0.05,
            "tiled GEMM too inaccurate: max_rel_err={:.4e} (threshold 5e-2)", max_rel_err);
        eprintln!("✓ tiled_gemm GPU PASSED ({}×{}×{}, max_rel_err={:.2e})", m, k_dim, n, max_rel_err);
    }

    // ═══════════════════════════════════════════════════════════
    // ElemChain Compilation Tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_elem_chain_swiglu_compile() {
        let func = ElemChain::swiglu(256);
        let lowered = lower_elementwise_1d(&func, 256, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("✓ swiglu: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_scale_add_compile() {
        let func = ElemChain::scale_add(256);
        let lowered = lower_elementwise_1d(&func, 256, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ scale_add: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_residual_scale_compile() {
        let func = ElemChain::residual_add_scale(256);
        let lowered = lower_elementwise_1d(&func, 256, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ residual_scale: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_weight_decay_compile() {
        let func = ElemChain::weight_decay_update(256);
        let lowered = lower_elementwise_1d(&func, 256, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ weight_decay: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_abs_clip_compile() {
        let func = ElemChain::abs_clip(256);
        let lowered = lower_elementwise_1d(&func, 256, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ abs_clip: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_axpy_compile() {
        let func = ElemChain::axpy(256);
        let lowered = lower_elementwise_1d(&func, 256, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ axpy: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_custom_relu_compile() {
        // Custom chain: acc = relu(x)
        let func = ElemChain::new("custom_relu")
            .input("x")
            .output("y")
            .relu()
            .build(128);
        let lowered = lower_elementwise_1d(&func, 128, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ custom_relu: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_custom_sigmoid_compile() {
        let func = ElemChain::new("custom_sigmoid")
            .input("x")
            .output("y")
            .sigmoid_op()
            .build(128);
        let lowered = lower_elementwise_1d(&func, 128, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ custom_sigmoid: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_inplace_compile() {
        // inplace mode: output writes back to input[0]
        let func = ElemChain::new("inplace_relu")
            .input("x")
            .inplace()
            .relu()
            .build(128);
        let lowered = lower_elementwise_1d(&func, 128, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ inplace_relu: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_elem_chain_ssa_dump() {
        // Verify the generated SSA IR makes sense
        let func = ElemChain::swiglu(64);
        let ir = func.dump();
        eprintln!("=== SwiGLU SSA IR ===\n{}", ir);
        assert!(ir.contains("program_id"), "should have program_id");
        assert!(ir.contains("load"), "should have load");
        assert!(ir.contains("Silu"), "should have Silu");
        assert!(ir.contains("Mul"), "should have Mul");
        assert!(ir.contains("store"), "should have store");
    }

    // ═══════════════════════════════════════════════════════════
    // ElemChain GPU E2E Correctness Tests
    // ═══════════════════════════════════════════════════════════

    /// Generic GPU E2E test harness for ElemChain-based kernels.
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn run_elem_chain_gpu_test(
        name: &str,
        func: &TileFunc,
        wg_size: u32,
        inputs: &[&[f32]],
        scalars: &[f32],
        expected: &[f32],
        tol: f32,
    ) {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let n = expected.len() as u32;
        let lowered = lower_elementwise_1d(func, wg_size, 1).unwrap();
        let elf = lowered.kernel.compile(Target::GFX1100).unwrap();

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        // Allocate and upload input buffers
        let mut gpu_bufs = Vec::new();
        for &input in inputs {
            let buf = device.alloc_vram(n as usize * 4).unwrap();
            buf.write(unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, n as usize * 4) });
            gpu_bufs.push(buf);
        }

        // Output buffer
        let out_buf = device.alloc_vram(n as usize * 4).unwrap();
        out_buf.write(&vec![0u8; n as usize * 4]);

        // Build kernargs: [in0_ptr, in1_ptr, ..., out_ptr, scalar0, scalar1, ..., n_elems]
        let n_ptrs = inputs.len() + 1; // inputs + output
        let n_scalars = scalars.len();
        let ka_size = n_ptrs * 8 + n_scalars * 4 + 4;
        let mut ka = vec![0u8; ka_size];
        let mut off = 0;
        for buf in &gpu_bufs {
            ka[off..off+8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
            off += 8;
        }
        ka[off..off+8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        off += 8;
        for &s in scalars {
            ka[off..off+4].copy_from_slice(&s.to_bits().to_le_bytes());
            off += 4;
        }
        ka[off..off+4].copy_from_slice(&n.to_le_bytes());

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [wg_size, 1, 1], lds_size: 0,
        }).unwrap();

        let n_wg = (n + wg_size - 1) / wg_size;
        let ka_buf = pool.write_kernargs(0, &ka);
        queue.submit(&gk, [n_wg * wg_size, 1, 1], ka_buf);
        queue.wait_idle().unwrap();

        // Read back results
        let mut result = vec![0f32; n as usize];
        unsafe { out_buf.read(std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n as usize * 4)); }

        // Compare
        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            let err = (result[i] - expected[i]).abs();
            let rel = if expected[i].abs() > 1e-6 { err / expected[i].abs() } else { err };
            max_err = max_err.max(rel);
            if rel > tol {
                panic!("{} [{}]: gpu={:.6} cpu={:.6} rel_err={:.2e}", name, i, result[i], expected[i], rel);
            }
        }
        eprintln!("✓ {} GPU PASSED (max_rel_err={:.2e}, {} elems)", name, max_err, n);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_elem_chain_gpu_relu() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let expected: Vec<f32> = input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();

        let func = ElemChain::new("relu")
            .input("x").output("y")
            .relu()
            .build(256);
        run_elem_chain_gpu_test("relu", &func, 256, &[&input], &[], &expected, 1e-6);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_elem_chain_gpu_sigmoid() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
        let expected: Vec<f32> = input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

        let func = ElemChain::new("sigmoid")
            .input("x").output("y")
            .sigmoid_op()
            .build(256);
        run_elem_chain_gpu_test("sigmoid", &func, 256, &[&input], &[], &expected, 1e-3);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_elem_chain_gpu_scale_add() {
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
        let alpha = 0.7f32;
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * alpha + bi).collect();

        let func = ElemChain::scale_add(256);
        run_elem_chain_gpu_test("scale_add", &func, 256, &[&a, &b], &[alpha], &expected, 1e-5);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_elem_chain_gpu_swiglu() {
        let n = 256;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let expected: Vec<f32> = gate.iter().zip(up.iter())
            .map(|(&g, &u)| {
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect();

        let func = ElemChain::swiglu(256);
        run_elem_chain_gpu_test("swiglu", &func, 256, &[&gate, &up], &[], &expected, 1e-3);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_elem_chain_gpu_abs_clip() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.5).collect();
        let threshold = 10.0f32;
        let expected: Vec<f32> = input.iter().map(|&x| x.abs().min(threshold)).collect();

        let func = ElemChain::abs_clip(256);
        run_elem_chain_gpu_test("abs_clip", &func, 256, &[&input], &[threshold], &expected, 1e-6);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_elem_chain_gpu_weight_decay() {
        let n = 256;
        let w: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let grad: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.001).collect();
        let decay = 0.999f32;   // 1 - lr*wd
        let neg_lr = -0.001f32; // -lr
        let expected: Vec<f32> = w.iter().zip(grad.iter())
            .map(|(&wi, &gi)| wi * decay + gi * neg_lr)
            .collect();

        let func = ElemChain::weight_decay_update(256);
        run_elem_chain_gpu_test("weight_decay", &func, 256, &[&w, &grad], &[decay, neg_lr], &expected, 1e-5);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_elem_chain_gpu_axpy() {
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
        let alpha = 2.5f32;
        let expected: Vec<f32> = a.iter().zip(b.iter())
            .map(|(&ai, &bi)| ai + alpha * bi)
            .collect();

        let func = ElemChain::axpy(256);
        run_elem_chain_gpu_test("axpy", &func, 256, &[&a, &b], &[alpha], &expected, 1e-5);
    }

    // ═══════════════════════════════════════════════════════════
    // ElemChain Performance Benchmark
    // ═══════════════════════════════════════════════════════════

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    #[ignore] // Run: cargo test --release --features rocm -- test_elem_chain_benchmark --ignored --nocapture
    fn test_elem_chain_benchmark() {
        use crate::ignis::gpu_context::GpuRuntime;

        let rt = GpuRuntime::new().expect("GpuRuntime");
        let n = 4 * 1024 * 1024u32; // 4M elements = 16 MB
        let wg_size = 256u32;

        eprintln!("═══════════════════════════════════════════════════════════");
        eprintln!("  ElemChain Fusion Benchmark: {} elements ({:.1} MB)", n, n as f64 * 4.0 / 1e6);
        eprintln!("═══════════════════════════════════════════════════════════");

        // Allocate buffers
        let a_buf = rt.alloc(n as usize * 4).unwrap();
        let b_buf = rt.alloc(n as usize * 4).unwrap();
        let out_buf = rt.alloc(n as usize * 4).unwrap();
        a_buf.zero(); b_buf.zero(); out_buf.zero();

        // ── Scale + Add: Fused vs Separate ──

        // Fused kernel: out = a * alpha + b (1 dispatch)
        let fused_func = ElemChain::scale_add(wg_size);
        let fused_lowered = lower_elementwise_1d(&fused_func, wg_size, 1).unwrap();
        let fused_elf = fused_lowered.kernel.compile(Target::GFX1100).unwrap();

        // Separate kernels: scale + add (2 dispatches)
        let scale_func = ElemChain::new("scale_only")
            .input("x").output("y").scalar("alpha")
            .scale("alpha").build(wg_size);
        let scale_lowered = lower_elementwise_1d(&scale_func, wg_size, 1).unwrap();
        let scale_elf = scale_lowered.kernel.compile(Target::GFX1100).unwrap();

        let add_func = ElemChain::new("add_only")
            .input("a").input("b").output("y")
            .add_input(1).build(wg_size);
        let add_lowered = lower_elementwise_1d(&add_func, wg_size, 1).unwrap();
        let add_elf = add_lowered.kernel.compile(Target::GFX1100).unwrap();

        use crate::gpu_backend::{GpuKernel, KernelLoadConfig};
        let load_cfg = KernelLoadConfig { workgroup_size: [wg_size, 1, 1], lds_size: 0 };

        let fused_kernel = GpuKernel::load(&rt.device, &fused_elf, &load_cfg).unwrap();
        let scale_kernel = GpuKernel::load(&rt.device, &scale_elf, &load_cfg).unwrap();
        let add_kernel = GpuKernel::load(&rt.device, &add_elf, &load_cfg).unwrap();

        let n_wg = (n + wg_size - 1) / wg_size;
        let grid = [n_wg * wg_size, 1, 1];
        let alpha_bits = 0.7f32.to_bits();

        // Fused kernarg: [a_ptr, b_ptr, out_ptr, alpha, n_elems]
        let mut fused_ka = vec![0u8; 32];
        fused_ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
        fused_ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
        fused_ka[16..24].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        fused_ka[24..28].copy_from_slice(&alpha_bits.to_le_bytes());
        fused_ka[28..32].copy_from_slice(&n.to_le_bytes());

        // Scale kernarg: [x_ptr, y_ptr, alpha, n_elems]
        let mut scale_ka = vec![0u8; 24];
        scale_ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
        scale_ka[8..16].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        scale_ka[16..20].copy_from_slice(&alpha_bits.to_le_bytes());
        scale_ka[20..24].copy_from_slice(&n.to_le_bytes());

        // Add kernarg: [a_ptr, b_ptr, out_ptr, n_elems]
        let mut add_ka = vec![0u8; 28];
        add_ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes()); // scaled a
        add_ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
        add_ka[16..24].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        add_ka[24..28].copy_from_slice(&n.to_le_bytes());

        // Warmup
        for _ in 0..10 {
            rt.dispatch(&fused_kernel, grid, &fused_ka).unwrap();
            rt.dispatch(&scale_kernel, grid, &scale_ka).unwrap();
            rt.dispatch(&add_kernel, grid, &add_ka).unwrap();
        }

        let n_iters = 50;

        // Benchmark: Fused
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            rt.dispatch_async(&fused_kernel, grid, &fused_ka);
        }
        rt.wait_idle().unwrap();
        let fused_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        // Benchmark: Separate (scale + add)
        let t1 = std::time::Instant::now();
        for _ in 0..n_iters {
            rt.dispatch_async(&scale_kernel, grid, &scale_ka);
            rt.dispatch_async(&add_kernel, grid, &add_ka);
        }
        rt.wait_idle().unwrap();
        let separate_us = t1.elapsed().as_micros() as f64 / n_iters as f64;

        // Bandwidth calculation
        // Fused: read 2 inputs (2*N*4) + write 1 output (N*4) = 12 bytes/elem
        // Separate: scale reads 1 input + writes 1 output (8B) + add reads 2 inputs + writes 1 output (12B) = 20 bytes/elem
        let fused_bytes = n as f64 * 12.0;
        let separate_bytes = n as f64 * 20.0;
        let fused_gbps = fused_bytes / (fused_us * 1e3);
        let separate_gbps = separate_bytes / (separate_us * 1e3);
        let speedup = separate_us / fused_us;

        eprintln!("  Scale+Add (fused):    {:.1} µs  ({:.1} GB/s)", fused_us, fused_gbps);
        eprintln!("  Scale+Add (separate): {:.1} µs  ({:.1} GB/s)", separate_us, separate_gbps);
        eprintln!("  Speedup: {:.2}×  ({:.0}% faster)", speedup, (speedup - 1.0) * 100.0);
        eprintln!("═══════════════════════════════════════════════════════════");

        rt.recycle(a_buf); rt.recycle(b_buf); rt.recycle(out_buf);
    }
}
