//! T0 Assembly Emitter — IR → GCN Assembly Text
//!
//! Converts allocated IR operations to GCN assembly text (.s format)
//! that can be assembled by LLVM/clang into a code object.

use std::fmt::Write;
use super::ir::*;
use super::regalloc::RegAlloc;

/// Assembly text emitter for GCN ISA.
pub struct AsmEmitter {
    buf: String,
    indent: &'static str,
    target: Target,
    // Waitcnt tracking: count outstanding memory ops to avoid redundant waits
    outstanding_vmcnt: u32,   // pending global loads
    outstanding_lgkmcnt: u32, // pending LDS / scalar loads
    outstanding_vscnt: u32,   // pending global stores
    waits_emitted: u32,       // total wait instructions emitted
    waits_elided: u32,        // waits skipped (already at 0)
    // s_delay_alu tracking: auto-inject VALU dependency hints
    valu_count: u32,                // monotonic VALU instruction counter (1-based; 0 = never written)
    last_writer: [u32; 257],        // last VALU that wrote each phys VGPR (0..255) + VCC (256)
    delay_alu_emitted: u32,         // stats: total s_delay_alu emitted
}

impl AsmEmitter {
    pub fn new() -> Self {
        Self {
            buf: String::with_capacity(8192),
            indent: "  ",
            target: Target::GFX1100,
            outstanding_vmcnt: 0,
            outstanding_lgkmcnt: 0,
            outstanding_vscnt: 0,
            waits_emitted: 0,
            waits_elided: 0,
            valu_count: 1,
            last_writer: [0; 257],
            delay_alu_emitted: 0,
        }
    }

    /// Emit a complete kernel assembly file.
    pub fn emit_kernel(
        &mut self,
        name: &str,
        ops: &[Op],
        alloc: &RegAlloc,
        target: Target,
        kernarg_size: u32,
        lds_size: u32,
        wgp_mode: bool,
    ) {
        self.target = target;

        // Header
        writeln!(self.buf, ".amdgcn_target \"amdgcn-amd-amdhsa--{}\"", target.mcpu_str()).unwrap();
        writeln!(self.buf).unwrap();

        // Text section
        writeln!(self.buf, ".text").unwrap();
        writeln!(self.buf, ".globl {}", name).unwrap();
        writeln!(self.buf, ".p2align 8").unwrap();
        writeln!(self.buf, ".type {},@function", name).unwrap();
        writeln!(self.buf, "{}:", name).unwrap();

        // Emit all ops
        for op in ops {
            self.emit_op(op, alloc);
        }

        // Function end label
        writeln!(self.buf, ".Lfunc_end_{}:", name).unwrap();
        writeln!(self.buf, "  .size {}, .Lfunc_end_{}-{}", name, name, name).unwrap();
        writeln!(self.buf).unwrap();

        // Kernel descriptor in .rodata
        writeln!(self.buf, ".rodata").unwrap();
        writeln!(self.buf, ".p2align 6").unwrap();
        writeln!(self.buf, ".amdhsa_kernel {}", name).unwrap();
        writeln!(self.buf, "  .amdhsa_group_segment_fixed_size {}", lds_size).unwrap();
        writeln!(self.buf, "  .amdhsa_private_segment_fixed_size 0").unwrap();
        writeln!(self.buf, "  .amdhsa_kernarg_size {}", kernarg_size).unwrap();
        writeln!(self.buf, "  .amdhsa_user_sgpr_kernarg_segment_ptr 1").unwrap();
        writeln!(self.buf, "  .amdhsa_next_free_vgpr {}", alloc.total_vgprs).unwrap();
        writeln!(self.buf, "  .amdhsa_next_free_sgpr {}", alloc.total_sgprs).unwrap();
        writeln!(self.buf, "  .amdhsa_wavefront_size32 1").unwrap();
        writeln!(self.buf, "  .amdhsa_system_sgpr_workgroup_id_x 1").unwrap();
        writeln!(self.buf, "  .amdhsa_system_sgpr_workgroup_id_y 1").unwrap();
        writeln!(self.buf, "  .amdhsa_system_sgpr_workgroup_id_z 1").unwrap();
        writeln!(self.buf, "  .amdhsa_float_denorm_mode_32 3").unwrap();
        writeln!(self.buf, "  .amdhsa_float_denorm_mode_16_64 3").unwrap();
        // CRITICAL: LLVM defaults .amdhsa_workgroup_processor_mode to 1 on GFX11!
        // We MUST emit this directive explicitly regardless of the desired value,
        // otherwise all kernels silently get WGP mode even when wgp_mode=false.
        // (Confirmed via llvm-objdump: KCP=0x0408 = bit10 set = WGP enabled by default)
        writeln!(self.buf, "  .amdhsa_workgroup_processor_mode {}",
            if wgp_mode { 1 } else { 0 }).unwrap();
        writeln!(self.buf, ".end_amdhsa_kernel").unwrap();
        writeln!(self.buf).unwrap();

        // NOTE: .amdgpu_metadata YAML is NOT emitted — KFD runtime reads
        // kernel descriptors directly from .rodata (.amdhsa_kernel above).
        // The metadata is only needed for HIP's hipModuleLoadData.
    }

    /// Emit a single IR operation as assembly text.
    fn emit_op(&mut self, op: &Op, a: &RegAlloc) {
        // ── s_delay_alu auto-injection ──
        // Track VALU writes to physical VGPRs and inject delay hints for RAW deps.
        // On control flow / sync, reset tracking (conservative but correct).
        if matches!(op, Op::Label(_) | Op::Branch(_) | Op::BranchScc0(_) | Op::BranchScc1(_)
            | Op::BranchVccz(_) | Op::Barrier | Op::SBarrier
            | Op::WaitVmcnt(_) | Op::WaitLgkmcnt(_) | Op::WaitVscnt(_)) {
            self.last_writer.fill(0);
        }

        if !matches!(op, Op::RawAsm(_)) {
            let lat = super::latency_model::op_latency(op);
            let is_valu = matches!(lat.pipeline,
                super::latency_model::Pipeline::VALU |
                super::latency_model::Pipeline::WMMA |
                super::latency_model::Pipeline::TRANS
            );

            if is_valu && !std::env::var("T0_SKIP_DELAY_ALU").is_ok() {
                // Check VGPR read dependencies
                let mut min_dep = 5u32; // > 4 means no dep
                for v in op.vreg_uses() {
                    let phys = a.phys_v(v) as usize;
                    if phys < 256 {
                        let last = self.last_writer[phys];
                        if last > 0 {
                            let dist = self.valu_count - last;
                            if dist >= 1 && dist <= 4 && dist < min_dep {
                                min_dep = dist;
                            }
                        }
                    }
                }
                // Also check multi-VGPR sources (WMMA uses v[N:N+7])
                if let Op::Wmma { a: va, b: vb, c: vc, format, .. } = op {
                    for (base_vreg, regs) in [
                        (va, format.a_vreg_count(self.target)),
                        (vb, format.b_vreg_count(self.target)),
                        (vc, format.c_vreg_count(self.target)),
                    ] {
                        let base_phys = a.phys_v(*base_vreg) as usize;
                        for off in 0..regs as usize {
                            let p = base_phys + off;
                            if p < 256 {
                                let last = self.last_writer[p];
                                if last > 0 {
                                    let dist = self.valu_count - last;
                                    if dist >= 1 && dist <= 4 && dist < min_dep {
                                        min_dep = dist;
                                    }
                                }
                            }
                        }
                    }
                }

                if min_dep <= 4 {
                    writeln!(self.buf, "{}s_delay_alu instid0(VALU_DEP_{})",
                        self.indent, min_dep).unwrap();
                    self.delay_alu_emitted += 1;
                }

                // Record this instruction's VGPR writes
                for v in op.vreg_defs() {
                    let phys = a.phys_v(v) as usize;
                    if phys < 256 {
                        self.last_writer[phys] = self.valu_count;
                    }
                }
                // Multi-VGPR defs (WMMA writes v[dst:dst+7])
                if let Op::Wmma { dst, format, .. } = op {
                    let base_phys = a.phys_v(*dst) as usize;
                    for off in 0..format.dst_vreg_count(self.target) as usize {
                        let p = base_phys + off;
                        if p < 256 {
                            self.last_writer[p] = self.valu_count;
                        }
                    }
                }
                // CvtPkBf16F32 emits 2 instructions (lshr + and_or), so mark as 2 ops
                if matches!(op, Op::CvtPkBf16F32 { .. }) {
                    self.valu_count += 2;
                } else {
                    self.valu_count += 1;
                }
            }
        }

        match op {
            // ── Global Memory ──
            Op::GlobalLoad { dst, addr, width, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*addr);
                let instr = match width {
                    Width::B16 => "global_load_u16",
                    Width::B32 => "global_load_b32",
                    Width::B64 => "global_load_b64",
                    Width::B128 => "global_load_b128",
                };
                let dst_str = vreg_range_str(vd, width.vreg_count());
                let addr_str = format!("v[{}:{}]", va, va + 1);
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, {}, off", self.indent, instr, dst_str, addr_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, {}, off offset:{}", self.indent, instr, dst_str, addr_str, offset).unwrap();
                }
                self.outstanding_vmcnt += 1;
            }

            Op::BufferLoad { dst, voffset, srsrc, width, offset, soffset } => {
                let vd = a.phys_v(*dst);
                let vo = a.phys_v(*voffset);
                let sr = a.phys_s(SReg(srsrc.0));
                let instr = match width {
                    Width::B16 => "buffer_load_u16",
                    Width::B32 => "buffer_load_b32",
                    Width::B64 => "buffer_load_b64",
                    Width::B128 => "buffer_load_b128",
                };
                let dst_str = vreg_range_str(vd, width.vreg_count());
                let soff_str = if *soffset == SOFFSET_ZERO {
                    match self.target {
                        Target::GFX1100 => "0".to_string(),
                        // gfx1201 llvm-mc rejects literal "0" for MUBUF soffset syntax.
                        // Use scalar null register (encodes zero) instead of s0.
                        Target::GFX1201 => "null".to_string(),
                    }
                } else {
                    format!("s{}", a.phys_s(*soffset))
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen",
                        self.indent, instr, dst_str, vo, sr, sr + 3, soff_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen offset:{}",
                        self.indent, instr, dst_str, vo, sr, sr + 3, soff_str, offset).unwrap();
                }
                self.outstanding_vmcnt += 1;
            }

            Op::BufferStore { voffset, src, srsrc, width, offset, soffset } => {
                let vo = a.phys_v(*voffset);
                let vs = a.phys_v(*src);
                let sr = a.phys_s(SReg(srsrc.0));
                let instr = match width {
                    Width::B16 => "buffer_store_b16",
                    Width::B32 => "buffer_store_b32",
                    Width::B64 => "buffer_store_b64",
                    Width::B128 => "buffer_store_b128",
                };
                let src_str = vreg_range_str(vs, width.vreg_count());
                let soff_str = if *soffset == SOFFSET_ZERO {
                    match self.target {
                        Target::GFX1100 => "0".to_string(),
                        // gfx1201 llvm-mc rejects literal "0" for MUBUF soffset syntax.
                        // Use scalar null register (encodes zero) instead of s0.
                        Target::GFX1201 => "null".to_string(),
                    }
                } else {
                    format!("s{}", a.phys_s(*soffset))
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen",
                        self.indent, instr, src_str, vo, sr, sr + 3, soff_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen offset:{}",
                        self.indent, instr, src_str, vo, sr, sr + 3, soff_str, offset).unwrap();
                }
                self.outstanding_vscnt += 1;
            }

            Op::GlobalStore { addr, src, width, offset } => {
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                let instr = match width {
                    Width::B16 => "global_store_b16",
                    Width::B32 => "global_store_b32",
                    Width::B64 => "global_store_b64",
                    Width::B128 => "global_store_b128",
                };
                let src_str = vreg_range_str(vs, width.vreg_count());
                let addr_str = format!("v[{}:{}]", va, va + 1);
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, {}, off", self.indent, instr, addr_str, src_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, {}, off offset:{}", self.indent, instr, addr_str, src_str, offset).unwrap();
                }
                self.outstanding_vscnt += 1;
            }

            // ── LDS ──
            Op::LdsLoad { dst, addr, width, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*addr);
                let instr = match width {
                    Width::B16 => "ds_load_u16",
                    Width::B32 => "ds_load_b32",
                    Width::B64 => "ds_load_b64",
                    Width::B128 => "ds_load_b128",
                };
                let dst_str = vreg_range_str(vd, width.vreg_count());
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, v{}", self.indent, instr, dst_str, va).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, v{} offset:{}", self.indent, instr, dst_str, va, offset).unwrap();
                }
                self.outstanding_lgkmcnt += 1;  // LDS loads use lgkmcnt
            }

            Op::LdsStore { addr, src, width, offset } => {
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                let instr = match width {
                    Width::B16 => "ds_store_b16",
                    Width::B32 => "ds_store_b32",
                    Width::B64 => "ds_store_b64",
                    Width::B128 => "ds_store_b128",
                };
                let src_str = vreg_range_str(vs, width.vreg_count());
                if *offset == 0 {
                    writeln!(self.buf, "{}{} v{}, {}", self.indent, instr, va, src_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} v{}, {} offset:{}", self.indent, instr, va, src_str, offset).unwrap();
                }
            }

            // ── Scalar Memory ──
            Op::ScalarLoad { dst, base, offset, width } => {
                let sd = a.phys_s(SReg(dst.0));
                // Sentinel detection: KERNARG_BASE_SENTINEL → hardware s[0:1]
                let sb = if base.0 >= super::compile::T0Kernel::KERNARG_BASE_SENTINEL - 100 {
                    0u8  // s[0:1] = kernarg_segment_ptr (hardware)
                } else {
                    a.phys_s(SReg(base.0))
                };
                let instr = match width {
                    Width::B32 => "s_load_b32",
                    Width::B64 => "s_load_b64",
                    Width::B128 => "s_load_b128",
                    _ => panic!("Unsupported scalar load width: {:?}", width),
                };
                let dst_str = sreg_range_str(sd, width.vreg_count());
                writeln!(self.buf, "{}{} {}, s[{}:{}], {:#x}",
                    self.indent, instr, dst_str, sb, sb + 1, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // scalar loads use lgkmcnt
            }

            // ── Vector ALU ──
            Op::VAddF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_add_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMulF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_mul_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VFmaF32 { dst, src0, src1, src2 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_fma_f32 v{}, {}, {}, {}",
                    self.indent, vd,
                    operand_str(src0, a), operand_str(src1, a), operand_str(src2, a)).unwrap();
            }
            Op::VMaxF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_max_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMinF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_min_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMinU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_min_u32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMov { dst, src } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_mov_b32 v{}, {}",
                    self.indent, vd, operand_str(src, a)).unwrap();
            }
            Op::VMovFromSgpr { dst, src } => {
                let vd = a.phys_v(*dst);
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}v_mov_b32 v{}, s{}", self.indent, vd, ss).unwrap();
            }
            Op::VAddU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_add_nc_u32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMulLoU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let v0 = a.phys_v(*src0);
                let v1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_mul_lo_u32 v{}, v{}, v{}", self.indent, vd, v0, v1).unwrap();
            }
            Op::VLshlrevB32 { dst, shift, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_lshlrev_b32 v{}, {}, v{}", self.indent, vd, shift, vs).unwrap();
            }
            Op::VLshrrevB32 { dst, shift, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_lshrrev_b32 v{}, {}, v{}", self.indent, vd, shift, vs).unwrap();
            }
            Op::VAndB32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_and_b32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VReadfirstlane { dst, src } => {
                let sd = a.phys_s(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_readfirstlane_b32 s{}, v{}", self.indent, sd, vs).unwrap();
            }

            // ── 64-bit address arithmetic ──
            Op::VAddCo { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let v0 = a.phys_v(*src0);
                let v1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_add_co_u32 v{}, vcc_lo, v{}, v{}", self.indent, vd, v0, v1).unwrap();
            }
            Op::VAddCoCi { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_add_co_ci_u32 v{}, vcc_lo, v{}, 0, vcc_lo", self.indent, vd, vs).unwrap();
            }

            // ── Scalar ALU ──
            Op::SAddU32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                writeln!(self.buf, "{}s_add_u32 s{}, s{}, {}", self.indent, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SAddcU32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                writeln!(self.buf, "{}s_addc_u32 s{}, s{}, {}", self.indent, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SSubU32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                writeln!(self.buf, "{}s_sub_u32 s{}, s{}, {}", self.indent, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SAndB32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                writeln!(self.buf, "{}s_and_b32 s{}, s{}, {}", self.indent, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SMulI32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                let s1 = a.phys_s(*src1);
                writeln!(self.buf, "{}s_mul_i32 s{}, s{}, s{}", self.indent, sd, s0, s1).unwrap();
            }
            Op::SLshlB32 { dst, src, shift } => {
                let sd = a.phys_s(*dst);
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}s_lshl_b32 s{}, s{}, {}", self.indent, sd, ss, shift).unwrap();
            }
            Op::SLshrB32 { dst, src, shift } => {
                let sd = a.phys_s(*dst);
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}s_lshr_b32 s{}, s{}, {}", self.indent, sd, ss, shift).unwrap();
            }
            Op::SMov { dst, src } => {
                let sd = a.phys_s(*dst);
                writeln!(self.buf, "{}s_mov_b32 s{}, {}", self.indent, sd, soperand_str(src, a)).unwrap();
            }
            Op::SCmpLtU32 { src0, src1 } => {
                let s0 = a.phys_s(*src0);
                let s1 = a.phys_s(*src1);
                writeln!(self.buf, "{}s_cmp_lt_u32 s{}, s{}", self.indent, s0, s1).unwrap();
            }
            Op::SCmpEqU32 { src0, src1 } => {
                let s0 = a.phys_s(*src0);
                match src1 {
                    SOperand::SReg(s) => {
                        let s1 = a.phys_s(*s);
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, s{}", self.indent, s0, s1).unwrap();
                    }
                    SOperand::InlineInt(v) => {
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, {}", self.indent, s0, v).unwrap();
                    }
                    SOperand::Literal(v) => {
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, 0x{:x}", self.indent, s0, v).unwrap();
                    }
                    SOperand::Vcc => {
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, vcc_lo", self.indent, s0).unwrap();
                    }
                }
            }
            Op::SCmpGeU32 { src0, src1 } => {
                let s0 = a.phys_s(*src0);
                let s1 = a.phys_s(*src1);
                writeln!(self.buf, "{}s_cmp_ge_u32 s{}, s{}", self.indent, s0, s1).unwrap();
            }

            // ── WMMA ──
            Op::Wmma { dst, a: va, b: vb, c: vc, format } => {
                let d = a.phys_v(*dst);
                let pa = a.phys_v(*va);
                let pb = a.phys_v(*vb);
                let pc = a.phys_v(*vc);
                let instr = match format {
                    WmmaFormat::BF16_F32 => "v_wmma_f32_16x16x16_bf16",
                    WmmaFormat::F16_F32 => "v_wmma_f32_16x16x16_f16",
                    WmmaFormat::BF16_BF16 => "v_wmma_bf16_16x16x16_bf16",
                };
                let dst_regs = format.dst_vreg_count(self.target);
                let a_regs = format.a_vreg_count(self.target);
                let b_regs = format.b_vreg_count(self.target);
                let c_regs = format.c_vreg_count(self.target);
                writeln!(self.buf, "{}{} v[{}:{}], v[{}:{}], v[{}:{}], v[{}:{}]",
                    self.indent, instr,
                    d, u32::from(d) + dst_regs - 1,
                    pa, u32::from(pa) + a_regs - 1,
                    pb, u32::from(pb) + b_regs - 1,
                    pc, u32::from(pc) + c_regs - 1).unwrap();
            }

            // ── Control flow ──
            Op::Label(name) => {
                writeln!(self.buf, ".L{}:", name).unwrap();
            }
            Op::BranchScc1(target) => {
                writeln!(self.buf, "{}s_cbranch_scc1 .L{}", self.indent, target).unwrap();
            }
            Op::Branch(target) => {
                writeln!(self.buf, "{}s_branch .L{}", self.indent, target).unwrap();
            }

            // ── Synchronization ──
            Op::Barrier => {
                match self.target {
                    Target::GFX1100 => {
                        writeln!(self.buf, "{}s_barrier", self.indent).unwrap();
                    }
                    Target::GFX1201 => {
                        writeln!(self.buf, "{}s_barrier_signal -1", self.indent).unwrap();
                        writeln!(self.buf, "{}s_barrier_wait -1", self.indent).unwrap();
                    }
                }
            }
            Op::WaitVmcnt(n) => {
                if self.outstanding_vmcnt > 0 || *n > 0 {
                    let actual = (*n as u32).min(self.outstanding_vmcnt);
                    writeln!(self.buf, "{}s_waitcnt vmcnt({})", self.indent, actual).unwrap();
                    self.outstanding_vmcnt = actual;
                    self.waits_emitted += 1;
                } else {
                    self.waits_elided += 1;
                }
            }
            Op::WaitLgkmcnt(n) => {
                if self.outstanding_lgkmcnt > 0 || *n > 0 {
                    let actual = (*n as u32).min(self.outstanding_lgkmcnt);
                    writeln!(self.buf, "{}s_waitcnt lgkmcnt({})", self.indent, actual).unwrap();
                    self.outstanding_lgkmcnt = actual;
                    self.waits_emitted += 1;
                } else {
                    self.waits_elided += 1;
                }
            }
            Op::WaitVscnt(n) => {
                if self.outstanding_vscnt > 0 || *n > 0 {
                    let actual = (*n as u32).min(self.outstanding_vscnt);
                    match self.target {
                        Target::GFX1100 => {
                            writeln!(self.buf, "{}s_waitcnt_vscnt null, {:#x}", self.indent, actual).unwrap();
                        }
                        Target::GFX1201 => {
                            writeln!(self.buf, "{}s_wait_storecnt {:#x}", self.indent, actual).unwrap();
                        }
                    }
                    self.outstanding_vscnt = actual;
                    self.waits_emitted += 1;
                } else {
                    self.waits_elided += 1;
                }
            }
            Op::ClearVcc => {
                writeln!(self.buf, "{}s_mov_b32 vcc_lo, 0", self.indent).unwrap();
            }
            Op::SMovToVcc { src } => {
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}s_mov_b32 vcc_lo, s{}", self.indent, ss).unwrap();
            }

            // ── Program structure ──
            Op::Endpgm => {
                writeln!(self.buf, "{}s_endpgm", self.indent).unwrap();
            }

            // ── Hardware register access ──
            Op::CaptureTgid { dst, axis } => {
                let sd = a.phys_s(*dst);
                let hw_sreg = 2 + axis;  // s2=TGID.x, s3=TGID.y, s4=TGID.z
                writeln!(self.buf, "{}s_mov_b32 s{}, s{}  ; capture TGID.{}",
                    self.indent, sd, hw_sreg,
                    match axis { 0 => "x", 1 => "y", _ => "z" }).unwrap();
            }

            Op::ComputeGlobalIdX { dst, wg_size } => {
                let vd = a.phys_v(*dst);
                // s2 = TGID.x (hardware), v0 = WORKITEM_ID_X (hardware)
                // Compute: dst = TGID.x * wg_size + v0
                // Clobbers s2 (OK since TGID is only needed once)
                writeln!(self.buf, "{}s_mul_i32 s2, s2, {}  ; TGID.x * WG_SIZE",
                    self.indent, wg_size).unwrap();
                writeln!(self.buf, "{}v_add_nc_u32 v{}, s2, v0  ; global_id = wg_offset + tid",
                    self.indent, vd).unwrap();
            }

            // ── Cross-lane operations ──
            Op::DsSwizzle { dst, src, offset } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_swizzle_b32 v{}, v{} offset:{:#06x}",
                    self.indent, vd, vs, offset).unwrap();
            }

            // ── Special math ──
            Op::VRsqF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_rsq_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VExpF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                // GFX11: v_exp_f32 computes 2^x (NOT e^x!)
                writeln!(self.buf, "{}v_exp_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VSinF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                // GFX11: v_sin_f32 computes sin(2π·x)
                writeln!(self.buf, "{}v_sin_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VCosF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                // GFX11: v_cos_f32 computes cos(2π·x)
                writeln!(self.buf, "{}v_cos_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VRcpF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_rcp_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VXorB32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_xor_b32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }
            Op::VSubF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_sub_f32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }
            Op::VMaxF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_max_f32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }
            Op::VAndB32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_and_b32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }

            // ── Wave-level butterfly reduction (Wave32) ──
            Op::WaveReduceAddF32 { val, tmp } => {
                let vv = a.phys_v(*val);
                let vt = a.phys_v(*tmp);
                for (offset, label) in &[
                    (0x401Fu16, "xor16"), (0x201F, "xor8"),
                    (0x101F, "xor4"), (0x081F, "xor2"), (0x041F, "xor1"),
                ] {
                    writeln!(self.buf, "{}ds_swizzle_b32 v{}, v{} offset:{:#06x}  ; {}",
                        self.indent, vt, vv, offset, label).unwrap();
                    writeln!(self.buf, "{}s_waitcnt lgkmcnt(0)", self.indent).unwrap();
                    writeln!(self.buf, "{}v_add_f32 v{}, v{}, v{}",
                        self.indent, vv, vv, vt).unwrap();
                }
            }
            Op::WaveReduceMaxF32 { val, tmp } => {
                let vv = a.phys_v(*val);
                let vt = a.phys_v(*tmp);
                for (offset, label) in &[
                    (0x401Fu16, "xor16"), (0x201F, "xor8"),
                    (0x101F, "xor4"), (0x081F, "xor2"), (0x041F, "xor1"),
                ] {
                    writeln!(self.buf, "{}ds_swizzle_b32 v{}, v{} offset:{:#06x}  ; {}",
                        self.indent, vt, vv, offset, label).unwrap();
                    writeln!(self.buf, "{}s_waitcnt lgkmcnt(0)", self.indent).unwrap();
                    writeln!(self.buf, "{}v_max_f32 v{}, v{}, v{}",
                        self.indent, vv, vv, vt).unwrap();
                }
            }

            // ── Data type conversion ──
            Op::CvtPkBf16F32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let v0 = a.phys_v(*src0);
                let v1 = a.phys_v(*src1);
                // GFX11 has no v_cvt_pk_bf16_f32! Use bit ops:
                // bf16 = f32[31:16] (truncate lower mantissa bits)
                // dst = (bf16(src1) << 16) | bf16(src0)
                //     = (src1 & 0xFFFF0000) | (src0 >> 16)
                // Step 1: dst = src0 >> 16
                writeln!(self.buf, "{}v_lshrrev_b32 v{}, 16, v{}",
                    self.indent, vd, v0).unwrap();
                // Step 2: dst = (src1 & 0xFFFF0000) | dst
                // v_and_or_b32 dst, src1, 0xFFFF0000, dst
                writeln!(self.buf, "{}v_and_or_b32 v{}, v{}, 0xffff0000, v{}",
                    self.indent, vd, v1, vd).unwrap();
            }
            Op::VCvtF32U32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_cvt_f32_u32 v{}, v{}",
                    self.indent, vd, vs).unwrap();
            }
            Op::VCvtU32F32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_cvt_u32_f32 v{}, v{}",
                    self.indent, vd, vs).unwrap();
            }
            Op::VSubU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_sub_u32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }

            // ── LDS (Local Data Share) ──
            Op::DsStoreB16 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b16 v{}, v{} offset:{}",
                    self.indent, va, vs, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsStoreB32 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b32 v{}, v{} offset:{}",
                    self.indent, va, vs, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsStoreB64 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b64 v{}, v[{}:{}] offset:{}",
                    self.indent, va, vs, vs + 1, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsStoreB128 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b128 v{}, v[{}:{}] offset:{}",
                    self.indent, va, vs, vs + 3, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsLoadB32 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_b32 v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
                self.outstanding_lgkmcnt += 1;
            }
            Op::DsLoadB64 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_b64 v[{}:{}], v{} offset:{}",
                    self.indent, vd, vd + 1, va, offset).unwrap();
                self.outstanding_lgkmcnt += 1;
            }
            Op::DsLoadB128 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_b128 v[{}:{}], v{} offset:{}",
                    self.indent, vd, vd + 3, va, offset).unwrap();
                self.outstanding_lgkmcnt += 1;
            }
            Op::DsLoadU16 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_u16 v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
            }
            Op::DsLoadU16D16 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_u16_d16 v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
            }
            Op::DsLoadU16D16Hi { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_u16_d16_hi v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
            }
            Op::SBarrier => {
                match self.target {
                    Target::GFX1100 => {
                        writeln!(self.buf, "{}s_barrier", self.indent).unwrap();
                    }
                    Target::GFX1201 => {
                        writeln!(self.buf, "{}s_barrier_signal -1", self.indent).unwrap();
                        writeln!(self.buf, "{}s_barrier_wait -1", self.indent).unwrap();
                    }
                }
            }

            Op::VCmpLtU32 { src0, src1 } => {
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_cmp_lt_u32 vcc_lo, {}, {}",
                    self.indent, s0, s1).unwrap();
            }
            Op::VCmpGeU32 { src0, src1 } => {
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_cmp_ge_u32 vcc_lo, {}, {}",
                    self.indent, s0, s1).unwrap();
            }
            Op::VCmpGtF32Imm0 { src } => {
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_cmp_gt_f32 vcc_lo, v{}, 0",
                    self.indent, vs).unwrap();
            }
            Op::VCndmaskB32 { dst, src_false, src_true } => {
                let vd = a.phys_v(*dst);
                let sf = operand_str(src_false, a);
                let st = operand_str(src_true, a);
                writeln!(self.buf, "{}v_cndmask_b32 v{}, {}, {}, vcc_lo",
                    self.indent, vd, sf, st).unwrap();
            }
            Op::SaveExec { dst } => {
                let sd = a.phys_s(*dst);
                // SOP1: s_and_saveexec_b32 s_dst, vcc_lo
                // Saves EXEC to dst, then EXEC &= VCC
                writeln!(self.buf, "{}s_and_saveexec_b32 s{}, vcc_lo",
                    self.indent, sd).unwrap();
            }
            Op::RestoreExec { src } => {
                let ss = a.phys_s(*src);
                // SOP1: s_mov_b32 exec_lo, s_src
                writeln!(self.buf, "{}s_mov_b32 exec_lo, s{}",
                    self.indent, ss).unwrap();
            }
            Op::XorExec { saved } => {
                let ss = a.phys_s(*saved);
                // SOP2: s_xor_b32 exec_lo, exec_lo, s_saved
                // Flips EXEC to else-branch lanes: (original & cond) XOR original = original & ~cond
                writeln!(self.buf, "{}s_xor_b32 exec_lo, exec_lo, s{}",
                    self.indent, ss).unwrap();
            }
            // ── Additional branch variants ──
            Op::BranchScc0(target) => {
                writeln!(self.buf, "{}s_cbranch_scc0 .L{}", self.indent, target).unwrap();
            }
            Op::BranchVccz(target) => {
                writeln!(self.buf, "{}s_cbranch_vccz .L{}", self.indent, target).unwrap();
            }

            // ── Additional ALU ──
            Op::VOrB32 { dst, src0, src1 } => {
                let d = a.phys_v(*dst);
                writeln!(self.buf, "{}v_or_b32 v{}, {}, {}",
                    self.indent, d, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VSqrtF32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_sqrt_f32 v{}, v{}", self.indent, d, s).unwrap();
            }
            Op::VLog2F32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_log_f32 v{}, v{}", self.indent, d, s).unwrap();
            }
            Op::VCmpGtU32Imm { src, imm } => {
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_cmp_gt_u32 vcc_lo, v{}, {}", self.indent, s, imm).unwrap();
            }
            Op::VCmpEqU32Imm { src, imm } => {
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_cmp_eq_u32 vcc_lo, v{}, {}", self.indent, s, imm).unwrap();
            }
            Op::VCmpGeI32 { src0, src1 } => {
                let s0 = a.phys_v(*src0);
                let s1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_cmp_ge_i32 vcc_lo, v{}, v{}", self.indent, s0, s1).unwrap();
            }

            // ── Global atomics ──
            Op::GlobalAtomicAddF32 { addr, src, offset } => {
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                if *offset == 0 {
                    writeln!(self.buf, "{}global_atomic_add_f32 v[{}:{}], v{}, off",
                        self.indent, va, va + 1, vs).unwrap();
                } else {
                    writeln!(self.buf, "{}global_atomic_add_f32 v[{}:{}], v{}, off offset:{}",
                        self.indent, va, va + 1, vs, offset).unwrap();
                }
            }

            Op::GlobalAtomicAddU32Rtn { dst, addr, src } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}global_atomic_add_u32 v{}, v[{}:{}], v{}, off glc",
                    self.indent, vd, va, va + 1, vs).unwrap();
            }

            // ── SMEM scalar load ──
            Op::SMemLoadDword { dst, base_lo, base_hi, offset } => {
                let sd = a.phys_s(*dst);
                let sb = a.phys_s(*base_lo);
                let sbh = a.phys_s(*base_hi);
                // SMEM requires even-aligned SBASE pair
                let (actual_lo, actual_hi) = if sb % 2 == 0 && sbh == sb + 1 {
                    (sb, sbh)
                } else {
                    // Copy to even-aligned scratch pair s4:s5
                    writeln!(self.buf, "{}s_mov_b32 s4, s{}",
                        self.indent, sb).unwrap();
                    writeln!(self.buf, "{}s_mov_b32 s5, s{}",
                        self.indent, sbh).unwrap();
                    (4u8, 5u8)
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}s_load_dword s{}, s[{}:{}], 0",
                        self.indent, sd, actual_lo, actual_hi).unwrap();
                } else {
                    writeln!(self.buf, "{}s_load_dword s{}, s[{}:{}], {}",
                        self.indent, sd, actual_lo, actual_hi, offset).unwrap();
                }
            }

            // ── 64-bit address arithmetic ──
            Op::VAddCOU32 { dst, src0, src1 } => {
                let d = a.phys_v(*dst);
                let s0 = a.phys_v(*src0);
                let s1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_add_co_u32 v{}, vcc_lo, v{}, v{}",
                    self.indent, d, s0, s1).unwrap();
            }
            Op::VAddCCU32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_add_co_ci_u32 v{}, vcc_lo, v{}, 0, vcc_lo",
                    self.indent, d, s).unwrap();
            }

            // ── Lane permute ──
            Op::VPermlanex16B32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_permlanex16_b32 v{}, v{}, s0, s0",
                    self.indent, d, s).unwrap();
            }

            // ── VOP3 three-source ──
            Op::VAndOrB32 { dst, src0, literal, src2 } => {
                let d = a.phys_v(*dst);
                let s0 = a.phys_v(*src0);
                let s2 = a.phys_v(*src2);
                writeln!(self.buf, "{}v_and_or_b32 v{}, v{}, 0x{:x}, v{}",
                    self.indent, d, s0, literal, s2).unwrap();
            }

            // ── Hardware performance counter ──
            Op::ReadShaderCycles { dst } => {
                let vd = a.phys_v(*dst);
                // Read 32-bit shader cycle counter into s2 (scratch), then move to VGPR
                // LLVM verified: encoding [0x1d,0xf8,0x80,0xb8]
                writeln!(self.buf, "{}s_getreg_b32 s2, hwreg(HW_REG_SHADER_CYCLES)  ; GPU cycle counter",
                    self.indent).unwrap();
                writeln!(self.buf, "{}v_mov_b32 v{}, s2", self.indent, vd).unwrap();
            }

            // ── Raw assembly passthrough ──
            Op::RawAsm(text) => {
                writeln!(self.buf, "{}{}", self.indent, text).unwrap();
            }
        }
    }

    /// Get the generated assembly text.
    pub fn finish(self) -> String {
        if self.waits_elided > 0 && super::verbose_diagnostics_enabled() {
            eprintln!(
                "[T0 AsmEmitter] Waitcnt stats: {} emitted, {} elided (redundant)",
                self.waits_emitted, self.waits_elided
            );
        }
        self.buf
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Format a VGPR range string: "v0" for single, "v[0:3]" for multi.
fn vreg_range_str(base: u8, count: u32) -> String {
    if count == 1 {
        format!("v{}", base)
    } else {
        format!("v[{}:{}]", base, base as u32 + count - 1)
    }
}

/// Format an SGPR range string.
fn sreg_range_str(base: u8, count: u32) -> String {
    if count == 1 {
        format!("s{}", base)
    } else {
        format!("s[{}:{}]", base, base as u32 + count - 1)
    }
}

/// Format a vector operand as assembly text.
fn operand_str(op: &Operand, a: &RegAlloc) -> String {
    match op {
        Operand::VReg(v) => format!("v{}", a.phys_v(*v)),
        Operand::InlineInt(n) => format!("{}", n),
        Operand::InlineFloat(f) => {
            // LLVM assembly uses specific float notation
            if *f == 0.0 { "0".to_string() }
            else if *f == 0.5 { "0.5".to_string() }
            else if *f == 1.0 { "1.0".to_string() }
            else if *f == 2.0 { "2.0".to_string() }
            else if *f == 4.0 { "4.0".to_string() }
            else if *f == -0.5 { "-0.5".to_string() }
            else if *f == -1.0 { "-1.0".to_string() }
            else if *f == -2.0 { "-2.0".to_string() }
            else if *f == -4.0 { "-4.0".to_string() }
            else { format!("{:#010x}", f.to_bits()) }
        }
        Operand::Literal(v) => format!("{:#x}", v),
    }
}

/// Format a scalar operand.
fn soperand_str(op: &SOperand, a: &RegAlloc) -> String {
    match op {
        SOperand::SReg(s) => format!("s{}", a.phys_s(*s)),
        SOperand::InlineInt(n) => format!("{}", n),
        SOperand::Literal(v) => format!("{:#x}", v),
        SOperand::Vcc => "vcc_lo".to_string(),
    }
}
