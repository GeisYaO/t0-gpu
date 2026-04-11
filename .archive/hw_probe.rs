//! HW Probe — RDNA3 GFX1100 微架构全指令穷举探测框架
//!
//! 通过 KFD 裸金属在 GPU 上运行微基准测试内核，
//! 测量每条指令的延迟 (latency) 和吞吐 (throughput)。
//!
//! # 计时机制
//! `s_getreg_b32 sN, hwreg(HW_REG_SHADER_CYCLES)` → 32-bit shader cycle counter
//!
//! # 设计原则
//! - `ProbeOp` 枚举统一所有指令类型
//! - `build_latency_probe` / `build_throughput_probe` 通用构建器
//! - 单 wave (32 threads) 执行，消除 wave 间干扰
//! - GPU sweep 测试输出完整指令表

use super::ir::*;
use super::compile::T0Kernel;

// ============================================================================
// ProbeOp — 要测量的指令类型枚举
// ============================================================================

/// Instruction type to probe on GFX1100.
#[derive(Clone, Copy, Debug)]
pub enum ProbeOp {
    // ── VALU Float ──
    VAddF32,
    VMulF32,
    VFmaF32,
    VMaxF32,
    VMinF32,
    VSubF32,

    // ── VALU Integer / Logic ──
    VAddU32,
    VSubU32,
    VMulLoU32,
    VAndB32,
    VOrB32,
    VXorB32,
    VLshlrevB32,
    VLshrrevB32,
    VCndmaskB32,

    // ── TRANS (transcendental) ──
    VRcpF32,
    VRsqF32,
    VExpF32,
    VLog2F32,
    VSqrtF32,

    // ── Type conversion ──
    VCvtF32U32,
    VCvtU32F32,
    CvtPkBf16F32,

    // ── LDS B32 ──
    DsLoadB32,
    DsStoreB32,
    // ── LDS wider ──
    DsLoadB64,
    DsLoadB128,
    DsStoreB64,
    DsStoreB128,
    DsLoadU16,
    DsStoreB16,

    // ── VMEM (Global Memory) ──
    GlobalLoadB32,
    GlobalLoadB64,
    GlobalLoadB128,
    GlobalStoreB32,

    // ── Lane permute ──
    DsSwizzleXor1,
    DsSwizzleXor2,
    DsSwizzleXor4,
    DsSwizzleXor8,
    DsSwizzleXor16,
    VPermlanex16,

    // ── Wave collective ──
    WaveReduceAddF32,

    // ── VOP3 multi-source ──
    VAndOrB32,

    // ── WMMA variants ──
    WmmaF32BF16,
    WmmaF32F16,
    WmmaBF16BF16,

    // ── SALU ──
    SAddU32,
    SMulI32,
}

impl ProbeOp {
    /// Human-readable instruction name for the output table.
    pub fn name(self) -> &'static str {
        match self {
            Self::VAddF32        => "v_add_f32",
            Self::VMulF32        => "v_mul_f32",
            Self::VFmaF32        => "v_fma_f32",
            Self::VMaxF32        => "v_max_f32",
            Self::VMinF32        => "v_min_f32",
            Self::VSubF32        => "v_sub_f32",
            Self::VAddU32        => "v_add_u32",
            Self::VSubU32        => "v_sub_u32",
            Self::VMulLoU32      => "v_mul_lo_u32",
            Self::VAndB32        => "v_and_b32",
            Self::VOrB32         => "v_or_b32",
            Self::VXorB32        => "v_xor_b32",
            Self::VLshlrevB32    => "v_lshlrev_b32",
            Self::VLshrrevB32    => "v_lshrrev_b32",
            Self::VCndmaskB32    => "v_cndmask_b32",
            Self::VRcpF32        => "v_rcp_f32",
            Self::VRsqF32        => "v_rsq_f32",
            Self::VExpF32        => "v_exp_f32",
            Self::VLog2F32       => "v_log_f32",
            Self::VSqrtF32       => "v_sqrt_f32",
            Self::VCvtF32U32     => "v_cvt_f32_u32",
            Self::VCvtU32F32     => "v_cvt_u32_f32",
            Self::CvtPkBf16F32   => "v_cvt_pk_bf16",
            Self::DsLoadB32      => "ds_load_b32",
            Self::DsStoreB32     => "ds_store_b32",
            Self::DsLoadB64      => "ds_load_b64",
            Self::DsLoadB128     => "ds_load_b128",
            Self::DsStoreB64     => "ds_store_b64",
            Self::DsStoreB128    => "ds_store_b128",
            Self::DsLoadU16      => "ds_load_u16",
            Self::DsStoreB16     => "ds_store_b16",
            Self::GlobalLoadB32  => "global_ld_b32",
            Self::GlobalLoadB64  => "global_ld_b64",
            Self::GlobalLoadB128 => "global_ld_b128",
            Self::GlobalStoreB32 => "global_st_b32",
            Self::DsSwizzleXor1  => "ds_swizzle_x1",
            Self::DsSwizzleXor2  => "ds_swizzle_x2",
            Self::DsSwizzleXor4  => "ds_swizzle_x4",
            Self::DsSwizzleXor8  => "ds_swizzle_x8",
            Self::DsSwizzleXor16 => "ds_swizzle_x16",
            Self::VPermlanex16   => "v_permlane_x16",
            Self::WaveReduceAddF32 => "wave_red_add",
            Self::VAndOrB32      => "v_and_or_b32",
            Self::WmmaF32BF16    => "wmma_f32_bf16",
            Self::WmmaF32F16     => "wmma_f32_f16",
            Self::WmmaBF16BF16   => "wmma_bf16_bf16",
            Self::SAddU32        => "s_add_u32",
            Self::SMulI32        => "s_mul_i32",
        }
    }

    /// Expected pipeline category.
    pub fn expected_pipeline(self) -> &'static str {
        match self {
            Self::VAddF32 | Self::VMulF32 | Self::VFmaF32 | Self::VMaxF32 |
            Self::VMinF32 | Self::VSubF32 | Self::VAddU32 | Self::VSubU32 |
            Self::VMulLoU32 | Self::VAndB32 | Self::VOrB32 | Self::VXorB32 |
            Self::VLshlrevB32 | Self::VLshrrevB32 | Self::VCndmaskB32 => "VALU",
            Self::VRcpF32 | Self::VRsqF32 | Self::VExpF32 |
            Self::VLog2F32 | Self::VSqrtF32 => "TRANS",
            Self::VCvtF32U32 | Self::VCvtU32F32 | Self::CvtPkBf16F32 => "CVT",
            Self::DsLoadB32 | Self::DsStoreB32 | Self::DsLoadB64 | Self::DsLoadB128 |
            Self::DsStoreB64 | Self::DsStoreB128 | Self::DsLoadU16 | Self::DsStoreB16 => "LDS",
            Self::GlobalLoadB32 | Self::GlobalLoadB64 | Self::GlobalLoadB128 |
            Self::GlobalStoreB32 => "VMEM",
            Self::DsSwizzleXor1 | Self::DsSwizzleXor2 | Self::DsSwizzleXor4 |
            Self::DsSwizzleXor8 | Self::DsSwizzleXor16 | Self::VPermlanex16 => "LANE",
            Self::WaveReduceAddF32 => "WAVE",
            Self::VAndOrB32 => "VOP3",
            Self::WmmaF32BF16 | Self::WmmaF32F16 | Self::WmmaBF16BF16 => "WMMA",
            Self::SAddU32 | Self::SMulI32 => "SALU",
        }
    }

    /// Default number of operations for this probe type.
    pub fn default_n_ops(self) -> u32 {
        match self {
            // WMMA / Wave reduce: few iterations
            Self::WmmaF32BF16 | Self::WmmaF32F16 | Self::WmmaBF16BF16 => 64,
            Self::WaveReduceAddF32 => 32,
            // TRANS / LDS / VMEM: moderate
            Self::VRcpF32 | Self::VRsqF32 | Self::VExpF32 |
            Self::VLog2F32 | Self::VSqrtF32 => 256,
            Self::DsLoadB32 | Self::DsStoreB32 | Self::DsLoadB64 | Self::DsLoadB128 |
            Self::DsStoreB64 | Self::DsStoreB128 | Self::DsLoadU16 | Self::DsStoreB16 => 256,
            Self::GlobalLoadB32 | Self::GlobalLoadB64 | Self::GlobalLoadB128 |
            Self::GlobalStoreB32 => 128,
            // Everything else
            _ => 512,
        }
    }

    /// Whether this op supports throughput probing.
    pub fn supports_throughput(self) -> bool {
        match self {
            // LDS/VMEM/WMMA/WaveReduce/SALU have special constraints
            Self::DsLoadB32 | Self::DsStoreB32 | Self::DsLoadB64 | Self::DsLoadB128 |
            Self::DsStoreB64 | Self::DsStoreB128 | Self::DsLoadU16 | Self::DsStoreB16 |
            Self::GlobalLoadB32 | Self::GlobalLoadB64 | Self::GlobalLoadB128 |
            Self::GlobalStoreB32 |
            Self::WmmaF32BF16 | Self::WmmaF32F16 | Self::WmmaBF16BF16 |
            Self::WaveReduceAddF32 |
            Self::SAddU32 | Self::SMulI32 => false,
            _ => true,
        }
    }

    /// Whether this op needs LDS space.
    fn needs_lds(self) -> bool {
        matches!(self,
            Self::DsLoadB32 | Self::DsStoreB32 | Self::DsLoadB64 | Self::DsLoadB128 |
            Self::DsStoreB64 | Self::DsStoreB128 | Self::DsLoadU16 | Self::DsStoreB16)
    }

    /// Whether this op needs VMEM buffer.
    fn needs_vmem(self) -> bool {
        matches!(self,
            Self::GlobalLoadB32 | Self::GlobalLoadB64 | Self::GlobalLoadB128 |
            Self::GlobalStoreB32)
    }
}

/// All probes to run in a sweep.
pub const ALL_PROBES: &[ProbeOp] = &[
    // VALU float
    ProbeOp::VAddF32, ProbeOp::VMulF32, ProbeOp::VFmaF32,
    ProbeOp::VMaxF32, ProbeOp::VMinF32, ProbeOp::VSubF32,
    // VALU int/logic
    ProbeOp::VAddU32, ProbeOp::VSubU32, ProbeOp::VMulLoU32,
    ProbeOp::VAndB32, ProbeOp::VOrB32, ProbeOp::VXorB32,
    ProbeOp::VLshlrevB32, ProbeOp::VLshrrevB32, ProbeOp::VCndmaskB32,
    // TRANS
    ProbeOp::VRcpF32, ProbeOp::VRsqF32, ProbeOp::VExpF32,
    ProbeOp::VLog2F32, ProbeOp::VSqrtF32,
    // CVT
    ProbeOp::VCvtF32U32, ProbeOp::VCvtU32F32, ProbeOp::CvtPkBf16F32,
    // LDS
    ProbeOp::DsLoadB32, ProbeOp::DsStoreB32,
    ProbeOp::DsLoadB64, ProbeOp::DsLoadB128,
    ProbeOp::DsStoreB64, ProbeOp::DsStoreB128,
    ProbeOp::DsLoadU16, ProbeOp::DsStoreB16,
    // VMEM
    ProbeOp::GlobalLoadB32, ProbeOp::GlobalLoadB64,
    ProbeOp::GlobalLoadB128, ProbeOp::GlobalStoreB32,
    // Lane
    ProbeOp::DsSwizzleXor1, ProbeOp::DsSwizzleXor2,
    ProbeOp::DsSwizzleXor4, ProbeOp::DsSwizzleXor8,
    ProbeOp::DsSwizzleXor16, ProbeOp::VPermlanex16,
    // Wave
    ProbeOp::WaveReduceAddF32,
    // VOP3
    ProbeOp::VAndOrB32,
    // WMMA
    ProbeOp::WmmaF32BF16, ProbeOp::WmmaF32F16, ProbeOp::WmmaBF16BF16,
    // SALU
    ProbeOp::SAddU32, ProbeOp::SMulI32,
];

// ============================================================================
// Emit helpers — generate the instruction-under-test
// ============================================================================

/// Helper: get XOR swizzle offset for ds_swizzle variants.
fn swizzle_offset(op: ProbeOp) -> u16 {
    match op {
        ProbeOp::DsSwizzleXor1  => 0x041F,
        ProbeOp::DsSwizzleXor2  => 0x081F,
        ProbeOp::DsSwizzleXor4  => 0x101F,
        ProbeOp::DsSwizzleXor8  => 0x201F,
        ProbeOp::DsSwizzleXor16 => 0x401F,
        _ => 0x401F,
    }
}

/// Emit one serial (latency) instruction: dst depends on dst (RAW chain).
/// `chain` is the primary data register, `aux` is a helper.
/// For SALU probes, uses `s_chain`/`s_aux` SGPRs.
fn emit_latency_op(
    k: &mut T0Kernel, op: ProbeOp,
    chain: VReg, aux: VReg,
    s_chain: SReg, s_aux: SReg,
) {
    match op {
        // ── VALU float ──
        ProbeOp::VAddF32 => k.push(Op::VAddF32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VMulF32 => k.push(Op::VMulF32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::InlineFloat(0.5),
        }),
        ProbeOp::VFmaF32 => k.push(Op::VFmaF32 {
            dst: chain, src0: Operand::VReg(chain),
            src1: Operand::InlineFloat(0.5), src2: Operand::VReg(aux),
        }),
        ProbeOp::VMaxF32 => k.push(Op::VMaxF32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VMinF32 => k.push(Op::VMinF32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VSubF32 => k.push(Op::VSubF32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),

        // ── VALU int/logic ──
        ProbeOp::VAddU32 => k.push(Op::VAddU32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VSubU32 => k.push(Op::VSubU32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VMulLoU32 => k.push(Op::VMulLoU32 {
            dst: chain, src0: chain, src1: aux,
        }),
        ProbeOp::VAndB32 => k.push(Op::VAndB32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VOrB32 => k.push(Op::VOrB32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VXorB32 => k.push(Op::VXorB32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
        }),
        ProbeOp::VLshlrevB32 => k.push(Op::VLshlrevB32 {
            dst: chain, shift: 0, src: chain,
        }),
        ProbeOp::VLshrrevB32 => k.push(Op::VLshrrevB32 {
            dst: chain, shift: 0, src: chain,
        }),
        ProbeOp::VCndmaskB32 => k.push(Op::VCndmaskB32 {
            dst: chain, src_false: Operand::VReg(chain), src_true: Operand::VReg(aux),
        }),

        // ── TRANS ──
        ProbeOp::VRcpF32  => k.push(Op::VRcpF32  { dst: chain, src: chain }),
        ProbeOp::VRsqF32  => k.push(Op::VRsqF32  { dst: chain, src: chain }),
        ProbeOp::VExpF32  => k.push(Op::VExpF32  { dst: chain, src: chain }),
        ProbeOp::VLog2F32 => k.push(Op::VLog2F32 { dst: chain, src: chain }),
        ProbeOp::VSqrtF32 => k.push(Op::VSqrtF32 { dst: chain, src: chain }),

        // ── CVT (roundtrip to stabilize value domain) ──
        ProbeOp::VCvtF32U32 => {
            k.push(Op::VCvtF32U32 { dst: chain, src: chain });
            k.push(Op::VCvtU32F32 { dst: chain, src: chain });
        },
        ProbeOp::VCvtU32F32 => {
            k.push(Op::VCvtU32F32 { dst: chain, src: chain });
            k.push(Op::VCvtF32U32 { dst: chain, src: chain });
        },
        ProbeOp::CvtPkBf16F32 => k.push(Op::CvtPkBf16F32 {
            dst: chain, src0: chain, src1: aux,
        }),

        // ── LDS (store → wait → load → wait = serial dependency) ──
        ProbeOp::DsLoadB32 => {
            k.push(Op::DsStoreB32 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
            k.push(Op::DsLoadB32 { dst: chain, vaddr: aux, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },
        ProbeOp::DsStoreB32 => {
            k.push(Op::DsStoreB32 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },
        ProbeOp::DsLoadB64 => {
            k.push(Op::DsStoreB64 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
            k.push(Op::DsLoadB64 { dst: chain, vaddr: aux, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },
        ProbeOp::DsLoadB128 => {
            k.push(Op::DsStoreB128 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
            k.push(Op::DsLoadB128 { dst: chain, vaddr: aux, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },
        ProbeOp::DsStoreB64 => {
            k.push(Op::DsStoreB64 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },
        ProbeOp::DsStoreB128 => {
            k.push(Op::DsStoreB128 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },
        ProbeOp::DsLoadU16 => {
            k.push(Op::DsStoreB16 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
            k.push(Op::DsLoadU16 { dst: chain, vaddr: aux, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },
        ProbeOp::DsStoreB16 => {
            k.push(Op::DsStoreB16 { vaddr: aux, src: chain, offset: 0 });
            k.push(Op::WaitLgkmcnt(0));
        },

        // ── VMEM (serial: store → wait → load → wait) ──
        ProbeOp::GlobalLoadB32 => {
            k.global_store(aux, chain, Width::B32, 0);
            k.wait_vscnt(0);
            k.global_load(chain, aux, Width::B32, 0);
            k.wait_vmcnt(0);
        },
        ProbeOp::GlobalLoadB64 => {
            k.global_store(aux, chain, Width::B64, 0);
            k.wait_vscnt(0);
            k.global_load(chain, aux, Width::B64, 0);
            k.wait_vmcnt(0);
        },
        ProbeOp::GlobalLoadB128 => {
            k.global_store(aux, chain, Width::B128, 0);
            k.wait_vscnt(0);
            k.global_load(chain, aux, Width::B128, 0);
            k.wait_vmcnt(0);
        },
        ProbeOp::GlobalStoreB32 => {
            k.global_store(aux, chain, Width::B32, 0);
            k.wait_vscnt(0);
        },

        // ── Lane permute (swizzle into aux, add back to chain for dependency) ──
        ProbeOp::DsSwizzleXor1 | ProbeOp::DsSwizzleXor2 |
        ProbeOp::DsSwizzleXor4 | ProbeOp::DsSwizzleXor8 |
        ProbeOp::DsSwizzleXor16 => {
            k.push(Op::DsSwizzle {
                dst: aux, src: chain, offset: swizzle_offset(op),
            });
            k.push(Op::VAddF32 {
                dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
            });
        },
        ProbeOp::VPermlanex16 => {
            k.push(Op::VPermlanex16B32 { dst: aux, src: chain });
            k.push(Op::VAddF32 {
                dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
            });
        },

        // ── Wave reduce (single full reduction = 5×swizzle+add) ──
        ProbeOp::WaveReduceAddF32 => {
            k.push(Op::WaveReduceAddF32 { val: chain, tmp: aux });
        },

        // ── VOP3 ──
        ProbeOp::VAndOrB32 => k.push(Op::VAndOrB32 {
            dst: chain, src0: chain, literal: 0xFFFFFFFF, src2: aux,
        }),

        // ── WMMA (chain = accumulator C, aux = A, aux+8 = B) ──
        ProbeOp::WmmaF32BF16 => k.push(Op::Wmma {
            dst: chain, a: aux, b: VReg(aux.0 + 8),
            c: chain, format: WmmaFormat::BF16_F32,
        }),
        ProbeOp::WmmaF32F16 => k.push(Op::Wmma {
            dst: chain, a: aux, b: VReg(aux.0 + 8),
            c: chain, format: WmmaFormat::F16_F32,
        }),
        ProbeOp::WmmaBF16BF16 => k.push(Op::Wmma {
            dst: chain, a: aux, b: VReg(aux.0 + 8),
            c: chain, format: WmmaFormat::BF16_BF16,
        }),

        // ── SALU (use SGPRs, VGPR chain unused) ──
        ProbeOp::SAddU32 => k.push(Op::SAddU32 {
            dst: s_chain, src0: s_chain, src1: SOperand::SReg(s_aux),
        }),
        ProbeOp::SMulI32 => k.push(Op::SMulI32 {
            dst: s_chain, src0: s_chain, src1: s_aux,
        }),
    }
}

/// Emit one throughput (independent) instruction using register `reg`.
fn emit_throughput_op(k: &mut T0Kernel, op: ProbeOp, reg: VReg, aux: VReg) {
    match op {
        ProbeOp::VAddF32 => k.push(Op::VAddF32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VMulF32 => k.push(Op::VMulF32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::InlineFloat(0.5),
        }),
        ProbeOp::VFmaF32 => k.push(Op::VFmaF32 {
            dst: reg, src0: Operand::VReg(reg),
            src1: Operand::InlineFloat(0.5), src2: Operand::VReg(aux),
        }),
        ProbeOp::VMaxF32 => k.push(Op::VMaxF32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VMinF32 => k.push(Op::VMinF32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VSubF32 => k.push(Op::VSubF32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VAddU32 => k.push(Op::VAddU32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VSubU32 => k.push(Op::VSubU32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VMulLoU32 => k.push(Op::VMulLoU32 {
            dst: reg, src0: reg, src1: aux,
        }),
        ProbeOp::VAndB32 => k.push(Op::VAndB32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VOrB32 => k.push(Op::VOrB32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VXorB32 => k.push(Op::VXorB32 {
            dst: reg, src0: Operand::VReg(reg), src1: Operand::VReg(aux),
        }),
        ProbeOp::VLshlrevB32 => k.push(Op::VLshlrevB32 {
            dst: reg, shift: 0, src: reg,
        }),
        ProbeOp::VLshrrevB32 => k.push(Op::VLshrrevB32 {
            dst: reg, shift: 0, src: reg,
        }),
        ProbeOp::VCndmaskB32 => k.push(Op::VCndmaskB32 {
            dst: reg, src_false: Operand::VReg(reg), src_true: Operand::VReg(aux),
        }),
        ProbeOp::VRcpF32  => k.push(Op::VRcpF32  { dst: reg, src: reg }),
        ProbeOp::VRsqF32  => k.push(Op::VRsqF32  { dst: reg, src: reg }),
        ProbeOp::VExpF32  => k.push(Op::VExpF32  { dst: reg, src: reg }),
        ProbeOp::VLog2F32 => k.push(Op::VLog2F32 { dst: reg, src: reg }),
        ProbeOp::VSqrtF32 => k.push(Op::VSqrtF32 { dst: reg, src: reg }),
        ProbeOp::VCvtF32U32 => k.push(Op::VCvtF32U32 { dst: reg, src: reg }),
        ProbeOp::VCvtU32F32 => k.push(Op::VCvtU32F32 { dst: reg, src: reg }),
        ProbeOp::CvtPkBf16F32 => k.push(Op::CvtPkBf16F32 {
            dst: reg, src0: reg, src1: aux,
        }),
        // Lane permute (throughput: independent per reg)
        ProbeOp::DsSwizzleXor1 | ProbeOp::DsSwizzleXor2 |
        ProbeOp::DsSwizzleXor4 | ProbeOp::DsSwizzleXor8 |
        ProbeOp::DsSwizzleXor16 => k.push(Op::DsSwizzle {
            dst: reg, src: reg, offset: swizzle_offset(op),
        }),
        ProbeOp::VPermlanex16 => k.push(Op::VPermlanex16B32 { dst: reg, src: reg }),
        ProbeOp::VAndOrB32 => k.push(Op::VAndOrB32 {
            dst: reg, src0: reg, literal: 0xFFFFFFFF, src2: aux,
        }),
        // Ops without throughput probe — fallback to latency
        _ => {
            let s_chain = SReg(10); let s_aux = SReg(11);
            emit_latency_op(k, op, reg, aux, s_chain, s_aux);
        }
    }
}

// ============================================================================
// Generic probe kernel builders
// ============================================================================

/// Build a latency probe kernel: serial dependency chain measuring per-op latency.
pub fn build_latency_probe(op: ProbeOp, n_ops: u32) -> T0Kernel {
    let name = format!("probe_lat_{}", op.name().replace(' ', ""));
    let mut k = T0Kernel::new(&name);
    k.set_wg_size(32);

    // Kernargs
    let out_ptr = k.arg_ptr("out");
    // VMEM probes need a scratch buffer
    let vmem_ptr = if op.needs_vmem() {
        Some(k.arg_ptr("vmem_buf"))
    } else {
        None
    };
    k.emit_arg_loads();

    let v_start = k.alloc_vreg();
    let v_end = k.alloc_vreg();

    // Allocate SGPR chain for SALU probes
    let s_chain = k.alloc_sreg();
    let s_aux = k.alloc_sreg();

    // Special setup per category
    let (chain, aux) = if matches!(op, ProbeOp::WmmaF32BF16 | ProbeOp::WmmaF32F16 | ProbeOp::WmmaBF16BF16) {
        let c = k.alloc_vreg_array(8, Alignment::Align2);
        let a = k.alloc_vreg_array(8, Alignment::Align2);
        let b = k.alloc_vreg_array(8, Alignment::Align2);
        k.push(Op::VCvtF32U32 { dst: c, src: VReg(0) });
        for i in 1..8 {
            k.push(Op::VMov { dst: VReg(c.0 + i), src: Operand::VReg(c) });
        }
        for i in 0..8 {
            k.push(Op::VMov { dst: VReg(a.0 + i), src: Operand::InlineFloat(0.5) });
            k.push(Op::VMov { dst: VReg(b.0 + i), src: Operand::InlineFloat(0.5) });
        }
        (c, a)
    } else if op.needs_lds() {
        k.set_lds_size(2048); // 2KB for wider loads
        let chain = k.alloc_vreg_array(4, Alignment::Align4); // up to b128
        let aux = k.alloc_vreg();
        // aux = tid * 16 (unique 16-byte aligned LDS address per lane)
        k.push(Op::VLshlrevB32 { dst: aux, shift: 4, src: VReg(0) });
        k.push(Op::VCvtF32U32 { dst: chain, src: VReg(0) });
        // Init extra regs for wider stores
        for i in 1..4u32 {
            k.push(Op::VMov { dst: VReg(chain.0 + i), src: Operand::VReg(chain) });
        }
        (chain, aux)
    } else if op.needs_vmem() {
        let chain = k.alloc_vreg_array(4, Alignment::Align4);
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        // Load VMEM buffer pointer into VGPRs
        let vp = vmem_ptr.unwrap();
        k.push(Op::VMovFromSgpr { dst: addr, src: SReg(vp.0) });
        k.push(Op::VMovFromSgpr { dst: VReg(addr.0 + 1), src: SReg(vp.0 + 1) });
        // Add tid*16 offset for per-lane addressing
        let v_off = k.alloc_vreg();
        k.push(Op::VLshlrevB32 { dst: v_off, shift: 4, src: VReg(0) });
        k.v_add_co(addr, addr, v_off);
        k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
        // Init data
        k.push(Op::VCvtF32U32 { dst: chain, src: VReg(0) });
        for i in 1..4u32 {
            k.push(Op::VMov { dst: VReg(chain.0 + i), src: Operand::VReg(chain) });
        }
        (chain, addr) // aux = addr pair for VMEM
    } else if matches!(op, ProbeOp::SAddU32 | ProbeOp::SMulI32) {
        // SALU: init SGPRs, VGPRs unused for actual work
        let chain = k.alloc_vreg();
        let aux = k.alloc_vreg();
        // Init SGPRs from inline constants
        k.push(Op::SMov { dst: s_chain, src: SOperand::InlineInt(7) });
        k.push(Op::SMov { dst: s_aux, src: SOperand::InlineInt(3) });
        // Still need chain VGPR for elapsed computation
        k.push(Op::VCvtF32U32 { dst: chain, src: VReg(0) });
        (chain, aux)
    } else {
        let chain = k.alloc_vreg();
        let aux = k.alloc_vreg();
        k.push(Op::VCvtF32U32 { dst: chain, src: VReg(0) });
        k.push(Op::VAddF32 {
            dst: chain, src0: Operand::VReg(chain), src1: Operand::InlineFloat(1.0),
        });
        k.push(Op::VAddF32 {
            dst: aux, src0: Operand::VReg(chain), src1: Operand::InlineFloat(0.5),
        });
        (chain, aux)
    };

    // === Read start ===
    k.push(Op::ReadShaderCycles { dst: v_start });

    // === Serial instruction chain ===
    for _ in 0..n_ops {
        emit_latency_op(&mut k, op, chain, aux, s_chain, s_aux);
    }

    // === Read end ===
    k.push(Op::ReadShaderCycles { dst: v_end });

    emit_store_elapsed(&mut k, v_start, v_end, out_ptr);
    k
}

/// Build a throughput probe kernel: N independent instructions, no RAW dependency.
pub fn build_throughput_probe(op: ProbeOp, n_ops: u32) -> T0Kernel {
    let name = format!("probe_thr_{}", op.name().replace(' ', ""));
    let mut k = T0Kernel::new(&name);
    k.set_wg_size(32);

    let out_ptr = k.arg_ptr("out");
    k.emit_arg_loads();

    let v_start = k.alloc_vreg();
    let v_end = k.alloc_vreg();

    let n_regs = n_ops.min(32) as usize;
    let mut regs: Vec<VReg> = Vec::with_capacity(n_regs);
    for _ in 0..n_regs {
        regs.push(k.alloc_vreg());
    }

    let aux = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: aux, src: VReg(0) });
    k.push(Op::VAddF32 {
        dst: aux, src0: Operand::VReg(aux), src1: Operand::InlineFloat(1.0),
    });

    let v_tid_f32 = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: v_tid_f32, src: VReg(0) });
    for (i, vr) in regs.iter().enumerate() {
        k.push(Op::VAddF32 {
            dst: *vr,
            src0: Operand::VReg(v_tid_f32),
            src1: Operand::InlineFloat((i + 1) as f32),
        });
    }

    // === Read start ===
    k.push(Op::ReadShaderCycles { dst: v_start });

    for i in 0..n_ops {
        let reg = regs[i as usize % n_regs];
        emit_throughput_op(&mut k, op, reg, aux);
    }

    // === Read end ===
    k.push(Op::ReadShaderCycles { dst: v_end });

    emit_store_elapsed(&mut k, v_start, v_end, out_ptr);
    k
}

/// Shared epilogue: compute elapsed = end - start, store to out[0] (lane 0 only).
fn emit_store_elapsed(k: &mut T0Kernel, v_start: VReg, v_end: VReg, out_ptr: SRegPair) {
    let v_elapsed = k.alloc_vreg();
    k.v_sub_u32(v_elapsed, v_end, v_start);

    k.v_cmp_eq_u32_imm(VReg(0), 0);
    let saved = k.alloc_sreg();
    k.save_exec(saved);

    let v_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.push(Op::VMovFromSgpr { dst: v_addr, src: SReg(out_ptr.0) });
    k.push(Op::VMovFromSgpr { dst: VReg(v_addr.0 + 1), src: SReg(out_ptr.0 + 1) });
    k.global_store(v_addr, v_elapsed, Width::B32, 0);

    k.restore_exec(saved);
    k.wait_vscnt(0);
    k.endpgm();
}

// ============================================================================
// Kept for backwards compatibility
// ============================================================================

/// Build a probe kernel that measures global_load_b32 latency via pointer chasing.
pub fn build_probe_vmem_latency(n_chases: u32) -> T0Kernel {
    let mut k = T0Kernel::new("probe_vmem_latency");
    k.set_wg_size(32);

    let data_ptr = k.arg_ptr("data");
    let out_ptr = k.arg_ptr("out");
    k.emit_arg_loads();

    let v_start = k.alloc_vreg();
    let v_end = k.alloc_vreg();

    let v_data_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.push(Op::VMovFromSgpr { dst: v_data_addr, src: SReg(data_ptr.0) });
    k.push(Op::VMovFromSgpr { dst: VReg(v_data_addr.0 + 1), src: SReg(data_ptr.0 + 1) });

    let v_idx = k.alloc_vreg();
    k.global_load(v_idx, v_data_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    k.push(Op::ReadShaderCycles { dst: v_start });

    let v_byte_off = k.alloc_vreg();
    let v_chase_addr = k.alloc_vreg_array(2, Alignment::Align2);

    for _ in 0..n_chases {
        k.v_lshlrev_b32(v_byte_off, 2, v_idx);
        k.push(Op::VMovFromSgpr { dst: v_chase_addr, src: SReg(data_ptr.0) });
        k.push(Op::VMovFromSgpr { dst: VReg(v_chase_addr.0 + 1), src: SReg(data_ptr.0 + 1) });
        k.v_add_co(v_chase_addr, v_chase_addr, v_byte_off);
        k.v_add_co_ci(VReg(v_chase_addr.0 + 1), VReg(v_chase_addr.0 + 1));
        k.global_load(v_idx, v_chase_addr, Width::B32, 0);
        k.wait_vmcnt(0);
    }

    k.push(Op::ReadShaderCycles { dst: v_end });

    emit_store_elapsed(&mut k, v_start, v_end, out_ptr);
    k
}

// ============================================================================
// Probe result + table
// ============================================================================

#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub op: &'static str,
    pub pipeline: &'static str,
    pub n_ops: u32,
    pub latency_cycles: u32,
    pub throughput_cycles: Option<u32>,
}

impl ProbeResult {
    pub fn lat_per_op(&self) -> f64 {
        self.latency_cycles as f64 / self.n_ops as f64
    }
    pub fn thr_per_op(&self) -> Option<f64> {
        self.throughput_cycles.map(|c| c as f64 / self.n_ops as f64)
    }
}

pub fn print_probe_table(results: &[ProbeResult]) {
    eprintln!("╔═══════════════════╦════════╦══════════╦══════════╦══════════╗");
    eprintln!("║ Instruction       ║ N_ops  ║ Lat/op   ║ Thr/op   ║ Pipeline ║");
    eprintln!("╠═══════════════════╬════════╬══════════╬══════════╬══════════╣");
    for r in results {
        let thr = match r.thr_per_op() {
            Some(t) => format!("{:>7.2}", t),
            None    => "    N/A".to_string(),
        };
        eprintln!("║ {:17} ║ {:>5} ║ {:>7.2} ║ {} ║ {:>8} ║",
            r.op, r.n_ops, r.lat_per_op(), thr, r.pipeline);
    }
    eprintln!("╚═══════════════════╩════════╩══════════╩══════════╩══════════╝");
}

// ============================================================================
// Pipeline Overlap Probes — 验证不同 pipeline 是否可并行执行
// ============================================================================
//
// 设计：构建 3 个内核：
//   1. baseline_A: 只执行 N 个 A 类指令 → T_a
//   2. baseline_B: 只执行 N 个 B 类指令 → T_b
//   3. interleave: 交替 A+B 指令 → T_mix
//
// 分析：
//   overlap_ratio = 1 - (T_mix - max(T_a, T_b)) / min(T_a, T_b)
//   ratio ≈ 1.0 → 完全并行
//   ratio ≈ 0.0 → 完全串行

/// Pipeline overlap test type.
#[derive(Clone, Copy, Debug)]
pub enum OverlapTest {
    /// VALU (v_add_f32) interleaved with VMEM (global_load_b32)
    ValuVmem,
    /// VALU (v_add_f32) interleaved with LDS (ds_load_b32)
    ValuLds,
    /// VALU (v_add_f32) interleaved with TRANS (v_rcp_f32)
    ValuTrans,
}

impl OverlapTest {
    pub fn name(self) -> &'static str {
        match self {
            Self::ValuVmem  => "VALU+VMEM",
            Self::ValuLds   => "VALU+LDS",
            Self::ValuTrans => "VALU+TRANS",
        }
    }
}

/// Result of an overlap test.
#[derive(Clone, Debug)]
pub struct OverlapResult {
    pub test: &'static str,
    pub n_pairs: u32,
    pub cycles_a_only: u32,      // baseline A
    pub cycles_b_only: u32,      // baseline B
    pub cycles_interleaved: u32, // interleaved A+B
    pub overlap_ratio: f64,      // 1.0 = full overlap, 0.0 = serial
}

/// Build a kernel that interleaves VALU ops with a second pipeline.
/// `n_pairs`: number of A+B instruction pairs.
///
/// Returns (baseline_a_kernel, baseline_b_kernel, interleaved_kernel).
pub fn build_overlap_probes(test: OverlapTest, n_pairs: u32) -> (T0Kernel, T0Kernel, T0Kernel) {
    let (mut ka, mut kb, mut kmix) = match test {
        OverlapTest::ValuVmem => build_overlap_valu_vmem(n_pairs),
        OverlapTest::ValuLds  => build_overlap_valu_lds(n_pairs),
        OverlapTest::ValuTrans => build_overlap_valu_trans(n_pairs),
    };
    // Overlap probes must bypass the optimizer — the scheduler would reorder
    // the interleaved instructions, defeating the measurement purpose.
    ka.set_skip_optimize(true);
    kb.set_skip_optimize(true);
    kmix.set_skip_optimize(true);
    (ka, kb, kmix)
}

/// VALU + VMEM overlap: independent v_add_f32 interleaved with global_load_b32.
fn build_overlap_valu_vmem(n_pairs: u32) -> (T0Kernel, T0Kernel, T0Kernel) {
    // === Baseline A: N×v_add_f32 (serial chain) ===
    let mut ka = T0Kernel::new("overlap_valu_only_vmem");
    ka.set_wg_size(32);
    let out_a = ka.arg_ptr("out");
    ka.emit_arg_loads();
    let v_start_a = ka.alloc_vreg();
    let v_end_a = ka.alloc_vreg();
    let chain_a = ka.alloc_vreg();
    let aux_a = ka.alloc_vreg();
    ka.push(Op::VCvtF32U32 { dst: chain_a, src: VReg(0) });
    ka.push(Op::VAddF32 { dst: aux_a, src0: Operand::VReg(chain_a), src1: Operand::InlineFloat(0.5) });
    ka.push(Op::ReadShaderCycles { dst: v_start_a });
    for _ in 0..n_pairs {
        ka.push(Op::VAddF32 {
            dst: chain_a, src0: Operand::VReg(chain_a), src1: Operand::VReg(aux_a),
        });
    }
    ka.push(Op::ReadShaderCycles { dst: v_end_a });
    emit_store_elapsed(&mut ka, v_start_a, v_end_a, out_a);

    // === Baseline B: N×global_load_b32 (independent, from same address) ===
    let mut kb = T0Kernel::new("overlap_vmem_only");
    kb.set_wg_size(32);
    let out_b = kb.arg_ptr("out");
    let vmem_ptr_b = kb.arg_ptr("vmem_buf");
    kb.emit_arg_loads();
    let v_start_b = kb.alloc_vreg();
    let v_end_b = kb.alloc_vreg();
    let addr_b = kb.alloc_vreg_array(2, Alignment::Align2);
    kb.push(Op::VMovFromSgpr { dst: addr_b, src: SReg(vmem_ptr_b.0) });
    kb.push(Op::VMovFromSgpr { dst: VReg(addr_b.0 + 1), src: SReg(vmem_ptr_b.0 + 1) });
    // Per-lane offset
    let v_off_b = kb.alloc_vreg();
    kb.push(Op::VLshlrevB32 { dst: v_off_b, shift: 2, src: VReg(0) });
    kb.v_add_co(addr_b, addr_b, v_off_b);
    kb.v_add_co_ci(VReg(addr_b.0 + 1), VReg(addr_b.0 + 1));
    let ld_dst_b = kb.alloc_vreg();
    kb.push(Op::ReadShaderCycles { dst: v_start_b });
    for _ in 0..n_pairs {
        kb.global_load(ld_dst_b, addr_b, Width::B32, 0);
    }
    kb.wait_vmcnt(0);
    kb.push(Op::ReadShaderCycles { dst: v_end_b });
    emit_store_elapsed(&mut kb, v_start_b, v_end_b, out_b);

    // === Interleaved: alternate v_add_f32 + global_load_b32 ===
    let mut km = T0Kernel::new("overlap_valu_vmem_mix");
    km.set_wg_size(32);
    let out_m = km.arg_ptr("out");
    let vmem_ptr_m = km.arg_ptr("vmem_buf");
    km.emit_arg_loads();
    let v_start_m = km.alloc_vreg();
    let v_end_m = km.alloc_vreg();
    let chain_m = km.alloc_vreg();
    let aux_m = km.alloc_vreg();
    km.push(Op::VCvtF32U32 { dst: chain_m, src: VReg(0) });
    km.push(Op::VAddF32 { dst: aux_m, src0: Operand::VReg(chain_m), src1: Operand::InlineFloat(0.5) });
    let addr_m = km.alloc_vreg_array(2, Alignment::Align2);
    km.push(Op::VMovFromSgpr { dst: addr_m, src: SReg(vmem_ptr_m.0) });
    km.push(Op::VMovFromSgpr { dst: VReg(addr_m.0 + 1), src: SReg(vmem_ptr_m.0 + 1) });
    let v_off_m = km.alloc_vreg();
    km.push(Op::VLshlrevB32 { dst: v_off_m, shift: 2, src: VReg(0) });
    km.v_add_co(addr_m, addr_m, v_off_m);
    km.v_add_co_ci(VReg(addr_m.0 + 1), VReg(addr_m.0 + 1));
    let ld_dst_m = km.alloc_vreg();
    km.push(Op::ReadShaderCycles { dst: v_start_m });
    for _ in 0..n_pairs {
        // VALU — serial chain (dependent)
        km.push(Op::VAddF32 {
            dst: chain_m, src0: Operand::VReg(chain_m), src1: Operand::VReg(aux_m),
        });
        // VMEM — independent load (no dependency on chain)
        km.global_load(ld_dst_m, addr_m, Width::B32, 0);
    }
    km.wait_vmcnt(0);
    km.push(Op::ReadShaderCycles { dst: v_end_m });
    emit_store_elapsed(&mut km, v_start_m, v_end_m, out_m);

    (ka, kb, km)
}

/// VALU + LDS overlap.
fn build_overlap_valu_lds(n_pairs: u32) -> (T0Kernel, T0Kernel, T0Kernel) {
    // === Baseline A: N×v_add_f32 (serial chain) ===
    let mut ka = T0Kernel::new("overlap_valu_only_lds");
    ka.set_wg_size(32);
    let out_a = ka.arg_ptr("out");
    ka.emit_arg_loads();
    let v_start_a = ka.alloc_vreg();
    let v_end_a = ka.alloc_vreg();
    let chain_a = ka.alloc_vreg();
    let aux_a = ka.alloc_vreg();
    ka.push(Op::VCvtF32U32 { dst: chain_a, src: VReg(0) });
    ka.push(Op::VAddF32 { dst: aux_a, src0: Operand::VReg(chain_a), src1: Operand::InlineFloat(0.5) });
    ka.push(Op::ReadShaderCycles { dst: v_start_a });
    for _ in 0..n_pairs {
        ka.push(Op::VAddF32 {
            dst: chain_a, src0: Operand::VReg(chain_a), src1: Operand::VReg(aux_a),
        });
    }
    ka.push(Op::ReadShaderCycles { dst: v_end_a });
    emit_store_elapsed(&mut ka, v_start_a, v_end_a, out_a);

    // === Baseline B: N×ds_load_b32 (independent) ===
    let mut kb = T0Kernel::new("overlap_lds_only");
    kb.set_wg_size(32);
    kb.set_lds_size(1024);
    let out_b = kb.arg_ptr("out");
    kb.emit_arg_loads();
    let v_start_b = kb.alloc_vreg();
    let v_end_b = kb.alloc_vreg();
    let lds_addr = kb.alloc_vreg();
    let lds_dst = kb.alloc_vreg();
    // Init LDS: store something first
    kb.push(Op::VLshlrevB32 { dst: lds_addr, shift: 2, src: VReg(0) }); // tid*4
    kb.push(Op::DsStoreB32 { vaddr: lds_addr, src: VReg(0), offset: 0 });
    kb.push(Op::WaitLgkmcnt(0));
    kb.push(Op::ReadShaderCycles { dst: v_start_b });
    for _ in 0..n_pairs {
        kb.push(Op::DsLoadB32 { dst: lds_dst, vaddr: lds_addr, offset: 0 });
    }
    kb.push(Op::WaitLgkmcnt(0));
    kb.push(Op::ReadShaderCycles { dst: v_end_b });
    emit_store_elapsed(&mut kb, v_start_b, v_end_b, out_b);

    // === Interleaved: alternate v_add_f32 + ds_load_b32 ===
    let mut km = T0Kernel::new("overlap_valu_lds_mix");
    km.set_wg_size(32);
    km.set_lds_size(1024);
    let out_m = km.arg_ptr("out");
    km.emit_arg_loads();
    let v_start_m = km.alloc_vreg();
    let v_end_m = km.alloc_vreg();
    let chain_m = km.alloc_vreg();
    let aux_m = km.alloc_vreg();
    let lds_addr_m = km.alloc_vreg();
    let lds_dst_m = km.alloc_vreg();
    km.push(Op::VCvtF32U32 { dst: chain_m, src: VReg(0) });
    km.push(Op::VAddF32 { dst: aux_m, src0: Operand::VReg(chain_m), src1: Operand::InlineFloat(0.5) });
    km.push(Op::VLshlrevB32 { dst: lds_addr_m, shift: 2, src: VReg(0) });
    km.push(Op::DsStoreB32 { vaddr: lds_addr_m, src: VReg(0), offset: 0 });
    km.push(Op::WaitLgkmcnt(0));
    km.push(Op::ReadShaderCycles { dst: v_start_m });
    for _ in 0..n_pairs {
        km.push(Op::VAddF32 {
            dst: chain_m, src0: Operand::VReg(chain_m), src1: Operand::VReg(aux_m),
        });
        km.push(Op::DsLoadB32 { dst: lds_dst_m, vaddr: lds_addr_m, offset: 0 });
    }
    km.push(Op::WaitLgkmcnt(0));
    km.push(Op::ReadShaderCycles { dst: v_end_m });
    emit_store_elapsed(&mut km, v_start_m, v_end_m, out_m);

    (ka, kb, km)
}

/// VALU + TRANS overlap.
fn build_overlap_valu_trans(n_pairs: u32) -> (T0Kernel, T0Kernel, T0Kernel) {
    // === Baseline A: N×v_add_f32 (serial chain) ===
    let mut ka = T0Kernel::new("overlap_valu_only_trans");
    ka.set_wg_size(32);
    let out_a = ka.arg_ptr("out");
    ka.emit_arg_loads();
    let v_start_a = ka.alloc_vreg();
    let v_end_a = ka.alloc_vreg();
    let chain_a = ka.alloc_vreg();
    let aux_a = ka.alloc_vreg();
    ka.push(Op::VCvtF32U32 { dst: chain_a, src: VReg(0) });
    ka.push(Op::VAddF32 { dst: aux_a, src0: Operand::VReg(chain_a), src1: Operand::InlineFloat(1.0) });
    ka.push(Op::ReadShaderCycles { dst: v_start_a });
    for _ in 0..n_pairs {
        ka.push(Op::VAddF32 {
            dst: chain_a, src0: Operand::VReg(chain_a), src1: Operand::VReg(aux_a),
        });
    }
    ka.push(Op::ReadShaderCycles { dst: v_end_a });
    emit_store_elapsed(&mut ka, v_start_a, v_end_a, out_a);

    // === Baseline B: N×v_rcp_f32 (serial chain — TRANS depends on TRANS) ===
    let mut kb = T0Kernel::new("overlap_trans_only");
    kb.set_wg_size(32);
    let out_b = kb.arg_ptr("out");
    kb.emit_arg_loads();
    let v_start_b = kb.alloc_vreg();
    let v_end_b = kb.alloc_vreg();
    let chain_b = kb.alloc_vreg();
    kb.push(Op::VCvtF32U32 { dst: chain_b, src: VReg(0) });
    kb.push(Op::VAddF32 { dst: chain_b, src0: Operand::VReg(chain_b), src1: Operand::InlineFloat(1.0) });
    kb.push(Op::ReadShaderCycles { dst: v_start_b });
    for _ in 0..n_pairs {
        kb.push(Op::VRcpF32 { dst: chain_b, src: chain_b });
    }
    kb.push(Op::ReadShaderCycles { dst: v_end_b });
    emit_store_elapsed(&mut kb, v_start_b, v_end_b, out_b);

    // === Interleaved: alternate v_add_f32 (chain_a) + v_rcp_f32 (chain_b) ===
    // These are independent chains — if VALU and TRANS are separate pipelines, they overlap.
    let mut km = T0Kernel::new("overlap_valu_trans_mix");
    km.set_wg_size(32);
    let out_m = km.arg_ptr("out");
    km.emit_arg_loads();
    let v_start_m = km.alloc_vreg();
    let v_end_m = km.alloc_vreg();
    let chain_m_valu = km.alloc_vreg();
    let aux_m = km.alloc_vreg();
    let chain_m_trans = km.alloc_vreg();
    km.push(Op::VCvtF32U32 { dst: chain_m_valu, src: VReg(0) });
    km.push(Op::VAddF32 { dst: aux_m, src0: Operand::VReg(chain_m_valu), src1: Operand::InlineFloat(0.5) });
    km.push(Op::VAddF32 { dst: chain_m_trans, src0: Operand::VReg(chain_m_valu), src1: Operand::InlineFloat(1.0) });
    km.push(Op::ReadShaderCycles { dst: v_start_m });
    for _ in 0..n_pairs {
        km.push(Op::VAddF32 {
            dst: chain_m_valu, src0: Operand::VReg(chain_m_valu), src1: Operand::VReg(aux_m),
        });
        km.push(Op::VRcpF32 { dst: chain_m_trans, src: chain_m_trans });
    }
    km.push(Op::ReadShaderCycles { dst: v_end_m });
    emit_store_elapsed(&mut km, v_start_m, v_end_m, out_m);

    (ka, kb, km)
}

/// Compute overlap ratio from three cycle counts.
/// Returns (overlap_ratio, OverlapResult).
pub fn compute_overlap(test: OverlapTest, n_pairs: u32,
                       t_a: u32, t_b: u32, t_mix: u32) -> OverlapResult {
    let t_max = t_a.max(t_b) as f64;
    let t_sum = (t_a as f64) + (t_b as f64);
    // overlap_ratio: 1.0 = perfect overlap, 0.0 = serial
    let ratio = if t_sum > t_max {
        1.0 - ((t_mix as f64) - t_max) / (t_sum - t_max)
    } else {
        1.0 // degenerate case
    };
    OverlapResult {
        test: test.name(),
        n_pairs,
        cycles_a_only: t_a,
        cycles_b_only: t_b,
        cycles_interleaved: t_mix,
        overlap_ratio: ratio.clamp(0.0, 1.5), // allow slightly >1 from measurement noise
    }
}

pub fn print_overlap_table(results: &[OverlapResult]) {
    eprintln!("╔═══════════════╦════════╦═════════╦═════════╦═══════════╦══════════════╗");
    eprintln!("║ Test          ║ N_pair ║ T_a     ║ T_b     ║ T_mix     ║ Overlap %    ║");
    eprintln!("╠═══════════════╬════════╬═════════╬═════════╬═══════════╬══════════════╣");
    for r in results {
        let verdict = if r.overlap_ratio > 0.75 { "✓ PARALLEL" }
                      else if r.overlap_ratio > 0.3 { "~ PARTIAL" }
                      else { "✗ SERIAL" };
        eprintln!("║ {:13} ║ {:>5} ║ {:>7} ║ {:>7} ║ {:>9} ║ {:>5.1}% {:>5} ║",
            r.test, r.n_pairs, r.cycles_a_only, r.cycles_b_only,
            r.cycles_interleaved, r.overlap_ratio * 100.0, verdict);
    }
    eprintln!("╚═══════════════╩════════╩═════════╩═════════╩═══════════╩══════════════╝");
}

// ============================================================================
// Multi-Wave Overlap Probes — 验证多 wave 占用率下的延迟隐藏效果
// ============================================================================
//
// 设计：
//   构建一个 "混合 compute+memory" 内核，每个 wave 做 N 次：
//     global_load → waitcnt → K×v_add_f32 → global_store → waitcnt
//   改变 WG 大小 (1/2/4/8 waves) 来测量吞吐缩放。
//
//   如果多 wave 隐藏有效：
//     2-wave 总时间 ≈ 1-wave 时间 → N 倍吞吐
//   如果无隐藏：
//     2-wave 总时间 ≈ 2 × 1-wave 时间 → 吞吐不变
//
//   timing: s_barrier 同步所有 wave → wave 0 读 shader cycles → 写输出

/// Result of a multi-wave throughput scaling test.
#[derive(Clone, Debug)]
pub struct MultiWaveResult {
    pub n_waves: u32,
    pub n_iters: u32,
    pub valu_per_iter: u32,
    pub total_cycles: u32,
    pub cycles_per_wave: f64,       // total / n_waves
    pub speedup_vs_1wave: f64,      // T_1wave / T_Nwave (ideal = n_waves)
    pub efficiency: f64,            // speedup / n_waves (1.0 = perfect scaling)
}

/// Build a multi-wave mixed compute+memory kernel.
///
/// Each wave does `n_iters` iterations of:
///   global_load_b32 → s_waitcnt vmcnt(0) → K×v_add_f32 → global_store_b32 → s_waitcnt vscnt(0)
///
/// `n_waves`: number of waves per WG (WG_SIZE = n_waves × 32)
/// `n_iters`: iterations per wave
/// `valu_per_iter`: number of v_add_f32 ops between load and store
pub fn build_multiwave_mixed_kernel(
    n_waves: u32,
    n_iters: u32,
    valu_per_iter: u32,
) -> T0Kernel {
    let name = format!("mw_mixed_{}w_{}i_{}v", n_waves, n_iters, valu_per_iter);
    let mut k = T0Kernel::new(&name);
    let wg_size = n_waves * 32;
    k.set_wg_size(wg_size);
    // NOTE: do NOT skip optimizer — skip_optimize causes GPU hang

    // Kernargs: out_ptr (u64), vmem_buf (u64)
    let out_ptr = k.arg_ptr("out");
    let vmem_ptr = k.arg_ptr("vmem_buf");
    k.emit_arg_loads();

    let v_start = k.alloc_vreg();
    let v_end = k.alloc_vreg();

    // Setup VMEM address: vmem_buf + thread_id * 4
    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.push(Op::VMovFromSgpr { dst: addr, src: SReg(vmem_ptr.0) });
    k.push(Op::VMovFromSgpr { dst: VReg(addr.0 + 1), src: SReg(vmem_ptr.0 + 1) });
    let v_off = k.alloc_vreg();
    k.push(Op::VLshlrevB32 { dst: v_off, shift: 2, src: VReg(0) }); // tid*4
    k.v_add_co(addr, addr, v_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

    // Data register for compute chain
    let chain = k.alloc_vreg();
    let aux = k.alloc_vreg();
    let ld_dst = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: chain, src: VReg(0) });
    k.push(Op::VAddF32 { dst: aux, src0: Operand::VReg(chain), src1: Operand::InlineFloat(0.5) });

    // Warmup barrier: ensure all waves are started before timing
    if n_waves > 1 {
        k.push(Op::RawAsm("s_barrier".to_string()));
    }

    // === Read start ===
    k.push(Op::ReadShaderCycles { dst: v_start });

    // === Main loop: N iterations of load-compute-store ===
    for _ in 0..n_iters {
        // Load from VMEM
        k.global_load(ld_dst, addr, Width::B32, 0);
        k.wait_vmcnt(0);

        // K × VALU (serial dependency chain — keeps the wave busy)
        for _ in 0..valu_per_iter {
            k.push(Op::VAddF32 {
                dst: chain, src0: Operand::VReg(chain), src1: Operand::VReg(aux),
            });
        }

        // Store back
        k.global_store(addr, chain, Width::B32, 0);
        k.wait_vscnt(0);
    }

    // Sync all waves so wave 0 doesn't finish early
    if n_waves > 1 {
        k.push(Op::RawAsm("s_barrier".to_string()));
    }

    // === Read end ===
    k.push(Op::ReadShaderCycles { dst: v_end });

    // Write result (wave 0 / lane 0 only)
    emit_store_elapsed(&mut k, v_start, v_end, out_ptr);
    k
}

/// Print multi-wave scaling results.
pub fn print_multiwave_table(results: &[MultiWaveResult]) {
    eprintln!("╔════════╦════════╦══════════════╦═══════════════╦═════════════╦════════════╗");
    eprintln!("║ Waves  ║ Iters  ║ Total cycles ║ Cycles/wave   ║ Speedup     ║ Efficiency ║");
    eprintln!("╠════════╬════════╬══════════════╬═══════════════╬═════════════╬════════════╣");
    for r in results {
        let verdict = if r.efficiency > 0.7 { "✓" } else if r.efficiency > 0.4 { "~" } else { "✗" };
        eprintln!("║ {:>5}w ║ {:>5} ║ {:>12} ║ {:>13.0} ║ {:>8.2}×   ║ {:>6.0}% {} ║",
            r.n_waves, r.n_iters, r.total_cycles,
            r.cycles_per_wave, r.speedup_vs_1wave,
            r.efficiency * 100.0, verdict);
    }
    eprintln!("╚════════╩════════╩══════════════╩═══════════════╩═════════════╩════════════╝");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_all_latency_probes() {
        for &op in ALL_PROBES {
            // Skip VMEM in assembly-only tests (needs 2nd kernarg)
            if op.needs_vmem() { continue; }
            let n = op.default_n_ops().min(8);
            let k = build_latency_probe(op, n);
            let asm = k.to_assembly(Target::GFX1100).expect(
                &format!("assembly gen failed for {}", op.name()));
            assert!(asm.contains("s_getreg_b32"),
                "{}: should have shader_cycles read", op.name());
            eprintln!("  ✓ lat_{} OK ({} bytes)", op.name(), asm.len());
        }
    }

    #[test]
    fn test_build_overlap_probes() {
        let tests = [OverlapTest::ValuTrans, OverlapTest::ValuLds];
        for &test in &tests {
            let (ka, kb, km) = build_overlap_probes(test, 16);
            for (label, k) in [("A", &ka), ("B", &kb), ("MIX", &km)] {
                let asm = k.to_assembly(Target::GFX1100).expect(
                    &format!("asm gen failed for {} {}", test.name(), label));
                let n_getreg = asm.matches("s_getreg").count();
                let n_lines = asm.lines().count();
                assert_eq!(n_getreg, 2, "{} {}: need exactly 2 timer reads", test.name(), label);
                eprintln!("  ✓ {} {} OK ({} lines)", test.name(), label, n_lines);
            }
        }
    }

    #[test]
    fn test_build_multiwave_kernel() {
        for n_waves in [1u32, 2, 4, 8] {
            let k = build_multiwave_mixed_kernel(n_waves, 4, 4);
            let asm = k.to_assembly(Target::GFX1100).expect(
                &format!("asm gen failed for {} waves", n_waves));
            let n_getreg = asm.matches("s_getreg").count();
            assert_eq!(n_getreg, 2, "{}w: need exactly 2 timer reads", n_waves);
            if n_waves > 1 {
                assert!(asm.contains("s_barrier"), "{}w: need barrier", n_waves);
            }
            eprintln!("  ✓ {}w multiwave kernel OK ({} lines)", n_waves, asm.lines().count());
        }
    }

    #[test]
    fn test_build_all_throughput_probes() {
        for &op in ALL_PROBES {
            if !op.supports_throughput() { continue; }
            let n = op.default_n_ops().min(8);
            let k = build_throughput_probe(op, n);
            let asm = k.to_assembly(Target::GFX1100).expect(
                &format!("assembly gen failed for {}", op.name()));
            assert!(asm.contains("s_getreg_b32"),
                "{}: should have shader_cycles read", op.name());
            eprintln!("  ✓ thr_{} OK ({} bytes)", op.name(), asm.len());
        }
    }

    #[test]
    fn test_build_probe_vmem_latency() {
        let k = build_probe_vmem_latency(32);
        let asm = k.to_assembly(Target::GFX1100).expect("assembly gen failed");
        assert!(asm.contains("s_getreg_b32"), "should have shader_cycles read");
        assert!(asm.contains("global_load_b32"), "should have global loads");
        eprintln!("  ✓ probe_vmem_latency OK ({} bytes)", asm.len());
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_compile_all_latency_probes() {
        for &op in ALL_PROBES {
            if op.needs_vmem() { continue; }
            let n = op.default_n_ops().min(8);
            let k = build_latency_probe(op, n);
            let elf = k.compile(Target::GFX1100).expect(
                &format!("compile failed for {}", op.name()));
            assert!(elf.len() > 100);
            eprintln!("  ✓ lat_{} ELF: {} bytes", op.name(), elf.len());
        }
    }

    // ── GPU E2E: full sweep ──

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    mod gpu_e2e {
        use super::*;
        use std::sync::{Arc, OnceLock};
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::t0::dsl::{CompiledKernel, KernArgMeta, KernArgType};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

        fn setup() -> Arc<GpuRuntime> {
            GPU_RT.get_or_init(|| SyncRt(GpuRuntime::new().expect("GPU init failed"))).0.clone()
        }

        fn run_probe_kernel(rt: &GpuRuntime, k: &T0Kernel, vmem_buf: Option<u64>) -> u32 {
            let elf = k.compile(Target::GFX1100).expect("compile failed");
            let mut args = vec![
                KernArgMeta { name: "out".to_string(), kind: KernArgType::Ptr, offset: 0 },
            ];
            if vmem_buf.is_some() {
                args.push(KernArgMeta { name: "vmem_buf".to_string(), kind: KernArgType::Ptr, offset: 8 });
            }
            let compiled = CompiledKernel {
                elf,
                kernarg_size: k.kernarg_size() as usize,
                workgroup_size: [32, 1, 1],
                lds_size: k.lds_size() as u32,
                name: k.name.clone(),
                args,
            };
            let out_buf = rt.alloc_f32(1).unwrap();
            let mut ka = vec![0u8; compiled.kernarg_size];
            ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
            if let Some(addr) = vmem_buf {
                ka[8..16].copy_from_slice(&addr.to_le_bytes());
            }
            let gpu_kernel = rt.compile_dsl(compiled).unwrap();
            rt.dispatch(&gpu_kernel, [32, 1, 1], &ka).unwrap();
            let result = rt.read_f32(&out_buf, 1);
            result[0].to_bits()
        }

        /// Run ALL instruction probes and print the complete table.
        #[test]
        fn test_gpu_probe_sweep() {
            let rt = setup();
            let mut results: Vec<ProbeResult> = Vec::new();

            // Allocate a shared VMEM scratch buffer for VMEM probes (32 lanes × 16 bytes = 512B)
            let vmem_scratch = rt.alloc_f32(128).unwrap(); // 512 bytes
            let vmem_addr = vmem_scratch.gpu_addr();

            for &op in ALL_PROBES {
                let n = op.default_n_ops();
                let vmem = if op.needs_vmem() { Some(vmem_addr) } else { None };

                // Latency probe
                let k_lat = build_latency_probe(op, n);
                let lat_cycles = run_probe_kernel(&rt, &k_lat, vmem);

                // Throughput probe (if supported)
                let thr_cycles = if op.supports_throughput() {
                    let k_thr = build_throughput_probe(op, n);
                    Some(run_probe_kernel(&rt, &k_thr, None))
                } else {
                    None
                };

                results.push(ProbeResult {
                    op: op.name(),
                    pipeline: op.expected_pipeline(),
                    n_ops: n,
                    latency_cycles: lat_cycles,
                    throughput_cycles: thr_cycles,
                });
            }

            eprintln!("\n");
            eprintln!("  GFX1100 (Navi 31, RDNA3) Instruction Probe Results");
            eprintln!("  Wave32, single WG, {} probes\n", results.len());
            print_probe_table(&results);
            eprintln!("");

            // Sanity checks: warn on suspicious values (don't hard-fail, cycle timer can overflow)
            for r in &results {
                if r.latency_cycles == 0 {
                    eprintln!("⚠ {}: latency = 0 (instruction may have been DCE'd)", r.op);
                }
                let lat_per = r.lat_per_op();
                if lat_per > 100_000.0 && r.pipeline != "VMEM" && r.pipeline != "WAVE" {
                    eprintln!("⚠ {}: lat/op = {:.0} (possible timer overflow)", r.op, lat_per);
                }
            }
        }

        /// Run pipeline overlap probes: VALU+VMEM, VALU+LDS, VALU+TRANS.
        #[test]
        fn test_gpu_overlap_probes() {
            let rt = setup();
            let vmem_scratch = rt.alloc_f32(128).unwrap();
            let vmem_addr = vmem_scratch.gpu_addr();
            let n_pairs = 256u32;

            let tests = [
                OverlapTest::ValuVmem,
                OverlapTest::ValuLds,
                OverlapTest::ValuTrans,
            ];

            let mut results: Vec<OverlapResult> = Vec::new();

            for &test in &tests {
                let (ka, kb, km) = build_overlap_probes(test, n_pairs);

                // Run baseline A (VALU only, no vmem buf needed)
                let t_a = run_probe_kernel(&rt, &ka, None);

                // Run baseline B
                let vmem = match test {
                    OverlapTest::ValuVmem => Some(vmem_addr),
                    _ => None,
                };
                let t_b = run_probe_kernel(&rt, &kb, vmem);

                // Run interleaved
                let t_mix = run_probe_kernel(&rt, &km, vmem);

                let r = compute_overlap(test, n_pairs, t_a, t_b, t_mix);
                results.push(r);
            }

            eprintln!("\n");
            eprintln!("  GFX1100 Pipeline Overlap Results (N={})\n", n_pairs);
            print_overlap_table(&results);
            eprintln!("");

            // Log raw data for analysis
            for r in &results {
                eprintln!("  {} — A={} B={} MIX={} → {:.1}%",
                    r.test, r.cycles_a_only, r.cycles_b_only,
                    r.cycles_interleaved, r.overlap_ratio * 100.0);
            }
        }

        /// Calibration verification: compare probe-measured latencies against
        /// `op_latency()` predictions in the latency model.
        #[test]
        fn test_latency_model_vs_probe() {
            use crate::t0::latency_model;

            let rt = setup();
            let vmem_scratch = rt.alloc_f32(128).unwrap();
            let vmem_addr = vmem_scratch.gpu_addr();

            // Collect reference probes (skip VAddF32 = baseline, comparing to itself is trivial)
            let reference_probes: Vec<(ProbeOp, Op)> = vec![
                (ProbeOp::VMulF32, Op::VMulF32 { dst: VReg(1), src0: Operand::VReg(VReg(2)), src1: Operand::InlineFloat(0.5) }),
                (ProbeOp::VFmaF32, Op::VFmaF32 { dst: VReg(1), src0: Operand::VReg(VReg(2)), src1: Operand::InlineFloat(0.5), src2: Operand::VReg(VReg(3)) }),
                (ProbeOp::VRcpF32, Op::VRcpF32 { dst: VReg(1), src: VReg(2) }),
                (ProbeOp::VCvtF32U32, Op::VCvtF32U32 { dst: VReg(1), src: VReg(2) }),
                (ProbeOp::CvtPkBf16F32, Op::CvtPkBf16F32 { dst: VReg(1), src0: VReg(2), src1: VReg(3) }),
                (ProbeOp::DsLoadB32, Op::DsLoadB32 { dst: VReg(1), vaddr: VReg(0), offset: 0 }),
                (ProbeOp::DsStoreB32, Op::DsStoreB32 { vaddr: VReg(0), src: VReg(1), offset: 0 }),
                (ProbeOp::GlobalLoadB32, Op::GlobalLoad { dst: VReg(1), addr: VReg(2), width: Width::B32, offset: 0 }),
                (ProbeOp::GlobalStoreB32, Op::GlobalStore { addr: VReg(2), src: VReg(1), width: Width::B32, offset: 0 }),
                (ProbeOp::WmmaF32BF16, Op::Wmma { dst: VReg(0), a: VReg(8), b: VReg(16), c: VReg(24), format: WmmaFormat::BF16_F32 }),
                (ProbeOp::SAddU32, Op::SAddU32 { dst: SReg(4), src0: SReg(4), src1: SOperand::InlineInt(1) }),
            ];

            // Run v_add_f32 baseline first to get the normalization factor
            let baseline_n = ProbeOp::VAddF32.default_n_ops();
            let k_baseline = build_latency_probe(ProbeOp::VAddF32, baseline_n);
            let baseline_cycles = run_probe_kernel(&rt, &k_baseline, None);
            let baseline_per_op = baseline_cycles as f64 / baseline_n as f64;

            eprintln!("\n  Calibration Verification: latency_model vs probe data");
            eprintln!("  Baseline: v_add_f32 = {:.1} shader cycles/op\n", baseline_per_op);
            eprintln!("  {:17} {:>6} {:>8} {:>8} {:>7}",
                "Instruction", "Model", "Probe", "Ratio", "Status");
            eprintln!("  {:17} {:>6} {:>8} {:>8} {:>7}",
                "───────────", "─────", "──────", "─────", "──────");

            let mut pass = 0usize;
            let mut fail = 0usize;

            for (probe_op, ir_op) in &reference_probes {
                let n = probe_op.default_n_ops();
                let vmem = if probe_op.needs_vmem() { Some(vmem_addr) } else { None };

                let k = build_latency_probe(*probe_op, n);
                let raw_cycles = run_probe_kernel(&rt, &k, vmem);
                let measured_per_op = raw_cycles as f64 / n as f64;
                let measured_valu_norm = measured_per_op / baseline_per_op;

                let model_info = latency_model::op_latency(ir_op);
                let model_valu_norm = model_info.latency as f64;

                let ratio = measured_valu_norm / model_valu_norm.max(0.01);
                let ok = ratio >= 0.5 && ratio <= 2.0; // ±50% tolerance

                let status = if ok { "✓" } else { "✗ DRIFT" };
                if ok { pass += 1; } else { fail += 1; }

                eprintln!("  {:17} {:>6} {:>8.1} {:>8.2} {:>7}",
                    probe_op.name(), model_valu_norm, measured_valu_norm, ratio, status);
            }

            eprintln!("\n  Result: {}/{} passed (±50% tolerance)", pass, pass + fail);
            assert!(fail == 0,
                "latency model has {} drifted entries (>50% deviation from probe)",
                fail);
        }

        /// Dispatch a probe kernel with configurable WG size.
        fn run_probe_kernel_wg(
            rt: &GpuRuntime,
            k: &T0Kernel,
            vmem_buf: Option<u64>,
            wg_size: u32,
        ) -> u32 {
            let elf = k.compile(Target::GFX1100).expect("compile failed");
            let mut args = vec![
                KernArgMeta { name: "out".to_string(), kind: KernArgType::Ptr, offset: 0 },
            ];
            if vmem_buf.is_some() {
                args.push(KernArgMeta {
                    name: "vmem_buf".to_string(), kind: KernArgType::Ptr, offset: 8,
                });
            }
            let compiled = CompiledKernel {
                elf,
                kernarg_size: k.kernarg_size() as usize,
                workgroup_size: [wg_size, 1, 1],
                lds_size: k.lds_size() as u32,
                name: k.name.clone(),
                args,
            };
            let out_buf = rt.alloc_f32(1).unwrap();
            let mut ka = vec![0u8; compiled.kernarg_size];
            ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
            if let Some(addr) = vmem_buf {
                ka[8..16].copy_from_slice(&addr.to_le_bytes());
            }
            let gpu_kernel = rt.compile_dsl(compiled).unwrap();
            rt.dispatch(&gpu_kernel, [wg_size, 1, 1], &ka).unwrap();
            let result = rt.read_f32(&out_buf, 1);
            result[0].to_bits()
        }

        /// Multi-wave overlap: does higher wave occupancy hide VMEM latency?
        #[test]
        fn test_gpu_multiwave_overlap() {
            let rt = setup();
            // Allocate scratch: max WG = 8×32 = 256 threads × 4 bytes = 1024 bytes
            let vmem_scratch = rt.alloc_f32(256).unwrap();
            let vmem_addr = vmem_scratch.gpu_addr();

            let n_iters = 64u32;
            let valu_per_iter = 16u32; // balanced compute/memory
            let wave_configs = [1u32, 2, 4, 8];

            eprintln!("\n  Multi-Wave Overlap ({} VALU / VMEM iter, N={})\n", valu_per_iter, n_iters);
            let mut results: Vec<MultiWaveResult> = Vec::new();
            let mut t1 = 0u32;

            for &n_waves in &wave_configs {
                let wg_size = n_waves * 32;
                eprintln!("  Building kernel: {} waves, WG={}", n_waves, wg_size);
                let k = build_multiwave_mixed_kernel(n_waves, n_iters, valu_per_iter);
                eprintln!("  Dispatching {} waves...", n_waves);
                let total = run_probe_kernel_wg(&rt, &k, Some(vmem_addr), wg_size);
                eprintln!("  {} waves → {} cycles", n_waves, total);

                if n_waves == 1 { t1 = total; }
                let speedup = if total > 0 { t1 as f64 / total as f64 } else { 0.0 };

                results.push(MultiWaveResult {
                    n_waves,
                    n_iters,
                    valu_per_iter,
                    total_cycles: total,
                    cycles_per_wave: total as f64 / n_waves as f64,
                    speedup_vs_1wave: speedup,
                    efficiency: speedup / n_waves as f64,
                });
            }
            print_multiwave_table(&results);

            // Summary
            eprintln!("\n  Summary:");
            for r in &results {
                if r.n_waves > 1 {
                    eprintln!("    {}w: {:.2}× speedup, {:.0}% efficiency",
                        r.n_waves, r.speedup_vs_1wave, r.efficiency * 100.0);
                }
            }
        }
    }
}
