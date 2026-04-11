//! Block-Level DSL — Triton 风格的 block kernel builder
//!
//! 提供 `BlockKernel` 构建器，让用户以 tile/block 粒度编写 GPU 内核。
//! 类比 Triton 的 `tl.load / tl.store / tl.program_id`。
//!
//! ## Example
//!
//! ```ignore
//! let mut kb = BlockKernel::new("vector_add", 256);
//! let x_ptr = kb.arg_ptr("x");
//! let y_ptr = kb.arg_ptr("y");
//! let out_ptr = kb.arg_ptr("out");
//! let n = kb.arg_u32("n");
//!
//! let pid = kb.program_id(0);
//! let offsets = kb.arange(0, 256).add(&mut kb, pid.mul_const(&mut kb, 256));
//! let mask = offsets.lt(&mut kb, n);
//! let a = kb.load(x_ptr, offsets, mask);
//! let b = kb.load(y_ptr, offsets, mask);
//! let c = a.add(&mut kb, b);
//! kb.store(out_ptr, offsets, c, mask);
//!
//! let compiled = kb.compile(Target::GFX1100)?;
//! ```

#[allow(unused_imports)]
use super::ir::{current_target, with_target_context, Target, VReg, SReg, SRegPair, Operand, Width, Alignment, WmmaFormat};
use super::compile::T0Kernel;
use super::dsl::CompiledKernel;
use super::gemm_gen;

// ════════════════════════════════════════════
//  TileGemmConfig — Tile-level GEMM specification
// ════════════════════════════════════════════

/// Configuration for a tile-level GEMM operation.
///
/// This is the tile-level abstraction: the user specifies _what_ (matrix multiply
/// with given tile sizes), and `compile()` decides _how_ (cooperative loading,
/// LDS layout, WMMA scheduling).
///
/// NT mode: Y[M,N] = X[M,K] @ WT[N,K]^T (bf16 in, f32 out)
#[derive(Clone, Debug)]
pub struct TileGemmConfig {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub wgp_mode: bool,
    pub split_k: u32,
    pub swap_grid: bool,
}

impl TileGemmConfig {
    /// Auto-select optimal tile configuration for an explicit target.
    pub fn auto_for_target(target: Target, m: u32, k: u32, n: u32) -> Self {
        let cfg = with_target_context(target, || gemm_gen::auto_select(m, k, n));
        Self {
            tile_m: cfg.tile_m,
            tile_n: cfg.tile_n,
            tile_k: cfg.tile_k,
            wgp_mode: cfg.wgp_mode,
            split_k: cfg.split_k.unwrap_or(1),
            swap_grid: cfg.swap_grid,
        }
    }

    /// Auto-select optimal tile configuration based on matrix dimensions.
    ///
    /// Uses the target currently installed in `with_target_context(...)`.
    /// Without an explicit target context, this preserves the historical
    /// `GFX1100` default behavior.
    pub fn auto(m: u32, k: u32, n: u32) -> Self {
        Self::auto_for_target(current_target(), m, k, n)
    }

    /// Convert to gemm_gen::GemmConfig for codegen.
    fn to_gemm_config(&self) -> gemm_gen::GemmConfig {
        gemm_gen::GemmConfig {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            tile_k: self.tile_k,
            wg_size: (self.tile_m / 32) * 32, // n_waves * 32
            use_lds: true,
            double_buffer: true,
            split_k: if self.split_k > 1 { Some(self.split_k) } else { None },
            lds_pad: 0,
            n_col_passes: 1,
            swap_grid: self.swap_grid,
            wgp_mode: self.wgp_mode,
            transpose: gemm_gen::GemmTranspose::NT,
            epilogue: gemm_gen::EpilogueOp::default(),
        }
    }

    /// Descriptive name
    pub fn name(&self) -> String {
        self.to_gemm_config().name()
    }
}

/// Compiled kernel with pre-computed grid size for auto dispatch.
///
/// Created by `BlockKernel::compile_auto()`.
#[derive(Debug)]
pub struct AutoCompiledKernel {
    pub kernel: CompiledKernel,
    pub grid: [u32; 3],
}

// ════════════════════════════════════════════
//  BVal — Block-level value handle
// ════════════════════════════════════════════

/// 不透明的值句柄。每个 BVal 对应 BlockKernel 中的一个 BNode。
///
/// 线程粒度：每个 Wave32 线程持有一个值。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BVal(pub usize);

/// Block-level 数据类型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BType {
    F32,
    U32,
    Mask,   // 1-bit per lane (VCC bitmask)
    Ptr,    // 64-bit GPU address
    LdsPtr, // LDS 内基地址（u32 byte offset）
    F32x8,  // 8 consecutive f32 VGPRs (WMMA accumulator fragment)
}

// ════════════════════════════════════════════
//  BNode — Block-level IR 节点
// ════════════════════════════════════════════

/// Block-level IR 节点。每个节点产生一个 BVal。
#[derive(Clone, Debug)]
pub enum BNode {
    // ── 参数（kernarg）──
    ArgPtr(String),           // 64-bit pointer
    ArgU32(String),           // scalar u32
    ArgF32(String),           // scalar f32

    // ── 索引 ──
    ProgramId(u8),            // tl.program_id(axis) → TGID.x/y/z
    ThreadId,                 // 平坦线程 ID（1D: WORKITEM_ID_X; 2D: flat_tid % block_size_x）
    ThreadIdY { block_x: u32 }, // 2D 工作组 Y 维度线程 ID：flat_tid / block_size_x
    Arange { start: u32, end: u32 }, // 生成 [start, start+1, ..., end-1]

    // ── 常量 ──
    ConstU32(u32),
    ConstF32(f32),

    // ── 算术（F32）──  
    AddF32(BVal, BVal),
    MulF32(BVal, BVal),
    SubF32(BVal, BVal),
    ExpF32(BVal),             // 2^x (hardware)
    Log2F32(BVal),            // log₂(x) (hardware)
    SqrtF32(BVal),
    RcpF32(BVal),
    RsqrtF32(BVal),           // 1/sqrt(x) (hardware)
    AbsF32(BVal),             // |x| (clear sign bit)
    NegF32(BVal),
    SinF32(BVal),             // sin(x) (hardware, computes sin(2π·x))
    CosF32(BVal),             // cos(x) (hardware, computes cos(2π·x))
    DivF32(BVal, BVal),       // a / b (rcp + mul)
    MaxF32(BVal, BVal),
    MinF32(BVal, BVal),
    FmaF32(BVal, BVal, BVal), // a*b + c

    // ── 算术（U32）──
    AddU32(BVal, BVal),
    SubU32(BVal, BVal),       // 整数减法
    MulU32(BVal, BVal),       // 低 32 位乘法
    ShrConstU32(BVal, u8),    // logical right shift by constant
    ShlConstU32(BVal, u8),    // logical left shift by constant
    AndConstU32(BVal, u32),   // bitwise AND with constant
    OrConstU32(BVal, u32),    // bitwise OR with constant
    XorConstU32(BVal, u32),   // bitwise XOR with constant

    // ── 比较 → Mask ──
    LtU32(BVal, BVal),        // a < b (unsigned) → VCC mask
    GeU32(BVal, BVal),        // a >= b (unsigned) → VCC mask
    CmpLtF32(BVal, BVal),     // a < b (float) → VCC mask
    CmpGtF32(BVal, BVal),     // a > b (float) → VCC mask
    AndBool(BVal, BVal),      // mask_a & mask_b → combined mask
    /// Conditional select: dst = mask ? true_val : false_val  (v_cndmask)
    Select { mask: BVal, true_val: BVal, false_val: BVal },

    // ── 内存 ──
    /// global_load with mask: load f32 from base+offset*4, masked by VCC
    Load { ptr: BVal, offsets: BVal, mask: BVal },
    /// global_load u32 with mask: load u32 from base+offset*4, masked
    LoadU32 { ptr: BVal, offsets: BVal, mask: BVal },
    /// global_store with mask: store f32 to base+offset*4, masked by VCC
    Store { ptr: BVal, offsets: BVal, val: BVal, mask: BVal },
    /// global_atomic_add_f32: atomically add val to ptr[offset], masked
    AtomicAddF32 { ptr: BVal, offsets: BVal, val: BVal, mask: BVal },
    /// global_atomic_add_u32 with return: old_val = atomicAdd(ptr, val)
    AtomicAddU32Rtn { ptr: BVal, val: BVal },
    /// global_load bf16 with mask: load bf16 from base+offset*2, convert to f32
    LoadBf16 { ptr: BVal, offsets: BVal, mask: BVal },
    /// global_store bf16 with mask: truncate f32 to bf16, store to base+offset*2
    StoreBf16 { ptr: BVal, offsets: BVal, val: BVal, mask: BVal },

    // ── 类型转换 ──
    CvtF32U32(BVal),          // u32 → f32
    CvtU32F32(BVal),          // f32 → u32 (truncate)
    CvtF32BF16(BVal),         // bf16 (lower 16 bits) → f32 (shift left 16)

    // ── Wave reduce ──
    WaveReduceAddF32(BVal),
    WaveReduceMaxF32(BVal),

    // ── LDS（共享内存）──
    LdsAlloc { size_bytes: u32 },                     // 分配 LDS 空间
    LdsLoad { base: BVal, offset: BVal },             // f32 from LDS
    LdsStore { base: BVal, offset: BVal, val: BVal }, // f32 to LDS

    // ── 同步 ──
    Barrier,                  // s_barrier + lgkmcnt(0)

    // ── 条件分支 ──
    /// if (mask): 仅 mask=1 的 lane 执行后续代码（SaveExec）
    IfMask(BVal),
    /// else: 切换到 mask=0 的 lane（XorExec）
    ElseMask,
    /// endif: 恢复原始 EXEC（RestoreExec）
    EndIf,

    // ── 循环 ──
    ForBegin { start: BVal, end: BVal, step: u32 },   // SGPR loop header
    ForEnd { begin_node: usize },                     // 跳回 ForBegin

    // ── 带累加器的循环（循环携带变量）──
    /// Loop with carried accumulator: for i in [start, end) { acc = f(acc, i) }
    ForAccBegin { start: BVal, end: BVal, step: u32, init_acc: BVal },
    /// Phi node for accumulator value inside loop body
    /// Pushed immediately after ForAccBegin to give acc its own BVal index
    ForAccPhi { begin_node: usize },
    /// End of accumulator loop; new_acc is the updated value from the body
    ForAccEnd { begin_node: usize, new_acc: BVal },
    /// Get the final result of accumulator loop after it exits
    ForAccResult { begin_node: usize },

    // ── WG 级归约（跨 wave，通过 LDS）──
    WgReduceAddF32(BVal),
    WgReduceMaxF32(BVal),

    // ── WMMA（Wave Matrix Multiply Accumulate）──
    /// Allocate 8×f32 zero-initialized accumulator (8-aligned VGPRs)
    ZeroAcc,
    /// Pack two f32 values into bf16x2: dst = (bf16(hi) << 16) | bf16(lo)
    CvtPkBf16F32 { lo: BVal, hi: BVal },
    /// v_wmma_f32_16x16x16_bf16: C += A * B
    /// a, b: 8×bf16x2 fragments; c: 8×f32 accumulator
    Wmma { a: BVal, b: BVal, c: BVal },
    /// Extract a single f32 from an F32x8 accumulator (idx 0..7)
    ExtractF32 { src: BVal, idx: u32 },
    /// Replicate a single U32 (bf16x2) value across 8 consecutive VGPRs → F32x8
    SplatFragment(BVal),

    // ── Tile-Level Operations ──
    /// Tile-level GEMM: Y[M,N] = X[M,K] @ WT[N,K]^T
    ///
    /// This is a **mega-op**: it takes over the entire kernel.
    /// `compile()` delegates to `gemm_gen::generate()` for code generation,
    /// inheriting its cooperative loading, LDS double-buffering, and
    /// interleaved WMMA scheduling optimizations.
    TileGemm {
        a_ptr: BVal,
        b_ptr: BVal,
        y_ptr: BVal,
        k_dim: BVal,
        n_dim: BVal,
        config: TileGemmConfig,
    },
}

impl BNode {
    /// 结果类型推导
    fn result_type(&self) -> BType {
        match self {
            BNode::ArgPtr(_) => BType::Ptr,
            BNode::ArgU32(_) | BNode::ConstU32(_) => BType::U32,
            BNode::ArgF32(_) | BNode::ConstF32(_) => BType::F32,
            BNode::ProgramId(_) | BNode::ThreadId | BNode::ThreadIdY { .. } => BType::U32,
            BNode::Arange { .. } => BType::U32,
            BNode::AddF32(..) | BNode::MulF32(..) | BNode::SubF32(..) => BType::F32,
            BNode::ExpF32(_) | BNode::Log2F32(_) | BNode::SqrtF32(_) => BType::F32,
            BNode::RcpF32(_) | BNode::RsqrtF32(_) | BNode::AbsF32(_) | BNode::NegF32(_) => BType::F32,
            BNode::SinF32(_) | BNode::CosF32(_) | BNode::DivF32(..) => BType::F32,
            BNode::MaxF32(..) | BNode::MinF32(..) | BNode::FmaF32(..) => BType::F32,
            BNode::AddU32(..) | BNode::SubU32(..) | BNode::MulU32(..) => BType::U32,
            BNode::ShrConstU32(..) | BNode::ShlConstU32(..) => BType::U32,
            BNode::AndConstU32(..) | BNode::OrConstU32(..) | BNode::XorConstU32(..) => BType::U32,
            BNode::LtU32(..) | BNode::GeU32(..) | BNode::CmpLtF32(..) | BNode::CmpGtF32(..) | BNode::AndBool(..) => BType::Mask,
            BNode::Select { .. } => BType::F32, // assumes f32 select; type depends on operands
            BNode::Load { .. } => BType::F32,
            BNode::LoadU32 { .. } => BType::U32,
            BNode::Store { .. } => BType::U32, // void, but need a type
            BNode::AtomicAddF32 { .. } => BType::U32, // void
            BNode::AtomicAddU32Rtn { .. } => BType::U32, // returns old value
            BNode::LoadBf16 { .. } => BType::F32,  // loads BF16, returns F32
            BNode::StoreBf16 { .. } => BType::U32,  // void
            BNode::CvtF32U32(_) => BType::F32,
            BNode::CvtU32F32(_) => BType::U32,
            BNode::CvtF32BF16(_) => BType::F32,
            BNode::WaveReduceAddF32(_) | BNode::WaveReduceMaxF32(_) => BType::F32,
            BNode::LdsAlloc { .. } => BType::LdsPtr,
            BNode::LdsLoad { .. } => BType::F32,
            BNode::LdsStore { .. } => BType::U32, // void
            BNode::Barrier => BType::U32,          // void
            BNode::IfMask(_) => BType::U32,        // void
            BNode::ElseMask => BType::U32,         // void
            BNode::EndIf => BType::U32,            // void
            BNode::ForBegin { .. } => BType::U32,  // iter var
            BNode::ForEnd { .. } => BType::U32,    // void
            BNode::ForAccBegin { .. } => BType::U32,  // iter var
            BNode::ForAccPhi { .. } => BType::F32,    // accumulator (current value)
            BNode::ForAccEnd { .. } => BType::U32,    // void
            BNode::ForAccResult { .. } => BType::F32, // final accumulator value
            BNode::WgReduceAddF32(_) | BNode::WgReduceMaxF32(_) => BType::F32,
            BNode::ZeroAcc => BType::F32x8,
            BNode::CvtPkBf16F32 { .. } => BType::U32, // bf16x2 packed in u32
            BNode::Wmma { .. } => BType::F32x8,
            BNode::ExtractF32 { .. } => BType::F32,
            BNode::SplatFragment(_) => BType::F32x8,
            BNode::TileGemm { .. } => BType::U32, // void (mega-op)
        }
    }
}

// ════════════════════════════════════════════
//  BVal 运算方法（链式 API）
// ════════════════════════════════════════════

impl BVal {
    pub fn add(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        let ty = kb.types[self.0];
        match ty {
            BType::F32 => kb.push(BNode::AddF32(self, other)),
            BType::U32 => kb.push(BNode::AddU32(self, other)),
            _ => panic!("add: unsupported type {:?}", ty),
        }
    }
    pub fn mul(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        let ty = kb.types[self.0];
        match ty {
            BType::F32 => kb.push(BNode::MulF32(self, other)),
            BType::U32 => kb.push(BNode::MulU32(self, other)),
            _ => panic!("mul: unsupported type {:?}", ty),
        }
    }
    pub fn sub(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        let ty = kb.types[self.0];
        match ty {
            BType::F32 => kb.push(BNode::SubF32(self, other)),
            BType::U32 => kb.push(BNode::SubU32(self, other)),
            _ => panic!("sub: unsupported type {:?}", ty),
        }
    }
    pub fn lt(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::LtU32(self, other))
    }
    pub fn ge(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::GeU32(self, other))
    }
    /// Boolean mask AND: result = mask_a & mask_b
    pub fn and_bool(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::AndBool(self, other))
    }
    pub fn exp2(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::ExpF32(self))
    }
    pub fn log2(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::Log2F32(self))
    }
    /// Natural log: ln(x) = log₂(x) * ln(2)
    /// Hardware v_log_f32 computes log₂(x), so we scale by ln(2) ≈ 0.6931
    pub fn log(self, kb: &mut BlockKernel) -> BVal {
        let log2_val = self.log2(kb);
        let ln2 = kb.const_f32(std::f32::consts::LN_2);
        log2_val.mul(kb, ln2)
    }
    pub fn sqrt(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::SqrtF32(self))
    }
    pub fn rcp(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::RcpF32(self))
    }
    /// Reciprocal square root: 1/sqrt(x) — single hardware instruction
    pub fn rsqrt(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::RsqrtF32(self))
    }
    /// Absolute value: |x| — clears sign bit
    pub fn abs(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::AbsF32(self))
    }
    pub fn neg(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::NegF32(self))
    }
    /// Division: a / b (via rcp + mul)
    pub fn div(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::DivF32(self, other))
    }
    /// Sine: sin(x) — hardware v_sin_f32 (computes sin(2π·x), pre-scaled)
    pub fn sin(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::SinF32(self))
    }
    /// Cosine: cos(x) — hardware v_cos_f32 (computes cos(2π·x), pre-scaled)
    pub fn cos(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::CosF32(self))
    }
    pub fn to_f32(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::CvtF32U32(self))
    }
    pub fn to_u32(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::CvtU32F32(self))
    }
    /// Logical right shift by constant amount
    pub fn shr(self, kb: &mut BlockKernel, amount: u8) -> BVal {
        kb.push(BNode::ShrConstU32(self, amount))
    }
    /// Logical left shift by constant amount
    pub fn shl(self, kb: &mut BlockKernel, amount: u8) -> BVal {
        kb.push(BNode::ShlConstU32(self, amount))
    }
    /// Bitwise AND with constant mask
    pub fn bitand(self, kb: &mut BlockKernel, mask: u32) -> BVal {
        kb.push(BNode::AndConstU32(self, mask))
    }
    /// Bitwise OR with constant
    pub fn bitor(self, kb: &mut BlockKernel, val: u32) -> BVal {
        kb.push(BNode::OrConstU32(self, val))
    }
    /// Bitwise XOR with constant
    pub fn bitxor(self, kb: &mut BlockKernel, val: u32) -> BVal {
        kb.push(BNode::XorConstU32(self, val))
    }
    /// Float less-than comparison: a < b → mask
    pub fn lt_f32(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::CmpLtF32(self, other))
    }
    /// Float greater-than comparison: a > b → mask
    pub fn gt_f32(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::CmpGtF32(self, other))
    }
    /// Conditional select: mask ? self : other
    pub fn select(self, kb: &mut BlockKernel, true_val: BVal, false_val: BVal) -> BVal {
        kb.push(BNode::Select { mask: self, true_val, false_val })
    }
    /// Convert bf16 (in lower 16 bits of u32) to f32 (shift left 16)
    pub fn bf16_to_f32(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::CvtF32BF16(self))
    }

    // ── Compound math operations ──

    /// Fused multiply-add: a*b + c
    pub fn fma(self, kb: &mut BlockKernel, b: BVal, c: BVal) -> BVal {
        kb.push(BNode::FmaF32(self, b, c))
    }

    /// Natural exponent: exp(x) = exp2(x * log2(e))
    /// Hardware v_exp_f32 computes 2^x, so we scale: exp(x) = 2^(x * log₂e)
    pub fn exp(self, kb: &mut BlockKernel) -> BVal {
        let log2e = kb.const_f32(std::f32::consts::LOG2_E);
        let scaled = self.mul(kb, log2e);
        scaled.exp2(kb)
    }

    /// Sigmoid: σ(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(self, kb: &mut BlockKernel) -> BVal {
        let neg_x = self.neg(kb);
        let exp_neg = neg_x.exp(kb);
        let one = kb.const_f32(1.0);
        let one_plus = one.add(kb, exp_neg);
        one_plus.rcp(kb)
    }

    /// Tanh: tanh(x) = 2 * sigmoid(2x) - 1
    pub fn tanh(self, kb: &mut BlockKernel) -> BVal {
        let two = kb.const_f32(2.0);
        let two_x = self.mul(kb, two);
        let sig = two_x.sigmoid(kb);
        let two2 = kb.const_f32(2.0);
        let scaled = sig.mul(kb, two2);
        let one = kb.const_f32(1.0);
        scaled.sub(kb, one)
    }

    /// SiLU: silu(x) = x * σ(x)
    pub fn silu(self, kb: &mut BlockKernel) -> BVal {
        let sig = self.sigmoid(kb);
        self.mul(kb, sig)
    }

    /// GELU (approximation): x * 0.5 * (1 + tanh(sqrt(2/π)(x + 0.044715*x³)))
    pub fn gelu(self, kb: &mut BlockKernel) -> BVal {
        let half = kb.const_f32(0.5);
        let one = kb.const_f32(1.0);
        let coeff = kb.const_f32(0.044715);
        let sqrt_2_pi = kb.const_f32(0.7978845608); // sqrt(2/π)
        // x³ = x * x * x
        let x2 = self.mul(kb, self);
        let x3 = x2.mul(kb, self);
        // inner = sqrt(2/π) * (x + 0.044715 * x³)
        let cx3 = coeff.mul(kb, x3);
        let x_plus = self.add(kb, cx3);
        let inner = sqrt_2_pi.mul(kb, x_plus);
        // tanh(inner)
        let th = inner.tanh(kb);
        // 0.5 * x * (1 + tanh(inner))
        let one_plus_th = one.add(kb, th);
        let half_x = half.mul(kb, self);
        half_x.mul(kb, one_plus_th)
    }

    /// ReLU: max(x, 0)
    pub fn relu(self, kb: &mut BlockKernel) -> BVal {
        let zero = kb.const_f32(0.0);
        kb.push(BNode::MaxF32(self, zero))
    }

    /// Clamp: max(lo, min(x, hi))
    pub fn clamp(self, kb: &mut BlockKernel, lo: f32, hi: f32) -> BVal {
        let lo_v = kb.const_f32(lo);
        let hi_v = kb.const_f32(hi);
        let min_hi = kb.push(BNode::MinF32(self, hi_v));
        kb.push(BNode::MaxF32(min_hi, lo_v))
    }

    /// Max with another BVal: max(self, other) — element-wise
    pub fn max(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::MaxF32(self, other))
    }
    /// Min with another BVal: min(self, other) — element-wise
    pub fn min(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        kb.push(BNode::MinF32(self, other))
    }
    /// Max with f32 constant
    pub fn max_const(self, kb: &mut BlockKernel, val: f32) -> BVal {
        let c = kb.const_f32(val);
        kb.push(BNode::MaxF32(self, c))
    }

    /// Min with f32 constant
    pub fn min_const(self, kb: &mut BlockKernel, val: f32) -> BVal {
        let c = kb.const_f32(val);
        kb.push(BNode::MinF32(self, c))
    }

    /// Dot product: sum(a * b) across the wave
    /// Multiplies elementwise then reduces across wave lanes.
    pub fn dot(self, kb: &mut BlockKernel, other: BVal) -> BVal {
        let prod = self.mul(kb, other);
        kb.wave_reduce_sum(prod)
    }

    /// WMMA: C += A * B (bf16 inputs, f32 accumulator)
    /// self = A fragment (8×bf16x2), b = B fragment, acc = accumulator (F32x8)
    pub fn wmma(self, kb: &mut BlockKernel, b: BVal, acc: BVal) -> BVal {
        kb.push(BNode::Wmma { a: self, b, c: acc })
    }
    /// Extract f32 element from F32x8 accumulator
    pub fn extract(self, kb: &mut BlockKernel, idx: u32) -> BVal {
        assert!(idx < 8, "extract index must be 0..7");
        kb.push(BNode::ExtractF32 { src: self, idx })
    }
    /// Build F32x8 WMMA fragment by replicating this U32 value across all 8 VGPRs
    pub fn splat_fragment(self, kb: &mut BlockKernel) -> BVal {
        kb.push(BNode::SplatFragment(self))
    }
}

// ════════════════════════════════════════════
//  BlockKernel Builder
// ════════════════════════════════════════════

/// Triton 风格的 block kernel 构建器。
///
/// 构建计算图后调用 `compile()` 降低到 T0Kernel → ELF。
pub struct BlockKernel {
    name: String,
    block_size: u32,           // BLOCK_SIZE（workgroup size, total threads）
    block_size_x: u32,         // X dimension of workgroup (default = block_size)
    block_size_y: u32,         // Y dimension of workgroup (default = 1)
    nodes: Vec<BNode>,
    types: Vec<BType>,
    /// 存储操作列表（需要按顺序执行）
    stores: Vec<BVal>,
}

impl BlockKernel {
    pub fn new(name: &str, block_size: u32) -> Self {
        assert!(block_size >= 32, "BLOCK_SIZE must be >= 32 (Wave32)");
        assert!(block_size <= 1024, "BLOCK_SIZE must be <= 1024");
        Self {
            name: name.to_string(),
            block_size,
            block_size_x: block_size,
            block_size_y: 1,
            nodes: Vec::new(),
            types: Vec::new(),
            stores: Vec::new(),
        }
    }

    // ── Accessors for SSA translator ──

    /// Get the kernel name.
    pub fn kernel_name(&self) -> &str { &self.name }
    /// Get the block size (total workgroup size = block_size_x * block_size_y).
    pub fn get_block_size(&self) -> u32 { self.block_size }
    /// Get the 2D block dimensions [x, y].
    pub fn get_block_size_2d(&self) -> [u32; 2] { [self.block_size_x, self.block_size_y] }
    /// Get all IR nodes.
    pub fn get_nodes(&self) -> &[BNode] { &self.nodes }
    /// Get the type of a BVal.
    pub fn get_type(&self, v: BVal) -> BType { self.types[v.0] }

    /// 设置 2D 工作组尺寸。
    ///
    /// Total threads = x * y. 必须满足 x*y >= 32 且 x*y <= 1024。
    /// 启用后，`thread_id()` 返回 X 维度的线程索引，`thread_id_y()` 返回 Y 维度。
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let mut kb = BlockKernel::new("kernel_2d", 128);
    /// kb.set_block_size_2d(32, 4);  // 32×4 = 128 threads
    /// ```
    pub fn set_block_size_2d(&mut self, x: u32, y: u32) {
        let total = x * y;
        assert!(total >= 32, "block_size_x * block_size_y must be >= 32");
        assert!(total <= 1024, "block_size_x * block_size_y must be <= 1024");
        assert!(x >= 1 && y >= 1, "dimensions must be >= 1");
        self.block_size = total;
        self.block_size_x = x;
        self.block_size_y = y;
    }

    // ── 内部：追加节点 ──

    fn push(&mut self, node: BNode) -> BVal {
        let ty = node.result_type();
        let id = self.nodes.len();
        self.nodes.push(node);
        self.types.push(ty);
        BVal(id)
    }

    // ── 参数声明（对应 Triton 的函数形参）──

    pub fn arg_ptr(&mut self, name: &str) -> BVal {
        self.push(BNode::ArgPtr(name.to_string()))
    }

    pub fn arg_u32(&mut self, name: &str) -> BVal {
        self.push(BNode::ArgU32(name.to_string()))
    }

    pub fn arg_f32(&mut self, name: &str) -> BVal {
        self.push(BNode::ArgF32(name.to_string()))
    }

    // ── 索引操作（对应 Triton 的 tl.program_id / tl.arange）──

    /// `tl.program_id(axis)` — 返回当前 workgroup 在 axis 维度的 ID
    pub fn program_id(&mut self, axis: u8) -> BVal {
        assert!(axis <= 2);
        self.push(BNode::ProgramId(axis))
    }

    /// `tl.arange(start, end)` — 返回 [start, start+1, ..., end-1]
    ///
    /// 每个线程持有一个值：thread 0 → start, thread 1 → start+1, ...
    /// `end - start` 应等于 BLOCK_SIZE。
    pub fn arange(&mut self, start: u32, end: u32) -> BVal {
        assert_eq!(end - start, self.block_size,
            "arange({}, {}) != BLOCK_SIZE({})", start, end, self.block_size);
        self.push(BNode::Arange { start, end })
    }

    // ── 常量 ──

    pub fn const_u32(&mut self, val: u32) -> BVal {
        self.push(BNode::ConstU32(val))
    }

    pub fn const_f32(&mut self, val: f32) -> BVal {
        self.push(BNode::ConstF32(val))
    }

    // ── 内存操作（对应 Triton 的 tl.load / tl.store）──

    /// `tl.load(ptr + offsets, mask=mask)`
    ///
    /// 加载 f32 值。每个线程通过 `ptr + offsets[thread_id] * sizeof(f32)` 寻址。
    /// mask 控制哪些线程参与加载（越界线程返回 0）。
    pub fn load(&mut self, ptr: BVal, offsets: BVal, mask: BVal) -> BVal {
        assert_eq!(self.types[ptr.0], BType::Ptr, "load: ptr must be Ptr type");
        assert_eq!(self.types[offsets.0], BType::U32, "load: offsets must be U32");
        assert_eq!(self.types[mask.0], BType::Mask, "load: mask must be Mask type");
        self.push(BNode::Load { ptr, offsets, mask })
    }

    /// `tl.store(ptr + offsets, val, mask=mask)`
    ///
    /// 存储 f32 值。mask 控制哪些线程参与存储。
    pub fn store(&mut self, ptr: BVal, offsets: BVal, val: BVal, mask: BVal) {
        assert_eq!(self.types[ptr.0], BType::Ptr, "store: ptr must be Ptr type");
        assert_eq!(self.types[offsets.0], BType::U32, "store: offsets must be U32");
        assert_eq!(self.types[val.0], BType::F32, "store: val must be F32");
        assert_eq!(self.types[mask.0], BType::Mask, "store: mask must be Mask type");
        let sv = self.push(BNode::Store { ptr, offsets, val, mask });
        self.stores.push(sv);
    }

    /// Load u32 values from global memory (for indices, masks, etc.)
    pub fn load_u32(&mut self, ptr: BVal, offsets: BVal, mask: BVal) -> BVal {
        assert_eq!(self.types[ptr.0], BType::Ptr, "load_u32: ptr must be Ptr type");
        assert_eq!(self.types[offsets.0], BType::U32, "load_u32: offsets must be U32");
        assert_eq!(self.types[mask.0], BType::Mask, "load_u32: mask must be Mask type");
        self.push(BNode::LoadU32 { ptr, offsets, mask })
    }

    /// Load u32 with auto bounds check: mask = offsets < n
    pub fn load_u32_checked(&mut self, ptr: BVal, offsets: BVal, n: BVal) -> BVal {
        let mask = offsets.lt(self, n);
        self.load_u32(ptr, offsets, mask)
    }

    /// Atomic f32 add to global memory: ptr[offsets] += val (masked)
    pub fn atomic_add_f32(&mut self, ptr: BVal, offsets: BVal, val: BVal, mask: BVal) {
        assert_eq!(self.types[ptr.0], BType::Ptr, "atomic_add: ptr must be Ptr");
        assert_eq!(self.types[offsets.0], BType::U32, "atomic_add: offsets must be U32");
        assert_eq!(self.types[val.0], BType::F32, "atomic_add: val must be F32");
        assert_eq!(self.types[mask.0], BType::Mask, "atomic_add: mask must be Mask");
        let sv = self.push(BNode::AtomicAddF32 { ptr, offsets, val, mask });
        self.stores.push(sv);
    }

    /// Atomic f32 add with auto bounds check
    pub fn atomic_add_f32_checked(&mut self, ptr: BVal, offsets: BVal, val: BVal, n: BVal) {
        let mask = offsets.lt(self, n);
        self.atomic_add_f32(ptr, offsets, val, mask);
    }

    /// Wave-level sum reduction (convenience, does not allocate LDS)
    pub fn wave_reduce_sum_val(&mut self, val: BVal) -> BVal {
        assert_eq!(self.types[val.0], BType::F32, "wave_reduce_sum: val must be F32");
        self.push(BNode::WaveReduceAddF32(val))
    }

    /// Wave-level max reduction (convenience)
    pub fn wave_reduce_max_val(&mut self, val: BVal) -> BVal {
        assert_eq!(self.types[val.0], BType::F32, "wave_reduce_max: val must be F32");
        self.push(BNode::WaveReduceMaxF32(val))
    }

    /// Thread ID within the workgroup — X dimension.
    ///
    /// - 1D mode: returns flat WORKITEM_ID_X (0..block_size-1)
    /// - 2D mode: returns flat_tid % block_size_x (0..block_size_x-1)
    pub fn thread_id(&mut self) -> BVal {
        self.push(BNode::ThreadId)
    }

    /// Thread ID within the workgroup — Y dimension (2D only).
    ///
    /// Returns flat_tid / block_size_x (0..block_size_y-1).
    /// Requires `set_block_size_2d()` to be called first.
    pub fn thread_id_y(&mut self) -> BVal {
        assert!(self.block_size_y > 1, "thread_id_y() requires set_block_size_2d()");
        self.push(BNode::ThreadIdY { block_x: self.block_size_x })
    }

    // ── Auto bounds-checked memory ops ──

    /// Load with automatic bounds checking: mask = offsets < n
    ///
    /// Equivalent to `let mask = offsets.lt(kb, n); kb.load(ptr, offsets, mask)`
    pub fn load_checked(&mut self, ptr: BVal, offsets: BVal, n: BVal) -> BVal {
        let mask = offsets.lt(self, n);
        self.load(ptr, offsets, mask)
    }

    /// Store with automatic bounds checking: mask = offsets < n
    ///
    /// Equivalent to `let mask = offsets.lt(kb, n); kb.store(ptr, offsets, val, mask)`
    pub fn store_checked(&mut self, ptr: BVal, offsets: BVal, val: BVal, n: BVal) {
        let mask = offsets.lt(self, n);
        self.store(ptr, offsets, val, mask);
    }

    /// Load BF16 from global memory and convert to F32.
    ///
    /// Reads a 16-bit BF16 value from `ptr + offsets * 2`, zero-extends to u32,
    /// then shifts left 16 to produce the F32 bit pattern.
    /// Result type is F32.
    pub fn load_bf16(&mut self, ptr: BVal, offsets: BVal, mask: BVal) -> BVal {
        assert_eq!(self.types[ptr.0], BType::Ptr, "load_bf16: ptr must be Ptr type");
        assert_eq!(self.types[offsets.0], BType::U32, "load_bf16: offsets must be U32");
        assert_eq!(self.types[mask.0], BType::Mask, "load_bf16: mask must be Mask type");
        self.push(BNode::LoadBf16 { ptr, offsets, mask })
    }

    /// Load BF16 with automatic bounds checking: mask = offsets < n
    pub fn load_bf16_checked(&mut self, ptr: BVal, offsets: BVal, n: BVal) -> BVal {
        let mask = offsets.lt(self, n);
        self.load_bf16(ptr, offsets, mask)
    }

    /// Store F32 value as BF16 to global memory.
    ///
    /// Truncates F32 to BF16 (takes upper 16 bits of f32 bit pattern),
    /// then stores the 16-bit value to `ptr + offsets * 2`.
    pub fn store_bf16(&mut self, ptr: BVal, offsets: BVal, val: BVal, mask: BVal) {
        assert_eq!(self.types[ptr.0], BType::Ptr, "store_bf16: ptr must be Ptr type");
        assert_eq!(self.types[offsets.0], BType::U32, "store_bf16: offsets must be U32");
        assert_eq!(self.types[val.0], BType::F32, "store_bf16: val must be F32");
        assert_eq!(self.types[mask.0], BType::Mask, "store_bf16: mask must be Mask type");
        let sv = self.push(BNode::StoreBf16 { ptr, offsets, val, mask });
        self.stores.push(sv);
    }

    /// Store F32 as BF16 with automatic bounds checking: mask = offsets < n
    pub fn store_bf16_checked(&mut self, ptr: BVal, offsets: BVal, val: BVal, n: BVal) {
        let mask = offsets.lt(self, n);
        self.store_bf16(ptr, offsets, val, mask);
    }

    // ── 归约操作 ──

    /// Wave-level sum reduction (32 lanes → broadcast result)
    pub fn wave_reduce_sum(&mut self, val: BVal) -> BVal {
        self.push(BNode::WaveReduceAddF32(val))
    }

    /// Wave-level max reduction (32 lanes → broadcast result)
    pub fn wave_reduce_max(&mut self, val: BVal) -> BVal {
        self.push(BNode::WaveReduceMaxF32(val))
    }

    // ── LDS（共享内存）操作 ──

    /// 分配 LDS 缓冲区（返回 LDS 基地址 handle）
    ///
    /// `size_bytes` 是总字节数。多次调用会分配不重叠的区域。
    pub fn lds_alloc(&mut self, size_bytes: u32) -> BVal {
        self.push(BNode::LdsAlloc { size_bytes })
    }

    /// LDS load: val = lds[base + offset * 4]
    pub fn lds_load(&mut self, base: BVal, offset: BVal) -> BVal {
        assert_eq!(self.types[base.0], BType::LdsPtr, "lds_load: base must be LdsPtr");
        assert_eq!(self.types[offset.0], BType::U32, "lds_load: offset must be U32");
        self.push(BNode::LdsLoad { base, offset })
    }

    /// LDS store: lds[base + offset * 4] = val
    pub fn lds_store(&mut self, base: BVal, offset: BVal, val: BVal) {
        assert_eq!(self.types[base.0], BType::LdsPtr, "lds_store: base must be LdsPtr");
        assert_eq!(self.types[offset.0], BType::U32, "lds_store: offset must be U32");
        assert_eq!(self.types[val.0], BType::F32, "lds_store: val must be F32");
        self.push(BNode::LdsStore { base, offset, val });
    }

    // ── 同步 ──

    /// Workgroup barrier (all waves in WG synchronize)
    pub fn barrier(&mut self) {
        self.push(BNode::Barrier);
    }

    // ── 条件分支 ──

    /// if (mask): 仅 mask=1 的 lane 执行后续代码
    ///
    /// 用法：
    /// ```ignore
    /// let mask = idx.lt_u32(kb, n);
    /// kb.if_mask(mask);
    ///   // ... if body （仅 mask=1 的 lane 执行）
    /// kb.else_mask();  // 可选
    ///   // ... else body （仅 mask=0 的 lane 执行）
    /// kb.end_if();
    /// ```
    pub fn if_mask(&mut self, mask: BVal) {
        self.push(BNode::IfMask(mask));
    }

    /// else: 切换到 mask=0 的 lane（与 if_mask 配对，可选）
    pub fn else_mask(&mut self) {
        self.push(BNode::ElseMask);
    }

    /// endif: 恢复原始 EXEC（与 if_mask 配对）
    pub fn end_if(&mut self) {
        self.push(BNode::EndIf);
    }

    // ── 循环 ──

    /// for i in range(start, end, step): ...
    ///
    /// 返回迭代变量 handle（SGPR 计数器）。
    /// 循环体内用 `end_for()` 关闭。
    pub fn for_range(&mut self, start: BVal, end: BVal, step: u32) -> BVal {
        assert!(step > 0, "for_range: step must be > 0");
        self.push(BNode::ForBegin { start, end, step })
    }

    /// 结束 for 循环（与 for_range 配对）
    pub fn end_for(&mut self, iter_var: BVal) {
        // Find the ForBegin node index from the iter_var
        let begin_node = iter_var.0;
        assert!(matches!(self.nodes[begin_node], BNode::ForBegin { .. }),
            "end_for: iter_var must come from for_range");
        self.push(BNode::ForEnd { begin_node });
    }

    /// for i in range(start, end, step) with accumulator:
    ///   acc starts at init, loop body updates it
    ///
    /// Returns (iter_var, acc_var).
    /// Use `end_for_acc()` to close the loop and get the final result.
    ///
    /// ```ignore
    /// let zero = kb.const_f32(0.0);
    /// let start = kb.const_u32(0);
    /// let (iter, acc) = kb.for_range_acc(start, n, 1, zero);
    /// let val = kb.load(ptr, iter, mask);
    /// let new_acc = acc.add(&mut kb, val);
    /// let result = kb.end_for_acc(iter, new_acc);
    /// // result = sum of all loaded values
    /// ```
    pub fn for_range_acc(&mut self, start: BVal, end: BVal, step: u32, init_acc: BVal) -> (BVal, BVal) {
        assert!(step > 0, "for_range_acc: step must be > 0");
        let iter_var = self.push(BNode::ForAccBegin { start, end, step, init_acc });
        let begin_node = iter_var.0;
        let acc_var = self.push(BNode::ForAccPhi { begin_node });
        (iter_var, acc_var)
    }

    /// End accumulator loop, return final result BVal
    pub fn end_for_acc(&mut self, iter_var: BVal, new_acc: BVal) -> BVal {
        let begin_node = iter_var.0;
        assert!(matches!(self.nodes[begin_node], BNode::ForAccBegin { .. }),
            "end_for_acc: iter_var must come from for_range_acc");
        self.push(BNode::ForAccEnd { begin_node, new_acc });
        self.push(BNode::ForAccResult { begin_node })
    }

    // ── WMMA 操作 ──

    /// Allocate 8×f32 zero-initialized WMMA accumulator (8-aligned VGPRs)
    pub fn zero_acc(&mut self) -> BVal {
        self.push(BNode::ZeroAcc)
    }

    /// Pack two f32 values into bf16x2: result = (bf16(hi) << 16) | bf16(lo)
    pub fn cvt_pk_bf16(&mut self, lo: BVal, hi: BVal) -> BVal {
        self.push(BNode::CvtPkBf16F32 { lo, hi })
    }

    // ── WG 级归约 ──

    /// WG-level sum reduction (cross-wave via LDS)
    ///
    /// 1. wave_reduce_add within each wave
    /// 2. wave leaders write partial sums to LDS
    /// 3. barrier
    /// 4. wave 0 loads + reduces partial sums
    /// 5. broadcast result to all lanes
    pub fn wg_reduce_sum(&mut self, val: BVal) -> BVal {
        assert_eq!(self.types[val.0], BType::F32, "wg_reduce_sum: val must be F32");
        self.push(BNode::WgReduceAddF32(val))
    }

    /// WG-level max reduction (cross-wave via LDS)
    ///
    /// Same pattern as wg_reduce_sum but uses max instead of add.
    pub fn wg_reduce_max(&mut self, val: BVal) -> BVal {
        assert_eq!(self.types[val.0], BType::F32, "wg_reduce_max: val must be F32");
        self.push(BNode::WgReduceMaxF32(val))
    }

    // ── Tile-Level Operations ──

    /// Tile-level GEMM: Y[M,N] = X[M,K] @ WT[N,K]^T
    ///
    /// This is a Triton-style tile operation: the user specifies _what_ (matrix
    /// multiply with pointers and dimensions), and `compile()` decides _how_
    /// (cooperative loading strategy, LDS layout, WMMA scheduling).
    ///
    /// **Note**: This is a mega-op that takes over the entire kernel. It cannot
    /// be composed with other BNodes in the same BlockKernel.
    ///
    /// # Example
    /// ```ignore
    /// let mut kb = BlockKernel::new("my_gemm", 128);
    /// let x = kb.arg_ptr("X");
    /// let w = kb.arg_ptr("W");
    /// let y = kb.arg_ptr("Y");
    /// let k = kb.arg_u32("K");
    /// let n = kb.arg_u32("N");
    /// let sk_shift = kb.arg_u32("split_k_shift");
    /// let y_stride = kb.arg_u32("y_split_stride");
    ///
    /// kb.tile_gemm(x, w, y, k, n, TileGemmConfig::auto(m, k_val, n_val));
    /// let compiled = kb.compile(Target::GFX1100)?;
    /// ```
    pub fn tile_gemm(
        &mut self,
        a_ptr: BVal, b_ptr: BVal, y_ptr: BVal,
        k_dim: BVal, n_dim: BVal,
        config: TileGemmConfig,
    ) {
        assert_eq!(self.types[a_ptr.0], BType::Ptr, "tile_gemm: a_ptr must be Ptr");
        assert_eq!(self.types[b_ptr.0], BType::Ptr, "tile_gemm: b_ptr must be Ptr");
        assert_eq!(self.types[y_ptr.0], BType::Ptr, "tile_gemm: y_ptr must be Ptr");
        assert_eq!(self.types[k_dim.0], BType::U32, "tile_gemm: k_dim must be U32");
        assert_eq!(self.types[n_dim.0], BType::U32, "tile_gemm: n_dim must be U32");
        let sv = self.push(BNode::TileGemm { a_ptr, b_ptr, y_ptr, k_dim, n_dim, config });
        self.stores.push(sv);
    }

    // ════════════════════════════════════════════
    //  compile() — 降低到 T0Kernel → ELF
    // ════════════════════════════════════════════

    /// 将 block kernel 编译为 GPU 可执行的 CompiledKernel。
    ///
    /// **统一编译路径**：所有调用走 SSA pipeline（compile_via_ssa）。
    pub fn compile(&self, target: Target) -> Result<CompiledKernel, String> {
        self.compile_via_ssa(target)
    }

    /// 打印 IR 摘要
    pub fn summary(&self) -> String {
        let mut s = format!("BlockKernel '{}' (BLOCK_SIZE={})\n", self.name, self.block_size);
        let args: Vec<_> = self.nodes.iter().filter(|n| matches!(n,
            BNode::ArgPtr(_) | BNode::ArgU32(_) | BNode::ArgF32(_)
        )).collect();
        s += &format!("  args: {}\n", args.len());
        s += &format!("  nodes: {}\n", self.nodes.len());
        s += &format!("  stores: {}\n", self.stores.len());
        s
    }
}

// ════════════════════════════════════════════
//  单元测试
// ════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_kernel_basic() {
        let mut kb = BlockKernel::new("test_add", 256);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let out = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let pid = kb.program_id(0);
        let bs = kb.const_u32(256);
        let base = pid.mul(&mut kb, bs);
        let offsets = kb.arange(0, 256).add(&mut kb, base);
        let mask = offsets.lt(&mut kb, n);
        let a = kb.load(x, offsets, mask);
        let b = kb.load(y, offsets, mask);
        let c = a.add(&mut kb, b);
        kb.store(out, offsets, c, mask);

        eprintln!("{}", kb.summary());
        assert!(kb.nodes.len() > 0);
        assert_eq!(kb.stores.len(), 1);
    }

    #[test]
    fn test_compile_vector_add() {
        let mut kb = BlockKernel::new("vector_add", 256);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let out = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let pid = kb.program_id(0);
        let bs = kb.const_u32(256);
        let base = pid.mul(&mut kb, bs);
        let offsets = kb.arange(0, 256).add(&mut kb, base);
        let mask = offsets.lt(&mut kb, n);
        let a = kb.load(x, offsets, mask);
        let b = kb.load(y, offsets, mask);
        let c = a.add(&mut kb, b);
        kb.store(out, offsets, c, mask);

        let compiled = kb.compile(Target::GFX1100).unwrap();
        eprintln!("compiled: {:?}", compiled);
        assert!(!compiled.elf.is_empty());
        assert!(compiled.name == "vector_add");
    }

    #[test]
    fn test_compile_softmax_partial() {
        // partial softmax within a wave: max, sub, exp
        let mut kb = BlockKernel::new("softmax_partial", 32);
        let x = kb.arg_ptr("x");
        let out = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let vals = kb.load(x, offsets, mask);
        let max_val = kb.wave_reduce_max(vals);
        let shifted = vals.sub(&mut kb, max_val);
        let exp_vals = shifted.exp2(&mut kb);
        kb.store(out, offsets, exp_vals, mask);

        let compiled = kb.compile(Target::GFX1100).unwrap();
        assert!(!compiled.elf.is_empty());
    }
}

// ════════════════════════════════════════════
//  GPU 测试
// ════════════════════════════════════════════

#[cfg(all(test, feature = "rocm"))]
mod gpu_tests {
    use super::*;
    use std::sync::{Arc, OnceLock};
    use crate::ignis::gpu_context::GpuRuntime;

    struct SyncRt(Arc<GpuRuntime>);
    unsafe impl Sync for SyncRt {}
    unsafe impl Send for SyncRt {}
    static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

    fn setup() -> Arc<GpuRuntime> {
        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
        });
        // Safety net: if a previous test caused GPU reset, the OnceLock'd runtime
        // is stale. Detect this early and fail fast instead of cascading hangs.
        if rt.0.is_poisoned() {
            panic!("[GPU SAFETY] Runtime poisoned after GPU hang — restart test process to recover");
        }
        rt.0.clone()
    }

    /// GPU test: vector_add using block DSL
    ///
    /// Equivalent Triton:
    /// ```python
    /// @triton.jit
    /// def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    ///     pid = tl.program_id(0)
    ///     offs = pid * BLOCK + tl.arange(0, BLOCK)
    ///     mask = offs < n
    ///     a = tl.load(x + offs, mask)
    ///     b = tl.load(y + offs, mask)
    ///     tl.store(out + offs, a + b, mask)
    /// ```
    #[test]
    fn test_gpu_vector_add() {
        let rt = setup();
        let n: usize = 1024;
        let block_size: u32 = 256;

        // Build kernel
        let mut kb = BlockKernel::new("gpu_vector_add", block_size);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let out = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");

        let pid = kb.program_id(0);
        let bs = kb.const_u32(block_size);
        let base = pid.mul(&mut kb, bs);
        let offsets = kb.arange(0, block_size).add(&mut kb, base);
        let mask = offsets.lt(&mut kb, n_arg);
        let a = kb.load(x, offsets, mask);
        let b = kb.load(y, offsets, mask);
        let c = a.add(&mut kb, b);
        kb.store(out, offsets, c, mask);

        let compiled = kb.compile(Target::GFX1100).unwrap();

        // Allocate GPU buffers
        let x_buf = rt.alloc_f32(n).unwrap();
        let y_buf = rt.alloc_f32(n).unwrap();
        let out_buf = rt.alloc_f32(n).unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
        rt.write_f32(&x_buf, &x_data);
        rt.write_f32(&y_buf, &y_data);

        // Build kernargs
        let mut ka = vec![0u8; compiled.kernarg_size];
        // Ptrs: x, y, out (each 8 bytes), then n (4 bytes)
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        ka[24..28].copy_from_slice(&(n as u32).to_le_bytes());

        // Dispatch
        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        let grid_x = ((n as u32 + block_size - 1) / block_size) * block_size;
        rt.dispatch(&gpu_kernel, [grid_x, 1, 1], &ka).unwrap();

        // Verify
        let result = rt.read_f32(&out_buf, n);
        let mut max_err = 0f32;
        for i in 0..n {
            let expected = x_data[i] + y_data[i];
            let err = (result[i] - expected).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("  block_dsl vector_add: max_err = {:.6e}", max_err);
        assert!(max_err < 1e-6, "max error {} too large", max_err);
    }

    /// GPU test: LDS write → barrier → read → multiply by 2
    ///
    /// Each thread writes its value to LDS, barrier, reads it back, multiplies by 2.
    /// This tests LDS alloc/store/load and barrier.
    #[test]
    fn test_gpu_lds_double() {
        let rt = setup();
        let n: usize = 256;
        let block_size: u32 = 256;

        let mut kb = BlockKernel::new("lds_double", block_size);
        let x = kb.arg_ptr("x");
        let out = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");

        let pid = kb.program_id(0);
        let bs = kb.const_u32(block_size);
        let base = pid.mul(&mut kb, bs);
        let offsets = kb.arange(0, block_size).add(&mut kb, base);
        let mask = offsets.lt(&mut kb, n_arg);

        // Load from global → LDS → load back → multiply
        let vals = kb.load(x, offsets, mask);

        // Allocate LDS for BLOCK_SIZE f32 values
        let lds = kb.lds_alloc(block_size * 4);
        let tid = kb.arange(0, block_size);

        // Write to LDS
        kb.lds_store(lds, tid, vals);

        // Barrier
        kb.barrier();

        // Read back from LDS
        let lds_vals = kb.lds_load(lds, tid);

        // Multiply by 2
        let two = kb.const_f32(2.0);
        let result = lds_vals.mul(&mut kb, two);

        kb.store(out, offsets, result, mask);

        let compiled = kb.compile(Target::GFX1100).unwrap();
        assert!(compiled.lds_size > 0, "LDS size should be > 0");
        eprintln!("  lds_double: lds_size = {}", compiled.lds_size);

        // Allocate GPU buffers
        let x_buf = rt.alloc_f32(n).unwrap();
        let out_buf = rt.alloc_f32(n).unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        rt.write_f32(&x_buf, &x_data);

        // Build kernargs
        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        ka[16..20].copy_from_slice(&(n as u32).to_le_bytes());

        // Dispatch
        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        rt.dispatch(&gpu_kernel, [block_size, 1, 1], &ka).unwrap();

        // Verify: output should be x * 2
        let result = rt.read_f32(&out_buf, n);
        let mut max_err = 0f32;
        for i in 0..n {
            let expected = x_data[i] * 2.0;
            let err = (result[i] - expected).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("  block_dsl lds_double: max_err = {:.6e}", max_err);
        assert!(max_err < 1e-6, "max error {} too large", max_err);
    }

    /// GPU test: LDS accumulation in a for-range loop
    ///
    /// out[tid] = 0; for k in 0..K: out[tid] += 1.0; → should be K
    #[test]
    fn test_gpu_lds_loop_accum() {
        let rt = setup();
        let block_size: u32 = 32;
        let k_val: u32 = 10;

        let mut kb = BlockKernel::new("lds_accum", block_size);
        let out = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");
        let k_arg = kb.arg_u32("K");

        let tid = kb.arange(0, block_size);
        let mask = tid.lt(&mut kb, n_arg);

        // LDS accumulator
        let lds = kb.lds_alloc(block_size * 4);
        let zero_f = kb.const_f32(0.0);
        kb.lds_store(lds, tid, zero_f);
        kb.barrier();

        // for k in 0..K: lds[tid] += 1.0
        let zero = kb.const_u32(0);
        let one_f = kb.const_f32(1.0);
        let iter = kb.for_range(zero, k_arg, 1);
        {
            let cur = kb.lds_load(lds, tid);
            let new_val = cur.add(&mut kb, one_f);
            kb.lds_store(lds, tid, new_val);
        }
        kb.end_for(iter);

        // Read and store
        let result = kb.lds_load(lds, tid);
        kb.store(out, tid, result, mask);

        let compiled = kb.compile(Target::GFX1100).unwrap();
        eprintln!("  lds_accum: elf={} lds={}", compiled.elf.len(), compiled.lds_size);

        let out_buf = rt.alloc_f32(block_size as usize).unwrap();

        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        ka[8..12].copy_from_slice(&block_size.to_le_bytes());
        ka[12..16].copy_from_slice(&k_val.to_le_bytes());

        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        rt.dispatch(&gpu_kernel, [block_size, 1, 1], &ka).unwrap();

        let result = rt.read_f32(&out_buf, block_size as usize);
        eprintln!("  lds_accum result[0..4] = {:?}", &result[0..4]);
        let expected = k_val as f32;
        let err = (result[0] - expected).abs();
        eprintln!("  lds_accum: result[0]={} expected={} err={:.6e}", result[0], expected, err);
        assert!(err < 0.01, "LDS accumulation failed: got {} expected {}", result[0], expected);
    }

    /// GPU test: TN GEMM (C = A^T @ B)
    ///
    /// A: [K, M] (transposed), B: [K, N]
    /// C[m, n] = sum_k A[k*M + m] * B[k*N + n]
    ///
    /// 2D grid: pid_m tiles over M, pid_n tiles over N
    /// Each WG: BLOCK_M * BLOCK_N threads = 256
    /// Thread computes one element: C[pid_m*BM + local_m, pid_n*BN + local_n]
    #[test]
    fn test_gpu_gemm_tn() {
        let rt = setup();
        const BM: u32 = 16;
        const BN: u32 = 16;
        const BLOCK_SIZE: u32 = BM * BN; // 256
        let m: usize = 32;
        let n: usize = 32;
        let k_dim: usize = 64;

        let mut kb = BlockKernel::new("gemm_tn", BLOCK_SIZE);

        // Kernel args: A_ptr, B_ptr, C_ptr, M, N, K
        let a_ptr = kb.arg_ptr("A");
        let b_ptr = kb.arg_ptr("B");
        let c_ptr = kb.arg_ptr("C");
        let m_arg = kb.arg_u32("M");
        let n_arg = kb.arg_u32("N");
        let k_arg = kb.arg_u32("K");

        // 2D tile indices
        let pid_m = kb.program_id(0);  // tile row
        let pid_n = kb.program_id(1);  // tile col

        // Thread → (local_m, local_n) within tile
        let tid = kb.arange(0, BLOCK_SIZE);
        let local_n = tid.bitand(&mut kb, (BN - 1) as u32); // tid & 15
        let local_m = tid.shr(&mut kb, 4);                    // tid >> 4

        // Global indices
        let bm_const = kb.const_u32(BM);
        let bn_const = kb.const_u32(BN);
        let global_m = pid_m.mul(&mut kb, bm_const).add(&mut kb, local_m);
        let global_n = pid_n.mul(&mut kb, bn_const).add(&mut kb, local_n);

        // Accumulator (initialized to 0)
        let acc = kb.const_f32(0.0);

        // For k in range(0, K, 1):   (scalar loop)
        let zero = kb.const_u32(0);
        let iter_k = kb.for_range(zero, k_arg, 1);
        {
            // A[k, m] offset: k * M + global_m
            // Need to move iter_k (SGPR) to VGPR for arithmetic
            // iter_k is in val_to_sreg, get_vreg_u32 will auto-promote
            let a_offset = iter_k.mul(&mut kb, m_arg).add(&mut kb, global_m);
            let b_offset = iter_k.mul(&mut kb, n_arg).add(&mut kb, global_n);

            // Load A[k,m] and B[k,n] (no mask needed, dimensions are exact)
            let mn_total = m_arg.mul(&mut kb, n_arg);
            let mask_a = global_m.lt(&mut kb, m_arg);
            let a_val = kb.load(a_ptr, a_offset, mask_a);
            let mask_b = global_n.lt(&mut kb, n_arg);
            let b_val = kb.load(b_ptr, b_offset, mask_b);

            // acc += a * b (FMA)
            let prod = a_val.mul(&mut kb, b_val);
            let _new_acc = acc.add(&mut kb, prod);
            // Note: in a real SSA graph we'd need phi nodes for loop-carried values.
            // For now, each iteration creates new values. We'll accumulate differently.
        }
        kb.end_for(iter_k);

        // Since block_dsl is SSA (no mutable accumulators in loops yet),
        // we use a simpler approach: compute only for K=1 element per iteration
        // and sum manually outside... Actually, let's use a flat approach instead.

        // --- SIMPLIFIED: compute C[m,n] = sum over all K ---
        // We'll restructure: load all K elements and sum
        // But that doesn't work for large K...
        //
        // PRACTICAL SOLUTION: Use LDS as accumulator scratch space.
        // But the cleanest approach is to rethink the kernel.
        //
        // For this test, use a fully unrolled approach for small K,
        // or accept that the DSL needs loop-carried mutable state.

        // ALTERNATIVE: Write a naive GEMM without for_range
        // Each thread loops over K using a CPU-unrolled sequence
        // For K=64, this would be 64 load-pairs + 64 FMAs. Fine for a test.

        // Let's restart with a cleaner approach below...
        drop(kb);

        // ═══════════════════════════════════════
        // Clean TN GEMM: naive per-element, CPU-side loop unroll
        // ═══════════════════════════════════════

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

        // Bounds mask
        let mask_m = gm.lt(&mut kb, m_arg);
        let mask_n = gn.lt(&mut kb, n_arg);

        // Accumulate in LDS: each thread has its own LDS slot
        let lds = kb.lds_alloc(BLOCK_SIZE * 4);
        let lds_tid = kb.arange(0, BLOCK_SIZE);

        // Init accumulator in LDS to 0
        let zero_f = kb.const_f32(0.0);
        kb.lds_store(lds, lds_tid, zero_f);
        kb.barrier();

        // for k in 0..K:
        let zero = kb.const_u32(0);
        let iter_k = kb.for_range(zero, k_arg, 1);
        {
            // a_off = k * M + gm
            let a_off = iter_k.mul(&mut kb, m_arg).add(&mut kb, gm);
            let a_val = kb.load(a_ptr, a_off, mask_m);

            // b_off = k * N + gn
            let b_off = iter_k.mul(&mut kb, n_arg).add(&mut kb, gn);
            let b_val = kb.load(b_ptr, b_off, mask_n);

            // product
            let prod = a_val.mul(&mut kb, b_val);

            // Load current acc from LDS, add product, store back
            let cur = kb.lds_load(lds, lds_tid);
            let new_acc = cur.add(&mut kb, prod);
            kb.lds_store(lds, lds_tid, new_acc);
            kb.barrier();
        }
        kb.end_for(iter_k);

        // Read final result from LDS
        let result = kb.lds_load(lds, lds_tid);

        // Store C: c_off = gm * N + gn
        let c_off = gm.mul(&mut kb, n_arg).add(&mut kb, gn);
        kb.store(c_ptr, c_off, result, mask_m);

        let compiled = kb.compile(Target::GFX1100).unwrap();
        eprintln!("  gemm_tn: elf={} bytes, lds={}, ka={}",
            compiled.elf.len(), compiled.lds_size, compiled.kernarg_size);

        // Setup data: A[K,M], B[K,N] in f32
        let a_buf = rt.alloc_f32(k_dim * m).unwrap();
        let b_buf = rt.alloc_f32(k_dim * n).unwrap();
        let c_buf = rt.alloc_f32(m * n).unwrap();

        // Fill with simple data
        let a_data: Vec<f32> = (0..k_dim*m).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..k_dim*n).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        rt.write_f32(&a_buf, &a_data);
        rt.write_f32(&b_buf, &b_data);

        // Build kernargs: A, B, C, M, N, K
        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&c_buf.gpu_addr().to_le_bytes());
        ka[24..28].copy_from_slice(&(m as u32).to_le_bytes());
        ka[28..32].copy_from_slice(&(n as u32).to_le_bytes());
        ka[32..36].copy_from_slice(&(k_dim as u32).to_le_bytes());

        // 2D grid: [M/BM * BLOCK_SIZE, N/BN]... wait, grid is total threads
        // Each WG = BLOCK_SIZE threads, we need ceil(M/BM) x ceil(N/BN) WGs
        let grid_m = ((m as u32 + BM - 1) / BM) * BLOCK_SIZE;
        let grid_n = (n as u32 + BN - 1) / BN;

        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        rt.dispatch(&gpu_kernel, [grid_m, grid_n, 1], &ka).unwrap();

        // CPU reference: C = A^T @ B
        let mut c_ref = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k_dim {
                    sum += a_data[ki * m + mi] * b_data[ki * n + ni];
                }
                c_ref[mi * n + ni] = sum;
            }
        }

        let c_result = rt.read_f32(&c_buf, m * n);
        let mut max_err = 0f32;
        let mut max_err_idx = 0;
        for i in 0..m*n {
            let err = (c_result[i] - c_ref[i]).abs();
            if err > max_err { max_err = err; max_err_idx = i; }
        }
        // Show first 4 elements for debugging
        eprintln!("  C_gpu[0..4] = {:?}", &c_result[0..4]);
        eprintln!("  C_ref[0..4] = {:?}", &c_ref[0..4]);
        eprintln!("  worst: idx={} gpu={} ref={}", max_err_idx, c_result[max_err_idx], c_ref[max_err_idx]);
        eprintln!("  gemm_tn {}x{}x{}: max_err = {:.6e}", m, n, k_dim, max_err);
        // FP32 tolerance: K=64 accumulations, expect ~1e-5 error
        assert!(max_err < 1e-4, "GEMM max error {} too large", max_err);
    }

    /// GPU test: WMMA 16×16×16 with all-ones inputs
    ///
    /// A[16,16] = all 1.0 (bf16), B[16,16] = all 1.0 (bf16)
    /// Expected: C[r,c] = 16.0 for all (r,c)
    ///
    /// WMMA output layout for Wave32:
    ///   lane l → row = l % 16
    ///   lanes 0..15  → columns 0..7  (VGPR[j] = C[row, j])
    ///   lanes 16..31 → columns 8..15 (VGPR[j] = C[row, 8+j])
    #[test]
    fn test_gpu_wmma_16x16() {
        let rt = setup();

        // Build kernel: each wave (32 threads) does one 16×16×16 WMMA
        let mut kb = BlockKernel::new("wmma_test", 32);
        let out = kb.arg_ptr("out");
        let n_arg = kb.arg_u32("n");  // output cols (16)

        let tid = kb.arange(0, 32);

        // bf16(1.0) = 0x3F80 (same top 16 bits as f32(1.0) = 0x3F800000)
        // bf16x2(1.0, 1.0) = 0x3F803F80
        // NOTE: bf16 ≠ fp16! bf16 has 8-bit exponent (bias 127), fp16 has 5-bit (bias 15)
        let ones_bf16x2 = kb.const_u32(0x3F803F80);

        // Build A and B fragments: 8 VGPRs all = bf16x2(1.0, 1.0)
        let a_frag = ones_bf16x2.splat_fragment(&mut kb);
        let b_frag = ones_bf16x2.splat_fragment(&mut kb);

        // Zero accumulator
        let acc = kb.zero_acc();

        // WMMA: C = A * B + acc
        let result = a_frag.wmma(&mut kb, b_frag, acc);

        // row = tid % 16, col_base = (tid / 16) * 8
        let row = tid.bitand(&mut kb, 15);
        let eight = kb.const_u32(8);
        let col_base = tid.shr(&mut kb, 4).mul(&mut kb, eight);

        // Store each of the 8 outputs: C[row, col_base + j]
        // offset = row * N + col_base + j
        let row_times_n = row.mul(&mut kb, n_arg);
        let base_off = row_times_n.add(&mut kb, col_base);

        // Need mask for all lanes (always true for this test)
        let n_threads = kb.const_u32(32);
        let mask = tid.lt(&mut kb, n_threads);

        for j in 0..8u32 {
            let val = result.extract(&mut kb, j);
            let j_val = kb.const_u32(j);
            let off = base_off.add(&mut kb, j_val);
            kb.store(out, off, val, mask);
        }

        let compiled = kb.compile(Target::GFX1100).unwrap();
        eprintln!("  wmma_test: elf={} bytes, ka={}", compiled.elf.len(), compiled.kernarg_size);

        // Allocate 16×16 f32 output
        let out_buf = rt.alloc_f32(16 * 16).unwrap();
        out_buf.zero();

        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());
        ka[8..12].copy_from_slice(&16u32.to_le_bytes()); // N=16

        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        rt.dispatch(&gpu_kernel, [32, 1, 1], &ka).unwrap();

        let result = rt.read_f32(&out_buf, 16 * 16);
        eprintln!("  wmma result[0..8] = {:?}", &result[0..8]);
        eprintln!("  wmma result[8..16] = {:?}", &result[8..16]);

        let mut max_err = 0f32;
        let mut max_err_idx = 0;
        for i in 0..256 {
            let err = (result[i] - 16.0).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = i;
            }
        }
        eprintln!("  wmma 16x16: max_err={:.6e} (idx={}, got={})",
            max_err, max_err_idx, result[max_err_idx]);
        assert!(max_err < 0.01, "WMMA max error {} too large (idx={}, got={})",
            max_err, max_err_idx, result[max_err_idx]);
    }

    /// GPU test: TileGemm via block_dsl — the Tile IR path.
    ///
    /// Verifies that `kb.tile_gemm()` produces correct results by comparing
    /// against CPU reference, then measures TFLOPS to confirm performance
    /// parity with direct gemm_gen invocation.
    #[test]
    fn test_tile_gemm_via_block_dsl() {
        use crate::t0::gemm_gen;

        let rt = setup();
        let m: u32 = 128;
        let k_val: u32 = 128;
        let n: u32 = 128;

        // ── Build kernel using tile_gemm API ──
        // Use 128×64 k16 no-split — the most basic config
        let config = TileGemmConfig {
            tile_m: 128, tile_n: 64, tile_k: 16,
            wgp_mode: false, split_k: 1, swap_grid: true,
        };
        let gemm_cfg = config.to_gemm_config();
        eprintln!("[tile_gemm] config: {} (wg_size={})", config.name(), gemm_cfg.wg_size);
        // Use config-unique name to avoid compile_dsl cache returning stale kernels
        let kname = format!("tile_gemm_{}", config.name());
        let mut kb = BlockKernel::new(&kname, gemm_cfg.wg_size);
        // gemm_gen expects args: X, WT, Y, K, N, split_k_shift, y_split_stride
        let _x = kb.arg_ptr("X");
        let _w = kb.arg_ptr("WT");
        let _y = kb.arg_ptr("Y");
        let _k = kb.arg_u32("K");
        let _n = kb.arg_u32("N");
        let _sks = kb.arg_u32("split_k_shift");
        let _yss = kb.arg_u32("y_split_stride");
        kb.tile_gemm(_x, _w, _y, _k, _n, config.clone());

        let compiled = kb.compile(Target::GFX1100).unwrap();
        let compiled_elf_len = compiled.elf.len();
        let compiled_elf_bytes = compiled.elf.clone();
        eprintln!("[tile_gemm] compiled: elf={} bytes, wg={}, lds={}, ka={}",
            compiled_elf_len, compiled.workgroup_size[0],
            compiled.lds_size, compiled.kernarg_size);

        // ── Prepare random bf16 data (seeded for reproducibility) ──
        let mut rng_state: u64 = 42;
        let mut next_f = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((rng_state >> 33) as u32) % 200;
            (bits as f32 - 100.0) * 0.01  // range [-1.0, 1.0]
        };

        let x_f32: Vec<f32> = (0..m*k_val).map(|_| next_f()).collect();
        let w_f32: Vec<f32> = (0..n*k_val).map(|_| next_f()).collect();

        // Convert f32 → bf16 (truncate mantissa)
        let x_bf16: Vec<u16> = x_f32.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let w_bf16: Vec<u16> = w_f32.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

        let x_bytes: Vec<u8> = x_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_bytes: Vec<u8> = w_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let x_buf = rt.alloc(x_bytes.len()).unwrap();
        x_buf.write(&x_bytes);
        let w_buf = rt.alloc(w_bytes.len()).unwrap();
        w_buf.write(&w_bytes);

        let sk = config.split_k;
        let y_elems = if sk > 1 { m * n * sk } else { m * n };
        let y_buf = rt.alloc_f32(y_elems as usize).unwrap();
        y_buf.zero();

        // ── Build kernargs (gemm_gen layout: 40 bytes) ──
        let ka = gemm_gen::build_kernargs(
            x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
            k_val, n, m, &gemm_cfg,
        );

        // ── Dispatch ──
        let (grid_x, grid_y) = gemm_gen::compute_grid_auto(&gemm_cfg, m, n);
        eprintln!("[tile_gemm] grid=({}, {}) for {}x{}x{}", grid_x, grid_y, m, k_val, n);
        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        rt.dispatch(&gpu_kernel, [grid_x, grid_y, 1], &ka).unwrap();

        // ── CPU reference: Y = X @ W^T (NT mode, bf16 precision) ──
        let bf16_to_f32 = |b: u16| -> f32 {
            f32::from_bits((b as u32) << 16)
        };
        let mut y_ref = vec![0.0f32; (m * n) as usize];
        for mi in 0..m as usize {
            for ni in 0..n as usize {
                let mut sum = 0.0f32;
                for ki in 0..k_val as usize {
                    let a = bf16_to_f32(x_bf16[mi * k_val as usize + ki]);
                    let b = bf16_to_f32(w_bf16[ni * k_val as usize + ki]);
                    sum += a * b;
                }
                y_ref[mi * n as usize + ni] = sum;
            }
        }

        // Read GPU result
        let y_result = rt.read_f32(&y_buf, (m * n) as usize);

        // Diagnostics: check values at tile boundaries
        let n_us = n as usize;
        for row in [0usize, 1, 63, 64, 127] {
            if row < m as usize {
                for col in [0usize, 63, 64, 127] {
                    if col < n_us {
                        let idx = row * n_us + col;
                        let err = (y_result[idx] - y_ref[idx]).abs();
                        if err > 0.01 {
                            eprintln!("[tile_gemm] MISMATCH [{},{}] idx={}: gpu={:.6} ref={:.6} err={:.2e}",
                                row, col, idx, y_result[idx], y_ref[idx], err);
                        }
                    }
                }
            }
        }

        let mut max_err = 0f32;
        let mut max_idx = 0;
        let mut n_bad = 0usize;
        for i in 0..(m * n) as usize {
            let err = (y_result[i] - y_ref[i]).abs();
            if err > max_err { max_err = err; max_idx = i; }
            if err > 1.0 { n_bad += 1; }
        }

        eprintln!("[tile_gemm] Y_gpu[0..4] = {:?}", &y_result[0..4]);
        eprintln!("[tile_gemm] Y_ref[0..4] = {:?}", &y_ref[0..4]);
        eprintln!("[tile_gemm] {}x{}x{}: max_err={:.6e} (idx={}) n_bad={}/{}",
            m, k_val, n, max_err, max_idx, n_bad, m*n);

        // bf16 GEMM tolerance: K=128 accumulations, bf16 precision → ~0.05 error
        if max_err >= 0.1 {
            // ── Comparison: try raw KFD dispatch with the same ELF ──
            // If raw KFD works, the bug is in compile_dsl/GpuRuntime.
            // If raw KFD also fails, the bug is in the kernel ELF itself.
            eprintln!("[tile_gemm] GpuRuntime path failed. Testing raw KFD...");

            use crate::gpu_backend::{GpuKernel as RawGpuKernel, KernelLoadConfig, DispatchPool};

            // Recompile fresh kernel directly from gemm_gen
            let t0k_raw = gemm_gen::generate(&gemm_cfg);
            let elf_raw = t0k_raw.compile(Target::GFX1100).unwrap();
            eprintln!("[tile_gemm] raw ELF: {} bytes (tile_gemm ELF: {} bytes, same={})",
                elf_raw.len(), compiled_elf_len,
                elf_raw == compiled_elf_bytes);

            let raw_gk = RawGpuKernel::load(&rt.device, &elf_raw, &KernelLoadConfig {
                workgroup_size: [gemm_cfg.wg_size, 1, 1],
                lds_size: gemm_cfg.lds_total(),
            }).unwrap();

            let pool2 = DispatchPool::new(&rt.device, 4).unwrap();
            let y_buf2 = rt.alloc_f32((m * n) as usize).unwrap();
            y_buf2.zero();

            let ka2 = gemm_gen::build_kernargs(
                x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf2.gpu_addr(),
                k_val, n, m, &gemm_cfg,
            );
            let ka2_buf = pool2.write_kernargs(0, &ka2);
            rt.queue.submit(&raw_gk, [grid_x, grid_y, 1], ka2_buf);
            rt.queue.wait_idle().unwrap();

            let y_raw = rt.read_f32(&y_buf2, (m * n) as usize);

            let mut raw_max_err = 0f32;
            let mut raw_n_bad = 0usize;
            for i in 0..(m * n) as usize {
                let err = (y_raw[i] - y_ref[i]).abs();
                if err > raw_max_err { raw_max_err = err; }
                if err > 1.0 { raw_n_bad += 1; }
            }
            eprintln!("[tile_gemm] RAW KFD: Y[0..4]={:?}", &y_raw[0..4]);
            eprintln!("[tile_gemm] RAW KFD: max_err={:.6e} n_bad={}/{}",
                raw_max_err, raw_n_bad, m*n);

            assert!(max_err < 0.1,
                "[tile_gemm] FAILED: max_err={:.6e} at idx={} (gpu={} ref={}) raw_max_err={:.6e}",
                max_err, max_idx, y_result[max_idx], y_ref[max_idx], raw_max_err);
        }

        // ── Performance measurement ──
        let warmup = 3;
        let iters = 10;
        for _ in 0..warmup {
            rt.dispatch(&gpu_kernel, [grid_x, grid_y, 1], &ka).ok();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            rt.dispatch(&gpu_kernel, [grid_x, grid_y, 1], &ka).ok();
        }
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let flops = 2.0 * m as f64 * k_val as f64 * n as f64;
        let tflops = flops / (us * 1e6);
        eprintln!("[tile_gemm] {}x{}x{}: {:.1} μs, {:.2} TFLOPS",
            m, k_val, n, us, tflops);

        eprintln!("[PASS] test_tile_gemm_via_block_dsl");
    }

    /// GPU E2E: if-only (no else) — masked multiply via ExecMaskPush/Pop
    ///
    /// For lane i: if i < n/2, y[i] = x[i] * 2.0; else y[i] = x[i] (untouched).
    /// Verifies that ExecMaskPush correctly disables lanes and ExecMaskPop restores them.
    #[test]
    fn test_gpu_if_only() {
        let rt = setup();
        let n: usize = 64;
        let block_size: u32 = 64;

        let mut kb = BlockKernel::new("gpu_if_only", block_size);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let n_arg = kb.arg_u32("n");
        let half_n = kb.arg_u32("half_n");

        let pid = kb.program_id(0);
        let bs = kb.const_u32(block_size);
        let base = pid.mul(&mut kb, bs);
        let offsets = kb.arange(0, block_size).add(&mut kb, base);
        let mask = offsets.lt(&mut kb, n_arg);

        // Load x
        let a = kb.load(x, offsets, mask);

        // First: store x as-is (default value for all lanes)
        kb.store(y, offsets, a, mask);

        // Then: if (idx < half_n) { y[idx] = x[idx] * 2.0 }
        let cond = offsets.lt(&mut kb, half_n);
        kb.if_mask(cond);
        let two = kb.const_f32(2.0);
        let doubled = a.mul(&mut kb, two);
        kb.store(y, offsets, doubled, mask);
        kb.end_if();

        let compiled = kb.compile(Target::GFX1100).unwrap();
        eprintln!("  gpu_if_only: elf={} bytes, ka={}", compiled.elf.len(), compiled.kernarg_size);

        let x_buf = rt.alloc_f32(n).unwrap();
        let y_buf = rt.alloc_f32(n).unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        rt.write_f32(&x_buf, &x_data);

        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        ka[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        ka[20..24].copy_from_slice(&((n / 2) as u32).to_le_bytes());

        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        rt.dispatch(&gpu_kernel, [block_size, 1, 1], &ka).unwrap();

        let result = rt.read_f32(&y_buf, n);
        let mut max_err = 0f32;
        for i in 0..n {
            let expected = if i < n / 2 { x_data[i] * 2.0 } else { x_data[i] };
            let err = (result[i] - expected).abs();
            if err > max_err { max_err = err; }
            if err > 1e-6 {
                eprintln!("  MISMATCH [{}]: gpu={} expected={} err={:.2e}", i, result[i], expected, err);
            }
        }
        eprintln!("  gpu_if_only: max_err={:.6e}", max_err);
        assert!(max_err < 1e-6, "if-only max error {} too large", max_err);
        eprintln!("[PASS] test_gpu_if_only");
    }

    /// GPU E2E: if/else — conditional branch with ExecMaskPush/Flip/Pop
    ///
    /// if (idx < half_n) { y[idx] = x[idx] * 3.0 }
    /// else              { y[idx] = x[idx] + 100.0 }
    #[test]
    fn test_gpu_if_else() {
        let rt = setup();
        let n: usize = 64;
        let block_size: u32 = 64;

        let mut kb = BlockKernel::new("gpu_if_else", block_size);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let n_arg = kb.arg_u32("n");
        let half_n = kb.arg_u32("half_n");

        let pid = kb.program_id(0);
        let bs = kb.const_u32(block_size);
        let base = pid.mul(&mut kb, bs);
        let offsets = kb.arange(0, block_size).add(&mut kb, base);
        let mask = offsets.lt(&mut kb, n_arg);

        let a = kb.load(x, offsets, mask);

        let cond = offsets.lt(&mut kb, half_n);
        kb.if_mask(cond);
        // Then branch: y = x * 3.0
        let three = kb.const_f32(3.0);
        let tripled = a.mul(&mut kb, three);
        kb.store(y, offsets, tripled, mask);
        kb.else_mask();
        // Else branch: y = x + 100.0
        let hundred = kb.const_f32(100.0);
        let shifted = a.add(&mut kb, hundred);
        kb.store(y, offsets, shifted, mask);
        kb.end_if();

        let compiled = kb.compile(Target::GFX1100).unwrap();
        eprintln!("  gpu_if_else: elf={} bytes, ka={}", compiled.elf.len(), compiled.kernarg_size);

        let x_buf = rt.alloc_f32(n).unwrap();
        let y_buf = rt.alloc_f32(n).unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        rt.write_f32(&x_buf, &x_data);

        let mut ka = vec![0u8; compiled.kernarg_size];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        ka[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        ka[20..24].copy_from_slice(&((n / 2) as u32).to_le_bytes());

        let gpu_kernel = rt.compile_dsl(compiled).unwrap();
        rt.dispatch(&gpu_kernel, [block_size, 1, 1], &ka).unwrap();

        let result = rt.read_f32(&y_buf, n);
        let mut max_err = 0f32;
        for i in 0..n {
            let expected = if i < n / 2 { x_data[i] * 3.0 } else { x_data[i] + 100.0 };
            let err = (result[i] - expected).abs();
            if err > max_err { max_err = err; }
            if err > 1e-6 {
                eprintln!("  MISMATCH [{}]: gpu={} expected={} err={:.2e}", i, result[i], expected, err);
            }
        }
        eprintln!("  gpu_if_else: max_err={:.6e}", max_err);
        assert!(max_err < 1e-6, "if/else max error {} too large", max_err);
        eprintln!("[PASS] test_gpu_if_else");
    }
}

// CPU-only tests for new features
#[cfg(test)]
mod tests_new {
    use super::*;
    use crate::t0::ir::Target;

    #[test]
    fn test_lds_basic_compile() {
        let mut kb = BlockKernel::new("lds_test", 32);
        let x = kb.arg_ptr("x");
        let out = kb.arg_ptr("out");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let vals = kb.load(x, offsets, mask);

        let lds = kb.lds_alloc(32 * 4);
        kb.lds_store(lds, offsets, vals);
        kb.barrier();
        let back = kb.lds_load(lds, offsets);
        kb.store(out, offsets, back, mask);

        let compiled = kb.compile(Target::GFX1100).unwrap();
        assert!(compiled.lds_size >= 128, "LDS size should be >= 128");
        eprintln!("  test_lds_basic: lds_size={}, elf={} bytes", compiled.lds_size, compiled.elf.len());
    }

    #[test]
    fn test_for_range_basic_compile() {
        // Simple test: loop that accumulates into LDS
        let mut kb = BlockKernel::new("loop_test", 32);
        let n = kb.arg_u32("n");

        let zero = kb.const_u32(0);
        let step1 = kb.const_u32(1);
        let iter = kb.for_range(zero, n, 1);
        // Loop body is just a noop for compile test
        kb.end_for(iter);

        // Need at least one store for endpgm
        // Just do a trivial store
        let out = kb.arg_ptr("out");
        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);
        let c = kb.const_f32(42.0);
        kb.store(out, offsets, c, mask);

        let compiled = kb.compile(Target::GFX1100).unwrap();
        assert!(!compiled.elf.is_empty());
        eprintln!("  test_for_range_basic: elf={} bytes", compiled.elf.len());
    }

    /// T7: if-only (no else) — masked store within bounds
    #[test]
    fn test_if_only_compile() {
        let mut kb = BlockKernel::new("t7_if_only", 32);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);

        // Load all elements
        let a = kb.load(x, offsets, mask);
        let two = kb.const_f32(2.0);
        let doubled = a.mul(&mut kb, two);

        // if (idx < n) { y[idx] = x[idx] * 2.0 }
        kb.if_mask(mask);
        kb.store(y, offsets, doubled, mask);
        kb.end_if();

        // compile() now goes through SSA path (ExecMaskPush/Pop)
        let compiled = kb.compile(Target::GFX1100).unwrap();
        assert!(!compiled.elf.is_empty(), "ELF should not be empty");
        eprintln!("  [PASS] test_if_only_compile: elf={} bytes (SSA path)", compiled.elf.len());
    }

    /// T8: if/else — conditional branch with both paths
    #[test]
    fn test_if_else_compile() {
        let mut kb = BlockKernel::new("t8_if_else", 32);
        let x = kb.arg_ptr("x");
        let y = kb.arg_ptr("y");
        let n = kb.arg_u32("n");

        let offsets = kb.arange(0, 32);
        let mask = offsets.lt(&mut kb, n);

        // Load x (all lanes)
        let a = kb.load(x, offsets, mask);

        // if (idx < n) { y[idx] = x[idx] }
        // else { y[idx] = 0.0 }
        kb.if_mask(mask);
        kb.store(y, offsets, a, mask);
        kb.else_mask();
        let zero = kb.const_f32(0.0);
        kb.store(y, offsets, zero, mask);
        kb.end_if();

        // compile() now goes through SSA path (ExecMaskPush/Flip/Pop)
        let compiled = kb.compile(Target::GFX1100).unwrap();
        assert!(!compiled.elf.is_empty(), "ELF should not be empty");
        eprintln!("  [PASS] test_if_else_compile: elf={} bytes (SSA path)", compiled.elf.len());
    }
}
