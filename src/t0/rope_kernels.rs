//! RoPE (Rotary Position Embedding) GPU kernels — forward and backward.
//!
//! # Algorithm
//!
//! RoPE applies position-dependent rotation to pairs of features (cos/sin rotation):
//! ```text
//! For each pair (x[2i], x[2i+1]) at position pos:
//!   θ = pos / 10000^(2i / d_model)
//!   x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
//!   x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
//! ```
//!
//! # Design
//!
//! Each thread processes one pair of features.
//! WG processes one row (one token's embedding vector).
//! WG_SIZE = 128 (handles d_model ≤ 256 since each thread does 2 elements).
//!
//! Grid: (n_tokens * WG_SIZE, 1, 1)

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 128;

/// Build RoPE forward kernel.
///
/// Kernarg layout: [x:u64, out:u64, pos_ptr:u64, d_model:u32, inv_freq_base:f32]
///
/// - x: (n_tokens × d_model) f32 — input embeddings
/// - out: (n_tokens × d_model) f32 — output (rotated)
/// - pos_ptr: (n_tokens,) u32 — position index per token
/// - d_model: embedding dimension (must be even, ≤ WG_SIZE*2)
/// - inv_freq_base: base for frequency (default: 10000.0)
///
/// Grid: (n_tokens * WG_SIZE, 1, 1)
pub fn build_rope_forward() -> BlockKernel {
    let mut kb = BlockKernel::new("rope_fwd", WG_SIZE);

    let x_ptr = kb.arg_ptr("x");
    let out_ptr = kb.arg_ptr("out");
    let d_model = kb.arg_u32("d_model");
    let n_tokens = kb.arg_u32("n_tokens");

    let tid = kb.thread_id();     // pair index within row (0..d_model/2-1)
    let pid = kb.program_id(0);   // token index

    // Compute indices for the pair
    // even_idx = 2 * tid, odd_idx = 2 * tid + 1
    let two_u = kb.const_u32(2);
    let even_idx = tid.mul(&mut kb, two_u);
    let one = kb.const_u32(1);
    let odd_idx = even_idx.add(&mut kb, one);

    // Mask: ensure odd_idx < d_model (both even and odd in bounds)
    let pair_mask = odd_idx.lt(&mut kb, d_model);

    // Row base offset
    let row_base = pid.mul(&mut kb, d_model);
    let even_off = row_base.add(&mut kb, even_idx);
    let odd_off = row_base.add(&mut kb, odd_idx);

    // Load even and odd elements
    let x_even = kb.load(x_ptr, even_off, pair_mask);
    let x_odd = kb.load(x_ptr, odd_off, pair_mask);

    // Compute frequency: θ = pos * (1 / base^(2i/d_model))
    // = pos * exp(-ln(base) * 2i/d_model)
    // Simplified: θ_i = pos / base^(2i/d_model)
    // Using float arithmetic: freq_i = 1.0 / base^(2*tid / d_model)
    //
    // For numerical stability, compute log-domain:
    // log(freq_i) = -(2*tid / d_model) * ln(base)
    // freq_i = exp(log(freq_i))
    //
    // Then θ = pos * freq_i

    // inv_freq[i] = 1.0 / (10000^(2*i / d_model))
    // We pre-compute using the formula: exp(-2*i/d * ln(10000))

    // Step 1: ratio = 2*tid / d_model (float)
    let tid_f32 = tid.to_f32(&mut kb);
    let two_f = kb.const_f32(2.0);
    let two_tid_f = tid_f32.mul(&mut kb, two_f);
    let d_model_f = d_model.to_f32(&mut kb);
    let inv_d_model = d_model_f.rcp(&mut kb);
    let ratio = two_tid_f.mul(&mut kb, inv_d_model);

    // Step 2: neg_ratio * ln(base) = -ratio * ln(10000) ≈ -ratio * 9.21034
    let neg_ln_base = kb.const_f32(-9.21034); // -ln(10000)
    let log_freq = ratio.mul(&mut kb, neg_ln_base);

    // Step 3: freq = exp(log_freq)
    let freq = log_freq.exp(&mut kb);

    // Step 4: theta = pos * freq (pos is just the row index = pid)
    let pos_f = pid.to_f32(&mut kb);
    let theta = pos_f.mul(&mut kb, freq);

    // Step 5: cos(θ) and sin(θ)
    let cos_theta = theta.cos(&mut kb);
    let sin_theta = theta.sin(&mut kb);

    // Step 6: Apply rotation
    // out_even = x_even * cos - x_odd * sin
    // out_odd  = x_even * sin + x_odd * cos
    let xc = x_even.mul(&mut kb, cos_theta);
    let xs = x_odd.mul(&mut kb, sin_theta);
    let out_even = xc.sub(&mut kb, xs);

    let xe_sin = x_even.mul(&mut kb, sin_theta);
    let xo_cos = x_odd.mul(&mut kb, cos_theta);
    let out_odd = xe_sin.add(&mut kb, xo_cos);

    // Store results
    kb.store(out_ptr, even_off, out_even, pair_mask);
    kb.store(out_ptr, odd_off, out_odd, pair_mask);

    kb
}

/// Build RoPE backward kernel.
///
/// The backward of RoPE is the inverse rotation (transpose of rotation matrix):
/// ```text
/// dx_even = dout_even * cos(θ) + dout_odd * sin(θ)
/// dx_odd  = -dout_even * sin(θ) + dout_odd * cos(θ)
/// ```
///
/// Kernarg layout: same as forward (dout replaces x, dx replaces out)
pub fn build_rope_backward() -> BlockKernel {
    let mut kb = BlockKernel::new("rope_bwd", WG_SIZE);

    let dout_ptr = kb.arg_ptr("dout");
    let dx_ptr = kb.arg_ptr("dx");
    let d_model = kb.arg_u32("d_model");
    let n_tokens = kb.arg_u32("n_tokens");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let two_u = kb.const_u32(2);
    let even_idx = tid.mul(&mut kb, two_u);
    let one = kb.const_u32(1);
    let odd_idx = even_idx.add(&mut kb, one);
    let pair_mask = odd_idx.lt(&mut kb, d_model);

    let row_base = pid.mul(&mut kb, d_model);
    let even_off = row_base.add(&mut kb, even_idx);
    let odd_off = row_base.add(&mut kb, odd_idx);

    let dout_even = kb.load(dout_ptr, even_off, pair_mask);
    let dout_odd = kb.load(dout_ptr, odd_off, pair_mask);

    // Same frequency computation as forward
    let tid_f32 = tid.to_f32(&mut kb);
    let two_f = kb.const_f32(2.0);
    let two_tid_f = tid_f32.mul(&mut kb, two_f);
    let d_model_f = d_model.to_f32(&mut kb);
    let inv_d_model = d_model_f.rcp(&mut kb);
    let ratio = two_tid_f.mul(&mut kb, inv_d_model);
    let neg_ln_base = kb.const_f32(-9.21034);
    let log_freq = ratio.mul(&mut kb, neg_ln_base);
    let freq = log_freq.exp(&mut kb);
    let pos_f = pid.to_f32(&mut kb);
    let theta = pos_f.mul(&mut kb, freq);
    let cos_theta = theta.cos(&mut kb);
    let sin_theta = theta.sin(&mut kb);

    // Inverse rotation (transpose)
    // dx_even = dout_even * cos + dout_odd * sin
    let dc = dout_even.mul(&mut kb, cos_theta);
    let ds = dout_odd.mul(&mut kb, sin_theta);
    let dx_even = dc.add(&mut kb, ds);

    // dx_odd = -dout_even * sin + dout_odd * cos
    let neg_de_sin = dout_even.mul(&mut kb, sin_theta).neg(&mut kb);
    let do_cos = dout_odd.mul(&mut kb, cos_theta);
    let dx_odd = neg_de_sin.add(&mut kb, do_cos);

    kb.store(dx_ptr, even_off, dx_even, pair_mask);
    kb.store(dx_ptr, odd_off, dx_odd, pair_mask);

    kb
}

/// CPU reference: RoPE forward
pub fn cpu_rope_forward(x: &[f32], out: &mut [f32], n_tokens: usize, d_model: usize, base: f32) {
    let half_d = d_model / 2;
    for t in 0..n_tokens {
        for i in 0..half_d {
            let freq = 1.0 / base.powf(2.0 * i as f32 / d_model as f32);
            let theta = t as f32 * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let even = t * d_model + 2 * i;
            let odd = even + 1;
            out[even] = x[even] * cos_t - x[odd] * sin_t;
            out[odd]  = x[even] * sin_t + x[odd] * cos_t;
        }
    }
}

/// CPU reference: RoPE backward
pub fn cpu_rope_backward(dout: &[f32], dx: &mut [f32], n_tokens: usize, d_model: usize, base: f32) {
    let half_d = d_model / 2;
    for t in 0..n_tokens {
        for i in 0..half_d {
            let freq = 1.0 / base.powf(2.0 * i as f32 / d_model as f32);
            let theta = t as f32 * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let even = t * d_model + 2 * i;
            let odd = even + 1;
            dx[even] =  dout[even] * cos_t + dout[odd] * sin_t;
            dx[odd]  = -dout[even] * sin_t + dout[odd] * cos_t;
        }
    }
}

pub fn rope_grid(n_tokens: u32) -> (u32, u32) { (n_tokens * WG_SIZE, 1) }
pub fn rope_wg_size() -> u32 { WG_SIZE }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_rope() {
        let d = 8; // 4 pairs
        let n = 2; // 2 tokens
        let x: Vec<f32> = (0..n*d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut out = vec![0.0f32; n * d];
        let mut dx = vec![0.0f32; n * d];

        cpu_rope_forward(&x, &mut out, n, d, 10000.0);

        // Token 0 (pos=0): θ=0 → cos=1, sin=0 → identity
        for i in 0..d {
            assert!((out[i] - x[i]).abs() < 1e-5,
                "pos=0: out[{}]={} expected {}", i, out[i], x[i]);
        }

        // Backward should be inverse
        cpu_rope_backward(&out, &mut dx, n, d, 10000.0);
        for i in 0..n*d {
            assert!((dx[i] - x[i]).abs() < 1e-4,
                "roundtrip: dx[{}]={} expected {}", i, dx[i], x[i]);
        }
    }

    #[test]
    fn test_rope_fwd_compiles() {
        let kb = build_rope_forward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("RoPE fwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ RoPE fwd: {} bytes ELF, wg={:?}", ck.elf.len(), ck.workgroup_size);
    }

    #[test]
    fn test_rope_bwd_compiles() {
        let kb = build_rope_backward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("RoPE bwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ RoPE bwd: {} bytes ELF, wg={:?}", ck.elf.len(), ck.workgroup_size);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_rope_fwd_gpu() {
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::gpu_backend::{GpuKernel, KernelLoadConfig};
        use std::sync::{Arc, OnceLock};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("GPU runtime"))
        }).0.clone();
        let _ = rt.wait_idle();

        let n_tokens: u32 = 4;
        let d_model: u32 = 64;
        let n = (n_tokens * d_model) as usize;

        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.17).sin() * 2.0)).collect();
        let mut expected = vec![0.0f32; n];
        cpu_rope_forward(&x, &mut expected, n_tokens as usize, d_model as usize, 10000.0);

        let x_buf = rt.upload_f32(&x).unwrap();
        let out_buf = rt.alloc_f32(n).unwrap();

        let kb = build_rope_forward();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let config = KernelLoadConfig {
            workgroup_size: ck.workgroup_size,
            lds_size: ck.lds_size,
        };
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &config).expect("load");

        let ka = crate::kernargs![
            x_buf.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            d_model => u32,
            n_tokens => u32
        ];
        let (grid_x, _) = rope_grid(n_tokens);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let gpu_out = rt.read_f32(&out_buf, n);

        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (gpu_out[i] - expected[i]).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-2,
                "RoPE[{}]: gpu={:.6} cpu={:.6} err={:.6}",
                i, gpu_out[i], expected[i], err);
        }

        let _ = rt.wait_idle();
        eprintln!("✓ RoPE fwd GPU: {}×{}, max_err={:.2e}", n_tokens, d_model, max_err);
    }
}
