//! RMSNorm GPU kernels — forward and backward.
//!
//! # Algorithm
//!
//! Forward: per-row RMSNorm
//! ```text
//! rms   = sqrt(mean(x²) + eps)
//! y[i]  = (x[i] / rms) * weight[i]
//! ```
//!
//! Backward: given dY (upstream gradient), X (input), Weight, and saved rstd
//! ```text
//! rstd     = 1 / rms
//! dx_hat   = dY * weight
//! dot_val  = mean(x * dx_hat)
//! dX[i]    = rstd * (dx_hat[i] - x[i] * rstd * dot_val)
//! dWeight += sum_over_rows(dY * x * rstd)  [via atomic_add]
//! ```
//!
//! # Design
//!
//! Each workgroup processes one row. Each thread handles one element.
//! WG-level reduce computes row mean(x²).
//! Constraint: cols ≤ WG_SIZE (256).

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build RMSNorm forward kernel.
///
/// Kernarg layout: [input:u64, weight:u64, output:u64, cols:u32, eps:f32]
/// Grid: (rows * WG_SIZE, 1, 1) — one WG per row
///
/// Constraint: cols ≤ WG_SIZE (256)
pub fn build_rmsnorm_forward() -> BlockKernel {
    let mut kb = BlockKernel::new("rmsnorm_fwd", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let weight_ptr = kb.arg_ptr("weight");
    let output_ptr = kb.arg_ptr("output");
    let cols = kb.arg_u32("cols");
    let eps = kb.arg_f32("eps");

    let tid = kb.thread_id();
    let pid = kb.program_id(0); // row index

    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols);

    // Load input element
    let x = kb.load(input_ptr, offset, mask);

    // Phase 1: mean(x²)
    let x_sq = x.mul(&mut kb, x);
    let zero_f = kb.const_f32(0.0);
    let x_sq_masked = mask.select(&mut kb, x_sq, zero_f);
    let sum_sq = kb.wg_reduce_sum(x_sq_masked);
    let cols_f = cols.to_f32(&mut kb);
    let mean_sq = sum_sq.div(&mut kb, cols_f);

    // Phase 2: rstd = rsqrt(mean_sq + eps)
    let mean_sq_eps = mean_sq.add(&mut kb, eps);
    let rstd = mean_sq_eps.rsqrt(&mut kb);

    // Phase 3: y = (x * rstd) * weight
    let x_norm = x.mul(&mut kb, rstd);
    let w = kb.load(weight_ptr, tid, mask); // weight[col]
    let y = x_norm.mul(&mut kb, w);
    kb.store(output_ptr, offset, y, mask);

    kb
}

/// Build RMSNorm backward kernel.
///
/// Kernarg layout: [dy:u64, x:u64, weight:u64, dx:u64, cols:u32, eps:f32]
/// Grid: (rows * WG_SIZE, 1, 1) — one WG per row
///
/// dX = rstd * (dx_hat - x * rstd² * mean(x * dx_hat))
/// where dx_hat = dY * weight
///
/// Note: dWeight accumulation (over rows) requires a separate reduce
/// or atomic_add pass; this kernel only computes dX.
pub fn build_rmsnorm_backward() -> BlockKernel {
    let mut kb = BlockKernel::new("rmsnorm_bwd", WG_SIZE);

    let dy_ptr = kb.arg_ptr("dy");
    let x_ptr = kb.arg_ptr("x");
    let weight_ptr = kb.arg_ptr("weight");
    let dx_ptr = kb.arg_ptr("dx");
    let cols = kb.arg_u32("cols");
    let eps = kb.arg_f32("eps");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols);

    // Load
    let dy = kb.load(dy_ptr, offset, mask);
    let x = kb.load(x_ptr, offset, mask);
    let w = kb.load(weight_ptr, tid, mask);

    // Recompute rstd from x
    let x_sq = x.mul(&mut kb, x);
    let zero_f = kb.const_f32(0.0);
    let x_sq_masked = mask.select(&mut kb, x_sq, zero_f);
    let sum_sq = kb.wg_reduce_sum(x_sq_masked);
    let cols_f = cols.to_f32(&mut kb);
    let mean_sq = sum_sq.div(&mut kb, cols_f);
    let mean_sq_eps = mean_sq.add(&mut kb, eps);
    let rstd = mean_sq_eps.rsqrt(&mut kb);

    // dx_hat = dy * weight
    let dx_hat = dy.mul(&mut kb, w);

    // mean(x * dx_hat)
    let x_dx_hat = x.mul(&mut kb, dx_hat);
    let x_dx_hat_masked = mask.select(&mut kb, x_dx_hat, zero_f);
    let dot_sum = kb.wg_reduce_sum(x_dx_hat_masked);
    let dot_mean = dot_sum.div(&mut kb, cols_f);

    // dX = rstd * (dx_hat - x * rstd² * dot_mean)
    let rstd_sq = rstd.mul(&mut kb, rstd);
    let x_rstd_sq = x.mul(&mut kb, rstd_sq);
    let correction = x_rstd_sq.mul(&mut kb, dot_mean);
    let dx_unnorm = dx_hat.sub(&mut kb, correction);
    let dx = dx_unnorm.mul(&mut kb, rstd);
    kb.store(dx_ptr, offset, dx, mask);

    kb
}

/// CPU reference: RMSNorm forward
pub fn cpu_rmsnorm_forward(
    input: &[f32], weight: &[f32], output: &mut [f32],
    rows: usize, cols: usize, eps: f32,
) {
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let out = &mut output[r * cols..(r + 1) * cols];
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / cols as f32;
        let rstd = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..cols {
            out[c] = row[c] * rstd * weight[c];
        }
    }
}

/// CPU reference: RMSNorm backward (dX only)
pub fn cpu_rmsnorm_backward(
    dy: &[f32], x: &[f32], weight: &[f32], dx: &mut [f32],
    rows: usize, cols: usize, eps: f32,
) {
    for r in 0..rows {
        let dy_row = &dy[r * cols..(r + 1) * cols];
        let x_row = &x[r * cols..(r + 1) * cols];
        let dx_row = &mut dx[r * cols..(r + 1) * cols];
        let mean_sq: f32 = x_row.iter().map(|&x| x * x).sum::<f32>() / cols as f32;
        let rstd = 1.0 / (mean_sq + eps).sqrt();
        // dx_hat = dy * weight
        let dx_hat: Vec<f32> = dy_row.iter().zip(weight.iter()).map(|(&d, &w)| d * w).collect();
        // dot_mean = mean(x * dx_hat)
        let dot_val: f32 = x_row.iter().zip(dx_hat.iter()).map(|(&x, &dh)| x * dh).sum::<f32>() / cols as f32;
        for c in 0..cols {
            dx_row[c] = rstd * (dx_hat[c] - x_row[c] * rstd * rstd * dot_val);
        }
    }
}

pub fn rmsnorm_grid(rows: u32) -> (u32, u32) { (rows * WG_SIZE, 1) }
pub fn rmsnorm_wg_size() -> u32 { WG_SIZE }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_rmsnorm_forward() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        cpu_rmsnorm_forward(&input, &weight, &mut output, 1, 4, 1e-5);
        // RMS = sqrt(mean(1+4+9+16)) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32 + 1e-5).sqrt();
        for c in 0..4 {
            let expected = input[c] / rms;
            assert!((output[c] - expected).abs() < 1e-5,
                "rmsnorm[{}]: got={} expected={}", c, output[c], expected);
        }
    }

    #[test]
    fn test_cpu_rmsnorm_backward() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let dy = vec![1.0, 0.0, 0.0, 0.0];
        let mut dx = vec![0.0; 4];
        cpu_rmsnorm_backward(&dy, &x, &weight, &mut dx, 1, 4, 1e-5);
        // dx should have near-zero sum (approximately)
        let dx_weighted_sum: f32 = dx.iter().zip(x.iter()).map(|(&d, &x)| d * x).sum();
        assert!(dx_weighted_sum.abs() < 1e-4, "dx·x should be ~0: {}", dx_weighted_sum);
    }

    #[test]
    fn test_rmsnorm_fwd_compiles() {
        let kb = build_rmsnorm_forward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("rmsnorm fwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ RMSNorm forward: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_rmsnorm_bwd_compiles() {
        let kb = build_rmsnorm_backward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("rmsnorm bwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ RMSNorm backward: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_rmsnorm_fwd_gpu() {
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

        let rows: u32 = 4;
        let cols: u32 = 64;
        let eps: f32 = 1e-5;
        let n = (rows * cols) as usize;

        let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.31).sin() * 2.0)).collect();
        let weight: Vec<f32> = (0..cols as usize).map(|i| 0.5 + (i as f32 * 0.01)).collect();
        let mut expected = vec![0.0f32; n];
        cpu_rmsnorm_forward(&input, &weight, &mut expected, rows as usize, cols as usize, eps);

        let input_buf = rt.upload_f32(&input).unwrap();
        let weight_buf = rt.upload_f32(&weight).unwrap();
        let output_buf = rt.alloc_f32(n).unwrap();

        let kb = build_rmsnorm_forward();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            input_buf.gpu_addr() => u64,
            weight_buf.gpu_addr() => u64,
            output_buf.gpu_addr() => u64,
            cols => u32,
            eps => f32
        ];
        let (grid_x, _) = rmsnorm_grid(rows);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let gpu_output = rt.read_f32(&output_buf, n);
        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (gpu_output[i] - expected[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-3, "RMSNorm fwd max_err={}", max_err);
        let _ = rt.wait_idle();
        eprintln!("✓ RMSNorm forward GPU: {}×{}, max_err={:.2e}", rows, cols, max_err);
    }
}
