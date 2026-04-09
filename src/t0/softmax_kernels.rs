//! Softmax GPU kernels — forward and backward.
//!
//! # Algorithm (Online Safe Softmax)
//!
//! Forward: per-row softmax on (rows × cols) matrix
//! ```text
//! max_val  = row_max(x)
//! exp_x    = exp(x - max_val)
//! sum_exp  = row_sum(exp_x)
//! y[i]     = exp_x[i] / sum_exp
//! ```
//!
//! Backward: given dY (upstream gradient) and Y (forward output)
//! ```text
//! dot_val  = row_sum(dY * Y)
//! dX[i]    = Y[i] * (dY[i] - dot_val)
//! ```
//!
//! # Design
//!
//! Each workgroup processes one row. Each thread handles one element.
//! WG-level reduce (wave reduce + LDS) computes row max and row sum.
//! Requires: cols ≤ WG_SIZE (256).
//! For larger cols, use multi-kernel approach (row_reduce_max + broadcast_sub_exp + etc.)

use super::block_dsl::*;
use super::ir::Target;

/// Workgroup size for softmax kernels.
const WG_SIZE: u32 = 256;

/// Build a Softmax forward kernel using block_dsl.
///
/// Kernarg layout: [input_ptr:u64, output_ptr:u64, cols:u32]
/// Grid: (rows * WG_SIZE, 1, 1) — one WG per row
///
/// Constraint: cols ≤ WG_SIZE (256)
pub fn build_softmax_forward() -> BlockKernel {
    let mut kb = BlockKernel::new("softmax_fwd", WG_SIZE);

    // Kernargs
    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let cols = kb.arg_u32("cols");

    // Thread/WG IDs
    let tid = kb.thread_id();
    let pid = kb.program_id(0); // row index

    // Offset = pid * cols + tid  (each thread loads one element)
    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols); // out-of-bounds lanes masked

    // Load element (masked: OOB threads get 0.0)
    let x = kb.load(input_ptr, offset, mask);

    // Phase 1: row max  — replace OOB with -inf for max reduction
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let x_for_max = mask.select(&mut kb, x, neg_inf);
    let row_max = kb.wg_reduce_max(x_for_max);

    // Phase 2: exp(x - max)
    let shifted = x.sub(&mut kb, row_max);
    let exp_x = shifted.exp(&mut kb);

    // Mask out-of-bounds to 0 for sum
    let zero_f = kb.const_f32(0.0);
    let exp_masked = mask.select(&mut kb, exp_x, zero_f);

    // Phase 3: row sum of exp
    let exp_sum = kb.wg_reduce_sum(exp_masked);
    let inv_sum = exp_sum.rcp(&mut kb);

    // Phase 4: output = exp(x - max) / sum
    let result = exp_x.mul(&mut kb, inv_sum);
    kb.store(output_ptr, offset, result, mask);

    kb
}

/// Build a Softmax backward kernel using block_dsl.
///
/// Kernarg layout: [dy_ptr:u64, y_ptr:u64, dx_ptr:u64, cols:u32]
/// Grid: (rows * WG_SIZE, 1, 1) — one WG per row
///
/// dX[i] = Y[i] * (dY[i] - dot(dY, Y))
///
/// Constraint: cols ≤ WG_SIZE (256)
pub fn build_softmax_backward() -> BlockKernel {
    let mut kb = BlockKernel::new("softmax_bwd", WG_SIZE);

    let dy_ptr = kb.arg_ptr("dy");
    let y_ptr = kb.arg_ptr("y");
    let dx_ptr = kb.arg_ptr("dx");
    let cols = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols);

    // Load dY and Y
    let dy = kb.load(dy_ptr, offset, mask);
    let y = kb.load(y_ptr, offset, mask);

    // dot(dY, Y) across row
    let prod = dy.mul(&mut kb, y);
    let zero_f = kb.const_f32(0.0);
    let prod_masked = mask.select(&mut kb, prod, zero_f);
    let dot_val = kb.wg_reduce_sum(prod_masked);

    // dX = Y * (dY - dot_val)
    let dy_shifted = dy.sub(&mut kb, dot_val);
    let dx = y.mul(&mut kb, dy_shifted);
    kb.store(dx_ptr, offset, dx, mask);

    kb
}

/// CPU reference: softmax forward
pub fn cpu_softmax_forward(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let out = &mut output[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        for c in 0..cols {
            out[c] = (row[c] - max_val).exp() / exp_sum;
        }
    }
}

/// CPU reference: softmax backward
pub fn cpu_softmax_backward(dy: &[f32], y: &[f32], dx: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let dy_row = &dy[r * cols..(r + 1) * cols];
        let y_row = &y[r * cols..(r + 1) * cols];
        let dx_row = &mut dx[r * cols..(r + 1) * cols];
        let dot_val: f32 = dy_row.iter().zip(y_row.iter()).map(|(&d, &y)| d * y).sum();
        for c in 0..cols {
            dx_row[c] = y_row[c] * (dy_row[c] - dot_val);
        }
    }
}

/// Get grid dimensions for softmax dispatch.
pub fn softmax_grid(rows: u32) -> (u32, u32) {
    (rows * WG_SIZE, 1)
}

/// Workgroup size for softmax kernels.
pub fn softmax_wg_size() -> u32 {
    WG_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_softmax_forward() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        cpu_softmax_forward(&input, &mut output, 2, 4);
        let sum_row0: f32 = output[0..4].iter().sum();
        let sum_row1: f32 = output[4..8].iter().sum();
        assert!((sum_row0 - 1.0).abs() < 1e-6, "row 0 sum: {}", sum_row0);
        assert!((sum_row1 - 1.0).abs() < 1e-6, "row 1 sum: {}", sum_row1);
    }

    #[test]
    fn test_cpu_softmax_backward() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        cpu_softmax_forward(&input, &mut y, 1, 4);
        let dy = vec![1.0, 0.0, 0.0, 0.0];
        let mut dx = vec![0.0; 4];
        cpu_softmax_backward(&dy, &y, &mut dx, 1, 4);
        let dx_sum: f32 = dx.iter().sum();
        assert!(dx_sum.abs() < 1e-6, "dx sum should be ~0: {}", dx_sum);
    }

    #[test]
    fn test_softmax_fwd_compiles() {
        let kb = build_softmax_forward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("softmax fwd should compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Softmax forward: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_softmax_bwd_compiles() {
        let kb = build_softmax_backward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("softmax bwd should compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Softmax backward: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_softmax_fwd_gpu() {
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
        let cols: u32 = 128;   // must be ≤ WG_SIZE (256)
        let n = (rows * cols) as usize;

        let input_data: Vec<f32> = (0..n).map(|i| {
            ((i as f32 * 0.37).sin() * 3.0)
        }).collect();

        let mut expected = vec![0.0f32; n];
        cpu_softmax_forward(&input_data, &mut expected, rows as usize, cols as usize);

        let input_buf = rt.upload_f32(&input_data).unwrap();
        let output_buf = rt.alloc_f32(n).unwrap();

        let kb = build_softmax_forward();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");

        let config = KernelLoadConfig {
            workgroup_size: ck.workgroup_size,
            lds_size: ck.lds_size,
        };
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &config).expect("load");

        let ka = crate::kernargs![
            input_buf.gpu_addr() => u64,
            output_buf.gpu_addr() => u64,
            cols => u32
        ];
        let (grid_x, _) = softmax_grid(rows);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let gpu_output = rt.read_f32(&output_buf, n);

        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (gpu_output[i] - expected[i]).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-3,
                "mismatch at {}: gpu={:.6} cpu={:.6}", i, gpu_output[i], expected[i]);
        }
        for r in 0..rows as usize {
            let sum: f32 = gpu_output[r * cols as usize..(r + 1) * cols as usize].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "row {} sum: {}", r, sum);
        }

        let _ = rt.wait_idle();
        eprintln!("✓ Softmax forward GPU: {}×{}, max_err={:.2e}", rows, cols, max_err);
    }
}
