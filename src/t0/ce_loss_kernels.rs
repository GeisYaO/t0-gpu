//! Cross-Entropy Loss GPU kernels — forward and backward.
//!
//! # Design (2-kernel approach)
//!
//! **Forward**: Two kernels:
//! 1. `build_log_softmax()` — computes log_softmax for all positions
//! 2. `build_nll_loss()` — extracts loss = -log_softmax[target] (simple gather)
//!
//! **Backward**: Single kernel:
//! - `build_ce_loss_backward()` — dX = (softmax - one_hot) * scale
//!
//! This avoids Triton's 3x wg_reduce + target comparison pattern
//! which causes GPU hangs with our current wg_reduce LDS implementation.

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build log-softmax kernel. Reuses the softmax algorithm but stores log(softmax).
///
/// Kernarg layout: [input:u64, output:u64, cols:u32]
/// Grid: (rows * WG_SIZE, 1, 1) — one WG per row
///
/// output[r,c] = x[r,c] - max_r - log(sum_r(exp(x - max_r)))
pub fn build_log_softmax() -> BlockKernel {
    let mut kb = BlockKernel::new("log_softmax", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let cols = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols);

    let x = kb.load(input_ptr, offset, mask);

    // Row max
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let x_for_max = mask.select(&mut kb, x, neg_inf);
    let row_max = kb.wg_reduce_max(x_for_max);

    // sum(exp(x - max))
    let shifted = x.sub(&mut kb, row_max);
    let exp_x = shifted.exp(&mut kb);
    let zero_f = kb.const_f32(0.0);
    let exp_masked = mask.select(&mut kb, exp_x, zero_f);
    let exp_sum = kb.wg_reduce_sum(exp_masked);
    let log_sum = exp_sum.log(&mut kb);

    // log_softmax = (x - max) - log(sum) = shifted - log_sum
    let log_softmax = shifted.sub(&mut kb, log_sum);
    kb.store(output_ptr, offset, log_softmax, mask);

    kb
}

/// Build NLL Loss kernel — simple gather: loss[r] = -log_softmax[r, target[r]]
///
/// Kernarg layout: [log_softmax:u64, targets:u64, losses:u64, cols:u32]
/// Grid: (rows * WG_SIZE, 1, 1) — one WG per row
///
/// Each thread checks if tid == target, then lane-reduces the result.
pub fn build_nll_loss() -> BlockKernel {
    let mut kb = BlockKernel::new("nll_loss", WG_SIZE);

    let log_softmax_ptr = kb.arg_ptr("log_softmax");
    let targets_ptr = kb.arg_ptr("targets");
    let losses_ptr = kb.arg_ptr("losses");
    let cols = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols);

    // Load log_softmax value for this thread's position
    let log_sm = kb.load(log_softmax_ptr, offset, mask);

    // Load target (broadcast pid to vector)
    let zero_u = kb.const_u32(0);
    let pid_vec = tid.mul(&mut kb, zero_u).add(&mut kb, pid);
    let target_val = kb.load_u32(targets_ptr, pid_vec, mask);

    // Check if tid == target: diff = tid - target, eq iff diff < 1
    let diff = tid.sub(&mut kb, target_val);
    let one_u = kb.const_u32(1);
    let tid_eq_target = diff.lt(&mut kb, one_u);

    // contribution = (tid == target) ? -log_sm : 0.0
    let zero_f = kb.const_f32(0.0);
    let neg_log_sm = log_sm.neg(&mut kb);
    let contribution = tid_eq_target.select(&mut kb, neg_log_sm, zero_f);

    // Sum contributions (only one thread contributes)
    let loss_val = kb.wg_reduce_sum(contribution);

    // Thread 0 writes loss
    let is_t0 = tid.lt(&mut kb, one_u);
    kb.store(losses_ptr, pid_vec, loss_val, is_t0);

    kb
}

/// Build CE backward kernel: dX = (softmax - one_hot) * scale
///
/// Kernarg layout: [logits:u64, targets:u64, dlogits:u64, cols:u32, scale:f32]
/// Grid: (rows * WG_SIZE, 1, 1)
pub fn build_ce_loss_backward() -> BlockKernel {
    let mut kb = BlockKernel::new("ce_loss_bwd", WG_SIZE);

    let logits_ptr = kb.arg_ptr("logits");
    let targets_ptr = kb.arg_ptr("targets");
    let dlogits_ptr = kb.arg_ptr("dlogits");
    let cols = kb.arg_u32("cols");
    let scale = kb.arg_f32("scale");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols);

    let x = kb.load(logits_ptr, offset, mask);

    // Softmax
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let x_for_max = mask.select(&mut kb, x, neg_inf);
    let row_max = kb.wg_reduce_max(x_for_max);
    let shifted = x.sub(&mut kb, row_max);
    let exp_x = shifted.exp(&mut kb);
    let zero_f = kb.const_f32(0.0);
    let exp_masked = mask.select(&mut kb, exp_x, zero_f);
    let exp_sum = kb.wg_reduce_sum(exp_masked);
    let inv_sum = exp_sum.rcp(&mut kb);
    let softmax_val = exp_x.mul(&mut kb, inv_sum);

    // Load target (broadcast pid → vector)
    let zero_u = kb.const_u32(0);
    let pid_vec = tid.mul(&mut kb, zero_u).add(&mut kb, pid);
    let target_val = kb.load_u32(targets_ptr, pid_vec, mask);

    // one_hot: tid == target → diff < 1
    let diff = tid.sub(&mut kb, target_val);
    let one_u = kb.const_u32(1);
    let tid_eq_target = diff.lt(&mut kb, one_u);
    let one_f = kb.const_f32(1.0);
    let one_hot = tid_eq_target.select(&mut kb, one_f, zero_f);

    // dLogits = (softmax - one_hot) * scale
    let grad = softmax_val.sub(&mut kb, one_hot);
    let grad_scaled = grad.mul(&mut kb, scale);
    kb.store(dlogits_ptr, offset, grad_scaled, mask);

    kb
}

/// CPU reference: CE loss forward
pub fn cpu_ce_loss_forward(logits: &[f32], targets: &[u32], losses: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        let target = targets[r] as usize;
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
        losses[r] = log_sum_exp + max_val - row[target];
    }
}

/// CPU reference: CE loss backward
pub fn cpu_ce_loss_backward(
    logits: &[f32], targets: &[u32], dlogits: &mut [f32],
    rows: usize, cols: usize, scale: f32,
) {
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        let target = targets[r] as usize;
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        for c in 0..cols {
            let softmax_val = (row[c] - max_val).exp() / exp_sum;
            let one_hot = if c == target { 1.0 } else { 0.0 };
            dlogits[r * cols + c] = (softmax_val - one_hot) * scale;
        }
    }
}

pub fn ce_loss_grid(rows: u32) -> (u32, u32) { (rows * WG_SIZE, 1) }
pub fn ce_loss_wg_size() -> u32 { WG_SIZE }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_ce_loss() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![2u32];
        let mut losses = vec![0.0f32; 1];
        cpu_ce_loss_forward(&logits, &targets, &mut losses, 1, 4);
        let expected = 1.44019;
        assert!((losses[0] - expected).abs() < 1e-3,
            "CE loss: {} expected ~{}", losses[0], expected);

        let mut dlogits = vec![0.0f32; 4];
        cpu_ce_loss_backward(&logits, &targets, &mut dlogits, 1, 4, 1.0);
        let dsum: f32 = dlogits.iter().sum();
        assert!(dsum.abs() < 1e-6, "dlogits sum: {}", dsum);
    }

    #[test]
    fn test_log_softmax_compiles() {
        let kb = build_log_softmax();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("log_softmax compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ log_softmax: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_nll_loss_compiles() {
        let kb = build_nll_loss();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("nll_loss compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ nll_loss: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_ce_bwd_compiles() {
        let kb = build_ce_loss_backward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("ce_bwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ ce_bwd: {} bytes ELF", ck.elf.len());
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_ce_loss_fwd_gpu() {
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
        let cols: u32 = 32;
        let n = (rows * cols) as usize;

        let logits: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.23).sin() * 5.0)).collect();
        let targets: Vec<u32> = (0..rows).map(|r| r % cols).collect();

        let mut expected_loss = vec![0.0f32; rows as usize];
        cpu_ce_loss_forward(&logits, &targets, &mut expected_loss, rows as usize, cols as usize);

        let logits_buf = rt.upload_f32(&logits).unwrap();
        let log_sm_buf = rt.alloc_f32(n).unwrap();
        let loss_buf = rt.alloc_f32(rows as usize).unwrap();
        // Upload targets as u32 bytes
        let targets_f32: Vec<f32> = targets.iter().map(|&t| f32::from_bits(t)).collect();
        let targets_buf = rt.upload_f32(&targets_f32).unwrap();

        // Kernel 1: log_softmax
        let kb1 = build_log_softmax();
        let ck1 = kb1.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile log_sm");
        let k1 = GpuKernel::load(&rt.device, &ck1.elf, &KernelLoadConfig {
            workgroup_size: ck1.workgroup_size, lds_size: ck1.lds_size,
        }).expect("load log_sm");

        let ka1 = crate::kernargs![
            logits_buf.gpu_addr() => u64,
            log_sm_buf.gpu_addr() => u64,
            cols => u32
        ];
        let (grid_x, _) = ce_loss_grid(rows);
        rt.dispatch(&k1, [grid_x, 1, 1], &ka1).expect("dispatch log_sm");

        // Wait for log_softmax to complete before nll_loss reads its output
        let _ = rt.wait_idle();

        // Kernel 2: nll_loss
        let kb2 = build_nll_loss();
        let ck2 = kb2.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile nll");
        let k2 = GpuKernel::load(&rt.device, &ck2.elf, &KernelLoadConfig {
            workgroup_size: ck2.workgroup_size, lds_size: ck2.lds_size,
        }).expect("load nll");

        let ka2 = crate::kernargs![
            log_sm_buf.gpu_addr() => u64,
            targets_buf.gpu_addr() => u64,
            loss_buf.gpu_addr() => u64,
            cols => u32
        ];
        rt.dispatch(&k2, [grid_x, 1, 1], &ka2).expect("dispatch nll");

        let gpu_loss = rt.read_f32(&loss_buf, rows as usize);

        let mut max_err: f32 = 0.0;
        for r in 0..rows as usize {
            let err = (gpu_loss[r] - expected_loss[r]).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-2,
                "CE loss[{}]: gpu={:.6} cpu={:.6} err={:.6}",
                r, gpu_loss[r], expected_loss[r], err);
        }

        let _ = rt.wait_idle();
        eprintln!("✓ CE loss fwd GPU: {}×{}, max_err={:.2e}", rows, cols, max_err);
    }

    /// Standalone log_softmax GPU test — isolates whether log_softmax alone hangs
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_log_softmax_gpu_only() {
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

        let rows: u32 = 2;
        let cols: u32 = 32;
        let n = (rows * cols) as usize;

        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 5.0).collect();
        let input_buf = rt.upload_f32(&input).unwrap();
        let out_buf = rt.alloc_f32(n).unwrap();

        let kb = build_log_softmax();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        eprintln!("log_softmax compiled: elf={} bytes, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);

        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            input_buf.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            cols => u32
        ];
        let grid_x = rows * ce_loss_wg_size();
        eprintln!("Dispatching log_softmax: grid=[{}, 1, 1]", grid_x);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch log_softmax");

        let gpu_out = rt.read_f32(&out_buf, n);

        // CPU reference: log_softmax
        let mut expected = vec![0.0f32; n];
        for r in 0..rows as usize {
            let row = &input[r * cols as usize..(r + 1) * cols as usize];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
            for c in 0..cols as usize {
                expected[r * cols as usize + c] = (row[c] - max_val) - log_sum;
            }
        }

        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (gpu_out[i] - expected[i]).abs();
            max_err = max_err.max(err);
            if err > 0.01 && i < 5 {
                eprintln!("  log_sm[{}]: gpu={:.6} cpu={:.6}", i, gpu_out[i], expected[i]);
            }
        }
        eprintln!("✓ log_softmax GPU: {}×{}, max_err={:.2e}", rows, cols, max_err);
        assert!(max_err < 0.01, "log_softmax max_err={}", max_err);
        let _ = rt.wait_idle();
    }

    /// Standalone nll_loss GPU test
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_nll_loss_gpu_only() {
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

        let rows: u32 = 2;
        let cols: u32 = 8;
        let n = (rows * cols) as usize;

        // Simple log_softmax values (CPU computed)
        let mut log_sm = vec![0.0f32; n];
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5) - 2.0).collect();
        for r in 0..rows as usize {
            let row = &input[r * cols as usize..(r + 1) * cols as usize];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
            for c in 0..cols as usize {
                log_sm[r * cols as usize + c] = (row[c] - max_val) - log_sum;
            }
        }

        let targets: Vec<u32> = vec![2, 5]; // target indices

        // Expected: loss[r] = -log_sm[r, target[r]]
        let expected_loss: Vec<f32> = (0..rows as usize)
            .map(|r| -log_sm[r * cols as usize + targets[r] as usize])
            .collect();

        let log_sm_buf = rt.upload_f32(&log_sm).unwrap();
        let targets_f32: Vec<f32> = targets.iter().map(|&t| f32::from_bits(t)).collect();
        let targets_buf = rt.upload_f32(&targets_f32).unwrap();
        let loss_buf = rt.alloc_f32(rows as usize).unwrap();

        let kb = build_nll_loss();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile nll");
        eprintln!("nll_loss: elf={} wg={:?} lds={}", ck.elf.len(), ck.workgroup_size, ck.lds_size);
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load nll");

        let ka = crate::kernargs![
            log_sm_buf.gpu_addr() => u64,
            targets_buf.gpu_addr() => u64,
            loss_buf.gpu_addr() => u64,
            cols => u32
        ];
        let grid_x = rows * ce_loss_wg_size();
        eprintln!("Dispatching nll_loss: grid=[{}, 1, 1], cols={}", grid_x, cols);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch nll_loss");

        let gpu_loss = rt.read_f32(&loss_buf, rows as usize);
        for r in 0..rows as usize {
            let err = (gpu_loss[r] - expected_loss[r]).abs();
            eprintln!("  loss[{}]: gpu={:.6} cpu={:.6} err={:.6}", r, gpu_loss[r], expected_loss[r], err);
            assert!(err < 0.01, "nll_loss[{}] error too large", r);
        }
        eprintln!("✓ nll_loss GPU OK");
        let _ = rt.wait_idle();
    }
}
