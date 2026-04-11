//! Causal Mask GPU kernel — applies upper-triangular mask to attention scores.
//!
//! # Algorithm
//!
//! For a (seq_len × seq_len) matrix of attention scores:
//! ```text
//! out[i,j] = scores[i,j]    if j ≤ i  (causal: can attend to past + current)
//!          = -inf             if j > i  (masked: cannot attend to future)
//! ```
//!
//! # Design
//!
//! 2D workgroup: each thread processes one element (row=pid, col=tid).
//! WG_SIZE = 256, seq_len ≤ 256.
//! Grid: (seq_len * WG_SIZE, 1, 1) — one WG per row.

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build causal mask kernel.
///
/// Kernarg layout: [scores:u64, output:u64, seq_len:u32]
/// Grid: (seq_len * WG_SIZE, 1, 1)
///
/// output[row, col] = scores[row, col] if col <= row, else -inf
pub fn build_causal_mask() -> BlockKernel {
    let mut kb = BlockKernel::new("causal_mask", WG_SIZE);

    let scores_ptr = kb.arg_ptr("scores");
    let output_ptr = kb.arg_ptr("output");
    let seq_len = kb.arg_u32("seq_len");

    let tid = kb.thread_id();   // column index
    let pid = kb.program_id(0); // row index

    let row_base = pid.mul(&mut kb, seq_len);
    let offset = row_base.add(&mut kb, tid);
    let in_bounds = tid.lt(&mut kb, seq_len);

    // Load score
    let score = kb.load(scores_ptr, offset, in_bounds);

    // Causal check: col <= row → tid <= pid
    // Equivalent: !(tid > pid) → pid >= tid → use GeU32
    let one_u = kb.const_u32(1);
    let pid_plus_one = pid.add(&mut kb, one_u);
    let is_causal = tid.lt(&mut kb, pid_plus_one); // tid < pid + 1 ≡ tid <= pid

    // Apply mask: out = is_causal ? score : -inf
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let masked_score = is_causal.select(&mut kb, score, neg_inf);

    // Store result
    kb.store(output_ptr, offset, masked_score, in_bounds);

    kb
}

/// Build causal mask backward kernel.
///
/// Kernarg layout: [dy:u64, dx:u64, seq_len:u32]
/// Grid: (seq_len * WG_SIZE, 1, 1)
///
/// dx[row, col] = dy[row, col] if col <= row, else 0.0
///
/// Same structure as forward, but uses 0.0 instead of -inf for masked positions
/// (gradient of -inf masking is zero — those positions don't contribute).
pub fn build_causal_mask_backward() -> BlockKernel {
    let mut kb = BlockKernel::new("causal_mask_bwd", WG_SIZE);

    let dy_ptr = kb.arg_ptr("dy");
    let dx_ptr = kb.arg_ptr("dx");
    let seq_len = kb.arg_u32("seq_len");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let row_base = pid.mul(&mut kb, seq_len);
    let offset = row_base.add(&mut kb, tid);
    let in_bounds = tid.lt(&mut kb, seq_len);

    let dy = kb.load(dy_ptr, offset, in_bounds);

    // Causal check: col <= row → tid < pid + 1
    let one_u = kb.const_u32(1);
    let pid_plus_one = pid.add(&mut kb, one_u);
    let is_causal = tid.lt(&mut kb, pid_plus_one);

    // dx = is_causal ? dy : 0.0
    let zero_f = kb.const_f32(0.0);
    let dx = is_causal.select(&mut kb, dy, zero_f);
    kb.store(dx_ptr, offset, dx, in_bounds);

    kb
}

/// CPU reference: causal mask forward
pub fn cpu_causal_mask(scores: &[f32], output: &mut [f32], seq_len: usize) {
    for row in 0..seq_len {
        for col in 0..seq_len {
            let idx = row * seq_len + col;
            output[idx] = if col <= row { scores[idx] } else { f32::NEG_INFINITY };
        }
    }
}

/// CPU reference: causal mask backward
pub fn cpu_causal_mask_backward(dy: &[f32], dx: &mut [f32], seq_len: usize) {
    for row in 0..seq_len {
        for col in 0..seq_len {
            let idx = row * seq_len + col;
            dx[idx] = if col <= row { dy[idx] } else { 0.0 };
        }
    }
}

pub fn causal_mask_grid(seq_len: u32) -> (u32, u32) { (seq_len * WG_SIZE, 1) }
pub fn causal_mask_wg_size() -> u32 { WG_SIZE }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_causal_mask() {
        let n = 4;
        let scores: Vec<f32> = (0..n*n).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; n * n];
        cpu_causal_mask(&scores, &mut output, n);

        // Row 0: [0, -inf, -inf, -inf]
        assert_eq!(output[0], 0.0);
        assert!(output[1].is_infinite() && output[1].is_sign_negative());
        // Row 1: [4, 5, -inf, -inf]
        assert_eq!(output[4], 4.0);
        assert_eq!(output[5], 5.0);
        assert!(output[6].is_infinite());
        // Row 3: [12, 13, 14, 15] — all unmasked
        assert_eq!(output[15], 15.0);
    }

    #[test]
    fn test_causal_mask_compiles() {
        let kb = build_causal_mask();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("causal mask compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ causal_mask: {} bytes ELF, wg={:?}", ck.elf.len(), ck.workgroup_size);
    }

    #[test]
    fn test_cpu_causal_mask_backward() {
        let n = 4;
        let dy: Vec<f32> = (0..n*n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut dx = vec![0.0f32; n * n];
        cpu_causal_mask_backward(&dy, &mut dx, n);

        // Row 0: [dy[0], 0, 0, 0]
        assert_eq!(dx[0], dy[0]);
        assert_eq!(dx[1], 0.0);
        // Row 1: [dy[4], dy[5], 0, 0]
        assert_eq!(dx[4], dy[4]);
        assert_eq!(dx[5], dy[5]);
        assert_eq!(dx[6], 0.0);
        // Row 3: all pass through
        assert_eq!(dx[15], dy[15]);
    }

    #[test]
    fn test_causal_mask_bwd_compiles() {
        let kb = build_causal_mask_backward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("causal mask bwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ causal_mask_bwd: {} bytes ELF, wg={:?}", ck.elf.len(), ck.workgroup_size);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_causal_mask_gpu() {
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

        let seq_len: u32 = 32;
        let n = (seq_len * seq_len) as usize;

        let scores: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.3).sin() * 3.0)).collect();
        let mut expected = vec![0.0f32; n];
        cpu_causal_mask(&scores, &mut expected, seq_len as usize);

        let scores_buf = rt.upload_f32(&scores).unwrap();
        let out_buf = rt.alloc_f32(n).unwrap();

        let kb = build_causal_mask();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            scores_buf.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            seq_len => u32
        ];
        let (grid_x, _) = causal_mask_grid(seq_len);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let gpu_out = rt.read_f32(&out_buf, n);

        let mut errors = 0;
        for i in 0..n {
            if expected[i].is_infinite() {
                if !gpu_out[i].is_infinite() || !gpu_out[i].is_sign_negative() {
                    errors += 1;
                    if errors <= 3 {
                        eprintln!("mask[{}]: gpu={} expected=-inf", i, gpu_out[i]);
                    }
                }
            } else {
                let err = (gpu_out[i] - expected[i]).abs();
                if err > 1e-6 {
                    errors += 1;
                    if errors <= 3 {
                        eprintln!("mask[{}]: gpu={:.6} cpu={:.6} err={:.6}", i, gpu_out[i], expected[i], err);
                    }
                }
            }
        }
        assert!(errors == 0, "Causal mask: {} errors", errors);
        let _ = rt.wait_idle();
        eprintln!("✓ causal_mask GPU: {}×{}, 0 errors", seq_len, seq_len);
    }
}
