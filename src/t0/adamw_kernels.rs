//! AdamW optimizer GPU kernel — fused parameter update.
//!
//! # Algorithm (AdamW)
//!
//! ```text
//! m = β₁ * m + (1 - β₁) * g          // first moment
//! v = β₂ * v + (1 - β₂) * g²         // second moment
//! m_hat = m / (1 - β₁ᵗ)              // bias-corrected first moment
//! v_hat = v / (1 - β₂ᵗ)              // bias-corrected second moment
//! p = p * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)
//! ```
//!
//! # Design
//!
//! Single fused kernel: each thread handles one parameter.
//! Grid: ceil(n / WG_SIZE) workgroups, WG_SIZE = 256.
//! All scalars (lr, β₁, β₂, eps, wd, bc1, bc2) passed as kernargs.

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build fused AdamW optimizer kernel.
///
/// Kernarg layout:
///   [params:u64, grads:u64, m:u64, v:u64, n:u32,
///    lr:f32, beta1:f32, beta2:f32, eps:f32, weight_decay:f32,
///    bc1:f32, bc2:f32]
///
/// bc1 = 1 / (1 - β₁ᵗ), bc2 = 1 / (1 - β₂ᵗ) — precomputed on CPU
///
/// Grid: (ceil(n / WG_SIZE) * WG_SIZE, 1, 1)
pub fn build_adamw_step() -> BlockKernel {
    let mut kb = BlockKernel::new("adamw_step", WG_SIZE);

    let params_ptr = kb.arg_ptr("params");
    let grads_ptr = kb.arg_ptr("grads");
    let m_ptr = kb.arg_ptr("m");       // first moment
    let v_ptr = kb.arg_ptr("v");       // second moment
    let n = kb.arg_u32("n");           // total parameters
    let lr = kb.arg_f32("lr");
    let beta1 = kb.arg_f32("beta1");
    let beta2 = kb.arg_f32("beta2");
    let eps = kb.arg_f32("eps");
    let wd = kb.arg_f32("weight_decay");
    let bc1 = kb.arg_f32("bc1");       // 1 / (1 - β₁ᵗ)
    let bc2 = kb.arg_f32("bc2");       // 1 / (1 - β₂ᵗ)

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    // Global index = pid * WG_SIZE + tid
    let wg_size_val = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size_val);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n);

    // Load
    let p = kb.load(params_ptr, gid, mask);
    let g = kb.load(grads_ptr, gid, mask);
    let m_val = kb.load(m_ptr, gid, mask);
    let v_val = kb.load(v_ptr, gid, mask);

    // m = β₁ * m + (1 - β₁) * g
    let one_f = kb.const_f32(1.0);
    let one_minus_b1 = one_f.sub(&mut kb, beta1);
    let m_decay = beta1.mul(&mut kb, m_val);
    let m_fresh = one_minus_b1.mul(&mut kb, g);
    let m_new = m_decay.add(&mut kb, m_fresh);

    // v = β₂ * v + (1 - β₂) * g²
    let one_minus_b2 = one_f.sub(&mut kb, beta2);
    let g_sq = g.mul(&mut kb, g);
    let v_decay = beta2.mul(&mut kb, v_val);
    let v_fresh = one_minus_b2.mul(&mut kb, g_sq);
    let v_new = v_decay.add(&mut kb, v_fresh);

    // Save updated moments
    kb.store(m_ptr, gid, m_new, mask);
    kb.store(v_ptr, gid, v_new, mask);

    // Bias-corrected: m_hat = m * bc1, v_hat = v * bc2
    let m_hat = m_new.mul(&mut kb, bc1);
    let v_hat = v_new.mul(&mut kb, bc2);

    // update = m_hat / (sqrt(v_hat) + eps)
    let v_hat_sqrt = v_hat.sqrt(&mut kb);
    let v_hat_eps = v_hat_sqrt.add(&mut kb, eps);
    let update = m_hat.div(&mut kb, v_hat_eps);

    // Weight decay: p = p * (1 - lr * wd)
    let lr_wd = lr.mul(&mut kb, wd);
    let decay_factor = one_f.sub(&mut kb, lr_wd);
    let p_decayed = p.mul(&mut kb, decay_factor);

    // p = p_decayed - lr * update
    let lr_update = lr.mul(&mut kb, update);
    let p_new = p_decayed.sub(&mut kb, lr_update);
    kb.store(params_ptr, gid, p_new, mask);

    kb
}

/// CPU reference: AdamW step
pub fn cpu_adamw_step(
    params: &mut [f32], grads: &[f32], m: &mut [f32], v: &mut [f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    t: u32, // step number (1-indexed)
) {
    let bc1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bc2 = 1.0 / (1.0 - beta2.powi(t as i32));
    for i in 0..params.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        let m_hat = m[i] * bc1;
        let v_hat = v[i] * bc2;
        params[i] = params[i] * (1.0 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + eps);
    }
}

pub fn adamw_grid(n: u32) -> (u32, u32) {
    let num_wgs = (n + WG_SIZE - 1) / WG_SIZE;
    (num_wgs * WG_SIZE, 1)
}
pub fn adamw_wg_size() -> u32 { WG_SIZE }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_adamw() {
        let mut params = vec![1.0, 2.0, 3.0, 4.0];
        let grads = vec![0.1, 0.2, 0.3, 0.4];
        let mut m = vec![0.0; 4];
        let mut v = vec![0.0; 4];

        let original = params.clone();
        cpu_adamw_step(&mut params, &grads, &mut m, &mut v,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1);

        // Check moments are updated: m = (1-β₁)*g, v = (1-β₂)*g²
        for i in 0..4 {
            let expected_m = (1.0 - 0.9) * grads[i];
            assert!((m[i] - expected_m).abs() < 1e-6,
                "m[{}]: got={} expected={}", i, m[i], expected_m);
            let expected_v = (1.0 - 0.999) * grads[i] * grads[i];
            assert!((v[i] - expected_v).abs() < 1e-8,
                "v[{}]: got={} expected={}", i, v[i], expected_v);
        }
        // Check params changed
        for i in 0..4 {
            assert!((params[i] - original[i]).abs() > 1e-6,
                "param[{}] should have changed", i);
        }
    }

    #[test]
    fn test_adamw_compiles() {
        let kb = build_adamw_step();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("adamw compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ AdamW: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_adamw_gpu() {
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

        let n: u32 = 256;
        let lr: f32 = 1e-3;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let wd: f32 = 0.01;
        let t: u32 = 1; // step 1

        let params_init: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 12.8).collect();
        let grads: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin() * 0.5)).collect();
        let m_init = vec![0.0f32; n as usize];
        let v_init = vec![0.0f32; n as usize];

        // CPU reference
        let mut cpu_params = params_init.clone();
        let mut cpu_m = m_init.clone();
        let mut cpu_v = v_init.clone();
        cpu_adamw_step(&mut cpu_params, &grads, &mut cpu_m, &mut cpu_v, lr, beta1, beta2, eps, wd, t);

        // GPU
        let params_buf = rt.upload_f32(&params_init).unwrap();
        let grads_buf = rt.upload_f32(&grads).unwrap();
        let m_buf = rt.upload_f32(&m_init).unwrap();
        let v_buf = rt.upload_f32(&v_init).unwrap();

        let bc1 = 1.0f32 / (1.0 - beta1.powi(t as i32));
        let bc2 = 1.0f32 / (1.0 - beta2.powi(t as i32));

        let kb = build_adamw_step();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            params_buf.gpu_addr() => u64,
            grads_buf.gpu_addr() => u64,
            m_buf.gpu_addr() => u64,
            v_buf.gpu_addr() => u64,
            n => u32,
            lr => f32,
            beta1 => f32,
            beta2 => f32,
            eps => f32,
            wd => f32,
            bc1 => f32,
            bc2 => f32
        ];
        let (grid_x, _) = adamw_grid(n);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let gpu_params = rt.read_f32(&params_buf, n as usize);
        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            let err = (gpu_params[i] - cpu_params[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-4, "AdamW max_err={}", max_err);
        let _ = rt.wait_idle();
        eprintln!("✓ AdamW GPU: n={}, max_err={:.2e}", n, max_err);
    }
}
