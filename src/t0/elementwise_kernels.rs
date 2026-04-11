//! Elementwise utility GPU kernels — memcpy, residual_add, scale, etc.
//!
//! These are infrastructure kernels used throughout the Ignis framework
//! for gradient accumulation, tensor copying, and basic elementwise ops.

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build memcpy kernel: output[i] = input[i]
///
/// Kernarg layout: [input:u64, output:u64, n:u32]
/// Grid: (ceil(n/WG_SIZE) * WG_SIZE, 1, 1)
pub fn build_memcpy() -> BlockKernel {
    let mut kb = BlockKernel::new("memcpy", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let n = kb.arg_u32("n");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n);

    let val = kb.load(input_ptr, gid, mask);
    kb.store(output_ptr, gid, val, mask);

    kb
}

/// Build residual_add kernel: y[i] += x[i]  (in-place)
///
/// Kernarg layout: [x:u64, y:u64, n:u32]
/// Grid: (ceil(n/WG_SIZE) * WG_SIZE, 1, 1)
///
/// Semantics: y[i] = y[i] + x[i]
pub fn build_residual_add() -> BlockKernel {
    let mut kb = BlockKernel::new("residual_add", WG_SIZE);

    let x_ptr = kb.arg_ptr("x");
    let y_ptr = kb.arg_ptr("y");
    let n = kb.arg_u32("n");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n);

    let x_val = kb.load(x_ptr, gid, mask);
    let y_val = kb.load(y_ptr, gid, mask);
    let result = y_val.add(&mut kb, x_val);
    kb.store(y_ptr, gid, result, mask);

    kb
}

/// Build scale kernel: output[i] = input[i] * scale
///
/// Kernarg layout: [input:u64, output:u64, n:u32, scale:f32]
/// Grid: (ceil(n/WG_SIZE) * WG_SIZE, 1, 1)
pub fn build_scale() -> BlockKernel {
    let mut kb = BlockKernel::new("scale", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let n = kb.arg_u32("n");
    let scale = kb.arg_f32("scale");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n);

    let val = kb.load(input_ptr, gid, mask);
    let result = val.mul(&mut kb, scale);
    kb.store(output_ptr, gid, result, mask);

    kb
}

/// Compute grid for elementwise kernels.
pub fn elementwise_grid(n: u32) -> u32 {
    ((n + WG_SIZE - 1) / WG_SIZE) * WG_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memcpy_compiles() {
        let kb = build_memcpy();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("memcpy compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ memcpy: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_residual_add_compiles() {
        let kb = build_residual_add();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("residual_add compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ residual_add: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_scale_compiles() {
        let kb = build_scale();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("scale compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ scale: {} bytes ELF", ck.elf.len());
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_memcpy_gpu() {
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

        let n: u32 = 1024;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let input_buf = rt.upload_f32(&input).unwrap();
        let output_buf = rt.alloc_f32(n as usize).unwrap();

        let kb = build_memcpy();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            input_buf.gpu_addr() => u64,
            output_buf.gpu_addr() => u64,
            n => u32
        ];
        let grid_x = elementwise_grid(n);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let output = rt.read_f32(&output_buf, n as usize);
        let max_err: f32 = input.iter().zip(output.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "memcpy max_err={}", max_err);
        eprintln!("✓ memcpy GPU: n={}, max_err={:.2e}", n, max_err);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_residual_add_gpu() {
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

        let n: u32 = 512;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();

        let x_buf = rt.upload_f32(&x).unwrap();
        let y_buf = rt.upload_f32(&y).unwrap();

        let kb = build_residual_add();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            x_buf.gpu_addr() => u64,
            y_buf.gpu_addr() => u64,
            n => u32
        ];
        let grid_x = elementwise_grid(n);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let result = rt.read_f32(&y_buf, n as usize);
        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            let expected = x[i] + y[i];
            max_err = max_err.max((result[i] - expected).abs());
        }
        assert!(max_err < 1e-5, "residual_add max_err={}", max_err);
        eprintln!("✓ residual_add GPU: n={}, max_err={:.2e}", n, max_err);
    }
}
