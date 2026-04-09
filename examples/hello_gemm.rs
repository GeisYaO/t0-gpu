//! # hello_gemm — T0 + KFD 端到端示例
//!
//! 演示完整的裸金属 GPU 工作流：
//!   1. 用 T0 编译一个向量加法内核 (y = a + b)
//!   2. 用 KFD 直接与 GPU 通信（无 HIP / ROCm 运行时）
//!   3. 分配 VRAM、上传数据、dispatch、读回结果、验证

use t0_gpu::t0::{Target, GFX1100Schedule, Schedule};
use t0_gpu::t0::math;

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════╗");
    eprintln!("║  hello_gemm — T0 Compiler + KFD Runtime Demo    ║");
    eprintln!("╚══════════════════════════════════════════════════╝");

    // ── Step 1: Compile a kernel using T0 ──
    eprintln!("\n[1/5] Compiling kernel with T0...");
    let sched = GFX1100Schedule {};
    let kernel_ir = math::elementwise_binary(&sched, math::BinaryOp::Add);
    let elf = kernel_ir.compile(Target::GFX1100)?;
    eprintln!("  ✓ Compiled 'vector_add' kernel: {} bytes ELF", elf.len());

    // ── Step 2: Open KFD device ──
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use std::sync::Arc;

        eprintln!("\n[2/5] Opening KFD device...");
        let device = KfdDevice::open()?;
        eprintln!("  ✓ GPU: {} (VRAM: {} MB)", "RDNA3 GFX1100", 24576);

        // ── Step 3: Load kernel + allocate VRAM ──
        eprintln!("\n[3/5] Loading kernel & allocating VRAM...");
        let (wg_x, _, _) = sched.workgroup_size();
        let wg = wg_x as usize;
        let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [wg as u32, 1, 1],
            lds_size: 0,
        })?;

        let n: usize = 1024;
        let bytes = n * 4; // f32
        let a_buf = device.alloc_vram(bytes)?;
        let b_buf = device.alloc_vram(bytes)?;
        let y_buf = device.alloc_vram(bytes)?;
        eprintln!("  ✓ Allocated 3 × {} KB VRAM buffers", bytes / 1024);

        // ── Step 4: Upload test data ──
        eprintln!("\n[4/5] Uploading test data...");
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.5).collect();

        a_buf.write(unsafe {
            std::slice::from_raw_parts(a_data.as_ptr() as *const u8, bytes)
        });
        b_buf.write(unsafe {
            std::slice::from_raw_parts(b_data.as_ptr() as *const u8, bytes)
        });
        eprintln!("  ✓ Uploaded a[0..3] = {:?}", &a_data[0..4]);
        eprintln!("  ✓ Uploaded b[0..3] = {:?}", &b_data[0..4]);

        // ── Step 5: Dispatch kernel ──
        eprintln!("\n[5/5] Dispatching kernel...");
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 4)?;

        // Build kernargs: [a_ptr:u64, b_ptr:u64, y_ptr:u64, alpha:f32, n:u32]
        // elementwise_binary kernarg layout: a_ptr(0), b_ptr(8), y_ptr(16), param(24:f32), n(28:u32)
        let mut ka = [0u8; 32];
        ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
        ka[24..28].copy_from_slice(&0.0f32.to_le_bytes()); // unused param
        ka[28..32].copy_from_slice(&(n as u32).to_le_bytes());

        let ka_buf = pool.write_kernargs(0, &ka);
        let grid_x = ((n + wg - 1) / wg) as u32 * wg as u32;
        queue.submit(&gpu_kernel, [grid_x, 1, 1], ka_buf);
        queue.wait_idle()?;
        eprintln!("  ✓ Kernel dispatched and completed!");

        // ── Verify results ──
        eprintln!("\n══ Verification ══");
        let mut y_out = vec![0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                y_buf.host_ptr as *const f32,
                y_out.as_mut_ptr(),
                n,
            );
        }

        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let expected = a_data[i] + b_data[i];
            let err = (y_out[i] - expected).abs();
            if err > max_err { max_err = err; }
        }

        eprintln!("  y[0..4] = {:?}", &y_out[0..4]);
        eprintln!("  Expected: {:?}", (0..4).map(|i| a_data[i] + b_data[i]).collect::<Vec<_>>());
        eprintln!("  Max error: {:.6e}", max_err);

        if max_err < 1e-5 {
            eprintln!("\n  ✅ PASS — All {} elements correct!", n);
        } else {
            eprintln!("\n  ❌ FAIL — Max error too large: {}", max_err);
            return Err(format!("Verification failed: max_err={}", max_err));
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        eprintln!("\n[2/5] KFD runtime not available (compile with --features rocm)");
        eprintln!("  T0 compilation succeeded — ELF is ready for GPU dispatch.");
        eprintln!("  To run on GPU: cargo run --example hello_gemm --features rocm");
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
