//! # bench_gemm — T0 bf16 WMMA GEMM Performance Benchmark
//!
//! Benchmarks T0-compiled GEMM kernels across multiple matrix sizes.
//! Measures TFLOPS and compares against theoretical peak (123 TFLOPS bf16 WMMA).
//!
//! Run: cargo run --example bench_gemm --features rocm --release
//!
//! ## Matrix layout
//! Y[M,N] = X[M,K] × W^T[N,K]   (X row-major bf16, W^T row-major bf16, Y f32)
//!
//! ## WMMA tile
//! 32×64 output tile, K_tile=16, workgroup = 64 threads (2 waves)

use t0_gpu::t0::Target;
use t0_gpu::t0::math;

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM Benchmark — AMD RX 7900 XTX (GFX1100, 96 CU)     ║");
    eprintln!("║  bf16 input → WMMA f32 accumulate → f32 output             ║");
    eprintln!("║  Theoretical peak: 123 TFLOPS (bf16 WMMA)                  ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Compile GEMM kernel
    use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid};
    let cfg = GemmConfig::tile_64x64_k16();
    eprintln!("\n[Compile] gemm_gen: {} ({}×{}, K={}, LDS={}B)...",
        cfg.name(), cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.lds_total());
    let kernel_ir = generate(&cfg);
    let elf = kernel_ir.compile(Target::GFX1100)?;
    eprintln!("  ✓ {} bytes ELF", elf.len());

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [cfg.wg_size, 1, 1],
            lds_size: cfg.lds_total(),
        })?;

        // Matrix sizes to sweep: (M, K, N)
        let sizes: Vec<(u32, u32, u32)> = vec![
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (128, 1024, 4096),
            (128, 4096, 1024),
            (256, 1024, 4096),
            (512, 1024, 4096),
            (1024, 1024, 4096),
        ];

        eprintln!("\n{:>6} {:>6} {:>6} | {:>8} {:>8} {:>8} | {:>7}",
            "M", "K", "N", "Time", "TFLOPS", "Util%", "VRAM");
        eprintln!("{}", "-".repeat(72));

        for &(m, k, n) in &sizes {
            // Allocate buffers (bf16 for X and W, f32 for Y)
            let x_bytes = (m as usize) * (k as usize) * 2; // bf16
            let w_bytes = (n as usize) * (k as usize) * 2; // bf16 (W^T[N,K])
            let y_bytes = (m as usize) * (n as usize) * 4; // f32 output
            let total_mb = (x_bytes + w_bytes + y_bytes) as f64 / (1024.0 * 1024.0);

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            let y_buf = device.alloc_vram(y_bytes)?;

            // Fill with bf16 data (0x3F80 = 1.0 in bf16)
            let x_data = vec![0x3F80u16; (m as usize) * (k as usize)];
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
            });
            let w_data = vec![0x3F80u16; (n as usize) * (k as usize)];
            w_buf.write(unsafe {
                std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
            });

            // Grid from gemm_gen::compute_grid
            let (grid_x, grid_y) = compute_grid(&cfg, m, n);

            // Kernargs: [X_ptr:u64, WT_ptr:u64, Y_ptr:u64, K:u32, N:u32]
            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());
            ka[36..40].copy_from_slice(&0u32.to_le_bytes());

            // Warmup
            for _ in 0..3 {
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                queue.wait_idle()?;
            }

            // Timed runs
            let n_iters = if m * k * n <= 1024 * 1024 * 1024 { 20 } else { 5 };
            let start = std::time::Instant::now();
            for i in 0..n_iters {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit(&gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                queue.wait_idle()?;
            }
            let elapsed = start.elapsed();
            let avg_us = elapsed.as_micros() as f64 / n_iters as f64;
            let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
            let tflops = flops / (avg_us * 1e6);
            let util = tflops / 123.0 * 100.0;

            eprintln!("{:>6} {:>6} {:>6} | {:>6.1}μs {:>6.2} TF {:>6.1}% | {:>5.1}MB",
                m, k, n, avg_us, tflops, util, total_mb);

            // Drop buffers
            drop(x_buf);
            drop(w_buf);
            drop(y_buf);
        }

        // Correctness check on small matrix
        eprintln!("\n── Correctness Check (64×64×64) ──");
        {
            let m = 64u32;
            let k = 64u32;
            let n = 64u32;

            let x_buf = device.alloc_vram((m * k * 2) as usize)?;
            let w_buf = device.alloc_vram((n * k * 2) as usize)?;
            let y_buf = device.alloc_vram((m * n * 4) as usize)?;  // f32 output

            // X = all 1.0: x[i,j] = 1.0 (bf16 = 0x3F80)
            let x_data = vec![0x3F80u16; (m * k) as usize];
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, (m * k * 2) as usize)
            });
            let w_data = vec![0x3F80u16; (n * k) as usize];
            w_buf.write(unsafe {
                std::slice::from_raw_parts(w_data.as_ptr() as *const u8, (n * k * 2) as usize)
            });

            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());
            ka[36..40].copy_from_slice(&k.to_le_bytes());

            let ka_buf = pool.write_kernargs(0, &ka);
            // Grid for 64×64: [N/64*64, M/64, 1]
            queue.submit(&gpu_kernel, [64, 1, 1], ka_buf);
            queue.wait_idle()?;

            // Read Y: all ones × all ones × K = K for each element
            let expected = k as f32;
            let mut y_out = vec![0f32; (m * n) as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    y_buf.host_ptr as *const f32,
                    y_out.as_mut_ptr(),
                    (m * n) as usize,
                );
            }

            let max_err = y_out.iter()
                .map(|&v| (v - expected).abs())
                .fold(0.0f32, f32::max);

            eprintln!("  Y[0] = {} (expected {})", y_out[0], expected);
            eprintln!("  Max error: {:.6e}", max_err);
            if max_err < 0.1 {
                eprintln!("  ✅ PASS");
            } else {
                eprintln!("  ❌ FAIL");
            }
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        eprintln!("\nKFD runtime not available. Compile with --features rocm");
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
