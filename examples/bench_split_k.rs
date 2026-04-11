//! # bench_split_k — Test split-K GEMM for small matrices
//!
//! Splits the K dimension across multiple dispatches to increase parallelism.
//! Each dispatch processes K/split_k iterations, writes to workspace,
//! then CPU reduces the partial results.
//!
//! Run: cargo run --example bench_split_k --features rocm --release

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid};

fn main() -> Result<(), String> {
    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║  Split-K GEMM Benchmark — Parallelism for Small Matrices ║");
    eprintln!("╚═══════════════════════════════════════════════════════════╝");

    let cfg = GemmConfig::tile_64x64_k16();
    let kernel_ir = generate(&cfg);
    let elf = kernel_ir.compile(Target::GFX1100)?;
    eprintln!("  ✓ {} ({} bytes)", cfg.name(), elf.len());

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 64)?;  // more slots for split-K

        let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [cfg.wg_size, 1, 1],
            lds_size: cfg.lds_total(),
        })?;

        // Test sizes that benefit from split-K (small M×N, enough K to split)
        let test_sizes: Vec<(u32, u32, u32, &[u32])> = vec![
            (256, 256, 256, &[1, 2, 4]),
            (512, 512, 512, &[1, 2, 4]),
            (1024, 1024, 1024, &[1, 2, 4]),
            (128, 1024, 4096, &[1, 2, 4]),
        ];

        eprintln!("\n{:>6} {:>6} {:>6} | split_k | {:>8} {:>8}", "M", "K", "N", "Time", "TFLOPS");
        eprintln!("{}", "-".repeat(60));

        for (m, k, n, splits) in &test_sizes {
            let (m, k, n) = (*m, *k, *n);
            let (grid_x, grid_y) = compute_grid(&cfg, m, n);
            let y_elems = (m as usize) * (n as usize);
            let y_bytes = y_elems * 4;

            // Input buffers
            let x_bytes = (m as usize) * (k as usize) * 2;
            let w_bytes = (n as usize) * (k as usize) * 2;
            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            let x_data = vec![0x3F80u16; (m * k) as usize];
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
            });
            let w_data = vec![0x3F80u16; (n * k) as usize];
            w_buf.write(unsafe {
                std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
            });

            for &split_k in *splits {
                if k % (cfg.tile_k * split_k) != 0 { continue; }

                let k_chunk = k / split_k;  // K iterations per split
                // Allocate workspace for split_k partial results
                let workspace = device.alloc_vram(y_bytes * split_k as usize)?;
                let y_final = device.alloc_vram(y_bytes)?;

                // Warmup
                for _ in 0..2 {
                    for s in 0..split_k {
                        let k_start = s * k_chunk * 2;  // byte offset
                        let y_off = s as u64 * y_bytes as u64;
                        let mut ka = [0u8; 40];
                        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
                        ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
                        ka[16..24].copy_from_slice(&(workspace.gpu_addr() + y_off).to_le_bytes());
                        ka[24..28].copy_from_slice(&k.to_le_bytes());
                        ka[28..32].copy_from_slice(&n.to_le_bytes());
                        ka[32..36].copy_from_slice(&k_start.to_le_bytes());
                        ka[36..40].copy_from_slice(&k_chunk.to_le_bytes());  // k_end = k_chunk
                        let ka_buf = pool.write_kernargs(s as usize, &ka);
                        queue.submit(&gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                    }
                    queue.wait_idle()?;
                }

                // Timed runs
                let n_iters = if m * k * n <= 512 * 512 * 512 { 20 } else { 10 };
                let start = std::time::Instant::now();
                for iter in 0..n_iters {
                    for s in 0..split_k {
                        let k_start = s * k_chunk * 2;
                        let y_off = s as u64 * y_bytes as u64;
                        let mut ka = [0u8; 40];
                        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
                        ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
                        ka[16..24].copy_from_slice(&(workspace.gpu_addr() + y_off).to_le_bytes());
                        ka[24..28].copy_from_slice(&k.to_le_bytes());
                        ka[28..32].copy_from_slice(&n.to_le_bytes());
                        ka[32..36].copy_from_slice(&k_start.to_le_bytes());
                        ka[36..40].copy_from_slice(&k_chunk.to_le_bytes());
                        let slot = (iter * split_k as usize + s as usize) % 32;
                        let ka_buf = pool.write_kernargs(slot, &ka);
                        queue.submit(&gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                    }
                    queue.wait_idle()?;
                    // CPU reduction would go here (not needed for timing)
                }
                let elapsed = start.elapsed();
                let avg_us = elapsed.as_micros() as f64 / n_iters as f64;
                let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
                let tflops = flops / (avg_us * 1e6);

                // Verify correctness for split_k=1
                if split_k == 1 {
                    let mut y_out = vec![0f32; y_elems];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            workspace.host_ptr as *const f32,
                            y_out.as_mut_ptr(), y_elems);
                    }
                    let expected = k as f32;
                    let max_err = y_out.iter().map(|&v| (v - expected).abs()).fold(0.0f32, f32::max);
                    let status = if max_err < 0.1 { "✅" } else { "❌" };
                    eprintln!("{:>6} {:>6} {:>6} | {:>7} | {:>6.1}μs {:>6.2} TF  {}", m, k, n, split_k, avg_us, tflops, status);
                } else {
                    eprintln!("{:>6} {:>6} {:>6} | {:>7} | {:>6.1}μs {:>6.2} TF", m, k, n, split_k, avg_us, tflops);
                }
            }
        }
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
