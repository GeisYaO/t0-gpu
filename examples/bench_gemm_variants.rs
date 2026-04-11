//! Benchmark T0 parameterized GEMM variants — tile size comparison
//!
//! Run: cargo run --example bench_gemm_variants --features rocm --release

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM Tile Variant Benchmark — RX 7900 XTX              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Compare different tile configurations from gemm_gen
    let variants: Vec<(&str, GemmConfig)> = vec![
        ("32x64_k16", GemmConfig::tile_32x64_k16()),
        ("64x64_k16", GemmConfig::tile_64x64_k16()),
        ("128x64_k16", GemmConfig::tile_128x64_k16()),
    ];

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use t0_gpu::t0::gemm_gen::compute_grid_auto;

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        let test_sizes: Vec<(u32, u32, u32)> = vec![
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ];

        for (name, cfg) in &variants {
            let kernel_ir = generate(cfg);
            let elf = kernel_ir.compile(Target::GFX1100)?;

            let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?;

            eprintln!("\n── {} ({} bytes, wg={}, lds={}) ──",
                name, elf.len(), cfg.wg_size, cfg.lds_total());
            eprintln!("{:>6} {:>6} {:>6} | {:>8} {:>8}", "M", "K", "N", "Time", "TFLOPS");

            for &(m, k, n) in &test_sizes {
                let x_buf = device.alloc_vram((m * k * 2) as usize)?;
                let w_buf = device.alloc_vram((n * k * 2) as usize)?;
                let sk = cfg.split_k.unwrap_or(1);
                let y_buf = device.alloc_vram((m * n * 4 * sk) as usize)?;

                // Fill with bf16 1.0
                let data = vec![0x3C00u16; (m * k) as usize];
                x_buf.write(unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, (m * k * 2) as usize)
                });
                let data_w = vec![0x3C00u16; (n * k) as usize];
                w_buf.write(unsafe {
                    std::slice::from_raw_parts(data_w.as_ptr() as *const u8, (n * k * 2) as usize)
                });

                let ka = t0_gpu::t0::gemm_gen::build_kernargs(
                    x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, cfg,
                );
                let (gx, gy) = compute_grid_auto(cfg, m, n);

                // Warmup
                for _ in 0..5 {
                    let ka_buf = pool.write_kernargs(0, &ka);
                    queue.submit(&gpu_kernel, [gx, gy, 1], ka_buf);
                    queue.wait_idle()?;
                }

                // Timed
                let n_iter = if m * k * n <= 1024u32.pow(3) { 20 } else { 5 };
                let start = std::time::Instant::now();
                for i in 0..n_iter {
                    let ka_buf = pool.write_kernargs(i % 16, &ka);
                    queue.submit(&gpu_kernel, [gx, gy, 1], ka_buf);
                    queue.wait_idle()?;
                }
                let avg_us = start.elapsed().as_micros() as f64 / n_iter as f64;
                let tflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / (avg_us * 1e6);

                eprintln!("{:>6} {:>6} {:>6} | {:>6.1}μs {:>6.2} TF", m, k, n, avg_us, tflops);

                drop(x_buf); drop(w_buf); drop(y_buf);
            }
        }
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
