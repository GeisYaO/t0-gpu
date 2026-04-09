//! Measure dispatch overhead and per-CU efficiency for small matrices
use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid, compute_grid_split_k};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Small Matrix Analysis: Dispatch Overhead + CU Efficiency   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // 1. Measure raw dispatch overhead with a trivial kernel (s_endpgm)
    // 2. For small matrices: measure total time, subtract dispatch, compute actual TFLOPS

    let configs = vec![
        ("16x64_sk2", GemmConfig { split_k: Some(2), ..GemmConfig::tile_16x64_k16() }),
        ("32x64_sk2", GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k16() }),
        ("32x64_sk4", GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x64_k16() }),
        ("32x64_sk8", GemmConfig { split_k: Some(8), ..GemmConfig::tile_32x64_k16() }),
        ("64x64_sk2", GemmConfig { split_k: Some(2), ..GemmConfig::tile_64x64_k16() }),
        ("64x64_sk4", GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() }),
        // WGP variants
        ("32x64_sk4_wgp", GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_32x64_k16() }),
        ("32x64_sk8_wgp", GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_32x64_k16() }),
        ("64x64_sk4_wgp", GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_64x64_k16() }),
    ];

    let sizes: Vec<(u32, u32, u32)> = vec![
        (64, 64, 64),
        (64, 128, 64),
        (128, 128, 128),
        (128, 256, 128),
        (256, 256, 256),
        (256, 512, 256),
        (384, 384, 384),
    ];

    // Compile all
    eprintln!("\n── Compiling {} kernel variants ──", configs.len());
    let mut compiled = Vec::new();
    for (name, cfg) in &configs {
        let kernel_ir = generate(cfg);
        let elf = kernel_ir.compile(Target::GFX1100)?;
        compiled.push(elf);
        eprint!(".");
    }
    eprintln!(" done");

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use std::time::Instant;

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        // 1. Measure raw dispatch overhead
        // Use the smallest kernel (16x64_sk2) with M=N=16, K=16 → 1 WG, minimal compute
        eprintln!("\n── Dispatch Overhead Measurement ──");
        {
            let gk = GpuKernel::load(&device, &compiled[0], &KernelLoadConfig {
                workgroup_size: [configs[0].1.wg_size, 1, 1],
                lds_size: configs[0].1.lds_total(),
            })?;
            // Minimal data
            let x_buf = device.alloc_vram(16*16*2)?;
            let w_buf = device.alloc_vram(16*16*2)?;
            let y_buf = device.alloc_vram(16*64*4)?;
            x_buf.write(&vec![0u8; 16*16*2]);
            w_buf.write(&vec![0u8; 16*16*2]);
            y_buf.write(&vec![0u8; 16*64*4]);

            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&16u32.to_le_bytes());
            ka[28..32].copy_from_slice(&64u32.to_le_bytes());

            // Warmup
            for _ in 0..5 {
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gk, [32, 1, 1], ka_buf);
                queue.wait_idle()?;
            }
            // Measure
            let iters = 50;
            let t0 = Instant::now();
            for _ in 0..iters {
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gk, [32, 1, 1], ka_buf);
                queue.wait_idle()?;
            }
            let overhead_us = t0.elapsed().as_micros() as f64 / iters as f64;
            eprintln!("  Dispatch overhead (1 WG, minimal compute): {:.1} μs", overhead_us);
        }

        // 2. Small matrix detailed analysis
        let mut gpu_kernels = Vec::new();
        for (ci, (_, cfg)) in configs.iter().enumerate() {
            let gk = GpuKernel::load(&device, &compiled[ci], &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?;
            gpu_kernels.push(gk);
        }

        eprintln!("\n══ Small Matrix Detailed Analysis ══\n");
        eprintln!("{:>16} | {:>22} | {:>5} | {:>7} | {:>7} | {:>7}",
            "Matrix", "Kernel", "WGs", "μs", "TFLOPS", "eff %");
        eprintln!("{}", "-".repeat(85));

        for &(m, k, n) in &sizes {
            let x_bytes = (m * k * 2) as usize;
            let w_bytes = (n * k * 2) as usize;
            let y_bytes = (m * n * 4) as usize;

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            x_buf.write(&vec![0u8; x_bytes]);
            w_buf.write(&vec![0u8; w_bytes]);

            let mut best_tf = 0.0f64;
            let mut best_name = "";
            let mut best_us = 0.0;
            let mut best_wgs = 0u32;

            for (ci, (name, cfg)) in configs.iter().enumerate() {
                let sk = cfg.split_k.unwrap_or(1);
                let effective_n = cfg.tile_n * cfg.n_col_passes;
                if m % cfg.tile_m != 0 || n % effective_n != 0 || k % (cfg.tile_k * sk) != 0 {
                    continue;
                }

                let (gx, gy) = if sk > 1 {
                    compute_grid_split_k(cfg, m, n, sk)
                } else {
                    compute_grid(cfg, m, n)
                };
                let total_wgs = (gx / cfg.wg_size) * gy;

                let y_stride = if sk > 1 { m * n * 4 } else { 0u32 };
                let y_alloc = if sk > 1 {
                    device.alloc_vram(y_bytes * sk as usize)?
                } else {
                    device.alloc_vram(y_bytes)?
                };
                y_alloc.write(&vec![0u8; if sk > 1 { y_bytes * sk as usize } else { y_bytes }]);

                let mut ka = [0u8; 40];
                ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
                ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
                ka[16..24].copy_from_slice(&y_alloc.gpu_addr().to_le_bytes());
                ka[24..28].copy_from_slice(&k.to_le_bytes());
                ka[28..32].copy_from_slice(&n.to_le_bytes());
                ka[32..36].copy_from_slice(&0u32.to_le_bytes());
                ka[36..40].copy_from_slice(&y_stride.to_le_bytes());

                // Warmup
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernels[ci], [gx, gy, 1], ka_buf);
                queue.wait_idle()?;

                // Benchmark
                let iters = 20;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let ka_buf = pool.write_kernargs(0, &ka);
                    queue.submit(&gpu_kernels[ci], [gx, gy, 1], ka_buf);
                    queue.wait_idle()?;
                }
                let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
                let flops = 2.0 * m as f64 * k as f64 * n as f64;
                let tflops = flops / (elapsed_us * 1e6);

                if tflops > best_tf {
                    best_tf = tflops;
                    best_name = name;
                    best_us = elapsed_us;
                    best_wgs = total_wgs;
                }

                drop(y_alloc);
            }

            // Efficiency: best_tf / (theoretical peak at this CU count)
            let active_cus = best_wgs.min(96) as f64;
            let peak_per_cu = 123.0 / 96.0; // TF per CU
            let peak_at_cus = active_cus * peak_per_cu;
            let eff = if peak_at_cus > 0.0 { best_tf / peak_at_cus * 100.0 } else { 0.0 };

            eprintln!("{:>5}×{:<4} ×{:<4} | {:>22} | {:>5} | {:>6.1} | {:>6.2} | {:>5.1}%",
                m, k, n, best_name, best_wgs, best_us, best_tf, eff);

            drop(x_buf);
            drop(w_buf);
        }
    }
    Ok(())
}
