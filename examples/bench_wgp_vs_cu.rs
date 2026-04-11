//! # WGP Mode vs CU Mode Comparison
//!
//! Experiment: compare GEMM performance with and without WGP mode
//! (Workgroup Processor mode: WG spans 2 CUs = 128KB LDS + 4 SIMDs)
//!
//! Also includes single-CU benchmark to measure per-CU efficiency.

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid, compute_grid_split_k};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  WGP Mode vs CU Mode — RDNA3 Architecture Experiment       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Test configs: same tile, CU mode vs WGP mode
    let pairs: Vec<(&str, GemmConfig)> = vec![
        // 64×64 k16 — CU mode (baseline)
        ("64x64_k16_CU", GemmConfig::tile_64x64_k16()),
        ("64x64_k16_WGP", GemmConfig { wgp_mode: true, ..GemmConfig::tile_64x64_k16() }),
        // 64×64 k16 sk8 — CU mode (best for 1024²)
        ("64x64_k16_sk8_CU", GemmConfig { split_k: Some(8), ..GemmConfig::tile_64x64_k16() }),
        ("64x64_k16_sk8_WGP", GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_64x64_k16() }),
        // 128×64 k32 — CU mode
        ("128x64_k32_CU", GemmConfig::tile_128x64_k32()),
        ("128x64_k32_WGP", GemmConfig { wgp_mode: true, ..GemmConfig::tile_128x64_k32() }),
    ];

    let sizes: Vec<(u32, u32, u32)> = vec![
        (64, 4096, 64),         // Single tile, large K → single-CU benchmark
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    // Compile all
    eprintln!("\n── Compiling {} kernel variants ──", pairs.len());
    let mut compiled = Vec::new();
    for (name, cfg) in &pairs {
        eprint!("  [{}] ... ", name);
        let kernel_ir = generate(cfg);
        let elf = kernel_ir.compile(Target::GFX1100)?;
        eprintln!("✓ {} bytes", elf.len());
        compiled.push(elf);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use std::time::Instant;

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        let mut gpu_kernels = Vec::new();
        for (ci, (_, cfg)) in pairs.iter().enumerate() {
            let gk = GpuKernel::load(&device, &compiled[ci], &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?;
            gpu_kernels.push(gk);
        }

        eprintln!("\n══ WGP Mode vs CU Mode Comparison ══\n");
        eprintln!("{:>20} | {:>26} | {:>8} | {:>8}",
            "Matrix", "Kernel", "TFLOPS", "μs");
        eprintln!("{}", "-".repeat(75));

        for &(m, k, n) in &sizes {
            let x_bytes = (m * k * 2) as usize;
            let w_bytes = (n * k * 2) as usize;
            let y_bytes = (m * n * 4) as usize;

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            x_buf.write(&vec![0u8; x_bytes]);
            w_buf.write(&vec![0u8; w_bytes]);

            for (ci, (name, cfg)) in pairs.iter().enumerate() {
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

                // Benchmark (5 iterations)
                let iters = 5;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let ka_buf = pool.write_kernargs(0, &ka);
                    queue.submit(&gpu_kernels[ci], [gx, gy, 1], ka_buf);
                    queue.wait_idle()?;
                }
                let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
                let flops = 2.0 * m as f64 * k as f64 * n as f64;
                let tflops = flops / (elapsed_us * 1e6);

                eprintln!("{:>5}×{:<5} ×{:<5} | {:>26} | {:>7.2} | {:>8.0}",
                    m, k, n, name, tflops, elapsed_us);

                drop(y_alloc);
            }
            eprintln!("{}", "-".repeat(75));

            drop(x_buf);
            drop(w_buf);
        }
    }
    Ok(())
}
