//! Focused thin-matrix (small M) benchmark
//! Tests many tile/WGP/split-K/grid combinations on M=128,256 × K=1024 × N=4096

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid, compute_grid_split_k};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Thin Matrix Optimization Sweep                            ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Build all combinations systematically
    let mut configs: Vec<(&str, GemmConfig)> = Vec::new();

    // Base tiles
    let tiles: Vec<(&str, fn() -> GemmConfig)> = vec![
        ("16x64_k16", GemmConfig::tile_16x64_k16),
        ("32x64_k16", GemmConfig::tile_32x64_k16),
        ("32x64_k32", GemmConfig::tile_32x64_k32),
        ("64x64_k16", GemmConfig::tile_64x64_k16),
        ("128x64_k16", GemmConfig::tile_128x64_k16),
        ("128x64_k32", GemmConfig::tile_128x64_k32),
    ];
    let split_ks = [1u32, 2, 4, 8];

    for (tname, tfn) in &tiles {
        for &sk in &split_ks {
            let base = tfn();
            // CU mode, swap_grid=true (N-on-X)
            let cfg = GemmConfig { split_k: if sk > 1 { Some(sk) } else { None }, ..base.clone() };
            configs.push((Box::leak(format!("{}_sk{}", tname, sk).into_boxed_str()), cfg));

            // CU mode, swap_grid=false (M-on-X) — often better for thin
            let cfg_mg = GemmConfig { swap_grid: false, split_k: if sk > 1 { Some(sk) } else { None }, ..base.clone() };
            configs.push((Box::leak(format!("{}_sk{}_mg", tname, sk).into_boxed_str()), cfg_mg));

            // WGP mode, swap_grid=true
            let cfg_wgp = GemmConfig { wgp_mode: true, split_k: if sk > 1 { Some(sk) } else { None }, ..base.clone() };
            configs.push((Box::leak(format!("{}_sk{}_wgp", tname, sk).into_boxed_str()), cfg_wgp));

            // WGP mode, swap_grid=false
            let cfg_wgp_mg = GemmConfig { swap_grid: false, wgp_mode: true, split_k: if sk > 1 { Some(sk) } else { None }, ..base.clone() };
            configs.push((Box::leak(format!("{}_sk{}_mg_wgp", tname, sk).into_boxed_str()), cfg_wgp_mg));
        }
    }

    let sizes: Vec<(u32, u32, u32)> = vec![
        (128, 1024, 4096),
        (256, 1024, 4096),
        (512, 1024, 4096),
        (128, 512, 2048),
        (256, 512, 2048),
    ];

    // Compile all
    eprintln!("\n── Compiling {} kernel variants ──", configs.len());
    let mut compiled = Vec::new();
    for (i, (name, cfg)) in configs.iter().enumerate() {
        let kernel_ir = generate(cfg);
        match kernel_ir.compile(Target::GFX1100) {
            Ok(elf) => compiled.push(Some(elf)),
            Err(e) => {
                compiled.push(None);
                // silently skip compile failures (VGPR overflow etc)
            }
        }
        if (i+1) % 20 == 0 { eprint!("."); }
    }
    eprintln!(" done ({} compiled)", compiled.iter().filter(|x| x.is_some()).count());

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use std::time::Instant;

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        // Load kernels
        let mut gpu_kernels: Vec<Option<GpuKernel>> = Vec::new();
        for (ci, (_, cfg)) in configs.iter().enumerate() {
            if let Some(elf) = &compiled[ci] {
                match GpuKernel::load(&device, elf, &KernelLoadConfig {
                    workgroup_size: [cfg.wg_size, 1, 1],
                    lds_size: cfg.lds_total(),
                }) {
                    Ok(gk) => gpu_kernels.push(Some(gk)),
                    Err(_) => gpu_kernels.push(None),
                }
            } else {
                gpu_kernels.push(None);
            }
        }

        eprintln!("\n══ Thin Matrix Sweep (top 5 per size) ══\n");

        for &(m, k, n) in &sizes {
            let x_bytes = (m * k * 2) as usize;
            let w_bytes = (n * k * 2) as usize;
            let y_bytes = (m * n * 4) as usize;

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            x_buf.write(&vec![0u8; x_bytes]);
            w_buf.write(&vec![0u8; w_bytes]);

            let mut results: Vec<(&str, f64, u32)> = Vec::new();

            for (ci, (name, cfg)) in configs.iter().enumerate() {
                if gpu_kernels[ci].is_none() { continue; }
                let gk = gpu_kernels[ci].as_ref().unwrap();

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

                let y_alloc = if sk > 1 {
                    device.alloc_vram(y_bytes * sk as usize)?
                } else {
                    device.alloc_vram(y_bytes)?
                };
                let y_stride = if sk > 1 { m * n * 4 } else { 0u32 };
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
                queue.submit(gk, [gx, gy, 1], ka_buf);
                queue.wait_idle()?;

                // Benchmark
                let iters = 10;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let ka_buf = pool.write_kernargs(0, &ka);
                    queue.submit(gk, [gx, gy, 1], ka_buf);
                    queue.wait_idle()?;
                }
                let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
                let flops = 2.0 * m as f64 * k as f64 * n as f64;
                let tflops = flops / (elapsed_us * 1e6);

                results.push((name, tflops, total_wgs));
                drop(y_alloc);
            }

            // Sort by TFLOPS descending, show top 5
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("── {}×{} ×{} ──", m, k, n);
            for (i, (name, tf, wgs)) in results.iter().take(5).enumerate() {
                let marker = if i == 0 { " 👑" } else { "" };
                eprintln!("  #{}: {:>30} | {:>7.2} TF | {:>5} WGs{}", i+1, name, tf, wgs, marker);
            }
            eprintln!();

            drop(x_buf);
            drop(w_buf);
        }
    }
    Ok(())
}
