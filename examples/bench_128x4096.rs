//! 128×1024×4096 deep-K optimization: k32/k64 tiles + WGP
use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid, compute_grid_split_k};

fn main() -> Result<(), String> {
    let m = 128u32; let k = 1024u32; let n = 4096u32;
    eprintln!("══ 128×1024×4096: Deep-K + WGP Optimization ══\n");

    let configs: Vec<(&str, GemmConfig)> = vec![
        // Baseline winners from previous run
        ("128x64_k16_sk8_wgp", GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_128x64_k16() }),
        ("128x64_k16_sk4_wgp", GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_128x64_k16() }),
        // Deep K: k32 (2× fewer K-loop iterations)
        ("128x64_k32_sk4_wgp", GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_128x64_k32() }),
        ("128x64_k32_sk8_wgp", GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_128x64_k32() }),
        ("128x64_k32_sk4", GemmConfig { split_k: Some(4), ..GemmConfig::tile_128x64_k32() }),
        ("128x64_k32_sk8", GemmConfig { split_k: Some(8), ..GemmConfig::tile_128x64_k32() }),
        // 64×64 k64 (4× fewer K-loop iters, but smaller tile)
        ("64x64_k64_sk4_wgp", GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_64x64_k64() }),
        ("64x64_k64_sk8_wgp", GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_64x64_k64() }),
        ("64x64_k64_sk4", GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k64() }),
        // 32×64 k32 (4 M-tiles + deeper K)
        ("32x64_k32_sk4_wgp", GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_32x64_k32() }),
        ("32x64_k32_sk8_wgp", GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_32x64_k32() }),
        ("32x64_k32_sk4", GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x64_k32() }),
    ];

    eprintln!("── Compiling {} configs ──", configs.len());
    let mut compiled = Vec::new();
    for (name, cfg) in &configs {
        let kernel_ir = generate(cfg);
        let elf = kernel_ir.compile(Target::GFX1100)?;
        compiled.push(elf);
    }
    eprintln!("  All compiled OK");

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use std::time::Instant;

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        let x_bytes = (m * k * 2) as usize;
        let w_bytes = (n * k * 2) as usize;
        let y_bytes = (m * n * 4) as usize;
        let x_buf = device.alloc_vram(x_bytes)?;
        let w_buf = device.alloc_vram(w_bytes)?;
        let y_workspace = device.alloc_vram(y_bytes * 16)?;
        x_buf.write(&vec![0u8; x_bytes]);
        w_buf.write(&vec![0u8; w_bytes]);

        // Pre-load all kernels
        let mut gpu_kernels = Vec::new();
        for (ci, (_, cfg)) in configs.iter().enumerate() {
            gpu_kernels.push(GpuKernel::load(&device, &compiled[ci], &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?);
        }

        // GPU warmup: run first kernel a few times to bring clocks up
        eprintln!("── GPU warmup ──");
        {
            let cfg = &configs[0].1;
            let sk = cfg.split_k.unwrap_or(1);
            let (gx, gy) = compute_grid_split_k(cfg, m, n, sk);
            let y_stride = if sk > 1 { m * n * 4 } else { 0u32 };
            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_workspace.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());
            ka[36..40].copy_from_slice(&y_stride.to_le_bytes());
            for _ in 0..10 {
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernels[0], [gx, gy, 1], ka_buf);
                queue.wait_idle()?;
            }
        }
        eprintln!("  Warmup complete\n");

        eprintln!("{:>25} | {:>7} | {:>5} | {:>7} | {:>6}", "Config", "TFLOPS", "WGs", "μs", "%rocB");
        eprintln!("{}", "-".repeat(65));

        for (ci, (name, cfg)) in configs.iter().enumerate() {
            let sk = cfg.split_k.unwrap_or(1);
            let (gx, gy) = compute_grid_split_k(cfg, m, n, sk);
            let total_wgs = (gx / cfg.wg_size) * gy;
            let y_stride = if sk > 1 { m * n * 4 } else { 0u32 };

            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_workspace.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());
            ka[36..40].copy_from_slice(&y_stride.to_le_bytes());

            // Warmup this specific kernel
            for _ in 0..3 {
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernels[ci], [gx, gy, 1], ka_buf);
                queue.wait_idle()?;
            }

            let iters = 20;
            let t0 = Instant::now();
            for i in 0..iters {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit(&gpu_kernels[ci], [gx, gy, 1], ka_buf);
                queue.wait_idle()?;
            }
            let elapsed_us = t0.elapsed().as_micros() as f64 / iters as f64;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let tflops = flops / (elapsed_us * 1e6);
            let pct = tflops / 51.0 * 100.0;

            eprintln!("{:>25} | {:>6.2} | {:>5} | {:>6.1} | {:>5.1}%",
                name, tflops, total_wgs, elapsed_us, pct);
        }

        queue.synchronize()?;
    }
    Ok(())
}
