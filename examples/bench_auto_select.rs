//! Validate auto_select: safe pattern with per-size Y allocation 
use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid_auto, auto_select, build_kernargs};

fn main() -> Result<(), String> {
    eprintln!("══ Auto-Select Validation ══\n");

    // Start with small sizes to warm up GPU, then ramp up
    let sizes: Vec<(u32, u32, u32)> = vec![
        (64,64,64), (128,128,128), (256,256,256), (384,384,384),
        (512,512,512), (1024,1024,1024), (2048,2048,2048),
        (128,1024,4096), (256,1024,4096), (512,1024,4096), (1024,1024,4096),
        // 4096² needs ~1GB just for X+W, test separately if needed
    ];

    // Phase 1: Collect configs and show what was selected
    let mut unique_configs: Vec<GemmConfig> = Vec::new();
    let mut size_to_cfg_idx: Vec<usize> = Vec::new();
    
    eprintln!("{:>18} | {:>35} | sk | tile", "Matrix", "Selected config");
    eprintln!("{}", "-".repeat(80));
    for &(m,k,n) in &sizes {
        let cfg = auto_select(m, k, n);
        let sk = cfg.split_k.unwrap_or(1);
        eprintln!("{:>6}×{:<6}×{:<4} | {:>35} | {:>2} | {}×{}",
            m, k, n, cfg.name().replace("t0_gemm_", ""), sk, cfg.tile_m, cfg.tile_n);
        
        let idx = unique_configs.iter().position(|c| c.name() == cfg.name());
        if let Some(i) = idx {
            size_to_cfg_idx.push(i);
        } else {
            size_to_cfg_idx.push(unique_configs.len());
            unique_configs.push(cfg);
        }
    }

    eprintln!("\n── Compiling {} unique configs ──", unique_configs.len());
    let mut compiled = Vec::new();
    for cfg in &unique_configs {
        let ir = generate(cfg);
        compiled.push(ir.compile(Target::GFX1100)?);
    }
    eprintln!("  OK");

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use std::time::Instant;

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        eprintln!("── Loading kernels ──");
        let mut gpu_kernels = Vec::new();
        for (i, cfg) in unique_configs.iter().enumerate() {
            gpu_kernels.push(GpuKernel::load(&device, &compiled[i], &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?);
        }
        eprintln!("  OK\n");

        let rocblas: Vec<((u32,u32,u32), f64)> = vec![
            ((256,256,256), 3.47), ((512,512,512), 12.50),
            ((1024,1024,1024), 27.89), ((2048,2048,2048), 36.65),
            ((4096,4096,4096), 58.72),
            ((128,1024,4096), 50.99), ((256,1024,4096), 44.43),
            ((512,1024,4096), 45.85), ((1024,1024,4096), 29.94),
        ];

        eprintln!("{:>18} | {:>8} | {:>6} | {:>25}", "Matrix", "TFLOPS", "%rocB", "config");
        eprintln!("{}", "-".repeat(65));

        for (si, &(m,k,n)) in sizes.iter().enumerate() {
            let ci = size_to_cfg_idx[si];
            let cfg = &unique_configs[ci];
            let sk = cfg.split_k.unwrap_or(1);

            // Validate dimensions
            if k % (cfg.tile_k * sk) != 0 || m % cfg.tile_m != 0 || n % cfg.tile_n != 0 {
                eprintln!("{:>6}×{:<6}×{:<4} | SKIP (dim)", m, k, n);
                continue;
            }

            let x_bytes = (m * k * 2) as usize;
            let w_bytes = (n * k * 2) as usize;
            let y_bytes = (m as usize) * (n as usize) * 4 * (sk as usize);

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            let y_buf = device.alloc_vram(y_bytes)?;
            x_buf.write(&vec![0u8; x_bytes]);
            w_buf.write(&vec![0u8; w_bytes]);

            let (gx, gy) = compute_grid_auto(cfg, m, n);
            let ka = build_kernargs(x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(), k, n, m, cfg);

            // Warmup
            for _ in 0..3 {
                let kb = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernels[ci], [gx, gy, 1], kb);
                queue.wait_idle()?;
            }

            let iters = if (m as u64)*(n as u64) <= 512*512 { 20 } else { 10 };
            let t0 = Instant::now();
            for i in 0..iters {
                let kb = pool.write_kernargs(i % 16, &ka);
                queue.submit(&gpu_kernels[ci], [gx, gy, 1], kb);
                queue.wait_idle()?;
            }
            let us = t0.elapsed().as_micros() as f64 / iters as f64;
            let tf = 2.0 * m as f64 * k as f64 * n as f64 / (us * 1e6);
            let rb = rocblas.iter().find(|((rm,rk,rn),_)| *rm==m && *rk==k && *rn==n)
                .map(|(_,v)| *v).unwrap_or(0.0);
            let pct = if rb > 0.0 { tf / rb * 100.0 } else { 0.0 };
            let mark = if pct > 100.0 { "🏆" } else { "  " };

            eprintln!("{:>6}×{:<6}×{:<4} | {:>6.2} TF | {:>5.1}% | {:>25} {}",
                m, k, n, tf, pct, cfg.name().replace("t0_gemm_", ""), mark);

            queue.synchronize()?;
            // Explicit drop order: Y first, then W, then X
            drop(y_buf);
            drop(w_buf);
            drop(x_buf);
        }
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
