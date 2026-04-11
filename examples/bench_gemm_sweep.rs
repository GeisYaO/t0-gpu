//! # bench_gemm_sweep — Sweep multiple T0 GEMM variants to find optimal per size
//!
//! Compiles all GemmConfig variants from gemm_gen::sweep_configs(),
//! benchmarks each on multiple matrix sizes, and reports the best variant.
//!
//! Run: cargo run --example bench_gemm_sweep --features rocm --release

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid, compute_grid_split_k};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM Sweep — Parameterized Kernel Auto-Tuning          ║");
    eprintln!("║  AMD RX 7900 XTX (GFX1100, 96 CU, 123 TF peak)           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Define configs to sweep
    let configs = vec![
        // ── 16×64 k16 (padded) + split-K ──
        GemmConfig::tile_16x64_k16(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_16x64_k16() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_16x64_k16() },
        GemmConfig { split_k: Some(8), ..GemmConfig::tile_16x64_k16() },
        GemmConfig { split_k: Some(16), ..GemmConfig::tile_16x64_k16() },
        // ── 32×64 k16 (padded) + split-K ──
        GemmConfig::tile_32x64_k16(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k16() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x64_k16() },
        GemmConfig { split_k: Some(8), ..GemmConfig::tile_32x64_k16() },
        GemmConfig { split_k: Some(16), ..GemmConfig::tile_32x64_k16() },
        // ── 32×64 k32 + split-K ──
        GemmConfig::tile_32x64_k32(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k32() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x64_k32() },
        // ── 64×64 k16 (padded, best for large) + split-K ──
        GemmConfig::tile_64x64_k16(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(8), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(16), ..GemmConfig::tile_64x64_k16() },
        // ── 32×128 k16 (padded) ──
        GemmConfig::tile_32x128_k16(),
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x128_k16() },
        // ── 128×64 k32 ──
        GemmConfig::tile_128x64_k32(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_128x64_k32() },
        // ── NEW: deeper K for fewer loop iterations ──
        GemmConfig::tile_64x64_k64(),                 // 4× fewer K-loop iters vs k16
        // ── M-on-X grid (better for rectangular M<N) ──
        GemmConfig { swap_grid: false, ..GemmConfig::tile_64x64_k16() },
        GemmConfig { swap_grid: false, split_k: Some(4), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { swap_grid: false, split_k: Some(8), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { swap_grid: false, ..GemmConfig::tile_128x64_k32() },
        GemmConfig { swap_grid: false, split_k: Some(2), ..GemmConfig::tile_128x64_k32() },
        // ── WGP mode (workgroup spans 2 CUs: 128KB LDS + 4 SIMDs) ──
        GemmConfig { wgp_mode: true, ..GemmConfig::tile_64x64_k16() },
        GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { wgp_mode: true, split_k: Some(16), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { wgp_mode: true, ..GemmConfig::tile_128x64_k32() },
        GemmConfig { wgp_mode: true, split_k: Some(2), ..GemmConfig::tile_128x64_k32() },
        // ── 128×64 WGP + deeper split-K (higher CU utilization) ──
        GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_128x64_k32() },
        GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_128x64_k16() },
        GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_128x64_k16() },
        // ── 64×64 WGP + split-K 4 (fill medium matrices) ──
        GemmConfig { wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_64x64_k16() },
        // ── Thin-matrix optimized: 128×64 k16, M-on-X + WGP ──
        GemmConfig { swap_grid: false, wgp_mode: true, ..GemmConfig::tile_128x64_k16() },
        GemmConfig { swap_grid: false, wgp_mode: true, split_k: Some(2), ..GemmConfig::tile_128x64_k16() },
        GemmConfig { swap_grid: false, wgp_mode: true, split_k: Some(4), ..GemmConfig::tile_128x64_k16() },
        GemmConfig { swap_grid: false, wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_128x64_k16() },
        // ── Deep-K CU mode for thin matrices ──
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_128x64_k32() },
        GemmConfig { split_k: Some(8), ..GemmConfig::tile_128x64_k32() },
        GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_128x64_k32() },
        // ── Higher split-K for thin matrices (128×4096: sk16→1024 WGs) ──
        GemmConfig { wgp_mode: true, split_k: Some(16), ..GemmConfig::tile_128x64_k16() },
        GemmConfig { swap_grid: false, wgp_mode: true, split_k: Some(16), ..GemmConfig::tile_128x64_k16() },
        GemmConfig { wgp_mode: true, split_k: Some(16), ..GemmConfig::tile_64x64_k16() },
        // ── Small-matrix focused (keep minimal) ──
        GemmConfig { wgp_mode: true, split_k: Some(8), ..GemmConfig::tile_32x64_k16() },
    ];

    // Matrix sizes: focus on small + representative large
    let sizes: Vec<(u32, u32, u32)> = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (128, 1024, 4096),
        (256, 1024, 4096),
        (512, 1024, 4096),
        (1024, 1024, 4096),
    ];

    // Compile all kernels
    eprintln!("\n── Compiling {} kernel variants ──", configs.len());
    let mut compiled = Vec::new();
    for cfg in &configs {
        eprint!("  [{}] ... ", cfg.name());
        let kernel_ir = generate(cfg);
        let elf = kernel_ir.compile(Target::GFX1100)?;
        eprintln!("✓ {} bytes", elf.len());
        compiled.push((cfg, elf));
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        // Load all kernels
        let mut gpu_kernels = Vec::new();
        for (cfg, elf) in &compiled {
            let gk = GpuKernel::load(&device, elf, &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?;
            gpu_kernels.push(gk);
        }

        // Results table: [size_idx][config_idx] = tflops
        let mut results: Vec<Vec<f64>> = vec![vec![0.0; configs.len()]; sizes.len()];

        // Header
        eprintln!("\n{:>6} {:>6} {:>6} |", "M", "K", "N");
        for cfg in &configs {
            eprint!(" {:>12} |", cfg.name().replace("t0_gemm_", ""));
        }
        eprintln!(" BEST");
        eprintln!("{}", "-".repeat(20 + configs.len() * 15 + 20));

        for (si, &(m, k, n)) in sizes.iter().enumerate() {
            eprint!("{:>6} {:>6} {:>6} |", m, k, n);

            let x_bytes = (m as usize) * (k as usize) * 2;
            let w_bytes = (n as usize) * (k as usize) * 2;
            let y_bytes = (m as usize) * (n as usize) * 4;

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            // Pre-allocate Y workspace for max split-K (16) to avoid per-config alloc/drop racing
            let max_sk = 16u32;
            let y_workspace = device.alloc_vram(y_bytes * max_sk as usize)?;

            // Fill with 1.0 bf16
            let x_data = vec![0x3F80u16; (m * k) as usize];
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
            });
            let w_data = vec![0x3F80u16; (n * k) as usize];
            w_buf.write(unsafe {
                std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
            });

            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_workspace.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());   // split_k_shift = 0 (no split)
            ka[36..40].copy_from_slice(&0u32.to_le_bytes());   // y_split_stride = 0

            let mut best_tf = 0.0f64;
            let mut best_idx = 0usize;

            for (ci, cfg) in configs.iter().enumerate() {
                // Check if this config is compatible with the matrix size
                let sk = cfg.split_k.unwrap_or(1);
                let effective_n = cfg.tile_n * cfg.n_col_passes;
                if m % cfg.tile_m != 0 || n % effective_n != 0 || k % (cfg.tile_k * sk) != 0 {
                    eprint!("      skip   |");
                    continue;
                }

                let (grid_x, grid_y) = if sk > 1 {
                    compute_grid_split_k(cfg, m, n, sk)
                } else {
                    compute_grid(cfg, m, n)
                };

                // Reuse pre-allocated workspace (already big enough for max split-K)
                let y_stride = if sk > 1 { (m * n * 4) as u32 } else { 0 };

                let mut ska = [0u8; 40];
                ska[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
                ska[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
                ska[16..24].copy_from_slice(&y_workspace.gpu_addr().to_le_bytes());
                ska[24..28].copy_from_slice(&k.to_le_bytes());
                ska[28..32].copy_from_slice(&n.to_le_bytes());
                ska[32..36].copy_from_slice(&0u32.to_le_bytes());
                ska[36..40].copy_from_slice(&y_stride.to_le_bytes());

                // Warmup
                for _ in 0..2 {
                    let ka_buf = pool.write_kernargs(0, &ska);
                    queue.submit(&gpu_kernels[ci], [grid_x, grid_y, 1], ka_buf);
                    queue.wait_idle()?;
                }

                // Timed (note: for split-K, this is GEMM only, no reduction cost)
                let n_iters = if m * n * k <= 512 * 512 * 512 { 20 } else { 10 };
                let start = std::time::Instant::now();
                for i in 0..n_iters {
                    let ka_buf = pool.write_kernargs(i % 16, &ska);
                    queue.submit(&gpu_kernels[ci], [grid_x, grid_y, 1], ka_buf);
                    queue.wait_idle()?;
                }
                let elapsed = start.elapsed();
                let avg_us = elapsed.as_micros() as f64 / n_iters as f64;
                let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
                let tflops = flops / (avg_us * 1e6);

                results[si][ci] = tflops;
                eprint!(" {:>8.2} TF |", tflops);

                if tflops > best_tf {
                    best_tf = tflops;
                    best_idx = ci;
                }
            }
            eprintln!(" ★ {} ({:.1} TF)", configs[best_idx].name().replace("t0_gemm_", ""), best_tf);

            // Synchronize GPU before dropping buffers — prevents use-after-free
            queue.synchronize()?;
        }

        // Summary: best config per size
        eprintln!("\n══ Optimal Kernel Selection ══");
        eprintln!("{:>18} | {:>20} | {:>8} | vs rocBLAS", "Matrix", "Best Kernel", "TFLOPS");
        eprintln!("{}", "-".repeat(65));
        // rocBLAS reference values (measured on RX 7900 XTX, ROCm 7.1.1)
        let rocblas_ref: Vec<((u32,u32,u32), f64)> = vec![
            ((256,256,256), 3.47), ((512,512,512), 12.50),
            ((1024,1024,1024), 27.89), ((2048,2048,2048), 36.65),
            ((4096,4096,4096), 58.72), ((8192,8192,8192), 71.71),
            ((128,1024,4096), 50.99), ((256,1024,4096), 44.43),
            ((512,1024,4096), 45.85), ((1024,1024,4096), 29.94),
        ];
        for (si, &(m, k, n)) in sizes.iter().enumerate() {
            let mut best_tf = 0.0f64;
            let mut best_name = String::new();
            for (ci, cfg) in configs.iter().enumerate() {
                if results[si][ci] > best_tf {
                    best_tf = results[si][ci];
                    best_name = cfg.name().replace("t0_gemm_", "");
                }
            }
            let rb = rocblas_ref.iter().find(|((rm,rk,rn),_)| *rm==m && *rk==k && *rn==n)
                .map(|(_,v)| *v).unwrap_or(0.0);
            let pct = if rb > 0.0 { best_tf / rb * 100.0 } else { 0.0 };
            let marker = if pct > 100.0 { "🏆" } else { "  " };
            eprintln!("{:>6}×{:<6}×{:<4} | {:>20} | {:>6.2} TF | {:>5.1}% {}", m, k, n, best_name, best_tf, pct, marker);
        }
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
