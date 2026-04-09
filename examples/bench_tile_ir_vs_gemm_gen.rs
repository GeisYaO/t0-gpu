//! # bench_tile_ir_vs_gemm_gen — Compare tile_ir compiler vs gemm_gen
//!
//! Both produce NT-mode GEMM (Y = A @ B^T), bf16 inputs, f32 output.
//! Uses raw KFD dispatch (same path as bench_gemm_sweep.rs).
//!
//! Run: cargo run --example bench_tile_ir_vs_gemm_gen --features rocm --release

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{self, GemmConfig};
use t0_gpu::t0::tile_ir::{self, TileGemm, TileTranspose};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  tile_ir vs gemm_gen — Performance Comparison               ║");
    eprintln!("║  AMD RX 7900 XTX (GFX1100, 96 CU, 123 TF bf16 peak)      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // Matrix sizes: (M, K, N)
    let sizes: Vec<(u32, u32, u32)> = vec![
        (128, 128, 64),
        (128, 256, 128),
        (128, 512, 128),
        (256, 256, 256),
        (256, 512, 256),
        (512, 512, 512),
        (128, 1024, 4096),
        (256, 1024, 4096),
        (512, 1024, 4096),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    // ── Step 1: Compile all kernels ──
    eprintln!("── Compiling kernels ──\n");

    struct KernelPair {
        tile_elf: Vec<u8>,
        tile_wg: u32,
        tile_lds: u32,
        tile_name: String,
        gemm_elf: Vec<u8>,
        gemm_wg: u32,
        gemm_lds: u32,
        gemm_name: String,
        gemm_cfg: GemmConfig,
        tile_spec: TileGemm,
    }

    let mut pairs: Vec<(u32, u32, u32, KernelPair)> = Vec::new();

    for &(m, k, n) in &sizes {
        // tile_ir
        let tile_spec = tile_ir::tile_auto_select(m, k, n, TileTranspose::NT);
        let tile_t0k = tile_ir::lower_gemm(&tile_spec);
        let tile_wg = tile_t0k.wg_size();
        let tile_lds = tile_spec.lds_total();
        let tile_name = tile_spec.name();
        eprint!("  tile_ir  {}×{}×{}: {} ... ", m, k, n, tile_name);
        let tile_elf = tile_t0k.compile(Target::GFX1100)?;
        eprintln!("✓ {} bytes", tile_elf.len());

        // gemm_gen
        let gemm_cfg = gemm_gen::auto_select(m, k, n);
        let gemm_t0k = gemm_gen::generate(&gemm_cfg);
        let gemm_wg = gemm_cfg.wg_size;
        let gemm_lds = gemm_cfg.lds_total();
        let gemm_name = gemm_cfg.name();
        eprint!("  gemm_gen {}×{}×{}: {} ... ", m, k, n, gemm_name);
        let gemm_elf = gemm_t0k.compile(Target::GFX1100)?;
        eprintln!("✓ {} bytes", gemm_elf.len());

        pairs.push((m, k, n, KernelPair {
            tile_elf, tile_wg, tile_lds, tile_name,
            gemm_elf, gemm_wg, gemm_lds, gemm_name,
            gemm_cfg, tile_spec,
        }));
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        eprintln!("\n── Loading kernels onto GPU ──\n");

        struct LoadedPair {
            tile_gk: GpuKernel,
            gemm_gk: GpuKernel,
        }

        let mut loaded: Vec<LoadedPair> = Vec::new();
        for (_m, _k, _n, pair) in &pairs {
            let tile_gk = GpuKernel::load(&device, &pair.tile_elf, &KernelLoadConfig {
                workgroup_size: [pair.tile_wg, 1, 1],
                lds_size: pair.tile_lds,
            })?;
            let gemm_gk = GpuKernel::load(&device, &pair.gemm_elf, &KernelLoadConfig {
                workgroup_size: [pair.gemm_wg, 1, 1],
                lds_size: pair.gemm_lds,
            })?;
            loaded.push(LoadedPair { tile_gk, gemm_gk });
        }

        // ── Benchmark ──
        eprintln!("\n{}", "=".repeat(95));
        eprintln!("  {:>15} | {:>26} | {:>26} | {:>8}",
            "Matrix", "tile_ir", "gemm_gen", "Speedup");
        eprintln!("  {:>15} | {:>12} {:>12} | {:>12} {:>12} | {:>8}",
            "(M×K×N)", "μs", "TFLOPS", "μs", "TFLOPS", "(t/g)");
        eprintln!("{}", "-".repeat(95));

        let warmup = 5;
        let n_iters = 20;

        for (idx, (m, k, n, pair)) in pairs.iter().enumerate() {
            let m = *m; let k = *k; let n = *n;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;

            // Allocate buffers
            let x_bytes = (m as usize) * (k as usize) * 2;
            let w_bytes = (n as usize) * (k as usize) * 2;
            let y_bytes = (m as usize) * (n as usize) * 4;

            let x_buf = device.alloc_vram(x_bytes.max(512))?;
            let w_buf = device.alloc_vram(w_bytes.max(512))?;
            // Extra space for split-K (max 16 splits)
            let y_tile = device.alloc_vram((y_bytes * 16).max(512))?;
            let y_gemm = device.alloc_vram((y_bytes * 16).max(512))?;

            // Fill with small non-zero bf16 values
            let x_data: Vec<u16> = (0..m*k).map(|i| {
                let v = ((i % 17) as f32 - 8.0) * 0.01;
                (v.to_bits() >> 16) as u16
            }).collect();
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
            });
            let w_data: Vec<u16> = (0..n*k).map(|i| {
                let v = ((i % 13) as f32 - 6.0) * 0.01;
                (v.to_bits() >> 16) as u16
            }).collect();
            w_buf.write(unsafe {
                std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
            });

            // ── tile_ir kernargs + grid ──
            let tile_ka = tile_ir::build_kernargs(
                x_buf.gpu_addr(), w_buf.gpu_addr(), y_tile.gpu_addr(),
                k, n, &pair.tile_spec,
            );
            let tile_grid = tile_ir::compute_grid(&pair.tile_spec, m, n);

            // ── gemm_gen kernargs + grid ──
            let gemm_ka = gemm_gen::build_kernargs(
                x_buf.gpu_addr(), w_buf.gpu_addr(), y_gemm.gpu_addr(),
                k, n, m, &pair.gemm_cfg,
            );
            let (gemm_gx, gemm_gy) = gemm_gen::compute_grid_auto(&pair.gemm_cfg, m, n);

            // ── Benchmark tile_ir ──
            for _ in 0..warmup {
                let ka_buf = pool.write_kernargs(0, &tile_ka);
                queue.submit(&loaded[idx].tile_gk, [tile_grid[0], tile_grid[1], tile_grid[2]], ka_buf);
                queue.wait_idle()?;
            }
            let t0 = std::time::Instant::now();
            for i in 0..n_iters {
                let ka_buf = pool.write_kernargs(i % 16, &tile_ka);
                queue.submit(&loaded[idx].tile_gk, [tile_grid[0], tile_grid[1], tile_grid[2]], ka_buf);
                queue.wait_idle()?;
            }
            let tile_us = t0.elapsed().as_micros() as f64 / n_iters as f64;
            let tile_tf = flops / (tile_us * 1e6);

            // ── Benchmark gemm_gen ──
            for _ in 0..warmup {
                let ka_buf = pool.write_kernargs(0, &gemm_ka);
                queue.submit(&loaded[idx].gemm_gk, [gemm_gx, gemm_gy, 1], ka_buf);
                queue.wait_idle()?;
            }
            let t1 = std::time::Instant::now();
            for i in 0..n_iters {
                let ka_buf = pool.write_kernargs(i % 16, &gemm_ka);
                queue.submit(&loaded[idx].gemm_gk, [gemm_gx, gemm_gy, 1], ka_buf);
                queue.wait_idle()?;
            }
            let gemm_us = t1.elapsed().as_micros() as f64 / n_iters as f64;
            let gemm_tf = flops / (gemm_us * 1e6);

            let speedup = tile_tf / gemm_tf;
            let marker = if speedup > 1.05 { "🟢" }
                else if speedup < 0.95 { "🔴" }
                else { "⚪" };

            eprintln!("  {:>4}×{:<4}×{:<4} | {:>10.1} {:>10.2} TF | {:>10.1} {:>10.2} TF | {:>5.2}x  {}  {} vs {}",
                m, k, n,
                tile_us, tile_tf,
                gemm_us, gemm_tf,
                speedup, marker,
                pair.tile_name, pair.gemm_name);

            queue.synchronize()?;
        }

        eprintln!("{}", "=".repeat(95));
        eprintln!("  🟢 = tile_ir faster | 🔴 = gemm_gen faster | ⚪ = comparable\n");
    }

    eprintln!("Done!");
    Ok(())
}
