//! # bench_tile_ir — tile_ir vs gemm_gen GEMM Performance Benchmark
//!
//! Compares tile_ir (compiler-generated) vs gemm_gen (hand-tuned) GEMM kernels
//! on AMD RX 7900 XTX (GFX1100, 96 CU).
//!
//! Uses GpuRuntime for dispatch (same path as correctness tests).
//!
//! Run: cargo run --example bench_tile_ir --features rocm --release

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen;
use t0_gpu::t0::tile_ir::{self, TileGemm};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  tile_ir vs gemm_gen — AMD RX 7900 XTX (GFX1100, 96 CU)   ║");
    eprintln!("║  bf16 input → WMMA f32 accumulate → f32 output (NT mode)   ║");
    eprintln!("║  Peak: 123 TFLOPS (bf16 WMMA)                             ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::ignis::gpu_context::GpuRuntime;
        use std::sync::Arc;

        let rt = GpuRuntime::new().map_err(|e| format!("GpuRuntime: {e}"))?;

        // Matrix sizes: (M, K, N) — NT GEMM: Y[M,N] = X[M,K] × W^T[N,K]
        let sizes: Vec<(u32, u32, u32)> = vec![
            (64,   64,   64),
            (128,  64,   64),
            (128, 128,  128),
            (256, 256,  256),
            (512, 512,  512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (128, 1024, 4096),
            (256, 1024, 4096),
            (512, 1024, 4096),
            (1024, 1024, 4096),
        ];

        eprintln!("\n{:>6} {:>6} {:>6} | {:>8} {:>8} {:>6} | {:>8} {:>8} {:>6} | {:>6}",
            "M", "K", "N",
            "tile_ir", "TFLOPS", "Util%",
            "gemm_gen", "TFLOPS", "Util%",
            "ratio");
        eprintln!("{}", "-".repeat(95));

        for &(m, k, n) in &sizes {
            // ── Allocate shared buffers ──
            let x_bytes = (m as usize) * (k as usize) * 2;
            let w_bytes = (n as usize) * (k as usize) * 2;
            let y_bytes = (m as usize) * (n as usize) * 4;

            let x_buf = rt.alloc(x_bytes)?;
            let w_buf = rt.alloc(w_bytes)?;

            // Fill with bf16 data (0x3F80 = 1.0 in bf16)
            let x_data: Vec<u8> = vec![0x3F80u16; (m as usize) * (k as usize)]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            x_buf.write(&x_data);
            let w_data: Vec<u8> = vec![0x3F80u16; (n as usize) * (k as usize)]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            w_buf.write(&w_data);

            let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);

            // ── Benchmark tile_ir ──
            let tile_us = {
                let spec = if m >= 128 {
                    TileGemm::tile_128x64_k16()
                } else {
                    TileGemm::tile_32x64_k16()
                };
                let kernel_name = format!("tile_bench_{}x{}", spec.tile_m, spec.tile_n);
                let spec_c = spec.clone();
                let kernel = rt.ensure_kernel_t0(
                    &kernel_name,
                    || tile_ir::lower_gemm(&spec_c),
                    [spec.wg_size(), 1, 1],
                    spec.lds_total(),
                )?;

                let y_buf = rt.alloc_zero(y_bytes)?;
                let ka = tile_ir::build_kernargs(
                    x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, &spec,
                );
                let grid = tile_ir::compute_grid(&spec, m, n);

                // Warmup
                for _ in 0..3 {
                    rt.dispatch(&kernel, grid, &ka)?;
                }

                // Timed
                let n_iters = if (m as u64) * (k as u64) * (n as u64) <= 512*512*512 { 50 } else { 10 };
                let start = std::time::Instant::now();
                for _ in 0..n_iters {
                    rt.dispatch(&kernel, grid, &ka)?;
                }
                start.elapsed().as_micros() as f64 / n_iters as f64
            };

            // ── Benchmark gemm_gen ──
            let gemm_us = {
                let cfg = gemm_gen::auto_select(m, k, n);
                let cfg_name = format!("gemm_bench_{}", cfg.name());
                let cfg_c = cfg.clone();
                let kernel = rt.ensure_kernel_t0(
                    &cfg_name,
                    || gemm_gen::generate(&cfg_c),
                    [cfg.wg_size, 1, 1],
                    cfg.lds_total(),
                )?;

                // split_k>1 writes split_k partial sums → need split_k × M×N×4 bytes
                let sk = cfg.split_k.unwrap_or(1).max(1) as usize;
                let y_alloc = y_bytes * sk;
                let y_buf = rt.alloc_zero(y_alloc)?;
                let ka = gemm_gen::build_kernargs(
                    x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, &cfg,
                );
                let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m, n);

                // Warmup
                for _ in 0..3 {
                    rt.dispatch(&kernel, [gx, gy, 1], &ka)?;
                }

                // Timed
                let n_iters = if (m as u64) * (k as u64) * (n as u64) <= 512*512*512 { 50 } else { 10 };
                let start = std::time::Instant::now();
                for _ in 0..n_iters {
                    rt.dispatch(&kernel, [gx, gy, 1], &ka)?;
                }
                start.elapsed().as_micros() as f64 / n_iters as f64
            };

            let tile_tflops = flops / (tile_us * 1e6);
            let gemm_tflops = flops / (gemm_us * 1e6);
            let tile_util = tile_tflops / 123.0 * 100.0;
            let gemm_util = gemm_tflops / 123.0 * 100.0;
            let ratio = tile_tflops / gemm_tflops;

            eprintln!("{:>6} {:>6} {:>6} | {:>6.1}μs {:>6.2} TF {:>5.1}% | {:>6.1}μs {:>6.2} TF {:>5.1}% | {:>5.2}x",
                m, k, n,
                tile_us, tile_tflops, tile_util,
                gemm_us, gemm_tflops, gemm_util,
                ratio);
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        eprintln!("\nKFD runtime not available. Compile with --features rocm");
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
