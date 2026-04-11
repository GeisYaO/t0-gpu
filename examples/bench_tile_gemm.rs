//! Benchmark: tile_ir GEMM (WMMA BF16→F32) across matrix sizes.
//!
//! This benchmarks the PRODUCTION tile_ir compilation path with full SSA optimization:
//!   buffer_load, graduated waitcnt, double-buffering, WMMA 16×16×16.
//!
//! Run:
//!   LIBRARY_PATH=/opt/rocm-7.1.1/lib LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib \
//!     cargo run --example bench_tile_gemm --features rocm --release

fn main() -> Result<(), String> {
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::ignis::gpu_context::GpuRuntime;
        use t0_gpu::t0::block_dsl::*;
        use t0_gpu::t0::gemm_gen::GemmConfig;
        use t0_gpu::t0::ir::Target;

        let rt = GpuRuntime::new().map_err(|e| format!("GpuRuntime: {e}"))?;

        eprintln!("╔═══════════════════════════════════════════════════════════════╗");
        eprintln!("║  T0 tile_ir GEMM — WMMA BF16→F32 Benchmark                  ║");
        eprintln!("║  AMD RX 7900 XTX (GFX1100, 96 CU, Wave32)                   ║");
        eprintln!("║  v_wmma_f32_16x16x16_bf16, buffer_load, double-buffered      ║");
        eprintln!("║  BF16 WMMA Peak: ~185 TFLOPS (96 CU × 512 ops/cyc × 4 GHz)  ║");
        eprintln!("╚═══════════════════════════════════════════════════════════════╝");

        fn make_bf16(n: usize, seed: u64) -> Vec<u16> {
            let mut s = seed;
            (0..n).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let f = (((s >> 33) as u32 % 200) as f32 - 100.0) * 0.01;
                (f.to_bits() >> 16) as u16
            }).collect()
        }

        fn c128() -> TileGemmConfig {
            TileGemmConfig { tile_m: 128, tile_n: 128, tile_k: 16,
                wgp_mode: false, split_k: 1, swap_grid: true }
        }

        let sizes: Vec<(u32, u32, u32, &str)> = vec![
            (256,  256,   256,  "256³"),
            (512,  512,   512,  "512³"),
            (1024, 1024,  1024, "1024³"),
            (2048, 2048,  2048, "2048³"),
            (4096, 4096,  4096, "4096³"),
            (1024, 4096,  1024, "1024×4096×1024"),
            (2048, 1024,  2048, "2048×1024×2048"),
        ];

        eprintln!("\n{:>20} | {:>8} {:>8} {:>6}",
            "Size", "μs", "TFLOPS", "Util%");
        eprintln!("{}", "─".repeat(55));

        for &(m, k, n, label) in &sizes {
            let cfg = c128();
            let gcfg = GemmConfig {
                tile_m: cfg.tile_m, tile_n: cfg.tile_n, tile_k: cfg.tile_k,
                wg_size: cfg.tile_m / 32 * 32,
                use_lds: true, double_buffer: true,
                split_k: None, lds_pad: 0, n_col_passes: 1,
                swap_grid: cfg.swap_grid, wgp_mode: cfg.wgp_mode,
                transpose: t0_gpu::t0::gemm_gen::GemmTranspose::NT,
                epilogue: t0_gpu::t0::gemm_gen::EpilogueOp::StoreF32,
            };

            let name = format!("bench_tgm_{}x{}x{}", m, k, n);
            let mut kb = BlockKernel::new(&name, gcfg.wg_size);
            let _x = kb.arg_ptr("X"); let _w = kb.arg_ptr("WT"); let _y = kb.arg_ptr("Y");
            let _k = kb.arg_u32("K"); let _n = kb.arg_u32("N");
            let _sks = kb.arg_u32("split_k_shift"); let _yss = kb.arg_u32("y_split_stride");
            let _m = kb.arg_u32("M");
            kb.tile_gemm(_x, _w, _y, _k, _n, cfg.clone());
            let compiled = kb.compile(Target::GFX1100)?;
            let gpu_kernel = rt.compile_dsl(compiled).map_err(|e| e.to_string())?;

            // Pad K to tile_k
            let k_padded = (k + cfg.tile_k - 1) & !(cfg.tile_k - 1);
            let m_padded = (m + cfg.tile_m - 1) & !(cfg.tile_m - 1);
            let n_padded = (n + cfg.tile_n - 1) & !(cfg.tile_n - 1);

            let xb = make_bf16((m * k) as usize, 42 + m as u64);
            let wb = make_bf16((n * k) as usize, 137 + n as u64);

            let xbuf = rt.alloc((m * k_padded * 2) as usize).map_err(|e| e.to_string())?;
            xbuf.zero();
            xbuf.write(&xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());

            let wbuf = rt.alloc((n * k_padded * 2) as usize).map_err(|e| e.to_string())?;
            wbuf.zero();
            wbuf.write(&wb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());

            let ybuf = rt.alloc_f32((m_padded * n_padded) as usize).map_err(|e| e.to_string())?;
            ybuf.zero();

            let mut ka = [0u8; 44];
            ka[0..8].copy_from_slice(&xbuf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&wbuf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&ybuf.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k_padded.to_le_bytes());
            ka[28..32].copy_from_slice(&n_padded.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());
            ka[36..40].copy_from_slice(&0u32.to_le_bytes());
            ka[40..44].copy_from_slice(&m.to_le_bytes());

            let tiles_m = (m + cfg.tile_m - 1) / cfg.tile_m;
            let tiles_n = (n + cfg.tile_n - 1) / cfg.tile_n;
            let (gx, gy) = if cfg.swap_grid {
                (tiles_n * gcfg.wg_size, tiles_m)
            } else {
                (tiles_m * gcfg.wg_size, tiles_n)
            };

            // Warmup
            for _ in 0..3 {
                rt.dispatch(&gpu_kernel, [gx, gy, 1], &ka)?;
            }

            // Timed runs
            let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
            let n_iters: u32 = if flops > 2e11 { 5 } else if flops > 2e9 { 20 } else { 50 };

            let start = std::time::Instant::now();
            for _ in 0..n_iters {
                rt.dispatch(&gpu_kernel, [gx, gy, 1], &ka)?;
            }
            let elapsed_us = start.elapsed().as_secs_f64() * 1e6 / n_iters as f64;
            let tflops = flops / (elapsed_us * 1e6);
            let util = tflops / 185.0 * 100.0;

            eprintln!("{:>20} | {:>6.0}μs {:>6.2} TF {:>5.1}%",
                label, elapsed_us, tflops, util);

            rt.recycle(xbuf);
            rt.recycle(wbuf);
            rt.recycle(ybuf);
        }

        eprintln!("\n══ Done ══");
    }

    #[cfg(not(feature = "rocm"))]
    eprintln!("Compile with --features rocm");

    Ok(())
}
