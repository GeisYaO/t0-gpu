//! Quick benchmark: block_dsl gemm_tn_naive across matrix sizes.
//!
//! Run: LIBRARY_PATH=/opt/rocm-7.1.1/lib LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib \
//!      cargo run --example bench_block_dsl_gemm --features rocm --release

fn main() -> Result<(), String> {
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::ignis::gpu_context::GpuRuntime;
        use t0_gpu::t0::block_dsl::BlockKernel;
        use t0_gpu::t0::Target;

        let rt = GpuRuntime::new().map_err(|e| format!("GpuRuntime: {e}"))?;

        const BM: u32 = 16;
        const BN: u32 = 16;
        const BLOCK_SIZE: u32 = BM * BN;

        // Build the gemm_tn_naive kernel once (generic over M,N,K via kernargs)
        let compiled = {
            let mut kb = BlockKernel::new("bench_gemm_tn", BLOCK_SIZE);
            let a_ptr = kb.arg_ptr("A");
            let b_ptr = kb.arg_ptr("B");
            let c_ptr = kb.arg_ptr("C");
            let m_arg = kb.arg_u32("M");
            let n_arg = kb.arg_u32("N");
            let k_arg = kb.arg_u32("K");

            let pid_m = kb.program_id(0);
            let pid_n = kb.program_id(1);
            let tid = kb.arange(0, BLOCK_SIZE);
            let local_n = tid.bitand(&mut kb, (BN - 1) as u32);
            let local_m = tid.shr(&mut kb, 4);
            let bm = kb.const_u32(BM);
            let bn = kb.const_u32(BN);
            let gm = pid_m.mul(&mut kb, bm).add(&mut kb, local_m);
            let gn = pid_n.mul(&mut kb, bn).add(&mut kb, local_n);

            let mask_m = gm.lt(&mut kb, m_arg);
            let mask_n = gn.lt(&mut kb, n_arg);

            let lds = kb.lds_alloc(BLOCK_SIZE * 4);
            let lds_tid = kb.arange(0, BLOCK_SIZE);
            let zero_f = kb.const_f32(0.0);
            kb.lds_store(lds, lds_tid, zero_f);
            kb.barrier();

            let zero = kb.const_u32(0);
            let iter_k = kb.for_range(zero, k_arg, 1);
            {
                let a_off = iter_k.mul(&mut kb, m_arg).add(&mut kb, gm);
                let a_val = kb.load(a_ptr, a_off, mask_m);
                let b_off = iter_k.mul(&mut kb, n_arg).add(&mut kb, gn);
                let b_val = kb.load(b_ptr, b_off, mask_n);
                let prod = a_val.mul(&mut kb, b_val);
                let cur = kb.lds_load(lds, lds_tid);
                let new_acc = cur.add(&mut kb, prod);
                kb.lds_store(lds, lds_tid, new_acc);
                kb.barrier();
            }
            kb.end_for(iter_k);

            let result = kb.lds_load(lds, lds_tid);
            let c_off = gm.mul(&mut kb, n_arg).add(&mut kb, gn);
            kb.store(c_ptr, c_off, result, mask_m);
            kb.compile(Target::GFX1100).unwrap()
        };

        let gpu_kernel = rt.compile_dsl(compiled.clone()).unwrap();

        eprintln!("╔═══════════════════════════════════════════════════════════╗");
        eprintln!("║  T0 SSA Compiler — block_dsl gemm_tn_naive Benchmark    ║");
        eprintln!("║  AMD RX 7900 XTX (GFX1100, 96 CU)                      ║");
        eprintln!("║  f32 scalar GEMM, 16×16 tile, WG=256, LDS accumulation  ║");
        eprintln!("║  Peak f32 VALU: ~49 TFLOPS                              ║");
        eprintln!("╚═══════════════════════════════════════════════════════════╝");

        eprintln!("\n{:>6} {:>6} {:>6} | {:>8} {:>8} {:>6}",
            "M", "K", "N", "μs", "GFLOPS", "Util%");
        eprintln!("{}", "-".repeat(55));

        let sizes: Vec<(u32, u32, u32)> = vec![
            (32,   64,   32),
            (64,   64,   64),
            (128, 128,  128),
            (256, 256,  256),
            (512, 512,  512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (128, 1024, 4096),
            (256, 1024, 4096),
        ];

        for &(m, k, n) in &sizes {
            let a_buf = rt.alloc_f32((k * m) as usize).map_err(|e| format!("alloc: {e}"))?;
            let b_buf = rt.alloc_f32((k * n) as usize).map_err(|e| format!("alloc: {e}"))?;
            let c_buf = rt.alloc_f32((m * n) as usize).map_err(|e| format!("alloc: {e}"))?;

            // Fill with ones (for simple verification)
            let a_data: Vec<f32> = vec![1.0; (k * m) as usize];
            let b_data: Vec<f32> = vec![1.0; (k * n) as usize];
            rt.write_f32(&a_buf, &a_data);
            rt.write_f32(&b_buf, &b_data);

            let mut ka = vec![0u8; compiled.kernarg_size];
            ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&c_buf.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&m.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&k.to_le_bytes());

            let grid_m = ((m + BM - 1) / BM) * BLOCK_SIZE;
            let grid_n = (n + BN - 1) / BN;

            // Warmup
            for _ in 0..3 {
                rt.dispatch(&gpu_kernel, [grid_m, grid_n, 1], &ka)
                    .map_err(|e| format!("dispatch: {e}"))?;
            }

            // Timed runs
            let n_iters = if (m as u64) * (k as u64) * (n as u64) <= 256*256*256 { 100 } else if (m as u64) * (k as u64) * (n as u64) <= 1024*1024*1024 { 20 } else { 5 };
            let start = std::time::Instant::now();
            for _ in 0..n_iters {
                rt.dispatch(&gpu_kernel, [grid_m, grid_n, 1], &ka)
                    .map_err(|e| format!("dispatch: {e}"))?;
            }
            let elapsed_us = start.elapsed().as_micros() as f64 / n_iters as f64;

            let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
            let gflops = flops / (elapsed_us * 1e3);
            let util = gflops / 49_000.0 * 100.0; // f32 VALU peak ~49 TFLOPS

            eprintln!("{:>6} {:>6} {:>6} | {:>6.1}μs {:>6.1} GF {:>5.1}%",
                m, k, n, elapsed_us, gflops, util);
        }

        eprintln!("\n══ Done ══");
    }

    #[cfg(not(feature = "rocm"))]
    eprintln!("Compile with --features rocm");

    Ok(())
}
