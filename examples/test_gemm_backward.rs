//! # test_gemm_backward — GEMM Backward Correctness Test
//!
//! Verifies that GEMM backward operations produce correct results:
//!   - dX = dY @ W^T  (backward data, uses NT GEMM directly)
//!   - dW = dY^T @ X  (backward weight, uses 2 transposes + NT GEMM)
//!
//! Run: cargo run --example test_gemm_backward --features rocm --release

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{
    generate, auto_select_backward_data, auto_select_backward_weight,
    build_kernargs_backward_data, build_kernargs_backward_weight,
    compute_grid_backward_data, compute_grid_backward_weight,
};

// ── bf16 helpers ──

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn rand_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut state = seed ^ 0xDEADBEEFCAFEBABE;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let f = ((state >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(f32_to_bf16(f));
    }
    out
}

// ── CPU reference implementations ──

/// CPU: dX[M,K] = dY[M,N] @ W[N,K]  (data backward)
fn cpu_backward_data(dy: &[u16], w: &[u16], m: usize, n: usize, k: usize) -> Vec<f32> {
    // dX[i,j] = sum_p dY[i,p] * W[p,j]   (p over N)
    let mut dx = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut acc = 0.0f32;
            for p in 0..n {
                let a = bf16_to_f32(dy[i * n + p]);
                let b = bf16_to_f32(w[p * k + j]);
                acc += a * b;
            }
            dx[i * k + j] = acc;
        }
    }
    dx
}

/// CPU: dW[N,K] = dY^T[N,M] @ X[M,K]  (weight backward)
fn cpu_backward_weight(dy: &[u16], x: &[u16], m: usize, n: usize, k: usize) -> Vec<f32> {
    // dW[i,j] = sum_p dY^T[i,p] * X[p,j] = sum_p dY[p,i] * X[p,j]  (p over M)
    let mut dw = vec![0.0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut acc = 0.0f32;
            for p in 0..m {
                let a = bf16_to_f32(dy[p * n + i]);  // dY^T[i,p] = dY[p,i]
                let b = bf16_to_f32(x[p * k + j]);
                acc += a * b;
            }
            dw[i * k + j] = acc;
        }
    }
    dw
}

/// CPU bf16 transpose: A[rows,cols] → A^T[cols,rows]
fn cpu_transpose_bf16(a: &[u16], rows: usize, cols: usize) -> Vec<u16> {
    let mut out = vec![0u16; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

fn compare(gpu: &[f32], cpu: &[f32], label: &str, k_dim: u32) -> bool {
    let n = gpu.len();
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f64;
    let mut sum_ref = 0.0f64;
    for i in 0..n {
        let err = (gpu[i] - cpu[i]).abs();
        if err > max_err { max_err = err; }
        sum_err += err as f64;
        sum_ref += cpu[i].abs() as f64;
    }
    let avg_err = sum_err / n as f64;
    let rel_err = if sum_ref > 0.0 { sum_err / sum_ref } else { 0.0 };
    let tol = (k_dim as f32) * 0.01;
    let pass = max_err < tol;
    let marker = if pass { "✅ PASS" } else { "❌ FAIL" };
    eprintln!("  {} max={:.4e} avg={:.4e} rel={:.4e} {}", label, max_err, avg_err, rel_err, marker);
    pass
}

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM Backward Correctness Test                     ║");
    eprintln!("║  dX = dY @ W (data)  |  dW = dY^T @ X (weight)        ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");

    // Test sizes: (M, K, N) — M=batch, K=input_dim, N=output_dim
    // All must be multiple of 64 (tile_m/tile_n alignment)
    let sizes: Vec<(u32, u32, u32)> = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (128, 256, 512),
        (64, 512, 128),
        (256, 128, 256),
        (512, 256, 128),
    ];

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;
        let mut total = 0u32;
        let mut passed = 0u32;

        // Use safe config that works for all sizes (64×64 tiles divide any multiple of 64)
        use t0_gpu::t0::gemm_gen::GemmConfig;

        for &(m, k, n) in &sizes {
            eprintln!("\n── M={} K={} N={} ──", m, k, n);

            // Generate random bf16 data
            let x_data = rand_bf16((m * k) as usize, m as u64 * 100 + k as u64);
            let w_data = rand_bf16((n * k) as usize, n as u64 * 200 + k as u64);
            let dy_data = rand_bf16((m * n) as usize, m as u64 * 300 + n as u64);

            // ════════════════════════════════════════════
            // Test 1: Backward Data — dX = dY @ WT
            // ════════════════════════════════════════════
            {
                let cpu_dx = cpu_backward_data(&dy_data, &w_data,
                    m as usize, n as usize, k as usize);

                // Forward used Y = X @ WT^T where WT is stored [N,K].
                // Backward: dX = dY @ WT, which needs NT GEMM with B = WT^T[K,N] = W[K,N].
                // So we must transpose WT[N,K] → W[K,N] first.
                let w_transposed = cpu_transpose_bf16(&w_data, n as usize, k as usize); // W[K,N]

                let cfg = GemmConfig::tile_64x64_k16();
                let kernel_ir = generate(&cfg);
                let elf = kernel_ir.compile(Target::GFX1100)?;
                let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
                    workgroup_size: [cfg.wg_size, 1, 1],
                    lds_size: cfg.lds_total(),
                })?;

                // Upload dY[M,N] and W[K,N] (transposed WT)
                let dy_buf = device.alloc_vram((m * n * 2) as usize)?;
                let w_buf = device.alloc_vram((k * n * 2) as usize)?;
                dy_buf.write(unsafe {
                    std::slice::from_raw_parts(dy_data.as_ptr() as *const u8, (m * n * 2) as usize)
                });
                w_buf.write(unsafe {
                    std::slice::from_raw_parts(w_transposed.as_ptr() as *const u8, (k * n * 2) as usize)
                });

                let sk = cfg.split_k.unwrap_or(1);
                let dx_elems = (m * k) as usize;
                let dx_buf = device.alloc_vram(dx_elems * 4 * sk as usize)?;
                dx_buf.zero();

                let ka = build_kernargs_backward_data(
                    dy_buf.gpu_addr(), w_buf.gpu_addr(), dx_buf.gpu_addr(),
                    m, n, k, &cfg
                );
                let (gx, gy) = compute_grid_backward_data(&cfg, m, k);

                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernel, [gx, gy, 1], ka_buf);
                queue.wait_idle()?;

                // Read back and reduce split-K partials
                let mut dx_gpu = vec![0f32; dx_elems];
                if sk > 1 {
                    let mut partials = vec![0f32; dx_elems * sk as usize];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            dx_buf.host_ptr as *const f32, partials.as_mut_ptr(),
                            dx_elems * sk as usize,
                        );
                    }
                    for s in 0..sk as usize {
                        for i in 0..dx_elems { dx_gpu[i] += partials[s * dx_elems + i]; }
                    }
                } else {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            dx_buf.host_ptr as *const f32, dx_gpu.as_mut_ptr(), dx_elems,
                        );
                    }
                }

                total += 1;
                // Contraction dimension for dX backward is N (sum over dY cols × W rows)
                if compare(&dx_gpu, &cpu_dx, "dX (bwd_data)", n) { passed += 1; }
            }

            // ════════════════════════════════════════════
            // Test 2: Backward Weight — dW = dY^T @ X^T^T
            // ════════════════════════════════════════════
            {
                let cpu_dw = cpu_backward_weight(&dy_data, &x_data,
                    m as usize, n as usize, k as usize);

                // Step 1: CPU transpose (in real usage, use T0 transpose kernel on GPU)
                let dy_t = cpu_transpose_bf16(&dy_data, m as usize, n as usize);  // [N,M]
                let x_t = cpu_transpose_bf16(&x_data, m as usize, k as usize);   // [K,M]

                // Step 2: dW = dY_T[N,M] @ X_T[K,M]^T → NT GEMM
                let cfg = GemmConfig::tile_64x64_k16();
                let kernel_ir = generate(&cfg);
                let elf = kernel_ir.compile(Target::GFX1100)?;
                let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
                    workgroup_size: [cfg.wg_size, 1, 1],
                    lds_size: cfg.lds_total(),
                })?;

                // Upload transposed buffers
                let dy_t_buf = device.alloc_vram((n * m * 2) as usize)?;
                let x_t_buf = device.alloc_vram((k * m * 2) as usize)?;
                dy_t_buf.write(unsafe {
                    std::slice::from_raw_parts(dy_t.as_ptr() as *const u8, (n * m * 2) as usize)
                });
                x_t_buf.write(unsafe {
                    std::slice::from_raw_parts(x_t.as_ptr() as *const u8, (k * m * 2) as usize)
                });

                let sk = cfg.split_k.unwrap_or(1);
                let dw_elems = (n * k) as usize;
                let dw_buf = device.alloc_vram(dw_elems * 4 * sk as usize)?;
                dw_buf.zero();

                let ka = build_kernargs_backward_weight(
                    dy_t_buf.gpu_addr(), x_t_buf.gpu_addr(), dw_buf.gpu_addr(),
                    m, n, k, &cfg
                );
                let (gx, gy) = compute_grid_backward_weight(&cfg, n, k);

                let ka_buf = pool.write_kernargs(1, &ka);
                queue.submit(&gpu_kernel, [gx, gy, 1], ka_buf);
                queue.wait_idle()?;

                let mut dw_gpu = vec![0f32; dw_elems];
                if sk > 1 {
                    let mut partials = vec![0f32; dw_elems * sk as usize];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            dw_buf.host_ptr as *const f32, partials.as_mut_ptr(),
                            dw_elems * sk as usize,
                        );
                    }
                    for s in 0..sk as usize {
                        for i in 0..dw_elems { dw_gpu[i] += partials[s * dw_elems + i]; }
                    }
                } else {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            dw_buf.host_ptr as *const f32, dw_gpu.as_mut_ptr(), dw_elems,
                        );
                    }
                }

                total += 1;
                // Contraction dimension for dW backward is M (sum over dY^T cols × X rows)
                if compare(&dw_gpu, &cpu_dw, "dW (bwd_weight)", m) { passed += 1; }
            }
        }

        eprintln!("\n══ Summary ══");
        eprintln!("  Total: {} tests, {} PASS, {} FAIL", total, passed, total - passed);
        if passed == total {
            eprintln!("  ✅ All backward GEMM tests passed!");
        } else {
            eprintln!("  ⚠️ {} tests FAILED!", total - passed);
            std::process::exit(1);
        }
    }

    #[cfg(not(feature = "rocm"))]
    eprintln!("KFD runtime not available. Compile with --features rocm");

    Ok(())
}
