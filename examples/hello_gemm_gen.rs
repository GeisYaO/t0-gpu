//! # hello_gemm_gen — T0 Auto-Select GEMM Demo
//!
//! Demonstrates the integrated GEMM generator:
//!   1. auto_select(M, K, N) → picks optimal GemmConfig
//!   2. generate() → compiles kernel
//!   3. Dispatches and verifies correctness
//!
//! Run: cargo run --example hello_gemm_gen --features rocm --release

use t0_gpu::t0::{Target, GemmConfig, auto_select, compute_grid_auto, build_kernargs};
use t0_gpu::t0::gemm_gen::generate;

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════╗");
    eprintln!("║  T0 Auto-Select GEMM — Integration Demo         ║");
    eprintln!("╚══════════════════════════════════════════════════╝");

    let test_cases: Vec<(u32, u32, u32)> = vec![
        (256, 256, 256),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (128, 1024, 4096),
        (512, 1024, 4096),
        (1024, 1024, 4096),
    ];

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};
        use std::collections::HashMap;

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        // Kernel cache: config name → compiled kernel
        let mut kernel_cache: HashMap<String, GpuKernel> = HashMap::new();

        eprintln!("\n{:>6} {:>6} {:>6} | {:>25} | {:>8} | {:>8} | Result",
                  "M", "K", "N", "Config", "Time", "TFLOPS");
        eprintln!("{}", "-".repeat(85));

        for (m, k, n) in &test_cases {
            let (m, k, n) = (*m, *k, *n);

            // Auto-select the best config
            let cfg = auto_select(m, k, n);
            let cfg_name = cfg.name();

            // 2. Compile kernel if not cached
            if !kernel_cache.contains_key(&cfg_name) {
                let kernel_ir = generate(&cfg);
                let elf = kernel_ir.compile(Target::GFX1100)?;
                let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
                    workgroup_size: [cfg.wg_size, 1, 1],
                    lds_size: cfg.lds_total(),
                })?;
                eprintln!("  [compiled] {} ({} bytes)", cfg_name, elf.len());
                kernel_cache.insert(cfg_name.clone(), gpu_kernel);
            }
            let gpu_kernel = &kernel_cache[&cfg_name];

            // 3. Allocate buffers
            let x_bytes = (m as usize) * (k as usize) * 2;
            let w_bytes = (n as usize) * (k as usize) * 2;
            let sk = cfg.split_k.unwrap_or(1);
            let y_bytes = (m as usize) * (n as usize) * 4 * sk as usize;
            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            let y_buf = device.alloc_vram(y_bytes)?;

            // Fill with bf16(1.0) = 0x3F80
            let x_data = vec![0x3F80u16; (m * k) as usize];
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
            });
            let w_data = vec![0x3F80u16; (n * k) as usize];
            w_buf.write(unsafe {
                std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
            });

            // 4. Build kernargs using the helper
            let ka = build_kernargs(
                x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
                k, n, m, &cfg
            );
            let (grid_x, grid_y) = compute_grid_auto(&cfg, m, n);

            // Warmup
            for _ in 0..2 {
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                queue.wait_idle()?;
            }

            // Timed
            let iters = if m * k * n <= 512 * 512 * 512 { 20 } else { 10 };
            let start = std::time::Instant::now();
            for i in 0..iters {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit(gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                queue.wait_idle()?;
            }
            let elapsed = start.elapsed();
            let avg_us = elapsed.as_micros() as f64 / iters as f64;
            let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
            let tflops = flops / (avg_us * 1e6);

            // 5. Reduce split-K partials and verify correctness
            let y_elems = (m as usize) * (n as usize);
            let mut y_out = vec![0f32; y_elems];
            if sk > 1 {
                // Sum sk partial results
                for s in 0..sk as usize {
                    let offset = s * y_elems;
                    let partial = unsafe {
                        std::slice::from_raw_parts(
                            (y_buf.host_ptr as *const f32).add(offset), y_elems)
                    };
                    for i in 0..y_elems {
                        y_out[i] += partial[i];
                    }
                }
            } else {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        y_buf.host_ptr as *const f32, y_out.as_mut_ptr(), y_elems);
                }
            }
            // For bf16(1.0) × bf16(1.0), expected = K (each of K elements = 1.0*1.0)
            let expected = k as f32;
            let max_err = y_out.iter().map(|v| (v - expected).abs()).fold(0.0f32, f32::max);
            let status = if max_err < 1.0 { "✅" } else { &format!("❌ err={:.1}", max_err) };

            eprintln!("{:>6} {:>6} {:>6} | {:>25} | {:>6.1}μs | {:>6.2} TF | {}",
                      m, k, n, cfg_name, avg_us, tflops, status);
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        // CPU-only: just show config selection
        for (m, k, n) in &test_cases {
            let cfg = auto_select(*m, *k, *n);
            eprintln!("  {}×{}×{} → {}", m, k, n, cfg.name());
        }
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
