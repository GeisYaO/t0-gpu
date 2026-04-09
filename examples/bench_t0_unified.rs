//! # bench_t0_unified — T0 GEMM 统一基准测试
//!
//! 测试 T0 编译器各编译链路的 GEMM 性能。
//! 
//! 当前链路:
//! - gemm_gen::generate() → T0Kernel → legacy regalloc → ELF (已验证可用)
//! - tile_ir::lower_gemm() → T0Kernel → regalloc → ELF (待 GPU 调试)
//!
//! 运行: cargo run --example bench_t0_unified --features rocm --release

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{self, GemmConfig, generate, compute_grid, compute_grid_split_k};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM 统一基准测试 — AMD RX 7900 XTX (GFX1100)         ║");
    eprintln!("║  BF16 WMMA GEMM (Y = X × W^T), F32 output                 ║");
    eprintln!("║  gemm_gen auto_select_legacy (validated pipeline)           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // README.md 中的 9 个标准矩阵尺寸
    let sizes: Vec<(u32, u32, u32)> = vec![
        (256,  256,  256),
        (512,  512,  512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (128,  1024, 4096),
        (256,  1024, 4096),
        (512,  1024, 4096),
        (1024, 1024, 4096),
    ];

    // 编译链路: gemm_gen (validated) + legacy regalloc (no spill)
    eprintln!("\n── 编译内核 (gemm_gen + ssa_regalloc=off) ──");
    let mut kernels_info: Vec<(GemmConfig, Vec<u8>)> = Vec::new();
    for &(m, k, n) in &sizes {
        let cfg = gemm_gen::auto_select_legacy(m, k, n);
        eprint!("  {}×{}×{} → {} ... ", m, k, n, cfg.name());
        let mut kernel = generate(&cfg);
        // gemm_gen 使用 skip_optimize=true + 手工指令序列,
        // SSA regalloc spill 会与 LDS double-buffer 冲突 → GPU hang.
        // 必须禁用使用 legacy linear scan.
        kernel.set_ssa_regalloc(false);
        let elf = kernel.compile(Target::GFX1100)?;
        eprintln!("✓ ({} bytes, sk={}, wgp={})",
            elf.len(), cfg.split_k.unwrap_or(1), cfg.wgp_mode);
        kernels_info.push((cfg, elf));
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        // Load all GPU kernels
        let mut gpu_kernels = Vec::new();
        for (cfg, elf) in &kernels_info {
            let gk = GpuKernel::load(&device, elf, &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?;
            gpu_kernels.push(gk);
        }

        // Results
        struct BenchResult {
            m: u32, k: u32, n: u32,
            tflops: f64,
            avg_us: f64,
            config_name: String,
        }
        let mut results: Vec<BenchResult> = Vec::new();

        // Header
        eprintln!("\n{:>20} | {:>20} | {:>10} | {:>10}",
            "Matrix", "Config", "TFLOPS", "μs");
        eprintln!("{}", "-".repeat(70));

        for (si, &(m, k, n)) in sizes.iter().enumerate() {
            let cfg = &kernels_info[si].0;
            let sk = cfg.split_k.unwrap_or(1);

            // Allocate GPU buffers
            let x_bytes = (m as usize) * (k as usize) * 2;   // BF16
            let w_bytes = (n as usize) * (k as usize) * 2;   // BF16
            let y_bytes = (m as usize) * (n as usize) * 4;   // F32

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            let y_workspace = device.alloc_vram(y_bytes * sk as usize)?;

            // Fill with 1.0 bf16 (0x3C00)
            let x_data = vec![0x3C00u16; (m * k) as usize];
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
            });
            let w_data = vec![0x3C00u16; (n * k) as usize];
            w_buf.write(unsafe {
                std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
            });

            // Compute grid
            let (grid_x, grid_y) = if sk > 1 {
                compute_grid_split_k(cfg, m, n, sk)
            } else {
                compute_grid(cfg, m, n)
            };

            // Build kernargs
            let y_stride = if sk > 1 { (m * n * 4) as u32 } else { 0 };
            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_workspace.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());
            ka[36..40].copy_from_slice(&y_stride.to_le_bytes());

            // Warmup (3 iterations)
            for _ in 0..3 {
                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernels[si], [grid_x, grid_y, 1], ka_buf);
                queue.wait_idle()?;
            }

            // Timed iterations
            let n_iters: usize = if (m as u64) * (n as u64) * (k as u64) <= 512 * 512 * 512 { 50 } else { 20 };
            let start = std::time::Instant::now();
            for i in 0..n_iters {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit(&gpu_kernels[si], [grid_x, grid_y, 1], ka_buf);
                queue.wait_idle()?;
            }
            let elapsed = start.elapsed();
            let avg_us = elapsed.as_micros() as f64 / n_iters as f64;
            let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
            let tflops = flops / (avg_us * 1e6);

            let config_name = cfg.name().replace("t0_gemm_", "");
            eprintln!("{:>6}×{:<6}×{:<4} | {:>20} | {:>8.2} TF | {:>8.1} μs",
                m, k, n, config_name, tflops, avg_us);

            results.push(BenchResult {
                m, k, n, tflops, avg_us, config_name,
            });

            queue.synchronize()?;
        }

        // Summary with Triton/rocBLAS comparison
        eprintln!("\n══ T0 vs Triton+rocBLAS 对比 ══");
        eprintln!("{:>20} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10}",
            "Matrix", "T0", "rocBLAS", "Triton", "Tri-AT", "T0/Best");
        eprintln!("{}", "-".repeat(80));

        let triton_baseline: Vec<((u32,u32,u32), f64, f64, f64)> = vec![
            ((256,256,256),    3.05, 1.41, 2.26),
            ((512,512,512),    13.29, 6.53, 17.63),
            ((1024,1024,1024), 51.96, 52.49, 54.21),
            ((2048,2048,2048), 71.46, 76.82, 76.33),
            ((4096,4096,4096), 90.78, 86.71, 77.30),
            ((128,1024,4096),  48.47, 27.22, 39.93),
            ((256,1024,4096),  51.47, 53.44, 56.17),
            ((512,1024,4096),  74.41, 76.97, 58.91),
            ((1024,1024,4096), 64.40, 75.57, 74.41),
        ];

        for r in &results {
            let (rb, tr, ta) = triton_baseline.iter()
                .find(|((rm,rk,rn), _, _, _)| *rm==r.m && *rk==r.k && *rn==r.n)
                .map(|(_, rb, tr, ta)| (*rb, *tr, *ta))
                .unwrap_or((0.0, 0.0, 0.0));

            let best_opponent = rb.max(tr).max(ta);
            let ratio = if best_opponent > 0.0 { r.tflops / best_opponent * 100.0 } else { 0.0 };
            let marker = if ratio > 100.0 { "🏆" } else { "  " };

            eprintln!("{:>6}×{:<6}×{:<4} | {:>6.2} TF | {:>6.2} TF | {:>6.2} TF | {:>6.2} TF | {:>5.1}% {}",
                r.m, r.k, r.n, r.tflops, rb, tr, ta, ratio, marker);
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        eprintln!("Error: 需要 --features rocm 才能运行 GPU 基准测试");
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
