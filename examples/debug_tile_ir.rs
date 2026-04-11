//! tile_ir vs gemm_gen assembly 对比 + GPU 最小测试
//!
//! 用于调试 tile_ir::lower_gemm() GPU hang
//! 
//! 策略：
//! 1. 对比 tile_ir 和 gemm_gen 对同一尺寸生成的 assembly
//! 2. 用最小尺寸单 tile 测试 GPU 执行
//!
//! cargo run --example debug_tile_ir --features rocm --release

use t0_gpu::t0::{Target, T0Kernel};
use t0_gpu::t0::tile_ir::{TileGemm, lower_gemm};
use t0_gpu::t0::gemm_gen;

fn main() -> Result<(), String> {
    eprintln!("══ tile_ir vs gemm_gen 对比调试 ══\n");

    // ─── 1. 生成同配置的两个内核 ───
    let spec = TileGemm::tile_128x64_k16();
    eprintln!("[tile_ir] spec: {} (wg={}, lds={})", spec.name(), spec.wg_size(), spec.lds_total());

    let tile_kernel = lower_gemm(&spec);
    let tile_asm = tile_kernel.to_assembly(Target::GFX1100)?;

    let gemm_cfg = gemm_gen::GemmConfig {
        tile_m: 128, tile_n: 64, tile_k: 16,
        wg_size: 128,
        use_lds: true,
        double_buffer: true,
        split_k: None,
        lds_pad: 0,
        n_col_passes: 1,
        swap_grid: true,
        wgp_mode: false,
        transpose: gemm_gen::GemmTranspose::NT,
        epilogue: Default::default(),
    };
    let mut gemm_kernel = gemm_gen::generate(&gemm_cfg);
    gemm_kernel.set_ssa_regalloc(false);
    let gemm_asm = gemm_kernel.to_assembly(Target::GFX1100)?;

    // ─── 2. 打印关键信息 ───
    eprintln!("\n[tile_ir] assembly: {} lines, {} bytes",
        tile_asm.lines().count(), tile_asm.len());
    eprintln!("[gemm_gen] assembly: {} lines, {} bytes",
        gemm_asm.lines().count(), gemm_asm.len());

    // 打印 tile_ir kernarg/LDS/VGPR 信息
    eprintln!("\n[tile_ir] kernel info:");
    eprintln!("  kernarg_size: {}", tile_kernel.kernarg_size());
    eprintln!("  lds_size: {}", tile_kernel.lds_size());
    eprintln!("  wg_size: {}", tile_kernel.wg_size());

    eprintln!("\n[gemm_gen] kernel info:");
    eprintln!("  kernarg_size: {}", gemm_kernel.kernarg_size());
    eprintln!("  lds_size: {}", gemm_kernel.lds_size());
    eprintln!("  wg_size: {}", gemm_kernel.wg_size());

    // 打印 kernarg 布局
    eprintln!("\n[tile_ir] args:");
    for a in tile_kernel.args() {
        eprintln!("  offset={:3}: {:?} '{}'", a.offset, a.kind, a.name);
    }
    eprintln!("\n[gemm_gen] args:");
    for a in gemm_kernel.args() {
        eprintln!("  offset={:3}: {:?} '{}'", a.offset, a.kind, a.name);
    }

    // 保存 assembly 到文件
    std::fs::write("/tmp/tile_ir_gemm.s", &tile_asm).unwrap();
    std::fs::write("/tmp/gemm_gen_gemm.s", &gemm_asm).unwrap();
    eprintln!("\n  Assembly saved to /tmp/tile_ir_gemm.s and /tmp/gemm_gen_gemm.s");
    eprintln!("  Run: diff /tmp/tile_ir_gemm.s /tmp/gemm_gen_gemm.s");

    // ─── 3. 编译 ELF ───
    let tile_elf = tile_kernel.compile(Target::GFX1100)?;
    let gemm_elf = gemm_kernel.compile(Target::GFX1100)?;
    eprintln!("\n[tile_ir] ELF: {} bytes", tile_elf.len());
    eprintln!("[gemm_gen] ELF: {} bytes", gemm_elf.len());

    // ─── 4. GPU 测试（最小尺寸：单 tile） ───
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        eprintln!("\n══ GPU 测试 ══\n");
        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 8)?;

        // 单 tile: M=128, K=16, N=64
        let m = 128u32;
        let k = 16u32;
        let n = 64u32;

        let x_bytes = (m * k * 2) as usize;   // BF16
        let w_bytes = (n * k * 2) as usize;
        let y_bytes = (m * n * 4) as usize;    // F32

        let x_buf = device.alloc_vram(x_bytes)?;
        let w_buf = device.alloc_vram(w_bytes)?;

        // Fill with 1.0 bf16 = 0x3F80 (wait, bf16 1.0 = 0x3F80)
        // Actually bf16 1.0 = 0x3F80
        let x_data = vec![0x3F80u16; (m * k) as usize];
        x_buf.write(unsafe {
            std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
        });
        let w_data = vec![0x3F80u16; (n * k) as usize];
        w_buf.write(unsafe {
            std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
        });

        // ── gemm_gen 先跑，验证基线 ──
        eprintln!("--- gemm_gen GPU test (M={}, K={}, N={}) ---", m, k, n);
        let y_gemm = device.alloc_vram(y_bytes)?;
        {
            let gk = GpuKernel::load(&device, &gemm_elf, &KernelLoadConfig {
                workgroup_size: [gemm_cfg.wg_size, 1, 1],
                lds_size: gemm_cfg.lds_total() as u32,
            })?;

            // Grid: single tile
            let (grid_x, grid_y) = gemm_gen::compute_grid(&gemm_cfg, m, n);
            eprintln!("  grid=({}, {}, 1), wg={}", grid_x, grid_y, gemm_cfg.wg_size);

            // kernargs: [X:u64, W:u64, Y:u64, K:u32, N:u32, sk_shift:u32, y_stride:u32]
            let mut ka = [0u8; 40];
            ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&y_gemm.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&k.to_le_bytes());
            ka[28..32].copy_from_slice(&n.to_le_bytes());
            ka[32..36].copy_from_slice(&0u32.to_le_bytes());
            ka[36..40].copy_from_slice(&0u32.to_le_bytes());

            let ka_buf = pool.write_kernargs(0, &ka);
            queue.submit(&gk, [grid_x, grid_y, 1], ka_buf);
            queue.wait_idle()?;

            // Read result
            let mut result = vec![0f32; (m * n) as usize];
            y_gemm.read(unsafe {
                std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, y_bytes)
            });
            eprintln!("  Y[0..8] = {:?}", &result[0..8]);
            eprintln!("  Expected: all = {} (K * 1.0 * 1.0)", k);
            eprintln!("  gemm_gen: ✅ OK");
        }

        // ── tile_ir 测试 ──
        eprintln!("\n--- tile_ir GPU test (M={}, K={}, N={}) ---", m, k, n);
        let y_tile = device.alloc_vram(y_bytes)?;

        // 使用 compile_with_info 获取实际 LDS 大小
        let (tile_elf2, final_lds) = tile_kernel.compile_with_info(Target::GFX1100)?;
        eprintln!("  lds_size(spec)={}, final_lds(with spill)={}", spec.lds_total(), final_lds);

        let gk = GpuKernel::load(&device, &tile_elf2, &KernelLoadConfig {
            workgroup_size: [spec.wg_size(), 1, 1],
            lds_size: final_lds,
        })?;

        // tile_ir grid: swap_grid=true → grid_x = n_tiles_n * wg_size, grid_y = n_tiles_m
        let n_tiles_m = (m + spec.tile_m - 1) / spec.tile_m;
        let n_tiles_n = (n + spec.tile_n - 1) / spec.tile_n;
        let (grid_x, grid_y) = if spec.swap_grid {
            (n_tiles_n * spec.wg_size(), n_tiles_m)
        } else {
            (n_tiles_m * spec.wg_size(), n_tiles_n)
        };
        eprintln!("  grid=({}, {}, 1), wg={}, swap_grid={}", grid_x, grid_y, spec.wg_size(), spec.swap_grid);
        eprintln!("  n_tiles_m={}, n_tiles_n={}", n_tiles_m, n_tiles_n);

        // kernargs: tile_ir 和 gemm_gen 布局应该一致
        let mut ka = [0u8; 40];
        ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&y_tile.gpu_addr().to_le_bytes());
        ka[24..28].copy_from_slice(&k.to_le_bytes());
        ka[28..32].copy_from_slice(&n.to_le_bytes());
        ka[32..36].copy_from_slice(&0u32.to_le_bytes());
        ka[36..40].copy_from_slice(&0u32.to_le_bytes());

        eprintln!("  Dispatching tile_ir...");
        let ka_buf = pool.write_kernargs(1, &ka);
        queue.submit(&gk, [grid_x, grid_y, 1], ka_buf);
        queue.wait_idle()?;

        let mut result = vec![0f32; (m * n) as usize];
        y_tile.read(unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, y_bytes)
        });
        eprintln!("  Y[0..8] = {:?}", &result[0..8]);
        eprintln!("  tile_ir: ✅ OK");
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
