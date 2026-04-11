//! KFD Dispatch Latency Benchmark — Final
//!
//! Run: cargo run --example bench_kfd_dispatch --features rocm --release

use t0_gpu::t0::{T0Kernel, Target};

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════╗");
    eprintln!("║  KFD Dispatch Latency Benchmark                 ║");
    eprintln!("╚══════════════════════════════════════════════════╝");

    // Compile empty kernel using T0
    eprintln!("\nCompiling empty kernel...");
    let mut k = T0Kernel::new("empty_kernel");
    let _n = k.arg_u32("dummy");
    k.emit_arg_loads();
    k.endpgm();
    let elf = k.compile(Target::GFX1100)?;
    eprintln!("  ✓ {} bytes ELF", elf.len());

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 16)?;

        let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [32, 1, 1],
            lds_size: 0,
        })?;

        let ka = [0u8; 4];

        // ── Warmup ──
        for _ in 0..200 {
            let ka_buf = pool.write_kernargs(0, &ka);
            queue.submit_fast(&gpu_kernel, [32, 1, 1], ka_buf);
            queue.wait_idle_spin();
        }

        // ── Test 1: Standard sync (submit + wait_idle) ──
        eprintln!("\n[1] Standard sync (submit + wait_idle, 1000 iter):");
        {
            let n = 1000;
            let start = std::time::Instant::now();
            for i in 0..n {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit(&gpu_kernel, [32, 1, 1], ka_buf);
                queue.wait_idle()?;
            }
            let us = start.elapsed().as_nanos() as f64 / 1000.0;
            eprintln!("  Per dispatch: {:.1} μs", us / n as f64);
        }

        // ── Test 2: Fast sync (submit_fast + wait_idle_spin) ──
        eprintln!("\n[2] Fast sync (submit_fast + wait_idle_spin, 10000 iter):");
        {
            let n = 10000;
            let start = std::time::Instant::now();
            for i in 0..n {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit_fast(&gpu_kernel, [32, 1, 1], ka_buf);
                queue.wait_idle_spin();
            }
            let us = start.elapsed().as_nanos() as f64 / 1000.0;
            eprintln!("  Per dispatch: {:.2} μs", us / n as f64);
        }

        // ── Test 3: Standard async batch ──
        eprintln!("\n[3] Standard async batch (10000 dispatches → 1 sync):");
        {
            let n = 10000;
            let start = std::time::Instant::now();
            for i in 0..n {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit(&gpu_kernel, [32, 1, 1], ka_buf);
            }
            queue.wait_idle()?;
            let us = start.elapsed().as_nanos() as f64 / 1000.0;
            eprintln!("  Per dispatch (amortized): {:.2} μs", us / n as f64);
        }

        // ── Test 4: Fast async batch ──
        eprintln!("\n[4] Fast async batch (10000 dispatches → 1 spin):");
        {
            let n = 10000;
            let start = std::time::Instant::now();
            for i in 0..n {
                let ka_buf = pool.write_kernargs(i % 16, &ka);
                queue.submit_fast(&gpu_kernel, [32, 1, 1], ka_buf);
            }
            queue.wait_idle_spin();
            let us = start.elapsed().as_nanos() as f64 / 1000.0;
            eprintln!("  Per dispatch (amortized): {:.2} μs", us / n as f64);
        }

        // ── Test 5: Ultra-fast — pre-write kernargs, submit_fast only ──
        eprintln!("\n[5] Ultra-fast async (pre-written kernargs, 10000 dispatches):");
        {
            // Pre-write all kernargs so submit loop is pure AQL writes
            for i in 0..16 {
                pool.write_kernargs(i, &ka);
            }
            let n = 10000;
            let start = std::time::Instant::now();
            for i in 0..n {
                let ka_buf = pool.get_kernargs(i % 16);
                queue.submit_fast(&gpu_kernel, [32, 1, 1], ka_buf);
            }
            queue.wait_idle_spin();
            let us = start.elapsed().as_nanos() as f64 / 1000.0;
            eprintln!("  Per dispatch (amortized): {:.2} μs", us / n as f64);
        }

        // ── Test 6: Raw AQL packet submission (inline everything) ──
        eprintln!("\n[6] Raw inline AQL batch (10000 dispatches → 1 spin):");
        {
            // Pre-write kernargs
            let ka_buf = pool.write_kernargs(0, &ka);
            let ka_addr = ka_buf.gpu_addr();
            let desc_va = gpu_kernel.descriptor_va;
            let lds = gpu_kernel.lds_size;
            let ring_mask = (queue.ring_size as u64 / 64) - 1;

            let n = 10000u64;
            let start = std::time::Instant::now();

            for _ in 0..n {
                unsafe {
                    let write_idx = std::ptr::read_volatile(queue.write_ptr_host);
                    let slot_idx = write_idx & ring_mask;
                    let base = queue.ring_buffer.host_ptr.add((slot_idx * 64) as usize);

                    // Write AQL packet body
                    std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
                    std::ptr::write_volatile(base.add(0x04) as *mut u16, 32u16); // wg_x
                    std::ptr::write_volatile(base.add(0x06) as *mut u16, 1u16);  // wg_y
                    std::ptr::write_volatile(base.add(0x08) as *mut u16, 1u16);  // wg_z
                    std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
                    std::ptr::write_volatile(base.add(0x0C) as *mut u32, 32u32); // grid_x
                    std::ptr::write_volatile(base.add(0x10) as *mut u32, 1u32);  // grid_y
                    std::ptr::write_volatile(base.add(0x14) as *mut u32, 1u32);  // grid_z
                    std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32);
                    std::ptr::write_volatile(base.add(0x1C) as *mut u32, lds);
                    std::ptr::write_volatile(base.add(0x20) as *mut u64, desc_va);
                    std::ptr::write_volatile(base.add(0x28) as *mut u64, ka_addr);
                    std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64);
                    std::ptr::write_volatile(base.add(0x38) as *mut u64, 0u64);

                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

                    // Header activates packet
                    let header: u16 = 2 | (1 << 8) | (3 << 9) | (3 << 11);
                    std::ptr::write_volatile(base as *mut u16, header);

                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

                    let nw = write_idx + 1;
                    std::ptr::write_volatile(queue.write_ptr_host, nw);
                    std::ptr::write_volatile(queue.doorbell_ptr, nw - 1);
                }
            }

            // Wait
            let target = unsafe { std::ptr::read_volatile(queue.write_ptr_host) };
            loop {
                let r = unsafe { std::ptr::read_volatile(queue.read_ptr_host) };
                if r >= target { break; }
                std::hint::spin_loop();
            }

            let us = start.elapsed().as_nanos() as f64 / 1000.0;
            eprintln!("  Per dispatch (amortized): {:.2} μs", us / n as f64);
        }
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
