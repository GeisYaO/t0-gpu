#[cfg(test)]
mod tests {
    use super::super::{
        AqlBarrierPacket,
        MemoryFlags,
        WslPm4CmdBuilder,
        WslAqlQueue,
        WslDxgDevice,
        CS_PARTIAL_FLUSH,
        HSA_PACKET_TYPE_BARRIER_AND,
        EVENT_INDEX_PARTIAL_FLUSH,
    };
    use crate::wsl_dxg::D3DKMT_HANDLE;
    use std::sync::Arc;
    use std::ptr;

    fn dxg_risky_tests_enabled() -> bool {
        matches!(
            std::env::var("T0_DXG_ENABLE_RISKY_TESTS").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    }

    fn enqueue_barrier_packet(queue: &WslAqlQueue) -> Result<u64, String> {
        queue.ensure_ring_space()?;

        let write_idx = unsafe { ptr::read_volatile(queue.write_ptr_host) };
        let ring_mask = (queue.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;
        let packet = AqlBarrierPacket {
            header: HSA_PACKET_TYPE_BARRIER_AND,
            ..Default::default()
        };

        unsafe {
            let base = queue.ring_buffer.cpu_ptr.add(pkt_offset) as *mut AqlBarrierPacket;
            ptr::write_volatile(base, packet);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            ptr::write_volatile(queue.write_ptr_host, write_idx + 1);
        }

        queue.notify_worker();
        Ok(write_idx + 1)
    }

    #[test]
    #[ignore] // Skip by default, run manually with: cargo test --features wsl_dxg -- --ignored
    fn test_device_open() {
        let device = WslDxgDevice::open()
            .expect("Failed to open DXG device");
        println!("Device opened: vendor_id=0x{:X} vram_size={}MB",
            device.vendor_id, device.vram_size / 1024 / 1024);
    }

    #[test]
    #[ignore]
    fn test_memory_allocations() {
        let device = WslDxgDevice::open().expect("Failed to open DXG device");

        // Test VRAM allocation
        let vram = match device.alloc_memory(4096, MemoryFlags {
            vram: true,
            ..Default::default()
        }) {
            Ok(buf) => {
                println!("Allocated VRAM: {} bytes", buf.size);
                buf
            }
            Err(e) => {
                println!("VRAM allocation not yet implemented: {}", e);
                return;
            }
        };

        // Test GART allocation
        let gart = match device.alloc_memory(4096, MemoryFlags {
            gart: true,
            ..Default::default()
        }) {
            Ok(buf) => {
                println!("Allocated GART: {} bytes", buf.size);
                buf
            }
            Err(e) => {
                println!("GART allocation not yet implemented: {}", e);
                return;
            }
        };

        // Test writing/reading
        let test_data = b"Hello WSL2 DXG!";
        vram.write(test_data);
        let mut read_buf = vec![0u8; test_data.len()];
        vram.read(&mut read_buf);
        assert_eq!(&read_buf, test_data);

        println!("Memory tests passed!");
    }

    #[test]
    #[ignore]
    fn test_sync_object() {
        let device = WslDxgDevice::open().expect("Failed to open DXG device");

        let sync: D3DKMT_HANDLE = device.create_sync_object().expect("Failed to create sync object");
        println!("Sync object created: 0x{:X}", sync as usize);
    }

    #[test]
    #[ignore]
    fn test_wsl_dxg_barrier_packet_smoke() {
        if !dxg_risky_tests_enabled() {
            eprintln!("[DXG][test] skip barrier_packet_smoke: set T0_DXG_ENABLE_RISKY_TESTS=1 to run");
            return;
        }
        let device = WslDxgDevice::open().expect("Failed to open DXG device");
        let queue = device.create_queue().expect("Failed to create DXG queue");
        let target = enqueue_barrier_packet(&queue).expect("Failed to enqueue barrier packet");
        queue
            .wait_read_ptr(target)
            .expect("Barrier packet did not retire");
        println!("Barrier packet retired: read_ptr target={}", target);
    }

    #[test]
    #[ignore]
    fn test_wsl_dxg_gpu_signal_only_smoke() {
        let device = WslDxgDevice::open().expect("Failed to open DXG device");
        let (sync_object, sync_cpu_va) = device
            .create_monitored_fence()
            .expect("Failed to create monitored fence");

        device
            .signal_sync_object_from_gpu(sync_object, 1)
            .expect("Failed to GPU-signal monitored fence");

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5);
        loop {
            let completed = unsafe { ptr::read_volatile(sync_cpu_va) };
            if completed == 1 {
                println!("GPU signal-only smoke retired: fence={}", completed);
                device.destroy_sync_object(sync_object);
                return;
            }
            if completed == u64::MAX {
                let state = device.describe_device_state();
                device.destroy_sync_object(sync_object);
                panic!("GPU signal-only smoke retired with invalid fence value UINT64_MAX: {}", state);
            }
            if start.elapsed() > timeout {
                let state = device.describe_device_state();
                device.destroy_sync_object(sync_object);
                panic!("GPU signal-only smoke did not retire: {}", state);
            }
            std::hint::spin_loop();
        }
    }

    #[test]
    #[ignore]
    fn test_wsl_dxg_event_write_submit_smoke() {
        if !dxg_risky_tests_enabled() {
            eprintln!("[DXG][test] skip event_write_submit_smoke: set T0_DXG_ENABLE_RISKY_TESTS=1 to run");
            return;
        }
        let device = WslDxgDevice::open().expect("Failed to open DXG device");
        let (sync_object, sync_cpu_va) = device
            .create_monitored_fence()
            .expect("Failed to create monitored fence");
        let cmd = device
            .alloc_system(4096)
            .expect("Failed to allocate command buffer");

        let mut pm4 = WslPm4CmdBuilder::new();
        pm4.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH);
        let cmds = pm4.finish();

        unsafe {
            let dst = cmd.cpu_ptr as *mut u32;
            for (idx, dword) in cmds.iter().enumerate() {
                ptr::write_volatile(dst.add(idx), *dword);
            }
        }

        device
            .submit_command(cmd.gpu_va, (cmds.len() * std::mem::size_of::<u32>()) as u32, sync_object, 1)
            .expect("Failed to submit event-write smoke command");

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5);
        loop {
            let completed = unsafe { ptr::read_volatile(sync_cpu_va) };
            if completed == 1 {
                println!("Event-write command retired: fence={}", completed);
                device.destroy_sync_object(sync_object);
                return;
            }
            if completed == u64::MAX {
                let state = device.describe_device_state();
                device.destroy_sync_object(sync_object);
                panic!("Event-write command retired with invalid fence value UINT64_MAX: {}", state);
            }
            if start.elapsed() > timeout {
                let state = device.describe_device_state();
                device.destroy_sync_object(sync_object);
                panic!("Event-write command did not retire: {}", state);
            }
            std::hint::spin_loop();
        }
    }
}
