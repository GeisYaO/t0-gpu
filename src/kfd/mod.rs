//! RDNA3 Bare-Metal KFD Runtime
//!
//! Direct GPU control via /dev/kfd ioctl, bypassing HIP/ROCm entirely.
//! Implements: VRAM allocation, AQL queue dispatch, doorbell ring, completion polling.
//!
//! Target: AMD RX 7900 XTX (GFX1100, RDNA3), KFD v1.14, Linux 6.17
//!
//! Architecture:
//!   /dev/kfd  → KFD ioctl (memory, queues, events)
//!   /dev/dri/renderD128 → DRM fd for acquire_vm
//!   AQL ring buffer → 64-byte dispatch packets → doorbell → GPU execution
//!
//! Reference: kfd_bare_metal_dispatch_architecture.md

use std::os::unix::io::RawFd;
use std::sync::Arc;

// =============================================================================
// SIGPIPE defense — prevent pipe-broken signals from killing the process
// =============================================================================

/// Ignore SIGPIPE so that `cargo test ... | head -N` doesn't kill the process
/// before Drop handlers can run (DESTROY_QUEUE, close fd, etc.).
/// Without this, piped commands leave GPU queues active with buggy kernels,
/// triggering KFD MODE1 reset and causing the next run to hard-hang.
fn ignore_sigpipe() {
    // SIG_IGN = 1, SIGPIPE = 13 on Linux x86_64
    extern "C" { fn signal(sig: i32, handler: usize) -> usize; }
    unsafe { signal(13, 1); }
}

// =============================================================================
// KFD IOCTL numbers (Linux _IOC encoding, type='K'=0x4B)
// =============================================================================

const AMDKFD_IOC_GET_VERSION: u64       = 0x80084B01;
const AMDKFD_IOC_CREATE_QUEUE: u64      = 0xC0604B02; // sizeof=96 (matches tinygrad)
const AMDKFD_IOC_DESTROY_QUEUE: u64     = 0xC0084B03;
const AMDKFD_IOC_ACQUIRE_VM: u64        = 0x40084B15;
const AMDKFD_IOC_ALLOC_MEMORY: u64      = 0xC0284B16;
const AMDKFD_IOC_FREE_MEMORY: u64       = 0x40084B17;
const AMDKFD_IOC_MAP_MEMORY: u64        = 0xC0184B18;
const AMDKFD_IOC_UNMAP_MEMORY: u64      = 0xC0184B19;
const AMDKFD_IOC_CREATE_EVENT: u64      = 0xC02C4B08;
const AMDKFD_IOC_WAIT_EVENTS: u64       = 0xC0204B0B;
const AMDKFD_IOC_RUNTIME_ENABLE: u64    = 0xC0104B25; // sizeof=16

// Memory allocation flags
const KFD_IOC_ALLOC_MEM_FLAGS_VRAM: u32       = 1 << 0;
const KFD_IOC_ALLOC_MEM_FLAGS_GTT: u32        = 1 << 1;
const KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE: u32   = 1 << 31;
const KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;
const KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC: u32     = 1 << 29;
const KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE: u32 = 1 << 28;
const KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM: u32 = 1 << 27;
const KFD_IOC_ALLOC_MEM_FLAGS_COHERENT: u32   = 1 << 26;
const KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED: u32   = 1 << 25;

// Queue types
const KFD_IOC_QUEUE_TYPE_COMPUTE: u32 = 0x0;     // PM4 compute queue
const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL: u32 = 0x2;  // AQL compute queue

// AQL packet types
const HSA_PACKET_TYPE_VENDOR_SPECIFIC: u16 = 0x0;
const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 0x2;

// Fence scopes
#[allow(dead_code)]
const HSA_FENCE_SCOPE_AGENT: u16 = 1;   // GPU-internal only (no PCIe sync)
const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;  // Full system coherency (PCIe writeback)

// PM4-in-AQL constants
const PACKET3_INDIRECT_BUFFER: u32 = 0x3F;
const INDIRECT_BUFFER_VALID: u32 = 1 << 23;
const PM4_IB_SIZE: usize = 256 * 1024; // 256KB for PM4 indirect buffers

// PM4 opcodes for compute dispatch
const PM4_SET_SH_REG: u32         = 0x76;
const PM4_DISPATCH_DIRECT: u32    = 0x15;
const PM4_RELEASE_MEM: u32        = 0x49;
const PM4_ACQUIRE_MEM: u32        = 0x58;
const PM4_EVENT_WRITE: u32        = 0x46;

// GFX11 Compute SH register offsets
const SH_REG_BASE: u32              = 0x2C00;
const REG_COMPUTE_PGM_LO: u32       = 0x2C0C;
const REG_COMPUTE_PGM_RSRC1: u32    = 0x2C44;
const REG_COMPUTE_PGM_RSRC3: u32    = 0x2C94;
const REG_COMPUTE_USER_DATA_0: u32  = 0x2C4C;
const REG_COMPUTE_NUM_THREAD_X: u32 = 0x2C78;
const REG_COMPUTE_RESOURCE_LIMITS: u32 = 0x2C14;
const REG_COMPUTE_START_X: u32      = 0x2C98;
const REG_COMPUTE_TMPRING_SIZE: u32 = 0x2C18;
const REG_COMPUTE_RESTART_X: u32    = 0x2C88;

// GFX11 event types
const CS_PARTIAL_FLUSH: u32     = 0x07;
const EVENT_INDEX_PARTIAL_FLUSH: u32 = 4;
const CACHE_FLUSH_AND_INV_TS_EVENT: u32 = 0x14;

// mmap constants
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_FIXED: i32 = 0x10;
const MAP_NORESERVE: i32 = 0x4000;
const MAP_FAILED: *mut u8 = usize::MAX as *mut u8;

// =============================================================================
// KFD IOCTL structs (must match kernel's kfd_ioctl.h exactly)
// =============================================================================

#[repr(C)]
#[derive(Default, Debug)]
struct KfdGetVersionArgs {
    major_version: u32,
    minor_version: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdAcquireVmArgs {
    drm_fd: u32,
    gpu_id: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdAllocMemoryArgs {
    va_addr: u64,
    size: u64,
    handle: u64,
    mmap_offset: u64,
    gpu_id: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdFreeMemoryArgs {
    handle: u64,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdMapMemoryArgs {
    handle: u64,
    device_ids_array_ptr: u64,
    n_devices: u32,
    n_success: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdUnmapMemoryArgs {
    handle: u64,
    device_ids_array_ptr: u64,
    n_devices: u32,
    n_success: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdCreateQueueArgs {
    ring_base_address: u64,
    write_pointer_address: u64,
    read_pointer_address: u64,
    doorbell_offset: u64,
    ring_size: u32,
    gpu_id: u32,
    queue_type: u32,
    queue_percentage: u32,
    queue_priority: u32,
    queue_id: u32,
    eop_buffer_address: u64,
    eop_buffer_size: u64,
    ctx_save_restore_address: u64,
    ctx_save_restore_size: u32,
    ctl_stack_size: u32,
    sdma_engine_id: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdDestroyQueueArgs {
    queue_id: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
struct KfdRuntimeEnableArgs {
    r_debug: u64,
    mode_mask: u32,
    capabilities_mask: u32,
}

// =============================================================================
// AQL Dispatch Packet (64 bytes, hardware format)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AqlDispatchPacket {
    pub header: u16,               // 0x00: type + barrier + fences (write LAST)
    pub setup: u16,                // 0x02: dimensions
    pub workgroup_size_x: u16,     // 0x04
    pub workgroup_size_y: u16,     // 0x06
    pub workgroup_size_z: u16,     // 0x08
    pub reserved0: u16,            // 0x0A: must be 0
    pub grid_size_x: u32,          // 0x0C
    pub grid_size_y: u32,          // 0x10
    pub grid_size_z: u32,          // 0x14
    pub private_segment_size: u32, // 0x18
    pub group_segment_size: u32,   // 0x1C: LDS size
    pub kernel_object: u64,        // 0x20: VA of kernel descriptor (NOT .text!)
    pub kernarg_address: u64,      // 0x28: VA of kernel arguments
    pub reserved2: u64,            // 0x30: must be 0
    pub completion_signal: u64,    // 0x38: VA of u64 for completion (0 = no signal)
}

const _: () = assert!(std::mem::size_of::<AqlDispatchPacket>() == 64);

// =============================================================================
// libc FFI
// =============================================================================

extern "C" {
    fn open(pathname: *const u8, flags: i32) -> i32;
    fn close(fd: i32) -> i32;
    fn mmap(addr: *mut u8, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, length: usize) -> i32;
    // Use syscall to avoid variadic ioctl() issues
    fn syscall(number: i64, ...) -> i64;
}

const SYS_IOCTL: i64 = 16; // x86-64: __NR_ioctl = 16

fn ioctl_safe(fd: i32, request: u64, arg: *mut u8) -> Result<(), String> {
    let ret = unsafe { syscall(SYS_IOCTL, fd as i64, request as i64, arg as i64) };
    if ret < 0 {
        let errno = std::io::Error::last_os_error();
        Err(format!("ioctl 0x{:X} failed: {} (ret={})", request, errno, ret))
    } else {
        Ok(())
    }
}

// =============================================================================
// KfdDevice — bare-metal GPU device handle
// =============================================================================

/// Bare-metal AMD GPU device accessed via /dev/kfd
pub struct KfdDevice {
    pub kfd_fd: RawFd,
    drm_fd: RawFd,
    pub gpu_id: u32,
    /// Base VA for user allocations (auto-incremented, 2MB aligned)
    next_va: std::sync::atomic::AtomicU64,
    /// Event page handle for cleanup (allocated during open)
    event_page_handle: u64,
    /// Event page VA for munmap during cleanup
    event_page_va: u64,
}

/// Global singleton: KFD driver only allows one ACQUIRE_VM per process.
/// All callers of KfdDevice::open() get the same Arc<KfdDevice>.
static GLOBAL_KFD_DEVICE: std::sync::OnceLock<Arc<KfdDevice>> = std::sync::OnceLock::new();

impl KfdDevice {
    pub fn target(&self) -> crate::t0::ir::Target {
        crate::t0::ir::Target::GFX1100
    }

    /// Open the GPU device and acquire VM.
    /// Returns a cached global singleton — KFD only allows one ACQUIRE_VM per process.
    pub fn open() -> Result<Arc<Self>, String> {
        Self::open_with_gpu_id(0) // auto-detect
    }

    /// Force a fresh device open (bypasses singleton cache).
    /// Use only when you know the previous device has been fully dropped.
    pub fn open_fresh() -> Result<Arc<Self>, String> {
        Self::open_fresh_with_gpu_id(0)
    }

    fn open_fresh_with_gpu_id(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        Self::open_device_impl(gpu_id_override)
    }

    pub fn open_with_gpu_id(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        // Return cached singleton if it exists (KFD only allows one ACQUIRE_VM per process)
        if let Some(dev) = GLOBAL_KFD_DEVICE.get() {
            return Ok(Arc::clone(dev));
        }
        let dev = Self::open_device_impl(gpu_id_override)?;
        // Try to store into OnceLock; if another thread raced us, use theirs
        match GLOBAL_KFD_DEVICE.set(Arc::clone(&dev)) {
            Ok(()) => Ok(dev),
            Err(_) => Ok(Arc::clone(GLOBAL_KFD_DEVICE.get().unwrap())),
        }
    }

    /// Internal: actual device open + VM acquire (called once per process)
    fn open_device_impl(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        // CRITICAL: ignore SIGPIPE before any GPU work.
        // When running under `cargo test ... | grep ... | head -N`, the `head`
        // command closes its stdin after reading enough lines, sending SIGPIPE
        // up the pipe chain. Without this, our process dies instantly and
        // Drop handlers (DESTROY_QUEUE, close fd) never run, leaving the GPU
        // executing buggy kernels → KFD MODE1 reset → next run hard-hangs.
        ignore_sigpipe();

        // Open /dev/kfd (with retry for GPU recovery after MODE1 reset)
        let kfd_fd = Self::open_kfd_with_retry()?;

        // Get KFD version
        let mut ver = KfdGetVersionArgs::default();
        ioctl_safe(kfd_fd, AMDKFD_IOC_GET_VERSION, &mut ver as *mut _ as *mut u8)?;
        eprintln!("[KFD] Version {}.{}", ver.major_version, ver.minor_version);

        // Determine gpu_id from topology
        let gpu_id = if gpu_id_override != 0 {
            gpu_id_override
        } else {
            Self::detect_gpu_id()?
        };
        eprintln!("[KFD] GPU ID: {}", gpu_id);

        // Open /dev/dri/renderDXXX — detect correct minor from KFD topology
        let render_minor = Self::detect_drm_render_minor(gpu_id).unwrap_or(128);
        let drm_path = format!("/dev/dri/renderD{}\0", render_minor);
        let drm_fd = unsafe { open(drm_path.as_ptr(), 2) };
        if drm_fd < 0 {
            unsafe { close(kfd_fd); }
            return Err(format!("Failed to open /dev/dri/renderD{}: {}", render_minor, std::io::Error::last_os_error()));
        }
        eprintln!("[KFD] Using /dev/dri/renderD{}", render_minor);

        // Acquire VM (bind DRM fd to KFD for this gpu)
        let mut acq = KfdAcquireVmArgs { drm_fd: drm_fd as u32, gpu_id };
        ioctl_safe(kfd_fd, AMDKFD_IOC_ACQUIRE_VM, &mut acq as *mut _ as *mut u8)?;
        eprintln!("[KFD] VM acquired");

        // RUNTIME_ENABLE - required on KFD >= 1.14 to activate AQL dispatch
        // Without this, doorbell writes for AQL queues are not processed by CP/MEC
        if ver.minor_version >= 14 {
            let mut rt = KfdRuntimeEnableArgs::default();
            ioctl_safe(kfd_fd, AMDKFD_IOC_RUNTIME_ENABLE, &mut rt as *mut _ as *mut u8)?;
            eprintln!("[KFD] Runtime enabled (caps=0x{:X})", rt.capabilities_mask);
        }

        // Event page + CREATE_EVENT — tinygrad does this before creating queues
        // This sets up the KFD event infrastructure required for MEC processing
        let event_page = {
            // Allocate a small uncached buffer for event page
            let va = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    0x8000, // 32KB event page
                    0, // PROT_NONE (reserve only)
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if va == MAP_FAILED || va.is_null() {
                return Err("Failed to reserve VA for event page".to_string());
            }
            let va_addr = va as u64;

            let mut alloc_args = KfdAllocMemoryArgs {
                va_addr,
                size: 0x8000,
                handle: 0,
                mmap_offset: 0,
                gpu_id,
                flags: KFD_IOC_ALLOC_MEM_FLAGS_GTT
                    | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                    | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
                    | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
                    | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
            };
            ioctl_safe(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY, &mut alloc_args as *mut _ as *mut u8)
                .map_err(|e| format!("Event page ALLOC_MEMORY failed: {}", e))?;

            // Map to GPU
            let mut gpu_ids = [gpu_id];
            let mut map_args = KfdMapMemoryArgs {
                handle: alloc_args.handle,
                device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
                n_devices: 1,
                n_success: 0,
            };
            ioctl_safe(kfd_fd, AMDKFD_IOC_MAP_MEMORY, &mut map_args as *mut _ as *mut u8)
                .map_err(|e| format!("Event page MAP_MEMORY failed: {}", e))?;

            // mmap to CPU
            let host_ptr = unsafe {
                mmap(
                    va,
                    0x8000,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED,
                    drm_fd,
                    alloc_args.mmap_offset as i64,
                )
            };
            if host_ptr == MAP_FAILED {
                return Err("Event page CPU mmap failed".to_string());
            }
            // Zero it
            unsafe { std::ptr::write_bytes(host_ptr, 0, 0x8000); }

            alloc_args.handle
        };

        // CREATE_EVENT with event_page_offset = handle
        // This initializes the KFD event page in the kernel
        #[repr(C)]
        #[derive(Default)]
        struct KfdCreateEventArgs {
            event_page_offset: u64,
            event_trigger_data: u32,
            event_type: u32,
            auto_reset: u32,
            node_id: u32,
            event_id: u32,
            event_slot_index: u32,
        }
        let mut ev = KfdCreateEventArgs {
            event_page_offset: event_page,
            ..Default::default()
        };
        ioctl_safe(kfd_fd, AMDKFD_IOC_CREATE_EVENT, &mut ev as *mut _ as *mut u8)
            .map_err(|e| format!("CREATE_EVENT failed: {}", e))?;
        eprintln!("[KFD] Event page created (event_id={}, slot={})", ev.event_id, ev.event_slot_index);

        let device = Arc::new(Self {
            kfd_fd,
            drm_fd,
            gpu_id,
            next_va: std::sync::atomic::AtomicU64::new(0x1_0000_0000),
            event_page_handle: event_page,
            event_page_va: 0, // VA is managed by kernel, not tracked for munmap
        });

        // GPU health probe: allocate a small buffer, write, read-back, verify.
        // If the GPU just recovered from MODE1 reset, VRAM may be unstable.
        // This catches it early instead of hanging on the first kernel dispatch.
        Self::gpu_health_probe(&device)?;

        Ok(device)
    }

    /// Open /dev/kfd with retry logic for GPU recovery after MODE1 reset.
    /// KFD driver may temporarily refuse open() while GPU is resetting.
    fn open_kfd_with_retry() -> Result<RawFd, String> {
        for attempt in 0..5 {
            let fd = unsafe { open(b"/dev/kfd\0".as_ptr(), 2 /* O_RDWR */) };
            if fd >= 0 {
                return Ok(fd);
            }
            let err = std::io::Error::last_os_error();
            if attempt < 4 {
                eprintln!("[KFD] /dev/kfd open failed (attempt {}): {} — GPU may be recovering from reset, retrying in 1s...",
                    attempt + 1, err);
                std::thread::sleep(std::time::Duration::from_secs(1));
            } else {
                return Err(format!("Failed to open /dev/kfd after 5 attempts: {}", err));
            }
        }
        unreachable!()
    }

    /// GPU health probe: allocate tiny GTT buffer, write pattern, read back.
    /// Catches post-MODE1-reset VRAM instability before any real work.
    fn gpu_health_probe(device: &Arc<Self>) -> Result<(), String> {
        for attempt in 0..3 {
            match Self::health_probe_once(device) {
                Ok(()) => {
                    eprintln!("[KFD] GPU health check PASSED");
                    return Ok(());
                }
                Err(e) => {
                    if attempt < 2 {
                        eprintln!("[KFD] GPU health check FAILED (attempt {}): {} — retrying in 2s...",
                            attempt + 1, e);
                        std::thread::sleep(std::time::Duration::from_secs(2));
                    } else {
                        return Err(format!("GPU health check failed after 3 attempts: {}", e));
                    }
                }
            }
        }
        unreachable!()
    }

    fn health_probe_once(device: &Arc<Self>) -> Result<(), String> {
        // Allocate a small GTT buffer (coherent, visible to both CPU and GPU)
        let probe_buf = device.alloc_gtt(4096)?;
        // Write a known pattern
        let pattern: u64 = 0xDEAD_BEEF_CAFE_F00D;
        probe_buf.write_val::<u64>(0, pattern);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // Read it back
        let readback: u64 = probe_buf.read_val(0);
        if readback != pattern {
            return Err(format!("GPU memory readback mismatch: wrote 0x{:X}, read 0x{:X}", pattern, readback));
        }
        // Buffer dropped here → RAII cleanup
        Ok(())
    }

    fn detect_gpu_id() -> Result<u32, String> {
        // Read gpu_id from sysfs topology
        // Try node 1 first (node 0 is usually CPU)
        for node in 1..=8 {
            let path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
            if let Ok(content) = std::fs::read_to_string(&path) {
                let id: u32 = content.trim().parse().unwrap_or(0);
                if id > 0 {
                    return Ok(id);
                }
            }
        }
        Err("No GPU found in KFD topology".to_string())
    }

    /// Detect DRM render minor for a given GPU ID from KFD topology
    fn detect_drm_render_minor(gpu_id: u32) -> Result<u32, String> {
        for node in 0..=8 {
            let gpu_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
            if let Ok(content) = std::fs::read_to_string(&gpu_path) {
                let id: u32 = content.trim().parse().unwrap_or(0);
                if id == gpu_id {
                    let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
                    if let Ok(props) = std::fs::read_to_string(&prop_path) {
                        for line in props.lines() {
                            if line.starts_with("drm_render_minor") {
                                if let Some(val) = line.split_whitespace().nth(1) {
                                    return val.parse::<u32>().map_err(|e| e.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        Err("drm_render_minor not found".to_string())
    }

    /// Pre-reserve a VA range via mmap(PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS)
    /// KFD requires the VA to be pre-reserved before ALLOC_MEMORY can use it.
    fn alloc_va(&self, size: usize) -> u64 {
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                0, // PROT_NONE
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                -1, // no fd
                0,
            )
        };
        if ptr == MAP_FAILED || ptr.is_null() {
            panic!("Failed to reserve VA space: {}", std::io::Error::last_os_error());
        }
        ptr as u64
    }

    /// Allocate VRAM buffer (writable, public, CPU-visible via mmap)
    pub fn alloc_vram(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)
    }

    /// Allocate executable VRAM (for kernel machine code)
    /// After writing code, call `hdp_flush()` or read back one byte to flush HDP.
    pub fn alloc_code(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)
    }

    /// Allocate GTT memory (host-visible, for kernargs, signals, etc.)
    pub fn alloc_gtt(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_COHERENT)
    }

    /// Allocate uncached GTT memory (for ring buffer, wr/rd ptrs, signals, kernargs).
    /// Keep this non-executable: EXECUTABLE GTT mappings can trigger CPF permission faults.
    pub fn alloc_uncached(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | // Required: CP fetches ring buffer as instructions
            KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC |
            KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
            KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
            KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    }

    /// Internal: allocate GPU memory via KFD ioctl
    fn alloc_memory(self: &Arc<Self>, size: usize, flags: u32) -> Result<GpuBuffer, String> {
        let page_size = 4096usize;
        let aligned_size = ((size + page_size - 1) / page_size) * page_size;
        let va_addr = self.alloc_va(aligned_size);

        let mut args = KfdAllocMemoryArgs {
            va_addr,
            size: aligned_size as u64,
            handle: 0,
            mmap_offset: 0,
            gpu_id: self.gpu_id,
            flags,
        };

        ioctl_safe(self.kfd_fd, AMDKFD_IOC_ALLOC_MEMORY, &mut args as *mut _ as *mut u8)
            .map_err(|e| format!("ALLOC_MEMORY failed (size={}, flags=0x{:X}): {}", aligned_size, flags, e))?;

        // Map memory to GPU
        let mut gpu_ids = [self.gpu_id];
        let mut map_args = KfdMapMemoryArgs {
            handle: args.handle,
            device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
            n_devices: 1,
            n_success: 0,
        };
        ioctl_safe(self.kfd_fd, AMDKFD_IOC_MAP_MEMORY, &mut map_args as *mut _ as *mut u8)
            .map_err(|e| format!("MAP_MEMORY failed: {}", e))?;
        if map_args.n_success != 1 {
            return Err(format!("MAP_MEMORY incomplete: n_success={}", map_args.n_success));
        }

        // mmap to CPU address space using MAP_FIXED on the pre-reserved VA
        // CRITICAL: VRAM mmap must use drm_fd (/dev/dri/renderD128), NOT kfd_fd!
        let host_ptr = unsafe {
            mmap(
                args.va_addr as *mut u8, // MAP_FIXED on pre-reserved address
                aligned_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_FIXED,
                self.drm_fd,
                args.mmap_offset as i64,
            )
        };
        if host_ptr == MAP_FAILED || host_ptr.is_null() {
            return Err(format!("mmap failed for KFD buffer: {}", std::io::Error::last_os_error()));
        }
        assert_eq!(host_ptr as u64, args.va_addr, "MAP_FIXED returned wrong address");

        Ok(GpuBuffer {
            handle: args.handle,
            va_addr: args.va_addr,
            host_ptr,
            size: aligned_size,
            device: Arc::clone(self),
        })
    }

    /// Create an AQL compute queue with default ring size (4MB = 65536 packets)
    pub fn create_queue(self: &Arc<Self>) -> Result<AqlQueue, String> {
        self.create_queue_sized(4 << 20)  // 4MB default
    }

    /// Create an AQL compute queue with specified ring buffer size in bytes.
    /// Ring size must be power of 2. Each packet is 64 bytes.
    /// Recommended: 1<<20 (1MB, 16K pkts), 4<<20 (4MB, 64K pkts), 16<<20 (16MB, 256K pkts)
    pub fn create_queue_sized(self: &Arc<Self>, ring_size: u32) -> Result<AqlQueue, String> {
        assert!(ring_size.is_power_of_two(), "AQL ring_size must be power of 2, got {}", ring_size);

        // Allocate ring buffer (uncached GTT — tinygrad pattern)
        let ring_buffer = self.alloc_uncached(ring_size as usize)?;

        // Zero the ring buffer, then initialize all packet headers to INVALID(1)
        // WARNING: Header=0 means HSA_PACKET_TYPE_VENDOR_SPECIFIC (not empty!)
        //          Only Header=1 (HSA_PACKET_TYPE_INVALID) marks a slot as free.
        //          CP prefetches slots and will choke on VENDOR_SPECIFIC(0) headers.
        unsafe {
            std::ptr::write_bytes(ring_buffer.host_ptr, 0, ring_buffer.size);
            let num_packets = ring_size as usize / 64;
            for i in 0..num_packets {
                let pkt_ptr = ring_buffer.host_ptr.add(i * 64) as *mut u16;
                std::ptr::write_volatile(pkt_ptr, 1u16); // HSA_PACKET_TYPE_INVALID
            }
        }

        // Allocate write/read pointer memory (uncached GTT)
        let wr_ptrs = self.alloc_uncached(4096)?; // page for write_ptr + read_ptr
        unsafe { std::ptr::write_bytes(wr_ptrs.host_ptr, 0, wr_ptrs.size); }
        // Write/read pointer addresses — use GPU VA (same as Tinygrad pattern)
        let write_ptr_va = wr_ptrs.va_addr;
        let read_ptr_va = wr_ptrs.va_addr + 8;

        // Allocate EOP buffer (uncached GTT)
        let eop_buffer = self.alloc_uncached(4096)?;

        // CWSR (Context Wave Save Restore) buffer
        // The hardcoded sizes below are specific to 96 CU (RX 7900 XTX).
        // On other GPUs, set KFD_NO_CWSR=1 to skip CWSR allocation.
        let cwsr_disabled = std::env::var("KFD_NO_CWSR").map(|v| v == "1").unwrap_or(false);
        let (cwsr_buffer, cwsr_size, ctl_stack_size) = if !cwsr_disabled {
            // Kernel formula (kfd_queue.c kfd_queue_ctx_save_restore_size):
            //   cu_num = 96, wave_num = 3072 (cu_num * 32 for gfxv >= 100100)
            //   ctl_stack = ALIGN(40 + 3072*12 + 8, 4096) = 40960
            //   wg_data = ALIGN(96 * (0x60000+0x4000+0x10000+0x1000), 4096) = 46006272
            //   cwsr_size = 40960 + 46006272 = 46047232
            //   debug_memory = ALIGN(3072 * 32, 64) = 98304
            //   total_alloc = ALIGN(cwsr_size + debug_memory, 4096) = 46145536
            let total_cwsr_alloc: usize = 46145536;    // 0x2C02000 — buffer alloc (incl debug)
            let buf = self.alloc_uncached(total_cwsr_alloc)?;
            (Some(buf), 46047232u32, 40960u32)
        } else {
            (None, 0u32, 0u32)
        };

        let mut args = KfdCreateQueueArgs {
            ring_base_address: ring_buffer.va_addr,
            write_pointer_address: write_ptr_va,
            read_pointer_address: read_ptr_va,
            doorbell_offset: 0, // returned by kernel
            ring_size,
            gpu_id: self.gpu_id,
            queue_type: KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
            queue_percentage: 100,
            queue_priority: 7, // medium priority
            queue_id: 0,       // returned by kernel
            eop_buffer_address: eop_buffer.va_addr,
            eop_buffer_size: eop_buffer.size as u64,
            ctx_save_restore_address: cwsr_buffer.as_ref().map(|b| b.va_addr).unwrap_or(0),
            ctx_save_restore_size: cwsr_size,
            ctl_stack_size,
            sdma_engine_id: 0,
            pad: 0,
        };

        if std::env::var("KFD_DEBUG").is_ok() {
            eprintln!("[KFD] CREATE_QUEUE args:");
            eprintln!("  ring_base=0x{:X} ring_size={}", args.ring_base_address, args.ring_size);
            eprintln!("  write_ptr=0x{:X} read_ptr=0x{:X}", args.write_pointer_address, args.read_pointer_address);
            eprintln!("  gpu_id={} queue_type={} pct={} pri={}", args.gpu_id, args.queue_type, args.queue_percentage, args.queue_priority);
            eprintln!("  eop_addr=0x{:X} eop_size={}", args.eop_buffer_address, args.eop_buffer_size);
            eprintln!("  cwsr_addr=0x{:X} cwsr_size={} ctl_stack={}", args.ctx_save_restore_address, args.ctx_save_restore_size, args.ctl_stack_size);
        }

        ioctl_safe(self.kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &mut args as *mut _ as *mut u8)
            .map_err(|e| format!("CREATE_QUEUE failed: {}", e))?;

        eprintln!("[KFD] Queue {} created (doorbell_offset=0x{:X})", args.queue_id, args.doorbell_offset);

        // mmap doorbell from /dev/kfd using the returned offset
        // doorbell_offset is the mmap offset from KFD — use directly
        // Use &!0x1FFF (8KB/two-page alignment) per tinygrad's proven approach
        let doorbell_base = args.doorbell_offset & !0x1FFF; // two-page aligned
        let doorbell_in_page = (args.doorbell_offset - doorbell_base) as usize;
        if std::env::var("KFD_DEBUG").is_ok() {
            eprintln!("[KFD] doorbell raw=0x{:X} base=0x{:X} in_page=0x{:X}",
                args.doorbell_offset, doorbell_base, doorbell_in_page);
        }
        let doorbell_mmap = unsafe {
            mmap(
                std::ptr::null_mut(),
                0x2000, // two pages
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.kfd_fd,
                doorbell_base as i64,
            )
        };
        if doorbell_mmap == MAP_FAILED || doorbell_mmap.is_null() {
            return Err(format!("mmap doorbell failed: {}", std::io::Error::last_os_error()));
        }
        let doorbell_ptr = unsafe { doorbell_mmap.add(doorbell_in_page) };
        if std::env::var("KFD_DEBUG").is_ok() {
            eprintln!("[KFD] doorbell mmap={:?} ptr={:?}", doorbell_mmap, doorbell_ptr);
        }

        // Allocate completion buffer for PM4-in-AQL — GPU writes seqno here after dispatch
        let completion_buf = self.alloc_uncached(64)?;
        // Zero it initially
        unsafe { std::ptr::write_bytes(completion_buf.host_ptr, 0, 64); }

        Ok(AqlQueue {
            queue_id: args.queue_id,
            ring_buffer,
            ring_size,
            write_ptr_host: wr_ptrs.host_ptr as *mut u64,
            read_ptr_host: unsafe { wr_ptrs.host_ptr.add(8) as *mut u64 },
            doorbell_ptr: doorbell_ptr as *mut u64,
            doorbell_mmap_base: doorbell_mmap,
            doorbell_mmap_size: 0x2000,
            pm4_ib: None,
            pm4_ib_offset: 0,
            completion_buf,
            completion_seqno: 0,
            _wr_ptrs: wr_ptrs,
            _eop_buffer: eop_buffer,
            _cwsr_buffer: cwsr_buffer,
            device: Arc::clone(self),
        })
    }

    /// Create a PM4 compute queue (type=0)
    /// PM4 queues use raw PACKET3 commands instead of AQL packets.
    /// Doorbell is u32 byte-offset into ring buffer.
    pub fn create_pm4_queue(self: &Arc<Self>) -> Result<Pm4Queue, String> {
        let ring_size: u32 = 16 << 20; // 16MB ring (same as tinygrad)
        let ring_buffer = self.alloc_uncached(ring_size as usize)?;
        // Zero ring
        unsafe { std::ptr::write_bytes(ring_buffer.host_ptr, 0, ring_buffer.size); }

        let wr_ptrs = self.alloc_uncached(4096)?;
        unsafe { std::ptr::write_bytes(wr_ptrs.host_ptr, 0, wr_ptrs.size); }
        let write_ptr_va = wr_ptrs.va_addr;
        let read_ptr_va = wr_ptrs.va_addr + 8;

        let eop_buffer = self.alloc_uncached(4096)?;
        unsafe { std::ptr::write_bytes(eop_buffer.host_ptr, 0, eop_buffer.size); }

        // CWSR — same pattern as AQL queue
        let cwsr_disabled = std::env::var("KFD_NO_CWSR").map(|v| v == "1").unwrap_or(false);
        let (cwsr_buffer, cwsr_size, ctl_stack_size) = if !cwsr_disabled {
            let total_cwsr_alloc: usize = 46145536;
            let buf = self.alloc_uncached(total_cwsr_alloc)?;
            unsafe { std::ptr::write_bytes(buf.host_ptr, 0, buf.size); }
            (Some(buf), 46047232u32, 40960u32)
        } else {
            (None, 0u32, 0u32)
        };

        let mut args = KfdCreateQueueArgs {
            ring_base_address: ring_buffer.va_addr,
            write_pointer_address: write_ptr_va,
            read_pointer_address: read_ptr_va,
            doorbell_offset: 0,
            ring_size,
            gpu_id: self.gpu_id,
            queue_type: KFD_IOC_QUEUE_TYPE_COMPUTE, // PM4!
            queue_percentage: 100,
            queue_priority: 7,
            queue_id: 0,
            eop_buffer_address: eop_buffer.va_addr,
            eop_buffer_size: eop_buffer.size as u64,
            ctx_save_restore_address: cwsr_buffer.as_ref().map(|b| b.va_addr).unwrap_or(0),
            ctx_save_restore_size: cwsr_size,
            ctl_stack_size,
            sdma_engine_id: 0,
            pad: 0,
        };

        println!("[KFD] PM4 CREATE_QUEUE args:");
        println!("  ring_base=0x{:X} ring_size={}", ring_buffer.va_addr, ring_size);
        println!("  queue_type=0 (COMPUTE/PM4)");

        ioctl_safe(self.kfd_fd, AMDKFD_IOC_CREATE_QUEUE,
            &mut args as *mut _ as *mut u8)?;

        println!("[KFD] PM4 Queue {} created (doorbell_offset=0x{:X})",
            args.queue_id, args.doorbell_offset);

        // Map doorbell — use offset directly (no >>1 shift!)
        // KFD returns doorbell_offset as a direct mmap offset for /dev/kfd.
        // Use &!0x1fff (8KB/two-page alignment) per tinygrad's proven approach.
        let db_offset_raw = args.doorbell_offset;
        let db_base = db_offset_raw & !0x1FFF; // two-page aligned
        let db_in_page = (db_offset_raw - db_base) as usize;
        eprintln!("[KFD] PM4 doorbell: raw=0x{:X} base=0x{:X} in_page=0x{:X}",
            db_offset_raw, db_base, db_in_page);
        let doorbell_base = unsafe {
            mmap(
                std::ptr::null_mut(),
                8192,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.kfd_fd,
                db_base as i64,
            )
        };
        if doorbell_base == MAP_FAILED {
            return Err("PM4 doorbell mmap failed".to_string());
        }
        let doorbell_ptr = unsafe { doorbell_base.add(db_in_page) as *mut u64 };

        Ok(Pm4Queue {
            queue_id: args.queue_id,
            ring_buffer,
            ring_size,
            write_ptr_host: wr_ptrs.host_ptr as *mut u64,
            read_ptr_host: unsafe { wr_ptrs.host_ptr.add(8) as *mut u64 },
            doorbell_ptr,
            write_offset: 0, // byte offset into ring
            doorbell_mmap_base: doorbell_base,
            _wr_ptrs: wr_ptrs,
            _eop_buffer: eop_buffer,
            _cwsr_buffer: cwsr_buffer,
            device: Arc::clone(self),
        })
    }

    /// Flush HDP (Host Data Path) cache
    /// Forces CPU writes to VRAM to be visible to GPU.
    /// Uses PCIe read-after-write ordering: reading any byte from the
    /// VRAM buffer forces all pending writes to drain.
    pub fn hdp_flush(buf: &GpuBuffer) {
        let _ = unsafe { std::ptr::read_volatile(buf.host_ptr) };
    }
}

impl Drop for KfdDevice {
    fn drop(&mut self) {
        // Release event page memory (allocated during open)
        if self.event_page_handle != 0 {
            // Unmap from GPU
            let mut gpu_ids = [self.gpu_id];
            let mut unmap = KfdUnmapMemoryArgs {
                handle: self.event_page_handle,
                device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
                n_devices: 1,
                n_success: 0,
            };
            let _ = ioctl_safe(self.kfd_fd, AMDKFD_IOC_UNMAP_MEMORY,
                &mut unmap as *mut _ as *mut u8);
            // Free GPU memory
            let mut free = KfdFreeMemoryArgs { handle: self.event_page_handle };
            let _ = ioctl_safe(self.kfd_fd, AMDKFD_IOC_FREE_MEMORY,
                &mut free as *mut _ as *mut u8);
        }
        unsafe {
            close(self.drm_fd);
            close(self.kfd_fd);
        }
    }
}

// =============================================================================
// GpuBuffer — RAII GPU memory with automatic cleanup
// =============================================================================

/// GPU memory buffer with automatic lifecycle management
pub struct GpuBuffer {
    handle: u64,
    pub va_addr: u64,
    pub host_ptr: *mut u8,
    pub size: usize,
    device: Arc<KfdDevice>,
}

// GpuBuffer is Send+Sync because it wraps GPU memory accessed via mmap
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Write data from CPU to GPU buffer
    pub fn write(&self, data: &[u8]) {
        assert!(data.len() <= self.size, "write overflow: {} > {}", data.len(), self.size);
        unsafe {
            // Use volatile writes to ensure WC (write-combine) mapped memory 
            // is properly flushed to GPU. Regular memcpy may leave data in
            // CPU write-combine buffers, causing GPU to read stale data.
            let dst = self.host_ptr;
            let src = data.as_ptr();
            // Write in 8-byte chunks for efficiency, then remaining bytes
            let n8 = data.len() / 8;
            let rem = data.len() % 8;
            for i in 0..n8 {
                let val = std::ptr::read_unaligned(src.add(i * 8) as *const u64);
                std::ptr::write_volatile(dst.add(i * 8) as *mut u64, val);
            }
            let base = n8 * 8;
            for i in 0..rem {
                std::ptr::write_volatile(dst.add(base + i), *src.add(base + i));
            }
            // Force WC flush: mfence ensures all prior stores are globally visible
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Read data from GPU buffer to CPU
    pub fn read(&self, buf: &mut [u8]) {
        assert!(buf.len() <= self.size, "read overflow: {} > {}", buf.len(), self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(self.host_ptr, buf.as_mut_ptr(), buf.len());
        }
    }

    /// Write typed data
    pub fn write_val<T: Copy>(&self, offset: usize, val: T) {
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe {
            let ptr = self.host_ptr.add(offset) as *mut T;
            std::ptr::write_volatile(ptr, val);
        }
    }

    /// Read typed data
    pub fn read_val<T: Copy>(&self, offset: usize) -> T {
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe {
            let ptr = self.host_ptr.add(offset) as *const T;
            std::ptr::read_volatile(ptr)
        }
    }

    /// GPU virtual address
    pub fn gpu_addr(&self) -> u64 {
        self.va_addr
    }

    // ── Safe accessors with offset (#17/#18) ──

    /// Write bytes at a specific offset with bounds checking.
    pub fn write_bytes(&self, offset: usize, data: &[u8]) {
        assert!(offset + data.len() <= self.size,
            "write_bytes overflow: offset={} len={} size={}", offset, data.len(), self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.host_ptr.add(offset), data.len());
        }
    }

    /// Read bytes from a specific offset with bounds checking.
    pub fn read_bytes(&self, offset: usize, len: usize) -> Vec<u8> {
        assert!(offset + len <= self.size,
            "read_bytes overflow: offset={} len={} size={}", offset, len, self.size);
        let mut buf = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(self.host_ptr.add(offset), buf.as_mut_ptr(), len);
        }
        buf
    }

    /// Zero the buffer
    pub fn zero(&self) {
        unsafe { std::ptr::write_bytes(self.host_ptr, 0, self.size); }
    }

    /// Create a sub-region view of this buffer (for pool allocation).
    /// The sub-region has handle=0, so Drop won't call KFD free.
    pub fn sub_region(parent: &GpuBuffer, offset: usize, size: usize) -> GpuBuffer {
        assert!(offset + size <= parent.size, "sub_region overflow: {}+{} > {}", offset, size, parent.size);
        GpuBuffer {
            handle: 0, // sentinel: sub-region, don't free
            va_addr: parent.va_addr + offset as u64,
            host_ptr: unsafe { parent.host_ptr.add(offset) },
            size,
            device: parent.device.clone(),
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // handle=0 means this is a sub-region view (pool allocation), don't free
        if self.handle == 0 {
            return;
        }
        unsafe {
            munmap(self.host_ptr, self.size);
        }
        // Unmap from GPU
        let mut gpu_ids = [self.device.gpu_id];
        let mut unmap = KfdUnmapMemoryArgs {
            handle: self.handle,
            device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
            n_devices: 1,
            n_success: 0,
        };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_UNMAP_MEMORY,
            &mut unmap as *mut _ as *mut u8);
        // Free GPU memory
        let mut free = KfdFreeMemoryArgs { handle: self.handle };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_FREE_MEMORY,
            &mut free as *mut _ as *mut u8);
    }
}

// =============================================================================
// Pre-dispatch kernarg validation (debug builds)
// =============================================================================

/// Scan kernarg bytes for obviously invalid GPU pointers.
///
/// Heuristic: every 8-byte aligned value > 0xFFFF is tested as a potential
/// GPU VA. Values that are NULL (0), unaligned pointers, or obviously out of
/// the 48-bit VA space are flagged.
///
/// This catches the most common hard-hang root causes:
/// - Host pointers passed instead of gpu_addr()
/// - Uninitialized kernarg fields (0x0 for pointer args)
/// - Use-after-free (GPU buffer dropped, same VA reused with different mapping)
///
/// Only active in debug builds via `#[cfg(debug_assertions)]` callers.
#[cfg(debug_assertions)]
fn validate_kernargs_bytes(ka: &[u8], ka_size: usize) {
    let mut offset = 0;
    while offset + 8 <= ka_size {
        let val = u64::from_le_bytes(ka[offset..offset + 8].try_into().unwrap());

        // Skip small values (likely u32 params padded to 8 bytes)
        if val <= 0xFFFF {
            offset += 8;
            continue;
        }

        // NULL pointer check (exact zero is caught above, this catches near-NULL)
        if val > 0xFFFF && val < 0x1000_0000 {
            // Suspicious: too large for a scalar, too small for a valid GPU VA
            // GPU VAs from mmap are typically > 0x1_0000_0000
            eprintln!(
                "[KFD VALIDATE] Warning: kernarg offset {} has value 0x{:016X} \
                 — suspicious (too small for GPU VA, too large for u32 param)",
                offset, val
            );
        }

        // 48-bit VA space check
        if val > 0x0000_FFFF_FFFF_FFFF {
            panic!(
                "[KFD SAFETY] Kernarg offset {} contains 0x{:016X} — \
                 outside 48-bit VA space. Likely a host pointer or uninitialized memory!\n\
                 This WILL cause a GPU page fault and hard hang.",
                offset, val
            );
        }

        offset += 8;
    }
}

// =============================================================================
// AqlQueue — AQL compute queue with doorbell dispatch
// =============================================================================

/// AQL hardware compute queue
pub struct AqlQueue {
    pub queue_id: u32,
    pub ring_buffer: GpuBuffer,
    pub ring_size: u32,
    pub write_ptr_host: *mut u64,
    pub read_ptr_host: *mut u64,
    pub doorbell_ptr: *mut u64,
    /// Original mmap base for doorbell (needed for correct munmap)
    doorbell_mmap_base: *mut u8,
    /// Size of doorbell mmap region
    doorbell_mmap_size: usize,
    // PM4-in-AQL: indirect buffer for PM4 commands
    pm4_ib: Option<GpuBuffer>,
    pm4_ib_offset: usize,
    // PM4-in-AQL: completion buffer — GPU writes seqno here after kernel finishes
    completion_buf: GpuBuffer,
    completion_seqno: u32,
    // Keep these alive (RAII)
    pub _wr_ptrs: GpuBuffer,
    pub _eop_buffer: GpuBuffer,
    pub _cwsr_buffer: Option<GpuBuffer>,
    pub device: Arc<KfdDevice>,
}

unsafe impl Send for AqlQueue {}
unsafe impl Sync for AqlQueue {}

impl AqlQueue {
    /// Dispatch a kernel. Returns after GPU completes execution.
    ///
    /// `kernel` = loaded GPU kernel
    /// `grid` = [grid_x, grid_y, grid_z] in threads (NOT workgroups)
    /// `kernargs` = kernel argument data (will be copied to GPU)
    pub fn dispatch(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) -> Result<(), String> {
        self.dispatch_signal(kernel, grid, kernargs, None)
    }

    /// Dispatch with explicit signal buffer for completion tracking.
    /// Validates kernarg size and ensures ring space before dispatch.
    pub fn dispatch_signal(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
        signal: Option<&GpuBuffer>,
    ) -> Result<(), String> {
        // Validate kernarg size matches kernel's declared requirement
        assert!(
            kernargs.size >= kernel.kernarg_size as usize,
            "kernarg too small: buffer={}B, kernel expects {}B",
            kernargs.size, kernel.kernarg_size
        );
        // Ensure ring buffer has space (prevents overflow)
        self.ensure_ring_space();
        // Get current write pointer
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1; // number of slots - 1
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // Build AQL dispatch packet (write header LAST for atomicity)
        let pkt_ptr = unsafe { self.ring_buffer.host_ptr.add(pkt_offset) as *mut AqlDispatchPacket };

        // Prepare completion signal using amd_signal_t layout:
        //   offset 0x00: kind (u64)   — must be 1 (AMD_SIGNAL_KIND_USER)
        //   offset 0x08: value (i64)  — CP will atomic_sub(1) upon completion
        //   offset 0x10: event_mailbox_ptr (u64) — must be 0 (no event)
        //   Total: 64 bytes, must be zeroed first
        let signal_va = if let Some(sig) = signal {
            // Zero entire 64-byte signal struct to clear any garbage event pointers
            unsafe { std::ptr::write_bytes(sig.host_ptr, 0, 64); }
            // kind = 1 (AMD_SIGNAL_KIND_USER) at offset 0
            sig.write_val::<u64>(0, 1);
            // value = 1 at offset 8 (CP will atomic_sub to make it 0)
            sig.write_val::<i64>(8, 1);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            sig.gpu_addr()
        } else {
            0
        };

        // Build header: type=DISPATCH(2), barrier=1, acquire=SYSTEM(2), release=SYSTEM(2)
        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |       // bits 0:7 = type
            (1 << 8) |                                        // bit 8 = barrier
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |         // bits 9:10 = acquire fence
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);         // bits 11:12 = release fence

        unsafe {
            // Write all fields EXCEPT header first (use addr_of_mut! for packed-safe access)
            let base = pkt_ptr as *mut u8;
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16); // setup: 3D (always, unused dims=1)
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16); // reserved0
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32); // private_segment_size
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size); // group_segment_size
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va); // kernel_object
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernargs.gpu_addr()); // kernarg_address
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64); // reserved2
            std::ptr::write_volatile(base.add(0x38) as *mut u64, signal_va); // completion_signal

            // Memory fence before writing header
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

            // Write header LAST (makes packet visible to CP atomically)
            std::ptr::write_volatile(base.add(0x00) as *mut u16, header);

            // Memory fence after header write
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            // Update write pointer (points PAST the last packet)
            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);

            // Memory fence to ensure write pointer is visible before doorbell
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            // doorbell = new_write_idx - 1 (index of the just-written packet)
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }

        // Wait for completion
        if let Some(sig) = signal {
            self.wait_signal(sig)?;
        } else {
            // No signal: poll read_ptr (less safe but functional)
            self.wait_read_ptr(write_idx + 1)?;
        }

        Ok(())
    }

    /// Submit a kernel without waiting — pipelined dispatch.
    /// 
    /// Writes AQL packet and rings doorbell, returns immediately.
    /// Call `wait_idle()` after submitting a batch to drain the queue.
    /// No signal overhead, no waiting — maximum throughput.
    ///
    /// Ring buffer overflow protection: spin-waits if ring is nearly full.
    pub fn submit(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) {
        // Pre-dispatch kernarg validation (debug builds only — zero cost in release)
        // Catches NULL pointers and invalid GPU VAs BEFORE they reach the GPU.
        #[cfg(debug_assertions)]
        {
            let ka_size = (kernel.kernarg_size as usize).min(kernargs.size);
            if ka_size >= 8 {
                let ka_bytes = kernargs.read_bytes(0, ka_size);
                validate_kernargs_bytes(&ka_bytes, ka_size);
            }
        }

        self.ensure_ring_space();
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // barrier=1 ensures previous kernel completes before this one starts
        // (critical for data dependencies between consecutive kernels)
        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |                                        // barrier bit
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size);
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va);
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernargs.gpu_addr());
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(base.add(0x38) as *mut u64, 0u64); // no signal

            // SeqCst = mfence on x86: flush WC buffers before doorbell
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(base as *mut u16, header);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
    }

    /// Submit kernel dispatch using raw kernargs GPU address (no signal, no wait).
    /// Similar to `submit()` but takes a `u64` address directly, enabling
    /// DispatchPool's single-buffer offset addressing.
    ///
    /// Ring buffer overflow protection: spin-waits if ring is nearly full.
    pub fn submit_at(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernarg_addr: u64,
    ) {
        self.ensure_ring_space();
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size);
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va);
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernarg_addr);
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(base.add(0x38) as *mut u64, 0u64);

            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(base as *mut u16, header);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
    }

    /// Batch-optimized submit: write AQL packet WITHOUT ringing doorbell.
    ///
    /// Call this N times, then call `ring_doorbell()` once.
    /// Avoids N MMIO writes to doorbell (each ~0.5-2μs via PCIe).
    /// Also optimizes packet construction: 1 memcpy instead of 13 write_volatile,
    /// and only 1 fence instead of 3.
    ///
    /// ```ignore
    /// for i in 0..100 {
    ///     queue.submit_batch(&kernel, grid, pool.get_kernargs(i));
    /// }
    /// queue.ring_doorbell();  // single MMIO write
    /// queue.wait_idle()?;
    /// ```
    pub fn submit_batch(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) {
        self.submit_batch_addr(kernel, grid, kernargs.gpu_addr());
    }

    /// Batch-optimized submit with raw kernarg address.
    ///
    /// Ring buffer overflow protection: spin-waits if ring is nearly full.
    pub fn submit_batch_addr(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernarg_addr: u64,
    ) {
        self.ensure_ring_space();
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // Build entire 64-byte AQL packet in a stack buffer, then memcpy once
        #[repr(C, packed)]
        struct AqlPkt {
            header: u16,        // 0x00
            setup: u16,         // 0x02
            wg_x: u16,          // 0x04
            wg_y: u16,          // 0x06
            wg_z: u16,          // 0x08
            reserved0: u16,     // 0x0A
            grid_x: u32,        // 0x0C
            grid_y: u32,        // 0x10
            grid_z: u32,        // 0x14
            private_seg: u32,   // 0x18
            group_seg: u32,     // 0x1C
            kernel_obj: u64,    // 0x20
            kernarg: u64,       // 0x28
            reserved2: u64,     // 0x30
            signal: u64,        // 0x38
        }

        // Header with INVALID type first — CP won't process until we flip header
        let real_header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        let pkt = AqlPkt {
            header: 1u16,   // INVALID — placeholder, overwritten below
            setup: 3,
            wg_x: kernel.workgroup_size[0] as u16,
            wg_y: kernel.workgroup_size[1] as u16,
            wg_z: kernel.workgroup_size[2] as u16,
            reserved0: 0,
            grid_x: grid[0],
            grid_y: grid[1],
            grid_z: grid[2],
            private_seg: 0,
            group_seg: kernel.lds_size,
            kernel_obj: kernel.descriptor_va,
            kernarg: kernarg_addr,
            reserved2: 0,
            signal: 0,
        };

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);

            // Single memcpy for the whole packet (header=INVALID so CP ignores it)
            std::ptr::copy_nonoverlapping(
                &pkt as *const AqlPkt as *const u8,
                base,
                64,
            );

            // SeqCst = mfence: flush WC buffers before making packet valid
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            // Atomically make packet valid by writing the real header
            std::ptr::write_volatile(base as *mut u16, real_header);

            // Update write pointer (no fence needed — x86 TSO guarantees
            // stores are visible in order, and Release fence above is sufficient)
            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            // NO doorbell write — caller must call ring_doorbell()
        }
    }

    /// Ring the doorbell once after a batch of submit_batch() calls.
    /// This triggers CP to process all queued packets.
    pub fn ring_doorbell(&self) {
        unsafe {
            let write_idx = std::ptr::read_volatile(self.write_ptr_host);
            // fence to ensure all packet writes + write_ptr are visible before doorbell MMIO
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(self.doorbell_ptr, write_idx - 1);
        }
    }

    // =========================================================================
    // Ring buffer overflow protection
    // =========================================================================

    /// Ensure there is space in the AQL ring buffer before writing a new packet.
    /// 
    /// Spin-waits if `write_idx - read_idx >= ring_slots - 64`.
    /// The 64-slot margin provides headroom so we never overwrite an unprocessed
    /// packet. With a 4MB ring (65536 slots), this effectively never triggers
    /// under normal workloads, but prevents hard hangs if thousands of kernels
    /// are submitted faster than the GPU can consume them.
    ///
    /// On timeout (5s): panics instead of exit(99) so callers can catch_unwind.
    fn ensure_ring_space(&self) {
        let ring_slots = self.ring_size as u64 / 64;
        // Leave 64 slots of headroom to avoid overwriting in-flight packets
        let max_inflight = ring_slots - 64;
        let start = std::time::Instant::now();
        let mut last_log = 0u64;
        loop {
            let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if write_idx.wrapping_sub(read_idx) < max_inflight {
                return;
            }
            let elapsed_s = start.elapsed().as_secs();
            // Periodic progress log every 2s so we can see if GPU is still alive
            if elapsed_s >= last_log + 2 {
                last_log = elapsed_s;
                eprintln!(
                    "[KFD] ensure_ring_space: waiting {elapsed_s}s — \
                     write={write_idx} read={read_idx} inflight={} ring_slots={ring_slots}",
                    write_idx.wrapping_sub(read_idx)
                );
            }
            // Timeout after 5 seconds — panic (catchable) instead of exit(99)
            // to avoid leaving GPU in hung state after process death.
            if elapsed_s >= 5 {
                let msg = format!(
                    "[KFD] ensure_ring_space TIMEOUT (5s): GPU likely hung!\n\
                     write_idx={} read_idx={} inflight={} ring_slots={} max_inflight={}\n\
                     This indicates a GPU page fault or kernel hang.",
                    write_idx, read_idx, write_idx.wrapping_sub(read_idx),
                    ring_slots, max_inflight
                );
                eprintln!("{}", msg);
                panic!("{}", msg);
            }
            std::hint::spin_loop();
        }
    }

    pub fn wait_idle(&self) -> Result<(), String> {
        let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        self.wait_read_ptr(target)
    }

    /// Wait for all pending dispatches + memory fence.
    /// This is the SAFE way to synchronize before dropping GPU buffers.
    /// Ensures all GPU stores (including L2 writeback) are complete.
    pub fn synchronize(&self) -> Result<(), String> {
        self.wait_idle()?;
        // Memory fence to ensure CPU sees GPU writes
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // Small sleep to allow L2 cache writeback to complete
        std::thread::sleep(std::time::Duration::from_micros(10));
        Ok(())
    }

    /// Wait for completion signal (poll amd_signal_t.value at offset 8)
    fn wait_signal(&self, signal: &GpuBuffer) -> Result<(), String> {
        let timeout_ns: u64 = 60_000_000_000; // 60 seconds (large GEMMs can be slow on first dispatch)
        let start = std::time::Instant::now();
        loop {
            // Read amd_signal_t.value at offset 8 (not offset 0 which is kind!)
            let val: i64 = signal.read_val(8);
            if val <= 0 {
                return Ok(());
            }
            if start.elapsed().as_nanos() as u64 > timeout_ns {
                return Err(format!("Kernel execution timeout (>{}s)", timeout_ns / 1_000_000_000));
            }
            std::hint::spin_loop();
        }
    }

    /// Fallback: wait by polling read pointer.
    /// Returns Err on timeout (5s) instead of exit(99) to allow graceful recovery.
    fn wait_read_ptr(&self, target: u64) -> Result<(), String> {
        let timeout_ns: u64 = 5_000_000_000;  // 5s: fast fail to prevent system hang
        let start = std::time::Instant::now();
        let mut last_log_s = 0u64;
        loop {
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target {
                return Ok(());
            }
            let elapsed = start.elapsed();
            let elapsed_s = elapsed.as_secs();
            // Progress log every 1s so we can see if GPU is making progress
            if elapsed_s >= last_log_s + 1 {
                last_log_s = elapsed_s;
                let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
                eprintln!(
                    "[KFD] wait_read_ptr: {elapsed_s}s — read={read_idx} target={target} \
                     write={write_idx} pending={}",
                    write_idx.wrapping_sub(read_idx)
                );
            }
            if elapsed.as_nanos() as u64 > timeout_ns {
                let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
                let msg = format!(
                    "[KFD] wait_read_ptr TIMEOUT (5s): GPU hung! \
                     read={}, target={}, write={}, pending={}",
                    read_idx, target, write_idx, write_idx.wrapping_sub(read_idx)
                );
                eprintln!("{}", msg);
                // Return Err instead of exit(99) so tests can continue
                // and GPU resources can potentially be reclaimed by KFD driver.
                return Err(msg);
            }
            std::hint::spin_loop();
        }
    }

    // =========================================================================
    // Optimized dispatch path — minimal overhead
    // =========================================================================

    /// Ultra-low-latency submit: skips ensure_ring_space, uses single Release fence.
    ///
    /// **Safety**: caller must guarantee ring buffer won't overflow (typical for
    /// benchmarks or pre-checked workloads). For production code, use `submit()`.
    ///
    /// Optimizations vs `submit()`:
    /// - 1× Release fence instead of 3× SeqCst (eliminates 2 x86 mfence @ ~33ns each)
    /// - No `ensure_ring_space()` check
    /// - Direct field writes without intermediate volatile reads where possible
    pub fn submit_fast(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) {
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // AQL header: dispatch + barrier + AGENT fences (GPU-internal only, no PCIe sync)
        // AGENT scope saves ~10-20μs per dispatch vs SYSTEM scope by avoiding L2 writeback.
        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_AGENT as u16) << 9) |
            ((HSA_FENCE_SCOPE_AGENT as u16) << 11);

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);

            // Write packet body first (header last = atomic activation)
            // setup = dims(3) at +0x02
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
            // workgroup_size at +0x04, +0x06, +0x08
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
            // grid_size at +0x0C, +0x10, +0x14
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            // private_segment_size + group_segment_size at +0x18, +0x1C
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size);
            // kernel_object (descriptor VA) at +0x20
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va);
            // kernarg_address at +0x28
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernargs.gpu_addr());
            // reserved + completion signal = 0
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(base.add(0x38) as *mut u64, 0u64);

            // Single Release fence: ensures all packet body writes are visible
            // before activating the header. On x86, this compiles to nothing
            // (x86 stores are naturally ordered) — only the final SeqCst for
            // the doorbell is needed.
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

            // Activate packet (header write)
            std::ptr::write_volatile(base as *mut u16, header);

            // SeqCst fence before doorbell write: this is the one fence we
            // truly need — it ensures the WC buffer is drained to memory
            // before the doorbell ring reaches the GPU's CP.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            // Doorbell: GPU CP reads this to discover new packets
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
    }

    /// Ultra-low-latency wait: tight spin on read_dispatch_id.
    ///
    /// No `Instant::now()`, no `.elapsed()`, no progress logging.
    /// Pure volatile read + spin_loop_hint. Typical exit: < 1μs for empty kernels.
    ///
    /// **Warning**: hangs forever if GPU is stuck. For production, use `wait_idle()`.
    #[inline]
    pub fn wait_idle_spin(&self) {
        let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        loop {
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target {
                return;
            }
            std::hint::spin_loop();
        }
    }

    // =========================================================================
    // GPU-accelerated buffer zeroing
    // =========================================================================

    /// Zero a GPU buffer using a GPU memset kernel.
    ///
    /// **~38x faster** than `buf.zero()` (CPU PCIe writes at ~25 GB/s):
    /// Uses `global_store_dwordx4` at VRAM bandwidth (~960 GB/s).
    ///
    /// The memset kernel is lazily built and cached on first call.
    ///
    /// ```ignore
    /// queue.gpu_zero(&d_output);  // zeros entire buffer on GPU
    /// queue.wait_idle()?;         // wait for completion
    /// ```
    pub fn gpu_zero(&self, buf: &GpuBuffer) {
        use std::sync::OnceLock;
        use crate::rdna3_asm::gfx11;
        use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

        // Lazy-init: build + load kernel once, reuse forever
        static MEMSET_KERNEL: OnceLock<GpuKernel> = OnceLock::new();
        static MEMSET_KA_BUF: OnceLock<GpuBuffer> = OnceLock::new();

        let kernel = MEMSET_KERNEL.get_or_init(|| {
            let mut asm = crate::rdna3_asm::Rdna3Assembler::new();

            // Kernarg: [ptr: u64(0), n_dwords: u32(8), pad: u32(12)]
            // SGPR: s[0:1]=kernarg_ptr, s2=workgroup_id_x (TGID_X_EN=1)
            // Load kernargs into s[4:7]
            let words = gfx11::s_load_dwordx4(4, 0, 0);
            asm.emit(words[0]); asm.emit(words[1]);
            asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

            // global_id = workgroup_id_x * 256 + thread_id
            asm.emit(gfx11::v_mov_b32_from_sgpr(1, 2));  // v1 = wg_id_x
            asm.emit(gfx11::v_lshlrev_b32(1, 8, 1));     // v1 *= 256
            let add = gfx11::v_add_co_u32_vcc(1, 1, 0);  // v1 += v0
            asm.emit(add[0]); asm.emit(add[1]);

            // Bounds: if global_id * 4 >= n_dwords → mask off
            asm.emit(gfx11::v_lshlrev_b32(2, 2, 1));     // v2 = global_id * 4
            asm.emit(gfx11::v_mov_b32_from_sgpr(3, 6));   // v3 = n_dwords
            asm.emit(gfx11::v_cmp_lt_u32(2, 3));          // vcc = v2 < v3
            asm.emit(gfx11::s_and_saveexec_b32_vcc(8));   // mask off OOB lanes

            // addr = ptr + global_id * 16
            asm.emit(gfx11::v_lshlrev_b32(4, 4, 1));     // v4 = byte offset
            asm.emit(gfx11::v_mov_b32_from_sgpr(5, 4));   // v5 = ptr_lo
            asm.emit(gfx11::v_mov_b32_from_sgpr(6, 5));   // v6 = ptr_hi
            let al = gfx11::v_add_co_u32_vcc(5, 5, 4);
            asm.emit(al[0]); asm.emit(al[1]);
            let ah = gfx11::v_add_co_ci_u32_zero_vcc(6, 6);
            asm.emit(ah[0]); asm.emit(ah[1]);

            // Write 16 bytes of zeros
            asm.emit(gfx11::v_mov_b32_imm(10, 0));
            asm.emit(gfx11::v_mov_b32_imm(11, 0));
            asm.emit(gfx11::v_mov_b32_imm(12, 0));
            asm.emit(gfx11::v_mov_b32_imm(13, 0));
            asm.global_store_dwordx4(5, 10, 0);

            asm.emit(gfx11::s_waitcnt_vmcnt(0));
            asm.emit(gfx11::s_waitcnt_vscnt(0));
            asm.emit(gfx11::S_ENDPGM);

            let co = AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
                name: "gpu_memset_zero".into(),
                lds_size: 0, kernarg_size: 16,
                vgpr_count: 16, sgpr_count: 16,
                workgroup_size_x: 256, workgroup_size_y: 1, workgroup_size_z: 1,
                scratch_size: 0,
            });
            let hsaco = co.to_code_object_llvm().expect("gpu_memset LLVM build");
            // Use the device from the buffer's allocation context
            GpuKernel::load(&buf.device, &hsaco, &KernelLoadConfig {
                lds_size: 0,
                workgroup_size: [256, 1, 1],
            }).expect("gpu_memset kernel load")
        });

        // Prepare kernargs: [ptr(8), n_dwords(4), pad(4)]
        let n_dwords = (buf.size / 4) as u32;
        let _ka_buf = MEMSET_KA_BUF.get_or_init(|| {
            buf.device.alloc_uncached(256).expect("memset ka buf")
        });
        let ka_buf = MEMSET_KA_BUF.get().unwrap();

        // Write kernargs directly
        let mut ka_data = [0u8; 16];
        ka_data[0..8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
        ka_data[8..12].copy_from_slice(&n_dwords.to_le_bytes());
        ka_buf.write(&ka_data);

        // Grid: ceil(n_dwords / 4 / 256) * 256 threads
        let threads_needed = ((n_dwords as usize + 3) / 4 + 255) / 256 * 256;
        let grid = [threads_needed as u32, 1, 1];

        self.submit(kernel, grid, ka_buf);
    }

    // =========================================================================
    // PM4 hardware synchronization primitives (Mega-IB pipeline)
    // =========================================================================

    /// Submit pure PM4 commands wrapped in a VENDOR_SPECIFIC AQL packet.
    /// Used for compute_barrier() + release_mem() synchronization
    /// that replaces the old wait_idle() pattern.
    pub fn submit_pm4(&mut self, pm4_cmds: &[u32]) -> Result<(), String> {
        if pm4_cmds.is_empty() {
            return Ok(());
        }
        // Ensure PM4 IB buffer is allocated
        if self.pm4_ib.is_none() {
            self.pm4_ib = Some(self.device.alloc_uncached(PM4_IB_SIZE)?);
            self.pm4_ib_offset = 0;
        }
        let pm4_byte_size = pm4_cmds.len() * 4;
        if self.pm4_ib_offset + pm4_byte_size > PM4_IB_SIZE {
            self.pm4_ib_offset = 0; // wrap around
        }
        let ib = self.pm4_ib.as_ref().unwrap();
        let ib_byte_offset = self.pm4_ib_offset;

        // Write PM4 commands to IB
        unsafe {
            let dst = ib.host_ptr.add(ib_byte_offset) as *mut u32;
            for (i, &dword) in pm4_cmds.iter().enumerate() {
                std::ptr::write_volatile(dst.add(i), dword);
            }
            // Read-back to flush write-combine buffers
            let _ = std::ptr::read_volatile(dst.add(pm4_cmds.len() - 1));
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let ib_va = ib.gpu_addr() + ib_byte_offset as u64;
        self.pm4_ib_offset += pm4_byte_size;

        // Write VENDOR_SPECIFIC AQL packet
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let slot = write_idx & ring_mask;
        let base = unsafe { self.ring_buffer.host_ptr.add((slot * 64) as usize) };

        // Header: barrier=1 (wait for prior compute), fence_scope=SYSTEM
        let header: u16 = HSA_PACKET_TYPE_VENDOR_SPECIFIC |
            (1 << 8) |  // barrier
            (HSA_FENCE_SCOPE_SYSTEM << 9) |
            (HSA_FENCE_SCOPE_SYSTEM << 11);

        // IB PACKET3 command: INDIRECT_BUFFER pointing to our PM4 commands
        let ib_pkt3 = (3u32 << 30) |
            (((3u32 - 1) & 0x3FFF) << 16) |  // 3 body dwords for IB: addr_lo, addr_hi, size|valid
            (PACKET3_INDIRECT_BUFFER << 8);

        unsafe {
            std::ptr::write_volatile(base as *mut u16, 1u16); // INVALID first
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_bytes(base.add(2), 0, 62);
            // VS packet layout: [0]=header, [2]=?, [4]=IB_pkt3, [8]=addr_lo, [12]=addr_hi, [16]=size|valid
            std::ptr::write_volatile(base.add(2)  as *mut u16, 1u16);
            std::ptr::write_volatile(base.add(4)  as *mut u32, ib_pkt3);
            std::ptr::write_volatile(base.add(8)  as *mut u32, ib_va as u32);
            std::ptr::write_volatile(base.add(12) as *mut u32, (ib_va >> 32) as u32);
            std::ptr::write_volatile(base.add(16) as *mut u32,
                pm4_cmds.len() as u32 | INDIRECT_BUFFER_VALID);
            std::ptr::write_volatile(base.add(20) as *mut u32, 10u32); // padding
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(base as *mut u16, header); // header LAST (atomically valid)

            // Ring doorbell — CRITICAL: write new_write_idx (NOT -1) to wake CP
            // after queue drain. When read_ptr == old_write_ptr, doorbell must be 
            // strictly greater than what MEC last consumed.
            let new_write_idx = write_idx + 1;
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
        Ok(())
    }

    /// Lock-free VRAM polling: wait for GPU to write seqno >= target via RELEASE_MEM.
    /// Replaces wait_idle() — never drains the AQL queue, avoids CP sleep/wakeup bug.
    pub fn wait_vram_seqno(sync_buf: &GpuBuffer, target_seqno: u32) -> Result<(), String> {
        let ptr = sync_buf.host_ptr as *const std::sync::atomic::AtomicU32;
        let atomic_ptr = unsafe { &*ptr };
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);
        loop {
            let current = atomic_ptr.load(std::sync::atomic::Ordering::Acquire);
            if current >= target_seqno {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(format!(
                    "VRAM seqno timeout: expected >= {}, got {} after {:?}",
                    target_seqno, current, start.elapsed()
                ));
            }
            std::hint::spin_loop();
        }
    }

    // =========================================================================
    // PM4-in-AQL: embed PM4 PACKET3 commands inside AQL VENDOR_SPECIFIC packets
    // =========================================================================

    /// Dispatch a kernel via PM4-in-AQL hybrid approach.
    ///
    /// Submits TWO AQL packets to the queue:
    ///   1. VENDOR_SPECIFIC: Contains INDIRECT_BUFFER with register setup PM4
    ///      (ACQUIRE_MEM + SET_SH_REG for all compute registers)
    ///   2. KERNEL_DISPATCH: Native AQL dispatch packet for actual kernel launch
    ///      (Uses MES's proven dispatch path + amd_signal_t completion)
    ///
    /// This hybrid approach is more reliable than pure PM4 dispatch because:
    /// - Register setup uses PM4 for fine-grained control (matching tinygrad)
    /// - Kernel launch uses native AQL for proper MES scheduling and completion
    /// - No IB-reuse race: VENDOR_SPECIFIC IB only contains register writes
    pub fn dispatch_pm4(
        &mut self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
        signal: Option<&GpuBuffer>,
    ) -> Result<(), String> {
        // Ensure PM4 IB buffer is allocated
        if self.pm4_ib.is_none() {
            self.pm4_ib = Some(self.device.alloc_uncached(PM4_IB_SIZE)?);
            self.pm4_ib_offset = 0;
        }

        // ── Step 1: Build register setup PM4 commands ───────────────────────
        let mut pm4 = Pm4CmdBuilder::new();

        // ACQUIRE_MEM — invalidate GPU caches (GFX10+ format with GCR_CNTL)
        pm4.acquire_mem_gfx10();

        // Set shader program address: COMPUTE_PGM_LO / COMPUTE_PGM_HI
        pm4.set_sh_reg(REG_COMPUTE_PGM_LO, &[
            (kernel.code_entry_va >> 8) as u32,
            (kernel.code_entry_va >> 40) as u32,
        ]);

        // Set RSRC1/RSRC2 (force rsrc1.priv=1 on GFX11 to workaround CWSR — tinygrad pattern)
        pm4.set_sh_reg(REG_COMPUTE_PGM_RSRC1, &[
            kernel.rsrc1 | (1 << 20),  // rsrc1.priv = 1 (GFX11 CWSR workaround)
            kernel.rsrc2,
        ]);

        // Set RSRC3 (GFX11 specific — wave slots / scratch)
        pm4.set_sh_reg(REG_COMPUTE_PGM_RSRC3, &[0]);

        // Set TMPRING_SIZE
        pm4.set_sh_reg(REG_COMPUTE_TMPRING_SIZE, &[0]);

        // Set RESTART_X/Y/Z = 0,0,0
        pm4.set_sh_reg(REG_COMPUTE_RESTART_X, &[0, 0, 0]);

        // Set USER_DATA_0 = kernargs pointer (lo, hi)
        pm4.set_sh_reg(REG_COMPUTE_USER_DATA_0, &[
            kernargs.gpu_addr() as u32,
            (kernargs.gpu_addr() >> 32) as u32,
        ]);

        // Set RESOURCE_LIMITS = 0
        pm4.set_sh_reg(REG_COMPUTE_RESOURCE_LIMITS, &[0]);

        // Set START_X/Y/Z = 0,0,0 + workgroup sizes (NUM_THREAD_X/Y/Z) + padding
        pm4.set_sh_reg(REG_COMPUTE_START_X, &[
            0, 0, 0,                        // start x, y, z
            kernel.workgroup_size[0],        // num_thread_x
            kernel.workgroup_size[1],        // num_thread_y
            kernel.workgroup_size[2],        // num_thread_z
            0, 0,                            // padding
        ]);

        let pm4_cmds = pm4.finish();

        // ── Step 2: Write PM4 register setup to IB buffer ───────────────────
        let ib_byte_offset = self.pm4_ib_offset;
        let pm4_byte_size = pm4_cmds.len() * 4;

        if ib_byte_offset + pm4_byte_size > PM4_IB_SIZE {
            self.pm4_ib_offset = 0; // wrap around
        }
        let ib_byte_offset = self.pm4_ib_offset;
        let ib = self.pm4_ib.as_ref().unwrap();

        unsafe {
            let dst = ib.host_ptr.add(ib_byte_offset) as *mut u32;
            for (i, &dword) in pm4_cmds.iter().enumerate() {
                std::ptr::write_volatile(dst.add(i), dword);
            }
            // Read-back to flush write-combine buffers before ringing doorbell
            let _ = std::ptr::read_volatile(dst.add(pm4_cmds.len() - 1));
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let ib_va = ib.gpu_addr() + ib_byte_offset as u64;
        self.pm4_ib_offset += pm4_byte_size;

        // ── Step 3+4: Write VS (reg setup) and KD (kernel launch) together ──
        // Pre-write both AQL slots, ring doorbell once for KD, wait for KD completion.
        // This avoids intermediate wait on VS read_ptr which can stall after 12+ packets.
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let vs_write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let kd_write_idx = vs_write_idx + 1;

        // ── Write VENDOR_SPECIFIC slot ────────────────────────────────────────
        let vs_slot = vs_write_idx & ring_mask;
        let vs_base = unsafe { self.ring_buffer.host_ptr.add((vs_slot * 64) as usize) };

        let vs_aql_hdr: u16 =
            HSA_PACKET_TYPE_VENDOR_SPECIFIC |
            (1 << 8) |
            (HSA_FENCE_SCOPE_SYSTEM << 9) |
            (HSA_FENCE_SCOPE_SYSTEM << 11);

        let ib_pkt3 = (3u32 << 30) | ((2u32 & 0x3FFF) << 16) | (PACKET3_INDIRECT_BUFFER << 8);

        unsafe {
            std::ptr::write_volatile(vs_base as *mut u16, 1u16);  // INVALID first
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_bytes(vs_base.add(2), 0, 62);
            std::ptr::write_volatile(vs_base.add(2)  as *mut u16, 1u16);
            std::ptr::write_volatile(vs_base.add(4)  as *mut u32, ib_pkt3);
            std::ptr::write_volatile(vs_base.add(8)  as *mut u32, ib_va as u32);
            std::ptr::write_volatile(vs_base.add(12) as *mut u32, (ib_va >> 32) as u32);
            std::ptr::write_volatile(vs_base.add(16) as *mut u32,
                pm4_cmds.len() as u32 | INDIRECT_BUFFER_VALID);
            std::ptr::write_volatile(vs_base.add(20) as *mut u32, 10u32);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(vs_base as *mut u16, vs_aql_hdr); // header LAST
        }

        // ── Write KERNEL_DISPATCH slot ────────────────────────────────────────
        let kd_slot = kd_write_idx & ring_mask;
        let kd_base = unsafe { self.ring_buffer.host_ptr.add((kd_slot * 64) as usize) };

        let signal_va = if let Some(sig) = signal {
            unsafe { std::ptr::write_bytes(sig.host_ptr, 0, 64); }
            sig.write_val::<u64>(0, 1);
            sig.write_val::<i64>(8, 1);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            sig.gpu_addr()
        } else {
            0u64
        };

        let kd_hdr: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        unsafe {
            std::ptr::write_volatile(kd_base.add(0x00) as *mut u16, 1u16);  // INVALID first
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(kd_base.add(0x02) as *mut u16, 3u16);
            std::ptr::write_volatile(kd_base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(kd_base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(kd_base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(kd_base.add(0x0A) as *mut u16, 0u16);
            std::ptr::write_volatile(kd_base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(kd_base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(kd_base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(kd_base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(kd_base.add(0x1C) as *mut u32, kernel.lds_size);
            std::ptr::write_volatile(kd_base.add(0x20) as *mut u64, kernel.descriptor_va);
            std::ptr::write_volatile(kd_base.add(0x28) as *mut u64, kernargs.gpu_addr());
            std::ptr::write_volatile(kd_base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(kd_base.add(0x38) as *mut u64, signal_va);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(kd_base.add(0x00) as *mut u16, kd_hdr); // header LAST
        }

        // ── Ring doorbell: update write_ptr to cover BOTH VS+KD ──────────────
        let new_write_idx = kd_write_idx + 1;
        unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            // tinygrad: write_ptr = put_value, doorbell = put_value - 1
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }

        // ── Wait for kernel completion ────────────────────────────────────────
        if let Some(sig) = signal {
            self.wait_signal(sig)?;
        } else {
            // No signal: poll read_ptr until both VS+KD consumed
            let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
            self.wait_read_ptr(target)?;
        }

        Ok(())
    }
}


// =============================================================================
// Pm4CmdBuilder — builds PM4 PACKET3 command sequences
// =============================================================================


/// PM4 command builder for constructing PACKET3 sequences
pub struct Pm4CmdBuilder {
    cmds: Vec<u32>,
}

impl Pm4CmdBuilder {
    pub fn new() -> Self {
        Self { cmds: Vec::with_capacity(64) }
    }

    /// Emit PACKET3: [31:30]=type3, [29:16]=count-1, [15:8]=opcode
    /// AMD PM4 spec: count field = number of body dwords - 1
    fn pkt3(&mut self, opcode: u32, body: &[u32]) {
        let header = (3u32 << 30) | (((body.len() as u32 - 1) & 0x3FFF) << 16) | (opcode << 8);
        self.cmds.push(header);
        self.cmds.extend_from_slice(body);
    }

    /// SET_SH_REG: write consecutive shader registers
    pub fn set_sh_reg(&mut self, reg_addr: u32, values: &[u32]) {
        let reg_offset = (reg_addr - SH_REG_BASE) >> 2;
        let mut body = Vec::with_capacity(1 + values.len());
        body.push(reg_offset);
        body.extend_from_slice(values);
        self.pkt3(PM4_SET_SH_REG, &body);
    }

    /// ACQUIRE_MEM for GFX10+ (with GCR_CNTL for cache invalidation)
    pub fn acquire_mem_gfx10(&mut self) {
        // GFX10+ ACQUIRE_MEM format: 7 body dwords
        // [0] = 0
        // [1:2] = coherence size (u64, all memory)
        // [3:4] = coherence base (u64, 0)
        // [5] = poll interval (0)
        // [6] = GCR_CNTL flags (invalidate all caches)
        let gcr_cntl: u32 =
            (1 << 0)  |  // GLI_INV
            (1 << 1)  |  // GLM_INV
            (1 << 2)  |  // GLM_WB
            (1 << 3)  |  // GLK_INV
            (1 << 4)  |  // GLK_WB
            (1 << 5)  |  // GLV_INV
            (1 << 6)  |  // GL1_INV
            (1 << 7)  |  // GL2_INV
            (1 << 8);    // GL2_WB
        self.pkt3(PM4_ACQUIRE_MEM, &[
            0,                  // cp_coher_cntl
            0xFFFF_FFFF, 0xFF,  // coherence size (large)
            0, 0,               // coherence base
            0,                  // poll interval
            gcr_cntl,           // GCR cache control flags
        ]);
    }

    /// DISPATCH_DIRECT: launch workgroups
    pub fn dispatch_direct(&mut self, grid: [u32; 3], dispatch_initiator: u32) {
        self.pkt3(PM4_DISPATCH_DIRECT, &[grid[0], grid[1], grid[2], dispatch_initiator]);
    }

    /// EVENT_WRITE: various GPU events
    pub fn event_write(&mut self, event_type: u32, event_index: u32) {
        let event_dw = event_type | (event_index << 8);
        self.pkt3(PM4_EVENT_WRITE, &[event_dw]);
    }

    /// RELEASE_MEM: write value to memory upon pipeline completion
    pub fn release_mem(&mut self, addr: u64, value: u32, data_sel: u32, int_sel: u32, cache_flush: bool) {
        let cache_flags = if cache_flush {
            (1 << 12) |  // GLV_INV
            (1 << 13) |  // GL1_INV
            (1 << 14) |  // GL2_INV
            (1 << 15) |  // GLM_WB
            (1 << 16) |  // GLM_INV
            (1 << 17) |  // GL2_WB
            (1 << 18)    // SEQ
        } else {
            0
        };
        let event_dw = CACHE_FLUSH_AND_INV_TS_EVENT | (5 << 8) | cache_flags; // event_index=5 for MEC end_of_pipe
        let data_dw = (data_sel << 29) | (int_sel << 24);
        self.pkt3(PM4_RELEASE_MEM, &[
            event_dw,
            data_dw,
            addr as u32,
            (addr >> 32) as u32,
            value,
            0,   // ctxid
            0,   // padding
        ]);
    }

    /// WRITE_DATA: write 32-bit value to GPU-visible memory address
    /// Opcode 0x37. dst_sel=2 (memory mapped via TC L2), wr_confirm=1, engine=ME
    pub fn write_data(&mut self, addr: u64, value: u32) {
        const PM4_WRITE_DATA: u32 = 0x37;
        // dst_sel=2 = "memory mapped" (used by tinygrad COPY_DATA with (2<<8)|4)
        // wr_confirm=bit20: wait for write to be acknowledged before proceeding
        // engine=MEC(1<<30): use MEC engine for compute
        let control_dw: u32 = (2 << 8) | (1 << 20);
        self.pkt3(PM4_WRITE_DATA, &[
            control_dw,
            addr as u32,
            (addr >> 32) as u32,
            value,
        ]);
    }

    /// Finish and return the PM4 dword sequence
    pub fn finish(self) -> Vec<u32> {
        self.cmds
    }

    /// Strict compute barrier: wait for all prior dispatches to complete + flush L1/L2 caches.
    /// PM4 equivalent of AQL barrier=1. Use between dependent kernel dispatches in an IB.
    pub fn compute_barrier(&mut self) {
        // 1. Wait for all compute shaders to finish
        self.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH);
        // 2. Flush/invalidate all GPU caches (same flags as acquire_mem_gfx10)
        let gcr_cntl: u32 =
            (1 << 0)  |  // GLI_INV
            (1 << 1)  |  // GLM_INV
            (1 << 2)  |  // GLM_WB
            (1 << 3)  |  // GLK_INV
            (1 << 4)  |  // GLK_WB
            (1 << 5)  |  // GLV_INV
            (1 << 6)  |  // GL1_INV
            (1 << 7)  |  // GL2_INV
            (1 << 8);    // GL2_WB
        self.pkt3(PM4_ACQUIRE_MEM, &[
            0,                  // cp_coher_cntl
            0xFFFF_FFFF, 0xFF,  // coherence size
            0, 0,               // coherence base
            0,                  // poll interval
            gcr_cntl,           // GCR flags
        ]);
    }
}

impl Drop for AqlQueue {
    fn drop(&mut self) {
        // Drain queue: wait for all in-flight dispatches to complete (500ms timeout)
        // This prevents DESTROY_QUEUE from killing active GPU work, which can
        // leave the KFD driver in a dirty state for the next KfdDevice::open().
        let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(500);
        loop {
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target { break; }
            if start.elapsed() > timeout {
                eprintln!("[KFD] WARNING: AqlQueue {} drain timeout (read={} target={})",
                    self.queue_id, read_idx, target);
                break;
            }
            std::hint::spin_loop();
        }
        let mut args = KfdDestroyQueueArgs { queue_id: self.queue_id, pad: 0 };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_DESTROY_QUEUE,
            &mut args as *mut _ as *mut u8);
        // Unmap doorbell — must use original mmap base address and size!
        // doorbell_ptr is offset within the mmap region, NOT the mmap return value.
        unsafe { munmap(self.doorbell_mmap_base, self.doorbell_mmap_size); }
    }
}

// =============================================================================
// Pm4Queue — PM4 compute queue with raw PACKET3 dispatch
// =============================================================================

/// PM4 hardware compute queue (uses PACKET3 commands, not AQL)
pub struct Pm4Queue {
    pub queue_id: u32,
    pub ring_buffer: GpuBuffer,
    pub ring_size: u32,
    pub write_ptr_host: *mut u64,
    pub read_ptr_host: *mut u64,
    pub doorbell_ptr: *mut u64, // KFD doorbell is u64 (same as AQL)
    pub write_offset: u32,      // current byte offset into ring
    /// Original mmap base for doorbell (needed for correct munmap)
    doorbell_mmap_base: *mut u8,
    pub _wr_ptrs: GpuBuffer,
    pub _eop_buffer: GpuBuffer,
    pub _cwsr_buffer: Option<GpuBuffer>,
    pub device: Arc<KfdDevice>,
}

unsafe impl Send for Pm4Queue {}
unsafe impl Sync for Pm4Queue {}

// PM4 PACKET3 opcodes
const PACKET3_SET_SH_REG: u32        = 0x76;
const PACKET3_DISPATCH_DIRECT: u32   = 0x15;
const PACKET3_RELEASE_MEM: u32       = 0x49;
const PACKET3_ACQUIRE_MEM: u32       = 0x58;

// GFX11 Compute SH register offsets (relative to 0x2C00)
const COMPUTE_PGM_LO: u32          = 0x2C0C;
const COMPUTE_PGM_HI: u32          = 0x2C10;
const COMPUTE_PGM_RSRC1: u32       = 0x2C44;
const COMPUTE_PGM_RSRC2: u32       = 0x2C48;
const COMPUTE_USER_DATA_0: u32     = 0x2C4C;
const COMPUTE_NUM_THREAD_X: u32    = 0x2C78;
const COMPUTE_NUM_THREAD_Y: u32    = 0x2C7C;
const COMPUTE_NUM_THREAD_Z: u32    = 0x2C80;
const COMPUTE_RESOURCE_LIMITS: u32 = 0x2C14;

impl Pm4Queue {
    /// Write a u32 dword to the ring buffer at current offset
    fn emit(&mut self, dword: u32) {
        let off = (self.write_offset as usize) % self.ring_size as usize;
        unsafe {
            let ptr = self.ring_buffer.host_ptr.add(off) as *mut u32;
            std::ptr::write_volatile(ptr, dword);
        }
        self.write_offset += 4;
    }

    /// Emit PACKET3 header: [31:30]=type3, [29:16]=body_dword_count, [15:8]=opcode
    /// AMD spec: count field = number of body dwords (NOT -1)
    /// tinygrad pkt3(op, *args): op | len(args)<<16  — len(args) = body dwords
    fn emit_packet3(&mut self, opcode: u32, body_dwords: u32) {
        let header = (3u32 << 30) | ((body_dwords & 0x3FFF) << 16) | (opcode << 8);
        self.emit(header);
    }

    /// SET_SH_REG: write consecutive registers starting at reg_addr
    fn emit_set_sh_reg(&mut self, reg_addr: u32, values: &[u32]) {
        let reg_offset = (reg_addr - 0x2C00) >> 2; // convert to dword offset from SH base
        self.emit_packet3(PACKET3_SET_SH_REG, values.len() as u32 + 1);
        self.emit(reg_offset);
        for &v in values {
            self.emit(v);
        }
    }

    /// DISPATCH_DIRECT: launch compute workgroups
    fn emit_dispatch_direct(&mut self, dim_x: u32, dim_y: u32, dim_z: u32) {
        self.emit_packet3(PACKET3_DISPATCH_DIRECT, 4);
        self.emit(dim_x);
        self.emit(dim_y);
        self.emit(dim_z);
        // dispatch_initiator: bit 0 = compute_shader_en
        self.emit(1u32);
    }

    /// RELEASE_MEM: write a value to memory after all prior work completes
    /// Used for completion signaling
    fn emit_release_mem(&mut self, dst_addr: u64, value: u64) {
        self.emit_packet3(PACKET3_RELEASE_MEM, 6);
        // event_cntl: EOP event, cache policy
        // [5:0] = event_type = 0x2F (BOTTOM_OF_PIPE_TS for compute)
        // [11:8] = event_index = 5 (write confirmation)
        let event_cntl = 0x2F | (5 << 8);
        self.emit(event_cntl);
        // data_cntl: [28:26]=data_sel(1=32bit,2=64bit,3=timestamp), [24:22]=int_sel
        // data_sel=2 (send 64-bit data), int_sel=0 (no interrupt)
        let data_cntl = 2 << 26;
        self.emit(data_cntl);
        // dst address (low, high)
        self.emit(dst_addr as u32);
        self.emit((dst_addr >> 32) as u32);
        // data (low, high)
        self.emit(value as u32);
        self.emit((value >> 32) as u32);
    }

    /// ACQUIRE_MEM: invalidate GPU caches (L1/L2) to see fresh data
    fn emit_acquire_mem(&mut self) {
        self.emit_packet3(PACKET3_ACQUIRE_MEM, 6);
        self.emit(0); // cp_coher_cntl (all caches)
        self.emit(0xFFFFFFFF); // coher_size (everything)
        self.emit(0); // coher_size_hi
        self.emit(0); // coher_base
        self.emit(0); // coher_base_hi
        self.emit(0); // poll_interval
    }

    /// Dispatch a kernel via PM4 commands
    pub fn dispatch_nop(&mut self, code_addr: u64, rsrc1: u32, rsrc2: u32,
                        wg_size: [u32; 3], grid: [u32; 3],
                        signal_buf: Option<&GpuBuffer>) -> Result<(), String> {
        // Record start offset
        self.write_offset = unsafe { std::ptr::read_volatile(self.write_ptr_host) } as u32 * 4;

        // 1. Acquire memory (flush GPU caches)
        self.emit_acquire_mem();

        // 2. Set shader program address (code_addr >> 8 because COMPUTE_PGM_LO only stores top bits)
        self.emit_set_sh_reg(COMPUTE_PGM_LO, &[
            (code_addr >> 8) as u32,  // PGM_LO
            (code_addr >> 40) as u32, // PGM_HI
        ]);

        // 3. Set resource limits (allow 1 CU)
        self.emit_set_sh_reg(COMPUTE_RESOURCE_LIMITS, &[0]);

        // 4. Set rsrc1/rsrc2
        self.emit_set_sh_reg(COMPUTE_PGM_RSRC1, &[rsrc1, rsrc2]);

        // 5. Set workgroup dimensions
        self.emit_set_sh_reg(COMPUTE_NUM_THREAD_X, &[wg_size[0], wg_size[1], wg_size[2]]);

        // 6. Dispatch
        self.emit_dispatch_direct(grid[0], grid[1], grid[2]);

        // 7. Release mem (signal completion)
        if let Some(sig) = signal_buf {
            self.emit_release_mem(sig.gpu_addr(), 0x12345678DEADBEEF);
        }

        // 8. Dump ring buffer for debugging
        eprintln!("[PM4] Ring buffer ({} dwords, {} bytes):", self.write_offset / 4, self.write_offset);
        for i in 0..(self.write_offset / 4) {
            let dword = unsafe {
                let ptr = self.ring_buffer.host_ptr.add(i as usize * 4) as *const u32;
                std::ptr::read_volatile(ptr)
            };
            if i % 4 == 0 { eprint!("  [{:3}]:", i); }
            eprint!(" {:08X}", dword);
            if i % 4 == 3 || i == self.write_offset / 4 - 1 { eprintln!(); }
        }

        // 9. Ring doorbell
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let new_wptr = self.write_offset / 4; // convert byte offset to dword count
        eprintln!("[PM4] Ringing doorbell with wptr={}", new_wptr);
        unsafe {
            std::ptr::write_volatile(self.write_ptr_host, new_wptr as u64);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.doorbell_ptr, new_wptr as u64);
        }

        // 9. Wait for completion
        if let Some(sig) = signal_buf {
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(5);
            loop {
                let val: u64 = sig.read_val(0);
                if val == 0x12345678DEADBEEF {
                    return Ok(());
                }
                if start.elapsed() > timeout {
                    let rp = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
                    return Err(format!("PM4 timeout: signal=0x{:X} rp={} wp={}", val, rp, new_wptr));
                }
                std::hint::spin_loop();
            }
        } else {
            // Poll read pointer
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(5);
            loop {
                let rp = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
                if rp >= new_wptr as u64 {
                    return Ok(());
                }
                if start.elapsed() > timeout {
                    return Err(format!("PM4 read_ptr timeout: rp={} wp={}", rp, new_wptr));
                }
                std::hint::spin_loop();
            }
        }
    }
}

impl Drop for Pm4Queue {
    fn drop(&mut self) {
        // Drain queue: wait for read_ptr to catch up with write_ptr (500ms timeout)
        let target_bytes = self.write_offset;
        if target_bytes > 0 {
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_millis(500);
            loop {
                let rp = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
                if rp >= target_bytes as u64 { break; }
                if start.elapsed() > timeout {
                    eprintln!("[KFD] WARNING: Pm4Queue {} drain timeout (read={} target={})",
                        self.queue_id, rp, target_bytes);
                    break;
                }
                std::hint::spin_loop();
            }
        }
        let mut args = KfdDestroyQueueArgs { queue_id: self.queue_id, pad: 0 };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_DESTROY_QUEUE,
            &mut args as *mut _ as *mut u8);
        // Unmap doorbell — use original mmap base, not the offset pointer
        unsafe { munmap(self.doorbell_mmap_base, 8192); }
    }
}

// =============================================================================
// GpuKernel — loaded kernel ready for dispatch
// =============================================================================

/// A GPU kernel loaded into VRAM, ready for AQL dispatch
pub struct GpuKernel {
    pub code_buffer: GpuBuffer,   // machine code + descriptor in VRAM
    pub descriptor_va: u64,       // GPU VA of 64-byte kernel descriptor
    pub code_entry_va: u64,       // GPU VA of actual code entry point
    pub rsrc1: u32,               // COMPUTE_PGM_RSRC1 (with PRIV bit set)
    pub rsrc2: u32,               // COMPUTE_PGM_RSRC2
    pub lds_size: u32,
    pub workgroup_size: [u32; 3],
    /// Kernel argument size in bytes (from kernel descriptor).
    /// Used for dispatch-time validation.
    pub kernarg_size: u32,
}

impl GpuKernel {
    /// Load a kernel from HSACO ELF bytes (produced by rdna3_code_object)
    ///
    /// Extracts .text + kernel descriptor from ELF, uploads to executable VRAM,
    /// and patches the kernel descriptor's PRIV bit and code entry offset.
    pub fn load(device: &Arc<KfdDevice>, hsaco: &[u8], config: &KernelLoadConfig) -> Result<Self, String> {
        // Parse ELF to find .text section and kernel descriptor (.kd symbol)
        let elf = ElfParser::parse(hsaco)?;

        // The HSACO contains .text (machine code) preceded/followed by kernel descriptor.
        // We load the entire LOAD segment into executable VRAM.
        // The kernel descriptor's `kernel_code_entry_byte_offset` already has the
        // correct relative offset from the LLVM linker.
        
        // Find the loadable content (everything between first LOAD phdr)
        let load_data = elf.loadable_content(hsaco)?;
        let kd_offset = elf.kernel_descriptor_offset()?;

        // Allocate executable VRAM and upload
        let code_buf = device.alloc_code(load_data.len())?;
        code_buf.write(&load_data);

        // PCIe read barrier: force HDP cache flush
        // Reading back one byte forces PCIe write-combine buffer to drain,
        // ensuring GPU's SQC (instruction cache) sees the latest code.
        // The AQL header's HSA_ACQUIRE_SYSTEM fence will also invalidate L1i/L2.
        let _ = unsafe { std::ptr::read_volatile(code_buf.host_ptr) };

        // Patch kernel descriptor's compute_pgm_rsrc1 (KD offset 0x30):
        //   bit 20: PRIV — Required for KFD bare-metal dispatch (CWSR context save/restore)
        //
        // NOTE on WGP mode (2026-03-31 investigation):
        //   LLVM's .amdhsa_workgroup_processor_mode directive sets RSRC1 bit 29
        //   (ENABLE_WGP_MODE) directly in the ELF. The hardware CP reads WGP mode
        //   from RSRC1 bit 29, NOT bit 27. We previously tried to propagate from
        //   KCP bit 10, but KCP bit 10 is actually USES_DYNAMIC_STACK (Code Object V5),
        //   not WGP mode. WGP mode has no KCP bit — it lives only in RSRC1 bit 29.
        //
        //   Therefore: trust LLVM's RSRC1 bit 29, do NOT override it.
        //   Only patch bit 20 (PRIV) which LLVM does not set.
        //
        // Reference: tinygrad desc.compute_pgm_rsrc1 |= (1 << 20)
        let (rsrc1, rsrc2, entry_offset);
        unsafe {
            let kd_host_ptr = code_buf.host_ptr.add(kd_offset);
            // Debug: dump first 64 bytes of KD
            let kd_bytes = std::slice::from_raw_parts(kd_host_ptr, 64);
            eprintln!("[KFD] KD at offset {} (0x{:X}) in code buffer:", kd_offset, kd_offset);
            for row in 0..4 {
                let off = row * 16;
                eprint!("  {:02X}:", off);
                for i in 0..16 {
                    eprint!(" {:02X}", kd_bytes[off + i]);
                }
                eprintln!();
            }
            let rsrc1_ptr = kd_host_ptr.add(0x30) as *mut u32;
            let raw_rsrc1 = std::ptr::read_volatile(rsrc1_ptr);
            let patched_rsrc1 = raw_rsrc1 | (1 << 20); // PRIV bit only

            // Log WGP status from RSRC1 bit 29 (the real hardware bit)
            let wgp_on = (patched_rsrc1 >> 29) & 1 == 1;
            eprintln!("[KFD] RSRC1=0x{:08X} WGP_MODE(bit29)={} MEM_ORD(bit30)={} FWD(bit31)={}",
                patched_rsrc1,
                (patched_rsrc1 >> 29) & 1,
                (patched_rsrc1 >> 30) & 1,
                (patched_rsrc1 >> 31) & 1);

            std::ptr::write_volatile(rsrc1_ptr, patched_rsrc1);
            rsrc1 = patched_rsrc1;
            rsrc2 = std::ptr::read_volatile(kd_host_ptr.add(0x34) as *const u32);
            entry_offset = std::ptr::read_volatile(kd_host_ptr.add(0x10) as *const i64);
        }
        // Extract kernarg_size from kernel descriptor (offset 8)
        let kd_kernarg_size = unsafe {
            let kd_host_ptr = code_buf.host_ptr.add(kd_offset);
            std::ptr::read_volatile(kd_host_ptr.add(8) as *const u32)
        };
        // Re-flush HDP after patching
        let _ = unsafe { std::ptr::read_volatile(code_buf.host_ptr) };

        let descriptor_va = code_buf.gpu_addr() + kd_offset as u64;
        let code_entry_va = (descriptor_va as i64 + entry_offset) as u64;

        eprintln!("[KFD] Kernel loaded: desc_va=0x{:X} code_va=0x{:X} rsrc1=0x{:08X} rsrc2=0x{:08X}",
            descriptor_va, code_entry_va, rsrc1, rsrc2);

        Ok(GpuKernel {
            code_buffer: code_buf,
            descriptor_va,
            code_entry_va,
            rsrc1,
            rsrc2,
            lds_size: config.lds_size,
            workgroup_size: config.workgroup_size,
            kernarg_size: kd_kernarg_size,
        })
    }
}

/// Configuration for kernel loading
pub struct KernelLoadConfig {
    pub lds_size: u32,
    pub workgroup_size: [u32; 3],
}

// =============================================================================
// Minimal ELF parser for HSACO
// =============================================================================

struct LoadSegment {
    offset: usize,
    vaddr: u64,
    filesz: usize,
    memsz: usize,
}

struct ElfParser {
    text_offset: usize,
    text_size: usize,
    loads: Vec<LoadSegment>,
    min_vaddr: u64,
    total_memsz: usize,
    kd_offset_in_load: usize,
}

impl ElfParser {
    fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 64 || &data[0..4] != b"\x7fELF" {
            return Err("Not a valid ELF file".to_string());
        }

        // ELF64 header
        let e_phoff = u64::from_le_bytes(data[32..40].try_into().unwrap()) as usize;
        let e_shoff = u64::from_le_bytes(data[40..48].try_into().unwrap()) as usize;
        let e_phentsize = u16::from_le_bytes(data[54..56].try_into().unwrap()) as usize;
        let e_phnum = u16::from_le_bytes(data[56..58].try_into().unwrap()) as usize;
        let e_shentsize = u16::from_le_bytes(data[58..60].try_into().unwrap()) as usize;
        let e_shnum = u16::from_le_bytes(data[60..62].try_into().unwrap()) as usize;
        let e_shstrndx = u16::from_le_bytes(data[62..64].try_into().unwrap()) as usize;

        // Collect ALL PT_LOAD segments to compute the total loadable range
        let mut loads = Vec::new();
        for i in 0..e_phnum {
            let ph = e_phoff + i * e_phentsize;
            let p_type = u32::from_le_bytes(data[ph..ph+4].try_into().unwrap());
            if p_type == 1 { // PT_LOAD
                loads.push(LoadSegment {
                    offset: u64::from_le_bytes(data[ph+8..ph+16].try_into().unwrap()) as usize,
                    vaddr: u64::from_le_bytes(data[ph+16..ph+24].try_into().unwrap()),
                    filesz: u64::from_le_bytes(data[ph+32..ph+40].try_into().unwrap()) as usize,
                    memsz: u64::from_le_bytes(data[ph+40..ph+48].try_into().unwrap()) as usize,
                });
            }
        }
        if loads.is_empty() {
            return Err("No PT_LOAD segments found".to_string());
        }

        // Compute the total virtual address range spanning all LOAD segments
        let min_vaddr = loads.iter().map(|l| l.vaddr).min().unwrap();
        let max_vaddr_end = loads.iter().map(|l| l.vaddr + l.memsz as u64).max().unwrap();
        let total_memsz = (max_vaddr_end - min_vaddr) as usize;

        // Find .text section and symbols
        let shstr_hdr = e_shoff + e_shstrndx * e_shentsize;
        let shstr_off = u64::from_le_bytes(data[shstr_hdr+24..shstr_hdr+32].try_into().unwrap()) as usize;

        let mut text_offset = 0usize;
        let mut text_size = 0usize;
        let mut _text_vaddr = 0u64;
        let mut symtab_off = 0usize;
        let mut symtab_size = 0usize;
        let mut symtab_entsize = 0usize;
        let mut strtab_off = 0usize;

        for i in 0..e_shnum {
            let sh = e_shoff + i * e_shentsize;
            let sh_name_idx = u32::from_le_bytes(data[sh..sh+4].try_into().unwrap()) as usize;
            let sh_type = u32::from_le_bytes(data[sh+4..sh+8].try_into().unwrap());
            let sh_off = u64::from_le_bytes(data[sh+24..sh+32].try_into().unwrap()) as usize;
            let sh_size = u64::from_le_bytes(data[sh+32..sh+40].try_into().unwrap()) as usize;
            let sh_addr = u64::from_le_bytes(data[sh+16..sh+24].try_into().unwrap());

            let name_start = shstr_off + sh_name_idx;
            let name_end = data[name_start..].iter().position(|&b| b == 0)
                .map(|p| name_start + p).unwrap_or(name_start);
            let name = std::str::from_utf8(&data[name_start..name_end]).unwrap_or("");

            if name == ".text" {
                text_offset = sh_off;
                text_size = sh_size;
                _text_vaddr = sh_addr;
            } else if sh_type == 2 { // SHT_SYMTAB
                symtab_off = sh_off;
                symtab_size = sh_size;
                symtab_entsize = u64::from_le_bytes(data[sh+56..sh+64].try_into().unwrap()) as usize;
                let sh_link = u32::from_le_bytes(data[sh+40..sh+44].try_into().unwrap()) as usize;
                let strtab_sh = e_shoff + sh_link * e_shentsize;
                strtab_off = u64::from_le_bytes(data[strtab_sh+24..strtab_sh+32].try_into().unwrap()) as usize;
            } else if sh_type == 11 && symtab_entsize == 0 { // SHT_DYNSYM — fallback if no SHT_SYMTAB
                symtab_off = sh_off;
                symtab_size = sh_size;
                symtab_entsize = u64::from_le_bytes(data[sh+56..sh+64].try_into().unwrap()) as usize;
                let sh_link = u32::from_le_bytes(data[sh+40..sh+44].try_into().unwrap()) as usize;
                let strtab_sh = e_shoff + sh_link * e_shentsize;
                strtab_off = u64::from_le_bytes(data[strtab_sh+24..strtab_sh+32].try_into().unwrap()) as usize;
            }
        }

        if text_size == 0 {
            return Err("No .text section found in HSACO".to_string());
        }

        // Find kernel descriptor symbol (ends with .kd)
        let mut kd_vaddr = 0u64;
        if symtab_entsize > 0 {
            let num_syms = symtab_size / symtab_entsize;
            for i in 0..num_syms {
                let sym = symtab_off + i * symtab_entsize;
                let st_name = u32::from_le_bytes(data[sym..sym+4].try_into().unwrap()) as usize;
                let st_value = u64::from_le_bytes(data[sym+8..sym+16].try_into().unwrap());

                let name_start = strtab_off + st_name;
                let name_end = data[name_start..].iter().position(|&b| b == 0)
                    .map(|p| name_start + p).unwrap_or(name_start);
                let name = std::str::from_utf8(&data[name_start..name_end]).unwrap_or("");

                if name.ends_with(".kd") {
                    kd_vaddr = st_value;
                    break;
                }
            }
        }

        // KD offset in the merged virtual address space
        let kd_offset_in_load = if kd_vaddr >= min_vaddr {
            (kd_vaddr - min_vaddr) as usize
        } else {
            0
        };

        Ok(ElfParser {
            text_offset,
            text_size,
            loads,
            min_vaddr,
            total_memsz,
            kd_offset_in_load,
        })
    }

    /// Build a contiguous buffer spanning all PT_LOAD segments
    fn loadable_content(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; self.total_memsz];
        for seg in &self.loads {
            let dst_offset = (seg.vaddr - self.min_vaddr) as usize;
            let src_end = seg.offset + seg.filesz;
            if src_end > data.len() {
                return Err(format!("PT_LOAD segment exceeds file: offset={:#x} filesz={:#x} file_len={:#x}",
                    seg.offset, seg.filesz, data.len()));
            }
            buf[dst_offset..dst_offset + seg.filesz]
                .copy_from_slice(&data[seg.offset..src_end]);
        }
        Ok(buf)
    }

    fn kernel_descriptor_offset(&self) -> Result<usize, String> {
        Ok(self.kd_offset_in_load)
    }
}

// =============================================================================
// Convenience: helpers for kernel launch
// =============================================================================

impl KfdDevice {
    /// Prepare kernel arguments in a GPU buffer.
    /// For our kernels: typically 6 args = 4 pointers (8B each) + 2 u32s (4B each) = 40 bytes
    pub fn prepare_kernargs(self: &Arc<Self>, args_data: &[u8]) -> Result<GpuBuffer, String> {
        let buf = self.alloc_uncached(std::cmp::max(args_data.len(), 256))?;
        buf.write(args_data);
        Ok(buf)
    }

    /// Allocate a completion signal buffer (8 bytes, GTT, coherent)
    pub fn alloc_signal(self: &Arc<Self>) -> Result<GpuBuffer, String> {
        self.alloc_uncached(4096) // full page, uncached for GPU-CPU coherence
    }
}

// =============================================================================
// DispatchPool — auto-growing kernargs pool (no fixed slot limit)
// =============================================================================

/// Pre-allocated dispatch resources for zero-overhead kernel launches.
/// Auto-grows: accessing any slot index automatically allocates if needed.
/// No hardcoded slot limits — just use any index you want.
pub struct DispatchPool {
    /// Single reusable signal buffer
    pub signal: GpuBuffer,
    /// Auto-growing ring of kernargs buffers (Mutex for thread-safe interior mutability)
    kernargs_ring: std::sync::Mutex<Vec<GpuBuffer>>,
    /// Device reference for on-demand allocation
    device: Arc<KfdDevice>,
}

impl DispatchPool {
    /// Create a pool. `initial_slots` kernargs buffers are pre-allocated.
    /// Additional slots are allocated on-demand when accessed.
    /// Pass 0 for default (1024 initial slots, auto-grows beyond that).
    pub fn new(device: &Arc<KfdDevice>, initial_slots: usize) -> Result<Self, String> {
        let signal = device.alloc_signal()?;
        let n = if initial_slots == 0 { 1024 } else { initial_slots };
        let mut ring = Vec::with_capacity(n);
        for _ in 0..n {
            ring.push(device.alloc_uncached(256)?);
        }
        Ok(Self {
            signal,
            kernargs_ring: std::sync::Mutex::new(ring),
            device: Arc::clone(device),
        })
    }

    /// Ensure slot `idx` exists, growing the pool if necessary.
    fn ensure_slot(&self, idx: usize) {
        let mut ring = self.kernargs_ring.lock().unwrap();
        while idx >= ring.len() {
            // Allocate new slot on demand
            match self.device.alloc_uncached(256) {
                Ok(buf) => ring.push(buf),
                Err(e) => panic!("DispatchPool: failed to grow to slot {}: {}", idx, e),
            }
        }
    }

    /// Get kernargs buffer for slot `idx`. Auto-allocates if slot doesn't exist.
    pub fn get_kernargs(&self, idx: usize) -> &GpuBuffer {
        self.ensure_slot(idx);
        let ring = self.kernargs_ring.lock().unwrap();
        // Safety: buffer lives as long as the pool (never removed from Vec)
        unsafe { &*(ring.get(idx).unwrap() as *const GpuBuffer) }
    }

    /// Write kernargs data to slot `idx` and return the buffer ref.
    /// Auto-allocates if slot doesn't exist yet.
    pub fn write_kernargs(&self, idx: usize, data: &[u8]) -> &GpuBuffer {
        self.ensure_slot(idx);
        let ring = self.kernargs_ring.lock().unwrap();
        let buf = unsafe { &*(ring.get(idx).unwrap() as *const GpuBuffer) };
        buf.write(data);
        buf
    }

    /// Dispatch with pre-allocated signal. Resets signal, dispatches, waits.
    pub fn dispatch(
        &self,
        queue: &AqlQueue,
        kernel: &GpuKernel,
        grid: [u32; 3],
        ka_idx: usize,
    ) -> Result<(), String> {
        self.signal.write_val::<u64>(0, 1);
        self.signal.write_val::<i64>(8, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        let ka = self.get_kernargs(ka_idx);
        queue.dispatch_signal(kernel, grid, ka, Some(&self.signal))
    }

    /// Current number of allocated slots.
    pub fn len(&self) -> usize { self.kernargs_ring.lock().unwrap().len() }

    /// Capacity in slots (same as len since auto-growing).
    pub fn capacity(&self) -> usize { self.kernargs_ring.lock().unwrap().capacity() }
}

// =============================================================================
// GpuMemset — GPU-side memory zeroing (replaces CPU PCIe writes)
// =============================================================================

/// GPU memset: zeros memory at VRAM bandwidth (~960 GB/s) vs CPU→PCIe (~25 GB/s).
/// For 16 MB: GPU ≈ 0.017 ms vs CPU ≈ 0.64 ms → 37× faster.
pub struct GpuMemset {
    kernel: GpuKernel,
    ka_buf: GpuBuffer,
}

impl GpuMemset {
    /// Build and load the GPU memset kernel.
    pub fn new(device: &Arc<KfdDevice>) -> Result<Self, String> {
        use crate::rdna3_asm::{Rdna3Assembler, gfx11};
        use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

        let mut asm = Rdna3Assembler::new();
        // s[0:1] = kernarg_ptr, s2 = wg_id_x, v0 = thread_id_x
        // Kernarg: [ptr:u64, n_dw4:u32, pad:u32]

        // Load kernargs -> s[4:7]
        let [w0, w1] = gfx11::s_load_dwordx4(4, 0, 0);
        asm.emit(w0); asm.emit(w1);
        asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

        // global_id = wg_id_x * 256 + thread_id
        asm.emit(gfx11::v_mov_b32_from_sgpr(1, 2));
        asm.emit(gfx11::v_lshlrev_b32(1, 8, 1));
        asm.emit(gfx11::v_add_u32(1, 1, 0));

        // EXEC mask: active if global_id < n_dw4
        asm.emit(gfx11::v_mov_b32_from_sgpr(2, 6));
        asm.emit(gfx11::v_cmp_lt_u32(1, 2));

        // addr = ptr + global_id * 16
        asm.emit(gfx11::v_mov_b32_from_sgpr(3, 4));
        asm.emit(gfx11::v_mov_b32_from_sgpr(4, 5));
        asm.emit(gfx11::v_lshlrev_b32(5, 4, 1));
        let [a0, a1] = gfx11::v_add_co_u32_vcc(3, 3, 5);
        asm.emit(a0); asm.emit(a1);
        let [b0, b1] = gfx11::v_add_co_ci_u32_zero_vcc(4, 4);
        asm.emit(b0); asm.emit(b1);

        // v[6:9] = 0
        asm.emit(gfx11::v_mov_b32_imm(6, 0));
        asm.emit(gfx11::v_mov_b32_imm(7, 0));
        asm.emit(gfx11::v_mov_b32_imm(8, 0));
        asm.emit(gfx11::v_mov_b32_imm(9, 0));

        // store zeros (only active EXEC lanes)
        asm.global_store_dwordx4(3, 6, 0);

        asm.emit(gfx11::s_waitcnt_vscnt(0));
        asm.emit(gfx11::S_ENDPGM);

        let co = AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
            name: "gpu_memset_zero".to_string(),
            lds_size: 0,
            kernarg_size: 16,
            vgpr_count: 10,
            sgpr_count: 8,
            workgroup_size_x: 256,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            scratch_size: 0,
        });

        let hsaco = co.to_code_object_llvm().map_err(|e| format!("memset LLVM: {e}"))?;
        let kernel = GpuKernel::load(device, &hsaco, &KernelLoadConfig {
            lds_size: 0,
            workgroup_size: [256, 1, 1],
        })?;
        let ka_buf = device.alloc_uncached(256)?;
        Ok(Self { kernel, ka_buf })
    }

    /// Zero `n_bytes` of `buf`. Waits for completion via signal.
    pub fn zero(
        &self,
        queue: &AqlQueue,
        buf: &GpuBuffer,
        n_bytes: usize,
        signal: &GpuBuffer,
    ) -> Result<(), String> {
        let n_dw4 = ((n_bytes + 15) / 16) as u32;
        let n_wg = (n_dw4 + 255) / 256;
        let mut ka = [0u8; 16];
        ka[0..8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
        ka[8..12].copy_from_slice(&n_dw4.to_le_bytes());
        self.ka_buf.write(&ka);
        signal.write_val::<u64>(0, 1);
        signal.write_val::<i64>(8, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        queue.dispatch_signal(&self.kernel, [n_wg * 256, 1, 1], &self.ka_buf, Some(signal))
    }

    /// Zero without signal — call queue.wait_idle() after.
    pub fn zero_async(&self, queue: &AqlQueue, buf: &GpuBuffer, n_bytes: usize) {
        let n_dw4 = ((n_bytes + 15) / 16) as u32;
        let n_wg = (n_dw4 + 255) / 256;
        let mut ka = [0u8; 16];
        ka[0..8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
        ka[8..12].copy_from_slice(&n_dw4.to_le_bytes());
        self.ka_buf.write(&ka);
        queue.submit(&self.kernel, [n_wg * 256, 1, 1], &self.ka_buf);
    }
}
