//! WSL2 DXG Backend
//!
//! Direct GPU control via WSL2 `/dev/dxg` and DXG KMT APIs.
//! Implements: Device initialization, VRAM allocation, AQL queue dispatch,
//! and synchronization without ROCm stack.
//!
//! Architecture:
//!   /dev/dxg → DXG KMT APIs (Device, Context, Allocation, SyncObject)
//!   AQL ring buffer → 64-byte dispatch packets → GPU execution
//!
//! Target: AMD RX 7900 XTX (GFX1100, RDNA3), Windows WDDM
//!
//! Reference: libdxg (libdxg/src/d3dkmt-wsl.cpp) and librocdxg WDDM implementation

// =============================================================================
// Submodules
// =============================================================================

pub mod memory_tests;
mod thunk_proxy;

use self::thunk_proxy::DxgThunkDeviceInfo;
use std::sync::Arc;
use std::ffi::c_void;
use std::ptr;

// =============================================================================
// Global Constants
// =============================================================================

/// AQL packet types
const HSA_PACKET_TYPE_VENDOR_SPECIFIC: u16 = 0x0;
const HSA_PACKET_TYPE_INVALID: u16 = 0x1;
const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 0x2;
const HSA_PACKET_TYPE_BARRIER_AND: u16 = 0x3;
const HSA_PACKET_TYPE_AGENT_DISPATCH: u16 = 0x4;
const HSA_PACKET_TYPE_BARRIER_OR: u16 = 0x5;
const HSA_PACKET_HEADER_BARRIER_BIT: u16 = 1 << 8;

/// Fence scopes
const HSA_FENCE_SCOPE_AGENT: u16 = 1;
const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;

/// Page size
const PAGE_SIZE: usize = 4096;
const D3DDDI_ID_UNINITIALIZED: u32 = u32::MAX;

/// Ring buffer overflow protection
const MAX_INFLIGHT: u64 = 64;

/// Default dispatch pool slots
const DEFAULT_DISPATCH_SLOTS: usize = 1024;

const D3DKMT_CLIENTHINT_OPENCL: u32 = 3;
const D3DDDI_CREATECONTEXTFLAGS_DISABLE_GPU_TIMEOUT: u32 = 1 << 2;
const D3DDDI_CREATECONTEXTFLAGS_HW_QUEUE_SUPPORTED: u32 = 1 << 4;
const D3DDDI_GPU_VA_PROTECTION_WRITE: u64 = 1 << 0;
const D3DDDI_GPU_VA_PROTECTION_EXECUTE: u64 = 1 << 1;
const D3DDDI_MAKERESIDENTFLAGS_CANT_TRIM_FURTHER: u32 = 1 << 0;
const T0_DXG_MEM_FLAG_FINE_GRAIN: u32 = 1 << 0;
const T0_DXG_MEM_FLAG_KERNARG: u32 = 1 << 1;
const DEFAULT_GPU_PAGE_SIZE_U64: u64 = 1 << 12;
const GPU_HUGE_PAGE_SIZE_U64: u64 = 2 << 20;
const DXG_HW_QUEUE_FRAME_SIZE: usize = 0x1000;
const DXG_HW_QUEUE_FRAME_COUNT: u64 = 0x1000;

const PM4_SET_SH_REG: u32 = 0x76;
const PM4_DISPATCH_DIRECT: u32 = 0x15;
const PM4_ATOMIC_MEM: u32 = 0x1E;
const PM4_COPY_DATA: u32 = 0x40;
const PM4_WRITE_DATA: u32 = 0x37;
const PM4_RELEASE_MEM: u32 = 0x49;
const PM4_ACQUIRE_MEM: u32 = 0x58;
const PM4_EVENT_WRITE: u32 = 0x46;
const PACKET3_INDIRECT_BUFFER: u32 = 0x3F;
const INDIRECT_BUFFER_VALID: u32 = 1 << 23;
const AMD_AQL_FORMAT_PM4_IB: u16 = 0x1;
const PM4_COMPUTE_SHADER_TYPE: u32 = 1 << 1;

const SH_REG_BASE: u32 = 0x2C00;
const REG_COMPUTE_NUM_THREAD_X: u32 = 0x2E07;
const REG_COMPUTE_PGM_LO: u32 = 0x2E0C;
const REG_COMPUTE_DISPATCH_SCRATCH_BASE_LO: u32 = 0x2E10;
const REG_COMPUTE_PGM_RSRC1: u32 = 0x2E12;
const REG_COMPUTE_RESOURCE_LIMITS: u32 = 0x2E15;
const REG_COMPUTE_TMPRING_SIZE: u32 = 0x2E18;
const REG_COMPUTE_PGM_RSRC3: u32 = 0x2E28;
const REG_COMPUTE_USER_DATA_0: u32 = 0x2E40;

const CS_PARTIAL_FLUSH: u32 = 0x07;
const EVENT_INDEX_PARTIAL_FLUSH: u32 = 4;
const CACHE_FLUSH_AND_INV_TS_EVENT: u32 = 0x14;

const KD_GROUP_SEGMENT_FIXED_SIZE_OFFSET: usize = 0x00;
const KD_PRIVATE_SEGMENT_FIXED_SIZE_OFFSET: usize = 0x04;
const KD_KERNARG_SIZE_OFFSET: usize = 0x08;
const KD_KERNEL_CODE_ENTRY_BYTE_OFFSET: usize = 0x10;
const KD_COMPUTE_PGM_RSRC3_OFFSET: usize = 0x2C;
const KD_COMPUTE_PGM_RSRC1_OFFSET: usize = 0x30;
const KD_COMPUTE_PGM_RSRC2_OFFSET: usize = 0x34;
const KD_KERNEL_CODE_PROPERTIES_OFFSET: usize = 0x38;

const HSA_QUEUE_TYPE_SINGLE: u32 = 1;
const HSA_QUEUE_FEATURE_KERNEL_DISPATCH: u32 = 1;
const AMD_QUEUE_PROPERTIES_IS_PTR64: u32 = 1 << 1;

const AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER: u16 = 1 << 0;
const AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR: u16 = 1 << 1;
const AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR: u16 = 1 << 2;
const AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR: u16 = 1 << 3;
const AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID: u16 = 1 << 4;
const AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT: u16 = 1 << 5;
const AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE: u16 = 1 << 6;
const AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32: u16 = 1 << 10;

const COMPUTE_RESOURCE_LIMITS_DEFAULT: u32 = 0x3FF;
const COMPUTE_STATIC_THREAD_MGMT_ENABLE_ALL: u32 = 0xFFFF_FFFF;
const COMPUTE_PGM_RSRC3_IMAGE_OP: u32 = 1 << 31;
const MEC_ATOMIC_MEM_CACHE_POLICY_STREAM: u32 = 1;
const MEC_ATOMIC_MEM_CACHE_POLICY_BYPASS: u32 = 3;
const TC_OP_ATOMIC_ADD_RTN_64: u32 = 0x2F;
const DISPATCH_INITIATOR_COMPUTE_SHADER_EN: u32 = 1 << 0;
const DISPATCH_INITIATOR_FORCE_START_AT_000: u32 = 1 << 2;
const DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS: u32 = 1 << 5;
const DISPATCH_INITIATOR_CS_W32_EN: u32 = 1 << 15;
const AMD_SIGNAL_KIND_USER: u64 = 1;
const AMD_SIGNAL_SIZE_BYTES: usize = 64;
const AMD_SIGNAL_KIND_OFFSET: usize = 0;
const AMD_SIGNAL_VALUE_OFFSET: usize = 8;
const AMD_SIGNAL_START_TS_OFFSET: usize = 32;
const AMD_SIGNAL_END_TS_OFFSET: usize = 40;
const AQL_RESERVED2_PROFILE_TS: u64 = 1u64 << 0;

#[repr(C, align(64))]
struct AmdSignalLayout {
    kind: u64,
    value: i64,
    event_mailbox_ptr: u64,
    event_id: u32,
    reserved1: u32,
    start_ts: u64,
    end_ts: u64,
    reserved2: u64,
    reserved3: [u32; 2],
}

fn verify_amd_signal_layout_once() {
    static VERIFIED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    VERIFIED.get_or_init(|| {
        use std::mem::{align_of, size_of, MaybeUninit};
        let sig = MaybeUninit::<AmdSignalLayout>::uninit();
        let base = sig.as_ptr() as usize;
        let kind_off = unsafe { std::ptr::addr_of!((*sig.as_ptr()).kind) as usize - base };
        let value_off = unsafe { std::ptr::addr_of!((*sig.as_ptr()).value) as usize - base };
        let start_off = unsafe { std::ptr::addr_of!((*sig.as_ptr()).start_ts) as usize - base };
        let end_off = unsafe { std::ptr::addr_of!((*sig.as_ptr()).end_ts) as usize - base };

        assert_eq!(size_of::<AmdSignalLayout>(), AMD_SIGNAL_SIZE_BYTES, "amd_signal size mismatch");
        assert_eq!(align_of::<AmdSignalLayout>(), 64, "amd_signal alignment mismatch");
        assert_eq!(kind_off, AMD_SIGNAL_KIND_OFFSET, "amd_signal.kind offset mismatch");
        assert_eq!(value_off, AMD_SIGNAL_VALUE_OFFSET, "amd_signal.value offset mismatch");
        assert_eq!(start_off, AMD_SIGNAL_START_TS_OFFSET, "amd_signal.start_ts offset mismatch");
        assert_eq!(end_off, AMD_SIGNAL_END_TS_OFFSET, "amd_signal.end_ts offset mismatch");
    });
}

#[inline]
fn align_up(value: usize, align: usize) -> usize {
    ((value + align - 1) / align) * align
}

#[inline]
fn align_up_u64(value: u64, align: u64) -> u64 {
    ((value + align - 1) / align) * align
}

#[inline]
fn low_part(value: u64) -> u32 {
    value as u32
}

#[inline]
fn high_part(value: u64) -> u32 {
    (value >> 32) as u32
}

#[inline]
fn make_u64(low: u32, high: u32) -> u64 {
    (low as u64) | ((high as u64) << 32)
}

#[inline]
fn ptr48_low32(addr: u64) -> u32 {
    ((addr & 0xFFFF_FFFF_FF) >> 8) as u32
}

#[inline]
fn ptr48_high8(addr: u64) -> u32 {
    ((addr >> 40) & 0xFF) as u32
}

#[inline]
fn lds_blocks(group_segment_size: u32) -> u32 {
    group_segment_size.saturating_add(511) / 512
}

#[inline]
fn nt_failed(status: NTSTATUS) -> bool {
    status < 0
}

fn dxg_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("T0_DXG_DEBUG").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

fn dxg_platform_atomic_override() -> Option<bool> {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| match std::env::var("T0_DXG_USE_PLATFORM_ATOMIC").ok().as_deref() {
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") => Some(true),
        Some("0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => Some(false),
        _ => None,
    })
}

fn dxg_force_hw_queue() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("T0_DXG_FORCE_HW_QUEUE").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

fn dxg_allow_wgp_legacy() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("T0_DXG_ALLOW_WGP_LEGACY").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

fn ignore_sigpipe_once() {
    static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    INIT.get_or_init(|| unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    });
}

macro_rules! dxg_debug {
    ($($arg:tt)*) => {
        if dxg_debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

fn hex_prefix(bytes: &[u8], len: usize) -> String {
    let mut out = String::new();
    for (i, byte) in bytes.iter().take(len).enumerate() {
        if i != 0 {
            out.push(' ');
        }
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02X}", byte);
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct HsaSignalHandle {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct HsaQueueAbi {
    queue_type: u32,
    features: u32,
    base_address: *mut c_void,
    doorbell_signal: HsaSignalHandle,
    size: u32,
    reserved1: u32,
    id: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ScratchLastUsedIndexXcc {
    main: u64,
    alt: u64,
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
struct AmdQueueV2 {
    hsa_queue: HsaQueueAbi,
    caps: u32,
    reserved1: [u32; 3],
    write_dispatch_id: u64,
    group_segment_aperture_base_hi: u32,
    private_segment_aperture_base_hi: u32,
    max_cu_id: u32,
    max_wave_id: u32,
    max_legacy_doorbell_dispatch_id_plus_1: u64,
    legacy_doorbell_lock: u32,
    reserved2: [u32; 9],
    read_dispatch_id: u64,
    read_dispatch_id_field_base_byte_offset: u32,
    compute_tmpring_size: u32,
    scratch_resource_descriptor: [u32; 4],
    scratch_backing_memory_location: u64,
    scratch_backing_memory_byte_size: u64,
    scratch_wave64_lane_byte_size: u32,
    queue_properties: u32,
    scratch_max_use_index: u64,
    queue_inactive_signal: HsaSignalHandle,
    alt_scratch_max_use_index: u64,
    alt_scratch_resource_descriptor: [u32; 4],
    alt_scratch_backing_memory_location: u64,
    alt_scratch_dispatch_limit_x: u32,
    alt_scratch_dispatch_limit_y: u32,
    alt_scratch_dispatch_limit_z: u32,
    alt_scratch_wave64_lane_byte_size: u32,
    alt_compute_tmpring_size: u32,
    reserved5: u32,
    scratch_last_used_index: [ScratchLastUsedIndexXcc; 128],
}

impl Default for AmdQueueV2 {
    fn default() -> Self {
        Self {
            hsa_queue: HsaQueueAbi::default(),
            caps: 0,
            reserved1: [0; 3],
            write_dispatch_id: 0,
            group_segment_aperture_base_hi: 0,
            private_segment_aperture_base_hi: 0,
            max_cu_id: 0,
            max_wave_id: 0,
            max_legacy_doorbell_dispatch_id_plus_1: 0,
            legacy_doorbell_lock: 0,
            reserved2: [0; 9],
            read_dispatch_id: 0,
            read_dispatch_id_field_base_byte_offset: 0,
            compute_tmpring_size: 0,
            scratch_resource_descriptor: [0; 4],
            scratch_backing_memory_location: 0,
            scratch_backing_memory_byte_size: 0,
            scratch_wave64_lane_byte_size: 0,
            queue_properties: 0,
            scratch_max_use_index: 0,
            queue_inactive_signal: HsaSignalHandle::default(),
            alt_scratch_max_use_index: 0,
            alt_scratch_resource_descriptor: [0; 4],
            alt_scratch_backing_memory_location: 0,
            alt_scratch_dispatch_limit_x: 0,
            alt_scratch_dispatch_limit_y: 0,
            alt_scratch_dispatch_limit_z: 0,
            alt_scratch_wave64_lane_byte_size: 0,
            alt_compute_tmpring_size: 0,
            reserved5: 0,
            scratch_last_used_index: [ScratchLastUsedIndexXcc::default(); 128],
        }
    }
}

fn init_amd_queue_metadata(
    queue: &mut AmdQueueV2,
    ring_base: *mut c_void,
    ring_packets: u32,
    queue_id: u64,
) {
    *queue = AmdQueueV2::default();
    queue.hsa_queue.queue_type = HSA_QUEUE_TYPE_SINGLE;
    queue.hsa_queue.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
    queue.hsa_queue.base_address = ring_base;
    queue.hsa_queue.size = ring_packets;
    queue.hsa_queue.id = queue_id;
    queue.queue_properties = AMD_QUEUE_PROPERTIES_IS_PTR64;
    queue.read_dispatch_id_field_base_byte_offset =
        std::mem::offset_of!(AmdQueueV2, read_dispatch_id) as u32;
}

// =============================================================================
// libc FFI
// =============================================================================

extern "C" {
    fn mmap(addr: *mut c_void, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> i32;
    fn mprotect(addr: *mut c_void, length: usize, prot: i32) -> i32;
    fn malloc(size: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}

const PROT_NONE: i32 = 0;
const PROT_READ: i32 = 1;
const PROT_EXEC: i32 = 4;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_NORESERVE: i32 = 0x4000;
const MAP_UNINITIALIZED: i32 = 0x4000000;
const MAP_FAILED: *mut c_void = !0 as *mut c_void;

// =============================================================================
// DXG KMT FFI — matched to libdxg/src/d3dkmt-wsl.cpp + d3dkmthk.h
// =============================================================================

/// D3DKMT_HANDLE = UINT (32-bit on WSL)
pub type D3DKMT_HANDLE = u32;

/// NTSTATUS return type
type NTSTATUS = i32;

// --- Function prototypes (match libdxg exported symbols) ---
#[link(name = "dxcore")]
extern "C" {
    fn D3DKMTOpenAdapterFromLuid(pArgs: *const D3DKMT_OPENADAPTERFROMLUID) -> NTSTATUS;
    fn D3DKMTCreateDevice(pArgs: *mut D3DKMT_CREATEDEVICE) -> NTSTATUS;
    fn D3DKMTCreateContextVirtual(pArgs: *mut D3DKMT_CREATECONTEXTVIRTUAL) -> NTSTATUS;
    fn D3DKMTCreatePagingQueue(pArgs: *mut D3DKMT_CREATEPAGINGQUEUE) -> NTSTATUS;
    fn D3DKMTDestroyPagingQueue(pArgs: *const D3DDDI_DESTROYPAGINGQUEUE) -> NTSTATUS;
    fn D3DKMTCreateAllocation2(pArgs: *mut D3DKMT_CREATEALLOCATION) -> NTSTATUS;
    fn D3DKMTDestroyAllocation2(pArgs: *const D3DKMT_DESTROYALLOCATION2) -> NTSTATUS;
    fn D3DKMTReserveGpuVirtualAddress(pArgs: *mut D3DDDI_RESERVEGPUVIRTUALADDRESS) -> NTSTATUS;
    fn D3DKMTFreeGpuVirtualAddress(pArgs: *const D3DKMT_FREEGPUVIRTUALADDRESS) -> NTSTATUS;
    fn D3DKMTMapGpuVirtualAddress(pArgs: *mut D3DDDI_MAPGPUVIRTUALADDRESS) -> NTSTATUS;
    fn D3DKMTMakeResident(pArgs: *mut D3DDDI_MAKERESIDENT) -> NTSTATUS;
    fn D3DKMTWaitForSynchronizationObjectFromCpu(
        pArgs: *const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU,
    ) -> NTSTATUS;
    fn D3DKMTWaitForSynchronizationObjectFromGpu(
        pArgs: *const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU,
    ) -> NTSTATUS;
    fn D3DKMTLock2(pArgs: *mut D3DKMT_LOCK2) -> NTSTATUS;
    fn D3DKMTUnlock2(pArgs: *const D3DKMT_UNLOCK2) -> NTSTATUS;
    fn D3DKMTCloseAdapter(pArgs: *const D3DKMT_CLOSEADAPTER) -> NTSTATUS;
    fn D3DKMTDestroyDevice(pArgs: *const D3DKMT_DESTROYDEVICE) -> NTSTATUS;
    fn D3DKMTDestroyContext(pArgs: *const D3DKMT_DESTROYCONTEXT) -> NTSTATUS;
    fn D3DKMTCreateSynchronizationObject2(pArgs: *mut D3DKMT_CREATESYNCHRONIZATIONOBJECT2) -> NTSTATUS;
    fn D3DKMTWaitForSynchronizationObject2(pArgs: *const D3DKMT_WAITFORSYNCHRONIZATIONOBJECT2) -> NTSTATUS;
    fn D3DKMTSignalSynchronizationObject2(pArgs: *const D3DKMT_SIGNALSYNCHRONIZATIONOBJECT2) -> NTSTATUS;
    fn D3DKMTDestroySynchronizationObject(pArgs: *const D3DKMT_DESTROYSYNCHRONIZATIONOBJECT) -> NTSTATUS;
    fn D3DKMTSignalSynchronizationObjectFromGpu(
        pArgs: *const D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU,
    ) -> NTSTATUS;
    fn D3DKMTSubmitCommand(pArgs: *const D3DKMT_SUBMITCOMMAND) -> NTSTATUS;
    fn D3DKMTEscape(pArgs: *const D3DKMT_ESCAPE) -> NTSTATUS;
    fn D3DKMTCreateHwQueue(pArgs: *mut D3DKMT_CREATEHWQUEUE) -> NTSTATUS;
    fn D3DKMTDestroyHwQueue(pArgs: *const D3DKMT_DESTROYHWQUEUE) -> NTSTATUS;
    fn D3DKMTSubmitCommandToHwQueue(pArgs: *const D3DKMT_SUBMITCOMMANDTOHWQUEUE) -> NTSTATUS;
    fn D3DKMTEnumAdapters2(pArgs: *mut D3DKMT_ENUMADAPTERS2) -> NTSTATUS;
    fn D3DKMTEnumAdapters3(pArgs: *mut D3DKMT_ENUMADAPTERS3) -> NTSTATUS;
    fn D3DKMTQueryAdapterInfo(pArgs: *const D3DKMT_QUERYADAPTERINFO) -> NTSTATUS;
    fn D3DKMTQueryClockCalibration(pArgs: *mut D3DKMT_QUERYCLOCKCALIBRATION) -> NTSTATUS;
    fn D3DKMTGetDeviceState(pArgs: *mut D3DKMT_GETDEVICESTATE) -> NTSTATUS;
}

// =============================================================================
// DXG KMT Structs — exact layout from libdxg/include/dxg/d3dkmthk.h
// =============================================================================

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_OPENADAPTERFROMLUID {
    pub AdapterLuid: LUID,
    pub hAdapter: D3DKMT_HANDLE,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct LUID {
    pub LowPart: u32,
    pub HighPart: i32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_CREATEDEVICEFLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_CREATEDEVICE {
    pub hAdapter: D3DKMT_HANDLE,
    pub pAdapter_Align: u32,
    pub Flags: D3DKMT_CREATEDEVICEFLAGS,
    pub hDevice: D3DKMT_HANDLE,
    pub pCommandBuffer: *mut c_void,
    pub CommandBufferSize: u32,
    pub pAllocationList: *mut c_void,
    pub AllocationListSize: u32,
    pub pPatchLocationList: *mut c_void,
    pub PatchLocationListSize: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DDDI_CREATECONTEXTFLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_CREATECONTEXTVIRTUAL {
    pub hDevice: D3DKMT_HANDLE,
    pub NodeOrdinal: u32,
    pub EngineAffinity: u32,
    pub Flags: D3DDDI_CREATECONTEXTFLAGS,
    pub pPrivateDriverData: *mut c_void,
    pub PrivateDriverDataSize: u32,
    pub ClientHint: u32,
    pub hContext: D3DKMT_HANDLE,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct D3DKMT_CREATEALLOCATIONFLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct D3DDDI_ALLOCATIONINFO2 {
    pub hAllocation: D3DKMT_HANDLE,
    pub pSystemMem: *mut c_void,
    pub pPrivateDriverData: *mut c_void,
    pub PrivateDriverDataSize: u32,
    pub VidPnSourceId: u32,
    pub Flags: u32,
    pub GpuVirtualAddress: u64,
    pub Priority: u64,
    pub Reserved: [u64; 5],
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_CREATEALLOCATION {
    pub hDevice: D3DKMT_HANDLE,
    pub hResource: D3DKMT_HANDLE,
    pub hGlobalShare: D3DKMT_HANDLE,
    pub pPrivateRuntimeData: *mut c_void,
    pub PrivateRuntimeDataSize: u32,
    pub pPrivateDriverData: *mut c_void,
    pub PrivateDriverDataSize: u32,
    pub NumAllocations: u32,
    pub pAllocationInfo2: *mut D3DDDI_ALLOCATIONINFO2,
    pub Flags: D3DKMT_CREATEALLOCATIONFLAGS,
    pub hPrivateRuntimeResourceHandle: *mut c_void,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DDDICB_DESTROYALLOCATION2FLAGS {
    pub Value: u32,
}

#[repr(C)]
pub struct D3DKMT_DESTROYALLOCATION2 {
    pub hDevice: D3DKMT_HANDLE,
    pub hResource: D3DKMT_HANDLE,
    pub phAllocationList: *const D3DKMT_HANDLE,
    pub AllocationCount: u32,
    pub Flags: D3DDDICB_DESTROYALLOCATION2FLAGS,
}

impl Default for D3DKMT_DESTROYALLOCATION2 {
    fn default() -> Self {
        Self {
            hDevice: 0,
            hResource: 0,
            phAllocationList: ptr::null(),
            AllocationCount: 0,
            Flags: D3DDDICB_DESTROYALLOCATION2FLAGS { Value: 0 },
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum D3DDDI_PAGINGQUEUE_PRIORITY {
    Normal = 0,
    AboveNormal = 1,
}

impl Default for D3DDDI_PAGINGQUEUE_PRIORITY {
    fn default() -> Self {
        Self::Normal
    }
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_CREATEPAGINGQUEUE {
    pub hDevice: D3DKMT_HANDLE,
    pub Priority: D3DDDI_PAGINGQUEUE_PRIORITY,
    pub hPagingQueue: D3DKMT_HANDLE,
    pub hSyncObject: D3DKMT_HANDLE,
    pub FenceValueCPUVirtualAddress: *mut c_void,
    pub PhysicalAdapterIndex: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DDDI_DESTROYPAGINGQUEUE {
    pub hPagingQueue: D3DKMT_HANDLE,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DDDIGPUVIRTUALADDRESS_PROTECTION_TYPE {
    pub Value: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DDDI_MAPGPUVIRTUALADDRESS {
    pub hPagingQueue: D3DKMT_HANDLE,
    pub BaseAddress: u64,
    pub MinimumAddress: u64,
    pub MaximumAddress: u64,
    pub hAllocation: D3DKMT_HANDLE,
    pub OffsetInPages: u64,
    pub SizeInPages: u64,
    pub Protection: D3DDDIGPUVIRTUALADDRESS_PROTECTION_TYPE,
    pub DriverProtection: u64,
    pub Reserved0: u32,
    pub Reserved1: u64,
    pub VirtualAddress: u64,
    pub PagingFenceValue: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DDDI_RESERVEGPUVIRTUALADDRESS {
    pub hAdapter: D3DKMT_HANDLE,
    pub BaseAddress: u64,
    pub MinimumAddress: u64,
    pub MaximumAddress: u64,
    pub Size: u64,
    pub ReservationType: u32,
    pub DriverProtection: u64,
    pub VirtualAddress: u64,
    pub PagingFenceValue: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DDDI_MAKERESIDENT_FLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DDDI_MAKERESIDENT {
    pub hPagingQueue: D3DKMT_HANDLE,
    pub NumAllocations: u32,
    pub AllocationList: *const D3DKMT_HANDLE,
    pub PriorityList: *const u32,
    pub Flags: D3DDDI_MAKERESIDENT_FLAGS,
    pub PagingFenceValue: u64,
    pub NumBytesToTrim: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_FREEGPUVIRTUALADDRESS {
    pub hAdapter: D3DKMT_HANDLE,
    pub BaseAddress: u64,
    pub Size: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DDDI_WAITFORSYNCHRONIZATIONOBJECTFROMCPU_FLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU {
    pub hDevice: D3DKMT_HANDLE,
    pub ObjectCount: u32,
    pub ObjectHandleArray: *const D3DKMT_HANDLE,
    pub FenceValueArray: *const u64,
    pub hAsyncEvent: *mut c_void,
    pub Flags: D3DDDI_WAITFORSYNCHRONIZATIONOBJECTFROMCPU_FLAGS,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union D3DKMT_WAITSYNCFGPU_DATA {
    pub MonitoredFenceValueArray: *const u64,
    pub FenceValue: u64,
    pub Reserved: [u64; 8],
}

impl Default for D3DKMT_WAITSYNCFGPU_DATA {
    fn default() -> Self {
        Self { Reserved: [0; 8] }
    }
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU {
    pub hContext: D3DKMT_HANDLE,
    pub ObjectCount: u32,
    pub ObjectHandleArray: *const D3DKMT_HANDLE,
    pub data: D3DKMT_WAITSYNCFGPU_DATA,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DDDICB_LOCK2FLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_LOCK2 {
    pub hDevice: D3DKMT_HANDLE,
    pub hAllocation: D3DKMT_HANDLE,
    pub Flags: D3DDDICB_LOCK2FLAGS,
    pub pData: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_UNLOCK2 {
    pub hDevice: D3DKMT_HANDLE,
    pub hAllocation: D3DKMT_HANDLE,
}

#[repr(C)]
pub struct D3DKMT_CLOSEADAPTER {
    pub hAdapter: D3DKMT_HANDLE,
}

#[repr(C)]
pub struct D3DKMT_DESTROYDEVICE {
    pub hDevice: D3DKMT_HANDLE,
}

#[repr(C)]
pub struct D3DKMT_DESTROYCONTEXT {
    pub hContext: D3DKMT_HANDLE,
}

// --- Synchronization Object Types ---

#[repr(u32)]
#[derive(Debug, Clone, Copy, Default)]
pub enum D3DDDI_SYNCHRONIZATIONOBJECT_TYPE {
    SynchronizationMutex = 1,
    Semaphore = 2,
    #[default]
    Fence = 3,
    CPUNotification = 4,
    MonitoredFence = 5,
    PeriodicMonitoredFence = 6,
    NativeFence = 7,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DDDI_SYNCHRONIZATIONOBJECT_FLAGS {
    pub Value: u32,
}

#[repr(C)]
pub struct D3DDDI_SYNCHRONIZATIONOBJECTINFO2 {
    pub Type: D3DDDI_SYNCHRONIZATIONOBJECT_TYPE,
    pub Flags: D3DDDI_SYNCHRONIZATIONOBJECT_FLAGS,
    pub info: SyncObjectInfoUnion,
    pub SharedHandle: D3DKMT_HANDLE,
}

impl Default for D3DDDI_SYNCHRONIZATIONOBJECTINFO2 {
    fn default() -> Self {
        Self {
            Type: D3DDDI_SYNCHRONIZATIONOBJECT_TYPE::Fence,
            Flags: D3DDDI_SYNCHRONIZATIONOBJECT_FLAGS { Value: 0 },
            info: SyncObjectInfoUnion {
                Reserved: [0; 8],
            },
            SharedHandle: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union SyncObjectInfoUnion {
    pub SynchronizationMutex: SyncMutexInfo,
    pub Semaphore: SemaphoreInfo,
    pub Fence: FenceInfo,
    pub CPUNotification: CpuNotificationInfo,
    pub MonitoredFence: MonitoredFenceInfo,
    pub Reserved: [u64; 8],
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct SyncMutexInfo {
    pub InitialState: i32, // BOOL
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct SemaphoreInfo {
    pub MaxCount: u32,
    pub InitialCount: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct FenceInfo {
    pub FenceValue: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct CpuNotificationInfo {
    pub Event: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MonitoredFenceInfo {
    pub InitialFenceValue: u64,
    pub FenceValueCPUVirtualAddress: *mut c_void,
    pub FenceValueGPUVirtualAddress: u64,
    pub EngineAffinity: u32,
    pub Padding: u32,
}

impl Default for MonitoredFenceInfo {
    fn default() -> Self {
        Self {
            InitialFenceValue: 0,
            FenceValueCPUVirtualAddress: ptr::null_mut(),
            FenceValueGPUVirtualAddress: 0,
            EngineAffinity: 0,
            Padding: 0,
        }
    }
}

#[repr(C)]
pub struct D3DKMT_CREATESYNCHRONIZATIONOBJECT2 {
    pub hDevice: D3DKMT_HANDLE,
    pub Info: D3DDDI_SYNCHRONIZATIONOBJECTINFO2,
    pub hSyncObject: D3DKMT_HANDLE,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_WAITFORSYNCHRONIZATIONOBJECT2 {
    pub hContext: D3DKMT_HANDLE,
    pub ObjectCount: u32,
    pub ObjectHandleArray: [D3DKMT_HANDLE; 32],
    pub data: D3DKMT_WAITSYNC2_DATA,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DDDICB_SIGNALFLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union D3DKMT_WAITSYNC2_DATA {
    pub FenceValue: u64,
    pub Reserved: [u64; 8],
}

impl Default for D3DKMT_WAITSYNC2_DATA {
    fn default() -> Self {
        Self { Reserved: [0; 8] }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union D3DKMT_SIGNALSYNC2_DATA {
    pub FenceValue: u64,
    pub CpuEventHandle: *mut c_void,
    pub Reserved: [u64; 8],
}

impl Default for D3DKMT_SIGNALSYNC2_DATA {
    fn default() -> Self {
        Self { Reserved: [0; 8] }
    }
}

#[repr(C)]
pub struct D3DKMT_SIGNALSYNCHRONIZATIONOBJECT2 {
    pub hContext: D3DKMT_HANDLE,
    pub ObjectCount: u32,
    pub ObjectHandleArray: [D3DKMT_HANDLE; 32],
    pub Flags: D3DDDICB_SIGNALFLAGS,
    pub BroadcastContextCount: u32,
    pub BroadcastContext: [D3DKMT_HANDLE; 64],
    pub data: D3DKMT_SIGNALSYNC2_DATA,
}

impl Default for D3DKMT_SIGNALSYNCHRONIZATIONOBJECT2 {
    fn default() -> Self {
        Self {
            hContext: 0,
            ObjectCount: 0,
            ObjectHandleArray: [0; 32],
            Flags: D3DDDICB_SIGNALFLAGS { Value: 0 },
            BroadcastContextCount: 0,
            BroadcastContext: [0; 64],
            data: D3DKMT_SIGNALSYNC2_DATA::default(),
        }
    }
}

#[repr(C)]
pub struct D3DKMT_DESTROYSYNCHRONIZATIONOBJECT {
    pub hSyncObject: D3DKMT_HANDLE,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union D3DKMT_SIGNALSYNCFGPU_DATA {
    pub MonitoredFenceValueArray: *const u64,
    pub Reserved: [u64; 8],
}

impl Default for D3DKMT_SIGNALSYNCFGPU_DATA {
    fn default() -> Self {
        Self { Reserved: [0; 8] }
    }
}

#[repr(C)]
pub struct D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU {
    pub hContext: D3DKMT_HANDLE,
    pub ObjectCount: u32,
    pub ObjectHandleArray: *const D3DKMT_HANDLE,
    pub data: D3DKMT_SIGNALSYNCFGPU_DATA,
}

impl Default for D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU {
    fn default() -> Self {
        Self {
            hContext: 0,
            ObjectCount: 0,
            ObjectHandleArray: ptr::null(),
            data: D3DKMT_SIGNALSYNCFGPU_DATA::default(),
        }
    }
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_SUBMITCOMMANDFLAGS {
    pub Value: u32,
}

#[repr(C)]
pub struct D3DKMT_SUBMITCOMMAND {
    pub Commands: u64,
    pub CommandLength: u32,
    pub Flags: D3DKMT_SUBMITCOMMANDFLAGS,
    pub PresentHistoryToken: u64,
    pub BroadcastContextCount: u32,
    pub BroadcastContext: [D3DKMT_HANDLE; 64],
    pub pPrivateDriverData: *mut c_void,
    pub PrivateDriverDataSize: u32,
    pub NumPrimaries: u32,
    pub WrittenPrimaries: [D3DKMT_HANDLE; 16],
    pub NumHistoryBuffers: u32,
    pub HistoryBufferArray: *mut D3DKMT_HANDLE,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DDDI_CREATEHWQUEUEFLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DKMT_CREATEHWQUEUE {
    pub hHwContext: D3DKMT_HANDLE,
    pub Flags: D3DDDI_CREATEHWQUEUEFLAGS,
    pub PrivateDriverDataSize: u32,
    pub pPrivateDriverData: *mut c_void,
    pub hHwQueue: D3DKMT_HANDLE,
    pub hHwQueueProgressFence: D3DKMT_HANDLE,
    pub HwQueueProgressFenceCPUVirtualAddress: *mut c_void,
    pub HwQueueProgressFenceGPUVirtualAddress: u64,
}

#[repr(C)]
pub struct D3DKMT_DESTROYHWQUEUE {
    pub hHwQueue: D3DKMT_HANDLE,
}

#[repr(C)]
pub struct D3DKMT_SUBMITCOMMANDTOHWQUEUE {
    pub hHwQueue: D3DKMT_HANDLE,
    pub HwQueueProgressFenceId: u64,
    pub CommandBuffer: u64,
    pub CommandLength: u32,
    pub PrivateDriverDataSize: u32,
    pub pPrivateDriverData: *mut c_void,
    pub NumPrimaries: u32,
    pub WrittenPrimaries: *const D3DKMT_HANDLE,
}

// --- Escape ---

#[repr(u32)]
pub enum D3DKMT_ESCAPETYPE {
    DriverPrivate = 0,
    VidMm = 1,
    Device = 4,
}

#[repr(C)]
#[derive(Default)]
pub struct D3DDDI_ESCAPEFLAGS {
    pub Value: u32,
}

#[repr(C)]
pub struct D3DKMT_ESCAPE {
    pub hAdapter: D3DKMT_HANDLE,
    pub hDevice: D3DKMT_HANDLE,
    pub Type: D3DKMT_ESCAPETYPE,
    pub Flags: D3DDDI_ESCAPEFLAGS,
    pub pPrivateDriverData: *mut c_void,
    pub PrivateDriverDataSize: u32,
    pub hContext: D3DKMT_HANDLE,
}

// --- Enum Adapters ---

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct D3DKMT_ADAPTERINFO {
    pub hAdapter: D3DKMT_HANDLE,
    pub AdapterLuid: LUID,
    pub NumOfSources: u32,
    pub bPrecisePresentRegionsPreferred: i32,
}

#[repr(C)]
pub struct D3DKMT_ENUMADAPTERS2 {
    pub NumAdapters: u32,
    pub pAdapters: *mut D3DKMT_ADAPTERINFO,
}

/// D3DKMT_ENUMADAPTERS3 - Extended adapter enumeration with filter support
/// Required for newer WSL2/DirectX versions where EnumAdapters2 returns 0 adapters.
#[repr(C)]
pub struct D3DKMT_ENUMADAPTERS3 {
    pub Filter: u64,
    pub NumAdapters: u32,
    pub pAdapters: *mut D3DKMT_ADAPTERINFO,
}

// --- Query Adapter Info ---

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum KMTQUERYADAPTERINFOTYPE {
    UmDriverPrivate = 0,
    UmdDriverName = 1,
    GetSegmentSize = 3,
    PhysicalAdapterDeviceIds = 31,
}

#[repr(C)]
pub struct D3DKMT_QUERYADAPTERINFO {
    pub hAdapter: D3DKMT_HANDLE,
    pub Type: KMTQUERYADAPTERINFOTYPE,
    pub pPrivateDriverData: *mut c_void,
    pub PrivateDriverDataSize: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DKMT_DEVICE_IDS {
    pub VendorID: u32,
    pub DeviceID: u32,
    pub SubVendorID: u32,
    pub SubSystemID: u32,
    pub RevisionID: u32,
    pub BusType: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DKMT_QUERY_DEVICE_IDS {
    pub PhysicalAdapterIndex: u32,
    pub DeviceIds: D3DKMT_DEVICE_IDS,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DXGK_GPUCLOCKDATA_FLAGS {
    pub Value: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DXGK_GPUCLOCKDATA {
    // Keep split 32-bit lanes to avoid host ABI packing differences.
    pub GpuFrequencyLow: u32,
    pub GpuFrequencyHigh: u32,
    pub GpuClockCounterLow: u32,
    pub GpuClockCounterHigh: u32,
    pub CpuClockCounterLow: u32,
    pub CpuClockCounterHigh: u32,
    pub Flags: DXGK_GPUCLOCKDATA_FLAGS,
}

impl DXGK_GPUCLOCKDATA {
    fn gpu_frequency_hz(self) -> u64 {
        make_u64(self.GpuFrequencyLow, self.GpuFrequencyHigh)
    }
    fn gpu_clock_counter(self) -> u64 {
        make_u64(self.GpuClockCounterLow, self.GpuClockCounterHigh)
    }
    fn cpu_clock_counter(self) -> u64 {
        make_u64(self.CpuClockCounterLow, self.CpuClockCounterHigh)
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct D3DKMT_QUERYCLOCKCALIBRATION {
    pub hAdapter: D3DKMT_HANDLE,
    pub NodeOrdinal: u32,
    pub PhysicalAdapterIndex: u32,
    pub ClockData: DXGK_GPUCLOCKDATA,
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum D3DKMT_DEVICESTATE_TYPE {
    Execution = 1,
    Reset = 3,
    PageFault = 5,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct D3DKMT_DEVICERESET_STATE {
    Value: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct D3DKMT_DEVICEPAGEFAULT_STATE {
    FaultedPrimitiveAPISequenceNumber: u64,
    FaultedPipelineStage: u32,
    FaultedBindTableEntry: u32,
    PageFaultFlags: u32,
    FaultErrorCode: u32,
    FaultedVirtualAddress: u64,
}

#[repr(C)]
union D3DKMT_GETDEVICESTATE_DATA {
    ExecutionState: u32,
    ResetState: D3DKMT_DEVICERESET_STATE,
    PageFaultState: D3DKMT_DEVICEPAGEFAULT_STATE,
}

impl Default for D3DKMT_GETDEVICESTATE_DATA {
    fn default() -> Self {
        Self { ExecutionState: 0 }
    }
}

#[repr(C)]
struct D3DKMT_GETDEVICESTATE {
    hDevice: D3DKMT_HANDLE,
    StateType: D3DKMT_DEVICESTATE_TYPE,
    data: D3DKMT_GETDEVICESTATE_DATA,
}

impl Default for D3DKMT_GETDEVICESTATE {
    fn default() -> Self {
        Self {
            hDevice: 0,
            StateType: D3DKMT_DEVICESTATE_TYPE::Execution,
            data: D3DKMT_GETDEVICESTATE_DATA::default(),
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum T0DxgAllocDomain {
    System = 0,
    Local = 1,
    UserMemory = 2,
    UserQueue = 3,
}

// --- AMD Private Escape Data for Queue Creation ---
// Reference: librocdxg src/queues.cpp — hsaKmtCreateQueue uses D3DKMT_ESCAPE
// with AMD's private driver data to create compute queues.

/// AMD private escape code for AQL queue creation
/// This is the escape code used by librocdxg for queue creation via D3DKMTEscape.
const DXG_ESCAPE_CREATE_QUEUE: u32 = 0x100;

#[repr(C)]
#[derive(Default)]
pub struct DxgkEscapeCreateQueue {
    pub escape_code: u32,     // DXG_ESCAPE_CREATE_QUEUE
    pub ring_buffer_va: u64,  // GPU VA of ring buffer
    pub ring_size: u32,       // Ring buffer size in bytes
    pub write_ptr_va: u64,    // GPU VA of write pointer
    pub read_ptr_va: u64,     // GPU VA of read pointer
    pub queue_id: u32,        // OUT: queue ID assigned by driver
    pub doorbell_offset: u32, // OUT: doorbell offset
    pub engine_index: u32,    // Compute engine index
    pub reserved: [u32; 5],
}

#[derive(Debug)]
struct DxgVaAllocator {
    free_segments: Vec<(u64, u64)>,
}

impl DxgVaAllocator {
    fn new(base: u64, size: u64) -> Self {
        Self {
            free_segments: vec![(base, size)],
        }
    }

    fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        for i in 0..self.free_segments.len() {
            let (base, len) = self.free_segments[i];
            let aligned = align_up_u64(base, align);
            let end = base.checked_add(len)?;
            let alloc_end = aligned.checked_add(size)?;
            if alloc_end > end {
                continue;
            }

            self.free_segments.remove(i);
            if aligned > base {
                self.free_segments.push((base, aligned - base));
            }
            if alloc_end < end {
                self.free_segments.push((alloc_end, end - alloc_end));
            }
            self.free_segments.sort_unstable_by_key(|seg| seg.0);
            return Some(aligned);
        }
        None
    }

    fn free(&mut self, addr: u64, size: u64) {
        self.free_segments.push((addr, size));
        self.free_segments.sort_unstable_by_key(|seg| seg.0);

        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.free_segments.len());
        for (base, len) in self.free_segments.drain(..) {
            if let Some((last_base, last_len)) = merged.last_mut() {
                if *last_base + *last_len == base {
                    *last_len += len;
                    continue;
                }
            }
            merged.push((base, len));
        }
        self.free_segments = merged;
    }
}

struct DxgVaHeap {
    base: u64,
    size: u64,
    alloc: std::sync::Mutex<DxgVaAllocator>,
}

impl DxgVaHeap {
    fn new(base: u64, size: u64) -> Self {
        Self {
            base,
            size,
            alloc: std::sync::Mutex::new(DxgVaAllocator::new(base, size)),
        }
    }

    fn alloc(&self, size: u64, align: u64) -> Option<u64> {
        self.alloc.lock().unwrap().alloc(size, align)
    }

    fn free(&self, addr: u64, size: u64) {
        self.alloc.lock().unwrap().free(addr, size);
    }
}

// =============================================================================
// WSL2 DXG Device
// =============================================================================

/// WSL2 DXG GPU device
pub struct WslDxgDevice {
    /// DXG adapter handle
    pub dxg_adapter: D3DKMT_HANDLE,
    /// DXG device handle
    pub dxg_device: D3DKMT_HANDLE,
    /// DXG context handle
    pub dxg_context: D3DKMT_HANDLE,
    /// Paging queue used for VA map/residency operations
    pub dxg_paging_queue: D3DKMT_HANDLE,
    /// Paging fence sync object
    pub dxg_paging_sync_object: D3DKMT_HANDLE,
    /// CPU mapping of the paging fence value
    pub dxg_paging_fence_cpu_va: *mut u64,
    /// GPU ID
    pub gpu_id: u32,
    /// Vendor ID (0x1002 = AMD)
    pub vendor_id: u32,
    /// Device ID
    pub device_id: u32,
    /// VRAM size in bytes
    pub vram_size: u64,
    /// GART size in bytes
    pub gart_size: u64,
    /// Segment ID for VRAM
    pub segment_vram: u32,
    /// Segment ID for GART
    pub segment_gart: u32,
    /// Compute engine index
    pub compute_engine: u32,
    /// Node ordinal used for compute queue creation and clock calibration.
    pub compute_node_ordinal: u32,
    /// Queue engine flag used by thunk-proxy allocation metadata
    pub compute_engine_flag: u32,
    /// Whether the compute engine supports hardware queue submission.
    pub compute_hws_enabled: bool,
    /// Adapter metadata parsed from thunk_proxy
    device_info: DxgThunkDeviceInfo,
    /// Fixed GPU timestamp counter frequency exposed by the DXG thunk.
    gpu_counter_frequency_hz: u64,
    /// Reserved CPU/GPU shared VA space for host-visible system allocations.
    system_heap: DxgVaHeap,
    /// Reserved CPU/GPU shared VA space for local VRAM allocations.
    local_heap: DxgVaHeap,
    /// Latest paging fence value observed from KMT calls
    paging_fence_value: std::sync::atomic::AtomicU64,
}

unsafe impl Send for WslDxgDevice {}
unsafe impl Sync for WslDxgDevice {}

#[derive(Clone, Copy, Debug)]
pub struct DxgClockCalibration {
    pub gpu_frequency_hz: u64,
    pub gpu_clock_counter: u64,
    pub cpu_clock_counter: u64,
}

/// Global singleton
static GLOBAL_WSL_DXG_DEVICE: std::sync::OnceLock<Arc<WslDxgDevice>> = std::sync::OnceLock::new();

impl WslDxgDevice {
    fn execution_state_name(state: u32) -> &'static str {
        match state {
            1 => "ACTIVE",
            2 => "RESET",
            3 => "HUNG",
            4 => "STOPPED",
            5 => "ERROR_OUTOFMEMORY",
            6 => "ERROR_DMAFAULT",
            7 => "ERROR_DMAPAGEFAULT",
            _ => "UNKNOWN",
        }
    }

    pub fn target(&self) -> crate::t0::ir::Target {
        match self.device_info.major() {
            11 => crate::t0::ir::Target::GFX1100,
            12 => crate::t0::ir::Target::GFX1201,
            major => panic!(
                "Unsupported gfx major {} for device 0x{:04X}: no fallback target mapping",
                major, self.device_id
            ),
        }
    }

    fn platform_atomic_support(&self) -> bool {
        dxg_platform_atomic_override().unwrap_or_else(|| self.device_info.platform_atomic_support())
    }

    fn record_paging_fence_value(&self, fence_value: u64) {
        if fence_value == 0 {
            return;
        }
        let mut current = self.paging_fence_value.load(std::sync::atomic::Ordering::Relaxed);
        while current < fence_value {
            match self.paging_fence_value.compare_exchange_weak(
                current,
                fence_value,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    pub fn gpu_counter_frequency_hz(&self) -> Result<u64, String> {
        if self.gpu_counter_frequency_hz == 0 {
            Err("DXG adapter did not expose gpu_counter_frequency".to_string())
        } else {
            Ok(self.gpu_counter_frequency_hz)
        }
    }

    pub fn query_clock_calibration(&self) -> Result<DxgClockCalibration, String> {
        let mut args = D3DKMT_QUERYCLOCKCALIBRATION {
            hAdapter: self.dxg_adapter,
            NodeOrdinal: self.compute_node_ordinal,
            PhysicalAdapterIndex: 0,
            ..Default::default()
        };
        let status = unsafe { D3DKMTQueryClockCalibration(&mut args) };
        if status != 0 {
            return Err(format!(
                "D3DKMTQueryClockCalibration failed: 0x{:08X}",
                status as u32
            ));
        }

        let gpu_frequency_hz = args.ClockData.gpu_frequency_hz();
        let gpu_clock_counter = args.ClockData.gpu_clock_counter();
        let cpu_clock_counter = args.ClockData.cpu_clock_counter();
        if gpu_frequency_hz == 0 {
            return Err("D3DKMTQueryClockCalibration returned zero GpuFrequency".to_string());
        }

        dxg_debug!(
            "[DXG] clock calibration: node={} gpu_freq={} gpu_counter={} cpu_counter={}",
            self.compute_node_ordinal,
            gpu_frequency_hz,
            gpu_clock_counter,
            cpu_clock_counter,
        );

        Ok(DxgClockCalibration {
            gpu_frequency_hz,
            gpu_clock_counter,
            cpu_clock_counter,
        })
    }

    pub fn open() -> Result<Arc<Self>, String> {
        Self::open_with_gpu_id(0)
    }

    pub fn open_fresh() -> Result<Arc<Self>, String> {
        Self::open_fresh_with_gpu_id(0)
    }

    pub fn open_with_gpu_id(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        if let Some(dev) = GLOBAL_WSL_DXG_DEVICE.get() {
            return Ok(Arc::clone(dev));
        }
        let dev = Self::open_device_impl(gpu_id_override)?;
        match GLOBAL_WSL_DXG_DEVICE.set(Arc::clone(&dev)) {
            Ok(()) => Ok(dev),
            Err(_) => Ok(Arc::clone(GLOBAL_WSL_DXG_DEVICE.get().unwrap())),
        }
    }

    pub fn open_fresh_with_gpu_id(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        Self::open_device_impl(gpu_id_override)
    }

    fn open_device_impl(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        // Keep process alive on broken pipe so queue/context Drop can clean up.
        ignore_sigpipe_once();

        let (adapter_handle, vendor_id, device_id) = Self::find_amd_adapter(gpu_id_override)?;
        dxg_debug!("[DXG] Found AMD GPU: vendor=0x{:04X} device=0x{:04X}", vendor_id, device_id);

        let device_info = DxgThunkDeviceInfo::new(adapter_handle)?;

        let compute_engine = device_info.compute_engine();
        let compute_engine_flag = device_info.queue_engine_flag(compute_engine)?;
        let node_ordinal = device_info.engine_ordinal(compute_engine)?;
        let hws_enabled = device_info.hws_enabled(compute_engine)?;
        let disable_gpu_timeout = device_info.should_disable_gpu_timeout(compute_engine)?;
        let gpu_counter_frequency_hz = device_info.gpu_counter_frequency();

        let mut create_device = D3DKMT_CREATEDEVICE {
            hAdapter: adapter_handle,
            ..Default::default()
        };
        let status = unsafe { D3DKMTCreateDevice(&mut create_device) };
        if nt_failed(status) {
            return Err(format!("D3DKMTCreateDevice failed: 0x{:08X}", status as u32));
        }
        let device = create_device.hDevice;
        dxg_debug!("[DXG] Device created: hDevice={}", device);
        Self::set_power_optimization(adapter_handle, device, false);

        let local_heap = match Self::reserve_local_heap_space(adapter_handle, &device_info) {
            Ok(heap) => heap,
            Err(err) => {
                unsafe { Self::close_handles(adapter_handle, device, 0, 0) };
                return Err(err);
            }
        };

        let system_heap = match Self::reserve_system_heap_space(adapter_handle) {
            Ok(heap) => heap,
            Err(err) => {
                Self::release_va_heap_space(adapter_handle, &local_heap, "local heap");
                unsafe { Self::close_handles(adapter_handle, device, 0, 0) };
                return Err(err);
            }
        };

        let context_priv_data = Self::build_context_priv_data(&device_info)?;
        let mut context_flags = 0u32;
        if hws_enabled {
            context_flags |= D3DDDI_CREATECONTEXTFLAGS_HW_QUEUE_SUPPORTED;
        } else if disable_gpu_timeout {
            context_flags |= D3DDDI_CREATECONTEXTFLAGS_DISABLE_GPU_TIMEOUT;
        }

        let mut create_context = D3DKMT_CREATECONTEXTVIRTUAL {
            hDevice: device,
            NodeOrdinal: node_ordinal,
            EngineAffinity: 1,
            Flags: D3DDDI_CREATECONTEXTFLAGS { Value: context_flags },
            pPrivateDriverData: context_priv_data.as_ptr() as *mut c_void,
            PrivateDriverDataSize: context_priv_data.len() as u32,
            ClientHint: D3DKMT_CLIENTHINT_OPENCL,
            ..Default::default()
        };
        let status = unsafe { D3DKMTCreateContextVirtual(&mut create_context) };
        if nt_failed(status) {
            Self::release_va_heap_space(adapter_handle, &system_heap, "system heap");
            Self::release_va_heap_space(adapter_handle, &local_heap, "local heap");
            unsafe { Self::close_handles(adapter_handle, device, 0, 0) };
            return Err(format!("D3DKMTCreateContextVirtual failed: 0x{:08X}", status as u32));
        }
        let context = create_context.hContext;
        dxg_debug!("[DXG] Context created: hContext={}", context);

        let (paging_queue, paging_sync_object, paging_fence_cpu_va) =
            match Self::create_paging_queue(device) {
                Ok(v) => v,
                Err(err) => {
                    Self::release_va_heap_space(adapter_handle, &system_heap, "system heap");
                    Self::release_va_heap_space(adapter_handle, &local_heap, "local heap");
                    unsafe { Self::close_handles(adapter_handle, device, context, 0) };
                    return Err(err);
                }
            };
        dxg_debug!(
            "[DXG] Paging queue created: hPagingQueue={} hSyncObject={}",
            paging_queue, paging_sync_object
        );

        let vram_size = device_info.local_visible_heap_size() + device_info.local_invisible_heap_size();
        let gart_size = device_info.non_local_heap_size();
        if vram_size == 0 || gart_size == 0 {
            Self::release_va_heap_space(adapter_handle, &system_heap, "system heap");
            Self::release_va_heap_space(adapter_handle, &local_heap, "local heap");
            unsafe { Self::close_handles(adapter_handle, device, context, paging_queue) };
            return Err(format!(
                "Failed to determine real heap sizes via adapter info: vram={} gart={}",
                vram_size, gart_size
            ));
        }

        let dev = Arc::new(Self {
            dxg_adapter: adapter_handle,
            dxg_device: device,
            dxg_context: context,
            dxg_paging_queue: paging_queue,
            dxg_paging_sync_object: paging_sync_object,
            dxg_paging_fence_cpu_va: paging_fence_cpu_va,
            gpu_id: gpu_id_override,
            vendor_id,
            device_id,
            vram_size,
            gart_size,
            segment_vram: 0,
            segment_gart: 1,
            compute_engine,
            compute_node_ordinal: node_ordinal,
            compute_engine_flag,
            compute_hws_enabled: hws_enabled,
            device_info,
            gpu_counter_frequency_hz,
            system_heap,
            local_heap,
            paging_fence_value: std::sync::atomic::AtomicU64::new(0),
        });

        dxg_debug!(
            "[DXG] Device initialized: device=0x{:04X} major={} node={} engine={} hws={} platform_atomic={} gpu_counter_hz={} vram={}MB gart={}MB",
            device_id,
            dev.device_info.major(),
            node_ordinal,
            compute_engine,
            hws_enabled,
            dev.platform_atomic_support(),
            gpu_counter_frequency_hz,
            vram_size / 1024 / 1024,
            gart_size / 1024 / 1024
        );

        Ok(dev)
    }

    fn find_amd_adapter(gpu_id_override: u32) -> Result<(D3DKMT_HANDLE, u32, u32), String> {
        // Hard precondition: /dev/dxg must be present and openable.
        let dxg_probe_fd = unsafe {
            let path = std::ffi::CString::new("/dev/dxg").unwrap();
            libc::open(path.as_ptr(), libc::O_RDWR)
        };
        if dxg_probe_fd >= 0 {
            dxg_debug!("[DEBUG] Opened /dev/dxg probe fd={}", dxg_probe_fd);
            unsafe {
                libc::close(dxg_probe_fd);
            }
        } else {
            return Err(format!(
                "/dev/dxg probe failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        if let Some(found) = Self::find_amd_adapter_via_enum2(gpu_id_override) {
            return Ok(found);
        }
        if gpu_id_override != 0 {
            Err(format!(
                "No AMD GPU found via EnumAdapters2 matching device_id=0x{:04X}",
                gpu_id_override
            ))
        } else {
            Err("No AMD GPU found via EnumAdapters2".to_string())
        }
    }

    fn find_amd_adapter_via_enum2(gpu_id_override: u32) -> Option<(D3DKMT_HANDLE, u32, u32)> {
        let mut enum_adapters = D3DKMT_ENUMADAPTERS2 {
            NumAdapters: 0,
            pAdapters: ptr::null_mut(),
        };
        let status = unsafe { D3DKMTEnumAdapters2(&mut enum_adapters) };
        dxg_debug!(
            "[DEBUG] D3DKMTEnumAdapters2(pass1) status: 0x{:08X}, NumAdapters: {}",
            status as u32,
            enum_adapters.NumAdapters
        );
        if status != 0 || enum_adapters.NumAdapters == 0 {
            return None;
        }

        let mut adapters = vec![D3DKMT_ADAPTERINFO::default(); enum_adapters.NumAdapters as usize];
        let mut enum_adapters_filled = D3DKMT_ENUMADAPTERS2 {
            NumAdapters: enum_adapters.NumAdapters,
            pAdapters: adapters.as_mut_ptr(),
        };
        let status = unsafe { D3DKMTEnumAdapters2(&mut enum_adapters_filled) };
        dxg_debug!(
            "[DEBUG] D3DKMTEnumAdapters2(pass2) status: 0x{:08X}, NumAdapters: {}",
            status as u32,
            enum_adapters_filled.NumAdapters
        );
        if status != 0 {
            return None;
        }

        Self::pick_amd_adapter(&adapters[..enum_adapters_filled.NumAdapters as usize], gpu_id_override)
    }

    fn find_amd_adapter_via_enum3(gpu_id_override: u32) -> Option<(D3DKMT_HANDLE, u32, u32)> {
        // Include compute-only, display-only and virtual GPU adapters.
        let filter = (1u64 << 0) | (1u64 << 1) | (1u64 << 2);
        let mut enum_adapters = D3DKMT_ENUMADAPTERS3 {
            Filter: filter,
            NumAdapters: 0,
            pAdapters: ptr::null_mut(),
        };
        let status = unsafe { D3DKMTEnumAdapters3(&mut enum_adapters) };
        dxg_debug!(
            "[DEBUG] D3DKMTEnumAdapters3(pass1) status: 0x{:08X}, NumAdapters: {}, Filter=0x{:X}",
            status as u32,
            enum_adapters.NumAdapters,
            filter
        );
        if status != 0 || enum_adapters.NumAdapters == 0 {
            return None;
        }

        let mut adapters = vec![D3DKMT_ADAPTERINFO::default(); enum_adapters.NumAdapters as usize];
        let mut enum_adapters_filled = D3DKMT_ENUMADAPTERS3 {
            Filter: filter,
            NumAdapters: enum_adapters.NumAdapters,
            pAdapters: adapters.as_mut_ptr(),
        };
        let status = unsafe { D3DKMTEnumAdapters3(&mut enum_adapters_filled) };
        dxg_debug!(
            "[DEBUG] D3DKMTEnumAdapters3(pass2) status: 0x{:08X}, NumAdapters: {}",
            status as u32,
            enum_adapters_filled.NumAdapters
        );
        if status != 0 {
            return None;
        }

        Self::pick_amd_adapter(&adapters[..enum_adapters_filled.NumAdapters as usize], gpu_id_override)
    }

    fn pick_amd_adapter(
        adapters: &[D3DKMT_ADAPTERINFO],
        gpu_id_override: u32,
    ) -> Option<(D3DKMT_HANDLE, u32, u32)> {
        let mut first_amd: Option<(D3DKMT_HANDLE, u32, u32)> = None;
        let mut preferred_gfx12: Option<(D3DKMT_HANDLE, u32, u32)> = None;
        for (i, adapter) in adapters.iter().enumerate() {
            let mut device_ids = D3DKMT_QUERY_DEVICE_IDS::default();
            let query = D3DKMT_QUERYADAPTERINFO {
                hAdapter: adapter.hAdapter,
                Type: KMTQUERYADAPTERINFOTYPE::PhysicalAdapterDeviceIds,
                pPrivateDriverData: &mut device_ids as *mut _ as *mut c_void,
                PrivateDriverDataSize: std::mem::size_of::<D3DKMT_QUERY_DEVICE_IDS>() as u32,
            };
            let qstatus = unsafe { D3DKMTQueryAdapterInfo(&query) };
            dxg_debug!(
                "[DEBUG] Adapter {}: hAdapter={} query=0x{:08X} vendor=0x{:04X} device=0x{:04X}",
                i,
                adapter.hAdapter,
                qstatus as u32,
                device_ids.DeviceIds.VendorID,
                device_ids.DeviceIds.DeviceID
            );
            if qstatus != 0 || device_ids.DeviceIds.VendorID != 0x1002 {
                continue;
            }

            let candidate = (
                adapter.hAdapter,
                device_ids.DeviceIds.VendorID,
                device_ids.DeviceIds.DeviceID,
            );

            if gpu_id_override != 0 && device_ids.DeviceIds.DeviceID == gpu_id_override {
                dxg_debug!(
                    "[DEBUG] Selected AMD adapter by override: device=0x{:04X}",
                    device_ids.DeviceIds.DeviceID
                );
                return Some(candidate);
            }

            // Prefer known gfx1201 class device IDs when no explicit override is provided.
            if gpu_id_override == 0 && matches!(device_ids.DeviceIds.DeviceID, 0x7550 | 0x7551) {
                preferred_gfx12 = Some(candidate);
            }
            if first_amd.is_none() {
                first_amd = Some(candidate);
            }
        }
        if gpu_id_override != 0 {
            return None;
        }
        preferred_gfx12.or(first_amd)
    }

    fn build_context_priv_data(device_info: &DxgThunkDeviceInfo) -> Result<Vec<u8>, String> {
        thunk_proxy::build_context_priv_data(device_info)
    }

    fn reserve_system_heap_space(adapter: D3DKMT_HANDLE) -> Result<DxgVaHeap, String> {
        let mut info = std::mem::MaybeUninit::<libc::sysinfo>::uninit();
        let ret = unsafe { libc::sysinfo(info.as_mut_ptr()) };
        if ret != 0 {
            return Err(format!("sysinfo failed: {}", std::io::Error::last_os_error()));
        }
        let info = unsafe { info.assume_init() };
        let total_ram = (info.totalram as u128).saturating_mul(info.mem_unit as u128);
        let alignment = 0x1_0000_0000u64;
        let max_size = 0x1000_0000_0000u64;
        let size = std::cmp::min(
            align_up_u64(total_ram.min(u64::MAX as u128) as u64, alignment).saturating_mul(2),
            max_size,
        );
        let base = Self::reserve_svm_space(adapter, size, alignment)?;
        Ok(DxgVaHeap::new(base, size))
    }

    fn reserve_local_heap_space(
        adapter: D3DKMT_HANDLE,
        device_info: &DxgThunkDeviceInfo,
    ) -> Result<DxgVaHeap, String> {
        let alignment = 0x1_0000_0000u64;
        let local_size = if device_info.is_dgpu() {
            device_info
                .local_visible_heap_size()
                .saturating_add(device_info.local_invisible_heap_size())
        } else {
            device_info
                .local_visible_heap_size()
                .saturating_add(device_info.local_invisible_heap_size())
                .saturating_add(device_info.non_local_heap_size())
        };
        if local_size == 0 {
            return Err("Adapter reports zero local heap size".to_string());
        }
        let size = align_up_u64(local_size, alignment).saturating_mul(4);
        let base = Self::reserve_svm_space(adapter, size, alignment)?;
        Ok(DxgVaHeap::new(base, size))
    }

    fn reserve_svm_space(adapter: D3DKMT_HANDLE, size: u64, align: u64) -> Result<u64, String> {
        let reservation_size = size
            .checked_add(align)
            .ok_or_else(|| format!("SVM reservation size overflow: size={} align={}", size, align))?;
        let mut last_err = None;

        for _ in 0..16 {
            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    reservation_size as usize,
                    PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr == MAP_FAILED || ptr.is_null() {
                last_err = Some(format!("mmap reserve_svm_space failed: {}", std::io::Error::last_os_error()));
                continue;
            }

            let cpu_base = ptr as u64;
            match Self::reserve_gpu_virtual_address_range(
                adapter,
                size,
                cpu_base,
                cpu_base.saturating_add(reservation_size).saturating_add(1),
            ) {
                Ok(gpu_base) => {
                    let left = gpu_base.saturating_sub(cpu_base);
                    let right = align.saturating_sub(left);
                    if left > 0 {
                        unsafe { munmap(cpu_base as *mut c_void, left as usize) };
                    }
                    if right > 0 {
                        unsafe { munmap((gpu_base + size) as *mut c_void, right as usize) };
                    }
                    return Ok(gpu_base);
                }
                Err(err) => {
                    unsafe { munmap(ptr, reservation_size as usize) };
                    last_err = Some(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| "Failed to reserve shared system heap space".to_string()))
    }

    fn reserve_gpu_virtual_address_range(
        adapter: D3DKMT_HANDLE,
        size: u64,
        minimum_address: u64,
        maximum_address: u64,
    ) -> Result<u64, String> {
        let mut args = D3DDDI_RESERVEGPUVIRTUALADDRESS {
            hAdapter: adapter,
            MinimumAddress: minimum_address,
            MaximumAddress: maximum_address,
            Size: size,
            ..Default::default()
        };
        let status = unsafe { D3DKMTReserveGpuVirtualAddress(&mut args) };
        if nt_failed(status) || args.VirtualAddress == 0 {
            return Err(format!(
                "D3DKMTReserveGpuVirtualAddress(range) failed: 0x{:08X} min=0x{:016X} max=0x{:016X} size=0x{:X}",
                status as u32,
                minimum_address,
                maximum_address,
                size
            ));
        }
        Ok(args.VirtualAddress)
    }

    fn commit_system_heap_space(addr: u64, size: usize) -> Result<*mut c_void, String> {
        let ptr = unsafe {
            mmap(
                addr as *mut c_void,
                size,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS | libc::MAP_FIXED | MAP_NORESERVE | MAP_UNINITIALIZED,
                -1,
                0,
            )
        };
        if ptr == MAP_FAILED || ptr.is_null() {
            return Err(format!("commit_system_heap_space failed: {}", std::io::Error::last_os_error()));
        }
        if ptr as u64 != addr {
            return Err(format!(
                "commit_system_heap_space returned wrong address: want=0x{:016X} got=0x{:016X}",
                addr,
                ptr as u64
            ));
        }
        let advise_ret = unsafe { libc::madvise(ptr, size, libc::MADV_DONTFORK) };
        if advise_ret != 0 {
            eprintln!(
                "[DXG] WARN: madvise(MADV_DONTFORK) failed for 0x{:016X}: {}",
                addr,
                std::io::Error::last_os_error()
            );
        }
        Ok(ptr)
    }

    fn decommit_system_heap_space(addr: *mut c_void, size: usize) {
        let ptr = unsafe {
            mmap(
                addr,
                size,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | libc::MAP_FIXED | MAP_NORESERVE | MAP_UNINITIALIZED,
                -1,
                0,
            )
        };
        if ptr == MAP_FAILED {
            eprintln!(
                "[DXG] WARN: decommit_system_heap_space failed for {:?}: {}",
                addr,
                std::io::Error::last_os_error()
            );
        }
    }

    fn release_va_heap_space(adapter: D3DKMT_HANDLE, heap: &DxgVaHeap, heap_name: &str) {
        let args = D3DKMT_FREEGPUVIRTUALADDRESS {
            hAdapter: adapter,
            BaseAddress: heap.base,
            Size: heap.size,
        };
        let status = unsafe { D3DKMTFreeGpuVirtualAddress(&args) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTFreeGpuVirtualAddress({}) failed: 0x{:08X}",
                heap_name,
                status as u32,
            );
        }
        let unmap_ret = unsafe { munmap(heap.base as *mut c_void, heap.size as usize) };
        if unmap_ret != 0 {
            eprintln!(
                "[DXG] WARN: munmap({}) failed: {}",
                heap_name,
                std::io::Error::last_os_error(),
            );
        }
    }

    fn create_paging_queue(
        device: D3DKMT_HANDLE,
    ) -> Result<(D3DKMT_HANDLE, D3DKMT_HANDLE, *mut u64), String> {
        let mut args = D3DKMT_CREATEPAGINGQUEUE {
            hDevice: device,
            Priority: D3DDDI_PAGINGQUEUE_PRIORITY::Normal,
            ..Default::default()
        };
        let status = unsafe { D3DKMTCreatePagingQueue(&mut args) };
        if nt_failed(status) {
            return Err(format!("D3DKMTCreatePagingQueue failed: 0x{:08X}", status as u32));
        }
        Ok((
            args.hPagingQueue,
            args.hSyncObject,
            args.FenceValueCPUVirtualAddress as *mut u64,
        ))
    }

    fn create_compute_context(&self) -> Result<D3DKMT_HANDLE, String> {
        let mut context_flags = 0u32;
        if self.compute_hws_enabled || dxg_force_hw_queue() {
            context_flags |= D3DDDI_CREATECONTEXTFLAGS_HW_QUEUE_SUPPORTED;
        } else if self.device_info.should_disable_gpu_timeout(self.compute_engine)? {
            context_flags |= D3DDDI_CREATECONTEXTFLAGS_DISABLE_GPU_TIMEOUT;
        }

        let mut context_priv_data = Self::build_context_priv_data(&self.device_info)?;
        let mut create_context = D3DKMT_CREATECONTEXTVIRTUAL {
            hDevice: self.dxg_device,
            NodeOrdinal: self.compute_node_ordinal,
            EngineAffinity: 1,
            Flags: D3DDDI_CREATECONTEXTFLAGS { Value: context_flags },
            pPrivateDriverData: context_priv_data.as_mut_ptr() as *mut c_void,
            PrivateDriverDataSize: context_priv_data.len() as u32,
            ClientHint: D3DKMT_CLIENTHINT_OPENCL,
            ..Default::default()
        };
        let status = unsafe { D3DKMTCreateContextVirtual(&mut create_context) };
        if nt_failed(status) {
            return Err(format!("D3DKMTCreateContextVirtual failed: 0x{:08X}", status as u32));
        }
        Ok(create_context.hContext)
    }

    fn set_power_optimization(
        adapter: D3DKMT_HANDLE,
        device: D3DKMT_HANDLE,
        restore: bool,
    ) {
        if adapter == 0 || device == 0 {
            return;
        }

        let mut priv_data = thunk_proxy::build_power_opt_priv_data(restore);
        let escape = D3DKMT_ESCAPE {
            hAdapter: adapter,
            hDevice: device,
            Type: D3DKMT_ESCAPETYPE::DriverPrivate,
            Flags: D3DDDI_ESCAPEFLAGS { Value: 0x1 },
            pPrivateDriverData: priv_data.as_mut_ptr() as *mut c_void,
            PrivateDriverDataSize: priv_data.len() as u32,
            hContext: 0,
        };
        let status = unsafe { D3DKMTEscape(&escape) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTEscape(power_opt restore={}) failed: 0x{:08X}",
                restore,
                status as u32,
            );
        } else {
            dxg_debug!("[DXG] power optimization restore={} applied", restore);
        }
    }

    fn create_hw_queue(
        &self,
        context: D3DKMT_HANDLE,
    ) -> Result<(D3DKMT_HANDLE, D3DKMT_HANDLE, *mut u64), String> {
        if !self.compute_hws_enabled && !dxg_force_hw_queue() {
            return Err(format!(
                "Compute engine {} does not expose HWS on this adapter",
                self.compute_engine
            ));
        }

        let mut priv_data = thunk_proxy::build_hw_queue_priv_data(
            self.device_info.state_shadowing_by_cpfw(),
            thunk_proxy::DxgSchedLevel::Normal,
        );
        let mut create = D3DKMT_CREATEHWQUEUE {
            hHwContext: context,
            Flags: D3DDDI_CREATEHWQUEUEFLAGS {
                Value: if self.device_info.should_disable_gpu_timeout(self.compute_engine)? {
                    1
                } else {
                    0
                },
            },
            PrivateDriverDataSize: priv_data.len() as u32,
            pPrivateDriverData: priv_data.as_mut_ptr() as *mut c_void,
            ..Default::default()
        };

        let status = unsafe { D3DKMTCreateHwQueue(&mut create) };
        if nt_failed(status) {
            return Err(format!("D3DKMTCreateHwQueue failed: 0x{:08X}", status as u32));
        }

        Ok((
            create.hHwQueue,
            create.hHwQueueProgressFence,
            create.HwQueueProgressFenceCPUVirtualAddress as *mut u64,
        ))
    }

    fn destroy_hw_queue(&self, hw_queue: D3DKMT_HANDLE) {
        if hw_queue == 0 {
            return;
        }
        let args = D3DKMT_DESTROYHWQUEUE { hHwQueue: hw_queue };
        let status = unsafe { D3DKMTDestroyHwQueue(&args) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTDestroyHwQueue failed: 0x{:08X}",
                status as u32
            );
        }
    }

    fn destroy_context(&self, context: D3DKMT_HANDLE) {
        if context == 0 {
            return;
        }
        let args = D3DKMT_DESTROYCONTEXT { hContext: context };
        let status = unsafe { D3DKMTDestroyContext(&args) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTDestroyContext failed: 0x{:08X}",
                status as u32
            );
        }
    }

    fn reserve_gpu_virtual_address(&self, size: usize) -> Result<u64, String> {
        let mut args = D3DDDI_RESERVEGPUVIRTUALADDRESS {
            hAdapter: self.dxg_adapter,
            Size: size as u64,
            ..Default::default()
        };
        let status = unsafe { D3DKMTReserveGpuVirtualAddress(&mut args) };
        if nt_failed(status) {
            return Err(format!(
                "D3DKMTReserveGpuVirtualAddress failed: 0x{:08X}",
                status as u32
            ));
        }
        Ok(args.VirtualAddress)
    }

    fn wait_for_sync_object_value(
        &self,
        sync_object: D3DKMT_HANDLE,
        fence_value: u64,
    ) -> Result<(), String> {
        if sync_object == 0 || fence_value == 0 {
            return Ok(());
        }

        dxg_debug!(
            "[DXG] wait_for_sync_object_value begin: sync_object={} fence_value={}",
            sync_object,
            fence_value
        );
        let handles = [sync_object];
        let fence_values = [fence_value];
        let args = D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU {
            hDevice: self.dxg_device,
            ObjectCount: 1,
            ObjectHandleArray: handles.as_ptr(),
            FenceValueArray: fence_values.as_ptr(),
            hAsyncEvent: ptr::null_mut(),
            Flags: D3DDDI_WAITFORSYNCHRONIZATIONOBJECTFROMCPU_FLAGS { Value: 0 },
        };
        let status = unsafe { D3DKMTWaitForSynchronizationObjectFromCpu(&args) };
        if nt_failed(status) {
            return Err(format!(
                "D3DKMTWaitForSynchronizationObjectFromCpu failed: 0x{:08X}",
                status as u32
            ));
        }
        dxg_debug!(
            "[DXG] wait_for_sync_object_value done: sync_object={} fence_value={}",
            sync_object,
            fence_value
        );
        Ok(())
    }

    fn wait_for_sync_object_value_from_gpu(
        &self,
        context: D3DKMT_HANDLE,
        sync_object: D3DKMT_HANDLE,
        fence_value: u64,
    ) -> Result<(), String> {
        if context == 0 || sync_object == 0 || fence_value == 0 {
            return Ok(());
        }

        let handles = [sync_object];
        let fence_values = [fence_value];
        let mut data = D3DKMT_WAITSYNCFGPU_DATA::default();
        data.MonitoredFenceValueArray = fence_values.as_ptr();
        let args = D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU {
            hContext: context,
            ObjectCount: 1,
            ObjectHandleArray: handles.as_ptr(),
            data,
        };
        let status = unsafe { D3DKMTWaitForSynchronizationObjectFromGpu(&args) };
        if nt_failed(status) {
            return Err(format!(
                "D3DKMTWaitForSynchronizationObjectFromGpu failed: 0x{:08X}",
                status as u32
            ));
        }
        Ok(())
    }

    fn wait_on_paging_fence(&self, fence_value: u64) -> Result<(), String> {
        if fence_value == 0 {
            return Ok(());
        }
        self.record_paging_fence_value(fence_value);
        self.wait_for_sync_object_value(self.dxg_paging_sync_object, fence_value)
    }

    fn wait_on_latest_paging_fence_from_gpu(&self, context: D3DKMT_HANDLE) -> Result<(), String> {
        let fence_value = self.paging_fence_value.load(std::sync::atomic::Ordering::Relaxed);
        if fence_value == 0 {
            return Ok(());
        }
        if !self.dxg_paging_fence_cpu_va.is_null() {
            let completed = unsafe { ptr::read_volatile(self.dxg_paging_fence_cpu_va) };
            if completed >= fence_value {
                return Ok(());
            }
        }
        self.wait_for_sync_object_value_from_gpu(context, self.dxg_paging_sync_object, fence_value)
    }

    fn submit_command_to_hw_queue_on_context(
        &self,
        context: D3DKMT_HANDLE,
        hw_queue: D3DKMT_HANDLE,
        command_buffer: u64,
        command_length: u32,
        fence_value: u64,
    ) -> Result<(), String> {
        let mut priv_data = thunk_proxy::build_submit_priv_data(
            hw_queue,
            command_buffer,
            command_length,
            true,
        );
        self.wait_on_latest_paging_fence_from_gpu(context)?;
        let args = D3DKMT_SUBMITCOMMANDTOHWQUEUE {
            hHwQueue: hw_queue,
            HwQueueProgressFenceId: fence_value,
            CommandBuffer: command_buffer,
            CommandLength: command_length,
            PrivateDriverDataSize: priv_data.len() as u32,
            pPrivateDriverData: priv_data.as_mut_ptr() as *mut c_void,
            NumPrimaries: 0,
            WrittenPrimaries: ptr::null(),
        };
        let status = unsafe { D3DKMTSubmitCommandToHwQueue(&args) };
        if nt_failed(status) {
            return Err(format!(
                "D3DKMTSubmitCommandToHwQueue failed: 0x{:08X}",
                status as u32
            ));
        }
        Ok(())
    }

    fn submit_command_to_hw_queue(
        &self,
        hw_queue: D3DKMT_HANDLE,
        command_buffer: u64,
        command_length: u32,
        fence_value: u64,
    ) -> Result<(), String> {
        self.submit_command_to_hw_queue_on_context(
            self.dxg_context,
            hw_queue,
            command_buffer,
            command_length,
            fence_value,
        )
    }

    fn submit_command_on_context(
        &self,
        context: D3DKMT_HANDLE,
        submit_queue_handle: D3DKMT_HANDLE,
        command_buffer: u64,
        command_length: u32,
        progress_sync_object: D3DKMT_HANDLE,
        fence_value: u64,
    ) -> Result<(), String> {
        let mut priv_data =
            thunk_proxy::build_submit_priv_data(
                submit_queue_handle,
                command_buffer,
                command_length,
                false,
            );
        self.wait_on_latest_paging_fence_from_gpu(context)?;
        let mut args = D3DKMT_SUBMITCOMMAND {
            Commands: command_buffer,
            CommandLength: command_length,
            Flags: D3DKMT_SUBMITCOMMANDFLAGS { Value: 0 },
            PresentHistoryToken: 0,
            BroadcastContextCount: 1,
            BroadcastContext: [0; 64],
            pPrivateDriverData: priv_data.as_mut_ptr() as *mut c_void,
            PrivateDriverDataSize: priv_data.len() as u32,
            NumPrimaries: 0,
            WrittenPrimaries: [0; 16],
            NumHistoryBuffers: 0,
            HistoryBufferArray: ptr::null_mut(),
        };
        args.BroadcastContext[0] = context;

        let status = unsafe { D3DKMTSubmitCommand(&args) };
        if nt_failed(status) {
            return Err(format!("D3DKMTSubmitCommand failed: 0x{:08X}", status as u32));
        }

        self.signal_sync_object_from_gpu_on_context(context, progress_sync_object, fence_value)
    }

    fn submit_command(
        &self,
        command_buffer: u64,
        command_length: u32,
        progress_sync_object: D3DKMT_HANDLE,
        fence_value: u64,
    ) -> Result<(), String> {
        self.submit_command_on_context(
            self.dxg_context,
            0,
            command_buffer,
            command_length,
            progress_sync_object,
            fence_value,
        )
    }

    fn query_execution_state(&self) -> Result<u32, String> {
        let mut args = D3DKMT_GETDEVICESTATE {
            hDevice: self.dxg_device,
            StateType: D3DKMT_DEVICESTATE_TYPE::Execution,
            ..Default::default()
        };
        let status = unsafe { D3DKMTGetDeviceState(&mut args) };
        if nt_failed(status) {
            return Err(format!("D3DKMTGetDeviceState(EXECUTION) failed: 0x{:08X}", status as u32));
        }
        Ok(unsafe { args.data.ExecutionState })
    }

    fn query_reset_state(&self) -> Result<D3DKMT_DEVICERESET_STATE, String> {
        let mut args = D3DKMT_GETDEVICESTATE {
            hDevice: self.dxg_device,
            StateType: D3DKMT_DEVICESTATE_TYPE::Reset,
            ..Default::default()
        };
        let status = unsafe { D3DKMTGetDeviceState(&mut args) };
        if nt_failed(status) {
            return Err(format!("D3DKMTGetDeviceState(RESET) failed: 0x{:08X}", status as u32));
        }
        Ok(unsafe { args.data.ResetState })
    }

    fn query_page_fault_state(&self) -> Result<D3DKMT_DEVICEPAGEFAULT_STATE, String> {
        let mut args = D3DKMT_GETDEVICESTATE {
            hDevice: self.dxg_device,
            StateType: D3DKMT_DEVICESTATE_TYPE::PageFault,
            ..Default::default()
        };
        let status = unsafe { D3DKMTGetDeviceState(&mut args) };
        if nt_failed(status) {
            return Err(format!("D3DKMTGetDeviceState(PAGE_FAULT) failed: 0x{:08X}", status as u32));
        }
        Ok(unsafe { args.data.PageFaultState })
    }

    fn describe_device_state(&self) -> String {
        let execution = match self.query_execution_state() {
            Ok(state) => format!(
                "execution={}({})",
                state,
                Self::execution_state_name(state),
            ),
            Err(err) => format!("execution_error={}", err),
        };
        let reset = match self.query_reset_state() {
            Ok(state) => format!("reset=0x{:08X}", state.Value),
            Err(err) => format!("reset_error={}", err),
        };
        let page_fault = match self.query_page_fault_state() {
            Ok(state) => format!(
                "page_fault_va=0x{:016X} flags=0x{:08X} error=0x{:08X} primitive_seq=0x{:016X}",
                state.FaultedVirtualAddress,
                state.PageFaultFlags,
                state.FaultErrorCode,
                state.FaultedPrimitiveAPISequenceNumber,
            ),
            Err(err) => format!("page_fault_error={}", err),
        };
        format!("{execution} {reset} {page_fault}")
    }

    fn signal_sync_object_from_gpu_on_context(
        &self,
        context: D3DKMT_HANDLE,
        sync_object: D3DKMT_HANDLE,
        fence_value: u64,
    ) -> Result<(), String> {
        let handles = [sync_object];
        let values = [fence_value];
        let mut data = D3DKMT_SIGNALSYNCFGPU_DATA::default();
        unsafe {
            data.MonitoredFenceValueArray = values.as_ptr();
        }
        let args = D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU {
            hContext: context,
            ObjectCount: 1,
            ObjectHandleArray: handles.as_ptr(),
            data,
        };
        let status = unsafe { D3DKMTSignalSynchronizationObjectFromGpu(&args) };
        if nt_failed(status) {
            return Err(format!(
                "D3DKMTSignalSynchronizationObjectFromGpu failed: 0x{:08X}",
                status as u32
            ));
        }
        Ok(())
    }

    fn signal_sync_object_from_gpu(
        &self,
        sync_object: D3DKMT_HANDLE,
        fence_value: u64,
    ) -> Result<(), String> {
        self.signal_sync_object_from_gpu_on_context(self.dxg_context, sync_object, fence_value)
    }

    fn free_gpu_virtual_address(&self, gpu_va: u64, size: usize) {
        if gpu_va == 0 {
            return;
        }
        let args = D3DKMT_FREEGPUVIRTUALADDRESS {
            hAdapter: self.dxg_adapter,
            BaseAddress: gpu_va,
            Size: size as u64,
        };
        let status = unsafe { D3DKMTFreeGpuVirtualAddress(&args) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTFreeGpuVirtualAddress failed: 0x{:08X}",
                status as u32
            );
        }
    }

    fn lock_allocation(&self, handle: D3DKMT_HANDLE) -> Result<*mut c_void, String> {
        let mut args = D3DKMT_LOCK2 {
            hDevice: self.dxg_device,
            hAllocation: handle,
            ..Default::default()
        };
        let status = unsafe { D3DKMTLock2(&mut args) };
        if nt_failed(status) {
            return Err(format!("D3DKMTLock2 failed: 0x{:08X}", status as u32));
        }
        Ok(args.pData)
    }

    fn unlock_allocation(&self, handle: D3DKMT_HANDLE) {
        let args = D3DKMT_UNLOCK2 {
            hDevice: self.dxg_device,
            hAllocation: handle,
        };
        let status = unsafe { D3DKMTUnlock2(&args) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTUnlock2 failed for allocation {}: 0x{:08X}",
                handle,
                status as u32
            );
        }
    }

    unsafe fn close_handles(
        adapter: D3DKMT_HANDLE,
        device: D3DKMT_HANDLE,
        context: D3DKMT_HANDLE,
        paging_queue: D3DKMT_HANDLE,
    ) {
        if context != 0 {
            let args = D3DKMT_DESTROYCONTEXT { hContext: context };
            let _ = D3DKMTDestroyContext(&args);
        }
        if paging_queue != 0 {
            let args = D3DDDI_DESTROYPAGINGQUEUE {
                hPagingQueue: paging_queue,
            };
            let _ = D3DKMTDestroyPagingQueue(&args);
        }
        if device != 0 {
            Self::set_power_optimization(adapter, device, true);
            let args = D3DKMT_DESTROYDEVICE { hDevice: device };
            let _ = D3DKMTDestroyDevice(&args);
        }
        if adapter != 0 {
            let args = D3DKMT_CLOSEADAPTER { hAdapter: adapter };
            let _ = D3DKMTCloseAdapter(&args);
        }
    }

    // ── Memory Allocation ───────────────────────────────────────────────────

    fn select_alloc_domain(flags: MemoryFlags) -> T0DxgAllocDomain {
        if flags.gart || flags.coherent || flags.uncached || flags.fine_grain || flags.kernarg {
            T0DxgAllocDomain::System
        } else if flags.vram {
            // Current WSL2 DXG path does not provide a stable CPU mapping for local
            // VRAM on AMD. Preserve the existing host-visible buffer contract by
            // backing public VRAM-style allocations with system memory for now.
            T0DxgAllocDomain::System
        } else {
            T0DxgAllocDomain::Local
        }
    }

    fn select_mem_flags(flags: MemoryFlags) -> u32 {
        let mut mem_flags = 0u32;
        if flags.coherent || flags.fine_grain {
            mem_flags |= T0_DXG_MEM_FLAG_FINE_GRAIN;
        }
        if flags.kernarg {
            mem_flags |= T0_DXG_MEM_FLAG_KERNARG;
        }
        mem_flags
    }

    fn select_map_protection(flags: MemoryFlags) -> D3DDDIGPUVIRTUALADDRESS_PROTECTION_TYPE {
        let mut value = 0u64;
        if flags.writable || flags.public || flags.gart || flags.coherent {
            value |= D3DDDI_GPU_VA_PROTECTION_WRITE;
        }
        if flags.executable {
            value |= D3DDDI_GPU_VA_PROTECTION_EXECUTE;
        }
        if value == 0 {
            value = D3DDDI_GPU_VA_PROTECTION_WRITE;
        }
        D3DDDIGPUVIRTUALADDRESS_PROTECTION_TYPE { Value: value }
    }

    fn destroy_allocation_handle(&self, handle: D3DKMT_HANDLE) {
        if handle == 0 {
            return;
        }
        let handles = [handle];
        let destroy = D3DKMT_DESTROYALLOCATION2 {
            hDevice: self.dxg_device,
            phAllocationList: handles.as_ptr(),
            AllocationCount: handles.len() as u32,
            ..Default::default()
        };
        let status = unsafe { D3DKMTDestroyAllocation2(&destroy) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTDestroyAllocation2 failed: 0x{:08X}",
                status as u32
            );
        }
    }

    /// Allocate GPU memory via D3DKMTCreateAllocation2.
    pub fn alloc_memory(self: &Arc<Self>, size: usize, flags: MemoryFlags) -> Result<WslGpuMemory, String> {
        self.alloc_memory_in_domain(size, flags, Self::select_alloc_domain(flags))
    }

    fn alloc_memory_in_domain(
        self: &Arc<Self>,
        size: usize,
        flags: MemoryFlags,
        alloc_domain: T0DxgAllocDomain,
    ) -> Result<WslGpuMemory, String> {
        let aligned_size = align_up(size, PAGE_SIZE);
        let mem_flags = Self::select_mem_flags(flags);
        let (reserved_gpu_va, sys_mem) = if matches!(alloc_domain, T0DxgAllocDomain::System | T0DxgAllocDomain::UserMemory) {
            let align = if aligned_size as u64 >= GPU_HUGE_PAGE_SIZE_U64 {
                GPU_HUGE_PAGE_SIZE_U64
            } else {
                64 * 1024
            };
            let addr = self
                .system_heap
                .alloc(aligned_size as u64, align)
                .ok_or_else(|| format!("Out of reserved system heap VA space for {} bytes", aligned_size))?;
            match Self::commit_system_heap_space(addr, aligned_size) {
                Ok(ptr) => (addr, ptr),
                Err(err) => {
                    self.system_heap.free(addr, aligned_size as u64);
                    return Err(err);
                }
            }
        } else if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
            let align = if aligned_size as u64 >= GPU_HUGE_PAGE_SIZE_U64 {
                GPU_HUGE_PAGE_SIZE_U64
            } else {
                64 * 1024
            };
            let addr = self
                .local_heap
                .alloc(aligned_size as u64, align)
                .ok_or_else(|| format!("Out of reserved local heap VA space for {} bytes", aligned_size))?;
            (addr, ptr::null_mut())
        } else {
            (self.reserve_gpu_virtual_address(aligned_size)?, ptr::null_mut())
        };

        if !sys_mem.is_null() && sys_mem as u64 != reserved_gpu_va {
            Self::decommit_system_heap_space(sys_mem, aligned_size);
            self.system_heap.free(reserved_gpu_va, aligned_size as u64);
            return Err(format!(
                "System heap CPU/GPU VA mismatch: gpu_va=0x{:016X} cpu_va=0x{:016X}",
                reserved_gpu_va,
                sys_mem as u64
            ));
        }

        let alloc_addr = if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
            reserved_gpu_va
        } else {
            0
        };
        let mut priv_drv_data = thunk_proxy::build_alloc_priv_drv_data();
        let mut priv_alloc_data = match thunk_proxy::build_alloc_priv_data(
            aligned_size as u64,
            alloc_domain as u32,
            alloc_addr,
            mem_flags,
            self.compute_engine_flag,
            &self.device_info,
        ) {
            Ok(data) => data,
            Err(err) => {
                if sys_mem.is_null() {
                    if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
                        self.local_heap.free(reserved_gpu_va, aligned_size as u64);
                    } else {
                        self.free_gpu_virtual_address(reserved_gpu_va, aligned_size);
                    }
                } else {
                    Self::decommit_system_heap_space(sys_mem, aligned_size);
                    self.system_heap.free(reserved_gpu_va, aligned_size as u64);
                }
                return Err(err);
            }
        };

        let mut alloc_info = D3DDDI_ALLOCATIONINFO2 {
            pSystemMem: if !sys_mem.is_null() {
                sys_mem
            } else {
                ptr::null_mut()
            },
            pPrivateDriverData: priv_alloc_data.as_mut_ptr() as *mut c_void,
            PrivateDriverDataSize: priv_alloc_data.len() as u32,
            VidPnSourceId: D3DDDI_ID_UNINITIALIZED,
            ..Default::default()
        };

        let mut create_alloc = D3DKMT_CREATEALLOCATION {
            hDevice: self.dxg_device,
            pPrivateDriverData: priv_drv_data.as_mut_ptr() as *mut c_void,
            PrivateDriverDataSize: priv_drv_data.len() as u32,
            NumAllocations: 1,
            pAllocationInfo2: &mut alloc_info,
            ..Default::default()
        };

        dxg_debug!(
            "[DXG] CreateAllocation2: size={} domain={} mem_flags=0x{:X} reserve_va=0x{:016X} sys_mem={:?} drv_size=0x{:X} alloc_size=0x{:X}",
            aligned_size,
            match alloc_domain {
                T0DxgAllocDomain::System => "system",
                T0DxgAllocDomain::Local => "local",
                T0DxgAllocDomain::UserMemory => "user",
                T0DxgAllocDomain::UserQueue => "queue",
            },
            mem_flags,
            reserved_gpu_va,
            alloc_info.pSystemMem,
            create_alloc.PrivateDriverDataSize,
            alloc_info.PrivateDriverDataSize,
        );
        dxg_debug!(
            "[DXG] alloc blobs: drv=[{}] alloc=[{}]",
            hex_prefix(&priv_drv_data, 32),
            hex_prefix(&priv_alloc_data, 64)
        );

        let status = unsafe { D3DKMTCreateAllocation2(&mut create_alloc) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] CreateAllocation2 failed: status=0x{:08X} reserve_va=0x{:016X} sys_mem={:?} alloc_gpu_va=0x{:016X} drv_size=0x{:X} alloc_size=0x{:X}",
                status as u32,
                reserved_gpu_va,
                sys_mem,
                alloc_info.GpuVirtualAddress,
                create_alloc.PrivateDriverDataSize,
                alloc_info.PrivateDriverDataSize,
            );
            if sys_mem.is_null() {
                if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
                    self.local_heap.free(reserved_gpu_va, aligned_size as u64);
                } else {
                    self.free_gpu_virtual_address(reserved_gpu_va, aligned_size);
                }
            } else {
                Self::decommit_system_heap_space(sys_mem, aligned_size);
                self.system_heap.free(reserved_gpu_va, aligned_size as u64);
            }
            return Err(format!("D3DKMTCreateAllocation2 failed: 0x{:08X}", status as u32));
        }

        let handle = alloc_info.hAllocation;
        let mut map_gpu_va = reserved_gpu_va;
        let mut map_args = D3DDDI_MAPGPUVIRTUALADDRESS {
            hPagingQueue: self.dxg_paging_queue,
            BaseAddress: reserved_gpu_va,
            hAllocation: handle,
            SizeInPages: (aligned_size / PAGE_SIZE) as u64,
            Protection: Self::select_map_protection(flags),
            ..Default::default()
        };
        let status = unsafe { D3DKMTMapGpuVirtualAddress(&mut map_args) };
        if nt_failed(status) {
            self.destroy_allocation_handle(handle);
            if sys_mem.is_null() {
                if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
                    self.local_heap.free(reserved_gpu_va, aligned_size as u64);
                } else {
                    self.free_gpu_virtual_address(reserved_gpu_va, aligned_size);
                }
            } else {
                Self::decommit_system_heap_space(sys_mem, aligned_size);
                self.system_heap.free(reserved_gpu_va, aligned_size as u64);
            }
            return Err(format!("D3DKMTMapGpuVirtualAddress failed: 0x{:08X}", status as u32));
        }
        if map_args.PagingFenceValue != 0 {
            if let Err(err) = self.wait_on_paging_fence(map_args.PagingFenceValue) {
                self.destroy_allocation_handle(handle);
                if sys_mem.is_null() {
                    if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
                        self.local_heap.free(reserved_gpu_va, aligned_size as u64);
                    } else {
                        self.free_gpu_virtual_address(reserved_gpu_va, aligned_size);
                    }
                } else {
                    Self::decommit_system_heap_space(sys_mem, aligned_size);
                    self.system_heap.free(reserved_gpu_va, aligned_size as u64);
                }
                return Err(err);
            }
        }
        if map_args.VirtualAddress != 0 {
            map_gpu_va = map_args.VirtualAddress;
        }

        let handles = [handle];
        let mut make_resident = D3DDDI_MAKERESIDENT {
            hPagingQueue: self.dxg_paging_queue,
            NumAllocations: handles.len() as u32,
            AllocationList: handles.as_ptr(),
            Flags: D3DDDI_MAKERESIDENT_FLAGS {
                Value: D3DDDI_MAKERESIDENTFLAGS_CANT_TRIM_FURTHER,
            },
            ..Default::default()
        };
        let status = unsafe { D3DKMTMakeResident(&mut make_resident) };
        if nt_failed(status) {
            self.destroy_allocation_handle(handle);
            if sys_mem.is_null() {
                if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
                    self.local_heap.free(reserved_gpu_va, aligned_size as u64);
                } else {
                    self.free_gpu_virtual_address(map_gpu_va, aligned_size);
                }
            } else {
                Self::decommit_system_heap_space(sys_mem, aligned_size);
                self.system_heap.free(reserved_gpu_va, aligned_size as u64);
            }
            return Err(format!("D3DKMTMakeResident failed: 0x{:08X}", status as u32));
        }
        if make_resident.PagingFenceValue != 0 {
            if let Err(err) = self.wait_on_paging_fence(make_resident.PagingFenceValue) {
                self.destroy_allocation_handle(handle);
                if sys_mem.is_null() {
                    if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
                        self.local_heap.free(reserved_gpu_va, aligned_size as u64);
                    } else {
                        self.free_gpu_virtual_address(map_gpu_va, aligned_size);
                    }
                } else {
                    Self::decommit_system_heap_space(sys_mem, aligned_size);
                    self.system_heap.free(reserved_gpu_va, aligned_size as u64);
                }
                return Err(err);
            }
        }

        let (cpu_ptr, cpu_locked) = if !sys_mem.is_null() {
            (sys_mem as *mut u8, false)
        } else if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
            match self.lock_allocation(handle) {
                Ok(ptr) => (ptr as *mut u8, true),
                Err(err) => {
                    self.destroy_allocation_handle(handle);
                    self.local_heap.free(reserved_gpu_va, aligned_size as u64);
                    return Err(format!(
                        "Failed to CPU-map local/user-queue allocation {}: {}",
                        handle, err
                    ));
                }
            }
        } else {
            (ptr::null_mut(), false)
        };

        dxg_debug!(
            "[DXG] Allocated {} bytes: va=0x{:016X} handle={} domain={:?}",
            aligned_size,
            map_gpu_va,
            handle,
            match alloc_domain {
                T0DxgAllocDomain::System => "system",
                T0DxgAllocDomain::Local => "local",
                T0DxgAllocDomain::UserMemory => "user",
                T0DxgAllocDomain::UserQueue => "queue",
            }
        );

        Ok(WslGpuMemory {
            handle,
            gpu_va: map_gpu_va,
            cpu_ptr,
            size: aligned_size,
            device: Arc::clone(self),
            flags,
            auto_free: true,
            sys_mem,
            cpu_locked,
            alloc_domain,
        })
    }

    fn free_memory_internal(
        &self,
        handle: D3DKMT_HANDLE,
        gpu_va: u64,
        size: usize,
        sys_mem: *mut c_void,
        cpu_locked: bool,
        alloc_domain: T0DxgAllocDomain,
    ) {
        if cpu_locked {
            self.unlock_allocation(handle);
        }
        if !sys_mem.is_null() {
            Self::decommit_system_heap_space(sys_mem, size);
            self.system_heap.free(gpu_va, size as u64);
        } else if matches!(alloc_domain, T0DxgAllocDomain::Local | T0DxgAllocDomain::UserQueue) {
            self.local_heap.free(gpu_va, size as u64);
        } else {
            self.free_gpu_virtual_address(gpu_va, size);
        }
        self.destroy_allocation_handle(handle);
        dxg_debug!("[DXG] Freed allocation {} ({} bytes)", handle, size);
    }

    // ── Synchronization ─────────────────────────────────────────────────────

    pub fn create_sync_object(&self) -> Result<D3DKMT_HANDLE, String> {
        let mut info = D3DDDI_SYNCHRONIZATIONOBJECTINFO2 {
            Type: D3DDDI_SYNCHRONIZATIONOBJECT_TYPE::Fence,
            ..Default::default()
        };
        let mut create = D3DKMT_CREATESYNCHRONIZATIONOBJECT2 {
            hDevice: self.dxg_device,
            Info: info,
            hSyncObject: 0,
        };
        let status = unsafe { D3DKMTCreateSynchronizationObject2(&mut create) };
        if status != 0 {
            return Err(format!("D3DKMTCreateSynchronizationObject2 failed: 0x{:08X}", status as u32));
        }
        Ok(create.hSyncObject)
    }

    pub fn create_monitored_fence(&self) -> Result<(D3DKMT_HANDLE, *mut u64), String> {
        let mut info = D3DDDI_SYNCHRONIZATIONOBJECTINFO2 {
            Type: D3DDDI_SYNCHRONIZATIONOBJECT_TYPE::MonitoredFence,
            ..Default::default()
        };
        unsafe {
            info.info.MonitoredFence.InitialFenceValue = 0;
            info.info.MonitoredFence.EngineAffinity = 1;
        }

        let mut create = D3DKMT_CREATESYNCHRONIZATIONOBJECT2 {
            hDevice: self.dxg_device,
            Info: info,
            hSyncObject: 0,
        };
        let status = unsafe { D3DKMTCreateSynchronizationObject2(&mut create) };
        if nt_failed(status) {
            return Err(format!(
                "D3DKMTCreateSynchronizationObject2(MonitoredFence) failed: 0x{:08X}",
                status as u32
            ));
        }
        let sync_cpu_va = unsafe { create.Info.info.MonitoredFence.FenceValueCPUVirtualAddress as *mut u64 };
        let sync_gpu_va = unsafe { create.Info.info.MonitoredFence.FenceValueGPUVirtualAddress };
        dxg_debug!(
            "[DXG] create_monitored_fence: hSyncObject={} cpu_va={:?} gpu_va=0x{:016X} initial_cpu_value={}",
            create.hSyncObject,
            sync_cpu_va,
            sync_gpu_va,
            if sync_cpu_va.is_null() {
                u64::MAX
            } else {
                unsafe { ptr::read_volatile(sync_cpu_va) }
            }
        );
        Ok((
            create.hSyncObject,
            sync_cpu_va,
        ))
    }

    pub fn destroy_sync_object(&self, sync_object: D3DKMT_HANDLE) {
        if sync_object == 0 {
            return;
        }
        let args = D3DKMT_DESTROYSYNCHRONIZATIONOBJECT { hSyncObject: sync_object };
        let status = unsafe { D3DKMTDestroySynchronizationObject(&args) };
        if nt_failed(status) {
            eprintln!(
                "[DXG] WARN: D3DKMTDestroySynchronizationObject failed: 0x{:08X}",
                status as u32
            );
        }
    }

    pub fn wait_sync(&self, sync_object: D3DKMT_HANDLE, timeout_ns: u64) -> Result<(), String> {
        let _ = timeout_ns;
        let mut wait = D3DKMT_WAITFORSYNCHRONIZATIONOBJECT2 {
            hContext: self.dxg_context,
            ObjectCount: 1,
            ObjectHandleArray: [sync_object; 32], // Will use only first ObjectCount
            ..Default::default()
        };
        // Zero out the rest of the array
        unsafe {
            ptr::write_bytes(wait.ObjectHandleArray.as_mut_ptr().add(1), 0, 31);
        }
        let status = unsafe { D3DKMTWaitForSynchronizationObject2(&wait) };
        if status != 0 {
            return Err(format!("D3DKMTWaitForSynchronizationObject2 failed: 0x{:08X}", status as u32));
        }
        Ok(())
    }

    pub fn signal_sync(&self, sync_object: D3DKMT_HANDLE) -> Result<(), String> {
        let mut signal = D3DKMT_SIGNALSYNCHRONIZATIONOBJECT2 {
            hContext: self.dxg_context,
            ObjectCount: 1,
            ObjectHandleArray: [sync_object; 32],
            ..Default::default()
        };
        unsafe {
            ptr::write_bytes(signal.ObjectHandleArray.as_mut_ptr().add(1), 0, 31);
        }
        let status = unsafe { D3DKMTSignalSynchronizationObject2(&signal) };
        if status != 0 {
            return Err(format!("D3DKMTSignalSynchronizationObject2 failed: 0x{:08X}", status as u32));
        }
        Ok(())
    }

    // ── Queue Creation ──────────────────────────────────────────────────────

    pub fn create_queue(self: &Arc<Self>) -> Result<WslAqlQueue, String> {
        self.create_queue_sized(4 << 20) // 4MB default = 65536 packets
    }

    pub fn create_queue_sized(self: &Arc<Self>, ring_size: u32) -> Result<WslAqlQueue, String> {
        assert!(ring_size.is_power_of_two(), "ring_size must be power of 2, got {}", ring_size);

        let ring_buffer = self.alloc_system(ring_size as usize)?;

        unsafe {
            ptr::write_bytes(ring_buffer.cpu_ptr, 0, ring_buffer.size);
            let num_packets = ring_size as usize / 64;
            for i in 0..num_packets {
                let pkt = ring_buffer.cpu_ptr.add(i * 64) as *mut u16;
                ptr::write_volatile(pkt, HSA_PACKET_TYPE_INVALID);
            }
        }

        let cmd_buffer = self.alloc_system(DXG_HW_QUEUE_FRAME_SIZE * DXG_HW_QUEUE_FRAME_COUNT as usize)?;
        cmd_buffer.zero();

        let queue_context = match self.create_compute_context() {
            Ok(context) => context,
            Err(err) => return Err(err),
        };

        let force_sw_queue = std::env::var_os("T0_DXG_FORCE_SW_QUEUE").is_some();
        let force_hw_queue = dxg_force_hw_queue();
        let (use_hw_queue, hw_queue, hw_queue_progress_fence, hw_queue_progress_fence_cpu_va) =
            if (self.compute_hws_enabled || force_hw_queue) && !force_sw_queue {
                match self.create_hw_queue(queue_context) {
                    Ok((hw_queue, progress_fence, progress_cpu_va)) => {
                        (true, hw_queue, progress_fence, progress_cpu_va)
                    }
                    Err(err) => {
                        self.destroy_context(queue_context);
                        return Err(err);
                    }
                }
            } else {
                match self.create_monitored_fence() {
                    Ok((sync_object, sync_cpu_va)) => (false, 0, sync_object, sync_cpu_va),
                    Err(err) => {
                        self.destroy_context(queue_context);
                        return Err(err);
                    }
                }
            };
        dxg_debug!(
            "[DXG] Queue mode: use_hw_queue={} compute_hws_enabled={} force_hw_queue={} force_sw_queue={}",
            use_hw_queue,
            self.compute_hws_enabled,
            force_hw_queue,
            force_sw_queue
        );
        // Match librocdxg default behavior:
        // SW queue submit uses queue handle 0 unless UMD queue alloc is explicitly enabled.
        let sw_queue_mem = None;
        let sw_queue_handle = 0;
        let queue_id = if use_hw_queue {
            hw_queue as u64
        } else {
            0
        };
        let amd_queue_mem = match self.alloc_system(PAGE_SIZE) {
            Ok(mem) => mem,
            Err(err) => {
                if use_hw_queue {
                    self.destroy_hw_queue(hw_queue);
                } else {
                    self.destroy_sync_object(hw_queue_progress_fence);
                }
                self.destroy_context(queue_context);
                return Err(err);
            }
        };
        amd_queue_mem.zero();
        let write_ptr_host = unsafe {
            let amd_queue_ptr = amd_queue_mem.cpu_ptr as *mut AmdQueueV2;
            init_amd_queue_metadata(
                &mut *amd_queue_ptr,
                ring_buffer.cpu_ptr as *mut c_void,
                ring_size / 64,
                queue_id,
            );
            ptr::addr_of_mut!((*amd_queue_ptr).write_dispatch_id)
        };
        let read_ptr_host = unsafe {
            let amd_queue_ptr = amd_queue_mem.cpu_ptr as *mut AmdQueueV2;
            ptr::addr_of_mut!((*amd_queue_ptr).read_dispatch_id)
        };
        let read_ptr_gpu_va =
            amd_queue_mem.gpu_va + std::mem::offset_of!(AmdQueueV2, read_dispatch_id) as u64;
        let worker_state = Arc::new(WslQueueWorkerState::new());
        let worker_thread = {
            let worker_state = Arc::clone(&worker_state);
            let device = Arc::clone(self);
            let ring_buffer_ptr = ring_buffer.cpu_ptr as usize;
            let ring_buffer_gpu_va = ring_buffer.gpu_va;
            let write_ptr_host = write_ptr_host as usize;
            let read_ptr_host = read_ptr_host as usize;
            let cmd_buffer_cpu_ptr = cmd_buffer.cpu_ptr as usize;
            let cmd_buffer_gpu_va = cmd_buffer.gpu_va;
            let hw_queue_progress_fence_cpu_va = hw_queue_progress_fence_cpu_va as usize;
            let amd_queue_gpu_va = amd_queue_mem.gpu_va;
            let scratch_base_gpu_va = 0u64;
            let device_major = self.device_info.major();
            let sw_queue_handle = sw_queue_handle;
            Some(std::thread::spawn(move || {
                if let Err(err) = run_wsl_queue_worker(
                    Arc::clone(&worker_state),
                    device,
                    use_hw_queue,
                    ring_buffer_ptr as *mut u8,
                    ring_buffer_gpu_va,
                    ring_size,
                    write_ptr_host as *mut u64,
                    read_ptr_host as *mut u64,
                    read_ptr_gpu_va,
                    scratch_base_gpu_va,
                    queue_context,
                    hw_queue,
                    sw_queue_handle,
                    hw_queue_progress_fence,
                    hw_queue_progress_fence_cpu_va as *mut u64,
                    cmd_buffer_cpu_ptr as *mut u8,
                    cmd_buffer_gpu_va,
                    amd_queue_gpu_va,
                    device_major,
                ) {
                    worker_state.store_error(err);
                }
            }))
        };
        let wait_idle_signal = self.alloc_signal()?;

        Ok(WslAqlQueue {
            queue_id: if use_hw_queue { hw_queue } else { hw_queue_progress_fence },
            ring_buffer,
            ring_size,
            write_ptr_host,
            read_ptr_host,
            doorbell_ptr: ptr::null_mut(),
            doorbell_mmap_base: ptr::null_mut(),
            doorbell_mmap_size: 0,
            use_hw_queue,
            queue_context,
            hw_queue,
            hw_queue_progress_fence,
            hw_queue_progress_fence_cpu_va,
            worker_state,
            worker_thread,
            wait_idle_signal,
            _sw_queue_mem: sw_queue_mem,
            _amd_queue_mem: amd_queue_mem,
            _cmd_buffer: cmd_buffer,
            _scratch_mem: None,
            device: Arc::clone(self),
        })
    }

    // ── Convenience allocators ──────────────────────────────────────────────

    pub fn alloc_vram(self: &Arc<Self>, size: usize) -> Result<WslGpuMemory, String> {
        self.alloc_memory(size, MemoryFlags {
            vram: true,
            writable: true,
            public: true,
            coherent: true,
            fine_grain: true,
            ..Default::default()
        })
    }

    pub fn alloc_code(self: &Arc<Self>, size: usize) -> Result<WslGpuMemory, String> {
        // Current gfx1201 + /dev/dxg path does not provide a valid LOCAL Lock2
        // CPU mapping on this setup (D3DKMTLock2 -> STATUS_INVALID_PARAMETER).
        // Keep code objects in host-visible system memory as an explicit policy.
        self.alloc_memory(size, MemoryFlags {
            vram: true,
            writable: true,
            executable: true,
            public: true,
            ..Default::default()
        })
    }

    pub(crate) fn alloc_system(self: &Arc<Self>, size: usize) -> Result<WslGpuMemory, String> {
        self.alloc_memory_in_domain(size, MemoryFlags {
            writable: true,
            public: true,
            ..Default::default()
        }, T0DxgAllocDomain::System)
    }

    pub fn alloc_gart(self: &Arc<Self>, size: usize) -> Result<WslGpuMemory, String> {
        self.alloc_memory(size, MemoryFlags {
            gart: true,
            writable: true,
            public: true,
            coherent: true,
            ..Default::default()
        })
    }

    pub fn alloc_uncached(self: &Arc<Self>, size: usize) -> Result<WslGpuMemory, String> {
        self.alloc_memory(size, MemoryFlags {
            gart: true, writable: true, executable: true,
            public: true, coherent: false, uncached: true, ..Default::default()
        })
    }

    pub fn alloc_kernargs(self: &Arc<Self>, size: usize) -> Result<WslGpuMemory, String> {
        self.alloc_memory(size, MemoryFlags {
            gart: true,
            writable: true,
            public: true,
            coherent: true,
            kernarg: true,
            ..Default::default()
        })
    }

    pub fn alloc_signal(self: &Arc<Self>) -> Result<WslGpuMemory, String> {
        self.alloc_memory(64, MemoryFlags {
            gart: true,
            writable: true,
            public: true,
            coherent: true,
            fine_grain: true,
            ..Default::default()
        })
    }
}

impl Drop for WslDxgDevice {
    fn drop(&mut self) {
        Self::release_va_heap_space(self.dxg_adapter, &self.system_heap, "system heap");
        Self::release_va_heap_space(self.dxg_adapter, &self.local_heap, "local heap");
        unsafe {
            WslDxgDevice::close_handles(
                self.dxg_adapter,
                self.dxg_device,
                self.dxg_context,
                self.dxg_paging_queue,
            );
        }
    }
}

// =============================================================================
// Memory Flags
// =============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryFlags {
    pub vram: bool,
    pub gart: bool,
    pub writable: bool,
    pub executable: bool,
    pub public: bool,
    pub coherent: bool,
    pub uncached: bool,
    pub fine_grain: bool,
    pub kernarg: bool,
}

// =============================================================================
// WslGpuMemory — RAII GPU buffer
// =============================================================================

pub struct WslGpuMemory {
    pub handle: D3DKMT_HANDLE,
    pub gpu_va: u64,
    pub cpu_ptr: *mut u8,
    pub size: usize,
    device: Arc<WslDxgDevice>,
    flags: MemoryFlags,
    auto_free: bool,
    sys_mem: *mut c_void, // Host allocation for GART (must be freed with allocation)
    cpu_locked: bool,
    alloc_domain: T0DxgAllocDomain,
}

unsafe impl Send for WslGpuMemory {}
unsafe impl Sync for WslGpuMemory {}

impl WslGpuMemory {
    pub fn alloc_vram(device: &Arc<WslDxgDevice>, size: usize) -> Result<Self, String> {
        device.alloc_vram(size)
    }

    pub fn alloc_code(device: &Arc<WslDxgDevice>, size: usize) -> Result<Self, String> {
        device.alloc_code(size)
    }

    pub fn alloc_gart(device: &Arc<WslDxgDevice>, size: usize) -> Result<Self, String> {
        device.alloc_gart(size)
    }

    pub fn alloc_uncached(device: &Arc<WslDxgDevice>, size: usize) -> Result<Self, String> {
        device.alloc_uncached(size)
    }

    pub fn gpu_addr(&self) -> u64 { self.gpu_va }

    pub fn host_ptr(&self) -> *mut u8 { self.cpu_ptr }

    pub fn write(&self, data: &[u8]) {
        assert!(!self.cpu_ptr.is_null(), "buffer is not CPU-mapped");
        assert!(data.len() <= self.size, "write overflow: {} > {}", data.len(), self.size);
        unsafe {
            let dst = self.cpu_ptr;
            let src = data.as_ptr();
            let n8 = data.len() / 8;
            let rem = data.len() % 8;
            for i in 0..n8 {
                let val = ptr::read_unaligned(src.add(i * 8) as *const u64);
                ptr::write_volatile(dst.add(i * 8) as *mut u64, val);
            }
            let base = n8 * 8;
            for i in 0..rem {
                ptr::write_volatile(dst.add(base + i), *src.add(base + i));
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn write_val<T>(&self, offset: usize, val: T) {
        assert!(!self.cpu_ptr.is_null(), "buffer is not CPU-mapped");
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe {
            ptr::write_volatile(self.cpu_ptr.add(offset) as *mut T, val);
        }
    }

    pub fn read(&self, buf: &mut [u8]) {
        assert!(!self.cpu_ptr.is_null(), "buffer is not CPU-mapped");
        assert!(buf.len() <= self.size);
        unsafe { ptr::copy_nonoverlapping(self.cpu_ptr, buf.as_mut_ptr(), buf.len()) };
    }

    pub fn read_val<T>(&self, offset: usize) -> T {
        assert!(!self.cpu_ptr.is_null(), "buffer is not CPU-mapped");
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe { ptr::read_volatile(self.cpu_ptr.add(offset) as *const T) }
    }

    pub fn zero(&self) {
        assert!(!self.cpu_ptr.is_null(), "buffer is not CPU-mapped");
        unsafe { ptr::write_bytes(self.cpu_ptr, 0, self.size) };
    }

    pub fn read_bytes(&self, offset: usize, len: usize) -> Vec<u8> {
        assert!(!self.cpu_ptr.is_null(), "buffer is not CPU-mapped");
        assert!(offset + len <= self.size);
        let mut buf = vec![0u8; len];
        unsafe { ptr::copy_nonoverlapping(self.cpu_ptr.add(offset), buf.as_mut_ptr(), len) };
        buf
    }
}

impl Drop for WslGpuMemory {
    fn drop(&mut self) {
        if self.auto_free {
            self.device
                .free_memory_internal(
                    self.handle,
                    self.gpu_va,
                    self.size,
                    self.sys_mem,
                    self.cpu_locked,
                    self.alloc_domain,
                );
        }
    }
}

// =============================================================================
// AQL Queue
// =============================================================================

struct WslQueueWorkerState {
    inner: std::sync::Mutex<WslQueueWorkerInner>,
    cv: std::sync::Condvar,
}

#[derive(Default)]
struct WslQueueWorkerInner {
    stop: bool,
    error: Option<String>,
}

impl WslQueueWorkerState {
    fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(WslQueueWorkerInner::default()),
            cv: std::sync::Condvar::new(),
        }
    }

    fn notify(&self) {
        self.cv.notify_one();
    }

    fn request_stop(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.stop = true;
        self.cv.notify_all();
    }

    fn store_error(&self, err: String) {
        let mut inner = self.inner.lock().unwrap();
        if inner.error.is_none() {
            inner.error = Some(err);
        }
        self.cv.notify_all();
    }

    fn error(&self) -> Option<String> {
        self.inner.lock().unwrap().error.clone()
    }
}

struct WslPm4CmdBuilder {
    cmds: Vec<u32>,
}

impl WslPm4CmdBuilder {
    fn new() -> Self {
        Self { cmds: Vec::with_capacity(64) }
    }

    fn pkt3(&mut self, opcode: u32, body: &[u32]) {
        let header =
            (3u32 << 30)
            | (((body.len() as u32 - 1) & 0x3FFF) << 16)
            | (opcode << 8)
            | PM4_COMPUTE_SHADER_TYPE;
        self.cmds.push(header);
        self.cmds.extend_from_slice(body);
    }

    fn set_sh_reg(&mut self, reg_addr: u32, values: &[u32]) {
        let reg_offset = reg_addr - SH_REG_BASE;
        let mut body = Vec::with_capacity(1 + values.len());
        body.push(reg_offset);
        body.extend_from_slice(values);
        self.pkt3(PM4_SET_SH_REG, &body);
    }

    fn acquire_mem_gfx10(&mut self) {
        let gcr_cntl: u32 =
            (1 << 16) |
            (1 << 15) |
            (1 << 14) |
            (1 << 9) |
            (1 << 8) |
            (1 << 7) |
            (1 << 6) |
            (1 << 5) |
            (1 << 4) |
            (1 << 0);
        self.pkt3(PM4_ACQUIRE_MEM, &[
            0,
            0xFFFF_FFFF,
            0xFF,
            0,
            0,
            0,
            gcr_cntl,
        ]);
    }

    fn event_write(&mut self, event_type: u32, event_index: u32) {
        self.pkt3(PM4_EVENT_WRITE, &[event_type | (event_index << 8)]);
    }

    fn compute_barrier(&mut self) {
        self.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH);
    }

    fn dispatch_direct(&mut self, grid: [u32; 3]) {
        self.dispatch_direct_with_initiator(grid, DISPATCH_INITIATOR_COMPUTE_SHADER_EN);
    }

    fn dispatch_direct_with_initiator(&mut self, grid: [u32; 3], initiator: u32) {
        self.pkt3(PM4_DISPATCH_DIRECT, &[grid[0], grid[1], grid[2], initiator]);
    }

    fn copy_gpu_clock_count(&mut self, addr: u64) {
        assert_eq!(
            addr & 0x7,
            0,
            "COPY_DATA timestamp destination must be 8-byte aligned: 0x{addr:016X}"
        );
        let control_dw =
            9 |
            (2 << 8) |
            (1 << 16) |
            (1 << 20) |
            (1 << 25);
        // NOTE:
        // PM4 COPY_DATA destination address here is emitted as the raw ordinal field.
        // Using the bitfield-style pre-shift (>>3) produced GPU page faults on gfx1201
        // because hardware consumed the ordinal value as a byte address in this path.
        let addr_lo = addr as u32;
        let addr_hi = (addr >> 32) as u32;
        self.pkt3(PM4_COPY_DATA, &[control_dw, 0, 0, addr_lo, addr_hi]);
    }

    fn write_data_64(&mut self, addr: u64, value: u64) {
        let control_dw =
            (5 << 8)  // dst_sel = memory
            | (1 << 20) // wr_confirm = wait for write confirmation
            | (3 << 25); // cache_policy = bypass
        let addr_lo = (addr as u32) >> 2;
        let addr_hi = (addr >> 32) as u32;
        self.pkt3(PM4_WRITE_DATA, &[
            control_dw,
            addr_lo,
            addr_hi,
            value as u32,
            (value >> 32) as u32,
        ]);
    }

    fn atomic_add_64(&mut self, addr: u64, value: u64, cache_policy: u32) {
        let control_dw = TC_OP_ATOMIC_ADD_RTN_64 | (cache_policy << 25);
        self.pkt3(PM4_ATOMIC_MEM, &[
            control_dw,
            addr as u32,
            (addr >> 32) as u32,
            value as u32,
            (value >> 32) as u32,
            0,
            0,
            0,
        ]);
    }

    fn release_mem_64(&mut self, addr: u64, value: u64, cache_flush: bool) {
        let cache_flags = if cache_flush {
            (1 << 12) |
            (1 << 13) |
            (1 << 14) |
            (1 << 15) |
            (1 << 16) |
            (1 << 17) |
            (1 << 18)
        } else {
            0
        };
        let event_dw = CACHE_FLUSH_AND_INV_TS_EVENT | (5 << 8) | cache_flags;
        let data_dw = 2 << 29;
        self.pkt3(PM4_RELEASE_MEM, &[
            event_dw,
            data_dw,
            addr as u32,
            (addr >> 32) as u32,
            value as u32,
            (value >> 32) as u32,
            0,
        ]);
    }

    fn finish(self) -> Vec<u32> {
        self.cmds
    }
}

fn wsl_queue_packet_base(ring_buffer: *mut u8, ring_size: u32, read_idx: u64) -> *mut u8 {
    let ring_mask = (ring_size as u64 / 64) - 1;
    let slot_idx = read_idx & ring_mask;
    unsafe { ring_buffer.add((slot_idx * 64) as usize) }
}

fn wsl_queue_packet_type(base: *mut u8) -> u16 {
    unsafe { ptr::read_volatile(base as *const u16) & 0xff }
}

fn build_dispatch_pm4(
    packet: &AqlDispatchPacket,
    read_ptr_gpu_va: u64,
    completed_read_idx: u64,
    scratch_base_gpu_va: u64,
    queue_va: u64,
    packet_gpu_va: u64,
    device_major: u32,
    platform_atomic_support: bool,
) -> Result<Vec<u32>, String> {
    if packet.kernel_object == 0 {
        return Err("AQL dispatch packet has null kernel_object".to_string());
    }
    if packet.kernarg_address == 0 {
        return Err("AQL dispatch packet has null kernarg_address".to_string());
    }
    let kd_ptr = packet.kernel_object as *const u8;
    let kd_group_segment_size =
        unsafe { ptr::read_volatile(kd_ptr.add(KD_GROUP_SEGMENT_FIXED_SIZE_OFFSET) as *const u32) };
    let kd_private_segment_size =
        unsafe { ptr::read_volatile(kd_ptr.add(KD_PRIVATE_SEGMENT_FIXED_SIZE_OFFSET) as *const u32) };
    let entry_offset =
        unsafe { ptr::read_volatile(kd_ptr.add(KD_KERNEL_CODE_ENTRY_BYTE_OFFSET) as *const i64) };
    let rsrc1 =
        unsafe { ptr::read_volatile(kd_ptr.add(KD_COMPUTE_PGM_RSRC1_OFFSET) as *const u32) };
    let rsrc2_base =
        unsafe { ptr::read_volatile(kd_ptr.add(KD_COMPUTE_PGM_RSRC2_OFFSET) as *const u32) };
    let rsrc3_base =
        unsafe { ptr::read_volatile(kd_ptr.add(KD_COMPUTE_PGM_RSRC3_OFFSET) as *const u32) };
    let kernel_code_properties =
        unsafe { ptr::read_volatile(kd_ptr.add(KD_KERNEL_CODE_PROPERTIES_OFFSET) as *const u16) };
    let enable_profile_timestamps = (packet.reserved2 & AQL_RESERVED2_PROFILE_TS) != 0;

    if packet.group_segment_size < kd_group_segment_size {
        return Err(format!(
            "dispatch group_segment_size={} is smaller than kernel descriptor requirement {}",
            packet.group_segment_size,
            kd_group_segment_size,
        ));
    }
    if packet.private_segment_size < kd_private_segment_size {
        return Err(format!(
            "dispatch private_segment_size={} is smaller than kernel descriptor requirement {}",
            packet.private_segment_size,
            kd_private_segment_size,
        ));
    }
    if packet.private_segment_size != 0 {
        return Err(format!(
            "private_segment_size={} requires scratch queue state, which is not wired into the WSL DXG path yet",
            packet.private_segment_size
        ));
    }

    let code_entry_va = (packet.kernel_object as i64)
        .checked_add(entry_offset)
        .ok_or_else(|| "kernel_code_entry_byte_offset overflow".to_string())? as u64;
    let wave32 =
        (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32) != 0;
    let dynamic_lds_blocks = lds_blocks(packet.group_segment_size);
    let wgp_mode = ((rsrc1 >> 29) & 1) != 0;
    if wgp_mode && device_major <= 11 && !dxg_allow_wgp_legacy() {
        return Err(
            "WGP mode dispatch is blocked on legacy gfx11 DXG path by default (known hang risk); \
             set T0_DXG_ALLOW_WGP_LEGACY=1 to force-enable"
                .to_string(),
        );
    }
    let max_lds_blocks = if wgp_mode { 256 } else { 128 };
    if dynamic_lds_blocks > max_lds_blocks {
        return Err(format!(
            "dispatch LDS allocation exceeds {} mode hardware limit: {} blocks > {}",
            if wgp_mode { "WGP" } else { "CU" },
            dynamic_lds_blocks,
            max_lds_blocks,
        ));
    }
    let rsrc2 = rsrc2_base | (dynamic_lds_blocks << 15);
    let rsrc3 = if device_major >= 11 {
        // Match librocdxg CmdUtil::BuildComputeShaderParams (gfx11+):
        // program RSRC3 as IMAGE_OP-only in PM4 dispatch stream.
        COMPUTE_PGM_RSRC3_IMAGE_OP
    } else {
        rsrc3_base
    };

    let mut compute_user_data = [0u32; 16];
    let mut user_data_count = 0usize;
    let mut push_user_data = |value: u32| -> Result<(), String> {
        if user_data_count >= compute_user_data.len() {
            return Err("kernel dispatch requires more than 16 user SGPR dwords".to_string());
        }
        compute_user_data[user_data_count] = value;
        user_data_count += 1;
        Ok(())
    };

    if (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0
    {
        push_user_data(0)?;
        push_user_data(0)?;
        push_user_data(0)?;
        push_user_data(0)?;
    }
    if (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR) != 0 {
        push_user_data(low_part(packet_gpu_va))?;
        push_user_data(high_part(packet_gpu_va))?;
    }
    if (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR) != 0 {
        push_user_data(low_part(queue_va))?;
        push_user_data(high_part(queue_va))?;
    }
    if (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR) != 0 {
        push_user_data(low_part(packet.kernarg_address))?;
        push_user_data(high_part(packet.kernarg_address))?;
    }
    if (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID) != 0 {
        push_user_data(0)?;
        push_user_data(0)?;
    }
    if (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT) != 0 {
        push_user_data(0)?;
        push_user_data(0)?;
    }
    if (kernel_code_properties & AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) != 0 {
        push_user_data(0)?;
    }

    let mut pm4 = WslPm4CmdBuilder::new();
    let use_atomic_progress = platform_atomic_support;

    if packet.completion_signal != 0 && enable_profile_timestamps {
        pm4.copy_gpu_clock_count(packet.completion_signal + AMD_SIGNAL_START_TS_OFFSET as u64);
    }

    if packet.header & HSA_PACKET_HEADER_BARRIER_BIT != 0 {
        pm4.compute_barrier();
    }

    pm4.acquire_mem_gfx10();
    if device_major >= 11 {
        pm4.set_sh_reg(
            REG_COMPUTE_DISPATCH_SCRATCH_BASE_LO,
            &[ptr48_low32(scratch_base_gpu_va), ptr48_high8(scratch_base_gpu_va)],
        );
        pm4.set_sh_reg(REG_COMPUTE_PGM_RSRC3, &[rsrc3]);
    }
    pm4.set_sh_reg(REG_COMPUTE_NUM_THREAD_X, &[
        packet.workgroup_size_x as u32,
        packet.workgroup_size_y as u32,
        packet.workgroup_size_z as u32,
    ]);
    pm4.set_sh_reg(REG_COMPUTE_PGM_LO, &[
        ptr48_low32(code_entry_va),
        ptr48_high8(code_entry_va),
    ]);
    pm4.set_sh_reg(REG_COMPUTE_PGM_RSRC1, &[rsrc1, rsrc2]);
    // mmCOMPUTE_RESOURCE_LIMITS..mmCOMPUTE_STATIC_THREAD_MGMT_SE3 are programmed as
    // one contiguous SET_SH_REG burst. Register order must match HW:
    //   RESOURCE_LIMITS, SE0, SE1, TMPRING_SIZE, SE2, SE3.
    pm4.set_sh_reg(REG_COMPUTE_RESOURCE_LIMITS, &[
        COMPUTE_RESOURCE_LIMITS_DEFAULT,
        COMPUTE_STATIC_THREAD_MGMT_ENABLE_ALL,
        COMPUTE_STATIC_THREAD_MGMT_ENABLE_ALL,
        0, // COMPUTE_TMPRING_SIZE
        COMPUTE_STATIC_THREAD_MGMT_ENABLE_ALL,
        COMPUTE_STATIC_THREAD_MGMT_ENABLE_ALL,
    ]);
    if user_data_count != 0 {
        pm4.set_sh_reg(REG_COMPUTE_USER_DATA_0, &compute_user_data[..user_data_count]);
    }
    let mut dispatch_initiator =
        DISPATCH_INITIATOR_COMPUTE_SHADER_EN
        | DISPATCH_INITIATOR_FORCE_START_AT_000
        | DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS;
    if wave32 {
        dispatch_initiator |= DISPATCH_INITIATOR_CS_W32_EN;
    }
    pm4.dispatch_direct_with_initiator(
        [packet.grid_size_x, packet.grid_size_y, packet.grid_size_z],
        dispatch_initiator,
    );

    if packet.completion_signal != 0 {
        pm4.compute_barrier();
        if enable_profile_timestamps {
            pm4.copy_gpu_clock_count(packet.completion_signal + AMD_SIGNAL_END_TS_OFFSET as u64);
        }
        pm4.acquire_mem_gfx10();
        if use_atomic_progress {
            pm4.atomic_add_64(
                packet.completion_signal + AMD_SIGNAL_VALUE_OFFSET as u64,
                u64::MAX,
                MEC_ATOMIC_MEM_CACHE_POLICY_BYPASS,
            );
        }
    }
    if use_atomic_progress {
        pm4.atomic_add_64(read_ptr_gpu_va, 1, MEC_ATOMIC_MEM_CACHE_POLICY_STREAM);
    } else {
        emit_non_atomic_progress_store(&mut pm4, device_major, read_ptr_gpu_va, completed_read_idx);
    }

    Ok(pm4.finish())
}

fn emit_non_atomic_progress_store(
    pm4: &mut WslPm4CmdBuilder,
    _device_major: u32,
    addr: u64,
    value: u64,
) {
    pm4.write_data_64(addr, value);
}

fn build_barrier_pm4(
    completion_signal: u64,
    read_ptr_gpu_va: u64,
    completed_read_idx: u64,
    device_major: u32,
    platform_atomic_support: bool,
) -> Vec<u32> {
    let mut pm4 = WslPm4CmdBuilder::new();
    let use_atomic_progress = platform_atomic_support;
    if completion_signal != 0 {
        pm4.compute_barrier();
        if use_atomic_progress {
            pm4.atomic_add_64(
                completion_signal + AMD_SIGNAL_VALUE_OFFSET as u64,
                u64::MAX,
                MEC_ATOMIC_MEM_CACHE_POLICY_BYPASS,
            );
        }
    }
    if use_atomic_progress {
        pm4.atomic_add_64(read_ptr_gpu_va, 1, MEC_ATOMIC_MEM_CACHE_POLICY_STREAM);
    } else {
        emit_non_atomic_progress_store(&mut pm4, device_major, read_ptr_gpu_va, completed_read_idx);
    }
    pm4.finish()
}

fn build_vendor_specific_pm4(
    packet: &AqlVendorSpecificPm4Packet,
    read_ptr_gpu_va: u64,
    completed_read_idx: u64,
    device_major: u32,
    platform_atomic_support: bool,
) -> Result<Vec<u32>, String> {
    let ib_jump_header = packet.ib_jump_cmd[0];
    let pkt_type = ib_jump_header >> 30;
    let opcode = (ib_jump_header >> 8) & 0xff;
    if packet.ven_hdr != AMD_AQL_FORMAT_PM4_IB {
        return Err(format!(
            "Unsupported vendor-specific packet format {}",
            packet.ven_hdr
        ));
    }
    if pkt_type != 3 || opcode != PACKET3_INDIRECT_BUFFER {
        return Err(format!(
            "Unsupported vendor-specific IB header: type={} opcode=0x{:02X}",
            pkt_type,
            opcode
        ));
    }
    if packet.ib_jump_cmd[3] & INDIRECT_BUFFER_VALID == 0 {
        return Err("Vendor-specific PM4 IB packet is missing INDIRECT_BUFFER_VALID".to_string());
    }

    let ib_addr =
        ((packet.ib_jump_cmd[2] as u64) << 32) | ((packet.ib_jump_cmd[1] as u64) & !0x3);
    let ib_dwords = (packet.ib_jump_cmd[3] & 0x000f_ffff) as usize;
    if ib_addr == 0 {
        return Err("Vendor-specific PM4 IB packet has null IB address".to_string());
    }
    if ib_dwords == 0 {
        return Err("Vendor-specific PM4 IB packet has zero-length IB".to_string());
    }

    let mut pm4 = Vec::with_capacity(ib_dwords + 16);
    if packet.header & HSA_PACKET_HEADER_BARRIER_BIT != 0 {
        let mut preamble = WslPm4CmdBuilder::new();
        preamble.compute_barrier();
        pm4.extend(preamble.finish());
    }

    let ib_ptr = ib_addr as *const u32;
    for idx in 0..ib_dwords {
        pm4.push(unsafe { ptr::read_volatile(ib_ptr.add(idx)) });
    }

    let mut tail = WslPm4CmdBuilder::new();
    let use_atomic_progress = platform_atomic_support;
    if packet.completion_signal != 0 {
        tail.compute_barrier();
        if use_atomic_progress {
            tail.atomic_add_64(
                packet.completion_signal + AMD_SIGNAL_VALUE_OFFSET as u64,
                u64::MAX,
                MEC_ATOMIC_MEM_CACHE_POLICY_BYPASS,
            );
        }
    }
    if use_atomic_progress {
        tail.atomic_add_64(read_ptr_gpu_va, 1, MEC_ATOMIC_MEM_CACHE_POLICY_STREAM);
    } else {
        emit_non_atomic_progress_store(&mut tail, device_major, read_ptr_gpu_va, completed_read_idx);
    }
    pm4.extend(tail.finish());

    Ok(pm4)
}

fn wait_barrier_dependencies(dep_signals: &[u64; 5], is_or: bool) -> Result<(), String> {
    if is_or && dep_signals.iter().all(|signal| *signal == 0) {
        return Err("AQL barrier-or packet has no non-null dependency signals".to_string());
    }

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let mut any_satisfied = false;
        let mut all_satisfied = true;

        for &signal in dep_signals {
            if signal == 0 {
                continue;
            }
            let signal_value = unsafe { ptr::read_volatile((signal + 8) as *const i64) };
            if signal_value == 0 {
                any_satisfied = true;
            } else {
                all_satisfied = false;
            }
        }

        if (!is_or && all_satisfied) || (is_or && any_satisfied) {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "AQL barrier-{} packet timed out waiting for dependency signals",
                if is_or { "or" } else { "and" }
            ));
        }
        std::thread::sleep(std::time::Duration::from_micros(10));
    }
}

fn run_wsl_queue_worker(
    worker_state: Arc<WslQueueWorkerState>,
    device: Arc<WslDxgDevice>,
    use_hw_queue: bool,
    ring_buffer: *mut u8,
    ring_buffer_gpu_va: u64,
    ring_size: u32,
    write_ptr_host: *mut u64,
    read_ptr_host: *mut u64,
    read_ptr_gpu_va: u64,
    scratch_base_gpu_va: u64,
    queue_context: D3DKMT_HANDLE,
    hw_queue: D3DKMT_HANDLE,
    submit_queue_handle: D3DKMT_HANDLE,
    hw_queue_progress_fence: D3DKMT_HANDLE,
    hw_queue_progress_fence_cpu_va: *mut u64,
    cmd_buffer_cpu_ptr: *mut u8,
    cmd_buffer_gpu_va: u64,
    amd_queue_gpu_va: u64,
    device_major: u32,
) -> Result<(), String> {
    if hw_queue_progress_fence_cpu_va.is_null() {
        return Err("HW queue progress fence CPU VA is null".to_string());
    }

    let mut submit_cursor = unsafe { ptr::read_volatile(read_ptr_host) };
    let platform_atomic_support = device.platform_atomic_support();

    loop {
        {
            let mut inner = worker_state.inner.lock().unwrap();
            while !inner.stop {
                let write_idx = unsafe { ptr::read_volatile(write_ptr_host) };
                let has_work = if submit_cursor < write_idx {
                    let base = wsl_queue_packet_base(ring_buffer, ring_size, submit_cursor);
                    wsl_queue_packet_type(base) != HSA_PACKET_TYPE_INVALID
                } else {
                    false
                };
                if has_work {
                    break;
                }
                inner = worker_state.cv.wait(inner).unwrap();
            }

            if inner.stop {
                return Ok(());
            }
        }

        let write_idx = unsafe { ptr::read_volatile(write_ptr_host) };
        if submit_cursor >= write_idx {
            continue;
        }

        let packet_base = wsl_queue_packet_base(ring_buffer, ring_size, submit_cursor);
        let packet_type = wsl_queue_packet_type(packet_base);
        if packet_type == HSA_PACKET_TYPE_INVALID {
            continue;
        }

        let completed_read_idx = submit_cursor + 1;
        let ring_mask = (ring_size as u64 / 64) - 1;
        let slot_idx = submit_cursor & ring_mask;
        let packet_gpu_va = ring_buffer_gpu_va + slot_idx * 64;
        let (pm4_cmds, completion_signal) = match packet_type {
            HSA_PACKET_TYPE_KERNEL_DISPATCH => {
                let packet = unsafe { ptr::read_unaligned(packet_base as *const AqlDispatchPacket) };
                (
                    build_dispatch_pm4(
                        &packet,
                        read_ptr_gpu_va,
                        completed_read_idx,
                        scratch_base_gpu_va,
                        amd_queue_gpu_va,
                        packet_gpu_va,
                        device_major,
                        platform_atomic_support,
                    )?,
                    packet.completion_signal,
                )
            }
            HSA_PACKET_TYPE_BARRIER_AND => {
                let packet = unsafe { ptr::read_unaligned(packet_base as *const AqlBarrierPacket) };
                wait_barrier_dependencies(&packet.dep_signal, false)?;
                (
                    build_barrier_pm4(
                        packet.completion_signal,
                        read_ptr_gpu_va,
                        completed_read_idx,
                        device_major,
                        platform_atomic_support,
                    ),
                    packet.completion_signal,
                )
            }
            HSA_PACKET_TYPE_BARRIER_OR => {
                let packet = unsafe { ptr::read_unaligned(packet_base as *const AqlBarrierPacket) };
                wait_barrier_dependencies(&packet.dep_signal, true)?;
                (
                    build_barrier_pm4(
                        packet.completion_signal,
                        read_ptr_gpu_va,
                        completed_read_idx,
                        device_major,
                        platform_atomic_support,
                    ),
                    packet.completion_signal,
                )
            }
            HSA_PACKET_TYPE_VENDOR_SPECIFIC => {
                let packet =
                    unsafe { ptr::read_unaligned(packet_base as *const AqlVendorSpecificPm4Packet) };
                (
                    build_vendor_specific_pm4(
                        &packet,
                        read_ptr_gpu_va,
                        completed_read_idx,
                        device_major,
                        platform_atomic_support,
                    )?,
                    packet.completion_signal,
                )
            }
            HSA_PACKET_TYPE_AGENT_DISPATCH => {
                return Err("AQL agent-dispatch packets are not implemented on WSL DXG".to_string());
            }
            other => {
                return Err(format!("Unsupported AQL packet type {}", other));
            }
        };

        let pm4_len_bytes = pm4_cmds.len() * std::mem::size_of::<u32>();
        if pm4_len_bytes > DXG_HW_QUEUE_FRAME_SIZE {
            return Err(format!(
                "PM4 frame overflow: {} bytes > frame size {}",
                pm4_len_bytes,
                DXG_HW_QUEUE_FRAME_SIZE
            ));
        }

        if dxg_debug_enabled() {
            let head: Vec<String> = pm4_cmds
                .iter()
                .take(12)
                .map(|dw| format!("{:08X}", dw))
                .collect();
            dxg_debug!(
                "[DXG] worker submit_cursor={} packet_type={} frame_dwords={} target={} progress_fence={} pm4_head=[{}]",
                submit_cursor,
                packet_type,
                pm4_cmds.len(),
                completed_read_idx,
                unsafe { ptr::read_volatile(hw_queue_progress_fence_cpu_va) },
                head.join(" ")
            );
            if submit_cursor == 0 {
                let dump: Vec<String> = pm4_cmds
                    .iter()
                    .take(64)
                    .enumerate()
                    .map(|(i, dw)| format!("{:02}:{:08X}", i, dw))
                    .collect();
                dxg_debug!(
                    "[DXG] first_dispatch_pm4_dwords={} [{}]",
                    pm4_cmds.len(),
                    dump.join(" ")
                );
            }
        }

        if completed_read_idx > DXG_HW_QUEUE_FRAME_COUNT {
            let min_completed = completed_read_idx - DXG_HW_QUEUE_FRAME_COUNT + 1;
            let wait_start = std::time::Instant::now();
            loop {
                let completed = unsafe { ptr::read_volatile(read_ptr_host) };
                if completed >= min_completed {
                    break;
                }
                if use_hw_queue {
                    device.wait_for_sync_object_value(hw_queue_progress_fence, min_completed)?;
                }
                if wait_start.elapsed() > std::time::Duration::from_secs(5) {
                    return Err(format!(
                        "Timed out waiting for reusable command frame: read_ptr={} target={} progress_fence={} {}",
                        completed,
                        min_completed,
                        unsafe { ptr::read_volatile(hw_queue_progress_fence_cpu_va) },
                        device.describe_device_state(),
                    ));
                }
                std::hint::spin_loop();
            }
        }

        let frame_slot = ((completed_read_idx - 1) % DXG_HW_QUEUE_FRAME_COUNT) as usize;
        let frame_offset = frame_slot * DXG_HW_QUEUE_FRAME_SIZE;
        unsafe {
            let dst = cmd_buffer_cpu_ptr.add(frame_offset) as *mut u32;
            for (idx, dword) in pm4_cmds.iter().enumerate() {
                ptr::write_volatile(dst.add(idx), *dword);
            }
            for idx in pm4_cmds.len()..(DXG_HW_QUEUE_FRAME_SIZE / std::mem::size_of::<u32>()) {
                ptr::write_volatile(dst.add(idx), 0);
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        if use_hw_queue {
            device.submit_command_to_hw_queue_on_context(
                queue_context,
                hw_queue,
                cmd_buffer_gpu_va + frame_offset as u64,
                pm4_len_bytes as u32,
                completed_read_idx,
            )?;
        } else {
            device.submit_command_on_context(
                queue_context,
                submit_queue_handle,
                cmd_buffer_gpu_va + frame_offset as u64,
                pm4_len_bytes as u32,
                hw_queue_progress_fence,
                completed_read_idx,
            )?;
        }

        if completion_signal != 0 && !platform_atomic_support {
            let wait_start = std::time::Instant::now();
            loop {
                let completed = unsafe { ptr::read_volatile(read_ptr_host) };
                if completed >= completed_read_idx {
                    break;
                }
                if use_hw_queue {
                    device.wait_for_sync_object_value(hw_queue_progress_fence, completed_read_idx)?;
                }
                if wait_start.elapsed() > std::time::Duration::from_secs(5) {
                    return Err(format!(
                        "Timed out waiting for completion signal retirement: read_ptr={} target={} progress_fence={} {}",
                        completed,
                        completed_read_idx,
                        unsafe { ptr::read_volatile(hw_queue_progress_fence_cpu_va) },
                        device.describe_device_state(),
                    ));
                }
                std::hint::spin_loop();
            }
            unsafe {
                let signal_ptr = (completion_signal + AMD_SIGNAL_VALUE_OFFSET as u64)
                    as *const std::sync::atomic::AtomicI64;
                (&*signal_ptr).fetch_sub(1, std::sync::atomic::Ordering::Release);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }

        dxg_debug!(
            "[DXG] worker submitted frame_slot={} target={} progress_fence_now={}",
            frame_slot,
            completed_read_idx,
            unsafe { ptr::read_volatile(hw_queue_progress_fence_cpu_va) }
        );

        unsafe {
            ptr::write_volatile(packet_base as *mut u16, HSA_PACKET_TYPE_INVALID);
        }
        submit_cursor = completed_read_idx;
    }
}

pub struct WslAqlQueue {
    pub queue_id: u32,
    pub ring_buffer: WslGpuMemory,
    pub ring_size: u32,
    pub write_ptr_host: *mut u64,
    pub read_ptr_host: *mut u64,
    pub doorbell_ptr: *mut u64,
    doorbell_mmap_base: *mut c_void,
    doorbell_mmap_size: usize,
    use_hw_queue: bool,
    queue_context: D3DKMT_HANDLE,
    hw_queue: D3DKMT_HANDLE,
    hw_queue_progress_fence: D3DKMT_HANDLE,
    hw_queue_progress_fence_cpu_va: *mut u64,
    worker_state: Arc<WslQueueWorkerState>,
    worker_thread: Option<std::thread::JoinHandle<()>>,
    wait_idle_signal: WslGpuMemory,
    _sw_queue_mem: Option<WslGpuMemory>,
    _amd_queue_mem: WslGpuMemory,
    _cmd_buffer: WslGpuMemory,
    _scratch_mem: Option<WslGpuMemory>,
    device: Arc<WslDxgDevice>,
}

unsafe impl Send for WslAqlQueue {}
unsafe impl Sync for WslAqlQueue {}

impl WslAqlQueue {
    fn notify_worker(&self) {
        self.worker_state.notify();
    }

    fn worker_error(&self) -> Option<String> {
        self.worker_state.error()
    }

    fn init_signal(&self, signal: &WslGpuMemory) {
        verify_amd_signal_layout_once();
        unsafe { ptr::write_bytes(signal.cpu_ptr, 0, AMD_SIGNAL_SIZE_BYTES) };
        signal.write_val::<u64>(AMD_SIGNAL_KIND_OFFSET, AMD_SIGNAL_KIND_USER);
        signal.write_val::<i64>(AMD_SIGNAL_VALUE_OFFSET, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }

    fn ensure_ring_space(&self) -> Result<(), String> {
        let max_inflight = (self.ring_size as u64 / 64) - MAX_INFLIGHT;
        loop {
            if let Some(err) = self.worker_error() {
                return Err(err);
            }
            let write_idx = unsafe { ptr::read_volatile(self.write_ptr_host) };
            let read_idx = unsafe { ptr::read_volatile(self.read_ptr_host) };
            if write_idx - read_idx < max_inflight {
                return Ok(());
            }
            std::hint::spin_loop();
        }
    }

    fn enqueue_dispatch_packet(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernarg_va: u64,
        signal_va: u64,
        profile_timestamps: bool,
        acquire_scope: u16,
        release_scope: u16,
    ) -> Result<u64, String> {
        self.ensure_ring_space()?;
        let write_idx = unsafe { ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        let header =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            HSA_PACKET_HEADER_BARRIER_BIT |
            (acquire_scope << 9) |
            (release_scope << 11);

        unsafe {
            let base = self.ring_buffer.cpu_ptr.add(pkt_offset);
            ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
            ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
            ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            ptr::write_volatile(base.add(0x18) as *mut u32, kernel.private_segment_size);
            ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size);
            ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va);
            ptr::write_volatile(base.add(0x28) as *mut u64, kernarg_va);
            let reserved2 = if profile_timestamps { AQL_RESERVED2_PROFILE_TS } else { 0 };
            ptr::write_volatile(base.add(0x30) as *mut u64, reserved2);
            ptr::write_volatile(base.add(0x38) as *mut u64, signal_va);

            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            ptr::write_volatile(base as *mut u16, header);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            ptr::write_volatile(self.write_ptr_host, new_write_idx);
        }

        dxg_debug!(
            "[DXG] enqueue write_idx={} slot={} grid=({},{},{}) wg=({},{},{}) kernarg=0x{:X} signal=0x{:X} desc=0x{:X} lds={} private={}",
            write_idx,
            slot_idx,
            grid[0],
            grid[1],
            grid[2],
            kernel.workgroup_size[0],
            kernel.workgroup_size[1],
            kernel.workgroup_size[2],
            kernarg_va,
            signal_va,
            kernel.descriptor_va,
            kernel.lds_size,
            kernel.private_segment_size
        );

        self.notify_worker();
        Ok(write_idx + 1)
    }

    fn enqueue_barrier_packet(
        &self,
        signal_va: u64,
        acquire_scope: u16,
        release_scope: u16,
    ) -> Result<u64, String> {
        self.ensure_ring_space()?;
        let write_idx = unsafe { ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        let header =
            (HSA_PACKET_TYPE_BARRIER_AND as u16) |
            (acquire_scope << 9) |
            (release_scope << 11);

        unsafe {
            let base = self.ring_buffer.cpu_ptr.add(pkt_offset);
            ptr::write_volatile(base.add(0x02) as *mut u16, 0u16);
            ptr::write_volatile(base.add(0x04) as *mut u32, 0u32);
            // dep_signal[0..5] = 0
            for i in 0..5usize {
                ptr::write_volatile(base.add(0x08 + i * 8) as *mut u64, 0u64);
            }
            ptr::write_volatile(base.add(0x30) as *mut u64, 0u64); // reserved2
            ptr::write_volatile(base.add(0x38) as *mut u64, signal_va);

            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            ptr::write_volatile(base as *mut u16, header);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            ptr::write_volatile(self.write_ptr_host, new_write_idx);
        }

        self.notify_worker();
        Ok(write_idx + 1)
    }

    /// Dispatch a kernel. Returns after GPU completes execution.
    pub fn dispatch(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &WslGpuMemory,
    ) -> Result<(), String> {
        self.dispatch_signal(kernel, grid, kernargs, None)
    }

    /// Dispatch with explicit signal buffer for completion tracking.
    pub fn dispatch_signal(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &WslGpuMemory,
        signal: Option<&WslGpuMemory>,
    ) -> Result<(), String> {
        self.dispatch_signal_internal(kernel, grid, kernargs, signal, false)
    }

    fn dispatch_signal_internal(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &WslGpuMemory,
        signal: Option<&WslGpuMemory>,
        profile_timestamps: bool,
    ) -> Result<(), String> {
        assert!(kernargs.size >= kernel.kernarg_size as usize,
            "kernarg too small: buffer={}B, kernel expects {}B",
            kernargs.size, kernel.kernarg_size);

        // Prepare completion signal (amd_signal_t layout)
        let signal_va = if let Some(sig) = signal {
            self.init_signal(sig);
            sig.gpu_va
        } else {
            0
        };

        if dxg_debug_enabled() {
            let mut words = [0u64; 8];
            if !kernargs.cpu_ptr.is_null() && kernargs.size >= 64 {
                unsafe {
                    for (i, w) in words.iter_mut().enumerate() {
                        *w = ptr::read_volatile(kernargs.cpu_ptr.add(i * 8) as *const u64);
                    }
                }
                dxg_debug!(
                    "[DXG] kernargs head: [{:016X} {:016X} {:016X} {:016X} {:016X} {:016X} {:016X} {:016X}]",
                    words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]
                );
            } else {
                dxg_debug!(
                    "[DXG] kernargs head unavailable: cpu_ptr={:?} size={}",
                    kernargs.cpu_ptr,
                    kernargs.size
                );
            }
        }

        let target = self.enqueue_dispatch_packet(
            kernel,
            grid,
            kernargs.gpu_va,
            signal_va,
            profile_timestamps,
            HSA_FENCE_SCOPE_SYSTEM,
            HSA_FENCE_SCOPE_SYSTEM,
        )?;

        // Wait for completion
        if let Some(sig) = signal {
            self.wait_signal(sig, target)?;
        } else {
            self.wait_read_ptr(target)?;
        }

        Ok(())
    }

    pub fn dispatch_signal_profiled(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &WslGpuMemory,
        signal: &WslGpuMemory,
    ) -> Result<(u64, u64), String> {
        self.dispatch_signal_internal(kernel, grid, kernargs, Some(signal), true)?;
        let timeout_ns: u64 = 10_000_000_000;
        let start = std::time::Instant::now();
        loop {
            if let Some(err) = self.worker_error() {
                return Err(err);
            }
            let start_ts: u64 = signal.read_val(AMD_SIGNAL_START_TS_OFFSET);
            let end_ts: u64 = signal.read_val(AMD_SIGNAL_END_TS_OFFSET);
            if start_ts != 0 && end_ts > start_ts {
                return Ok((start_ts, end_ts));
            }
            if start.elapsed().as_nanos() as u64 > timeout_ns {
                let signal_value: i64 = signal.read_val(AMD_SIGNAL_VALUE_OFFSET);
                return Err(format!(
                    "DXG profiling timestamp wait timeout: signal={} start_ts={} end_ts={}",
                    signal_value,
                    start_ts,
                    end_ts,
                ));
            }
            std::hint::spin_loop();
        }
    }

    /// Submit without waiting — pipelined dispatch.
    pub fn submit(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &WslGpuMemory,
    ) {
        self.enqueue_dispatch_packet(
            kernel,
            grid,
            kernargs.gpu_va,
            0,
            false,
            HSA_FENCE_SCOPE_SYSTEM,
            HSA_FENCE_SCOPE_SYSTEM,
        ).expect("failed to enqueue DXG dispatch packet");
    }

    /// Wait for all pending dispatches.
    pub fn wait_idle(&self) -> Result<(), String> {
        if let Some(err) = self.worker_error() {
            return Err(err);
        }
        // Precise queue drain: enqueue a barrier packet and wait its completion signal.
        // This avoids relying on read_ptr pseudo-completion under DXG.
        self.init_signal(&self.wait_idle_signal);
        let target = self.enqueue_barrier_packet(
            self.wait_idle_signal.gpu_va,
            HSA_FENCE_SCOPE_SYSTEM,
            HSA_FENCE_SCOPE_SYSTEM,
        )?;
        self.wait_signal(&self.wait_idle_signal, target)
    }

    /// Wait for all pending dispatches + memory fence.
    /// This is the SAFE way to synchronize before dropping GPU buffers.
    pub fn synchronize(&self) -> Result<(), String> {
        self.wait_idle()?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_micros(10));
        Ok(())
    }

    /// Submit without waiting — pipelined dispatch with AGENT fences.
    /// Uses GPU-internal-only fences (no PCIe sync) for lower latency.
    pub fn submit_fast(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &WslGpuMemory,
    ) {
        self.enqueue_dispatch_packet(
            kernel,
            grid,
            kernargs.gpu_va,
            0,
            false,
            HSA_FENCE_SCOPE_AGENT,
            HSA_FENCE_SCOPE_AGENT,
        ).expect("failed to enqueue DXG fast dispatch packet");
    }

    /// Ultra-low-latency wait: tight spin on read_dispatch_id.
    /// No timeout — hangs forever if GPU is stuck.
    #[inline]
    pub fn wait_idle_spin(&self) {
        let target = unsafe { ptr::read_volatile(self.write_ptr_host) };
        loop {
            if let Some(err) = self.worker_error() {
                panic!("DXG queue worker failed: {}", err);
            }
            let read_idx = unsafe { ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target {
                return;
            }
            std::hint::spin_loop();
        }
    }

    fn wait_read_ptr(&self, target: u64) -> Result<(), String> {
        let timeout_ns: u64 = 5_000_000_000;
        let start = std::time::Instant::now();
        loop {
            if let Some(err) = self.worker_error() {
                return Err(err);
            }
            let read_idx = unsafe { ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target {
                dxg_debug!(
                    "[DXG] wait_read_ptr complete: read_idx={} target={} progress_fence={}",
                    read_idx,
                    target,
                    unsafe { ptr::read_volatile(self.hw_queue_progress_fence_cpu_va) }
                );
                return Ok(());
            }
            if start.elapsed().as_nanos() as u64 > timeout_ns {
                let device_state = self.device.describe_device_state();
                return Err(format!(
                    "Queue wait timeout (read_idx={} target={} progress_fence={} {})",
                    read_idx,
                    target,
                    unsafe { ptr::read_volatile(self.hw_queue_progress_fence_cpu_va) },
                    device_state,
                ));
            }
            std::hint::spin_loop();
        }
    }

    fn wait_signal(&self, signal: &WslGpuMemory, target: u64) -> Result<(), String> {
        let timeout_ns: u64 = 10_000_000_000;
        let start = std::time::Instant::now();
        loop {
            if let Some(err) = self.worker_error() {
                return Err(err);
            }
            let val: i64 = signal.read_val(8);
            let read_idx = unsafe { ptr::read_volatile(self.read_ptr_host) };
            if val == 0 && read_idx >= target {
                return Ok(());
            }
            if start.elapsed().as_nanos() as u64 > timeout_ns {
                let device_state = self.device.describe_device_state();
                return Err(format!(
                    "Signal wait timeout: signal={} read_idx={} target={} progress_fence={} {}",
                    val,
                    read_idx,
                    target,
                    unsafe { ptr::read_volatile(self.hw_queue_progress_fence_cpu_va) },
                    device_state
                ));
            }
            std::hint::spin_loop();
        }
    }
}

impl Drop for WslAqlQueue {
    fn drop(&mut self) {
        // Drain queue
        let target = unsafe { ptr::read_volatile(self.write_ptr_host) };
        let timeout = std::time::Duration::from_millis(500);
        let start = std::time::Instant::now();
        loop {
            let read_idx = unsafe { ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target { break; }
            if start.elapsed() > timeout {
                eprintln!("[DXG] WARN: Queue drain timeout");
                break;
            }
            std::hint::spin_loop();
        }

        self.worker_state.request_stop();
        if let Some(worker_thread) = self.worker_thread.take() {
            let _ = worker_thread.join();
        }

        if self.use_hw_queue {
            self.device.destroy_hw_queue(self.hw_queue);
        } else {
            self.device.destroy_sync_object(self.hw_queue_progress_fence);
        }
        self.device.destroy_context(self.queue_context);
    }
}

// =============================================================================
// AQL Dispatch Packet (64 bytes, hardware format)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AqlDispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: u64,
    pub reserved2: u64,
    pub completion_signal: u64,
}

const _: () = assert!(std::mem::size_of::<AqlDispatchPacket>() == 64);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AqlBarrierPacket {
    pub header: u16,
    pub reserved0: u16,
    pub reserved1: u32,
    pub dep_signal: [u64; 5],
    pub reserved2: u64,
    pub completion_signal: u64,
}

const _: () = assert!(std::mem::size_of::<AqlBarrierPacket>() == 64);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AqlVendorSpecificPm4Packet {
    pub header: u16,
    pub ven_hdr: u16,
    pub ib_jump_cmd: [u32; 4],
    pub dw_cnt_remain: u32,
    pub reserved: [u32; 8],
    pub completion_signal: u64,
}

const _: () = assert!(std::mem::size_of::<AqlVendorSpecificPm4Packet>() == 64);

// =============================================================================
// GpuKernel — loaded GPU kernel (DXG version)
// =============================================================================

pub struct GpuKernel {
    pub code_buffer: WslGpuMemory,
    pub descriptor_va: u64,
    pub code_entry_va: u64,
    pub rsrc3: u32,
    pub rsrc1: u32,
    pub rsrc2: u32,
    pub kernel_code_properties: u16,
    pub private_segment_size: u32,
    pub lds_size: u32,
    pub workgroup_size: [u32; 3],
    pub kernarg_size: u32,
}

/// Minimal ELF parser for HSACO code objects
struct ElfParser {
    text_offset: usize,
    text_size: usize,
    loads: Vec<LoadSegment>,
    min_vaddr: u64,
    total_memsz: usize,
    kd_offset_in_load: usize,
}

struct LoadSegment {
    offset: usize,
    vaddr: u64,
    filesz: usize,
    memsz: usize,
}

impl ElfParser {
    fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 64 || &data[0..4] != b"\x7fELF" {
            return Err("Not a valid ELF file".to_string());
        }

        let e_shoff = u64::from_le_bytes(data[40..48].try_into().unwrap()) as usize;
        let e_phoff = u64::from_le_bytes(data[32..40].try_into().unwrap()) as usize;
        let e_phentsize = u16::from_le_bytes(data[54..56].try_into().unwrap()) as usize;
        let e_phnum = u16::from_le_bytes(data[56..58].try_into().unwrap()) as usize;
        let e_shentsize = u16::from_le_bytes(data[58..60].try_into().unwrap()) as usize;
        let e_shnum = u16::from_le_bytes(data[60..62].try_into().unwrap()) as usize;
        let e_shstrndx = u16::from_le_bytes(data[62..64].try_into().unwrap()) as usize;

        let mut loads = Vec::new();
        for i in 0..e_phnum {
            let ph = e_phoff + i * e_phentsize;
            let p_type = u32::from_le_bytes(data[ph..ph + 4].try_into().unwrap());
            if p_type == 1 { // PT_LOAD
                loads.push(LoadSegment {
                    offset: u64::from_le_bytes(data[ph + 8..ph + 16].try_into().unwrap()) as usize,
                    vaddr: u64::from_le_bytes(data[ph + 16..ph + 24].try_into().unwrap()),
                    filesz: u64::from_le_bytes(data[ph + 32..ph + 40].try_into().unwrap()) as usize,
                    memsz: u64::from_le_bytes(data[ph + 40..ph + 48].try_into().unwrap()) as usize,
                });
            }
        }
        if loads.is_empty() {
            return Err("No PT_LOAD segments found".to_string());
        }

        let min_vaddr = loads.iter().map(|l| l.vaddr).min().unwrap();
        let max_vaddr_end = loads.iter().map(|l| l.vaddr + l.memsz as u64).max().unwrap();
        let total_memsz = (max_vaddr_end - min_vaddr) as usize;

        let shstr_hdr = e_shoff + e_shstrndx * e_shentsize;
        let shstr_off = u64::from_le_bytes(data[shstr_hdr + 24..shstr_hdr + 32].try_into().unwrap()) as usize;

        let mut text_offset = 0usize;
        let mut text_size = 0usize;
        let mut symtab_off = 0usize;
        let mut symtab_size = 0usize;
        let mut symtab_entsize = 0usize;
        let mut strtab_off = 0usize;

        for i in 0..e_shnum {
            let sh = e_shoff + i * e_shentsize;
            let sh_name_idx = u32::from_le_bytes(data[sh..sh + 4].try_into().unwrap()) as usize;
            let sh_type = u32::from_le_bytes(data[sh + 4..sh + 8].try_into().unwrap());
            let sh_off = u64::from_le_bytes(data[sh + 24..sh + 32].try_into().unwrap()) as usize;
            let sh_size = u64::from_le_bytes(data[sh + 32..sh + 40].try_into().unwrap()) as usize;

            let name_start = shstr_off + sh_name_idx;
            let name_end = data[name_start..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| name_start + p)
                .unwrap_or(name_start);
            let name = std::str::from_utf8(&data[name_start..name_end]).unwrap_or("");

            if name == ".text" {
                text_offset = sh_off;
                text_size = sh_size;
            } else if sh_type == 2 || (sh_type == 11 && symtab_entsize == 0) {
                symtab_off = sh_off;
                symtab_size = sh_size;
                symtab_entsize =
                    u64::from_le_bytes(data[sh + 56..sh + 64].try_into().unwrap()) as usize;
                let sh_link = u32::from_le_bytes(data[sh + 40..sh + 44].try_into().unwrap()) as usize;
                let strtab_sh = e_shoff + sh_link * e_shentsize;
                strtab_off =
                    u64::from_le_bytes(data[strtab_sh + 24..strtab_sh + 32].try_into().unwrap()) as usize;
            }
        }

        if text_size == 0 {
            return Err("No .text section found in HSACO".to_string());
        }

        let mut kd_vaddr = 0u64;
        if symtab_entsize > 0 {
            let num_syms = symtab_size / symtab_entsize;
            for i in 0..num_syms {
                let sym = symtab_off + i * symtab_entsize;
                let st_name = u32::from_le_bytes(data[sym..sym + 4].try_into().unwrap()) as usize;
                let st_value = u64::from_le_bytes(data[sym + 8..sym + 16].try_into().unwrap());

                let name_start = strtab_off + st_name;
                let name_end = data[name_start..]
                    .iter()
                    .position(|&b| b == 0)
                    .map(|p| name_start + p)
                    .unwrap_or(name_start);
                let name = std::str::from_utf8(&data[name_start..name_end]).unwrap_or("");

                if name.ends_with(".kd") {
                    kd_vaddr = st_value;
                    break;
                }
            }
        }

        if kd_vaddr < min_vaddr {
            return Err("Could not find kernel descriptor (.kd) symbol in ELF".to_string());
        }

        let kd_offset_in_load = (kd_vaddr - min_vaddr) as usize;

        Ok(Self {
            text_offset,
            text_size,
            loads,
            min_vaddr,
            total_memsz,
            kd_offset_in_load,
        })
    }

    fn loadable_content(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; self.total_memsz];
        for seg in &self.loads {
            let dst_offset = (seg.vaddr - self.min_vaddr) as usize;
            let src_end = seg.offset + seg.filesz;
            if src_end > data.len() {
                return Err(format!(
                    "PT_LOAD segment exceeds file: offset={:#x} filesz={:#x} file_len={:#x}",
                    seg.offset,
                    seg.filesz,
                    data.len()
                ));
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

/// Configuration for kernel loading
pub struct KernelLoadConfig {
    pub lds_size: u32,
    pub workgroup_size: [u32; 3],
}

impl GpuKernel {
    /// Load a kernel from HSACO ELF bytes
    pub fn load(device: &Arc<WslDxgDevice>, hsaco: &[u8], config: &KernelLoadConfig) -> Result<Self, String> {
        let elf = ElfParser::parse(hsaco)?;
        let load_data = elf.loadable_content(hsaco)?;
        let kd_offset = elf.kernel_descriptor_offset()?;

        let code_buf = device.alloc_code(load_data.len())?;
        code_buf.write(&load_data);

        // Read back to flush WC buffers
        let _ = unsafe { std::ptr::read_volatile(code_buf.cpu_ptr) };

        // Patch kernel descriptor: set RSRC1.PRIV (bit 20) only on gfx11.
        // Align with librocdxg CmdUtil::BuildDispatch behavior.
        let (rsrc1, rsrc2, rsrc3, entry_offset, kd_kernarg_size, kernel_code_properties, private_segment_size, kd_group_segment_size);
        unsafe {
            let kd_ptr = code_buf.cpu_ptr.add(kd_offset);
            // Debug dump
            if dxg_debug_enabled() {
                eprintln!("[DXG] KD at offset {} (0x{:X}) in code buffer:", kd_offset, kd_offset);
                for row in 0..4 {
                    let off = row * 16;
                    eprint!("  {:02X}:", off);
                    for i in 0..16 {
                        eprint!(" {:02X}", *kd_ptr.add(off + i));
                    }
                    eprintln!();
                }
            }

            let rsrc1_ptr = kd_ptr.add(KD_COMPUTE_PGM_RSRC1_OFFSET) as *mut u32;
            let raw_rsrc1 = ptr::read_volatile(rsrc1_ptr);
            let patched_rsrc1 = if device.device_info.major() == 11 {
                raw_rsrc1 | (1 << 20) // PRIV bit required on current DXG compute path
            } else {
                raw_rsrc1
            };

            let wgp_on = (patched_rsrc1 >> 29) & 1 == 1;
            dxg_debug!("[DXG] RSRC1=0x{:08X} WGP_MODE(bit29)={}", patched_rsrc1, wgp_on);

            ptr::write_volatile(rsrc1_ptr, patched_rsrc1);
            rsrc1 = patched_rsrc1;
            rsrc2 = ptr::read_volatile(kd_ptr.add(KD_COMPUTE_PGM_RSRC2_OFFSET) as *const u32);
            rsrc3 = ptr::read_volatile(kd_ptr.add(KD_COMPUTE_PGM_RSRC3_OFFSET) as *const u32);
            entry_offset = ptr::read_volatile(
                kd_ptr.add(KD_KERNEL_CODE_ENTRY_BYTE_OFFSET) as *const i64
            );
            kd_kernarg_size =
                ptr::read_volatile(kd_ptr.add(KD_KERNARG_SIZE_OFFSET) as *const u32);
            kernel_code_properties = ptr::read_volatile(
                kd_ptr.add(KD_KERNEL_CODE_PROPERTIES_OFFSET) as *const u16
            );
            kd_group_segment_size = ptr::read_volatile(
                kd_ptr.add(KD_GROUP_SEGMENT_FIXED_SIZE_OFFSET) as *const u32
            );
            private_segment_size = ptr::read_volatile(
                kd_ptr.add(KD_PRIVATE_SEGMENT_FIXED_SIZE_OFFSET) as *const u32
            );
        }

        let _ = unsafe { std::ptr::read_volatile(code_buf.cpu_ptr) }; // Flush HDP

        let wgp_on = ((rsrc1 >> 29) & 1) != 0;
        let effective_lds_size = config.lds_size.max(kd_group_segment_size);
        let max_lds_blocks = if wgp_on { 256 } else { 128 };
        let effective_lds_blocks = lds_blocks(effective_lds_size);
        if effective_lds_blocks > max_lds_blocks {
            return Err(format!(
                "kernel requires {}B LDS ({} blocks), exceeding {} mode limit of {} blocks; config={}B descriptor={}B",
                effective_lds_size,
                effective_lds_blocks,
                if wgp_on { "WGP" } else { "CU" },
                max_lds_blocks,
                config.lds_size,
                kd_group_segment_size,
            ));
        }

        let descriptor_va = code_buf.gpu_va + kd_offset as u64;
        let code_entry_va = (descriptor_va as i64 + entry_offset) as u64;

        dxg_debug!("[DXG] Kernel loaded: desc_va=0x{:X} code_va=0x{:X} rsrc1=0x{:08X}",
            descriptor_va, code_entry_va, rsrc1);

        Ok(GpuKernel {
            code_buffer: code_buf,
            descriptor_va,
            code_entry_va,
            rsrc3,
            rsrc1,
            rsrc2,
            kernel_code_properties,
            private_segment_size,
            lds_size: effective_lds_size,
            workgroup_size: config.workgroup_size,
            kernarg_size: kd_kernarg_size,
        })
    }
}

// =============================================================================
// DispatchPool — pre-allocated kernargs + signal buffers
// =============================================================================

pub struct DispatchPool {
    pub signal: WslGpuMemory,
    kernargs_ring: std::sync::Mutex<Vec<WslGpuMemory>>,
    device: Arc<WslDxgDevice>,
}

impl DispatchPool {
    pub fn new(device: &Arc<WslDxgDevice>, initial_slots: usize) -> Result<Self, String> {
        let signal = device.alloc_signal()?;
        let n = if initial_slots == 0 { DEFAULT_DISPATCH_SLOTS } else { initial_slots };
        let mut ring = Vec::with_capacity(n);
        for _ in 0..n {
            ring.push(device.alloc_kernargs(256)?);
        }
        Ok(Self {
            signal,
            kernargs_ring: std::sync::Mutex::new(ring),
            device: Arc::clone(device),
        })
    }

    fn ensure_slot(&self, idx: usize) {
        let mut ring = self.kernargs_ring.lock().unwrap();
        while idx >= ring.len() {
            match self.device.alloc_kernargs(256) {
                Ok(buf) => ring.push(buf),
                Err(e) => panic!("DispatchPool: failed to grow to slot {}: {}", idx, e),
            }
        }
    }

    pub fn get_kernargs(&self, idx: usize) -> &WslGpuMemory {
        self.ensure_slot(idx);
        let ring = self.kernargs_ring.lock().unwrap();
        unsafe { &*(ring.get(idx).unwrap() as *const WslGpuMemory) }
    }

    pub fn write_kernargs(&self, idx: usize, data: &[u8]) -> &WslGpuMemory {
        self.ensure_slot(idx);
        let ring = self.kernargs_ring.lock().unwrap();
        let buf = unsafe { &*(ring.get(idx).unwrap() as *const WslGpuMemory) };
        buf.write(data);
        buf
    }

    pub fn dispatch(
        &self,
        queue: &WslAqlQueue,
        kernel: &GpuKernel,
        grid: [u32; 3],
        ka_idx: usize,
    ) -> Result<(), String> {
        self.signal.write_val::<u64>(AMD_SIGNAL_KIND_OFFSET, AMD_SIGNAL_KIND_USER);
        self.signal.write_val::<i64>(AMD_SIGNAL_VALUE_OFFSET, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        let ka = self.get_kernargs(ka_idx);
        queue.dispatch_signal(kernel, grid, ka, Some(&self.signal))
    }

    pub fn len(&self) -> usize { self.kernargs_ring.lock().unwrap().len() }
    pub fn capacity(&self) -> usize { self.kernargs_ring.lock().unwrap().capacity() }
}
