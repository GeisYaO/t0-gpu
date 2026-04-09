use std::ffi::c_void;

use super::{
    D3DKMT_HANDLE, D3DKMT_QUERYADAPTERINFO, D3DKMTQueryAdapterInfo, KMTQUERYADAPTERINFOTYPE,
};

const RAW_ADAPTER_INFO_SIZE: usize = 0x45c0;
const ADAPTER_INFO_EX_SIZE: usize = 0x3a00;
const ADAPTER_INFO_EX_QUERY_SIZE: usize = 0x4360;
const ATI_ADAPTER_INFO_OFFSET: usize = 0x3a00;
const ATI_ADAPTER_INFO_SIZE: usize = 0x260;
const ATI_ADAPTER_INFO_QUERY_SIZE: usize = 0xbc0;
const ATI_ADAPTER_INFO_QUERY_OFFSET: usize = 0x960;
const PROXY_ADAPTER_INFO_OFFSET: usize = 0x3c60;
const PROXY_ADAPTER_INFO_SIZE: usize = 0x960;
const PROXY_ADAPTER_INFO_QUERY_SIZE: usize = 0xbc0;

const ENGINE_ORDINAL_TABLE_OFFSET: usize = 0x89c;
const ENGINE_ORDINAL_TABLE_LEN: usize = 32;
const LOCAL_VISIBLE_HEAP_SIZE_OFFSET: usize = 0xa30;
const LOCAL_INVISIBLE_HEAP_SIZE_OFFSET: usize = 0xa38;
const NON_LOCAL_HEAP_SIZE_OFFSET: usize = 0xa40;
const HWS_MASK_GATE_OFFSET: usize = 0x2e03;
const HWS_ORDINAL_MASK_SOURCE_OFFSET: usize = 0x35c4;
const DISABLE_GPU_TIMEOUT_MASK_OFFSET: usize = 0x35d8;
const ADAPTER_DEVICE_ID_OFFSET: usize = 0x3a2c;
const DGPU_FLAG_OFFSET: usize = 0x3c8c;
const DGPU_STATE_SHADOWING_OFFSET: usize = 0xc46;
const IGPU_STATE_SHADOWING_OFFSET: usize = 0x208;

const CONTEXT_PRIV_DATA_SIZE: usize = 0x40;
const ALLOC_PRIV_DRV_DATA_SIZE: usize = 0x40;
const ALLOC_PRIV_DATA_SIZE: usize = 0x218;

const DEFAULT_GFX_MAJOR: u32 = 12;
const COMPUTE_ENGINE: u32 = 5;
const SUPPORTED_HWS_ENGINES: [u32; 4] = [0, 5, 4, 7];

#[repr(C)]
#[derive(Default)]
struct D3DKMT_SEGMENTSIZEINFO {
    dedicated_video_memory_size: u64,
    dedicated_system_memory_size: u64,
    shared_system_memory_size: u64,
}

#[derive(Clone)]
pub(crate) struct DxgThunkDeviceInfo {
    raw: Vec<u8>,
    major: u32,
    is_dgpu: bool,
    local_visible_heap_size: u64,
    local_invisible_heap_size: u64,
    non_local_heap_size: u64,
    compute_schedid: u32,
    state_shadowing_by_cpfw: bool,
    hws_engine_ordinal_mask: u64,
    disable_gpu_timeout_ordinal_mask: u64,
}

impl DxgThunkDeviceInfo {
    pub(crate) fn new(adapter: D3DKMT_HANDLE) -> Result<Self, String> {
        let mut raw = vec![0u8; RAW_ADAPTER_INFO_SIZE];
        let adapter_info_ex = query_umd_private(adapter, ADAPTER_INFO_EX_QUERY_SIZE, "ADAPTERINFOEX")?;
        raw[..ADAPTER_INFO_EX_SIZE]
            .copy_from_slice(&adapter_info_ex[ADAPTER_INFO_EX_QUERY_SIZE - ADAPTER_INFO_EX_SIZE..]);

        let ati_adapter_info = query_umd_private(adapter, ATI_ADAPTER_INFO_QUERY_SIZE, "ATIADAPTERINFO")?;
        raw[ATI_ADAPTER_INFO_OFFSET..ATI_ADAPTER_INFO_OFFSET + ATI_ADAPTER_INFO_SIZE]
            .copy_from_slice(
                &ati_adapter_info[ATI_ADAPTER_INFO_QUERY_OFFSET..ATI_ADAPTER_INFO_QUERY_OFFSET + ATI_ADAPTER_INFO_SIZE],
            );

        let proxy_adapter_info = query_umd_private(adapter, PROXY_ADAPTER_INFO_QUERY_SIZE, "PROXY_ADAPTER_INFO")?;
        raw[PROXY_ADAPTER_INFO_OFFSET..PROXY_ADAPTER_INFO_OFFSET + PROXY_ADAPTER_INFO_SIZE]
            .copy_from_slice(&proxy_adapter_info[..PROXY_ADAPTER_INFO_SIZE]);

        let device_id = read_u32(&raw, ADAPTER_DEVICE_ID_OFFSET);
        let is_dgpu = read_u32(&raw, DGPU_FLAG_OFFSET) == 0;
        let mut local_visible_heap_size = read_u64(&raw, LOCAL_VISIBLE_HEAP_SIZE_OFFSET);
        let local_invisible_heap_size = read_u64(&raw, LOCAL_INVISIBLE_HEAP_SIZE_OFFSET);
        let mut non_local_heap_size = read_u64(&raw, NON_LOCAL_HEAP_SIZE_OFFSET);

        if (local_visible_heap_size == 0 && local_invisible_heap_size == 0) || non_local_heap_size == 0 {
            if let Ok(segment_size) = query_segment_size(adapter) {
                if local_visible_heap_size == 0 && local_invisible_heap_size == 0 {
                    local_visible_heap_size = segment_size.dedicated_video_memory_size;
                }
                if non_local_heap_size == 0 {
                    non_local_heap_size = segment_size.shared_system_memory_size;
                }
            }
        }

        let compute_schedid = find_engine_ordinal(&raw, COMPUTE_ENGINE)
            .map(|_| COMPUTE_ENGINE)
            .ok_or_else(|| format!("No compute queue engine {} found in adapter info", COMPUTE_ENGINE))?;

        let state_shadowing_offset = if is_dgpu {
            DGPU_STATE_SHADOWING_OFFSET
        } else {
            IGPU_STATE_SHADOWING_OFFSET
        };
        let state_shadowing_by_cpfw = (read_u8(&raw, state_shadowing_offset) & 0x20) != 0;

        let mut hws_engine_ordinal_mask = 0u64;
        if (read_u8(&raw, HWS_MASK_GATE_OFFSET) & 0x01) != 0 {
            let raw_hws_mask = read_u64(&raw, HWS_ORDINAL_MASK_SOURCE_OFFSET);
            for engine in SUPPORTED_HWS_ENGINES {
                if let Some(ordinal) = find_engine_ordinal(&raw, engine) {
                    let bit = ordinal_bit(ordinal);
                    if raw_hws_mask & bit != 0 {
                        hws_engine_ordinal_mask |= bit;
                    }
                }
            }
        }

        let disable_gpu_timeout_ordinal_mask = read_u64(&raw, DISABLE_GPU_TIMEOUT_MASK_OFFSET);

        Ok(Self {
            raw,
            major: infer_gfx_major(device_id),
            is_dgpu,
            local_visible_heap_size,
            local_invisible_heap_size,
            non_local_heap_size,
            compute_schedid,
            state_shadowing_by_cpfw,
            hws_engine_ordinal_mask,
            disable_gpu_timeout_ordinal_mask,
        })
    }

    pub(crate) fn compute_engine(&self) -> u32 {
        self.compute_schedid
    }

    pub(crate) fn major(&self) -> u32 {
        self.major
    }

    pub(crate) fn is_dgpu(&self) -> bool {
        self.is_dgpu
    }

    pub(crate) fn state_shadowing_by_cpfw(&self) -> bool {
        self.state_shadowing_by_cpfw
    }

    pub(crate) fn engine_ordinal(&self, engine: u32) -> Result<u32, String> {
        find_engine_ordinal(&self.raw, engine)
            .ok_or_else(|| format!("EngineOrdinal failed for engine {}", engine))
    }

    pub(crate) fn hws_enabled(&self, engine: u32) -> bool {
        self.engine_ordinal(engine)
            .ok()
            .map(|ordinal| self.hws_engine_ordinal_mask & ordinal_bit(ordinal) != 0)
            .unwrap_or(false)
    }

    pub(crate) fn should_disable_gpu_timeout(&self, engine: u32) -> bool {
        self.engine_ordinal(engine)
            .ok()
            .map(|ordinal| self.disable_gpu_timeout_ordinal_mask & ordinal_bit(ordinal) != 0)
            .unwrap_or(false)
    }

    pub(crate) fn queue_engine_flag(&self, queue_engine: u32) -> Result<u32, String> {
        queue_engine_to_engine_flag(queue_engine)
            .ok_or_else(|| format!("Unsupported queue engine {}", queue_engine))
    }

    pub(crate) fn local_visible_heap_size(&self) -> u64 {
        self.local_visible_heap_size
    }

    pub(crate) fn local_invisible_heap_size(&self) -> u64 {
        self.local_invisible_heap_size
    }

    pub(crate) fn non_local_heap_size(&self) -> u64 {
        self.non_local_heap_size
    }
}

pub(crate) fn build_context_priv_data(device_info: &DxgThunkDeviceInfo) -> Result<Vec<u8>, String> {
    let mut priv_data = vec![0u8; CONTEXT_PRIV_DATA_SIZE];
    write_u64(&mut priv_data, 0x00, 0x0000_0100_0000_0040);
    let mut flags = read_u8(&priv_data, 0x3a) & !0x20;
    if device_info.state_shadowing_by_cpfw() {
        flags |= 0x20;
    }
    priv_data[0x3a] = flags;
    Ok(priv_data)
}

pub(crate) fn build_alloc_priv_drv_data() -> Vec<u8> {
    let mut priv_data = vec![0u8; ALLOC_PRIV_DRV_DATA_SIZE];
    priv_data[0x08] |= 0x80;
    write_u32(&mut priv_data, 0x3c, ALLOC_PRIV_DATA_SIZE as u32);
    priv_data
}

pub(crate) fn build_alloc_priv_data(
    size: u64,
    domain: u32,
    addr: u64,
    mem_flags: u32,
    engine_flag: u32,
    device_info: &DxgThunkDeviceInfo,
) -> Result<Vec<u8>, String> {
    let mut data = vec![0u8; ALLOC_PRIV_DATA_SIZE];
    write_u32(&mut data, 0x00, 0x1d8);
    write_u32(&mut data, 0x10, 0);
    write_u64(&mut data, 0x3c, 0x0000_01d8_0000_0001);
    write_u64(&mut data, 0x44, 0x0000_0080_0000_0001);
    write_u32(&mut data, 0x54, size as u32);
    write_u32(&mut data, 0x58, 0x1000);
    write_u64(&mut data, 0xa8, size);
    write_u32(&mut data, 0xf0, 0xc4);
    write_u32(&mut data, 0x104, 0x114);

    match domain {
        0 | 2 => {
            or_u32(&mut data, 0x50, 0x0400_0008);
            write_u16(&mut data, 0x60, 4);
            data[0x5c] = (data[0x5c] & !0x07) | 0x01;
            if device_info.major <= 11 && (mem_flags & 0x3) != 0 {
                write_u32(&mut data, 0x1d0, 4);
            }
        }
        1 => {
            if !device_info.is_dgpu {
                or_u32(&mut data, 0x50, 0x9);
                write_u16(&mut data, 0x60, 0x401);
                data[0x5c] = (data[0x5c] & !0x07) | 0x02;
            } else if device_info.local_invisible_heap_size == 0 {
                or_u32(&mut data, 0x50, 0x1);
                data[0x60] = 0x1;
                data[0x5c] = (data[0x5c] & !0x07) | 0x01;
            } else {
                or_u32(&mut data, 0x50, 0x3);
                write_u16(&mut data, 0x60, 0x102);
                data[0x5c] = (data[0x5c] & !0x07) | 0x02;
            }
        }
        3 => {
            let engine_id = engine_flag_to_queue_engine_id(engine_flag)
                .ok_or_else(|| format!("Unsupported engine flag {}", engine_flag))?;
            or_u32(&mut data, 0x50, 0x8);
            data[0xa6] |= 0x81;
            write_u64(&mut data, 0xb0, addr);
            data[0x60] = 4;
            data[0x5c] = (data[0x5c] & !0x07) | 0x01;
            data[0xf4] = (data[0xf4] & !0x7f) | engine_id;
        }
        other => {
            return Err(format!("Unsupported allocation domain {}", other));
        }
    }

    Ok(data)
}

#[derive(Clone, Copy)]
pub(crate) enum DxgSchedLevel {
    Low = 0,
    Normal = 1,
    High = 2,
}

pub(crate) fn build_submit_priv_data(
    queue: D3DKMT_HANDLE,
    command_addr: u64,
    command_size: u32,
    is_hw_queue: bool,
) -> Vec<u8> {
    let mut priv_data = vec![0u8; 0x48];
    write_u32(&mut priv_data, 0x00, 0x06);
    write_u32(&mut priv_data, 0x04, 0x01);
    write_u32(&mut priv_data, 0x0c, 0x40);
    write_u32(&mut priv_data, 0x10, command_size);
    write_u64(&mut priv_data, 0x20, command_addr);
    if !is_hw_queue {
        write_u32(&mut priv_data, 0x28, queue);
    }
    priv_data
}

pub(crate) fn build_hw_queue_priv_data(
    fw_managed_gfx_state: bool,
    level: DxgSchedLevel,
) -> Vec<u8> {
    let mut priv_data = vec![0u8; 0x40];
    write_u32(&mut priv_data, 0x00, 0x40);

    let prio_selector = match level {
        DxgSchedLevel::Low | DxgSchedLevel::Normal => 1u32,
        DxgSchedLevel::High => 2u32,
    };
    let fw_bit = if fw_managed_gfx_state { 1u32 << 17 } else { 0 };
    let queue_flags = ((prio_selector << 12) & 0x7000) | fw_bit;
    write_u32(&mut priv_data, 0x10, queue_flags & 0x2700f);

    priv_data
}

fn query_umd_private(adapter: D3DKMT_HANDLE, size: usize, name: &str) -> Result<Vec<u8>, String> {
    let mut buffer = vec![0u8; size];
    let query = D3DKMT_QUERYADAPTERINFO {
        hAdapter: adapter,
        Type: KMTQUERYADAPTERINFOTYPE::UmDriverPrivate,
        pPrivateDriverData: buffer.as_mut_ptr() as *mut c_void,
        PrivateDriverDataSize: buffer.len() as u32,
    };
    let status = unsafe { D3DKMTQueryAdapterInfo(&query) };
    if status != 0 {
        return Err(format!(
            "D3DKMTQueryAdapterInfo({}) failed: 0x{:08X}",
            name,
            status as u32
        ));
    }
    Ok(buffer)
}

fn query_segment_size(adapter: D3DKMT_HANDLE) -> Result<D3DKMT_SEGMENTSIZEINFO, String> {
    let mut segment_size = D3DKMT_SEGMENTSIZEINFO::default();
    let query = D3DKMT_QUERYADAPTERINFO {
        hAdapter: adapter,
        Type: KMTQUERYADAPTERINFOTYPE::GetSegmentSize,
        pPrivateDriverData: &mut segment_size as *mut _ as *mut c_void,
        PrivateDriverDataSize: std::mem::size_of::<D3DKMT_SEGMENTSIZEINFO>() as u32,
    };
    let status = unsafe { D3DKMTQueryAdapterInfo(&query) };
    if status != 0 {
        return Err(format!(
            "D3DKMTQueryAdapterInfo(GetSegmentSize) failed: 0x{:08X}",
            status as u32
        ));
    }
    Ok(segment_size)
}

fn queue_engine_to_engine_flag(queue_engine: u32) -> Option<u32> {
    match queue_engine {
        4 => Some(2),
        5 => Some(1),
        6 => Some(0),
        7 => Some(4),
        _ => None,
    }
}

fn engine_flag_to_queue_engine_id(engine_flag: u32) -> Option<u8> {
    match engine_flag {
        1 => Some(5),
        2 => Some(4),
        4 => Some(7),
        _ => None,
    }
}

fn infer_gfx_major(_device_id: u32) -> u32 {
    DEFAULT_GFX_MAJOR
}

fn find_engine_ordinal(raw: &[u8], engine: u32) -> Option<u32> {
    raw[ENGINE_ORDINAL_TABLE_OFFSET..ENGINE_ORDINAL_TABLE_OFFSET + ENGINE_ORDINAL_TABLE_LEN]
        .iter()
        .position(|value| *value == engine as u8)
        .map(|ordinal| ordinal as u32)
}

fn ordinal_bit(ordinal: u32) -> u64 {
    1u64.checked_shl(ordinal).unwrap_or(0)
}

fn read_u8(data: &[u8], offset: usize) -> u8 {
    data[offset]
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

fn read_u64(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

fn write_u16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn or_u32(data: &mut [u8], offset: usize, mask: u32) {
    let value = read_u32(data, offset) | mask;
    write_u32(data, offset, value);
}
