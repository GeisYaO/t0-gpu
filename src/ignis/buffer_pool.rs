//! BufferPool — Power-of-2 GPU buffer cache for reducing allocation overhead.

#[cfg(feature = "rocm")]
use std::collections::HashMap;
#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::gpu_backend::{GpuBuffer, GpuDevice};

/// GPU buffer pool with power-of-2 bucket caching.
///
/// Reuses freed buffers to avoid expensive KFD alloc/free ioctls.
#[cfg(feature = "rocm")]
pub struct BufferPool {
    device: Arc<GpuDevice>,
    buckets: HashMap<usize, Vec<GpuBuffer>>,
    hits: u64,
    misses: u64,
}

#[cfg(feature = "rocm")]
impl BufferPool {
    pub fn new(device: &Arc<GpuDevice>) -> Self {
        Self {
            device: device.clone(),
            buckets: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Round up to next power of 2, with minimum 4096 bytes.
    /// Must match KFD's page-aligned allocation size (alloc_memory rounds to 4096).
    fn bucket_size(size: usize) -> usize {
        let min_size = 4096; // KFD page size — all allocations are at least this
        let size = size.max(min_size);
        size.next_power_of_two()
    }

    /// Allocate a buffer from the pool (or create new).
    pub fn allocate(&mut self, size: usize) -> Result<GpuBuffer, String> {
        let bucket = Self::bucket_size(size);
        if let Some(bufs) = self.buckets.get_mut(&bucket) {
            if let Some(buf) = bufs.pop() {
                self.hits += 1;
                buf.zero();
                return Ok(buf);
            }
        }
        self.misses += 1;
        // Allocate bucket-sized buffer (already page-aligned ≥4096)
        let buf = self.device.alloc_vram(bucket)?;
        buf.zero();
        Ok(buf)
    }

    /// Return a buffer to the pool for reuse.
    /// Uses buf.size (KFD-aligned) as bucket key for exact match on next allocate.
    pub fn release(&mut self, buf: GpuBuffer) {
        let bucket = Self::bucket_size(buf.size);
        self.buckets.entry(bucket).or_insert_with(Vec::new).push(buf);
    }

    /// Pool statistics.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Total cached buffers across all buckets.
    pub fn cached_count(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }

    /// Flush all cached buffers (free memory).
    pub fn flush(&mut self) {
        self.buckets.clear();
    }
}
