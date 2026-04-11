//! Unified GPU Backend Interface
//!
//! Provides consistent type names (`GpuDevice`, `GpuBuffer`, `GpuQueue`,
//! `GpuKernel`, `DispatchPool`, `KernelLoadConfig`) regardless of whether
//! the `rocm` (KFD) or `wsl_dxg` (DXG) feature is enabled.
//!
//! This allows all higher-level code (ignis, t0 kernels, prelude) to
//! import from a single place without `#[cfg]` spaghetti.

// =============================================================================
// Re-export types from the active backend
// =============================================================================

#[cfg(feature = "rocm")]
pub use crate::kfd::{
    KfdDevice as GpuDevice,
    GpuBuffer,
    AqlQueue as GpuQueue,
    GpuKernel,
    KernelLoadConfig,
    DispatchPool,
};

#[cfg(all(feature = "wsl_dxg", not(feature = "rocm")))]
pub use crate::wsl_dxg::{
    WslDxgDevice as GpuDevice,
    WslGpuMemory as GpuBuffer,
    WslAqlQueue as GpuQueue,
    GpuKernel,
    KernelLoadConfig,
    DispatchPool,
};

// =============================================================================
// Common trait: anything that can act as a dispatch queue
// =============================================================================

/// Abstract interface for GPU dispatch operations.
/// Implemented by both KFD's AqlQueue and DXG's WslAqlQueue.
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub trait DispatchQueue {
    type Buffer;
    type Kernel;

    fn dispatch(
        &self,
        kernel: &Self::Kernel,
        grid: [u32; 3],
        kernargs: &Self::Buffer,
    ) -> Result<(), String>;

    fn dispatch_signal(
        &self,
        kernel: &Self::Kernel,
        grid: [u32; 3],
        kernargs: &Self::Buffer,
        signal: Option<&Self::Buffer>,
    ) -> Result<(), String>;

    fn submit(
        &self,
        kernel: &Self::Kernel,
        grid: [u32; 3],
        kernargs: &Self::Buffer,
    );

    fn wait_idle(&self) -> Result<(), String>;
}

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
impl DispatchQueue for crate::kfd::AqlQueue {
    type Buffer = crate::kfd::GpuBuffer;
    type Kernel = crate::kfd::GpuKernel;

    fn dispatch(&self, kernel: &Self::Kernel, grid: [u32; 3], kernargs: &Self::Buffer) -> Result<(), String> {
        crate::kfd::AqlQueue::dispatch(self, kernel, grid, kernargs)
    }

    fn dispatch_signal(&self, kernel: &Self::Kernel, grid: [u32; 3], kernargs: &Self::Buffer, signal: Option<&Self::Buffer>) -> Result<(), String> {
        crate::kfd::AqlQueue::dispatch_signal(self, kernel, grid, kernargs, signal)
    }

    fn submit(&self, kernel: &Self::Kernel, grid: [u32; 3], kernargs: &Self::Buffer) {
        crate::kfd::AqlQueue::submit(self, kernel, grid, kernargs)
    }

    fn wait_idle(&self) -> Result<(), String> {
        crate::kfd::AqlQueue::wait_idle(self)
    }
}

#[cfg(all(feature = "wsl_dxg", not(feature = "rocm")))]
impl DispatchQueue for crate::wsl_dxg::WslAqlQueue {
    type Buffer = crate::wsl_dxg::WslGpuMemory;
    type Kernel = crate::wsl_dxg::GpuKernel;

    fn dispatch(&self, kernel: &Self::Kernel, grid: [u32; 3], kernargs: &Self::Buffer) -> Result<(), String> {
        crate::wsl_dxg::WslAqlQueue::dispatch(self, kernel, grid, kernargs)
    }

    fn dispatch_signal(&self, kernel: &Self::Kernel, grid: [u32; 3], kernargs: &Self::Buffer, signal: Option<&Self::Buffer>) -> Result<(), String> {
        crate::wsl_dxg::WslAqlQueue::dispatch_signal(self, kernel, grid, kernargs, signal)
    }

    fn submit(&self, kernel: &Self::Kernel, grid: [u32; 3], kernargs: &Self::Buffer) {
        crate::wsl_dxg::WslAqlQueue::submit(self, kernel, grid, kernargs)
    }

    fn wait_idle(&self) -> Result<(), String> {
        crate::wsl_dxg::WslAqlQueue::wait_idle(self)
    }
}

// =============================================================================
// Common trait: GPU buffer abstraction
// =============================================================================

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub trait GpuBufferT {
    fn gpu_addr(&self) -> u64;
    fn host_ptr(&self) -> *mut u8;
    fn size(&self) -> usize;
    fn write(&self, data: &[u8]);
    fn read(&self, buf: &mut [u8]);
    fn zero(&self);
}

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
impl GpuBufferT for crate::kfd::GpuBuffer {
    fn gpu_addr(&self) -> u64 { self.va_addr }
    fn host_ptr(&self) -> *mut u8 { self.host_ptr }
    fn size(&self) -> usize { self.size }
    fn write(&self, data: &[u8]) { crate::kfd::GpuBuffer::write(self, data) }
    fn read(&self, buf: &mut [u8]) { crate::kfd::GpuBuffer::read(self, buf) }
    fn zero(&self) { self.write(&vec![0u8; self.size]) }
}

#[cfg(all(feature = "wsl_dxg", not(feature = "rocm")))]
impl GpuBufferT for crate::wsl_dxg::WslGpuMemory {
    fn gpu_addr(&self) -> u64 { crate::wsl_dxg::WslGpuMemory::gpu_addr(self) }
    fn host_ptr(&self) -> *mut u8 { crate::wsl_dxg::WslGpuMemory::host_ptr(self) }
    fn size(&self) -> usize { self.size }
    fn write(&self, data: &[u8]) { crate::wsl_dxg::WslGpuMemory::write(self, data) }
    fn read(&self, buf: &mut [u8]) { crate::wsl_dxg::WslGpuMemory::read(self, buf) }
    fn zero(&self) { crate::wsl_dxg::WslGpuMemory::zero(self) }
}
