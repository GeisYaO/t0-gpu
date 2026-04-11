//! Tensor — GPU-backed tensor with autodiff support.
//!
//! Core data structure modeled after PyTorch's Tensor:
//! - `Arc<GpuBuffer>` for VRAM data (reference-counted sharing)
//! - Unique `TensorId` for tape tracking
//! - `grad` buffer for gradient accumulation
//! - `tape_node` linking to computation graph

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use std::cell::{Cell, RefCell};
#[cfg(feature = "rocm")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "rocm")]
use crate::gpu_backend::{GpuBuffer, GpuDevice};
#[cfg(feature = "rocm")]
use super::gpu_context::GpuRuntime;

/// Re-export DType from canonical definition in t0::dsl
pub use crate::t0::dsl::DType;

/// Unique tensor identifier (monotonically increasing).
pub type TensorId = u64;

/// Tape node identifier.
pub type NodeId = usize;

/// Global counter for unique tensor IDs.
#[cfg(feature = "rocm")]
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "rocm")]
fn next_tensor_id() -> TensorId {
    NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed)
}

/// GPU-backed tensor with automatic differentiation support.
///
/// Like PyTorch's Tensor:
/// - Immutable data buffer (shared via Arc)
/// - Mutable gradient (RefCell, lazy-allocated)
/// - Optional tape node linking to computation graph
#[cfg(feature = "rocm")]
pub struct Tensor {
    id: TensorId,
    buf: Arc<GpuBuffer>,
    runtime: Arc<GpuRuntime>,
    shape: Vec<usize>,
    dtype: DType,
    label: String,
    // Autodiff state
    grad: RefCell<Option<Arc<GpuBuffer>>>,
    tape_node: Cell<Option<NodeId>>,
    requires_grad: bool,
}

#[cfg(feature = "rocm")]
impl Tensor {
    // ── Constructors ──

    /// Create a tensor from f32 data on GPU.
    pub fn from_f32(
        runtime: &Arc<GpuRuntime>,
        data: &[f32],
        shape: &[usize],
        label: &str,
    ) -> Result<Self, String> {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "data length {} != shape product {}", data.len(), n);
        // Pad allocation to 512 bytes (128 f32s) min — vectorized kernels
        // (adamw_1d, frobenius_norm_partial) use dwordx4 loads without bounds checks.
        let alloc_bytes = ((n * 4).max(512) + 511) & !511;
        let buf = runtime.device.alloc_vram(alloc_bytes)?;
        buf.zero(); // zero-fill first to ensure padding is clean
        buf.write(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, n * 4)
        });
        Ok(Tensor {
            id: next_tensor_id(),
            buf: Arc::new(buf),
            runtime: runtime.clone(),
            shape: shape.to_vec(),
            dtype: DType::F32,
            label: label.to_string(),
            grad: RefCell::new(None),
            tape_node: Cell::new(None),
            requires_grad: false,
        })
    }

    /// Create a tensor from an existing GpuBuffer.
    pub fn from_buffer(
        buf: Arc<GpuBuffer>,
        runtime: &Arc<GpuRuntime>,
        shape: &[usize],
        dtype: DType,
        label: &str,
    ) -> Self {
        Tensor {
            id: next_tensor_id(),
            buf,
            runtime: runtime.clone(),
            shape: shape.to_vec(),
            dtype,
            label: label.to_string(),
            grad: RefCell::new(None),
            tape_node: Cell::new(None),
            requires_grad: false,
        }
    }

    /// Create a zero-initialized f32 tensor.
    pub fn zeros(
        runtime: &Arc<GpuRuntime>,
        shape: &[usize],
        label: &str,
    ) -> Result<Self, String> {
        let n: usize = shape.iter().product();
        let buf = runtime.device.alloc_vram(n * 4)?;
        buf.zero();
        Ok(Tensor {
            id: next_tensor_id(),
            buf: Arc::new(buf),
            runtime: runtime.clone(),
            shape: shape.to_vec(),
            dtype: DType::F32,
            label: label.to_string(),
            grad: RefCell::new(None),
            tape_node: Cell::new(None),
            requires_grad: false,
        })
    }

    // ── Autodiff control ──

    /// Mark this tensor as requiring gradients. Returns self for chaining.
    pub fn requires_grad_(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Set requires_grad on mutable reference.
    pub fn set_requires_grad(&mut self, v: bool) {
        self.requires_grad = v;
    }

    /// Check if this tensor requires gradients.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Detach from computation graph — creates a new tensor sharing data but no grad/tape.
    pub fn detach(&self) -> Self {
        Tensor {
            id: next_tensor_id(),
            buf: self.buf.clone(),
            runtime: self.runtime.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            label: format!("{}_detached", self.label),
            grad: RefCell::new(None),
            tape_node: Cell::new(None),
            requires_grad: false,
        }
    }

    /// Get runtime reference.
    pub fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.runtime
    }

    /// Get device reference (convenience, delegates to runtime).
    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.runtime.device
    }

    // ── Accessors ──

    /// Unique tensor identifier.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Get underlying GPU buffer reference.
    pub fn buffer(&self) -> &GpuBuffer {
        &self.buf
    }

    /// Get underlying GPU buffer as Arc.
    pub fn buffer_arc(&self) -> &Arc<GpuBuffer> {
        &self.buf
    }

    /// GPU virtual address.
    pub fn gpu_addr(&self) -> u64 {
        self.buf.gpu_addr()
    }

    /// Shape as slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Label for debugging.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Byte size of the tensor data.
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    // ── Gradient management ──

    /// Get gradient buffer (if computed).
    pub fn grad(&self) -> Option<Arc<GpuBuffer>> {
        self.grad.borrow().clone()
    }

    /// Set gradient buffer (replaces existing).
    pub fn set_grad(&self, grad: Arc<GpuBuffer>) {
        *self.grad.borrow_mut() = Some(grad);
    }

    /// Accumulate gradient: grad += incoming.
    /// If no existing grad, just copies the buffer on GPU. Otherwise dispatches GPU add kernel.
    pub fn accumulate_grad(
        &self,
        incoming: &Arc<GpuBuffer>,
        _device: &Arc<GpuDevice>,
    ) -> Result<(), String> {
        let mut grad_ref = self.grad.borrow_mut();
        let n = self.numel();
        match grad_ref.as_ref() {
            None => {
                // First gradient — GPU copy via BlockDSL memcpy kernel
                let new_buf = self.runtime.alloc_f32(n)?;
                let kernel = self.runtime.ensure_kernel_blockdsl(
                    "grad_memcpy",
                    || crate::t0::elementwise_kernels::build_memcpy(),
                )?;
                let ka = crate::kernargs![
                    incoming.gpu_addr() => u64,
                    new_buf.gpu_addr() => u64,
                    n as u32 => u32
                ];
                let grid_x = crate::t0::elementwise_kernels::elementwise_grid(n as u32);
                self.runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
                *grad_ref = Some(Arc::new(new_buf));
            }
            Some(existing) => {
                // GPU in-place add: existing += incoming via BlockDSL residual_add
                let kernel = self.runtime.ensure_kernel_blockdsl(
                    "grad_accumulate",
                    || crate::t0::elementwise_kernels::build_residual_add(),
                )?;
                let ka = crate::kernargs![
                    incoming.gpu_addr() => u64,
                    existing.gpu_addr() => u64,
                    n as u32 => u32
                ];
                let grid_x = crate::t0::elementwise_kernels::elementwise_grid(n as u32);
                self.runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
            }
        }
        Ok(())
    }

    /// Clear gradient.
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    // ── Tape ──

    /// Get tape node ID.
    pub fn tape_node(&self) -> Option<NodeId> {
        self.tape_node.get()
    }

    /// Set tape node ID.
    pub fn set_tape_node(&self, id: NodeId) {
        self.tape_node.set(Some(id));
    }

    // ── Data transfer ──

    /// Read data back to CPU as f32 vector.
    /// Synchronizes GPU before reading to ensure all dispatches are complete.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        // Lazy sync: ensure all GPU work is done before CPU reads
        let _ = self.runtime.queue.wait_idle();
        let n = self.numel();
        let mut data = vec![0f32; n];
        self.buf.read(unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4)
        });
        data
    }

    /// Read a single f32 value (index 0).
    pub fn to_f32_scalar(&self) -> f32 {
        let v = self.to_f32_vec();
        v[0]
    }

    // ── Convenience ops (use stored device) ──

    /// Matrix multiply: self @ rhs.
    /// self: [M, K], rhs: [K, N] → [M, N].
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        super::ops::bf16_matmul::matmul(self, rhs, &self.device())
            .expect("Tensor.matmul() failed")
    }

    /// Sum all elements → scalar tensor.
    pub fn sum(&self) -> Tensor {
        super::ops::add::sum(self, &self.device())
            .expect("Tensor.sum() failed")
    }

    /// Vector dot product: sum(self * rhs).
    pub fn dot(&self, rhs: &Tensor) -> Tensor {
        let product = super::ops::add::elementwise_mul(self, rhs, &self.device())
            .expect("Tensor.dot() mul failed");
        super::ops::add::sum(&product, &self.device())
            .expect("Tensor.dot() sum failed")
    }

    /// Reshape (zero-copy if same numel).
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        super::ops::shape_ops::reshape(self, new_shape, &self.device())
            .expect("Tensor.reshape() failed")
    }

    /// Transpose 2D: [M,N] → [N,M].
    pub fn t(&self) -> Tensor {
        super::ops::shape_ops::transpose(self, &self.device())
            .expect("Tensor.t() failed")
    }

    /// Mean of all elements → scalar.
    pub fn mean(&self) -> Tensor {
        super::ops::shape_ops::mean(self, &self.device())
            .expect("Tensor.mean() failed")
    }

    /// ReLU activation.
    pub fn relu(&self) -> Tensor {
        super::ops::shape_ops::relu(self, &self.device())
            .expect("Tensor.relu() failed")
    }

    /// Softmax along last dim.
    pub fn softmax(&self) -> Tensor {
        super::ops::shape_ops::softmax(self, &self.device())
            .expect("Tensor.softmax() failed")
    }

    /// Negate: -self.
    pub fn neg(&self) -> Tensor {
        super::ops::shape_ops::neg(self, &self.device())
            .expect("Tensor.neg() failed")
    }

    /// Slice rows: self[start..end].
    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        super::ops::shape_ops::slice_rows(self, start, end, &self.device())
            .expect("Tensor.slice() failed")
    }
}

#[cfg(feature = "rocm")]
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            id: self.id,
            buf: self.buf.clone(),
            runtime: self.runtime.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            label: self.label.clone(),
            grad: RefCell::new(self.grad.borrow().clone()),
            tape_node: Cell::new(self.tape_node.get()),
            requires_grad: self.requires_grad,
        }
    }
}

#[cfg(feature = "rocm")]
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(id={}, '{}', {:?}, {}, grad={})",
            self.id, self.label, self.shape, self.dtype.name(),
            if self.grad.borrow().is_some() { "yes" } else { "no" })
    }
}

// ── Operator overloading ──

/// &Tensor + &Tensor → Tensor (element-wise add)
#[cfg(feature = "rocm")]
impl<'a, 'b> std::ops::Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor {
        super::ops::add::add(self, rhs, &self.runtime.device)
            .expect("Tensor + Tensor failed")
    }
}

/// &Tensor * f32 → Tensor (scalar multiply)
#[cfg(feature = "rocm")]
impl<'a> std::ops::Mul<f32> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        super::ops::add::scale(self, rhs, &self.runtime.device)
            .expect("Tensor * f32 failed")
    }
}

/// &Tensor * &Tensor → Tensor (element-wise multiply)
#[cfg(feature = "rocm")]
impl<'a, 'b> std::ops::Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        super::ops::add::elementwise_mul(self, rhs, &self.runtime.device)
            .expect("Tensor * Tensor failed")
    }
}

/// &Tensor - &Tensor → Tensor (subtraction)
#[cfg(feature = "rocm")]
impl<'a, 'b> std::ops::Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor {
        super::ops::shape_ops::sub(self, rhs, &self.runtime.device)
            .expect("Tensor - Tensor failed")
    }
}

/// -&Tensor → Tensor (negation)
#[cfg(feature = "rocm")]
impl<'a> std::ops::Neg for &'a Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        super::ops::shape_ops::neg(self, &self.runtime.device)
            .expect("-Tensor failed")
    }
}
