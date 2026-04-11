//! Tape — Thread-local computation graph for reverse-mode automatic differentiation.
//!
//! Modeled after PyTorch's autograd engine:
//! 1. Forward ops record `TapeNode` entries with backward closures
//! 2. `backward()` traverses nodes in reverse, dispatching backward kernels
//! 3. Gradients accumulate onto `Tensor.grad` via `accumulate_grad()`
//!
//! Key design:
//! - Thread-local tape (no locking needed for single-GPU training)
//! - Each node stores `backward_fn` as `FnOnce` closure
//! - Gradient buffers are tracked by TensorId in a global registry

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use std::cell::RefCell;
#[cfg(feature = "rocm")]
use std::collections::HashMap;

#[cfg(feature = "rocm")]
use crate::gpu_backend::GpuBuffer;
#[cfg(feature = "rocm")]
use super::tensor::{Tensor, TensorId};
#[cfg(feature = "rocm")]
use super::gpu_context::GpuRuntime;

// ── Backward function type ──

/// Backward function: (grad_output, saved_tensors, runtime) → input_grads
/// Returns one Option<Arc<GpuBuffer>> per input (None if input doesn't need grad).
#[cfg(feature = "rocm")]
pub type BackwardFn = Box<dyn FnOnce(
    &GpuBuffer,                    // grad_output
    &[Arc<GpuBuffer>],             // saved tensors from forward
    &Arc<GpuRuntime>,              // runtime (for GPU dispatch + allocation)
) -> Result<Vec<Option<Arc<GpuBuffer>>>, String>>;

// ── Tape node ──

/// A node in the computation tape — records one forward operation.
#[cfg(feature = "rocm")]
pub struct TapeNode {
    pub label: String,
    /// TensorId of this node's output (used to find its grad during backward)
    pub output_id: TensorId,
    /// TensorIds of inputs that need gradients
    pub input_ids: Vec<Option<TensorId>>,
    /// Whether each input requires grad
    pub input_requires_grad: Vec<bool>,
    /// Saved tensors/buffers from forward (e.g., activations needed for backward)
    pub saved_tensors: Vec<Arc<GpuBuffer>>,
    /// The backward function (consumed once during backward())
    pub backward_fn: Option<BackwardFn>,
}

// ── Thread-local tape storage ──

#[cfg(feature = "rocm")]
thread_local! {
    /// The computation tape: list of forward operations in order.
    static TAPE_NODES: RefCell<Vec<TapeNode>> = RefCell::new(Vec::new());
    /// Whether tape recording is enabled (false = inference mode / no_grad).
    static TAPE_RECORDING: RefCell<bool> = RefCell::new(false);
    /// Gradient registry: TensorId → grad buffer.
    /// Populated during backward() so nodes can find their output's gradient.
    static GRAD_REGISTRY: RefCell<HashMap<TensorId, Arc<GpuBuffer>>> = RefCell::new(HashMap::new());
}

// ── Tape API ──

/// Tape — static methods for managing the thread-local computation graph.
#[cfg(feature = "rocm")]
pub struct Tape;

#[cfg(feature = "rocm")]
impl Tape {
    // ── Recording control (like torch.set_grad_enabled) ──

    /// Enable tape recording (training mode).
    pub fn enable() {
        TAPE_RECORDING.with(|r| *r.borrow_mut() = true);
    }

    /// Disable tape recording (inference / no_grad mode).
    pub fn disable() {
        TAPE_RECORDING.with(|r| *r.borrow_mut() = false);
    }

    /// Check if recording is active.
    pub fn is_recording() -> bool {
        TAPE_RECORDING.with(|r| *r.borrow())
    }


    // ── Compatibility aliases (used by tests.rs) ──

    /// Alias for clear() — reset the tape.
    pub fn reset() { Self::clear(); }

    /// Alias for enable() — begin recording.
    pub fn start_recording() { Self::enable(); }

    /// Alias for disable() — stop recording.
    pub fn stop_recording() { Self::disable(); }

    /// No-grad guard — disables recording, returns guard that re-enables on drop.
    pub fn no_grad() -> NoGradGuard {
        Self::disable();
        NoGradGuard { _prev: true }
    }

    // ── Node recording ──

    /// Record a forward operation on the tape.
    ///
    /// Called by each op's forward() implementation.
    /// Returns the node ID (used as the output tensor's tape_node).
    ///
    /// # Arguments
    /// - `label`: human-readable op name (e.g., "matmul", "add")
    /// - `output_id`: TensorId of the output tensor
    /// - `input_ids`: TensorId of each input (None if not tracked)
    /// - `input_requires_grad`: whether each input needs gradient
    /// - `saved_tensors`: buffers saved for backward (activations, weights, etc.)
    /// - `backward_fn`: closure that computes input gradients from output gradient
    pub fn record(
        label: &str,
        output_id: TensorId,
        input_ids: Vec<Option<TensorId>>,
        input_requires_grad: Vec<bool>,
        saved_tensors: Vec<Arc<GpuBuffer>>,
        backward_fn: BackwardFn,
    ) -> usize {
        TAPE_NODES.with(|nodes| {
            let mut nodes = nodes.borrow_mut();
            let id = nodes.len();
            nodes.push(TapeNode {
                label: label.to_string(),
                output_id,
                input_ids,
                input_requires_grad,
                saved_tensors,
                backward_fn: Some(backward_fn),
            });
            id
        })
    }

    // ── Backward pass (core autograd engine) ──

    /// Run backward pass from a loss tensor.
    ///
    /// Like PyTorch's `loss.backward()`:
    /// 1. Seeds loss gradient with 1.0
    /// 2. Traverses tape in reverse order
    /// 3. For each node: looks up output grad → calls backward_fn → accumulates input grads
    /// 4. After completion, all parameter tensors have `.grad()` populated
    pub fn backward(loss: &Tensor, runtime: &Arc<GpuRuntime>) -> Result<(), String> {
        // Step 1: Seed the loss gradient (∂L/∂L = 1.0)
        let n = loss.numel();
        let seed = runtime.device.alloc_vram(n * 4)?;
        let ones = vec![1.0f32; n];
        seed.write(unsafe {
            std::slice::from_raw_parts(ones.as_ptr() as *const u8, n * 4)
        });
        let seed_arc = Arc::new(seed);
        loss.set_grad(seed_arc.clone());

        // Register the loss gradient so tape nodes can find it
        GRAD_REGISTRY.with(|reg| {
            reg.borrow_mut().insert(loss.id(), seed_arc.clone());
        });

        // Step 2: Get the number of nodes to traverse
        let node_count = TAPE_NODES.with(|nodes| nodes.borrow().len());

        // Step 3: Reverse traverse
        for i in (0..node_count).rev() {
            // Extract node data (taking backward_fn out for FnOnce consumption)
            let (backward_fn, saved, output_id, input_ids, input_requires_grad, _label) =
                TAPE_NODES.with(|nodes| {
                    let mut nodes = nodes.borrow_mut();
                    let node = &mut nodes[i];
                    (
                        node.backward_fn.take(),
                        node.saved_tensors.clone(),
                        node.output_id,
                        node.input_ids.clone(),
                        node.input_requires_grad.clone(),
                        node.label.clone(),
                    )
                });

            // Look up the gradient for this node's output
            let grad_output = GRAD_REGISTRY.with(|reg| {
                reg.borrow().get(&output_id).cloned()
            });

            let grad_output = match grad_output {
                Some(g) => g,
                None => {
                    // No gradient for this output — skip (e.g., detached tensor)
                    continue;
                }
            };

            // Call backward_fn to compute input gradients
            if let Some(bfn) = backward_fn {
                let input_grads = bfn(&grad_output, &saved, runtime)?;

                // Accumulate gradients onto input tensors
                for (j, maybe_grad) in input_grads.into_iter().enumerate() {
                    if j >= input_ids.len() || !input_requires_grad[j] {
                        continue;
                    }
                    if let (Some(grad_buf), Some(input_id)) = (maybe_grad, input_ids[j]) {
                        // Check if a gradient already exists for this input
                        let existing = GRAD_REGISTRY.with(|reg| {
                            reg.borrow().get(&input_id).cloned()
                        });

                        if let Some(existing_grad) = existing {
                            // GPU accumulate: existing_grad += grad_buf (in-place)
                            Self::gpu_accumulate_grad(&existing_grad, &grad_buf, runtime)?;
                            // existing_grad is already in the registry — no need to re-insert
                        } else {
                            // First gradient for this input — just register it
                            GRAD_REGISTRY.with(|reg| {
                                reg.borrow_mut().insert(input_id, grad_buf);
                            });
                        }
                    }
                }
            }
        }

        // Final sync: ensure all backward GPU dispatches are complete
        // before CPU reads gradients (lazy execution barrier)
        runtime.synchronize()?;

        Ok(())
    }

    /// Run backward pass from a mid-graph tensor with a pre-computed gradient.
    ///
    /// Used by DCI mode: loss is computed on CPU, grad_h is written back to GPU,
    /// then backward traverses from the tensor's tape node backward.
    pub fn backward_from_grad(
        tensor: &Tensor,
        grad: Arc<GpuBuffer>,
        runtime: &Arc<GpuRuntime>,
    ) -> Result<(), String> {
        // Register the gradient for this tensor
        tensor.set_grad(grad.clone());
        GRAD_REGISTRY.with(|reg| {
            reg.borrow_mut().insert(tensor.id(), grad);
        });

        // Profiling: track time per op label
        use std::cell::Cell;
        thread_local! { static BWD_CALL: Cell<u32> = Cell::new(0); }
        let call_idx = BWD_CALL.with(|c| { let v = c.get(); c.set(v + 1); v });
        let do_prof = call_idx >= 1 && call_idx < 4; // steps 2-4
        let mut label_times: std::collections::HashMap<String, (f64, u32)> = std::collections::HashMap::new();

        // Same traversal as backward()
        let node_count = TAPE_NODES.with(|nodes| nodes.borrow().len());

        for i in (0..node_count).rev() {
            let (backward_fn, saved, output_id, input_ids, input_requires_grad, label) =
                TAPE_NODES.with(|nodes| {
                    let mut nodes = nodes.borrow_mut();
                    let node = &mut nodes[i];
                    (
                        node.backward_fn.take(),
                        node.saved_tensors.clone(),
                        node.output_id,
                        node.input_ids.clone(),
                        node.input_requires_grad.clone(),
                        node.label.clone(),
                    )
                });

            let grad_output = GRAD_REGISTRY.with(|reg| {
                reg.borrow().get(&output_id).cloned()
            });

            let grad_output = match grad_output {
                Some(g) => g,
                None => continue,
            };

            if let Some(bfn) = backward_fn {
                let nt0 = std::time::Instant::now();
                let input_grads = bfn(&grad_output, &saved, runtime)?;
                let node_ms = nt0.elapsed().as_secs_f64() * 1e3;

                if do_prof {
                    let entry = label_times.entry(label.clone()).or_insert((0.0, 0));
                    entry.0 += node_ms;
                    entry.1 += 1;
                }

                for (j, maybe_grad) in input_grads.into_iter().enumerate() {
                    if j >= input_ids.len() || !input_requires_grad[j] {
                        continue;
                    }
                    if let (Some(grad_buf), Some(input_id)) = (maybe_grad, input_ids[j]) {
                        let existing = GRAD_REGISTRY.with(|reg| {
                            reg.borrow().get(&input_id).cloned()
                        });

                        if let Some(existing_grad) = existing {
                            Self::gpu_accumulate_grad(&existing_grad, &grad_buf, runtime)?;
                        } else {
                            GRAD_REGISTRY.with(|reg| {
                                reg.borrow_mut().insert(input_id, grad_buf);
                            });
                        }
                    }
                }
            }
        }

        runtime.synchronize()?;

        if do_prof {
            let mut sorted: Vec<_> = label_times.into_iter().collect();
            sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let parts: Vec<String> = sorted.iter()
                .map(|(label, (ms, count))| format!("{}={:.1}×{}", label, ms, count))
                .collect();
            eprintln!("  [BWD] {}", parts.join(" "));
        }
        Ok(())
    }

    /// GPU gradient accumulation: existing_buf += new_buf (in-place).
    ///
    /// Uses BlockDSL residual_add kernel for element-wise y[i] += x[i].
    /// This is called during backward when a tensor's gradient is contributed
    /// to by multiple ops (e.g., a parameter used in multiple layers).
    fn gpu_accumulate_grad(
        existing: &GpuBuffer,
        new_grad: &GpuBuffer,
        runtime: &Arc<GpuRuntime>,
    ) -> Result<(), String> {
        let n_bytes = existing.size.min(new_grad.size);
        let n_elems = (n_bytes / 4) as u32;
        if n_elems == 0 { return Ok(()); }

        let kernel = runtime.ensure_kernel_blockdsl(
            "grad_accumulate",
            || crate::t0::elementwise_kernels::build_residual_add(),
        )?;

        // residual_add kernarg: [x_ptr(u64), y_ptr(u64), n(u32)]
        // Semantics: y[i] += x[i]
        let ka = crate::kernargs![
            new_grad.gpu_addr() => u64,   // x = new gradient (source)
            existing.gpu_addr() => u64,   // y = existing gradient (accumulate target)
            n_elems => u32                 // element count
        ];

        let grid_x = crate::t0::elementwise_kernels::elementwise_grid(n_elems);
        runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)
    }

    /// After backward(), copy gradients from registry onto Tensor.grad fields.
    /// Call this with all parameter tensors to populate their .grad().
    pub fn sync_grads(params: &[&Tensor]) {
        GRAD_REGISTRY.with(|reg| {
            let reg = reg.borrow();
            for p in params {
                if let Some(grad) = reg.get(&p.id()) {
                    p.set_grad(grad.clone());
                }
            }
        });
    }

    /// Clear all tape nodes and gradient registry (call between training steps).
    /// Buffer deallocation is deferred to a background thread to avoid blocking
    /// the training loop with KFD ioctl syscalls (~0.3ms per buffer × dozens of buffers).
    pub fn clear() {
        // Extract Arc<GpuBuffer> from saved_tensors before clearing nodes
        let mut deferred_bufs: Vec<Arc<GpuBuffer>> = Vec::new();
        TAPE_NODES.with(|nodes| {
            let mut nodes = nodes.borrow_mut();
            for node in nodes.iter_mut() {
                deferred_bufs.append(&mut node.saved_tensors);
            }
            nodes.clear();
        });
        // Also extract grad registry buffers
        GRAD_REGISTRY.with(|reg| {
            let mut reg = reg.borrow_mut();
            deferred_bufs.extend(reg.drain().map(|(_, buf)| buf));
        });
        // Drop synchronously — async drop causes VRAM race between steps
        drop(deferred_bufs);
    }

    /// Number of recorded nodes.
    pub fn len() -> usize {
        TAPE_NODES.with(|nodes| nodes.borrow().len())
    }
}

// ── no_grad context ──

/// RAII guard for `torch.no_grad()` equivalent.
/// Disables tape recording on creation, restores on drop.
#[cfg(feature = "rocm")]
pub struct NoGrad {
    was_recording: bool,
}

#[cfg(feature = "rocm")]
impl NoGrad {
    pub fn new() -> Self {
        let was = Tape::is_recording();
        Tape::disable();
        NoGrad { was_recording: was }
    }
}

#[cfg(feature = "rocm")]
impl Drop for NoGrad {
    fn drop(&mut self) {
        if self.was_recording {
            Tape::enable();
        }
    }
}

/// Guard that re-enables recording when dropped.
#[cfg(feature = "rocm")]
pub struct NoGradGuard { _prev: bool }

#[cfg(feature = "rocm")]
impl Drop for NoGradGuard {
    fn drop(&mut self) {
        Tape::enable();
    }
}
