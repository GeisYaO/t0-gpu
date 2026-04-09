//! Cross-entropy loss — softmax + negative log-likelihood with autodiff.
//!
//! Forward: loss = -sum(one_hot(targets) * log(softmax(logits))) / batch
//! Backward: d_logits = (softmax(logits) - one_hot(targets)) / batch
//!
//! Uses T0 BlockDSL softmax_forward kernel (single fused WG-reduce dispatch).

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::gpu_backend::GpuBuffer;
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::tape::Tape;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Softmax Cross-Entropy Loss (fused forward + backward gradient pre-computation)
///
/// Computes:
///   probs = softmax(logits)    per row  (single BlockDSL kernel)
///   loss  = -mean(log(probs[target]))   (CPU, tiny)
///   grad  = (probs - one_hot(target)) / batch_size  (CPU, saved for backward)
///
/// # Arguments
/// - logits: [batch, vocab_size] f32
/// - targets: [batch] u32 (token indices, stored as raw u32 bits)
/// - vocab_size: vocabulary dimension (must be ≤ 256 for single-WG softmax)
///
/// # Returns
/// - Scalar loss tensor [1]
#[cfg(feature = "rocm")]
pub fn cross_entropy(
    logits: &Tensor,
    targets_buf: &GpuBuffer,
    vocab_size: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    let batch = logits.numel() / vocab_size;
    let cols = vocab_size;

    // ═══ Step 1: GPU Softmax via single BlockDSL kernel ═══
    // Replaces 4 separate math.rs kernels (row_max, exp_sub, row_sum, div)
    // with a single fused wg_reduce softmax dispatch.
    let probs_buf = runtime.alloc_f32(batch * cols)?;
    let k_softmax = runtime.ensure_kernel_blockdsl("softmax_fwd",
        || crate::t0::softmax_kernels::build_softmax_forward())?;

    let (grid_x, _) = crate::t0::softmax_kernels::softmax_grid(batch as u32);
    let ka = crate::kernargs![
        logits.gpu_addr() => u64,
        probs_buf.gpu_addr() => u64,
        cols as u32 => u32
    ];
    runtime.dispatch(&k_softmax, [grid_x, 1, 1], &ka)?;

    // ═══ Step 2: CPU loss + grad (batch-level, tiny) ═══
    runtime.wait_idle()?;
    let probs = read_f32_vec(&probs_buf, batch * cols);

    let mut targets_bytes = vec![0u8; batch * 4];
    targets_buf.read(&mut targets_bytes);
    let targets: Vec<u32> = targets_bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Loss: -mean(log(probs[target]))
    let mut total_loss = 0f32;
    for b in 0..batch {
        let target = targets[b] as usize;
        if target < cols {
            total_loss += -probs[b * cols + target].max(1e-10).ln();
        }
    }
    total_loss /= batch as f32;

    let loss = Tensor::from_f32(runtime, &[total_loss], &[1], "ce_loss")?;

    if Tape::is_recording() && logits.requires_grad() {
        let logits_id = Some(logits.id());
        let v = cols;
        let bs = batch;

        // Grad: (softmax - one_hot) / batch → upload to GPU
        let mut grad_data = probs;
        for b in 0..bs {
            let target = targets[b] as usize;
            if target < v {
                grad_data[b * v + target] -= 1.0;
            }
            for c in 0..v {
                grad_data[b * v + c] /= bs as f32;
            }
        }
        let grad_buf = runtime.alloc_f32(bs * v)?;
        write_f32(&grad_buf, &grad_data);
        let grad_arc = Arc::new(grad_buf);

        let node_id = Tape::record(
            "cross_entropy",
            loss.id(),
            vec![logits_id],
            vec![true],
            vec![grad_arc],
            Box::new(move |_grad_output, saved, _runtime| {
                Ok(vec![Some(saved[0].clone())])
            }),
        );
        loss.set_tape_node(node_id);
    }

    Ok(loss)
}

#[cfg(feature = "rocm")]
fn read_f32_vec(buf: &GpuBuffer, n: usize) -> Vec<f32> {
    let mut data = vec![0f32; n];
    buf.read(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) });
    data
}

#[cfg(feature = "rocm")]
fn write_f32(buf: &GpuBuffer, data: &[f32]) {
    buf.write(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) });
}
