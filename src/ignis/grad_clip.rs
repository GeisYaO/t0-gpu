//! Gradient clipping — global L2 norm clipping.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::gpu_backend::GpuBuffer;
#[cfg(feature = "rocm")]
use super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::gpu_context::GpuRuntime;

/// Clip gradients by global L2 norm.
///
/// If the total L2 norm of all gradients exceeds `max_norm`,
/// scale all gradients down by (max_norm / total_norm).
///
/// Returns the total norm before clipping.
#[cfg(feature = "rocm")]
pub fn clip_grad_norm(
    params: &[&Tensor],
    max_norm: f32,
    runtime: &Arc<GpuRuntime>,
) -> Result<f32, String> {
    let _ = runtime.synchronize();

    // Compute total L2 norm across all gradients
    let mut total_norm_sq = 0f64;

    for param in params {
        if let Some(grad) = param.grad() {
            let n = param.numel();
            let grad_data = read_f32(&grad, n);
            let norm_sq: f64 = grad_data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            total_norm_sq += norm_sq;
        }
    }

    let total_norm = (total_norm_sq as f32).sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);

        // Scale all gradients
        for param in params {
            if let Some(grad) = param.grad() {
                let n = param.numel();
                let mut grad_data = read_f32(&grad, n);
                for v in &mut grad_data {
                    *v *= scale;
                }
                write_f32(&grad, &grad_data);
            }
        }
    }

    Ok(total_norm)
}

/// Check if any gradient contains NaN or Inf.
#[cfg(feature = "rocm")]
pub fn check_grad_health(params: &[&Tensor]) -> (bool, bool) {
    let mut has_nan = false;
    let mut has_inf = false;

    for param in params {
        if let Some(grad) = param.grad() {
            let n = param.numel();
            let data = read_f32(&grad, n);
            for &v in &data {
                if v.is_nan() { has_nan = true; }
                if v.is_infinite() { has_inf = true; }
            }
        }
    }

    (has_nan, has_inf)
}

#[cfg(feature = "rocm")]
fn read_f32(buf: &GpuBuffer, n: usize) -> Vec<f32> {
    let mut data = vec![0f32; n];
    buf.read(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) });
    data
}

#[cfg(feature = "rocm")]
fn write_f32(buf: &GpuBuffer, data: &[f32]) {
    buf.write(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) });
}
