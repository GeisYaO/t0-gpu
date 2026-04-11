//! Dynamic loss scaler for mixed-precision training.
//!
//! Maintains a scaling factor that grows when training is stable
//! and backs off when NaN/Inf is detected.

#[cfg(feature = "rocm")]
use crate::gpu_backend::GpuBuffer;
#[cfg(feature = "rocm")]
use super::tensor::Tensor;

/// Dynamic loss scaler.
///
/// - Scales loss by `scale` before backward
/// - Unscales gradients after backward (divide by `scale`)
/// - If gradients contain NaN/Inf: skip optimizer step, reduce `scale`
/// - If N consecutive steps are clean: increase `scale`
pub struct LossScaler {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
    pub consecutive_clean: usize,
    pub max_scale: f32,
    pub min_scale: f32,
}

impl LossScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0,       // initial scale
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 200, // grow after 200 clean steps
            consecutive_clean: 0,
            max_scale: 2f32.powi(24),
            min_scale: 1.0,
        }
    }

    pub fn with_scale(initial_scale: f32) -> Self {
        let mut s = Self::new();
        s.scale = initial_scale;
        s
    }

    /// Scale a loss value before backward.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Unscale gradients after backward.
    /// Returns true if gradients are clean (no NaN/Inf).
    #[cfg(feature = "rocm")]
    pub fn unscale_grads(&mut self, params: &[&Tensor]) -> bool {
        let inv_scale = 1.0 / self.scale;
        let mut is_clean = true;

        for param in params {
            if let Some(grad) = param.grad() {
                let n = param.numel();
                let mut data = read_f32(&grad, n);

                for v in &mut data {
                    *v *= inv_scale;
                    if v.is_nan() || v.is_infinite() {
                        is_clean = false;
                    }
                }

                if is_clean {
                    write_f32(&grad, &data);
                }
            }
        }

        is_clean
    }

    /// Update scale based on gradient health.
    ///
    /// Call after every step:
    /// - If clean: increment counter, possibly grow
    /// - If dirty: reset counter, backoff
    pub fn update(&mut self, grads_are_clean: bool) {
        if grads_are_clean {
            self.consecutive_clean += 1;
            if self.consecutive_clean >= self.growth_interval {
                self.scale = (self.scale * self.growth_factor).min(self.max_scale);
                self.consecutive_clean = 0;
            }
        } else {
            self.scale = (self.scale * self.backoff_factor).max(self.min_scale);
            self.consecutive_clean = 0;
        }
    }

    /// Current scale factor.
    pub fn current_scale(&self) -> f32 { self.scale }
}

impl Default for LossScaler {
    fn default() -> Self { Self::new() }
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
