//! BF16 MatMul — GEMM forward and backward via T0 gemm_gen.
//!
//! Forward:  Y[M,N] = X[M,K] @ W[K,N]   (f32 in → bf16 convert → WMMA → f32 out)
//! Backward: dX[M,K] = dY[M,N] @ WT      (transpose WT, then NT GEMM)
//!           dW[K,N] = X^T @ dY           (transpose X + dY, then NT GEMM)
//!
//! Uses T0's gemm_gen for auto-selected tile configs. All bf16 buffers are
//! padded to tile-aligned sizes to prevent GPU page faults.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::gpu_backend::{GpuBuffer, GpuDevice};
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::tape::Tape;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Matrix multiply: Y = X @ W
///
/// Inputs:
///   x: [M, K] f32 — activations
///   w: [K, N] f32 — weight matrix
///
/// Output:
///   y: [M, N] f32
///
/// Internally converts to bf16 for WMMA, accumulates in f32.
/// Pads M/N to tile boundaries to handle any dimension.
#[cfg(feature = "rocm")]
pub fn matmul(x: &Tensor, w: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let x_shape = x.shape();
    let w_shape = w.shape();
    assert_eq!(x_shape.len(), 2, "matmul: X must be 2D, got {:?}", x_shape);
    assert_eq!(w_shape.len(), 2, "matmul: W must be 2D, got {:?}", w_shape);

    let m = x_shape[0];
    let k = x_shape[1];
    let n = w_shape[1];
    assert_eq!(w_shape[0], k, "matmul: K mismatch X[{},{}] @ W[{},{}]", m, k, w_shape[0], n);

    let runtime = x.runtime().clone();

    // Forward GEMM: Y = X @ W
    let y_buf = dispatch_gemm_forward(&runtime, x.buffer(), w.buffer(), m, k, n)?;

    let y_arc = Arc::new(y_buf);
    let mut output = Tensor::from_buffer(y_arc, &runtime, &[m, n], DType::F32, "matmul_out");
    // Propagate requires_grad: output needs grad if any input needs grad
    if x.requires_grad() || w.requires_grad() {
        output.set_requires_grad(true);
    }

    // Record backward on tape
    if Tape::is_recording() && (x.requires_grad() || w.requires_grad()) {
        let x_id = Some(x.id());
        let w_id = Some(w.id());
        let x_needs = x.requires_grad();
        let w_needs = w.requires_grad();
        let x_buf_saved = x.buffer_arc().clone();
        let w_buf_saved = w.buffer_arc().clone();
        let mm = m; let kk = k; let nn = n;

        let node_id = Tape::record(
            "matmul",
            output.id(),
            vec![x_id, w_id],
            vec![x_needs, w_needs],
            vec![x_buf_saved, w_buf_saved],
            Box::new(move |grad_output, saved, runtime| {
                let mut grads = Vec::new();

                // dX = dY @ WT
                if x_needs {
                    let dx = gemm_backward_data(runtime, grad_output, &saved[1], mm, kk, nn)?;
                    grads.push(Some(Arc::new(dx)));
                } else {
                    grads.push(None);
                }

                // dW = X^T @ dY
                if w_needs {
                    let dw = gemm_backward_weight(runtime, &saved[0], grad_output, mm, kk, nn)?;
                    grads.push(Some(Arc::new(dw)));
                } else {
                    grads.push(None);
                }

                Ok(grads)
            }),
        );
        output.set_tape_node(node_id);
    }

    Ok(output)
}

/// MatMul with pre-transposed bf16 weight (for inference/repeated forward).
#[cfg(feature = "rocm")]
pub fn matmul_with_wt_bf16(
    x: &Tensor,
    wt_bf16: &GpuBuffer,
    m: usize, k: usize, n: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(m);
    let m_pad = pad_tile(m, cfg.tile_m);
    let n_pad = pad_tile(n, cfg.tile_n);

    let x_bf16 = f32_to_bf16_gpu_padded(runtime, x.buffer(), m * k, m_pad * k)?;

    let kernel = runtime.ensure_kernel_t0(
        &cfg.name(),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let y_buf = runtime.alloc_f32(m_pad * n_pad)?;
    y_buf.zero();

    let ka = gemm_gen::build_kernargs(
        x_bf16.gpu_addr(), wt_bf16.gpu_addr(), y_buf.gpu_addr(),
        k as u32, n as u32, m as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m as u32, n as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    Ok(Tensor::from_buffer(Arc::new(y_buf), runtime, &[m, n], DType::F32, "matmul_wt_out"))
}

// ── Config selection and padding ──

/// Select tile config based on M dimension
#[cfg(feature = "rocm")]
fn select_config(m: usize) -> crate::t0::gemm_gen::GemmConfig {
    use crate::t0::gemm_gen::GemmConfig;
    if m <= 16 {
        GemmConfig::tile_16x64_k16()
    } else if m <= 32 {
        GemmConfig::tile_32x64_k16()
    } else {
        GemmConfig::tile_64x64_k16()
    }
}

/// Padded size for tile alignment
fn pad_tile(size: usize, tile: u32) -> usize {
    let t = tile as usize;
    (size + t - 1) / t * t
}

// ── Core GEMM dispatch ──

/// Forward GEMM: Y[M,N] = X[M,K] @ W[K,N]
/// Accepts raw f32 buffers, handles bf16 conversion + padding internally.
#[cfg(feature = "rocm")]
fn dispatch_gemm_forward(
    runtime: &Arc<GpuRuntime>,
    x_f32: &GpuBuffer,    // [M, K] f32
    w_f32: &GpuBuffer,    // [K, N] f32
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(m);
    let m_pad = pad_tile(m, cfg.tile_m);
    let n_pad = pad_tile(n, cfg.tile_n);

    let k_pad_contraction = pad_tile(k, cfg.tile_k);

    // Convert A: X[M,K] → bf16 with padding to [m_pad, k_pad_contraction]
    let x_bf16 = f32_to_bf16_gpu_padded(runtime, x_f32, m * k, m_pad * k_pad_contraction)?;
    // Convert B: W[K,N] → transpose → WT[N,K] → bf16 with padding to [n_pad, k_pad_contraction]
    let wt_bf16 = f32_to_bf16_transpose_gpu_padded(runtime, w_f32, k, n, n_pad, k_pad_contraction)?;

    let kernel = runtime.ensure_kernel_t0(
        &cfg.name(),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let y_buf = runtime.alloc_f32(m_pad * n_pad)?;
    y_buf.zero();

    // CRITICAL: pass n_pad as N (output stride) to prevent out-of-bounds column
    // writes from wrapping to the next row when N < tile_n.
    // Pass k_pad_contraction as K to match padded input buffers.
    let ka = gemm_gen::build_kernargs(
        x_bf16.gpu_addr(), wt_bf16.gpu_addr(), y_buf.gpu_addr(),
        k_pad_contraction as u32, n_pad as u32, m as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m as u32, n_pad as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    // Unpad output: copy [m, n] from padded [m_pad, n_pad] with stride=n_pad
    if m == m_pad && n == n_pad {
        Ok(y_buf)
    } else {
        unpad_f32(runtime, &y_buf, m, n, m_pad, n_pad)
    }
}

// ── Backward helpers ──

/// dX = dY @ W  (backward data)
/// dY: [M, N] f32, W: [K, N] f32
/// NT GEMM: A=dY[M,N], B=W[K,N], compute A @ B^T → dX[M,K]
#[cfg(feature = "rocm")]
fn gemm_backward_data(
    runtime: &Arc<GpuRuntime>,
    grad_y: &GpuBuffer,      // [M, N] f32
    w_buf: &Arc<GpuBuffer>,   // [K, N] f32
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(m);
    let m_pad = pad_tile(m, cfg.tile_m);
    let k_pad = pad_tile(k, cfg.tile_n);
    let n_pad_k = pad_tile(n, cfg.tile_k); // pad contraction dim to tile_k

    // A = dY[M,N] → bf16 padded to [m_pad, n_pad_k]
    let dy_bf16 = f32_to_bf16_gpu_padded(runtime, grad_y, m * n, m_pad * n_pad_k)?;
    // B = W[K,N] → bf16 padded to [k_pad, n_pad_k] (B will be transposed by WMMA: B^T)
    let w_bf16 = f32_to_bf16_gpu_padded(runtime, w_buf, k * n, k_pad * n_pad_k)?;

    let kernel = runtime.ensure_kernel_t0(
        &format!("gemm_bwd_data_{}", cfg.name()),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let dx_buf = runtime.alloc_f32(m_pad * k_pad)?;
    dx_buf.zero();

    // Use k_pad as N (output stride) to prevent column wraparound
    // Use n_pad_k as K (contraction dim) since inputs are padded to tile_k
    let ka = gemm_gen::build_kernargs(
        dy_bf16.gpu_addr(), w_bf16.gpu_addr(), dx_buf.gpu_addr(),
        n_pad_k as u32, k_pad as u32, m as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m as u32, k_pad as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    // Unpad output: copy [m, k] from padded [m_pad, k_pad]
    if m == m_pad && k == k_pad {
        Ok(dx_buf)
    } else {
        unpad_f32(runtime, &dx_buf, m, k, m_pad, k_pad)
    }
}

/// dW = X^T @ dY  (backward weight)
/// X: [M, K] f32, dY: [M, N] f32 → dW: [K, N]
/// Steps: transpose X→X_T[K,M], transpose dY→dY_T[N,M]
/// NT GEMM: A=X_T[K,M], B=dY_T[N,M] → A @ B^T = dW[K,N]
#[cfg(feature = "rocm")]
fn gemm_backward_weight(
    runtime: &Arc<GpuRuntime>,
    x_buf: &Arc<GpuBuffer>,  // [M, K] f32
    grad_y: &GpuBuffer,      // [M, N] f32
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(k);
    let k_pad = pad_tile(k, cfg.tile_m);
    let n_pad = pad_tile(n, cfg.tile_n);
    let m_pad_k = pad_tile(m, cfg.tile_k); // pad contraction dim to tile_k

    // A = X_T[K,M] bf16 with padding to [k_pad, m_pad_k]
    let xt_bf16 = f32_to_bf16_transpose_gpu_padded(runtime, x_buf, m, k, k_pad, m_pad_k)?;
    // B = dY_T[N,M] bf16 with padding to [n_pad, m_pad_k]
    let dyt_bf16 = f32_to_bf16_transpose_gpu_padded(runtime, grad_y, m, n, n_pad, m_pad_k)?;

    let kernel = runtime.ensure_kernel_t0(
        &format!("gemm_bwd_wt_{}", cfg.name()),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let dw_buf = runtime.alloc_f32(k_pad * n_pad)?;
    dw_buf.zero();

    // Use n_pad as N (output stride) to prevent column wraparound
    // Use m_pad_k as K (contraction dim) since inputs are padded to tile_k
    let ka = gemm_gen::build_kernargs(
        xt_bf16.gpu_addr(), dyt_bf16.gpu_addr(), dw_buf.gpu_addr(),
        m_pad_k as u32, n_pad as u32, k as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, k as u32, n_pad as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    // Unpad output: copy [k, n] from padded [k_pad, n_pad]
    if k == k_pad && n == n_pad {
        Ok(dw_buf)
    } else {
        unpad_f32(runtime, &dw_buf, k, n, k_pad, n_pad)
    }
}

// ── Output unpadding helper ──

/// Strip padding from a GEMM output buffer (GPU kernel).
///
/// The GEMM kernel writes to a padded [m_pad, n_pad] buffer (stride=n_pad).
/// This copies only the valid [m, n] portion into a contiguous buffer (stride=n).
/// Uses t0_unpad_2d GPU kernel — no CPU roundtrip.
#[cfg(feature = "rocm")]
fn unpad_f32(
    runtime: &Arc<GpuRuntime>,
    padded: &GpuBuffer,
    m: usize, n: usize,
    _m_pad: usize, n_pad: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::math;

    let kernel = runtime.ensure_kernel_t0(
        "t0_unpad_2d",
        || math::t0_unpad_2d(),
        [32, 1, 1],
        0,
    )?;

    let out_buf = runtime.alloc_f32(m * n)?;

    let ka = crate::kernargs![
        padded.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        m as u32 => u32,
        n as u32 => u32,
        n_pad as u32 => u32,
        0u32 => u32  // padding to 32 bytes
    ];

    let total = m * n;
    let grid_x = ((total as u32 + 31) / 32) * 32;
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    Ok(out_buf)
}

// ── BF16 conversion helpers with padding ──

/// Convert f32 → bf16 on CPU, with padded output buffer.
/// Converts `n_real` elements from src GPU buffer, allocates bf16 buffer
/// for `n_padded` elements (extra elements = 0).
#[cfg(feature = "rocm")]
fn f32_to_bf16_gpu_padded(
    runtime: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    n_real: usize,
    n_padded: usize,
) -> Result<GpuBuffer, String> {
    // Read f32 data from GPU
    let mut f32_data = vec![0f32; n_real];
    src.read(unsafe {
        std::slice::from_raw_parts_mut(f32_data.as_mut_ptr() as *mut u8, n_real * 4)
    });

    // Convert to bf16 with padding
    let mut bf16_data = vec![0u16; n_padded]; // zeros for padding
    for i in 0..n_real {
        let bits = f32_data[i].to_bits();
        bf16_data[i] = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    }

    let bf16_bytes = n_padded * 2;
    let alloc_bytes = (bf16_bytes + 255) & !255;
    let dst = runtime.alloc(alloc_bytes)?;
    dst.write(unsafe {
        std::slice::from_raw_parts(bf16_data.as_ptr() as *const u8, bf16_bytes)
    });

    Ok(dst)
}

/// Convert f32 → bf16, transposed: [rows, cols] → [cols_padded, rows_padded] bf16.
/// Pads output to `cols_padded` rows and `rows_padded` columns.
/// Padding values are zero (bf16 0x0000).
#[cfg(feature = "rocm")]
fn f32_to_bf16_transpose_gpu_padded(
    runtime: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    rows: usize,           // original rows (becomes cols after transpose)
    cols: usize,            // original cols (becomes rows after transpose)
    cols_padded: usize,     // padded cols (= padded rows of transposed result)
    rows_padded: usize,     // padded rows (= padded cols of transposed result = K dimension)
) -> Result<GpuBuffer, String> {
    // CPU path: read f32, transpose, convert to bf16
    let n = rows * cols;
    let mut f32_data = vec![0f32; n];
    src.read(unsafe {
        std::slice::from_raw_parts_mut(f32_data.as_mut_ptr() as *mut u8, n * 4)
    });

    // Transpose and convert to bf16, with padding on both dimensions
    let n_padded = cols_padded * rows_padded;
    let mut bf16_data = vec![0u16; n_padded]; // zeros for padding
    for r in 0..rows {
        for c in 0..cols {
            let val = f32_data[r * cols + c];
            let bits = val.to_bits();
            let bf16 = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
            bf16_data[c * rows_padded + r] = bf16; // transposed: [c, r] with stride=rows_padded
        }
    }

    let bf16_bytes = ((n_padded * 2) + 255) & !255;
    let dst = runtime.alloc(bf16_bytes)?;
    dst.write(unsafe {
        std::slice::from_raw_parts(bf16_data.as_ptr() as *const u8, n_padded * 2)
    });

    Ok(dst)
}
