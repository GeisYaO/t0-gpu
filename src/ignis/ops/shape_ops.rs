//! Shape ops — reshape, transpose, slice, neg, sub, mean, relu, softmax
//!
//! Most are zero-copy or simple element-wise with tape recording.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::gpu_backend::{GpuBuffer, GpuDevice};
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::tape::Tape;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

// ── Reshape (zero-copy) ──

/// Reshape tensor (zero-copy if same numel).
/// Backward: reshape grad back to original shape.
#[cfg(feature = "rocm")]
pub fn reshape(a: &Tensor, new_shape: &[usize], _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let old_numel: usize = a.shape().iter().product();
    let new_numel: usize = new_shape.iter().product();
    assert_eq!(old_numel, new_numel, "reshape: numel mismatch {} vs {}", old_numel, new_numel);

    // Zero-copy: share the same buffer
    let output = Tensor::from_buffer(
        a.buffer_arc().clone(), a.runtime(), new_shape, a.dtype(), "reshape_out",
    );

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let _old_shape = a.shape().to_vec();
        let node_id = Tape::record(
            "reshape", output.id(),
            vec![a_id], vec![true], vec![],
            Box::new(move |grad_output, _saved, _runtime| {
                // Grad is the same buffer, just different shape interpretation
                Ok(vec![Some(Arc::new(clone_buf(grad_output, _runtime)?))])
            }),
        );
        output.set_tape_node(node_id);
    }
    Ok(output)
}

// ── Transpose 2D ──

/// Transpose 2D: [M, N] → [N, M]
/// Backward: transpose grad back
#[cfg(feature = "rocm")]
pub fn transpose(a: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let shape = a.shape();
    assert_eq!(shape.len(), 2, "transpose: need 2D tensor, got {:?}", shape);
    let (m, n) = (shape[0], shape[1]);
    let runtime = a.runtime().clone();

    let data = a.to_f32_vec();
    let mut t_data = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            t_data[c * m + r] = data[r * n + c];
        }
    }

    let output = Tensor::from_f32(&runtime, &t_data, &[n, m], "transpose_out")?;

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let node_id = Tape::record(
            "transpose", output.id(),
            vec![a_id], vec![true], vec![],
            Box::new(move |grad_output, _saved, runtime| {
                // Transpose the gradient back: [N,M] → [M,N]
                let gdata = read_f32(grad_output, n * m);
                let mut dt = vec![0f32; m * n];
                for r in 0..n {
                    for c in 0..m {
                        dt[c * n + r] = gdata[r * m + c];
                    }
                }
                let buf = runtime.alloc_f32(m * n)?;
                write_f32(&buf, &dt);
                Ok(vec![Some(Arc::new(buf))])
            }),
        );
        output.set_tape_node(node_id);
    }
    Ok(output)
}

// ── Slice rows ──

/// Slice rows: output = a[start..end, :]
#[cfg(feature = "rocm")]
pub fn slice_rows(a: &Tensor, start: usize, end: usize, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let shape = a.shape();
    assert!(shape.len() >= 2, "slice: need at least 2D");
    let cols = shape[shape.len() - 1];
    let total_rows: usize = a.numel() / cols;
    assert!(start <= end && end <= total_rows, "slice: invalid range {}..{} for {} rows", start, end, total_rows);

    let runtime = a.runtime().clone();
    let n_rows = end - start;

    // Copy the slice
    let data = a.to_f32_vec();
    let slice_data: Vec<f32> = data[start * cols..end * cols].to_vec();
    let mut new_shape = shape.to_vec();
    new_shape[0] = n_rows;

    let output = Tensor::from_f32(&runtime, &slice_data, &new_shape, "slice_out")?;

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let full_n = a.numel();
        let s = start;
        let c = cols;
        let nr = n_rows;

        let node_id = Tape::record(
            "slice", output.id(),
            vec![a_id], vec![true], vec![],
            Box::new(move |grad_output, _saved, runtime| {
                // Backward: scatter grad into a zero buffer at the sliced positions
                let mut grad_full = vec![0f32; full_n];
                let grad_slice = read_f32(grad_output, nr * c);
                grad_full[s * c..(s + nr) * c].copy_from_slice(&grad_slice);
                let buf = runtime.alloc_f32(full_n)?;
                write_f32(&buf, &grad_full);
                Ok(vec![Some(Arc::new(buf))])
            }),
        );
        output.set_tape_node(node_id);
    }
    Ok(output)
}

// ── Negation ──

/// Negate: output = -a
#[cfg(feature = "rocm")]
pub fn neg(a: &Tensor, device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    super::add::scale(a, -1.0, device)
}

// ── Subtraction ──

/// Subtract: output = a - b
#[cfg(feature = "rocm")]
pub fn sub(a: &Tensor, b: &Tensor, device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let neg_b = super::add::scale(b, -1.0, device)?;
    super::add::add(a, &neg_b, device)
}

// ── Mean ──

/// Mean of all elements → scalar
#[cfg(feature = "rocm")]
pub fn mean(a: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let n = a.numel();
    let runtime = a.runtime().clone();

    let data = a.to_f32_vec();
    let avg: f32 = data.iter().sum::<f32>() / n as f32;

    let output = Tensor::from_f32(&runtime, &[avg], &[1], "mean_out")?;

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let num = n;

        let node_id = Tape::record(
            "mean", output.id(),
            vec![a_id], vec![true], vec![],
            Box::new(move |grad_output, _saved, runtime| {
                // d(mean)/da[i] = 1/n
                let g = read_f32(grad_output, 1)[0];
                let val = g / num as f32;
                let grad_data = vec![val; num];
                let buf = runtime.alloc_f32(num)?;
                write_f32(&buf, &grad_data);
                Ok(vec![Some(Arc::new(buf))])
            }),
        );
        output.set_tape_node(node_id);
    }
    Ok(output)
}

// ── ReLU ──

/// ReLU: output[i] = max(0, a[i])
#[cfg(feature = "rocm")]
pub fn relu(a: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let n = a.numel();
    let runtime = a.runtime().clone();

    // Build kernel via block_dsl: out = max(0, x)
    let kernel = {
        let cached = runtime.get_kernel("bdsl_relu");
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new("bdsl_relu", 256);
            let a_ptr = kb.arg_ptr("a");
            let out_ptr = kb.arg_ptr("out");
            let n_arg = kb.arg_u32("n");

            let pid = kb.program_id(0);
            let bs = kb.const_u32(256);
            let base = pid.mul(&mut kb, bs);
            let tid = kb.arange(0, 256);
            let off = tid.add(&mut kb, base);

            let x = kb.load_checked(a_ptr, off, n_arg);
            let result = x.relu(&mut kb);
            kb.store_checked(out_ptr, off, result, n_arg);

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    let out_buf = runtime.alloc_f32(n)?;
    let grid_x = ((n as u32 + 255) / 256) * 256;
    let ka = crate::kernargs![
        a.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        n as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let output = Tensor::from_buffer(Arc::new(out_buf), &runtime, a.shape(), super::super::tensor::DType::F32, "relu_out");

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let a_buf = a.buffer_arc().clone();
        let num = n;

        let node_id = Tape::record(
            "relu", output.id(),
            vec![a_id], vec![true],
            vec![a_buf], // save input for backward mask
            Box::new(move |grad_output, saved, runtime| {
                // d(relu)/da = 1 if a > 0, else 0
                let a_data = read_f32(&saved[0], num);
                let g_data = read_f32(grad_output, num);
                let grad: Vec<f32> = a_data.iter().zip(g_data.iter())
                    .map(|(&a, &g)| if a > 0.0 { g } else { 0.0 })
                    .collect();
                let buf = runtime.alloc_f32(num)?;
                write_f32(&buf, &grad);
                Ok(vec![Some(Arc::new(buf))])
            }),
        );
        output.set_tape_node(node_id);
    }
    Ok(output)
}

// ── Softmax ──

/// Softmax along last dimension.
/// Backward: d_input = softmax * (d_output - sum(d_output * softmax))
#[cfg(feature = "rocm")]
pub fn softmax(a: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let shape = a.shape().to_vec();
    let dim = *shape.last().unwrap();
    let rows = a.numel() / dim;
    let runtime = a.runtime().clone();

    let data = a.to_f32_vec();
    let mut out_data = vec![0f32; rows * dim];

    for r in 0..rows {
        let row = &data[r * dim..(r + 1) * dim];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0f32;
        for c in 0..dim {
            let e = (row[c] - max_val).exp();
            out_data[r * dim + c] = e;
            sum_exp += e;
        }
        for c in 0..dim {
            out_data[r * dim + c] /= sum_exp;
        }
    }

    let output = Tensor::from_f32(&runtime, &out_data, &shape, "softmax_out")?;

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let out_buf = output.buffer_arc().clone();
        let d = dim;
        let r = rows;

        let node_id = Tape::record(
            "softmax", output.id(),
            vec![a_id], vec![true],
            vec![out_buf], // save softmax output for backward
            Box::new(move |grad_output, saved, runtime| {
                let s_data = read_f32(&saved[0], r * d);
                let g_data = read_f32(grad_output, r * d);
                let mut dx = vec![0f32; r * d];

                for row in 0..r {
                    // dot = sum(grad * softmax) for this row
                    let mut dot = 0f32;
                    for c in 0..d {
                        dot += g_data[row * d + c] * s_data[row * d + c];
                    }
                    for c in 0..d {
                        dx[row * d + c] = s_data[row * d + c] * (g_data[row * d + c] - dot);
                    }
                }
                let buf = runtime.alloc_f32(r * d)?;
                write_f32(&buf, &dx);
                Ok(vec![Some(Arc::new(buf))])
            }),
        );
        output.set_tape_node(node_id);
    }
    Ok(output)
}

// ── Helpers ──

#[cfg(feature = "rocm")]
fn clone_buf(src: &GpuBuffer, runtime: &Arc<GpuRuntime>) -> Result<GpuBuffer, String> {
    let dst = runtime.alloc(src.size)?;
    let mut tmp = vec![0u8; src.size];
    src.read(&mut tmp);
    dst.write(&tmp);
    Ok(dst)
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
