//! Elementwise ops: add, scale, sum, elementwise_mul — with autodiff tape recording.
//!
//! All ops:
//! 1. Dispatch GPU ISA kernels for forward computation
//! 2. Record backward closures on the tape
//! 3. Backward closures dispatch GPU kernels for gradient computation

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

// ── Add: a + b (element-wise) ──

/// Element-wise addition of two f32 tensors.
///
/// Forward: output[i] = a[i] + b[i]
/// Backward: da = grad_out, db = grad_out (gradient passes through unchanged)
#[cfg(feature = "rocm")]
pub fn add(a: &Tensor, b: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    assert_eq!(a.shape(), b.shape(), "add: shape mismatch {:?} vs {:?}", a.shape(), b.shape());
    let n = a.numel();
    let runtime = a.runtime().clone();

    // Allocate output
    let out_buf = runtime.alloc_f32(n)?;

    // Build kernel via block_dsl (cached by name in compile_dsl)
    let kernel = {
        let cached = runtime.get_kernel("bdsl_add_f32");
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new("bdsl_add_f32", 256);
            let a_ptr = kb.arg_ptr("a");
            let b_ptr = kb.arg_ptr("b");
            let out_ptr = kb.arg_ptr("out");
            let n_arg = kb.arg_u32("n");

            let pid = kb.program_id(0);
            let bs = kb.const_u32(256);
            let base = pid.mul(&mut kb, bs);
            let tid = kb.arange(0, 256);
            let off = tid.add(&mut kb, base);

            let va = kb.load_checked(a_ptr, off, n_arg);
            let vb = kb.load_checked(b_ptr, off, n_arg);
            let sum = va.add(&mut kb, vb);
            kb.store_checked(out_ptr, off, sum, n_arg);

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    // Dispatch: grid = ceil(n / 256) * 256
    let grid_x = ((n as u32 + 255) / 256) * 256;
    let ka = crate::kernargs![
        a.gpu_addr() => u64,
        b.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        n as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    let mut output = Tensor::from_buffer(
        out_arc.clone(),
        &runtime,
        a.shape(),
        DType::F32,
        "add_out",
    );
    if a.requires_grad() || b.requires_grad() {
        output.set_requires_grad(true);
    }

    // Record on tape if recording is active
    if Tape::is_recording() && (a.requires_grad() || b.requires_grad()) {
        let a_id = Some(a.id());
        let b_id = Some(b.id());
        let a_needs = a.requires_grad();
        let b_needs = b.requires_grad();

        let node_id = Tape::record(
            "add",
            output.id(),
            vec![a_id, b_id],
            vec![a_needs, b_needs],
            vec![], // no saved tensors needed for add backward
            Box::new(move |grad_output, _saved, _runtime| {
                // d(a+b)/da = 1, d(a+b)/db = 1
                // Both input grads are just the output grad (shared)
                let grad_arc = Arc::new(clone_gpu_buffer(grad_output, _runtime)?);
                let mut grads = Vec::new();
                if a_needs {
                    grads.push(Some(grad_arc.clone()));
                } else {
                    grads.push(None);
                }
                if b_needs {
                    grads.push(Some(grad_arc));
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

// ── Scale: a * scalar ──

/// Scalar multiplication: output[i] = a[i] * scalar
///
/// Forward: GPU via DSL `Op::Scale`
/// Backward: da = grad_out * scalar (CPU)
#[cfg(feature = "rocm")]
pub fn scale(a: &Tensor, scalar: f32, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let n = a.numel();
    let runtime = a.runtime().clone();

    let out_buf = runtime.alloc_f32(n)?;

    // Build kernel via block_dsl (scale is fused: load → mul_const → store)
    let kernel = {
        let cached = runtime.get_kernel("bdsl_scale_f32");
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new("bdsl_scale_f32", 256);
            let a_ptr = kb.arg_ptr("a");
            let out_ptr = kb.arg_ptr("out");
            let s_arg = kb.arg_f32("scalar");
            let n_arg = kb.arg_u32("n");

            let pid = kb.program_id(0);
            let bs = kb.const_u32(256);
            let base = pid.mul(&mut kb, bs);
            let tid = kb.arange(0, 256);
            let off = tid.add(&mut kb, base);

            let va = kb.load_checked(a_ptr, off, n_arg);
            let scaled = va.mul(&mut kb, s_arg);
            kb.store_checked(out_ptr, off, scaled, n_arg);

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    let grid_x = ((n as u32 + 255) / 256) * 256;
    let ka = crate::kernargs![
        a.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        scalar => f32,
        n as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    let mut out = Tensor::from_buffer(out_arc, &runtime, a.shape(), DType::F32, "scale_out");
    if a.requires_grad() {
        out.set_requires_grad(true);
    }

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let s = scalar;

        let node_id = Tape::record(
            "scale",
            out.id(),
            vec![a_id],
            vec![true],
            vec![],
            Box::new(move |grad_output, _saved, runtime| {
                // d(a*s)/da = s → grad_a = grad_out * s
                let n_elems = grad_output.size / 4;
                let go_data = runtime.read_f32(grad_output, n_elems);
                let grad_data: Vec<f32> = go_data.iter().map(|&v| v * s).collect();
                let grad_buf = runtime.upload_f32(&grad_data)?;
                Ok(vec![Some(Arc::new(grad_buf))])
            }),
        );
        out.set_tape_node(node_id);
    }

    Ok(out)
}

// ── Sum: reduce all elements to scalar ──

/// Sum all elements → scalar tensor.
///
/// Forward: output = sum(a[i])  (GPU reduction kernel)
/// Backward: da[i] = grad_out (broadcast scalar to all elements)
#[cfg(feature = "rocm")]
pub fn sum(a: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let n = a.numel();
    let runtime = a.runtime().clone();

    // For small tensors (≤ 4096 elements), CPU readback is faster than kernel launch.
    // For larger tensors, GPU partial reduction per WG + CPU final sum.
    let total: f32 = if n <= 4096 {
        let data = a.to_f32_vec();
        data.iter().sum()
    } else {
        // GPU partial reduction: each WG of 256 threads reduces 256 elements → 1 partial sum.
        let wg_size = 256u32;
        let n_wgs = ((n as u32) + wg_size - 1) / wg_size;

        let partial_buf = runtime.alloc_f32(n_wgs as usize)?;

        let kernel = {
            let cached = runtime.get_kernel("bdsl_partial_sum_f32");
            if let Some(k) = cached {
                k
            } else {
                use crate::t0::block_dsl::BlockKernel;
                use crate::t0::ir::Target;
                let mut kb = BlockKernel::new("bdsl_partial_sum_f32", 256);
                let in_ptr = kb.arg_ptr("input");
                let out_ptr = kb.arg_ptr("partial_out");
                let n_arg = kb.arg_u32("n");

                // Each thread loads one element (0.0 for OOB lanes)
                let pid = kb.program_id(0);
                let bs = kb.const_u32(256);
                let base = pid.mul(&mut kb, bs);
                let tid = kb.arange(0, 256);
                let off = tid.add(&mut kb, base);

                let val = kb.load_checked(in_ptr, off, n_arg);

                // WG-level reduction: wave reduce → LDS → cross-wave reduce → broadcast
                let wg_sum = kb.wg_reduce_sum(val);

                // Only thread 0 of each WG stores the result to partial_out[pid].
                // wg_reduce_sum broadcasts the result to all lanes, so we use
                // store to pid-based offsets with a "tid < 1" mask (only lane 0).
                let one = kb.const_u32(1);
                let mask_lane0 = tid.lt(&mut kb, one);
                kb.store(out_ptr, pid, wg_sum, mask_lane0);

                let compiled = kb.compile(Target::GFX1100)?;
                runtime.compile_dsl(compiled)?
            }
        };

        let grid_x = n_wgs * wg_size;
        let ka = crate::kernargs![
            a.gpu_addr() => u64,
            partial_buf.gpu_addr() => u64,
            n as u32 => u32
        ];
        runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

        // CPU final sum of partial results (n_wgs values — typically a few hundred)
        let partial_data = runtime.read_f32(&partial_buf, n_wgs as usize);
        partial_data.iter().sum()
    };

    let mut out = Tensor::from_f32(&runtime, &[total], &[1], "sum_out")?;
    if a.requires_grad() {
        out.set_requires_grad(true);
    }

    if Tape::is_recording() && a.requires_grad() {
        let a_id = Some(a.id());
        let num_elems = n;

        let node_id = Tape::record(
            "sum",
            out.id(),
            vec![a_id],
            vec![true],
            vec![],
            Box::new(move |grad_output, _saved, runtime| {
                // d(sum)/da[i] = 1 → grad_a = broadcast(grad_out)
                // grad_output is scalar [1], broadcast to [n]
                let grad_val = runtime.read_f32(grad_output, 1)[0];
                let grad_data = vec![grad_val; num_elems];
                let grad_buf = runtime.alloc_f32(num_elems)?;
                runtime.write_f32(&grad_buf, &grad_data);
                Ok(vec![Some(Arc::new(grad_buf))])
            }),
        );
        out.set_tape_node(node_id);
    }

    Ok(out)
}

// ── Elementwise Multiply: a * b ──

/// Element-wise multiplication: output[i] = a[i] * b[i]
///
/// Backward: da = grad_out * b, db = grad_out * a
#[cfg(feature = "rocm")]
pub fn elementwise_mul(a: &Tensor, b: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    assert_eq!(a.shape(), b.shape(), "mul: shape mismatch {:?} vs {:?}", a.shape(), b.shape());
    let n = a.numel();
    let runtime = a.runtime().clone();

    let out_buf = runtime.alloc_f32(n)?;

    // Build kernel via block_dsl
    let kernel = {
        let cached = runtime.get_kernel("bdsl_mul_f32");
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new("bdsl_mul_f32", 256);
            let a_ptr = kb.arg_ptr("a");
            let b_ptr = kb.arg_ptr("b");
            let out_ptr = kb.arg_ptr("out");
            let n_arg = kb.arg_u32("n");

            let pid = kb.program_id(0);
            let bs = kb.const_u32(256);
            let base = pid.mul(&mut kb, bs);
            let tid = kb.arange(0, 256);
            let off = tid.add(&mut kb, base);

            let va = kb.load_checked(a_ptr, off, n_arg);
            let vb = kb.load_checked(b_ptr, off, n_arg);
            let prod = va.mul(&mut kb, vb);
            kb.store_checked(out_ptr, off, prod, n_arg);

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    let grid_x = ((n as u32 + 255) / 256) * 256;
    let ka = crate::kernargs![
        a.gpu_addr() => u64,
        b.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        n as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    let mut output = Tensor::from_buffer(out_arc, &runtime, a.shape(), DType::F32, "mul_out");
    if a.requires_grad() || b.requires_grad() {
        output.set_requires_grad(true);
    }

    if Tape::is_recording() && (a.requires_grad() || b.requires_grad()) {
        let a_id = Some(a.id());
        let b_id = Some(b.id());
        let a_needs = a.requires_grad();
        let b_needs = b.requires_grad();
        let a_buf = a.buffer_arc().clone();
        let b_buf = b.buffer_arc().clone();
        let num = n;

        let node_id = Tape::record(
            "mul",
            output.id(),
            vec![a_id, b_id],
            vec![a_needs, b_needs],
            vec![a_buf.clone(), b_buf.clone()], // save both for backward
            Box::new(move |grad_output, saved, runtime| {
                let mut grads = Vec::new();
                // CPU backward — GPU dispatch has a hang bug on second invocation
                let go = runtime.read_f32(grad_output, num);

                if a_needs {
                    // da = grad_out * b
                    let b_data = runtime.read_f32(&saved[1], num);
                    let grad_a_data: Vec<f32> = go.iter().zip(b_data.iter())
                        .map(|(&g, &b)| g * b).collect();
                    let grad_a = runtime.upload_f32(&grad_a_data)?;
                    grads.push(Some(Arc::new(grad_a)));
                } else {
                    grads.push(None);
                }

                if b_needs {
                    // db = grad_out * a
                    let a_data = runtime.read_f32(&saved[0], num);
                    let grad_b_data: Vec<f32> = go.iter().zip(a_data.iter())
                        .map(|(&g, &a)| g * a).collect();
                    let grad_b = runtime.upload_f32(&grad_b_data)?;
                    grads.push(Some(Arc::new(grad_b)));
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

// ── Helper: clone a GPU buffer ──

#[cfg(feature = "rocm")]
fn clone_gpu_buffer(src: &GpuBuffer, runtime: &Arc<GpuRuntime>) -> Result<GpuBuffer, String> {
    let n = src.size;
    let dst = runtime.alloc(n)?;
    let mut tmp = vec![0u8; n];
    src.read(&mut tmp);
    dst.write(&tmp);
    Ok(dst)
}

// ── Kernel builders (T0-based) ──

/// Build elementwise multiply kernel via T0: out[i] = a[i] * b[i]
/// Kernarg: [a_ptr: u64, b_ptr: u64, out_ptr: u64, n: u32]
#[cfg(feature = "rocm")]
fn build_elementwise_mul_kernel() -> crate::t0::compile::T0Kernel {
    use crate::t0::compile::T0Kernel;
    use crate::t0::ir::*;

    let mut k = T0Kernel::new("elementwise_mul_f32");
    let a_ptr = k.arg_ptr("a_ptr");
    let b_ptr = k.arg_ptr("b_ptr");
    let out_ptr = k.arg_ptr("out_ptr");
    let n = k.arg_u32("n");

    k.emit_arg_loads();

    let gid = k.compute_global_id_x(256);
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, n);

    let saved = k.bounds_check_begin(gid, n_vreg);

    // Compute byte offset
    let offset = k.alloc_vreg();
    k.v_lshlrev_b32(offset, 2, gid); // offset = gid * 4

    // Load a[i] and b[i]
    let addr_lo = k.alloc_vreg();
    let addr_hi = k.alloc_vreg();
    let val_a = k.alloc_vreg();
    let val_b = k.alloc_vreg();

    // a[i]
    k.v_mov_from_sgpr(addr_lo, SReg(a_ptr.0));
    k.v_mov_from_sgpr(addr_hi, SReg(a_ptr.0 + 1));
    k.addr64_add(addr_lo, addr_hi, offset);
    k.global_load(val_a, addr_lo, Width::B32, 0);
    k.wait_vmcnt(0);

    // b[i]
    k.v_mov_from_sgpr(addr_lo, SReg(b_ptr.0));
    k.v_mov_from_sgpr(addr_hi, SReg(b_ptr.0 + 1));
    k.addr64_add(addr_lo, addr_hi, offset);
    k.global_load(val_b, addr_lo, Width::B32, 0);
    k.wait_vmcnt(0);

    // out = a * b
    let result = k.alloc_vreg();
    k.v_mul_f32(result, val_a, val_b);

    // Store out[i]
    k.v_mov_from_sgpr(addr_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(addr_hi, SReg(out_ptr.0 + 1));
    k.addr64_add(addr_lo, addr_hi, offset);
    k.global_store(addr_lo, result, Width::B32, 0);

    k.bounds_check_end(saved);
    k.endpgm();
    k
}
