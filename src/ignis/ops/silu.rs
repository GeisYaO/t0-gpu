//! SiLU-gated activation: output = SiLU(gate) * up
//!
//! Forward: out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//! Backward: Complex chain rule through sigmoid and product
//!
//! Uses build_silu_mul(epl) ISA kernel.
//! Kernarg: [gate_ptr: u64, up_ptr: u64, out_ptr: u64]

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::gpu_backend::{GpuBuffer, GpuDevice};
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::tape::Tape;

/// Fused SiLU-gate multiplication: output = silu(gate) * up
///
/// This is the FFN computation: gate_proj → SiLU → elementwise_mul(up_proj)
///
/// # Arguments
/// - `gate`: gate projection output [batch*seq, ffn_dim]
/// - `up`: up projection output [batch*seq, ffn_dim]
#[cfg(feature = "rocm")]
pub fn silu_gate(gate: &Tensor, up: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    assert_eq!(gate.shape(), up.shape(), "silu_gate: shape mismatch");
    let n = gate.numel();
    let runtime = gate.runtime().clone();

    // Build kernel via block_dsl: out = silu(gate) * up
    let kernel = {
        let cached = runtime.get_kernel("bdsl_silu_gate");
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new("bdsl_silu_gate", 256);
            let gate_ptr = kb.arg_ptr("gate");
            let up_ptr = kb.arg_ptr("up");
            let out_ptr = kb.arg_ptr("out");
            let n_arg = kb.arg_u32("n");

            let pid = kb.program_id(0);
            let bs = kb.const_u32(256);
            let base = pid.mul(&mut kb, bs);
            let tid = kb.arange(0, 256);
            let off = tid.add(&mut kb, base);

            let g = kb.load_checked(gate_ptr, off, n_arg);
            let u = kb.load_checked(up_ptr, off, n_arg);
            let result = g.silu(&mut kb).mul(&mut kb, u); // silu(gate) * up
            kb.store_checked(out_ptr, off, result, n_arg);

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    let out_buf = runtime.alloc_f32(n)?;

    let grid_x = ((n as u32 + 255) / 256) * 256;
    let ka = crate::kernargs![
        gate.gpu_addr() => u64,
        up.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        n as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    let output = Tensor::from_buffer(out_arc, &runtime, gate.shape(), DType::F32, "silu_out");

    if Tape::is_recording() && (gate.requires_grad() || up.requires_grad()) {
        let gate_id = Some(gate.id());
        let up_id = Some(up.id());
        let gate_needs = gate.requires_grad();
        let up_needs = up.requires_grad();
        let gate_buf = gate.buffer_arc().clone();
        let up_buf = up.buffer_arc().clone();
        let num = n;

        let node_id = Tape::record(
            "silu_gate",
            output.id(),
            vec![gate_id, up_id],
            vec![gate_needs, up_needs],
            vec![gate_buf, up_buf],
            Box::new(move |grad_output, saved, runtime| {
                // GPU backward via block_dsl: d_up = grad * silu(gate), d_gate = grad * up * silu'(gate)
                let ne = num;
                let kernel = {
                    let cached = runtime.get_kernel("bdsl_silu_bwd");
                    if let Some(k) = cached {
                        k
                    } else {
                        use crate::t0::block_dsl::BlockKernel;
                        use crate::t0::ir::Target;
                        let mut kb = BlockKernel::new("bdsl_silu_bwd", 256);
                        let grad_ptr = kb.arg_ptr("grad");
                        let gate_ptr = kb.arg_ptr("gate");
                        let up_ptr = kb.arg_ptr("up");
                        let dgate_ptr = kb.arg_ptr("dgate");
                        let dup_ptr = kb.arg_ptr("dup");
                        let n_arg = kb.arg_u32("n");

                        let pid = kb.program_id(0);
                        let bs = kb.const_u32(256);
                        let base = pid.mul(&mut kb, bs);
                        let tid = kb.arange(0, 256);
                        let off = tid.add(&mut kb, base);

                        let g = kb.load_checked(grad_ptr, off, n_arg);
                        let gate = kb.load_checked(gate_ptr, off, n_arg);
                        let up = kb.load_checked(up_ptr, off, n_arg);

                        // silu(gate) = gate * σ(gate)
                        let sig = gate.sigmoid(&mut kb);
                        let silu_val = gate.mul(&mut kb, sig);

                        // d_up = grad * silu(gate)
                        let d_up = g.mul(&mut kb, silu_val);

                        // d_gate = grad * up * σ(gate) * (1 + gate * (1 - σ(gate)))
                        let one = kb.const_f32(1.0);
                        let one_minus_sig = one.sub(&mut kb, sig);
                        let gate_term = gate.mul(&mut kb, one_minus_sig);
                        let bracket = one.add(&mut kb, gate_term);  // 1 + gate*(1-σ)
                        let d_gate = g.mul(&mut kb, up).mul(&mut kb, sig).mul(&mut kb, bracket);

                        kb.store_checked(dgate_ptr, off, d_gate, n_arg);
                        kb.store_checked(dup_ptr, off, d_up, n_arg);

                        let compiled = kb.compile(Target::GFX1100)?;
                        runtime.compile_dsl(compiled)?
                    }
                };

                let dgate_buf = runtime.alloc_f32(ne)?;
                let dup_buf = runtime.alloc_f32(ne)?;

                let grid_x = ((ne as u32 + 255) / 256) * 256;
                let ka = crate::kernargs![
                    grad_output.gpu_addr() => u64,
                    saved[0].gpu_addr() => u64,
                    saved[1].gpu_addr() => u64,
                    dgate_buf.gpu_addr() => u64,
                    dup_buf.gpu_addr() => u64,
                    ne as u32 => u32
                ];
                runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

                let mut grads = Vec::new();
                if gate_needs {
                    grads.push(Some(Arc::new(dgate_buf)));
                } else {
                    grads.push(None);
                }
                if up_needs {
                    grads.push(Some(Arc::new(dup_buf)));
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
