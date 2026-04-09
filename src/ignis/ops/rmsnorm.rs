//! RMSNorm op — forward and backward with autodiff tape.
//!
//! RMSNorm(x, γ) = x / RMS(x) * γ
//! where RMS(x) = sqrt(mean(x²) + ε)
//!
//! Forward uses GPU ISA kernel when available, CPU fallback otherwise.
//! Backward computes gradients for both x and γ.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::gpu_backend::{GpuBuffer, GpuDevice};
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::tape::Tape;

const EPSILON: f32 = 1e-5;

/// RMSNorm forward: y = (x / rms(x)) * gamma
///
/// - x: [rows, dim] f32
/// - gamma: [dim] f32 (per-channel scale)
/// - output: [rows, dim] f32
#[cfg(feature = "rocm")]
pub fn rmsnorm(x: &Tensor, gamma: &Tensor, _device: &Arc<GpuDevice>) -> Result<Tensor, String> {
    let runtime = x.runtime().clone();
    let shape = x.shape().to_vec();
    assert!(shape.len() >= 1, "rmsnorm: need at least 1D");
    let dim = *shape.last().unwrap();
    let rows = x.numel() / dim;

    assert_eq!(gamma.numel(), dim, "rmsnorm: gamma dim mismatch");

    // GPU forward via block_dsl: 1 wave per row
    let out_buf = runtime.alloc_f32(rows * dim)?;
    let rms_buf = runtime.alloc_f32(rows)?; // save inv_rms for backward

    // Build or reuse kernel
    let kernel = {
        let name = format!("bdsl_rmsnorm_d{}", dim);
        let cached = runtime.get_kernel(&name);
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new(&name, 32); // 1 wave per WG
            let x_ptr = kb.arg_ptr("x");
            let gamma_ptr = kb.arg_ptr("gamma");
            let out_ptr = kb.arg_ptr("out");
            let rms_ptr = kb.arg_ptr("rms");
            let dim_arg = kb.arg_u32("dim");

            let row_id = kb.program_id(0);     // 1 WG per row
            let lane_id = kb.thread_id();       // 0..31

            // row_base = row_id * dim (element offset within x/out)
            let row_base = row_id.mul(&mut kb, dim_arg);

            // Pass 1: accumulate sum of x²
            // Each lane handles elements lane_id, lane_id+32, lane_id+64, ...
            let epl = ((dim + 31) / 32) as u32; // ceil(dim/32)
            let mut sum_sq = kb.const_f32(0.0);
            for j in 0..epl {
                let col = if j == 0 {
                    lane_id
                } else {
                    let offset = kb.const_u32(j * 32);
                    lane_id.add(&mut kb, offset)
                };
                let idx = row_base.add(&mut kb, col);
                let mask = col.lt(&mut kb, dim_arg);
                let val = kb.load(x_ptr, idx, mask);
                sum_sq = val.fma(&mut kb, val, sum_sq); // sum_sq += x²
            }

            // Wave reduce sum
            sum_sq = kb.wave_reduce_sum_val(sum_sq);

            // inv_rms = 1/sqrt(sum_sq/dim + eps)
            let dim_f = dim_arg.to_f32(&mut kb);
            let inv_dim = dim_f.rcp(&mut kb);
            let mean_sq = sum_sq.mul(&mut kb, inv_dim);
            let eps = kb.const_f32(EPSILON);
            let rms = mean_sq.add(&mut kb, eps).sqrt(&mut kb);
            let inv_rms = rms.rcp(&mut kb);

            // Save inv_rms (only lane 0 writes)
            let zero_u = kb.const_u32(0);
            let one_u = kb.const_u32(1);
            let lane_mask = lane_id.lt(&mut kb, one_u);
            kb.store(rms_ptr, row_id, inv_rms, lane_mask);

            // Pass 2: normalize and store
            for j in 0..epl {
                let col = if j == 0 {
                    lane_id
                } else {
                    let offset = kb.const_u32(j * 32);
                    lane_id.add(&mut kb, offset)
                };
                let idx = row_base.add(&mut kb, col);
                let mask = col.lt(&mut kb, dim_arg);
                let val = kb.load(x_ptr, idx, mask);
                let g = kb.load(gamma_ptr, col, mask);
                let normed = val.mul(&mut kb, inv_rms).mul(&mut kb, g);
                kb.store(out_ptr, idx, normed, mask);
            }

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    // Dispatch: 1 WG per row, each WG = 32 threads
    let ka = crate::kernargs![
        x.gpu_addr() => u64,
        gamma.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        rms_buf.gpu_addr() => u64,
        dim as u32 => u32
    ];
    runtime.dispatch(&kernel, [rows as u32 * 32, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    let output = Tensor::from_buffer(out_arc, &runtime, &shape, super::super::tensor::DType::F32, "rmsnorm_out");

    if Tape::is_recording() && (x.requires_grad() || gamma.requires_grad()) {
        let x_id = Some(x.id());
        let g_id = Some(gamma.id());
        let x_needs = x.requires_grad();
        let g_needs = gamma.requires_grad();
        let x_buf = x.buffer_arc().clone();
        let g_buf = gamma.buffer_arc().clone();
        let d = dim;
        let r = rows;

        // rms_buf already allocated on GPU — contains inv_rms
        let rms_arc = Arc::new(rms_buf);

        let node_id = Tape::record(
            "rmsnorm",
            output.id(),
            vec![x_id, g_id],
            vec![x_needs, g_needs],
            vec![x_buf, g_buf, rms_arc],
            Box::new(move |grad_output, saved, runtime| {
                let mut grads = Vec::new();

                // ═══ dx: block_dsl GPU kernel (1 wave per row) ═══
                if x_needs {
                    let dx_kernel = {
                        let name = format!("bdsl_rmsnorm_dx_d{}", d);
                        let cached = runtime.get_kernel(&name);
                        if let Some(k) = cached {
                            k
                        } else {
                            use crate::t0::block_dsl::BlockKernel;
                            use crate::t0::ir::Target;
                            let mut kb = BlockKernel::new(&name, 32);
                            let dy_ptr = kb.arg_ptr("dy");
                            let x_ptr = kb.arg_ptr("x");
                            let gamma_ptr = kb.arg_ptr("gamma");
                            let irms_ptr = kb.arg_ptr("inv_rms");
                            let dx_ptr = kb.arg_ptr("dx");
                            let dim_arg = kb.arg_u32("dim");

                            let row_id = kb.program_id(0);
                            let lane_id = kb.thread_id();
                            let row_base = row_id.mul(&mut kb, dim_arg);

                            // Load inv_rms for this row
                            let max_u = kb.const_u32(u32::MAX);
                            let always = row_id.lt(&mut kb, max_u);
                            let inv_rms = kb.load(irms_ptr, row_id, always);
                            let inv_rms3 = inv_rms.mul(&mut kb, inv_rms).mul(&mut kb, inv_rms);

                            // Pass 1: dot_sum = sum(dy * gamma * x)
                            let epl = ((d + 31) / 32) as u32;
                            let mut dot = kb.const_f32(0.0);
                            for j in 0..epl {
                                let col = if j == 0 { lane_id } else {
                                    let o = kb.const_u32(j * 32);
                                    lane_id.add(&mut kb, o)
                                };
                                let mask = col.lt(&mut kb, dim_arg);
                                let idx = row_base.add(&mut kb, col);
                                let dy = kb.load(dy_ptr, idx, mask);
                                let x = kb.load(x_ptr, idx, mask);
                                let g = kb.load(gamma_ptr, col, mask);
                                let term = dy.mul(&mut kb, g).mul(&mut kb, x);
                                dot = dot.add(&mut kb, term);
                            }
                            dot = kb.wave_reduce_sum_val(dot);

                            // Pass 2: dx = dy*gamma*inv_rms - x * dot_sum * inv_rms^3 / dim
                            let dim_f = dim_arg.to_f32(&mut kb);
                            let inv_dim = dim_f.rcp(&mut kb);
                            let scale2 = dot.mul(&mut kb, inv_rms3).mul(&mut kb, inv_dim);
                            for j in 0..epl {
                                let col = if j == 0 { lane_id } else {
                                    let o = kb.const_u32(j * 32);
                                    lane_id.add(&mut kb, o)
                                };
                                let mask = col.lt(&mut kb, dim_arg);
                                let idx = row_base.add(&mut kb, col);
                                let dy = kb.load(dy_ptr, idx, mask);
                                let x = kb.load(x_ptr, idx, mask);
                                let g = kb.load(gamma_ptr, col, mask);
                                // dx = dy*g*inv_rms - x*scale2
                                let term1 = dy.mul(&mut kb, g).mul(&mut kb, inv_rms);
                                let term2 = x.mul(&mut kb, scale2);
                                let result = term1.sub(&mut kb, term2);
                                kb.store(dx_ptr, idx, result, mask);
                            }

                            let compiled = kb.compile(Target::GFX1100)?;
                            runtime.compile_dsl(compiled)?
                        }
                    };

                    let dx_buf = runtime.alloc_f32(r * d)?;
                    let ka = crate::kernargs![
                        grad_output.gpu_addr() => u64,
                        saved[0].gpu_addr() => u64,  // x
                        saved[1].gpu_addr() => u64,  // gamma
                        saved[2].gpu_addr() => u64,  // inv_rms
                        dx_buf.gpu_addr() => u64,
                        d as u32 => u32
                    ];
                    runtime.dispatch(&dx_kernel, [r as u32 * 32, 1, 1], &ka)?;
                    grads.push(Some(Arc::new(dx_buf)));
                } else {
                    grads.push(None);
                }

                // ═══ dgamma: block_dsl GPU kernel (atomic add across rows) ═══
                if g_needs {
                    let dg_kernel = {
                        let name = format!("bdsl_rmsnorm_dg_d{}", d);
                        let cached = runtime.get_kernel(&name);
                        if let Some(k) = cached {
                            k
                        } else {
                            use crate::t0::block_dsl::BlockKernel;
                            use crate::t0::ir::Target;
                            let mut kb = BlockKernel::new(&name, 32);
                            let dy_ptr = kb.arg_ptr("dy");
                            let x_ptr = kb.arg_ptr("x");
                            let irms_ptr = kb.arg_ptr("inv_rms");
                            let dg_ptr = kb.arg_ptr("dgamma");
                            let dim_arg = kb.arg_u32("dim");

                            let row_id = kb.program_id(0);
                            let lane_id = kb.thread_id();
                            let row_base = row_id.mul(&mut kb, dim_arg);

                            let max_u = kb.const_u32(u32::MAX);
                            let always = row_id.lt(&mut kb, max_u);
                            let inv_rms = kb.load(irms_ptr, row_id, always);

                            // dgamma[c] += dy[row,c] * x[row,c] * inv_rms
                            let epl = ((d + 31) / 32) as u32;
                            for j in 0..epl {
                                let col = if j == 0 { lane_id } else {
                                    let o = kb.const_u32(j * 32);
                                    lane_id.add(&mut kb, o)
                                };
                                let mask = col.lt(&mut kb, dim_arg);
                                let idx = row_base.add(&mut kb, col);
                                let dy = kb.load(dy_ptr, idx, mask);
                                let x = kb.load(x_ptr, idx, mask);
                                let contrib = dy.mul(&mut kb, x).mul(&mut kb, inv_rms);
                                kb.atomic_add_f32(dg_ptr, col, contrib, mask);
                            }

                            let compiled = kb.compile(Target::GFX1100)?;
                            runtime.compile_dsl(compiled)?
                        }
                    };

                    let dg_buf = runtime.alloc_f32(d)?;
                    dg_buf.zero();
                    let ka = crate::kernargs![
                        grad_output.gpu_addr() => u64,
                        saved[0].gpu_addr() => u64,  // x
                        saved[2].gpu_addr() => u64,  // inv_rms
                        dg_buf.gpu_addr() => u64,
                        d as u32 => u32
                    ];
                    runtime.dispatch(&dg_kernel, [r as u32 * 32, 1, 1], &ka)?;
                    grads.push(Some(Arc::new(dg_buf)));
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
