//! Embedding op — gather forward + scatter_add backward.
//!
//! Forward: for each token_id, copy the embedding row from the table
//! Backward: scatter_add gradients back to the embedding table

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

/// Embedding gather: table[token_ids] → output
///
/// - table: [vocab_size, dim] f32
/// - token_ids: [seq_len] u32 (as f32 cast, or separate u32 buffer)
/// - output: [seq_len, dim] f32
///
/// Kernarg layout (build_embedding_gather):
///   [0:8]   table_ptr
///   [8:16]  ids_ptr 
///   [16:24] out_ptr
///   [24:28] dim
#[cfg(feature = "rocm")]
pub fn embedding_forward(
    table: &Tensor,
    ids_buf: &GpuBuffer,
    seq_len: usize,
    dim: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    // Build gather kernel: 1 wave per token
    let kernel = {
        let name = format!("bdsl_emb_gather_d{}", dim);
        let cached = runtime.get_kernel(&name);
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new(&name, 32);
            let table_ptr = kb.arg_ptr("table");
            let ids_ptr = kb.arg_ptr("ids");
            let out_ptr = kb.arg_ptr("out");
            let dim_arg = kb.arg_u32("dim");

            let token_id = kb.program_id(0); // 1 WG per token
            let lane_id = kb.thread_id();

            // Load index: ids[token_id]
            let one_u = kb.const_u32(1);
            let max_u = kb.const_u32(u32::MAX);
            let id_mask = token_id.lt(&mut kb, max_u); // always true
            let word_idx = kb.load_u32(ids_ptr, token_id, id_mask);

            // table_base = word_idx * dim
            let table_base = word_idx.mul(&mut kb, dim_arg);
            // out_base = token_id * dim
            let out_base = token_id.mul(&mut kb, dim_arg);

            // Copy dim elements
            let epl = ((dim + 31) / 32) as u32;
            for j in 0..epl {
                let col = if j == 0 {
                    lane_id
                } else {
                    let offset = kb.const_u32(j * 32);
                    lane_id.add(&mut kb, offset)
                };
                let mask = col.lt(&mut kb, dim_arg);
                let t_idx = table_base.add(&mut kb, col);
                let o_idx = out_base.add(&mut kb, col);
                let val = kb.load(table_ptr, t_idx, mask);
                kb.store(out_ptr, o_idx, val, mask);
            }

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    let out_buf = runtime.alloc_f32(seq_len * dim)?;

    let ka = crate::kernargs![
        table.gpu_addr() => u64,
        ids_buf.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        dim as u32 => u32
    ];
    runtime.dispatch(&kernel, [seq_len as u32 * 32, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    let output = Tensor::from_buffer(out_arc, runtime, &[seq_len, dim], DType::F32, "emb_out");

    if Tape::is_recording() && table.requires_grad() {
        let table_id = Some(table.id());
        let ids_arc = Arc::new(clone_buf(ids_buf, runtime)?);
        let vocab_size = table.shape()[0];
        let d = dim;
        let sl = seq_len;
        let vs = vocab_size;

        let node_id = Tape::record(
            "embedding",
            output.id(),
            vec![table_id],
            vec![true],
            vec![ids_arc],
            Box::new(move |grad_output, saved, runtime| {
                // scatter_add via block_dsl with atomic_add_f32
                let ids = &saved[0];
                let grad_table = runtime.alloc_f32(vs * d)?;
                grad_table.zero();

                let kernel = {
                    let name = format!("bdsl_emb_scatter_d{}", d);
                    let cached = runtime.get_kernel(&name);
                    if let Some(k) = cached {
                        k
                    } else {
                        use crate::t0::block_dsl::BlockKernel;
                        use crate::t0::ir::Target;
                        let mut kb = BlockKernel::new(&name, 32);
                        let grad_out_ptr = kb.arg_ptr("grad_out");
                        let ids_ptr = kb.arg_ptr("ids");
                        let grad_table_ptr = kb.arg_ptr("grad_table");
                        let dim_arg = kb.arg_u32("dim");

                        let token_id = kb.program_id(0);
                        let lane_id = kb.thread_id();

                        let max_u2 = kb.const_u32(u32::MAX);
                        let always = token_id.lt(&mut kb, max_u2);
                        let word_idx = kb.load_u32(ids_ptr, token_id, always);

                        let grad_base = token_id.mul(&mut kb, dim_arg);
                        let table_base = word_idx.mul(&mut kb, dim_arg);

                        let epl = ((d + 31) / 32) as u32;
                        for j in 0..epl {
                            let col = if j == 0 {
                                lane_id
                            } else {
                                let offset = kb.const_u32(j * 32);
                                lane_id.add(&mut kb, offset)
                            };
                            let mask = col.lt(&mut kb, dim_arg);
                            let g_idx = grad_base.add(&mut kb, col);
                            let t_idx = table_base.add(&mut kb, col);
                            let gval = kb.load(grad_out_ptr, g_idx, mask);
                            kb.atomic_add_f32(grad_table_ptr, t_idx, gval, mask);
                        }

                        let compiled = kb.compile(Target::GFX1100)?;
                        runtime.compile_dsl(compiled)?
                    }
                };

                let ka = crate::kernargs![
                    grad_output.gpu_addr() => u64,
                    ids.gpu_addr() => u64,
                    grad_table.gpu_addr() => u64,
                    d as u32 => u32
                ];
                runtime.dispatch(&kernel, [sl as u32 * 32, 1, 1], &ka)?;
                Ok(vec![Some(Arc::new(grad_table))])
            }),
        );
        output.set_tape_node(node_id);
    }

    Ok(output)
}

#[cfg(feature = "rocm")]
fn clone_buf(src: &GpuBuffer, runtime: &Arc<GpuRuntime>) -> Result<GpuBuffer, String> {
    let dst = runtime.alloc(src.size)?;
    let mut tmp = vec![0u8; src.size];
    src.read(&mut tmp);
    dst.write(&tmp);
    Ok(dst)
}
