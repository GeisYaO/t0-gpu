//! T0 Legacy Math Layer — Hand-written ISA kernels
//!
//! Contains OCPA attention kernels and row-reduce operations that
//! require manual ISA control beyond BlockDSL capabilities.

use super::ir::*;
use super::compile::T0Kernel;
use super::schedule::Schedule;

/// Binary elementwise operation type.
#[derive(Clone, Copy, Debug)]
pub enum BinaryOp {
    /// y = a + b
    Add,
    /// y = a * b
    Mul,
    /// y = a + alpha * b  (axpy)
    Axpy(f32),
}

#[derive(Clone, Copy, Debug)]
struct Bf16F32WmmaShape {
    frag_regs: u32,
    frag_align: Alignment,
    acc_regs: u32,
    acc_align: Alignment,
}

impl Bf16F32WmmaShape {
    fn current() -> Self {
        let target = current_target();
        let format = WmmaFormat::BF16_F32;
        let frag_regs = format.a_vreg_count(target);
        debug_assert_eq!(frag_regs % 4, 0, "WMMA BF16 fragment must be a multiple of 4 VGPRs");
        Self {
            frag_regs,
            frag_align: format.a_alignment(target),
            acc_regs: format.dst_vreg_count(target),
            acc_align: format.dst_alignment(target),
        }
    }

    fn frag_loads(self) -> u32 { self.frag_regs / 4 }
    fn frag_bytes(self) -> u32 { self.frag_regs * 4 }
    fn frag_span(self, groups: u32) -> u32 { self.frag_regs * groups }
    fn acc_span(self, groups: u32) -> u32 { self.acc_regs * groups }
}

fn zero_vreg_span(k: &mut T0Kernel, base: VReg, regs: u32) {
    for i in 0..regs {
        k.v_mov_imm(VReg(base.0 + i), 0);
    }
}

fn load_b128_span(k: &mut T0Kernel, dst: VReg, addr: VReg, regs: u32) {
    debug_assert_eq!(regs % 4, 0, "B128 span load requires 4-VGPR chunks");
    for i in 0..(regs / 4) {
        k.global_load(VReg(dst.0 + i * 4), addr, Width::B128, (i * 16) as i32);
    }
}

fn ds_load_bf16_fragment(
    k: &mut T0Kernel,
    dst: VReg,
    vaddr: VReg,
    regs: u32,
    base_offset: u32,
    row_stride_bytes: u32,
) {
    for i in 0..regs {
        let off_lo = base_offset + (2 * i) * row_stride_bytes;
        let off_hi = base_offset + (2 * i + 1) * row_stride_bytes;
        k.ds_load_u16_d16(VReg(dst.0 + i), vaddr, off_lo as u16);
        k.ds_load_u16_d16_hi(VReg(dst.0 + i), vaddr, off_hi as u16);
    }
}


// ============================================================================
// unpad_2d: strip padding from a 2D f32 buffer
// ============================================================================

/// Generate f32 unpad kernel: dst[i * n + j] = src[i * n_pad + j]
///
/// Copies valid [m, n] portion from a padded [m_pad, n_pad] buffer.
/// Each thread handles one element (same pattern as transpose_f32).
///
/// Kernargs (32 bytes): src_ptr(0), dst_ptr(8), m(16), n(20), n_pad(24), _pad(28)
/// Grid: [ceil(m*n/32)*32, 1, 1], WG: 32
pub fn t0_unpad_2d() -> T0Kernel {
    let mut k = T0Kernel::new("t0_unpad_2d");

    let src_ptr = k.arg_ptr("src");
    let dst_ptr = k.arg_ptr("dst");
    let dim_m = k.arg_u32("m");
    let dim_n = k.arg_u32("n");
    let stride_src = k.arg_u32("n_pad");
    k.emit_arg_loads();

    let tile_s = k.alloc_sreg();
    k.capture_tgid_x(tile_s);

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // global_id = TGID.x * 32 + lane_id
    let gid = k.alloc_vreg();
    k.v_mov_from_sgpr(gid, tile_s);
    k.v_lshlrev_b32(gid, 5, gid);
    k.v_add_u32(gid, gid, lane_id);

    // Bounds check: gid < m * n
    let total = k.alloc_vreg();
    let m_v = k.alloc_vreg();
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(m_v, SReg(dim_m.0));
    k.v_mov_from_sgpr(n_v, SReg(dim_n.0));
    k.v_mul_lo_u32(total, m_v, n_v);
    let saved_exec = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(gid), src1: Operand::VReg(total) });
    k.push(Op::SaveExec { dst: saved_exec });

    // Divmod: i = gid / n, j = gid % n (float rcp approximation + fixup)
    let n_f = k.alloc_vreg();
    k.v_cvt_f32_u32(n_f, n_v);
    let rcp_n = k.alloc_vreg();
    k.v_rcp_f32(rcp_n, n_f);
    let gid_f = k.alloc_vreg();
    k.v_cvt_f32_u32(gid_f, gid);
    let i_approx_f = k.alloc_vreg();
    k.push(Op::VMulF32 { dst: i_approx_f, src0: Operand::VReg(gid_f), src1: Operand::VReg(rcp_n) });
    let i_v = k.alloc_vreg();
    k.v_cvt_u32_f32(i_v, i_approx_f);
    let j_v = k.alloc_vreg();
    let tmp = k.alloc_vreg();
    k.v_mul_lo_u32(tmp, i_v, n_v);
    k.v_sub_u32(j_v, gid, tmp);

    // Fix rounding: if j >= n then i++, j -= n
    let fix = k.alloc_vreg();
    k.v_cmp_ge_u32(Operand::VReg(j_v), Operand::VReg(n_v));
    k.v_mov_imm(fix, 0);
    let one_v = k.alloc_vreg();
    k.v_mov_imm(one_v, 1);
    k.v_cndmask_b32(fix, Operand::VReg(fix), Operand::VReg(one_v));
    k.v_add_u32(i_v, i_v, fix);
    let fix_n = k.alloc_vreg();
    k.v_mul_lo_u32(fix_n, fix, n_v);
    k.v_sub_u32(j_v, j_v, fix_n);

    // Read src: src_ptr + (i * n_pad + j) * 4
    let n_pad_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_pad_v, SReg(stride_src.0));
    let src_off = k.alloc_vreg();
    k.v_mul_lo_u32(src_off, i_v, n_pad_v);  // i * n_pad
    k.v_add_u32(src_off, src_off, j_v);       // i * n_pad + j
    k.v_lshlrev_b32(src_off, 2, src_off);     // * 4 (f32 bytes)
    let src_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(src_addr, SReg(src_ptr.0));
    k.v_mov_from_sgpr(VReg(src_addr.0 + 1), SReg(src_ptr.0 + 1));
    k.v_add_co(src_addr, src_addr, src_off);
    k.v_add_co_ci(VReg(src_addr.0 + 1), VReg(src_addr.0 + 1));
    let val = k.alloc_vreg();
    k.global_load(val, src_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // Write dst: dst_ptr + gid * 4 (contiguous output)
    let dst_off = k.alloc_vreg();
    k.v_lshlrev_b32(dst_off, 2, gid);
    let dst_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dst_addr, SReg(dst_ptr.0));
    k.v_mov_from_sgpr(VReg(dst_addr.0 + 1), SReg(dst_ptr.0 + 1));
    k.v_add_co(dst_addr, dst_addr, dst_off);
    k.v_add_co_ci(VReg(dst_addr.0 + 1), VReg(dst_addr.0 + 1));
    k.global_store(dst_addr, val, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved_exec });
    k.wait_vscnt(0);
    k.endpgm();
    k
}



/// Generate a binary elementwise kernel: y[i] = op(a[i], b[i])
///
/// Grid: [ceil(n/wg_size), 1, 1]
/// Kernargs: a_ptr(0), b_ptr(8), y_ptr(16), param(24:f32), n(28:u32)
pub fn elementwise_binary(sched: &dyn Schedule, op: BinaryOp) -> T0Kernel {
    let name = match op {
        BinaryOp::Add => "t0_add",
        BinaryOp::Mul => "t0_mul",
        BinaryOp::Axpy(_) => "t0_axpy",
    };

    let mut k = T0Kernel::new(name);
    let (wg_x, _, _) = sched.workgroup_size();

    // ── Args ──
    let a_ptr = k.arg_ptr("a");
    let b_ptr = k.arg_ptr("b");
    let y_ptr = k.arg_ptr("y");
    let alpha = k.arg_f32("alpha");
    let _n = k.arg_u32("n");
    k.emit_arg_loads();

    let global_id = k.compute_global_id_x(wg_x as u32);

    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, global_id);

    // Load a[i]
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_addr.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_addr, a_addr, byte_off);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
    let a_val = k.alloc_vreg();
    k.global_load(a_val, a_addr, Width::B32, 0);

    // Load b[i]
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_addr.0 + 1), SReg(b_ptr.0 + 1));
    k.v_add_co(b_addr, b_addr, byte_off);
    k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
    let b_val = k.alloc_vreg();
    k.global_load(b_val, b_addr, Width::B32, 0);

    k.wait_vmcnt(0);

    // Apply op
    let result = k.alloc_vreg();
    match op {
        BinaryOp::Add => {
            k.v_add_f32(result, a_val, b_val);
        }
        BinaryOp::Mul => {
            k.v_mul_f32(result, a_val, b_val);
        }
        BinaryOp::Axpy(_) => {
            // y = a + alpha * b
            let alpha_v = k.alloc_vreg();
            k.v_mov_from_sgpr(alpha_v, SReg(alpha.0));
            k.v_fma_f32(result, alpha_v, b_val, a_val);
        }
    }

    // Store y[i]
    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
    k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
    k.v_add_co(y_addr, y_addr, byte_off);
    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
    k.global_store(y_addr, result, Width::B32, 0);

    k.wait_vscnt(0);
    k.endpgm();
    k
}




// ============================================================================
// OCPA Forward Intra: 1D Tile-Stealing with Causal GEMM
//
// Computes: O_intra_c = (CausalMask(Q_c @ K_c^T)) @ V_c
// Accumulates: O_c += O_intra_c via global_atomic_add_f32
//
// Architecture:
//   Phase A₀: Build 1D→2D LUT (lower triangle tiles)
//   Phase A:  512 threads cooperatively load V_c into LDS ZCLT (132B padded)
//   Phase B:  1D Tile-Stealing: wave steals tiles from pool
//             LUT lookup → conditional Q reload → K tile → QK^T WMMA →
//             causal mask → transpose → V-tile WMMA → atomic add O
// ============================================================================

/// OCPA Forward Intra kernel (C=256)
pub fn ocpa_forward_intra() -> T0Kernel {
    ocpa_forward_intra_with_c(256)
}



/// Forward Intra, parameterized by chunk size
pub fn ocpa_forward_intra_with_c(c_chunk: u32) -> T0Kernel {
    let n_tile_rows = c_chunk / 16;
    let total_tiles = n_tile_rows * (n_tile_rows + 1) / 2;
    let lds_zclt_size = c_chunk * 132;
    let lut_base = lds_zclt_size;
    let lds_total = ((lut_base + total_tiles * 2 + 255) / 256) * 256;

    let mut k = T0Kernel::new(&format!("t0_ocpa_forward_intra_c{}", c_chunk));
    k.set_lds_size(lds_total);

    let q_ptr = k.arg_ptr("Q");
    let k_ptr = k.arg_ptr("K");
    let v_ptr = k.arg_ptr("V");
    let o_ptr = k.arg_ptr("O");
    let seq_len = k.arg_u32("seq_len");
    k.emit_arg_loads();

    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, tid, 15);
    let wmma = Bf16F32WmmaShape::current();

    // base_row = head_id * seq_len + chunk_id * C
    let base_row = k.alloc_sreg();
    k.s_mul_i32(base_row, head_id, SReg(seq_len.0));
    let c_off = k.alloc_sreg();
    let c_shift = match c_chunk { 32=>5, 64=>6, 128=>7, _=>8 };
    k.s_lshl_b32(c_off, chunk_id, c_shift);
    k.s_add_u32_ss(base_row, base_row, c_off);

    // ═════════════════════════════════════════════════════════════
    // Phase A₀: Build 1D→2D LUT (only first total_tiles threads)
    // r = floor((sqrt(8k+1) - 1) / 2), c = k - r*(r+1)/2
    // ═════════════════════════════════════════════════════════════
    let total_v = k.alloc_vreg();
    k.v_mov_imm(total_v, total_tiles as i32);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(total_v) });
    k.branch_vccz("lut_done");

    // r = floor((sqrt(8*tid + 1) - 1) / 2)
    // r = floor((isqrt(8*tid + 1) - 1) / 2) — integer-only via float sqrt
    let k8 = k.alloc_vreg();
    k.v_lshlrev_b32(k8, 3, tid); // k8 = tid * 8
    let k8p1 = k.alloc_vreg();
    k.v_mov_imm(k8p1, 1);
    k.v_add_u32(k8p1, k8, k8p1); // k8p1 = 8*tid + 1
    let fk = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: fk, src: k8p1 });
    k.v_sqrt_f32(fk, fk);
    k.push(Op::VCvtU32F32 { dst: fk, src: fk }); // fk = floor(sqrt(8k+1))
    let neg1 = k.alloc_vreg();
    k.v_mov_imm(neg1, -1i32);
    k.v_add_u32(fk, fk, neg1); // fk = isqrt(8k+1) - 1
    let r_reg = k.alloc_vreg();
    k.v_lshrrev_b32(r_reg, 1, fk); // r = (isqrt-1) / 2

    // Correction: tri = r*(r+1)/2, if tri > k then r--
    let rp1 = k.alloc_vreg();
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    let tri = k.alloc_vreg();
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(tri) });
    k.branch_vccz("r_ok");
    k.v_add_u32(r_reg, r_reg, neg1); // r--
    // Recompute tri
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.label("r_ok");

    // c = k - tri
    let c_reg = k.alloc_vreg();
    // c = tid - tri: use two's complement
    let neg_tri = k.alloc_vreg();
    k.push(Op::VXorB32 { dst: neg_tri, src0: Operand::VReg(tri), src1: Operand::VReg(neg1) }); // ~tri
    let one_v = k.alloc_vreg();
    k.v_mov_imm(one_v, 1);
    k.v_add_u32(neg_tri, neg_tri, one_v); // -tri
    k.v_add_u32(c_reg, tid, neg_tri); // c = tid - tri

    // Pack (r << 8 | c) and write to LDS
    let packed = k.alloc_vreg();
    k.v_lshlrev_b32(packed, 8, r_reg);
    k.v_or_b32(packed, Operand::VReg(packed), Operand::VReg(c_reg));

    let lut_addr = k.alloc_vreg();
    k.v_lshlrev_b32(lut_addr, 1, tid); // k * 2
    let lut_base_v = k.alloc_vreg();
    k.v_mov_imm(lut_base_v, lut_base as i32);
    k.v_add_u32(lut_addr, lut_addr, lut_base_v);
    k.push(Op::DsStoreB16 { vaddr: lut_addr, src: packed, offset: 0 });

    k.label("lut_done");
    k.wait_lgkmcnt(0);

    // ═════════════════════════════════════════════════════════════
    // Phase A: Cooperative V_c ZCLT load (512 threads, 132B stride)
    // ═════════════════════════════════════════════════════════════
    let v_row = k.alloc_vreg();
    k.v_lshrrev_b32(v_row, 1, tid); // row = tid / 2
    let col_grp = k.alloc_vreg();
    k.v_and_b32_imm(col_grp, tid, 1);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 6, col_grp); // col_grp * 64

    // Clamp row for C < 256
    let lds_row = if c_chunk < 256 {
        let clamped = k.alloc_vreg();
        k.v_and_b32_imm(clamped, v_row, c_chunk - 1);
        clamped
    } else {
        v_row
    };

    // HBM addr = V_ptr + (base_row + row) * 128 + col_bytes
    let v_off = k.alloc_vreg();
    k.v_mov_from_sgpr(v_off, base_row);
    k.v_add_u32(v_off, v_off, lds_row);
    k.v_lshlrev_b32(v_off, 7, v_off); // * 128
    k.v_add_u32(v_off, v_off, col_bytes);

    let v_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(v_addr, SReg(v_ptr.0));
    k.v_mov_from_sgpr(VReg(v_addr.0 + 1), SReg(v_ptr.0 + 1));
    k.v_add_co(v_addr, v_addr, v_off);
    k.v_add_co_ci(VReg(v_addr.0 + 1), VReg(v_addr.0 + 1));

    let v_data = k.alloc_vreg_array(16, Alignment::Align4);
    for i in 0..4u32 {
        k.global_load(VReg(v_data.0 + i * 4), v_addr, Width::B128, (i * 16) as i32);
    }

    // LDS write: lds_row * 132 + col_bytes
    let s132 = k.alloc_sreg();
    k.s_mov_imm(s132, 132);
    let stride132 = k.alloc_vreg();
    k.v_mov_from_sgpr(stride132, s132);
    let lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(lds_wr, lds_row, stride132);
    k.v_add_u32(lds_wr, lds_wr, col_bytes);

    k.wait_vmcnt(0);
    for i in 0..4u32 {
        k.ds_store_b128(lds_wr, VReg(v_data.0 + i * 4), (i * 16) as u16);
    }
    k.wait_lgkmcnt(0);
    k.s_barrier(); // Sync LUT + ZCLT

    // ═════════════════════════════════════════════════════════════
    // Prepare O output base
    // ═════════════════════════════════════════════════════════════
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });

    let o_base_lo = k.alloc_sreg();
    let o_base_hi = k.alloc_sreg();
    let tmp_off = k.alloc_sreg();
    k.s_lshl_b32(tmp_off, base_row, 8); // base_row * 256
    k.s_add_u32_ss(o_base_lo, SReg(o_ptr.0), tmp_off);
    k.s_addc_u32_imm(o_base_hi, SReg(o_ptr.0 + 1), 0);

    // ═════════════════════════════════════════════════════════════
    // 1D Tile-Stealing Main Loop
    // k_task = wave_id, step += 16
    // ═════════════════════════════════════════════════════════════
    let k_task = k.alloc_sreg();
    k.push(Op::SMov { dst: k_task, src: SOperand::SReg(wave_id_s) });
    let cur_r = k.alloc_sreg();
    k.s_mov_imm(cur_r, -1i32); // invalid, force first Q load

    // Skip idle waves
    let s_total = k.alloc_sreg();
    k.s_mov_imm(s_total, total_tiles as i32);
    k.s_cmp_ge_u32(k_task, s_total);
    k.branch_scc1("epilogue");

    // Big arrays: 4 K-groups, 4 V-groups, target-specific BF16 fragment width.
    let q_frag = k.alloc_vreg_array(wmma.frag_span(4), wmma.frag_align);
    let k_frag = k.alloc_vreg_array(wmma.frag_span(4), wmma.frag_align);
    let p_acc = k.alloc_vreg_array(wmma.acc_regs, wmma.acc_align);
    let o_acc = k.alloc_vreg_array(wmma.acc_span(4), wmma.acc_align);
    let p_trans = k.alloc_vreg_array(wmma.frag_regs, wmma.frag_align);
    let v_tile = k.alloc_vreg_array(wmma.frag_regs, wmma.frag_align);

    k.label("main_loop");

    // ── LUT lookup: read (r, c) ──
    let lut_off = k.alloc_sreg();
    k.s_lshl_b32(lut_off, k_task, 1);
    let lut_base_s = k.alloc_sreg();
    k.s_mov_imm(lut_base_s, lut_base as i32);
    k.s_add_u32_ss(lut_off, lut_off, lut_base_s);
    let lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(lut_v, lut_off);
    let lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: lut_val, vaddr: lut_v, offset: 0 });
    k.wait_lgkmcnt(0);

    // Unpack: tile_r, tile_c
    let tile_r_v = k.alloc_vreg();
    k.v_lshrrev_b32(tile_r_v, 8, lut_val);
    let tile_r = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: tile_r, src: tile_r_v });
    let tile_c_v = k.alloc_vreg();
    k.v_and_b32_imm(tile_c_v, lut_val, 0xFF);
    let tile_c = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: tile_c, src: tile_c_v });

    // ── Conditional Q_slice load (when r changes) ──
    k.push(Op::SCmpEqU32 { src0: tile_r, src1: SOperand::SReg(cur_r) });
    k.branch_scc1("skip_q_load");

    k.push(Op::SMov { dst: cur_r, src: SOperand::SReg(tile_r) });
    // Q addr = Q_ptr + (base_row + r*16 + lane_row) * 128
    let q_row = k.alloc_sreg();
    k.s_lshl_b32(q_row, tile_r, 4);
    k.s_add_u32_ss(q_row, q_row, base_row);
    let q_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(q_row_v, q_row);
    k.v_add_u32(q_row_v, q_row_v, lane_row);
    k.v_lshlrev_b32(q_row_v, 7, q_row_v);
    let q_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(q_addr, SReg(q_ptr.0));
    k.v_mov_from_sgpr(VReg(q_addr.0 + 1), SReg(q_ptr.0 + 1));
    k.v_add_co(q_addr, q_addr, q_row_v);
    k.v_add_co_ci(VReg(q_addr.0 + 1), VReg(q_addr.0 + 1));

    load_b128_span(&mut k, q_frag, q_addr, wmma.frag_span(4));
    // Zero O_acc when r changes
    zero_vreg_span(&mut k, o_acc, wmma.acc_span(4));
    k.wait_vmcnt(0);
    k.label("skip_q_load");

    // ── Load K_tile[c] ──
    let k_row = k.alloc_sreg();
    k.s_lshl_b32(k_row, tile_c, 4);
    k.s_add_u32_ss(k_row, k_row, base_row);
    let k_off = k.alloc_sreg();
    k.s_lshl_b32(k_off, k_row, 7);
    let k_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(k_off_v, 7, lane_row);
    let k_off_s_v = k.alloc_vreg();
    k.v_mov_from_sgpr(k_off_s_v, k_off);
    k.v_add_u32(k_off_v, k_off_v, k_off_s_v);
    let k_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(k_addr, SReg(k_ptr.0));
    k.v_mov_from_sgpr(VReg(k_addr.0 + 1), SReg(k_ptr.0 + 1));
    k.v_add_co(k_addr, k_addr, k_off_v);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));

    load_b128_span(&mut k, k_frag, k_addr, wmma.frag_span(4));

    zero_vreg_span(&mut k, p_acc, wmma.acc_regs);
    k.wait_vmcnt(0);

    // ── WMMA: P^T = K @ Q^T (4 k-groups) ──
    for kg in 0..4u32 {
        k.wmma_bf16_f32(
            p_acc,
            VReg(k_frag.0 + kg * wmma.frag_regs),
            VReg(q_frag.0 + kg * wmma.frag_regs),
            p_acc,
        );
    }

    // ── Causal mask (diagonal: r == c) ──
    k.push(Op::SCmpEqU32 { src0: tile_r, src1: SOperand::SReg(tile_c) });
    k.branch_scc0("mask_done");

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let zero_f = k.alloc_vreg();
    k.v_mov_imm(zero_f, 0);
    let one_f = k.alloc_vreg();
    k.v_mov_imm(one_f, 0x3F800000u32 as i32); // 1.0f32

    for vk in 0..wmma.acc_regs {
        // row_in_tile = 2*vk + lane_half
        let row_v = k.alloc_vreg();
        if vk == 0 {
            k.push(Op::VMov { dst: row_v, src: Operand::VReg(lane_half) });
        } else {
            k.v_mov_imm(row_v, (2 * vk) as i32);
            k.v_add_u32(row_v, lane_half, row_v);
        }
        // Lower-triangular mask on P (stored as P^T in C-register):
        // Keep P[row,col] where row >= col. In P^T: keep P^T[col,row] where col <= row.
        // C-register: row_v = row index, lane_row = col index.
        // For P^T, we need col_in_P^T <= row_in_P^T to mask the UPPER part of P^T.
        k.v_cmp_ge_i32(lane_row, row_v);
        k.push(Op::VCndmaskB32 { dst: row_v, src_false: Operand::VReg(zero_f), src_true: Operand::VReg(one_f) });
        k.push(Op::VMulF32 { dst: VReg(p_acc.0 + vk), src0: Operand::VReg(VReg(p_acc.0 + vk)), src1: Operand::VReg(row_v) });
    }
    k.label("mask_done");
    k.clear_vcc(); // prevent VCC residual from mask cmp affecting v_add_co_ci

    // ── C-layout → A-operand transpose (pure VALU, no LDS) ──
    // v_tile[0:2] used as temp (free before V-tile loads)
    k.reg_transpose_c_to_ab(p_trans, p_acc, v_tile);

    // ── V-tile WMMA: O_acc += P_trans @ V_tile (4 v-groups) ──
    let lds_read_base = k.alloc_vreg();
    k.v_and_b32_imm(lds_read_base, lane_row, 15);
    k.v_lshlrev_b32(lds_read_base, 1, lds_read_base);
    // LDS base for column c: c * 16 * 132 = c * 2112
    let s2112 = k.alloc_sreg();
    k.s_mov_imm(s2112, 2112);
    let v_lds_base = k.alloc_sreg();
    k.s_mul_i32(v_lds_base, tile_c, s2112);

    for vg in 0..4u32 {
        let col_off = vg * wmma.frag_bytes();
        let lds_off = k.alloc_sreg();
        k.s_mov_imm(lds_off, (col_off) as i32);
        k.s_add_u32_ss(lds_off, v_lds_base, lds_off);
        let lds_addr = k.alloc_vreg();
        k.v_mov_from_sgpr(lds_addr, lds_off);
        k.v_add_u32(lds_addr, lds_read_base, lds_addr);

        ds_load_bf16_fragment(&mut k, v_tile, lds_addr, wmma.frag_regs, 0, 132);
        k.wait_lgkmcnt(0);
        k.wmma_bf16_f32(
            VReg(o_acc.0 + vg * wmma.acc_regs),
            p_trans,
            v_tile,
            VReg(o_acc.0 + vg * wmma.acc_regs),
        );
    }

    // ── Check if next task has different r → flush O_acc ──
    let next_k = k.alloc_sreg();
    k.s_add_u32(next_k, k_task, 16);
    k.s_cmp_ge_u32(next_k, s_total);
    k.branch_scc1("flush"); // last iteration → must flush

    // Read next tile's r
    let next_lut_off = k.alloc_sreg();
    k.s_lshl_b32(next_lut_off, next_k, 1);
    k.s_add_u32_ss(next_lut_off, next_lut_off, lut_base_s);
    let next_lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(next_lut_v, next_lut_off);
    let next_lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: next_lut_val, vaddr: next_lut_v, offset: 0 });
    k.wait_lgkmcnt(0);
    let next_r_v = k.alloc_vreg();
    k.v_lshrrev_b32(next_r_v, 8, next_lut_val);
    let next_r = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: next_r, src: next_r_v });
    k.push(Op::SCmpEqU32 { src0: next_r, src1: SOperand::SReg(tile_r) });
    k.branch_scc1("skip_flush"); // same r → accumulate more

    // ── FLUSH: Atomic add O_acc to HBM ──
    k.label("flush");
    let flush_lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(flush_lane_half, 4, lane_id);
    let r16 = k.alloc_sreg();
    k.s_lshl_b32(r16, tile_r, 4);
    let r16_v = k.alloc_vreg();
    k.v_mov_from_sgpr(r16_v, r16);
    k.v_add_u32(r16_v, r16_v, flush_lane_half);
    let row_off = k.alloc_vreg();
    k.v_lshlrev_b32(row_off, 8, r16_v); // * 256 bytes/row
    let col_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(col_off_v, 2, lane_row);

    let o_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_addr, o_base_lo);
    k.v_mov_from_sgpr(VReg(o_addr.0 + 1), o_base_hi);
    k.v_add_co(o_addr, o_addr, row_off);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));
    k.v_add_co(o_addr, o_addr, col_off_v);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));

    for vg in 0..4u32 {
        for vk in 0..wmma.acc_regs {
            let off = (vk as i32) * 512 + (vg as i32) * 64;
            k.global_atomic_add_f32(o_addr, VReg(o_acc.0 + vg * wmma.acc_regs + vk), off);
        }
    }
    k.wait_vmcnt(0);

    k.label("skip_flush");

    // k += 16, loop if k < total_tiles
    k.s_add_u32(k_task, k_task, 16);
    k.s_cmp_lt_u32(k_task, s_total);
    k.branch_scc1("main_loop");

    k.label("epilogue");
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}



// ============================================================================
// OCPA Backward Intra: 1D Tile-Stealing with Causal GEMM
//
// Mathematical Isomorphism with Forward Intra:
//   Intra(A, B, C, Mask) = (Mask ⊙ (A @ B^T)) @ C
//   dQ_intra = Intra(dO, V, K, Lower)   ← identical to forward
//   dK_intra = Intra(V, dO, Q, Upper)   ← time-arrow reversed
//   dV_intra = Intra(K, Q, dO, Upper)   ← time-arrow reversed
//
// Kernarg: A_ptr(0), B_ptr(8), C_ptr(16), Out_ptr(24), seq_len(32)
// ============================================================================

/// Backward Intra dQ (lower mask)
pub fn ocpa_backward_intra_dq() -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dq", false, 256)
}



/// Backward Intra dQ with parameterized c_chunk
pub fn ocpa_backward_intra_dq_c(c_chunk: u32) -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dq", false, c_chunk)
}



/// Backward Intra dK/dV (upper mask)
pub fn ocpa_backward_intra_dkdv() -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dkdv", true, 256)
}



/// Backward Intra dK/dV with parameterized c_chunk
pub fn ocpa_backward_intra_dkdv_c(c_chunk: u32) -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dkdv", true, c_chunk)
}



fn build_backward_intra(name: &str, is_upper: bool, c_chunk: u32) -> T0Kernel {
    let n_tile_rows = c_chunk / 16;
    let total_tiles = n_tile_rows * (n_tile_rows + 1) / 2;
    let lds_zclt_size = c_chunk * 132;
    let lut_base = lds_zclt_size;
    let lds_total = ((lut_base + total_tiles * 2 + 255) / 256) * 256;

    let mut k = T0Kernel::new(name);
    k.set_lds_size(lds_total);

    // Kernarg: A_ptr, B_ptr, C_ptr, Out_ptr, seq_len
    let a_ptr = k.arg_ptr("A");
    let b_ptr = k.arg_ptr("B");
    let c_ptr = k.arg_ptr("C");
    let out_ptr = k.arg_ptr("Out");
    let seq_len = k.arg_u32("seq_len");
    k.emit_arg_loads();

    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, tid, 15);
    let wmma = Bf16F32WmmaShape::current();

    let base_row = k.alloc_sreg();
    k.s_mul_i32(base_row, head_id, SReg(seq_len.0));
    let c_off = k.alloc_sreg();
    let c_shift = match c_chunk { 32=>5, 64=>6, 128=>7, _=>8 };
    k.s_lshl_b32(c_off, chunk_id, c_shift);
    k.s_add_u32_ss(base_row, base_row, c_off);

    // ── Phase A₀: Build 1D→2D LUT (same as forward) ──
    let total_v = k.alloc_vreg();
    k.v_mov_imm(total_v, total_tiles as i32);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(total_v) });
    k.branch_vccz("lut_done");

    let k8 = k.alloc_vreg();
    k.v_lshlrev_b32(k8, 3, tid);
    let k8p1 = k.alloc_vreg();
    k.v_mov_imm(k8p1, 1);
    k.v_add_u32(k8p1, k8, k8p1);
    let fk = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: fk, src: k8p1 });
    k.v_sqrt_f32(fk, fk);
    k.push(Op::VCvtU32F32 { dst: fk, src: fk });
    let neg1 = k.alloc_vreg();
    k.v_mov_imm(neg1, -1i32);
    k.v_add_u32(fk, fk, neg1);
    let r_reg = k.alloc_vreg();
    k.v_lshrrev_b32(r_reg, 1, fk);

    let rp1 = k.alloc_vreg();
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    let tri = k.alloc_vreg();
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(tri) });
    k.branch_vccz("r_ok");
    k.v_add_u32(r_reg, r_reg, neg1);
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.label("r_ok");

    let neg_tri = k.alloc_vreg();
    k.push(Op::VXorB32 { dst: neg_tri, src0: Operand::VReg(tri), src1: Operand::VReg(neg1) });
    let one_v = k.alloc_vreg();
    k.v_mov_imm(one_v, 1);
    k.v_add_u32(neg_tri, neg_tri, one_v);
    let c_reg = k.alloc_vreg();
    k.v_add_u32(c_reg, tid, neg_tri);

    let packed = k.alloc_vreg();
    k.v_lshlrev_b32(packed, 8, r_reg);
    k.v_or_b32(packed, Operand::VReg(packed), Operand::VReg(c_reg));

    let lut_addr = k.alloc_vreg();
    k.v_lshlrev_b32(lut_addr, 1, tid);
    let lut_base_v = k.alloc_vreg();
    k.v_mov_imm(lut_base_v, lut_base as i32);
    k.v_add_u32(lut_addr, lut_addr, lut_base_v);
    k.push(Op::DsStoreB16 { vaddr: lut_addr, src: packed, offset: 0 });
    k.label("lut_done");
    k.wait_lgkmcnt(0);

    // ── Phase A: ZCLT-load C to LDS (132B stride) ──
    let v_row = k.alloc_vreg();
    k.v_lshrrev_b32(v_row, 1, tid);
    let col_grp = k.alloc_vreg();
    k.v_and_b32_imm(col_grp, tid, 1);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 6, col_grp);

    let lds_row = if c_chunk < 256 {
        let clamped = k.alloc_vreg();
        k.v_and_b32_imm(clamped, v_row, c_chunk - 1);
        clamped
    } else { v_row };

    let v_off = k.alloc_vreg();
    k.v_mov_from_sgpr(v_off, base_row);
    k.v_add_u32(v_off, v_off, lds_row);
    k.v_lshlrev_b32(v_off, 7, v_off);
    k.v_add_u32(v_off, v_off, col_bytes);

    let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(c_addr, SReg(c_ptr.0));
    k.v_mov_from_sgpr(VReg(c_addr.0 + 1), SReg(c_ptr.0 + 1));
    k.v_add_co(c_addr, c_addr, v_off);
    k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));

    let c_data = k.alloc_vreg_array(16, Alignment::Align4);
    for i in 0..4u32 {
        k.global_load(VReg(c_data.0 + i * 4), c_addr, Width::B128, (i * 16) as i32);
    }

    let s132 = k.alloc_sreg();
    k.s_mov_imm(s132, 132);
    let stride132 = k.alloc_vreg();
    k.v_mov_from_sgpr(stride132, s132);
    let lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(lds_wr, lds_row, stride132);
    k.v_add_u32(lds_wr, lds_wr, col_bytes);
    k.wait_vmcnt(0);
    for i in 0..4u32 {
        k.ds_store_b128(lds_wr, VReg(c_data.0 + i * 4), (i * 16) as u16);
    }
    k.wait_lgkmcnt(0);
    k.s_barrier();

    // ── Prepare Out base ──
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let o_base_lo = k.alloc_sreg();
    let o_base_hi = k.alloc_sreg();
    let tmp_off = k.alloc_sreg();
    k.s_lshl_b32(tmp_off, base_row, 8);
    k.s_add_u32_ss(o_base_lo, SReg(out_ptr.0), tmp_off);
    k.s_addc_u32_imm(o_base_hi, SReg(out_ptr.0 + 1), 0);

    // ── Main loop setup ──
    let k_task = k.alloc_sreg();
    k.push(Op::SMov { dst: k_task, src: SOperand::SReg(wave_id_s) });
    let cur_a = k.alloc_sreg();
    k.s_mov_imm(cur_a, -1i32);
    let s_total = k.alloc_sreg();
    k.s_mov_imm(s_total, total_tiles as i32);
    k.s_cmp_ge_u32(k_task, s_total);
    k.branch_scc1("epilogue");

    let a_frag = k.alloc_vreg_array(wmma.frag_span(4), wmma.frag_align);
    let b_frag = k.alloc_vreg_array(wmma.frag_span(4), wmma.frag_align);
    let p_acc = k.alloc_vreg_array(wmma.acc_regs, wmma.acc_align);
    let o_acc = k.alloc_vreg_array(wmma.acc_span(4), wmma.acc_align);
    let p_trans = k.alloc_vreg_array(wmma.frag_regs, wmma.frag_align);
    let c_tile = k.alloc_vreg_array(wmma.frag_regs, wmma.frag_align);

    k.label("main_loop");

    // ── LUT lookup ──
    let lut_off = k.alloc_sreg();
    k.s_lshl_b32(lut_off, k_task, 1);
    let lut_base_s = k.alloc_sreg();
    k.s_mov_imm(lut_base_s, lut_base as i32);
    k.s_add_u32_ss(lut_off, lut_off, lut_base_s);
    let lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(lut_v, lut_off);
    let lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: lut_val, vaddr: lut_v, offset: 0 });
    k.wait_lgkmcnt(0);

    let lut_r_v = k.alloc_vreg();
    k.v_lshrrev_b32(lut_r_v, 8, lut_val);
    let lut_r = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: lut_r, src: lut_r_v });
    let lut_c_v = k.alloc_vreg();
    k.v_and_b32_imm(lut_c_v, lut_val, 0xFF);
    let lut_c = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: lut_c, src: lut_c_v });

    // For upper triangle: A_tile = lut_c, B_tile = lut_r
    // For lower triangle: A_tile = lut_r, B_tile = lut_c
    let a_tile = k.alloc_sreg();
    let b_tile = k.alloc_sreg();
    if is_upper {
        k.push(Op::SMov { dst: a_tile, src: SOperand::SReg(lut_c) });
        k.push(Op::SMov { dst: b_tile, src: SOperand::SReg(lut_r) });
    } else {
        k.push(Op::SMov { dst: a_tile, src: SOperand::SReg(lut_r) });
        k.push(Op::SMov { dst: b_tile, src: SOperand::SReg(lut_c) });
    }

    // ── Conditional A_slice load ──
    k.push(Op::SCmpEqU32 { src0: a_tile, src1: SOperand::SReg(cur_a) });
    k.branch_scc1("skip_a_load");
    k.push(Op::SMov { dst: cur_a, src: SOperand::SReg(a_tile) });

    let a_row = k.alloc_sreg();
    k.s_lshl_b32(a_row, a_tile, 4);
    k.s_add_u32_ss(a_row, a_row, base_row);
    let a_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(a_row_v, a_row);
    k.v_add_u32(a_row_v, a_row_v, lane_row);
    k.v_lshlrev_b32(a_row_v, 7, a_row_v);
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_addr.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_addr, a_addr, a_row_v);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
    load_b128_span(&mut k, a_frag, a_addr, wmma.frag_span(4));
    zero_vreg_span(&mut k, o_acc, wmma.acc_span(4));
    k.wait_vmcnt(0);
    k.label("skip_a_load");

    // ── Load B_tile ──
    let b_row = k.alloc_sreg();
    k.s_lshl_b32(b_row, b_tile, 4);
    k.s_add_u32_ss(b_row, b_row, base_row);
    let b_off = k.alloc_sreg();
    k.s_lshl_b32(b_off, b_row, 7);
    let b_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(b_off_v, 7, lane_row);
    let b_off_s_v = k.alloc_vreg();
    k.v_mov_from_sgpr(b_off_s_v, b_off);
    k.v_add_u32(b_off_v, b_off_v, b_off_s_v);
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_addr.0 + 1), SReg(b_ptr.0 + 1));
    k.v_add_co(b_addr, b_addr, b_off_v);
    k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
    load_b128_span(&mut k, b_frag, b_addr, wmma.frag_span(4));
    zero_vreg_span(&mut k, p_acc, wmma.acc_regs);
    k.wait_vmcnt(0);

    // ── WMMA: P^T = B @ A^T ──
    for kg in 0..4u32 {
        k.wmma_bf16_f32(
            p_acc,
            VReg(b_frag.0 + kg * wmma.frag_regs),
            VReg(a_frag.0 + kg * wmma.frag_regs),
            p_acc,
        );
    }

    // ── Causal mask (diagonal only: lut_r == lut_c) ──
    k.push(Op::SCmpEqU32 { src0: lut_r, src1: SOperand::SReg(lut_c) });
    k.branch_scc0("mask_done");

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let zero_f = k.alloc_vreg();
    k.v_mov_imm(zero_f, 0);
    let one_f = k.alloc_vreg();
    k.v_mov_imm(one_f, 0x3F800000u32 as i32);

    for vk in 0..wmma.acc_regs {
        let row_v = k.alloc_vreg();
        if vk == 0 {
            k.push(Op::VMov { dst: row_v, src: Operand::VReg(lane_half) });
        } else {
            k.v_mov_imm(row_v, (2 * vk) as i32);
            k.v_add_u32(row_v, lane_half, row_v);
        }
        if is_upper {
            // Upper: keep where col >= row in P^T → row_v >= lane_row
            k.v_cmp_ge_i32(row_v, lane_row);
        } else {
            // Lower: keep where col <= row in P^T → lane_row >= row_v
            k.v_cmp_ge_i32(lane_row, row_v);
        }
        k.push(Op::VCndmaskB32 { dst: row_v, src_false: Operand::VReg(zero_f), src_true: Operand::VReg(one_f) });
        k.push(Op::VMulF32 { dst: VReg(p_acc.0 + vk), src0: Operand::VReg(VReg(p_acc.0 + vk)), src1: Operand::VReg(row_v) });
    }
    k.label("mask_done");
    k.clear_vcc(); // prevent VCC residual from mask cmp affecting v_add_co_ci

    // ── C-layout → A-operand transpose (pure VALU, no LDS) ──
    k.reg_transpose_c_to_ab(p_trans, p_acc, c_tile);

    // ── C-tile from LDS and WMMA (4 v-groups) ──
    let lds_read_base = k.alloc_vreg();
    k.v_and_b32_imm(lds_read_base, lane_row, 15);
    k.v_lshlrev_b32(lds_read_base, 1, lds_read_base);
    let s2112 = k.alloc_sreg();
    k.s_mov_imm(s2112, 2112);
    let c_lds_base = k.alloc_sreg();
    k.s_mul_i32(c_lds_base, b_tile, s2112);

    for vg in 0..4u32 {
        let col_off = vg * wmma.frag_bytes();
        let lds_off = k.alloc_sreg();
        k.s_mov_imm(lds_off, col_off as i32);
        k.s_add_u32_ss(lds_off, c_lds_base, lds_off);
        let lds_addr = k.alloc_vreg();
        k.v_mov_from_sgpr(lds_addr, lds_off);
        k.v_add_u32(lds_addr, lds_read_base, lds_addr);

        ds_load_bf16_fragment(&mut k, c_tile, lds_addr, wmma.frag_regs, 0, 132);
        k.wait_lgkmcnt(0);
        k.wmma_bf16_f32(
            VReg(o_acc.0 + vg * wmma.acc_regs),
            p_trans,
            c_tile,
            VReg(o_acc.0 + vg * wmma.acc_regs),
        );
    }

    // ── Flush check ──
    let next_k = k.alloc_sreg();
    k.s_add_u32(next_k, k_task, 16);
    k.s_cmp_ge_u32(next_k, s_total);
    k.branch_scc1("flush");

    let next_lut_off = k.alloc_sreg();
    k.s_lshl_b32(next_lut_off, next_k, 1);
    k.s_add_u32_ss(next_lut_off, next_lut_off, lut_base_s);
    let next_lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(next_lut_v, next_lut_off);
    let next_lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: next_lut_val, vaddr: next_lut_v, offset: 0 });
    k.wait_lgkmcnt(0);

    // Determine next A_tile
    let next_a_v = k.alloc_vreg();
    if is_upper {
        k.v_and_b32_imm(next_a_v, next_lut_val, 0xFF); // next_lut_c = A_tile for upper
    } else {
        k.v_lshrrev_b32(next_a_v, 8, next_lut_val); // next_lut_r = A_tile for lower
    }
    let next_a = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: next_a, src: next_a_v });
    k.push(Op::SCmpEqU32 { src0: next_a, src1: SOperand::SReg(a_tile) });
    k.branch_scc1("skip_flush");

    // ── Flush ──
    k.label("flush");
    let flush_lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(flush_lane_half, 4, lane_id);
    let a16 = k.alloc_sreg();
    k.s_lshl_b32(a16, a_tile, 4);
    let a16_v = k.alloc_vreg();
    k.v_mov_from_sgpr(a16_v, a16);
    k.v_add_u32(a16_v, a16_v, flush_lane_half);
    let row_off = k.alloc_vreg();
    k.v_lshlrev_b32(row_off, 8, a16_v);
    let col_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(col_off_v, 2, lane_row);

    let o_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_addr, o_base_lo);
    k.v_mov_from_sgpr(VReg(o_addr.0 + 1), o_base_hi);
    k.v_add_co(o_addr, o_addr, row_off);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));
    k.v_add_co(o_addr, o_addr, col_off_v);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));

    for vg in 0..4u32 {
        for vk in 0..wmma.acc_regs {
            let off = (vk as i32) * 512 + (vg as i32) * 64;
            k.global_atomic_add_f32(o_addr, VReg(o_acc.0 + vg * wmma.acc_regs + vk), off);
        }
    }
    k.wait_vmcnt(0);
    k.label("skip_flush");

    k.s_add_u32(k_task, k_task, 16);
    k.s_cmp_lt_u32(k_task, s_total);
    k.branch_scc1("main_loop");

    k.label("epilogue");
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}



// ============================================================================
// OCPA State Update: W_c = K_c^T @ V_c  (ZCLT + WMMA)
// ============================================================================

/// Generate OCPA state update kernel using ZCLT (Zero-Conflict LDS Transpose).
///
/// Computes 64×64 outer product W_c = K_c^T @ V_c per chunk per head.
/// ZCLT: 132-byte padded LDS stride + bf16 column tearing (zero-VALU transpose).
///
/// Kernargs (40 bytes):
///   K_ptr(0), V_ptr(8), W_ptr(16), C_chunk(24), d_head(28),
///   seq_len(32), n_chunks(36)
/// Grid: (n_chunks, num_heads, 1), WG: 32
pub fn ocpa_state_update() -> T0Kernel {
    let mut k = T0Kernel::new("t0_ocpa_state_update");
    k.set_lds_size(4224); // ZCLT 金库: 16 rows × 132 bytes × 2 matrices

    // Kernargs
    let k_ptr = k.arg_ptr("K");          // [0:7]
    let v_ptr = k.arg_ptr("V");          // [8:15]
    let w_ptr = k.arg_ptr("W");          // [16:23]
    let c_chunk = k.arg_u32("C_chunk");  // [24:27]
    let d_head = k.arg_u32("d_head");    // [28:31]
    let seq_len = k.arg_u32("seq_len");  // [32:35]
    let n_chunks = k.arg_u32("n_chunks"); // [36:39]
    k.emit_arg_loads();

    // TGID.x = chunk_id, TGID.y = head_id
    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0); // hardware thread_id
    let wmma = Bf16F32WmmaShape::current();

    // ── Pointer math: K/V base = ptr + (head_id*seq_len + chunk_id*C_chunk) * 128 ──
    let row_start = k.alloc_sreg();
    k.s_mul_i32(row_start, head_id, SReg(seq_len.0));
    let tmp_s = k.alloc_sreg();
    k.s_mul_i32(tmp_s, chunk_id, SReg(c_chunk.0));
    k.s_add_u32_ss(row_start, row_start, tmp_s);

    // offset_lo = row_start << 7 (128 bytes per row for d_head=64 bf16)
    // offset_hi = row_start >> 25
    let off_lo = k.alloc_sreg();
    let off_hi = k.alloc_sreg();
    k.s_lshl_b32(off_lo, row_start, 7);
    k.s_lshr_b32(off_hi, row_start, 25);

    // K_base = K_ptr + offset
    let k_base_lo = k.alloc_sreg();
    let k_base_hi = k.alloc_sreg();
    k.s_add_u32_ss(k_base_lo, SReg(k_ptr.0), off_lo);
    k.s_addc_u32(k_base_hi, SReg(k_ptr.0 + 1), off_hi);

    // V_base = V_ptr + offset
    let v_base_lo = k.alloc_sreg();
    let v_base_hi = k.alloc_sreg();
    k.s_add_u32_ss(v_base_lo, SReg(v_ptr.0), off_lo);
    k.s_addc_u32(v_base_hi, SReg(v_ptr.0 + 1), off_hi);

    // Thread byte offset = thread_id * 64
    let t_off = k.alloc_vreg();
    k.v_lshlrev_b32(t_off, 6, tid);

    // K_addr = K_base + thread_offset
    let k_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(k_addr, k_base_lo);
    k.v_mov_from_sgpr(VReg(k_addr.0 + 1), k_base_hi);
    k.v_add_co(k_addr, k_addr, t_off);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));

    // V_addr = V_base + thread_offset
    let v_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(v_addr, v_base_lo);
    k.v_mov_from_sgpr(VReg(v_addr.0 + 1), v_base_hi);
    k.v_add_co(v_addr, v_addr, t_off);
    k.v_add_co_ci(VReg(v_addr.0 + 1), VReg(v_addr.0 + 1));

    // ── ZCLT LDS topology: 132-byte padded stride ──
    // r = thread_id / 2, c = thread_id % 2
    let r_v = k.alloc_vreg();
    k.v_lshrrev_b32(r_v, 1, tid);
    let c_v = k.alloc_vreg();
    k.v_and_b32_imm(c_v, tid, 1);
    k.v_lshlrev_b32(c_v, 6, c_v); // c_bytes = c * 64

    // K_lds_write = r * 132 + c_bytes
    let stride_132 = k.alloc_vreg();
    let s132 = k.alloc_sreg();
    k.s_mov_imm(s132, 132);
    k.v_mov_from_sgpr(stride_132, s132);
    let k_lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(k_lds_wr, r_v, stride_132);
    k.v_add_u32(k_lds_wr, k_lds_wr, c_v);

    // V_lds_write = K_lds_write + 2112
    let v_lds_wr = k.alloc_vreg();
    let s2112 = k.alloc_sreg();
    k.s_mov_imm(s2112, 2112);
    let v2112 = k.alloc_vreg();
    k.v_mov_from_sgpr(v2112, s2112);
    k.v_add_u32(v_lds_wr, k_lds_wr, v2112);

    // LDS read base = (thread_id % 16) * 2
    let lds_read_base = k.alloc_vreg();
    k.v_and_b32_imm(lds_read_base, tid, 15);
    k.v_lshlrev_b32(lds_read_base, 1, lds_read_base);

    // HBM step = 16 rows × 128 bytes = 2048
    let hbm_step = k.alloc_vreg();
    let s2048 = k.alloc_sreg();
    k.s_mov_imm(s2048, 2048);
    k.v_mov_from_sgpr(hbm_step, s2048);

    // ── Big VGPR arrays (allocated AFTER individual VGPRs to avoid VReg(0) conflict) ──
    let acc = k.alloc_vreg_array(wmma.acc_span(16), wmma.acc_align);
    let a_regs = k.alloc_vreg_array(wmma.frag_span(4), wmma.frag_align);
    let b_regs = k.alloc_vreg_array(wmma.frag_span(4), wmma.frag_align);

    // Zero accumulators
    zero_vreg_span(&mut k, acc, wmma.acc_span(16));

    // ════════════════════════════════════════════════════════════════
    // Main loop: process C_chunk/16 iterations
    // ════════════════════════════════════════════════════════════════
    let m_idx = k.alloc_sreg();
    k.s_mov_imm(m_idx, 0);
    k.label("m_loop");

    // A. Load K → a_regs[0:15], V → b_regs[0:15] (reuse, will be overwritten by LDS reads)
    for i in 0..4u32 {
        k.global_load(VReg(a_regs.0 + i * 4), k_addr, Width::B128, (i * 16) as i32);
    }
    for i in 0..4u32 {
        k.global_load(VReg(b_regs.0 + i * 4), v_addr, Width::B128, (i * 16) as i32);
    }

    // B. Advance HBM pointers (hide latency)
    k.v_add_co(k_addr, k_addr, hbm_step);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));
    k.v_add_co(v_addr, v_addr, hbm_step);
    k.v_add_co_ci(VReg(v_addr.0 + 1), VReg(v_addr.0 + 1));

    k.wait_vmcnt(0);

    // C. Write to LDS from a_regs/b_regs (data will be overwritten by column tearing reads next)
    for i in 0..4u32 {
        k.ds_store_b128(k_lds_wr, VReg(a_regs.0 + i * 4), (i * 16) as u16);
    }
    for i in 0..4u32 {
        k.ds_store_b128(v_lds_wr, VReg(b_regs.0 + i * 4), (i * 16) as u16);
    }
    k.wait_lgkmcnt(0);

    // D. Column tearing: transpose read via ds_load_u16_d16/hi
    // K^T/V fragments use target-specific BF16 WMMA fragment width.
    for g in 0..4u32 {
        ds_load_bf16_fragment(
            &mut k,
            VReg(a_regs.0 + g * wmma.frag_regs),
            lds_read_base,
            wmma.frag_regs,
            g * wmma.frag_bytes(),
            132,
        );
    }
    for v in 0..4u32 {
        ds_load_bf16_fragment(
            &mut k,
            VReg(b_regs.0 + v * wmma.frag_regs),
            lds_read_base,
            wmma.frag_regs,
            2112 + v * wmma.frag_bytes(),
            132,
        );
    }
    k.wait_lgkmcnt(0);

    // E. 16× WMMA: K^T[g] × V[v] → acc[g*4+v]
    for g in 0..4u32 {
        for v in 0..4u32 {
            let acc_base = VReg(acc.0 + g * (4 * wmma.acc_regs) + v * wmma.acc_regs);
            let a_base = VReg(a_regs.0 + g * wmma.frag_regs);
            let b_base = VReg(b_regs.0 + v * wmma.frag_regs);
            k.wmma_bf16_f32(acc_base, a_base, b_base, acc_base);
        }
    }

    // F. Loop control
    k.s_add_u32(m_idx, m_idx, 16);
    k.s_cmp_lt_u32(m_idx, SReg(c_chunk.0));
    k.branch_scc1("m_loop");

    // ════════════════════════════════════════════════════════════════
    // Store 64×64 result to W[head][chunk]
    // ════════════════════════════════════════════════════════════════
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    // W_base = W_ptr + (head_id * n_chunks + chunk_id) * d_head^2 * 4
    let w_idx = k.alloc_sreg();
    k.s_mul_i32(w_idx, head_id, SReg(n_chunks.0));
    k.s_add_u32_ss(w_idx, w_idx, chunk_id);
    let d_sq = k.alloc_sreg();
    k.s_mul_i32(d_sq, SReg(d_head.0), SReg(d_head.0));
    let w_off = k.alloc_sreg();
    k.s_mul_i32(w_off, w_idx, d_sq);
    k.s_lshl_b32(w_off, w_off, 2); // × 4 bytes

    let w_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(w_addr, SReg(w_ptr.0));
    k.v_mov_from_sgpr(VReg(w_addr.0 + 1), SReg(w_ptr.0 + 1));
    let w_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(w_off_v, w_off);
    k.v_add_co(w_addr, w_addr, w_off_v);
    k.v_add_co_ci(VReg(w_addr.0 + 1), VReg(w_addr.0 + 1));

    // For each k_grp (0..3) × v_tile (0..3), store 8 rows
    let d_v = k.alloc_vreg();
    k.v_mov_from_sgpr(d_v, SReg(d_head.0));

    let st_addr = k.alloc_vreg_array(2, Alignment::Align2);

    // Pre-allocate scratch VGPRs ONCE (reuse dead a_regs slots after main loop)
    let scratch_base_row = VReg(a_regs.0);       // reuse dead a_regs[0]
    let scratch_row_off = VReg(a_regs.0 + 1);    // reuse dead a_regs[1]
    let scratch_imm = VReg(a_regs.0 + 2);        // reuse dead a_regs[2]

    // Pre-compute lane_row * 4
    let lr_off = VReg(a_regs.0 + 3);             // reuse dead a_regs[3]
    k.v_lshlrev_b32(lr_off, 2, lane_row);

    for k_grp in 0..4u32 {
        // base_row = lane_half + k_grp * 16
        if k_grp == 0 {
            k.push(Op::VMov { dst: scratch_base_row, src: Operand::VReg(lane_half) });
        } else {
            k.v_mov_imm(scratch_imm, (k_grp * 16) as i32);
            k.v_add_u32(scratch_base_row, lane_half, scratch_imm);
        }
        // row_byte_offset = base_row * d_head * 4
        k.v_mul_lo_u32(scratch_row_off, scratch_base_row, d_v);
        k.v_lshlrev_b32(scratch_row_off, 2, scratch_row_off);

        for v_tile in 0..4u32 {
            let acc_base_idx = acc.0 + k_grp * (4 * wmma.acc_regs) + v_tile * wmma.acc_regs;
            let col_offset = v_tile * 16 * 4;

            // st_addr = w_addr + row_offset + lr_off + col_offset
            k.push(Op::VMov { dst: st_addr, src: Operand::VReg(w_addr) });
            k.push(Op::VMov { dst: VReg(st_addr.0 + 1), src: Operand::VReg(VReg(w_addr.0 + 1)) });
            k.v_add_co(st_addr, st_addr, scratch_row_off);
            k.v_add_co_ci(VReg(st_addr.0 + 1), VReg(st_addr.0 + 1));
            k.v_add_u32(st_addr, st_addr, lr_off);

            if col_offset > 0 {
                k.v_mov_imm(scratch_imm, col_offset as i32);
                k.v_add_u32(st_addr, st_addr, scratch_imm);
            }

            // Store WMMA result rows for one accumulator fragment.
            for r in 0..wmma.acc_regs {
                let r_offset = (r as i32) * 512;
                k.global_store(st_addr, VReg(acc_base_idx + r), Width::B32, r_offset);
            }
        }
    }

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}



// ══════════════════════════════════════════════════════════════
//  Row-wise reductions: each WG (32 threads) processes one row
//  Kernarg: [in_ptr(u64), out_ptr(u64), cols(u32)]
//  Grid: [n_rows, 1, 1]  WG: [32, 1, 1]
// ══════════════════════════════════════════════════════════════

/// Row-wise sum: out[row] = sum(in[row, 0..cols])
///
/// Algorithm:
/// 1. wg_id = row index
/// 2. Loop stride=32: each lane loads in[row, col_base + lane_id]
/// 3. EXEC mask for bounds (col < cols)
/// 4. Accumulate v_add_f32
/// 5. wave_reduce_add_f32 → lane 0 has total
/// 6. Lane 0 stores out[row]
pub fn t0_row_reduce_sum() -> T0Kernel {
    t0_row_reduce_generic(RowReduceOp::Sum)
}



/// Row-wise max: out[row] = max(in[row, 0..cols])
pub fn t0_row_reduce_max() -> T0Kernel {
    t0_row_reduce_generic(RowReduceOp::Max)
}



/// Row-wise sum of squares: out[row] = sum(in[row, :]²)
/// Used by RMSNorm backward.
pub fn t0_row_reduce_sum_sq() -> T0Kernel {
    t0_row_reduce_generic(RowReduceOp::SumSq)
}



// ══════════════════════════════════════════════════════════════
//  Row-wise broadcast ops: out[r,c] = x[r,c] op vec[r]
//  Kernarg: [x_ptr(u64), vec_ptr(u64), out_ptr(u64), cols(u32)]
//  Grid: [n_rows * 32, 1, 1]  WG: [32, 1, 1]
// ══════════════════════════════════════════════════════════════

/// Row broadcast sub: out[r,c] = x[r,c] - vec[r]
/// Used in Softmax: shifted = x - row_max(x)
pub fn t0_row_broadcast_sub() -> T0Kernel {
    t0_row_broadcast_generic(RowBroadcastOp::Sub)
}



/// Row broadcast div: out[r,c] = x[r,c] / vec[r]
/// Used in Softmax: softmax = exp_shifted / row_sum(exp_shifted)
pub fn t0_row_broadcast_div() -> T0Kernel {
    t0_row_broadcast_generic(RowBroadcastOp::Div)
}



/// Row broadcast exp(sub): out[r,c] = exp(x[r,c] - vec[r])
/// Fused shift+exp for Softmax (saves one kernel launch + memory pass)
pub fn t0_row_broadcast_exp_sub() -> T0Kernel {
    t0_row_broadcast_generic(RowBroadcastOp::ExpSub)
}


enum RowReduceOp { Sum, Max, SumSq }



fn t0_row_reduce_generic(op: RowReduceOp) -> T0Kernel {
    use super::ir::*;

    let name = match op {
        RowReduceOp::Sum   => "t0_row_reduce_sum",
        RowReduceOp::Max   => "t0_row_reduce_max",
        RowReduceOp::SumSq => "t0_row_reduce_sum_sq",
    };
    let mut k = T0Kernel::new(name);
    k.set_wg_size(32); // single wave per WG

    // ── Kernarg: [in_ptr(u64), out_ptr(u64), cols(u32)] ──
    let in_ptr = k.arg_ptr("in");
    let out_ptr = k.arg_ptr("out");
    let cols_arg = k.arg_u32("cols");
    k.emit_arg_loads();

    // wg_id = row index
    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // row byte offset = wg_id * cols * 4
    let row_byte_off = k.alloc_sreg();
    k.s_mul_i32(row_byte_off, wg_id, SReg(cols_arg.0));
    k.s_lshl_b32(row_byte_off, row_byte_off, 2); // * 4 bytes

    // in_row_ptr = in_ptr + row_byte_off
    let in_row_lo = k.alloc_sreg();
    let in_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(in_row_lo, SReg(in_ptr.0), row_byte_off);
    k.s_addc_u32_imm(in_row_hi, SReg(in_ptr.0 + 1), 0);

    // ── Initialize accumulator ──
    let acc = k.alloc_vreg();
    match op {
        RowReduceOp::Sum | RowReduceOp::SumSq => k.v_mov_imm(acc, 0), // 0.0
        RowReduceOp::Max => k.v_mov_imm(acc, 0xFF800000u32 as i32),    // -inf
    }

    // ── Main loop: stride=32 over columns ──
    let loop_base = k.alloc_sreg();
    k.s_mov_imm(loop_base, 0);
    let loop_label = k.make_label("row_loop");
    k.label(&loop_label);

    // col = loop_base + lane_id
    let col = k.alloc_vreg();
    k.v_mov_from_sgpr(col, loop_base);
    k.v_add_u32(col, col, lane_id);

    // Bounds check: col < cols (EXEC mask)
    let cols_v = k.alloc_vreg();
    k.v_mov_from_sgpr(cols_v, SReg(cols_arg.0));
    let saved_exec = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(col), src1: Operand::VReg(cols_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    // Load in[row, col]
    let byte_col = k.alloc_vreg();
    k.v_lshlrev_b32(byte_col, 2, col); // col * 4
    let load_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(load_addr, in_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), in_row_hi);
    k.v_add_co(load_addr, load_addr, byte_col);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    let val = k.alloc_vreg();
    k.global_load(val, load_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // Accumulate
    match op {
        RowReduceOp::Sum => k.v_add_f32(acc, acc, val),
        RowReduceOp::Max => k.v_max_f32(acc, acc, val),
        RowReduceOp::SumSq => {
            let sq = k.alloc_vreg();
            k.v_mul_f32(sq, val, val);
            k.v_add_f32(acc, acc, sq);
        }
    }

    // Restore EXEC
    k.push(Op::RestoreExec { src: saved_exec });

    // loop_base += 32
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(cols_arg.0));
    k.branch_scc1(&loop_label);

    // ── Wave32 butterfly reduce ──
    let tmp = k.alloc_vreg();
    match op {
        RowReduceOp::Sum | RowReduceOp::SumSq => k.wave_reduce_add_f32(acc, tmp),
        RowReduceOp::Max => k.wave_reduce_max_f32(acc, tmp),
    }

    // ── Lane 0 writes out[row] ──
    // byte_off = wg_id * 4
    let out_byte = k.alloc_sreg();
    k.s_lshl_b32(out_byte, wg_id, 2);
    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0));
    k.v_mov_from_sgpr(VReg(out_addr.0 + 1), SReg(out_ptr.0 + 1));
    let out_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(out_off_v, out_byte);
    k.v_add_co(out_addr, out_addr, out_off_v);
    k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));

    // Only lane 0 writes
    let exec_save2 = k.alloc_sreg();
    k.push(Op::VCmpEqU32Imm { src: lane_id, imm: 0 });
    k.push(Op::SaveExec { dst: exec_save2 });
    k.global_store(out_addr, acc, Width::B32, 0);
    k.push(Op::RestoreExec { src: exec_save2 });

    k.wait_vmcnt(0); k.wait_vscnt(0);
    k.endpgm();
    k
}


enum RowBroadcastOp { Sub, Div, ExpSub }



fn t0_row_broadcast_generic(op: RowBroadcastOp) -> T0Kernel {
    use super::ir::*;

    let name = match op {
        RowBroadcastOp::Sub    => "t0_row_broadcast_sub",
        RowBroadcastOp::Div    => "t0_row_broadcast_div",
        RowBroadcastOp::ExpSub => "t0_row_broadcast_exp_sub",
    };
    let mut k = T0Kernel::new(name);
    k.set_wg_size(32);

    // Kernarg: [x_ptr(u64), vec_ptr(u64), out_ptr(u64), cols(u32)]
    let x_ptr = k.arg_ptr("x");
    let vec_ptr = k.arg_ptr("vec");
    let out_ptr = k.arg_ptr("out");
    let cols_arg = k.arg_u32("cols");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // row byte offset = wg_id * cols * 4
    let row_byte_off = k.alloc_sreg();
    k.s_mul_i32(row_byte_off, wg_id, SReg(cols_arg.0));
    k.s_lshl_b32(row_byte_off, row_byte_off, 2);

    // x_row = x_ptr + row_byte_off
    let x_row_lo = k.alloc_sreg();
    let x_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(x_row_lo, SReg(x_ptr.0), row_byte_off);
    k.s_addc_u32_imm(x_row_hi, SReg(x_ptr.0 + 1), 0);

    // out_row = out_ptr + row_byte_off
    let out_row_lo = k.alloc_sreg();
    let out_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(out_row_lo, SReg(out_ptr.0), row_byte_off);
    k.s_addc_u32_imm(out_row_hi, SReg(out_ptr.0 + 1), 0);

    // Load vec[row] — scalar, same for all lanes
    let vec_byte_off = k.alloc_sreg();
    k.s_lshl_b32(vec_byte_off, wg_id, 2); // wg_id * 4
    let vec_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(vec_addr, SReg(vec_ptr.0));
    k.v_mov_from_sgpr(VReg(vec_addr.0 + 1), SReg(vec_ptr.0 + 1));
    let vec_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(vec_off_v, vec_byte_off);
    k.v_add_co(vec_addr, vec_addr, vec_off_v);
    k.v_add_co_ci(VReg(vec_addr.0 + 1), VReg(vec_addr.0 + 1));
    let row_scalar = k.alloc_vreg();
    k.global_load(row_scalar, vec_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // For div: pre-compute rcp(vec[row]) = 1/vec[row]
    if matches!(op, RowBroadcastOp::Div) {
        k.v_rcp_f32(row_scalar, row_scalar);
    }

    // For exp_sub: pre-compute log2(e) constant for v_exp_f32 (which computes 2^x)
    // IMPORTANT: only allocate this VReg when needed!
    // An unused VReg has first_use=MAX which triggers incorrect expiry of
    // row_scalar in the register allocator's liveness analysis.
    let log2e = if matches!(op, RowBroadcastOp::ExpSub) {
        let v = k.alloc_vreg();
        k.v_mov_imm(v, 0x3FB8AA3Bu32 as i32); // log2(e) = 1.4426950..
        v
    } else {
        VReg(0) // dummy, never used
    };

    // ── Main loop: stride=32 ──
    let loop_base = k.alloc_sreg();
    k.s_mov_imm(loop_base, 0);
    let loop_label = k.make_label("bcast_loop");
    k.label(&loop_label);

    let col = k.alloc_vreg();
    k.v_mov_from_sgpr(col, loop_base);
    k.v_add_u32(col, col, lane_id);

    // Bounds check
    let cols_v = k.alloc_vreg();
    k.v_mov_from_sgpr(cols_v, SReg(cols_arg.0));
    let saved_exec = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(col), src1: Operand::VReg(cols_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    // Load x[row, col]
    let byte_col = k.alloc_vreg();
    k.v_lshlrev_b32(byte_col, 2, col);
    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(x_addr.0 + 1), x_row_hi);
    k.v_add_co(x_addr, x_addr, byte_col);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));
    let x_val = k.alloc_vreg();
    k.global_load(x_val, x_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // Apply operation
    let result = k.alloc_vreg();
    match op {
        RowBroadcastOp::Sub => k.v_sub_f32(result, x_val, row_scalar),
        RowBroadcastOp::Div => k.v_mul_f32(result, x_val, row_scalar), // x * rcp(vec)
        RowBroadcastOp::ExpSub => {
            k.v_sub_f32(result, x_val, row_scalar);     // x - max
            k.v_mul_f32(result, result, log2e);          // (x - max) * log2(e)
            k.v_exp_f32(result, result);                 // 2^((x-max)*log2(e)) = e^(x-max)
        }
    }

    // Store out[row, col]
    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, out_row_lo);
    k.v_mov_from_sgpr(VReg(out_addr.0 + 1), out_row_hi);
    k.v_add_co(out_addr, out_addr, byte_col);
    k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));
    k.global_store(out_addr, result, Width::B32, 0);

    // Restore EXEC
    k.push(Op::RestoreExec { src: saved_exec });

    // loop_base += 32
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(cols_arg.0));
    k.branch_scc1(&loop_label);

    k.wait_vmcnt(0); k.wait_vscnt(0);
    k.endpgm();
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocpa_forward_intra_gfx1201_wmma_operands() {
        let kernel = with_target_context(Target::GFX1201, || ocpa_forward_intra_with_c(64));
        let asm = kernel.to_assembly(Target::GFX1201).expect("GFX1201 OCPA forward assembly failed");
        let wmma_line = asm.lines()
            .find(|line| line.contains("v_wmma_f32_16x16x16_bf16"))
            .expect("missing WMMA instruction");
        let spans: Vec<u32> = wmma_line
            .split("v[")
            .skip(1)
            .map(|part| {
                let regs = part.split(']').next().expect("unterminated register range");
                let (lo, hi) = regs.split_once(':').expect("expected register range");
                hi.parse::<u32>().unwrap() - lo.parse::<u32>().unwrap() + 1
            })
            .collect();
        assert_eq!(spans, vec![8, 4, 4, 8], "unexpected GFX1201 WMMA operand widths");
    }
}
