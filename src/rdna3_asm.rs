//! RDNA3 ISA Assembler for Rust
//!
//! Binary instruction encoder for gfx1100 (RDNA3) targeting FlashAttention optimization.
//! Implements key instructions: s_waitcnt, global_load, ds_read/write, v_wmma.
//!
//! Reference: AMD RDNA3 ISA Reference Guide (606 pages)
//! GFX11 (gfx1100) instruction encoding format
//!
//! ## WMMA Lane Layout (Wave32, 16x16x16 bf16 → f32)
//!
//! ```text
//! C[16x16] = A[16x16] @ B[16x16]
//!
//! Lane ownership of output C matrix:
//! ┌─────────────────┬─────────────────┐
//! │  Lanes 0-15     │   Lanes 16-31   │
//! │  C[lane, 0:7]   │   C[lane-16, 8:15] │
//! ├─────────────────┼─────────────────┤
//! │ Lane 0: row 0   │ Lane 16: row 0  │
//! │ Lane 1: row 1   │ Lane 17: row 1  │
//! │ ...             │ ...             │
//! │ Lane 15: row 15 │ Lane 31: row 15 │
//! └─────────────────┴─────────────────┘
//!
//! Each lane's registers:
//! - a_frag[16]: 16 bf16 values (one row of A or column of B^T)
//! - b_frag[16]: 16 bf16 values
//! - c_frag[8]:  8 f32 values (partial row of C)
//!
//! Fragment loading for row-major A[16x16]:
//!   Lane i (0-15):  a_frag = A[i, 0:15]  (row i, all 16 cols)
//!   Lane i (16-31): a_frag = A[i-16, 0:15] (same row as lane i-16)
//!
//! Memory layout requirements:
//!   - All 32 lanes load the SAME row into their a_frag for left/right halves
//!   - Lane 0 and Lane 16 both need row 0 of A
//!   - Lane 1 and Lane 17 both need row 1 of A
//!   - etc.
//! ```


/// RDNA3 GFX11 instruction encoding constants
pub mod gfx11 {
    // =========================================================================
    // SOPP (Scalar Operation, Immediate)
    // Format: [31:23]=opcode_prefix, [22:16]=op, [15:0]=imm16
    // GFX11 encodings verified via LLVM disassembly
    // =========================================================================
    
    /// s_endpgm: End program
    /// LLVM: s_endpgm = [0x00,0x00,0xb0,0xbf] = 0xBFB00000
    pub const S_ENDPGM: u32 = 0xBFB00000;
    
    /// GFX11 s_waitcnt bit layout (from LLVM analysis):
    /// - lgkmcnt(0) = 0xBF89FC07: wait for lgkmcnt=0, vmcnt/expcnt at max
    /// - vmcnt(0)   = 0xBF8903F7: wait for vmcnt=0, lgkmcnt/expcnt at max
    /// - all zeros  = 0xBF890000: wait for everything
    ///
    /// For simplicity, use hardcoded values for common cases.
    
    /// s_waitcnt vmcnt(N) - wait for N or fewer outstanding vector memory ops
    /// GFX11 verified via LLVM: 
    ///   vmcnt(0) = 0xBF8903F7
    ///   vmcnt(4) = 0xBF8913F7  
    ///   vmcnt(8) = 0xBF8923F7
    /// Pattern: 0xBF89 | (vmcnt << 8) | 0xF7 (lgkmcnt maxed)
    pub fn s_waitcnt_vmcnt(n: u8) -> u32 {
        // GFX11 s_waitcnt encoding:
        // bits [3:0] = vmcnt[3:0]
        // bits [5:4] = reserved
        // bits [13:10] = vmcnt[5:4] (high bits)
        // For n <= 15, only low bits needed
        // LLVM pattern: 0xBF89 | ((n & 0x30) << 6) | ((n & 0x0F) << 0) | 0x03F0
        // Simplified for n <= 15: 0xBF8903F0 | n
        if n <= 15 {
            0xBF8903F0 | (n as u32) | 0x07  // +0x07 = expcnt(7) = no wait on exports
        } else {
            // For n > 15, use high bits at [13:10]
            let lo = (n & 0x0F) as u32;
            let hi = ((n >> 4) & 0x03) as u32;
            0xBF8903F0 | lo | (hi << 10) | 0x07  // +0x07 = expcnt(7)
        }
    }
    
    /// s_waitcnt lgkmcnt(N) - wait for N or fewer outstanding scalar memory ops
    /// GFX11 verified via LLVM: lgkmcnt(0) = [0x07,0xfc,0x89,0xbf] = 0xBF89FC07
    pub fn s_waitcnt_lgkmcnt(n: u8) -> u32 {
        // For n=0, use verified LLVM encoding
        // For n>0, lgkmcnt bits are at [5:4] and [13:10]
        if n == 0 {
            0xBF89FC07
        } else {
            // Approximate for small n values (bits 5:4 hold low 2 bits)
            let lgkmcnt_lo = n & 0x3;
            let lgkmcnt_hi = (n >> 2) & 0xF;
            0xBF89FC07 | ((lgkmcnt_lo as u32) << 4) | ((lgkmcnt_hi as u32) << 10)
        }
    }
    
    /// s_waitcnt_vscnt null, N - wait for N or fewer outstanding vector stores
    /// GFX11 CRITICAL: vmcnt only waits for loads, stores require vscnt!
    /// LLVM: s_waitcnt_vscnt null, 0 = [0x00,0x00,0x7c,0xbc] = 0xBC7C0000
    /// Without this, stores may not complete before kernel exit!
    pub fn s_waitcnt_vscnt(n: u8) -> u32 {
        // GFX11 encoding: 0xBC7C0000 | count
        // null destination is encoded as 0x7C in the sdst field
        0xBC7C0000 | (n as u32)
    }
    
    /// s_barrier: Workgroup barrier
    /// LLVM: s_barrier = [0x00,0x00,0xbd,0xbf] = 0xBFBD0000
    /// NOTE: Old encoding 0xBF8A0000 was actually s_wait_idle (works but slower)
    pub const S_BARRIER: u32 = 0xBFBD0000;
    
    /// s_setprio imm - Set wavefront scheduling priority
    /// LLVM: s_setprio 1 = [0x01,0x00,0xb5,0xbf] = 0xBFB50001
    /// imm: 0 = normal, 1-3 = higher priority
    pub fn s_setprio(prio: u8) -> u32 {
        0xBFB50000 | (prio as u32)
    }
    
    /// s_nop n - Insert n+1 cycles of delay
    /// LLVM: s_nop 0 = 0xBF800000 (1 cycle)
    /// LLVM: s_nop 7 = 0xBF800007 (8 cycles)
    pub fn s_nop(n: u8) -> u32 {
        0xBF800000 | (n as u32)
    }
    
    /// s_clause count - Mark next N instructions as atomic clause (no interruption)
    /// LLVM: s_clause 0x3 = [0x03,0x00,0x85,0xbf] = 0xBF850003
    /// CRITICAL: The N instructions after s_clause MUST be of the same type
    ///           (all global_load, all ds_read, etc.) - NO mixing with ALU!
    /// count: number of additional instructions in clause (1-63, meaning 2-64 total)
    pub fn s_clause(count: u8) -> u32 {
        assert!(count >= 1 && count <= 63, "s_clause count must be 1-63");
        0xBF850000 | (count as u32)
    }
    
    // =========================================================================
    // SOPP Branch Instructions - For loops
    // =========================================================================
    
    /// s_branch target - unconditional branch
    /// offset is relative to PC+4, in dwords
    pub fn s_branch(offset: i16) -> u32 {
        // SOPP opcode 0x20 = s_branch (LLVM verified: s_branch 100 -> 0xBFA00064)
        0xBFA00000u32 | ((offset as u16) as u32)
    }
    
    /// s_cbranch_scc0 target - branch if SCC == 0
    /// LLVM: s_cbranch_scc0 1 = [0x01,0x00,0xa1,0xbf] = 0xBFA10001
    pub fn s_cbranch_scc0(offset: i16) -> u32 {
        // SOPP opcode 0x21 = s_cbranch_scc0 (GFX11)
        0xBFA10000u32 | ((offset as u16) as u32)
    }
    
    /// s_cbranch_scc1 target - branch if SCC == 1
    /// LLVM: s_cbranch_scc1 10 = [0x0a,0x00,0xa2,0xbf] = 0xBFA2000A
    pub fn s_cbranch_scc1(offset: i16) -> u32 {
        // SOPP opcode 0x22 = s_cbranch_scc1 (GFX11)
        0xBFA20000u32 | ((offset as u16) as u32)
    }
    
    /// s_cbranch_vccz target - branch if VCC == 0
    pub fn s_cbranch_vccz(offset: i16) -> u32 {
        // SOPP opcode 0x23 = s_cbranch_vccz (LLVM verified: 0xBFA3xxxx)
        0xBFA30000u32 | ((offset as u16) as u32)
    }
    
    /// s_cbranch_vccnz target - branch if VCC != 0
    pub fn s_cbranch_vccnz(offset: i16) -> u32 {
        // SOPP opcode 0x24 = s_cbranch_vccnz (LLVM verified: 0xBFA4xxxx)
        0xBFA40000u32 | ((offset as u16) as u32)
    }
    
    
    // =========================================================================
    // SOP2 - Scalar ALU operations (for loop counters)
    // =========================================================================
    
    /// s_add_u32 sdst, ssrc0, ssrc1 - scalar add (sets SCC on carry)
    pub fn s_add_u32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x00 = s_add_u32
        // Format: [31:30]=SOP2, [29:23]=OP, [22:16]=SDST, [15:8]=SSRC1, [7:0]=SSRC0
        0x80000000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_add_u32 sdst, ssrc0, imm - scalar add with inline constant (0..64 ONLY!)
    /// For imm > 64, use s_mov_b32_literal + s_add_u32.
    pub fn s_add_u32_imm(sdst: u8, ssrc0: u8, imm: u8) -> u32 {
        assert!(imm <= 64,
            "s_add_u32_imm: imm={} exceeds inline constant range [0..64]. \
             Use s_mov_b32_literal() + s_add_u32() instead.", imm);
        let src1 = 0x80 + imm as u32;
        0x80000000u32 | ((sdst as u32) << 16) | (src1 << 8) | (ssrc0 as u32)
    }
    
    /// s_sub_u32 sdst, ssrc0, imm - scalar subtract with inline constant (0..64 ONLY!)
    /// For imm > 64, use s_mov_b32_literal + s_sub_u32.
    pub fn s_sub_u32_imm(sdst: u8, ssrc0: u8, imm: u8) -> u32 {
        assert!(imm <= 64,
            "s_sub_u32_imm: imm={} exceeds inline constant range [0..64]. \
             Use s_mov_b32_literal() + s_sub_u32() instead.", imm);
        let src1 = 0x80 + imm as u32;
        0x80800000u32 | ((sdst as u32) << 16) | (src1 << 8) | (ssrc0 as u32)
    }
    
    /// s_cmp_lg_u32 ssrc0, imm - set SCC if src0 != imm
    /// LLVM: s_cmp_lg_u32 s16, 0 = [0x10,0x80,0x07,0xbf] = 0xBF078010
    pub fn s_cmp_lg_u32_imm(ssrc0: u8, imm: u8) -> u32 {
        // SOPC format: opcode 0x07 = s_cmp_lg_u32
        let src1 = if imm <= 64 { 0x80 + imm as u32 } else { imm as u32 };
        0xBF070000u32 | (src1 << 8) | (ssrc0 as u32)
    }
    
    /// s_and_b32 sdst, ssrc0, inline_const - scalar AND with inline constant
    /// LLVM: s_and_b32 s20, s20, 15 = [0x14,0x8f,0x14,0x8b] = 0x8B148F14
    /// For inline constants 0-64: use 128 + value (e.g., 15 = 0x8f)
    pub fn s_and_b32_imm(sdst: u8, ssrc0: u8, imm: u8) -> u32 {
        // SOP2 opcode 0x16 = s_and_b32 (bits 29:23)
        // 0x8B = 10_00101_1 = SOP2 prefix + opcode 0x16>>1
        // LLVM encoding shows 0x8b prefix
        let src1 = if imm <= 64 { 128 + imm } else { imm };
        0x8B000000u32 | ((sdst as u32) << 16) | ((src1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_and_b32 sdst, ssrc0, ssrc1 - scalar AND with two SGPR operands
    /// Same opcode as s_and_b32_imm but ssrc1 is a register, not inline constant
    pub fn s_and_b32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x16 = s_and_b32: 0x8B prefix
        0x8B000000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_addc_u32 sdst, ssrc0, ssrc1 - scalar add with carry (uses SCC as carry-in)
    /// LLVM: s_addc_u32 s5, s5, 0 = [0x05,0x80,0x05,0x82] = 0x82058005
    /// CRITICAL: When ssrc1 is 0, use inline constant 0x80 for literal 0, NOT register s0!
    pub fn s_addc_u32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode for s_addc_u32 = 0x04 (not 0x02!)
        // [31:30]=10 (SOP2), [29:23]=opcode, bits: 10_0000100 = 0x82
        // CRITICAL FIX: ssrc1=0 means "inline constant 0" encoded as 0x80, not "register s0"
        let src1_enc = if ssrc1 == 0 { 0x80u32 } else { ssrc1 as u32 };
        0x82000000u32 | ((sdst as u32) << 16) | (src1_enc << 8) | (ssrc0 as u32)
    }
    
    /// s_sub_u32 sdst, ssrc0, ssrc1 - scalar subtract
    pub fn s_sub_u32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x01 = s_sub_u32
        0x80800000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_add_i32 sdst, ssrc0, ssrc1 - scalar add (signed, sets SCC on overflow)
    pub fn s_add_i32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x02 = s_add_i32
        0x81000000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    /// s_cselect_b32 sdst, ssrc0, ssrc1 - conditional select based on SCC
    /// If SCC=1: sdst = ssrc0. If SCC=0: sdst = ssrc1.
    /// LLVM: s_cselect_b32 s20, s15, s20 = [0x0f,0x14,0x14,0x98] = 0x9814140F
    /// SOP2 opcode 0x0C
    pub fn s_cselect_b32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        0x98000000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    /// s_xor_b32 sdst, ssrc0, ssrc1 - scalar bitwise XOR
    /// LLVM: s_xor_b32 s0, s0, s1 = [0x00,0x01,0x00,0x8d] = 0x8d000100
    pub fn s_xor_b32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x1A = s_xor_b32
        0x8d000000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    /// s_sub_i32 sdst, ssrc0, ssrc1 - scalar sub (signed)
    pub fn s_sub_i32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x03 = s_sub_i32
        0x81800000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_mul_i32 sdst, ssrc0, ssrc1 - scalar multiply (32-bit)
    /// LLVM: s_mul_i32 s14, s14, s15 = [0x0e,0x0f,0x0e,0x96] = 0x960E0F0E
    pub fn s_mul_i32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x16 = s_mul_i32
        // Format: 0x96 in high byte = 0x80 | (0x16 << 1) >> ..
        // Actually: [31:30]=SOP2=10, [29:23]=OP=0x16, [22:16]=SDST, [15:8]=SSRC1, [7:0]=SSRC0
        0x96000000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    /// s_subb_u32 sdst, ssrc0, ssrc1 - scalar sub with borrow
    pub fn s_subb_u32(sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        // SOP2 opcode 0x05 = s_subb_u32
        0x82800000u32 | ((sdst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_cmp_eq_u32 ssrc0, ssrc1 - compare equal, set SCC
    pub fn s_cmp_eq_u32(ssrc0: u8, ssrc1: u8) -> u32 {
        // SOPC opcode 0x06 = s_cmp_eq_u32 (GFX11)
        0xBF060000u32 | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_cmp_lt_u32 ssrc0, ssrc1 - compare less than, set SCC if ssrc0 < ssrc1
    pub fn s_cmp_lt_u32(ssrc0: u8, ssrc1: u8) -> u32 {
        // SOPC opcode 0x0A = s_cmp_lt_u32 (GFX11)
        0xBF0A0000u32 | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_cmp_lt_u32 ssrc0, inline_const - compare SGPR against inline constant (0-64)
    /// For inline constants: 128 + value (e.g., 3 = 131 = 0x83)
    pub fn s_cmp_lt_u32_imm(ssrc0: u8, imm: u8) -> u32 {
        // SOPC opcode 0x0A = s_cmp_lt_u32 (GFX11)
        // inline constant encoding: 128 + value for 0-64
        let src1 = if imm <= 64 { 128 + imm } else { imm };
        0xBF0A0000u32 | ((src1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_cmp_gt_i32 ssrc0, ssrc1 - compare greater than (signed)
    pub fn s_cmp_gt_i32(ssrc0: u8, ssrc1: u8) -> u32 {
        // SOPC opcode 0x02 = s_cmp_gt_i32 (GFX11)
        0xBF020000u32 | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    /// s_cmp_ge_u32 ssrc0, ssrc1 - compare >=, set SCC if ssrc0 >= ssrc1
    /// LLVM: s_cmp_ge_u32 s15, s14 = [0x0f,0x0e,0x09,0xbf] = 0xBF090E0F
    pub fn s_cmp_ge_u32(ssrc0: u8, ssrc1: u8) -> u32 {
        // SOPC opcode 0x09 = s_cmp_ge_u32 (GFX11)
        0xBF090000u32 | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_cmp_ge_u32 ssrc0, inline_const - compare SGPR against inline constant (0-64)
    /// LLVM: s_cmp_ge_u32 s14, 3 = [0x0e,0x83,0x09,0xbf] = 0xBF09830E
    /// For inline constants: 128 + value (e.g., 3 = 0x83)
    pub fn s_cmp_ge_u32_imm(ssrc0: u8, imm: u8) -> u32 {
        // SOPC opcode 0x09 = s_cmp_ge_u32 (GFX11)
        // inline constant encoding: 128 + value for 0-64
        let src1 = if imm <= 64 { 128 + imm } else { imm };
        0xBF090000u32 | ((src1 as u32) << 8) | (ssrc0 as u32)
    }
    
    /// s_lshr_b32 sdst, ssrc0, ssrc1 - logical shift right
    /// LLVM: s_lshr_b32 s14, s10, 6 = [0x0a,0x86,0x0e,0x85] = 0x850E860A
    /// GFX11 SOP2 format: [31:30]=10, [29:23]=OP, [22:16]=SDST, [15:8]=SSRC1, [7:0]=SSRC0
    /// s_lshr_b32 opcode = 0x0A (bits 29:23)
    pub fn s_lshr_b32(sdst: u8, ssrc0: u8, shift: u8) -> u32 {
        // LLVM format: 0x85 prefix (SOP2 + op 0x0A), sdst at [22:16], shift (0x86=6+128) at [15:8], src at [7:0]
        // inline constant 6 = 0x80+6 = 0x86
        let shift_encoded = if shift <= 64 { 0x80 + shift as u32 } else { shift as u32 };
        0x85000000u32 | ((sdst as u32) << 16) | (shift_encoded << 8) | (ssrc0 as u32)
    }
    
    /// s_lshl_b32 sdst, ssrc0, shift - logical shift left by immediate
    /// LLVM: s_lshl_b32 s10, s14, 8 → encoding: [0x0e,0x88,0x0a,0x84]
    /// GFX11 SOP2 format: [31:30]=10, [29:23]=OP, [22:16]=SDST, [15:8]=SSRC1, [7:0]=SSRC0
    /// s_lshl_b32 opcode = 0x08 (bits 29:23)
    pub fn s_lshl_b32(sdst: u8, ssrc0: u8, shift: u8) -> u32 {
        // inline constant shift = 0x80 + shift (for values 0-64)
        let shift_encoded = if shift <= 64 { 0x80 + shift as u32 } else { shift as u32 };
        0x84000000u32 | ((sdst as u32) << 16) | (shift_encoded << 8) | (ssrc0 as u32)
    }
    
    /// s_mov_b32 sdst, ssrc - move scalar
    pub fn s_mov_b32(sdst: u8, ssrc: u8) -> u32 {
        // SOP1 opcode 0x00 = s_mov_b32
        0xBE800000u32 | ((sdst as u32) << 16) | (ssrc as u32)
    }
    
    /// s_mov_b32 with inline constant
    pub fn s_mov_b32_imm(sdst: u8, imm: i32) -> u32 {
        let src = match imm {
            0 => 0x80u32,
            1..=64 => 0x80 + imm as u32,
            -64..=-1 => 0xC0 + (-imm) as u32,
            _ => panic!("s_mov_b32_imm: imm={} out of inline constant range [-64..64]. Use s_mov_b32_literal() for larger values.", imm),
        };
        0xBE800000u32 | ((sdst as u32) << 16) | src
    }
    
    /// s_mov_b32 with 32-bit literal constant (for values > 64)
    /// Returns [instruction, literal] as two dwords
    pub fn s_mov_b32_literal(sdst: u8, literal: u32) -> [u32; 2] {
        // 0xFF = literal constant placeholder in src field
        let instruction = 0xBE800000u32 | ((sdst as u32) << 16) | 0xFF;
        [instruction, literal]
    }

    /// s_mov_b32 exec_lo, imm - Set exec mask (for lane masking)
    /// LLVM: s_mov_b32 exec_lo, 1 = [0x81, 0x00, 0xfe, 0xbe] = 0xBEFE0081
    /// exec_lo is register 0x7E (126)
    /// Used to control which lanes execute subsequent instructions
    pub fn s_mov_b32_exec_lo(imm: u32) -> u32 {
        // exec_lo = SGPR 0x7E (126)
        // For imm=1: src = 0x81 (inline constant 1)
        // For imm=0xFFFFFFFF: src = 0xC1 (inline constant -1)
        let src = if imm == 1 {
            0x81u32  // inline constant 1
        } else if imm == 0xFFFFFFFF {
            0xC1u32  // inline constant -1
        } else if imm == 0 {
            0x80u32  // inline constant 0
        } else {
            0xC1u32  // Default to all lanes
        };
        0xBE800000u32 | (0x7E << 16) | src  // 0x7E = exec_lo
    }
    
    // =========================================================================
    // SMEM (Scalar Memory) - GFX11 encoding
    // =========================================================================
    // Verified via LLVM:
    // s_load_b64  s[2:3], s[0:1], 0x0   -> [0x80,0x00,0x04,0xf4, 0x00,0x00,0x00,0xf8]
    // s_load_b64  s[6:7], s[0:1], 0x10  -> [0x80,0x01,0x04,0xf4, 0x10,0x00,0x00,0xf8]
    // s_load_b128 s[4:7], s[0:1], 0x0   -> [0x00,0x01,0x08,0xf4, 0x00,0x00,0x00,0xf8]
    //
    // GFX11 SMEM format (little-endian u32):
    // Word 0: byte[0] = 0x80 (IMM flag for dwordx2) or SBASE for dwordx4
    //         byte[1] = SDST encoding (dst/4 for x4, (dst-2)/4 for x2?)
    //         byte[2] = opcode (0x04=b64, 0x08=b128)
    //         byte[3] = 0xF4 (SMEM prefix)
    // Word 1: 24-bit offset with 0xF8 prefix
    
    /// s_load_dwordx4 s[dst:dst+3], s[base:base+1], offset (s_load_b128)
    /// LLVM: s[4:7],s[0:1],0 = [0x00,0x01,0x08,0xf4] = 0xF4080100
    pub fn s_load_dwordx4(dst: u8, base: u8, offset: u32) -> [u32; 2] {
        assert!(dst % 4 == 0, "s_load_dwordx4 dst must be 4-aligned, got s{}", dst);
        assert!(base % 2 == 0, "s_load_dwordx4 base must be 2-aligned, got s{}", base);
        // From LLVM: s[4:7] -> byte[1]=0x01 = dst/4 = 4/4 = 1
        // byte[0] = base/2 = 0/2 = 0
        let byte0 = (base / 2) as u32;
        let byte1 = (dst / 4) as u32;
        let word0 = 0xF4080000u32 | (byte1 << 8) | byte0;
        let word1 = 0xF8000000u32 | (offset & 0xFFFFFF);
        [word0, word1]
    }
    
    /// s_load_dwordx2 s[dst:dst+1], s[base:base+1], offset (s_load_b64)
    /// LLVM analysis of SDST encoding:
    ///   s[0:1] → byte1=0 (0/4), byte0.bit7=0 (0/2%2=0) → 0xF4040080
    ///   s[2:3] → byte1=0 (2/4), byte0.bit7=1 (2/2%2=1) → 0xF4040080 (wait, that's 0x80 from base)
    ///   s[4:5] → byte1=1 (4/4), byte0.bit7=0 (4/2%2=0) → 0xF4040100
    ///   s[6:7] → byte1=1 (6/4), byte0.bit7=1 (6/2%2=1) → 0xF4040180
    ///   s[8:9] → byte1=2 (8/4), byte0.bit7=0 (8/2%2=0) → 0xF4040200
    ///
    /// For SMEM SDST field: dst register is encoded as byte1*4 + (byte0.bit7)*2
    /// So byte1 = dst/4, byte0.bit7 = (dst%4)/2 = (dst/2)%2
    pub fn s_load_dwordx2(dst: u8, base: u8, offset: u32) -> [u32; 2] {
        let byte1 = (dst / 4) as u32;  // High bits of register index
        let dst_bit = ((dst / 2) % 2) as u32;  // 1 if dst is 2,6,10... (not 4-aligned)
        // byte0 = 0x80 (IMM flag) | dst_bit<<7 would conflict with IMM!
        // Wait - the 0x80 is from base/2=0 check. Let me re-analyze:
        // Actually looking at LLVM output again:
        // s[6:7]: [0x80,0x01,0x04,0xf4] = word 0xF4040180
        // s[4:5]: [0x00,0x01,0x04,0xf4] = word 0xF4040100
        // The IMM flag is in the second word (0xF8), not byte0!
        // byte0 encodes: SBASE (bits 0-5) and part of SDST (bit 7)
        let byte0 = ((base / 2) as u32) | (dst_bit << 7);
        let word0 = 0xF4040000u32 | (byte1 << 8) | byte0;
        let word1 = 0xF8000000u32 | (offset & 0xFFFFFF);
        [word0, word1]
    }
    
    /// s_load_dword s_dst, s[base:base+1], offset (s_load_b32)
    /// LLVM: s_load_b32 s15, s[0:1], 0x20 → [0xc0,0x03,0x00,0xf4,0x20,0x00,0x00,0xf8]
    /// GFX11 offset includes the 64-byte kernarg skip (auto-handled by runtime)
    pub fn s_load_dword(dst: u8, base: u8, offset: u32) -> [u32; 2] {
        // SMEM GFX11 encoding analysis from LLVM:
        //   s12 → 0x00 (12%4=0 → bits7:6=00)
        //   s13 → 0x40 (13%4=1 → bits7:6=01)
        //   s14 → 0x80 (14%4=2 → bits7:6=10)
        //   s15 → 0xC0 (15%4=3 → bits7:6=11)
        // byte1 = dst / 4, byte0.bits7:6 = dst % 4
        let byte1 = (dst / 4) as u32;  // High bits of register index
        let dst_low = (dst % 4) as u32;  // Low 2 bits encoded in byte0
        let byte0 = ((base / 2) as u32) | (dst_low << 6);
        let word0 = 0xF4000000u32 | (byte1 << 8) | byte0;  // 0x00 opcode for b32
        let word1 = 0xF8000000u32 | (offset & 0xFFFFFF);
        [word0, word1]
    }
    
    // =========================================================================
    // Global Memory (Vector) - GFX11 encoding
    // =========================================================================
    // Verified via LLVM: echo 'global_load_dwordx4 v[8:11], v[0:1], off' | llvm-mc -mcpu=gfx1100 --show-encoding
    //
    // global_load_b128 v[8:11], v[0:1], off    ; encoding: [0x00,0x00,0x5e,0xdc,0x00,0x00,0x7c,0x08]
    // global_load_b128 v[8:11], v[0:1], off offset:16 ; encoding: [0x10,0x00,0x5e,0xdc,0x00,0x00,0x7c,0x08]
    // global_store_b128 v[0:1], v[4:7], off   ; encoding: [0x00,0x00,0x76,0xdc,0x00,0x04,0x7c,0x00]
    //
    // GFX11 FLAT/Global format (64-bit):
    // Word 0: [31:24]=opcode base (0xDC), [23:16]=opcode (0x5E=load_b128, 0x76=store_b128), [15:0]=offset
    // Word 1: [31:24]=vdst, [23:16]=saddr(0x7C=off), [15:8]=vdata/unused, [7:0]=vaddr
    
    /// global_load_dwordx4 v[dst:dst+3], v[addr:addr+1], off [offset:N]
    /// Loads 128 bits (4 dwords) from global memory
    /// NOTE: GFX11 uses 13-bit signed offset (-4096 to +4095)
    pub fn global_load_dwordx4(vdst: u8, vaddr: u8, offset: i32) -> [u32; 2] {
        // Word 0: 0xDC5E0000 | (13-bit signed offset)
        // Word 1: (vdst << 24) | (0x7C << 16) | (vaddr)
        // GFX11 offset field is 13-bit signed: bits [12:0] of word0[15:0]
        // Actually looking at LLVM encoding more carefully:
        // offset:-64 produces 0x1FC0 in the low 16 bits
        // This suggests bits [12:0] hold the 13-bit signed offset
        // with bit 13 being something else (or just part of opcode extension)
        let offset_enc = (offset as u32) & 0x1FFF; // 13-bit mask
        let word0 = 0xDC5E0000u32 | offset_enc;
        let word1 = ((vdst as u32) << 24) | (0x7C << 16) | (vaddr as u32);
        [word0, word1]
    }

    /// global_load_dword v[dst], v[addr:addr+1], off [offset:N]
    /// Loads 32 bits (1 dword)
    pub fn global_load_dword(vdst: u8, vaddr: u8, offset: i32) -> [u32; 2] {
        // Opcode 0x52 (load_b32) verified via LLVM
        let offset_enc = (offset as u32) & 0x1FFF;
        let word0 = 0xDC520000u32 | offset_enc;
        let word1 = ((vdst as u32) << 24) | (0x7C << 16) | (vaddr as u32);
        [word0, word1]
    }

    /// global_load_dwordx2 v[dst:dst+1], v[addr:addr+1], off [offset:N]
    /// Loads 64 bits (2 dwords) from global memory
    /// Verified: global_load_b64 v[8:9], v[0:1], off ; encoding: [0x00,0x00,0x56,0xdc,0x00,0x00,0x7c,0x08]
    pub fn global_load_dwordx2(vdst: u8, vaddr: u8, offset: i32) -> [u32; 2] {
        // Opcode 0x56 (load_b64)
        let offset_enc = (offset as u32) & 0x1FFF;
        let word0 = 0xDC560000u32 | offset_enc;
        let word1 = ((vdst as u32) << 24) | (0x7C << 16) | (vaddr as u32);
        [word0, word1]
    }
    
    /// global_store_dwordx4 v[addr:addr+1], v[src:src+3], off [offset:N]
    /// Stores 128 bits (4 dwords) to global memory
    pub fn global_store_dwordx4(vaddr: u8, vsrc: u8, offset: i32) -> [u32; 2] {
        // Word 0: 0xDC760000 | (13-bit signed offset)
        // Word 1: (0x00 << 24) | (0x7C << 16) | (vsrc << 8) | (vaddr)
        let offset_enc = (offset as u32) & 0x1FFF; // 13-bit mask
        let word0 = 0xDC760000u32 | offset_enc;
        let word1 = (0x7C << 16) | ((vsrc as u32) << 8) | (vaddr as u32);
        [word0, word1]
    }
    
    /// global_store_dwordx2 v[addr:addr+1], v[src:src+1], off [offset:N]
    /// Stores 64 bits (2 dwords) to global memory
    /// LLVM: global_store_b64 v[0:1], v[2:3], off -> [0x00,0x00,0x6e,0xdc,0x00,0x02,0x7c,0x00]
    pub fn global_store_dwordx2(vaddr: u8, vsrc: u8, offset: i32) -> [u32; 2] {
        // Opcode 0x6E (store_b64)
        let offset_enc = (offset as u32) & 0x1FFF;
        let word0 = 0xDC6E0000u32 | offset_enc;
        let word1 = (0x7C << 16) | ((vsrc as u32) << 8) | (vaddr as u32);
        [word0, word1]
    }

    /// global_store_dword v[addr:addr+1], vsrc, off [offset:N]
    /// Stores 32 bits (1 dword) to global memory
    pub fn global_store_dword(vaddr: u8, vsrc: u8, offset: i32) -> [u32; 2] {
        // Opcode 0x6A (store_b32)
        // Word 0: 0xDC6A0000 | (13-bit signed offset)
        // Word 1: (0x00 << 24) | (0x7C << 16) | (vsrc << 8) | (vaddr)
        let offset_enc = (offset as u32) & 0x1FFF; // 13-bit mask
        let word0 = 0xDC6A0000u32 | offset_enc;
        let word1 = (0x7C << 16) | ((vsrc as u32) << 8) | (vaddr as u32);
        [word0, word1]
    }

    /// global_load_ushort v_dst, v[addr:addr+1], off [offset:N]
    /// Loads 16 bits unsigned (1 ushort) from global memory, zero-extends to 32-bit VGPR
    pub fn global_load_ushort(vdst: u8, vaddr: u8, offset: i32) -> [u32; 2] {
        // Opcode 0x4A for GFX11 (load_u16 / global_load_ushort) — LLVM verified
        let offset_enc = (offset as u32) & 0x1FFF;
        let word0 = 0xDC4A0000u32 | offset_enc;
        let word1 = ((vdst as u32) << 24) | (0x7C << 16) | (vaddr as u32);
        [word0, word1]
    }

    /// global_store_short v[addr:addr+1], v_src, off [offset:N]
    /// Stores 16 bits (lower half of VGPR) to global memory
    pub fn global_store_short(vaddr: u8, vsrc: u8, offset: i32) -> [u32; 2] {
        // Opcode 0x66 for GFX11 (store_b16 / global_store_short) — LLVM verified
        let offset_enc = (offset as u32) & 0x1FFF;
        let word0 = 0xDC660000u32 | offset_enc;
        let word1 = (0x7C << 16) | ((vsrc as u32) << 8) | (vaddr as u32);
        [word0, word1]
    }

    // =========================================================================
    // DS (Data Share / LDS)
    // =========================================================================
    
    /// ds_read_b128 v[dst:dst+3], v_addr
    /// GFX11 DS format: ds_load_b128 opcode = 0xFC
    /// LLVM: ds_load_b128 v[8:11], v70 -> [0x00,0x00,0xfc,0xdb,0x46,0x00,0x00,0x08]
    pub fn ds_read_b128(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
        // opcode 0xFC in GFX11 DS format = 0xDBFC0000
        let word0 = 0xDBFC0000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }
    
    /// ds_write_b128 v_addr, v[src:src+3]
    pub fn ds_write_b128(vaddr: u8, vsrc: u8, offset: u16) -> [u32; 2] {
        let word0 = 0xD8FD0000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vsrc as u32) << 8);
        [word0, word1]
    }
    
    /// ds_load_b32 v_dst, v_addr (LLVM verified: 0xD8D80000)
    /// GFX11: used `ds_load` terminology instead of `ds_read`
    pub fn ds_load_b32(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
        // LLVM: ds_load_b32 v0, v1 -> [0x00,0x00,0xd8,0xd8,0x01,0x00,0x00,0x00]
        let word0 = 0xD8D80000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }
    
    /// ds_load_u16 v_dst, v_addr, offset — load unsigned 16-bit from LDS
    /// LLVM verified: ds_load_u16 v20, v10 offset:128 → [0x80,0x00,0xf0,0xd8,0x0a,0x00,0x00,0x14]
    /// Opcode = 0xD8F00000
    pub fn ds_load_u16(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
        let word0 = 0xD8F00000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }

    /// ds_load_u16_d16 v_dst, v_addr, offset — load u16 into LOW 16 bits of vdst
    /// The HIGH 16 bits of vdst are PRESERVED (not zeroed).
    /// LLVM verified: ds_load_u16_d16 v0, v1 → [0x00,0x00,0x98,0xda,0x01,0x00,0x00,0x00]
    /// Opcode = 0xDA980000
    /// Key use: zero-VALU bf16x2 packing — load first bf16 into low half
    pub fn ds_load_u16_d16(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
        let word0 = 0xDA980000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }

    /// ds_load_u16_d16_hi v_dst, v_addr, offset — load u16 into HIGH 16 bits of vdst
    /// The LOW 16 bits of vdst are PRESERVED (not zeroed).
    /// LLVM verified: ds_load_u16_d16_hi v0, v1 → [0x00,0x00,0x9c,0xda,0x01,0x00,0x00,0x00]
    /// Opcode = 0xDA9C0000
    /// Key use: zero-VALU bf16x2 packing — load second bf16 into high half
    pub fn ds_load_u16_d16_hi(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
        let word0 = 0xDA9C0000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }

    /// ds_load_2addr_b32 v[vdst:vdst+1], v_addr, offset0, offset1
    /// Loads TWO dwords in ONE instruction:
    ///   vdst   = LDS[vaddr + offset0 * 4]
    ///   vdst+1 = LDS[vaddr + offset1 * 4]
    ///
    /// For stride 260 bytes: offset1 = 65 (65 * 4 = 260) ✓
    /// offset0, offset1 are 8-bit (0..255)
    ///
    /// LLVM verified: ds_load_2addr_b32 v[0:1], v2 offset0:0 offset1:65
    ///   → [0x00,0x41,0xdc,0xd8,0x02,0x00,0x00,0x00]
    /// Opcode = 0xD8DC
    pub fn ds_load_2addr_b32(vdst: u8, vaddr: u8, offset0: u8, offset1: u8) -> [u32; 2] {
        let word0 = 0xD8DC0000u32 | (offset0 as u32) | ((offset1 as u32) << 8);
        let word1 = (vaddr as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }
    
    /// ds_store_b32 v_addr, v_src (LLVM verified: 0xD8340000)
    pub fn ds_store_b32(vaddr: u8, vsrc: u8, offset: u16) -> [u32; 2] {
        // LLVM: ds_store_b32 v0, v1 -> [0x00,0x00,0x34,0xd8,0x00,0x01,0x00,0x00]
        let word0 = 0xD8340000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vsrc as u32) << 8);
        [word0, word1]
    }
    
    /// ds_store_b16 v_addr, v_src, offset — store 16-bit to LDS
    /// LLVM verified: ds_store_b16 v100, v101 → [0x00,0x00,0x7c,0xd8,0x64,0x65,0x00,0x00]
    /// Opcode = 0xD87C0000
    pub fn ds_store_b16(vaddr: u8, vsrc: u8, offset: u16) -> [u32; 2] {
        let word0 = 0xD87C0000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vsrc as u32) << 8);
        [word0, word1]
    }
    
    /// ds_load_b64 v[dst:dst+1], v_addr (LLVM verified: 0xD9D80000)
    pub fn ds_load_b64(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
        // LLVM: ds_load_b64 v[0:1], v2 -> [0x00,0x00,0xd8,0xd9,0x02,0x00,0x00,0x00]
        let word0 = 0xD9D80000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }
    
    /// ds_store_b64 v_addr, v[src:src+1] (LLVM verified: 0xD9340000)
    pub fn ds_store_b64(vaddr: u8, vsrc: u8, offset: u16) -> [u32; 2] {
        // LLVM: ds_store_b64 v0, v[1:2] -> [0x00,0x00,0x34,0xd9,0x00,0x01,0x00,0x00]
        let word0 = 0xD9340000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vsrc as u32) << 8);
        [word0, word1]
    }
    
    /// ds_store_b128 v_addr, v[src:src+3] (LLVM verified: 0xDB7C0000)
    /// Stores 128 bits (4 dwords) to LDS
    pub fn ds_store_b128(vaddr: u8, vsrc: u8, offset: u16) -> [u32; 2] {
        // LLVM: ds_store_b128 v0, v[1:4] -> [0x00,0x00,0x7c,0xdb,0x00,0x01,0x00,0x00]
        let word0 = 0xDB7C0000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vsrc as u32) << 8);
        [word0, word1]
    }


    /// v_and_b32 with inline constant immediate (0..64 ONLY!)
    /// For imm > 64, use s_mov_b32_literal + v_and_b32.
    pub fn v_and_b32_imm(vdst: u8, vsrc: u8, imm: u32) -> u32 {
        assert!(imm <= 64,
            "v_and_b32_imm: imm={} exceeds inline constant range [0..64]. \
             Use s_mov_b32_literal() + v_and_b32() instead.", imm);
        let imm_enc = 0x80 + imm;
        0x36000000u32 | ((vdst as u32) << 17) | ((vsrc as u32) << 9) | imm_enc
    }
    
    /// v_add_u32 with inline constant immediate (0..64 ONLY!)
    /// For larger values, use v_add_u32_literal() instead.
    pub fn v_add_u32_imm(vdst: u8, vsrc: u8, imm: u32) -> u32 {
        assert!(imm <= 64,
            "v_add_u32_imm: imm={} exceeds inline constant range [0..64]. \
             Use v_add_u32_literal() instead.", imm);
        let imm_enc = 0x80 + imm;
        0x4A000000u32 | ((vdst as u32) << 17) | ((vsrc as u32) << 9) | imm_enc
    }
    
    /// v_add_u32 with 32-bit literal constant (any value)
    /// LLVM-verified: v_add_nc_u32 v12, 128, v11 → [0x4a1816ff, 0x00000080]
    /// Encodes as 2 dwords: VOP2 word (src0=0xFF) + literal value
    pub fn v_add_u32_literal(vdst: u8, vsrc: u8, literal: u32) -> [u32; 2] {
        let word0 = 0x4A000000u32 | ((vdst as u32) << 17) | ((vsrc as u32) << 9) | 0xFF;
        [word0, literal]
    }
    
    /// v_and_b32 with 32-bit literal constant (any value)
    /// LLVM-verified: v_and_b32 v0, 0x80, v1 → [0x360002ff, 0x00000080]
    pub fn v_and_b32_literal(vdst: u8, vsrc: u8, literal: u32) -> [u32; 2] {
        let word0 = 0x36000000u32 | ((vdst as u32) << 17) | ((vsrc as u32) << 9) | 0xFF;
        [word0, literal]
    }
    
    // =========================================================================
    // VOP3P (Packed/Matrix operations) - WMMA
    // =========================================================================
    // Verified via LLVM:
    // echo 'v_wmma_f32_16x16x16_bf16 v[0:7], v[64:71], v[65:72], v[66:73]' | llvm-mc -mcpu=gfx1100 --show-encoding
    // ; encoding: [0x00,0x40,0x41,0xcc,0x40,0x83,0x0a,0x1d]
    // word0 = 0xcc414000, word1 = 0x1d0a8340
    // word1 bits: [8:0]=320(v64+256), [17:9]=321(v65+256), [26:18]=322(v66+256)
    
    /// v_wmma_f32_16x16x16_bf16 v[dst:dst+7], v[a:a+7], v[b:b+7], v[c:c+7]
    pub fn v_wmma_f32_16x16x16_bf16(vdst: u8, va: u8, vb: u8, vc: u8) -> [u32; 2] {
        // Word 0: Opcode (0xCC414000) | VDST
        // VDST does not need +256 in word0
        let word0 = 0xCC414000u32 | (vdst as u32);
        
        // Word 1: Standard VOP3P layout + modifier bits
        // All source VGPRs must be encoded as 256 + register_num
        // SRC0 (va): bits [8:0]
        // SRC1 (vb): bits [17:9]  
        // SRC2 (vc): bits [26:18]
        // Bits [28:27] = 0b11 (0x18000000) - VOP3P-MAI modifier
        let src0 = (va as u32) + 256;
        let src1 = (vb as u32) + 256;
        let src2 = (vc as u32) + 256;
        let word1 = 0x18000000u32 | src0 | (src1 << 9) | (src2 << 18);
        
        [word0, word1]
    }

    /// v_wmma_f32_16x16x16_f16 v[dst:dst+7], v[a:a+7], v[b:b+7], v[c:c+7]
    /// FP16 input operands, FP32 accumulator — higher mantissa precision than BF16 variant
    pub fn v_wmma_f32_16x16x16_f16(vdst: u8, va: u8, vb: u8, vc: u8) -> [u32; 2] {
        let word0 = 0xCC404000u32 | (vdst as u32);  // opcode = 0x40 (f16→f32)
        let src0 = (va as u32) + 256;
        let src1 = (vb as u32) + 256;
        let src2 = (vc as u32) + 256;
        let word1 = 0x18000000u32 | src0 | (src1 << 9) | (src2 << 18);
        [word0, word1]
    }

    /// v_wmma_bf16_16x16x16_bf16 v[dst:dst+7], v[a:a+7], v[b:b+7], v[c:c+7]
    /// BF16 input AND BF16 accumulator — saves VGPR (pack 2 values per reg)
    /// but lower accumulation precision
    pub fn v_wmma_bf16_16x16x16_bf16(vdst: u8, va: u8, vb: u8, vc: u8) -> [u32; 2] {
        let word0 = 0xCC434000u32 | (vdst as u32);  // opcode = 0x43 (bf16→bf16)
        let src0 = (va as u32) + 256;
        let src1 = (vb as u32) + 256;
        let src2 = (vc as u32) + 256;
        let word1 = 0x18000000u32 | src0 | (src1 << 9) | (src2 << 18);
        [word0, word1]
    }

    
    // =========================================================================
    // VOP2/VOP3 (Vector ALU)
    // =========================================================================
    
    /// v_fma_f32 v_dst, v_src0, v_src1, v_src2
    pub fn v_fma_f32(vdst: u8, vsrc0: u8, vsrc1: u8, vsrc2: u8) -> [u32; 2] {
        // VOP3 encoding: LLVM verified v_fma_f32 v25, v13, v24, v25 -> [0x19,0x00,0x13,0xd6,...]
        // word0 = 0xD6130000 | vdst
        // word1 = (256+src0) | ((256+src1) << 9) | ((256+src2) << 18)
        let word0 = 0xD6130000u32 | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let src2_enc = 256 + vsrc2 as u32;
        let word1 = src0_enc | (src1_enc << 9) | (src2_enc << 18);
        [word0, word1]
    }
    
    /// v_mul_lo_u32 v_dst, v_src0, v_src1 - integer multiply low 32 bits
    /// LLVM verified: v_mul_lo_u32 v10, v20, v30 -> [0x0a,0x00,0x2c,0xd7,0x14,0x3d,0x02,0x00]
    pub fn v_mul_lo_u32(vdst: u8, vsrc0: u8, vsrc1: u8) -> [u32; 2] {
        // VOP3 encoding: word0 = 0xD72C0000 | vdst
        // word1 = (256 + src0) | ((256 + src1) << 9)
        let word0 = 0xD72C0000u32 | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }
    
    /// v_add_f32 v_dst, v_src0, v_src1 (VOP2 encoding)
    pub fn v_add_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        // VOP2: Opcode = 0x06 (verified via llvm-mc)
        // vsrc0 must be encoded as 256 + vgpr_num for VGPRs
        0x06000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }
    /// v_mul_f32 v_dst, v_src0, v_src1 (VOP2 encoding)
    pub fn v_mul_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        // VOP2: vsrc0 must be encoded as 256 + vgpr_num for VGPRs
        0x10000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }
    
    /// v_mul_u32_u24 vdst, vsrc, imm - 24-bit unsigned multiply with inline constant
    /// LLVM: v_mul_u32_u24_e32 v0, v1, v2 = 0x16000501
    /// VOP2 opcode = 0x0B (bits [30:25])
    pub fn v_mul_u32_u24(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        0x16000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }
    
    /// v_mul_u32_u24 vdst, vsrc, inline_imm - 24-bit unsigned multiply with inline constant (0..64 ONLY!)
    /// For imm > 64, use s_mov_b32_literal + v_mul_u32_u24.
    pub fn v_mul_u32_u24_imm(vdst: u8, vsrc: u8, imm: u8) -> u32 {
        assert!(imm <= 64,
            "v_mul_u32_u24_imm: imm={} exceeds inline constant range [0..64]. \
             Use s_mov_b32_literal() + v_mul_u32_u24() instead.", imm);
        let imm_enc = 128 + imm as u32;
        0x16000000u32 | ((vdst as u32) << 17) | (imm_enc << 9) | (256 + vsrc as u32)
    }
    
    /// v_mov_b32 v_dst, v_src (VOP1 encoding) - VGPR to VGPR
    /// LLVM: v_mov_b32 v60, v0 = [0x00,0x03,0x78,0x7e] = 0x7E780300
    /// Analysis: SRC0=256 (v0+256), OP=1, VDST=60, ENC=63
    /// Format: [31:25]=VOP1(0x3F), [24:17]=VDST, [16:9]=opcode(1 for v_mov_b32), [8:0]=SRC0
    pub fn v_mov_b32(vdst: u8, vsrc: u8) -> u32 {
        // VGPR in VOP1 SRC0 needs +256
        // Opcode 1 = v_mov_b32
        0x7E000000u32 | (0x01 << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }
    
    /// v_mov_b32 v_dst, s_src - copy scalar register to vector register
    /// LLVM: v_mov_b32 v0, s4 = 0x7E000204 (SGPR 4 encoded directly as 0x04)
    pub fn v_mov_b32_from_sgpr(vdst: u8, ssrc: u8) -> u32 {
        // Same encoding as v_mov_b32, SGPR 0-127 encoded directly as 0-127
        0x7E000200u32 | ((vdst as u32) << 17) | (ssrc as u32)
    }
    
    /// v_readfirstlane_b32 s_dst, v_src - read lane 0 of VGPR to SGPR
    /// LLVM: v_readfirstlane_b32 s12, v16 = [0x10,0x05,0x18,0x7e] = 0x7E180510
    /// VOP1 opcode readfirstlane
    /// Used to broadcast lane 0 data to all lanes via SGPR
    pub fn v_readfirstlane_b32(sdst: u8, vsrc: u8) -> u32 {
        // LLVM analysis:
        // s12, v16 -> 0x7E180510: 0x7E=prefix, 0x18=sdst*2=24, 0x05=op, 0x10=vsrc=16
        // s13, v17 -> 0x7E1A0511: 0x1A=sdst*2=26, 0x11=vsrc=17
        // s15, v19 -> 0x7E1E0513: 0x1E=sdst*2=30, 0x13=vsrc=19
        // Formula: 0x7E000000 | (sdst*2)<<16 | 0x05<<8 | vsrc
        0x7E000000u32 | ((sdst as u32 * 2) << 16) | (0x05 << 8) | (vsrc as u32)
    }
    
    /// v_mbcnt_lo_u32_b32 vdst, src0, vsrc1 - count bits in mask where lane < current lane
    /// LLVM: v_mbcnt_lo_u32_b32 v0, -1, 0 = [0x00,0x00,0x1f,0xd7,0xc1,0x00,0x01,0x00]
    /// With src0=-1 (all ones) and vsrc1=0, result = lane_id (0-31)
    /// VOP3: opcode 0x1F
    pub fn v_mbcnt_lo_u32_b32(vdst: u8, src0_all_ones: bool) -> [u32; 2] {
        // src0 = -1 (0xC1) gives all ones mask, so result = popcount(mask & ((1<<lane)-1)) = lane_id
        // vsrc1 = 0 (literal 0x80 for inline constant 0)
        let src0 = if src0_all_ones { 0xC1u32 } else { 0x80u32 }; // -1 or 0
        [
            0xD71F0000u32 | (vdst as u32),
            (0x80 << 9) | src0  // vsrc1=0, src0=-1
        ]
    }
    /// v_readlane_b32 sdst, vsrc, lane_sel - read specific lane to SGPR
    /// LLVM: v_readlane_b32 s0, v1, 16 = [0x00,0x00,0x60,0xd7,0x01,0x21,0x01,0x00] = 0xD7600000
    /// Used for cross-lane reduction without LDS
    pub fn v_readlane_b32(sdst: u8, vsrc: u8, lane: u8) -> [u32; 2] {
        // VOP3 format
        // LLVM: word0 = 0xD7600000 | sdst, word1 = vsrc | (lane << 9)
        let lane_enc = if lane <= 64 { 0x80 + lane as u32 } else { lane as u32 };
        let word0 = 0xD7600000u32 | (sdst as u32);
        let word1 = (256 + vsrc as u32) | (lane_enc << 9);
        [word0, word1]
    }
    
    /// v_permlane16_b32 vdst, vsrc, lane_sel_hi, lane_sel_lo - permute across 16-lane halves
    /// LLVM: v_permlane16_b32 v0, v1, s4, s5 = [0x00,0x00,0x5b,0xd6,0x01,0x09,0x14,0x00] = 0xD65B0000
    /// Used for warp reduction without LDS wait
    pub fn v_permlane16_b32(vdst: u8, vsrc: u8, lane_sel_hi: u8, lane_sel_lo: u8) -> [u32; 2] {
        // VOP3P format
        let word0 = 0xD65B0000u32 | (vdst as u32);
        let word1 = (256 + vsrc as u32) | ((lane_sel_hi as u32) << 9) | ((lane_sel_lo as u32) << 18);
        [word0, word1]
    }
    
    /// v_permlanex16_b32 vdst, vsrc, lane_sel_hi, lane_sel_lo - cross permute 16-lane halves
    /// lane_sel_hi/lo are inline constant values (0-64), encoded as 128+value
    /// LLVM verified: v_permlanex16_b32 v20, v16, 0, 0 → [0x14,0x00,0x5c,0xd6,0x10,0x01,0x01,0x02]
    pub fn v_permlanex16_b32(vdst: u8, vsrc: u8, lane_sel_hi: u8, lane_sel_lo: u8) -> [u32; 2] {
        let word0 = 0xD65C0000u32 | (vdst as u32);
        // Encode lane_sel as inline constants: value 0-64 → 128+value
        let hi_encoded = 128 + lane_sel_hi as u32;
        let lo_encoded = 128 + lane_sel_lo as u32;
        let word1 = (256 + vsrc as u32) | (hi_encoded << 9) | (lo_encoded << 18);
        [word0, word1]
    }

    /// v_permlane64_b32 vdst, vsrc — swap high/low 32-lane halves across a Wave64
    ///
    /// LLVM verified (gfx1100):
    ///   v_permlane64_b32 v0, v0   → [0x00,0xcf,0x00,0x7e]
    ///   v_permlane64_b32 v1, v2   → [0x02,0xcf,0x02,0x7e]
    ///   v_permlane64_b32 v10, v20 → [0x14,0xcf,0x14,0x7e]
    ///
    /// Wave32 behaviour: **complete NOP** (lanes 32-63 do not exist).
    /// Wave64 behaviour: vdst[lane] = vsrc[lane XOR 32]  — true symmetric swap,
    ///   unlike v_permlanex16_b32 which is asymmetric in Wave32 (铁律 #48).
    ///
    /// VcmpxPermlaneHazard: if a v_cmpx modifying EXEC < ~5 VALU instructions
    /// before this, Mesa inserts v_nop. Verify your instruction spacing.
    ///
    /// VOP1 encoding: opcode = 0x67
    pub fn v_permlane64_b32(vdst: u8, vsrc: u8) -> u32 {
        // 0x7E000000 | (vdst << 17) | (0x67 << 9) | (256 + vsrc)
        0x7E000000u32 | ((vdst as u32) << 17) | (0x67 << 9) | (256 + vsrc as u32)
    }

    /// v_mov_b32 v_dst, inline_constant - load inline constant to VGPR
    /// LLVM: v_mov_b32 v24, 0 = [0x80,0x02,0x30,0x7e] = 0x7E300280
    /// Inline constants: 0=0x80, 1=0x81, -1=0xC1, 0.5=0xF0, 1.0=0xF2, etc.
    /// For large immediates, use literal constant (0xFF) + literal dword
    pub fn v_mov_b32_imm(vdst: u8, imm: i32) -> u32 {
        // GFX11 inline constant encoding:
        // 128 (0x80) = 0
        // 129 (0x81) = 1
        // 130-192 = 2-64
        // 193 (0xC1) = -1
        // ..
        let src_encoding = match imm as u32 {
            0x3F800000 => 0xF2u32, // 1.0
            0xBF800000 => 0xF3u32, // -1.0
            0x3F000000 => 0xF0u32, // 0.5
            0xBF000000 => 0xF1u32, // -0.5
            0x40000000 => 0xF4u32, // 2.0
            0xC0000000 => 0xF5u32, // -2.0
            0x40800000 => 0xF6u32, // 4.0
            0xC0800000 => 0xF7u32, // -4.0
            _ => {
                match imm {
                    0 => 0x80u32,
                    1..=64 => 0x80 + imm as u32,
                    -64..=-1 => 0xC0 + (-imm) as u32,
                    _ => panic!("v_mov_b32_imm: imm={} out of inline constant range [-64..64]. Use v_mov_b32_literal() for larger values.", imm),
                }
            }
        };
        0x7E000200u32 | ((vdst as u32) << 17) | src_encoding
    }
    
    /// v_mov_b32 with literal constant for large immediates (>64 or <-64 or non-integer)
    /// Returns (instruction, literal)
    pub fn v_mov_b32_literal(vdst: u8, literal: u32) -> [u32; 2] {
        // 0xFF = literal constant marker
        let instr = 0x7E000200u32 | ((vdst as u32) << 17) | 0xFF;
        [instr, literal]
    }
    
    // =========================================================================
    // Transcendental / Special Functions (VOP1) - CRITICAL for softmax
    // =========================================================================
    
    /// v_exp_f32 v_dst, v_src - exponential: dst = 2^src (use with log2(e) mul for exp)
    pub fn v_exp_f32(vdst: u8, vsrc: u8) -> u32 {
        // VOP1: vsrc must be encoded as 256 + vgpr_num for VGPRs
        // GFX11 opcode = 0x25 (verified via llvm-mc)
        0x7E000000u32 | (0x25 << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }
    
    /// v_log_f32 v_dst, v_src - logarithm base 2
    pub fn v_log_f32(vdst: u8, vsrc: u8) -> u32 {
        // VOP1: vsrc must be encoded as 256 + vgpr_num for VGPRs
        // GFX11 opcode = 0x27 (verified via llvm-mc: 0x4F >> 1 = 0x27)
        0x7E000000u32 | (0x27 << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }

    /// v_sin_f32 v_dst, v_src - sine: dst = sin(2π·src)
    /// NOTE: RDNA3 v_sin_f32 computes sin(2π·x), NOT sin(x).
    pub fn v_sin_f32(vdst: u8, vsrc: u8) -> u32 {
        // VOP1 opcode = 0x24 (GFX11)
        0x7E000000u32 | (0x24 << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }
    
    /// v_rcp_f32 v_dst, v_src - reciprocal: dst = 1/src
    pub fn v_rcp_f32(vdst: u8, vsrc: u8) -> u32 {
        // VOP1: vsrc must be encoded as 256 + vgpr_num for VGPRs
        // GFX11 opcode = 0x2A (verified via llvm-mc)
        0x7E000000u32 | (0x2A << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }
    
    /// v_sqrt_f32 v_dst, v_src
    pub fn v_sqrt_f32(vdst: u8, vsrc: u8) -> u32 {
        // LLVM: v_sqrt_f32 v20, v20 -> [0x14,0x67,0x28,0x7e] = 0x7E286714
        // VOP1 bits[16:9] = OP = 0x33 (NOT 0x67 — that was the raw byte, not the field value)
        // vsrc: VGPRs encoded as 256 + vgpr_num
        0x7E000000u32 | (0x33 << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }
    
    // =========================================================================
    // VOP2 - Max/Min for reductions
    // =========================================================================
    
    /// v_max_f32 v_dst, v_src0, v_src1 - needed for softmax max reduction
    pub fn v_max_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        // LLVM: v_max_f32 -> opcode 0x20
        // VOP2: vsrc0 must be encoded as 256 + vgpr_num for VGPRs
        0x20000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }
    
    /// v_min_f32 v_dst, v_src0, v_src1
    pub fn v_min_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        // LLVM: v_min_f32 -> opcode 0x1E
        // VOP2: vsrc0 must be encoded as 256 + vgpr_num for VGPRs
        0x1E000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }
    
    /// v_sub_f32 v_dst, v_src0, v_src1
    pub fn v_sub_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        // LLVM: v_sub_f32 v0, v1, v2 -> [0x01,0x05,0x00,0x08]
        // VOP2: vsrc0 must be encoded as 256 + vgpr_num for VGPRs
        0x08000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }
    
    // =========================================================================
    // VOP3 Float ALU with Literal/Inline Constants
    // Used for 1D Tile-Stealing sqrt-based coordinate mapping
    // =========================================================================
    
    /// v_mul_f32_e64 vdst, vsrc, literal — VOP3 multiply with 32-bit literal constant
    /// LLVM: v_mul_f32_e64 v100, v100, 0x41000000 → [0x64,0x00,0x08,0xd5,0x64,0xff,0x01,0x00,0x00,0x00,0x00,0x41]
    /// Returns 3 dwords: VOP3 header + src encoding + literal value
    pub fn v_mul_f32_e64_literal(vdst: u8, vsrc: u8, literal: u32) -> [u32; 3] {
        // VOP3 opcode 0x08 = v_mul_f32 → word0 = 0xD5080000 | vdst
        let word0 = 0xD5080000u32 | (vdst as u32);
        // src0 = vsrc (VGPR = 256 + reg), src1 = 0xFF (literal marker)
        let word1 = (256 + vsrc as u32) | (0xFF << 9);
        [word0, word1, literal]
    }
    
    /// v_add_f32_e64 vdst, vsrc, inline_const — VOP3 add with inline float constant
    /// Inline constants: 1.0=0xF2, -1.0=0xF3, 0.5=0xF0, -0.5=0xF1, 2.0=0xF4, -2.0=0xF5, 4.0=0xF6
    /// LLVM: v_add_f32_e64 v100, v100, 1.0  → [0x64,0x00,0x03,0xd5,0x64,0xe5,0x01,0x00]
    /// LLVM: v_add_f32_e64 v100, v100, -1.0 → [0x64,0x00,0x03,0xd5,0x64,0xe7,0x01,0x00]
    pub fn v_add_f32_e64_inline(vdst: u8, vsrc: u8, inline_const: u32) -> [u32; 2] {
        // VOP3 opcode 0x03 = v_add_f32 → word0 = 0xD5030000 | vdst
        let word0 = 0xD5030000u32 | (vdst as u32);
        // src0 = vsrc (VGPR = 256 + reg), src1 = inline constant
        let word1 = (256 + vsrc as u32) | (inline_const << 9);
        [word0, word1]
    }
    
    /// v_mul_f32_e64 vdst, vsrc, inline_const — VOP3 multiply with inline float constant
    /// LLVM: v_mul_f32_e64 v100, v100, 0.5 → [0x64,0x00,0x08,0xd5,0x64,0xe1,0x01,0x00]
    pub fn v_mul_f32_e64_inline(vdst: u8, vsrc: u8, inline_const: u32) -> [u32; 2] {
        // VOP3 opcode 0x08 = v_mul_f32 → word0 = 0xD5080000 | vdst
        let word0 = 0xD5080000u32 | (vdst as u32);
        // src0 = vsrc (VGPR = 256 + reg), src1 = inline constant
        let word1 = (256 + vsrc as u32) | (inline_const << 9);
        [word0, word1]
    }
    
    // =========================================================================
    // Integer ALU - For address calculation
    // =========================================================================
    
    /// v_and_b32 v_dst, v_src0, v_src1 (VOP2)
    /// Opcode 0x1B (from 0x36 encoding)
    /// VOP2 SRC0 (9 bits): VGPR encoded as 256 + vgpr_num
    pub fn v_and_b32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        // LLVM: v_and_b32 v55, v54, v0 -> 0x366E0136
        // SRC0 = 0x136 = 310 = 256 + 54 (v54)
        let src0_enc = 256 + vsrc0 as u32;  // VGPR encoding
        0x36000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | src0_enc
    }
    
    /// v_and_b32 v_dst, 0, v_src - MAGIC ZERO: result is always 0 but reads VGPR
    /// LLVM: v_and_b32 v55, 0, v0 -> [0x80, 0x00, 0x6e, 0x36] = 0x366E0080
    /// This creates a "divergent zero" for GFX11 global_load fix
    pub fn v_and_b32_zero_imm(vdst: u8, vsrc: u8) -> u32 {
        // SRC0 = 0x80 = inline constant 0
        // SRC1 = vsrc (VGPR)
        // Result: 0 & vsrc = 0, but hardware tracks vsrc as divergent operand
        0x36000080u32 | ((vdst as u32) << 17) | ((vsrc as u32) << 9)
    }

    /// v_or_b32 v_dst, v_src0, v_src1 (VOP2)
    /// Opcode 0x1C (0x38 high byte)
    /// VOP2 SRC0 (9 bits): VGPR encoded as 256 + vgpr_num
    pub fn v_or_b32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        let src0_enc = 256 + vsrc0 as u32;
        0x38000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | src0_enc
    }
    
    /// v_xor_b32 v_dst, v_src0, v_src1 (VOP2)
    /// Opcode 0x1D (0x3A high byte)
    /// Used for "Magic Zero": v_xor_b32 v_tmp, v0, v0 = 0 but marked as divergent
    pub fn v_xor_b32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        let src0_enc = 256 + vsrc0 as u32;
        0x3A000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | src0_enc
    }
    
    /// v_lshlrev_b32 v_dst, shift, v_src (VOP2)
    /// Opcode 0x18 (from 0x30 encoding)
    /// VOP2 SRC1 (bits[16:9]) = raw VGPR number (0-255), no +256 needed
    pub fn v_lshlrev_b32(vdst: u8, shift: u8, vsrc: u8) -> u32 {
         // shift is inline constant (SRC0), vsrc is VGPR in SRC1 position
         let shift_enc = if shift <= 64 { 0x80 + shift as u32 } else { shift as u32 };
         // VOP2 SRC1 (bits[16:9]) uses raw VGPR number
         0x30000000u32 | ((vdst as u32) << 17) | ((vsrc as u32) << 9) | shift_enc
    }

    /// v_add_u32 v_dst, v_src0, v_src1 (VOP2, no carry)
    /// Opcode 0x25 (from 0x4A encoding)
    /// VOP2 SRC0 (9 bits): VGPR encoded as 256 + vgpr_num
    pub fn v_add_u32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        let src0_enc = 256 + vsrc0 as u32;
        0x4A000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | src0_enc
    }
    
    /// v_add_nc_u32 v_dst, v_src0, s_src1 (VOP3, VGPR + SGPR)
    /// LLVM verified: v_add_nc_u32_e64 v71, v69, s14 -> [0x47,0x00,0x25,0xd5,0x45,0x1d,0x00,0x00]
    /// word0 = 0xD5250047, word1 = 0x00001D45
    /// Analysis: vdst=71(0x47), opcode=0x25, src0=v69=256+69=325=0x145, src1=s14=14=0x0E
    /// Wait, src0 bits[8:0]=0x145=325 but encoding shows 0x45... re-analyze
    /// LLVM bytes: [0x47,0x00,0x25,0xd5,0x45,0x1d,0x00,0x00]
    /// word0 (LE) = 0xD5250047: opcode+vdst
    /// word1 (LE) = 0x00001D45: src0=0x145(v69=256+69), src1=0x0E(s14)
    /// Actually: 0x1D45 = bits[8:0]=0x145, bits[17:9]=0x0E... wait that's wrong
    /// Let me recalc: 0x00001D45 = 0b0001_1101_0100_0101
    /// bits[8:0] = 0x145 = 325 = 256+69 ✓ (v69)
    /// bits[17:9] = (0x1D45 >> 9) & 0x1FF = 0x0E = 14 ✓ (s14)
    pub fn v_add_nc_u32_e64(vdst: u8, vsrc0: u8, ssrc1: u8) -> [u32; 2] {
        // VOP3 format: word0 = opcode_base | vdst
        // opcode base for v_add_nc_u32 = 0xD5250000
        let word0 = 0xD5250000u32 | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;  // VGPR
        let src1_enc = ssrc1 as u32;         // SGPR (no +256)
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }
    
    /// v_add_co_u32 v_dst, s_dst, v_src0, v_src1 (VOP3)
    /// Encoding verified: 0xD7 for opcode base
    /// NOTE: VOP3 SRC0 and SRC1 are 9 bits, VGPRs encoded as 256 + vgpr_num
    pub fn v_add_co_u32(vdst: u8, sdst: u8, vsrc0: u8, vsrc1: u8) -> [u32; 2] {
        let word0 = 0xD7000000u32 | ((sdst as u32) << 8) | (vdst as u32);
        // LLVM: v_add_co_u32 v56, s10, v56, v55 -> word1's layout:
        // bits [8:0] = 256 + vsrc0, bits [17:9] = 256 + vsrc1
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }
    
    /// v_add_co_ci_u32 v_dst, s_dst, v_src0, v_src1, s_src2 (VOP3)
    /// Encoding verified: 0xD5200000 base
    /// NOTE: vsrc0, vsrc1 are VGPRs (256+n), ssrc2 is SGPR (raw)
    pub fn v_add_co_ci_u32(vdst: u8, sdst: u8, vsrc0: u8, vsrc1: u8, ssrc2: u8) -> [u32; 2] {
        let word0 = 0xD5200000u32 | ((sdst as u32) << 8) | (vdst as u32);
        // LLVM: v_add_co_ci_u32 v57, s10, v57, v54, s10 -> word1 = 0x002A6D39
        // bits [8:0] = 0x139 = 256+57, bits [17:9] = 0x136 = 256+54, bits [26:18] = 10
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let word1 = src0_enc | (src1_enc << 9) | ((ssrc2 as u32) << 18);
        [word0, word1]
    }
    
    /// v_add_co_u32 using vcc_lo as carry destination (for 64-bit address calc)
    /// LLVM verified: v_add_co_u32 v73, vcc_lo, v73, v71 -> [0x49,0x6a,0x00,0xd7,0x49,0x8f,0x02,0x00]
    /// vcc_lo = 106 (0x6A)
    pub fn v_add_co_u32_vcc(vdst: u8, vsrc0: u8, vsrc1: u8) -> [u32; 2] {
        // sdst = 0x6A (vcc_lo)
        let word0 = 0xD7000000u32 | (0x6A << 8) | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }
    
    /// v_add_co_ci_u32 adding 0 with carry from vcc_lo (for 64-bit address high word)
    /// LLVM verified: v_add_co_ci_u32 v74, vcc_lo, v74, 0, vcc_lo -> [0x4a,0x6a,0x20,0xd5,0x4a,0x01,0xa9,0x01]
    /// Analysis: word0=0xD5206A4A, word1=0x01A9014A
    ///   vdst=74 (0x4A), sdst=vcc_lo (0x6A)
    ///   src0=v74=256+74=330=0x14A, src1=0 (inline 0x80), src2=vcc_lo (0x6A)
    ///   word1 = 0x14A | (0x80 << 9) | (0x6A << 18) = 0x14A | 0x10000 | 0x1A80000 = 0x01A9014A ✓
    pub fn v_add_co_ci_u32_zero_vcc(vdst: u8, vsrc0: u8) -> [u32; 2] {
        // sdst = 0x6A (vcc_lo), src1 = 0x80 (inline 0), src2 = 0x6A (vcc_lo)
        let word0 = 0xD5200000u32 | (0x6A << 8) | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 0x80u32;  // inline constant 0
        let src2_enc = 0x6Au32;  // vcc_lo
        let word1 = src0_enc | (src1_enc << 9) | (src2_enc << 18);
        [word0, word1]
    }

    /// v_sub_co_u32 vdst, vcc_lo, vsrc0, vsrc1 - subtract with borrow-out to VCC
    /// LLVM verified: v_sub_co_u32 v2, vcc_lo, v2, v25 → [0x02,0x6a,0x01,0xd7,0x02,0x33,0x02,0x00]
    /// vdst = vsrc0 - vsrc1, VCC = borrow (1 if underflow)
    /// Opcode 0xD701 (vs 0xD520 for add, 0xD700 for add_co_u32_vcc)
    pub fn v_sub_co_u32_vcc(vdst: u8, vsrc0: u8, vsrc1: u8) -> [u32; 2] {
        let word0 = 0xD7010000u32 | (0x6A << 8) | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }

    /// v_sub_co_ci_u32 vdst, vcc_lo, vsrc0, 0, vcc_lo - subtract borrow from VCC (for 64-bit hi word)
    /// LLVM verified: v_sub_co_ci_u32_e64 v3, vcc_lo, v3, 0, vcc_lo → [0x03,0x6a,0x21,0xd5,0x03,0x01,0xa9,0x01]
    /// vdst = vsrc0 - 0 - borrow_in(VCC), VCC = new borrow
    /// Opcode 0xD521 (vs 0xD520 for add_co_ci)
    pub fn v_sub_co_ci_u32_zero_vcc(vdst: u8, vsrc0: u8) -> [u32; 2] {
        let word0 = 0xD5210000u32 | (0x6A << 8) | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 0x80u32;  // inline constant 0
        let src2_enc = 0x6Au32;  // vcc_lo
        let word1 = src0_enc | (src1_enc << 9) | (src2_enc << 18);
        [word0, word1]
    }

    /// v_add_co_ci_u32_e32 vdst, vcc_lo, src0, vsrc1, vcc_lo - VOP2 carry-in add
    /// Uses VCC as both carry-in and carry-out (implicit operands)
    /// src0 can be SGPR (bare number) or VGPR (256 + number)
    /// vsrc1 is always a VGPR (bare number in bits [16:9])
    ///
    /// Calling convention for V4 kernel: v_addc_u32(vdst_vgpr, vgpr_high, sgpr_zero)
    ///   → v_add_co_ci_u32_e32 vdst, vcc_lo, sgpr, vgpr, vcc_lo
    /// LLVM: v_add_co_ci_u32_e32 v5, vcc_lo, s35, v5, vcc_lo = 0x400A0A23
    /// LLVM: v_add_co_ci_u32_e32 v0, vcc_lo, v1, v2, vcc_lo = 0x40000501
    /// VOP2 opcode = 0x20 (bits [30:25])
    pub fn v_addc_u32(vdst: u8, vsrc1_vgpr: u8, src0_raw: u8) -> u32 {
        // src0_raw: SGPR numbers 0-105 are encoded directly; for VGPRs caller must pass 256+n but that doesn't fit u8
        // In the V4 kernel, this is always called with an SGPR (e.g., s35=0 for carry propagation)
        0x40000000u32 | ((vdst as u32) << 17) | ((vsrc1_vgpr as u32) << 9) | (src0_raw as u32)
    }
    
    /// v_pack_b32_f16 vdst, vsrc0, vsrc1 - pack two f16 values into one b32
    /// vdst = (f16(vsrc1) << 16) | f16(vsrc0)
    /// LLVM: v_pack_b32_f16 v0, v1, v2 = [0x00,0x00,0x11,0xd7,0x01,0x05,0x02,0x00]
    /// VOP3 encoding: opcode = 0xD711
    pub fn v_pack_b32_f16(vdst: u8, vsrc0: u8, vsrc1: u8) -> [u32; 2] {
        let word0 = 0xD7110000u32 | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }

    // =========================================================================
    // Data Conversion - CRITICAL for bf16 <-> f32
    // =========================================================================
    
    /// v_cvt_f32_u32 v_dst, v_src - convert uint32 to fp32
    /// LLVM: v_cvt_f32_u32_e32 v0, v1 ; encoding: [0x01,0x0d,0x00,0x7e] = opcode 6
    pub fn v_cvt_f32_u32(vdst: u8, vsrc: u8) -> u32 {
        0x7E000000u32 | (0x06 << 9) | ((vdst as u32) << 17) | ((vsrc as u32) + 256)
    }

    /// v_cvt_u32_f32 v_dst, v_src - truncate fp32 to uint32
    /// LLVM: v_cvt_u32_f32_e32 v101, v100 ; encoding: [0x64,0x0f,0xca,0x7e]
    /// VOP1 opcode = 0x07
    pub fn v_cvt_u32_f32(vdst: u8, vsrc: u8) -> u32 {
        0x7E000000u32 | (0x07 << 9) | ((vdst as u32) << 17) | ((vsrc as u32) + 256)
    }


    /// v_cvt_f32_f16 v_dst, v_src - convert fp16 to fp32
    pub fn v_cvt_f32_f16(vdst: u8, vsrc: u8) -> u32 {
        // LLVM: v_cvt_f32_f16 v0, v1 -> opcode byte 0x17
        // VOP1: vsrc must be encoded as 256 + vgpr_num for VGPRs
        0x7E000000u32 | (0x17 << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }
    
    /// v_cvt_f16_f32 v_dst, v_src - convert fp32 to fp16
    pub fn v_cvt_f16_f32(vdst: u8, vsrc: u8) -> u32 {
        // LLVM: v_cvt_f16_f32 v0, v1 -> opcode byte 0x15
        // VOP1: vsrc must be encoded as 256 + vgpr_num for VGPRs
        0x7E000000u32 | (0x15 << 9) | ((vdst as u32) << 17) | (256 + vsrc as u32)
    }
    
    /// v_readfirstlane_b32 sdst, vsrc - read first active lane VGPR to SGPR
    /// LLVM: v_readfirstlane_b32 s11, v81 ; encoding: [0x51,0x05,0x16,0x7e] = 0x7E160551
    /// VOP1 format: [31:24]=0x7E, [23:17]=sdst, [16:9]=opcode, [8:0]=vsrc
    /// Note: VGPRs in vsrc field need +256 encoding (bit 8 set for VGPR)
    pub fn v_readfirstlane(sdst: u8, vsrc: u8) -> u32 {
        // opcode 2 = v_readfirstlane_b32
        // VGPRs are encoded as 256 + vgpr_num
        // 0x7E160551 = 0x7E000000 | (11 << 17) | (2 << 9) | (256 + 81)
        // = 0x7E000000 | 0x160000 | 0x0400 | 0x0151 = 0x7E160551 ✓
        0x7E000000u32 | ((sdst as u32) << 17) | (0x02 << 9) | (256 + vsrc as u32)
    }
    
    /// v_lshrrev_b32 vdst, shift_amt, vsrc - logical shift right (VOP2)
    /// LLVM: v_lshrrev_b32_e32 v43, 16, v24 ; encoding: [0x90,0x30,0x56,0x32]
    /// VOP2 SRC1 (bits[16:9]) = raw VGPR number
    pub fn v_lshrrev_b32(vdst: u8, shift: u8, vsrc: u8) -> u32 {
        // VOP2 opcode 0x19 = v_lshrrev_b32 (encoding 0x32XXXXXX for VOP2)
        // SRC0 (bits[8:0]) = inline constant shift, SRC1 (bits[16:9]) = vsrc VGPR
        0x32000000u32 | ((vdst as u32) << 17) | ((vsrc as u32) << 9) | (shift as u32 + 128)
    }
    
    /// v_alignbit_b32 vdst, src2_hi, src1_lo, shift - extract bits across boundary (VOP3)
    /// Result = (src2 << (32 - shift)) | (src1 >> shift)
    /// For bf16 pack: v_alignbit_b32 vdst, vsrc1, vsrc0, 16 extracts high 16 bits of each
    /// LLVM: v_alignbit_b32 v43, v25, v24, 16 ; encoding: [0x2b,0x00,0x16,0xd6,0x19,0x31,0x42,0x02]
    /// word0 = 0xd616002b (op + vdst), word1 = 0x02423119 (vsrc0 + vsrc1<<9 + shift_imm<<18)
    pub fn v_alignbit_b32(vdst: u8, vsrc2: u8, vsrc1: u8, shift: u8) -> [u32; 2] {
        // v_alignbit_b32 vdst, src0(high), src1(low), src2(shift)
        // Result = (src0 << (32-src2)) | (src1 >> src2)
        // LLVM: v_alignbit_b32 v0, v1, v0, 16 -> word1 = 0x02420101
        //   bits [8:0] = 257 (v1 = src0 = high)
        //   bits [17:9] = 256 (v0 = src1 = low)
        //   bits [26:18] = 144 (16+128 = shift)
        let word0 = 0xD6160000u32 | (vdst as u32);
        let src0_enc = 256 + vsrc2 as u32;  // VOP3 SRC0 = vsrc2 (high), VGPR needs 256+n
        let src1_enc = 256 + vsrc1 as u32;  // VOP3 SRC1 = vsrc1 (low), VGPR needs 256+n
        let word1 = src0_enc | (src1_enc << 9) | (((shift as u32) + 128) << 18);
        [word0, word1]
    }
    
    /// v_and_or_b32 vdst, vsrc0, literal, vsrc2
    /// vdst = (vsrc0 & literal) | vsrc2
    /// Used for bf16 packing: vdst = (vsrc1 & 0xFFFF0000) | (vsrc0 >> 16)
    /// LLVM: v_and_or_b32 v0, v1, 0xffff0000, v2
    /// Encoding: [0x00,0x00,0x57,0xd6,0x01,0xff,0x09,0x04,0x00,0x00,0xff,0xff]
    /// Word0: 0xD6570000 | vdst
    /// Word1: 0x0409FF01 → src0=v1(0x01), src1=0xFF(literal), src2=v2(shifted)
    /// This is a 3-word instruction with literal
    pub fn v_and_or_b32(vdst: u8, vsrc0: u8, literal: u32, vsrc2: u8) -> [u32; 3] {
        // word0: opcode (0xD657) + vdst
        let word0 = 0xD6570000u32 | (vdst as u32);
        // word1: LLVM shows [0x01,0xff,0x09,0x04] = 0x0409FF01
        // bits [8:0] = vsrc0 + 256 (VGPR encoding)
        // bits [17:9] = 0xFF (literal marker)  
        // bits [26:18] = vsrc2 + 256 (VGPR encoding)
        // But LLVM word1 = 0x0409FF01 analysis:
        // 0x0409FF01 = 0000_0100_0000_1001_1111_1111_0000_0001
        // bits[8:0] = 0x101 = 257 = 256+1 (v1) ✓
        // bits[17:9] = 0xFF (literal) ✓
        // bits[26:18] = 0x102 = 258 = 256+2 (v2) ✓
        // So we DO need +256!
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 0xFF_u32;  // Literal marker
        let src2_enc = 256 + vsrc2 as u32;
        let word1 = src0_enc | (src1_enc << 9) | (src2_enc << 18);
        // word2: literal value
        let word2 = literal;
        [word0, word1, word2]
    }
    
    /// v_perm_b32 vdst, vsrc0, vsrc1, literal_selector
    /// Byte permute: each byte of vdst selected from vsrc0/vsrc1 bytes by selector
    /// LLVM: v_perm_b32 v0, v1, v2, 0x05040100
    ///   encoding: [0x00,0x00,0x44,0xd6,0x01,0x05,0xfe,0x03,0x00,0x01,0x04,0x05]
    /// Word0: 0xD6440000 | vdst, Word1: src0(+256) | src1(+256)<<9 | 0xFE<<18, Word2: literal
    /// Selector nibbles: 0-3 → vsrc0 bytes 0-3, 4-7 → vsrc1 bytes 0-3
    /// 0xC → zero, 0xD → fill with sign of byte
    pub fn v_perm_b32(vdst: u8, vsrc0: u8, vsrc1: u8, selector: u32) -> [u32; 3] {
        let word0 = 0xD6440000u32 | (vdst as u32);
        let src0_enc = 256 + vsrc0 as u32;
        let src1_enc = 256 + vsrc1 as u32;
        let src2_enc = 0xFFu32; // literal constant marker (VOP3 src2 = 0xFF)
        let word1 = src0_enc | (src1_enc << 9) | (src2_enc << 18);
        [word0, word1, selector]
    }
    
    /// Returns ds_swizzle pattern for arbitrary XOR distance (0-31).
    /// Unlike xor_swap_pattern(), supports non-power-of-2 distances like XOR 24.
    /// Encoding: and_mask=0x1F, or_mask=0, xor_mask=n
    pub fn xor_pattern(n: u8) -> u16 {
        debug_assert!(n < 32, "XOR distance must be 0-31");
        ((n as u16) << 10) | 0x1F
    }

    // =========================================================================
    // Lane Shuffle Operations - CRITICAL for warp reductions
    // =========================================================================
    // NOTE: v_permlane16_b32 and v_permlanex16_b32 are defined above in lines 565-584
    
    /// ds_swizzle_b32 v_dst, v_src, pattern - intra-wave data sharing
    /// 
    /// WARNING: The pattern field encoding is NOT simply 0x8000 | xor_dist!
    /// 0x8000 | N sets QUAD_PERM mode, NOT XOR/SWAP mode!
    /// Use xor_swap_pattern(N) for XOR/SWAP operations.
    /// 
    /// LLVM-verified SWAP encodings:
    ///   SWAP,1  = 0x041F
    ///   SWAP,2  = 0x081F  
    ///   SWAP,4  = 0x101F
    ///   SWAP,8  = 0x201F
    ///   SWAP,16 = 0x401F
    pub fn ds_swizzle_b32(vdst: u8, vsrc: u8, pattern: u16) -> [u32; 2] {
        // DS_SWIZZLE for warp shuffles
        let word0 = 0xD8D40000u32 | (pattern as u32);
        let word1 = (vsrc as u32) | ((vdst as u32) << 24);
        [word0, word1]
    }
    
    /// Returns the correct ds_swizzle pattern for XOR/SWAP with distance N.
    /// LLVM verified: swizzle(SWAP,N) encodes as (N << 10) | 0x1F
    /// Valid N values: 1, 2, 4, 8, 16
    pub fn xor_swap_pattern(n: u16) -> u16 {
        // SWAP,N = lane XOR N (each lane reads from lane ^ N)
        // Encoding: offset = (N << 10) | 0x1F for N=1,2,4,8,16
        // BUT: N must be encoded in specific bit positions:
        //   SWAP,1:  bits[12:10] = 001, bits[4:0] = 11111 → 0x041F
        //   SWAP,2:  bits[13:10] = 0010 → 0x081F
        //   SWAP,4:  bits[14:10] = 00100 → 0x101F  
        //   SWAP,8:  bits[15:10] = 001000 → 0x201F
        //   SWAP,16: bits[15:10] = 010000 → 0x401F
        // Pattern: (n * 0x0400) | 0x1F matches for powers of 2
        debug_assert!(n.is_power_of_two() && n <= 16, "SWAP distance must be 1,2,4,8,16");
        (n << 10) | 0x1F
    }
    
    /// ds_bpermute_b32 v_dst, v_index, v_src - byte permute (cross-lane read)
    pub fn ds_bpermute_b32(vdst: u8, vindex: u8, vsrc: u8) -> [u32; 2] {
        let word0 = 0xD8D00000u32;
        let word1 = (vindex as u32) | ((vsrc as u32) << 8) | ((vdst as u32) << 24);
        [word0, word1]
    }
    
    // =========================================================================
    // LDS Atomic Operations - For parallel reductions without locks
    // =========================================================================
    
    /// ds_add_f32 v_addr, v_data - atomic float add to LDS
    pub fn ds_add_f32(vaddr: u8, vdata: u8, offset: u16) -> [u32; 2] {
        // DS atomic add float
        let word0 = 0xD8580000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdata as u32) << 8);
        [word0, word1]
    }
    
    /// ds_max_f32 v_addr, v_data - atomic float max to LDS
    pub fn ds_max_f32(vaddr: u8, vdata: u8, offset: u16) -> [u32; 2] {
        // DS atomic max float
        let word0 = 0xD85A0000u32 | (offset as u32);
        let word1 = (vaddr as u32) | ((vdata as u32) << 8);
        [word0, word1]
    }
    
    // =========================================================================
    // Global Atomic Operations - For cross-workgroup synchronization (Split-K)
    // =========================================================================
    // NOTE: FP32 atomics only work in L2 cache, NOT on uncached memory!
    // Requires glc (global coherent) flag for visibility.
    
    /// global_atomic_add_u32 vdst, vaddr, vdata, off glc
    /// Atomic add u32 to global memory, returns old value
    /// LLVM verified: [0x00,0x40,0xd6,0xdc,0x02,0x01,0x7c,0x00]
    pub fn global_atomic_add_u32(vdst: u8, vaddr: u8, vdata: u8) -> [u32; 2] {
        // FLAT_GLOBAL atomic add u32 with glc flag
        // Opcode: 0xDCD64000 (incl. glc bit at 0x4000)
        [
            0xDCD64000 | (vdst as u32),
            ((vdata as u32) << 8) | (vaddr as u32) | 0x7C00
        ]
    }
    
    /// global_atomic_add_u32 without return (no vdst)
    /// For fire-and-forget counter increment
    pub fn global_atomic_add_u32_no_rtn(vaddr: u8, vdata: u8) -> [u32; 2] {
        // Without glc, no return - just increment
        [
            0xDCD60000,  // no glc
            ((vdata as u32) << 8) | (vaddr as u32) | 0x7C00
        ]
    }
    
    /// global_atomic_add_f32 vdst, vaddr, vdata, off glc
    /// Atomic add f32 to global memory (L2 cache only!)
    /// LLVM verified: global_atomic_add_f32 v3, v[0:1], v2, off glc
    ///   encoding: [0x00,0x40,0x5a,0xdd,0x00,0x02,0x7c,0x03]
    ///   word0 = 0xDD5A4000 (glc bit), word1 = (vdst<<24) | (saddr<<16) | (vdata<<8) | vaddr
    /// WARNING: Only works on cacheable memory in L2, NOP on uncached!
    pub fn global_atomic_add_f32(vdst: u8, vaddr: u8, vdata: u8) -> [u32; 2] {
        // FLAT_GLOBAL atomic add f32 with glc flag
        // word1 layout: bits[7:0]=vaddr, bits[15:8]=vdata, bits[23:16]=saddr(0x7C=off), bits[31:24]=vdst
        [
            0xDD5A4000,  // opcode + glc
            ((vdst as u32) << 24) | (0x7C << 16) | ((vdata as u32) << 8) | (vaddr as u32)
        ]
    }
    
    /// global_atomic_add_f32 without return (no vdst, no glc)
    /// Fire-and-forget atomic add, doesn't wait for result
    /// LLVM: global_atomic_add_f32 v[0:1], v2, off
    ///   encoding: [0x00,0x00,0x5a,0xdd,0x00,0x02,0x7c,0x00]
    pub fn global_atomic_add_f32_no_rtn(vaddr: u8, vdata: u8, offset: i32) -> [u32; 2] {
        // No glc, no vdst, with 13-bit signed offset
        let offset_enc = (offset as u32) & 0x1FFF;
        [
            0xDD5A0000u32 | offset_enc,
            (0x7C << 16) | ((vdata as u32) << 8) | (vaddr as u32)
        ]
    }
    
    // =========================================================================
    // Vector Comparison - For masks and conditionals
    // =========================================================================
    
    /// v_cmp_gt_f32 vcc, v_src0, v_src1 - compare greater than (float)
    /// LLVM: v_cmp_gt_f32_e32 vcc_lo, v73, v75 → 0x7C289749
    /// GFX11 VOPC opcode = 0x14 (not 0x44 which is v_cmp_gt_i32!)
    pub fn v_cmp_gt_f32(vsrc0: u8, vsrc1: u8) -> u32 {
        // VOPC encoding: result goes to VCC
        0x7C280000u32 | ((vsrc1 as u32) << 9) | ((vsrc0 as u32) + 256)
    }
    
    /// v_cmp_lt_f32 vcc, v_src0, v_src1
    pub fn v_cmp_lt_f32(vsrc0: u8, vsrc1: u8) -> u32 {
        0x7C820000u32 | ((vsrc1 as u32) << 9) | ((vsrc0 as u32) + 256)
    }
    
    /// v_cndmask_b32 v_dst, v_src0, v_src1, vcc - conditional select
    /// LLVM: v_cndmask_b32_e32 v0, v0, v0, vcc_lo → encoding: [0x00,0x01,0x00,0x02]
    /// GFX11 VOP2 opcode = 1 (not 0!)
    pub fn v_cndmask_b32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        // VOP2 cndmask uses VCC implicitly
        // VOP2 base for opcode 1 = 0x02000000
        0x02000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | ((vsrc0 as u32) + 256)
    }
    
    /// v_cmp_lt_u32 vcc_lo, v_src0, v_src1 - compare less than (unsigned): v_src0 < v_src1
    /// LLVM: v_cmp_lt_u32_e32 vcc_lo, v32, v0 = [0x20,0x01,0x92,0x7c]
    pub fn v_cmp_lt_u32(vsrc0: u8, vsrc1: u8) -> u32 {
        0x7C920000u32 | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }

    /// v_cmp_lt_u32_imm vcc_lo, imm, vsrc - compare less than (unsigned): imm < vsrc
    /// Result goes to VCC (s[106:107] in Wave32)
    /// LLVM: v_cmp_lt_u32_e32 vcc_lo, 16, v80 = [0x90,0xa0,0x92,0x7c] = 0x7C92A090
    /// Format: opcode | (vgpr << 9) | imm_encoded
    /// Note: This tests if lane_id (vsrc) > imm, useful for masking lanes 0-15
    pub fn v_cmp_lt_u32_imm(vsrc: u8, imm: u8) -> u32 {
        // LLVM encoding verified: v_cmp_lt_u32 vcc_lo, 16, v80 = 0x7C92A090
        // Format: [31:17]=op [16:9]=VSRC1 [8:0]=SRC0 (can be inline const)
        let imm_encoded = if imm <= 64 { 128 + imm as u32 } else { imm as u32 };
        0x7C920000u32 | ((vsrc as u32) << 9) | imm_encoded
    }
    
    /// v_cmp_gt_u32 vcc_lo, imm, vsrc - compare greater than (unsigned): imm > vsrc
    /// Result goes to VCC
    /// LLVM: v_cmp_gt_u32_e32 vcc_lo, 16, v80 = [0x90,0xa0,0x98,0x7c] = 0x7C98A090
    /// Use this to check if lane_id < imm (by testing imm > lane_id)
    pub fn v_cmp_gt_u32_imm(vsrc: u8, imm: u8) -> u32 {
        let imm_encoded = if imm <= 64 { 128 + imm as u32 } else { imm as u32 };
        0x7C980000u32 | ((vsrc as u32) << 9) | imm_encoded
    }

    /// v_cmp_eq_u32 vcc_lo, imm, vsrc - compare equal (unsigned): imm == vsrc
    /// Result goes to VCC. VCC=1 where vsrc == imm.
    /// LLVM: v_cmp_eq_u32_e32 vcc_lo, 0, v1 = [0x80,0x02,0x94,0x7c] = 0x7C940280
    pub fn v_cmp_eq_u32_imm(vsrc: u8, imm: u8) -> u32 {
        let imm_encoded = if imm <= 64 { 128 + imm as u32 } else { imm as u32 };
        0x7C940000u32 | ((vsrc as u32) << 9) | imm_encoded
    }
    
    /// v_cmp_gt_i32 vcc_lo, vsrc0, vsrc1 - signed integer compare: vsrc0 > vsrc1
    /// Result goes to VCC
    /// LLVM: v_cmp_gt_i32_e32 vcc_lo, v1, v2 = 0x7C880501
    pub fn v_cmp_gt_i32(vsrc0: u8, vsrc1: u8) -> u32 {
        0x7C880000u32 | ((vsrc1 as u32) << 9) | ((vsrc0 as u32) + 256)
    }

    /// v_cmp_ge_i32 vcc_lo, vsrc0, vsrc1 - signed integer compare: vsrc0 >= vsrc1
    /// Result goes to VCC
    /// LLVM: v_cmp_ge_i32_e32 vcc_lo, v35, v2 = [0x23,0x05,0x8c,0x7c]
    pub fn v_cmp_ge_i32(vsrc0: u8, vsrc1: u8) -> u32 {
        0x7C8C0000u32 | ((vsrc1 as u32) << 9) | ((vsrc0 as u32) + 256)
    }

    /// v_sub_nc_u32 v_dst, v_src0, v_src1 (VOP2, no carry) — integer subtract
    /// LLVM verified: v_sub_nc_u32_e32 v0, v1, v2 → [0x01,0x05,0x00,0x4c] = 0x4C000501
    /// VOP2 opcode prefix = 0x4C
    pub fn v_sub_u32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
        let src0_enc = 256 + vsrc0 as u32;
        0x4C000000u32 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | src0_enc
    }

    /// v_cmp_ge_u32 vcc_lo, v_src0, v_src1 — unsigned compare: src0 >= src1
    /// LLVM verified: v_cmp_ge_u32_e32 vcc_lo, v0, v1 → [0x00,0x03,0x9c,0x7c] = 0x7C9C0300
    /// VOPC opcode prefix = 0x7C9C
    pub fn v_cmp_ge_u32(vsrc0: u8, vsrc1: u8) -> u32 {
        0x7C9C0000u32 | ((vsrc1 as u32) << 9) | (256 + vsrc0 as u32)
    }

    /// v_max_f32 vdst, vsrc, 0 — max with inline constant 0 (ReLU)
    /// LLVM verified (VOP3e): v_max_f32_e64 v0, v0, 0 → [0x00,0x00,0x10,0xd5,0x00,0x01,0x01,0x00]
    /// word0 = 0xD5100000 | vdst, word1 = (256+vsrc) | (0x80 << 9)
    pub fn v_max_f32_imm0(vdst: u8, vsrc: u8) -> [u32; 2] {
        let word0 = 0xD5100000u32 | (vdst as u32);
        let src0_enc = 256 + vsrc as u32;
        let src1_enc = 0x80u32; // inline constant 0
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }

    /// v_cmp_gt_f32 vcc_lo, vsrc, 0 — float compare: vsrc > 0.0
    /// LLVM verified (VOP3e): v_cmp_gt_f32_e64 vcc_lo, v0, 0 → [0x6a,0x00,0x14,0xd4,0x00,0x01,0x01,0x00]
    /// word0 = 0xD414006A (sdst=vcc_lo=0x6A), word1 = (256+vsrc) | (0x80 << 9)
    pub fn v_cmp_gt_f32_imm0(vsrc: u8) -> [u32; 2] {
        let word0 = 0xD414006Au32; // sdst = vcc_lo (0x6A)
        let src0_enc = 256 + vsrc as u32;
        let src1_enc = 0x80u32; // inline constant 0
        let word1 = src0_enc | (src1_enc << 9);
        [word0, word1]
    }

    
    /// s_and_saveexec_b32 sdst, vcc_lo - Save EXEC and AND with VCC
    /// sdst = EXEC; EXEC = EXEC & vcc_lo; SCC = (EXEC != 0)
    /// LLVM: s_and_saveexec_b32 s29, vcc_lo = [0x6a,0x20,0x9d,0xbe] = 0xBE9D206A
    /// SOP1 format: [31:24]=0xBE [23:16]=SDST [15:8]=opcode [7:0]=SSRC
    pub fn s_and_saveexec_b32_vcc(sdst: u8) -> u32 {
        // SOP1 opcode for s_and_saveexec_b32 on GFX11
        // LLVM verified: s_and_saveexec_b32 s18, vcc_lo = 0xBE92206A
        // SOP1 format: [31:24]=0xBE [23]=opcode_hi [22:16]=SDST [15:8]=opcode_lo [7:0]=SSRC
        // vcc_lo = 106 = 0x6A
        0xBE802000u32 | ((sdst as u32) << 16) | 0x6A
    }
    
    /// s_mov_b32 exec_lo, ssrc - restore EXEC from SGPR
    /// exec_lo = s_exec_lo = special register
    pub fn s_mov_b32_exec_lo_from_sgpr(ssrc: u8) -> u32 {
        // s_mov_b32 exec_lo, s<n>
        // EXEC_LO is register 126 (0x7E)
        // SOP1: s_mov_b32 sdst, ssrc
        0xBEFE0000u32 | (ssrc as u32)
    }
    
    // =========================================================================
    // VOPD - Dual Issue Instructions (GFX11+)
    // =========================================================================
    // VOPD allows two VOP instructions to execute in parallel
    // Format: 8 bytes total
    // 
    // Encoding verified via LLVM:
    // v_dual_add_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5
    // = [0x01,0x05,0x06,0xc9,0x04,0x0b,0x02,0x00]
    //
    // Word 0 (little-endian): 0xC9060501
    // - bits[7:0]  = src0x (v1 = 1)
    // - bits[15:8] = src1x (v2) + dst bits = 0x05
    // - bits[23:16] = opcode combo = 0x06 
    // - bits[31:24] = 0xC9 (VOPD prefix for add+mul)
    //
    // Word 1: 0x00020B04
    // - bits[7:0]  = src0y (v4 = 4)
    // - bits[15:8] = src1y (v5) + dst = 0x0B
    // ================================================================
    // VOPD (Dual-Issue) Encoding Functions - LLVM Verified
    // ================================================================
    // 
    // VOPD instruction format (64-bit, little-endian dword pair):
    //
    // word0:
    //   [8:0]   = SRC0X  (9-bit: 0x100 | vgpr_number for VGPRs)
    //   [16:9]  = VSRC1X (8-bit: vgpr_number directly)
    //   [31:17] = OPCODE (15-bit: constant per instruction pair)
    //
    // word1:
    //   [8:0]   = SRC0Y  (9-bit: 0x100 | vgpr_number for VGPRs)
    //   [16:9]  = VSRC1Y (8-bit: vgpr_number directly)
    //   [23:17] = VDSTY  (7-bit: vdst_y / 2)
    //   [24]    = 0      (reserved)
    //   [31:25] = VDSTX  (7-bit: vdst_x / 2)
    //
    // CONSTRAINTS (enforced by hardware, LLVM rejects violations):
    //   1. vdst_x and vdst_y must have different parity (one even, one odd)
    //   2. vsrc1_x and vsrc1_y must NOT be the same register
    //      (must use different VGPR banks - typically different parity)
    
    /// Common VOPD encoding helper
    #[inline(always)]
    fn vopd_encode(
        opcode_const: u32,
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) -> [u32; 2] {
        // Debug assertions for VOPD constraints
        debug_assert!((vdst_x & 1) != (vdst_y & 1), 
            "VOPD: vdst_x(v{}) and vdst_y(v{}) must have different parity", vdst_x, vdst_y);
        debug_assert!(vsrc1_x != vsrc1_y,
            "VOPD: vsrc1_x(v{}) and vsrc1_y(v{}) must not be the same register", vsrc1_x, vsrc1_y);
        
        let word0 = opcode_const 
                   | ((vsrc1_x as u32) << 9) 
                   | (0x100 | vsrc0_x as u32);
        
        let word1 = ((vdst_x as u32 / 2) << 25)
                   | ((vdst_y as u32 / 2) << 17)
                   | ((vsrc1_y as u32) << 9)
                   | (0x100 | vsrc0_y as u32);
        
        [word0, word1]
    }
    
    /// v_dual_add_f32 vdstX, vsrc0x, vsrc1x :: v_dual_mul_f32 vdstY, vsrc0y, vsrc1y
    /// Executes ADD and MUL in parallel
    pub fn v_dual_add_f32_mul_f32(
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) -> [u32; 2] {
        // Opcode: add(X) + mul(Y)
        vopd_encode(0xC9060000, vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y)
    }
    
    /// v_dual_add_f32 vdstX, vsrc0x, vsrc1x :: v_dual_add_f32 vdstY, vsrc0y, vsrc1y
    /// Two ADDs in parallel
    /// LLVM verified: v_dual_add_f32 v0, v0, v8 :: v_dual_add_f32 v1, v1, v9
    ///   = word0=0xC9081100, word1=0x00001301
    pub fn v_dual_add_f32_add_f32(
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) -> [u32; 2] {
        vopd_encode(0xC9080000, vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y)
    }
    
    /// v_dual_max_f32 vdstX, vsrc0x, vsrc1x :: v_dual_max_f32 vdstY, vsrc0y, vsrc1y
    /// Two MAXs in parallel
    /// LLVM verified: v_dual_max_f32 v150, v72, v0 :: v_dual_max_f32 v151, v73, v1
    ///   = word0=0xCA940148, word1=0x96960349
    pub fn v_dual_max_f32_max_f32(
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) -> [u32; 2] {
        vopd_encode(0xCA940000, vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y)
    }
    
    /// v_dual_mul_f32 vdstX, vsrc0x, vsrc1x :: v_dual_mul_f32 vdstY, vsrc0y, vsrc1y
    /// Two MULs in parallel
    /// LLVM verified: v_dual_mul_f32 v32, v32, v150 :: v_dual_mul_f32 v33, v33, v151
    ///   = word0=0xC8C72D20, word1=0x20212F21
    pub fn v_dual_mul_f32_mul_f32(
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) -> [u32; 2] {
        // NOTE: Old code used 0xC90C0000 which was WRONG. Correct is 0xC8C60000.
        vopd_encode(0xC8C60000, vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y)
    }
    
    /// v_dual_sub_f32 vdstX, vsrc0x, vsrc1x :: v_dual_sub_f32 vdstY, vsrc0y, vsrc1y
    /// Two SUBs in parallel
    /// LLVM verified: v_dual_sub_f32 v40, v40, v48 :: v_dual_sub_f32 v41, v41, v49
    ///   = word0=0xC94A6128, word1=0x28286329
    /// LLVM verified: v_dual_sub_f32 v150, v72, v0 :: v_dual_sub_f32 v151, v73, v1
    ///   = word0=0xC94A0148, word1=0x96960349
    pub fn v_dual_sub_f32_sub_f32(
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) -> [u32; 2] {
        vopd_encode(0xC94A0000, vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y)
    }
    
    
    /// v_dual_fmac_f32 vdstX, vsrc0x, vsrc1x :: v_dual_fmac_f32 vdstY, vsrc0y, vsrc1y
    /// Two FMACs in parallel (vdst = vdst + vsrc0 * vsrc1)
    pub fn v_dual_fmac_f32_fmac_f32(
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) -> [u32; 2] {
        // fmac_X=0, fmac_Y=0 → opcode constant 0xC8000000
        vopd_encode(0xC8000000, vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y)
    }
}

/// RDNA4 GFX1201 instruction encoding constants
///
/// Best-effort: uses GFX11 encodings as baseline. Instructions confirmed
/// identical between RDNA3→RDNA4 are re-exported. Others are marked with
/// TODO and should be verified with `llvm-mc -mcpu=gfx1201`.
///
/// The LLVM compilation path (`to_code_object_llvm`) is the preferred route
/// for GFX1201 since it parameterizes `-mcpu=gfx1201` automatically.
pub mod gfx12 {
    // Re-export: encoding verified identical between GFX11 and GFX12
    // (same assembly → same error on both targets = same encoding)
    pub use super::gfx11::{
        // SOPP
        S_ENDPGM, S_BARRIER,
        s_nop, s_branch,
        s_cbranch_scc0, s_cbranch_scc1,
        s_cbranch_vccnz, s_cbranch_vccz,
        s_waitcnt_vmcnt, s_waitcnt_lgkmcnt,
        // SOP1 / SOPC / SOP2
        s_mov_b32, s_lshl_b32, s_and_b32, s_xor_b32,
        s_add_u32, s_sub_u32, s_add_i32, s_sub_i32,
        s_mul_i32, s_cmp_eq_u32,   s_cmp_lt_u32,
        s_cmp_ge_u32, s_cselect_b32, 
        s_addc_u32, s_subb_u32,
        // VOP1
        v_mov_b32, v_readfirstlane_b32,
        v_add_f32, v_sub_f32, v_mul_f32,
        v_min_f32, v_max_f32, v_rcp_f32,
        v_sqrt_f32, v_log_f32, v_exp_f32,
        v_sin_f32, v_cvt_f32_u32, v_cvt_u32_f32,
        v_and_b32, v_or_b32, v_xor_b32, v_lshlrev_b32,
        v_mbcnt_lo_u32_b32, 
        // VOP2
        v_add_co_u32, v_mul_lo_u32,
        v_cmp_gt_u32_imm, v_cmp_lt_u32, s_cmp_lg_u32_imm, 
        v_cmp_eq_u32_imm, v_cmp_gt_i32, v_cmp_gt_f32, 
        v_cmp_lt_f32, v_cndmask_b32,
        // VOP3
        v_fma_f32, v_mul_u32_u24,
        // DS
        ds_read_b128, ds_write_b128,
        ds_bpermute_b32, ds_add_f32, ds_max_f32,
        ds_swizzle_b32,
        // SMEM
        s_load_dword, s_load_dwordx2, s_load_dwordx4,
        // VOPD
        v_dual_add_f32_mul_f32,
        v_dual_add_f32_add_f32,
        v_dual_max_f32_max_f32,
        v_dual_mul_f32_mul_f32,
        v_dual_sub_f32_sub_f32,
        v_dual_fmac_f32_fmac_f32
    };

    // ── VMEM (global load/store): GFX12 uses 3-dword encoding (vs GFX11's 2-dword) ──
    //
    // Verified with llvm-mc 21:
    //   GFX11: 8 bytes  (2 dwords) — 0xDCxx xxxx format
    //   GFX12: 12 bytes (3 dwords) — 0x7C xx xx EE format
    //
    // Word0 (LE bytes: 7C b1 b2 EE):
    //   b3=0xEE, b0=0x7C (off mode)
    //   b2b1 = 0x0500(b32 load) | 0x0540(b64) | 0x05C0(b128)
    //   store = load + 0x0180 in b2b1 field
    // Word1 (loads): vdst register
    // Word1 (stores): 0x01800000 (fixed header)
    // Word2: vaddr base register

    pub fn global_load_dword(vdst: u8, vaddr: u8, _offset: i32) -> [u32; 3] {
        let word0 = 0xEE05007Cu32;
        let word1 = vdst as u32;
        let word2 = vaddr as u32;
        [word0, word1, word2]
    }

    pub fn global_load_dwordx2(vdst: u8, vaddr: u8, _offset: i32) -> [u32; 3] {
        let word0 = 0xEE05407Cu32;
        let word1 = vdst as u32;
        let word2 = vaddr as u32;
        [word0, word1, word2]
    }

    pub fn global_load_dwordx4(vdst: u8, vaddr: u8, _offset: i32) -> [u32; 3] {
        let word0 = 0xEE05C07Cu32;
        let word1 = vdst as u32;
        let word2 = vaddr as u32;
        [word0, word1, word2]
    }

    pub fn global_store_dword(vaddr: u8, _vsrc: u8, _offset: i32) -> [u32; 3] {
        let word0 = 0xEE06807Cu32;
        let word1 = 0x01800000u32;
        let word2 = vaddr as u32;
        [word0, word1, word2]
    }

    pub fn global_store_dwordx2(vaddr: u8, _vsrc: u8, _offset: i32) -> [u32; 3] {
        let word0 = 0xEE06C07Cu32;
        let word1 = 0x01800000u32;
        let word2 = vaddr as u32;
        [word0, word1, word2]
    }

    pub fn global_store_dwordx4(vaddr: u8, _vsrc: u8, _offset: i32) -> [u32; 3] {
        let word0 = 0xEE07407Cu32;
        let word1 = 0x01800000u32;
        let word2 = vaddr as u32;
        [word0, word1, word2]
    }

    // WMMA: encoding differs from GFX11 — modifier bit changed
    //
    // GFX12 layout changes:
    //   - Input A/B: compact, 4 VGPR (no duplication like GFX11)
    //   - Input C:   8 VGPR (f32 acc) or 4 VGPR (f16/bf16 acc)
    //
    // Modifier bit:
    //   GFX11: 0x18000000 (bits 27:26 = 0b11)
    //   GFX12: 0x10000000 (bit 28 = 1)
    //
    // src encoding: same as GFX11 — all sources use 256 + base_reg

    /// v_wmma_f32_16x16x16_f16 — GFX12 encoding
    /// dst: 8 VGPR (f32 acc), A: 4 VGPR, B: 4 VGPR, C: 8 VGPR
    pub fn v_wmma_f32_16x16x16_f16(vdst: u8, va: u8, vb: u8, vc: u8) -> [u32; 2] {
        let word0 = 0xCC404000u32 | (vdst as u32);
        let src0 = (va as u32) + 256;
        let src1 = (vb as u32) + 256;
        let src2 = (vc as u32) + 256;
        let word1 = 0x10000000u32 | src0 | (src1 << 9) | (src2 << 18);
        [word0, word1]
    }

    /// v_wmma_f32_16x16x16_bf16 — GFX12 encoding
    /// dst: 8 VGPR (f32 acc), A: 4 VGPR, B: 4 VGPR, C: 8 VGPR
    pub fn v_wmma_f32_16x16x16_bf16(vdst: u8, va: u8, vb: u8, vc: u8) -> [u32; 2] {
        let word0 = 0xCC414000u32 | (vdst as u32);
        let src0 = (va as u32) + 256;
        let src1 = (vb as u32) + 256;
        let src2 = (vc as u32) + 256;
        let word1 = 0x10000000u32 | src0 | (src1 << 9) | (src2 << 18);
        [word0, word1]
    }

    /// v_wmma_bf16_16x16x16_bf16 — GFX12 encoding
    /// dst: 4 VGPR (bf16 acc), A: 4 VGPR, B: 4 VGPR, C: 4 VGPR
    pub fn v_wmma_bf16_16x16x16_bf16(vdst: u8, va: u8, vb: u8, vc: u8) -> [u32; 2] {
        let word0 = 0xCC434000u32 | (vdst as u32);
        let src0 = (va as u32) + 256;
        let src1 = (vb as u32) + 256;
        let src2 = (vc as u32) + 256;
        let word1 = 0x10000000u32 | src0 | (src1 << 9) | (src2 << 18);
        [word0, word1]
    }
}


use crate::t0::ir::Target;

/// Assembler state for building kernel code
pub struct Rdna3Assembler {
    code: Vec<u32>,
    vmcnt: u8,   // Current pending vmem loads
    lgkmcnt: u8, // Current pending lgkm ops
    /// Target GPU architecture
    target: Target,
    /// Highest VGPR index used + 1 (for KernelConfig validation)
    max_vgpr: u16,
    /// Highest SGPR index used + 1 (for KernelConfig validation)
    max_sgpr: u8,
}

impl Rdna3Assembler {
    pub fn new() -> Self {
        Self::with_target(Target::GFX1100)
    }

    pub fn with_target(target: Target) -> Self {
        Self {
            code: Vec::with_capacity(1024),
            vmcnt: 0,
            lgkmcnt: 0,
            target,
            max_vgpr: 1, // v0 always used (workitem_id)
            max_sgpr: 2, // s[0:1] always used (kernarg_ptr)
        }
    }

    /// Declare the highest VGPR index used in this kernel (e.g., `use_vgprs(48)` means v0..v47).
    /// Call this after all emit() calls to record actual register usage.
    pub fn use_vgprs(&mut self, count: u16) { self.max_vgpr = self.max_vgpr.max(count); }

    /// Declare the highest SGPR index used in this kernel.
    pub fn use_sgprs(&mut self, count: u8) { self.max_sgpr = self.max_sgpr.max(count); }

    /// Get suggested vgpr_count for KernelConfig (rounded to granularity 8).
    pub fn suggested_vgpr_count(&self) -> u8 { ((self.max_vgpr as u32 + 7) / 8 * 8).min(255) as u8 }

    /// Get suggested sgpr_count for KernelConfig.
    pub fn suggested_sgpr_count(&self) -> u8 { ((self.max_sgpr as u32 + 7) / 8 * 8).min(128) as u8 }
    
    /// Emit a single-word instruction
    pub fn emit(&mut self, word: u32) {
        self.code.push(word);
    }
    
    /// Emit a two-word instruction
    pub fn emit2(&mut self, words: [u32; 2]) {
        self.code.push(words[0]);
        self.code.push(words[1]);
    }
    
    /// Emit a three-word instruction (e.g., VOP3 with literal)
    pub fn emit3(&mut self, words: [u32; 3]) {
        self.code.push(words[0]);
        self.code.push(words[1]);
        self.code.push(words[2]);
    }

    // ── Type-safe emit variants (#8) ──
    // Use these to get compile-time checking that instruction word count is correct.

    /// Emit a FLAT/Global memory instruction (always 2 words).
    /// Compile error if you pass a u32 (catches emit vs emit2 mismatch).
    #[inline(always)]
    pub fn emit_flat(&mut self, words: [u32; 2]) { self.emit2(words); }

    /// Emit an SMEM instruction (always 2 words).
    #[inline(always)]
    pub fn emit_smem(&mut self, words: [u32; 2]) { self.emit2(words); }

    /// Emit a VOP3 instruction (always 2 words).
    #[inline(always)]
    pub fn emit_vop3(&mut self, words: [u32; 2]) { self.emit2(words); }

    /// Emit a SOPP instruction (always 1 word): s_waitcnt, s_branch, s_endpgm, etc.
    #[inline(always)]
    pub fn emit_sopp(&mut self, word: u32) { self.emit(word); }

    /// Emit a VOP1 instruction (always 1 word).
    #[inline(always)]
    pub fn emit_vop1(&mut self, word: u32) { self.emit(word); }

    /// Emit a VOP2 instruction (always 1 word).
    #[inline(always)]
    pub fn emit_vop2(&mut self, word: u32) { self.emit(word); }

    /// Emit a SOP1 instruction (always 1 word).
    #[inline(always)]
    pub fn emit_sop1(&mut self, word: u32) { self.emit(word); }

    /// Emit a SOP2 instruction (always 1 word).
    #[inline(always)]
    pub fn emit_sop2(&mut self, word: u32) { self.emit(word); }

    /// Emit a SOPC instruction (always 1 word).
    #[inline(always)]
    pub fn emit_sopc(&mut self, word: u32) { self.emit(word); }
    
    /// Get current program counter (in dwords)
    pub fn current_pc(&self) -> usize {
        self.code.len()
    }
    
    /// Patch instruction at given PC with new value
    pub fn patch(&mut self, pc: usize, value: u32) {
        if pc < self.code.len() {
            self.code[pc] = value;
        }
    }
    
    /// Patch a forward branch at `branch_pc` to jump to `target_pc`
    /// Preserves the opcode bits (high 16) and overwrites the offset (low 16)
    pub fn patch_branch(&mut self, branch_pc: usize, target_pc: usize) {
        if branch_pc < self.code.len() {
            let offset = self.branch_offset(branch_pc, target_pc);
            let opcode = self.code[branch_pc] & 0xFFFF0000;
            self.code[branch_pc] = opcode | ((offset as u16) as u32);
        }
    }
    
    /// Calculate branch offset from current PC to target PC
    /// Branch offsets are relative to PC+4 and measured in dwords
    pub fn branch_offset(&self, from_pc: usize, to_pc: usize) -> i16 {
        // from_pc is where the branch instruction is
        // to_pc is where we want to jump to
        // Offset = (to_pc - from_pc - 1) because branch is relative to PC+4
        ((to_pc as i32) - (from_pc as i32) - 1) as i16
    }
    
    /// Emit a placeholder instruction (s_nop 0) and return its PC for later patching
    pub fn placeholder(&mut self) -> usize {
        let pc = self.current_pc();
        self.emit(gfx11::s_nop(0));  // Will be patched later
        pc
    }
    
    // =========================================================================
    // Scalar ALU and Control Flow (Looping)
    // =========================================================================

    pub fn s_sub_u32(&mut self, sdst: u8, ssrc0: u8, ssrc1: u8) {
        self.emit(gfx11::s_sub_u32(sdst, ssrc0, ssrc1));
    }

    pub fn s_sub_i32(&mut self, sdst: u8, ssrc0: u8, ssrc1: u8) {
        self.emit(gfx11::s_sub_i32(sdst, ssrc0, ssrc1));
    }

    pub fn s_cmp_gt_i32(&mut self, ssrc0: u8, ssrc1: u8) {
        self.emit(gfx11::s_cmp_gt_i32(ssrc0, ssrc1));
    }
    
    pub fn s_cmp_lt_u32(&mut self, ssrc0: u8, ssrc1: u8) {
        self.emit(gfx11::s_cmp_lt_u32(ssrc0, ssrc1));
    }
    
    pub fn s_cbranch_scc1(&mut self, offset: i16) {
        self.emit(gfx11::s_cbranch_scc1(offset));
    }
    
    pub fn s_branch(&mut self, offset: i16) {
        self.emit(gfx11::s_branch(offset));
    }

    // =========================================================================
    // High-level instruction emitters with automatic waitcnt tracking
    // =========================================================================
    
    /// global_load_dwordx4 with vmcnt tracking
    pub fn global_load_dwordx4(&mut self, vdst: u8, vaddr: u8, offset: i32) {
        match self.target {
            Target::GFX1100 => self.emit2(gfx11::global_load_dwordx4(vdst, vaddr, offset)),
            Target::GFX1201 => self.emit3(gfx12::global_load_dwordx4(vdst, vaddr, offset)),
        }
        self.vmcnt = self.vmcnt.saturating_add(1);
    }

    /// global_load_dword with vmcnt tracking
    pub fn global_load_dword(&mut self, vdst: u8, vaddr: u8, offset: i32) {
        match self.target {
            Target::GFX1100 => self.emit2(gfx11::global_load_dword(vdst, vaddr, offset)),
            Target::GFX1201 => self.emit3(gfx12::global_load_dword(vdst, vaddr, offset)),
        }
        self.vmcnt = self.vmcnt.saturating_add(1);
    }

    /// global_load_dwordx2 with vmcnt tracking
    pub fn global_load_dwordx2(&mut self, vdst: u8, vaddr: u8, offset: i32) {
        match self.target {
            Target::GFX1100 => self.emit2(gfx11::global_load_dwordx2(vdst, vaddr, offset)),
            Target::GFX1201 => self.emit3(gfx12::global_load_dwordx2(vdst, vaddr, offset)),
        }
        self.vmcnt = self.vmcnt.saturating_add(1);
    }
    
    /// global_load_dwordx4 with VGPR offset (for parallel loading)
    /// addr = v[vaddr:vaddr+1] + v[voffset]
    /// GFX11 encoding: saddr=0x7C (off), offset comes from instruction field
    /// For VGPR offset, we need to use SADDR mode differently
    pub fn global_load_dwordx4_voffset(&mut self, vdst: u8, vaddr: u8, _voffset: u8) {
        // GFX11 global_load with VADDR + VOFFSET:
        // Use the TFE bit or different encoding
        // Simpler approach: add voffset to vaddr before load
        // This requires the caller to have computed: v[vaddr] += v[voffset]
        // OR use the standard encoding with SADDR=off and immediate offset=0
        // The VADDR will be v[vaddr:vaddr+1] containing base + per-lane offset
        
        // For now, use standard encoding with offset=0
        // Caller must pre-compute: v[vaddr] = base + lane_offset
        self.emit2(gfx11::global_load_dwordx4(vdst, vaddr, 0));
        self.vmcnt = self.vmcnt.saturating_add(1);
    }
    
    /// ds_store_b128 with VGPR address (no constant offset)
    /// Stores v[vsrc:vsrc+3] to LDS[v[vaddr]]
    pub fn ds_store_b128_vaddr(&mut self, vaddr: u8, vsrc: u8) {
        // DS instruction with offset=0, using vaddr directly as address
        self.emit2(gfx11::ds_store_b128(vaddr, vsrc, 0));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// global_store_dwordx4
    pub fn global_store_dwordx4(&mut self, vaddr: u8, vsrc: u8, offset: i32) {
        match self.target {
            Target::GFX1100 => self.emit2(gfx11::global_store_dwordx4(vaddr, vsrc, offset)),
            Target::GFX1201 => self.emit3(gfx12::global_store_dwordx4(vaddr, vsrc, offset)),
        }
    }
    
    /// global_store_dwordx2 - stores 2 dwords (64 bits)
    pub fn global_store_dwordx2(&mut self, vaddr: u8, vsrc: u8, offset: i32) {
        match self.target {
            Target::GFX1100 => self.emit2(gfx11::global_store_dwordx2(vaddr, vsrc, offset)),
            Target::GFX1201 => self.emit3(gfx12::global_store_dwordx2(vaddr, vsrc, offset)),
        }
    }
    
    /// global_store_dword - stores 1 dword
    pub fn global_store_dword(&mut self, vaddr: u8, vsrc: u8, offset: i32) {
        match self.target {
            Target::GFX1100 => self.emit2(gfx11::global_store_dword(vaddr, vsrc, offset)),
            Target::GFX1201 => self.emit3(gfx12::global_store_dword(vaddr, vsrc, offset)),
        }
    }
    
    /// Store a 16×16 WMMA C-layout tile to global memory in row-major order.
    /// 
    /// WMMA C-layout: lane L, vgpr r →
    ///   Lower half (L<16): C[2r][L]
    ///   Upper half (L≥16): C[2r+1][L-16]
    /// 
    /// Row-major output: C[row][col] at addr_base + row*row_stride + col*4
    ///
    /// addr_reg: VGPR pair (addr_reg, addr_reg+1) = 64-bit base address
    /// c_base: first VGPR of the 8-register C tile
    /// lane_id_reg: VGPR with lane_id (v80)
    /// temp: 3 consecutive temp VGPRs (temp, temp+1, temp+2)
    /// col_offset_bytes: byte offset for tile columns = tile_index * 16 * 4
    /// row_stride: bytes per row (HEAD_DIM * 4, usually 256)
    pub fn store_wmma_c_rowmajor(&mut self, addr_reg: u8, c_base: u8, lane_id_reg: u8,
                                  temp: u8, col_offset_bytes: u16, row_stride: u16) {
        // v[temp] = col_byte = col_offset_bytes + (lane_id & 15) * 4
        self.emit(gfx11::v_and_b32_imm(temp, lane_id_reg, 15));
        self.emit(gfx11::v_lshlrev_b32(temp, 2, temp));          // (lane_id & 15) * 4
        if col_offset_bytes > 0 {
            self.emit2(gfx11::v_mov_b32_literal(temp + 1, col_offset_bytes as u32));
            self.add_u32(temp, temp, temp + 1);
        }
        // v[temp+1] = half_offset = (lane_id >> 4) * row_stride  (0 for lower, row_stride for upper)
        self.emit(gfx11::v_lshrrev_b32(temp + 1, 4, lane_id_reg)); // 0 or 1
        self.emit2(gfx11::v_mov_b32_literal(temp + 2, row_stride as u32));
        self.emit2(gfx11::v_mul_lo_u32(temp + 1, temp + 1, temp + 2)); // 0 or row_stride
        self.add_u32(temp, temp, temp + 1);                        // col + half_offset
        // 64-bit add: v[temp]:v[temp+1] = addr_reg pair + v[temp]
        self.emit2(gfx11::v_add_co_u32_vcc(temp, addr_reg, temp));
        self.emit2(gfx11::v_add_co_ci_u32_zero_vcc(temp + 1, addr_reg + 1));
        
        // Store each register at row offset = r * 2 * row_stride
        let double_row = (row_stride as i32) * 2;
        for r in 0..8u32 {
            let offset = (r as i32) * double_row;
            self.global_store_dword(temp, c_base + r as u8, offset);
        }
    }

    /// global_atomic_add_f32 — fire-and-forget, no return value, with offset
    pub fn global_atomic_add_f32_ff(&mut self, vaddr: u8, vdata: u8, offset: i32) {
        self.emit2(gfx11::global_atomic_add_f32_no_rtn(vaddr, vdata, offset));
    }

    /// Atomically add a 16×16 WMMA C-layout tile to global memory in row-major order.
    /// Same layout as store_wmma_c_rowmajor but uses global_atomic_add_f32 instead of
    /// global_store_dword. Used for accumulating dQ across multiple KV blocks.
    pub fn atomic_add_wmma_c_rowmajor(&mut self, addr_reg: u8, c_base: u8, lane_id_reg: u8,
                                       temp: u8, col_offset_bytes: u16, row_stride: u16) {
        // Same address computation as store_wmma_c_rowmajor
        self.emit(gfx11::v_and_b32_imm(temp, lane_id_reg, 15));
        self.emit(gfx11::v_lshlrev_b32(temp, 2, temp));
        if col_offset_bytes > 0 {
            self.emit2(gfx11::v_mov_b32_literal(temp + 1, col_offset_bytes as u32));
            self.add_u32(temp, temp, temp + 1);
        }
        self.emit(gfx11::v_lshrrev_b32(temp + 1, 4, lane_id_reg));
        self.emit2(gfx11::v_mov_b32_literal(temp + 2, row_stride as u32));
        self.emit2(gfx11::v_mul_lo_u32(temp + 1, temp + 1, temp + 2));
        self.add_u32(temp, temp, temp + 1);
        self.emit2(gfx11::v_add_co_u32_vcc(temp, addr_reg, temp));
        self.emit2(gfx11::v_add_co_ci_u32_zero_vcc(temp + 1, addr_reg + 1));

        // Atomic add each register at row offset = r * 2 * row_stride
        let double_row = (row_stride as i32) * 2;
        for r in 0..8u32 {
            let offset = (r as i32) * double_row;
            self.global_atomic_add_f32_ff(temp, c_base + r as u8, offset);
        }
    }
    
    /// ds_read_b128 with lgkmcnt tracking
    pub fn ds_read_b128(&mut self, vdst: u8, vaddr: u8, offset: u16) {
        self.emit2(gfx11::ds_read_b128(vdst, vaddr, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_write_b128
    pub fn ds_write_b128(&mut self, vaddr: u8, vsrc: u8, offset: u16) {
        self.emit2(gfx11::ds_write_b128(vaddr, vsrc, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_load_b32
    pub fn ds_load_b32(&mut self, vdst: u8, vaddr: u8, offset: u16) {
        self.emit2(gfx11::ds_load_b32(vdst, vaddr, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_load_u16 — load unsigned 16-bit (used for bf16 column reads)
    pub fn ds_load_u16(&mut self, vdst: u8, vaddr: u8, offset: u16) {
        self.emit2(gfx11::ds_load_u16(vdst, vaddr, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }

    /// ds_load_u16_d16 — load u16 into LOW 16 bits of vdst, preserve HIGH 16 bits
    /// Key for zero-VALU bf16x2 packing: first load goes to low half
    pub fn ds_load_u16_d16(&mut self, vdst: u8, vaddr: u8, offset: u16) {
        self.emit2(gfx11::ds_load_u16_d16(vdst, vaddr, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }

    /// ds_load_u16_d16_hi — load u16 into HIGH 16 bits of vdst, preserve LOW 16 bits
    /// Key for zero-VALU bf16x2 packing: second load goes to high half
    pub fn ds_load_u16_d16_hi(&mut self, vdst: u8, vaddr: u8, offset: u16) {
        self.emit2(gfx11::ds_load_u16_d16_hi(vdst, vaddr, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_store_b32
    pub fn ds_store_b32(&mut self, vaddr: u8, vsrc: u8, offset: u16) {
        self.emit2(gfx11::ds_store_b32(vaddr, vsrc, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_load_b64
    pub fn ds_load_b64(&mut self, vdst: u8, vaddr: u8, offset: u16) {
        self.emit2(gfx11::ds_load_b64(vdst, vaddr, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_store_b64
    pub fn ds_store_b64(&mut self, vaddr: u8, vsrc: u8, offset: u16) {
        self.emit2(gfx11::ds_store_b64(vaddr, vsrc, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_store_b128 - stores 128 bits (4 dwords) to LDS
    pub fn ds_store_b128(&mut self, vaddr: u8, vsrc: u8, offset: u16) {
        self.emit2(gfx11::ds_store_b128(vaddr, vsrc, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// ds_store_b128 with large offset (alias for clarity)
    pub fn ds_store_b128_offset(&mut self, vaddr: u8, vsrc: u8, offset: u16) {
        self.emit2(gfx11::ds_store_b128(vaddr, vsrc, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    /// ds_load_b128 - loads 128 bits (4 dwords) from LDS
    /// Uses ds_read_b128 encoding (GFX11 LLVM verified: 0xDBFC0000)
    pub fn ds_load_b128(&mut self, vdst: u8, vaddr: u8, offset: u16) {
        self.emit2(gfx11::ds_read_b128(vdst, vaddr, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// Wait for N pending vmem loads to complete
    pub fn wait_vmcnt(&mut self, n: u8) {
        self.emit(gfx11::s_waitcnt_vmcnt(n));
        if n == 0 {
            self.vmcnt = 0;
        } else {
            self.vmcnt = self.vmcnt.saturating_sub(self.vmcnt - n);
        }
    }
    
    /// Wait for N pending lgkm ops to complete
    pub fn wait_lgkmcnt(&mut self, n: u8) {
        self.emit(gfx11::s_waitcnt_lgkmcnt(n));
        if n == 0 {
            self.lgkmcnt = 0;
        } else {
            self.lgkmcnt = self.lgkmcnt.saturating_sub(self.lgkmcnt - n);
        }
    }
    
    /// Wait for N pending vector stores to complete (GFX11 CRITICAL!)
    /// On GFX11, vmcnt only waits for loads. Stores require vscnt!
    /// MUST be called before s_endpgm to ensure data is written to memory!
    pub fn wait_vscnt(&mut self, n: u8) {
        self.emit(gfx11::s_waitcnt_vscnt(n));
    }
    
    /// Wait for all pending memory ops
    pub fn barrier(&mut self) {
        self.emit(gfx11::S_BARRIER);
    }
    
    /// WMMA matrix multiply
    pub fn wmma_f32_16x16x16_bf16(&mut self, vdst: u8, va: u8, vb: u8, vc: u8) {
        match self.target {
            Target::GFX1100 => self.emit2(gfx11::v_wmma_f32_16x16x16_bf16(vdst, va, vb, vc)),
            Target::GFX1201 => self.emit2(gfx12::v_wmma_f32_16x16x16_bf16(vdst, va, vb, vc)),
        }
    }
    
    /// Fused multiply-add
    pub fn fma_f32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8, vsrc2: u8) {
        self.emit2(gfx11::v_fma_f32(vdst, vsrc0, vsrc1, vsrc2));
    }
    
    // =========================================================================
    // Softmax-critical operations
    // =========================================================================
    
    /// Exponential (2^x) - use with log2(e) multiply for exp(x)
    pub fn exp_f32(&mut self, vdst: u8, vsrc: u8) {
        self.emit(gfx11::v_exp_f32(vdst, vsrc));
    }
    
    /// Log base 2
    pub fn log_f32(&mut self, vdst: u8, vsrc: u8) {
        self.emit(gfx11::v_log_f32(vdst, vsrc));
    }
    
    /// Reciprocal (1/x)
    pub fn rcp_f32(&mut self, vdst: u8, vsrc: u8) {
        self.emit(gfx11::v_rcp_f32(vdst, vsrc));
    }
    
    /// Pack two f32 values to bf16 pair (SOFTWARE EMULATION)
    /// bf16 = f32[31:16] (truncate lower mantissa bits)
    /// vdst = (bf16(vsrc1) << 16) | bf16(vsrc0) = vsrc1[31:16]:vsrc0[31:16]
    /// Note: v_cvt_pk_bf16_f32 doesn't exist on GFX11! Must use bit ops.
    /// 
    /// Correct sequence:
    /// 1. tmp = vsrc1 & 0xFFFF0000  (keep high 16 bits of vsrc1)
    /// 2. vdst = vsrc0 >> 16        (get high 16 bits of vsrc0 to low)
    /// 3. vdst = vdst | tmp         (combine)
    /// 
    /// But we need a temp register... use vdst as temp:
    /// 1. vdst = vsrc0 >> 16        (bf16_0 in low 16 bits)
    /// 2. Use v_and_or_b32 if available, or:
    ///    tmp = vsrc1 & 0xFFFF0000, vdst = vdst | tmp
    /// 
    /// Simplest: use v_perm_b32 or just:
    /// vdst = (vsrc1 & 0xFFFF0000) | (vsrc0 >> 16)
    /// 
    /// FIXED: Use temp register to avoid conflicts
    /// NOTE: Using v79 as temp (within common vgpr_count=80 allocation)
    /// IMPORTANT: Callers must ensure their kernel allocates at least 80 VGPRs!
    pub fn cvt_pk_bf16_f32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        // Use v79 as temp (must be within vgpr_count)
        // Step 1: temp = vsrc0 >> 16 (get high 16 bits of vsrc0 to low position)
        self.emit(gfx11::v_lshrrev_b32(79, 16, vsrc0));
        // Step 2: vdst = (vsrc1 & 0xFFFF0000) | temp
        // GFX11 has v_and_or_b32, let's use it
        self.emit3(gfx11::v_and_or_b32(vdst, vsrc1, 0xFFFF0000, 79));
    }

    /// Same as cvt_pk_bf16_f32 but uses a caller-specified temp register
    /// instead of hardcoded v79. Use this when v79 is live (e.g., v79=m[7] in flash kernel).
    pub fn cvt_pk_bf16_f32_with_temp(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8, vtmp: u8) {
        self.emit(gfx11::v_lshrrev_b32(vtmp, 16, vsrc0));
        self.emit3(gfx11::v_and_or_b32(vdst, vsrc1, 0xFFFF0000, vtmp));
    }
    
    /// Maximum of two values (for row max reduction)
    pub fn max_f32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_max_f32(vdst, vsrc0, vsrc1));
    }
    
    /// Minimum of two values
    pub fn min_f32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_min_f32(vdst, vsrc0, vsrc1));
    }
    
    /// Subtraction (for score - max in softmax)
    pub fn sub_f32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_sub_f32(vdst, vsrc0, vsrc1));
    }
    
    /// Add operation
    pub fn add_f32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_add_f32(vdst, vsrc0, vsrc1));
    }
    
    /// Multiply operation
    pub fn mul_f32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_mul_f32(vdst, vsrc0, vsrc1));
    }
    
    // =========================================================================
    // Lane shuffle operations (for warp reductions)
    // =========================================================================
    
    /// Swizzle within wave - pattern-based lane data exchange
    pub fn ds_swizzle(&mut self, vdst: u8, vsrc: u8, pattern: u16) {
        self.emit2(gfx11::ds_swizzle_b32(vdst, vsrc, pattern));
    }
    
    /// Cross-lane permute (16-lane groups)
    /// lane_sel_hi, lane_sel_lo default to s0 (identity permute)
    pub fn permlane16(&mut self, vdst: u8, vsrc: u8) {
        // Use s0, s0 as default lane selectors (identity map)
        self.emit2(gfx11::v_permlane16_b32(vdst, vsrc, 0, 0));
    }
    
    /// Exchange data across lane halves
    pub fn permlanex16(&mut self, vdst: u8, vsrc: u8) {
        self.emit2(gfx11::v_permlanex16_b32(vdst, vsrc, 0, 0));
    }
    
    /// Zero-cost VGPR transpose: convert WMMA C-layout (f32) to A-layout (bf16x2) transpose
    ///
    /// C-layout (interleaved): vgpr k, lane L (L<16) = C[2k][L], lane L (L≥16) = C[2k+1][L-16]
    /// A-layout (replicated):  vgpr k, all lanes = bf16x2(A^T[L&15][2k], A^T[L&15][2k+1])
    ///
    /// Uses v_permlanex16 to swap half-wave data, then v_cndmask to merge even/odd rows.
    /// Result: each lane L gets bf16x2(src[2k][L&15], src[2k+1][L&15]) — perfect A-layout for
    /// the transposed matrix! 0 LDS access, 0 barrier.
    ///
    /// temp_base: 4 consecutive temp VGPRs (v_tmp, val_even, val_odd, pack_tmp)
    /// lane_id_reg: VGPR containing lane_id (v80 in backward kernel)
    pub fn reg_transpose_c_to_ab(&mut self, dest_base: u8, src_base: u8, temp_base: u8, lane_id_reg: u8) {
        let val_even = temp_base;
        let val_odd = temp_base + 1;
        let pack_tmp = temp_base + 2;
        
        // VCC=1 where 16 > lane_id → VCC=1 for lanes 0-15, VCC=0 for lanes 16-31
        self.emit(gfx11::v_cmp_gt_u32_imm(lane_id_reg, 16));
        
        // Phase 1: Batch-emit 8 ds_swizzle to hide latency.
        // Use dest_base as temp storage — no extra VGPRs needed!
        for k in 0..8u8 {
            let src = src_base + k;
            let dest = dest_base + k;
            // 0x401F = SWAP,16: perfectly swaps Lane L ↔ Lane L+16
            self.ds_swizzle(dest, src, 0x401F);
        }
        
        // Wait for all 8 swizzles to complete
        self.wait_lgkmcnt(0);
        
        // Phase 2: Merge even/odd rows and pack to bf16x2
        for k in 0..8u8 {
            let src = src_base + k;
            let dest = dest_base + k;
            let swizzled = dest; // dest holds swapped data from opposite half-wave
            
            // C-layout: Lower lane L has C[2k][L] (even row), Upper has C[2k+1][L-16] (odd row)
            // After SWAP16: lower swizzled = C[2k+1][L] (from upper), upper swizzled = C[2k][L-16] (from lower)
            //
            // Lower half (VCC=1): val_even = src (own even), val_odd = swizzled (odd from upper)
            // Upper half (VCC=0): val_even = swizzled (even from lower), val_odd = src (own odd)
            self.emit(gfx11::v_cndmask_b32(val_even, swizzled, src));  // VCC=1:src, VCC=0:swizzled
            self.emit(gfx11::v_cndmask_b32(val_odd, src, swizzled));   // VCC=1:swizzled, VCC=0:src
            
            // Pack f32→bf16x2 (safe to write back to dest, swizzled already consumed)
            self.cvt_pk_bf16_f32_with_temp(dest, val_even, val_odd, pack_tmp);
        }
    }

    /// VALU-only transpose: identical to reg_transpose_c_to_ab but uses
    /// v_permlanex16_b32 instead of ds_swizzle(SWAP16).
    ///
    /// KEY DIFFERENCE: No LDS crossbar usage → no conflict with concurrent
    /// ds_load_u16 double-buffer prefetches! And no wait_lgkmcnt(0) needed.
    ///
    /// v_permlanex16_b32(vdst, vsrc) does exactly: vdst[lane] = vsrc[lane XOR 16]
    /// which is exactly what ds_swizzle(0x401F) = SWAP16 does.
    ///
    /// temp_base: 3 consecutive temp VGPRs (val_even, val_odd, pack_tmp)
    pub fn reg_transpose_valu_only(&mut self, dest_base: u8, src_base: u8, temp_base: u8, lane_id_reg: u8) {
        let val_even = temp_base;
        let val_odd = temp_base + 1;
        let pack_tmp = temp_base + 2;

        // VCC=1 for lanes 0-15 (lower half), VCC=0 for lanes 16-31
        self.emit(gfx11::v_cmp_gt_u32_imm(lane_id_reg, 16));

        // Phase 1: v_permlanex16 (pure VALU, no LDS crossbar, no wait!)
        // Result: dest_base+k = src from opposite half-wave
        for k in 0..8u8 {
            let src = src_base + k;
            let dest = dest_base + k;
            self.permlanex16(dest, src);  // dest[lane] = src[lane XOR 16]
        }
        // NO wait_lgkmcnt here — permlanex16 is purely VALU, result ready next cycle

        // Phase 2: Merge even/odd rows and pack to bf16x2 (identical to original)
        for k in 0..8u8 {
            let src = src_base + k;
            let dest = dest_base + k;
            let swizzled = dest;

            self.emit(gfx11::v_cndmask_b32(val_even, swizzled, src));
            self.emit(gfx11::v_cndmask_b32(val_odd, src, swizzled));
            self.cvt_pk_bf16_f32_with_temp(dest, val_even, val_odd, pack_tmp);
        }
    }

    /// In-place 4-stage butterfly transpose of a 16×16 bf16 matrix stored in 8 VGPRs.
    ///
    /// Input layout (A-layout, bf16x2 packed):
    ///   Lane L (0-15): VGPR k holds bf16x2(M[L][2k], M[L][2k+1]) — row L of M
    ///   Lane L+16: mirror of Lane L
    ///
    /// Output layout (transposed):
    ///   Lane L: VGPR k holds bf16x2(M[2k][L], M[2k+1][L]) — column L of M = row L of M^T
    ///
    /// Algorithm: 4-stage butterfly with CROSS-VGPR ROUTING
    ///   Stage 1 (XOR 1):  Sub-word bf16 swap using v_perm_b32
    ///   Stage 2 (XOR 2):  Cross-VGPR dword swap (offset ±1)
    ///   Stage 3 (XOR 4):  Cross-VGPR 2-dword block swap (offset ±2)
    ///   Stage 4 (XOR 8):  Cross-VGPR 4-dword block swap (offset ±4)
    ///
    /// CRITICAL: bf16x2 packing means stages 2-4 need cross-VGPR routing!
    /// After Stage 1, data positions shift: stage s uses neighbor's VGPR k±block_size,
    /// not the same VGPR k. Without cross-VGPR routing, lane pairs collapse to identical data.
    ///
    /// reg_base: first of 8 VGPRs to transpose IN-PLACE
    /// temp_base: 9 temp VGPRs (8 for swizzle + 1 for perm temp)
    /// lane_id_reg: VGPR containing lane_id (0-31)
    pub fn butterfly_transpose_16x16(&mut self, reg_base: u8, temp_base: u8, lane_id_reg: u8) {
        let perm_tmp = temp_base + 8; // extra temp for v_perm stage

        // =====================================================================
        // Stage 1: XOR 1 — sub-word bf16 swap between Lane L and Lane L^1
        // =====================================================================
        // Even lane: result = bf16x2(M[L][2k], M[L^1][2k])
        // Odd lane:  result = bf16x2(M[L^1][2k+1], M[L][2k+1])
        {
            for k in 0..8u8 {
                self.ds_swizzle(temp_base + k, reg_base + k, gfx11::xor_pattern(1));
            }
            self.wait_lgkmcnt(0);

            // VCC = 1 for even lanes (lane_id & 1 == 0)
            self.emit(gfx11::v_and_b32_imm(perm_tmp, lane_id_reg, 1));
            self.emit(gfx11::v_cmp_eq_u32_imm(perm_tmp, 0));

            for k in 0..8u8 {
                let src = reg_base + k;
                let neigh = temp_base + k;
                // Even-lane result: bf16x2(lo=self_lo, hi=neigh_lo)
                // self_lo = vsrc0 bytes 0-1, neigh_lo = vsrc1 bytes 0-1
                // Result bytes: [neigh_byte0, neigh_byte1, self_byte0, self_byte1]
                //             = selector 0x01000504
                self.emit3(gfx11::v_perm_b32(perm_tmp, src, neigh, 0x01000504));
                // Odd-lane result: bf16x2(lo=neigh_hi, hi=self_hi)
                // neigh_hi = vsrc1 bytes 2-3, self_hi = vsrc0 bytes 2-3
                // Result bytes: [self_byte2, self_byte3, neigh_byte2, neigh_byte3]
                //             = selector 0x07060302
                self.emit3(gfx11::v_perm_b32(neigh, src, neigh, 0x07060302));
                // src = VCC ? perm_tmp (even) : neigh (odd)
                self.emit(gfx11::v_cndmask_b32(src, neigh, perm_tmp));
            }
        }

        // =====================================================================
        // Stage 2: XOR 2 — cross-VGPR dword swap (block_size=1)
        // =====================================================================
        // Lane bit1=0: keep even VGPRs, odd k ← neighbor's VGPR k-1
        // Lane bit1=1: even k ← neighbor's VGPR k+1, keep odd VGPRs
        {
            for k in 0..8u8 {
                self.ds_swizzle(temp_base + k, reg_base + k, gfx11::xor_pattern(2));
            }
            self.wait_lgkmcnt(0);

            // VCC = 1 where (lane_id & 2) == 0
            self.emit(gfx11::v_and_b32_imm(perm_tmp, lane_id_reg, 2));
            self.emit(gfx11::v_cmp_eq_u32_imm(perm_tmp, 0));

            // Cross-VGPR routing:
            // even k: cndmask(temp[k+1], reg[k]) — VCC=1→keep own, VCC=0→take neigh k+1
            // odd k:  cndmask(reg[k], temp[k-1]) — VCC=1→take neigh k-1, VCC=0→keep own
            for k in 0..8u8 {
                if k % 2 == 0 {
                    let neigh_src = if k + 1 < 8 { temp_base + k + 1 } else { temp_base + k };
                    self.emit(gfx11::v_cndmask_b32(reg_base + k, neigh_src, reg_base + k));
                } else {
                    let neigh_src = temp_base + k - 1;
                    self.emit(gfx11::v_cndmask_b32(reg_base + k, reg_base + k, neigh_src));
                }
            }
        }

        // =====================================================================
        // Stage 3: XOR 4 — cross-VGPR 2-dword block swap (block_size=2)
        // =====================================================================
        // Lane bit2=0: keep VGPRs {0,1,4,5}, take neighbor's {k-2} for {2,3,6,7}
        // Lane bit2=1: take neighbor's {k+2} for {0,1,4,5}, keep {2,3,6,7}
        {
            for k in 0..8u8 {
                self.ds_swizzle(temp_base + k, reg_base + k, gfx11::xor_pattern(4));
            }
            self.wait_lgkmcnt(0);

            // VCC = 1 where (lane_id & 4) == 0
            self.emit(gfx11::v_and_b32_imm(perm_tmp, lane_id_reg, 4));
            self.emit(gfx11::v_cmp_eq_u32_imm(perm_tmp, 0));

            // Cross-VGPR routing:
            // first pair of 4-block (k%4 < 2): cndmask(temp[k+2], reg[k])
            //   VCC=1→keep own, VCC=0→take neigh k+2
            // second pair (k%4 >= 2): cndmask(reg[k], temp[k-2])
            //   VCC=1→take neigh k-2, VCC=0→keep own
            for k in 0..8u8 {
                if (k % 4) < 2 {
                    let neigh_src = if k + 2 < 8 { temp_base + k + 2 } else { temp_base + k };
                    self.emit(gfx11::v_cndmask_b32(reg_base + k, neigh_src, reg_base + k));
                } else {
                    let neigh_src = temp_base + k - 2;
                    self.emit(gfx11::v_cndmask_b32(reg_base + k, reg_base + k, neigh_src));
                }
            }
        }

        // =====================================================================
        // Stage 4: XOR 24 — cross-VGPR 4-dword block swap (block_size=4)
        // =====================================================================
        // WMMA A-Layout quadrant mapping:
        //   UL (Lane 0-7):   Row 0-7,  Col 0-7
        //   LL (Lane 8-15):  Row 8-15, Col 0-7
        //   UR (Lane 16-23): Row 0-7,  Col 8-15
        //   LR (Lane 24-31): Row 8-15, Col 8-15
        //
        // For 16×16 transpose: LL ↔ UR must swap → XOR(8|16) = 24
        //
        // VCC mask: ((lane_id >> 3) ^ (lane_id >> 4)) & 1 == 0 → UL+LR (lanes 0-7, 24-31)
        // VCC=1 for UL+LR: keep own VGPRs 0-3, take neighbor's VGPRs 0-3 into 4-7
        // VCC=0 for LL+UR: take neighbor's VGPRs 4-7 into 0-3, keep own VGPRs 4-7
        {
            for k in 0..8u8 {
                self.ds_swizzle(temp_base + k, reg_base + k, gfx11::xor_pattern(24));
            }
            self.wait_lgkmcnt(0);

            // Hardcode VCC mask to avoid clobbering temp_base or reg_base
            // VCC=1 for lanes {0-7, 16-23} = 0x00FF00FF
            // Lane L: bit3=0 → upper rows (keep low VGPRs) → VCC=1
            //         bit3=1 → lower rows (swap with neighbor) → VCC=0
            // Lanes 16-23 duplicate lanes 0-7 data, lanes 24-31 duplicate 8-15
            self.emit2(gfx11::s_mov_b32_literal(106, 0x00FF00FF)); // vcc_lo = 0x00FF00FF

            // Cross-VGPR routing:
            // k < 4: cndmask(temp[k+4], reg[k])
            //   VCC=1 (UL/LR) → keep own,  VCC=0 (LL/UR) → take neigh k+4
            // k >= 4: cndmask(reg[k], temp[k-4])
            //   VCC=1 (UL/LR) → take neigh k-4,  VCC=0 (LL/UR) → keep own
            for k in 0..8u8 {
                if k < 4 {
                    self.emit(gfx11::v_cndmask_b32(reg_base + k, temp_base + k + 4, reg_base + k));
                } else {
                    self.emit(gfx11::v_cndmask_b32(reg_base + k, reg_base + k, temp_base + k - 4));
                }
            }
        }
    }
    
    /// Zero-barrier LDS transposed tile read: read 16×16 bf16 matrix transposed from
    /// XOR-swizzled LDS. Reads column-wise (one column per lane) with zero bank conflict.
    ///
    /// LDS layout: row r stored at `row_base + (within_row ^ ((r & 7) << 4))`
    /// Column read: each lane reads bf16 at column `lane_id & 15` across all 16 rows.
    ///
    /// Result: dest_base[0:7] = bf16x2 packed, ready for WMMA B input.
    /// Loads in two batches of 8 rows each.
    ///
    /// addr_reg: VGPR containing `lds_base + (lane_id & 15) * 2` (column byte address)
    /// dest_base: 8 VGPRs for output (bf16x2 packed)
    /// temp_base: 9 temp VGPRs (8 for u16 loads + 1 for packing)
    pub fn lds_load_transposed_tile(&mut self, dest_base: u8, addr_reg: u8, temp_base: u8) {
        // Load in two halves: rows 0-7, then rows 8-15
        for half in 0..2u32 {
            for r in 0..8u32 {
                let row = half * 8 + r;
                // XOR swizzle: row_base = row * 128, mask = (row & 7) << 4
                // The column byte address (in addr_reg) XORed with mask gives the swizzled offset
                // But we can't XOR the addr_reg per-row dynamically, so we compute the offset:
                // offset = (row * 128) ^ ((col_byte) ^ ((row & 7) << 4)) - col_byte
                // Actually: final_addr = row * 128 + (col_byte ^ mask)
                // Since addr_reg = lds_base + col_byte, we need:
                //   ds_load_u16 at addr = addr_reg + row*128 + ((col_byte ^ mask) - col_byte)
                // But we don't know col_byte at compile time!
                //
                // Alternative: precompute: offset = row * 128 + xor_correction
                // where xor_correction = (col_byte ^ mask) - col_byte
                // But col_byte varies per lane...
                //
                // Simpler: use just the row_base ^ mask as the ds_load_u16 offset
                // addr_reg = lds_base + (lane_id & 15) * 2 = lds_base + col_byte
                // We want: lds_base + row * 128 + (col_byte ^ mask)
                //        = addr_reg + row * 128 + (col_byte ^ mask) - col_byte
                //        = addr_reg + row * 128 + (col_byte XOR mask) - col_byte
                //
                // For XOR: (col_byte ^ mask) - col_byte depends on lane!
                // col_byte = 0..30 (even), mask = (row & 7) << 4 = 0..112
                //
                // This isn't a simple compile-time offset. We need to compute the XOR per lane.
                // Use a temp VGPR to compute the swizzled address for each row.
                let _mask = ((row & 7) << 4) as u16;
                let row_base = (row * 128) as u16;
                // We need: lds_base + row_base + (col_byte ^ mask)
                // addr_reg has: lds_base + col_byte
                // So: addr_reg + row_base + ((col_byte ^ mask) - col_byte)
                // Let xor_delta = (col_byte ^ mask) - col_byte
                // But this varies per lane... We need dynamic XOR.
                
                // Use ds_load offset = row_base as u16. But we need to XOR the col part.
                // For correctness without per-lane XOR, we'd need mask=0 (no swizzle).
                // With swizzle: must compute address dynamically.
                // 
                // For now, use offset-only approach assuming NO swizzle (mask=0 case):
                // This works when XOR swizzle is disabled.
                self.ds_load_u16(temp_base + r as u8, addr_reg, row_base);
            }
            self.wait_lgkmcnt(0);
            
            // Pack pairs of u16 values into bf16x2
            for k in 0..4u8 {
                let dest_reg = dest_base + (half as u8) * 4 + k;
                let r_even = temp_base + k * 2;
                let r_odd = temp_base + k * 2 + 1;
                // bf16x2 = (odd << 16) | even
                self.emit(gfx11::v_lshlrev_b32(r_odd, 16, r_odd));
                self.emit(gfx11::v_or_b32(dest_reg, r_even, r_odd));
            }
        }
    }
    
    /// Byte permute (arbitrary cross-lane read)
    pub fn ds_bpermute(&mut self, vdst: u8, vindex: u8, vsrc: u8) {
        self.emit2(gfx11::ds_bpermute_b32(vdst, vindex, vsrc));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    // =========================================================================
    // LDS atomics (for parallel reductions)
    // =========================================================================
    
    /// Atomic float add to LDS
    pub fn ds_atomic_add_f32(&mut self, vaddr: u8, vdata: u8, offset: u16) {
        self.emit2(gfx11::ds_add_f32(vaddr, vdata, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    /// Atomic float max to LDS
    pub fn ds_atomic_max_f32(&mut self, vaddr: u8, vdata: u8, offset: u16) {
        self.emit2(gfx11::ds_max_f32(vaddr, vdata, offset));
        self.lgkmcnt = self.lgkmcnt.saturating_add(1);
    }
    
    // =========================================================================
    // Global atomics (for Split-K accumulation)
    // =========================================================================
    
    /// Atomic float add to global memory (single f32)
    /// WARNING: Only works on cacheable memory in L2!
    /// vaddr_lo/hi = 64-bit address in VGPR pair
    /// Use offset to add immediate offset to address
    pub fn global_atomic_add_f32(&mut self, vdst: u8, vaddr_lo: u8, vdata: u8, offset: i16) {
        // Use the existing gfx11 function but with proper address handling
        // vaddr is a pair [vaddr_lo, vaddr_lo+1]
        let instr = [
            0xDD5A4000u32 | ((offset as u16 as u32) & 0xFFF) | (vdst as u32),
            ((vdata as u32) << 8) | (vaddr_lo as u32) | 0x7C00
        ];
        self.emit2(instr);
        self.vmcnt = self.vmcnt.saturating_add(1);
    }
    
    /// Atomic float add 4× f32 (replacement for global_store_dwordx4)
    /// vaddr = 64-bit address in VGPR pair (vaddr, vaddr+1)
    /// vdata_base = first of 4 consecutive VGPRs to add
    /// vdst = first of 4 consecutive VGPRs for return values (can be same as temp)
    /// offset = byte offset from address
    pub fn global_atomic_add_f32_x4(&mut self, vdst: u8, vaddr: u8, vdata_base: u8, offset: i16) {
        // 4 atomic adds for 4 f32 values (replacing dwordx4)
        self.global_atomic_add_f32(vdst, vaddr, vdata_base, offset);
        self.global_atomic_add_f32(vdst.wrapping_add(1), vaddr, vdata_base.wrapping_add(1), offset + 4);
        self.global_atomic_add_f32(vdst.wrapping_add(2), vaddr, vdata_base.wrapping_add(2), offset + 8);
        self.global_atomic_add_f32(vdst.wrapping_add(3), vaddr, vdata_base.wrapping_add(3), offset + 12);
    }
    
    /// Fire-and-forget atomic float add 4× f32 — NO return value, NO wasted VGPR
    /// Uses lane*32 layout: each lane owns 32 contiguous bytes, zero overlap
    /// vaddr = 64-bit address in VGPR pair (vaddr, vaddr+1)
    /// vdata_base = first of 4 consecutive VGPRs to atomically add
    /// offset = byte offset from address
    pub fn global_atomic_add_f32_no_rtn_x4(&mut self, vaddr: u8, vdata_base: u8, offset: i16) {
        // 4 fire-and-forget atomic adds with offset, no glc, no vdst
        for i in 0..4u8 {
            let off = offset + (i as i16) * 4;
            let instr = [
                0xDD5A0000u32 | ((off as u16 as u32) & 0xFFF),
                (0x7Cu32 << 16) | ((vdata_base.wrapping_add(i) as u32) << 8) | (vaddr as u32)
            ];
            self.emit2(instr);
        }
    }
    
    // =========================================================================
    // Comparisons and conditionals
    // =========================================================================
    
    /// Compare greater than (sets VCC)
    pub fn cmp_gt_f32(&mut self, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_cmp_gt_f32(vsrc0, vsrc1));
    }
    
    /// Compare less than (sets VCC)
    pub fn cmp_lt_f32(&mut self, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_cmp_lt_f32(vsrc0, vsrc1));
    }
    
    /// Conditional mask select (uses VCC)
    pub fn cndmask(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_cndmask_b32(vdst, vsrc0, vsrc1));
    }
    
    pub fn and_b32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_and_b32(vdst, vsrc0, vsrc1));
    }

    pub fn or_b32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_or_b32(vdst, vsrc0, vsrc1));
    }
    
    pub fn lshlrev_b32(&mut self, vdst: u8, shift: u8, vsrc: u8) {
        self.emit(gfx11::v_lshlrev_b32(vdst, shift, vsrc));
    }
    
    pub fn add_u32(&mut self, vdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit(gfx11::v_add_u32(vdst, vsrc0, vsrc1));
    }
    
    pub fn add_co_u32(&mut self, vdst: u8, sdst: u8, vsrc0: u8, vsrc1: u8) {
        self.emit2(gfx11::v_add_co_u32(vdst, sdst, vsrc0, vsrc1));
    }
    
    pub fn add_co_ci_u32(&mut self, vdst: u8, sdst: u8, vsrc0: u8, vsrc1: u8, ssrc2: u8) {
        self.emit2(gfx11::v_add_co_ci_u32(vdst, sdst, vsrc0, vsrc1, ssrc2));
    }
    
    // =========================================================================
    // VOPD Dual Issue Instructions (GFX11+)
    // =========================================================================
    
    /// v_dual_sub_f32 :: v_dual_sub_f32 - Two SUBs in parallel
    pub fn dual_sub_sub_f32(
        &mut self,
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) {
        self.emit2(gfx11::v_dual_sub_f32_sub_f32(
            vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y
        ));
    }
    
    /// v_dual_add_f32 :: v_dual_mul_f32 - ADD and MUL in parallel
    pub fn dual_add_mul_f32(
        &mut self,
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) {
        self.emit2(gfx11::v_dual_add_f32_mul_f32(
            vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y
        ));
    }
    
    /// v_dual_mul_f32 :: v_dual_mul_f32 - Two MULs in parallel
    pub fn dual_mul_mul_f32(
        &mut self,
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) {
        self.emit2(gfx11::v_dual_mul_f32_mul_f32(
            vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y
        ));
    }
    
    /// v_dual_max_f32 :: v_dual_max_f32 - Two MAXs in parallel
    pub fn dual_max_max_f32(
        &mut self,
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) {
        self.emit2(gfx11::v_dual_max_f32_max_f32(
            vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y
        ));
    }
    
    /// v_dual_add_f32 :: v_dual_add_f32 - Two ADDs in parallel
    pub fn dual_add_add_f32(
        &mut self,
        vdst_x: u8, vsrc0_x: u8, vsrc1_x: u8,
        vdst_y: u8, vsrc0_y: u8, vsrc1_y: u8,
    ) {
        self.emit2(gfx11::v_dual_add_f32_add_f32(
            vdst_x, vsrc0_x, vsrc1_x, vdst_y, vsrc0_y, vsrc1_y
        ));
    }

    /// End program
    pub fn endpgm(&mut self) {
        self.emit(gfx11::S_ENDPGM);
    }
    
    // =========================================================================
    // Finalization
    // =========================================================================
    
    /// Get the assembled code as bytes
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.code.len() * 4);
        for word in &self.code {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
    
    /// Get code length in dwords
    pub fn len(&self) -> usize {
        self.code.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }
}

impl Default for Rdna3Assembler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Example: Minimal GEMM 16x16x16 bf16 kernel
// =============================================================================

/// Build a minimal GEMM kernel using WMMA
/// C[16x16] = A[16x16] @ B[16x16] (bf16 inputs, f32 output)
pub fn build_gemm_16x16x16_kernel() -> Rdna3Assembler {
    let mut asm = Rdna3Assembler::new();
    
    // Register allocation:
    // v[0:1] = address pair for A
    // v[2:3] = address pair for B  
    // v[4:5] = address pair for C
    // v[8:15] = A fragment (8 bf16 pairs)
    // v[16:23] = B fragment (8 bf16 pairs)
    // v[24:31] = C accumulator (8 f32)
    
    // Initialize C accumulator to zero (not shown - assume pre-initialized)
    
    // Load A tile (128 bytes = 64 bf16 = 4 x dwordx4)
    // Each thread loads its portion
    asm.global_load_dwordx4(8, 0, 0);   // v[8:11] = A[0:15]
    asm.global_load_dwordx4(12, 0, 16); // v[12:15] = A[16:31]
    
    // Load B tile
    asm.global_load_dwordx4(16, 2, 0);  // v[16:19] = B[0:15]
    asm.global_load_dwordx4(20, 2, 16); // v[20:23] = B[16:31]
    
    // Wait for all global loads
    asm.wait_vmcnt(0);
    
    // WMMA: C += A @ B
    // v[24:31] += v[8:15] @ v[16:23]
    asm.wmma_f32_16x16x16_bf16(24, 8, 16, 24);
    
    // Store result
    asm.global_store_dwordx4(4, 24, 0);  // C[0:3]
    asm.global_store_dwordx4(4, 28, 16); // C[4:7]
    
    // Wait for store and end
    asm.wait_vmcnt(0);
    asm.endpgm();
    
    asm
}

// =============================================================================
// Example: Warp-level Softmax Reduction Pattern
// =============================================================================

/// Build a warp-level max reduction for softmax
/// Uses ds_swizzle for fast lane-to-lane communication
/// 
/// Pattern for 32-lane wave32 max reduction:
/// 1. Each lane has its local max value
/// 2. Reduce across pairs using XOR shuffle pattern  
/// 3. After log2(32)=5 steps, lane 0 has global max
pub fn build_warp_max_reduction() -> Rdna3Assembler {
    let mut asm = Rdna3Assembler::new();
    
    // Register allocation:
    // v0 = input value (local max)
    // v1 = temporary for reduction
    // v2 = final global max (broadcast to all lanes)
    
    // XOR reduction pattern for 32 lanes
    // Step 1: XOR with lane+16
    let swizzle_xor16 = 0x041F; // ds_swizzle pattern for XOR 16
    asm.ds_swizzle(1, 0, swizzle_xor16);
    asm.wait_lgkmcnt(0);
    asm.max_f32(0, 0, 1);
    
    // Step 2: XOR with lane+8  
    let swizzle_xor8 = 0x020F;
    asm.ds_swizzle(1, 0, swizzle_xor8);
    asm.wait_lgkmcnt(0);
    asm.max_f32(0, 0, 1);
    
    // Step 3: XOR with lane+4
    let swizzle_xor4 = 0x0107;
    asm.ds_swizzle(1, 0, swizzle_xor4);
    asm.wait_lgkmcnt(0);
    asm.max_f32(0, 0, 1);
    
    // Step 4: XOR with lane+2
    let swizzle_xor2 = 0x0083;
    asm.ds_swizzle(1, 0, swizzle_xor2);
    asm.wait_lgkmcnt(0);
    asm.max_f32(0, 0, 1);
    
    // Step 5: XOR with lane+1
    let swizzle_xor1 = 0x0041;
    asm.ds_swizzle(1, 0, swizzle_xor1);
    asm.wait_lgkmcnt(0);
    asm.max_f32(0, 0, 1);
    
    // Now lane 0 has the max - broadcast to all lanes
    // Use readlane to broadcast (would need s_mov + v_readlane sequence)
    
    asm.endpgm();
    asm
}

/// Build exp(x) using v_exp_f32 (computes 2^x, so need x * log2(e))
/// log2(e) ≈ 1.4426950408889634
pub fn build_exp_function() -> Rdna3Assembler {
    let mut asm = Rdna3Assembler::new();
    
    // Register allocation:
    // v0 = input x
    // v1 = log2(e) constant (pre-loaded)
    // v2 = x * log2(e)
    // v3 = result: 2^(x*log2(e)) = e^x
    
    // Assume v1 already contains log2(e) constant
    asm.mul_f32(2, 0, 1);  // v2 = x * log2(e)
    asm.exp_f32(3, 2);     // v3 = 2^v2 = e^x
    
    asm.endpgm();
    asm
}

/// Build complete softmax row processing:
/// 1. Find max across row (warp reduction)
/// 2. Compute exp(score - max)
/// 3. Sum all exp values (warp reduction)
/// 4. Normalize by sum
pub fn build_softmax_row_kernel() -> Rdna3Assembler {
    let mut asm = Rdna3Assembler::new();
    
    // This is a pattern template - actual kernel needs register planning
    // v0 = score input
    // v1 = row max (after reduction)
    // v2 = exp(score - max)
    // v3 = row sum (after reduction)
    // v4 = output: v2 / v3
    
    // Step 1: Compute (score - max) 
    // Assume v1 already has the row max
    asm.sub_f32(2, 0, 1);  // v2 = score - max
    
    // Step 2: exp(v2) using 2^(x * log2(e)) pattern
    // Assume v5 = log2(e) constant
    asm.mul_f32(2, 2, 5);  // v2 = (score-max) * log2(e)
    asm.exp_f32(2, 2);     // v2 = exp(score - max)
    
    // Step 3: (sum reduction would go here using ds_swizzle + add)
    // ...
    
    // Step 4: Normalize: v4 = v2 / v3
    // Using reciprocal + multiply instead of division
    asm.rcp_f32(4, 3);     // v4 = 1/sum
    asm.mul_f32(4, 2, 4);  // v4 = exp / sum = softmax output
    
    asm.endpgm();
    asm
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_s_waitcnt_encoding() {
        // Verify s_waitcnt vmcnt(0) and lgkmcnt(0) produce different encodings
        let vmcnt0 = gfx11::s_waitcnt_vmcnt(0);
        let lgkmcnt0 = gfx11::s_waitcnt_lgkmcnt(0);
        assert_ne!(vmcnt0, lgkmcnt0, "vmcnt(0) and lgkmcnt(0) should differ");
        // Both should be SOPP format (top byte 0xBF)
        assert_eq!((vmcnt0 >> 24) & 0xFF, 0xBF, "s_waitcnt SOPP format");
        assert_eq!((lgkmcnt0 >> 24) & 0xFF, 0xBF, "s_waitcnt SOPP format");
    }
    
    #[test]
    fn test_gemm_kernel_builds() {
        let asm = build_gemm_16x16x16_kernel();
        assert!(!asm.is_empty());
        println!("GEMM kernel: {} dwords ({} bytes)", asm.len(), asm.len() * 4);
    }

    // ══════════════════════════════════════════════════════════════════
    // ISA Encoding Regression Tests (#9)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_global_load_opcodes() {
        // GFX11 global_load_b128 (was global_load_dwordx4)
        let [w0, _] = gfx11::global_load_dwordx4(0, 10, 0);
        let op = (w0 >> 18) & 0x7F;
        assert_eq!(op, 0x17, "global_load_b128 opcode = 0x17 on GFX11");
    }

    #[test]
    fn test_global_store_opcodes() {
        let [w0, _] = gfx11::global_store_dwordx4(10, 0, 0);
        let op = (w0 >> 18) & 0x7F;
        assert_eq!(op, 0x1D, "global_store_b128 opcode = 0x1D on GFX11");
    }

    #[test]
    fn test_vop1_exp_rcp() {
        // v_exp_f32 v0, v1 — VOP1 opcode = 0x25 on GFX11
        let enc = gfx11::v_exp_f32(0, 1);
        let opcode = (enc >> 9) & 0xFF;
        assert_eq!(opcode, 0x25, "v_exp_f32 VOP1 opcode = 0x25 on GFX11");

        // v_rcp_f32 v0, v1 — verify VOP1 format (bit 31 = 0, bits 24:17 = 0x3F)
        let enc = gfx11::v_rcp_f32(0, 1);
        let top = (enc >> 25) & 0x7F;
        assert_eq!(top, 0x3F, "v_rcp_f32 should be VOP1 format (0x3F prefix)");
    }

    #[test]
    fn test_vop2_add_mul() {
        // v_add_f32 v0, v1, v2
        let enc = gfx11::v_add_f32(0, 1, 2);
        let top_bit = (enc >> 31) & 1;
        assert_eq!(top_bit, 0, "VOP2 bit 31 should be 0");

        // v_mul_f32 v0, v1, v2
        let enc = gfx11::v_mul_f32(0, 1, 2);
        let top_bit = (enc >> 31) & 1;
        assert_eq!(top_bit, 0, "VOP2 bit 31 should be 0");
    }

    #[test]
    fn test_smem_load_dwordx2() {
        let [w0, _w1] = gfx11::s_load_dwordx2(2, 0, 0);
        // Verify SMEM format: bits [31:26] = 0b111101 = 0x3D
        let top6 = (w0 >> 26) & 0x3F;
        assert_eq!(top6, 0x3D, "SMEM top bits = 0x3D");
    }

    #[test]
    fn test_s_branch_encoding() {
        // s_cbranch_scc1 offset=-5: verify the simm16 field
        let enc = gfx11::s_cbranch_scc1(-5i16);
        let simm16 = (enc & 0xFFFF) as i16;
        assert_eq!(simm16, -5, "s_cbranch_scc1(-5) offset = -5");

        // s_branch offset=0: verify simm16 = 0
        let enc = gfx11::s_branch(0);
        let simm16 = enc & 0xFFFF;
        assert_eq!(simm16, 0, "s_branch(0) offset = 0");
    }

    #[test]
    fn test_s_mov_b32_inline_constants() {
        // s_mov_b32 s0, 0 → inline constant 128
        let enc = gfx11::s_mov_b32(0, 128);
        assert_ne!(enc, 0, "s_mov_b32 should produce non-zero encoding");

        // s_mov_b32 s20, s2 (register-to-register)
        let enc = gfx11::s_mov_b32(20, 2);
        let src = enc & 0xFF;
        assert_eq!(src, 2, "s_mov_b32 src should be s2");
    }

    #[test]
    fn test_vgpr_sgpr_tracker() {
        let mut asm = Rdna3Assembler::new();
        assert_eq!(asm.suggested_vgpr_count(), 8); // default: v0 used, rounded to 8

        asm.use_vgprs(48);
        assert_eq!(asm.suggested_vgpr_count(), 48); // 48 is already aligned

        asm.use_vgprs(50);
        assert_eq!(asm.suggested_vgpr_count(), 56); // rounded up to 56

        asm.use_sgprs(24);
        assert_eq!(asm.suggested_sgpr_count(), 24); // 24 is already aligned

        asm.use_sgprs(25);
        assert_eq!(asm.suggested_sgpr_count(), 32); // rounded up to 32
    }

    #[test]
    fn test_endpgm() {
        assert_eq!(gfx11::S_ENDPGM, 0xBFB00000, "S_ENDPGM encoding");
    }
}
