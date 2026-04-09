//! GPU Printf — ring buffer based printf for KFD bare-metal kernels.
//!
//! Since KFD bypasses the HIP runtime, there's no built-in printf support.
//! This module provides a lightweight printf mechanism:
//!
//! ## Architecture
//! ```text
//! GPU side:
//!   1. atomic_add(counter, msg_size) → get write offset
//!   2. global_store format_id + args to ring_buf[offset..]
//!
//! Host side:
//!   1. Allocate ring buffer (64KB) + counter (u32)
//!   2. After dispatch: read counter, decode messages
//! ```
//!
//! ## Message format (16 bytes fixed per message)
//! | Bytes  | Content                          |
//! |--------|----------------------------------|
//! | 0..4   | format_id: u32 (registered fmt)  |
//! | 4..8   | arg0: u32 (bit pattern)          |
//! | 8..12  | arg1: u32                        |
//! | 12..16 | arg2: u32                        |

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
use std::sync::Arc;
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
use crate::kfd::{GpuBuffer, KfdDevice};

/// Size of printf ring buffer in bytes (64KB = 4096 messages).
const RING_BUF_SIZE: usize = 64 * 1024;

/// Size of each printf message in bytes.
pub const MSG_SIZE: u32 = 16;

/// Maximum number of messages before ring buffer wraps.
const MAX_MESSAGES: u32 = (RING_BUF_SIZE as u32) / MSG_SIZE;

/// GPU printf context — manages ring buffer + counter on GPU side,
/// decodes messages on host side.
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub struct GpuPrintfCtx {
    /// Ring buffer for printf messages (64KB, GPU-visible).
    ring_buf: GpuBuffer,
    /// Atomic counter (u32) — current write offset in bytes.
    counter_buf: GpuBuffer,
    /// Registered format strings: format_id → format string.
    formats: Vec<String>,
}

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
impl GpuPrintfCtx {
    /// Create a new GPU printf context.
    pub fn new(device: &Arc<KfdDevice>) -> Result<Self, String> {
        let ring_buf = device.alloc_vram(RING_BUF_SIZE)?;
        ring_buf.zero();

        let counter_buf = device.alloc_vram(256)?;  // 256-byte aligned for atomic
        counter_buf.zero();

        Ok(Self {
            ring_buf,
            counter_buf,
            formats: Vec::new(),
        })
    }

    /// Register a printf format string, returns its format_id.
    ///
    /// Supported format specifiers:
    /// - `%u` — u32 decimal
    /// - `%d` — i32 decimal  
    /// - `%x` — u32 hex (lowercase)
    /// - `%08x` — u32 hex (zero-padded 8 digits)
    /// - `%f` — f32 (reinterpret bits as float)
    /// - `%e` — f32 scientific notation
    pub fn register_format(&mut self, fmt: &str) -> u32 {
        let id = self.formats.len() as u32;
        self.formats.push(fmt.to_string());
        id
    }

    /// GPU address of the ring buffer (pass as kernarg).
    pub fn ring_buf_addr(&self) -> u64 {
        self.ring_buf.gpu_addr()
    }

    /// GPU address of the atomic counter (pass as kernarg).
    pub fn counter_addr(&self) -> u64 {
        self.counter_buf.gpu_addr()
    }

    /// Reset counter to 0 (call before each dispatch).
    pub fn reset(&self) {
        self.counter_buf.zero();
    }

    /// Read message count (how many messages were written).
    pub fn message_count(&self) -> u32 {
        let mut buf = [0u8; 4];
        self.counter_buf.read(&mut buf);
        let byte_offset = u32::from_le_bytes(buf);
        (byte_offset / MSG_SIZE).min(MAX_MESSAGES)
    }

    /// Flush: read all messages from ring buffer, format and print to stderr.
    /// Returns the formatted messages as a Vec<String>.
    pub fn flush(&self) -> Vec<String> {
        let count = self.message_count();
        if count == 0 {
            return Vec::new();
        }

        let read_bytes = (count * MSG_SIZE) as usize;
        let mut raw = vec![0u8; read_bytes];
        self.ring_buf.read(&mut raw);

        let mut messages = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let base = i * MSG_SIZE as usize;
            let fmt_id = u32::from_le_bytes(raw[base..base+4].try_into().unwrap());
            let arg0 = u32::from_le_bytes(raw[base+4..base+8].try_into().unwrap());
            let arg1 = u32::from_le_bytes(raw[base+8..base+12].try_into().unwrap());
            let arg2 = u32::from_le_bytes(raw[base+12..base+16].try_into().unwrap());

            let msg = self.format_message(fmt_id, [arg0, arg1, arg2]);
            eprintln!("[gpu_printf] {}", msg);
            messages.push(msg);
        }

        messages
    }

    /// Format a single message using registered format string.
    pub fn format_message(&self, fmt_id: u32, args: [u32; 3]) -> String {
        let fmt = match self.formats.get(fmt_id as usize) {
            Some(f) => f,
            None => return format!("<unknown fmt_id={} args=[{}, {}, {}]>",
                fmt_id, args[0], args[1], args[2]),
        };

        let mut result = String::new();
        let mut arg_idx = 0usize;
        let mut chars = fmt.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '%' && chars.peek().is_some() {
                let arg = if arg_idx < 3 { args[arg_idx] } else { 0 };

                // Check for zero-pad width prefix
                let mut zero_pad = false;
                let mut width = 0u32;
                if chars.peek() == Some(&'0') {
                    zero_pad = true;
                    chars.next();
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            width = width * 10 + (d as u32 - '0' as u32);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }

                match chars.next() {
                    Some('u') => {
                        result.push_str(&format!("{}", arg));
                        arg_idx += 1;
                    }
                    Some('d') => {
                        result.push_str(&format!("{}", arg as i32));
                        arg_idx += 1;
                    }
                    Some('x') => {
                        if zero_pad && width > 0 {
                            result.push_str(&format!("{:0>width$x}", arg, width = width as usize));
                        } else {
                            result.push_str(&format!("{:x}", arg));
                        }
                        arg_idx += 1;
                    }
                    Some('f') => {
                        let f = f32::from_bits(arg);
                        result.push_str(&format!("{:.6}", f));
                        arg_idx += 1;
                    }
                    Some('e') => {
                        let f = f32::from_bits(arg);
                        result.push_str(&format!("{:.4e}", f));
                        arg_idx += 1;
                    }
                    Some('%') => result.push('%'),
                    Some(other) => {
                        result.push('%');
                        result.push(other);
                    }
                    None => result.push('%'),
                }
            } else {
                result.push(c);
            }
        }

        result
    }
}

// ============================================================================
// T0Kernel integration — emit printf ISA sequence
// ============================================================================

use super::ir::*;
use super::compile::T0Kernel;

/// Emit a printf call into a T0Kernel.
///
/// This generates an ISA sequence that:
/// 1. Masks EXEC to lane 0 only (SaveExec)
/// 2. Atomic-adds MSG_SIZE to the counter → gets write offset
/// 3. Bounds-checks offset < RING_BUF_SIZE
/// 4. Stores format_id + up to 3 args to ring_buf[offset]
/// 5. Restores EXEC
///
/// # Arguments
/// * `k` — T0Kernel being built
/// * `counter_ptr` — SGPR pair holding counter GPU address (from kernargs)
/// * `ring_buf_ptr` — SGPR pair holding ring buffer GPU address
/// * `format_id` — Pre-registered format ID
/// * `args` — Up to 3 VRegs containing u32/f32 values to print
pub fn emit_printf(
    k: &mut T0Kernel,
    counter_ptr: SRegPair,
    ring_buf_ptr: SRegPair,
    format_id: u32,
    args: &[VReg],
) {
    assert!(args.len() <= 3, "gpu_printf: max 3 args, got {}", args.len());

    // Only lane 0 performs the printf (SaveExec masks to VCC)
    let saved_exec = k.alloc_sreg();
    let lane0_flag = k.alloc_vreg();
    k.v_and_b32_imm(lane0_flag, VReg(0), 31);
    k.push(Op::VCmpEqU32Imm { src: lane0_flag, imm: 0 });
    k.push(Op::SaveExec { dst: saved_exec });

    // Step 1: Atomic add MSG_SIZE to counter → old_offset
    let counter_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(counter_addr, SReg(counter_ptr.0));
    k.v_mov_from_sgpr(VReg(counter_addr.0 + 1), SReg(counter_ptr.0 + 1));

    let msg_size_v = k.alloc_vreg();
    k.v_mov_imm(msg_size_v, MSG_SIZE as i32);

    // Atomic add u32 with return: old_offset = atomicAdd(counter, MSG_SIZE)
    let old_offset = k.alloc_vreg();
    k.push(Op::GlobalAtomicAddU32Rtn { dst: old_offset, addr: counter_addr, src: msg_size_v });
    k.wait_vmcnt(0);

    // Step 2: Bounds check
    let ring_size_v = k.alloc_vreg();
    k.v_mov_imm(ring_size_v, RING_BUF_SIZE as i32);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(old_offset), src1: Operand::VReg(ring_size_v) });
    let saved_exec2 = k.alloc_sreg();
    k.push(Op::SaveExec { dst: saved_exec2 });

    // Step 3: Store message
    let write_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(write_addr, SReg(ring_buf_ptr.0));
    k.v_mov_from_sgpr(VReg(write_addr.0 + 1), SReg(ring_buf_ptr.0 + 1));
    k.v_add_co(write_addr, write_addr, old_offset);
    k.v_add_co_ci(VReg(write_addr.0 + 1), VReg(write_addr.0 + 1));

    let fmt_v = k.alloc_vreg();
    k.v_mov_imm(fmt_v, format_id as i32);
    k.global_store(write_addr, fmt_v, Width::B32, 0);

    for i in 0..3u32 {
        let val = if (i as usize) < args.len() {
            args[i as usize]
        } else {
            let z = k.alloc_vreg();
            k.v_mov_imm(z, 0);
            z
        };
        k.global_store(write_addr, val, Width::B32, (i as i32 + 1) * 4);
    }

    // Restore EXEC
    k.push(Op::RestoreExec { src: saved_exec2 });
    k.push(Op::RestoreExec { src: saved_exec });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_printf_compiles() {
        // Verify emit_printf generates valid T0Kernel ops (no GPU needed)
        let mut k = T0Kernel::new("printf_test");
        let _x_ptr = k.arg_ptr("x");
        let counter_ptr = k.arg_ptr("printf_counter");
        let ring_buf_ptr = k.arg_ptr("printf_ring_buf");
        k.emit_arg_loads();

        let tid = k.alloc_vreg();
        k.v_and_b32_imm(tid, VReg(0), 31);

        emit_printf(&mut k, counter_ptr, ring_buf_ptr, 0, &[tid]);

        k.endpgm();

        // Should produce valid assembly
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("global_atomic_add_u32"), "should contain atomic: {}", asm);
        assert!(asm.contains("s_endpgm"), "should contain endpgm");
        eprintln!("printf kernel:\n{}", asm);
    }

    /// GPU integration test: dispatch a printf kernel and verify host-side decode.
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_gpu_printf_e2e() {
        use std::sync::OnceLock;
        use crate::ignis::gpu_context::GpuRuntime;

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
        }).0.clone();

        let mut pctx = GpuPrintfCtx::new(&rt.device).unwrap();
        let fmt_id = pctx.register_format("hello from GPU: tid=%u");
        pctx.reset();

        // Build and compile kernel via ensure_kernel_t0
        let captured_fmt_id = fmt_id;
        let kernel = rt.ensure_kernel_t0(
            "printf_e2e_test",
            move || {
                let mut k = T0Kernel::new("printf_e2e_test");
                let counter_ptr = k.arg_ptr("printf_counter");
                let ring_buf_ptr = k.arg_ptr("printf_ring_buf");
                k.emit_arg_loads();

                let tid = k.alloc_vreg();
                k.v_and_b32_imm(tid, VReg(0), 31);
                emit_printf(&mut k, counter_ptr, ring_buf_ptr, captured_fmt_id, &[tid]);
                k.wait_vscnt(0);
                k.endpgm();
                k
            },
            [32, 1, 1],
            0,
        ).expect("ensure_kernel_t0 failed");

        let ka = crate::kernargs![
            pctx.counter_addr() => u64,
            pctx.ring_buf_addr() => u64
        ];
        rt.dispatch(&kernel, [32, 1, 1], &ka).expect("dispatch");

        // Flush and verify
        let messages = pctx.flush();
        assert_eq!(messages.len(), 1, "expected 1 message (lane 0), got {}", messages.len());
        assert!(messages[0].contains("tid=0"), "expected tid=0, got: {}", messages[0]);
        eprintln!("✓ GPU printf E2E passed: {:?}", messages);
    }

    /// Test format_message (pure CPU — needs GPU only for buffer allocation).
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[test]
    fn test_format_message() {
        use crate::kfd::KfdDevice;

        let device = KfdDevice::open().expect("Need GPU");
        let mut ctx = GpuPrintfCtx::new(&device).unwrap();

        // Basic format
        let id = ctx.register_format("tid=%u val=%f hex=%08x");
        let msg = ctx.format_message(id, [42, 0x3F800000, 0xDEADBEEF]);
        assert!(msg.contains("tid=42"), "got: {}", msg);
        assert!(msg.contains("1.000000"), "f32 1.0, got: {}", msg);
        assert!(msg.contains("deadbeef"), "hex, got: {}", msg);

        // Signed
        let id2 = ctx.register_format("signed=%d");
        let msg2 = ctx.format_message(id2, [(-5i32 as u32), 0, 0]);
        assert!(msg2.contains("signed=-5"), "got: {}", msg2);

        // Literal %%
        let id3 = ctx.register_format("100%%");
        let msg3 = ctx.format_message(id3, [0, 0, 0]);
        assert!(msg3.contains("100%"), "got: {}", msg3);

        eprintln!("✓ format_message tests passed");
    }
}
