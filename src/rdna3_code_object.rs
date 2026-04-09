//! AMD GPU Code Object Generator for RDNA3
//!
//! Generates AMD HSA Code Object (ELF format) from assembled RDNA3 instructions.
//! The generated binary can be loaded via `hipModuleLoadData()`.
//!
//! ## AMD GPU Code Object Structure
//!
//! ```text
//! ┌─────────────────────┐
//! │    ELF Header       │  (64 bytes for ELF64)
//! ├─────────────────────┤
//! │  Program Headers    │  (describe segments)
//! ├─────────────────────┤
//! │  .text section      │  (kernel machine code)
//! ├─────────────────────┤
//! │  .rodata section    │  (constants)
//! ├─────────────────────┤
//! │  .note section      │  (AMDGPU metadata)
//! ├─────────────────────┤
//! │  Section Headers    │
//! └─────────────────────┘
//! ```
//!
//! Reference: AMD ROCm Documentation, LLVM AMDGPU Backend

use crate::llvm_toolchain::{find_clang, find_ld_lld};
use crate::rdna3_asm::Rdna3Assembler;
use crate::t0::ir::Target;

/// Kernel configuration for code object generation
#[derive(Clone, Debug)]
pub struct KernelConfig {
    /// Kernel name (symbol name)
    pub name: String,
    /// Local data share (LDS) size in bytes
    pub lds_size: u32,
    /// Kernel argument size in bytes
    pub kernarg_size: u32,
    /// Number of VGPRs used
    pub vgpr_count: u8,
    /// Number of SGPRs used
    pub sgpr_count: u8,
    /// Workgroup size X
    pub workgroup_size_x: u16,
    /// Workgroup size Y
    pub workgroup_size_y: u16,
    /// Workgroup size Z
    pub workgroup_size_z: u16,
    /// Scratch (private segment) size per work-item in bytes (0 = no scratch)
    pub scratch_size: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            name: "kernel".to_string(),
            lds_size: 0,
            kernarg_size: 64, // 8 pointers
            vgpr_count: 32,
            sgpr_count: 16,
            workgroup_size_x: 256,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            scratch_size: 0,
        }
    }
}

/// AMD GPU Kernel Descriptor (64 bytes)
/// See: https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct KernelDescriptor {
    /// GROUP_SEGMENT_FIXED_SIZE (LDS size)
    pub group_segment_fixed_size: u32,
    /// PRIVATE_SEGMENT_FIXED_SIZE (scratch size per work-item)
    pub private_segment_fixed_size: u32,
    /// KERNARG_SIZE
    pub kernarg_size: u32,
    /// Reserved
    pub reserved0: u32,
    /// Kernel code entry byte offset from descriptor
    pub kernel_code_entry_byte_offset: i64,
    /// Reserved
    pub reserved1: [u32; 5],
    /// COMPUTE_PGM_RSRC3 (GFX10+)
    pub compute_pgm_rsrc3: u32,
    /// COMPUTE_PGM_RSRC1
    pub compute_pgm_rsrc1: u32,
    /// COMPUTE_PGM_RSRC2
    pub compute_pgm_rsrc2: u32,
    /// KERNEL_CODE_PROPERTIES
    pub kernel_code_properties: u16,
    /// KERNARG_PRELOAD
    pub kernarg_preload: u16,
    /// Reserved
    pub reserved2: [u32; 1],
}

impl KernelDescriptor {
    /// Create kernel descriptor for the given target.
    pub fn for_target(config: &KernelConfig, code_size: usize, target: Target) -> Self {
        // COMPUTE_PGM_RSRC1 encoding:
        // [5:0]   = GRANULATED_WORKITEM_VGPR_COUNT
        //           Wave32: (VGPRs + 7) / 8 - 1 (Granularity 8)
        //           Wave64: (VGPRs + 3) / 4 - 1 (Granularity 4)
        // [9:6]   = GRANULATED_WAVEFRONT_SGPR_COUNT (must be 0 on GFX10+)
        // [11:10] = PRIORITY
        // [13:12] = FLOAT_MODE (3 = ROUND_NEAREST_EVEN for both FP32 and FP16/64)
        // [21:16] = Various flags
        
        // We use Wave32 (ENABLE_WAVEFRONT_SIZE32 = 1), so granularity is 8
        let vgpr_granularity = 8_u32;
        let vgprs = (config.vgpr_count as u32 + vgpr_granularity - 1) / vgpr_granularity - 1;
        let vgprs = vgprs.min(63); // Saturate to 6-bit max
        
        // SGPR must be 0 on GFX10+
        let sgprs = 0_u32;
        
        // FLOAT_MODE encoding (8 bits, bits [19:12] of RSRC1):
        //   [13:12] = FP32_ROUND:     3 = Round Nearest Even
        //   [15:14] = FP16_64_ROUND:  3 = Round Nearest Even  
        //   [17:16] = FP32_DENORM:    0 = Flush to Zero (fast path)
        //   [19:18] = FP16_64_DENORM: 3 = Preserve (required for BF16)
        // = 0b11_00_11_11 = 0xCF
        // Benchmarked: 0xCF=59.1 TF vs 0xF0(LLVM)=58.2 TF → 0xCF is optimal
        let float_mode = 0xCF_u32;
        // DX10_CLAMP (bit 21): 1 = clamp NaN to zero (standard for compute)
        // bit 20 = PRIV: MUST be 0 (privilege trap handler mode, set by CP only)
        // bit 22 = reserved, must be 0
        // IEEE_MODE (bit 23): 1 = IEEE 754-2008 compliant NaN handling
        let dx10_clamp = 1_u32;
        let ieee_mode = 1_u32;
        
        let rsrc1 = vgprs | (sgprs << 6) | (float_mode << 12) | (dx10_clamp << 21) | (ieee_mode << 23);
        
        // COMPUTE_PGM_RSRC2 encoding:
        // [0]     = ENABLE_PRIVATE_SEGMENT
        // [5:1]   = USER_SGPR (number of user SGPRs) = 2 for kernarg ptr only
        //           修复：之前是 4，导致 s2-s3 是空洞，workgroup_id.x 在 s4
        // [6]     = TRAP_HANDLER
        // [7]     = TGID_X_EN (1 = enable workgroup ID X)
        // [8]     = TGID_Y_EN
        // [9]     = TGID_Z_EN
        // [15:10] = LDS_SIZE - GFX11: MUST BE 0! CP reads from group_segment_fixed_size
        //           ref: LLVM AMDGPU documentation, GFX11 changes
        let enable_private_segment = if config.scratch_size > 0 { 1u32 } else { 0u32 };
        let rsrc2 = enable_private_segment | (2 << 1) | (1 << 7) | (1 << 8) | (1 << 9); // LDS_SIZE = 0 for GFX11
        
        // COMPUTE_PGM_RSRC3 encoding for GFX11:
        // [3:0]   = SHARED_VGPR_COUNT (usually 0)
        // [9:4]   = INST_PREF_SIZE - Instruction Prefetch Size
        //           Formula: min(ceil(code_size / 128), 63)
        //           Unit: 128-byte cache lines
        //           CRITICAL: If 0, prefetch is disabled, causing pipeline starvation!
        // [10]    = TRAP_ON_START
        // [11]    = TRAP_ON_END
        // [31:12] = Reserved (must be 0)
        let inst_pref_lines = ((code_size as u32 + 127) / 128).min(63);
        let rsrc3 = (inst_pref_lines & 0x3F) << 4;
        
        // KERNEL_CODE_PROPERTIES:
        // [0] = ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
        // [1] = ENABLE_SGPR_DISPATCH_PTR
        // [2] = ENABLE_SGPR_QUEUE_PTR
        // [3] = ENABLE_SGPR_KERNARG_SEGMENT_PTR (needed for kernel args)
        // [4] = ENABLE_SGPR_DISPATCH_ID
        // [5] = ENABLE_SGPR_FLAT_SCRATCH_INIT
        // [6] = ENABLE_SGPR_PRIVATE_SEGMENT_SIZE
        // [9:7] = ENABLE_VGPR_WORKITEM_ID (0=none, 1=X, 2=X+Y, 3=X+Y+Z)
        //         CRITICAL for Magic Zero: v0 must contain thread ID!
        // [10] = ENABLE_WAVEFRONT_SIZE32 (GFX10+) - NOTE: bit 10, not 8!
        let enable_workitem_id = 1_u16;  // Enable X only (1 VGPR = v0)
        let code_props: u16 = (1 << 3) | (enable_workitem_id << 7) | (1 << 10); // kernarg ptr + workitem_id + wavefront32
        
        Self {
            group_segment_fixed_size: config.lds_size,
            private_segment_fixed_size: config.scratch_size,
            kernarg_size: config.kernarg_size,
            reserved0: 0,
            kernel_code_entry_byte_offset: 64, // Code starts after descriptor
            reserved1: [0; 5],
            compute_pgm_rsrc3: rsrc3,
            compute_pgm_rsrc1: rsrc1,
            compute_pgm_rsrc2: rsrc2,
            kernel_code_properties: code_props,
            kernarg_preload: 0,
            reserved2: [0; 1],
        }
    }
    
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<Self>()
            )
        }
    }
}

/// ELF64 Header
#[repr(C, packed)]
struct Elf64Header {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

/// ELF64 Section Header
#[repr(C, packed)]
struct Elf64Shdr {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}

/// ELF64 Program Header
#[repr(C, packed)]
struct Elf64Phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

/// ELF64 Symbol Table Entry (24 bytes)
#[repr(C, packed)]
struct Elf64Sym {
    st_name: u32,      // Symbol name (index into string table)
    st_info: u8,       // Type and binding attributes
    st_other: u8,      // Reserved (visibility)
    st_shndx: u16,     // Section header index
    st_value: u64,     // Symbol value (address)
    st_size: u64,      // Size of object
}

impl Elf64Sym {
    /// Create STT_NOTYPE symbol (null entry)
    fn null() -> Self {
        Self {
            st_name: 0, st_info: 0, st_other: 0, st_shndx: 0,
            st_value: 0, st_size: 0,
        }
    }
    
    /// Create kernel descriptor symbol (.kd suffix)
    /// STT_OBJECT (type 1), STB_GLOBAL (bind 1) => info = (1 << 4) | 1 = 0x11
    fn kernel_descriptor(name_offset: u32, section_idx: u16, offset: u64, size: u64) -> Self {
        Self {
            st_name: name_offset,
            st_info: 0x11,  // STB_GLOBAL | STT_OBJECT
            st_other: 0,    // STV_DEFAULT
            st_shndx: section_idx,
            st_value: offset,
            st_size: size,
        }
    }
    
    /// Create kernel entry point symbol
    /// STT_FUNC (type 2), STB_GLOBAL (bind 1) => info = (1 << 4) | 2 = 0x12
    fn kernel_entry(name_offset: u32, section_idx: u16, offset: u64, size: u64) -> Self {
        Self {
            st_name: name_offset,
            st_info: 0x12,  // STB_GLOBAL | STT_FUNC
            st_other: 0,    // STV_DEFAULT
            st_shndx: section_idx,
            st_value: offset,
            st_size: size,
        }
    }
    
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<Self>()
            )
        }
    }
}

/// AMD GPU Code Object builder
pub struct AmdGpuCodeObject {
    /// Kernel configuration
    pub config: KernelConfig,
    /// Assembled kernel code
    code: Vec<u8>,
    /// Constants section
    rodata: Vec<u8>,
    /// Target architecture
    target: Target,
}

impl AmdGpuCodeObject {
    /// Create code object from assembler output.
    ///
    /// SAFETY: Panics if ISA code exceeds 64KB (would corrupt GPU I$ state).
    /// Warns if code exceeds 32KB (GFX1100 SQC L1I cache size).
    pub fn from_assembler(asm: &Rdna3Assembler, config: KernelConfig) -> Self {
        Self::from_assembler_with_target(asm, config, Target::GFX1100)
    }

    /// Create code object with explicit target.
    pub fn from_assembler_with_target(asm: &Rdna3Assembler, config: KernelConfig, target: Target) -> Self {
        let code = asm.as_bytes();
        let code_kb = code.len() / 1024;
        if code.len() > 64 * 1024 {
            panic!(
                "[ISA] FATAL: kernel '{}' code size = {}KB (>64KB limit). \
                 Fully unrolled loops with large epl? Use ISA loop instructions instead.",
                config.name, code_kb
            );
        }
        if code.len() > 32 * 1024 {
            eprintln!(
                "[ISA] WARNING: kernel '{}' code size = {}KB (>32KB L1I cache). \
                 May cause instruction cache thrashing and GPU hangs.",
                config.name, code_kb
            );
        }
        Self {
            config,
            code,
            rodata: Vec::new(),
            target,
        }
    }
    
    /// Add constant data (e.g., log2(e) for exp)
    pub fn add_constant(&mut self, data: &[u8]) {
        self.rodata.extend_from_slice(data);
    }
    
    /// Add log2(e) constant for exp(x) = 2^(x * log2(e))
    pub fn add_log2e_constant(&mut self) {
        let log2e: f32 = std::f32::consts::LOG2_E;
        self.rodata.extend_from_slice(&log2e.to_le_bytes());
    }
    
    /// Get the raw code bytes (for debugging/comparison)
    pub fn get_code_bytes(&self) -> &[u8] {
        &self.code
    }
    
    /// Build the complete code object as bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8192);
        
        // Build kernel descriptor
        let kd = KernelDescriptor::for_target(&self.config, self.code.len(), self.target);
        
        // Sizes
        let elf_header_size = 64;
        let phdr_size = 56;
        let shdr_size = 64;
        let sym_size = 24;  // sizeof(Elf64Sym)
        let num_phdrs = 2;  // PT_LOAD for code, PT_NOTE
        let num_shdrs = 7;  // NULL, .text, .rodata, .note, .symtab, .strtab, .shstrtab
        
        // Build symbol string table (.strtab)
        // Format: \0 + kernel_name + \0 + kernel_name.kd + \0
        let mut strtab = Vec::new();
        strtab.push(0u8);  // Null string at index 0
        let kernel_name_offset = strtab.len() as u32;
        strtab.extend_from_slice(self.config.name.as_bytes());
        strtab.push(0u8);
        let kernel_kd_name_offset = strtab.len() as u32;
        strtab.extend_from_slice(self.config.name.as_bytes());
        strtab.extend_from_slice(b".kd");
        strtab.push(0u8);
        
        // Build symbol table (.symtab)
        // Entry 0: null symbol
        // Entry 1: kernel_name (STT_FUNC, points to entry after descriptor)
        // Entry 2: kernel_name.kd (STT_OBJECT, points to kernel descriptor)
        let sym_null = Elf64Sym::null();
        let text_section_idx: u16 = 1;  // .text is section index 1
        
        // Section string table (.shstrtab)
        let shstrtab = b"\0.text\0.rodata\0.note\0.symtab\0.strtab\0.shstrtab\0";
        let shstrtab_text_idx = 1;
        let shstrtab_rodata_idx = 7;
        let shstrtab_note_idx = 15;
        let shstrtab_symtab_idx = 21;
        let shstrtab_strtab_idx = 29;
        let shstrtab_shstrtab_idx = 37;
        
        // Calculate offsets (layout: header, phdrs, shdrs, data sections)
        let phdrs_offset = elf_header_size;
        let shdrs_offset = phdrs_offset + phdr_size * num_phdrs;
        let data_start = shdrs_offset + shdr_size * num_shdrs;
        
        // Data section offsets
        let shstrtab_offset = data_start;
        let strtab_offset = shstrtab_offset + shstrtab.len();
        let symtab_offset = strtab_offset + strtab.len();
        let symtab_size = 3 * sym_size;  // 3 symbols
        
        // CRITICAL: Kernel entry point must be 256-byte aligned.
        // Entry point = text_offset + 64 (after kernel descriptor)
        // So we need: (text_offset + 64) % 256 == 0
        // => text_offset % 256 == 192
        let raw_text_offset = symtab_offset + symtab_size;
        let text_offset = if (raw_text_offset + 64) % 256 == 0 {
            raw_text_offset
        } else {
            // Pad to make (text_offset + 64) 256-aligned
            let current_entry = raw_text_offset + 64;
            let next_aligned_entry = (current_entry + 255) & !255;
            next_aligned_entry - 64
        };
        let text_padding = text_offset - raw_text_offset;
        
        let text_size = 64 + self.code.len(); // Kernel descriptor + code
        let rodata_offset = text_offset + text_size;
        let note_offset = rodata_offset + self.rodata.len();
        
        // AMDGPU note data - ELF note format:
        // namesz (4) + descsz (4) + type (4) + name (aligned to 4) + desc (aligned to 4)
        let note_name_raw = b"AMDGPU\0";  // 7 bytes including null terminator
        let note_desc = self.build_note_descriptor();
        // Align name and desc to 4 bytes
        let note_name_aligned_len = (note_name_raw.len() + 3) & !3;  // 7 -> 8
        let note_desc_aligned_len = (note_desc.len() + 3) & !3;
        let note_size = 12 + note_name_aligned_len + note_desc_aligned_len;
        
        // Create symbols (now that we know text_offset)
        // Kernel entry point is at text_offset + 64 (after descriptor)
        let sym_entry = Elf64Sym::kernel_entry(
            kernel_name_offset,
            text_section_idx,
            64,  // Offset within .text section (after kernel descriptor)
            self.code.len() as u64
        );
        // Kernel descriptor is at text_offset + 0
        let sym_kd = Elf64Sym::kernel_descriptor(
            kernel_kd_name_offset,
            text_section_idx,
            0,   // Offset within .text section
            64   // Kernel descriptor is 64 bytes
        );
        
        // ELF Header
        let elf_header = Elf64Header {
            e_ident: [
                0x7f, b'E', b'L', b'F',
                2,    // ELFCLASS64
                1,    // ELFDATA2LSB
                1,    // EV_CURRENT
                64,   // ELFOSABI_AMDGPU_HSA = 0x40
                4,    // ELFABIVERSION_AMDGPU_HSA_V5 = 4 (required for modern ROCm)
                0, 0, 0, 0, 0, 0, 0,
            ],
            e_type: 3,      // ET_DYN (shared object) - note: ET_EXEC=2, ET_DYN=3
            e_machine: 224, // EM_AMDGPU
            e_version: 1,
            e_entry: 0,
            e_phoff: phdrs_offset as u64,
            e_shoff: shdrs_offset as u64,
            e_flags: self.target.elf_flags(),
            e_ehsize: elf_header_size as u16,
            e_phentsize: phdr_size as u16,
            e_phnum: num_phdrs as u16,
            e_shentsize: shdr_size as u16,
            e_shnum: num_shdrs as u16,
            e_shstrndx: 6,  // .shstrtab is section 6
        };
        
        // Write ELF header
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&elf_header as *const _ as *const u8, elf_header_size)
        });
        
        // Program headers
        let phdr_load = Elf64Phdr {
            p_type: 1,  // PT_LOAD
            p_flags: 5, // PF_R | PF_X
            p_offset: text_offset as u64,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: text_size as u64,
            p_memsz: text_size as u64,
            p_align: 256,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&phdr_load as *const _ as *const u8, phdr_size)
        });
        
        let phdr_note = Elf64Phdr {
            p_type: 4,  // PT_NOTE
            p_flags: 4, // PF_R
            p_offset: note_offset as u64,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: note_size as u64,
            p_memsz: note_size as u64,
            p_align: 4,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&phdr_note as *const _ as *const u8, phdr_size)
        });
        
        // Section headers (7 sections)
        
        // 0: SHN_UNDEF
        let shdr_null = Elf64Shdr {
            sh_name: 0, sh_type: 0, sh_flags: 0, sh_addr: 0,
            sh_offset: 0, sh_size: 0, sh_link: 0, sh_info: 0,
            sh_addralign: 0, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_null as *const _ as *const u8, shdr_size)
        });
        
        // 1: .text
        let shdr_text = Elf64Shdr {
            sh_name: shstrtab_text_idx as u32,
            sh_type: 1,  // SHT_PROGBITS
            sh_flags: 6, // SHF_ALLOC | SHF_EXECINSTR
            sh_addr: 0,
            sh_offset: text_offset as u64,
            sh_size: text_size as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 256, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_text as *const _ as *const u8, shdr_size)
        });
        
        // 2: .rodata
        let shdr_rodata = Elf64Shdr {
            sh_name: shstrtab_rodata_idx as u32,
            sh_type: 1,  // SHT_PROGBITS
            sh_flags: 2, // SHF_ALLOC
            sh_addr: 0,
            sh_offset: rodata_offset as u64,
            sh_size: self.rodata.len() as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 4, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_rodata as *const _ as *const u8, shdr_size)
        });
        
        // 3: .note
        let shdr_note_sec = Elf64Shdr {
            sh_name: shstrtab_note_idx as u32,
            sh_type: 7,  // SHT_NOTE
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: note_offset as u64,
            sh_size: note_size as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 4, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_note_sec as *const _ as *const u8, shdr_size)
        });
        
        // 4: .symtab
        let shdr_symtab = Elf64Shdr {
            sh_name: shstrtab_symtab_idx as u32,
            sh_type: 2,  // SHT_SYMTAB
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: symtab_offset as u64,
            sh_size: symtab_size as u64,
            sh_link: 5,  // Link to .strtab (section 5)
            sh_info: 1,  // Index of first non-local symbol
            sh_addralign: 8, sh_entsize: sym_size as u64,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_symtab as *const _ as *const u8, shdr_size)
        });
        
        // 5: .strtab
        let shdr_strtab = Elf64Shdr {
            sh_name: shstrtab_strtab_idx as u32,
            sh_type: 3,  // SHT_STRTAB
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: strtab_offset as u64,
            sh_size: strtab.len() as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 1, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_strtab as *const _ as *const u8, shdr_size)
        });
        
        // 6: .shstrtab
        let shdr_shstrtab = Elf64Shdr {
            sh_name: shstrtab_shstrtab_idx as u32,
            sh_type: 3,  // SHT_STRTAB
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: shstrtab_offset as u64,
            sh_size: shstrtab.len() as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 1, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_shstrtab as *const _ as *const u8, shdr_size)
        });
        
        // Data sections (in order)
        
        // .shstrtab data
        buf.extend_from_slice(shstrtab);
        
        // .strtab data
        buf.extend_from_slice(&strtab);
        
        // .symtab data (3 symbols)
        buf.extend_from_slice(sym_null.as_bytes());
        buf.extend_from_slice(sym_entry.as_bytes());
        buf.extend_from_slice(sym_kd.as_bytes());
        
        // Alignment padding before .text (for 256-byte aligned kernel entry)
        for _ in 0..text_padding {
            buf.push(0);
        }
        
        // .text data (kernel descriptor + code)
        buf.extend_from_slice(kd.as_bytes());
        buf.extend_from_slice(&self.code);
        
        // .rodata data
        buf.extend_from_slice(&self.rodata);
        
        // .note data (with proper alignment)
        buf.extend_from_slice(&(note_name_raw.len() as u32).to_le_bytes());  // namesz = 7
        buf.extend_from_slice(&(note_desc.len() as u32).to_le_bytes());      // descsz
        buf.extend_from_slice(&(32u32).to_le_bytes());                       // type = NT_AMDGPU_METADATA
        buf.extend_from_slice(note_name_raw);
        // Pad name to 4-byte alignment
        let name_padding = note_name_aligned_len - note_name_raw.len();
        for _ in 0..name_padding {
            buf.push(0);
        }
        buf.extend_from_slice(&note_desc);
        // Pad desc to 4-byte alignment
        let desc_padding = note_desc_aligned_len - note_desc.len();
        for _ in 0..desc_padding {
            buf.push(0);
        }
        
        buf
    }
    
    /// Build code object with full dynamic linking support (v2)
    /// This includes .dynsym, .dynstr, .hash, .dynamic sections
    /// Required for hipModuleLoadData to work without LLVM
    pub fn to_bytes_v2(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16384);
        
        // Build kernel descriptor
        let kd = KernelDescriptor::for_target(&self.config, self.code.len(), self.target);
        
        // Sizes
        let elf_header_size: usize = 64;
        let phdr_size: usize = 56;
        let shdr_size: usize = 64;
        let sym_size: usize = 24;
        let dyn_entry_size: usize = 16;  // Elf64_Dyn
        
        // Build dynstr (dynamic string table)
        // Format: \0 + kernel_name + \0 + kernel_name.kd + \0
        let mut dynstr = Vec::new();
        dynstr.push(0u8);
        let kernel_name_offset = dynstr.len() as u32;
        dynstr.extend_from_slice(self.config.name.as_bytes());
        dynstr.push(0u8);
        let kernel_kd_name_offset = dynstr.len() as u32;
        dynstr.extend_from_slice(self.config.name.as_bytes());
        dynstr.extend_from_slice(b".kd");
        dynstr.push(0u8);
        
        // Build dynsym (dynamic symbol table) - same as symtab but for dynamic
        // 3 symbols: null, kernel entry, kernel descriptor
        let dynsym_count = 3usize;
        
        // Build hash table (simple sysv hash)
        // nbucket=1, nchain=3 (simplified)
        let hash_nbucket = 1u32;
        let hash_nchain = dynsym_count as u32;
        let hash_size = 8 + hash_nbucket * 4 + hash_nchain * 4;  // header + buckets + chains
        
        // Build .dynamic section entries
        let dynamic_entries = vec![
            (6u64, 0u64),   // DT_SYMTAB - will be patched
            (5u64, 0u64),   // DT_STRTAB - will be patched
            (10u64, dynstr.len() as u64),  // DT_STRSZ
            (11u64, sym_size as u64),      // DT_SYMENT
            (4u64, 0u64),   // DT_HASH - will be patched
            (0u64, 0u64),   // DT_NULL (terminator)
        ];
        let dynamic_size = dynamic_entries.len() * dyn_entry_size;
        
        // Section string table
        let shstrtab = b"\0.note\0.dynsym\0.hash\0.dynstr\0.text\0.rodata\0.dynamic\0.shstrtab\0";
        // Indices: .note=1, .dynsym=7, .hash=15, .dynstr=21, .text=29, .rodata=35, .dynamic=43, .shstrtab=52
        
        // Note section
        let note_name_raw = b"AMDGPU\0";
        let note_desc = self.build_note_descriptor();
        let note_name_aligned = (note_name_raw.len() + 3) & !3;
        let note_desc_aligned = (note_desc.len() + 3) & !3;
        let note_size = 12 + note_name_aligned + note_desc_aligned;
        
        // Calculate layout (like LLVM output):
        // PHDR, then data sections contiguous
        let num_phdrs = 4;  // PHDR, LOAD (readable), LOAD (code), NOTE
        let num_shdrs = 9;  // NULL, .note, .dynsym, .hash, .dynstr, .text, .rodata, .dynamic, .shstrtab
        
        let phdrs_offset = elf_header_size;
        let data_start = phdrs_offset + phdr_size * num_phdrs;
        
        // Align data_start to 4 bytes
        let data_start = (data_start + 3) & !3;
        
        // Data section layout (contiguous for first LOAD segment)
        let note_offset = data_start;
        let dynsym_offset = note_offset + note_size;
        let dynsym_offset = (dynsym_offset + 7) & !7;  // 8-byte align
        let hash_offset = dynsym_offset + dynsym_count * sym_size;
        let hash_offset = (hash_offset + 3) & !3;  // 4-byte align
        let dynstr_offset = hash_offset + hash_size as usize;
        let rodata_offset = dynstr_offset + dynstr.len();
        let rodata_offset = (rodata_offset + 63) & !63;  // 64-byte align for rodata
        
        // Text section needs 256-byte alignment for kernel entry
        let text_base = rodata_offset + self.rodata.len();
        // Kernel entry = text_offset + 64, needs 256-align
        let text_offset = if (text_base + 64) % 256 == 0 {
            text_base
        } else {
            let entry = text_base + 64;
            let aligned_entry = (entry + 255) & !255;
            aligned_entry - 64
        };
        let text_size = 64 + self.code.len();
        
        // Dynamic section after text
        let dynamic_offset = text_offset + text_size;
        let dynamic_offset = (dynamic_offset + 7) & !7;  // 8-byte align
        
        // Section headers at end
        let shdrs_offset = dynamic_offset + dynamic_size;
        let shdrs_offset = (shdrs_offset + 7) & !7;
        
        let shstrtab_offset = shdrs_offset + shdr_size * num_shdrs;
        
        // Virtual addresses for segments
        let _readable_vaddr = 0u64;
        let code_vaddr = 0x1000u64 + text_offset as u64;  // Page-aligned
        let dynamic_vaddr = 0x2000u64 + dynamic_offset as u64;
        
        // Create dynamic symbols
        let text_section_idx = 5u16;  // .text is section 5
        
        // ELF Header
        let elf_header = Elf64Header {
            e_ident: [
                0x7f, b'E', b'L', b'F',
                2,    // ELFCLASS64
                1,    // ELFDATA2LSB
                1,    // EV_CURRENT
                64,   // ELFOSABI_AMDGPU_HSA
                4,    // ELFABIVERSION_AMDGPU_HSA_V5
                0, 0, 0, 0, 0, 0, 0,
            ],
            e_type: 3,      // ET_DYN
            e_machine: 224, // EM_AMDGPU
            e_version: 1,
            e_entry: 0,
            e_phoff: phdrs_offset as u64,
            e_shoff: shdrs_offset as u64,
            e_flags: self.target.elf_flags(),
            e_ehsize: elf_header_size as u16,
            e_phentsize: phdr_size as u16,
            e_phnum: num_phdrs as u16,
            e_shentsize: shdr_size as u16,
            e_shnum: num_shdrs as u16,
            e_shstrndx: 8,  // .shstrtab is section 8
        };
        
        // Write ELF header
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&elf_header as *const _ as *const u8, elf_header_size)
        });
        
        // Program Headers
        // 1. PHDR
        let phdr_phdr = Elf64Phdr {
            p_type: 6,  // PT_PHDR
            p_flags: 4, // PF_R
            p_offset: phdrs_offset as u64,
            p_vaddr: phdrs_offset as u64,
            p_paddr: phdrs_offset as u64,
            p_filesz: (phdr_size * num_phdrs) as u64,
            p_memsz: (phdr_size * num_phdrs) as u64,
            p_align: 8,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&phdr_phdr as *const _ as *const u8, phdr_size)
        });
        
        // 2. LOAD (readable data: note, dynsym, hash, dynstr, rodata)
        let readable_end = rodata_offset + self.rodata.len();
        let phdr_readable = Elf64Phdr {
            p_type: 1,  // PT_LOAD
            p_flags: 4, // PF_R
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: readable_end as u64,
            p_memsz: readable_end as u64,
            p_align: 0x1000,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&phdr_readable as *const _ as *const u8, phdr_size)
        });
        
        // 3. LOAD (code)
        let phdr_code = Elf64Phdr {
            p_type: 1,  // PT_LOAD
            p_flags: 5, // PF_R | PF_X
            p_offset: text_offset as u64,
            p_vaddr: code_vaddr,
            p_paddr: code_vaddr,
            p_filesz: text_size as u64,
            p_memsz: text_size as u64,
            p_align: 0x1000,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&phdr_code as *const _ as *const u8, phdr_size)
        });
        
        // 4. NOTE
        let phdr_note = Elf64Phdr {
            p_type: 4,  // PT_NOTE
            p_flags: 4, // PF_R
            p_offset: note_offset as u64,
            p_vaddr: note_offset as u64,
            p_paddr: note_offset as u64,
            p_filesz: note_size as u64,
            p_memsz: note_size as u64,
            p_align: 4,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&phdr_note as *const _ as *const u8, phdr_size)
        });
        
        // Pad to data_start
        while buf.len() < data_start {
            buf.push(0);
        }
        
        // .note section data
        buf.extend_from_slice(&(note_name_raw.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(note_desc.len() as u32).to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());  // NT_AMDGPU_METADATA
        buf.extend_from_slice(note_name_raw);
        while buf.len() < note_offset + 12 + note_name_aligned {
            buf.push(0);
        }
        buf.extend_from_slice(&note_desc);
        while buf.len() < note_offset + note_size {
            buf.push(0);
        }
        
        // Pad to dynsym
        while buf.len() < dynsym_offset {
            buf.push(0);
        }
        
        // .dynsym section data
        // Symbol 0: null
        buf.extend_from_slice(Elf64Sym::null().as_bytes());
        // Symbol 1: kernel entry
        let sym_entry = Elf64Sym::kernel_entry(
            kernel_name_offset,
            text_section_idx,
            64 + code_vaddr,  // Virtual address of entry
            self.code.len() as u64
        );
        buf.extend_from_slice(sym_entry.as_bytes());
        // Symbol 2: kernel descriptor
        let sym_kd = Elf64Sym::kernel_descriptor(
            kernel_kd_name_offset,
            text_section_idx,
            code_vaddr,  // Virtual address of descriptor
            64
        );
        buf.extend_from_slice(sym_kd.as_bytes());
        
        // Pad to hash
        while buf.len() < hash_offset {
            buf.push(0);
        }
        
        // .hash section data (simple sysv hash)
        buf.extend_from_slice(&hash_nbucket.to_le_bytes());
        buf.extend_from_slice(&hash_nchain.to_le_bytes());
        // Bucket[0] = 1 (first non-null symbol)
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Chain[0] = 0, Chain[1] = 2, Chain[2] = 0
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        
        // Pad to dynstr
        while buf.len() < dynstr_offset {
            buf.push(0);
        }
        
        // .dynstr section data
        buf.extend_from_slice(&dynstr);
        
        // Pad to rodata
        while buf.len() < rodata_offset {
            buf.push(0);
        }
        
        // .rodata section data
        // For AMDGPU, rodata contains kernel descriptor (64 bytes aligned)
        buf.extend_from_slice(&self.rodata);
        
        // Pad to text
        while buf.len() < text_offset {
            buf.push(0);
        }
        
        // .text section data (kernel descriptor + code)
        buf.extend_from_slice(kd.as_bytes());
        buf.extend_from_slice(&self.code);
        
        // Pad to dynamic
        while buf.len() < dynamic_offset {
            buf.push(0);
        }
        
        // .dynamic section data
        for (tag, val) in &dynamic_entries {
            let actual_val = match *tag {
                6 => dynsym_offset as u64,   // DT_SYMTAB
                5 => dynstr_offset as u64,   // DT_STRTAB
                4 => hash_offset as u64,     // DT_HASH
                _ => *val,
            };
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&actual_val.to_le_bytes());
        }
        
        // Pad to section headers
        while buf.len() < shdrs_offset {
            buf.push(0);
        }
        
        // Section Headers (9 sections)
        // 0: NULL
        let shdr_null = Elf64Shdr {
            sh_name: 0, sh_type: 0, sh_flags: 0, sh_addr: 0,
            sh_offset: 0, sh_size: 0, sh_link: 0, sh_info: 0,
            sh_addralign: 0, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_null as *const _ as *const u8, shdr_size)
        });
        
        // 1: .note
        let shdr_note = Elf64Shdr {
            sh_name: 1,  // ".note"
            sh_type: 7,  // SHT_NOTE
            sh_flags: 2, // SHF_ALLOC
            sh_addr: note_offset as u64,
            sh_offset: note_offset as u64,
            sh_size: note_size as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 4, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_note as *const _ as *const u8, shdr_size)
        });
        
        // 2: .dynsym
        let shdr_dynsym = Elf64Shdr {
            sh_name: 7,  // ".dynsym"
            sh_type: 11, // SHT_DYNSYM
            sh_flags: 2, // SHF_ALLOC
            sh_addr: dynsym_offset as u64,
            sh_offset: dynsym_offset as u64,
            sh_size: (dynsym_count * sym_size) as u64,
            sh_link: 4,  // link to .dynstr
            sh_info: 1,  // first non-local symbol
            sh_addralign: 8, sh_entsize: sym_size as u64,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_dynsym as *const _ as *const u8, shdr_size)
        });
        
        // 3: .hash
        let shdr_hash = Elf64Shdr {
            sh_name: 15, // ".hash"
            sh_type: 5,  // SHT_HASH
            sh_flags: 2, // SHF_ALLOC
            sh_addr: hash_offset as u64,
            sh_offset: hash_offset as u64,
            sh_size: hash_size as u64,
            sh_link: 2,  // link to .dynsym
            sh_info: 0,
            sh_addralign: 4, sh_entsize: 4,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_hash as *const _ as *const u8, shdr_size)
        });
        
        // 4: .dynstr
        let shdr_dynstr = Elf64Shdr {
            sh_name: 21, // ".dynstr"
            sh_type: 3,  // SHT_STRTAB
            sh_flags: 2, // SHF_ALLOC
            sh_addr: dynstr_offset as u64,
            sh_offset: dynstr_offset as u64,
            sh_size: dynstr.len() as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 1, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_dynstr as *const _ as *const u8, shdr_size)
        });
        
        // 5: .text
        let shdr_text = Elf64Shdr {
            sh_name: 29, // ".text"
            sh_type: 1,  // SHT_PROGBITS
            sh_flags: 6, // SHF_ALLOC | SHF_EXECINSTR
            sh_addr: code_vaddr,
            sh_offset: text_offset as u64,
            sh_size: text_size as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 256, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_text as *const _ as *const u8, shdr_size)
        });
        
        // 6: .rodata
        let shdr_rodata = Elf64Shdr {
            sh_name: 35, // ".rodata"
            sh_type: 1,  // SHT_PROGBITS
            sh_flags: 2, // SHF_ALLOC
            sh_addr: rodata_offset as u64,
            sh_offset: rodata_offset as u64,
            sh_size: self.rodata.len() as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 64, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_rodata as *const _ as *const u8, shdr_size)
        });
        
        // 7: .dynamic
        let shdr_dynamic = Elf64Shdr {
            sh_name: 43, // ".dynamic"
            sh_type: 6,  // SHT_DYNAMIC
            sh_flags: 3, // SHF_WRITE | SHF_ALLOC
            sh_addr: dynamic_vaddr,
            sh_offset: dynamic_offset as u64,
            sh_size: dynamic_size as u64,
            sh_link: 4, // link to .dynstr
            sh_info: 0,
            sh_addralign: 8, sh_entsize: dyn_entry_size as u64,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_dynamic as *const _ as *const u8, shdr_size)
        });
        
        // 8: .shstrtab
        let shdr_shstrtab = Elf64Shdr {
            sh_name: 52, // ".shstrtab"
            sh_type: 3,  // SHT_STRTAB
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: shstrtab_offset as u64,
            sh_size: shstrtab.len() as u64,
            sh_link: 0, sh_info: 0,
            sh_addralign: 1, sh_entsize: 0,
        };
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&shdr_shstrtab as *const _ as *const u8, shdr_size)
        });
        
        // .shstrtab data
        buf.extend_from_slice(shstrtab);
        
        buf
    }
    
    /// Build AMDGPU note descriptor (MSGPACK format metadata)
    fn build_note_descriptor(&self) -> Vec<u8> {
        // MSGPACK encoding for kernel metadata matching LLVM output
        let mut msg = Vec::new();
        
        // Top-level map with 3 entries (amdhsa.kernels, amdhsa.target, amdhsa.version)
        // CRITICAL: Order matters! LLVM puts kernels FIRST
        msg.extend_from_slice(b"\x83"); // fixmap with 3 entries
        
        // 1. amdhsa.kernels (MUST BE FIRST for ROCm)
        msg.extend_from_slice(b"\xAEamdhsa.kernels"); // key (14 chars)
        msg.extend_from_slice(b"\x91"); // array with 1 element
        
        // Kernel entry
        msg.extend_from_slice(b"\x8A"); // fixmap with 10 entries
        
        // .name
        msg.extend_from_slice(b"\xA5.name");
        let name_bytes = self.config.name.as_bytes();
        if name_bytes.len() < 32 {
            msg.push(0xA0 | name_bytes.len() as u8);
        } else {
            msg.push(0xD9);
            msg.push(name_bytes.len() as u8);
        }
        msg.extend_from_slice(name_bytes);
        
        // .symbol
        msg.extend_from_slice(b"\xA7.symbol");
        let symbol = format!("{}.kd", self.config.name);
        let sym_bytes = symbol.as_bytes();
        if sym_bytes.len() < 32 {
            msg.push(0xA0 | sym_bytes.len() as u8);
        } else {
            msg.push(0xD9);
            msg.push(sym_bytes.len() as u8);
        }
        msg.extend_from_slice(sym_bytes);
        
        // .kernarg_segment_size
        msg.extend_from_slice(b"\xB5.kernarg_segment_size");
        msg.extend_from_slice(&[0xCD]); // uint16
        msg.extend_from_slice(&(self.config.kernarg_size as u16).to_be_bytes());
        
        // .group_segment_fixed_size
        msg.extend_from_slice(b"\xB8.group_segment_fixed_size");
        msg.extend_from_slice(&[0xCE]); // uint32
        msg.extend_from_slice(&self.config.lds_size.to_be_bytes());
        
        // .private_segment_fixed_size
        msg.extend_from_slice(b"\xBA.private_segment_fixed_size");
        msg.push(0x00); // 0
        
        // .wavefront_size
        msg.extend_from_slice(b"\xAF.wavefront_size");
        msg.push(0x20); // 32
        
        // .sgpr_count
        msg.extend_from_slice(b"\xAB.sgpr_count");
        msg.push(self.config.sgpr_count);
        
        // .vgpr_count
        msg.extend_from_slice(b"\xAB.vgpr_count");
        msg.push(self.config.vgpr_count);
        
        // .max_flat_workgroup_size
        msg.extend_from_slice(b"\xB7.max_flat_workgroup_size");
        let wg_size = (self.config.workgroup_size_x as u32)
            * (self.config.workgroup_size_y as u32)
            * (self.config.workgroup_size_z as u32);
        msg.extend_from_slice(&[0xCD]); // uint16
        msg.extend_from_slice(&(wg_size as u16).to_be_bytes());
        
        // .kernarg_segment_align
        msg.extend_from_slice(b"\xB6.kernarg_segment_align");
        msg.push(0x08); // 8 bytes

        // .args
        msg.extend_from_slice(b"\xA5.args");
        msg.push(0x90 | (self.config.kernarg_size / 8) as u8); // array of size N

        let num_args = self.config.kernarg_size / 8;
        for i in 0..num_args {
            msg.extend_from_slice(b"\x84"); // map with 4 entries
            
            // .address_space: global
            msg.extend_from_slice(b"\xAE.address_space");
            msg.extend_from_slice(b"\xA6global"); // "global"

            // .offset: i * 8
            msg.extend_from_slice(b"\xA7.offset");
            msg.extend_from_slice(&[0xCD]); // uint16
            msg.extend_from_slice(&((i * 8) as u16).to_be_bytes());

            // .size: 8
            msg.extend_from_slice(b"\xA5.size");
            msg.push(0x08);

            // .value_kind: global_buffer
            msg.extend_from_slice(b"\xAB.value_kind");
            msg.extend_from_slice(b"\xADglobal_buffer");
        }
        
        // 2. amdhsa.target (required for ROCm)
        msg.extend_from_slice(b"\xADamdhsa.target"); // key (13 chars)
        let isa_triple = self.target.isa_triple();
        let isa_bytes = isa_triple.as_bytes();
        msg.push(0xA0 | isa_bytes.len() as u8);
        msg.extend_from_slice(isa_bytes);
        
        // 3. amdhsa.version
        msg.extend_from_slice(b"\xAEamdhsa.version"); // key (14 chars)
        msg.extend_from_slice(b"\x92\x01\x02"); // [1, 2] array
        
        msg
    }
    
    /// Generate AMDGPU assembly source file compatible with llvm-mc
    /// This is the recommended approach for production use
    pub fn to_assembly(&self) -> String {
        let _wg_size = (self.config.workgroup_size_x as u32)
            * (self.config.workgroup_size_y as u32)
            * (self.config.workgroup_size_z as u32);
        
        let mut asm = String::new();
        
        // Header
        asm.push_str(&format!("    .amdgcn_target \"{}\"\n\n", self.target.isa_triple()));
        
        // Text section with kernel code
        asm.push_str("    .text\n");
        asm.push_str(&format!("    .globl {}\n", self.config.name));
        asm.push_str("    .p2align 8\n");
        asm.push_str(&format!("    .type {},@function\n", self.config.name));
        asm.push_str(&format!("{}:\n", self.config.name));
        
        // Convert our assembled bytes to assembly mnemonics
        // For now, emit raw .long directives for the code
        let mut pc = 0;
        while pc < self.code.len() {
            if pc + 4 <= self.code.len() {
                let dword = u32::from_le_bytes([
                    self.code[pc], self.code[pc+1], 
                    self.code[pc+2], self.code[pc+3]
                ]);
                asm.push_str(&format!("    .long 0x{:08x}\n", dword));
                pc += 4;
            } else {
                // Handle remaining bytes
                for b in &self.code[pc..] {
                    asm.push_str(&format!("    .byte 0x{:02x}\n", b));
                }
                break;
            }
        }
        
        // Kernel descriptor in rodata
        asm.push_str("\n.rodata\n");
        asm.push_str("    .p2align 6\n");
        asm.push_str(&format!("    .amdhsa_kernel {}\n", self.config.name));
        asm.push_str(&format!("        .amdhsa_group_segment_fixed_size {}\n", self.config.lds_size));
        asm.push_str(&format!("        .amdhsa_private_segment_fixed_size {}\n", self.config.scratch_size));
        if self.config.scratch_size > 0 {
            asm.push_str("        .amdhsa_enable_private_segment 1\n");
        }
        asm.push_str(&format!("        .amdhsa_kernarg_size {}\n", self.config.kernarg_size));
        asm.push_str("        .amdhsa_user_sgpr_kernarg_segment_ptr 1\n");
        // Removed: amdhsa_system_vgpr_workitem_id 1 - may affect rsrc1 bits
        asm.push_str(&format!("        .amdhsa_next_free_vgpr {}\n", self.config.vgpr_count.max(4)));
        asm.push_str(&format!("        .amdhsa_next_free_sgpr {}\n", self.config.sgpr_count.max(6)));
        asm.push_str("        .amdhsa_wavefront_size32 1\n");
        // 关键：必须显式启用 workgroup_id SGPRs！
        // 否则 LLVM 可能优化掉它们，导致 wg.y/z 不可用
        asm.push_str("        .amdhsa_system_sgpr_workgroup_id_x 1\n");
        asm.push_str("        .amdhsa_system_sgpr_workgroup_id_y 1\n");
        asm.push_str("        .amdhsa_system_sgpr_workgroup_id_z 1\n");
        asm.push_str(&format!("    .end_amdhsa_kernel\n"));
        
        // Metadata in YAML format (required for ROCm runtime)
        // Critical: must include amdhsa.target and .args for proper kernel dispatch
        asm.push_str("\n    .amdgpu_metadata\n");
        asm.push_str("---\n");
        asm.push_str(&format!("amdhsa.target: {}\n", self.target.isa_triple()));
        asm.push_str("amdhsa.version:\n");
        asm.push_str("  - 1\n");
        asm.push_str("  - 2\n");
        asm.push_str("amdhsa.kernels:\n");
        asm.push_str(&format!("  - .name: {}\n", self.config.name));
        asm.push_str(&format!("    .symbol: {}.kd\n", self.config.name));
        asm.push_str(&format!("    .kernarg_segment_size: {}\n", self.config.kernarg_size));
        asm.push_str(&format!("    .group_segment_fixed_size: {}\n", self.config.lds_size));
        asm.push_str(&format!("    .private_segment_fixed_size: {}\n", self.config.scratch_size));
        asm.push_str("    .kernarg_segment_align: 16\n");  // Increased from 8
        asm.push_str("    .wavefront_size: 32\n");
        asm.push_str(&format!("    .sgpr_count: {}\n", self.config.sgpr_count.max(2)));
        asm.push_str(&format!("    .vgpr_count: {}\n", self.config.vgpr_count.max(4)));
        asm.push_str("    .max_flat_workgroup_size: 1024\n");
        asm.push_str("    .workgroup_processor_mode: 1\n");
        // Each arg: address_space, offset, size, value_kind
        let num_args = self.config.kernarg_size / 8;  // Assume 8-byte pointers
        asm.push_str("    .args:\n");
        for i in 0..num_args {
            asm.push_str("      - .address_space: global\n");
            asm.push_str(&format!("        .offset: {}\n", i * 8));
            asm.push_str("        .size: 8\n");
            asm.push_str("        .value_kind: global_buffer\n");
        }
        asm.push_str("...\n");
        asm.push_str("    .end_amdgpu_metadata\n");
        
        asm
    }
    
    /// Build code object using LLVM toolchain (clang + ld.lld).
    /// This generates a properly formatted code object that the runtime can load.
    pub fn to_code_object_llvm(&self) -> Result<Vec<u8>, String> {
        use std::process::Command;
        use std::fs;
        
        let temp_dir = std::env::temp_dir();
        let asm_path = temp_dir.join(format!("{}.s", self.config.name));
        let obj_path = temp_dir.join(format!("{}.o", self.config.name));
        let co_path = temp_dir.join(format!("{}.co", self.config.name));
        
        // Write assembly file
        let asm_content = self.to_assembly();
        fs::write(&asm_path, &asm_content)
            .map_err(|e| format!("Failed to write assembly file: {}", e))?;

        let clang = find_clang()?;
        let lld = find_ld_lld()?;

        let mc_result = Command::new(&clang)
            .args([
                "-x", "assembler",
                "-target", "amdgcn-amd-amdhsa",
                &format!("-mcpu={}", self.target.mcpu_str()),
                "-c",
                &asm_path.to_string_lossy(),
                "-o",
                &obj_path.to_string_lossy(),
            ])
            .output()
            .map_err(|e| format!("Failed to run clang: {}", e))?;
        
        if !mc_result.status.success() {
            return Err(format!(
                "clang failed: {}",
                String::from_utf8_lossy(&mc_result.stderr)
            ));
        }
        
        let lld_result = Command::new(&lld)
            .args([
                "-flavor", "gnu",
                "-shared",
                &obj_path.to_string_lossy(),
                "-o",
                &co_path.to_string_lossy(),
            ])
            .output()
            .map_err(|e| format!("Failed to run ld.lld: {}", e))?;
        
        if !lld_result.status.success() {
            return Err(format!(
                "ld.lld failed: {}",
                String::from_utf8_lossy(&lld_result.stderr)
            ));
        }
        
        // Read the code object
        let co_bytes = fs::read(&co_path)
            .map_err(|e| format!("Failed to read code object: {}", e))?;
        
        // Cleanup temp files (disabled for debugging)
        // let _ = fs::remove_file(&asm_path);
        // let _ = fs::remove_file(&obj_path);
        // let _ = fs::remove_file(&co_path);
        
        Ok(co_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna3_asm::Rdna3Assembler;
    
    #[test]
    fn test_kernel_descriptor_size() {
        assert_eq!(std::mem::size_of::<KernelDescriptor>(), 64);
    }
    
    #[test]
    fn test_simple_code_object() {
        let mut asm = Rdna3Assembler::new();
        asm.endpgm();
        
        let config = KernelConfig::default();
        let co = AmdGpuCodeObject::from_assembler(&asm, config);
        let bytes = co.to_bytes();
        
        // Verify ELF magic
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);
        
        println!("Code object size: {} bytes", bytes.len());
    }
}
