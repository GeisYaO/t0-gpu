//! T0 Machine-Level SSA IR
//!
//! Provides SSA-form wrapper around the existing `Vec<Op>` machine IR.
//! Explicit modeling of VCC/SCC/EXEC implicit state registers enables
//! safe instruction scheduling and future register spilling.
//!
//! # Architecture
//!
//! ```text
//! Vec<Op>  ──lift_to_ssa()──→  MachFunc (SSA)
//!                                  │
//!                            [optimizations]
//!                                  │
//!                          lower_from_ssa()──→  Vec<Op>  → regalloc → asm
//! ```
//!
//! # Design decisions
//!
//! - `MachInst.op` retains the original `Op` for backend compatibility
//! - VCC/SCC are modeled as `implicit_defs`/`implicit_uses` on each inst
//! - Phi nodes use block-param style (entries from predecessor blocks)

use std::collections::{HashMap, HashSet};
use super::ir::*;

// ============================================================================
// SSA Value Handle
// ============================================================================

/// Machine SSA value — unique per definition.
/// Conceptually equivalent to `tile_ssa::Value` but at machine-instruction level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MVal(pub u32);

impl std::fmt::Display for MVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "m{}", self.0)
    }
}

// ============================================================================
// Implicit State Registers
// ============================================================================

/// Hardware implicit state register.
///
/// GFX1100 has two implicit condition codes:
/// - **VCC**: written by v_cmp_*, v_add_co_*; read by v_cndmask, branches
/// - **SCC**: written by s_add/s_sub/s_cmp; read by s_cbranch_scc*, s_addc
/// - **EXEC**: modified by SaveExec/RestoreExec
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImplicitReg {
    Vcc,
    Scc,
    Exec,
}

// ============================================================================
// Machine SSA Instruction
// ============================================================================

/// A machine instruction in SSA form.
///
/// Wraps the original `Op` with explicit SSA def/use information
/// and explicit implicit-state tracking.
#[derive(Clone, Debug)]
pub struct MachInst {
    /// Original machine Op (preserved for backend emission)
    pub op: Op,
    /// SSA values defined by this instruction
    pub defs: Vec<MVal>,
    /// SSA values used by this instruction
    pub uses: Vec<MVal>,
    /// Implicit state registers written (VCC, SCC, EXEC)
    pub implicit_defs: Vec<ImplicitReg>,
    /// Implicit state registers read
    pub implicit_uses: Vec<ImplicitReg>,
    /// Wide-op coalesced group: instructions with the same non-None group ID
    /// must keep their defs physically contiguous. Opt passes (CopyProp, DCE)
    /// must not break the contiguity of these instructions.
    /// Used for WMMA fragments (8 consecutive VGPRs), 128-bit loads, etc.
    pub coalesced_group: Option<u32>,
}

// ============================================================================
// Phi Node
// ============================================================================

/// Phi node at block entry — merges values from predecessor blocks.
#[derive(Clone, Debug)]
pub struct PhiNode {
    /// SSA value defined by this phi
    pub dst: MVal,
    /// (predecessor block ID, incoming value)
    pub entries: Vec<(u32, MVal)>,
}

// ============================================================================
// Machine Basic Block
// ============================================================================

/// A basic block in the machine SSA CFG.
#[derive(Clone, Debug)]
pub struct MachBlock {
    /// Block ID (0-indexed)
    pub id: u32,
    /// Instruction indices into `MachFunc.insts`
    pub insts: Vec<usize>,
    /// Predecessor block IDs
    pub preds: Vec<u32>,
    /// Successor block IDs
    pub succs: Vec<u32>,
    /// Phi nodes at block entry
    pub phis: Vec<PhiNode>,
    /// Label name (if this block starts with a Label op)
    pub label: Option<String>,
}

// ============================================================================
// Machine SSA Function
// ============================================================================

/// A complete machine function in SSA form.
#[derive(Clone, Debug)]
pub struct MachFunc {
    /// All basic blocks (block 0 = entry)
    pub blocks: Vec<MachBlock>,
    /// All instructions (referenced by block.insts indices)
    pub insts: Vec<MachInst>,
    /// Next available MVal ID
    pub val_count: u32,
}

impl MachFunc {
    /// Allocate a fresh SSA value.
    fn alloc_val(&mut self) -> MVal {
        let v = MVal(self.val_count);
        self.val_count += 1;
        v
    }

    /// Get the number of basic blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the number of instructions.
    pub fn num_insts(&self) -> usize {
        self.insts.len()
    }

    /// Pretty-print the MachFunc for debugging.
    pub fn dump(&self) -> String {
        let mut s = String::new();
        for blk in &self.blocks {
            if let Some(lbl) = &blk.label {
                s += &format!("BB{} (label: {}):\n", blk.id, lbl);
            } else {
                s += &format!("BB{}:\n", blk.id);
            }
            s += &format!("  preds: {:?}, succs: {:?}\n", blk.preds, blk.succs);
            for phi in &blk.phis {
                s += &format!("  PHI {} = {:?}\n", phi.dst, phi.entries);
            }
            for &idx in &blk.insts {
                let inst = &self.insts[idx];
                s += &format!("  [{}] defs={:?} uses={:?}", idx, inst.defs, inst.uses);
                if !inst.implicit_defs.is_empty() {
                    s += &format!(" imp_def={:?}", inst.implicit_defs);
                }
                if !inst.implicit_uses.is_empty() {
                    s += &format!(" imp_use={:?}", inst.implicit_uses);
                }
                s += &format!("  {:?}\n", inst.op);
            }
        }
        s
    }
}

// ============================================================================
// Lift: Vec<Op> → MachFunc
// ============================================================================

/// Lift a linear `Vec<Op>` into SSA form `MachFunc`.
///
/// Steps:
/// 1. Split into basic blocks at Label/Branch/Endpgm boundaries
/// 2. Build CFG (pred/succ edges from branch targets)
/// 3. Assign SSA values: each VReg def gets a fresh MVal
/// 4. Resolve uses: map VReg uses to the dominating MVal definition
/// 5. Annotate implicit state (VCC/SCC) from Op metadata
/// 6. Insert Phi nodes for VRegs defined in multiple predecessors
pub fn lift_to_ssa(ops: &[Op]) -> MachFunc {
    if ops.is_empty() {
        return MachFunc { blocks: vec![], insts: vec![], val_count: 0 };
    }

    // ── Step 1: Split into basic blocks ──
    // A new block starts:
    //   - at the beginning
    //   - after any branch/endpgm
    //   - at any Label
    let mut block_starts: Vec<usize> = vec![0]; // first op always starts a block
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::Label(_) => {
                if i > 0 && !block_starts.contains(&i) {
                    block_starts.push(i);
                }
            }
            Op::Branch(_) | Op::BranchScc1(_) | Op::BranchScc0(_) |
            Op::BranchVccz(_) | Op::Endpgm => {
                if i + 1 < ops.len() && !block_starts.contains(&(i + 1)) {
                    block_starts.push(i + 1);
                }
            }
            _ => {}
        }
    }
    block_starts.sort();
    block_starts.dedup();

    // Build block ranges: [start, end) for each block
    let mut block_ranges: Vec<(usize, usize)> = Vec::new();
    for (bi, &start) in block_starts.iter().enumerate() {
        let end = if bi + 1 < block_starts.len() {
            block_starts[bi + 1]
        } else {
            ops.len()
        };
        block_ranges.push((start, end));
    }

    // Map label name → block ID
    let mut label_to_block: HashMap<String, u32> = HashMap::new();
    for (bi, &(start, _end)) in block_ranges.iter().enumerate() {
        if let Op::Label(name) = &ops[start] {
            label_to_block.insert(name.clone(), bi as u32);
        }
    }

    // ── Step 2: Build CFG ──
    let num_blocks = block_ranges.len();
    let mut blocks: Vec<MachBlock> = Vec::with_capacity(num_blocks);

    for (bi, &(start, end)) in block_ranges.iter().enumerate() {
        let label = if let Op::Label(name) = &ops[start] {
            Some(name.clone())
        } else {
            None
        };
        blocks.push(MachBlock {
            id: bi as u32,
            insts: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
            phis: Vec::new(),
            label,
        });

        // Determine successors from the last instruction in this block
        if end > start {
            let last_op = &ops[end - 1];
            match last_op {
                Op::Branch(target) => {
                    if let Some(&target_bid) = label_to_block.get(target) {
                        blocks[bi].succs.push(target_bid);
                    }
                }
                Op::BranchScc1(target) | Op::BranchScc0(target) => {
                    // Conditional: fall-through + target
                    if let Some(&target_bid) = label_to_block.get(target) {
                        blocks[bi].succs.push(target_bid);
                    }
                    // Fall-through to next block
                    if bi + 1 < num_blocks {
                        blocks[bi].succs.push((bi + 1) as u32);
                    }
                }
                Op::BranchVccz(target) => {
                    if let Some(&target_bid) = label_to_block.get(target) {
                        blocks[bi].succs.push(target_bid);
                    }
                    if bi + 1 < num_blocks {
                        blocks[bi].succs.push((bi + 1) as u32);
                    }
                }
                Op::Endpgm => {
                    // No successors
                }
                _ => {
                    // Fall-through to next block
                    if bi + 1 < num_blocks {
                        blocks[bi].succs.push((bi + 1) as u32);
                    }
                }
            }
        }
    }

    // Build predecessors from successors
    // (must do this after all blocks have succs populated)
    let succ_pairs: Vec<(u32, u32)> = blocks.iter()
        .flat_map(|b| b.succs.iter().map(move |&s| (b.id, s)))
        .collect();
    for (pred, succ) in succ_pairs {
        blocks[succ as usize].preds.push(pred);
    }

    // Dedup pred/succ lists
    for blk in &mut blocks {
        blk.preds.sort();
        blk.preds.dedup();
        blk.succs.sort();
        blk.succs.dedup();
    }

    // ── Step 3 & 4: SSA value assignment ──
    let mut func = MachFunc {
        blocks,
        insts: Vec::with_capacity(ops.len()),
        val_count: 0,
    };

    // Map: VReg → MVal for the latest definition
    let mut vreg_to_mval: HashMap<VReg, MVal> = HashMap::new();
    // Map: SReg → MVal for the latest definition (CRITICAL for LICM correctness)
    // Without this, SReg data dependencies are invisible to SSA analysis, and LICM
    // incorrectly hoists instructions that depend on loop induction variables.
    let mut sreg_to_mval: HashMap<SReg, MVal> = HashMap::new();

    for (bi, &(start, end)) in block_ranges.iter().enumerate() {
        for i in start..end {
            let op = &ops[i];

            // Get defs and uses from the Op (both VReg and SReg)
            let vreg_defs = op.vreg_defs();
            let vreg_uses: Vec<VReg> = op.vreg_uses();
            let sreg_defs_list = op.sreg_defs();
            let sreg_uses_list = op.sreg_uses();

            // Resolve VReg uses → MVal
            let mut use_mvals: Vec<MVal> = vreg_uses.iter()
                .map(|v| {
                    *vreg_to_mval.get(v).unwrap_or(&MVal(u32::MAX))
                })
                .collect();

            // Resolve SReg uses → MVal (append to use_mvals)
            for s in &sreg_uses_list {
                if let Some(&mv) = sreg_to_mval.get(s) {
                    use_mvals.push(mv);
                }
                // If SReg not in map, it's a kernarg/hw reg — no MVal dependency needed
            }

            // Allocate fresh MVal for each VReg def
            let mut def_mvals: Vec<MVal> = vreg_defs.iter()
                .map(|v| {
                    let mv = func.alloc_val();
                    vreg_to_mval.insert(*v, mv);
                    mv
                })
                .collect();

            // Allocate fresh MVal for each SReg def (append to def_mvals)
            for s in &sreg_defs_list {
                let mv = func.alloc_val();
                sreg_to_mval.insert(*s, mv);
                def_mvals.push(mv);
            }

            // Implicit state
            let mut implicit_defs = Vec::new();
            let mut implicit_uses = Vec::new();

            if op.writes_vcc() { implicit_defs.push(ImplicitReg::Vcc); }
            if op.reads_vcc()  { implicit_uses.push(ImplicitReg::Vcc); }
            if op.writes_scc() { implicit_defs.push(ImplicitReg::Scc); }
            if op.reads_scc()  { implicit_uses.push(ImplicitReg::Scc); }

            // EXEC mask modifications
            match op {
                Op::SaveExec { .. }    => {
                    implicit_defs.push(ImplicitReg::Exec);
                    implicit_uses.push(ImplicitReg::Exec);
                }
                Op::RestoreExec { .. } | Op::XorExec { .. } => {
                    implicit_defs.push(ImplicitReg::Exec);
                }
                _ => {}
            }

            let inst_idx = func.insts.len();
            func.insts.push(MachInst {
                op: op.clone(),
                defs: def_mvals,
                uses: use_mvals,
                implicit_defs,
                implicit_uses,
                coalesced_group: None,
            });
            func.blocks[bi].insts.push(inst_idx);
        }
    }

    func
}

// ============================================================================
// Coalesced Group Annotation
// ============================================================================

/// Annotate MachInsts whose VRegs belong to coalesced groups.
///
/// After `lift_to_ssa`, scans all instructions and marks those that define
/// or use VRegs within a CoalescedGroup's range. This prevents CopyProp
/// and DCE from breaking the physical contiguity required by WMMA, etc.
///
/// Call this immediately after `lift_to_ssa` and before running opt passes.
pub fn annotate_coalesced_groups(
    func: &mut MachFunc,
    groups: &[super::compile::CoalescedGroup],
) {
    if groups.is_empty() { return; }

    // Build a VReg → group_id lookup
    let mut vreg_to_group: HashMap<VReg, u32> = HashMap::new();
    for g in groups {
        for i in 0..g.count {
            vreg_to_group.insert(VReg(g.base_vreg.0 + i), g.id);
        }
    }

    // Scan all instructions: if any def or use VReg is in a group, mark the inst
    for blk in &func.blocks {
        for &idx in &blk.insts {
            let inst = &func.insts[idx];
            let mut group_id: Option<u32> = None;

            // Check VReg defs
            for vr in inst.op.vreg_defs() {
                if let Some(&gid) = vreg_to_group.get(&vr) {
                    group_id = Some(gid);
                    break;
                }
            }
            // Check VReg uses if no def match
            if group_id.is_none() {
                for vr in inst.op.vreg_uses() {
                    if let Some(&gid) = vreg_to_group.get(&vr) {
                        group_id = Some(gid);
                        break;
                    }
                }
            }

            if let Some(gid) = group_id {
                func.insts[idx].coalesced_group = Some(gid);
            }
        }
    }
}

// ============================================================================
// MachSSA Instruction Scheduler (D3)
// ============================================================================

/// Check if two MachInsts have an implicit-state conflict.
///
/// Returns true if inst_a writes an implicit register that inst_b reads (or vice versa),
/// or both write the same implicit register. This is **more precise** than the old
/// `touches_implicit_state()` which conservatively blocked all reordering.
fn implicit_conflict(a: &MachInst, b: &MachInst) -> bool {
    // a.def conflicts with b.use or b.def
    for d in &a.implicit_defs {
        if b.implicit_uses.contains(d) || b.implicit_defs.contains(d) {
            return true;
        }
    }
    // b.def conflicts with a.use
    for d in &b.implicit_defs {
        if a.implicit_uses.contains(d) {
            return true;
        }
    }
    false
}

/// Check if two MachInsts have an SSA value conflict (data dependency).
fn ssa_conflict(a: &MachInst, b: &MachInst) -> bool {
    // a defines something b uses
    for d in &a.defs {
        if b.uses.contains(d) { return true; }
    }
    // b defines something a uses
    for d in &b.defs {
        if a.uses.contains(d) { return true; }
    }
    // both define the same value (WAW)
    for d in &a.defs {
        if b.defs.contains(d) { return true; }
    }
    false
}

/// Are two MachInsts fully independent? (No data or implicit-state conflict)
fn are_independent(a: &MachInst, b: &MachInst) -> bool {
    !ssa_conflict(a, b) && !implicit_conflict(a, b)
}

/// Schedule instructions within a MachFunc for latency hiding and pressure reduction.
///
/// Phase 1: Within each basic block, move independent ALU ops between memory loads
///          and their corresponding waitcnt to overlap load latency.
///
/// Phase 2: When live MVal count > 96 (VGPR pressure threshold), swap adjacent
///          independent instructions to prefer those consuming dying values.
///
/// Returns the number of instructions reordered.
pub fn schedule_mach_func(func: &mut MachFunc) -> usize {
    use super::latency_model;
    let mut total_reordered = 0;
    let mut total_movable_cycles: u32 = 0;
    let mut total_load_latency: u32 = 0;
    let mut total_loads: usize = 0;

    for blk in &mut func.blocks {
        let block_insts = std::mem::take(&mut blk.insts);
        let len = block_insts.len();
        if len < 3 {
            blk.insts = block_insts;
            continue;
        }

        // ── Phase 1: Latency-hiding reorder ──
        let mut scheduled: Vec<usize> = Vec::with_capacity(len);
        let mut i = 0;

        while i < len {
            let idx = block_insts[i];
            let inst = &func.insts[idx];

            // Detect memory loads
            let is_vmem = matches!(&inst.op, Op::GlobalLoad { .. } | Op::BufferLoad { .. });
            let is_lds = matches!(&inst.op,
                Op::LdsLoad { .. } | Op::DsLoadB32 { .. } | Op::DsLoadB64 { .. } |
                Op::DsLoadB128 { .. } | Op::DsLoadU16 { .. } |
                Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. }
            );

            if is_vmem || is_lds {
                let wait_match = if is_vmem {
                    |op: &Op| matches!(op, Op::WaitVmcnt(_))
                } else {
                    |op: &Op| matches!(op, Op::WaitLgkmcnt(_))
                };

                // Collect consecutive loads of the same type
                let mut load_indices: Vec<usize> = Vec::new();
                let mut load_defs: Vec<MVal> = Vec::new();

                while i < len {
                    let ci = block_insts[i];
                    let cinst = &func.insts[ci];
                    let cur_vmem = matches!(&cinst.op, Op::GlobalLoad { .. } | Op::BufferLoad { .. });
                    let cur_lds = matches!(&cinst.op,
                        Op::LdsLoad { .. } | Op::DsLoadB32 { .. } | Op::DsLoadB64 { .. } |
                        Op::DsLoadB128 { .. } | Op::DsLoadU16 { .. } |
                        Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. }
                    );
                    if (is_vmem && cur_vmem) || (is_lds && cur_lds) {
                        load_defs.extend_from_slice(&cinst.defs);
                        load_indices.push(ci);
                        i += 1;
                    } else {
                        break;
                    }
                }

                // Track load latency budget for diagnostic
                for &li in &load_indices {
                    let lat = latency_model::op_latency(&func.insts[li].op).latency;
                    total_load_latency += lat;
                    total_loads += 1;
                    scheduled.push(li);
                }

                // Collect until wait: split into movable vs non-movable.
                // CRITICAL: an op is only movable if it has no dependency on
                // any PRECEDING non-movable op. Otherwise reordering breaks
                // data flow (e.g. v_add_co_u32 writes VCC, v_add_co_ci_u32
                // reads VCC — moving the reader before a non-movable writer
                // produces wrong results).
                let mut movable: Vec<usize> = Vec::new();
                let mut non_movable: Vec<usize> = Vec::new();
                // Track accumulated defs from non-movable ops to catch
                // transitive dependencies.
                let mut nm_defs: Vec<MVal> = Vec::new();
                let mut nm_implicit_defs: Vec<ImplicitReg> = Vec::new();

                while i < len {
                    let ci = block_insts[i];
                    let cinst = &func.insts[ci];

                    if wait_match(&cinst.op) { break; }

                    // Stop at control flow / sync boundaries
                    let is_boundary = matches!(&cinst.op,
                        Op::Barrier | Op::SBarrier |
                        Op::Label(_) |
                        Op::Branch(_) | Op::BranchScc0(_) | Op::BranchScc1(_) |
                        Op::BranchVccz(_)
                    );
                    if is_boundary { break; }

                    // Check: does this inst depend on the loaded values?
                    let depends_on_load = cinst.uses.iter().any(|u| load_defs.contains(u));

                    // Can overlap with memory?
                    let can_overlap = if is_vmem {
                        latency_model::can_overlap_vmem(&cinst.op)
                    } else {
                        latency_model::can_overlap_lds(&cinst.op)
                    };

                    // Use precise implicit-state conflict check against all loads
                    let implicit_ok = load_indices.iter().all(|&li| {
                        !implicit_conflict(&func.insts[li], cinst)
                    });

                    // Check dependency on accumulated non-movable ops
                    let depends_on_nm = cinst.uses.iter().any(|u| nm_defs.contains(u))
                        || cinst.implicit_uses.iter().any(|u| nm_implicit_defs.contains(u));

                    if can_overlap && !depends_on_load && !depends_on_nm
                        && !cinst.op.has_side_effects() && implicit_ok
                    {
                        movable.push(ci);
                        total_reordered += 1;
                    } else {
                        // Track this op's defs so future ops can check dependency
                        nm_defs.extend_from_slice(&cinst.defs);
                        nm_implicit_defs.extend_from_slice(&cinst.implicit_defs);
                        non_movable.push(ci);
                    }
                    i += 1;
                }

                // Sort movable ops by throughput descending: expensive compute
                // first to maximally fill the memory latency window.
                movable.sort_by(|&a, &b| {
                    let la = latency_model::op_latency(&func.insts[a].op).throughput;
                    let lb = latency_model::op_latency(&func.insts[b].op).throughput;
                    lb.cmp(&la)
                });

                // Track movable cycles for utilization diagnostic
                for &idx in &movable {
                    total_movable_cycles += latency_model::op_latency(&func.insts[idx].op).throughput;
                }

                // Emit: movable first (sorted), then non-movable, then wait
                for idx in movable { scheduled.push(idx); }
                for idx in non_movable { scheduled.push(idx); }

                if i < len && wait_match(&func.insts[block_insts[i]].op) {
                    scheduled.push(block_insts[i]);
                    i += 1;
                }
            } else {
                scheduled.push(idx);
                i += 1;
            }
        }

        // ── Phase 2: Pressure-aware local swaps ──
        // Compute last-use of each MVal in this block
        let slen = scheduled.len();
        let mut last_use: HashMap<MVal, usize> = HashMap::new();
        for (pos, &idx) in scheduled.iter().enumerate() {
            for u in &func.insts[idx].uses {
                last_use.insert(*u, pos);
            }
        }

        // Count live MVals at each position
        let mut live_count: Vec<usize> = vec![0; slen];
        let mut live: std::collections::HashSet<MVal> = std::collections::HashSet::new();
        for (pos, &idx) in scheduled.iter().enumerate() {
            for d in &func.insts[idx].defs {
                live.insert(*d);
            }
            live_count[pos] = live.len();
            live.retain(|v| last_use.get(v).map_or(true, |&lu| lu > pos));
        }

        // Swap adjacent independent ops in high-pressure regions
        for pos in 0..slen.saturating_sub(1) {
            if live_count[pos] <= 96 { continue; }

            let idx_a = scheduled[pos];
            let idx_b = scheduled[pos + 1];
            let inst_a = &func.insts[idx_a];
            let inst_b = &func.insts[idx_b];

            if inst_a.op.has_side_effects() || inst_b.op.has_side_effects() { continue; }
            if inst_a.defs.is_empty() || inst_b.defs.is_empty() { continue; }
            if !are_independent(inst_a, inst_b) { continue; }

            // Prefer the inst that consumes more dying values
            let dying_a = inst_a.uses.iter().filter(|v| {
                last_use.get(v).map_or(false, |&lu| lu == pos)
            }).count();
            let dying_b = inst_b.uses.iter().filter(|v| {
                last_use.get(v).map_or(false, |&lu| lu == pos + 1)
            }).count();

            // Tie-break by throughput: prefer scheduling higher-latency ops earlier
            let tp_a = latency_model::op_latency(&inst_a.op).throughput;
            let tp_b = latency_model::op_latency(&inst_b.op).throughput;
            let should_swap = dying_b > dying_a
                || (dying_b == dying_a && tp_b > tp_a);

            if should_swap {
                scheduled.swap(pos, pos + 1);
                // Update last_use for swapped
                for u in &func.insts[scheduled[pos]].uses {
                    last_use.entry(*u).and_modify(|lu| { if *lu == pos + 1 { *lu = pos; } });
                }
                for u in &func.insts[scheduled[pos + 1]].uses {
                    last_use.entry(*u).and_modify(|lu| { if *lu == pos { *lu = pos + 1; } });
                }
                total_reordered += 1;
            }
        }

        blk.insts = scheduled;
    }

    // Diagnostic output when T0_DUMP_ASM=1
    if total_loads > 0 {
        let utilization = if total_load_latency > 0 {
            total_movable_cycles as f32 / total_load_latency as f32
        } else {
            0.0
        };
        if std::env::var("T0_DUMP_ASM").is_ok() {
            eprintln!("  [schedule] reordered={} loads={} movable_cy={} load_lat={} util={:.0}%",
                total_reordered, total_loads, total_movable_cycles, total_load_latency,
                utilization * 100.0);
        }
    }

    total_reordered
}

// ============================================================================
// MachSSA Dead Code Elimination (D4a)
// ============================================================================

/// SSA-based Dead Code Elimination.
///
/// An instruction is dead if:
/// - It has no side effects (`!op.has_side_effects()`)
/// - None of its defined MVals are used by any other instruction
/// - It doesn't define implicit state (VCC/SCC/EXEC)
///   unless that implicit state is also unused
///
/// Complexity: O(n) via use-count — vs old DCE which was O(n²).
/// Returns the number of instructions removed.
pub fn dce_mach_func(func: &mut MachFunc) -> usize {
    // ── Worklist-based DCE with transitive liveness propagation ──
    //
    // Algorithm:
    //   1. Build MVal → defining instruction index map
    //   2. Mark "root live" instructions: side-effect ops, control flow,
    //      implicit_defs, no-def ops
    //   3. Worklist: for each live instruction, mark all MVal uses as needed,
    //      then mark their defining instructions as live too
    //   4. Remove all non-live instructions

    // Step 1: MVal → defining inst index
    let mut mval_def_inst: HashMap<MVal, usize> = HashMap::new();
    let all_inst_indices: Vec<usize> = func.blocks.iter()
        .flat_map(|blk| blk.insts.iter().copied())
        .collect();

    for &idx in &all_inst_indices {
        for &d in &func.insts[idx].defs {
            mval_def_inst.insert(d, idx);
        }
    }

    // Step 2: Mark root-live instructions
    let mut live: HashSet<usize> = HashSet::new();
    let mut worklist: Vec<usize> = Vec::new();

    for &idx in &all_inst_indices {
        let inst = &func.insts[idx];
        let is_root = inst.op.has_side_effects()
            || matches!(inst.op,
                Op::Branch(_) | Op::BranchScc0(_) | Op::BranchScc1(_) |
                Op::BranchVccz(_) | Op::Endpgm | Op::Label(_) |
                Op::Barrier | Op::SBarrier)
            || !inst.implicit_defs.is_empty()
            || inst.defs.is_empty() // no defs = keep (stores, etc.)
            // Coalesced group members (WMMA fragments, etc.) must be kept alive
            // to preserve physical contiguity of their VReg allocations.
            || inst.coalesced_group.is_some();

        if is_root {
            if live.insert(idx) {
                worklist.push(idx);
            }
        }
    }

    // Step 2b: Loop-carried liveness — mark VReg defs inside loops as live
    //
    // T0's SSA is single-pass linear (no phi nodes). Instructions at the
    // end of a loop body that update a register (e.g. `v153 += 32` for
    // K-loop increment) create a new MVal that the loop header instruction
    // uses via the backedge. But the header's MVal reference points to
    // the prologue definition, NOT the loop body's new MVal. This makes
    // the loop body's definition appear dead → DCE removes it → broken.
    //
    // Fix: For each loop (backward branch → label), collect all VRegs
    // used inside the loop. Any instruction inside the loop that DEFINES
    // one of those VRegs is loop-carried and must be kept alive.
    {
        // Linearize: build position map for inst indices
        let mut idx_to_pos: HashMap<usize, usize> = HashMap::new();
        for (pos, &idx) in all_inst_indices.iter().enumerate() {
            idx_to_pos.insert(idx, pos);
        }

        // Find label positions
        let mut label_pos: HashMap<String, usize> = HashMap::new();
        for (pos, &idx) in all_inst_indices.iter().enumerate() {
            if let Op::Label(name) = &func.insts[idx].op {
                label_pos.insert(name.clone(), pos);
            }
        }

        // Find loop ranges (backward branches)
        let mut loop_ranges: Vec<(usize, usize)> = Vec::new(); // (start_pos, end_pos)
        for (pos, &idx) in all_inst_indices.iter().enumerate() {
            let target = match &func.insts[idx].op {
                Op::BranchScc1(t) | Op::BranchScc0(t) | Op::Branch(t) |
                Op::BranchVccz(t) => Some(t.as_str()),
                _ => None,
            };
            if let Some(t) = target {
                if let Some(&lp) = label_pos.get(t) {
                    if lp < pos {
                        // Backward branch = loop: lp..pos
                        loop_ranges.push((lp, pos));
                    }
                }
            }
        }

        // For each loop, collect used VRegs and mark defs as root-live
        for &(loop_start, loop_end) in &loop_ranges {
            // Collect all VRegs USED inside the loop
            let mut loop_used_vregs: HashSet<VReg> = HashSet::new();
            for pos in loop_start..=loop_end {
                let idx = all_inst_indices[pos];
                for vr in func.insts[idx].op.vreg_uses() {
                    loop_used_vregs.insert(vr);
                }
            }

            // Mark any instruction inside the loop that DEFINES a loop-used VReg
            for pos in loop_start..=loop_end {
                let idx = all_inst_indices[pos];
                let inst = &func.insts[idx];
                // Skip instructions already marked live
                if live.contains(&idx) { continue; }
                // Check if any def VReg is used inside the loop
                for vr in inst.op.vreg_defs() {
                    if loop_used_vregs.contains(&vr) {
                        if live.insert(idx) {
                            worklist.push(idx);
                        }
                        break;
                    }
                }
            }
        }
    }

    // Step 3: Propagate liveness backward through use→def chains
    while let Some(idx) = worklist.pop() {
        let uses = func.insts[idx].uses.clone();
        for u in &uses {
            if let Some(&def_idx) = mval_def_inst.get(u) {
                if live.insert(def_idx) {
                    worklist.push(def_idx);
                }
            }
        }
    }

    // Step 4: Remove non-live instructions from blocks
    let mut total_removed = 0;
    for blk in &mut func.blocks {
        let original_len = blk.insts.len();
        blk.insts.retain(|&idx| live.contains(&idx));
        total_removed += original_len - blk.insts.len();
    }

    total_removed
}

// ============================================================================
// MachSSA Copy Propagation (D4b)
// ============================================================================

/// SSA-based Copy Propagation.
///
/// For `VMov { dst, src: Operand::VReg(src_vreg) }`:
/// Replace all uses of `dst`'s MVal with `src_vreg`'s MVal throughout the function.
/// The VMov itself becomes dead and will be cleaned up by DCE.
///
/// Returns the number of copies propagated.
pub fn copy_propagate_mach_func(func: &mut MachFunc) -> usize {
    // Step 1: Find VMov copies: MVal(dst) → MVal(src)
    let mut copy_map: HashMap<MVal, MVal> = HashMap::new();

    for blk in &func.blocks {
        for &idx in &blk.insts {
            let inst = &func.insts[idx];
            if let Op::VMov { dst: _, src: Operand::VReg(_) } = &inst.op {
                // Skip: VMov belonging to a coalesced group (e.g. SplatFragment
                // for WMMA) must NOT be propagated — it would break the
                // physical contiguity of the 8-VGPR fragment.
                if inst.coalesced_group.is_some() { continue; }
                // This is a copy: dst_mval → src_mval
                if inst.defs.len() == 1 && inst.uses.len() == 1 {
                    let dst_mval = inst.defs[0];
                    let src_mval = inst.uses[0];
                    copy_map.insert(dst_mval, src_mval);
                }
            }
        }
    }

    if copy_map.is_empty() { return 0; }

    // Step 2: Transitively resolve copy chains (a → b → c becomes a → c)
    let mut resolved: HashMap<MVal, MVal> = HashMap::new();
    for &start in copy_map.keys() {
        let mut current = start;
        let mut seen = std::collections::HashSet::new();
        while let Some(&next) = copy_map.get(&current) {
            if !seen.insert(current) { break; } // cycle guard
            current = next;
        }
        if current != start {
            resolved.insert(start, current);
        }
    }

    // Step 3: Replace uses throughout all instructions
    let mut propagated = 0;
    for blk in &func.blocks {
        for &idx in &blk.insts {
            let inst = &mut func.insts[idx];
            for u in &mut inst.uses {
                if let Some(&replacement) = resolved.get(u) {
                    *u = replacement;
                    propagated += 1;
                }
            }
        }
    }

    propagated
}

// ============================================================================
// MachSSA Constant Folding (D4c)
// ============================================================================

/// SSA-based Constant Folding.
///
/// Tracks MVals with known constant values (from `VMov { .., src: InlineFloat(v) }`).
/// When an ALU op's inputs are all known constants, replaces it with a VMov of
/// the computed result.
///
/// Returns the number of instructions folded.
pub fn constant_fold_mach_func(func: &mut MachFunc) -> usize {
    // Map: MVal → known f32 constant value
    let mut known_f32: HashMap<MVal, f32> = HashMap::new();
    let mut folded = 0;

    for blk in &func.blocks {
        for &idx in &blk.insts {
            let inst = &func.insts[idx];

            // Track VMov from inline constant
            if let Op::VMov { dst: _, src: Operand::InlineFloat(val) } = &inst.op {
                if inst.defs.len() == 1 {
                    known_f32.insert(inst.defs[0], *val);
                }
                continue;
            }

            // Try to fold ALU ops with known constant inputs
            let result_val = match &inst.op {
                Op::VAddF32 { .. } => {
                    if inst.uses.len() == 2 {
                        let a = known_f32.get(&inst.uses[0]);
                        let b = known_f32.get(&inst.uses[1]);
                        if let (Some(&a), Some(&b)) = (a, b) { Some(a + b) } else { None }
                    } else { None }
                }
                Op::VMulF32 { .. } => {
                    if inst.uses.len() == 2 {
                        let a = known_f32.get(&inst.uses[0]);
                        let b = known_f32.get(&inst.uses[1]);
                        if let (Some(&a), Some(&b)) = (a, b) { Some(a * b) } else { None }
                    } else { None }
                }
                Op::VSubF32 { .. } => {
                    if inst.uses.len() == 2 {
                        let a = known_f32.get(&inst.uses[0]);
                        let b = known_f32.get(&inst.uses[1]);
                        if let (Some(&a), Some(&b)) = (a, b) { Some(a - b) } else { None }
                    } else { None }
                }
                Op::VFmaF32 { .. } => {
                    if inst.uses.len() == 3 {
                        let a = known_f32.get(&inst.uses[0]);
                        let b = known_f32.get(&inst.uses[1]);
                        let c = known_f32.get(&inst.uses[2]);
                        if let (Some(&a), Some(&b), Some(&c)) = (a, b, c) {
                            Some(a.mul_add(b, c))
                        } else { None }
                    } else { None }
                }
                _ => None,
            };

            if let Some(val) = result_val {
                if inst.defs.len() == 1 {
                    let dst_mval = inst.defs[0];
                    known_f32.insert(dst_mval, val);

                    // Replace instruction with VMov of constant
                    let dst_vreg = match &inst.op {
                        Op::VAddF32 { dst, .. } | Op::VMulF32 { dst, .. } |
                        Op::VSubF32 { dst, .. } | Op::VFmaF32 { dst, .. } => *dst,
                        _ => unreachable!(),
                    };
                    func.insts[idx] = MachInst {
                        op: Op::VMov { dst: dst_vreg, src: Operand::InlineFloat(val) },
                        defs: vec![dst_mval],
                        uses: vec![],
                        implicit_defs: vec![],
                        implicit_uses: vec![],
                        coalesced_group: None,
                    };
                    folded += 1;
                }
            }
        }
    }

    folded
}

// ============================================================================
// MachSSA Algebraic Simplification (D4d)
// ============================================================================

/// SSA-based Algebraic Simplification.
///
/// Peephole rules applied per instruction:
/// - x + 0 → mov dst, x
/// - x * 1 → mov dst, x
/// - x * 0 → mov dst, 0
/// - x - 0 → mov dst, x
/// - x - x → mov dst, 0
/// - fma(0,b,c) → mov dst, c
/// - fma(a,1,c) → add dst, a, c
/// - fma(a,b,0) → mul dst, a, b
///
/// Returns number of instructions simplified.
pub fn algebraic_simplify_mach_func(func: &mut MachFunc) -> usize {
    let mut simplified = 0;

    for blk in &func.blocks {
        for &idx in &blk.insts {
            let inst = &func.insts[idx];
            let replacement = match &inst.op {
                Op::VAddF32 { dst, src0, src1 } => {
                    if is_zero_op(src1) {
                        Some(Op::VMov { dst: *dst, src: src0.clone() })
                    } else if is_zero_op(src0) {
                        Some(Op::VMov { dst: *dst, src: src1.clone() })
                    } else { None }
                }
                Op::VMulF32 { dst, src0, src1 } => {
                    if is_one_op(src1) {
                        Some(Op::VMov { dst: *dst, src: src0.clone() })
                    } else if is_one_op(src0) {
                        Some(Op::VMov { dst: *dst, src: src1.clone() })
                    } else if is_zero_op(src0) || is_zero_op(src1) {
                        Some(Op::VMov { dst: *dst, src: Operand::InlineFloat(0.0) })
                    } else { None }
                }
                Op::VSubF32 { dst, src0, src1 } => {
                    if is_zero_op(src1) {
                        Some(Op::VMov { dst: *dst, src: src0.clone() })
                    } else if inst.uses.len() == 2 && inst.uses[0] == inst.uses[1] {
                        // x - x → 0 (SSA: same MVal for both uses)
                        Some(Op::VMov { dst: *dst, src: Operand::InlineFloat(0.0) })
                    } else { None }
                }
                Op::VFmaF32 { dst, src0, src1, src2 } => {
                    if is_zero_op(src0) || is_zero_op(src1) {
                        Some(Op::VMov { dst: *dst, src: src2.clone() })
                    } else if is_one_op(src1) {
                        Some(Op::VAddF32 { dst: *dst, src0: src0.clone(), src1: src2.clone() })
                    } else if is_one_op(src0) {
                        Some(Op::VAddF32 { dst: *dst, src0: src1.clone(), src1: src2.clone() })
                    } else if is_zero_op(src2) {
                        Some(Op::VMulF32 { dst: *dst, src0: src0.clone(), src1: src1.clone() })
                    } else { None }
                }
                _ => None,
            };

            if let Some(new_op) = replacement {
                // Rebuild MachInst with new op but preserve defs
                let defs = func.insts[idx].defs.clone();
                let new_uses = super::ssa_ir::extract_vreg_uses_from_op(&new_op);
                // For simplified ops, uses change — but we keep the same MVal mapping.
                // Since we're replacing the Op in-place and the defs stay the same,
                // the MVal chain remains valid. Uses are recalculated from the new op.
                func.insts[idx] = MachInst {
                    op: new_op,
                    defs,
                    uses: func.insts[idx].uses.clone(), // keep original uses (SSA values)
                    implicit_defs: vec![],
                    implicit_uses: vec![],
                    coalesced_group: None,
                };
                simplified += 1;
            }
        }
    }

    simplified
}

/// Helper: check if operand is 0.0 or inline int 0.
fn is_zero_op(op: &Operand) -> bool {
    matches!(op, Operand::InlineFloat(f) if *f == 0.0)
        || matches!(op, Operand::InlineInt(i) if *i == 0)
}

/// Helper: check if operand is 1.0 or inline int 1.
fn is_one_op(op: &Operand) -> bool {
    matches!(op, Operand::InlineFloat(f) if *f == 1.0)
        || matches!(op, Operand::InlineInt(i) if *i == 1)
}

/// Extract VReg uses from an Op (for rebuilding MachInst).
/// This is a public helper used by algebraic_simplify_mach_func.
pub fn extract_vreg_uses_from_op(_op: &Op) -> Vec<VReg> {
    // Note: this is intentionally not used directly since we preserve
    // the original MVal uses during simplification. The SSA value chain
    // is maintained by keeping the same MVal references.
    vec![]
}

// ============================================================================
// MachSSA Common Subexpression Elimination (D4e)
// ============================================================================

/// SSA-based CSE using MVal keys.
///
/// Key insight: in SSA, two instructions with the same opcode and same
/// MVal inputs compute the same value. No need for the complex CseOperand
/// hashing — MVal identity IS the hash.
///
/// Returns the number of instructions eliminated.
pub fn cse_mach_func(func: &mut MachFunc) -> usize {
    // Map: (opcode_tag, sorted_use_mvals) → defining MVal
    let mut seen: HashMap<(u8, Vec<MVal>), MVal> = HashMap::new();
    let mut eliminated = 0;

    // CSE opcode tags (must match the patterns we handle)
    const ADD_F32: u8 = 1;  const MUL_F32: u8 = 2;  const SUB_F32: u8 = 3;
    const MAX_F32: u8 = 4;  const MIN_F32: u8 = 5;  const FMA_F32: u8 = 6;
    const ADD_U32: u8 = 7;  const SUB_U32: u8 = 8;
    const RSQ_F32: u8 = 10; const EXP_F32: u8 = 11; const RCP_F32: u8 = 12;
    const SQRT_F32: u8 = 13; const LOG2_F32: u8 = 14;
    const AND_B32: u8 = 15; const XOR_B32: u8 = 16; const OR_B32: u8 = 17;

    fn is_commutative(tag: u8) -> bool {
        matches!(tag, 1 | 2 | 4 | 5 | 7 | 15 | 16 | 17) // add, mul, max, min, add_u32, and, xor, or
    }

    for blk in &func.blocks {
        // Clear at block boundaries (conservative — don't CSE across blocks)
        seen.clear();

        for &idx in &blk.insts {
            let inst = &func.insts[idx];

            // Control flow: clear map
            if matches!(inst.op,
                Op::Label(_) | Op::Branch(_) | Op::BranchScc0(_) |
                Op::BranchScc1(_) | Op::BranchVccz(_) | Op::Barrier | Op::SBarrier
            ) {
                seen.clear();
                continue;
            }

            // Skip side-effect ops
            if inst.op.has_side_effects() { continue; }

            // Determine CSE tag
            let tag = match &inst.op {
                Op::VAddF32 { .. } => Some(ADD_F32),
                Op::VMulF32 { .. } => Some(MUL_F32),
                Op::VSubF32 { .. } => Some(SUB_F32),
                Op::VMaxF32 { .. } => Some(MAX_F32),
                Op::VMinF32 { .. } => Some(MIN_F32),
                Op::VFmaF32 { .. } => Some(FMA_F32),
                Op::VAddU32 { .. } => Some(ADD_U32),
                Op::VSubU32 { .. } => Some(SUB_U32),
                Op::VRsqF32 { .. } => Some(RSQ_F32),
                Op::VExpF32 { .. } => Some(EXP_F32),
                Op::VRcpF32 { .. } => Some(RCP_F32),
                Op::VSqrtF32 { .. } => Some(SQRT_F32),
                Op::VLog2F32 { .. } => Some(LOG2_F32),
                Op::VAndB32 { .. } => Some(AND_B32),
                Op::VXorB32 { .. } => Some(XOR_B32),
                Op::VOrB32 { .. }  => Some(OR_B32),
                _ => None,
            };

            let tag = match tag { Some(t) => t, None => continue };

            if inst.defs.len() != 1 { continue; }
            let dst_mval = inst.defs[0];

            // Build key from MVal uses (sorted for commutative ops)
            let mut key_uses = inst.uses.clone();
            if is_commutative(tag) && key_uses.len() == 2 && key_uses[1] < key_uses[0] {
                key_uses.swap(0, 1);
            }
            // For FMA: commutative in first two args only
            if tag == FMA_F32 && key_uses.len() == 3 && key_uses[1] < key_uses[0] {
                key_uses.swap(0, 1);
            }

            let key = (tag, key_uses);

            if let Some(&prev_mval) = seen.get(&key) {
                // Redundant! Replace with copy
                let dst_vreg = match &inst.op {
                    Op::VAddF32 { dst, .. } | Op::VMulF32 { dst, .. } |
                    Op::VSubF32 { dst, .. } | Op::VMaxF32 { dst, .. } |
                    Op::VMinF32 { dst, .. } | Op::VFmaF32 { dst, .. } |
                    Op::VAddU32 { dst, .. } | Op::VSubU32 { dst, .. } |
                    Op::VAndB32 { dst, .. } | Op::VXorB32 { dst, .. } |
                    Op::VOrB32 { dst, .. } => *dst,
                    Op::VRsqF32 { dst, .. } | Op::VExpF32 { dst, .. } |
                    Op::VRcpF32 { dst, .. } | Op::VSqrtF32 { dst, .. } |
                    Op::VLog2F32 { dst, .. } => *dst,
                    _ => unreachable!(),
                };
                func.insts[idx] = MachInst {
                    op: Op::VMov { dst: dst_vreg, src: Operand::VReg(VReg(0)) }, // VReg placeholder — lowering uses MVal
                    defs: vec![dst_mval],
                    uses: vec![prev_mval],
                    implicit_defs: vec![],
                    implicit_uses: vec![],
                    coalesced_group: None,
                };
                eliminated += 1;
            } else {
                seen.insert(key, dst_mval);
            }
        }
    }

    eliminated
}

// ============================================================================
// MachSSA Cross-Block CSE using Dominator Tree (D4e+)
// ============================================================================

/// SSA-based CSE with dominator-tree-guided scope inheritance.
///
/// Upgrade over `cse_mach_func`: instead of clearing the `seen` table
/// at every block boundary, traverses blocks in domtree preorder.
/// Each block inherits its idom's `seen` table, so expressions computed
/// in a dominating block can be reused by dominated blocks.
///
/// Safety: if block A dominates block B, then any expression first seen
/// in A is guaranteed to have already been computed at B's entry point.
///
/// Returns the number of instructions eliminated.
pub fn cse_mach_func_domtree(func: &mut MachFunc) -> usize {
    let dt = func.build_domtree();
    let preorder = dt.preorder();

    // CSE opcode tags (same as cse_mach_func)
    const ADD_F32: u8 = 1;  const MUL_F32: u8 = 2;  const SUB_F32: u8 = 3;
    const MAX_F32: u8 = 4;  const MIN_F32: u8 = 5;  const FMA_F32: u8 = 6;
    const ADD_U32: u8 = 7;  const SUB_U32: u8 = 8;
    const RSQ_F32: u8 = 10; const EXP_F32: u8 = 11; const RCP_F32: u8 = 12;
    const SQRT_F32: u8 = 13; const LOG2_F32: u8 = 14;
    const AND_B32: u8 = 15; const XOR_B32: u8 = 16; const OR_B32: u8 = 17;

    fn is_commutative(tag: u8) -> bool {
        matches!(tag, 1 | 2 | 4 | 5 | 7 | 15 | 16 | 17)
    }

    // Per-block seen tables, indexed by block ID
    let mut block_seen: Vec<HashMap<(u8, Vec<MVal>), MVal>> = vec![HashMap::new(); func.blocks.len()];
    let mut eliminated = 0;

    // Build MVal → VReg map so CSE replacement VMov uses the correct source VReg.
    // lower_from_ssa clones Op directly (no VReg remapping), so the VReg in the Op
    // must match the original instruction's VReg, not a placeholder.
    let mval_vreg = build_mval_to_vreg(func);

    for &blk_id in &preorder {
        // Inherit idom's seen table (clone for scope isolation)
        let idom_id = dt.idom(blk_id);
        if idom_id != blk_id {
            block_seen[blk_id as usize] = block_seen[idom_id as usize].clone();
        }

        let inst_indices: Vec<usize> = func.blocks[blk_id as usize].insts.clone();
        for &idx in &inst_indices {
            let inst = &func.insts[idx];

            // ── Barrier-aware CSE: clear seen table at sync points ──
            // Expressions computed before a barrier may produce different results
            // after it (e.g., LDS values written by other waves between barriers).
            // Without this, CSE merges pre-barrier and post-barrier expressions
            // with identical operands, causing GPU hangs in cooperative GEMM kernels.
            if matches!(inst.op, Op::Barrier | Op::SBarrier) {
                block_seen[blk_id as usize].clear();
                continue;
            }

            // Skip side-effect and control-flow ops
            if inst.op.has_side_effects() { continue; }
            if matches!(inst.op,
                Op::Label(_) | Op::Branch(_) | Op::BranchScc0(_) |
                Op::BranchScc1(_) | Op::BranchVccz(_)
            ) { continue; }

            let tag = match &inst.op {
                Op::VAddF32 { .. } => Some(ADD_F32),
                Op::VMulF32 { .. } => Some(MUL_F32),
                Op::VSubF32 { .. } => Some(SUB_F32),
                Op::VMaxF32 { .. } => Some(MAX_F32),
                Op::VMinF32 { .. } => Some(MIN_F32),
                Op::VFmaF32 { .. } => Some(FMA_F32),
                Op::VAddU32 { .. } => Some(ADD_U32),
                Op::VSubU32 { .. } => Some(SUB_U32),
                Op::VRsqF32 { .. } => Some(RSQ_F32),
                Op::VExpF32 { .. } => Some(EXP_F32),
                Op::VRcpF32 { .. } => Some(RCP_F32),
                Op::VSqrtF32 { .. } => Some(SQRT_F32),
                Op::VLog2F32 { .. } => Some(LOG2_F32),
                Op::VAndB32 { .. } => Some(AND_B32),
                Op::VXorB32 { .. } => Some(XOR_B32),
                Op::VOrB32 { .. }  => Some(OR_B32),
                _ => None,
            };
            let tag = match tag { Some(t) => t, None => continue };

            if inst.defs.len() != 1 { continue; }
            let dst_mval = inst.defs[0];

            let mut key_uses = inst.uses.clone();

            // Skip expressions with undefined MVal references (u32::MAX sentinel).
            // lift_to_ssa uses MVal(u32::MAX) for VRegs without tracked definitions.
            // Two different operations can both have MVal(u32::MAX) uses, making
            // them look identical to CSE despite being unrelated computations.
            if key_uses.iter().any(|m| m.0 == u32::MAX) { continue; }

            // Encode inline constant operands into the key.
            // Without this, v_add_u32(d, s, 64) and v_add_u32(d, s, 128) would
            // have identical keys because inline constants don't produce MVal uses.
            // We encode each inline constant as MVal(0xFE000000 | bits) in the key.
            fn encode_inline_operands(op: &Op) -> Vec<MVal> {
                let mut inlines = Vec::new();
                let operands: Vec<&Operand> = match op {
                    Op::VAddF32 { src0, src1, .. } | Op::VMulF32 { src0, src1, .. } |
                    Op::VSubF32 { src0, src1, .. } | Op::VMaxF32 { src0, src1, .. } |
                    Op::VMinF32 { src0, src1, .. } |
                    Op::VAddU32 { src0, src1, .. } | Op::VSubU32 { src0, src1, .. } => {
                        vec![src0, src1]
                    }
                    Op::VFmaF32 { src0, src1, src2, .. } => vec![src0, src1, src2],
                    _ => vec![],
                };
                for (i, op) in operands.iter().enumerate() {
                    match op {
                        Operand::InlineInt(v) => {
                            inlines.push(MVal(0xFE000000 | (i as u32) << 20 | (*v as u32 & 0xFFFFF)));
                        }
                        Operand::InlineFloat(f) => {
                            let bits = f.to_bits();
                            inlines.push(MVal(0xFD000000 | (i as u32) << 20 | (bits >> 12)));
                        }
                        _ => {}
                    }
                }
                inlines
            }
            key_uses.extend(encode_inline_operands(&inst.op));

            if is_commutative(tag) && key_uses.len() >= 2 && key_uses[1] < key_uses[0] {
                key_uses.swap(0, 1);
            }
            if tag == FMA_F32 && key_uses.len() >= 3 && key_uses[1] < key_uses[0] {
                key_uses.swap(0, 1);
            }

            let key = (tag, key_uses);
            let seen = &mut block_seen[blk_id as usize];

            if let Some(&prev_mval) = seen.get(&key) {
                // Look up the VReg that defined prev_mval.
                // lower_from_ssa clones Op directly — VRegs are NOT remapped.
                // Using VReg(0) would emit WORKITEM_ID_X instead of the CSE source!
                let src_vreg = match mval_vreg.get(&prev_mval) {
                    Some(&vr) => vr,
                    None => {
                        // Can't find source VReg — skip this CSE to avoid corruption
                        seen.insert(key, dst_mval);
                        continue;
                    }
                };

                let dst_vreg = match &inst.op {
                    Op::VAddF32 { dst, .. } | Op::VMulF32 { dst, .. } |
                    Op::VSubF32 { dst, .. } | Op::VMaxF32 { dst, .. } |
                    Op::VMinF32 { dst, .. } | Op::VFmaF32 { dst, .. } |
                    Op::VAddU32 { dst, .. } | Op::VSubU32 { dst, .. } |
                    Op::VAndB32 { dst, .. } | Op::VXorB32 { dst, .. } |
                    Op::VOrB32 { dst, .. } => *dst,
                    Op::VRsqF32 { dst, .. } | Op::VExpF32 { dst, .. } |
                    Op::VRcpF32 { dst, .. } | Op::VSqrtF32 { dst, .. } |
                    Op::VLog2F32 { dst, .. } => *dst,
                    _ => unreachable!(),
                };
                func.insts[idx] = MachInst {
                    op: Op::VMov { dst: dst_vreg, src: Operand::VReg(src_vreg) },
                    defs: vec![dst_mval],
                    uses: vec![prev_mval],
                    implicit_defs: vec![],
                    implicit_uses: vec![],
                    coalesced_group: None,
                };
                if std::env::var("T0_DUMP_ASM").is_ok() {
                    eprintln!("[CSE] blk{} inst{}: tag={} v_mov v{}, v{} (MVal {}→{}) key_uses={:?}",
                        blk_id, idx, key.0, dst_vreg.0, src_vreg.0, dst_mval.0, prev_mval.0, &key.1);
                }
                eliminated += 1;
            } else {
                seen.insert(key, dst_mval);
            }
        }
    }

    eliminated
}

// ============================================================================
// MachSSA Loop-Invariant Code Motion (LICM) using Dominator Tree
// ============================================================================

/// SSA-based LICM with dominator-tree-guided loop detection.
///
/// Algorithm:
/// 1. Build domtree, detect back edges (B→H where H dominates B)
/// 2. For each natural loop, collect body blocks via reverse BFS from B to H
/// 3. Build producer map: MVal → block_id (which block defines this value)
/// 4. An instruction is loop-invariant if:
///    - No side effects, no memory ops, no implicit state (VCC/SCC/EXEC)
///    - All MVal uses are defined outside the loop body
/// 5. Hoist invariant instructions to preheader (idom(header))
/// 6. VReg renaming: if hoisted dst VReg is multi-defined in loop, allocate
///    a fresh VReg and patch all downstream uses to prevent register reuse corruption.
///
/// Returns the number of instructions hoisted.

fn remap_group_base(base: &mut VReg, count: u32, old: VReg, new: VReg) {
    if old.0 < base.0 {
        return;
    }
    let offset = old.0 - base.0;
    if offset >= count {
        return;
    }
    if let Some(new_base) = new.0.checked_sub(offset) {
        *base = VReg(new_base);
    }
}

fn remap_group_base_from_map(base: &mut VReg, count: u32, map: &HashMap<VReg, VReg>) {
    let original = *base;
    for offset in 0..count {
        let member = VReg(original.0 + offset);
        if let Some(&new_member) = map.get(&member) {
            if let Some(new_base) = new_member.0.checked_sub(offset) {
                *base = VReg(new_base);
            }
            return;
        }
    }
}

/// Rename destination VRegs in an Op according to the rename map.
fn rename_op_defs(op: &mut Op, map: &HashMap<VReg, VReg>) {
    match op {
        // VALU ops with dst
        Op::VAddF32 { dst, .. } | Op::VMulF32 { dst, .. } |
        Op::VFmaF32 { dst, .. } | Op::VMaxF32 { dst, .. } |
        Op::VMinF32 { dst, .. } | Op::VMinU32 { dst, .. } |
        Op::VMov { dst, .. } |
        Op::VMovFromSgpr { dst, .. } | Op::VAddU32 { dst, .. } |
        Op::VMulLoU32 { dst, .. } | Op::VLshlrevB32 { dst, .. } |
        Op::VLshrrevB32 { dst, .. } | Op::VAndB32 { dst, .. } |
        Op::VXorB32 { dst, .. } | Op::VSubF32 { dst, .. } |
        Op::VSubU32 { dst, .. } | Op::VOrB32 { dst, .. } |
        Op::VRsqF32 { dst, .. } | Op::VExpF32 { dst, .. } |
        Op::VSinF32 { dst, .. } | Op::VCosF32 { dst, .. } |
        Op::VRcpF32 { dst, .. } | Op::VSqrtF32 { dst, .. } |
        Op::VLog2F32 { dst, .. } | Op::VCvtF32U32 { dst, .. } |
        Op::VCvtU32F32 { dst, .. } | Op::VCndmaskB32 { dst, .. } |
        Op::CvtPkBf16F32 { dst, .. } | Op::VAndOrB32 { dst, .. } |
        Op::VPermlanex16B32 { dst, .. } | Op::DsSwizzle { dst, .. } |
        Op::ComputeGlobalIdX { dst, .. } | Op::ReadShaderCycles { dst, .. } |
        Op::VAddCOU32 { dst, .. } | Op::VAddCCU32 { dst, .. } => {
            if let Some(&new) = map.get(dst) { *dst = new; }
        }
        Op::VAddCo { dst, .. } | Op::VAddCoCi { dst, .. } => {
            if let Some(&new) = map.get(dst) { *dst = new; }
        }
        // Memory loads (dst is VReg)
        Op::GlobalLoad { dst, width, .. } |
        Op::BufferLoad { dst, width, .. } |
        Op::LdsLoad { dst, width, .. } => {
            remap_group_base_from_map(dst, width.vreg_count(), map);
        }
        Op::DsLoadB32 { dst, .. } | Op::DsLoadU16 { dst, .. } |
        Op::DsLoadU16D16 { dst, .. } | Op::DsLoadU16D16Hi { dst, .. } => {
            if let Some(&new) = map.get(dst) { *dst = new; }
        }
        Op::DsLoadB64 { dst, .. } => {
            remap_group_base_from_map(dst, 2, map);
        }
        Op::DsLoadB128 { dst, .. } => {
            remap_group_base_from_map(dst, 4, map);
        }
        // Atomics with return
        Op::GlobalAtomicAddU32Rtn { dst, .. } => {
            if let Some(&new) = map.get(dst) { *dst = new; }
        }
        // WMMA (dst is VReg)
        Op::Wmma { dst, format, .. } => {
            remap_group_base_from_map(dst, format.dst_vreg_count(current_target()), map);
        }
        // Wave reductions (val is read/write, tmp is scratch)
        Op::WaveReduceAddF32 { val, tmp } |
        Op::WaveReduceMaxF32 { val, tmp } => {
            if let Some(&new) = map.get(val) { *val = new; }
            if let Some(&new) = map.get(tmp) { *tmp = new; }
        }
        _ => {} // Non-VReg-defining ops don't need renaming
    }
}

/// Rename a specific source VReg in an Op's source operands.
fn rename_op_uses(op: &mut Op, old: VReg, new: VReg) {
    // Helper: rename VReg in Operand
    fn rename_operand(o: &mut Operand, old: VReg, new: VReg) {
        if let Operand::VReg(v) = o {
            if *v == old { *v = new; }
        }
    }

    match op {
        // VALU 2-src (Operand based)
        Op::VAddF32 { src0, src1, .. } | Op::VMulF32 { src0, src1, .. } |
        Op::VMaxF32 { src0, src1, .. } | Op::VMinF32 { src0, src1, .. } |
        Op::VMinU32 { src0, src1, .. } |
        Op::VAddU32 { src0, src1, .. } | Op::VAndB32 { src0, src1, .. } |
        Op::VXorB32 { src0, src1, .. } | Op::VSubF32 { src0, src1, .. } |
        Op::VSubU32 { src0, src1, .. } | Op::VOrB32 { src0, src1, .. } => {
            rename_operand(src0, old, new);
            rename_operand(src1, old, new);
        }
        // VALU 3-src
        Op::VFmaF32 { src0, src1, src2, .. } => {
            rename_operand(src0, old, new);
            rename_operand(src1, old, new);
            rename_operand(src2, old, new);
        }
        // Moves
        Op::VMov { src, .. } => { rename_operand(src, old, new); }
        // Integer ops (VReg-direct)
        Op::VMulLoU32 { src0, src1, .. } => {
            if *src0 == old { *src0 = new; }
            if *src1 == old { *src1 = new; }
        }
        Op::VLshlrevB32 { src, .. } | Op::VLshrrevB32 { src, .. } => {
            if *src == old { *src = new; }
        }
        // 64-bit addr
        Op::VAddCo { src0, src1, .. } | Op::VAddCOU32 { src0, src1, .. } => {
            if *src0 == old { *src0 = new; }
            if *src1 == old { *src1 = new; }
        }
        Op::VAddCoCi { src, .. } | Op::VAddCCU32 { src, .. } => {
            if *src == old { *src = new; }
        }
        // Comparisons (Operand based)
        Op::VCmpLtU32 { src0, src1 } | Op::VCmpGeU32 { src0, src1 } => {
            rename_operand(src0, old, new);
            rename_operand(src1, old, new);
        }
        Op::VCmpGtF32Imm0 { src } => { if *src == old { *src = new; } }
        Op::VCmpGtU32Imm { src, .. } | Op::VCmpEqU32Imm { src, .. } => {
            if *src == old { *src = new; }
        }
        // VCmpGeI32 (VReg-direct)
        Op::VCmpGeI32 { src0, src1 } => {
            if *src0 == old { *src0 = new; }
            if *src1 == old { *src1 = new; }
        }
        Op::VCndmaskB32 { src_false, src_true, .. } => {
            rename_operand(src_false, old, new);
            rename_operand(src_true, old, new);
        }
        // Global memory
        Op::GlobalLoad { addr, .. } => {
            remap_group_base(addr, 2, old, new);
        }
        Op::GlobalStore { addr, src, width, .. } => {
            remap_group_base(addr, 2, old, new);
            remap_group_base(src, width.vreg_count(), old, new);
        }
        Op::BufferLoad { voffset, .. } => {
            if *voffset == old { *voffset = new; }
        }
        Op::BufferStore { voffset, src, width, .. } => {
            if *voffset == old { *voffset = new; }
            remap_group_base(src, width.vreg_count(), old, new);
        }
        Op::GlobalAtomicAddF32 { addr, src, .. } => {
            remap_group_base(addr, 2, old, new);
            if *src == old { *src = new; }
        }
        Op::GlobalAtomicAddU32Rtn { addr, src, .. } => {
            remap_group_base(addr, 2, old, new);
            if *src == old { *src = new; }
        }
        // LDS (legacy)
        Op::LdsLoad { addr, .. } => {
            if *addr == old { *addr = new; }
        }
        Op::LdsStore { addr, src, width, .. } => {
            if *addr == old { *addr = new; }
            remap_group_base(src, width.vreg_count(), old, new);
        }
        // DS stores (all widths)
        Op::DsStoreB16 { vaddr, src, .. } |
        Op::DsStoreB32 { vaddr, src, .. } => {
            if *vaddr == old { *vaddr = new; }
            if *src == old { *src = new; }
        }
        Op::DsStoreB64 { vaddr, src, .. } => {
            if *vaddr == old { *vaddr = new; }
            remap_group_base(src, 2, old, new);
        }
        Op::DsStoreB128 { vaddr, src, .. } => {
            if *vaddr == old { *vaddr = new; }
            remap_group_base(src, 4, old, new);
        }
        // DS loads (all widths)
        Op::DsLoadB32 { vaddr, .. } |
        Op::DsLoadB64 { vaddr, .. } |
        Op::DsLoadB128 { vaddr, .. } |
        Op::DsLoadU16 { vaddr, .. } |
        Op::DsLoadU16D16 { vaddr, .. } |
        Op::DsLoadU16D16Hi { vaddr, .. } => {
            if *vaddr == old { *vaddr = new; }
        }
        // Special math (unary VReg src)
        Op::VRsqF32 { src, .. } | Op::VExpF32 { src, .. } |
        Op::VSinF32 { src, .. } | Op::VCosF32 { src, .. } |
        Op::VRcpF32 { src, .. } | Op::VSqrtF32 { src, .. } |
        Op::VLog2F32 { src, .. } | Op::VCvtF32U32 { src, .. } |
        Op::VCvtU32F32 { src, .. } => {
            if *src == old { *src = new; }
        }
        // Data conversion
        Op::CvtPkBf16F32 { src0, src1, .. } => {
            if *src0 == old { *src0 = new; }
            if *src1 == old { *src1 = new; }
        }
        // Lane permute
        Op::DsSwizzle { src, .. } | Op::VPermlanex16B32 { src, .. } => {
            if *src == old { *src = new; }
        }
        Op::VAndOrB32 { src0, src2, .. } => {
            if *src0 == old { *src0 = new; }
            if *src2 == old { *src2 = new; }
        }
        // VReadfirstlane (VReg → SReg)
        Op::VReadfirstlane { src, .. } => {
            if *src == old { *src = new; }
        }
        // WMMA (rename a, b, c inputs)
        Op::Wmma { a, b, c, format, .. } => {
            let target = current_target();
            remap_group_base(a, format.a_vreg_count(target), old, new);
            remap_group_base(b, format.b_vreg_count(target), old, new);
            remap_group_base(c, format.c_vreg_count(target), old, new);
        }
        // Wave reductions (val and tmp are both read/write)
        Op::WaveReduceAddF32 { val, tmp } |
        Op::WaveReduceMaxF32 { val, tmp } => {
            if *val == old { *val = new; }
            if *tmp == old { *tmp = new; }
        }
        _ => {} // Ops without VReg sources (scalar, control flow, etc.)
    }
}

pub fn licm_mach_func(func: &mut MachFunc) -> usize {
    if func.blocks.len() < 2 { return 0; }

    let dt = func.build_domtree();
    let n = func.blocks.len();

    // ── Step 1: Detect back edges → natural loops ──
    // A back edge is B→H where H dominates B (H is the loop header).
    struct NaturalLoop {
        header: u32,
        body: HashSet<u32>,    // all blocks in the loop (including header)
        preheader: u32,        // idom(header) — where to hoist to
    }

    let mut loops: Vec<NaturalLoop> = Vec::new();

    for blk_id in 0..n as u32 {
        for &succ in &func.blocks[blk_id as usize].succs {
            // Back edge: succ dominates blk_id
            if dt.dominates(succ, blk_id) {
                let header = succ;
                let preheader = dt.idom(header);

                // Don't hoist to self (entry block has idom == self)
                if preheader == header { continue; }

                // Collect loop body via reverse BFS from blk_id to header
                let mut body = HashSet::new();
                body.insert(header);
                if blk_id != header {
                    let mut worklist = vec![blk_id];
                    body.insert(blk_id);
                    while let Some(b) = worklist.pop() {
                        for &pred in &func.blocks[b as usize].preds {
                            if !body.contains(&pred) {
                                body.insert(pred);
                                worklist.push(pred);
                            }
                        }
                    }
                }

                loops.push(NaturalLoop { header, body, preheader });
            }
        }
    }

    if loops.is_empty() { return 0; }

    // ── Step 2: Build producer map: MVal → block_id ──
    let mut mval_block: HashMap<MVal, u32> = HashMap::new();
    for blk in &func.blocks {
        for &idx in &blk.insts {
            for d in &func.insts[idx].defs {
                mval_block.insert(*d, blk.id);
            }
        }
        // Phi defs
        for phi in &blk.phis {
            mval_block.insert(phi.dst, blk.id);
        }
    }

    // ── Step 3: Find invariant instructions and hoist ──
    let mut total_hoisted = 0;

    // Track max VReg across the entire program for fresh VReg allocation
    let mut max_vreg: u32 = 0;
    for inst in &func.insts {
        for vr in inst.op.vreg_refs() {
            if vr.0 > max_vreg && vr.0 < u32::MAX - 100 {
                max_vreg = vr.0;
            }
        }
    }

    for lp in &loops {
        // Pre-compute: which VRegs are defined by MULTIPLE instructions in the loop body?
        let mut vreg_def_count: HashMap<VReg, u32> = HashMap::new();
        for &blk_id in &lp.body {
            for &idx in &func.blocks[blk_id as usize].insts {
                for vr in func.insts[idx].op.vreg_defs() {
                    *vreg_def_count.entry(vr).or_insert(0) += 1;
                }
            }
        }

        // Collect all MVal defs from phi nodes at loop header.
        // Phi-defined values are loop-carried — instructions using them are NOT invariant.
        let phi_defs: HashSet<MVal> = func.blocks[lp.header as usize].phis.iter()
            .map(|phi| phi.dst)
            .collect();

        // Conservative: skip large loops (GEMM K-loops have complex double-buffering
        // with Phase A/B alternation that invariant analysis can't safely handle).
        // Small loops (elementwise, reduction) still benefit from LICM.
        let loop_inst_count: usize = lp.body.iter()
            .map(|&b| func.blocks[b as usize].insts.len())
            .sum();
        if loop_inst_count > 100 {
            continue; // too complex for safe LICM
        }

        // Skip loops containing BufferLoad/BufferStore (GEMM K-loops).
        // These loops have hand-scheduled double-buffering with address computations
        // that appear MVal-invariant but are actually loop-variant via self-updating
        // VRegs in T0's phi-less SSA model. Hoisting address ops from these loops
        // corrupts the K-loop's memory access pattern.
        let has_buffer_ops = lp.body.iter().any(|&b| {
            func.blocks[b as usize].insts.iter().any(|&idx| {
                matches!(&func.insts[idx].op,
                    Op::BufferLoad { .. } | Op::BufferStore { .. })
            })
        });
        if has_buffer_ops {
            continue; // BufferLoad/Store loops need exact scheduling
        }

        // Build MVal → VReg map for the entire function.
        // Used to check if an MVal's underlying VReg is redefined inside the loop.
        let mut mval_to_vreg: HashMap<MVal, VReg> = HashMap::new();
        for blk in &func.blocks {
            for &idx in &blk.insts {
                let inst = &func.insts[idx];
                let vreg_defs = inst.op.vreg_defs();
                for (i, d) in inst.defs.iter().enumerate() {
                    if i < vreg_defs.len() {
                        mval_to_vreg.insert(*d, vreg_defs[i]);
                    }
                }
            }
        }

        // Collect ALL VRegs defined inside the loop body (not just multi-def).
        // In T0's phi-less SSA, a VReg defined inside the loop makes ALL MVals
        // of that VReg loop-variant (even MVals defined in the preheader).
        let loop_defined_vregs: HashSet<VReg> = vreg_def_count.keys().copied().collect();

        // Iteratively find invariant instructions
        let mut invariant_insts: HashSet<usize> = HashSet::new(); // inst indices
        let mut invariant_defs: HashSet<MVal> = HashSet::new();   // MVal defs by invariant insts
        let mut changed = true;

        while changed {
            changed = false;
            for &blk_id in &lp.body {
                // Don't hoist from header (has phi nodes, loop control)
                if blk_id == lp.header { continue; }

                let block = &func.blocks[blk_id as usize];
                for &idx in &block.insts {
                    if invariant_insts.contains(&idx) { continue; }

                    let inst = &func.insts[idx];

                    // Safety filters
                    if inst.op.has_side_effects() { continue; }
                    if inst.defs.is_empty() { continue; }
                    if !inst.implicit_defs.is_empty() || !inst.implicit_uses.is_empty() {
                        continue;
                    }
                    // Skip memory ops (ordering-sensitive)
                    // Skip WMMA (alignment-sensitive: dst/c must be 8-aligned)
                    // Skip atomics, wave reductions, lane ops (side-effects or ordering)
                    if matches!(inst.op,
                        Op::GlobalLoad { .. } | Op::GlobalStore { .. } |
                        Op::BufferLoad { .. } | Op::BufferStore { .. } |
                        Op::LdsLoad { .. } | Op::LdsStore { .. } |
                        Op::DsLoadB32 { .. } | Op::DsLoadB64 { .. } |
                        Op::DsLoadB128 { .. } | Op::DsLoadU16 { .. } |
                        Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. } |
                        Op::DsStoreB16 { .. } | Op::DsStoreB32 { .. } |
                        Op::DsStoreB64 { .. } | Op::DsStoreB128 { .. } |
                        Op::GlobalAtomicAddF32 { .. } | Op::GlobalAtomicAddU32Rtn { .. } |
                        Op::Wmma { .. } |  // WMMA: 8-aligned VReg constraint
                        Op::WaveReduceAddF32 { .. } | Op::WaveReduceMaxF32 { .. } |
                        Op::DsSwizzle { .. } | Op::VPermlanex16B32 { .. } |
                        Op::VReadfirstlane { .. } |
                        Op::Label(_) | Op::Branch(_) | Op::BranchScc0(_) |
                        Op::BranchScc1(_) | Op::BranchVccz(_) |
                        Op::Barrier | Op::SBarrier
                    ) { continue; }

                    // Safety: skip self-update patterns (dst == src, e.g. v += const).
                    // These are loop induction variables and are NEVER invariant.
                    let vreg_uses_set: HashSet<VReg> = inst.op.vreg_uses().into_iter().collect();
                    let vreg_defs_set: HashSet<VReg> = inst.op.vreg_defs().into_iter().collect();
                    if vreg_uses_set.iter().any(|u| vreg_defs_set.contains(u)) {
                        continue;  // self-update → loop induction variable
                    }

                    // Safety: skip if any def VReg has multiple definitions in the loop body.
                    // Multiply-defined VRegs indicate loop-carried values that are NOT invariant.
                    let has_multi_def = inst.op.vreg_defs().iter().any(|vr| {
                        vreg_def_count.get(vr).copied().unwrap_or(0) > 1
                    });
                    if has_multi_def {
                        continue;  // multi-def → loop-carried value
                    }

                    // Safety: skip if any MVal use is defined by a phi at the loop header.
                    // Phi-defined values change each iteration (loop-carried), so instructions
                    // reading them are NOT invariant.
                    let uses_phi_val = inst.uses.iter().any(|u| phi_defs.contains(u));
                    if uses_phi_val {
                        continue;  // uses loop-carried phi value
                    }

                    // Check: all MVal uses defined outside loop OR by invariant inst
                    // CRITICAL FIX: also check that the VReg underlying each MVal is NOT
                    // redefined anywhere inside the loop body. In T0's phi-less SSA,
                    // an MVal may point to a preheader definition, but at runtime the
                    // VReg holds the loop-updated value after the first iteration.
                    let all_invariant = inst.uses.iter().all(|u| {
                        // MVal(u32::MAX) = undefined reference (vreg/sreg not in maps).
                        // Conservatively treat as loop-dependent to prevent incorrect hoisting.
                        if u.0 == u32::MAX { return false; }
                        if invariant_defs.contains(u) { return true; }
                        // Check if the VReg underlying this MVal is redefined in the loop
                        if let Some(vr) = mval_to_vreg.get(u) {
                            if loop_defined_vregs.contains(vr) {
                                return false; // VReg is loop-variant → not invariant
                            }
                        }
                        match mval_block.get(u) {
                            Some(def_blk) => !lp.body.contains(def_blk),
                            None => false, // unknown def location → conservatively NOT invariant
                        }
                    });

                    if all_invariant {
                        invariant_insts.insert(idx);
                        for d in &inst.defs {
                            invariant_defs.insert(*d);
                        }
                        changed = true;
                    }
                }
            }
        }

        if invariant_insts.is_empty() { continue; }

        // ── LICM diagnostic: show what we're hoisting ──
        if std::env::var("T0_DUMP_ASM").is_ok() || std::env::var("T0_LICM_DEBUG").is_ok() {
            eprintln!("[LICM] loop header=bb{}, preheader=bb{}, hoisting {} instructions:",
                lp.header, lp.preheader, invariant_insts.len());
            for &idx in &invariant_insts {
                let inst = &func.insts[idx];
                eprintln!("  [LICM hoist] inst#{}: {:?}", idx, inst.op);
            }
        }

        // ── VReg renaming for multi-def VRegs ──
        // When a hoisted instruction's dst VReg is also defined by another instruction
        // in the loop body, we must allocate a fresh VReg to prevent register reuse
        // corruption. Example: VMovFromSgpr{dst:v22} (hoisted) + VAddCo{dst:v22} (stays)
        // → rename hoisted to VMovFromSgpr{dst:v_NEW} and patch all uses of this MVal.
        let mut rename_map: HashMap<VReg, VReg> = HashMap::new(); // old → new

        for &idx in &invariant_insts {
            let inst = &func.insts[idx];
            let vreg_defs = inst.op.vreg_defs();
            for vr in &vreg_defs {
                if vreg_def_count.get(vr).copied().unwrap_or(0) > 1 {
                    if !rename_map.contains_key(vr) {
                        max_vreg += 1;
                        rename_map.insert(*vr, VReg(max_vreg));
                    }
                }
            }
        }

        if !rename_map.is_empty() {
            // Build MVal→VReg mapping for hoisted instructions' defs
            let mut hoisted_mval_vreg: HashMap<MVal, VReg> = HashMap::new();
            for &idx in &invariant_insts {
                let inst = &func.insts[idx];
                let vreg_defs = inst.op.vreg_defs();
                for (i, vr) in vreg_defs.iter().enumerate() {
                    if rename_map.contains_key(vr) {
                        if i < inst.defs.len() {
                            hoisted_mval_vreg.insert(inst.defs[i], rename_map[vr]);
                        }
                    }
                }
            }

            // 1. Rename dst in hoisted instructions
            for &idx in &invariant_insts {
                let inst = &mut func.insts[idx];
                rename_op_defs(&mut inst.op, &rename_map);
            }

            // 2. Patch downstream uses: find instructions that use hoisted MVal
            //    and replace their source VReg references
            for &blk_id in &lp.body {
                for &idx in &func.blocks[blk_id as usize].insts {
                    if invariant_insts.contains(&idx) { continue; }
                    let inst = &func.insts[idx];
                    // Check if any use references a hoisted MVal
                    let needs_patch: Vec<(MVal, VReg)> = inst.uses.iter()
                        .filter_map(|u| hoisted_mval_vreg.get(u).map(|&new_vr| (*u, new_vr)))
                        .collect();
                    if !needs_patch.is_empty() {
                        let inst = &mut func.insts[idx];
                        for (_, new_vr) in &needs_patch {
                            // Find the old VReg that maps to this new VReg
                            for (old, new) in &rename_map {
                                if new == new_vr {
                                    rename_op_uses(&mut inst.op, *old, *new_vr);
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Hoist: move invariant insts to preheader ──
        let preheader_id = lp.preheader as usize;

        // Remove from original blocks
        for &blk_id in &lp.body {
            let block = &mut func.blocks[blk_id as usize];
            block.insts.retain(|idx| !invariant_insts.contains(idx));
        }

        // Insert hoisted instructions BEFORE the terminator (last inst) in preheader.
        let hoisted_indices: Vec<usize> = invariant_insts.iter().copied().collect();
        let ph_insts = &mut func.blocks[preheader_id].insts;
        let insert_pos = if ph_insts.is_empty() {
            0
        } else {
            let last_idx = *ph_insts.last().unwrap();
            let is_term = matches!(func.insts[last_idx].op,
                Op::Branch(_) | Op::BranchScc0(_) | Op::BranchScc1(_) |
                Op::BranchVccz(_) | Op::Endpgm
            );
            if is_term {
                ph_insts.len() - 1
            } else {
                ph_insts.len()
            }
        };
        for (i, &idx) in hoisted_indices.iter().enumerate() {
            ph_insts.insert(insert_pos + i, idx);
        }

        total_hoisted += invariant_insts.len();
    }

    total_hoisted
}

// ============================================================================
// MachSSA Instruction Combining (D4f)
// ============================================================================

/// SSA-based Instruction Combining: mul + add → fma.
///
/// In SSA, single-use detection is trivial: count how many instructions
/// reference each MVal. If a mul's result MVal has exactly one use (an add),
/// combine them.
///
/// Returns the number of instructions combined.
pub fn instruction_combine_mach_func(func: &mut MachFunc) -> usize {
    // Step 1: Build use-count for each MVal
    let mut use_count: HashMap<MVal, usize> = HashMap::new();
    for blk in &func.blocks {
        for &idx in &blk.insts {
            for u in &func.insts[idx].uses {
                *use_count.entry(*u).or_insert(0) += 1;
            }
        }
    }

    // Step 2: Build producer map: MVal → inst_index
    let mut producer: HashMap<MVal, usize> = HashMap::new();
    for blk in &func.blocks {
        for &idx in &blk.insts {
            for d in &func.insts[idx].defs {
                producer.insert(*d, idx);
            }
        }
    }

    // Step 3: Scan for combinable patterns
    let mut combined = 0;
    let mut dead_insts: HashSet<usize> = HashSet::new();

    for blk in &func.blocks {
        for &idx in &blk.insts {
            if dead_insts.contains(&idx) { continue; }

            let inst = &func.insts[idx];
            if let Op::VAddF32 { dst, src0, src1 } = &inst.op {
                // Try each MVal use: if it comes from a single-use VMulF32, combine
                let mut did_combine = false;
                for use_pos in 0..inst.uses.len() {
                    let mul_mval = inst.uses[use_pos];
                    if use_count.get(&mul_mval).copied().unwrap_or(0) != 1 { continue; }

                    let prod_idx = match producer.get(&mul_mval) {
                        Some(&pi) => pi,
                        None => continue,
                    };
                    if dead_insts.contains(&prod_idx) { continue; }

                    let prod = &func.insts[prod_idx];
                    if let Op::VMulF32 { src0: mul_a, src1: mul_b, .. } = &prod.op {
                        // Determine which src of the add is the mul result, and which is "other"
                        // We match by checking which Operand is a VReg referencing mul's dst
                        let (other_operand, other_mval) = if matches!(src0, Operand::VReg(_)) && use_pos == 0 {
                            // src0 is the mul result → other is src1
                            (src1.clone(), inst.uses.get(1).copied())
                        } else if inst.uses.len() >= 2 && use_pos == 1 {
                            // src1 is the mul result → other is src0
                            (src0.clone(), Some(inst.uses[0]))
                        } else if inst.uses.len() == 1 {
                            // Only 1 MVal use (the mul result). The other operand must be inline.
                            // Determine which src is the VReg (mul result) by checking the Op
                            if matches!(src0, Operand::VReg(_)) {
                                (src1.clone(), None) // src0 is VReg(mul), src1 is inline constant
                            } else {
                                (src0.clone(), None) // src1 is VReg(mul), src0 is inline constant
                            }
                        } else {
                            continue;
                        };

                        // Build FMA uses: mul's MVal uses + other MVal (if any)
                        let fma_defs = func.insts[idx].defs.clone();
                        let mut fma_uses = prod.uses.clone(); // mul's uses (a, b)
                        if let Some(other_mv) = other_mval {
                            fma_uses.push(other_mv);
                        }

                        func.insts[idx] = MachInst {
                            op: Op::VFmaF32 {
                                dst: *dst,
                                src0: mul_a.clone(),
                                src1: mul_b.clone(),
                                src2: other_operand,
                            },
                            defs: fma_defs,
                            uses: fma_uses,
                            implicit_defs: vec![],
                            implicit_uses: vec![],
                            coalesced_group: None,
                        };
                        dead_insts.insert(prod_idx);
                        combined += 1;
                        did_combine = true;
                        break;
                    }
                }
                if did_combine { continue; }
            }
        }
    }

    // Remove dead mul instructions from blocks
    if !dead_insts.is_empty() {
        for blk in &mut func.blocks {
            blk.insts.retain(|idx| !dead_insts.contains(idx));
        }
    }

    combined
}

// ============================================================================
// MachSSA Waitcnt Optimization (D4g)
// ============================================================================

/// SSA-based waitcnt optimization.
///
/// Tracks pending memory operation counts per block and removes
/// redundant s_waitcnt when the counter is already zero.
///
/// Returns the number of waitcnts removed.
pub fn optimize_waitcnt_mach_func(func: &mut MachFunc) -> usize {
    let mut total_removed = 0;

    for blk in &mut func.blocks {
        let mut pending_vmcnt: u32 = 0;
        let mut pending_lgkmcnt: u32 = 0;
        let mut pending_vscnt: u32 = 0;
        let mut to_remove: Vec<usize> = Vec::new();

        for &idx in &blk.insts {
            let inst = &func.insts[idx];

            // Control flow: reset counters
            if matches!(inst.op,
                Op::Label(_) | Op::Branch(_) | Op::BranchScc0(_) |
                Op::BranchScc1(_) | Op::BranchVccz(_) | Op::Barrier | Op::SBarrier
            ) {
                pending_vmcnt = u32::MAX;
                pending_lgkmcnt = u32::MAX;
                pending_vscnt = u32::MAX;
                continue;
            }

            // Track memory ops
            // CRITICAL: must track ALL memory instructions, including BufferLoad/BufferStore
            // which tile_ir uses instead of GlobalLoad for cooperative GMEM→LDS loads.
            match &inst.op {
                Op::GlobalLoad { .. } | Op::BufferLoad { .. } => {
                    pending_vmcnt = pending_vmcnt.saturating_add(1);
                }
                Op::GlobalStore { .. } | Op::BufferStore { .. } => {
                    pending_vmcnt = pending_vmcnt.saturating_add(1);
                    pending_vscnt = pending_vscnt.saturating_add(1);
                }
                Op::GlobalAtomicAddF32 { .. } |
                Op::GlobalAtomicAddU32Rtn { .. } => {
                    pending_vmcnt = pending_vmcnt.saturating_add(1);
                }
                Op::LdsLoad { .. } | Op::DsLoadB32 { .. } | Op::DsLoadB64 { .. } |
                Op::DsLoadB128 { .. } | Op::DsLoadU16 { .. } |
                Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. } |
                Op::ScalarLoad { .. } | Op::SMemLoadDword { .. } => {
                    pending_lgkmcnt = pending_lgkmcnt.saturating_add(1);
                }
                Op::LdsStore { .. } | Op::DsStoreB16 { .. } | Op::DsStoreB32 { .. } |
                Op::DsStoreB64 { .. } | Op::DsStoreB128 { .. } => {
                    pending_lgkmcnt = pending_lgkmcnt.saturating_add(1);
                }
                _ => {}
            }

            // Check waitcnt — graduated wait handling:
            // wait_vmcnt(N) means "wait until at most N operations outstanding".
            // After wait_vmcnt(N), pending = min(pending, N), NOT 0.
            // Only wait_vmcnt(0) fully drains the counter.
            // A wait_vmcnt(N) is redundant if pending <= N (already satisfied).
            match &inst.op {
                Op::WaitVmcnt(n) => {
                    let n = *n as u32;
                    if pending_vmcnt <= n {
                        // Already at or below target — waitcnt is redundant
                        to_remove.push(idx);
                    } else {
                        pending_vmcnt = n;
                    }
                }
                Op::WaitLgkmcnt(n) => {
                    let n = *n as u32;
                    if pending_lgkmcnt <= n {
                        to_remove.push(idx);
                    } else {
                        pending_lgkmcnt = n;
                    }
                }
                Op::WaitVscnt(n) => {
                    let n = *n as u32;
                    if pending_vscnt <= n {
                        to_remove.push(idx);
                    } else {
                        pending_vscnt = n;
                    }
                }
                _ => {}
            }
        }

        if !to_remove.is_empty() {
            let remove_set: HashSet<usize> = to_remove.into_iter().collect();
            let before = blk.insts.len();
            blk.insts.retain(|idx| !remove_set.contains(idx));
            total_removed += before - blk.insts.len();
        }
    }

    total_removed
}

// ============================================================================
// Wrapper passes: delegate to existing Vec<Op> implementations (D4h)
// ============================================================================

// Loop-based passes (loop_unroll, LICM, strength_reduce, software_pipeline)
// and memory-pattern passes (coalesce_loads) work on linear loop structure
// detected via Label/Branch patterns. These don't benefit from SSA and are
// kept as Vec<Op> implementations, called directly from optimize().

// ============================================================================
// SSA Live Interval Analysis (Phase E1)
// ============================================================================

/// Live interval for a single MVal in the SSA graph.
///
/// Represents the range of instruction indices where this value is alive.
/// Used by the SSA register allocator to determine interference.
#[derive(Clone, Debug)]
pub struct LiveInterval {
    /// The SSA value this interval represents
    pub mval: MVal,
    /// Global instruction sequence number where this value is defined
    pub def_point: u32,
    /// Global instruction sequence number of the last use
    pub last_use: u32,
    /// The original VReg that this MVal corresponds to (for constraint propagation)
    pub vreg: VReg,
    /// If this MVal is part of a consecutive register group, the group ID.
    /// All MVal with the same group_id must be allocated to consecutive physical regs.
    pub group_id: Option<u32>,
    /// Position within the group (0-based). E.g., for an 8-wide WMMA accumulator,
    /// index 0 gets the base register, index 1 gets base+1, etc.
    pub group_index: u32,
    /// Alignment constraint inherited from VRegAlloc
    pub alignment: Alignment,
    /// Number of physical registers in the group (1 for scalar, 8 for WMMA)
    pub group_size: u32,
}

/// Build a mapping from MVal → VReg by inspecting the original Op in each MachInst.
///
/// The lift_to_ssa() creates MVal in the same order as vreg_defs(), so we can
/// reconstruct which MVal corresponds to which VReg from the def list + op.
pub fn build_mval_to_vreg(func: &MachFunc) -> HashMap<MVal, VReg> {
    let mut map = HashMap::new();
    for inst in &func.insts {
        let vreg_defs = inst.op.vreg_defs();
        // defs and vreg_defs are parallel arrays (same order from lift_to_ssa)
        for (mval, vreg) in inst.defs.iter().zip(vreg_defs.iter()) {
            map.insert(*mval, *vreg);
        }
    }
    map
}

/// Compute live intervals for all MVal defined in a MachFunc.
///
/// # Algorithm
///
/// 1. **Linearize**: Assign a global sequence number to each instruction
///    (block 0 insts first, then block 1, etc.)
/// 2. **Def/Use scan**: For each MVal, record def_point and last_use
/// 3. **Loop extension**: For backward edges (succ.id < block.id), extend
///    live ranges of any MVal live inside the loop body to the loop end
/// 4. **Constraint propagation**: Map MVal → VReg → VRegAlloc to inherit
///    alignment and consecutive-register group constraints
///
/// # Returns
///
/// A Vec<LiveInterval> sorted by def_point (ascending).
/// Only MVal that are actually used (have at least one use) are included.
pub fn compute_live_intervals(
    func: &MachFunc,
    vreg_allocs: &[VRegAlloc],
) -> Vec<LiveInterval> {
    if func.blocks.is_empty() || func.insts.is_empty() {
        return Vec::new();
    }

    // ── Step 1: Linearize instruction indices ──
    // Map global_seq → inst_index (into func.insts)
    let mut seq_to_inst: Vec<usize> = Vec::with_capacity(func.num_insts());
    // Map inst_index → global_seq
    let mut inst_to_seq: HashMap<usize, u32> = HashMap::new();
    // Map inst_index → block_id
    let mut inst_to_block: HashMap<usize, u32> = HashMap::new();

    for blk in &func.blocks {
        for &inst_idx in &blk.insts {
            let seq = seq_to_inst.len() as u32;
            inst_to_seq.insert(inst_idx, seq);
            inst_to_block.insert(inst_idx, blk.id);
            seq_to_inst.push(inst_idx);
        }
    }

    let total_insts = seq_to_inst.len() as u32;

    // ── Step 2: Compute def_point and last_use for each MVal ──
    let mut def_points: HashMap<MVal, u32> = HashMap::new();
    let mut last_uses: HashMap<MVal, u32> = HashMap::new();

    for (inst_idx, inst) in func.insts.iter().enumerate() {
        let seq = match inst_to_seq.get(&inst_idx) {
            Some(&s) => s,
            None => continue, // inst not in any block (shouldn't happen)
        };

        // Record defs
        for mval in &inst.defs {
            def_points.entry(*mval).or_insert(seq);
        }

        // Record uses (update last_use to the maximum seq)
        for mval in &inst.uses {
            if mval.0 == u32::MAX { continue; } // unresolved use, skip
            let entry = last_uses.entry(*mval).or_insert(seq);
            if seq > *entry {
                *entry = seq;
            }
        }
    }

    // ── Step 3: Loop backedge extension ──
    // Find loop ranges from CFG: if block B has a successor A where A.id < B.id,
    // that's a backward edge defining a loop [A_first_seq, B_last_seq].
    let mut loop_ranges: Vec<(u32, u32)> = Vec::new(); // (start_seq, end_seq)

    // Pre-compute block seq ranges
    let mut block_seq_ranges: Vec<(u32, u32)> = Vec::new(); // (first_seq, last_seq) per block
    for blk in &func.blocks {
        if blk.insts.is_empty() {
            block_seq_ranges.push((0, 0));
            continue;
        }
        let first = inst_to_seq[&blk.insts[0]];
        let last = inst_to_seq[blk.insts.last().unwrap()];
        block_seq_ranges.push((first, last));
    }

    for blk in &func.blocks {
        for &succ_id in &blk.succs {
            if succ_id <= blk.id {
                // Backward edge (or self-loop): loop from succ_id to blk.id
                let loop_start = block_seq_ranges[succ_id as usize].0;
                let loop_end = block_seq_ranges[blk.id as usize].1;
                loop_ranges.push((loop_start, loop_end));
            }
        }
    }

    // Extend live ranges for values used inside loops.
    //
    // CRITICAL DISTINCTION:
    //   - Loop-CARRIED values (defined BEFORE the loop, used inside) → MUST extend
    //     to loop_end because the value persists across iterations.
    //   - In-place updates (value defined inside loop that belongs to a VReg
    //     also defined before the loop, e.g., accumulators) → MUST extend.
    //   - Loop-LOCAL temporaries (VReg only defined inside loop, never before it)
    //     → do NOT extend. These are re-created each iteration.
    //
    // The key insight: after Step 6 (VReg merge), all MVal versions of the same
    // VReg become ONE interval. If any MVal version isn't extended, the merged
    // interval's last_use may be too short → regalloc overwrites the register
    // mid-loop → silent data corruption.

    // Build MVal → VReg mapping (needed to detect in-place updates)
    let mval_to_vreg_tmp = build_mval_to_vreg(func);

    // Identify VRegs that have at least one def BEFORE any loop
    let earliest_loop_start = loop_ranges.iter().map(|r| r.0).min().unwrap_or(u32::MAX);
    let mut pre_loop_vregs: std::collections::HashSet<VReg> = std::collections::HashSet::new();
    for (&mv, &def) in def_points.iter() {
        if def < earliest_loop_start {
            if let Some(&vreg) = mval_to_vreg_tmp.get(&mv) {
                pre_loop_vregs.insert(vreg);
            }
        }
    }

    for &(loop_start, loop_end) in &loop_ranges {
        for (mval, last_use) in last_uses.iter_mut() {
            if *last_use >= loop_start && *last_use <= loop_end {
                if let Some(&def) = def_points.get(mval) {
                    // Case 1: Defined BEFORE the loop → loop-carried → extend
                    if def < loop_start {
                        *last_use = loop_end;
                        continue;
                    }
                    // Case 2: Defined INSIDE loop — check if in-place update
                    // (same VReg has a definition before the loop)
                    if def >= loop_start && def <= loop_end {
                        if let Some(&vreg) = mval_to_vreg_tmp.get(mval) {
                            if pre_loop_vregs.contains(&vreg) {
                                // In-place update of a loop-carried VReg → extend
                                *last_use = loop_end;
                                continue;
                            }
                        }
                        // Pure loop-local temporary → do NOT extend
                    }
                }
            }
        }
    }

    // ── Step 4: Build MVal → VReg mapping and propagate constraints ──
    let mval_to_vreg = build_mval_to_vreg(func);

    // Build VReg → VRegAlloc index mapping
    let mut vreg_to_alloc_idx: HashMap<VReg, usize> = HashMap::new();
    for (idx, va) in vreg_allocs.iter().enumerate() {
        for i in 0..va.count {
            vreg_to_alloc_idx.insert(VReg(va.vreg.0 + i), idx);
        }
    }

    // ── Step 5: Build live intervals ──
    let mut intervals: Vec<LiveInterval> = Vec::new();

    for (&mval, &def_seq) in &def_points {
        // Skip values that are never used (dead defs)
        let last_use_seq = match last_uses.get(&mval) {
            Some(&lu) => lu,
            None => continue, // dead value — no interval needed
        };

        // Ensure last_use >= def_point (a value is alive at its def point)
        let last_use_seq = last_use_seq.max(def_seq);

        // Look up the original VReg
        let vreg = mval_to_vreg.get(&mval).copied().unwrap_or(VReg(u32::MAX));

        // Look up allocation constraints
        let (alignment, group_id, group_index, group_size) =
            if let Some(&alloc_idx) = vreg_to_alloc_idx.get(&vreg) {
                let va = &vreg_allocs[alloc_idx];
                let idx_in_group = vreg.0 - va.vreg.0;
                (va.alignment, Some(alloc_idx as u32), idx_in_group, va.count)
            } else {
                // VReg not in any alloc (e.g., v0 = hardware TID) — default constraints
                (Alignment::None, None, 0, 1)
            };

        intervals.push(LiveInterval {
            mval,
            def_point: def_seq,
            last_use: last_use_seq,
            vreg,
            group_id,
            group_index,
            alignment,
            group_size,
        });
    }

    // ── Step 6: Merge intervals for the same VReg ──
    // CRITICAL: In-place ops (e.g., v_add_co(v5, v5, v3)) create multiple
    // MVal versions of the same VReg. Each MVal gets its own LiveInterval.
    // But to_legacy_regalloc maps VReg → ONE physical register, so all
    // MVal versions of the same VReg MUST get the same physical register.
    // We achieve this by merging all intervals for the same VReg into one
    // interval with def_point = min(all defs), last_use = max(all uses).
    // This is equivalent to "coalescing" the SSA versions.
    let mut vreg_merged: HashMap<VReg, usize> = HashMap::new(); // VReg → index in merged
    let mut merged: Vec<LiveInterval> = Vec::new();

    for interval in &intervals {
        if interval.vreg.0 == u32::MAX { continue; } // skip unresolved
        
        if let Some(&merged_idx) = vreg_merged.get(&interval.vreg) {
            // Merge: widen the existing interval
            let m = &mut merged[merged_idx];
            m.def_point = m.def_point.min(interval.def_point);
            m.last_use = m.last_use.max(interval.last_use);
        } else {
            // First interval for this VReg
            vreg_merged.insert(interval.vreg, merged.len());
            merged.push(interval.clone());
        }
    }

    // Sort by def_point ascending (tie-break by MVal ID for determinism)
    merged.sort_by(|a, b| {
        a.def_point.cmp(&b.def_point)
            .then(a.mval.0.cmp(&b.mval.0))
    });

    merged
}

// ============================================================================
// Lower: MachFunc → Vec<Op>
// ============================================================================

/// Lower a MachFunc back to a linear `Vec<Op>`.
///
/// Steps:
/// 1. Emit blocks in order (block 0, 1, 2, ...)
/// 2. Phi nodes → VMov copies inserted at the end of predecessor blocks
///    (for now, skip phi lowering since D1 doesn't generate phis)
/// 3. Extract `MachInst.op` for each instruction
pub fn lower_from_ssa(func: &MachFunc) -> Vec<Op> {
    if func.blocks.is_empty() {
        return Vec::new();
    }

    // Build MVal → VReg mapping from all definitions.
    // This is the ground truth: each MVal was produced by an instruction
    // whose Op has a specific VReg as its destination.
    let mval_to_vreg = build_mval_to_vreg(func);

    let mut result: Vec<Op> = Vec::with_capacity(func.num_insts());

    for blk in &func.blocks {
        for &idx in &blk.insts {
            let inst = &func.insts[idx];
            let mut op = inst.op.clone();

            // ── Remap USE VRegs ──
            // After SSA passes (CSE, CopyProp), an instruction's `uses` MVal list
            // may point to different MVal than the Op's VRegs suggest.
            // CRITICAL: recalculate op_vreg_uses after each rename because
            // rename_op_uses mutates the op in place. Using stale values causes
            // subsequent renames to search for VRegs that no longer exist in the op
            // (especially consecutive VRegs like addr+1 in global_load/store).
            for (i, &use_mval) in inst.uses.iter().enumerate() {
                let current_uses = op.vreg_uses();
                if i < current_uses.len() {
                    if let Some(&correct_vreg) = mval_to_vreg.get(&use_mval) {
                        let current_vreg = current_uses[i];
                        if correct_vreg != current_vreg {
                            rename_op_uses(&mut op, current_vreg, correct_vreg);
                        }
                    }
                }
            }

            // ── Remap DEF VRegs ──
            // Less common, but SSA passes can also change what VReg a def maps to.
            let op_vreg_defs = op.vreg_defs();
            let mut def_remap: HashMap<VReg, VReg> = HashMap::new();
            for (i, &def_mval) in inst.defs.iter().enumerate() {
                if i < op_vreg_defs.len() {
                    if let Some(&correct_vreg) = mval_to_vreg.get(&def_mval) {
                        let current_vreg = op_vreg_defs[i];
                        if correct_vreg != current_vreg {
                            def_remap.insert(current_vreg, correct_vreg);
                        }
                    }
                }
            }
            if !def_remap.is_empty() {
                rename_op_defs(&mut op, &def_remap);
            }

            result.push(op);
        }
    }

    // Diagnostic: count how many ops had VReg remaps
    if std::env::var("T0_DUMP_ASM").is_ok() {
        let mut n_use_remaps = 0usize;
        let mut n_def_remaps = 0usize;
        for blk in &func.blocks {
            for &idx in &blk.insts {
                let inst = &func.insts[idx];
                let op_uses = inst.op.vreg_uses();
                for (i, &use_mval) in inst.uses.iter().enumerate() {
                    if i < op_uses.len() {
                        if let Some(&cv) = mval_to_vreg.get(&use_mval) {
                            if cv != op_uses[i] { n_use_remaps += 1; }
                        }
                    }
                }
                let op_defs = inst.op.vreg_defs();
                for (i, &def_mval) in inst.defs.iter().enumerate() {
                    if i < op_defs.len() {
                        if let Some(&cv) = mval_to_vreg.get(&def_mval) {
                            if cv != op_defs[i] { n_def_remaps += 1; }
                        }
                    }
                }
            }
        }
        if n_use_remaps > 0 || n_def_remaps > 0 {
            eprintln!("[lower_from_ssa] {} USE remaps, {} DEF remaps", n_use_remaps, n_def_remaps);
        }
    }

    result
}

// ============================================================================
// MachFunc — CfgProvider integration (Dominator Tree)
// ============================================================================

use super::domtree::{CfgProvider, DomTree};

impl CfgProvider for MachFunc {
    fn num_blocks(&self) -> usize { self.blocks.len() }
    fn entry(&self) -> u32 { 0 }
    fn preds(&self, block: u32) -> &[u32] { &self.blocks[block as usize].preds }
    fn succs(&self, block: u32) -> &[u32] { &self.blocks[block as usize].succs }
}

impl MachFunc {
    /// Build Dominator Tree from the current MachFunc CFG.
    pub fn build_domtree(&self) -> DomTree {
        DomTree::build(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lift_basic() {
        // Straight-line code → 1 block, correct def/use MVal
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(0.0) },
            Op::VAddF32 {
                dst: VReg(2),
                src0: Operand::VReg(VReg(1)),
                src1: Operand::InlineFloat(1.0),
            },
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let func = lift_to_ssa(&ops);

        // Should have 1 block (no labels/branches except endpgm)
        assert_eq!(func.num_blocks(), 1, "straight-line code → 1 block");
        assert_eq!(func.num_insts(), 4, "4 instructions");

        // VMov defines v1 → MVal(0)
        assert_eq!(func.insts[0].defs, vec![MVal(0)]);
        assert!(func.insts[0].uses.is_empty(), "VMov from inline has no VReg uses");

        // VAddF32 uses v1(MVal(0)), defines v2 → MVal(1)
        assert_eq!(func.insts[1].defs, vec![MVal(1)]);
        assert_eq!(func.insts[1].uses, vec![MVal(0)], "VAddF32 uses v1=MVal(0)");

        // GlobalStore uses v10(addr) and v2(src) — v10 has no def → MVal(MAX)
        assert!(func.insts[2].defs.is_empty(), "store defines nothing");
        // v10 and v11 (addr pair) + v2 (src) — v10,v11 undefined → MAX, v2 → MVal(1)
        let store_uses = &func.insts[2].uses;
        assert!(store_uses.contains(&MVal(1)), "store uses v2=MVal(1)");
    }

    #[test]
    fn test_lift_branch() {
        // Code with Label + Branch → multiple blocks, correct CFG
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(0.0) },
            Op::Label("loop".to_string()),
            Op::VAddF32 {
                dst: VReg(1),
                src0: Operand::VReg(VReg(1)),
                src1: Operand::InlineFloat(1.0),
            },
            Op::SCmpLtU32 { src0: SReg(0), src1: SReg(1) },
            Op::BranchScc1("loop".to_string()),
            Op::Endpgm,
        ];

        let func = lift_to_ssa(&ops);

        // Expect 3 blocks: [VMov], [Label..BranchScc1], [Endpgm]
        assert_eq!(func.num_blocks(), 3,
            "expected 3 blocks, got {}: {}", func.num_blocks(), func.dump());

        // Block 0 → Block 1 (fall-through)
        assert!(func.blocks[0].succs.contains(&1), "BB0 → BB1");
        // Block 1 → Block 1 (back-edge: BranchScc1 → "loop") + Block 2 (fall-through)
        assert!(func.blocks[1].succs.contains(&1), "BB1 → BB1 (back-edge)");
        assert!(func.blocks[1].succs.contains(&2), "BB1 → BB2 (fall-through)");
        // Block 1 preds: Block 0 + Block 1 (self-loop)
        assert!(func.blocks[1].preds.contains(&0), "BB1 pred: BB0");
        assert!(func.blocks[1].preds.contains(&1), "BB1 pred: BB1 (back-edge)");
    }

    #[test]
    fn test_lift_vcc() {
        // VCmpLtU32 writes VCC, VCndmask reads VCC
        let ops = vec![
            Op::VCmpLtU32 {
                src0: Operand::VReg(VReg(1)),
                src1: Operand::VReg(VReg(2)),
            },
            Op::VCndmaskB32 {
                dst: VReg(3),
                src_false: Operand::InlineFloat(0.0),
                src_true: Operand::InlineFloat(1.0),
            },
            Op::Endpgm,
        ];

        let func = lift_to_ssa(&ops);
        assert_eq!(func.num_insts(), 3);

        // VCmpLtU32: implicit_defs = [Vcc]
        assert_eq!(func.insts[0].implicit_defs, vec![ImplicitReg::Vcc],
            "VCmpLtU32 should write VCC");
        assert!(func.insts[0].implicit_uses.is_empty(),
            "VCmpLtU32 should not read VCC");

        // VCndmaskB32: implicit_uses = [Vcc]
        assert!(func.insts[1].implicit_defs.is_empty(),
            "VCndmask should not write VCC");
        assert_eq!(func.insts[1].implicit_uses, vec![ImplicitReg::Vcc],
            "VCndmask should read VCC");
    }

    #[test]
    fn test_lift_scc() {
        // SAddU32 writes SCC, BranchScc1 reads SCC
        let ops = vec![
            Op::SAddU32 { dst: SReg(0), src0: SReg(0), src1: SOperand::InlineInt(1) },
            Op::SCmpLtU32 { src0: SReg(0), src1: SReg(1) },
            Op::BranchScc1("target".to_string()),
            Op::Label("target".to_string()),
            Op::Endpgm,
        ];

        let func = lift_to_ssa(&ops);

        // SAddU32: writes SCC
        assert!(func.insts[0].implicit_defs.contains(&ImplicitReg::Scc),
            "SAddU32 should write SCC");

        // SCmpLtU32: writes SCC
        assert!(func.insts[1].implicit_defs.contains(&ImplicitReg::Scc),
            "SCmpLtU32 should write SCC");

        // BranchScc1: reads SCC
        assert!(func.insts[2].implicit_uses.contains(&ImplicitReg::Scc),
            "BranchScc1 should read SCC");
    }

    #[test]
    fn test_roundtrip() {
        // lift → lower should produce semantically equivalent ops
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(0.0) },
            Op::VAddF32 {
                dst: VReg(2),
                src0: Operand::VReg(VReg(1)),
                src1: Operand::InlineFloat(1.0),
            },
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let func = lift_to_ssa(&ops);
        let lowered = lower_from_ssa(&func);

        assert_eq!(lowered.len(), ops.len(), "round-trip should preserve op count");

        // Verify each op matches (using Debug representation since Op doesn't impl PartialEq)
        for (i, (orig, low)) in ops.iter().zip(lowered.iter()).enumerate() {
            assert_eq!(
                format!("{:?}", orig),
                format!("{:?}", low),
                "op {} mismatch after round-trip", i
            );
        }
    }

    #[test]
    fn test_rename_op_uses_wmma_group_member_updates_base_on_gfx1201() {
        with_target_context(Target::GFX1201, || {
            let mut op = Op::Wmma {
                dst: VReg(100),
                a: VReg(1),
                b: VReg(5),
                c: VReg(9),
                format: WmmaFormat::BF16_F32,
            };
            rename_op_uses(&mut op, VReg(8), VReg(28));
            match op {
                Op::Wmma { b, .. } => assert_eq!(b, VReg(25)),
                _ => panic!("expected WMMA op"),
            }
        });
    }

    #[test]
    fn test_rename_op_defs_wmma_group_member_updates_base_on_gfx1201() {
        with_target_context(Target::GFX1201, || {
            let mut op = Op::Wmma {
                dst: VReg(100),
                a: VReg(1),
                b: VReg(5),
                c: VReg(9),
                format: WmmaFormat::BF16_F32,
            };
            let mut map = std::collections::HashMap::new();
            map.insert(VReg(107), VReg(207));
            rename_op_defs(&mut op, &map);
            match op {
                Op::Wmma { dst, .. } => assert_eq!(dst, VReg(200)),
                _ => panic!("expected WMMA op"),
            }
        });
    }

    #[test]
    fn test_schedule_latency_hiding() {
        // Load → independent ALU → waitcnt → use
        // Scheduler should move independent ALU before waitcnt
        let ops = vec![
            Op::GlobalLoad { dst: VReg(1), addr: VReg(10), width: Width::B32, offset: 0 },
            Op::VAddF32 {
                dst: VReg(3),
                src0: Operand::VReg(VReg(20)),
                src1: Operand::InlineFloat(1.0),
            },
            Op::WaitVmcnt(0),
            Op::VAddF32 {
                dst: VReg(4),
                src0: Operand::VReg(VReg(1)),
                src1: Operand::InlineFloat(2.0),
            },
            Op::GlobalStore { addr: VReg(10), src: VReg(4), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let mut func = lift_to_ssa(&ops);
        let reordered = schedule_mach_func(&mut func);
        let result = lower_from_ssa(&func);

        // The independent VAddF32(v3) should have been moved before waitcnt
        assert!(reordered > 0, "scheduler should reorder at least 1 op");
        assert_eq!(result.len(), ops.len(), "op count should be preserved");
    }

    #[test]
    fn test_schedule_vcc_safe() {
        // VCmpLtU32 writes VCC, VCndmask reads VCC — must NOT be reordered
        let ops = vec![
            Op::GlobalLoad { dst: VReg(1), addr: VReg(10), width: Width::B32, offset: 0 },
            Op::VCmpLtU32 {
                src0: Operand::VReg(VReg(20)),
                src1: Operand::VReg(VReg(21)),
            },
            Op::VCndmaskB32 {
                dst: VReg(3),
                src_false: Operand::InlineFloat(0.0),
                src_true: Operand::InlineFloat(1.0),
            },
            Op::WaitVmcnt(0),
            Op::VAddF32 {
                dst: VReg(4),
                src0: Operand::VReg(VReg(1)),
                src1: Operand::VReg(VReg(3)),
            },
            Op::GlobalStore { addr: VReg(10), src: VReg(4), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let mut func = lift_to_ssa(&ops);
        schedule_mach_func(&mut func);
        let result = lower_from_ssa(&func);

        // VCmpLtU32 must appear before VCndmaskB32 (VCC dependency)
        let cmp_pos = result.iter().position(|op| matches!(op, Op::VCmpLtU32 { .. })).unwrap();
        let cndmask_pos = result.iter().position(|op| matches!(op, Op::VCndmaskB32 { .. })).unwrap();
        assert!(cmp_pos < cndmask_pos,
            "VCmpLtU32 must remain before VCndmaskB32 (VCC dependency)");
    }

    #[test]
    fn test_schedule_roundtrip_with_branches() {
        // Scheduling should preserve all ops including control flow
        let ops = vec![
            Op::GlobalLoad { dst: VReg(1), addr: VReg(10), width: Width::B32, offset: 0 },
            Op::WaitVmcnt(0),
            Op::SCmpLtU32 { src0: SReg(0), src1: SReg(1) },
            Op::BranchScc1("done".to_string()),
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(0.0) },
            Op::Label("done".to_string()),
            Op::Endpgm,
        ];

        let mut func = lift_to_ssa(&ops);
        schedule_mach_func(&mut func);
        let result = lower_from_ssa(&func);

        assert_eq!(result.len(), ops.len(), "scheduling must preserve all ops");
    }

    // ═══════════════════════════════════════════════════
    //  D4d: Algebraic Simplification tests
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_ssa_algsim_add_zero() {
        // v1 + 0.0 → v_mov v2, v1
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(3.0) },
            Op::VAddF32 { dst: VReg(2), src0: Operand::VReg(VReg(1)), src1: Operand::InlineFloat(0.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];
        let mut func = lift_to_ssa(&ops);
        let n = algebraic_simplify_mach_func(&mut func);
        assert!(n >= 1, "should simplify x+0 → mov");
        let result = lower_from_ssa(&func);
        // The VAddF32 should have become a VMov
        assert!(result.iter().all(|op| !matches!(op, Op::VAddF32 { .. })),
            "VAddF32 should have been simplified away");
    }

    #[test]
    fn test_ssa_algsim_mul_zero() {
        // x * 0 → 0
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(5.0) },
            Op::VMulF32 { dst: VReg(2), src0: Operand::VReg(VReg(1)), src1: Operand::InlineFloat(0.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];
        let mut func = lift_to_ssa(&ops);
        let n = algebraic_simplify_mach_func(&mut func);
        assert!(n >= 1, "should simplify x*0 → 0");
    }

    // ═══════════════════════════════════════════════════
    //  D4e: CSE tests
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_ssa_cse_duplicate() {
        // two identical adds with same inputs → second becomes mov
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(2.0) },
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::VAddF32 { dst: VReg(4), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 0 },
            Op::GlobalStore { addr: VReg(10), src: VReg(4), width: Width::B32, offset: 4 },
            Op::Endpgm,
        ];
        let mut func = lift_to_ssa(&ops);
        let n = cse_mach_func(&mut func);
        assert!(n >= 1, "CSE should eliminate at least 1 duplicate computation");
    }

    // ═══════════════════════════════════════════════════
    //  D4f: Instruction Combine tests
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_ssa_combine_mul_add_to_fma() {
        // mul(a,b) + c → fma(a,b,c) when mul result is single-use
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(2.0) },
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(3.0) },
            Op::VMulF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::VAddF32 { dst: VReg(4), src0: Operand::VReg(VReg(3)), src1: Operand::InlineFloat(1.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(4), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];
        let mut func = lift_to_ssa(&ops);
        let n = instruction_combine_mach_func(&mut func);
        assert!(n >= 1, "should combine mul+add → fma");
        let result = lower_from_ssa(&func);
        assert!(result.iter().any(|op| matches!(op, Op::VFmaF32 { .. })),
            "result should contain an FMA instruction");
    }

    // ═══════════════════════════════════════════════════
    //  D4g: Waitcnt Optimization tests
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_ssa_waitcnt_remove_redundant() {
        // Two consecutive waitcnts with no memory op between → second is redundant
        let ops = vec![
            Op::GlobalLoad { dst: VReg(1), addr: VReg(10), width: Width::B32, offset: 0 },
            Op::WaitVmcnt(0),
            Op::WaitVmcnt(0), // redundant
            Op::VAddF32 { dst: VReg(2), src0: Operand::VReg(VReg(1)), src1: Operand::InlineFloat(1.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];
        let mut func = lift_to_ssa(&ops);
        let n = optimize_waitcnt_mach_func(&mut func);
        assert!(n >= 1, "should remove at least 1 redundant waitcnt");
    }

    // ═══════════════════════════════════════════════════
    //  Full SSA pipeline test
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_ssa_full_pipeline() {
        // Exercise all SSA passes in sequence
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(2.0) },
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(3.0) },
            Op::VMulF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::VAddF32 { dst: VReg(4), src0: Operand::VReg(VReg(3)), src1: Operand::InlineFloat(0.0) }, // algsim: +0
            Op::VMov { dst: VReg(5), src: Operand::VReg(VReg(4)) }, // copy prop target
            Op::VMov { dst: VReg(6), src: Operand::InlineFloat(99.0) }, // dead (DCE)
            Op::GlobalStore { addr: VReg(10), src: VReg(5), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let mut func = lift_to_ssa(&ops);

        // Run all passes
        let cf = constant_fold_mach_func(&mut func);
        let as_ = algebraic_simplify_mach_func(&mut func);
        let cp = copy_propagate_mach_func(&mut func);
        let cse = cse_mach_func(&mut func);
        let ic = instruction_combine_mach_func(&mut func);
        let dce = dce_mach_func(&mut func);
        let wc = optimize_waitcnt_mach_func(&mut func);
        let sched = schedule_mach_func(&mut func);

        let result = lower_from_ssa(&func);
        let total = cf + as_ + cp + cse + ic + dce + wc + sched;

        assert!(total > 0, "full pipeline should optimize something (got {})", total);
        assert!(result.len() <= ops.len(),
            "optimized should be ≤ original ({} > {})", result.len(), ops.len());
        // Must still end with Endpgm
        assert!(matches!(result.last(), Some(Op::Endpgm)));
    }

    // ═══════════════════════════════════════════════════
    //  Phase E1: Live Interval tests
    // ═══════════════════════════════════════════════════

    #[test]
    fn test_live_intervals_basic() {
        // Straight-line code: v1 = const, v2 = const, v3 = v1 + v2, store v3
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },       // seq 0
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(2.0) },       // seq 1
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)),        // seq 2
                          src1: Operand::VReg(VReg(2)) },
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 0 }, // seq 3
            Op::Endpgm,                                                       // seq 4
        ];

        let func = lift_to_ssa(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(3), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);

        // v1: def@0, last_use@2; v2: def@1, last_use@2; v3: def@2, last_use@3
        assert!(intervals.len() >= 3, "expected at least 3 intervals, got {}", intervals.len());

        // Find interval for v1 (defined at seq 0)
        let v1_int = intervals.iter().find(|i| i.vreg == VReg(1)).expect("v1 interval missing");
        assert_eq!(v1_int.def_point, 0, "v1 def_point");
        assert_eq!(v1_int.last_use, 2, "v1 last_use");

        // Find interval for v3 (defined at seq 2)
        let v3_int = intervals.iter().find(|i| i.vreg == VReg(3)).expect("v3 interval missing");
        assert_eq!(v3_int.def_point, 2, "v3 def_point");
        assert_eq!(v3_int.last_use, 3, "v3 last_use");

        // Intervals should be sorted by def_point
        for pair in intervals.windows(2) {
            assert!(pair[0].def_point <= pair[1].def_point,
                "intervals not sorted: def {} > {}", pair[0].def_point, pair[1].def_point);
        }
    }

    #[test]
    fn test_live_intervals_loop() {
        // Code with a loop:
        //   v1 = const
        //   v2 = const
        //   label "loop":
        //     v3 = v1 + v2   (inside loop body)
        //     s_cmp ...
        //     branch_scc1 "loop"
        //   store v3
        //   endpgm
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },       // seq 0
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(2.0) },       // seq 1
            Op::Label("loop".into()),                                         // seq 2
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)),        // seq 3
                          src1: Operand::VReg(VReg(2)) },
            Op::SCmpLtU32 { src0: SReg(10), src1: SReg(11) },               // seq 4
            Op::BranchScc1("loop".into()),                                    // seq 5
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 0 }, // seq 6
            Op::Endpgm,                                                       // seq 7
        ];

        let func = lift_to_ssa(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(3), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);

        // v1 is used inside the loop → its last_use should be extended to loop_end (seq 5)
        let v1_int = intervals.iter().find(|i| i.vreg == VReg(1)).expect("v1 interval");
        assert!(v1_int.last_use >= 5,
            "v1 last_use should be extended to loop end, got {}", v1_int.last_use);

        // v3 is defined inside the loop → its range should also span the loop
        let v3_int = intervals.iter().find(|i| i.vreg == VReg(3)).expect("v3 interval");
        assert!(v3_int.last_use >= 5,
            "v3 last_use should be extended to loop end, got {}", v3_int.last_use);
    }

    #[test]
    fn test_live_intervals_alignment() {
        // WMMA accumulator: 8 consecutive VGPRs with Align8
        let ops = vec![
            // Zero-init 8 VGPRs to simulate WMMA accumulators
            Op::VMov { dst: VReg(8), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(9), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(10), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(11), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(12), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(13), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(14), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(15), src: Operand::InlineInt(0) },
            // Store first and last
            Op::GlobalStore { addr: VReg(20), src: VReg(8), width: Width::B32, offset: 0 },
            Op::GlobalStore { addr: VReg(20), src: VReg(15), width: Width::B32, offset: 4 },
            Op::Endpgm,
        ];

        let func = lift_to_ssa(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(8), count: 8, alignment: Alignment::Align8 },
        ];
        let intervals = compute_live_intervals(&func, &allocs);

        // Check that v8 has Align8 and group_size=8
        let v8_int = intervals.iter().find(|i| i.vreg == VReg(8)).expect("v8 interval");
        assert_eq!(v8_int.alignment, Alignment::Align8, "v8 should have Align8");
        assert_eq!(v8_int.group_size, 8, "v8 should have group_size=8");
        assert_eq!(v8_int.group_index, 0, "v8 should be group_index=0");

        // Check that v15 has the same group_id but group_index=7
        let v15_int = intervals.iter().find(|i| i.vreg == VReg(15)).expect("v15 interval");
        assert_eq!(v15_int.group_id, v8_int.group_id, "v8 and v15 should share group_id");
        assert_eq!(v15_int.group_index, 7, "v15 should be group_index=7");
    }

    #[test]
    fn test_live_intervals_dead_vreg() {
        // v1 is defined but never used → should NOT appear in intervals
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(42.0) }, // dead
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(1.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let func = lift_to_ssa(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);

        // v1 should not appear (no uses)
        let v1_int = intervals.iter().find(|i| i.vreg == VReg(1));
        assert!(v1_int.is_none(), "dead VReg(1) should not have an interval");

        // v2 should appear
        let v2_int = intervals.iter().find(|i| i.vreg == VReg(2));
        assert!(v2_int.is_some(), "live VReg(2) should have an interval");
    }

    /// Regression test: VReg self-update must not be eliminated by DCE.
    ///
    /// Pattern: v10 = v10 + 2048 (stride advance for multi-pass LDS store)
    /// If DCE eliminates this, all LDS stores go to the same address.
    #[test]
    fn test_self_update_dce() {
        let ops: Vec<Op> = vec![
            // addr = base + offset
            Op::VAddU32 {
                dst: VReg(10),
                src0: Operand::VReg(VReg(7)),
                src1: Operand::VReg(VReg(8)),
            },
            // store pass 0
            Op::DsStoreB128 { vaddr: VReg(10), src: VReg(0), offset: 0 },
            // addr += stride (self-update)
            Op::VAddU32 {
                dst: VReg(10),
                src0: Operand::VReg(VReg(10)),
                src1: Operand::InlineInt(2048),
            },
            // store pass 1 (must use UPDATED addr)
            Op::DsStoreB128 { vaddr: VReg(10), src: VReg(4), offset: 0 },
            Op::Endpgm,
        ];

        let mut func = lift_to_ssa(&ops);

        // Verify lift_to_ssa gave distinct MVals
        let all_insts: Vec<usize> = func.blocks.iter()
            .flat_map(|b| b.insts.iter().copied()).collect();
        
        // Inst 0: VAddU32(v10, v7, v8) → defs MVal(a)
        // Inst 1: DsStoreB128(v10, ...) → uses MVal(a)
        // Inst 2: VAddU32(v10, v10, 2048) → defs MVal(b), uses MVal(a)
        // Inst 3: DsStoreB128(v10, ...) → uses MVal(b)
        let inst2 = &func.insts[all_insts[2]];
        let inst3 = &func.insts[all_insts[3]];
        
        eprintln!("Inst 2 (stride add): defs={:?}, uses={:?}", inst2.defs, inst2.uses);
        eprintln!("Inst 3 (store pass1): defs={:?}, uses={:?}", inst3.defs, inst3.uses);
        
        // The store pass 1 should use the SAME MVal as the stride add's def
        assert!(!inst2.defs.is_empty(), "stride add must define an MVal");
        let stride_def_mval = inst2.defs[0];
        let store1_uses_stride = inst3.uses.contains(&stride_def_mval);
        eprintln!("store pass1 uses stride_def MVal({})? {}", stride_def_mval.0, store1_uses_stride);
        assert!(store1_uses_stride,
            "BUG in lift_to_ssa: store pass 1 must use the stride add's MVal! \
             stride_def={:?}, store1_uses={:?}",
            inst2.defs, inst3.uses);

        // Now run DCE
        let dce_removed = dce_mach_func(&mut func);
        let result = lower_from_ssa(&func);

        eprintln!("DCE removed {} instructions", dce_removed);
        for (i, op) in result.iter().enumerate() {
            eprintln!("  [{}] {:?}", i, op);
        }

        // The stride add MUST survive
        let has_stride = result.iter().any(|op| {
            matches!(op, Op::VAddU32 { src1: Operand::InlineInt(2048), .. })
        });
        assert!(has_stride,
            "BUG: v_add_u32(v10, v10, 2048) was eliminated by DCE!");
    }

    /// Test: VReg self-update through full optimize pipeline
    #[test]
    fn test_self_update_full_optimize() {
        let ops: Vec<Op> = vec![
            Op::VAddU32 {
                dst: VReg(10),
                src0: Operand::VReg(VReg(7)),
                src1: Operand::VReg(VReg(8)),
            },
            Op::DsStoreB128 { vaddr: VReg(10), src: VReg(0), offset: 0 },
            Op::VAddU32 {
                dst: VReg(10),
                src0: Operand::VReg(VReg(10)),
                src1: Operand::InlineInt(2048),
            },
            Op::DsStoreB128 { vaddr: VReg(10), src: VReg(4), offset: 0 },
            Op::Endpgm,
        ];

        let (result, stats) = super::super::opt_passes::optimize(ops, &[]);

        eprintln!("=== Full optimize ===");
        eprintln!("Stats: DCE={}, CSE={}, Copy={}, AlgSimp={}",
            stats.dead_ops_removed, stats.cse_eliminated,
            stats.copies_propagated, stats.algebraic_simplified);
        for (i, op) in result.iter().enumerate() {
            eprintln!("  [{}] {:?}", i, op);
        }

        let has_stride = result.iter().any(|op| {
            matches!(op, Op::VAddU32 { src1: Operand::InlineInt(2048), .. })
        });
        assert!(has_stride,
            "BUG: v_add_u32(v10, v10, 2048) eliminated by full optimize!");
    }

    /// Bisect: isolate which pass eliminates the self-update
    #[test]
    fn test_self_update_per_pass_bisect() {
        let make_ops = || vec![
            Op::VAddU32 {
                dst: VReg(10),
                src0: Operand::VReg(VReg(7)),
                src1: Operand::VReg(VReg(8)),
            },
            Op::DsStoreB128 { vaddr: VReg(10), src: VReg(0), offset: 0 },
            Op::VAddU32 {
                dst: VReg(10),
                src0: Operand::VReg(VReg(10)),
                src1: Operand::InlineInt(2048),
            },
            Op::DsStoreB128 { vaddr: VReg(10), src: VReg(4), offset: 0 },
            Op::Endpgm,
        ];

        let has_stride = |ops: &[Op]| -> bool {
            ops.iter().any(|op| {
                matches!(op, Op::VAddU32 { src1: Operand::InlineInt(2048), .. })
            })
        };

        // Phase A: lift → individual passes
        let ops = make_ops();
        let mut func = lift_to_ssa(&ops);
        eprintln!("After lift: stride present = {}", has_stride(&lower_from_ssa(&func)));

        // 1. ConstFold
        constant_fold_mach_func(&mut func);
        eprintln!("After constfold: stride present = {}", has_stride(&lower_from_ssa(&func)));

        // 2. AlgSimp
        algebraic_simplify_mach_func(&mut func);
        eprintln!("After algsimp: stride present = {}", has_stride(&lower_from_ssa(&func)));

        // 3. CopyProp
        copy_propagate_mach_func(&mut func);
        eprintln!("After copyprop: stride present = {}", has_stride(&lower_from_ssa(&func)));

        // 4. CSE (domtree)
        cse_mach_func_domtree(&mut func);
        eprintln!("After CSE: stride present = {}", has_stride(&lower_from_ssa(&func)));

        // 5. Combine
        instruction_combine_mach_func(&mut func);
        eprintln!("After combine: stride present = {}", has_stride(&lower_from_ssa(&func)));

        // 6. LICM
        licm_mach_func(&mut func);
        eprintln!("After LICM: stride present = {}", has_stride(&lower_from_ssa(&func)));

        // Lower to Vec<Op> for Phase B
        let ops = lower_from_ssa(&func);
        eprintln!("After Phase A lower: stride present = {}", has_stride(&ops));

        // Phase C: re-lift
        let mut func2 = lift_to_ssa(&ops);
        eprintln!("After Phase C re-lift: stride present = {}", has_stride(&lower_from_ssa(&func2)));

        // Phase C DCE
        let removed = dce_mach_func(&mut func2);
        let final_ops = lower_from_ssa(&func2);
        eprintln!("After Phase C DCE (removed {}): stride present = {}", removed, has_stride(&final_ops));

        for (i, op) in final_ops.iter().enumerate() {
            eprintln!("  [{}] {:?}", i, op);
        }

        assert!(has_stride(&final_ops),
            "Self-update was eliminated! Check per-pass output above.");
    }
}
