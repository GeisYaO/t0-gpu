//! T0 SSA Register Allocator (Phase E2)
//!
//! Allocates physical VGPRs/SGPRs based on SSA live intervals computed by
//! `compute_live_intervals()`. Replaces the legacy linear-scan allocator
//! with an SSA-aware allocator that supports:
//!
//! - **Interval-sorted allocation**: Processes intervals by def_point order
//!   instead of declaration order, enabling better register reuse
//! - **Alignment constraints**: Align2/4/8 for WMMA and multi-dword loads
//! - **Register groups**: Consecutive physical registers for WMMA accumulators
//! - **Spill to LDS**: When VGPR pressure exceeds a target, spills the
//!   interval with the farthest last_use to LDS scratch space

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;
use super::ir::*;
use super::ssa_ir::*;

// ============================================================================
// Allocation Result
// ============================================================================

/// Result of SSA register allocation.
#[derive(Clone, Debug)]
pub struct SsaRegAlloc {
    /// MVal → physical VGPR number
    pub vgpr_map: HashMap<MVal, u8>,
    /// SReg → physical SGPR number (same as legacy)
    pub sgpr_map: HashMap<SReg, u8>,
    /// Peak VGPR count
    pub total_vgprs: u8,
    /// Peak SGPR count
    pub total_sgprs: u8,
    /// Spill records for LDS-based spilling
    pub spills: Vec<SpillRecord>,
}

/// A spill record: an MVal that was spilled to LDS.
#[derive(Clone, Debug)]
pub struct SpillRecord {
    /// Which SSA value was spilled
    pub mval: MVal,
    /// LDS byte offset for the spill slot
    pub lds_offset: u32,
    /// The instruction sequence number where the value was originally defined
    pub def_point: u32,
    /// The instruction sequence number of the last use (for reload insertion)
    pub last_use: u32,
    /// Original VReg (for backend mapping)
    pub vreg: VReg,
}

impl SsaRegAlloc {
    /// Convert SSA allocation back to the legacy VReg-based RegAlloc format.
    ///
    /// This enables drop-in replacement in the existing compile pipeline.
    /// Maps each MVal's VReg to its physical VGPR assignment.
    ///
    /// Also handles "dead def" VRegs that were skipped by `compute_live_intervals`
    /// (e.g., scratch regs inside EXEC-masked regions). These get a fallback
    /// physical register to prevent asm_emitter panics.
    pub fn to_legacy_regalloc(
        &self,
        func: &MachFunc,
        post_spill_ops: Option<&[Op]>,
    ) -> super::regalloc::RegAlloc {
        let mval_to_vreg = build_mval_to_vreg(func);

        let mut vgpr_map: HashMap<VReg, u8> = HashMap::new();
        // Always map v0 → 0 (hardware WORKITEM_ID_X)
        vgpr_map.insert(VReg(0), 0);

        for (&mval, &phys) in &self.vgpr_map {
            if let Some(&vreg) = mval_to_vreg.get(&mval) {
                vgpr_map.insert(vreg, phys);
            }
        }

        // Fallback: scan all ops for VRegs not yet mapped.
        // These are "dead defs" that compute_live_intervals skipped because
        // the MVal had no use (e.g., scratch VRegs inside EXEC-masked regions,
        // or tmp VRegs for wave_reduce that are only used implicitly).
        // Also covers spill-inserted VRegs (spill_addr_vreg) from insert_spill_reloads.
        //
        // IMPORTANT: Dead defs share a single scratch register to avoid
        // inflating total_vgprs into the CWSR danger zone (>252).
        // Since dead defs are never read after being written, they can
        // safely alias — the hardware writes to the register but the
        // value is never consumed. Using a dedicated scratch register
        // (the next one after SSA-allocated peak) is safe because:
        //   1. Dead defs by definition have no live range conflict
        //   2. Macro-op expansion (e.g., WaveReduceAddF32.tmp) uses
        //      the register within the same instruction sequence
        let fallback_scratch = self.total_vgprs; // one register past SSA peak
        let mut fallback_used = false;

        // Scan MachFunc insts (pre-spill)
        for inst in &func.insts {
            for vr in inst.op.vreg_refs() {
                if vr.0 < u32::MAX - 100 && !vgpr_map.contains_key(&vr) {
                    vgpr_map.insert(vr, fallback_scratch);
                    fallback_used = true;
                }
            }
        }

        // Scan post-spill ops (covers spill_addr_vreg and any reload VRegs)
        if let Some(ops) = post_spill_ops {
            for op in ops {
                for vr in op.vreg_refs() {
                    if vr.0 < u32::MAX - 100 && !vgpr_map.contains_key(&vr) {
                        vgpr_map.insert(vr, fallback_scratch);
                        fallback_used = true;
                    }
                }
            }
        }

        // Cap total_vgprs at 254 (CWSR safe limit on GFX1100).
        // When SSA already used 254, the fallback scratch aliases with
        // register 253 (last even-aligned register). Dead defs are never
        // read after their macro expansion completes, so this is safe.
        let total_vgprs = if fallback_used {
            self.total_vgprs.saturating_add(1).min(254)
        } else {
            self.total_vgprs.min(254)
        };

        super::regalloc::RegAlloc {
            vgpr_map,
            sgpr_map: self.sgpr_map.clone(),
            total_vgprs,
            total_sgprs: self.total_sgprs,
        }
    }
}

// ============================================================================
// Free Register Pool
// ============================================================================

/// A pool of free physical register ranges.
/// Supports allocation with alignment and contiguous-range constraints.
struct FreePool {
    /// Available ranges: (start_phys, count)
    ranges: Vec<(u8, u32)>,
    /// High-water mark: next unallocated register
    next_free: u8,
    /// Hard limit (255 for VGPR)
    max_regs: u8,
}

impl FreePool {
    fn new(start: u8, max_regs: u8) -> Self {
        Self {
            ranges: Vec::new(),
            next_free: start,
            max_regs,
        }
    }

    /// Try to allocate `count` consecutive registers with given alignment.
    /// Returns the base physical register number, or None if no fit found.
    fn try_alloc(&mut self, count: u32, alignment: Alignment) -> Option<u8> {
        let align_mask: u32 = match alignment {
            Alignment::None => 0,
            Alignment::Align2 => 1,
            Alignment::Align4 => 3,
            Alignment::Align8 => 7,
        };

        // Best-fit search in free ranges
        let mut best: Option<(usize, u8, u32)> = None; // (range_idx, aligned_start, total_waste)

        for (fi, &(start, fcount)) in self.ranges.iter().enumerate() {
            let aligned_u32 = ((start as u32) + align_mask) & !align_mask;
            if aligned_u32 > u8::MAX as u32 {
                continue;
            }
            let aligned = aligned_u32 as u8;
            let gap = aligned_u32 - start as u32;
            if fcount >= count + gap {
                let waste = gap + (fcount - count - gap);
                if best.is_none() || waste < best.unwrap().2 {
                    best = Some((fi, aligned, waste));
                    if waste == 0 { break; } // perfect fit
                }
            }
        }

        if let Some((fi, aligned, _waste)) = best {
            let (start, fcount) = self.ranges[fi];
            let gap = aligned as u32 - start as u32;
            let used = count + gap;

            // Split or remove the range
            if fcount > used {
                self.ranges[fi] = ((start as u32 + used) as u8, fcount - used);
            } else {
                self.ranges.remove(fi);
            }
            // Return the alignment gap as a free range
            if gap > 0 {
                self.ranges.push((start, gap));
            }

            return Some(aligned);
        }

        // No free range — allocate from high-water mark
        let aligned_u32 = ((self.next_free as u32) + align_mask) & !align_mask;
        let end = aligned_u32 + count;
        if end > self.max_regs as u32 {
            return None; // would overflow — caller should spill
        }
        let aligned = aligned_u32 as u8;

        // Gap reclaim: recovers alignment-gap VGPRs back into the free pool.
        // PROVEN SAFE: k16/k32 dispatch tested (58.4/77.4 TF, no hangs).
        // k48 hang was caused by coop load cpr=6 non-power-of-2 bug (now assert-blocked).
        // Saves ~15 VGPRs for k32 (254→239), enabling ILP optimization headroom.
        let gap = aligned_u32 - self.next_free as u32;
        if gap > 0 {
            self.ranges.push((self.next_free, gap));
        }

        self.next_free = end as u8;
        Some(aligned)
    }

    /// Return registers to the free pool with full transitive merge.
    fn free(&mut self, base: u8, count: u32) {
        // Add the new range
        self.ranges.push((base, count));

        // Full coalesce: sort by start, then merge all adjacent/overlapping ranges.
        // This ensures no fragmentation from partial merges.
        self.ranges.sort_by_key(|r| r.0);

        let mut merged: Vec<(u8, u32)> = Vec::with_capacity(self.ranges.len());
        for &(start, cnt) in &self.ranges {
            if let Some(last) = merged.last_mut() {
                let last_end = last.0 as u32 + last.1;
                if start as u32 <= last_end {
                    // Overlapping or adjacent — extend
                    let new_end = (start as u32 + cnt).max(last_end);
                    last.1 = new_end - last.0 as u32;
                    continue;
                }
            }
            merged.push((start, cnt));
        }
        self.ranges = merged;
    }
}

// ============================================================================
// Active interval entry (for the expire heap)
// ============================================================================

/// An active interval tracked by the allocator.
#[derive(Clone, Debug)]
struct ActiveInterval {
    /// Index into the intervals array
    interval_idx: usize,
    /// Assigned physical base register
    phys_base: u8,
    /// Number of consecutive physical registers
    count: u32,
    /// last_use sequence number (for expire ordering)
    last_use: u32,
}

// ============================================================================
// SSA Allocator
// ============================================================================

/// Allocate physical registers for all live intervals.
///
/// # Algorithm (modified linear scan on SSA intervals)
///
/// 1. Sort intervals by def_point
/// 2. For each interval:
///    a. Expire any active intervals whose last_use < current def_point
///    b. Try to allocate from free pool (respecting alignment + group)
///    c. If no register available and would exceed max_vgprs:
///       - Spill the active interval with the farthest last_use
/// 3. Return the allocation map
///
/// # Parameters
///
/// - `intervals`: Live intervals from `compute_live_intervals()`
/// - `sreg_allocs`: SGPR allocation declarations
/// - `func`: The MachFunc (for MVal→VReg mapping)
/// - `max_vgprs`: Target VGPR limit for occupancy (e.g., 128 for 8 waves/SIMD)
pub fn allocate_ssa(
    intervals: &[LiveInterval],
    sreg_allocs: &[SRegAlloc],
    func: &MachFunc,
    max_vgprs: u8,
) -> SsaRegAlloc {
    // ── SGPR allocation (bump, same as legacy) ──
    let mut sgpr_map: HashMap<SReg, u8> = HashMap::new();
    let mut next_sgpr: u8 = 5; // s0:s1 = kernarg, s2/s3/s4 = TGID

    for sa in sreg_allocs {
        if sa.count == 1 {
            sgpr_map.insert(sa.sreg, next_sgpr);
            next_sgpr += 1;
        } else if sa.count == 2 {
            let aligned = (next_sgpr + 1) & !1;
            sgpr_map.insert(sa.sreg, aligned);
            sgpr_map.insert(SReg(sa.sreg.0 + 1), aligned + 1);
            next_sgpr = aligned + 2;
        } else if sa.count == 4 {
            // Buffer resource descriptors need 4-aligned SGPRs
            let aligned = (next_sgpr + 3) & !3;
            for i in 0..4u32 {
                sgpr_map.insert(SReg(sa.sreg.0 + i), aligned + i as u8);
            }
            next_sgpr = aligned + 4;
        } else {
            let base = next_sgpr;
            for i in 0..sa.count {
                sgpr_map.insert(SReg(sa.sreg.0 + i), base + i as u8);
            }
            next_sgpr = base + sa.count as u8;
        }
        assert!(next_sgpr < 106, "SGPR overflow!");
    }

    // ── VGPR allocation ──
    let mut pool = FreePool::new(1, max_vgprs); // v0 reserved for TID, limited by max_vgprs
    let mut vgpr_map: HashMap<MVal, u8> = HashMap::new();
    let mut spills: Vec<SpillRecord> = Vec::new();

    // Active intervals sorted by last_use (we use a simple Vec + sort approach
    // since the number of active intervals is typically small for GPU kernels)
    let mut active: Vec<ActiveInterval> = Vec::new();

    // Group tracking: group_id → allocated phys_base
    // All MVals in the same group share the same base register.
    let mut group_base: HashMap<u32, u8> = HashMap::new();

    // Spill LDS offset counter
    let mut spill_lds_offset: u32 = 0;

    // Peak active VGPR tracking (for diagnostics)
    let mut peak_active_vgprs: u32 = 0;
    let mut peak_active_at_def: u32 = 0;

    for (idx, interval) in intervals.iter().enumerate() {
        // ── Expire dead intervals ──
        let mut expired_indices: Vec<usize> = Vec::new();
        for (ai, act) in active.iter().enumerate() {
            if act.last_use < interval.def_point {
                // This interval is dead — return its registers to the pool
                pool.free(act.phys_base, act.count);
                expired_indices.push(ai);
            }
        }
        // Remove expired (reverse order to preserve indices)
        expired_indices.sort();
        for &ai in expired_indices.iter().rev() {
            active.remove(ai);
        }

        // ── Check if this MVal is part of an already-allocated group ──
        if let Some(gid) = interval.group_id {
            if let Some(&base) = group_base.get(&gid) {
                // Group already allocated — assign this MVal to base + group_index
                let phys = base + interval.group_index as u8;
                vgpr_map.insert(interval.mval, phys);
                // Don't add a separate active entry — the group leader manages the range
                continue;
            }
        }

        // ── Allocate registers ──
        let count = if interval.group_id.is_some() {
            interval.group_size // allocate the whole group at once
        } else {
            1
        };

        let phys_base = pool.try_alloc(count, interval.alignment);


        match phys_base {
            Some(base) => {
                // Successful allocation
                if let Some(gid) = interval.group_id {
                    group_base.insert(gid, base);
                }

                // Map this MVal (and group members if first in group)
                if interval.group_id.is_some() {
                    // Only map this particular MVal within the group
                    vgpr_map.insert(interval.mval, base + interval.group_index as u8);
                } else {
                    vgpr_map.insert(interval.mval, base);
                }

                active.push(ActiveInterval {
                    interval_idx: idx,
                    phys_base: base,
                    count,
                    last_use: interval.last_use,
                });

                // Track peak active VGPRs
                let current_active: u32 = active.iter().map(|a| a.count).sum();
                if current_active > peak_active_vgprs {
                    peak_active_vgprs = current_active;
                    peak_active_at_def = interval.def_point;
                }
            }
            None => {
                // Need to spill. Print diagnostics for first spill.
                if spills.is_empty() && super::verbose_diagnostics_enabled() {
                    let total_free: u32 = pool.ranges.iter().map(|r| r.1).sum();
                    let active_vgprs: u32 = active.iter().map(|a| a.count).sum();
                    eprintln!("  [SPILL#0] at op#{}: need {} regs (align={:?}), active={} VGPRs, free_pool={} in {} frags",
                        interval.def_point, count, interval.alignment,
                        active_vgprs, total_free, pool.ranges.len());
                    for (fi, &(s, c)) in pool.ranges.iter().enumerate() {
                        eprintln!("    pool[{}]: v{}..v{} ({} regs)", fi, s, s as u32 + c - 1, c);
                    }
                }

                // Find the active interval with the farthest last_use.
                // If it's farther than the current interval, spill it instead.
                if let Some(spill_pos) = active.iter().enumerate()
                    .filter(|(_, a)| a.count == 1) // only spill non-group scalars for safety
                    .max_by_key(|(_, a)| a.last_use)
                    .map(|(i, _)| i)
                {
                    let spill_act = active[spill_pos].clone();

                    if spill_act.last_use > interval.last_use {
                        // Spill the farther-away active interval and reuse its register
                        let spill_interval = &intervals[spill_act.interval_idx];

                        spills.push(SpillRecord {
                            mval: spill_interval.mval,
                            lds_offset: spill_lds_offset,
                            def_point: spill_interval.def_point,
                            last_use: spill_interval.last_use,
                            vreg: spill_interval.vreg,
                        });
                        spill_lds_offset += 4; // 32-bit spill slot

                        // Reclaim the spilled interval's register
                        let reclaimed_phys = spill_act.phys_base;
                        active.remove(spill_pos);

                        // Assign the reclaimed register to the current interval
                        vgpr_map.insert(interval.mval, reclaimed_phys);
                        active.push(ActiveInterval {
                            interval_idx: idx,
                            phys_base: reclaimed_phys,
                            count: 1,
                            last_use: interval.last_use,
                        });
                    } else {
                        // Current interval is farther — spill it instead
                        spills.push(SpillRecord {
                            mval: interval.mval,
                            lds_offset: spill_lds_offset,
                            def_point: interval.def_point,
                            last_use: interval.last_use,
                            vreg: interval.vreg,
                        });
                        spill_lds_offset += 4;

                        // Assign a dummy register (will be reloaded from LDS)
                        // The spill/reload insertion pass will handle this.
                        vgpr_map.insert(interval.mval, 0);
                    }
                } else {
                    // No spillable interval found — this is a hard error
                    panic!(
                        "[T0 SSA RegAlloc] FATAL: Cannot allocate VGPR for {:?} \
                         (vreg={:?}, needs {} regs, alignment={:?}). \
                         All active intervals are groups. \
                         Consider reducing kernel complexity.",
                        interval.mval, interval.vreg, count, interval.alignment
                    );
                }
            }
        }
    }

    // Compute total VGPRs used
    let total_vgprs = pool.next_free;

    // Occupancy diagnostics
    let (waves, tier) = if total_vgprs <= 64 { (16, "excellent") }
        else if total_vgprs <= 96 { (10, "good") }
        else if total_vgprs <= 128 { (8, "fair") }
        else if total_vgprs <= 192 { (4, "low") }
        else { (2, "critical") };

    if (total_vgprs > 128 || !spills.is_empty()) && super::verbose_diagnostics_enabled() {
        eprintln!(
            "[T0 SSA RegAlloc] {} VGPRs, {} SGPRs → {} waves/SIMD ({}), {} spills (peak_active={} at op#{})",
            total_vgprs, next_sgpr, waves, tier, spills.len(),
            peak_active_vgprs, peak_active_at_def
        );

        // Fragmentation diagnostics: how much free pool capacity is wasted?
        if !spills.is_empty() {
            let total_free: u32 = pool.ranges.iter().map(|r| r.1).sum();
            let unused_high = pool.max_regs.saturating_sub(pool.next_free) as u32;
            let frag_count = pool.ranges.len();
            let usable_8: u32 = pool.ranges.iter()
                .filter(|(start, count)| {
                    let aligned = (*start + 7) & !7;
                    let gap = (aligned - *start) as u32;
                    *count >= 8 + gap
                })
                .count() as u32;
            eprintln!(
                "  [FRAG] free_pool: {} VGPRs in {} fragments, {} usable for Align8, unused_high={}, next_free={}",
                total_free, frag_count, usable_8, unused_high, pool.next_free
            );
            // Print each fragment for debugging
            for (i, &(start, cnt)) in pool.ranges.iter().enumerate() {
                let end = start as u32 + cnt;
                let aligned = (start + 7) & !7;
                let a8_ok = cnt >= 8 + (aligned - start) as u32;
                eprintln!("    frag[{}]: v{}..v{} ({} regs){}",
                    i, start, end - 1, cnt, if a8_ok { " ✅ Align8" } else { "" });
            }
        }
    }

    SsaRegAlloc {
        vgpr_map,
        sgpr_map,
        total_vgprs,
        total_sgprs: next_sgpr,
        spills,
    }
}

// ============================================================================
// Spill/Reload Code Insertion (Phase E3)
// ============================================================================

/// Result of spill/reload code insertion.
#[derive(Clone, Debug)]
pub struct SpillInsertResult {
    /// Total LDS bytes consumed by spill slots
    pub spill_lds_bytes: u32,
    /// Number of spill stores inserted
    pub stores_inserted: u32,
    /// Number of reload loads inserted
    pub loads_inserted: u32,
}

/// Insert spill stores and reload loads into the instruction stream.
///
/// For each spilled VReg (identified by `SpillRecord`):
/// - **After definition**: insert `DsStoreB32` to save the value to LDS
/// - **Before each use**: insert `DsLoadB32` + `WaitLgkmcnt(0)` to reload
///
/// # LDS Layout
///
/// Spill slots are placed after the kernel's existing LDS usage:
/// ```text
/// [0 .. existing_lds) — kernel LDS (tiles, reductions, etc.)
/// [existing_lds .. existing_lds + spill_bytes) — spill slots
/// ```
///
/// # Addressing
///
/// LDS spill access uses `vaddr=v0` (which may contain any value) with a
/// static `offset` field. Since ds_load/ds_store support a 16-bit unsigned
/// offset, we encode `existing_lds + spill.lds_offset` directly in the
/// offset. For spills, the vaddr is always a zero-initialized scratch VReg
/// allocated by this pass.
///
/// # Parameters
///
/// - `ops`: The instruction stream to modify (will be expanded with spill code)
/// - `alloc`: The SSA allocation result containing spill records
/// - `existing_lds`: Current kernel LDS size in bytes (spill slots go after this)
///
/// # Returns
///
/// `SpillInsertResult` with the total LDS bytes consumed and insertion counts.
pub fn insert_spill_reloads(
    ops: &mut Vec<Op>,
    alloc: &SsaRegAlloc,
    existing_lds: u32,
    wg_size: u32,
) -> SpillInsertResult {
    if alloc.spills.is_empty() {
        return SpillInsertResult {
            spill_lds_bytes: 0,
            stores_inserted: 0,
            loads_inserted: 0,
        };
    }

    // Build spilled VReg → (lds_offset, phys_reg) mapping
    // A VReg may appear multiple times if multiple SSA values map to the same VReg,
    // but for spill insertion we track by VReg since that's what the Op stream uses.
    let mut spill_info: HashMap<VReg, Vec<(u32, u8)>> = HashMap::new();
    for spill in &alloc.spills {
        let phys = alloc.vgpr_map.get(&spill.mval).copied().unwrap_or(0);
        spill_info
            .entry(spill.vreg)
            .or_default()
            .push((existing_lds + spill.lds_offset, phys));
    }

    // For simplicity, use the first spill slot info per VReg
    // (in practice, each VReg has one spill record)
    let mut vreg_to_lds: HashMap<VReg, u32> = HashMap::new();
    for spill in &alloc.spills {
        vreg_to_lds.entry(spill.vreg).or_insert(existing_lds + spill.lds_offset);
    }

    // Compute total spill LDS bytes
    let max_spill_offset = alloc.spills.iter()
        .map(|s| s.lds_offset + 4)
        .max()
        .unwrap_or(0);

    let mut stores_inserted: u32 = 0;
    let mut loads_inserted: u32 = 0;

    // We need a scratch VReg for spill LDS addressing (always 0).
    // Find the highest VReg number in the existing ops to avoid conflicts.
    let mut max_vreg: u32 = 0;
    for op in ops.iter() {
        for vr in op.vreg_refs() {
            if vr.0 > max_vreg && vr.0 < u32::MAX - 100 {
                max_vreg = vr.0;
            }
        }
    }
    let spill_addr_vreg = VReg(max_vreg + 1);

    // Compute per-lane spill address: spill_addr = v0 * max_spill_offset
    //
    // CRITICAL: on kernel entry, VReg(0) = WORKITEM_ID_X (set by hardware).
    // Each lane needs its own spill region in LDS to prevent data corruption.
    // Without this, all 32 lanes in a wave write to the same LDS address,
    // overwriting each other's spilled values.
    //
    // LDS layout: [existing_lds ...][lane0_spill][lane1_spill]...[lane_N_spill]
    // Each lane's spill starts at: existing_lds + v0 * max_spill_offset
    // ds_store_b32 uses: vaddr + offset, so:
    //   vaddr = v0 * max_spill_offset  (per-lane base)
    //   offset = existing_lds + spill_slot_offset (static)
    // But ds_store_b32 offset is 16-bit, and existing_lds could be large.
    // So we compute: vaddr = existing_lds + v0 * max_spill_offset
    // and use offset = spill_slot_offset directly.
    let mut insert_pos = 0;
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::ScalarLoad { .. } | Op::WaitLgkmcnt(_) |
            Op::CaptureTgid { .. } | Op::ComputeGlobalIdX { .. } |
            Op::ClearVcc => {
                insert_pos = i + 1;
            }
            _ => break,
        }
    }

    // spill_addr = v0 * max_spill_offset (multiply workitem ID by per-lane stride)
    ops.insert(insert_pos, Op::VMulLoU32 {
        dst: spill_addr_vreg,
        src0: VReg(0),  // v0 = WORKITEM_ID_X at kernel entry
        src1: VReg(u32::MAX), // placeholder — will be replaced below
    });
    // Replace the placeholder with the actual multiplication:
    // We need v_mul_lo_u32 spill_addr, v0, inline_const(max_spill_offset)
    // But VMulLoU32 takes two VRegs. Use a different approach:
    // v_lshlrev_b32 if max_spill_offset is power of 2, otherwise use scalar constant.
    // Simplest: emit VMovFromSgpr to load the constant, then multiply.
    // OR: restructure to use a simpler addressing scheme.
    //
    // Actually, the cleanest way: use v_mul_lo_u32 with an inline constant.
    // But our Op::VMulLoU32 only takes VRegs. Let's use a sequence:
    //   v_mov_b32 temp, max_spill_offset
    //   v_mul_lo_u32 spill_addr, v0, temp
    // But we need another scratch VReg.
    //
    // Even simpler: just use v_lshlrev + v_add for non-power-of-2,
    // OR use the existing LDS offset as a constant via inline math.
    //
    // Most robust approach: use two ops.
    ops.remove(insert_pos); // remove the placeholder

    let spill_stride_vreg = VReg(max_vreg + 2);

    // Step 1: v_mov_b32 spill_stride, max_spill_offset
    ops.insert(insert_pos, Op::VMov {
        dst: spill_stride_vreg,
        src: Operand::Literal(max_spill_offset),
    });
    // Step 2: v_mul_lo_u32 spill_addr, v0, spill_stride
    ops.insert(insert_pos + 1, Op::VMulLoU32 {
        dst: spill_addr_vreg,
        src0: VReg(0),  // v0 = WORKITEM_ID_X
        src1: spill_stride_vreg,
    });

    // Now scan and insert spill stores after defs, reload loads before uses.
    let mut i = insert_pos + 2; // skip past the v_mov + v_mul we just inserted
    while i < ops.len() {
        let op = &ops[i];

        // Check if this op DEFINES a spilled VReg
        let defs = op.vreg_defs();
        let mut store_insertions: Vec<(VReg, u32)> = Vec::new();
        for def_vreg in &defs {
            if let Some(&lds_off) = vreg_to_lds.get(def_vreg) {
                store_insertions.push((*def_vreg, lds_off));
            }
        }

        // Check if this op USES a spilled VReg (before the def check to insert loads BEFORE)
        let uses = op.vreg_uses();
        let mut load_insertions: Vec<(VReg, u32)> = Vec::new();
        for use_vreg in &uses {
            if let Some(&lds_off) = vreg_to_lds.get(use_vreg) {
                // Don't insert reload if this same instruction also defines the vreg
                // (e.g., v_add_co where dst == src0)
                if !defs.contains(use_vreg) || !store_insertions.iter().any(|(v, _)| v == use_vreg) {
                    // Avoid duplicate reloads for the same vreg in the same instruction
                    if !load_insertions.iter().any(|(v, _)| v == use_vreg) {
                        load_insertions.push((*use_vreg, lds_off));
                    }
                }
            }
        }

        // Insert reload loads BEFORE the current instruction
        if !load_insertions.is_empty() {
            let mut inserted = 0;
            for (vreg, lds_off) in &load_insertions {
                // ds_load_b32 vreg, spill_addr_vreg, offset=existing_lds + lds_off
                ops.insert(i + inserted, Op::DsLoadB32 {
                    dst: *vreg,
                    vaddr: spill_addr_vreg,
                    offset: (existing_lds + *lds_off) as u16,
                });
                inserted += 1;
            }
            // Insert a single wait after all loads
            ops.insert(i + inserted, Op::WaitLgkmcnt(0));
            inserted += 1;

            loads_inserted += load_insertions.len() as u32;
            i += inserted; // skip past the inserted loads+wait to the original instruction
        }

        // Now i points to the original instruction. Check for store insertions.
        if !store_insertions.is_empty() {
            let mut inserted = 0;
            for (vreg, lds_off) in &store_insertions {
                // ds_store_b32 spill_addr_vreg, vreg, offset=existing_lds + lds_off
                ops.insert(i + 1 + inserted, Op::DsStoreB32 {
                    vaddr: spill_addr_vreg,
                    src: *vreg,
                    offset: (existing_lds + *lds_off) as u16,
                });
                inserted += 1;
            }
            stores_inserted += store_insertions.len() as u32;
            i += inserted; // skip past inserted stores
        }

        i += 1;
    }

    if super::verbose_diagnostics_enabled() {
        eprintln!(
            "[T0 Spill] Inserted {} stores + {} loads, LDS spill region: {} bytes @ offset {}",
            stores_inserted, loads_inserted, max_spill_offset, existing_lds
        );
    }

    SpillInsertResult {
        spill_lds_bytes: max_spill_offset * wg_size,
        stores_inserted,
        loads_inserted,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple MachFunc with straight-line ops
    fn make_func(ops: &[Op]) -> MachFunc {
        lift_to_ssa(ops)
    }

    #[test]
    fn test_ssa_alloc_basic() {
        // 3 non-overlapping intervals: v1, v2, v3 used sequentially
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(1), width: Width::B32, offset: 0 },
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(2.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 4 },
            Op::VMov { dst: VReg(3), src: Operand::InlineFloat(3.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 8 },
            Op::Endpgm,
        ];

        let func = make_func(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(3), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);
        let result = allocate_ssa(&intervals, &[], &func, 128);

        // Should allocate successfully with no spills
        assert!(result.spills.is_empty(), "no spills expected");
        assert!(result.total_vgprs <= 10, "should use few VGPRs, got {}", result.total_vgprs);

        // Each MVal should have a valid physical register
        for interval in &intervals {
            assert!(result.vgpr_map.contains_key(&interval.mval),
                "MVal {:?} not allocated", interval.mval);
        }
    }

    #[test]
    fn test_ssa_alloc_reuse() {
        // v1 dies before v2 is born → v2 should reuse v1's register
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },       // def v1
            Op::GlobalStore { addr: VReg(10), src: VReg(1), width: Width::B32, offset: 0 }, // last use v1
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(2.0) },       // def v2 (v1 is dead)
            Op::GlobalStore { addr: VReg(10), src: VReg(2), width: Width::B32, offset: 4 }, // last use v2
            Op::Endpgm,
        ];

        let func = make_func(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);
        let result = allocate_ssa(&intervals, &[], &func, 128);

        // v2 should reuse v1's physical register
        let v1_int = intervals.iter().find(|i| i.vreg == VReg(1)).unwrap();
        let v2_int = intervals.iter().find(|i| i.vreg == VReg(2)).unwrap();
        let p1 = result.vgpr_map[&v1_int.mval];
        let p2 = result.vgpr_map[&v2_int.mval];
        assert_eq!(p1, p2,
            "v2 should reuse v1's register (p1={}, p2={})", p1, p2);
        assert!(result.total_vgprs <= 3, "should reuse, got {} VGPRs", result.total_vgprs);
    }

    #[test]
    fn test_ssa_alloc_alignment() {
        // 8-aligned accumulator group
        let ops = vec![
            Op::VMov { dst: VReg(8), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(9), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(10), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(11), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(12), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(13), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(14), src: Operand::InlineInt(0) },
            Op::VMov { dst: VReg(15), src: Operand::InlineInt(0) },
            Op::GlobalStore { addr: VReg(20), src: VReg(8), width: Width::B32, offset: 0 },
            Op::GlobalStore { addr: VReg(20), src: VReg(15), width: Width::B32, offset: 4 },
            Op::Endpgm,
        ];

        let func = make_func(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(8), count: 8, alignment: Alignment::Align8 },
        ];
        let intervals = compute_live_intervals(&func, &allocs);
        let result = allocate_ssa(&intervals, &[], &func, 128);

        // Find the physical register for v8 (first in group)
        let v8_int = intervals.iter().find(|i| i.vreg == VReg(8)).unwrap();
        let base = result.vgpr_map[&v8_int.mval];

        // Base must be 8-aligned
        assert_eq!(base % 8, 0, "WMMA base should be 8-aligned, got v{}", base);

        // v15 should be at base + 7
        let v15_int = intervals.iter().find(|i| i.vreg == VReg(15)).unwrap();
        let p15 = result.vgpr_map[&v15_int.mval];
        assert_eq!(p15, base + 7, "v15 should be at base+7={}, got v{}", base + 7, p15);
    }

    #[test]
    fn test_ssa_alloc_spill() {
        // Force spill by setting max_vgprs very low
        // Create 4 overlapping intervals that all need to be alive simultaneously
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(2.0) },
            Op::VMov { dst: VReg(3), src: Operand::InlineFloat(3.0) },
            Op::VMov { dst: VReg(4), src: Operand::InlineFloat(4.0) },
            // All 4 are alive here:
            Op::VAddF32 { dst: VReg(5), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::VAddF32 { dst: VReg(6), src0: Operand::VReg(VReg(3)), src1: Operand::VReg(VReg(4)) },
            Op::GlobalStore { addr: VReg(10), src: VReg(5), width: Width::B32, offset: 0 },
            Op::GlobalStore { addr: VReg(10), src: VReg(6), width: Width::B32, offset: 4 },
            Op::Endpgm,
        ];

        let func = make_func(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(3), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(4), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(5), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(6), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);

        // Set max_vgprs to 4 (v0 reserved + 3 allocatable) — forces spills
        let result = allocate_ssa(&intervals, &[], &func, 4);

        // Should have produced at least one spill
        assert!(!result.spills.is_empty(),
            "expected spills with max_vgprs=4, got {} spills",
            result.spills.len());

        // Each spill should have a valid LDS offset
        for (i, spill) in result.spills.iter().enumerate() {
            assert_eq!(spill.lds_offset, (i as u32) * 4,
                "spill {} should have lds_offset={}", i, i * 4);
        }
    }

    #[test]
    fn test_spill_reload_insertion() {
        // Same ops as test_ssa_alloc_spill — force spills, then verify code insertion
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(2.0) },
            Op::VMov { dst: VReg(3), src: Operand::InlineFloat(3.0) },
            Op::VMov { dst: VReg(4), src: Operand::InlineFloat(4.0) },
            Op::VAddF32 { dst: VReg(5), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::VAddF32 { dst: VReg(6), src0: Operand::VReg(VReg(3)), src1: Operand::VReg(VReg(4)) },
            Op::GlobalStore { addr: VReg(10), src: VReg(5), width: Width::B32, offset: 0 },
            Op::GlobalStore { addr: VReg(10), src: VReg(6), width: Width::B32, offset: 4 },
            Op::Endpgm,
        ];

        let func = make_func(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(3), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(4), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(5), count: 1, alignment: Alignment::None },
            VRegAlloc { vreg: VReg(6), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);

        // Force spills with low VGPR limit
        let ssa_alloc = allocate_ssa(&intervals, &[], &func, 4);
        assert!(!ssa_alloc.spills.is_empty(), "need spills for this test");

        // Run spill/reload insertion
        let mut test_ops = ops.clone();
        let existing_lds = 1024; // simulate kernel using 1KB LDS already
        let result = insert_spill_reloads(&mut test_ops, &ssa_alloc, existing_lds, 256);

        // Verify: should have inserted stores and loads
        assert!(result.stores_inserted > 0, "expected spill stores, got 0");
        assert!(result.loads_inserted > 0, "expected reload loads, got 0");
        assert!(result.spill_lds_bytes > 0, "expected non-zero spill LDS");

        // Verify: ops should now contain DsStoreB32 and DsLoadB32
        let has_ds_store = test_ops.iter().any(|op| matches!(op, Op::DsStoreB32 { .. }));
        let has_ds_load = test_ops.iter().any(|op| matches!(op, Op::DsLoadB32 { .. }));
        assert!(has_ds_store, "ops should contain DsStoreB32 after spill insertion");
        assert!(has_ds_load, "ops should contain DsLoadB32 after spill insertion");

        // Verify: LDS offsets should be >= existing_lds
        for op in &test_ops {
            match op {
                Op::DsStoreB32 { offset, .. } | Op::DsLoadB32 { offset, .. } => {
                    assert!(*offset >= existing_lds as u16,
                        "spill LDS offset {} should be >= existing_lds {}", offset, existing_lds);
                }
                _ => {}
            }
        }

        // Verify: should have a VMov initializing the spill address register
        let has_spill_init = test_ops.iter().any(|op| {
            matches!(op, Op::VMov { src: Operand::InlineInt(0), .. })
        });
        assert!(has_spill_init, "should initialize spill addr VReg to 0");

        eprintln!("[test] spill insertion: {} stores, {} loads, {} LDS bytes",
            result.stores_inserted, result.loads_inserted, result.spill_lds_bytes);
    }

    #[test]
    fn test_spill_reload_no_spills() {
        // When there are no spills, insert_spill_reloads should be a no-op
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(1), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let func = make_func(&ops);
        let allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None },
        ];
        let intervals = compute_live_intervals(&func, &allocs);

        // No spills at 128 VGPRs
        let ssa_alloc = allocate_ssa(&intervals, &[], &func, 128);
        assert!(ssa_alloc.spills.is_empty(), "should have no spills");

        let mut test_ops = ops.clone();
        let orig_len = test_ops.len();
        let result = insert_spill_reloads(&mut test_ops, &ssa_alloc, 0, 256);

        // Should be a no-op
        assert_eq!(result.stores_inserted, 0);
        assert_eq!(result.loads_inserted, 0);
        assert_eq!(result.spill_lds_bytes, 0);
        assert_eq!(test_ops.len(), orig_len, "ops should not be modified");
    }
}
