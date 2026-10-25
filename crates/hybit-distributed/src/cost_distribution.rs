//! H1-C3E: opt-in root-owned cost-aware decision, broadcast to MPI ranks.
//!
//! Only MPI rank 0 needs full CSR and scoped timing evidence. Every rank must
//! provide the identical global field layout, options and environment tag.
//! Collective input/selection errors are resolved before Halo peer exchanges.
//! This does NOT provide distributed matrix ingestion or automatic solver routing.

use crate::{
    CalibrationGuardDecision, ContiguousPartition, CostAwarePartitionDecision,
    CostAwarePartitionKind, CostAwarePartitionOptions, DistributedTopologyError,
    PartitionCalibrationIdentity, ScopedCostPartitionResult, ScopedPartitionCostCalibration,
    cost_aware_contiguous_partition_scoped, validate_field_contiguous_partition,
};
use hybit_matrix::{Csr32Matrix, FieldAwareDofLayout};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::io;

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn digest(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
fn mix_u64(h: u64, v: u64) -> u64 {
    digest(h, &v.to_le_bytes())
}
fn layout_digest(layout: &FieldAwareDofLayout) -> u64 {
    let mut h = digest(FNV_BASIS, b"HYBIT-H1C3D-LAYOUT-v1");
    h = mix_u64(h, layout.total_dofs());
    h = mix_u64(h, layout.blocks().len() as u64);
    for block in layout.blocks() {
        h = mix_u64(h, u64::from(block.entity_kind as u32));
        h = mix_u64(h, block.entity_id);
        h = mix_u64(h, u64::from(block.field_id));
        h = mix_u64(h, u64::from(block.components));
    }
    h
}
fn environment_digest(tag: &str) -> u64 {
    let h = digest(FNV_BASIS, b"HYBIT-H1C3D-ENVIRONMENT-v1");
    let h = mix_u64(h, tag.len() as u64);
    digest(h, tag.as_bytes())
}

fn decode_kind(code: u64) -> Option<CostAwarePartitionKind> {
    match code {
        0 => Some(CostAwarePartitionKind::Equal),
        1 => Some(CostAwarePartitionKind::Field),
        2 => Some(CostAwarePartitionKind::Graph),
        _ => None,
    }
}
fn decode_guard(code: u64) -> Option<CalibrationGuardDecision> {
    match code {
        0 => Some(CalibrationGuardDecision::Missing),
        1 => Some(CalibrationGuardDecision::ScopeMismatch),
        2 => Some(CalibrationGuardDecision::Matched),
        _ => None,
    }
}
fn decode_decision(code: u64) -> Option<CostAwarePartitionDecision> {
    match code {
        0 => Some(CostAwarePartitionDecision::SingleRank),
        1 => Some(CostAwarePartitionDecision::NoExpectedReuse),
        2 => Some(CostAwarePartitionDecision::NoCalibration),
        3 => Some(CostAwarePartitionDecision::ZeroSearchRadius),
        4 => Some(CostAwarePartitionDecision::NoCalibratedSpeedup),
        5 => Some(CostAwarePartitionDecision::UnamortizedPreparation),
        6 => Some(CostAwarePartitionDecision::GraphProposalRejected),
        7 => Some(CostAwarePartitionDecision::GraphNotDistinct),
        8 => Some(CostAwarePartitionDecision::GraphDoesNotDominateBaseline),
        9 => Some(CostAwarePartitionDecision::LoadImbalanceLimit),
        10 => Some(CostAwarePartitionDecision::GraphSelected),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct RootMpiCostSelection {
    pub partition: ContiguousPartition,
    pub selected_kind: CostAwarePartitionKind,
    pub baseline_kind: CostAwarePartitionKind,
    pub decision: CostAwarePartitionDecision,
    pub calibration_guard: CalibrationGuardDecision,
    pub root_identity: PartitionCalibrationIdentity,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Collectively distribute rank-0's *already attempted* scoped selection.
///
/// Rank 0 must pass `Some(&Result<...>)` even on selector failure; peers pass
/// `None`. Do NOT use `?` to early-return on rank 0 before this collective.
/// The root alone owns full CSR and calibrations. All ranks hold an identical
/// global `FieldAwareDofLayout`, options and environment tag. Non-root ranks
/// need only their owned CSR rows *after* this returns successfully.
///
/// The whole offset vector is Bcast, then reconstructed and checked against
/// field boundaries on *all* ranks before any Halo exchange. Error handling
/// is collective for the checked conditions, not arbitrary rank failure.
#[allow(clippy::too_many_lines)]
pub fn broadcast_root_cost_selection_mpi(
    root_result: Option<&Result<ScopedCostPartitionResult, DistributedTopologyError>>,
    layout: &FieldAwareDofLayout,
    options: CostAwarePartitionOptions,
    environment_tag: &str,
) -> io::Result<RootMpiCostSelection> {
    let world = mpi::topology::SimpleCommunicator::world();
    let rank = world.rank();
    let size = world.size();
    let root = world.process_at_rank(0);
    let is_root = rank == 0;
    let valid_tag = !environment_tag.is_empty()
        && environment_tag.len() <= 256
        && !environment_tag.chars().any(char::is_control);
    let valid_options = options.max_owned_dof_imbalance.is_finite()
        && options.max_owned_dof_imbalance >= 1.0
        && options.max_local_nnz_imbalance.is_finite()
        && options.max_local_nnz_imbalance >= 1.0;
    let local_invalid = size < 1
        || !valid_tag
        || !valid_options
        || (is_root != root_result.is_some())
        || layout.blocks().len() < size as usize;
    let mut invalid_count = 0u64;
    world.all_reduce_into(
        &u64::from(local_invalid),
        &mut invalid_count,
        SystemOperation::sum(),
    );
    if invalid_count != 0 {
        return Err(invalid("MPI root selector collective preflight failed"));
    }

    // Fixed-size collective BEFORE variable-length broadcast. A different
    // option, layout or environment tag cannot silently reuse root evidence.
    let timings = options.calibration;
    let keys: [u64; 13] = [
        layout_digest(layout),
        environment_digest(environment_tag),
        layout.total_dofs(),
        layout.blocks().len() as u64,
        options.expected_spmv_calls,
        options.search_radius_blocks as u64,
        options.max_owned_dof_imbalance.to_bits(),
        options.max_local_nnz_imbalance.to_bits(),
        options.minimum_net_savings_ns,
        u64::from(timings.is_some()),
        timings.map_or(0, |t| t.baseline_spmv_ns),
        timings.map_or(0, |t| t.graph_spmv_ns),
        timings.map_or(0, |t| t.graph_prepare_budget_ns),
    ];
    let mut mins = [0u64; 13];
    let mut maxs = [0u64; 13];
    world.all_reduce_into(&keys[..], &mut mins[..], SystemOperation::min());
    world.all_reduce_into(&keys[..], &mut maxs[..], SystemOperation::max());
    if mins != maxs {
        return Err(invalid(
            "MPI root selector options, layout or environment mismatch",
        ));
    }

    let mut root_failed = if is_root {
        u64::from(!root_result.is_some_and(Result::is_ok))
    } else {
        0
    };
    root.broadcast_into(&mut root_failed);
    if root_failed != 0 {
        return Err(invalid("MPI root cost partition selection failed"));
    }

    // Header is root-owned and a fixed size. No rank chooses policy locally.
    let mut header = [0u64; 12];
    if is_root {
        let selection = root_result
            .expect("root checked above")
            .as_ref()
            .expect("successful selection checked above");
        let id = &selection.identity;
        header = [
            selection.partition.global_dofs(),
            u64::from(selection.partition.rank_count()),
            selection.partition.offsets().len() as u64,
            selection.report.selected_kind as u64,
            selection.guard as u64,
            selection.report.decision as u64,
            id.matrix_pattern,
            id.matrix_values,
            id.physical_layout,
            id.environment,
            id.search_radius_blocks as u64,
            id.baseline_kind as u64,
        ];
    }
    root.broadcast_into(&mut header[..]);
    let header_ok = header[0] == layout.total_dofs()
        && header[1] == size as u64
        && header[2] == size as u64 + 1
        && header[8] == keys[0]
        && header[9] == keys[1]
        && header[10] == options.search_radius_blocks as u64
        && decode_kind(header[3]).is_some()
        && decode_kind(header[11]).is_some()
        && decode_guard(header[4]).is_some()
        && decode_decision(header[5]).is_some();
    let mut header_failures = 0u64;
    world.all_reduce_into(
        &u64::from(!header_ok),
        &mut header_failures,
        SystemOperation::sum(),
    );
    if header_failures != 0 {
        return Err(invalid("MPI broadcast partition header failed validation"));
    }

    // Header is validated on every rank; Bcast count is now fixed.
    let mut offsets = vec![0u64; size as usize + 1];
    if is_root {
        offsets.copy_from_slice(
            root_result
                .expect("root checked above")
                .as_ref()
                .expect("selection checked above")
                .partition
                .offsets(),
        );
    }
    root.broadcast_into(offsets.as_mut_slice());
    let local_partition = ContiguousPartition::from_offsets(offsets).and_then(|partition| {
        validate_field_contiguous_partition(layout, &partition)?;
        Ok(partition)
    });
    let mut failures = 0u64;
    world.all_reduce_into(
        &u64::from(local_partition.is_err()),
        &mut failures,
        SystemOperation::sum(),
    );
    if failures != 0 {
        return Err(invalid(
            "MPI broadcast ownership splits a field block or is invalid",
        ));
    }
    Ok(RootMpiCostSelection {
        partition: local_partition.expect("all-rank validation succeeded"),
        selected_kind: decode_kind(header[3]).expect("header validated"),
        baseline_kind: decode_kind(header[11]).expect("header validated"),
        decision: decode_decision(header[5]).expect("header validated"),
        calibration_guard: decode_guard(header[4]).expect("header validated"),
        root_identity: PartitionCalibrationIdentity {
            matrix_pattern: header[6],
            matrix_values: header[7],
            physical_layout: header[8],
            environment: header[9],
            ranks: size as u32,
            search_radius_blocks: header[10] as usize,
            baseline_kind: decode_kind(header[11]).expect("header validated"),
        },
    })
}

/// Rank 0 alone evaluates `cost_aware_contiguous_partition_scoped` using full
/// CSR and supplied scoped evidence. Workers do NOT construct ABTM or CSR.
/// Every process must call this routine in the same collective epoch.
pub fn select_and_broadcast_cost_partition_mpi(
    root_matrix: Option<&Csr32Matrix>,
    layout: &FieldAwareDofLayout,
    options: CostAwarePartitionOptions,
    environment_tag: &str,
    root_calibration: Option<&ScopedPartitionCostCalibration>,
) -> io::Result<RootMpiCostSelection> {
    let world = mpi::topology::SimpleCommunicator::world();
    let local_result = if world.rank() == 0 {
        Some(match root_matrix {
            Some(matrix) => cost_aware_contiguous_partition_scoped(
                matrix,
                layout,
                world.size() as u32,
                options,
                environment_tag,
                root_calibration,
            ),
            None => Err(DistributedTopologyError::InvalidPartition(
                "rank 0 requires a complete CSR matrix to select ownership",
            )),
        })
    } else {
        None
    };
    broadcast_root_cost_selection_mpi(local_result.as_ref(), layout, options, environment_tag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hybit_matrix::{DofBlock, DofEntityKind};

    #[test]
    fn layout_digest_changes_with_field_metadata() {
        let mk = |field_id| {
            FieldAwareDofLayout::new(vec![DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: 1,
                field_id,
                components: 3,
            }])
            .unwrap()
        };
        assert_ne!(layout_digest(&mk(0)), layout_digest(&mk(1)));
        assert_eq!(layout_digest(&mk(0)), layout_digest(&mk(0)));
    }

    #[test]
    fn environment_digest_is_deterministic() {
        assert_eq!(environment_digest("msmpi"), environment_digest("msmpi"));
        assert_ne!(environment_digest("msmpi"), environment_digest("other"));
    }
}
