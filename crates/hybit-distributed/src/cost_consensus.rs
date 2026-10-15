//! H1-C3D opt-in MPI consensus gate. Invoke collectively on every rank
//! *before* any peer-to-peer Halo traffic. This gate checks identical scoped
//! input/decision metadata and compares the complete selected offset vector.

use crate::{
    CalibrationGuardDecision, CostAwarePartitionDecision, CostAwarePartitionKind,
    CostAwarePartitionOptions, ScopedCostPartitionResult, ScopedPartitionCostCalibration,
};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::io;

fn kind_code(value: CostAwarePartitionKind) -> u64 {
    match value {
        CostAwarePartitionKind::Equal => 0,
        CostAwarePartitionKind::Field => 1,
        CostAwarePartitionKind::Graph => 2,
    }
}
fn guard_code(value: CalibrationGuardDecision) -> u64 {
    match value {
        CalibrationGuardDecision::Missing => 0,
        CalibrationGuardDecision::ScopeMismatch => 1,
        CalibrationGuardDecision::Matched => 2,
    }
}
fn decision_code(value: CostAwarePartitionDecision) -> u64 {
    match value {
        CostAwarePartitionDecision::SingleRank => 0,
        CostAwarePartitionDecision::NoExpectedReuse => 1,
        CostAwarePartitionDecision::NoCalibration => 2,
        CostAwarePartitionDecision::ZeroSearchRadius => 3,
        CostAwarePartitionDecision::NoCalibratedSpeedup => 4,
        CostAwarePartitionDecision::UnamortizedPreparation => 5,
        CostAwarePartitionDecision::GraphProposalRejected => 6,
        CostAwarePartitionDecision::GraphNotDistinct => 7,
        CostAwarePartitionDecision::GraphDoesNotDominateBaseline => 8,
        CostAwarePartitionDecision::LoadImbalanceLimit => 9,
        CostAwarePartitionDecision::GraphSelected => 10,
    }
}

/// Collectively compare selected partition and calibration/decision parameters.
///
/// Precondition: all ranks initialized the same MPI_COMM_WORLD and entered this
/// exact collective epoch. Do not call on only the root, and do not start
/// point-to-point Halo exchange before this returns `Ok(())` on every rank.
///
/// Uses exact offsets via Allgather (not merely a hash). Scope digests are
/// noncryptographic; this guards accidental divergence, not hostile peers.
#[allow(clippy::too_many_lines)]
pub fn verify_cost_partition_mpi_consensus(
    selection: &ScopedCostPartitionResult,
    options: CostAwarePartitionOptions,
    scoped: Option<&ScopedPartitionCostCalibration>,
) -> io::Result<()> {
    let world = mpi::topology::SimpleCommunicator::world();
    let size = world.size();
    let id = &selection.identity;
    // All fields have a fixed count, so the first allreduce is safe even if a
    // local partition has a different rank count or offset length.
    let scoped_timings = scoped.map(|value| value.timings);
    let local: [u64; 34] = [
        id.matrix_pattern,
        id.matrix_values,
        id.physical_layout,
        id.environment,
        u64::from(id.ranks),
        id.search_radius_blocks as u64,
        kind_code(id.baseline_kind),
        options.expected_spmv_calls,
        options.search_radius_blocks as u64,
        options.max_owned_dof_imbalance.to_bits(),
        options.max_local_nnz_imbalance.to_bits(),
        options.minimum_net_savings_ns,
        u64::from(scoped.is_some()),
        scoped_timings.map_or(0, |x| x.baseline_spmv_ns),
        scoped_timings.map_or(0, |x| x.graph_spmv_ns),
        scoped_timings.map_or(0, |x| x.graph_prepare_budget_ns),
        guard_code(selection.guard),
        kind_code(selection.report.baseline_kind),
        kind_code(selection.report.selected_kind),
        decision_code(selection.report.decision),
        u64::from(selection.partition.rank_count()),
        selection.partition.global_dofs(),
        selection.partition.offsets().len() as u64,
        u64::from(options.calibration.is_some()),
        options.calibration.map_or(0, |x| x.baseline_spmv_ns),
        options.calibration.map_or(0, |x| x.graph_spmv_ns),
        options.calibration.map_or(0, |x| x.graph_prepare_budget_ns),
        scoped.map_or(0, |x| x.identity.matrix_pattern),
        scoped.map_or(0, |x| x.identity.matrix_values),
        scoped.map_or(0, |x| x.identity.physical_layout),
        scoped.map_or(0, |x| x.identity.environment),
        scoped.map_or(0, |x| u64::from(x.identity.ranks)),
        scoped.map_or(0, |x| x.identity.search_radius_blocks as u64),
        scoped.map_or(0, |x| kind_code(x.identity.baseline_kind)),
    ];
    let mut minimum = [0u64; 34];
    let mut maximum = [0u64; 34];
    world.all_reduce_into(&local[..], &mut minimum[..], SystemOperation::min());
    world.all_reduce_into(&local[..], &mut maximum[..], SystemOperation::max());
    let invalid_rank_count = size < 1 || id.ranks != size as u32;
    if minimum != maximum || invalid_rank_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "MPI calibration identity, options, decision, or offset count differs across ranks",
        ));
    }
    // At this point every rank has an identical offset count; a fixed-size
    // Allgather safely checks exact bytes rather than just a digest.
    let offsets = selection.partition.offsets();
    let count = offsets.len();
    let total = count.checked_mul(size as usize).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "MPI offset gather size overflow",
        )
    })?;
    let mut all_offsets = vec![0u64; total];
    world.all_gather_into(offsets, all_offsets.as_mut_slice());
    if all_offsets
        .chunks_exact(count)
        .any(|slice| slice != offsets)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "MPI selected partition offset vectors differ across ranks",
        ));
    }
    Ok(())
}
