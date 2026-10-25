//! H1-C3: ABTM-symbolic, field-block-aligned contiguous partition refinement.
//!
//! Existing MPI halo transport requires contiguous owned global DOFs. Keep
//! ownership contiguous; use ABTM's deduplicated structural row masks to
//! propose block-boundary moves and the *existing* exact CSR HaloPlan to
//! accept/reject a proposal. No numeric SpMV backend is changed.

use crate::{
    ContiguousPartition, DistributedTopologyError, HaloPlan, PartitionTelemetry, RankId,
    RankLocalCsr, build_contiguous_halo_plans, field_aligned_contiguous_partition,
    partition_telemetry, prepare_rank_local_csrs, validate_field_contiguous_partition,
};
use hybit_matrix::{AbtmDualTopology, Csr32Matrix, FieldAwareDofLayout};

/// Exact (global CSR reference) comparison; not measured MPI communication or time.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldGraphPartitionComparison {
    pub candidate_accepted: bool,
    pub baseline: PartitionTelemetry,
    pub candidate: PartitionTelemetry,
    pub selected: PartitionTelemetry,
    pub topology_nnz: usize,
    pub examined_boundary_candidates: usize,
    pub search_radius_blocks: usize,
}

/// An explicit, metadata-first partition experiment.
///
/// Refine H1-C2's block-aligned *contiguous* partition using the number of
/// directed ABTM structural edges crossing candidate block boundaries.
/// Numerical zeros remain structural, and duplicated entries count once.
/// Each boundary is moved by at most `search_radius_blocks` physical blocks
/// around its H1-C2 starting position. The output preserves at least one block
/// per rank and never exceeds the worse of the baseline maximum rank size or
/// 115% of mean rank DOFs. The exact HaloPlan is recomputed once for the
/// candidate and once for the baseline. The candidate is accepted only if
/// communication_volume, cut_nnz and directional peer relations do not regress
/// and at least one strictly improves. Otherwise the baseline is returned.
///
/// This local heuristic cannot promise a global optimum or speedup, and does
/// not prepare noncontiguous MPI-owned rows.
pub fn field_graph_contiguous_partition(
    matrix: &Csr32Matrix,
    layout: &FieldAwareDofLayout,
    ranks: RankId,
    search_radius_blocks: usize,
) -> Result<(ContiguousPartition, FieldGraphPartitionComparison), DistributedTopologyError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    if matrix.nrows() as u64 != layout.total_dofs() {
        return Err(DistributedTopologyError::MatrixPartitionMismatch {
            matrix_rows: matrix.nrows(),
            global_dofs: layout.total_dofs(),
        });
    }
    let baseline = field_aligned_contiguous_partition(layout, ranks)?;
    let baseline_plans = build_contiguous_halo_plans(matrix, &baseline)?;
    let baseline_stats = partition_telemetry(matrix, &baseline, &baseline_plans)?;
    let blocks = layout.blocks().len();
    let mut boundaries = Vec::with_capacity(blocks + 1);
    boundaries.push(0u64);
    let mut dof_to_block = Vec::with_capacity(matrix.nrows());
    for index in 0..blocks {
        let range = layout.block_range(index).expect("validated field block");
        let count = usize::try_from(range.end - range.start)
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        dof_to_block.resize(dof_to_block.len() + count, index);
        boundaries.push(range.end);
    }
    if dof_to_block.len() != matrix.nrows() {
        return Err(DistributedTopologyError::InvalidPartition(
            "field block index map does not cover matrix rows",
        ));
    }

    // Difference array for exact ABTM structural-edge pressure over block cuts.
    // An arc between blocks a and b crosses each boundary min(a,b)+1 .. max(a,b).
    let topology = AbtmDualTopology::from_csr32(matrix).map_err(|_| {
        DistributedTopologyError::InvalidPartition("cannot prepare ABTM dual topology")
    })?;
    let mut pressure_delta = vec![0i64; blocks + 1];
    for (row, &from) in dof_to_block.iter().enumerate() {
        let top_row = topology.row(row).map_err(|_| {
            DistributedTopologyError::InvalidPartition("invalid ABTM structural row")
        })?;
        for word in top_row.words() {
            let mut bits = word.mask();
            while bits != 0 {
                let col = word.base_col() + bits.trailing_zeros() as usize;
                let to = dof_to_block[col];
                if from != to {
                    let lower = from.min(to) + 1;
                    let upper = from.max(to) + 1;
                    pressure_delta[lower] += 1;
                    pressure_delta[upper] -= 1;
                }
                bits &= bits - 1;
            }
        }
    }
    let mut pressure = vec![0i64; blocks + 1];
    let mut current = 0i64;
    for (index, &delta) in pressure_delta.iter().enumerate() {
        current += delta;
        pressure[index] = current;
    }

    let rank_count = ranks as usize;
    let mut baseline_cuts = Vec::with_capacity(rank_count + 1);
    baseline_cuts.push(0usize);
    for &cut in baseline.offsets().iter().skip(1).take(rank_count - 1) {
        let block = layout
            .locate(cut)
            .ok_or(DistributedTopologyError::InvalidPartition(
                "baseline cut not in DOF layout",
            ))?;
        if block.component != 0 {
            return Err(DistributedTopologyError::InvalidPartition(
                "baseline cut splits a block",
            ));
        }
        baseline_cuts.push(block.block_index);
    }
    baseline_cuts.push(blocks);
    let mut cuts = baseline_cuts.clone();
    let mut max_owned = 0u64;
    for pair in baseline.offsets().windows(2) {
        max_owned = max_owned.max(pair[1] - pair[0]);
    }
    let cap_115 = (u128::from(layout.total_dofs()) * 115).div_ceil(u128::from(ranks) * 100);
    max_owned = max_owned.max(u64::try_from(cap_115).unwrap_or(u64::MAX));
    let mut examined = 0usize;

    // Coordinate descent. Every move retains one whole block per rank and
    // respects the *starting* baseline search radius; it cannot oscillate.
    for _ in 0..3 {
        let mut any_change = false;
        for (i, &initial_cut) in baseline_cuts.iter().enumerate().take(rank_count).skip(1) {
            let min_cut = (cuts[i - 1] + 1).max(initial_cut.saturating_sub(search_radius_blocks));
            let max_cut = (cuts[i + 1] - 1).min(initial_cut.saturating_add(search_radius_blocks));
            let mut winner = cuts[i];
            let target = (u128::from(layout.total_dofs()) * i as u128) / u128::from(ranks);
            for candidate in min_cut..=max_cut {
                let left = boundaries[candidate] - boundaries[cuts[i - 1]];
                let right = boundaries[cuts[i + 1]] - boundaries[candidate];
                if left > max_owned || right > max_owned {
                    continue;
                }
                examined += 1;
                let new_key = (
                    pressure[candidate],
                    u128::from(boundaries[candidate]).abs_diff(target),
                    candidate,
                );
                let old_key = (
                    pressure[winner],
                    u128::from(boundaries[winner]).abs_diff(target),
                    winner,
                );
                if new_key < old_key {
                    winner = candidate;
                }
            }
            if winner != cuts[i] {
                cuts[i] = winner;
                any_change = true;
            }
        }
        if !any_change {
            break;
        }
    }
    let proposed_offsets: Vec<u64> = cuts.iter().map(|&i| boundaries[i]).collect();
    let candidate = ContiguousPartition::from_offsets(proposed_offsets)?;
    validate_field_contiguous_partition(layout, &candidate)?;
    let candidate_plans = build_contiguous_halo_plans(matrix, &candidate)?;
    let candidate_stats = partition_telemetry(matrix, &candidate, &candidate_plans)?;
    let accepted = candidate_stats.communication_volume <= baseline_stats.communication_volume
        && candidate_stats.cut_nnz <= baseline_stats.cut_nnz
        && candidate_stats.directional_peer_relations <= baseline_stats.directional_peer_relations
        && (candidate_stats.communication_volume < baseline_stats.communication_volume
            || candidate_stats.cut_nnz < baseline_stats.cut_nnz
            || candidate_stats.directional_peer_relations
                < baseline_stats.directional_peer_relations);
    let selected = if accepted { candidate } else { baseline };
    let selected_stats = if accepted {
        candidate_stats.clone()
    } else {
        baseline_stats.clone()
    };
    Ok((
        selected,
        FieldGraphPartitionComparison {
            candidate_accepted: accepted,
            baseline: baseline_stats,
            candidate: candidate_stats,
            selected: selected_stats,
            topology_nnz: topology.stats().structural_nnz,
            examined_boundary_candidates: examined,
            search_radius_blocks,
        },
    ))
}

/// Prepared *contiguous* halo for the chosen field-aware graph partition.
/// Uses the existing reciprocal, unique-ghost HaloPlan and rank-local CSR.
#[derive(Clone, Debug)]
pub struct PreparedFieldGraphHalo {
    partition: ContiguousPartition,
    plans: Vec<HaloPlan>,
    locals: Vec<RankLocalCsr>,
    comparison: FieldGraphPartitionComparison,
}

impl PreparedFieldGraphHalo {
    pub fn prepare(
        matrix: &Csr32Matrix,
        layout: &FieldAwareDofLayout,
        ranks: RankId,
        search_radius_blocks: usize,
    ) -> Result<Self, DistributedTopologyError> {
        let (partition, comparison) =
            field_graph_contiguous_partition(matrix, layout, ranks, search_radius_blocks)?;
        let plans = build_contiguous_halo_plans(matrix, &partition)?;
        let locals = prepare_rank_local_csrs(matrix, &plans)?;
        Ok(Self {
            partition,
            plans,
            locals,
            comparison,
        })
    }
    pub fn partition(&self) -> &ContiguousPartition {
        &self.partition
    }
    pub fn plans(&self) -> &[HaloPlan] {
        &self.plans
    }
    pub fn local_operators(&self) -> &[RankLocalCsr] {
        &self.locals
    }
    pub fn comparison(&self) -> &FieldGraphPartitionComparison {
        &self.comparison
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PartitionAssignment, simulate_halo_exchange, validate_field_assignment};
    use hybit_matrix::{DofBlock, DofEntityKind};

    fn fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
        let sizes = [3u32, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1];
        let mut blocks = Vec::new();
        for (i, &components) in sizes.iter().enumerate() {
            blocks.push(DofBlock {
                entity_kind: if i == 10 {
                    DofEntityKind::Multiplier
                } else {
                    DofEntityKind::Node
                },
                entity_id: i as u64 / 2,
                field_id: (i % 2) as u32,
                components,
            });
        }
        let layout = FieldAwareDofLayout::new(blocks).unwrap();
        let n = layout.total_dofs() as usize;
        let mut row_ptr = vec![0u32];
        let mut cols = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            let block = layout.locate(row as u64).unwrap().block_index;
            let start = if block < 5 { 0 } else { 11 };
            let end = if block < 5 { 11 } else { n };
            for col in start..end {
                // Dense structural references (some explicit numeric zeros),
                // but *no* connection between the two physical clusters.
                cols.push(col as u32);
                values.push(if col == row {
                    4.0
                } else if (row + col) % 7 == 0 {
                    0.0
                } else {
                    -0.1
                });
            }
            row_ptr.push(cols.len() as u32);
        }
        (
            layout,
            Csr32Matrix::new(n, n, row_ptr, cols, values).unwrap(),
        )
    }

    #[test]
    fn graph_cut_can_improve_ghost_and_cut_without_splitting_fields() {
        let (layout, matrix) = fixture();
        let baseline = field_aligned_contiguous_partition(&layout, 2).unwrap();
        assert_eq!(baseline.offsets(), &[0, 12, 24]);
        let (best, comparison) = field_graph_contiguous_partition(&matrix, &layout, 2, 3).unwrap();
        assert_eq!(best.offsets(), &[0, 11, 24]);
        assert!(comparison.candidate_accepted);
        assert!(
            comparison.selected.communication_volume < comparison.baseline.communication_volume
        );
        assert!(comparison.selected.cut_nnz < comparison.baseline.cut_nnz);
        validate_field_contiguous_partition(&layout, &best).unwrap();
        let owners = PartitionAssignment::from_contiguous(&best).unwrap();
        assert_eq!(
            validate_field_assignment(&layout, &owners).unwrap().len(),
            12
        );
        let (_, second) = field_graph_contiguous_partition(&matrix, &layout, 2, 3).unwrap();
        assert_eq!(comparison, second);
    }

    #[test]
    fn prepared_graph_halo_spmv_equals_direct_csr_and_deduplicates_ghosts() {
        let (layout, matrix) = fixture();
        let prepared = PreparedFieldGraphHalo::prepare(&matrix, &layout, 2, 3).unwrap();
        let x: Vec<f64> = (0..matrix.nrows())
            .map(|i| (i as f64 * 0.19).sin())
            .collect();
        let reference = matrix.spmv(&x).unwrap();
        let extended = simulate_halo_exchange(&x, prepared.plans()).unwrap();
        for ((plan, local), vector) in prepared
            .plans()
            .iter()
            .zip(prepared.local_operators())
            .zip(&extended)
        {
            let mut result = vec![0.0; plan.owned_len()];
            local.apply_extended(vector, &mut result).unwrap();
            let range = plan.owned_range();
            assert_eq!(result, reference[range.start as usize..range.end as usize]);
            let mut unique = plan.ghost_globals().to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), plan.ghost_len());
        }
        assert_eq!(prepared.comparison().selected.communication_volume, 0);
    }

    #[test]
    fn zero_radius_falls_back_and_invalid_inputs_are_rejected() {
        let (layout, matrix) = fixture();
        let (partition, stats) = field_graph_contiguous_partition(&matrix, &layout, 2, 0).unwrap();
        assert_eq!(partition.offsets(), &[0, 12, 24]);
        assert!(!stats.candidate_accepted);
        assert_eq!(
            stats.selected.communication_volume,
            stats.baseline.communication_volume
        );
        assert!(field_graph_contiguous_partition(&matrix, &layout, 0, 8).is_err());
        assert!(field_graph_contiguous_partition(&matrix, &layout, 13, 8).is_err());
        let wrong = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
        assert!(field_graph_contiguous_partition(&wrong, &layout, 2, 8).is_err());
    }
}
