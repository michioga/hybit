//! H1-C2: field-block-aligned ownership plugged into the existing contiguous
//! HaloPlan / rank-local CSR implementation. MPI transport is unchanged.
//!
//! This is an explicit preparation API, NOT an automatic repartitioner or a
//! promise of improved communication. Every block has a single owner; different
//! fields of one physical entity may have different owners.

use crate::{
    ContiguousPartition, DistributedTopologyError, HaloPlan, PartitionAssignment,
    PartitionTelemetry, RankId, RankLocalCsr, build_contiguous_halo_plans, partition_telemetry,
    prepare_rank_local_csrs,
};
use hybit_matrix::{Csr32Matrix, FieldAwareDofLayout, field_block_owners};

/// Validate an existing (possibly noncontiguous) owner assignment against
/// variable-size physical field blocks. Returns one owner per block.
/// Does NOT turn a graph partition into a contiguous HaloPlan.
pub fn validate_field_assignment(
    layout: &FieldAwareDofLayout,
    assignment: &PartitionAssignment,
) -> Result<Vec<RankId>, DistributedTopologyError> {
    let expected = usize::try_from(layout.total_dofs())
        .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
    if assignment.owners().len() != expected {
        return Err(DistributedTopologyError::VectorLengthMismatch {
            expected,
            actual: assignment.owners().len(),
        });
    }
    field_block_owners(layout, assignment.owners(), assignment.rank_count()).map_err(|_| {
        DistributedTopologyError::InvalidPartition(
            "an ownership assignment splits a field block or uses an invalid rank",
        )
    })
}

/// Validate a contiguous partition without allocating a per-DOF owner vector.
/// Each internal rank boundary must be the START of some physical field block.
pub fn validate_field_contiguous_partition(
    layout: &FieldAwareDofLayout,
    partition: &ContiguousPartition,
) -> Result<(), DistributedTopologyError> {
    if layout.total_dofs() != partition.global_dofs() {
        return Err(DistributedTopologyError::InvalidPartition(
            "partition DOF count differs from field layout",
        ));
    }
    for &cut in &partition.offsets()[1..partition.offsets().len() - 1] {
        if !layout
            .locate(cut)
            .is_some_and(|address| address.component == 0)
        {
            return Err(DistributedTopologyError::InvalidPartition(
                "contiguous rank boundary splits a physical field block",
            ));
        }
    }
    Ok(())
}

/// Choose deterministic contiguous, **block-aligned** ownership ranges.
/// Boundaries approximate uniform DOF targets; the candidate on either side
/// of each target is chosen with at least one whole block per remaining rank.
/// Ties choose the earlier boundary. This is not graph/nnz optimized.
pub fn field_aligned_contiguous_partition(
    layout: &FieldAwareDofLayout,
    ranks: RankId,
) -> Result<ContiguousPartition, DistributedTopologyError> {
    let blocks = layout.blocks().len();
    if ranks == 0 || (ranks as usize) > blocks {
        return Err(DistributedTopologyError::InvalidRankCount);
    }
    let mut boundaries = Vec::with_capacity(blocks + 1);
    boundaries.push(0u64);
    for index in 0..blocks {
        let range = layout
            .block_range(index)
            .expect("validated DOF block index");
        boundaries.push(range.end);
    }
    let mut offsets = Vec::with_capacity(ranks as usize + 1);
    offsets.push(0u64);
    let mut previous_block = 0usize;
    for cut in 1..ranks as usize {
        let min_block = previous_block + 1;
        let max_block = blocks - (ranks as usize - cut);
        let target = ((u128::from(layout.total_dofs()) * cut as u128) / u128::from(ranks)) as u64;
        let insertion = boundaries.partition_point(|&offset| offset < target);
        let upper = insertion.clamp(min_block, max_block);
        let lower = insertion.saturating_sub(1).clamp(min_block, max_block);
        let chosen = if boundaries[lower].abs_diff(target) <= boundaries[upper].abs_diff(target) {
            lower
        } else {
            upper
        };
        offsets.push(boundaries[chosen]);
        previous_block = chosen;
    }
    offsets.push(layout.total_dofs());
    let partition = ContiguousPartition::from_offsets(offsets)?;
    validate_field_contiguous_partition(layout, &partition)?;
    Ok(partition)
}

/// Prepared reference plan for block-aligned ownership and current CSR values.
/// Halo lists use the existing unique-ghost / reciprocal-peer algorithm.
/// No MPI calls are made; the matching `mpi_input::prepare_owned_rows` accepts
/// this object's `partition()` on real MPI ranks.
#[derive(Clone, Debug)]
pub struct PreparedFieldAwareHalo {
    partition: ContiguousPartition,
    plans: Vec<HaloPlan>,
    local: Vec<RankLocalCsr>,
    telemetry: PartitionTelemetry,
}

impl PreparedFieldAwareHalo {
    pub fn prepare(
        matrix: &Csr32Matrix,
        layout: &FieldAwareDofLayout,
        ranks: RankId,
    ) -> Result<Self, DistributedTopologyError> {
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
        let partition = field_aligned_contiguous_partition(layout, ranks)?;
        let plans = build_contiguous_halo_plans(matrix, &partition)?;
        let local = prepare_rank_local_csrs(matrix, &plans)?;
        let telemetry = partition_telemetry(matrix, &partition, &plans)?;
        Ok(Self {
            partition,
            plans,
            local,
            telemetry,
        })
    }

    pub fn partition(&self) -> &ContiguousPartition {
        &self.partition
    }
    pub fn plans(&self) -> &[HaloPlan] {
        &self.plans
    }
    pub fn local_operators(&self) -> &[RankLocalCsr] {
        &self.local
    }
    pub fn telemetry(&self) -> &PartitionTelemetry {
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContiguousPartition, simulate_halo_exchange};
    use hybit_matrix::{DofBlock, DofEntityKind};

    fn layout() -> FieldAwareDofLayout {
        let kinds = [
            (DofEntityKind::Node, 1, 1, 3),
            (DofEntityKind::Node, 1, 2, 1), // same node, other field
            (DofEntityKind::Node, 2, 1, 6),
            (DofEntityKind::Node, 3, 2, 5),
            (DofEntityKind::Multiplier, 9, 8, 3),
            (DofEntityKind::Node, 4, 3, 1),
            (DofEntityKind::Node, 5, 1, 3),
            (DofEntityKind::Constraint, 2, 8, 2),
        ];
        FieldAwareDofLayout::new(
            kinds
                .iter()
                .map(|&(entity_kind, entity_id, field_id, components)| DofBlock {
                    entity_kind,
                    entity_id,
                    field_id,
                    components,
                })
                .collect(),
        )
        .unwrap()
    }

    fn sample_matrix(n: usize) -> Csr32Matrix {
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            // Directed remote links, repeated remote columns across owned rows,
            // and a stored zero which still requires a symbolic ghost.
            col_idx.push(row as u32);
            values.push(3.0 + row as f64 / 50.0);
            col_idx.push(((row + 3) % n) as u32);
            values.push(-0.5);
            col_idx.push(((row + 5) % n) as u32);
            values.push(if row % 3 == 0 { 0.0 } else { 0.25 });
            row_ptr.push(col_idx.len() as u32);
        }
        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn field_aligned_partition_is_deterministic_and_never_splits_blocks() {
        let layout = layout();
        assert_eq!(layout.total_dofs(), 24);
        for ranks in [1, 2, 3, 4, 8] {
            let first = field_aligned_contiguous_partition(&layout, ranks).unwrap();
            let second = field_aligned_contiguous_partition(&layout, ranks).unwrap();
            assert_eq!(first, second);
            validate_field_contiguous_partition(&layout, &first).unwrap();
            let owners = PartitionAssignment::from_contiguous(&first).unwrap();
            assert_eq!(
                validate_field_assignment(&layout, &owners).unwrap().len(),
                8
            );
            assert_eq!(first.offsets().len(), ranks as usize + 1);
        }
        assert!(field_aligned_contiguous_partition(&layout, 0).is_err());
        assert!(field_aligned_contiguous_partition(&layout, 9).is_err());
        let split = ContiguousPartition::from_offsets(vec![0, 2, 24]).unwrap();
        assert!(validate_field_contiguous_partition(&layout, &split).is_err());
        let assignment = PartitionAssignment::from_contiguous(&split).unwrap();
        assert!(validate_field_assignment(&layout, &assignment).is_err());
        // Separate fields on node 1 can be owned by separate ranks.
        let valid = ContiguousPartition::from_offsets(vec![0, 3, 24]).unwrap();
        assert!(validate_field_contiguous_partition(&layout, &valid).is_ok());
    }

    #[test]
    fn noncontiguous_block_owners_are_validated_without_halo_conversion() {
        let layout = layout();
        let mut owners = vec![0u32; layout.total_dofs() as usize];
        for index in 0..layout.blocks().len() {
            let range = layout.block_range(index).unwrap();
            owners[range.start as usize..range.end as usize].fill((index % 2) as u32);
        }
        let assignment = PartitionAssignment::from_owners(2, owners).unwrap();
        assert_eq!(
            validate_field_assignment(&layout, &assignment).unwrap(),
            vec![0, 1, 0, 1, 0, 1, 0, 1]
        );
        // General ownership is intentionally NOT fed into contiguous HaloPlan.
    }

    #[test]
    fn packed_ghosts_reciprocal_peers_and_spmv_equal_csr() {
        let layout = layout();
        let a = sample_matrix(layout.total_dofs() as usize);
        for ranks in [1, 2, 4, 8] {
            let prepared = PreparedFieldAwareHalo::prepare(&a, &layout, ranks).unwrap();
            let plans = prepared.plans();
            let telemetry = prepared.telemetry();
            assert_eq!(
                telemetry.communication_volume,
                plans.iter().map(HaloPlan::ghost_len).sum()
            );
            assert_eq!(plans.len(), ranks as usize);
            for plan in plans {
                let mut sorted = plan.ghost_globals().to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), plan.ghost_len());
                for peer in plan.peers() {
                    let reciprocal = plans[peer.peer() as usize].peer(plan.rank()).unwrap();
                    assert_eq!(peer.recv_globals(), reciprocal.send_globals());
                    assert_eq!(peer.send_globals(), reciprocal.recv_globals());
                }
            }
            for trial in 0..3 {
                let x: Vec<f64> = (0..a.nrows())
                    .map(|index| ((index + 1 + trial) as f64 * 0.031).sin())
                    .collect();
                let reference = a.spmv(&x).unwrap();
                let extended = simulate_halo_exchange(&x, plans).unwrap();
                let mut actual = vec![0.0; x.len()];
                for ((plan, operator), local_x) in
                    plans.iter().zip(prepared.local_operators()).zip(&extended)
                {
                    let mut y = vec![0.0; plan.owned_len()];
                    operator.apply_extended(local_x, &mut y).unwrap();
                    let owned = plan.owned_range();
                    actual[owned.start as usize..owned.end as usize].copy_from_slice(&y);
                }
                assert_eq!(actual, reference);
            }
        }
    }

    #[test]
    fn invalid_matrix_and_assignment_are_rejected() {
        let layout = layout();
        let wrong = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
        assert!(PreparedFieldAwareHalo::prepare(&wrong, &layout, 2).is_err());
        let rectangular = Csr32Matrix::new(1, 2, vec![0, 1], vec![1], vec![1.0]).unwrap();
        assert!(PreparedFieldAwareHalo::prepare(&rectangular, &layout, 2).is_err());
        let assignment = PartitionAssignment::from_owners(2, vec![0, 1]).unwrap();
        assert!(validate_field_assignment(&layout, &assignment).is_err());
        let a = sample_matrix(layout.total_dofs() as usize);
        assert!(PreparedFieldAwareHalo::prepare(&a, &layout, 9).is_err());
    }
}
