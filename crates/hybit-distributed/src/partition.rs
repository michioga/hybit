use crate::{
    ContiguousPartition, DistributedTopologyError, GlobalDofId, PartitionTelemetry, RankId,
};
use hybit_matrix::{AbtmDualTopology, Csr32Matrix};
use std::collections::{BTreeSet, VecDeque};

const UNASSIGNED: RankId = RankId::MAX;

/// General per-DOF partition ownership.
///
/// Unlike `ContiguousPartition`, this representation permits arbitrary graph
/// partitions such as ABTM, METIS, or Scotch assignments while keeping global
/// DOF identifiers unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionAssignment {
    ranks: RankId,
    owners: Vec<RankId>,
}

impl PartitionAssignment {
    pub fn from_owners(
        ranks: RankId,
        owners: Vec<RankId>,
    ) -> Result<Self, DistributedTopologyError> {
        if ranks == 0 || ranks == RankId::MAX || owners.is_empty() {
            return Err(DistributedTopologyError::InvalidRankCount);
        }
        if owners.len() < ranks as usize {
            return Err(DistributedTopologyError::InvalidRankCount);
        }

        let mut counts = vec![0usize; ranks as usize];
        for &owner in &owners {
            if owner >= ranks {
                return Err(DistributedTopologyError::RankOutOfRange { rank: owner, ranks });
            }
            counts[owner as usize] += 1;
        }

        if counts.contains(&0) {
            return Err(DistributedTopologyError::InvalidPartition(
                "partition assignment contains an empty rank",
            ));
        }

        Ok(Self { ranks, owners })
    }

    pub fn from_contiguous(
        partition: &ContiguousPartition,
    ) -> Result<Self, DistributedTopologyError> {
        let n = usize::try_from(partition.global_dofs())
            .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
        let mut owners = vec![0; n];

        for rank in 0..partition.rank_count() {
            let owned = partition.owned_range(rank)?;
            let start = usize::try_from(owned.start)
                .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
            let end = usize::try_from(owned.end)
                .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?;
            owners[start..end].fill(rank);
        }

        Self::from_owners(partition.rank_count(), owners)
    }

    pub fn rank_count(&self) -> RankId {
        self.ranks
    }

    pub fn global_dofs(&self) -> GlobalDofId {
        self.owners.len() as GlobalDofId
    }

    pub fn owners(&self) -> &[RankId] {
        &self.owners
    }

    pub fn owner_of(&self, global: GlobalDofId) -> Result<RankId, DistributedTopologyError> {
        let index =
            usize::try_from(global).map_err(|_| DistributedTopologyError::GlobalDofOutOfRange {
                global,
                global_dofs: self.global_dofs(),
            })?;

        self.owners
            .get(index)
            .copied()
            .ok_or(DistributedTopologyError::GlobalDofOutOfRange {
                global,
                global_dofs: self.global_dofs(),
            })
    }

    pub fn owned_count(&self, rank: RankId) -> Result<usize, DistributedTopologyError> {
        if rank >= self.ranks {
            return Err(DistributedTopologyError::RankOutOfRange {
                rank,
                ranks: self.ranks,
            });
        }
        Ok(self.owners.iter().filter(|&&owner| owner == rank).count())
    }
}

/// Structural work counters for the first ABTM partition prototype.
///
/// These counters describe topology traversal only. They are intentionally
/// separate from partition quality telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmPartitionStats {
    pub ranks: RankId,
    pub seeds_started: usize,
    pub disconnected_restarts: usize,
    pub frontier_nodes_expanded: usize,
    pub topology_words_visited: usize,
    pub candidate_neighbor_bits: usize,
    pub dual_topology_metadata_bytes: usize,
}

fn node_degree(
    topology: &AbtmDualTopology,
    node: usize,
) -> Result<usize, DistributedTopologyError> {
    let row = topology.row(node).map_err(|_| {
        DistributedTopologyError::InvalidPartition("ABTM row topology lookup failed")
    })?;
    let column = topology.column(node).map_err(|_| {
        DistributedTopologyError::InvalidPartition("ABTM column topology lookup failed")
    })?;

    Ok(row.popcount().saturating_add(column.popcount()))
}

fn choose_seed(owners: &[RankId], degree: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;

    for (node, (&owner, &node_degree)) in owners.iter().zip(degree).enumerate() {
        if owner != UNASSIGNED {
            continue;
        }
        match best {
            None => best = Some((node_degree, node)),
            Some((best_degree, best_node))
                if node_degree > best_degree
                    || (node_degree == best_degree && node < best_node) =>
            {
                best = Some((node_degree, node));
            }
            _ => {}
        }
    }

    best.map(|(_, node)| node)
}

fn balanced_targets(n: usize, ranks: RankId) -> Vec<usize> {
    let ranks_usize = ranks as usize;
    let base = n / ranks_usize;
    let remainder = n % ranks_usize;

    (0..ranks_usize)
        .map(|rank| base + usize::from(rank < remainder))
        .collect()
}

/// Deterministic ABTM dual-topology region-growth partition prototype.
///
/// This is deliberately a first prototype, not yet a multilevel competitor to
/// METIS/Scotch. It establishes:
///
/// - arbitrary per-DOF ownership rather than contiguous ranges;
/// - exact balanced DOF targets;
/// - graph-connected growth when the topology permits it;
/// - deterministic high-degree seeds and tie-breaking;
/// - ABTM row/column word traversal without reading numerical values.
///
/// When one connected component cannot fill a rank target, another unassigned
/// seed is started. The final result always obeys the exact balanced target
/// sizes.
pub fn abtm_region_grow_partition(
    matrix: &Csr32Matrix,
    ranks: RankId,
) -> Result<(PartitionAssignment, AbtmPartitionStats), DistributedTopologyError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    if ranks == 0 || ranks == RankId::MAX || matrix.nrows() == 0 || ranks as usize > matrix.nrows()
    {
        return Err(DistributedTopologyError::InvalidRankCount);
    }

    let topology = AbtmDualTopology::from_csr32(matrix).map_err(|_| {
        DistributedTopologyError::InvalidPartition("ABTM dual topology construction failed")
    })?;
    let topology_stats = topology.stats();

    let mut degree = Vec::with_capacity(matrix.nrows());
    for node in 0..matrix.nrows() {
        degree.push(node_degree(&topology, node)?);
    }

    let targets = balanced_targets(matrix.nrows(), ranks);
    let mut owners = vec![UNASSIGNED; matrix.nrows()];
    let mut stats = AbtmPartitionStats {
        ranks,
        dual_topology_metadata_bytes: topology_stats.total_metadata_bytes(),
        ..AbtmPartitionStats::default()
    };

    for (rank_index, &target) in targets.iter().enumerate() {
        let rank = rank_index as RankId;
        let mut assigned = 0usize;
        let mut first_seed_for_rank = true;
        let mut frontier = VecDeque::new();

        while assigned < target {
            if frontier.is_empty() {
                let seed = choose_seed(&owners, &degree).ok_or(
                    DistributedTopologyError::InvalidPartition(
                        "ABTM partition ran out of unassigned seed nodes",
                    ),
                )?;

                owners[seed] = rank;
                assigned += 1;
                frontier.push_back(seed);
                stats.seeds_started += 1;
                if !first_seed_for_rank {
                    stats.disconnected_restarts += 1;
                }
                first_seed_for_rank = false;

                if assigned == target {
                    continue;
                }
            }

            let Some(node) = frontier.pop_front() else {
                continue;
            };
            stats.frontier_nodes_expanded += 1;

            let row = topology.row(node).map_err(|_| {
                DistributedTopologyError::InvalidPartition(
                    "ABTM row topology lookup failed during partition growth",
                )
            })?;

            for word in row.words() {
                stats.topology_words_visited += 1;
                stats.candidate_neighbor_bits += word.popcount();

                let mut bits = word.mask();
                while bits != 0 && assigned < target {
                    let bit = bits.trailing_zeros() as usize;
                    let neighbor = word.base_col() + bit;
                    bits &= bits - 1;

                    if neighbor < owners.len() && owners[neighbor] == UNASSIGNED {
                        owners[neighbor] = rank;
                        assigned += 1;
                        frontier.push_back(neighbor);
                    }
                }
                if assigned == target {
                    break;
                }
            }

            if assigned == target {
                continue;
            }

            let column = topology.column(node).map_err(|_| {
                DistributedTopologyError::InvalidPartition(
                    "ABTM column topology lookup failed during partition growth",
                )
            })?;

            for word in column.words() {
                stats.topology_words_visited += 1;
                stats.candidate_neighbor_bits += word.popcount();

                let mut bits = word.mask();
                while bits != 0 && assigned < target {
                    let bit = bits.trailing_zeros() as usize;
                    let neighbor = word.base_col() + bit;
                    bits &= bits - 1;

                    if neighbor < owners.len() && owners[neighbor] == UNASSIGNED {
                        owners[neighbor] = rank;
                        assigned += 1;
                        frontier.push_back(neighbor);
                    }
                }
                if assigned == target {
                    break;
                }
            }
        }
    }

    if owners.contains(&UNASSIGNED) {
        return Err(DistributedTopologyError::InvalidPartition(
            "ABTM partition left unassigned DOFs",
        ));
    }

    let assignment = PartitionAssignment::from_owners(ranks, owners)?;

    for (rank, &target) in targets.iter().enumerate() {
        if assignment.owned_count(rank as RankId)? != target {
            return Err(DistributedTopologyError::InvalidPartition(
                "ABTM partition did not preserve balanced target sizes",
            ));
        }
    }

    Ok((assignment, stats))
}

/// Compute partition quality for arbitrary ownership labels.
///
/// This uses the same metric semantics as G8-A2's contiguous/halo telemetry:
///
/// - `cut_nnz`: stored CSR references whose row/column owners differ;
/// - `communication_volume`: unique remote DOF values required by each
///   receiving rank for one halo refresh;
/// - directional peer relations / maximum neighbors;
/// - owned-DOF and local-nnz balance.
pub fn partition_telemetry_assignment(
    matrix: &Csr32Matrix,
    assignment: &PartitionAssignment,
) -> Result<PartitionTelemetry, DistributedTopologyError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    if matrix.nrows() as GlobalDofId != assignment.global_dofs() {
        return Err(DistributedTopologyError::MatrixPartitionMismatch {
            matrix_rows: matrix.nrows(),
            global_dofs: assignment.global_dofs(),
        });
    }

    let ranks = assignment.rank_count() as usize;
    let mut owned_counts = vec![0usize; ranks];
    let mut local_nnz = vec![0usize; ranks];
    let mut remote_globals: Vec<BTreeSet<GlobalDofId>> =
        (0..ranks).map(|_| BTreeSet::new()).collect();
    let mut remote_ranks: Vec<BTreeSet<RankId>> = (0..ranks).map(|_| BTreeSet::new()).collect();

    let mut cut_nnz = 0usize;

    for row in 0..matrix.nrows() {
        let row_rank = assignment.owners()[row];
        let row_rank_index = row_rank as usize;
        owned_counts[row_rank_index] += 1;

        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        local_nnz[row_rank_index] += end - start;

        for &col in &matrix.col_idx()[start..end] {
            let global_col = GlobalDofId::from(col);
            let col_rank = assignment.owner_of(global_col)?;
            if col_rank != row_rank {
                cut_nnz += 1;
                remote_globals[row_rank_index].insert(global_col);
                remote_ranks[row_rank_index].insert(col_rank);
            }
        }
    }

    let communication_volume = remote_globals.iter().map(BTreeSet::len).sum();
    let directional_peer_relations = remote_ranks.iter().map(BTreeSet::len).sum();
    let max_neighbors = remote_ranks.iter().map(BTreeSet::len).max().unwrap_or(0);

    let min_owned_dofs = *owned_counts.iter().min().unwrap_or(&0);
    let max_owned_dofs = *owned_counts.iter().max().unwrap_or(&0);
    let min_local_nnz = *local_nnz.iter().min().unwrap_or(&0);
    let max_local_nnz = *local_nnz.iter().max().unwrap_or(&0);

    let ranks_f64 = ranks as f64;
    let average_owned = matrix.nrows() as f64 / ranks_f64;
    let average_nnz = matrix.nnz() as f64 / ranks_f64;

    let owned_dof_imbalance = if average_owned > 0.0 {
        max_owned_dofs as f64 / average_owned
    } else {
        1.0
    };
    let local_nnz_imbalance = if average_nnz > 0.0 {
        max_local_nnz as f64 / average_nnz
    } else {
        1.0
    };

    Ok(PartitionTelemetry {
        ranks: assignment.rank_count(),
        global_dofs: assignment.global_dofs(),
        global_nnz: matrix.nnz(),
        cut_nnz,
        communication_volume,
        directional_peer_relations,
        max_neighbors,
        min_owned_dofs,
        max_owned_dofs,
        owned_dof_imbalance,
        min_local_nnz,
        max_local_nnz,
        local_nnz_imbalance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_contiguous_halo_plans, partition_telemetry};

    fn chain_matrix(n: usize) -> Csr32Matrix {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in 0..n {
            if row > 0 {
                col_idx.push((row - 1) as u32);
                values.push(-1.0);
            }
            col_idx.push(row as u32);
            values.push(2.0);
            if row + 1 < n {
                col_idx.push((row + 1) as u32);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    fn interleaved_two_chain_matrix() -> Csr32Matrix {
        let n = 8usize;
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];

        for (row, row_cols) in rows.iter_mut().enumerate() {
            row_cols.push(row as u32);

            if row >= 2 {
                row_cols.push((row - 2) as u32);
            }
            if row + 2 < n {
                row_cols.push((row + 2) as u32);
            }

            row_cols.sort_unstable();
        }

        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in rows {
            for col in row {
                col_idx.push(col);
                values.push(if col_idx.len() % 3 == 0 { 2.0 } else { -1.0 });
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn assignment_telemetry_matches_contiguous_g8_a2_metrics() {
        let matrix = chain_matrix(11);
        let contiguous = ContiguousPartition::balanced(11, 3).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &contiguous).unwrap();
        let reference = partition_telemetry(&matrix, &contiguous, &plans).unwrap();

        let assignment = PartitionAssignment::from_contiguous(&contiguous).unwrap();
        let generic = partition_telemetry_assignment(&matrix, &assignment).unwrap();

        assert_eq!(generic, reference);
    }

    #[test]
    fn abtm_region_partition_is_balanced_and_deterministic() {
        let matrix = chain_matrix(17);

        let (a, stats_a) = abtm_region_grow_partition(&matrix, 4).unwrap();
        let (b, stats_b) = abtm_region_grow_partition(&matrix, 4).unwrap();

        assert_eq!(a, b);
        assert_eq!(stats_a, stats_b);
        assert_eq!(a.owned_count(0).unwrap(), 5);
        assert_eq!(a.owned_count(1).unwrap(), 4);
        assert_eq!(a.owned_count(2).unwrap(), 4);
        assert_eq!(a.owned_count(3).unwrap(), 4);
    }

    #[test]
    fn abtm_region_growth_beats_contiguous_cut_on_interleaved_components() {
        let matrix = interleaved_two_chain_matrix();

        let contiguous = ContiguousPartition::balanced(8, 2).unwrap();
        let contiguous_assignment = PartitionAssignment::from_contiguous(&contiguous).unwrap();
        let contiguous_metrics =
            partition_telemetry_assignment(&matrix, &contiguous_assignment).unwrap();

        let (abtm, _) = abtm_region_grow_partition(&matrix, 2).unwrap();
        let abtm_metrics = partition_telemetry_assignment(&matrix, &abtm).unwrap();

        assert!(contiguous_metrics.cut_nnz > 0);
        assert_eq!(abtm_metrics.cut_nnz, 0);
        assert_eq!(abtm_metrics.communication_volume, 0);
        assert!(abtm_metrics.cut_nnz < contiguous_metrics.cut_nnz);
    }

    #[test]
    fn explicit_owner_labels_reject_empty_rank() {
        let error = PartitionAssignment::from_owners(3, vec![0, 0, 1, 1]).unwrap_err();
        assert!(matches!(
            error,
            DistributedTopologyError::InvalidPartition(_)
        ));
    }
}
