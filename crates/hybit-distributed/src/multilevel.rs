use crate::{
    partition_telemetry_assignment, DistributedTopologyError, GlobalDofId, PartitionAssignment,
    PartitionTelemetry, RankId,
};
use hybit_matrix::{AbtmDualTopology, Csr32Matrix};
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::time::Instant;

const UNASSIGNED: RankId = RankId::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbtmMultilevelOptions {
    /// Maximum number of pair-coarsening levels.
    pub max_levels: usize,
    /// Stop coarsening once the graph has at most this many vertices per rank.
    pub coarse_vertices_per_rank: usize,
    /// Maximum fine-DOF load above the exact average, in per-mille.
    ///
    /// 30 corresponds to the 3% tolerance used by the first G8-A6 prototype.
    pub imbalance_per_mille: u32,
    /// Greedy cut-refinement passes after every uncoarsening step.
    pub refinement_passes: usize,
}

impl Default for AbtmMultilevelOptions {
    fn default() -> Self {
        Self {
            max_levels: 12,
            coarse_vertices_per_rank: 64,
            imbalance_per_mille: 30,
            refinement_passes: 2,
        }
    }
}

impl AbtmMultilevelOptions {
    fn validate(self) -> Result<(), DistributedTopologyError> {
        if self.max_levels == 0
            || self.coarse_vertices_per_rank == 0
            || self.refinement_passes == 0
            || self.imbalance_per_mille > 500
        {
            return Err(DistributedTopologyError::InvalidPartition(
                "invalid G8-A6 multilevel options",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmMultilevelStats {
    pub levels_built: usize,
    pub coarsest_vertices: usize,
    pub matched_pairs: usize,
    pub singleton_aggregates: usize,
    pub coarse_restarts: usize,
    pub refinement_moves: usize,
    pub final_cut_nnz: usize,
    pub final_communication_volume: usize,
}

#[derive(Clone, Debug)]
struct HierarchyLevel {
    matrix: Csr32Matrix,
    weights: Vec<usize>,
    fine_to_coarse: Vec<usize>,
}

fn invalid(message: &'static str) -> DistributedTopologyError {
    DistributedTopologyError::InvalidPartition(message)
}

fn build_undirected_adjacency_sorted(
    matrix: &Csr32Matrix,
) -> Result<Vec<Vec<usize>>, DistributedTopologyError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }

    let topology =
        AbtmDualTopology::from_csr32(matrix).map_err(|_| invalid("ABTM dual topology failed"))?;
    let mut adjacency = Vec::with_capacity(matrix.nrows());

    for node in 0..matrix.nrows() {
        let mut neighbors = Vec::new();

        let row = topology
            .row(node)
            .map_err(|_| invalid("ABTM row lookup failed"))?;
        for word in row.words() {
            let mut bits = word.mask();
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let neighbor = word.base_col() + bit;
                if neighbor < matrix.nrows() && neighbor != node {
                    neighbors.push(neighbor);
                }
            }
        }

        let column = topology
            .column(node)
            .map_err(|_| invalid("ABTM column lookup failed"))?;
        for word in column.words() {
            let mut bits = word.mask();
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let neighbor = word.base_col() + bit;
                if neighbor < matrix.nrows() && neighbor != node {
                    neighbors.push(neighbor);
                }
            }
        }

        neighbors.sort_unstable();
        neighbors.dedup();
        adjacency.push(neighbors);
    }

    Ok(adjacency)
}

// F12_MERGED_ABTM_V1: alternate sorted-union extractor. Both ABTM views
// enumerate bitmap words in increasing column order. Merge and unique the
// streams instead of appending, sorting and deduplicating each row.
fn build_undirected_adjacency_merged(
    matrix: &Csr32Matrix,
) -> Result<Vec<Vec<usize>>, DistributedTopologyError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }

    let topology =
        AbtmDualTopology::from_csr32(matrix).map_err(|_| invalid("ABTM dual topology failed"))?;
    let mut adjacency = Vec::with_capacity(matrix.nrows());

    for node in 0..matrix.nrows() {
        let row = topology
            .row(node)
            .map_err(|_| invalid("ABTM row lookup failed"))?;
        let column = topology
            .column(node)
            .map_err(|_| invalid("ABTM column lookup failed"))?;

        let mut from_row = row.words().flat_map(|word| {
            let base = word.base_col();
            let mut bits = word.mask();
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let col = base + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(col)
            })
        });
        let mut from_column = column.words().flat_map(|word| {
            let base = word.base_col();
            let mut bits = word.mask();
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let col = base + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(col)
            })
        });

        let mut next_row = from_row.next();
        let mut next_column = from_column.next();
        let mut neighbors = Vec::new();
        while next_row.is_some() || next_column.is_some() {
            let next = match (next_row, next_column) {
                (Some(a), Some(b)) if a < b => {
                    next_row = from_row.next();
                    a
                }
                (Some(a), Some(b)) if a > b => {
                    next_column = from_column.next();
                    b
                }
                (Some(a), Some(_)) => {
                    next_row = from_row.next();
                    next_column = from_column.next();
                    a
                }
                (Some(a), None) => {
                    next_row = from_row.next();
                    a
                }
                (None, Some(b)) => {
                    next_column = from_column.next();
                    b
                }
                (None, None) => break,
            };
            if next != node && neighbors.last().copied() != Some(next) {
                neighbors.push(next);
            }
        }
        adjacency.push(neighbors);
    }
    Ok(adjacency)
}

fn build_undirected_adjacency(
    matrix: &Csr32Matrix,
) -> Result<Vec<Vec<usize>>, DistributedTopologyError> {
    if std::env::var_os("HYBIT_A6_F12_MERGE").is_some() {
        build_undirected_adjacency_merged(matrix)
    } else {
        build_undirected_adjacency_sorted(matrix)
    }
}
#[cfg(test)]
fn common_neighbor_count(a: &[usize], b: &[usize]) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut count = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }

    count
}

fn matrix_from_adjacency(
    adjacency: &[Vec<usize>],
) -> Result<Csr32Matrix, DistributedTopologyError> {
    let mut row_ptr = Vec::with_capacity(adjacency.len() + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0u32);

    for neighbors in adjacency {
        for &neighbor in neighbors {
            col_idx.push(
                u32::try_from(neighbor)
                    .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?,
            );
        }
        row_ptr.push(
            u32::try_from(col_idx.len())
                .map_err(|_| DistributedTopologyError::LocalIndexOverflow)?,
        );
    }

    let values = vec![1.0; col_idx.len()];
    Csr32Matrix::new(adjacency.len(), adjacency.len(), row_ptr, col_idx, values)
        .map_err(|_| invalid("coarse CSR construction failed"))
}

type CoarsenResult = (Csr32Matrix, Vec<usize>, Vec<usize>, usize, usize);
type CoarsenCachedResult = (CoarsenResult, Vec<Vec<usize>>);

fn coarsen_once_cached(
    matrix: &Csr32Matrix,
    weights: &[usize],
    adjacency: &[Vec<usize>],
) -> Result<CoarsenCachedResult, DistributedTopologyError> {
    if matrix.nrows() != weights.len() {
        return Err(invalid("multilevel matrix/weight dimension mismatch"));
    }

    if adjacency.len() != matrix.nrows() {
        return Err(invalid("cached coarsening adjacency length mismatch"));
    }
    // A6_F10_COARSEN_STAGES_V1: opt-in subphase measurements only.
    let f10_enabled = std::env::var_os("HYBIT_A6_F10_PHASE").is_some();
    let order_started = Instant::now();
    let n = matrix.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&node| (Reverse(adjacency[node].len()), node));
    let order_ms = order_started.elapsed().as_secs_f64() * 1.0e3;
    let pairing_started = Instant::now();

    let mut fine_to_coarse = vec![usize::MAX; n];
    // F11_STAMPED_AFFINITY_V1: one mark vector reused for every unmatched node.
    // The current node ID is a unique stamp, so there is no epoch overflow.
    let mut neighbor_stamps = vec![usize::MAX; n];
    let mut coarse_weights = Vec::new();
    let mut matched_pairs = 0usize;
    let mut singleton_aggregates = 0usize;

    for node in order {
        if fine_to_coarse[node] != usize::MAX {
            continue;
        }

        // Mark each neighbor of the current node once, instead of walking
        // both sorted adjacency lists for every candidate match.
        for &neighbor in &adjacency[node] {
            neighbor_stamps[neighbor] = node;
        }
        let mut best_neighbor: Option<(usize, usize, usize)> = None;

        for &neighbor in &adjacency[node] {
            if fine_to_coarse[neighbor] != usize::MAX {
                continue;
            }

            // Exactly the same common-neighbor count as sorted intersection:
            // adjacency lists are sorted and deduplicated by construction.
            let affinity = adjacency[neighbor]
                .iter()
                .filter(|&&shared| neighbor_stamps[shared] == node)
                .count();
            let candidate = (
                affinity,
                usize::MAX - adjacency[neighbor].len(),
                usize::MAX - neighbor,
            );

            let replace = match best_neighbor {
                None => true,
                Some(current) => candidate > current,
            };
            if replace {
                best_neighbor = Some(candidate);
            }
        }

        let coarse = coarse_weights.len();
        fine_to_coarse[node] = coarse;
        let mut aggregate_weight = weights[node];

        if let Some((_, _, reverse_neighbor)) = best_neighbor {
            let neighbor = usize::MAX - reverse_neighbor;
            fine_to_coarse[neighbor] = coarse;
            aggregate_weight = aggregate_weight.saturating_add(weights[neighbor]);
            matched_pairs += 1;
        } else {
            singleton_aggregates += 1;
        }

        coarse_weights.push(aggregate_weight);
    }

    let pair_ms = pairing_started.elapsed().as_secs_f64() * 1.0e3;
    let scan_started = Instant::now();
    let mut coarse_adjacency = vec![Vec::<usize>::new(); coarse_weights.len()];

    for node in 0..n {
        let coarse_node = fine_to_coarse[node];
        for &neighbor in &adjacency[node] {
            let coarse_neighbor = fine_to_coarse[neighbor];
            if coarse_neighbor != coarse_node {
                coarse_adjacency[coarse_node].push(coarse_neighbor);
            }
        }
    }

    let scan_ms = scan_started.elapsed().as_secs_f64() * 1.0e3;
    let dedup_started = Instant::now();
    for neighbors in &mut coarse_adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let dedup_ms = dedup_started.elapsed().as_secs_f64() * 1.0e3;
    let csr_started = Instant::now();
    let coarse_matrix = matrix_from_adjacency(&coarse_adjacency)?;
    let csr_ms = csr_started.elapsed().as_secs_f64() * 1.0e3;
    if f10_enabled {
        eprintln!(
            "A6 F10 stage order_ms={order_ms:.3} pair_ms={pair_ms:.3} scan_ms={scan_ms:.3} dedup_ms={dedup_ms:.3} csr_ms={csr_ms:.3}"
        );
    }
    Ok((
        (
            coarse_matrix,
            coarse_weights,
            fine_to_coarse,
            matched_pairs,
            singleton_aggregates,
        ),
        coarse_adjacency,
    ))
}

#[cfg(test)]
fn coarsen_once(
    matrix: &Csr32Matrix,
    weights: &[usize],
) -> Result<CoarsenResult, DistributedTopologyError> {
    let adjacency = build_undirected_adjacency(matrix)?;
    let (result, _) = coarsen_once_cached(matrix, weights, &adjacency)?;
    Ok(result)
}

fn load_cap(total_weight: usize, ranks: RankId, imbalance_per_mille: u32) -> usize {
    let denominator = ranks as usize * 1000usize;
    let numerator = total_weight.saturating_mul(1000usize + imbalance_per_mille as usize);
    numerator / denominator
}

fn minimum_load(total_weight: usize, ranks: RankId, imbalance_per_mille: u32) -> usize {
    let numerator =
        total_weight.saturating_mul(1000usize.saturating_sub(imbalance_per_mille as usize));
    numerator / (ranks as usize * 1000usize)
}

fn farthest_seeds(
    adjacency: &[Vec<usize>],
    ranks: RankId,
) -> Result<Vec<usize>, DistributedTopologyError> {
    if adjacency.len() < ranks as usize || ranks == 0 {
        return Err(DistributedTopologyError::InvalidRankCount);
    }

    let first = adjacency
        .iter()
        .enumerate()
        .max_by(|(node_a, a), (node_b, b)| a.len().cmp(&b.len()).then_with(|| node_b.cmp(node_a)))
        .map(|(node, _)| node)
        .ok_or(invalid("cannot seed empty coarse graph"))?;

    let mut seeds = vec![first];

    while seeds.len() < ranks as usize {
        let mut distance = vec![usize::MAX; adjacency.len()];
        let mut queue = VecDeque::new();

        for &seed in &seeds {
            distance[seed] = 0;
            queue.push_back(seed);
        }

        while let Some(node) = queue.pop_front() {
            let next = distance[node].saturating_add(1);
            for &neighbor in &adjacency[node] {
                if distance[neighbor] == usize::MAX {
                    distance[neighbor] = next;
                    queue.push_back(neighbor);
                }
            }
        }

        let mut best: Option<(bool, usize, usize, usize)> = None;
        for node in 0..adjacency.len() {
            if seeds.contains(&node) {
                continue;
            }
            let unreachable = distance[node] == usize::MAX;
            let candidate = (
                unreachable,
                if unreachable { 0 } else { distance[node] },
                adjacency[node].len(),
                usize::MAX - node,
            );
            let replace = match best {
                None => true,
                Some(current) => candidate > current,
            };
            if replace {
                best = Some(candidate);
            }
        }

        let (_, _, _, reverse_node) = best.ok_or(invalid("cannot choose enough coarse seeds"))?;
        seeds.push(usize::MAX - reverse_node);
    }

    Ok(seeds)
}

fn weighted_multisource_partition(
    matrix: &Csr32Matrix,
    weights: &[usize],
    ranks: RankId,
    imbalance_per_mille: u32,
) -> Result<(PartitionAssignment, usize), DistributedTopologyError> {
    if matrix.nrows() != weights.len() {
        return Err(invalid("multilevel matrix/weight dimension mismatch"));
    }

    let adjacency = build_undirected_adjacency(matrix)?;
    let total_weight: usize = weights.iter().copied().sum();
    let cap = load_cap(total_weight, ranks, imbalance_per_mille);

    if weights.iter().any(|&weight| weight > cap) {
        return Err(invalid("coarse aggregate exceeds balance cap"));
    }

    let seeds = farthest_seeds(&adjacency, ranks)?;
    let mut owners = vec![UNASSIGNED; matrix.nrows()];
    let mut loads = vec![0usize; ranks as usize];
    let mut frontiers: Vec<VecDeque<usize>> = (0..ranks).map(|_| VecDeque::new()).collect();

    for (rank, &seed) in seeds.iter().enumerate() {
        owners[seed] = rank as RankId;
        loads[rank] = weights[seed];
    }

    for (rank, &seed) in seeds.iter().enumerate() {
        for &neighbor in &adjacency[seed] {
            if owners[neighbor] == UNASSIGNED {
                frontiers[rank].push_back(neighbor);
            }
        }
    }

    let mut assigned = seeds.len();
    let mut restarts = 0usize;

    while assigned < matrix.nrows() {
        let mut changed = false;

        for rank in 0..ranks as usize {
            let mut deferred = VecDeque::new();
            let mut claimed = None;

            while let Some(candidate) = frontiers[rank].pop_front() {
                if owners[candidate] != UNASSIGNED {
                    continue;
                }
                if loads[rank].saturating_add(weights[candidate]) <= cap {
                    claimed = Some(candidate);
                    break;
                }
                deferred.push_back(candidate);
            }
            frontiers[rank].append(&mut deferred);

            if claimed.is_none() {
                let mut best: Option<(usize, usize)> = None;
                for node in 0..matrix.nrows() {
                    if owners[node] != UNASSIGNED || loads[rank].saturating_add(weights[node]) > cap
                    {
                        continue;
                    }
                    let candidate = (adjacency[node].len(), usize::MAX - node);
                    let replace = match best {
                        None => true,
                        Some(current) => candidate > current,
                    };
                    if replace {
                        best = Some(candidate);
                    }
                }

                if let Some((_, reverse_node)) = best {
                    claimed = Some(usize::MAX - reverse_node);
                    restarts += 1;
                }
            }

            if let Some(node) = claimed {
                owners[node] = rank as RankId;
                loads[rank] = loads[rank].saturating_add(weights[node]);
                assigned += 1;
                changed = true;

                for &neighbor in &adjacency[node] {
                    if owners[neighbor] == UNASSIGNED {
                        frontiers[rank].push_back(neighbor);
                    }
                }
            }
        }

        if !changed {
            // Capacity exists globally, but coarse aggregate granularity can
            // leave the round-robin frontier unable to fit the last item.
            // Assign the next item to the lightest rank that can still accept it.
            let mut fallback: Option<(usize, usize, usize)> = None;

            for node in 0..matrix.nrows() {
                if owners[node] != UNASSIGNED {
                    continue;
                }
                for rank in 0..ranks as usize {
                    if loads[rank].saturating_add(weights[node]) > cap {
                        continue;
                    }
                    let candidate = (
                        usize::MAX - loads[rank],
                        adjacency[node].len(),
                        usize::MAX - node,
                    );
                    let replace = match fallback {
                        None => true,
                        Some((best_rank, best_node, best_score)) => {
                            let current = (
                                usize::MAX - loads[best_rank],
                                adjacency[best_node].len(),
                                best_score,
                            );
                            candidate > current
                        }
                    };
                    if replace {
                        fallback = Some((rank, node, usize::MAX - node));
                    }
                }
            }

            let (rank, node, _) = fallback.ok_or(invalid(
                "weighted coarse partition cannot satisfy balance cap",
            ))?;
            owners[node] = rank as RankId;
            loads[rank] = loads[rank].saturating_add(weights[node]);
            assigned += 1;
            restarts += 1;

            for &neighbor in &adjacency[node] {
                if owners[neighbor] == UNASSIGNED {
                    frontiers[rank].push_back(neighbor);
                }
            }
        }
    }

    PartitionAssignment::from_owners(ranks, owners).map(|assignment| (assignment, restarts))
}

fn refine_by_cut(
    matrix: &Csr32Matrix,
    weights: &[usize],
    assignment: &mut PartitionAssignment,
    imbalance_per_mille: u32,
    passes: usize,
) -> Result<usize, DistributedTopologyError> {
    let adjacency = build_undirected_adjacency(matrix)?;
    let ranks = assignment.rank_count() as usize;
    let total_weight: usize = weights.iter().copied().sum();
    let cap = load_cap(total_weight, assignment.rank_count(), imbalance_per_mille);
    let floor = minimum_load(total_weight, assignment.rank_count(), imbalance_per_mille);

    let mut owners = assignment.owners().to_vec();
    let mut loads = vec![0usize; ranks];

    for (node, &owner) in owners.iter().enumerate() {
        loads[owner as usize] = loads[owner as usize].saturating_add(weights[node]);
    }

    let mut moves = 0usize;

    for _ in 0..passes {
        let mut changed = false;

        for node in 0..matrix.nrows() {
            let source = owners[node] as usize;
            let weight = weights[node];

            if loads[source] < weight || loads[source] - weight < floor {
                continue;
            }

            let mut links = vec![0usize; ranks];
            for &neighbor in &adjacency[node] {
                links[owners[neighbor] as usize] += 1;
            }

            let internal = links[source];
            let mut best_target: Option<(usize, usize)> = None;

            for target in 0..ranks {
                if target == source
                    || links[target] <= internal
                    || loads[target].saturating_add(weight) > cap
                {
                    continue;
                }

                let gain = links[target] - internal;
                let candidate = (gain, usize::MAX - target);
                let replace = match best_target {
                    None => true,
                    Some(current) => candidate > current,
                };
                if replace {
                    best_target = Some(candidate);
                }
            }

            if let Some((_, reverse_target)) = best_target {
                let target = usize::MAX - reverse_target;
                owners[node] = target as RankId;
                loads[source] -= weight;
                loads[target] = loads[target].saturating_add(weight);
                moves += 1;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    *assignment = PartitionAssignment::from_owners(assignment.rank_count(), owners)?;
    Ok(moves)
}

/// Bounded, deterministic boundary pair exchange.
///
/// This deliberately uses a single cached adjacency and ranks candidate
/// vertices by their immediate change of undirected cut. Swapping equal
/// fine-DOF weights preserves the balance constraint exactly. The pair
/// correction handles a direct edge between the swapped vertices.
fn refine_fine_pair_swaps(
    matrix: &Csr32Matrix,
    assignment: &mut PartitionAssignment,
    imbalance_per_mille: u32,
    passes: usize,
) -> Result<usize, DistributedTopologyError> {
    let adjacency = build_undirected_adjacency(matrix)?;
    let ranks = assignment.rank_count() as usize;
    let n = matrix.nrows();
    let cap = load_cap(n, assignment.rank_count(), imbalance_per_mille);
    let floor = minimum_load(n, assignment.rank_count(), imbalance_per_mille);
    let mut owners = assignment.owners().to_vec();
    let mut loads = vec![0usize; ranks];
    for &owner in &owners {
        loads[owner as usize] += 1;
    }
    let mut total_swaps = 0usize;

    for _ in 0..passes {
        // Scoring uses the ownership at the start of this sweep.
        // Each rank pair has at most eight strong candidates in each direction.
        let mut candidates: Vec<Vec<Vec<(isize, usize)>>> = vec![vec![Vec::new(); ranks]; ranks];

        for node in 0..n {
            let source = owners[node] as usize;
            let mut links = vec![0usize; ranks];
            for &neighbor in &adjacency[node] {
                links[owners[neighbor] as usize] += 1;
            }
            let internal = links[source] as isize;
            for target in 0..ranks {
                if target == source || links[target] == 0 {
                    continue;
                }
                let gain = links[target] as isize - internal;
                candidates[source][target].push((gain, node));
            }
        }

        for row in &mut candidates {
            for items in row {
                items.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                items.truncate(8);
            }
        }

        let mut changed = false;
        let mut used = vec![false; n];

        for a in 0..ranks {
            for b in a + 1..ranks {
                let forward = &candidates[a][b];
                let reverse = &candidates[b][a];
                let mut proposals = Vec::new();

                for &(gain_a, node_a) in forward {
                    for &(gain_b, node_b) in reverse {
                        // Both vertices remain cut across each other after a
                        // swap, therefore the edge contributes zero net gain.
                        let shared = usize::from(adjacency[node_a].binary_search(&node_b).is_ok());
                        let net = gain_a + gain_b - (shared as isize * 2);
                        if net > 0 {
                            proposals.push((net, node_a, node_b));
                        }
                    }
                }

                proposals.sort_unstable_by(|a, b| {
                    b.0.cmp(&a.0)
                        .then_with(|| a.1.cmp(&b.1))
                        .then_with(|| a.2.cmp(&b.2))
                });

                for (_, node_a, node_b) in proposals {
                    if used[node_a] || used[node_b] {
                        continue;
                    }
                    if owners[node_a] != a as RankId || owners[node_b] != b as RankId {
                        continue;
                    }

                    // Recompute exact local gain against updated ownership;
                    // stale candidate rankings can otherwise accept bad swaps.
                    let mut gain_a = 0isize;
                    for &neighbor in &adjacency[node_a] {
                        if neighbor == node_b {
                            continue;
                        }
                        if owners[neighbor] == b as RankId {
                            gain_a += 1;
                        } else if owners[neighbor] == a as RankId {
                            gain_a -= 1;
                        }
                    }
                    let mut gain_b = 0isize;
                    for &neighbor in &adjacency[node_b] {
                        if neighbor == node_a {
                            continue;
                        }
                        if owners[neighbor] == a as RankId {
                            gain_b += 1;
                        } else if owners[neighbor] == b as RankId {
                            gain_b -= 1;
                        }
                    }

                    if gain_a + gain_b <= 0 {
                        continue;
                    }

                    // Fine-level nodes have unit weight, so this pair swap
                    // leaves rank loads unchanged.
                    if loads[a] < floor || loads[a] > cap || loads[b] < floor || loads[b] > cap {
                        continue;
                    }

                    owners[node_a] = b as RankId;
                    owners[node_b] = a as RankId;
                    used[node_a] = true;
                    used[node_b] = true;
                    total_swaps += 1;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    *assignment = PartitionAssignment::from_owners(assignment.rank_count(), owners)?;
    Ok(total_swaps)
}
/// G8-A6 multilevel ABTM partition prototype.
///
/// The hierarchy uses deterministic structural-affinity pair aggregation.
/// Partitioning happens on the coarsest weighted graph; assignments are then
/// prolonged and greedily cut-refined at every finer level while respecting
/// an explicit fine-DOF balance tolerance.
pub fn abtm_multilevel_partition(
    matrix: &Csr32Matrix,
    ranks: RankId,
    options: AbtmMultilevelOptions,
) -> Result<(PartitionAssignment, AbtmMultilevelStats), DistributedTopologyError> {
    options.validate()?;
    // A6_PHASE_PROFILE_V1: enabled only in the explicit F7 experiment.
    let phase_profile = std::env::var_os("HYBIT_A6_PHASE_PROFILE").is_some();
    let total_started = Instant::now();

    if matrix.nrows() != matrix.ncols() {
        return Err(DistributedTopologyError::MatrixMustBeSquare {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    if matrix.nrows() == 0 || ranks == 0 || ranks as usize > matrix.nrows() {
        return Err(DistributedTopologyError::InvalidRankCount);
    }

    let stop_vertices = (ranks as usize)
        .saturating_mul(options.coarse_vertices_per_rank)
        .max(ranks as usize);

    let mut current_matrix = matrix.clone();
    let mut current_weights = vec![1usize; matrix.nrows()];
    let mut levels = Vec::<HierarchyLevel>::new();
    let mut stats = AbtmMultilevelStats::default();

    let coarsening_started = Instant::now();
    // Every coarse matrix is constructed from the exact sorted undirected
    // adjacency; carry it forward instead of rebuilding an ABTM dual topology.
    let f10_init_started = Instant::now();
    let mut current_adjacency = build_undirected_adjacency(&current_matrix)?;
    if std::env::var_os("HYBIT_A6_F10_PHASE").is_some() {
        eprintln!(
            "A6 F10 initial_adjacency_ms={:.3}",
            f10_init_started.elapsed().as_secs_f64() * 1.0e3
        );
    }
    while levels.len() < options.max_levels && current_matrix.nrows() > stop_vertices {
        let level_started = Instant::now();
        let previous_n = current_matrix.nrows();
        let ((coarse_matrix, coarse_weights, fine_to_coarse, pairs, singletons), coarse_adjacency) =
            coarsen_once_cached(&current_matrix, &current_weights, &current_adjacency)?;

        if phase_profile {
            eprintln!(
                "A6 PROFILE level={} fine_vertices={} coarse_vertices={} fine_arcs={} coarse_arcs={} matched={} singletons={} ms={:.3}",
                levels.len() + 1,
                previous_n,
                coarse_matrix.nrows(),
                current_matrix.nnz(),
                coarse_matrix.nnz(),
                pairs,
                singletons,
                level_started.elapsed().as_secs_f64() * 1.0e3
            );
        }

        if coarse_matrix.nrows() >= previous_n {
            break;
        }

        stats.matched_pairs = stats.matched_pairs.saturating_add(pairs);
        stats.singleton_aggregates = stats.singleton_aggregates.saturating_add(singletons);

        levels.push(HierarchyLevel {
            matrix: current_matrix,
            weights: current_weights,
            fine_to_coarse,
        });

        current_matrix = coarse_matrix;
        current_weights = coarse_weights;
        current_adjacency = coarse_adjacency;
    }

    drop(current_adjacency);
    stats.levels_built = levels.len();
    stats.coarsest_vertices = current_matrix.nrows();
    if phase_profile {
        eprintln!(
            "A6 PROFILE coarsening_ms={:.3}",
            coarsening_started.elapsed().as_secs_f64() * 1.0e3
        );
    }
    let coarse_started = Instant::now();

    let (mut assignment, coarse_restarts) = weighted_multisource_partition(
        &current_matrix,
        &current_weights,
        ranks,
        options.imbalance_per_mille,
    )?;
    stats.coarse_restarts = coarse_restarts;
    if phase_profile {
        eprintln!(
            "A6 PROFILE coarse_growth_ms={:.3}",
            coarse_started.elapsed().as_secs_f64() * 1.0e3
        );
    }
    let coarse_refine_started = Instant::now();

    stats.refinement_moves = stats.refinement_moves.saturating_add(refine_by_cut(
        &current_matrix,
        &current_weights,
        &mut assignment,
        options.imbalance_per_mille,
        options.refinement_passes,
    )?);

    if phase_profile {
        eprintln!(
            "A6 PROFILE coarse_refine_ms={:.3}",
            coarse_refine_started.elapsed().as_secs_f64() * 1.0e3
        );
    }
    let uncoarsening_started = Instant::now();

    for level in levels.iter().rev() {
        let mut owners = Vec::with_capacity(level.fine_to_coarse.len());
        for &coarse in &level.fine_to_coarse {
            owners.push(assignment.owners()[coarse]);
        }

        assignment = PartitionAssignment::from_owners(ranks, owners)?;
        stats.refinement_moves = stats.refinement_moves.saturating_add(refine_by_cut(
            &level.matrix,
            &level.weights,
            &mut assignment,
            options.imbalance_per_mille,
            options.refinement_passes,
        )?);
    }

    if phase_profile {
        eprintln!(
            "A6 PROFILE uncoarsen_refine_ms={:.3}",
            uncoarsening_started.elapsed().as_secs_f64() * 1.0e3
        );
    }

    if assignment.global_dofs() != matrix.nrows() as GlobalDofId {
        return Err(invalid("multilevel prolongation dimension mismatch"));
    }
    let pair_swap_started = Instant::now();

    stats.refinement_moves = stats
        .refinement_moves
        .saturating_add(refine_fine_pair_swaps(
            matrix,
            &mut assignment,
            options.imbalance_per_mille,
            options.refinement_passes.saturating_mul(4),
        )?);

    if phase_profile {
        eprintln!(
            "A6 PROFILE pair_swap_ms={:.3}",
            pair_swap_started.elapsed().as_secs_f64() * 1.0e3
        );
    }
    let telemetry_started = Instant::now();
    let final_metrics: PartitionTelemetry = partition_telemetry_assignment(matrix, &assignment)?;
    if phase_profile {
        eprintln!(
            "A6 PROFILE final_telemetry_ms={:.3}",
            telemetry_started.elapsed().as_secs_f64() * 1.0e3
        );
        eprintln!(
            "A6 PROFILE total_ms={:.3}",
            total_started.elapsed().as_secs_f64() * 1.0e3
        );
    }
    stats.final_cut_nnz = final_metrics.cut_nnz;
    stats.final_communication_volume = final_metrics.communication_volume;

    Ok((assignment, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_matrix(n: usize) -> Csr32Matrix {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in 0..n {
            if row > 0 {
                col_idx.push((row - 1) as u32);
                values.push(1.0);
            }
            if row + 1 < n {
                col_idx.push((row + 1) as u32);
                values.push(1.0);
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    fn two_cluster_matrix(cluster: usize) -> Csr32Matrix {
        let n = cluster * 2;
        let mut adjacency = vec![Vec::<usize>::new(); n];

        for base in [0usize, cluster] {
            for (i, neighbors) in adjacency.iter_mut().enumerate().skip(base).take(cluster) {
                for j in base..base + cluster {
                    if i != j {
                        neighbors.push(j);
                    }
                }
            }
        }

        adjacency[cluster - 1].push(cluster);
        adjacency[cluster].push(cluster - 1);

        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        matrix_from_adjacency(&adjacency).unwrap()
    }

    #[test]
    fn abtm_merged_union_matches_sorted_reference() {
        let asymmetric =
            Csr32Matrix::new(4, 4, vec![0, 2, 3, 4, 4], vec![1, 2, 2, 1], vec![1.0; 4]).unwrap();
        for matrix in [chain_matrix(128), two_cluster_matrix(16), asymmetric] {
            let reference = build_undirected_adjacency_sorted(&matrix).unwrap();
            let merged = build_undirected_adjacency_merged(&matrix).unwrap();
            assert_eq!(reference, merged);
        }
    }
    #[test]
    fn stamped_affinity_matches_sorted_reference_on_dense_and_sparse_graphs() {
        for matrix in [chain_matrix(128), two_cluster_matrix(16)] {
            let adjacency = build_undirected_adjacency(&matrix).unwrap();
            let mut neighbor_stamps = vec![usize::MAX; adjacency.len()];
            for (node, neighbors) in adjacency.iter().enumerate() {
                for &neighbor in neighbors {
                    neighbor_stamps[neighbor] = node;
                }
                for &candidate in neighbors {
                    let stamped = adjacency[candidate]
                        .iter()
                        .filter(|&&shared| neighbor_stamps[shared] == node)
                        .count();
                    assert_eq!(
                        stamped,
                        common_neighbor_count(neighbors, &adjacency[candidate])
                    );
                }
            }
        }
    }

    #[test]
    fn cached_adjacency_equals_rebuilt_dual_across_two_levels() {
        let matrix = chain_matrix(128);
        let first_adjacency = build_undirected_adjacency(&matrix).unwrap();
        let ((coarse, coarse_weights, _, _, _), cached_adjacency) =
            coarsen_once_cached(&matrix, &vec![1usize; 128], &first_adjacency).unwrap();
        assert_eq!(
            cached_adjacency,
            build_undirected_adjacency(&coarse).unwrap()
        );
        let ((next, _, _, _, _), next_cached_adjacency) =
            coarsen_once_cached(&coarse, &coarse_weights, &cached_adjacency).unwrap();
        assert_eq!(
            next_cached_adjacency,
            build_undirected_adjacency(&next).unwrap()
        );
    }

    #[test]
    fn coarsening_reduces_chain_graph() {
        let matrix = chain_matrix(32);
        let weights = vec![1usize; 32];

        let (coarse, coarse_weights, mapping, pairs, _) = coarsen_once(&matrix, &weights).unwrap();

        assert!(coarse.nrows() < matrix.nrows());
        assert_eq!(mapping.len(), 32);
        assert_eq!(coarse_weights.iter().sum::<usize>(), 32);
        assert!(pairs > 0);
    }

    #[test]
    fn multilevel_partition_is_deterministic_and_within_balance_cap() {
        let matrix = chain_matrix(128);
        let options = AbtmMultilevelOptions {
            coarse_vertices_per_rank: 8,
            ..AbtmMultilevelOptions::default()
        };

        let (a, stats_a) = abtm_multilevel_partition(&matrix, 4, options).unwrap();
        let (b, stats_b) = abtm_multilevel_partition(&matrix, 4, options).unwrap();

        assert_eq!(a, b);
        assert_eq!(stats_a, stats_b);

        let metrics = partition_telemetry_assignment(&matrix, &a).unwrap();
        assert!(metrics.owned_dof_imbalance <= 1.03125);
    }

    #[test]
    fn load_cap_never_exceeds_configured_tolerance() {
        assert_eq!(load_cap(32, 2, 30), 16);
        assert_eq!(load_cap(127_224, 4, 30), 32_760);
    }
    #[test]
    fn multilevel_separates_two_dense_clusters() {
        let matrix = two_cluster_matrix(16);
        let options = AbtmMultilevelOptions {
            coarse_vertices_per_rank: 4,
            ..AbtmMultilevelOptions::default()
        };

        let (assignment, stats) = abtm_multilevel_partition(&matrix, 2, options).unwrap();
        let metrics = partition_telemetry_assignment(&matrix, &assignment).unwrap();
        println!("G8-A6 DENSE CLUSTER DIAGNOSTIC");
        println!("owners={:?}", assignment.owners());
        println!("stats={:?}", stats);
        println!("cut_nnz={}", metrics.cut_nnz);
        println!("communication_volume={}", metrics.communication_volume);
        println!("owned_dof_imbalance={:.6}", metrics.owned_dof_imbalance);
        println!("local_nnz_imbalance={:.6}", metrics.local_nnz_imbalance);

        assert!(metrics.cut_nnz <= 2);
    }
}
