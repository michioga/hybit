use hybit_core::HybitError;

use crate::{AbtmDualTopology, Csr32Matrix, DofMask};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmRegionGrowthStats {
    pub hops_requested: usize,
    pub hops_completed: usize,
    pub seed_nodes: usize,
    pub region_nodes: usize,
    pub frontier_nodes_visited: usize,
    pub row_words_visited: usize,
    pub column_words_visited: usize,
    pub candidate_neighbor_bits: usize,
}

impl AbtmRegionGrowthStats {
    pub fn topology_words_visited(self) -> usize {
        self.row_words_visited
            .saturating_add(self.column_words_visited)
    }

    pub fn new_nodes(self) -> usize {
        self.region_nodes.saturating_sub(self.seed_nodes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbtmRegionGrowth {
    region: DofMask,
    frontier: DofMask,
    stats: AbtmRegionGrowthStats,
}

impl AbtmRegionGrowth {
    pub fn region(&self) -> &DofMask {
        &self.region
    }

    pub fn frontier(&self) -> &DofMask {
        &self.frontier
    }

    pub fn stats(&self) -> AbtmRegionGrowthStats {
        self.stats
    }
}

/// Grow a seed set through the undirected structural graph `A union A^T`.
///
/// The dual topology supplies outgoing neighbors from `row(node)` and incoming
/// neighbors from `column(node)`. The seed set is always included in the
/// returned region. Each hop expands only the previous hop's frontier, and
/// nodes already present in the region are removed before the next frontier is
/// formed.
///
/// The operation is structural: numerical matrix values are never read.
pub fn grow_undirected_region(
    topology: &AbtmDualTopology,
    seeds: &DofMask,
    hops: usize,
) -> Result<AbtmRegionGrowth, HybitError> {
    if topology.nrows() != topology.ncols() {
        return Err(HybitError::InvalidMatrix(
            "ABTM region growth requires a square topology",
        ));
    }
    if seeds.len() != topology.nrows() {
        return Err(HybitError::DimensionMismatch {
            expected: topology.nrows(),
            actual: seeds.len(),
        });
    }

    let mut region = seeds.clone();
    let mut frontier = seeds.clone();
    let mut stats = AbtmRegionGrowthStats {
        hops_requested: hops,
        seed_nodes: seeds.count_ones(),
        region_nodes: seeds.count_ones(),
        ..AbtmRegionGrowthStats::default()
    };

    for _ in 0..hops {
        if frontier.is_empty() {
            break;
        }

        let frontier_indices = frontier.indices();
        stats.frontier_nodes_visited = stats
            .frontier_nodes_visited
            .saturating_add(frontier_indices.len());

        let mut next = DofMask::new(topology.nrows());

        for node in frontier_indices {
            let row = topology.row(node)?;
            for word in row.words() {
                stats.row_words_visited = stats.row_words_visited.saturating_add(1);
                stats.candidate_neighbor_bits = stats
                    .candidate_neighbor_bits
                    .saturating_add(word.popcount());
                next.or_word(word.word_index() as usize, word.mask());
            }

            let column = topology.column(node)?;
            for word in column.words() {
                stats.column_words_visited = stats.column_words_visited.saturating_add(1);
                stats.candidate_neighbor_bits = stats
                    .candidate_neighbor_bits
                    .saturating_add(word.popcount());
                next.or_word(word.word_index() as usize, word.mask());
            }
        }

        next.and_not_assign(&region)?;
        region.union_assign(&next)?;
        frontier = next;
        stats.hops_completed = stats.hops_completed.saturating_add(1);
    }

    stats.region_nodes = region.count_ones();

    Ok(AbtmRegionGrowth {
        region,
        frontier,
        stats,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmRegionMultiplicityStats {
    pub region_count: usize,
    pub node_count: usize,
    pub total_memberships: u64,
    pub covered_nodes: usize,
    pub overlap_nodes: usize,
    pub max_multiplicity: u32,
    pub pair_overlap_memberships: u64,
}

impl AbtmRegionMultiplicityStats {
    pub fn extra_memberships(self) -> u64 {
        self.total_memberships
            .saturating_sub(self.covered_nodes as u64)
    }

    pub fn average_multiplicity_on_covered(self) -> f64 {
        if self.covered_nodes == 0 {
            0.0
        } else {
            self.total_memberships as f64 / self.covered_nodes as f64
        }
    }

    pub fn overlap_fraction_on_covered(self) -> f64 {
        if self.covered_nodes == 0 {
            0.0
        } else {
            self.overlap_nodes as f64 / self.covered_nodes as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbtmRegionMultiplicity {
    counts: Vec<u32>,
    stats: AbtmRegionMultiplicityStats,
}

impl AbtmRegionMultiplicity {
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    pub fn multiplicity(&self, node: usize) -> Option<u32> {
        self.counts.get(node).copied()
    }

    pub fn stats(&self) -> AbtmRegionMultiplicityStats {
        self.stats
    }
}

/// Count how many regions contain each node.
///
/// Regions are represented as `DofMask`s over the same node universe. The
/// operation is structural and value-independent. `pair_overlap_memberships`
/// is the exact sum of pairwise region-intersection sizes, computed from the
/// multiplicity identity `sum_v C(m_v, 2)`.
pub fn region_multiplicity(
    node_count: usize,
    regions: &[DofMask],
) -> Result<AbtmRegionMultiplicity, HybitError> {
    let mut counts = vec![0u32; node_count];
    let mut total_memberships = 0u64;

    for region in regions {
        if region.len() != node_count {
            return Err(HybitError::DimensionMismatch {
                expected: node_count,
                actual: region.len(),
            });
        }

        for (word_index, &word) in region.words().iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let node = word_index
                    .checked_mul(64)
                    .and_then(|base| base.checked_add(bit))
                    .ok_or(HybitError::SizeOverflow)?;
                if node >= node_count {
                    return Err(HybitError::InvalidMatrix(
                        "region mask contains a bit beyond the node universe",
                    ));
                }
                counts[node] = counts[node]
                    .checked_add(1)
                    .ok_or(HybitError::SizeOverflow)?;
                total_memberships = total_memberships
                    .checked_add(1)
                    .ok_or(HybitError::SizeOverflow)?;
                bits &= bits - 1;
            }
        }
    }

    let mut covered_nodes = 0usize;
    let mut overlap_nodes = 0usize;
    let mut max_multiplicity = 0u32;
    let mut pair_overlap_memberships = 0u64;

    for &count in &counts {
        if count == 0 {
            continue;
        }
        covered_nodes = covered_nodes.saturating_add(1);
        if count >= 2 {
            overlap_nodes = overlap_nodes.saturating_add(1);
        }
        max_multiplicity = max_multiplicity.max(count);

        let count = u64::from(count);
        pair_overlap_memberships = pair_overlap_memberships
            .checked_add(
                count
                    .checked_mul(count.saturating_sub(1))
                    .ok_or(HybitError::SizeOverflow)?
                    / 2,
            )
            .ok_or(HybitError::SizeOverflow)?;
    }

    Ok(AbtmRegionMultiplicity {
        counts,
        stats: AbtmRegionMultiplicityStats {
            region_count: regions.len(),
            node_count,
            total_memberships,
            covered_nodes,
            overlap_nodes,
            max_multiplicity,
            pair_overlap_memberships,
        },
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmLocalSubmatrixPatternStats {
    pub global_nodes: usize,
    pub local_nodes: usize,
    pub local_nnz: usize,
    pub topology_words_visited: usize,
    pub candidate_neighbor_bits: usize,
    pub kept_neighbor_bits: usize,
    pub mapping_scratch_bytes: usize,
    pub pattern_bytes: usize,
}

impl AbtmLocalSubmatrixPatternStats {
    pub fn structural_pruning_ratio(self) -> f64 {
        if self.candidate_neighbor_bits == 0 {
            0.0
        } else {
            1.0 - self.kept_neighbor_bits as f64 / self.candidate_neighbor_bits as f64
        }
    }

    pub fn average_local_row_nnz(self) -> f64 {
        if self.local_nodes == 0 {
            0.0
        } else {
            self.local_nnz as f64 / self.local_nodes as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbtmLocalSubmatrixPattern {
    global_nodes: Vec<u32>,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    stats: AbtmLocalSubmatrixPatternStats,
}

impl AbtmLocalSubmatrixPattern {
    pub fn global_nodes(&self) -> &[u32] {
        &self.global_nodes
    }

    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }

    pub fn col_idx(&self) -> &[u32] {
        &self.col_idx
    }

    pub fn nrows(&self) -> usize {
        self.global_nodes.len()
    }

    pub fn ncols(&self) -> usize {
        self.global_nodes.len()
    }

    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    pub fn stats(&self) -> AbtmLocalSubmatrixPatternStats {
        self.stats
    }
}

/// Extract the structural pattern of `A[R,R]` from a square dual topology.
///
/// Local row/column numbering follows ascending global node order. Duplicate
/// structural entries have already been collapsed by `AbtmTopology`.
///
/// The dense global-to-local map is temporary scratch storage. G3c measures
/// that cost explicitly so a later prepared-region representation can decide
/// whether the mapping should be cached and reused.
pub fn extract_local_submatrix_pattern(
    topology: &AbtmDualTopology,
    region: &DofMask,
) -> Result<AbtmLocalSubmatrixPattern, HybitError> {
    if topology.nrows() != topology.ncols() {
        return Err(HybitError::InvalidMatrix(
            "ABTM local submatrix extraction requires a square topology",
        ));
    }
    if region.len() != topology.nrows() {
        return Err(HybitError::DimensionMismatch {
            expected: topology.nrows(),
            actual: region.len(),
        });
    }

    let global_indices = region.indices();
    if global_indices.len() > u32::MAX as usize {
        return Err(HybitError::SizeOverflow);
    }

    let mut global_nodes = Vec::with_capacity(global_indices.len());
    let mut global_to_local = vec![u32::MAX; topology.nrows()];

    for (local, &global) in global_indices.iter().enumerate() {
        let global_u32 = u32::try_from(global).map_err(|_| HybitError::SizeOverflow)?;
        let local_u32 = u32::try_from(local).map_err(|_| HybitError::SizeOverflow)?;
        global_nodes.push(global_u32);
        global_to_local[global] = local_u32;
    }

    let mut row_ptr = Vec::with_capacity(global_nodes.len() + 1);
    let mut col_idx = Vec::new();
    let mut topology_words_visited = 0usize;
    let mut candidate_neighbor_bits = 0usize;
    let mut kept_neighbor_bits = 0usize;

    row_ptr.push(0);

    for &global_row in &global_nodes {
        let row = topology.row(global_row as usize)?;

        for word in row.words() {
            topology_words_visited = topology_words_visited.saturating_add(1);
            candidate_neighbor_bits = candidate_neighbor_bits.saturating_add(word.popcount());

            let word_index = word.word_index() as usize;
            let region_word = region.words().get(word_index).copied().unwrap_or(0);
            let mut bits = word.mask() & region_word;
            kept_neighbor_bits = kept_neighbor_bits.saturating_add(bits.count_ones() as usize);

            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let global_col = word
                    .base_col()
                    .checked_add(bit)
                    .ok_or(HybitError::SizeOverflow)?;
                let local_col =
                    global_to_local
                        .get(global_col)
                        .copied()
                        .ok_or(HybitError::InvalidMatrix(
                            "ABTM local pattern produced an out-of-range column",
                        ))?;
                if local_col == u32::MAX {
                    return Err(HybitError::InvalidMatrix(
                        "ABTM local pattern retained a column outside the region",
                    ));
                }
                col_idx.push(local_col);
                bits &= bits - 1;
            }
        }

        if col_idx.len() > u32::MAX as usize {
            return Err(HybitError::SizeOverflow);
        }
        row_ptr.push(col_idx.len() as u32);
    }

    let pattern_bytes = global_nodes
        .len()
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(row_ptr.len().saturating_mul(std::mem::size_of::<u32>()))
        .saturating_add(col_idx.len().saturating_mul(std::mem::size_of::<u32>()));

    let stats = AbtmLocalSubmatrixPatternStats {
        global_nodes: topology.nrows(),
        local_nodes: global_nodes.len(),
        local_nnz: col_idx.len(),
        topology_words_visited,
        candidate_neighbor_bits,
        kept_neighbor_bits,
        mapping_scratch_bytes: global_to_local
            .len()
            .saturating_mul(std::mem::size_of::<u32>()),
        pattern_bytes,
    };

    Ok(AbtmLocalSubmatrixPattern {
        global_nodes,
        row_ptr,
        col_idx,
        stats,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbtmLocalNumericPlanStats {
    pub local_nodes: usize,
    pub local_nnz: usize,
    pub source_terms: usize,
    pub duplicate_source_terms: usize,
    pub direct_source_map: bool,
    pub plan_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbtmLocalNumericPlan {
    pattern: AbtmLocalSubmatrixPattern,
    source_ptr: Vec<u32>,
    source_indices: Vec<u32>,
    source_value_len: usize,
    direct_source_map: bool,
    stats: AbtmLocalNumericPlanStats,
}

impl AbtmLocalNumericPlan {
    pub fn pattern(&self) -> &AbtmLocalSubmatrixPattern {
        &self.pattern
    }

    pub fn stats(&self) -> AbtmLocalNumericPlanStats {
        self.stats
    }

    /// Gather a new local numerical value vector without rebuilding topology.
    ///
    /// `values` must use the same structural CSR ordering that was used when
    /// the plan was prepared. For the common canonical-CSR case every local
    /// structural entry has exactly one source and the gather is a direct
    /// indexed copy. Duplicate source entries fall back to summation.
    pub fn gather_values(&self, values: &[f64]) -> Result<Vec<f64>, HybitError> {
        if values.len() != self.source_value_len {
            return Err(HybitError::DimensionMismatch {
                expected: self.source_value_len,
                actual: values.len(),
            });
        }

        if self.direct_source_map {
            let mut local = Vec::with_capacity(self.source_indices.len());
            for &source in &self.source_indices {
                local.push(values[source as usize]);
            }
            return Ok(local);
        }

        let mut local = Vec::with_capacity(self.pattern.nnz());
        for pair in self.source_ptr.windows(2) {
            let start = pair[0] as usize;
            let end = pair[1] as usize;
            let mut sum = 0.0;
            for &source in &self.source_indices[start..end] {
                sum += values[source as usize];
            }
            local.push(sum);
        }
        Ok(local)
    }
}

/// Prepare symbolic/numerical addressing for repeated local value refreshes.
///
/// The structural pattern comes from ABTM topology. CSR is consulted once to
/// bind every local structural entry to one or more source value positions.
/// This separates symbolic region extraction from later numerical
/// refactorization when the sparsity pattern is unchanged.
pub fn prepare_local_numeric_plan(
    topology: &AbtmDualTopology,
    matrix: &Csr32Matrix,
    region: &DofMask,
) -> Result<AbtmLocalNumericPlan, HybitError> {
    if topology.nrows() != matrix.nrows() || topology.ncols() != matrix.ncols() {
        return Err(HybitError::InvalidArgument(
            "ABTM local numeric plan topology/matrix dimensions differ",
        ));
    }

    let pattern = extract_local_submatrix_pattern(topology, region)?;

    let mut global_to_local = vec![u32::MAX; matrix.ncols()];
    for (local, &global) in pattern.global_nodes().iter().enumerate() {
        global_to_local[global as usize] =
            u32::try_from(local).map_err(|_| HybitError::SizeOverflow)?;
    }

    let mut source_ptr = Vec::with_capacity(pattern.nnz().saturating_add(1));
    let mut source_indices = Vec::new();
    source_ptr.push(0);

    for local_row in 0..pattern.nrows() {
        let global_row = pattern.global_nodes()[local_row] as usize;
        let start = matrix.row_ptr()[global_row] as usize;
        let end = matrix.row_ptr()[global_row + 1] as usize;

        let mut by_local_col: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();

        for source in start..end {
            let global_col = matrix.col_idx()[source] as usize;
            let local_col = global_to_local[global_col];
            if local_col == u32::MAX {
                continue;
            }
            by_local_col
                .entry(local_col)
                .or_default()
                .push(u32::try_from(source).map_err(|_| HybitError::SizeOverflow)?);
        }

        let p_start = pattern.row_ptr()[local_row] as usize;
        let p_end = pattern.row_ptr()[local_row + 1] as usize;

        for &local_col in &pattern.col_idx()[p_start..p_end] {
            let sources = by_local_col
                .remove(&local_col)
                .ok_or(HybitError::InvalidMatrix(
                    "ABTM local numeric plan could not bind a structural entry",
                ))?;
            source_indices.extend_from_slice(&sources);
            if source_indices.len() > u32::MAX as usize {
                return Err(HybitError::SizeOverflow);
            }
            source_ptr.push(source_indices.len() as u32);
        }

        if !by_local_col.is_empty() {
            return Err(HybitError::InvalidMatrix(
                "CSR local structure contains entries absent from ABTM topology",
            ));
        }
    }

    if source_ptr.len() != pattern.nnz().saturating_add(1) {
        return Err(HybitError::InvalidMatrix(
            "ABTM local numeric source pointer length is inconsistent",
        ));
    }

    let direct_source_map = source_indices.len() == pattern.nnz()
        && source_ptr
            .iter()
            .enumerate()
            .all(|(index, &ptr)| ptr as usize == index);

    let duplicate_source_terms = source_indices.len().saturating_sub(pattern.nnz());

    let plan_bytes = pattern
        .stats()
        .pattern_bytes
        .saturating_add(source_ptr.len().saturating_mul(std::mem::size_of::<u32>()))
        .saturating_add(
            source_indices
                .len()
                .saturating_mul(std::mem::size_of::<u32>()),
        );

    let stats = AbtmLocalNumericPlanStats {
        local_nodes: pattern.nrows(),
        local_nnz: pattern.nnz(),
        source_terms: source_indices.len(),
        duplicate_source_terms,
        direct_source_map,
        plan_bytes,
    };

    Ok(AbtmLocalNumericPlan {
        pattern,
        source_ptr,
        source_indices,
        source_value_len: matrix.values().len(),
        direct_source_map,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Csr32Matrix;

    fn directed_chain_with_incoming_edge() -> AbtmDualTopology {
        // 0 -> 1 -> 2 <- 3 ; 4 isolated.
        let csr = Csr32Matrix::new(
            5,
            5,
            vec![0, 1, 2, 2, 3, 3],
            vec![1, 2, 2],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        AbtmDualTopology::from_csr32(&csr).unwrap()
    }

    #[test]
    fn zero_hops_returns_seed_region() {
        let topology = directed_chain_with_incoming_edge();
        let seed = DofMask::from_indices(5, &[0]).unwrap();
        let growth = grow_undirected_region(&topology, &seed, 0).unwrap();
        assert_eq!(growth.region().indices(), vec![0]);
        assert_eq!(growth.frontier().indices(), vec![0]);
        assert_eq!(growth.stats().hops_completed, 0);
    }

    #[test]
    fn undirected_growth_uses_rows_and_columns() {
        let topology = directed_chain_with_incoming_edge();
        let seed = DofMask::from_indices(5, &[0]).unwrap();

        let hop1 = grow_undirected_region(&topology, &seed, 1).unwrap();
        assert_eq!(hop1.region().indices(), vec![0, 1]);
        assert_eq!(hop1.frontier().indices(), vec![1]);

        let hop2 = grow_undirected_region(&topology, &seed, 2).unwrap();
        assert_eq!(hop2.region().indices(), vec![0, 1, 2]);
        assert_eq!(hop2.frontier().indices(), vec![2]);

        let hop3 = grow_undirected_region(&topology, &seed, 3).unwrap();
        assert_eq!(hop3.region().indices(), vec![0, 1, 2, 3]);
        assert_eq!(hop3.frontier().indices(), vec![3]);
    }

    #[test]
    fn incoming_edge_can_expand_from_seed() {
        let topology = directed_chain_with_incoming_edge();
        let seed = DofMask::from_indices(5, &[2]).unwrap();
        let hop1 = grow_undirected_region(&topology, &seed, 1).unwrap();
        assert_eq!(hop1.region().indices(), vec![1, 2, 3]);
    }

    #[test]
    fn seed_dimension_mismatch_is_rejected() {
        let topology = directed_chain_with_incoming_edge();
        let seed = DofMask::from_indices(4, &[0]).unwrap();
        assert!(matches!(
            grow_undirected_region(&topology, &seed, 1),
            Err(HybitError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn multiplicity_counts_overlap_and_pair_memberships() {
        let regions = vec![
            DofMask::from_indices(8, &[0, 1, 2, 5]).unwrap(),
            DofMask::from_indices(8, &[1, 2, 3, 5]).unwrap(),
            DofMask::from_indices(8, &[2, 4, 5]).unwrap(),
        ];
        let multiplicity = region_multiplicity(8, &regions).unwrap();
        assert_eq!(multiplicity.counts(), &[1, 2, 3, 1, 1, 3, 0, 0]);
        assert_eq!(multiplicity.multiplicity(2), Some(3));
        assert_eq!(multiplicity.multiplicity(8), None);

        let stats = multiplicity.stats();
        assert_eq!(stats.region_count, 3);
        assert_eq!(stats.node_count, 8);
        assert_eq!(stats.total_memberships, 11);
        assert_eq!(stats.covered_nodes, 6);
        assert_eq!(stats.overlap_nodes, 3);
        assert_eq!(stats.max_multiplicity, 3);
        // C(2,2) + C(3,2) + C(3,2) = 1 + 3 + 3.
        assert_eq!(stats.pair_overlap_memberships, 7);
    }

    #[test]
    fn multiplicity_rejects_mismatched_region_universe() {
        let regions = vec![DofMask::from_indices(7, &[1, 2]).unwrap()];
        assert!(matches!(
            region_multiplicity(8, &regions),
            Err(HybitError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn local_submatrix_pattern_uses_sorted_local_numbering() {
        let csr = Csr32Matrix::new(
            5,
            5,
            vec![0, 3, 5, 8, 10, 12],
            vec![0, 1, 4, 1, 2, 0, 2, 3, 3, 4, 1, 4],
            vec![1.0; 12],
        )
        .unwrap();
        let topology = AbtmDualTopology::from_csr32(&csr).unwrap();
        let region = DofMask::from_indices(5, &[0, 2, 3]).unwrap();

        let pattern = extract_local_submatrix_pattern(&topology, &region).unwrap();
        assert_eq!(pattern.global_nodes(), &[0, 2, 3]);
        assert_eq!(pattern.row_ptr(), &[0, 1, 4, 5]);
        assert_eq!(pattern.col_idx(), &[0, 0, 1, 2, 2]);
        assert_eq!(pattern.nrows(), 3);
        assert_eq!(pattern.ncols(), 3);
        assert_eq!(pattern.nnz(), 5);

        let stats = pattern.stats();
        assert_eq!(stats.local_nodes, 3);
        assert_eq!(stats.local_nnz, 5);
        assert_eq!(stats.kept_neighbor_bits, 5);
        assert!(stats.candidate_neighbor_bits >= stats.kept_neighbor_bits);
    }

    #[test]
    fn local_submatrix_pattern_rejects_region_dimension_mismatch() {
        let csr = Csr32Matrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0; 3]).unwrap();
        let topology = AbtmDualTopology::from_csr32(&csr).unwrap();
        let region = DofMask::from_indices(2, &[0]).unwrap();

        assert!(matches!(
            extract_local_submatrix_pattern(&topology, &region),
            Err(HybitError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn local_numeric_plan_gathers_values_and_sums_duplicates() {
        let csr = Csr32Matrix::new(
            3,
            3,
            vec![0, 3, 5, 7],
            vec![0, 1, 1, 0, 2, 1, 2],
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let topology = AbtmDualTopology::from_csr32(&csr).unwrap();
        let region = DofMask::from_indices(3, &[0, 1]).unwrap();

        let plan = prepare_local_numeric_plan(&topology, &csr, &region).unwrap();
        assert_eq!(plan.pattern().global_nodes(), &[0, 1]);
        assert_eq!(plan.pattern().row_ptr(), &[0, 2, 3]);
        assert_eq!(plan.pattern().col_idx(), &[0, 1, 0]);

        let gathered = plan.gather_values(csr.values()).unwrap();
        assert_eq!(gathered, vec![2.0, 7.0, 5.0]);

        let stats = plan.stats();
        assert_eq!(stats.local_nnz, 3);
        assert_eq!(stats.source_terms, 4);
        assert_eq!(stats.duplicate_source_terms, 1);
        assert!(!stats.direct_source_map);
    }

    #[test]
    fn local_numeric_plan_uses_direct_source_map_for_canonical_rows() {
        let csr = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 2, 2],
            vec![2.0, 3.0, 5.0, 6.0, 8.0],
        )
        .unwrap();
        let topology = AbtmDualTopology::from_csr32(&csr).unwrap();
        let region = DofMask::from_indices(3, &[0, 1]).unwrap();

        let plan = prepare_local_numeric_plan(&topology, &csr, &region).unwrap();
        assert!(plan.stats().direct_source_map);
        assert_eq!(
            plan.gather_values(csr.values()).unwrap(),
            vec![2.0, 3.0, 5.0]
        );
    }
}
