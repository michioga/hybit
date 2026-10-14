//! H1-B: block-closed mixed-field regions for ABTM symbolic planning.
//!
//! This metadata-only module does not alter Krylov routing or directly perform
//! contact mechanics, physical assembly, partitioning, or factorization.
//! A complete physical block is kept intact in a Schwarz candidate region.
//! Different fields of the same entity are *not* silently coupled: only
//! structural edges or explicit seeds can connect them.

use crate::{
    extract_local_submatrix_pattern, grow_undirected_region, AbtmDualTopology,
    AbtmLocalSubmatrixPattern, DofMask, FieldAwareDofLayout,
};
use hybit_core::HybitError;

fn check_mask_layout(layout: &FieldAwareDofLayout, mask: &DofMask) -> Result<(), HybitError> {
    let len = usize::try_from(layout.total_dofs()).map_err(|_| HybitError::SizeOverflow)?;
    if len != mask.len() {
        return Err(HybitError::DimensionMismatch {
            expected: len,
            actual: mask.len(),
        });
    }
    Ok(())
}

/// Expand each selected DOF to its complete `(entity_kind, entity_id, field_id)` block.
///
/// Supports blocks spanning multiple 64-bit words. No numeric matrix entries
/// are read and separate fields on one entity stay separate.
pub fn close_to_field_blocks(
    layout: &FieldAwareDofLayout,
    selected: &DofMask,
) -> Result<DofMask, HybitError> {
    check_mask_layout(layout, selected)?;
    let mut result = DofMask::new(selected.len());
    let words = selected.words();
    for (block_index, _) in layout.blocks().iter().enumerate() {
        let range = layout
            .block_range(block_index)
            .expect("validated block index");
        let start = usize::try_from(range.start).map_err(|_| HybitError::SizeOverflow)?;
        let end = usize::try_from(range.end).map_err(|_| HybitError::SizeOverflow)?;
        let first = start / 64;
        let last = (end - 1) / 64;
        let mut hit = false;
        /*
        for word in first..=last {
            let low = if word == first {
                u64::MAX << (start % 64)
            } else {
                u64::MAX
            };
            let high = if word == last && end % 64 != 0 {
                (1u64 << (end % 64)) - 1
            } else {
                u64::MAX
            };
            if words[word] & low & high != 0 {
                hit = true;
                break;
            }
        }
        */
        for (word, &selected_bits) in words.iter().enumerate().take(last + 1).skip(first) {
            let low = if word == first {
                u64::MAX << (start % 64)
            } else {
                u64::MAX
            };
            let high = if word == last && end % 64 != 0 {
                (1u64 << (end % 64)) - 1
            } else {
                u64::MAX
            };
            if selected_bits & low & high != 0 {
                hit = true;
                break;
            }
        }
        if hit {
            for word in first..=last {
                let low = if word == first {
                    u64::MAX << (start % 64)
                } else {
                    u64::MAX
                };
                let high = if word == last && end % 64 != 0 {
                    (1u64 << (end % 64)) - 1
                } else {
                    u64::MAX
                };
                result.or_word(word, low & high);
            }
        }
    }
    Ok(result)
}

/// Exact statistics for block-closed symbolic growth. Frontier traversal metrics
/// count only ABTM graph expansion, not implicit intra-block closure edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldAwareRegionStats {
    pub hops_requested: usize,
    pub hops_completed: usize,
    pub seed_dofs: usize,
    pub block_closed_seed_dofs: usize,
    pub region_dofs: usize,
    pub region_blocks: usize,
    pub frontier_dofs_visited: usize,
    pub row_words_visited: usize,
    pub column_words_visited: usize,
    pub candidate_neighbor_bits: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldAwareRegionGrowth {
    region: DofMask,
    frontier: DofMask,
    stats: FieldAwareRegionStats,
}

impl FieldAwareRegionGrowth {
    pub fn region(&self) -> &DofMask {
        &self.region
    }
    pub fn frontier(&self) -> &DofMask {
        &self.frontier
    }
    pub fn stats(&self) -> FieldAwareRegionStats {
        self.stats
    }

    /// Reuse G3's ABTM `A[R,R]` symbolic extractor for a block-closed region.
    /// `global_nodes` in the returned pattern are global DOF indices, not FEM node IDs.
    pub fn local_pattern(
        &self,
        topology: &AbtmDualTopology,
    ) -> Result<AbtmLocalSubmatrixPattern, HybitError> {
        extract_local_submatrix_pattern(topology, &self.region)
    }
}

/// Grow the undirected structural graph `A union A^T` by `hops` and close
/// each new frontier over entire physical field blocks.
///
/// A hop means **one structural matrix edge**, followed by a free block closure.
/// Seed closure occurs even when `hops == 0`. This is deliberately different
/// from scalar DOF graph growth, and must be the same in CSR reference tests.
pub fn grow_field_aware_region(
    topology: &AbtmDualTopology,
    layout: &FieldAwareDofLayout,
    seeds: &DofMask,
    hops: usize,
) -> Result<FieldAwareRegionGrowth, HybitError> {
    layout.validate_square_matrix_size(topology.nrows(), topology.ncols())?;
    check_mask_layout(layout, seeds)?;
    let mut region = close_to_field_blocks(layout, seeds)?;
    let mut frontier = region.clone();
    let mut stats = FieldAwareRegionStats {
        hops_requested: hops,
        seed_dofs: seeds.count_ones(),
        block_closed_seed_dofs: region.count_ones(),
        ..FieldAwareRegionStats::default()
    };

    for _ in 0..hops {
        if frontier.is_empty() {
            break;
        }
        let step = grow_undirected_region(topology, &frontier, 1)?;
        let s = step.stats();
        stats.frontier_dofs_visited = stats
            .frontier_dofs_visited
            .saturating_add(s.frontier_nodes_visited);
        stats.row_words_visited = stats.row_words_visited.saturating_add(s.row_words_visited);
        stats.column_words_visited = stats
            .column_words_visited
            .saturating_add(s.column_words_visited);
        stats.candidate_neighbor_bits = stats
            .candidate_neighbor_bits
            .saturating_add(s.candidate_neighbor_bits);

        let mut next = close_to_field_blocks(layout, step.region())?;
        next.and_not_assign(&region)?;
        region.union_assign(&next)?;
        frontier = next;
        stats.hops_completed += 1;
    }
    stats.region_dofs = region.count_ones();
    stats.region_blocks = layout
        .blocks()
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let r = layout.block_range(*i).expect("valid block");
            region.contains(r.start as usize)
        })
        .count();
    Ok(FieldAwareRegionGrowth {
        region,
        frontier,
        stats,
    })
}

/// Validate an optional MPI ownership assignment for all physical DOFs.
///
/// Ownership must not split a field block; distinct fields of one entity may
/// belong to different ranks. This does **not** construct a HaloPlan or change
/// the current contiguous/graph partition. A failing assignment must be
/// revised before it can be used for block-aware distributed preparation.
pub fn field_block_owners(
    layout: &FieldAwareDofLayout,
    dof_owners: &[u32],
    ranks: u32,
) -> Result<Vec<u32>, HybitError> {
    let n = usize::try_from(layout.total_dofs()).map_err(|_| HybitError::SizeOverflow)?;
    if dof_owners.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: dof_owners.len(),
        });
    }
    if ranks == 0 {
        return Err(HybitError::InvalidArgument(
            "MPI rank count must be positive",
        ));
    }
    let mut block_owners = Vec::with_capacity(layout.blocks().len());
    for i in 0..layout.blocks().len() {
        let r = layout.block_range(i).expect("valid block");
        let start = usize::try_from(r.start).map_err(|_| HybitError::SizeOverflow)?;
        let end = usize::try_from(r.end).map_err(|_| HybitError::SizeOverflow)?;
        let owner = dof_owners[start];
        if owner >= ranks {
            return Err(HybitError::InvalidArgument("DOF owner rank out of range"));
        }
        if dof_owners[start..end].iter().any(|&v| v != owner) {
            return Err(HybitError::InvalidArgument(
                "MPI ownership splits a physical field block",
            ));
        }
        block_owners.push(owner);
    }
    Ok(block_owners)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Csr32Matrix, DofBlock, DofEntityKind};

    fn block(kind: DofEntityKind, id: u64, field: u32, components: u32) -> DofBlock {
        DofBlock {
            entity_kind: kind,
            entity_id: id,
            field_id: field,
            components,
        }
    }

    #[test]
    fn closure_crosses_64bit_words_without_merging_fields() {
        use DofEntityKind::*;
        let layout = FieldAwareDofLayout::new(vec![
            block(Node, 1, 1, 63),       // 0..63
            block(Node, 1, 2, 4),        // 63..67; same entity, separate field
            block(Multiplier, 9, 3, 65), // 67..132
        ])
        .unwrap();
        let seed = DofMask::from_indices(132, &[65, 129]).unwrap();
        let closed = close_to_field_blocks(&layout, &seed).unwrap();
        assert_eq!(closed.count_ones(), 69);
        assert!(!closed.contains(62));
        assert!(closed.contains(63));
        assert!(closed.contains(66));
        assert!(closed.contains(67));
        assert!(closed.contains(131));
        assert!(close_to_field_blocks(&layout, &DofMask::new(132))
            .unwrap()
            .is_empty());
        assert!(close_to_field_blocks(&layout, &DofMask::new(131)).is_err());
    }

    fn mixed_fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
        use DofEntityKind::*;
        let layout = FieldAwareDofLayout::new(vec![
            block(Node, 1, 1, 3),       // 0..3 solid translation
            block(Node, 1, 2, 1),       // 3..4 temperature (same entity)
            block(Node, 2, 1, 2),       // 4..6 shell rotation
            block(Multiplier, 3, 5, 3), // 6..9 contact multipliers
        ])
        .unwrap();
        // directed structure; row0->4, row6->5. The second is incoming to DOF 5.
        // An explicit stored zero still represents a structural dependency.
        let mut rows = vec![vec![]; 9];
        for (i, row) in rows.iter_mut().enumerate() {
            row.push((i as u32, 1.0));
        }
        rows[0].push((4, 0.0));
        rows[6].push((5, -2.0));
        let mut ptr = vec![0u32];
        let mut cols = vec![];
        let mut vals = vec![];
        for row in rows {
            for (col, val) in row {
                cols.push(col);
                vals.push(val);
            }
            ptr.push(cols.len() as u32);
        }
        (layout, Csr32Matrix::new(9, 9, ptr, cols, vals).unwrap())
    }

    #[test]
    fn directed_zero_edge_and_contact_closure_match_reference() {
        let (layout, a) = mixed_fixture();
        let topo = AbtmDualTopology::from_csr32(&a).unwrap();
        let seeds = DofMask::from_indices(9, &[0]).unwrap();
        let initial = grow_field_aware_region(&topo, &layout, &seeds, 0).unwrap();
        assert_eq!(initial.region().indices(), vec![0, 1, 2]);
        let one = grow_field_aware_region(&topo, &layout, &seeds, 1).unwrap();
        assert_eq!(one.region().indices(), vec![0, 1, 2, 4, 5]);
        assert_eq!(one.frontier().indices(), vec![4, 5]);
        let two = grow_field_aware_region(&topo, &layout, &seeds, 2).unwrap();
        assert_eq!(two.region().indices(), vec![0, 1, 2, 4, 5, 6, 7, 8]);
        assert!(!two.region().contains(3)); // do not auto-merge temperature
        assert_eq!(two.stats().region_blocks, 3);
        assert_eq!(two.stats().hops_completed, 2);

        // Independent scalar CSR adjacency+block-closure reference.
        let mut reference = close_to_field_blocks(&layout, &seeds).unwrap();
        let mut frontier = reference.clone();
        for _ in 0..2 {
            let mut nbr = frontier.clone();
            for row in 0..a.nrows() {
                for k in a.row_ptr()[row] as usize..a.row_ptr()[row + 1] as usize {
                    let col = a.col_idx()[k] as usize;
                    if frontier.contains(row) {
                        nbr.set(col, true).unwrap();
                    }
                    if frontier.contains(col) {
                        nbr.set(row, true).unwrap();
                    }
                }
            }
            let mut next = close_to_field_blocks(&layout, &nbr).unwrap();
            next.and_not_assign(&reference).unwrap();
            reference.union_assign(&next).unwrap();
            frontier = next;
        }
        assert_eq!(two.region(), &reference);

        let local = two.local_pattern(&topo).unwrap();
        assert_eq!(local.global_nodes(), &[0, 1, 2, 4, 5, 6, 7, 8]);
        assert_eq!(local.nrows(), 8);
        // Verify every local symbolic row against a direct CSR restriction.
        for (li, &global) in local.global_nodes().iter().enumerate() {
            let global = global as usize;
            let first = local.row_ptr()[li] as usize;
            let last = local.row_ptr()[li + 1] as usize;
            let actual: Vec<_> = local.col_idx()[first..last]
                .iter()
                .map(|&j| local.global_nodes()[j as usize])
                .collect();
            let mut expected: Vec<_> = (a.row_ptr()[global] as usize
                ..a.row_ptr()[global + 1] as usize)
                .map(|k| a.col_idx()[k])
                .filter(|&col| reference.contains(col as usize))
                .collect();
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn owner_checks_reject_split_blocks_not_mixed_entities() {
        let (layout, _) = mixed_fixture();
        assert_eq!(
            field_block_owners(&layout, &[0, 0, 0, 1, 1, 1, 2, 2, 2], 3).unwrap(),
            [0, 1, 1, 2]
        );
        assert!(field_block_owners(&layout, &[0, 0, 1, 1, 1, 1, 2, 2, 2], 3).is_err());
        assert!(field_block_owners(&layout, &[0, 0, 0, 1, 1, 1, 3, 3, 3], 3).is_err());
        assert!(field_block_owners(&layout, &[0; 8], 3).is_err());
        assert!(field_block_owners(&layout, &[0; 9], 0).is_err());
    }

    #[test]
    fn growth_rejects_mismatched_dimensions_and_empty_seeds() {
        let (layout, a) = mixed_fixture();
        let topo = AbtmDualTopology::from_csr32(&a).unwrap();
        let none = grow_field_aware_region(&topo, &layout, &DofMask::new(9), 4).unwrap();
        assert_eq!(none.stats().region_dofs, 0);
        assert_eq!(none.stats().hops_completed, 0);
        assert!(grow_field_aware_region(&topo, &layout, &DofMask::new(8), 1).is_err());
        let rectangular =
            Csr32Matrix::new(2, 3, vec![0, 1, 2], vec![1, 2], vec![1.0, 1.0]).unwrap();
        let rect_topo = AbtmDualTopology::from_csr32(&rectangular).unwrap();
        assert!(grow_field_aware_region(&rect_topo, &layout, &DofMask::new(9), 1).is_err());
    }
}
