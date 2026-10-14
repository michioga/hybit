//! H1-C1: explicitly prepared mixed-field ABTM symbolic/numeric-address plans.
//!
//! A single `AbtmDualTopology` is built for a borrowed, immutable CSR matrix.
//! Any number of field-block-closed Schwarz candidate regions may then reuse it.
//! Numeric values can change in *the same CSR structural ordering* without
//! rebuilding the prepared plans. No factorization, MPI Halo, or automatic
//! Krylov/preconditioner routing is performed here.

use crate::{
    field_block_owners, grow_field_aware_region, prepare_local_numeric_plan, AbtmDualTopology,
    AbtmDualTopologyStats, AbtmLocalNumericPlan, AbtmLocalNumericPlanStats,
    AbtmLocalSubmatrixPattern, Csr32Matrix, DofMask, FieldAwareDofLayout, FieldAwareRegionGrowth,
    FieldAwareRegionStats,
};
use hybit_core::HybitError;

/// Prepared ABTM topology and physical field layout for one immutable CSR
/// structure. The borrow ties this object to the input matrix's lifetime;
/// it intentionally does not duplicate its CSR numerical values.
#[derive(Debug)]
pub struct PreparedFieldAwareTopology<'matrix> {
    matrix: &'matrix Csr32Matrix,
    layout: FieldAwareDofLayout,
    topology: AbtmDualTopology,
}

/// One reusable symbolic/local-numeric addressing plan. Contains no copy of
/// matrix numerical values and performs no direct factorization.
#[derive(Clone, Debug)]
pub struct PreparedFieldAwareLocalRegion {
    growth: FieldAwareRegionGrowth,
    numeric_plan: AbtmLocalNumericPlan,
}

impl<'matrix> PreparedFieldAwareTopology<'matrix> {
    /// Build the ABTM forward+transpose structure *once* for this CSR matrix.
    /// The layout is copied so callers may release their layout object.
    pub fn new(
        matrix: &'matrix Csr32Matrix,
        layout: &FieldAwareDofLayout,
    ) -> Result<Self, HybitError> {
        matrix.validate()?;
        layout.validate_square_matrix_size(matrix.nrows(), matrix.ncols())?;
        let topology = AbtmDualTopology::from_csr32(matrix)?;
        Ok(Self {
            matrix,
            layout: layout.clone(),
            topology,
        })
    }

    pub fn topology(&self) -> &AbtmDualTopology {
        &self.topology
    }

    pub fn layout(&self) -> &FieldAwareDofLayout {
        &self.layout
    }

    pub fn topology_stats(&self) -> AbtmDualTopologyStats {
        self.topology.stats()
    }

    /// Reuse the same ABTM topology to prepare a different Schwarz candidate.
    /// Hop semantics are exactly those of H1-B's `grow_field_aware_region`.
    pub fn prepare_region(
        &self,
        seeds: &DofMask,
        hops: usize,
    ) -> Result<PreparedFieldAwareLocalRegion, HybitError> {
        let growth = grow_field_aware_region(&self.topology, &self.layout, seeds, hops)?;
        let numeric_plan =
            prepare_local_numeric_plan(&self.topology, self.matrix, growth.region())?;
        Ok(PreparedFieldAwareLocalRegion {
            growth,
            numeric_plan,
        })
    }

    /// Multiple independently selected regions share the single prepared
    /// topology. An error in any candidate returns `Err` rather than silently
    /// accepting a partial set of regions.
    pub fn prepare_regions(
        &self,
        seed_masks: &[DofMask],
        hops: usize,
    ) -> Result<Vec<PreparedFieldAwareLocalRegion>, HybitError> {
        seed_masks
            .iter()
            .map(|seeds| self.prepare_region(seeds, hops))
            .collect()
    }

    /// Validate owner ranks at **physical-field-block** granularity. This is
    /// metadata validation only: no distributed HaloPlan is constructed here.
    pub fn validate_block_owners(
        &self,
        dof_owners: &[u32],
        ranks: u32,
    ) -> Result<Vec<u32>, HybitError> {
        field_block_owners(&self.layout, dof_owners, ranks)
    }
}

impl PreparedFieldAwareLocalRegion {
    pub fn region(&self) -> &DofMask {
        self.growth.region()
    }

    pub fn growth_stats(&self) -> FieldAwareRegionStats {
        self.growth.stats()
    }

    pub fn pattern(&self) -> &AbtmLocalSubmatrixPattern {
        self.numeric_plan.pattern()
    }

    pub fn numeric_plan_stats(&self) -> AbtmLocalNumericPlanStats {
        self.numeric_plan.stats()
    }

    /// Gather local values for unchanged CSR row-pointer and column-index
    /// arrays. The caller is responsible for using the same structural CSR
    /// ordering; the length is checked but semantic structure cannot be
    /// verified from a value-only slice. Refactorization remains separate.
    pub fn gather_values(&self, csr_values: &[f64]) -> Result<Vec<f64>, HybitError> {
        self.numeric_plan.gather_values(csr_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DofBlock, DofEntityKind};

    fn fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
        use DofEntityKind::{Multiplier, Node};
        let blocks = [
            (Node, 1, 1, 3),       // displacement 0..3
            (Node, 1, 2, 1),       // temperature 3..4, distinct field
            (Node, 2, 1, 2),       // shell 4..6
            (Multiplier, 7, 3, 3), // contact multiplier 6..9
        ];
        let layout = FieldAwareDofLayout::new(
            blocks
                .iter()
                .map(|&(entity_kind, entity_id, field_id, components)| DofBlock {
                    entity_kind,
                    entity_id,
                    field_id,
                    components,
                })
                .collect(),
        )
        .unwrap();
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); 9];
        for (i, row) in rows.iter_mut().enumerate() {
            row.push((i as u32, 10.0 + i as f64));
        }
        rows[0].push((4, 0.0)); // stored zero remains structural
        rows[6].push((5, -2.0)); // directed coupling to multiplier
        let mut row_ptr = vec![0_u32];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for row in rows {
            for (c, v) in row {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let a = Csr32Matrix::new(9, 9, row_ptr, col_idx, values).unwrap();
        (layout, a)
    }

    fn assert_local_matches_direct_csr(
        a: &Csr32Matrix,
        region: &PreparedFieldAwareLocalRegion,
        values: &[f64],
    ) {
        let pat = region.pattern();
        let got = region.gather_values(values).unwrap();
        let mut expected_entries = Vec::new();
        /*
        for &global_row in pat.global_nodes() {
            let mut row = Vec::new();
            for k in a.row_ptr()[global_row as usize] as usize
                ..a.row_ptr()[global_row as usize + 1] as usize
            {
                let global_col = a.col_idx()[k];
                if let Some(local_col) = pat.global_nodes().iter().position(|&j| j == global_col) {
                    row.push((local_col as u32, values[k]));
                }
            }
            row.sort_by_key(|&(col, _)| col);
            expected_entries.extend(row);
        }
        */
        for &global_row in pat.global_nodes() {
            let mut row = Vec::new();

            let start = a.row_ptr()[global_row as usize] as usize;
            let end = a.row_ptr()[global_row as usize + 1] as usize;

            for (&global_col, &value) in a.col_idx()[start..end].iter().zip(&values[start..end]) {
                if let Some(local_col) = pat.global_nodes().iter().position(|&j| j == global_col) {
                    row.push((local_col as u32, value));
                }
            }

            row.sort_by_key(|&(col, _)| col);
            expected_entries.extend(row);
        }
        let actual_entries: Vec<_> = pat.col_idx().iter().copied().zip(got).collect();
        assert_eq!(actual_entries, expected_entries);
    }

    #[test]
    fn prepared_regions_reuse_topology_and_match_direct_csr() {
        let (layout, a) = fixture();
        let prepared = PreparedFieldAwareTopology::new(&a, &layout).unwrap();
        let initial_bytes = prepared.topology_stats().total_metadata_bytes();
        assert!(initial_bytes > 0);
        let seeds = [
            DofMask::from_indices(9, &[0]).unwrap(),
            DofMask::from_indices(9, &[8]).unwrap(),
            DofMask::new(9),
        ];
        let plans = prepared.prepare_regions(&seeds, 1).unwrap();
        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].region().indices(), vec![0, 1, 2, 4, 5]);
        assert_eq!(plans[1].region().indices(), vec![4, 5, 6, 7, 8]);
        assert!(plans[2].region().is_empty());
        for plan in &plans {
            assert_local_matches_direct_csr(&a, plan, a.values());
            let updated: Vec<f64> = a.values().iter().map(|v| 3.0 * v + 1.0).collect();
            assert_local_matches_direct_csr(&a, plan, &updated);
            assert!(plan.gather_values(&updated[..updated.len() - 1]).is_err());
        }
        assert_eq!(
            prepared.topology_stats().total_metadata_bytes(),
            initial_bytes
        );
        assert_eq!(
            prepared.prepare_region(&seeds[0], 1).unwrap().pattern(),
            plans[0].pattern()
        );
        // Same physical entity but a different field is not silently added.
        assert!(!plans[0].region().contains(3));
    }

    #[test]
    fn prepared_regions_reject_dimension_errors_and_split_owners() {
        let (layout, a) = fixture();
        let prepared = PreparedFieldAwareTopology::new(&a, &layout).unwrap();
        assert!(prepared.prepare_region(&DofMask::new(8), 1).is_err());
        assert!(prepared
            .prepare_regions(&[DofMask::new(9), DofMask::new(8)], 1)
            .is_err());
        let mut bad_owners = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        assert_eq!(
            prepared.validate_block_owners(&bad_owners, 3).unwrap(),
            vec![0, 1, 1, 2]
        );
        bad_owners[1] = 1;
        assert!(prepared.validate_block_owners(&bad_owners, 3).is_err());
        let short_layout = FieldAwareDofLayout::new(vec![DofBlock {
            entity_kind: DofEntityKind::Node,
            entity_id: 42,
            field_id: 1,
            components: 8,
        }])
        .unwrap();
        assert!(PreparedFieldAwareTopology::new(&a, &short_layout).is_err());
    }
}
