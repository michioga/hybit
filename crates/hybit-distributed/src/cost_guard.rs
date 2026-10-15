//! H1-C3D: deterministic calibration identity, fail-closed, opt-in selector.
//!
//! This is a non-cryptographic *accidental mismatch* guard, not an attestation
//! of performance, files, or hardware. Calibration provenance must be supplied
//! by the caller; no timing is inferred from graph sparsity or rank-local time.

use crate::{
    cost_aware_contiguous_partition, ContiguousPartition, CostAwarePartitionKind,
    CostAwarePartitionOptions, CostAwarePartitionReport, DistributedTopologyError,
    PartitionCostCalibration, RankId,
};
use hybit_matrix::{Csr32Matrix, FieldAwareDofLayout};

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
struct StableDigest(u64);
impl StableDigest {
    fn new(domain: &[u8]) -> Self {
        let mut digest = Self(FNV_BASIS);
        digest.bytes(domain);
        digest
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }
    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }
    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }
    fn text(&mut self, value: &str) {
        self.usize(value.len());
        self.bytes(value.as_bytes());
    }
    fn finish(self) -> u64 {
        self.0
    }
}

/// Ordered calibration identity. All keys must match before the graph
/// preparation can run; numerical CSR values are deliberately included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionCalibrationIdentity {
    pub matrix_pattern: u64,
    pub matrix_values: u64,
    pub physical_layout: u64,
    pub environment: u64,
    pub ranks: RankId,
    pub search_radius_blocks: usize,
    pub baseline_kind: CostAwarePartitionKind,
}

/// A caller-declared calibration with a fingerprint of the conditions under
/// which the *same baseline and graph candidate* were benchmarked.
/// No runtime hardware discovery is performed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedPartitionCostCalibration {
    pub identity: PartitionCalibrationIdentity,
    pub timings: PartitionCostCalibration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationGuardDecision {
    Missing,
    ScopeMismatch,
    Matched,
}

#[derive(Clone, Debug)]
pub struct ScopedCostPartitionResult {
    pub partition: ContiguousPartition,
    pub report: CostAwarePartitionReport,
    pub guard: CalibrationGuardDecision,
    /// Fingerprint of the *actual* matrix, layout, environment and baseline.
    pub identity: PartitionCalibrationIdentity,
}

fn validate_environment_tag(environment_tag: &str) -> Result<(), DistributedTopologyError> {
    if environment_tag.is_empty()
        || environment_tag.len() > 256
        || environment_tag.chars().any(char::is_control)
    {
        return Err(DistributedTopologyError::InvalidPartition(
            "calibration environment tag must be 1..=256 bytes without control characters",
        ));
    }
    Ok(())
}

fn identity(
    matrix: &Csr32Matrix,
    layout: &FieldAwareDofLayout,
    ranks: RankId,
    radius: usize,
    baseline_kind: CostAwarePartitionKind,
    environment_tag: &str,
) -> Result<PartitionCalibrationIdentity, DistributedTopologyError> {
    validate_environment_tag(environment_tag)?;
    let mut pattern = StableDigest::new(b"HYBIT-H1C3D-PATTERN-v1");
    pattern.usize(matrix.nrows());
    pattern.usize(matrix.ncols());
    pattern.usize(matrix.nnz());
    for &v in matrix.row_ptr() {
        pattern.u64(u64::from(v));
    }
    for &v in matrix.col_idx() {
        pattern.u64(u64::from(v));
    }

    let mut numeric = StableDigest::new(b"HYBIT-H1C3D-VALUES-v1");
    numeric.usize(matrix.values().len());
    for &value in matrix.values() {
        numeric.u64(value.to_bits());
    }

    let mut physical = StableDigest::new(b"HYBIT-H1C3D-LAYOUT-v1");
    physical.u64(layout.total_dofs());
    physical.usize(layout.blocks().len());
    for block in layout.blocks() {
        physical.u64(block.entity_kind as u32 as u64);
        physical.u64(block.entity_id);
        physical.u64(u64::from(block.field_id));
        physical.u64(u64::from(block.components));
    }

    let mut environment = StableDigest::new(b"HYBIT-H1C3D-ENVIRONMENT-v1");
    environment.text(environment_tag);
    Ok(PartitionCalibrationIdentity {
        matrix_pattern: pattern.finish(),
        matrix_values: numeric.finish(),
        physical_layout: physical.finish(),
        environment: environment.finish(),
        ranks,
        search_radius_blocks: radius,
        baseline_kind,
    })
}

fn baseline_snapshot(
    matrix: &Csr32Matrix,
    layout: &FieldAwareDofLayout,
    ranks: RankId,
    mut options: CostAwarePartitionOptions,
) -> Result<(ContiguousPartition, CostAwarePartitionReport), DistributedTopologyError> {
    // Deliberately suppress all old, unscoped calibration on this entry point.
    options.calibration = None;
    cost_aware_contiguous_partition(matrix, layout, ranks, options)
}

/// Construct an identity for externally measured *matching* baselines/graphs.
/// This does not measure or verify `timings`; their provenance is the caller's
/// responsibility. A different search radius, baseline, CSR values, or machine
/// tag yields a different key.
#[allow(clippy::too_many_arguments)]
pub fn capture_partition_calibration(
    matrix: &Csr32Matrix,
    layout: &FieldAwareDofLayout,
    ranks: RankId,
    options: CostAwarePartitionOptions,
    environment_tag: &str,
    timings: PartitionCostCalibration,
) -> Result<ScopedPartitionCostCalibration, DistributedTopologyError> {
    if timings.baseline_spmv_ns == 0 || timings.graph_spmv_ns == 0 {
        return Err(DistributedTopologyError::InvalidPartition(
            "calibration SpMV values must be positive",
        ));
    }
    let (_, report) = baseline_snapshot(matrix, layout, ranks, options)?;
    let identity = identity(
        matrix,
        layout,
        ranks,
        options.search_radius_blocks,
        report.baseline_kind,
        environment_tag,
    )?;
    Ok(ScopedPartitionCostCalibration { identity, timings })
}

/// Always recompute a block-safe baseline; never trust the caller's old
/// unscoped `options.calibration`. Only exactly matching scoped evidence can
/// authorize graph preparation. Mismatch is a normal, fail-closed result.
#[allow(clippy::too_many_arguments)]
pub fn cost_aware_contiguous_partition_scoped(
    matrix: &Csr32Matrix,
    layout: &FieldAwareDofLayout,
    ranks: RankId,
    options: CostAwarePartitionOptions,
    environment_tag: &str,
    scoped: Option<&ScopedPartitionCostCalibration>,
) -> Result<ScopedCostPartitionResult, DistributedTopologyError> {
    let (fallback, mut report) = baseline_snapshot(matrix, layout, ranks, options)?;
    let expected = identity(
        matrix,
        layout,
        ranks,
        options.search_radius_blocks,
        report.baseline_kind,
        environment_tag,
    )?;
    let guard = match scoped {
        None => CalibrationGuardDecision::Missing,
        Some(value) if value.identity == expected => CalibrationGuardDecision::Matched,
        Some(_) => CalibrationGuardDecision::ScopeMismatch,
    };
    if guard != CalibrationGuardDecision::Matched {
        // The baseline snapshot already has a NoCalibration/early-exit report;
        // `guard` disambiguates a missing calibration from scope mismatch.
        return Ok(ScopedCostPartitionResult {
            partition: fallback,
            report,
            guard,
            identity: expected,
        });
    }
    let mut calibrated_options = options;
    calibrated_options.calibration = Some(scoped.expect("matched scope").timings);
    let (chosen, calibrated_report) =
        cost_aware_contiguous_partition(matrix, layout, ranks, calibrated_options)?;
    report = calibrated_report;
    Ok(ScopedCostPartitionResult {
        partition: chosen,
        report,
        guard,
        identity: expected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hybit_matrix::{DofBlock, DofEntityKind};

    fn fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
        let layout = FieldAwareDofLayout::new(
            (0..12)
                .map(|i| DofBlock {
                    entity_kind: DofEntityKind::Node,
                    entity_id: i,
                    field_id: 0,
                    components: 1,
                })
                .collect(),
        )
        .unwrap();
        let mut ptr = vec![0u32];
        let mut col = Vec::new();
        let mut val = Vec::new();
        for row in 0..12usize {
            for c in [row, (row + 1) % 12] {
                col.push(c as u32);
                val.push(if c == row { 2.0 } else { -1.0 });
            }
            ptr.push(col.len() as u32);
        }
        (layout, Csr32Matrix::new(12, 12, ptr, col, val).unwrap())
    }
    fn opts() -> CostAwarePartitionOptions {
        CostAwarePartitionOptions {
            expected_spmv_calls: 10_000,
            search_radius_blocks: 2,
            max_owned_dof_imbalance: 2.0,
            max_local_nnz_imbalance: 2.0,
            ..CostAwarePartitionOptions::default()
        }
    }
    fn calibration() -> PartitionCostCalibration {
        PartitionCostCalibration {
            baseline_spmv_ns: 1000,
            graph_spmv_ns: 900,
            graph_prepare_budget_ns: 100,
        }
    }
    #[test]
    fn matching_scope_is_accepted_and_offsets_are_deterministic() {
        let (layout, matrix) = fixture();
        let scoped = capture_partition_calibration(
            &matrix,
            &layout,
            4,
            opts(),
            "win11-msmpi10-r7",
            calibration(),
        )
        .unwrap();
        let a = cost_aware_contiguous_partition_scoped(
            &matrix,
            &layout,
            4,
            opts(),
            "win11-msmpi10-r7",
            Some(&scoped),
        )
        .unwrap();
        let b = cost_aware_contiguous_partition_scoped(
            &matrix,
            &layout,
            4,
            opts(),
            "win11-msmpi10-r7",
            Some(&scoped),
        )
        .unwrap();
        assert_eq!(a.guard, CalibrationGuardDecision::Matched);
        assert_eq!(a.partition, b.partition);
        assert_eq!(a.report.decision, b.report.decision);
    }
    #[test]
    fn changed_radius_and_ranks_reject_before_graph() {
        let (layout, matrix) = fixture();
        let scoped =
            capture_partition_calibration(&matrix, &layout, 4, opts(), "win11", calibration())
                .unwrap();
        let mut radius = opts();
        radius.search_radius_blocks += 1;
        let a = cost_aware_contiguous_partition_scoped(
            &matrix,
            &layout,
            4,
            radius,
            "win11",
            Some(&scoped),
        )
        .unwrap();
        let b = cost_aware_contiguous_partition_scoped(
            &matrix,
            &layout,
            2,
            opts(),
            "win11",
            Some(&scoped),
        )
        .unwrap();
        for rejected in [&a, &b] {
            assert_eq!(rejected.guard, CalibrationGuardDecision::ScopeMismatch);
            assert!(rejected.report.local_graph_prepare_ns.is_none());
            assert!(rejected.report.graph_candidate.is_none());
        }
    }
    #[test]
    fn changed_environment_and_numeric_values_reject() {
        let (layout, matrix) = fixture();
        let scoped =
            capture_partition_calibration(&matrix, &layout, 4, opts(), "win11", calibration())
                .unwrap();
        let wrong_env = cost_aware_contiguous_partition_scoped(
            &matrix,
            &layout,
            4,
            opts(),
            "ubuntu",
            Some(&scoped),
        )
        .unwrap();
        let mut vals = matrix.values().to_vec();
        vals[0] = 3.0;
        let changed = Csr32Matrix::new(
            matrix.nrows(),
            matrix.ncols(),
            matrix.row_ptr().to_vec(),
            matrix.col_idx().to_vec(),
            vals,
        )
        .unwrap();
        let wrong_vals = cost_aware_contiguous_partition_scoped(
            &changed,
            &layout,
            4,
            opts(),
            "win11",
            Some(&scoped),
        )
        .unwrap();
        assert_eq!(wrong_env.guard, CalibrationGuardDecision::ScopeMismatch);
        assert_eq!(wrong_vals.guard, CalibrationGuardDecision::ScopeMismatch);
        assert!(wrong_vals.report.local_graph_prepare_ns.is_none());
    }
    #[test]
    fn changed_field_metadata_and_pattern_reject() {
        let (layout, matrix) = fixture();
        let scoped =
            capture_partition_calibration(&matrix, &layout, 4, opts(), "win11", calibration())
                .unwrap();
        let mut blocks = layout.blocks().to_vec();
        blocks[0].field_id = 42;
        let alternative = FieldAwareDofLayout::new(blocks).unwrap();
        let wrong_field = cost_aware_contiguous_partition_scoped(
            &matrix,
            &alternative,
            4,
            opts(),
            "win11",
            Some(&scoped),
        )
        .unwrap();
        let mut col = matrix.col_idx().to_vec();
        col[1] = 6;
        let changed = Csr32Matrix::new(
            matrix.nrows(),
            matrix.ncols(),
            matrix.row_ptr().to_vec(),
            col,
            matrix.values().to_vec(),
        )
        .unwrap();
        let wrong_pattern = cost_aware_contiguous_partition_scoped(
            &changed,
            &layout,
            4,
            opts(),
            "win11",
            Some(&scoped),
        )
        .unwrap();
        assert_eq!(wrong_field.guard, CalibrationGuardDecision::ScopeMismatch);
        assert_eq!(wrong_pattern.guard, CalibrationGuardDecision::ScopeMismatch);
    }
    #[test]
    fn no_scoped_evidence_ignores_legacy_timing() {
        let (layout, matrix) = fixture();
        let mut options = opts();
        options.calibration = Some(calibration());
        let result =
            cost_aware_contiguous_partition_scoped(&matrix, &layout, 4, options, "win11", None)
                .unwrap();
        assert_eq!(result.guard, CalibrationGuardDecision::Missing);
        assert!(result.report.local_graph_prepare_ns.is_none());
    }
    #[test]
    fn invalid_environment_and_empty_measurement_rejected() {
        let (layout, matrix) = fixture();
        let invalid = capture_partition_calibration(&matrix, &layout, 4, opts(), "", calibration());
        assert!(invalid.is_err());
        let mut zero = calibration();
        zero.baseline_spmv_ns = 0;
        assert!(capture_partition_calibration(&matrix, &layout, 4, opts(), "win11", zero).is_err());
    }
}
