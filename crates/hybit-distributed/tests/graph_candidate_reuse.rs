//! H1-C3J: prepared-Graph replay must preserve the exact legacy decision.
use hybit_distributed::{
    CalibrationGuardDecision, CostAwarePartitionDecision, CostAwarePartitionKind,
    CostAwarePartitionOptions, PartitionCostCalibration, capture_partition_calibration,
    cost_aware_contiguous_partition_scoped, cost_aware_contiguous_partition_scoped_prepared,
    prepare_cost_graph_candidate_scoped,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};

const ENV: &str = "h1c3j-test-msmpi-cpu";

fn fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
    let components = [3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1];
    let layout = FieldAwareDofLayout::new(
        (0..12u64)
            .map(|i| DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: i,
                field_id: 0,
                components: components[i as usize],
            })
            .collect(),
    )
    .unwrap();
    let n = layout.total_dofs() as usize;
    let mut ptr = vec![0u32];
    let mut col = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        let block = layout.locate(row as u64).unwrap().block_index;
        let (start, end) = if block < 5 { (0, 11) } else { (11, n) };
        for c in start..end {
            col.push(c as u32);
            values.push(if c == row { 5.0 } else { -0.25 });
        }
        ptr.push(values.len() as u32);
    }
    let matrix = Csr32Matrix::new(n, n, ptr, col, values).unwrap();
    (layout, matrix)
}
fn options() -> CostAwarePartitionOptions {
    CostAwarePartitionOptions {
        expected_spmv_calls: 10_000,
        search_radius_blocks: 5,
        max_owned_dof_imbalance: 2.0,
        max_local_nnz_imbalance: 2.0,
        ..CostAwarePartitionOptions::default()
    }
}
fn timings() -> PartitionCostCalibration {
    PartitionCostCalibration {
        baseline_spmv_ns: 100_000,
        graph_spmv_ns: 50_000,
        graph_prepare_budget_ns: 200_000,
    }
}

#[test]
fn reuse_matches_legacy_graph_partition_decision_and_telemetry() {
    let (layout, matrix) = fixture();
    let candidate =
        prepare_cost_graph_candidate_scoped(&matrix, &layout, 2, options(), ENV).unwrap();
    let scoped =
        capture_partition_calibration(&matrix, &layout, 2, options(), ENV, timings()).unwrap();
    let legacy =
        cost_aware_contiguous_partition_scoped(&matrix, &layout, 2, options(), ENV, Some(&scoped))
            .unwrap();
    let reused = cost_aware_contiguous_partition_scoped_prepared(
        &matrix,
        &layout,
        2,
        options(),
        ENV,
        Some(&scoped),
        &candidate,
    )
    .unwrap();
    assert_eq!(legacy.guard, CalibrationGuardDecision::Matched);
    assert_eq!(reused.guard, CalibrationGuardDecision::Matched);
    assert_eq!(
        legacy.report.decision,
        CostAwarePartitionDecision::GraphSelected
    );
    assert_eq!(reused.report.selected_kind, CostAwarePartitionKind::Graph);
    assert_eq!(legacy.partition, reused.partition);
    assert_eq!(candidate.partition(), &reused.partition);
    assert_eq!(legacy.report.baseline, reused.report.baseline);
    assert_eq!(legacy.report.selected, reused.report.selected);
    assert_eq!(legacy.report.graph_candidate, reused.report.graph_candidate);
    assert_eq!(
        legacy.report.modeled_net_savings_ns,
        reused.report.modeled_net_savings_ns
    );
    assert_eq!(
        reused.report.local_graph_prepare_ns,
        Some(candidate.preparation_ns())
    );
}

#[test]
fn changed_numeric_values_and_environment_fail_closed() {
    let (layout, matrix) = fixture();
    let candidate =
        prepare_cost_graph_candidate_scoped(&matrix, &layout, 2, options(), ENV).unwrap();
    let scoped =
        capture_partition_calibration(&matrix, &layout, 2, options(), ENV, timings()).unwrap();
    let mut values = matrix.values().to_vec();
    values[0] += 0.125;
    let changed = Csr32Matrix::new(
        matrix.nrows(),
        matrix.ncols(),
        matrix.row_ptr().to_vec(),
        matrix.col_idx().to_vec(),
        values,
    )
    .unwrap();
    let result = cost_aware_contiguous_partition_scoped_prepared(
        &changed,
        &layout,
        2,
        options(),
        ENV,
        Some(&scoped),
        &candidate,
    )
    .unwrap();
    assert_eq!(result.guard, CalibrationGuardDecision::ScopeMismatch);
    assert_ne!(result.report.selected_kind, CostAwarePartitionKind::Graph);
    assert!(result.report.graph_candidate.is_none());
    let result = cost_aware_contiguous_partition_scoped_prepared(
        &matrix,
        &layout,
        2,
        options(),
        "different-runtime",
        Some(&scoped),
        &candidate,
    )
    .unwrap();
    assert_eq!(result.guard, CalibrationGuardDecision::ScopeMismatch);
}

#[test]
fn changed_radius_rejects_even_with_newly_matching_timing_scope() {
    let (layout, matrix) = fixture();
    let candidate =
        prepare_cost_graph_candidate_scoped(&matrix, &layout, 2, options(), ENV).unwrap();
    let mut changed = options();
    changed.search_radius_blocks += 1;
    let scoped =
        capture_partition_calibration(&matrix, &layout, 2, changed, ENV, timings()).unwrap();
    let result = cost_aware_contiguous_partition_scoped_prepared(
        &matrix,
        &layout,
        2,
        changed,
        ENV,
        Some(&scoped),
        &candidate,
    )
    .unwrap();
    assert_eq!(result.guard, CalibrationGuardDecision::ScopeMismatch);
    assert_ne!(result.report.selected_kind, CostAwarePartitionKind::Graph);
}

#[test]
fn missing_timing_scope_retains_reference_baseline() {
    let (layout, matrix) = fixture();
    let candidate =
        prepare_cost_graph_candidate_scoped(&matrix, &layout, 2, options(), ENV).unwrap();
    let result = cost_aware_contiguous_partition_scoped_prepared(
        &matrix,
        &layout,
        2,
        options(),
        ENV,
        None,
        &candidate,
    )
    .unwrap();
    assert_eq!(result.guard, CalibrationGuardDecision::Missing);
    assert_eq!(
        result.report.decision,
        CostAwarePartitionDecision::NoCalibration
    );
    assert_ne!(result.report.selected_kind, CostAwarePartitionKind::Graph);
}

#[test]
fn zero_reuse_is_still_rejected_despite_prepared_proposal() {
    let (layout, matrix) = fixture();
    let candidate =
        prepare_cost_graph_candidate_scoped(&matrix, &layout, 2, options(), ENV).unwrap();
    let mut zero = options();
    zero.expected_spmv_calls = 0;
    let scoped = capture_partition_calibration(&matrix, &layout, 2, zero, ENV, timings()).unwrap();
    let result = cost_aware_contiguous_partition_scoped_prepared(
        &matrix,
        &layout,
        2,
        zero,
        ENV,
        Some(&scoped),
        &candidate,
    )
    .unwrap();
    assert_eq!(
        result.report.decision,
        CostAwarePartitionDecision::NoExpectedReuse
    );
    assert_ne!(result.report.selected_kind, CostAwarePartitionKind::Graph);
}
