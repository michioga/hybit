//! H1-C3C: conservative, explicitly requested cost-aware contiguous partitioning.
//!
//! In particular, *local* elapsed time never participates in the partition
//! decision. A native MPI caller must supply the same calibrated timings on
//! every rank or select on one root and broadcast its decision and offsets.
//! Calibration is evidence from earlier representative runs, not a prediction
//! derived from fewer ghosts. No automatic solver routing is changed.

use crate::{
    build_contiguous_halo_plans, field_aligned_contiguous_partition,
    field_graph_contiguous_partition, partition_telemetry, validate_field_contiguous_partition,
    ContiguousPartition, DistributedTopologyError, PartitionTelemetry, RankId,
};
use hybit_matrix::{Csr32Matrix, FieldAwareDofLayout};
use std::time::Instant;

/// Shared, previously measured maximum-rank SpMV timings and a conservative
/// *estimated* cost of graph preparation. Nanoseconds; all ranks must agree.
/// Supply only when a representative calibration exists; never synthesize
/// timings from the number of ghost DOFs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionCostCalibration {
    pub baseline_spmv_ns: u64,
    pub graph_spmv_ns: u64,
    pub graph_prepare_budget_ns: u64,
}

/// An explicitly invoked experiment; not a global/default solver policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostAwarePartitionOptions {
    pub expected_spmv_calls: u64,
    pub calibration: Option<PartitionCostCalibration>,
    pub search_radius_blocks: usize,
    pub max_owned_dof_imbalance: f64,
    pub max_local_nnz_imbalance: f64,
    /// Positive net savings required by the shared *estimated* cost model.
    pub minimum_net_savings_ns: u64,
}

impl Default for CostAwarePartitionOptions {
    fn default() -> Self {
        Self {
            expected_spmv_calls: 0,
            calibration: None,
            search_radius_blocks: 5,
            max_owned_dof_imbalance: 1.15,
            max_local_nnz_imbalance: 1.15,
            minimum_net_savings_ns: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostAwarePartitionKind {
    Equal,
    Field,
    Graph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostAwarePartitionDecision {
    SingleRank,
    NoExpectedReuse,
    NoCalibration,
    ZeroSearchRadius,
    NoCalibratedSpeedup,
    UnamortizedPreparation,
    GraphProposalRejected,
    GraphNotDistinct,
    GraphDoesNotDominateBaseline,
    LoadImbalanceLimit,
    GraphSelected,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CostAwarePartitionReport {
    pub baseline_kind: CostAwarePartitionKind,
    pub selected_kind: CostAwarePartitionKind,
    pub decision: CostAwarePartitionDecision,
    pub baseline: PartitionTelemetry,
    pub selected: PartitionTelemetry,
    /// Includes rejected H1-C3 graph proposal, when ABTM was prepared.
    pub graph_candidate: Option<PartitionTelemetry>,
    /// Actual local wall time is diagnostic only, NEVER an input to selection.
    pub local_graph_prepare_ns: Option<u128>,
    /// Estimated savings after prep cost, based only on shared calibration.
    pub modeled_net_savings_ns: Option<u128>,
}

fn structurally_dominates(a: &PartitionTelemetry, b: &PartitionTelemetry) -> bool {
    a.communication_volume <= b.communication_volume
        && a.cut_nnz <= b.cut_nnz
        && a.directional_peer_relations <= b.directional_peer_relations
        && a.max_local_nnz <= b.max_local_nnz
        && a.max_owned_dofs <= b.max_owned_dofs
}

/// Returns a *field-block-valid* contiguous partition and its decision report.
///
/// Equal is considered only if no physical field block is split; otherwise
/// Field is the baseline. Equal can replace Field only when it dominates in
/// exact CSR Halo/nnz telemetry. ABTM is not built before the cheap model gate.
/// The graph candidate is allowed only if it (a) has a strictly favorable
/// calibrated cost model, (b) does not regress exact halo/cut/peer metrics
/// against the chosen baseline, and (c) meets load-imbalance caps.
///
/// Calibration is not a guarantee of wall-clock speedup. In an MPI caller,
/// provide identical options on every rank; do not use rank-local timings as
/// calibration. The current API assumes global CSR is available in this
/// *experimental* preparation step and does not redistribute MPI-owned rows.
/// Shared legacy selector; an opt-in H1-C3J caller may reuse a prepared Graph
/// proposal after a matching scoped guard has checked its provenance.
/// No production or default selection policy is modified.
pub fn cost_aware_contiguous_partition(
    matrix: &Csr32Matrix,
    layout: &FieldAwareDofLayout,
    ranks: RankId,
    options: CostAwarePartitionOptions,
) -> Result<(ContiguousPartition, CostAwarePartitionReport), DistributedTopologyError> {
    cost_aware_contiguous_partition_with_graph(matrix, layout, ranks, options, None)
}

#[allow(clippy::too_many_lines)]
pub(crate) fn cost_aware_contiguous_partition_with_graph(
    matrix: &Csr32Matrix,
    layout: &FieldAwareDofLayout,
    ranks: RankId,
    options: CostAwarePartitionOptions,
    prepared: Option<(
        &ContiguousPartition,
        &crate::FieldGraphPartitionComparison,
        u128,
    )>,
) -> Result<(ContiguousPartition, CostAwarePartitionReport), DistributedTopologyError> {
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
    if !options.max_owned_dof_imbalance.is_finite()
        || options.max_owned_dof_imbalance < 1.0
        || !options.max_local_nnz_imbalance.is_finite()
        || options.max_local_nnz_imbalance < 1.0
    {
        return Err(DistributedTopologyError::InvalidPartition(
            "cost-aware imbalance caps must be finite and >= 1.0",
        ));
    }
    if let Some(calibration) = options.calibration {
        if calibration.baseline_spmv_ns == 0 || calibration.graph_spmv_ns == 0 {
            return Err(DistributedTopologyError::InvalidPartition(
                "calibrated SpMV timings must be positive",
            ));
        }
    }

    let field = field_aligned_contiguous_partition(layout, ranks)?;
    let field_plans = build_contiguous_halo_plans(matrix, &field)?;
    let field_stats = partition_telemetry(matrix, &field, &field_plans)?;
    let equal = ContiguousPartition::balanced(layout.total_dofs(), ranks)?;
    let equal_stats = if validate_field_contiguous_partition(layout, &equal).is_ok() {
        let equal_plans = build_contiguous_halo_plans(matrix, &equal)?;
        Some(partition_telemetry(matrix, &equal, &equal_plans)?)
    } else {
        None
    };
    let (baseline, baseline_kind, baseline_stats) = match equal_stats {
        Some(stats) if structurally_dominates(&stats, &field_stats) => {
            (equal, CostAwarePartitionKind::Equal, stats)
        }
        _ => (field, CostAwarePartitionKind::Field, field_stats),
    };
    let make_report = |decision,
                       graph_candidate: Option<PartitionTelemetry>,
                       local_graph_prepare_ns: Option<u128>,
                       modeled_net_savings_ns: Option<u128>| {
        CostAwarePartitionReport {
            baseline_kind,
            selected_kind: baseline_kind,
            decision,
            baseline: baseline_stats.clone(),
            selected: baseline_stats.clone(),
            graph_candidate,
            local_graph_prepare_ns,
            modeled_net_savings_ns,
        }
    };

    // Cheap early exit: no ABTM dual topology, no graph boundary search.
    let reason = if ranks == 1 {
        Some(CostAwarePartitionDecision::SingleRank)
    } else if options.expected_spmv_calls == 0 {
        Some(CostAwarePartitionDecision::NoExpectedReuse)
    } else if options.search_radius_blocks == 0 {
        Some(CostAwarePartitionDecision::ZeroSearchRadius)
    } else if options.calibration.is_none() {
        Some(CostAwarePartitionDecision::NoCalibration)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Ok((baseline, make_report(reason, None, None, None)));
    }
    let calibration = options.calibration.expect("calibration checked above");
    if calibration.graph_spmv_ns >= calibration.baseline_spmv_ns {
        return Ok((
            baseline,
            make_report(
                CostAwarePartitionDecision::NoCalibratedSpeedup,
                None,
                None,
                None,
            ),
        ));
    }
    let gross = u128::from(calibration.baseline_spmv_ns - calibration.graph_spmv_ns)
        * u128::from(options.expected_spmv_calls);
    let prep = u128::from(calibration.graph_prepare_budget_ns);
    let net = gross.saturating_sub(prep);
    if gross <= prep || net <= u128::from(options.minimum_net_savings_ns) {
        return Ok((
            baseline,
            make_report(
                CostAwarePartitionDecision::UnamortizedPreparation,
                None,
                None,
                Some(net),
            ),
        ));
    }

    // The only consumer of `prepared` is the scoped, identity-checked H1-C3J
    // path. Exact structural and load-cap admission below remains identical.
    let (graph, graph_report, measured_ns) = if let Some((partition, comparison, prior_ns)) =
        prepared
    {
        if comparison.search_radius_blocks != options.search_radius_blocks {
            return Err(DistributedTopologyError::InvalidPartition(
                "prepared Graph search radius does not match options",
            ));
        }
        (partition.clone(), comparison.clone(), prior_ns)
    } else {
        let started = Instant::now();
        let (graph, comparison) =
            field_graph_contiguous_partition(matrix, layout, ranks, options.search_radius_blocks)?;
        (graph, comparison, started.elapsed().as_nanos())
    };
    let candidate = Some(graph_report.candidate.clone());
    if !graph_report.candidate_accepted {
        return Ok((
            baseline,
            make_report(
                CostAwarePartitionDecision::GraphProposalRejected,
                candidate,
                Some(measured_ns),
                Some(net),
            ),
        ));
    }
    if graph == baseline {
        return Ok((
            baseline,
            make_report(
                CostAwarePartitionDecision::GraphNotDistinct,
                candidate,
                Some(measured_ns),
                Some(net),
            ),
        ));
    }
    let graph_stats = &graph_report.selected;
    if graph_stats.communication_volume > baseline_stats.communication_volume
        || graph_stats.cut_nnz > baseline_stats.cut_nnz
        || graph_stats.directional_peer_relations > baseline_stats.directional_peer_relations
    {
        return Ok((
            baseline,
            make_report(
                CostAwarePartitionDecision::GraphDoesNotDominateBaseline,
                candidate,
                Some(measured_ns),
                Some(net),
            ),
        ));
    }
    if graph_stats.owned_dof_imbalance > options.max_owned_dof_imbalance
        || graph_stats.local_nnz_imbalance > options.max_local_nnz_imbalance
    {
        return Ok((
            baseline,
            make_report(
                CostAwarePartitionDecision::LoadImbalanceLimit,
                candidate,
                Some(measured_ns),
                Some(net),
            ),
        ));
    }
    Ok((
        graph,
        CostAwarePartitionReport {
            baseline_kind,
            selected_kind: CostAwarePartitionKind::Graph,
            decision: CostAwarePartitionDecision::GraphSelected,
            baseline: baseline_stats,
            selected: graph_report.selected,
            graph_candidate: candidate,
            local_graph_prepare_ns: Some(measured_ns),
            modeled_net_savings_ns: Some(net),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prepare_rank_local_csrs, simulate_halo_exchange};
    use hybit_matrix::{DofBlock, DofEntityKind};

    fn fixture() -> (FieldAwareDofLayout, Csr32Matrix) {
        let blocks = (0..12u64)
            .map(|i| DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: i,
                field_id: 0,
                components: [3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1][i as usize],
            })
            .collect();
        let layout = FieldAwareDofLayout::new(blocks).unwrap();
        let n = layout.total_dofs() as usize;
        let mut row_ptr = vec![0u32];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for row in 0..n {
            let block = layout.locate(row as u64).unwrap().block_index;
            let (begin, end) = if block < 5 { (0, 11) } else { (11, n) };
            for col in begin..end {
                cols.push(col as u32);
                vals.push(if row == col { 5.0 } else { -0.25 });
            }
            row_ptr.push(vals.len() as u32);
        }
        (layout, Csr32Matrix::new(n, n, row_ptr, cols, vals).unwrap())
    }

    fn profitable() -> CostAwarePartitionOptions {
        CostAwarePartitionOptions {
            expected_spmv_calls: 10_000,
            calibration: Some(PartitionCostCalibration {
                baseline_spmv_ns: 100_000,
                graph_spmv_ns: 50_000,
                graph_prepare_budget_ns: 200_000,
            }),
            max_owned_dof_imbalance: 2.0,
            max_local_nnz_imbalance: 2.0,
            ..CostAwarePartitionOptions::default()
        }
    }

    #[test]
    fn default_skips_abtm_and_returns_block_safe_baseline() {
        let (layout, matrix) = fixture();
        let (partition, report) = cost_aware_contiguous_partition(
            &matrix,
            &layout,
            2,
            CostAwarePartitionOptions::default(),
        )
        .unwrap();
        assert_eq!(report.decision, CostAwarePartitionDecision::NoExpectedReuse);
        assert!(report.graph_candidate.is_none());
        assert!(report.local_graph_prepare_ns.is_none());
        validate_field_contiguous_partition(&layout, &partition).unwrap();
    }

    #[test]
    fn one_rank_and_unamortized_skip_before_abtm() {
        let (layout, matrix) = fixture();
        let (_, one) = cost_aware_contiguous_partition(&matrix, &layout, 1, profitable()).unwrap();
        assert_eq!(one.decision, CostAwarePartitionDecision::SingleRank);
        let mut opts = profitable();
        opts.expected_spmv_calls = 1;
        let (_, two) = cost_aware_contiguous_partition(&matrix, &layout, 2, opts).unwrap();
        assert_eq!(
            two.decision,
            CostAwarePartitionDecision::UnamortizedPreparation
        );
        assert!(two.local_graph_prepare_ns.is_none());
    }

    #[test]
    fn profitable_graph_is_field_safe_and_spmv_exact() {
        let (layout, matrix) = fixture();
        let (partition, report) =
            cost_aware_contiguous_partition(&matrix, &layout, 2, profitable()).unwrap();
        assert_eq!(report.decision, CostAwarePartitionDecision::GraphSelected);
        assert_eq!(report.selected_kind, CostAwarePartitionKind::Graph);
        assert_eq!(partition.offsets(), &[0, 11, 24]);
        validate_field_contiguous_partition(&layout, &partition).unwrap();
        let plans = build_contiguous_halo_plans(&matrix, &partition).unwrap();
        let locals = prepare_rank_local_csrs(&matrix, &plans).unwrap();
        let x: Vec<f64> = (0..matrix.nrows())
            .map(|i| ((i + 1) as f64 * 0.087).sin())
            .collect();
        let reference = matrix.spmv(&x).unwrap();
        let extended = simulate_halo_exchange(&x, &plans).unwrap();
        for ((plan, local), ext) in plans.iter().zip(&locals).zip(&extended) {
            let mut actual = vec![0.0; plan.owned_len()];
            local.apply_extended(ext, &mut actual).unwrap();
            let owned = plan.owned_range();
            assert_eq!(actual, reference[owned.start as usize..owned.end as usize]);
        }
    }

    #[test]
    fn load_cap_rejects_even_when_graph_cuts_ghosts() {
        let (layout, matrix) = fixture();
        let mut opts = profitable();
        opts.max_owned_dof_imbalance = 1.05;
        let (partition, report) =
            cost_aware_contiguous_partition(&matrix, &layout, 2, opts).unwrap();
        assert_eq!(
            report.decision,
            CostAwarePartitionDecision::LoadImbalanceLimit
        );
        assert_ne!(report.selected_kind, CostAwarePartitionKind::Graph);
        assert_eq!(partition.offsets(), &[0, 12, 24]);
        assert!(report.local_graph_prepare_ns.is_some());
    }

    #[test]
    fn missing_calibration_and_slower_graph_are_rejected() {
        let (layout, matrix) = fixture();
        let mut opts = profitable();
        opts.calibration = None;
        let (_, a) = cost_aware_contiguous_partition(&matrix, &layout, 2, opts).unwrap();
        assert_eq!(a.decision, CostAwarePartitionDecision::NoCalibration);
        opts.calibration = Some(PartitionCostCalibration {
            baseline_spmv_ns: 90_000,
            graph_spmv_ns: 100_000,
            graph_prepare_budget_ns: 0,
        });
        let (_, b) = cost_aware_contiguous_partition(&matrix, &layout, 2, opts).unwrap();
        assert_eq!(b.decision, CostAwarePartitionDecision::NoCalibratedSpeedup);
    }

    #[test]
    fn invalid_caps_and_dimensions_are_rejected() {
        let (layout, matrix) = fixture();
        let mut opts = profitable();
        opts.max_local_nnz_imbalance = f64::NAN;
        assert!(cost_aware_contiguous_partition(&matrix, &layout, 2, opts).is_err());
        opts.max_local_nnz_imbalance = 1.0;
        opts.max_owned_dof_imbalance = 0.9;
        assert!(cost_aware_contiguous_partition(&matrix, &layout, 2, opts).is_err());
        let wrong = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
        assert!(cost_aware_contiguous_partition(&wrong, &layout, 2, profitable()).is_err());
    }

    #[test]
    fn equal_partition_never_splits_multicomponent_block() {
        let layout = FieldAwareDofLayout::new(vec![
            DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: 0,
                field_id: 0,
                components: 3,
            },
            DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: 1,
                field_id: 0,
                components: 3,
            },
            DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: 2,
                field_id: 0,
                components: 3,
            },
        ])
        .unwrap();
        let n = layout.total_dofs() as usize;
        let row_ptr = (0..=n as u32).collect();
        let matrix =
            Csr32Matrix::new(n, n, row_ptr, (0..n as u32).collect(), vec![1.0; n]).unwrap();
        let (partition, report) = cost_aware_contiguous_partition(
            &matrix,
            &layout,
            2,
            CostAwarePartitionOptions::default(),
        )
        .unwrap();
        assert_eq!(report.baseline_kind, CostAwarePartitionKind::Field);
        assert!(partition.offsets().contains(&3) || partition.offsets().contains(&6));
        validate_field_contiguous_partition(&layout, &partition).unwrap();
    }
}
