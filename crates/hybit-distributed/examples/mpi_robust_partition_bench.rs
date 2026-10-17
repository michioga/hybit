//! H1-C3I: paired-epoch robust preflight veto on native MPI evidence.
//!
//! Root-only candidate generation; native MPI probe and guarded selection.
//! Preregister a conservative second-selector budget, measure four paired
//! calibration epochs and apply the H1-C3I robust cost veto BEFORE the
//! H1-C3D scoped selector. Independent holdout remains out of the decision.
//! Never use holdout results to pick the policy. A full-N solve is NOT timed.
//! A synthetic sparse problem is used, not a real mixed-field FEM benchmark.
//! This does NOT change HyBIT's automatic routing or production MPI solver.

use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit_distributed::{
    capture_partition_calibration, evaluate_robust_cost, field_graph_contiguous_partition,
    select_and_broadcast_cost_partition_mpi, validate_field_contiguous_partition,
    CalibrationGuardDecision, ContiguousPartition, CostAwarePartitionDecision,
    CostAwarePartitionKind, CostAwarePartitionOptions, PartitionCostCalibration,
    RobustCostDecision, RobustCostInputs, RootMpiCostSelection,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::ops::Range;
use std::time::Instant;

const CLUSTER_BLOCKS: [usize; 8] = [12, 11, 10, 11, 10, 10, 10, 10];
const WARMUPS: usize = 5;
const SAMPLES: usize = 40;
const CALIBRATION_EPOCHS: usize = 4;
const SAMPLES_PER_EPOCH: usize = SAMPLES / CALIBRATION_EPOCHS;
const SELECTOR_MIN_BUDGET_NS: u64 = 1_000_000;
const HOLDOUT: usize = 20;
const HOLDOUT_EPOCHS: usize = 3;
const ENVIRONMENT: &str = "H1C3I-synthetic-paired-calibration-native-msmpi-v1";

fn ranges() -> Vec<Range<usize>> {
    let mut begin = 0usize;
    CLUSTER_BLOCKS
        .iter()
        .map(|&blocks| {
            let end = begin + 2 * blocks;
            let range = begin..end;
            begin = end;
            range
        })
        .collect()
}

fn layout() -> Result<FieldAwareDofLayout, Box<dyn Error>> {
    let blocks = (0..CLUSTER_BLOCKS.iter().sum::<usize>())
        .map(|block| DofBlock {
            entity_kind: if block % 7 == 0 {
                DofEntityKind::Multiplier
            } else {
                DofEntityKind::Node
            },
            entity_id: block as u64,
            field_id: (block % 3) as u32,
            components: 2,
        })
        .collect();
    Ok(FieldAwareDofLayout::new(blocks)?)
}

fn row_entries(row: usize, clusters: &[Range<usize>]) -> Vec<(u32, f64)> {
    let cluster_id = clusters
        .iter()
        .position(|range| range.contains(&row))
        .expect("row covered by exactly one cluster");
    let cluster = &clusters[cluster_id];
    let mut entries = Vec::with_capacity(cluster.len() + 1);
    for col in cluster.clone() {
        entries.push((
            col as u32,
            if col == row {
                cluster.len() as f64 + 1.0
            } else {
                -0.0625
            },
        ));
    }
    // One directed bridge per region, retaining real inter-rank halo traffic
    // after the Graph partition aligns cuts to the eight weak boundaries.
    if row == cluster.start {
        let other = clusters[(cluster_id + 1) % clusters.len()].start;
        entries.push((other as u32, -0.125));
    }
    entries
}

fn root_csr(clusters: &[Range<usize>]) -> Result<Csr32Matrix, Box<dyn Error>> {
    let n = clusters.last().expect("clusters").end;
    let mut row_ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        for (col, value) in row_entries(row, clusters) {
            cols.push(col);
            values.push(value);
        }
        row_ptr.push(u32::try_from(cols.len())?);
    }
    Ok(Csr32Matrix::new(n, n, row_ptr, cols, values)?)
}

fn rank_owned_rows(clusters: &[Range<usize>], owned: Range<u64>) -> OwnedCsrRows {
    let mut row_ptr = vec![0u32];
    let mut global_col_idx = Vec::new();
    let mut values = Vec::new();
    for row in owned.start as usize..owned.end as usize {
        for (col, value) in row_entries(row, clusters) {
            global_col_idx.push(u64::from(col));
            values.push(value);
        }
        row_ptr.push(values.len() as u32);
    }
    OwnedCsrRows {
        global_dofs: clusters.last().expect("clusters").end as u64,
        owned,
        row_ptr,
        global_col_idx,
        values,
    }
}

struct MpiCase {
    halo: hybit_distributed::HaloPlan,
    local: hybit_distributed::RankLocalCsr,
    extended: Vec<f64>,
    output: Vec<f64>,
    ghosts: u64,
    sent: u64,
    received: u64,
    total_nnz: u64,
    setup_max_ms: f64,
    max_rank_spmv_ms: Vec<f64>,
    worst_error: f64,
}

fn collective_max_ms(local_ms: f64) -> f64 {
    let mut maximum = 0.0;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &local_ms,
        &mut maximum,
        SystemOperation::max(),
    );
    maximum
}

fn prepare_case(
    mpi: &MpiRuntime,
    clusters: &[Range<usize>],
    selection: RootMpiCostSelection,
) -> Result<MpiCase, Box<dyn Error>> {
    let owned = selection.partition.owned_range(mpi.rank() as u32)?;
    let rows = rank_owned_rows(clusters, owned);
    let started = Instant::now();
    let (halo, local, stats) = prepare_owned_rows(mpi, &selection.partition, &rows)?;
    let setup_max_ms = collective_max_ms(started.elapsed().as_secs_f64() * 1e3);
    Ok(MpiCase {
        extended: vec![0.0; halo.extended_len()],
        output: vec![0.0; halo.owned_len()],
        ghosts: mpi.all_reduce_sum_u64(stats.ghosts as u64),
        sent: 0,
        received: 0,
        total_nnz: mpi.all_reduce_sum_u64(stats.local_nnz as u64),
        setup_max_ms,
        max_rank_spmv_ms: Vec::with_capacity(SAMPLES.max(HOLDOUT)),
        worst_error: 0.0,
        halo,
        local,
    })
}

fn measure_once(
    mpi: &MpiRuntime,
    case: &mut MpiCase,
    input: &[f64],
    clusters: &[Range<usize>],
    measured: bool,
) -> Result<(), Box<dyn Error>> {
    let owned = case.halo.owned_range();
    let started = Instant::now();
    let traffic = mpi.spmv_local(
        &case.halo,
        &case.local,
        &input[owned.start as usize..owned.end as usize],
        &mut case.extended,
        &mut case.output,
    )?;
    let max_rank_ms = collective_max_ms(started.elapsed().as_secs_f64() * 1e3);
    if measured {
        case.max_rank_spmv_ms.push(max_rank_ms);
    }
    case.sent += traffic.sent_values as u64;
    case.received += traffic.received_values as u64;
    for (local_row, &actual) in case.output.iter().enumerate() {
        let global_row = owned.start as usize + local_row;
        let reference = row_entries(global_row, clusters)
            .iter()
            .fold(0.0, |sum, &(col, value)| sum + value * input[col as usize]);
        case.worst_error = case.worst_error.max((actual - reference).abs());
    }
    for (slot, &global) in case.halo.ghost_globals().iter().enumerate() {
        case.worst_error = case
            .worst_error
            .max((case.extended[case.halo.owned_len() + slot] - input[global as usize]).abs());
    }
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "nonempty timing samples");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}

fn milliseconds_to_ns(ms: f64) -> Result<u64, Box<dyn Error>> {
    let ns = ms * 1.0e6;
    if !ns.is_finite() || ns <= 0.0 || ns > u64::MAX as f64 {
        return Err("invalid measured time for nanosecond calibration".into());
    }
    Ok((ns.ceil() as u64).max(1))
}

fn break_even_calls(additional_ns: i128, savings_ns: i128) -> Option<u64> {
    if savings_ns <= 0 {
        return None;
    }
    if additional_ns <= 0 {
        return Some(0);
    }
    let calls = (additional_ns as u128).div_ceil(savings_ns as u128);
    u64::try_from(calls).ok()
}

fn full_break_even_calls(extra_ms: f64, savings_ms: f64) -> Option<u64> {
    if !extra_ms.is_finite() || !savings_ms.is_finite() || savings_ms <= 0.0 {
        return None;
    }
    if extra_ms <= 0.0 {
        return Some(0);
    }
    let extra_ns = (extra_ms * 1.0e6).ceil();
    let saved_ns = (savings_ms * 1.0e6).floor();
    if !extra_ns.is_finite()
        || !saved_ns.is_finite()
        || saved_ns < 1.0
        || extra_ns > i128::MAX as f64
        || saved_ns > i128::MAX as f64
    {
        return None;
    }
    break_even_calls(extra_ns as i128, saved_ns as i128)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 && args.len() != 3 {
        return Err(
            "usage: mpi_robust_partition_bench <ranks:1|2|4|8> [expected_spmv_calls]".into(),
        );
    }
    let expected: i32 = args[1].parse()?;
    let expected_calls: u64 = if args.len() == 3 {
        args[2].parse()?
    } else {
        10_000
    };
    if expected_calls == 0 || expected_calls > 10_000_000 {
        return Err("expected_spmv_calls must be 1..=10000000".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI world size must be 1, 2, 4, or 8 as requested".into());
    }
    let rank = mpi.rank();
    let world = mpi::topology::SimpleCommunicator::world();
    let root = world.process_at_rank(0);
    let clusters = ranges();
    let physical = layout()?;
    let n = physical.total_dofs() as usize;
    let global = if rank == 0 {
        Some(root_csr(&clusters)?)
    } else {
        None
    };
    let options = CostAwarePartitionOptions {
        expected_spmv_calls: expected_calls,
        search_radius_blocks: 3,
        max_owned_dof_imbalance: 1.5,
        max_local_nnz_imbalance: 1.8,
        ..CostAwarePartitionOptions::default()
    };

    // The probe ledger includes baseline broadcast, root candidate generation,
    // native setup for BOTH policies, and all calibration warmup/sample calls.
    mpi.barrier();
    let probe_started = Instant::now();
    // Always first select the real Equal/Field baseline using NO calibration.
    let baseline_selected = select_and_broadcast_cost_partition_mpi(
        global.as_ref(),
        &physical,
        options,
        ENVIRONMENT,
        None,
    )?;
    if baseline_selected.selected_kind == CostAwarePartitionKind::Graph {
        return Err("uncalibrated baseline chose Graph".into());
    }
    let baseline_offsets = baseline_selected.partition.offsets().to_vec();

    // Candidate-generation is explicitly NOT the guarded final selector.
    // Root only constructs and times the ABTM graph candidate; on failure,
    // broadcast a status before any rank can enter a different collective.
    let graph_prepare_start = Instant::now();
    let root_candidate = if rank == 0 {
        Some(field_graph_contiguous_partition(
            global.as_ref().expect("root CSR"),
            &physical,
            expected as u32,
            options.search_radius_blocks,
        ))
    } else {
        None
    };
    let graph_prepare_ns = if rank == 0 {
        u64::try_from(graph_prepare_start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    } else {
        0
    };
    let mut root_error = if rank == 0 {
        u64::from(root_candidate.as_ref().expect("root result").is_err())
    } else {
        0
    };
    root.broadcast_into(&mut root_error);
    // All ranks must use exactly the same conservative selector budget.
    let mut shared_graph_prepare_ns = graph_prepare_ns;
    root.broadcast_into(&mut shared_graph_prepare_ns);
    if root_error != 0 {
        return Err("root ABTM proposal generation failed collectively".into());
    }
    let mut candidate_offsets = vec![0u64; expected as usize + 1];
    if rank == 0 {
        let (partition, _) = root_candidate
            .as_ref()
            .expect("root result")
            .as_ref()
            .expect("successful root proposal");
        candidate_offsets.copy_from_slice(partition.offsets());
    }
    root.broadcast_into(candidate_offsets.as_mut_slice());
    let proposed = ContiguousPartition::from_offsets(candidate_offsets.clone()).and_then(|p| {
        validate_field_contiguous_partition(&physical, &p)?;
        Ok(p)
    });
    let mut invalid = 0u64;
    world.all_reduce_into(
        &u64::from(proposed.is_err()),
        &mut invalid,
        SystemOperation::sum(),
    );
    if invalid != 0 {
        return Err("invalid Graph candidate offsets broadcast to MPI ranks".into());
    }
    let graph_partition = proposed?;
    // The candidate is used for timing only. Its policy label is not
    // an authorization for the guarded final selection.
    let mut trial_graph_selection = baseline_selected.clone();
    trial_graph_selection.partition = graph_partition.clone();
    trial_graph_selection.selected_kind = CostAwarePartitionKind::Graph;

    let mut baseline = prepare_case(&mpi, &clusters, baseline_selected)?;
    let mut candidate = prepare_case(&mpi, &clusters, trial_graph_selection)?;
    let x: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.017).sin()).collect();
    // Four PAIRED calibration epochs; never use the later holdout to admit.
    let mut paired_epoch_ns = Vec::with_capacity(CALIBRATION_EPOCHS);
    let mut calibration_base_epochs = Vec::with_capacity(CALIBRATION_EPOCHS);
    let mut calibration_graph_epochs = Vec::with_capacity(CALIBRATION_EPOCHS);
    for epoch in 0..CALIBRATION_EPOCHS {
        baseline.max_rank_spmv_ms.clear();
        candidate.max_rank_spmv_ms.clear();
        for trial in 0..(WARMUPS + SAMPLES_PER_EPOCH) {
            let measured = trial >= WARMUPS;
            if (epoch + trial) % 2 == 0 {
                measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
                measure_once(&mpi, &mut candidate, &x, &clusters, measured)?;
            } else {
                measure_once(&mpi, &mut candidate, &x, &clusters, measured)?;
                measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
            }
        }
        let baseline_epoch_ms = median(&baseline.max_rank_spmv_ms);
        let graph_epoch_ms = median(&candidate.max_rank_spmv_ms);
        calibration_base_epochs.push(baseline_epoch_ms);
        calibration_graph_epochs.push(graph_epoch_ms);
        paired_epoch_ns.push((
            milliseconds_to_ns(baseline_epoch_ms)?,
            milliseconds_to_ns(graph_epoch_ms)?,
        ));
    }
    let baseline_error = mpi.all_reduce_sum_f64(baseline.worst_error);
    let candidate_error = mpi.all_reduce_sum_f64(candidate.worst_error);
    let baseline_sent = mpi.all_reduce_sum_u64(baseline.sent);
    let baseline_recv = mpi.all_reduce_sum_u64(baseline.received);
    let graph_sent = mpi.all_reduce_sum_u64(candidate.sent);
    let graph_recv = mpi.all_reduce_sum_u64(candidate.received);
    let owned_sum = mpi.all_reduce_sum_u64(candidate.halo.owned_len() as u64);
    let workers_global_csr = mpi.all_reduce_sum_u64(u64::from(rank != 0 && global.is_some()));
    let performed = (CALIBRATION_EPOCHS * (WARMUPS + SAMPLES_PER_EPOCH)) as u64;
    if baseline_error > 1e-10
        || candidate_error > 1e-10
        || baseline_sent != baseline_recv
        || graph_sent != graph_recv
        || baseline_recv != baseline.ghosts * performed
        || graph_recv != candidate.ghosts * performed
        || owned_sum != n as u64
        || workers_global_csr != 0
        || baseline.total_nnz != candidate.total_nnz
    {
        return Err("native MPI candidate/benchmark numerical or traffic mismatch".into());
    }

    // Calibration is formed ONLY from actually recorded native MPI times.
    let baseline_ms = median(&calibration_base_epochs);
    let graph_ms = median(&calibration_graph_epochs);
    // Include the measured positive native-setup delta, if any. This model
    // excludes the *calibration run itself*, which is a real up-front expense
    // and must be considered before operational deployment.
    let extra_setup_ms = (candidate.setup_max_ms - baseline.setup_max_ms).max(0.0);
    let extra_setup_ns = if extra_setup_ms > 0.0 {
        milliseconds_to_ns(extra_setup_ms)?
    } else {
        0
    };
    let preparation_budget_ns = graph_prepare_ns.saturating_add(extra_setup_ns).max(1);
    let calibration = PartitionCostCalibration {
        baseline_spmv_ns: milliseconds_to_ns(baseline_ms)?,
        graph_spmv_ns: milliseconds_to_ns(graph_ms)?,
        graph_prepare_budget_ns: preparation_budget_ns,
    };

    mpi.barrier();
    let probe_wall_ms = collective_max_ms(probe_started.elapsed().as_secs_f64() * 1.0e3);
    // Preregister a generous floor and a multiple of the root proposal time.
    // This is only a *forecast* of the second selector, audited after execution.
    let selector_budget_ns = SELECTOR_MIN_BUDGET_NS.max(shared_graph_prepare_ns.saturating_mul(4));
    let robust = evaluate_robust_cost(RobustCostInputs {
        epochs: &paired_epoch_ns,
        expected_spmv_calls: expected_calls,
        probe_wall_ns: milliseconds_to_ns(probe_wall_ms)?,
        baseline_native_setup_ns: milliseconds_to_ns(baseline.setup_max_ms)?,
        graph_native_setup_ns: milliseconds_to_ns(candidate.setup_max_ms)?,
        second_selector_budget_ns: selector_budget_ns,
        minimum_net_savings_ns: 0,
    });
    // Hard fail-closed: a nonpositive gain in ANY epoch, or an unamortized
    // total budget, suppresses scoped evidence and hence the Graph search.
    let robust_admit = robust.decision == RobustCostDecision::Admit && expected > 1;
    let local_admit = u64::from(robust_admit);
    let mut admit_min = 0u64;
    let mut admit_max = 0u64;
    world.all_reduce_into(&local_admit, &mut admit_min, SystemOperation::min());
    world.all_reduce_into(&local_admit, &mut admit_max, SystemOperation::max());
    if admit_min != admit_max {
        return Err("robust MPI evidence disagreement".into());
    }
    mpi.barrier();
    let decision_started = Instant::now();

    // Failed scope capture on rank 0 must propagate collectively; no rank
    // can return before the next MPI selection broadcast.
    let root_scope = if rank == 0 && robust_admit {
        Some(capture_partition_calibration(
            global.as_ref().expect("root CSR"),
            &physical,
            expected as u32,
            options,
            ENVIRONMENT,
            calibration,
        ))
    } else {
        None
    };
    let mut scope_failed = if rank == 0 {
        u64::from(root_scope.as_ref().is_some_and(Result::is_err))
    } else {
        0
    };
    root.broadcast_into(&mut scope_failed);
    if scope_failed != 0 {
        return Err("root measured calibration scope capture failed".into());
    }
    let scope = root_scope.as_ref().and_then(|value| value.as_ref().ok());
    let selected = select_and_broadcast_cost_partition_mpi(
        global.as_ref(),
        &physical,
        options,
        ENVIRONMENT,
        scope,
    )?;
    mpi.barrier();
    // Includes scoped evidence capture, an additional ABTM graph search when
    // the cost gate allows it, and root broadcast/collective validation.
    let decision_wall_ms = collective_max_ms(decision_started.elapsed().as_secs_f64() * 1.0e3);
    let accepted_graph = selected.selected_kind == CostAwarePartitionKind::Graph;
    if (robust_admit && selected.calibration_guard != CalibrationGuardDecision::Matched)
        || (!robust_admit && selected.calibration_guard != CalibrationGuardDecision::Missing)
        || (accepted_graph
            && (!robust_admit
                || selected.decision != CostAwarePartitionDecision::GraphSelected
                || selected.partition != graph_partition))
        || (!accepted_graph && selected.partition.offsets() != baseline_offsets)
    {
        return Err("measured guarded selector returned inconsistent policy/offsets".into());
    }

    // Entirely independent paired holdout. Each epoch refreshes warmup state;
    // rotating invocation order prevents a consistent first-policy advantage.
    // Candidate and baseline are measured even when the gate selects fallback.
    let mut holdout = prepare_case(&mpi, &clusters, selected.clone())?;
    let mut base_epochs = Vec::with_capacity(HOLDOUT_EPOCHS);
    let mut graph_epochs = Vec::with_capacity(HOLDOUT_EPOCHS);
    let mut selected_epochs = Vec::with_capacity(HOLDOUT_EPOCHS);
    for epoch in 0..HOLDOUT_EPOCHS {
        baseline.max_rank_spmv_ms.clear();
        candidate.max_rank_spmv_ms.clear();
        holdout.max_rank_spmv_ms.clear();
        for trial in 0..(WARMUPS + HOLDOUT) {
            let measured = trial >= WARMUPS;
            match (epoch + trial) % 3 {
                0 => {
                    measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
                    measure_once(&mpi, &mut candidate, &x, &clusters, measured)?;
                    measure_once(&mpi, &mut holdout, &x, &clusters, measured)?;
                }
                1 => {
                    measure_once(&mpi, &mut candidate, &x, &clusters, measured)?;
                    measure_once(&mpi, &mut holdout, &x, &clusters, measured)?;
                    measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
                }
                _ => {
                    measure_once(&mpi, &mut holdout, &x, &clusters, measured)?;
                    measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
                    measure_once(&mpi, &mut candidate, &x, &clusters, measured)?;
                }
            }
        }
        base_epochs.push(median(&baseline.max_rank_spmv_ms));
        graph_epochs.push(median(&candidate.max_rank_spmv_ms));
        selected_epochs.push(median(&holdout.max_rank_spmv_ms));
    }
    let base_holdout_ms = median(&base_epochs);
    let graph_holdout_ms = median(&graph_epochs);
    let selected_holdout_ms = median(&selected_epochs);
    let candidate_wins_all_epochs = base_epochs
        .iter()
        .zip(&graph_epochs)
        .all(|(&base, &graph)| graph < base);

    // Verify BOTH probe and holdout traffic (no imaginary communication
    // savings) as well as every independently prepared selected policy.
    let baseline_error_final = mpi.all_reduce_sum_f64(baseline.worst_error);
    let graph_error_final = mpi.all_reduce_sum_f64(candidate.worst_error);
    let holdout_error = mpi.all_reduce_sum_f64(holdout.worst_error);
    let baseline_sent_final = mpi.all_reduce_sum_u64(baseline.sent);
    let baseline_recv_final = mpi.all_reduce_sum_u64(baseline.received);
    let graph_sent_final = mpi.all_reduce_sum_u64(candidate.sent);
    let graph_recv_final = mpi.all_reduce_sum_u64(candidate.received);
    let holdout_sent = mpi.all_reduce_sum_u64(holdout.sent);
    let holdout_recv = mpi.all_reduce_sum_u64(holdout.received);
    let probe_calls = (CALIBRATION_EPOCHS * (WARMUPS + SAMPLES_PER_EPOCH)) as u64;
    let extra_calls = (HOLDOUT_EPOCHS * (WARMUPS + HOLDOUT)) as u64;
    if baseline_error_final > 1e-10
        || graph_error_final > 1e-10
        || holdout_error > 1e-10
        || baseline_sent_final != baseline_recv_final
        || graph_sent_final != graph_recv_final
        || baseline_recv_final != baseline.ghosts * (probe_calls + extra_calls)
        || graph_recv_final != candidate.ghosts * (probe_calls + extra_calls)
        || holdout_sent != holdout_recv
        || holdout_recv != holdout.ghosts * extra_calls
    {
        return Err("H1-C3I held-out numerical / reciprocal Halo traffic invariant failed".into());
    }

    // These are MODELLED total times for N calls; N actual solver iterations
    // were NOT run. Account for the full probe and the second selector once,
    // without adding the root's exploratory graph preparation a second time.
    // The baseline-only alternative does not pay to calibrate Graph.
    let baseline_only_ms = baseline.setup_max_ms + expected_calls as f64 * base_holdout_ms;
    let selected_after_calibration_ms =
        decision_wall_ms + holdout.setup_max_ms + expected_calls as f64 * selected_holdout_ms;
    let selected_from_probe_start_ms = probe_wall_ms + selected_after_calibration_ms;
    let extra_full_ms =
        probe_wall_ms + decision_wall_ms + holdout.setup_max_ms - baseline.setup_max_ms;
    let full_break_even = if accepted_graph {
        full_break_even_calls(extra_full_ms, base_holdout_ms - selected_holdout_ms)
    } else {
        None
    };
    let calibrated_break_even = break_even_calls(
        i128::from(calibration.graph_prepare_budget_ns),
        i128::from(calibration.baseline_spmv_ns) - i128::from(calibration.graph_spmv_ns),
    );
    let calibration_holdout_ratio = if graph_holdout_ms > 0.0 {
        graph_ms / graph_holdout_ms
    } else {
        f64::NAN
    };
    let workers_global_csr_final = mpi.all_reduce_sum_u64(u64::from(rank != 0 && global.is_some()));
    if workers_global_csr_final != 0 {
        return Err("H1-C3I nonroot unexpectedly holds whole CSR".into());
    }
    let selector_budget_exceeded = decision_wall_ms * 1.0e6 > selector_budget_ns as f64;
    mpi.barrier();
    if rank == 0 {
        println!(
            "PASS H1-C3I ranks={expected} n={n} nnz={} expected_calls={expected_calls} decision={:?} selected={:?} guard={:?} robust={:?} robust_admit={robust_admit} robust_min_paired_gain_ns={:?} robust_required_calls={:?} second_selector_budget_ms={:.6} selector_budget_exceeded={selector_budget_exceeded} workers_global_csr={workers_global_csr_final} baseline_ghosts={} graph_candidate_ghosts={} selected_ghosts={} calibration_baseline_us={:.3} calibration_graph_us={:.3} holdout_baseline_us={:.3} holdout_graph_us={:.3} holdout_selected_us={:.3} all_epochs_graph_faster={candidate_wins_all_epochs} calibration_to_holdout_graph_ratio={calibration_holdout_ratio:.3} probe_wall_ms={probe_wall_ms:.6} root_candidate_prepare_ms={:.6} second_selector_wall_ms={decision_wall_ms:.6} baseline_native_setup_ms={:.6} graph_native_setup_ms={:.6} selected_native_setup_ms={:.6} gate_break_even_calls={calibrated_break_even:?} fully_accounted_break_even_calls={full_break_even:?} modeled_baseline_only_ms={baseline_only_ms:.6} modeled_selected_after_calibration_ms={selected_after_calibration_ms:.6} modeled_selected_from_probe_start_ms={selected_from_probe_start_ms:.6} baseline_abs_err={baseline_error_final:.3e} graph_abs_err={graph_error_final:.3e} selected_abs_err={holdout_error:.3e}",
            candidate.total_nnz, selected.decision, selected.selected_kind,
            selected.calibration_guard, robust.decision, robust.minimum_paired_gain_ns,
            robust.required_calls, selector_budget_ns as f64 * 1e-6, baseline.ghosts, candidate.ghosts, holdout.ghosts,
            baseline_ms * 1e3, graph_ms * 1e3, base_holdout_ms * 1e3,
            graph_holdout_ms * 1e3, selected_holdout_ms * 1e3,
            graph_prepare_ns as f64 * 1e-6,
            baseline.setup_max_ms, candidate.setup_max_ms, holdout.setup_max_ms,
        );
        println!(
            "H1-C3I holdout_epochs baseline_us={:?} graph_us={:?} selected_us={:?}",
            base_epochs.iter().map(|ms| ms * 1e3).collect::<Vec<_>>(),
            graph_epochs.iter().map(|ms| ms * 1e3).collect::<Vec<_>>(),
            selected_epochs
                .iter()
                .map(|ms| ms * 1e3)
                .collect::<Vec<_>>(),
        );
        println!(
            "H1-C3I calibration_epochs baseline_us={:?} graph_us={:?} robust_extra_budget_ms={:.6} robust_gross_savings_ms={:.6}",
            calibration_base_epochs.iter().map(|ms| ms * 1e3).collect::<Vec<_>>(),
            calibration_graph_epochs.iter().map(|ms| ms * 1e3).collect::<Vec<_>>(),
            robust.conservative_extra_budget_ns as f64 * 1e-6,
            robust.estimated_gross_savings_ns as f64 * 1e-6,
        );
        println!(
            "H1-C3I offsets baseline={baseline_offsets:?} candidate={candidate_offsets:?} selected={:?} warmups_per_cal_epoch={WARMUPS} calibration_epochs={CALIBRATION_EPOCHS} calibration_samples={SAMPLES} holdout_epochs={HOLDOUT_EPOCHS} holdout_samples_per_epoch={HOLDOUT} note=modeled-N-not-full-N-measured;paired-epoch-admission;probe-and-selector-budget-included;synthetic-problem",
            selected.partition.offsets(),
        );
    }
    Ok(())
}
