//! H1-C3G: measured rather than invented calibration for rank-0 MPI partition selection.
//!
//! Probe an explicit Graph candidate without a cost calibration, time the SAME
//! baseline/candidate on native MPI, then feed actual maximum-rank medians and
//! root preparation time to the existing scoped cost selector. A held-out
//! post-decision timing is independent from the timing used in the decision.
//! A synthetic sparse *problem* is used, but calibration timings are measured.
//! This is an experiment, not an automatic solver policy or performance claim.

use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit_distributed::{
    capture_partition_calibration, field_graph_contiguous_partition,
    select_and_broadcast_cost_partition_mpi, validate_field_contiguous_partition,
    CalibrationGuardDecision, ContiguousPartition, CostAwarePartitionDecision,
    CostAwarePartitionKind, CostAwarePartitionOptions, PartitionCostCalibration,
    RootMpiCostSelection,
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
const HOLDOUT: usize = 20;
const ENVIRONMENT: &str = "H1C3G-synthetic-problem-native-msmpi-measured-calibration-v1";

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
        max_rank_spmv_ms: Vec::with_capacity(SAMPLES + HOLDOUT),
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
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 && args.len() != 3 {
        return Err(
            "usage: mpi_root_measured_calibration_bench <ranks:1|2|4|8> [expected_spmv_calls]"
                .into(),
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
    for trial in 0..(WARMUPS + SAMPLES) {
        let measured = trial >= WARMUPS;
        if trial % 2 == 0 {
            measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
            measure_once(&mpi, &mut candidate, &x, &clusters, measured)?;
        } else {
            measure_once(&mpi, &mut candidate, &x, &clusters, measured)?;
            measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
        }
    }
    let baseline_error = mpi.all_reduce_sum_f64(baseline.worst_error);
    let candidate_error = mpi.all_reduce_sum_f64(candidate.worst_error);
    let baseline_sent = mpi.all_reduce_sum_u64(baseline.sent);
    let baseline_recv = mpi.all_reduce_sum_u64(baseline.received);
    let graph_sent = mpi.all_reduce_sum_u64(candidate.sent);
    let graph_recv = mpi.all_reduce_sum_u64(candidate.received);
    let owned_sum = mpi.all_reduce_sum_u64(candidate.halo.owned_len() as u64);
    let workers_global_csr = mpi.all_reduce_sum_u64(u64::from(rank != 0 && global.is_some()));
    let performed = (WARMUPS + SAMPLES) as u64;
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
    let baseline_ms = median(&baseline.max_rank_spmv_ms);
    let graph_ms = median(&candidate.max_rank_spmv_ms);
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

    // Failed scope capture on rank 0 must propagate collectively; no rank
    // can return before the next MPI selection broadcast.
    let root_scope = if rank == 0 {
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
        u64::from(root_scope.as_ref().expect("root scope").is_err())
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
    let accepted_graph = selected.selected_kind == CostAwarePartitionKind::Graph;
    if selected.calibration_guard != CalibrationGuardDecision::Matched
        || (accepted_graph
            && (selected.decision != CostAwarePartitionDecision::GraphSelected
                || selected.partition != graph_partition))
        || (!accepted_graph && selected.partition.offsets() != baseline_offsets)
    {
        return Err("measured guarded selector returned inconsistent policy/offsets".into());
    }

    // Held-out native measurement AFTER the decision. These results were not
    // used in capture_partition_calibration and may diverge from calibration.
    let mut holdout = prepare_case(&mpi, &clusters, selected.clone())?;
    for trial in 0..(WARMUPS + HOLDOUT) {
        measure_once(&mpi, &mut holdout, &x, &clusters, trial >= WARMUPS)?;
    }
    let holdout_error = mpi.all_reduce_sum_f64(holdout.worst_error);
    let holdout_sent = mpi.all_reduce_sum_u64(holdout.sent);
    let holdout_recv = mpi.all_reduce_sum_u64(holdout.received);
    if holdout_error > 1e-10
        || holdout_sent != holdout_recv
        || holdout_recv != holdout.ghosts * (WARMUPS + HOLDOUT) as u64
    {
        return Err("held-out selected MPI SpMV / traffic mismatch".into());
    }

    let baseline_ns = i128::from(calibration.baseline_spmv_ns);
    let graph_ns = i128::from(calibration.graph_spmv_ns);
    let additional_ns = i128::from(calibration.graph_prepare_budget_ns);
    let break_even = break_even_calls(additional_ns, baseline_ns - graph_ns);
    let modeled_baseline_ms = baseline.setup_max_ms + expected_calls as f64 * baseline_ms;
    let modeled_graph_ms =
        candidate.setup_max_ms + graph_prepare_ns as f64 * 1e-6 + expected_calls as f64 * graph_ms;
    mpi.barrier();
    if rank == 0 {
        println!(
            "PASS H1-C3G ranks={expected} n={n} nnz={} expected_calls={expected_calls} decision={:?} selected={:?} guard={:?} workers_global_csr={workers_global_csr} baseline_ghosts={} graph_candidate_ghosts={} selected_ghosts={} calibration_baseline_max_rank_median_ms={baseline_ms:.6} calibration_graph_max_rank_median_ms={graph_ms:.6} root_graph_prepare_ms={:.6} extra_native_setup_ns={extra_setup_ns} calibration_preparation_budget_ns={preparation_budget_ns} break_even_calls={break_even:?} modeled_baseline_total_ms={modeled_baseline_ms:.6} modeled_graph_total_ms={modeled_graph_ms:.6} holdout_selected_max_rank_median_ms={:.6} baseline_abs_err={baseline_error:.3e} graph_abs_err={candidate_error:.3e} holdout_abs_err={holdout_error:.3e}",
            candidate.total_nnz, selected.decision, selected.selected_kind,
            selected.calibration_guard, baseline.ghosts, candidate.ghosts,
            holdout.ghosts, graph_prepare_ns as f64 * 1e-6,
            median(&holdout.max_rank_spmv_ms),
        );
        println!(
            "H1-C3G offsets baseline={baseline_offsets:?} candidate={candidate_offsets:?} selected={:?} warmups={WARMUPS} calibration_samples={SAMPLES} holdout_samples={HOLDOUT} note=native-measured-calibration-on-synthetic-problem-excludes-calibration-probe-cost",
            selected.partition.offsets(),
        );
    }
    Ok(())
}
