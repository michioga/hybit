//! H1-C3F: native MPI Graph-selected root-owned partition experiment.
//!
//! Rank 0 alone owns the full CSR, captures an illustrative (NOT measured)
//! calibration and performs symbolic ABTM refinement. Every nonroot rank owns
//! only its own CSR rows and shared O(blocks) field metadata. Both the
//! baseline and Graph candidate run the unchanged MPI Halo/CSR SpMV path.
//! The synthetic calibration demonstrates the branch, NOT a speedup claim.

use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit_distributed::{
    broadcast_root_cost_selection_mpi, capture_partition_calibration,
    cost_aware_contiguous_partition_scoped, select_and_broadcast_cost_partition_mpi,
    CalibrationGuardDecision, CostAwarePartitionDecision, CostAwarePartitionKind,
    CostAwarePartitionOptions, PartitionCostCalibration, RootMpiCostSelection,
    ScopedCostPartitionResult,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::ops::Range;
use std::time::Instant;

const CLUSTER_BLOCKS: [usize; 8] = [12, 11, 10, 11, 10, 10, 10, 10];
const WARMUPS: usize = 5;
const SAMPLES: usize = 20;
const ENVIRONMENT: &str = "H1C3F-synthetic-graph-msmpi-v1-ILLUSTRATIVE";

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
    selection: RootMpiCostSelection,
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
        selection,
        extended: vec![0.0; halo.extended_len()],
        output: vec![0.0; halo.owned_len()],
        ghosts: mpi.all_reduce_sum_u64(stats.ghosts as u64),
        sent: 0,
        received: 0,
        total_nnz: mpi.all_reduce_sum_u64(stats.local_nnz as u64),
        setup_max_ms,
        max_rank_spmv_ms: Vec::with_capacity(SAMPLES),
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: mpi_root_graph_selection_bench <ranks: 1|2|4|8>".into());
    }
    let expected: i32 = args[1].parse()?;
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI ranks must match 1, 2, 4 or 8".into());
    }
    let rank = mpi.rank();
    let clusters = ranges();
    let physical = layout()?;
    let n = physical.total_dofs() as usize;
    let global = if rank == 0 {
        Some(root_csr(&clusters)?)
    } else {
        None
    };
    let options = CostAwarePartitionOptions {
        expected_spmv_calls: 10_000,
        search_radius_blocks: 3,
        max_owned_dof_imbalance: 1.5,
        max_local_nnz_imbalance: 1.8,
        ..CostAwarePartitionOptions::default()
    };

    // Baseline and Graph selection have identical options and matrix/layout.
    // Only the root's matched calibration differs, allowing graph preparation.
    let baseline_selection = select_and_broadcast_cost_partition_mpi(
        global.as_ref(),
        &physical,
        options,
        ENVIRONMENT,
        None,
    )?;
    if baseline_selection.selected_kind == CostAwarePartitionKind::Graph {
        return Err("no-calibration control unexpectedly chose Graph".into());
    }

    // These illustrative timings are intentionally not measured on this setup.
    // They exercise GraphSelected and cannot certify any speedup or ROI.
    // Capture/calibration failures are returned collectively through Bcast;
    // root must not `?` early while workers are waiting in MPI.
    let root_started = Instant::now();
    let root_selection: Option<Result<ScopedCostPartitionResult, _>> = if rank == 0 {
        let matrix = global.as_ref().expect("root global matrix");
        Some(
            capture_partition_calibration(
                matrix,
                &physical,
                expected as u32,
                options,
                ENVIRONMENT,
                PartitionCostCalibration {
                    baseline_spmv_ns: 20_000,
                    graph_spmv_ns: 10_000,
                    graph_prepare_budget_ns: 100_000,
                },
            )
            .and_then(|scope| {
                cost_aware_contiguous_partition_scoped(
                    matrix,
                    &physical,
                    expected as u32,
                    options,
                    ENVIRONMENT,
                    Some(&scope),
                )
            }),
        )
    } else {
        None
    };
    let root_select_ms = if rank == 0 {
        root_started.elapsed().as_secs_f64() * 1e3
    } else {
        0.0
    };
    // Collective even if rank 0 returned Err; never `?` the root selector.
    let selected = broadcast_root_cost_selection_mpi(
        root_selection.as_ref(),
        &physical,
        options,
        ENVIRONMENT,
    )?;

    if selected.calibration_guard != CalibrationGuardDecision::Matched {
        return Err("H1-C3F matched calibration unexpectedly rejected".into());
    }
    let expect_graph = expected > 1;
    if expect_graph
        && (selected.selected_kind != CostAwarePartitionKind::Graph
            || selected.decision != CostAwarePartitionDecision::GraphSelected
            || selected.partition == baseline_selection.partition)
    {
        return Err(format!(
            "H1-C3F GraphSelected fixture failed: ranks={expected} decision={:?} kind={:?} baseline_offsets={:?} selected_offsets={:?}",
            selected.decision,
            selected.selected_kind,
            baseline_selection.partition.offsets(),
            selected.partition.offsets(),
        ).into());
    }
    if !expect_graph && selected.decision != CostAwarePartitionDecision::SingleRank {
        return Err("one-rank fixture did not use SingleRank".into());
    }
    let baseline_offsets = baseline_selection.partition.offsets().to_vec();
    let graph_offsets = selected.partition.offsets().to_vec();
    let mut baseline = prepare_case(&mpi, &clusters, baseline_selection)?;
    let mut graph = prepare_case(&mpi, &clusters, selected)?;

    // Alternating order prevents deterministic baseline-first bias. Report
    // 20 medians of GLOBAL MAX rank wall time; timings include native Halo.
    let x: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.017).sin()).collect();
    for trial in 0..(WARMUPS + SAMPLES) {
        let measured = trial >= WARMUPS;
        if trial % 2 == 0 {
            measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
            measure_once(&mpi, &mut graph, &x, &clusters, measured)?;
        } else {
            measure_once(&mpi, &mut graph, &x, &clusters, measured)?;
            measure_once(&mpi, &mut baseline, &x, &clusters, measured)?;
        }
    }
    let baseline_error = mpi.all_reduce_sum_f64(baseline.worst_error);
    let graph_error = mpi.all_reduce_sum_f64(graph.worst_error);
    let baseline_sent = mpi.all_reduce_sum_u64(baseline.sent);
    let baseline_recv = mpi.all_reduce_sum_u64(baseline.received);
    let graph_sent = mpi.all_reduce_sum_u64(graph.sent);
    let graph_recv = mpi.all_reduce_sum_u64(graph.received);
    let count = (WARMUPS + SAMPLES) as u64;
    let total_owned = mpi.all_reduce_sum_u64(graph.halo.owned_len() as u64);
    let workers_global_csr = mpi.all_reduce_sum_u64(u64::from(rank != 0 && global.is_some()));
    if baseline_error > 1e-10
        || graph_error > 1e-10
        || baseline_sent != baseline_recv
        || graph_sent != graph_recv
        || baseline_recv != baseline.ghosts * count
        || graph_recv != graph.ghosts * count
        || total_owned != n as u64
        || workers_global_csr != 0
        || baseline.total_nnz != graph.total_nnz
    {
        return Err("H1-C3F MPI SpMV / traffic / ownership invariant failed".into());
    }
    if expect_graph && (graph.ghosts == 0 || graph.ghosts >= baseline.ghosts) {
        return Err("H1-C3F fixture did not preserve nonzero traffic and reduce ghosts".into());
    }
    mpi.barrier();
    if rank == 0 {
        let baseline_median = median(&baseline.max_rank_spmv_ms);
        let graph_median = median(&graph.max_rank_spmv_ms);
        println!(
            "PASS H1-C3F ranks={expected} n={n} nnz={} policy={:?} decision={:?} guard={:?} workers_global_csr={workers_global_csr} baseline_ghosts={} selected_ghosts={} baseline_max_rank_spmv_median_ms={baseline_median:.6} selected_max_rank_spmv_median_ms={graph_median:.6} root_select_ms={root_select_ms:.6} baseline_native_setup_max_ms={:.6} selected_native_setup_max_ms={:.6} baseline_abs_err={baseline_error:.3e} selected_abs_err={graph_error:.3e}",
            graph.total_nnz,
            graph.selection.selected_kind,
            graph.selection.decision,
            graph.selection.calibration_guard,
            baseline.ghosts,
            graph.ghosts,
            baseline.setup_max_ms,
            graph.setup_max_ms,
        );
        println!("H1-C3F offsets baseline={baseline_offsets:?} selected={graph_offsets:?} warmups={WARMUPS} measured={SAMPLES} note=illustrative-calibration-not-speedup-proof");
    }
    Ok(())
}
