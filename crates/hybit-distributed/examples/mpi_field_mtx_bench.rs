//! H1-C3B: native MS-MPI SpMV comparison on real Matrix Market matrices.
//!
//! This is a bounded, replicated-global-CSR *experiment*, not a scalable
//! distributed reader or a production partition selector. Matrix Market files
//! have no field/element layout: one scalar DOF is one synthetic field block.
//! No SPD/PCG assumption is made; both general and symmetric inputs are valid.

use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit_distributed::{
    build_contiguous_halo_plans, field_aligned_contiguous_partition,
    field_graph_contiguous_partition, partition_telemetry, ContiguousPartition, HaloPlan,
    PartitionTelemetry, RankLocalCsr,
};
use hybit_matrix::{
    read_matrix_market, Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout,
    MatrixMarketSymmetry,
};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const MAX_INPUT_BYTES: u64 = 110 * 1024 * 1024;
const MAX_MATRIX_DOFS: usize = 200_000;
const MAX_EXPANDED_NNZ: usize = 8_000_000;

fn max_f64(value: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut result = 0.0f64;
    world.all_reduce_into(&value, &mut result, SystemOperation::max());
    result
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn scalar_control_layout(n: usize) -> Result<FieldAwareDofLayout, Box<dyn Error>> {
    let blocks = (0..n)
        .map(|i| DofBlock {
            entity_kind: DofEntityKind::Node,
            entity_id: i as u64,
            field_id: 0,
            components: 1,
        })
        .collect();
    Ok(FieldAwareDofLayout::new(blocks)?)
}

fn owned_rows(matrix: &Csr32Matrix, plan: &HaloPlan) -> OwnedCsrRows {
    let owned = plan.owned_range();
    let first = owned.start as usize;
    let end = owned.end as usize;
    let nnz_first = matrix.row_ptr()[first] as usize;
    let nnz_end = matrix.row_ptr()[end] as usize;
    OwnedCsrRows {
        global_dofs: matrix.nrows() as u64,
        owned,
        row_ptr: matrix.row_ptr()[first..=end]
            .iter()
            .map(|&ptr| ptr - matrix.row_ptr()[first])
            .collect(),
        global_col_idx: matrix.col_idx()[nnz_first..nnz_end]
            .iter()
            .map(|&col| u64::from(col))
            .collect(),
        values: matrix.values()[nnz_first..nnz_end].to_vec(),
    }
}

struct Case {
    label: &'static str,
    halo: HaloPlan,
    local: RankLocalCsr,
    telemetry: PartitionTelemetry,
    partition_prepare_max_ms: f64,
    native_setup_max_ms: f64,
    resident_bytes_estimate: u64,
    extended: Vec<f64>,
    output: Vec<f64>,
    samples_ms: Vec<f64>,
    worst_abs_error: f64,
    worst_relative_error: f64,
    sent_per_apply: u64,
    received_per_apply: u64,
}

fn prepare_case(
    mpi: &MpiRuntime,
    matrix: &Csr32Matrix,
    label: &'static str,
    partition: &ContiguousPartition,
    partition_prepare_max_ms: f64,
) -> Result<Case, Box<dyn Error>> {
    let rank = mpi.rank() as usize;
    let reference_plans = build_contiguous_halo_plans(matrix, partition)?;
    let telemetry = partition_telemetry(matrix, partition, &reference_plans)?;
    let rows = owned_rows(matrix, &reference_plans[rank]);

    mpi.barrier();
    let started = Instant::now();
    let (halo, local, stats) = prepare_owned_rows(mpi, partition, &rows)?;
    let native_setup_max_ms = max_f64(started.elapsed().as_secs_f64() * 1e3);

    // No rank may quit before the other ranks reach their next collective.
    let mismatch = halo != reference_plans[rank]
        || local.row_ptr() != rows.row_ptr.as_slice()
        || local.values() != rows.values.as_slice()
        || stats.ghosts != halo.ghost_len();
    if mpi.all_reduce_sum_u64(u64::from(mismatch)) != 0 {
        return Err(format!("{label}: native owned-Halo differs from reference").into());
    }
    let resident_bytes_estimate =
        mpi.all_reduce_sum_u64(stats.operator_and_halo_bytes_estimate as u64);
    let ghosts = mpi.all_reduce_sum_u64(halo.ghost_len() as u64);
    let peers = mpi.all_reduce_sum_u64(halo.peers().len() as u64);
    let consistent = ghosts as usize == telemetry.communication_volume
        && peers as usize == telemetry.directional_peer_relations;
    if mpi.all_reduce_sum_u64(u64::from(!consistent)) != 0 {
        return Err(format!("{label}: native Halo traffic topology differs from reference").into());
    }
    Ok(Case {
        label,
        telemetry,
        partition_prepare_max_ms,
        native_setup_max_ms,
        resident_bytes_estimate,
        extended: vec![0.0; halo.extended_len()],
        output: vec![0.0; halo.owned_len()],
        halo,
        local,
        samples_ms: Vec::new(),
        worst_abs_error: 0.0,
        worst_relative_error: 0.0,
        sent_per_apply: 0,
        received_per_apply: 0,
    })
}

fn run_one_spmv(
    mpi: &MpiRuntime,
    case: &mut Case,
    x: &[f64],
    reference: &[f64],
    measured: bool,
) -> Result<(), Box<dyn Error>> {
    let owned = case.halo.owned_range();
    let start = owned.start as usize;
    let end = owned.end as usize;
    mpi.barrier();
    let started = Instant::now();
    let traffic = mpi.spmv_local(
        &case.halo,
        &case.local,
        &x[start..end],
        &mut case.extended,
        &mut case.output,
    )?;
    let max_elapsed_ms = max_f64(started.elapsed().as_secs_f64() * 1e3);
    let total_sent = mpi.all_reduce_sum_u64(traffic.sent_values as u64);
    let total_recv = mpi.all_reduce_sum_u64(traffic.received_values as u64);
    let reference_scale = reference[start..end]
        .iter()
        .map(|v| v.abs())
        .fold(1.0f64, f64::max);
    let mut local_abs_error = 0.0f64;
    let mut finite = true;
    for (&actual, &truth) in case.output.iter().zip(&reference[start..end]) {
        finite &= actual.is_finite() && truth.is_finite();
        local_abs_error = local_abs_error.max((actual - truth).abs());
    }
    for (slot, &global) in case.halo.ghost_globals().iter().enumerate() {
        let v = case.extended[case.halo.owned_len() + slot];
        finite &= v.is_finite();
        local_abs_error = local_abs_error.max((v - x[global as usize]).abs());
    }
    let absolute = max_f64(local_abs_error);
    let global_scale = max_f64(reference_scale);
    let relative = absolute / global_scale;
    let bad = !finite
        || !absolute.is_finite()
        || relative > 1e-11
        || total_sent != total_recv
        || total_recv as usize != case.telemetry.communication_volume;
    if mpi.all_reduce_sum_u64(u64::from(bad)) != 0 {
        return Err(format!(
            "{} native SpMV/traffic mismatch: abs={absolute:.3e} rel={relative:.3e} send={total_sent} recv={total_recv}",
            case.label
        )
        .into());
    }
    if measured {
        case.samples_ms.push(max_elapsed_ms);
        case.worst_abs_error = case.worst_abs_error.max(absolute);
        case.worst_relative_error = case.worst_relative_error.max(relative);
        case.sent_per_apply = total_sent;
        case.received_per_apply = total_recv;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 7 {
        return Err("usage: mpi_field_mtx_bench <ranks> <matrix.mtx> <label> <radius_blocks> <warmups> <repeats> <results.csv>".into());
    }
    let expected: i32 = args[0].parse()?;
    let path = Path::new(&args[1]);
    let label = &args[2];
    let radius: usize = args[3].parse()?;
    let warmups: usize = args[4].parse()?;
    let repeats: usize = args[5].parse()?;
    let csv = Path::new(&args[6]);
    if !matches!(expected, 1 | 2 | 4 | 8)
        || radius > 256
        || warmups > 20
        || repeats == 0
        || repeats > 100
        || label.contains(',')
    {
        return Err("invalid benchmark ranks/radius/warmups/repeats/label".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected {
        return Err("MS-MPI rank count does not equal requested count".into());
    }
    let ranks = expected as u32;
    let file_size = std::fs::metadata(path)?.len();
    if file_size > MAX_INPUT_BYTES {
        return Err(format!(
            "file is larger than {MAX_INPUT_BYTES} bytes; excluded from replicated-CSR H1-C3B smoke"
        )
        .into());
    }
    mpi.barrier();
    let read_started = Instant::now();
    let (matrix, info) = read_matrix_market(path)?;
    let read_max_ms = max_f64(read_started.elapsed().as_secs_f64() * 1e3);
    let n = matrix.nrows();
    if n == 0
        || n != matrix.ncols()
        || n > MAX_MATRIX_DOFS
        || matrix.nnz() > MAX_EXPANDED_NNZ
        || n < ranks as usize
    {
        return Err(
            "matrix outside H1-C3B smoke bounds (square, n <= 200000, expanded nnz <= 8000000)"
                .into(),
        );
    }
    let kind = match info.symmetry {
        MatrixMarketSymmetry::General => "general",
        MatrixMarketSymmetry::Symmetric => "symmetric",
    };
    // MTX has no physical field metadata. This surrogate is topology-only.
    let layout = scalar_control_layout(n)?;
    let started = Instant::now();
    let equal = ContiguousPartition::balanced(n as u64, ranks)?;
    let equal_partition_max_ms = max_f64(started.elapsed().as_secs_f64() * 1e3);
    let started = Instant::now();
    let field = field_aligned_contiguous_partition(&layout, ranks)?;
    let field_partition_max_ms = max_f64(started.elapsed().as_secs_f64() * 1e3);
    let started = Instant::now();
    let (graph, comparison) = field_graph_contiguous_partition(&matrix, &layout, ranks, radius)?;
    let graph_partition_max_ms = max_f64(started.elapsed().as_secs_f64() * 1e3);

    let mut cases = vec![
        prepare_case(&mpi, &matrix, "equal", &equal, equal_partition_max_ms)?,
        prepare_case(&mpi, &matrix, "field", &field, field_partition_max_ms)?,
        prepare_case(&mpi, &matrix, "graph", &graph, graph_partition_max_ms)?,
    ];
    if comparison.baseline != cases[1].telemetry || comparison.selected != cases[2].telemetry {
        return Err("graph selection telemetry does not match Halo reference".into());
    }

    for trial in 0..(warmups + repeats) {
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let v = (i as f64 + 1.0) * (trial as f64 + 1.0);
                (v * 0.017).sin() + 0.125 * (v * 0.0031).cos()
            })
            .collect();
        let reference = matrix.spmv(&x)?;
        // Alternating case order limits systematic warm-cache bias. All ranks
        // use the same order, retaining the MPI peer scheduling contract.
        let order = if trial % 2 == 0 {
            [0usize, 1, 2]
        } else {
            [2usize, 1, 0]
        };
        for index in order {
            run_one_spmv(&mpi, &mut cases[index], &x, &reference, trial >= warmups)?;
        }
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        let new_csv = !csv.exists() || std::fs::metadata(csv)?.len() == 0;
        let mut writer = OpenOptions::new().create(true).append(true).open(csv)?;
        if new_csv {
            writeln!(writer, "matrix,format,ranks,n,input_nnz,expanded_csr_nnz,policy,graph_accepted,radius,warmups,repeats,read_max_ms,partition_max_ms,native_halo_max_ms,median_max_spmv_ms,min_max_spmv_ms,max_max_spmv_ms,ghosts,cut_nnz,peer_relations,owned_imbalance,nnz_imbalance,estimated_rank_operator_halo_sum_bytes,abs_error,rel_error,sent_f64_per_apply,received_f64_per_apply")?;
        }
        for case in &cases {
            let minimum = case
                .samples_ms
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let maximum = case.samples_ms.iter().copied().fold(0.0f64, f64::max);
            let median_ms = median(&case.samples_ms);
            let stats = &case.telemetry;
            writeln!(writer, "{label},{kind},{ranks},{n},{},{},{},{},{radius},{warmups},{repeats},{read_max_ms:.6},{:.6},{:.6},{median_ms:.6},{minimum:.6},{maximum:.6},{},{},{},{:.6},{:.6},{},{:.12e},{:.12e},{},{}",
                info.input_entries, matrix.nnz(), case.label, comparison.candidate_accepted,
                case.partition_prepare_max_ms, case.native_setup_max_ms,
                stats.communication_volume, stats.cut_nnz, stats.directional_peer_relations,
                stats.owned_dof_imbalance, stats.local_nnz_imbalance,
                case.resident_bytes_estimate, case.worst_abs_error, case.worst_relative_error,
                case.sent_per_apply, case.received_per_apply)?;
            println!("PASS H1-C3B {label} ranks={ranks} policy={} accepted={} n={n} nnz={} ghosts={} cut={} peers={} owned_imbalance={:.3} nnz_imbalance={:.3} median_max_spmv_ms={median_ms:.6} partition_max_ms={:.6} halo_max_ms={:.6} rel_err={:.3e}",
                case.label, comparison.candidate_accepted, matrix.nnz(),
                stats.communication_volume, stats.cut_nnz, stats.directional_peer_relations,
                stats.owned_dof_imbalance, stats.local_nnz_imbalance,
                case.partition_prepare_max_ms, case.native_setup_max_ms,
                case.worst_relative_error);
        }
    }
    mpi.barrier();
    Ok(())
}
