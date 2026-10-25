//! H1-C3K5: experimental noncontiguous-ownership Jacobi-PCG on native MPI.
//!
//! This is NOT production MPI-PCG. All ranks read the same Matrix Market CSR,
//! and construct a *synthetic SPD surrogate from its undirected sparsity graph*.
//! Original matrix coefficients are NOT solved. The surrogate has symmetric
//! negative off-diagonals and strict positive diagonal dominance.
//! The K3 MPI assignment Halo/SpMV is reused without modifying its code.
//! Results apply to single-node MPI until separately validated on >=2 hosts.

use hybit_distributed::mpi_assignment::{
    AssignmentHaloPlan, AssignmentLocalCsr, AssignmentOwnedCsrRows, prepare_assignment_owned_rows,
    spmv_assignment,
};
use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::{
    AbtmMultilevelOptions, ContiguousPartition, PartitionAssignment, PartitionTelemetry,
    abtm_balanced_multisource_partition, abtm_multilevel_partition, abtm_region_grow_partition,
    field_graph_contiguous_partition, partition_telemetry_assignment,
};
use hybit_matrix::{Csr32Matrix, DofBlock, DofEntityKind, FieldAwareDofLayout, read_matrix_market};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;
const MAX_INPUT_BYTES: u64 = 110 * 1024 * 1024;
const MAX_DOFS: usize = 200_000;
const MAX_NNZ: usize = 8_000_000;

#[derive(Clone, Copy, Debug)]
enum Policy {
    Equal,
    Graph,
    Region,
    Multisource,
    Multilevel,
}

impl Policy {
    fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Graph => "graph_contiguous",
            Self::Region => "abtm_region_growth",
            Self::Multisource => "abtm_multisource",
            Self::Multilevel => "abtm_multilevel",
        }
    }
}

fn world_max(value: f64) -> f64 {
    let mut result = 0.0f64;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &value,
        &mut result,
        SystemOperation::max(),
    );
    result
}

fn world_sum(value: f64) -> f64 {
    let mut result = 0.0f64;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &value,
        &mut result,
        SystemOperation::sum(),
    );
    result
}

/// Times ONLY the scalar MPI Allreduce portion of every PCG dot product.
/// Local dot evaluation is independently timed; setup and reporting reductions
/// stay outside this scope. This is observational instrumentation, not pipelined PCG.
#[derive(Default, Debug)]
struct DotTiming {
    local_ns: u128,
    allreduce_ns: u128,
    calls: usize,
}

fn world_dot_timed(a: &[f64], b: &[f64], timer: &mut DotTiming) -> f64 {
    let local_started = Instant::now();
    let local = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>();
    timer.local_ns += local_started.elapsed().as_nanos();
    let collective_started = Instant::now();
    let global = world_sum(local);
    timer.allreduce_ns += collective_started.elapsed().as_nanos();
    timer.calls += 1;
    global
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    }
}

fn x_exact(global: u64) -> f64 {
    let t = global as f64 + 1.0;
    (t * 0.017).sin() + 0.125 * (t * 0.0031).cos()
}

/// Preserve the *undirected input sparsity*, but not the input coefficients.
/// Each off-diagonal pair has equal negative weight. Diagonal = row sum of
/// off-diagonal absolute weights + positive `shift`. Hence the matrix is
/// symmetric strictly diagonally dominant with positive diagonal => SPD.
fn make_spd_surrogate(input: &Csr32Matrix, shift: f64) -> BenchResult<Csr32Matrix> {
    if !shift.is_finite() || shift <= 0.0 || input.nrows() != input.ncols() {
        return Err("surrogate requires a square input and positive finite shift".into());
    }
    let n = input.nrows();
    let mut neighbors = vec![BTreeSet::<u32>::new(); n];
    for row in 0..n {
        let first = input.row_ptr()[row] as usize;
        let last = input.row_ptr()[row + 1] as usize;
        for &col in &input.col_idx()[first..last] {
            if col as usize != row {
                neighbors[row].insert(col);
                neighbors[col as usize].insert(row as u32);
            }
        }
    }
    let degree: Vec<usize> = neighbors.iter().map(BTreeSet::len).collect();
    let mut row_ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for (row, neighbors_row) in neighbors.iter().enumerate() {
        let mut diagonal = shift;
        let mut entries = Vec::with_capacity(neighbors_row.len() + 1);
        for &col in neighbors_row {
            let weight =
                -1.0 / (((degree[row] + 1) as f64 * (degree[col as usize] + 1) as f64).sqrt());
            diagonal -= weight;
            entries.push((col, weight));
        }
        entries.push((row as u32, diagonal));
        entries.sort_unstable_by_key(|(col, _)| *col);
        for (col, value) in entries {
            cols.push(col);
            vals.push(value);
        }
        row_ptr.push(u32::try_from(cols.len()).map_err(|_| "surrogate nnz exceeds u32")?);
    }
    if cols.len() > MAX_NNZ {
        return Err("symmetrized surrogate exceeds K5 expanded nnz cap".into());
    }
    Ok(Csr32Matrix::new(n, n, row_ptr, cols, vals)?)
}

fn scalar_layout(n: usize) -> BenchResult<FieldAwareDofLayout> {
    Ok(FieldAwareDofLayout::new(
        (0..n)
            .map(|i| DofBlock {
                entity_kind: DofEntityKind::Node,
                entity_id: i as u64,
                field_id: 0,
                components: 1,
            })
            .collect(),
    )?)
}

fn propose(
    matrix: &Csr32Matrix,
    ranks: u32,
    radius: usize,
    policy: Policy,
) -> BenchResult<PartitionAssignment> {
    Ok(match policy {
        Policy::Equal => PartitionAssignment::from_contiguous(&ContiguousPartition::balanced(
            matrix.nrows() as u64,
            ranks,
        )?)?,
        Policy::Graph => {
            let (p, _) = field_graph_contiguous_partition(
                matrix,
                &scalar_layout(matrix.nrows())?,
                ranks,
                radius,
            )?;
            PartitionAssignment::from_contiguous(&p)?
        }
        Policy::Region => abtm_region_grow_partition(matrix, ranks)?.0,
        Policy::Multisource => abtm_balanced_multisource_partition(matrix, ranks)?.0,
        Policy::Multilevel => {
            abtm_multilevel_partition(matrix, ranks, AbtmMultilevelOptions::default())?.0
        }
    })
}

fn inverse_diagonal(rows: &AssignmentOwnedCsrRows) -> BenchResult<Vec<f64>> {
    if rows.row_ptr.len() != rows.owned_globals.len() + 1 {
        return Err("invalid local row pointer for Jacobi".into());
    }
    let mut inverse = Vec::with_capacity(rows.owned_globals.len());
    for (row, &global) in rows.owned_globals.iter().enumerate() {
        let first = rows.row_ptr[row] as usize;
        let last = rows.row_ptr[row + 1] as usize;
        if last > rows.values.len() || last > rows.global_col_idx.len() || first >= last {
            return Err("invalid Jacobi row bounds".into());
        }
        let mut diagonal = None;
        for k in first..last {
            if rows.global_col_idx[k] == global && diagonal.replace(rows.values[k]).is_some() {
                return Err("duplicate Jacobi diagonal".into());
            }
        }
        let value = diagonal.ok_or("missing Jacobi diagonal")?;
        if !value.is_finite() || value <= 0.0 {
            return Err("Jacobi requires strictly positive finite diagonal".into());
        }
        inverse.push(1.0 / value);
    }
    Ok(inverse)
}

#[derive(Debug)]
struct SolveResult {
    iterations: usize,
    true_rel: f64,
    max_solution_error: f64,
    elapsed_ms: f64,
    exchange_ms: f64,
    numeric_ms: f64,
    spmv_calls: usize,
    dot_reductions: usize,
    dot_local_ms: f64,
    dot_allreduce_ms: f64,
    validation_allreduce_ms: f64,
}

/// Collective PCG experiment. Fixed positive Jacobi, zero initial guess,
/// globally reduced dot products, positive-curvature guard, true-residual
/// certification against b-Ax. No automatic solver routing/fallback.
fn solve_pcg(
    mpi: &MpiRuntime,
    halo: &AssignmentHaloPlan,
    operator: &AssignmentLocalCsr,
    invdiag: &[f64],
    rhs: &[f64],
    max_iters: usize,
    tol: f64,
) -> BenchResult<SolveResult> {
    let n = halo.owned_len();
    if mpi.all_reduce_sum_u64(u64::from(
        rhs.len() != n || invdiag.len() != n || !tol.is_finite() || tol <= 0.0,
    )) != 0
    {
        return Err("invalid K5 PCG input".into());
    }
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    let mut scratch = halo.workspace();
    let mut exchange_ns = 0u128;
    let mut numeric_ns = 0u128;
    let mut spmv_calls = 0usize;
    let mut dot_reductions = 0usize;
    let mut dot_timer = DotTiming::default();

    mpi.barrier();
    let started = Instant::now();
    let norm_b_sq = world_dot_timed(rhs, rhs, &mut dot_timer);
    dot_reductions += 1;
    if !norm_b_sq.is_finite() || norm_b_sq <= 0.0 {
        return Err("K5 manufactured RHS must have finite positive norm".into());
    }
    let norm_b = norm_b_sq.sqrt();
    for ((zi, pi), (&d, &ri)) in z.iter_mut().zip(&mut p).zip(invdiag.iter().zip(&r)) {
        let value = d * ri;
        *zi = value;
        *pi = value;
    }
    let mut rz = world_dot_timed(&r, &z, &mut dot_timer);
    dot_reductions += 1;
    if !rz.is_finite() || rz <= 0.0 {
        return Err("PCG invalid initial preconditioned residual".into());
    }
    let mut iterations = 0usize;
    let mut recursive_converged = false;
    for iteration in 0..max_iters {
        let timing = spmv_assignment(mpi, halo, operator, &p, &mut scratch, &mut ap)?;
        exchange_ns += timing.halo_ns;
        numeric_ns += timing.numeric_ns;
        spmv_calls += 1;
        let curvature = world_dot_timed(&p, &ap, &mut dot_timer);
        dot_reductions += 1;
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err("PCG nonpositive curvature (SPD contract violated)".into());
        }
        let alpha = rz / curvature;
        if !alpha.is_finite() {
            return Err("PCG nonfinite alpha".into());
        }
        for ((xi, ri), (&pi, &api)) in x.iter_mut().zip(&mut r).zip(p.iter().zip(&ap)) {
            *xi += alpha * pi;
            *ri -= alpha * api;
        }
        iterations = iteration + 1;
        let rr = world_dot_timed(&r, &r, &mut dot_timer);
        dot_reductions += 1;
        if !rr.is_finite() || rr < 0.0 {
            return Err("PCG nonfinite recursive residual".into());
        }
        if rr.sqrt() <= 0.5 * tol * norm_b {
            recursive_converged = true;
            break;
        }
        for (zi, (&d, &ri)) in z.iter_mut().zip(invdiag.iter().zip(&r)) {
            *zi = d * ri;
        }
        let next_rz = world_dot_timed(&r, &z, &mut dot_timer);
        dot_reductions += 1;
        if !next_rz.is_finite() || next_rz <= 0.0 {
            return Err("PCG invalid preconditioned recurrence".into());
        }
        let beta = next_rz / rz;
        if !beta.is_finite() {
            return Err("PCG nonfinite beta".into());
        }
        for (pi, &zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
        rz = next_rz;
    }
    if !recursive_converged {
        return Err("PCG did not converge within max_iters".into());
    }
    // Independently recompute the true residual rather than trusting the
    // recursive estimate. Include this final SpMV in measured solve cost.
    let timing = spmv_assignment(mpi, halo, operator, &x, &mut scratch, &mut ap)?;
    exchange_ns += timing.halo_ns;
    numeric_ns += timing.numeric_ns;
    spmv_calls += 1;
    for (residual, (&b, &ax)) in r.iter_mut().zip(rhs.iter().zip(&ap)) {
        *residual = b - ax;
    }
    let true_rr = world_dot_timed(&r, &r, &mut dot_timer);
    dot_reductions += 1;
    let true_rel = true_rr.sqrt() / norm_b;
    let local_error = x
        .iter()
        .zip(&halo.owned_globals)
        .map(|(&value, &global)| (value - x_exact(global)).abs())
        .fold(0.0_f64, f64::max);
    let validation_started = Instant::now();
    let max_solution_error = world_max(local_error);
    let validation_allreduce_ms = validation_started.elapsed().as_secs_f64() * 1000.0;
    if dot_timer.calls != dot_reductions {
        return Err("K5 dot-reduction count/timer mismatch".into());
    }
    if !true_rel.is_finite() || true_rel > tol || !max_solution_error.is_finite() {
        return Err(format!("K5 true-residual/solution certification failed: rel={true_rel:.3e} error={max_solution_error:.3e}").into());
    }
    Ok(SolveResult {
        iterations,
        true_rel,
        max_solution_error,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        exchange_ms: exchange_ns as f64 * 1e-6,
        numeric_ms: numeric_ns as f64 * 1e-6,
        spmv_calls,
        dot_reductions,
        dot_local_ms: dot_timer.local_ns as f64 * 1e-6,
        dot_allreduce_ms: dot_timer.allreduce_ns as f64 * 1e-6,
        validation_allreduce_ms,
    })
}

struct Case {
    policy: Policy,
    telemetry: PartitionTelemetry,
    halo: AssignmentHaloPlan,
    local: AssignmentLocalCsr,
    inverse: Vec<f64>,
    rhs: Vec<f64>,
    partition_ms: f64,
    extraction_ms: f64,
    halo_setup_ms: f64,
    rhs_setup_ms: f64,
    resident_bytes_sum: u64,
    samples_ms: Vec<f64>,
    exchange_samples_ms: Vec<f64>,
    numeric_samples_ms: Vec<f64>,
    true_rel_samples: Vec<f64>,
    dot_local_samples_ms: Vec<f64>,
    dot_allreduce_samples_ms: Vec<f64>,
    validation_allreduce_samples_ms: Vec<f64>,
    reference_iters: Option<usize>,
    worst_true_rel: f64,
    worst_solution_error: f64,
    spmv_calls: usize,
    dot_reductions: usize,
}

fn prepare_case(
    mpi: &MpiRuntime,
    matrix: &Csr32Matrix,
    ranks: u32,
    radius: usize,
    policy: Policy,
) -> BenchResult<Case> {
    mpi.barrier();
    let partition_started = Instant::now();
    let root = if mpi.rank() == 0 {
        Some(propose(matrix, ranks, radius, policy))
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(root.as_ref().is_some_and(Result::is_err))) != 0 {
        if let Some(Err(err)) = root.as_ref() {
            eprintln!("K5 root partition {} failed: {err}", policy.label());
        }
        return Err("partition proposal failed on rank zero".into());
    }
    let mut owners = if let Some(Ok(assignment)) = root {
        assignment.owners().to_vec()
    } else {
        vec![0u32; matrix.nrows()]
    };
    mpi::topology::SimpleCommunicator::world()
        .process_at_rank(0)
        .broadcast_into(&mut owners[..]);
    let assignment = PartitionAssignment::from_owners(ranks, owners)?;
    let partition_ms = world_max(partition_started.elapsed().as_secs_f64() * 1000.0);
    let telemetry = partition_telemetry_assignment(matrix, &assignment)?;

    mpi.barrier();
    let extraction_started = Instant::now();
    let rows = AssignmentOwnedCsrRows::from_global(matrix, &assignment, mpi.rank() as u32);
    if mpi.all_reduce_sum_u64(u64::from(rows.is_err())) != 0 {
        return Err("K5 local CSR extraction failed on one or more ranks".into());
    }
    let rows = rows?;
    let inverse = inverse_diagonal(&rows);
    if mpi.all_reduce_sum_u64(u64::from(inverse.is_err())) != 0 {
        return Err("K5 Jacobi preparation failed on one or more ranks".into());
    }
    let inverse = inverse?;
    let extraction_ms = world_max(extraction_started.elapsed().as_secs_f64() * 1000.0);

    mpi.barrier();
    let halo_started = Instant::now();
    let (halo, local, stats) = prepare_assignment_owned_rows(mpi, &assignment, &rows)?;
    let halo_setup_ms = world_max(halo_started.elapsed().as_secs_f64() * 1000.0);
    let ghosts = mpi.all_reduce_sum_u64(stats.ghosts as u64);
    let peers = mpi.all_reduce_sum_u64(stats.peer_ranks as u64);
    let sent = mpi.all_reduce_sum_u64(stats.sent_f64 as u64);
    let received = mpi.all_reduce_sum_u64(stats.received_f64 as u64);
    let resident_bytes_sum = mpi.all_reduce_sum_u64(stats.rank_operator_halo_bytes as u64);
    if ghosts != telemetry.communication_volume as u64
        || peers != telemetry.directional_peer_relations as u64
        || sent != received
        || sent != ghosts
    {
        return Err(format!("K5 {} Halo telemetry mismatch", policy.label()).into());
    }

    mpi.barrier();
    let rhs_started = Instant::now();
    let known = halo
        .owned_globals
        .iter()
        .map(|&g| x_exact(g))
        .collect::<Vec<_>>();
    let mut rhs = vec![0.0; halo.owned_len()];
    spmv_assignment(mpi, &halo, &local, &known, &mut halo.workspace(), &mut rhs)?;
    let rhs_setup_ms = world_max(rhs_started.elapsed().as_secs_f64() * 1000.0);
    Ok(Case {
        policy,
        telemetry,
        halo,
        local,
        inverse,
        rhs,
        partition_ms,
        extraction_ms,
        halo_setup_ms,
        rhs_setup_ms,
        resident_bytes_sum,
        samples_ms: Vec::new(),
        exchange_samples_ms: Vec::new(),
        numeric_samples_ms: Vec::new(),
        true_rel_samples: Vec::new(),
        dot_local_samples_ms: Vec::new(),
        dot_allreduce_samples_ms: Vec::new(),
        validation_allreduce_samples_ms: Vec::new(),
        reference_iters: None,
        worst_true_rel: 0.0,
        worst_solution_error: 0.0,
        spmv_calls: 0,
        dot_reductions: 0,
    })
}

fn run_once(
    mpi: &MpiRuntime,
    case: &mut Case,
    max_iters: usize,
    tol: f64,
    measured: bool,
) -> BenchResult<()> {
    let result = solve_pcg(
        mpi,
        &case.halo,
        &case.local,
        &case.inverse,
        &case.rhs,
        max_iters,
        tol,
    )?;
    let wall = world_max(result.elapsed_ms);
    let exchange = world_max(result.exchange_ms);
    let numeric = world_max(result.numeric_ms);
    let dot_local = world_max(result.dot_local_ms);
    let dot_allreduce = world_max(result.dot_allreduce_ms);
    let validation_allreduce = world_max(result.validation_allreduce_ms);
    if let Some(previous) = case.reference_iters {
        if previous != result.iterations {
            return Err("K5 nondeterministic iteration count across identical RHS".into());
        }
    } else {
        case.reference_iters = Some(result.iterations);
    }
    if measured {
        case.samples_ms.push(wall);
        case.exchange_samples_ms.push(exchange);
        case.numeric_samples_ms.push(numeric);
        case.true_rel_samples.push(result.true_rel);
        case.dot_local_samples_ms.push(dot_local);
        case.dot_allreduce_samples_ms.push(dot_allreduce);
        case.validation_allreduce_samples_ms
            .push(validation_allreduce);
        case.worst_true_rel = case.worst_true_rel.max(result.true_rel);
        case.worst_solution_error = case.worst_solution_error.max(result.max_solution_error);
        case.spmv_calls = result.spmv_calls;
        case.dot_reductions = result.dot_reductions;
    }
    Ok(())
}

fn main() -> BenchResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 9 {
        return Err("usage: mpi_k5_assignment_pcg <ranks> <matrix.mtx> <radius> <warmups> <repeats> <max_iters> <relative_tol> <positive_shift> <results.csv>".into());
    }
    let expected: i32 = args[0].parse()?;
    let path = Path::new(&args[1]);
    let radius: usize = args[2].parse()?;
    let warmups: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    let max_iters: usize = args[5].parse()?;
    let tol: f64 = args[6].parse()?;
    let shift: f64 = args[7].parse()?;
    let csv = Path::new(&args[8]);
    if !matches!(expected, 1 | 2 | 4 | 8)
        || !(1..=256).contains(&radius)
        || warmups > 10
        || !(1..=30).contains(&repeats)
        || !(1..=5000).contains(&max_iters)
        || !(1.0e-14..=1.0e-3).contains(&tol)
        || !(1.0e-5..=10.0).contains(&shift)
    {
        return Err("K5 args outside guarded bounds".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected {
        return Err("MPI rank count differs from K5 input".into());
    }
    // Paired per-trial raw file is mandatory for run-to-run variability checks.
    let raw_csv = csv.with_extension("samples.csv");
    if mpi.all_reduce_sum_u64(u64::from(
        mpi.rank() == 0 && (csv.exists() || raw_csv.exists()),
    )) != 0
    {
        return Err("K5 summary/raw CSV already exists; refuse to overwrite".into());
    }
    mpi.barrier();
    let read_started = Instant::now();
    let loaded: BenchResult<_> = (|| {
        if std::fs::metadata(path)?.len() > MAX_INPUT_BYTES {
            return Err("K5 global input exceeds replicated-CSR size cap".into());
        }
        Ok(read_matrix_market(path)?)
    })();
    if mpi.all_reduce_sum_u64(u64::from(loaded.is_err())) != 0 {
        return Err("K5 Matrix Market input failed on one or more ranks".into());
    }
    let (input, info) = loaded?;
    let n = input.nrows();
    if mpi.all_reduce_sum_u64(u64::from(
        n == 0
            || n != input.ncols()
            || n > MAX_DOFS
            || n < expected as usize
            || input.nnz() > MAX_NNZ,
    )) != 0
    {
        return Err("K5 input outside square/n/nnz/rank bounds".into());
    }
    let read_max_ms = world_max(read_started.elapsed().as_secs_f64() * 1000.0);

    mpi.barrier();
    let surrogate_started = Instant::now();
    let surrogate = make_spd_surrogate(&input, shift);
    if mpi.all_reduce_sum_u64(u64::from(surrogate.is_err())) != 0 {
        return Err("K5 SPD surrogate generation failed on at least one rank".into());
    }
    let matrix = surrogate?;
    let surrogate_max_ms = world_max(surrogate_started.elapsed().as_secs_f64() * 1000.0);
    let cases_policy = [
        Policy::Equal,
        Policy::Graph,
        Policy::Region,
        Policy::Multisource,
        Policy::Multilevel,
    ];
    let mut cases = Vec::with_capacity(cases_policy.len());
    for policy in cases_policy {
        cases.push(prepare_case(
            &mpi,
            &matrix,
            expected as u32,
            radius,
            policy,
        )?);
    }
    for trial in 0..(warmups + repeats) {
        for shift in 0..cases.len() {
            let index = (shift + trial) % cases.len();
            run_once(&mpi, &mut cases[index], max_iters, tol, trial >= warmups)?;
        }
    }
    let benchmark_iterations = cases[0].reference_iters.unwrap_or(0);
    if mpi.all_reduce_sum_u64(u64::from(
        cases
            .iter()
            .any(|c| c.reference_iters != Some(benchmark_iterations)),
    )) != 0
    {
        return Err("K5 PCG iteration count differs across ownership policies".into());
    }

    mpi.barrier();
    if mpi.rank() == 0 {
        let mut output = OpenOptions::new().write(true).create_new(true).open(csv)?;
        let mut raw = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&raw_csv)?;
        writeln!(
            output,
            "matrix,scope,ranks,n,input_nnz,surrogate_nnz,shift,method,radius,warmups,repeats,max_iters,tolerance,read_max_ms,surrogate_max_ms,partition_max_ms,owned_extract_max_ms,halo_setup_max_ms,rhs_setup_max_ms,median_max_solve_ms,median_max_exchange_sum_ms,median_max_numeric_sum_ms,median_max_dot_local_sum_ms,median_max_dot_allreduce_sum_ms,median_max_validation_allreduce_ms,iterations,spmv_calls,explicit_dot_reductions,true_rel_residual,max_abs_solution_error,ghosts,cut_nnz,peer_relations,owned_imbalance,nnz_imbalance,rank_operator_halo_sum_bytes"
        )?;
        writeln!(
            raw,
            "matrix,scope,ranks,method,trial,solve_max_ms,exchange_sum_max_ms,numeric_sum_max_ms,dot_local_sum_max_ms,dot_allreduce_sum_max_ms,validation_allreduce_max_ms,iterations,true_rel_residual"
        )?;
        for case in &cases {
            let solve = median(&case.samples_ms);
            let exchange = median(&case.exchange_samples_ms);
            let numeric = median(&case.numeric_samples_ms);
            let dot_local = median(&case.dot_local_samples_ms);
            let dot_collective = median(&case.dot_allreduce_samples_ms);
            let validation_collective = median(&case.validation_allreduce_samples_ms);
            for trial in 0..case.samples_ms.len() {
                writeln!(
                    raw,
                    "{},synthetic_spd_single_node_mpi_experiment,{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.12e}",
                    path.display(),
                    expected,
                    case.policy.label(),
                    trial + 1,
                    case.samples_ms[trial],
                    case.exchange_samples_ms[trial],
                    case.numeric_samples_ms[trial],
                    case.dot_local_samples_ms[trial],
                    case.dot_allreduce_samples_ms[trial],
                    case.validation_allreduce_samples_ms[trial],
                    case.reference_iters.unwrap_or(0),
                    case.true_rel_samples[trial],
                )?;
            }
            let s = &case.telemetry;
            let iters = case.reference_iters.unwrap_or(0);
            writeln!(
                output,
                "{},synthetic_spd_single_node_mpi_experiment,{},{},{},{},{:.8},{},{},{},{},{},{:.3e},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{solve:.6},{exchange:.6},{numeric:.6},{dot_local:.6},{dot_collective:.6},{validation_collective:.6},{iters},{},{},{:.12e},{:.12e},{},{},{},{:.8},{:.8},{}",
                path.display(),
                expected,
                n,
                info.input_entries,
                matrix.nnz(),
                shift,
                case.policy.label(),
                radius,
                warmups,
                repeats,
                max_iters,
                tol,
                read_max_ms,
                surrogate_max_ms,
                case.partition_ms,
                case.extraction_ms,
                case.halo_setup_ms,
                case.rhs_setup_ms,
                case.spmv_calls,
                case.dot_reductions,
                case.worst_true_rel,
                case.worst_solution_error,
                s.communication_volume,
                s.cut_nnz,
                s.directional_peer_relations,
                s.owned_dof_imbalance,
                s.local_nnz_imbalance,
                case.resident_bytes_sum
            )?;
            println!(
                "K5 NATIVE PASS method={} ranks={} iterations={} solve_median_ms={solve:.6} exchange_sum_median_ms={exchange:.6} dot_allreduce_sum_median_ms={dot_collective:.6} dot_local_sum_median_ms={dot_local:.6} true_rel={:.3e} max_solution_err={:.3e} ghosts={} cut={} partition_ms={:.3} extraction_ms={:.3} halo_ms={:.3}",
                case.policy.label(),
                expected,
                iters,
                case.worst_true_rel,
                case.worst_solution_error,
                s.communication_volume,
                s.cut_nnz,
                case.partition_ms,
                case.extraction_ms,
                case.halo_setup_ms
            );
        }
        output.flush()?;
        raw.flush()?;
        println!(
            "=== H1-C3K5 NATIVE MPI-PCG PROFILE PASS: {} + {} ===",
            csv.display(),
            raw_csv.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_and_even_are_exact() {
        assert_eq!(median(&[3.0, 1.0, 5.0]), 3.0);
        assert_eq!(median(&[4.0, 2.0, 3.0, 1.0]), 2.5);
    }

    #[test]
    fn spd_surrogate_is_symmetric_and_strictly_diagonally_dominant() {
        let input = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 7],
            vec![0, 1, 1, 2, 2, 3, 3],
            vec![1.0; 7],
        )
        .unwrap();
        let surrogate = make_spd_surrogate(&input, 0.02).unwrap();
        let mut entries = std::collections::BTreeMap::new();
        for row in 0..surrogate.nrows() {
            let mut sum_abs = 0.0;
            let mut diagonal = 0.0;
            for k in surrogate.row_ptr()[row] as usize..surrogate.row_ptr()[row + 1] as usize {
                let col = surrogate.col_idx()[k] as usize;
                let value = surrogate.values()[k];
                entries.insert((row, col), value);
                if row == col {
                    diagonal = value;
                } else {
                    sum_abs += value.abs();
                }
            }
            assert!(diagonal > sum_abs);
        }
        for ((row, col), value) in &entries {
            assert_eq!(entries.get(&(*col, *row)), Some(value));
        }
    }

    #[test]
    fn jacobi_extracts_only_owned_diagonals() {
        let input = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 1, 2, 2, 3, 0, 3],
            vec![4., -1., 5., -1., 6., -1., -1., 7.],
        )
        .unwrap();
        let owners = PartitionAssignment::from_owners(2, vec![0, 1, 0, 1]).unwrap();
        let rows = AssignmentOwnedCsrRows::from_global(&input, &owners, 0).unwrap();
        assert_eq!(inverse_diagonal(&rows).unwrap(), vec![0.25, 1.0 / 6.0]);
    }
}
