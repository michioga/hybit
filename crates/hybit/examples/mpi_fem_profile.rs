//! G8-C5: MPI PCG cost decomposition, owned-CSR diagnostics, and FEM candidate probe.
//!
//! The experimental policy checks symmetry and positive diagonals, but does NOT
//! certify SPD. A converged manufactured solve is also NOT an SPD certificate.
//! Timings are rank-local inclusive durations; halo_wait is PART of SpMV time.
//! CSV medians of per-run maxima are independent and must not be summed.

use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit::distributed::mpi_mtx::{distribute_matrix_market_with_policy, MtxPcgPolicy};
use hybit::distributed::mpi_pcg::{DistributedPcg, DistributedPcgOptions, DistributedPcgStatus};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn max_f64(v: f64) -> f64 {
    let mut global = 0.0;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &v,
        &mut global,
        SystemOperation::max(),
    );
    global
}

fn min_f64(v: f64) -> f64 {
    let mut global = 0.0;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &v,
        &mut global,
        SystemOperation::min(),
    );
    global
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let middle = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[middle - 1] + v[middle]) * 0.5
    } else {
        v[middle]
    }
}

fn truth(global: u64) -> f64 {
    let u = (global + 1) as f64;
    (u * 0.017).sin() + 0.21 * (u * 0.0061).cos()
}

fn diagonal_diagnostics(rows: &OwnedCsrRows) -> (u64, f64) {
    let mut strictly_dominant = 0u64;
    let mut min_margin = f64::INFINITY;
    for (row, span) in rows.row_ptr.windows(2).enumerate() {
        let global_row = rows.owned.start + row as u64;
        let first = span[0] as usize;
        let last = span[1] as usize;
        let mut diag = 0.0;
        let mut off = 0.0;
        for pos in first..last {
            if rows.global_col_idx[pos] == global_row {
                diag += rows.values[pos];
            } else {
                off += rows.values[pos].abs();
            }
        }
        let margin = diag - off;
        if margin > 0.0 {
            strictly_dominant += 1;
        }
        min_margin = min_margin.min(margin);
    }
    (strictly_dominant, min_margin)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err("usage: mpi_fem_profile <ranks> <csv> <matrix.mtx> <label> <certified|experimental> <warmups> <repeats> <max_iters>".into());
    }
    let expected: i32 = args[0].parse()?;
    let csv = Path::new(&args[1]);
    let matrix = Path::new(&args[2]);
    let label = &args[3];
    let (policy, policy_name) = match args[4].as_str() {
        "certified" => (MtxPcgPolicy::StrictCertified, "certified"),
        "experimental" => (
            MtxPcgPolicy::ExperimentalUnverified,
            "experimental_unverified",
        ),
        _ => return Err("policy must be certified or experimental".into()),
    };
    let warmups: usize = args[5].parse()?;
    let repeats: usize = args[6].parse()?;
    let max_iters: usize = args[7].parse()?;
    if repeats == 0 || repeats > 100 || warmups > 100 || !(1..=100_000).contains(&max_iters) {
        return Err("invalid warmup/repeats/max_iters".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if expected != mpi.size() || !matches!(expected, 1 | 2 | 4 | 8) {
        return Err("MPI rank mismatch (supported: 1/2/4/8)".into());
    }
    if mpi.rank() == 0 && (!csv.exists() || std::fs::metadata(csv)?.len() == 0) {
        let mut file = OpenOptions::new().append(true).create(true).open(csv)?;
        writeln!(file, "case,policy,ranks,n,nnz,strict_dd_rows,min_diag_margin,iterations,status,true_relative_residual,max_abs_error,ghosts,root_read_ms,max_distribution_ms,max_halo_prepare_ms,max_pcg_prepare_ms,median_max_solve_ms,median_max_spmv_ms,median_max_allreduce_ms,median_max_halo_wait_ms,median_max_interior_ms,median_max_boundary_ms,median_max_unattributed_ms,median_max_over_mean,spmv_calls,allreduce_calls")?;
    }
    mpi.barrier();
    let (partition, owned, ingest) = distribute_matrix_market_with_policy(&mpi, matrix, policy)?;
    let (local_dd, local_margin) = diagonal_diagnostics(&owned);
    let dd_rows = mpi.all_reduce_sum_u64(local_dd);
    let min_margin = min_f64(local_margin);
    let started = Instant::now();
    let (plan, local, halo_stats) = prepare_owned_rows(&mpi, &partition, &owned)?;
    let halo_setup_ms = max_f64(started.elapsed().as_secs_f64() * 1000.0);
    let started = Instant::now();
    let mut pcg = DistributedPcg::prepare(&mpi, &plan, &local)?;
    let pcg_setup_ms = max_f64(started.elapsed().as_secs_f64() * 1000.0);
    let read_ms = mpi.all_reduce_sum_f64(ingest.root_read_ns as f64 * 1e-6);
    let distribute_ms = max_f64(ingest.distribution_ns as f64 * 1e-6);
    let ghosts = mpi.all_reduce_sum_u64(halo_stats.ghosts as u64);

    let mut rhs = vec![0.0; plan.owned_len()];
    for (row, value) in rhs.iter_mut().enumerate() {
        let first = owned.row_ptr[row] as usize;
        let last = owned.row_ptr[row + 1] as usize;
        for position in first..last {
            *value += owned.values[position] * truth(owned.global_col_idx[position]);
        }
    }
    let options = DistributedPcgOptions {
        relative_tolerance: 1e-9,
        absolute_tolerance: 1e-13,
        max_iterations: max_iters,
    };
    let mut solve = Vec::with_capacity(repeats);
    let mut spmv = Vec::with_capacity(repeats);
    let mut reduce = Vec::with_capacity(repeats);
    let mut wait = Vec::with_capacity(repeats);
    let mut interior = Vec::with_capacity(repeats);
    let mut boundary = Vec::with_capacity(repeats);
    let mut other = Vec::with_capacity(repeats);
    let mut imbalance = Vec::with_capacity(repeats);
    let mut iter_count = 0usize;
    let mut residual = f64::INFINITY;
    let mut error = f64::INFINITY;
    let mut status = DistributedPcgStatus::MaxIterations;
    let mut spmv_calls = 0usize;
    let mut allreduce_calls = 0usize;

    for run in 0..warmups + repeats {
        let mut x = vec![0.0; plan.owned_len()];
        mpi.barrier();
        let report = pcg.solve(&mpi, &local, &rhs, &mut x, options)?;
        let local_time = report.elapsed_ns as f64 * 1e-6;
        let elapsed = max_f64(local_time);
        let mean = mpi.all_reduce_sum_f64(local_time) / f64::from(expected);
        let local_error = x
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - truth(owned.owned.start + i as u64)).abs())
            .fold(0.0_f64, f64::max);
        let global_error = max_f64(local_error);
        let spmv_ms = max_f64(report.spmv_elapsed_ns as f64 * 1e-6);
        let reduce_ms = max_f64(report.allreduce_elapsed_ns as f64 * 1e-6);
        let wait_ms = max_f64(report.halo_wait_ns as f64 * 1e-6);
        let interior_ms = max_f64(report.interior_compute_ns as f64 * 1e-6);
        let boundary_ms = max_f64(report.boundary_compute_ns as f64 * 1e-6);
        let unattributed = report
            .elapsed_ns
            .saturating_sub(report.spmv_elapsed_ns)
            .saturating_sub(report.allreduce_elapsed_ns);
        let unattributed_ms = max_f64(unattributed as f64 * 1e-6);
        if run >= warmups {
            solve.push(elapsed);
            spmv.push(spmv_ms);
            reduce.push(reduce_ms);
            wait.push(wait_ms);
            interior.push(interior_ms);
            boundary.push(boundary_ms);
            other.push(unattributed_ms);
            imbalance.push(if mean > 0.0 { elapsed / mean } else { 1.0 });
            iter_count = report.iterations;
            status = report.status;
            residual = report.true_relative_residual;
            error = global_error;
            spmv_calls = report.spmv_calls;
            allreduce_calls = report.allreduce_calls;
        }
    }
    let median_solve = median(&mut solve);
    let median_spmv = median(&mut spmv);
    let median_reduce = median(&mut reduce);
    let median_wait = median(&mut wait);
    let median_interior = median(&mut interior);
    let median_boundary = median(&mut boundary);
    let median_other = median(&mut other);
    let median_imbalance = median(&mut imbalance);
    let success = status == DistributedPcgStatus::Converged
        && residual.is_finite()
        && residual <= 1.1e-9
        && error.is_finite()
        && error <= 2e-7;
    mpi.barrier();
    if mpi.rank() == 0 {
        let mut file = OpenOptions::new().append(true).open(csv)?;
        writeln!(file, "{label},{policy_name},{expected},{},{},{dd_rows},{min_margin:.12e},{iter_count},{status:?},{residual:.12e},{error:.12e},{ghosts},{read_ms:.6},{distribute_ms:.6},{halo_setup_ms:.6},{pcg_setup_ms:.6},{median_solve:.6},{median_spmv:.6},{median_reduce:.6},{median_wait:.6},{median_interior:.6},{median_boundary:.6},{median_other:.6},{median_imbalance:.6},{spmv_calls},{allreduce_calls}", partition.global_dofs(), ingest.global_nnz)?;
        let verdict = if success {
            "CONVERGED"
        } else {
            "NOT_CONVERGED"
        };
        println!("G8-C5 {verdict} {label}: ranks={expected} policy={policy_name} n={} strict_dd={dd_rows} iters={iter_count} rel={residual:.3e} maxerr={error:.3e} solve={median_solve:.4}ms spmv={median_spmv:.4}ms allreduce={median_reduce:.4}ms halo_wait={median_wait:.4}ms", partition.global_dofs());
        if policy == MtxPcgPolicy::ExperimentalUnverified {
            println!("G8-C5 WARNING: experimental input does not certify SPD; convergence is not a proof");
        }
    }
    mpi.barrier();
    if policy == MtxPcgPolicy::StrictCertified && !success {
        return Err("G8-C5 certified SPD test failed correctness gate".into());
    }
    Ok(())
}
