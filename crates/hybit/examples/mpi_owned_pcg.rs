//! G8-C2: rank-owned matrix assembly + real MPI halo + Jacobi-PCG.
//!
//! The synthetic SPD rows are generated ONLY for the range owned by each
//! rank. No rank creates a global CSR matrix, global solution or global RHS.
//! Reference x is an analytic function evaluated at requested global DOFs.
//! Tests 1/2/4/8 ranks; output is correctness-oriented, not a speedup claim.

use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_input::{prepare_owned_rows, OwnedCsrRows};
use hybit::distributed::mpi_overlap::OverlapSpmv;
use hybit::distributed::mpi_pcg::{DistributedPcg, DistributedPcgOptions, DistributedPcgStatus};
use hybit::distributed::ContiguousPartition;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::time::Instant;

fn manufactured(global: u64, rhs: usize) -> f64 {
    let g = global as f64 + 1.0;
    let phase = (rhs + 1) as f64;
    (g * phase * 0.017).sin() + 0.25 * (g * 0.0037).cos()
}

fn row(global: u64, n: u64, kind: &str, side: u64) -> Vec<(u64, f64)> {
    let mut out = Vec::<(u64, f64)>::new();
    match kind {
        "chain" => {
            if global > 0 {
                out.push((global - 1, -1.0));
            }
            out.push((global, 3.2));
            if global + 1 < n {
                out.push((global + 1, -1.0));
            }
        }
        "banded" => {
            for d in (1..=3u64).rev() {
                if global >= d {
                    out.push((global - d, -0.4));
                }
            }
            out.push((global, 7.0));
            for d in 1..=3u64 {
                if global + d < n {
                    out.push((global + d, -0.4));
                }
            }
        }
        "heterogeneous" => {
            if global > 0 {
                out.push((global - 1, -0.7));
            }
            out.push((global, 3.0 + (global % 23) as f64 * 0.25));
            if global + 1 < n {
                out.push((global + 1, -0.7));
            }
        }
        "poisson2d" => {
            if global >= side {
                out.push((global - side, -1.0));
            }
            if global % side != 0 {
                out.push((global - 1, -1.0));
            }
            out.push((global, 4.5));
            if global % side + 1 < side {
                out.push((global + 1, -1.0));
            }
            if global + side < n {
                out.push((global + side, -1.0));
            }
        }
        "disconnected" => {
            if global % 32 != 0 {
                out.push((global - 1, -0.6));
            }
            out.push((global, 3.0));
            if global + 1 < n && global % 32 != 31 {
                out.push((global + 1, -0.6));
            }
        }
        _ => unreachable!(),
    }
    out
}

fn owned_rows(
    part: &ContiguousPartition,
    rank: u32,
    kind: &str,
    side: u64,
) -> Result<OwnedCsrRows, Box<dyn Error>> {
    let owned = part.owned_range(rank)?;
    let n = part.global_dofs();
    let mut row_ptr = vec![0u32];
    let mut global_col_idx = Vec::new();
    let mut values = Vec::new();
    for g in owned.clone() {
        for (col, value) in row(g, n, kind, side) {
            global_col_idx.push(col);
            values.push(value);
        }
        row_ptr.push(u32::try_from(values.len())?);
    }
    Ok(OwnedCsrRows {
        global_dofs: n,
        owned,
        row_ptr,
        global_col_idx,
        values,
    })
}

fn run(
    mpi: &MpiRuntime,
    kind: &str,
    side: u64,
    rhs: usize,
    csv_path: &str,
) -> Result<(), Box<dyn Error>> {
    let n = side * side;
    let ranks = mpi.size() as usize;
    let rank = mpi.rank() as u32;
    let partition = ContiguousPartition::balanced(n, ranks as u32)?;
    let rows = owned_rows(&partition, rank, kind, side)?;
    let t_prepare = Instant::now();
    let (plan, local, stats) = prepare_owned_rows(mpi, &partition, &rows)?;
    let input_ms = t_prepare.elapsed().as_secs_f64() * 1.0e3;
    let t_solver = Instant::now();
    let mut solver = DistributedPcg::prepare(mpi, &plan, &local)?;
    let solver_setup_ms = t_solver.elapsed().as_secs_f64() * 1.0e3;
    let mut halo_probe = OverlapSpmv::prepare(mpi, &plan, &local)?;

    let mut x_truth_local = Vec::with_capacity(stats.owned_rows);
    let mut rhs_local = Vec::with_capacity(stats.owned_rows);
    for g in rows.owned.clone() {
        x_truth_local.push(manufactured(g, rhs));
        let sum = row(g, n, kind, side)
            .iter()
            .map(|(col, coefficient)| coefficient * manufactured(*col, rhs))
            .sum::<f64>();
        rhs_local.push(sum);
    }
    // Independently exercise communication-plan construction and local SpMV.
    let mut distributed_b = vec![0.0; stats.owned_rows];
    halo_probe.spmv_overlap(mpi, &local, &x_truth_local, &mut distributed_b)?;
    let halo_err_local = rhs_local
        .iter()
        .zip(&distributed_b)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let halo_error_sum = mpi.all_reduce_sum_f64(halo_err_local);
    if halo_error_sum > 1.0e-12 {
        return Err(format!("halo mismatch {kind} N={n}: {halo_error_sum:e}").into());
    }

    let mut x = vec![0.0; stats.owned_rows];
    let opts = DistributedPcgOptions {
        relative_tolerance: 1.0e-9,
        absolute_tolerance: 1.0e-13,
        max_iterations: 1800,
    };
    let report = solver.solve(mpi, &local, &rhs_local, &mut x, opts)?;
    let local_error = x
        .iter()
        .zip(&x_truth_local)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let truth_error_sum = mpi.all_reduce_sum_f64(local_error);
    let count = mpi.all_reduce_sum_u64(stats.owned_rows as u64);
    let nnz = mpi.all_reduce_sum_u64(stats.local_nnz as u64);
    let ghosts = mpi.all_reduce_sum_u64(stats.ghosts as u64);
    let sent = mpi.all_reduce_sum_u64(stats.sent_halo_values_per_spmv as u64);
    let received = mpi.all_reduce_sum_u64(stats.received_halo_values_per_spmv as u64);
    let resident = mpi.all_reduce_sum_u64(stats.operator_and_halo_bytes_estimate as u64);
    let input_avg = mpi.all_reduce_sum_f64(input_ms) / ranks as f64;
    let setup_avg = mpi.all_reduce_sum_f64(solver_setup_ms) / ranks as f64;
    let solve_avg = mpi.all_reduce_sum_f64(report.elapsed_ns as f64 * 1.0e-6) / ranks as f64;
    let pass = count == n
        && sent == received
        && report.status == DistributedPcgStatus::Converged
        && report.true_relative_residual <= opts.relative_tolerance * 1.05
        && report.true_relative_residual.is_finite()
        && truth_error_sum <= 2.0e-7
        && halo_error_sum <= 1.0e-12;
    if mpi.all_reduce_sum_u64(u64::from(!pass)) != 0 {
        return Err(format!("G8-C2 failed {kind} N={n} ranks={ranks}: status={:?}, true_rel={:.3e}, truth_err={truth_error_sum:.3e}",report.status,report.true_relative_residual).into());
    }
    mpi.barrier();
    if rank == 0 {
        let mut csv = OpenOptions::new().append(true).open(csv_path)?;
        writeln!(csv, "{kind},{ranks},{n},{rhs},{nnz},{ghosts},{sent},{received},{resident},{},{},{:.12e},{:.12e},{:.12e},{:.6},{:.6},{:.6}",
            report.iterations, report.allreduce_calls,report.true_relative_residual,truth_error_sum,halo_error_sum,input_avg,setup_avg,solve_avg)?;
        println!("PASS C2 {kind} N={n} rhs={rhs} ranks={ranks} iter={} rel={:.3e} truth_err={truth_error_sum:.3e} ghosts={ghosts} nnz={nnz} avg_solve_ms={solve_avg:.3}",report.iterations,report.true_relative_residual);
    }
    mpi.barrier();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let expected: i32 = args
        .next()
        .ok_or("usage: mpi_owned_pcg <ranks> <csv>")?
        .parse()?;
    let csv_path = args.next().ok_or("usage: mpi_owned_pcg <ranks> <csv>")?;
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != expected || !matches!(mpi.size(), 1 | 2 | 4 | 8) {
        return Err("expected rank count 1/2/4/8".into());
    }
    if mpi.rank() == 0 {
        let mut csv = File::create(&csv_path)?;
        writeln!(csv, "case,ranks,n,rhs,global_nnz,total_ghosts,total_halo_sent,total_halo_recv,sum_operator_bytes,iterations,allreduces,true_rel,truth_error_sum,halo_error_sum,mean_rank_input_ms,mean_rank_setup_ms,mean_rank_solve_ms")?;
    }
    mpi.barrier();
    for side in [64u64, 128u64] {
        for kind in [
            "chain",
            "banded",
            "heterogeneous",
            "poisson2d",
            "disconnected",
        ] {
            for rhs in 0..2usize {
                run(&mpi, kind, side, rhs, &csv_path)?;
            }
        }
    }
    mpi.barrier();
    if mpi.rank() == 0 {
        println!(
            "=== HYBIT 0.9 G8-C2 MPI {}-RANK OWNED-CSR/PCG PASS ===",
            mpi.size()
        );
    }
    Ok(())
}
