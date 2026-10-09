//! G8-C1 native distributed Jacobi PCG: 1/2/4 ranks, SPD matrix families.
//!
//! All MPI ranks construct the full CSR ONLY for deterministic validation.
//! The production DistributedPcg::solve API receives a rank-local operator,
//! local RHS and locally-owned solution vector; no replicated CSR is needed.

use hybit::distributed::mpi_backend::MpiRuntime;
use hybit::distributed::mpi_pcg::{DistributedPcg, DistributedPcgOptions, DistributedPcgStatus};
use hybit::distributed::{
    build_contiguous_halo_plans, prepare_rank_local_csr, ContiguousPartition,
};
use hybit::Csr32Matrix;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;

fn matrix(n: usize, kind: &str) -> Csr32Matrix {
    let mut ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        let mut entries = Vec::<(usize, f64)>::new();
        match kind {
            "chain" => {
                if i > 0 {
                    entries.push((i - 1, -1.0));
                }
                entries.push((i, 2.01));
                if i + 1 < n {
                    entries.push((i + 1, -1.0));
                }
            }
            "banded" => {
                for d in (1..=3).rev() {
                    if i >= d {
                        entries.push((i - d, -0.4));
                    }
                }
                entries.push((i, 6.0));
                for d in 1..=3 {
                    if i + d < n {
                        entries.push((i + d, -0.4));
                    }
                }
            }
            "heterogeneous" => {
                if i > 0 {
                    entries.push((i - 1, -0.7));
                }
                entries.push((i, 3.0 + ((i % 23) as f64) * 0.25));
                if i + 1 < n {
                    entries.push((i + 1, -0.7));
                }
            }
            "disconnected" => {
                if i % 32 != 0 {
                    entries.push((i - 1, -0.6));
                }
                entries.push((i, 3.0));
                if i % 32 != 31 && i + 1 < n {
                    entries.push((i + 1, -0.6));
                }
            }
            _ => unreachable!(),
        }
        for (col, val) in entries {
            cols.push(col as u32);
            vals.push(val);
        }
        ptr.push(cols.len() as u32);
    }
    Csr32Matrix::new(n, n, ptr, cols, vals).expect("valid SPD CSR")
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Independent serial Jacobi-PCG, for verification only.
fn serial_pcg(a: &Csr32Matrix, b: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
    let n = b.len();
    let mut diag_inverse = vec![0.0; n];
    for (i, d_inv) in diag_inverse.iter_mut().enumerate() {
        let start = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        let mut diagonal = 0.0;
        for pos in start..end {
            if a.col_idx()[pos] as usize == i {
                diagonal += a.values()[pos];
            }
        }
        if diagonal <= 0.0 {
            return Err("non-positive reference diagonal".into());
        }
        *d_inv = 1.0 / diagonal;
    }
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut z: Vec<f64> = r.iter().zip(&diag_inverse).map(|(&v, &d)| v * d).collect();
    let mut p = z.clone();
    let b_norm = dot(b, b).sqrt();
    let target = 1.0e-13_f64.max(1.0e-11 * b_norm);
    if b_norm <= target {
        return Ok(x);
    }
    let mut rz = dot(&r, &z);
    for _ in 0..4000 {
        let ap = a.spmv(&p)?;
        let denom = dot(&p, &ap);
        if denom <= 0.0 || !denom.is_finite() {
            return Err("serial PCG curvature breakdown".into());
        }
        let alpha = rz / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= target {
            return Ok(x);
        }
        for i in 0..n {
            z[i] = diag_inverse[i] * r[i];
        }
        let next_rz = dot(&r, &z);
        let beta = next_rz / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = next_rz;
    }
    Err("serial PCG failed to converge".into())
}

fn run_case(
    mpi: &MpiRuntime,
    kind: &'static str,
    n: usize,
    rhs_kind: &'static str,
    trial: usize,
    csv: &str,
) -> Result<(), Box<dyn Error>> {
    let ranks = mpi.size() as usize;
    let rank = mpi.rank() as usize;
    let a = matrix(n, kind);
    let partition = ContiguousPartition::balanced(n as u64, ranks as u32)?;
    let plans = build_contiguous_halo_plans(&a, &partition)?;
    let plan = &plans[rank];
    let local = prepare_rank_local_csr(&a, plan)?;
    let begin = plan.owned_range().start as usize;
    let end = plan.owned_range().end as usize;
    let mut pcg = DistributedPcg::prepare(mpi, plan, &local)?;
    let true_x: Vec<f64> = match rhs_kind {
        "zero" => vec![0.0; n],
        _ => (0..n)
            .map(|i| {
                let seed = ((i + 1) * (trial + 2)) as f64;
                (seed * 0.19).sin() + 0.3 * (seed * 0.037).cos()
            })
            .collect(),
    };
    let b = a.spmv(&true_x)?;
    let reference_x = serial_pcg(&a, &b)?;
    let mut x = vec![0.0; plan.owned_len()];
    let options = DistributedPcgOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 1.0e-13,
        max_iterations: 2000,
    };
    let report = pcg.solve(mpi, &local, &b[begin..end], &mut x, options)?;
    let mut local_reference_error = 0.0_f64;
    let mut local_truth_error = 0.0_f64;
    for ((&got, &expected), &truth) in x
        .iter()
        .zip(&reference_x[begin..end])
        .zip(&true_x[begin..end])
    {
        local_reference_error = local_reference_error.max((got - expected).abs());
        local_truth_error = local_truth_error.max((got - truth).abs());
    }
    let global_reference_error = mpi.all_reduce_sum_f64(local_reference_error);
    let global_truth_error = mpi.all_reduce_sum_f64(local_truth_error);
    let serial_ax = a.spmv(&reference_x)?;
    let serial_residual_norm: f64 = b
        .iter()
        .zip(&serial_ax)
        .map(|(&bi, &axi)| (bi - axi) * (bi - axi))
        .sum::<f64>()
        .sqrt();
    let b_norm = dot(&b, &b).sqrt();
    let serial_relative = if b_norm > 0.0 {
        serial_residual_norm / b_norm
    } else {
        serial_residual_norm
    };
    let elapsed_mean_ms = mpi.all_reduce_sum_f64(report.elapsed_ns as f64 * 1.0e-6) / ranks as f64;
    let status_ok = report.status == DistributedPcgStatus::Converged;
    let rel_ok =
        report.true_relative_residual.is_finite() && report.true_relative_residual <= 1.0e-9;
    let error_ok = global_reference_error <= 2.0e-7 && global_truth_error <= 2.0e-7;
    let expected_zero = rhs_kind != "zero" || report.iterations == 0;
    let local_fail = u64::from(!(status_ok && rel_ok && error_ok && expected_zero));
    if mpi.all_reduce_sum_u64(local_fail) != 0 {
        return Err(format!(
            "C1 failed {kind}/{rhs_kind}: status={:?} iterations={} rel={:.3e} serial_error={:.3e} truth_error={:.3e}",
            report.status, report.iterations, report.true_relative_residual,
            global_reference_error, global_truth_error
        ).into());
    }
    mpi.barrier();
    if rank == 0 {
        let mut f = OpenOptions::new().append(true).open(csv)?;
        writeln!(
            f,
            "{kind},{ranks},{n},{rhs_kind},{},{:?},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{},{:.6}",
            report.iterations,
            report.status,
            report.true_relative_residual,
            serial_relative,
            global_reference_error,
            global_truth_error,
            report.spmv_calls,
            report.allreduce_calls,
            pcg.interior_rows(),
            pcg.boundary_rows(),
            elapsed_mean_ms,
        )?;
        println!(
            "PASS C1 {kind}/{rhs_kind}: ranks={ranks} n={n} iters={} true_rel={:.3e} serial_err={:.3e} truth_err={:.3e} interior={} boundary={}",
            report.iterations, report.true_relative_residual,
            global_reference_error, global_truth_error,
            pcg.interior_rows(), pcg.boundary_rows(),
        );
    }
    mpi.barrier();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let expected: i32 = args
        .next()
        .ok_or("usage: mpi_distributed_pcg <ranks> <csv>")?
        .parse()?;
    let csv = args
        .next()
        .ok_or("usage: mpi_distributed_pcg <ranks> <csv>")?;
    let mpi = MpiRuntime::initialize()?;
    if !matches!(mpi.size(), 1 | 2 | 4) || mpi.size() != expected {
        return Err("MPI rank count must match 1/2/4".into());
    }
    if mpi.rank() == 0 {
        let mut f = File::create(&csv)?;
        writeln!(f, "case,ranks,n,rhs,iterations,status,true_relative_residual,serial_relative_residual,serial_solution_error,manufactured_solution_error,spmv_calls,allreduce_calls,interior_rows,boundary_rows,avg_elapsed_ms")?;
    }
    mpi.barrier();
    for (kind, n) in [
        ("chain", 257),
        ("banded", 260),
        ("heterogeneous", 257),
        ("disconnected", 256),
    ] {
        run_case(&mpi, kind, n, "manufactured_a", 0, &csv)?;
        run_case(&mpi, kind, n, "manufactured_b", 1, &csv)?;
    }
    run_case(&mpi, "chain", 257, "zero", 0, &csv)?;
    mpi.barrier();
    if mpi.rank() == 0 {
        println!(
            "=== HYBIT 0.9 G8-C1 MPI {}-RANK DISTRIBUTED PCG PASS ===",
            mpi.size()
        );
    }
    Ok(())
}
