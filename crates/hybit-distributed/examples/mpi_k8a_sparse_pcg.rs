//! H1-C3K8-A experimental original-coefficient Root-only MS-MPI Jacobi-PCG.
//! Certified SPD subset: exactly symmetric, real, strictly row-diagonally dominant,
//! strictly positive diagonal. Does not claim all SPD or general FEM support.
//! Collective MPI errors are synchronized where possible. Experimental only.

use hybit_distributed::mpi_assignment::{
    AssignmentHaloPlan, AssignmentLocalCsr, AssignmentOwnedCsrRows,
    prepare_assignment_owned_rows_sparse, spmv_assignment,
};
use hybit_distributed::mpi_assignment_sparse::scatter_sparse_owners_from_root;
use hybit_distributed::mpi_assignment_sparse_verify::verify_sparse_assignment_scatter_exact;
use hybit_distributed::mpi_backend::MpiRuntime;
use hybit_distributed::{
    ContiguousPartition, PartitionAssignment, abtm_balanced_multisource_partition,
    abtm_region_grow_partition,
};
use hybit_matrix::{Csr32Matrix, read_matrix_market};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;
const MAX_INPUT_BYTES: u64 = 384 * 1024 * 1024;
const MAX_DOFS: usize = 1_200_000;
const MAX_NNZ: usize = 16_000_000;

#[derive(Clone, Copy, Debug)]
enum Policy {
    Equal,
    Region,
    Multisource,
}
impl Policy {
    fn parse(s: &str) -> BenchResult<Self> {
        match s {
            "equal" => Ok(Self::Equal),
            "region" => Ok(Self::Region),
            "multisource" => Ok(Self::Multisource),
            _ => Err("policy must be equal, region, or multisource".into()),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Region => "abtm_region_growth",
            Self::Multisource => "abtm_multisource",
        }
    }
}

fn partition(matrix: &Csr32Matrix, ranks: u32, policy: Policy) -> BenchResult<PartitionAssignment> {
    Ok(match policy {
        Policy::Equal => PartitionAssignment::from_contiguous(&ContiguousPartition::balanced(
            matrix.nrows() as u64,
            ranks,
        )?)?,
        Policy::Region => abtm_region_grow_partition(matrix, ranks)?.0,
        Policy::Multisource => abtm_balanced_multisource_partition(matrix, ranks)?.0,
    })
}

fn world_max(value: f64) -> f64 {
    let world = mpi::topology::SimpleCommunicator::world();
    let mut out = 0.0f64;
    world.all_reduce_into(&value, &mut out, SystemOperation::max());
    out
}

fn x_exact(global: u64) -> f64 {
    let t = global as f64 + 1.0;
    (t * 0.017).sin() + 0.125 * (t * 0.0031).cos()
}

/// A sufficient (not necessary) SPD certificate: A=A^T exactly, finite,
/// positive diagonal and strict row diagonal dominance. It intentionally
/// rejects many valid SPD FEM matrices; never confuses PCG convergence with
/// certification of arbitrary input.
fn certify_strict_spd(a: &Csr32Matrix) -> BenchResult<()> {
    if a.nrows() == 0 || a.nrows() != a.ncols() {
        return Err("SPD certificate requires nonempty square matrix".into());
    }
    let n = a.nrows();
    for i in 0..n {
        let start = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        let cols = &a.col_idx()[start..end];
        let vals = &a.values()[start..end];
        if cols.is_empty() || cols.windows(2).any(|w| w[0] >= w[1]) {
            return Err("SPD certificate requires sorted unique CSR columns".into());
        }
        let mut diag = None;
        let mut offsum = 0.0f64;
        for (&j, &v) in cols.iter().zip(vals) {
            if !v.is_finite() || j as usize >= n {
                return Err("SPD certificate: invalid CSR coefficient".into());
            }
            if j as usize == i {
                diag = Some(v);
            } else {
                offsum += v.abs();
                let j0 = a.row_ptr()[j as usize] as usize;
                let j1 = a.row_ptr()[j as usize + 1] as usize;
                let reverse = a.col_idx()[j0..j1]
                    .binary_search(&(i as u32))
                    .map_err(|_| "SPD certificate: missing transpose coefficient")?;
                if a.values()[j0 + reverse].to_bits() != v.to_bits() {
                    return Err("SPD certificate: asymmetric coefficients".into());
                }
            }
        }
        let d = diag.ok_or("SPD certificate: diagonal missing")?;
        if !offsum.is_finite() || d <= 0.0 || d <= offsum + 1.0e-12 * (d + offsum) {
            return Err(
                "SPD certificate: not strictly diagonally dominant with positive diagonal".into(),
            );
        }
    }
    Ok(())
}

fn inverse_diagonal(rows: &AssignmentOwnedCsrRows) -> BenchResult<Vec<f64>> {
    if rows.row_ptr.len() != rows.owned_globals.len() + 1 {
        return Err("bad local row_ptr".into());
    }
    let mut inv = Vec::with_capacity(rows.owned_globals.len());
    for (i, &global) in rows.owned_globals.iter().enumerate() {
        let mut diagonal = None;
        for k in rows.row_ptr[i] as usize..rows.row_ptr[i + 1] as usize {
            if rows.global_col_idx[k] == global && diagonal.replace(rows.values[k]).is_some() {
                return Err("duplicate diagonal".into());
            }
        }
        let value = diagonal.ok_or("Jacobi diagonal missing")?;
        if !value.is_finite() || value <= 0.0 {
            return Err("Jacobi diagonal nonpositive".into());
        }
        inv.push(1.0 / value);
    }
    Ok(inv)
}

fn world_dot(a: &[f64], b: &[f64]) -> f64 {
    let local = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>();
    let mut result = 0.0f64;
    mpi::topology::SimpleCommunicator::world().all_reduce_into(
        &local,
        &mut result,
        SystemOperation::sum(),
    );
    result
}

struct Solution {
    x: Vec<f64>,
    iterations: usize,
    true_rel: f64,
    max_error: f64,
    solve_ms: f64,
    spmv_calls: usize,
    dot_reductions: usize,
}

fn solve_pcg(
    mpi: &MpiRuntime,
    halo: &AssignmentHaloPlan,
    op: &AssignmentLocalCsr,
    invdiag: &[f64],
    rhs: &[f64],
    max_iters: usize,
    tol: f64,
) -> BenchResult<Solution> {
    let n = halo.owned_len();
    if mpi.all_reduce_sum_u64(u64::from(n == 0 || rhs.len() != n || invdiag.len() != n)) != 0 {
        return Err("incompatible PCG local vectors".into());
    }
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    let mut scratch = halo.workspace();
    let mut dots = 0usize;
    let mut spmvs = 0usize;
    mpi.barrier();
    let started = Instant::now();
    let norm_b2 = world_dot(rhs, rhs);
    dots += 1;
    if !norm_b2.is_finite() || norm_b2 <= 0.0 {
        return Err("invalid manufactured RHS".into());
    }
    let norm_b = norm_b2.sqrt();
    for (((zi, pi), &d), &ri) in z.iter_mut().zip(&mut p).zip(invdiag).zip(&r) {
        let value = d * ri;
        *zi = value;
        *pi = value;
    }
    let mut rz = world_dot(&r, &z);
    dots += 1;
    if !rz.is_finite() || rz <= 0.0 {
        return Err("invalid initial preconditioned norm".into());
    }
    let mut iters = 0usize;
    let mut converged = false;
    for it in 0..max_iters {
        spmv_assignment(mpi, halo, op, &p, &mut scratch, &mut ap)?;
        spmvs += 1;
        let pap = world_dot(&p, &ap);
        dots += 1;
        if !pap.is_finite() || pap <= 0.0 {
            return Err("PCG nonpositive curvature".into());
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            return Err("PCG nonfinite alpha".into());
        }
        for (((xi, ri), &pi), &api) in x.iter_mut().zip(&mut r).zip(&p).zip(&ap) {
            *xi += alpha * pi;
            *ri -= alpha * api;
        }
        iters = it + 1;
        let rr = world_dot(&r, &r);
        dots += 1;
        if !rr.is_finite() || rr < 0.0 {
            return Err("invalid PCG residual".into());
        }
        if rr.sqrt() <= tol * norm_b * 0.5 {
            converged = true;
            break;
        }
        for (zi, (&d, &ri)) in z.iter_mut().zip(invdiag.iter().zip(&r)) {
            *zi = d * ri;
        }
        let next = world_dot(&r, &z);
        dots += 1;
        if !next.is_finite() || next <= 0.0 {
            return Err("invalid PCG recurrence".into());
        }
        let beta = next / rz;
        if !beta.is_finite() {
            return Err("invalid PCG beta".into());
        }
        for (pi, &zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
        rz = next;
    }
    if !converged {
        return Err("Jacobi-PCG failed to converge within max_iters".into());
    }
    spmv_assignment(mpi, halo, op, &x, &mut scratch, &mut ap)?;
    spmvs += 1;
    let residual = rhs
        .iter()
        .zip(&ap)
        .map(|(&b, &ax)| b - ax)
        .collect::<Vec<_>>();
    let rel = (world_dot(&residual, &residual) / norm_b2).sqrt();
    dots += 1;
    let local_err = x
        .iter()
        .zip(&halo.owned_globals)
        .map(|(&v, &g)| (v - x_exact(g)).abs())
        .fold(0.0f64, f64::max);
    let max_err = world_max(local_err);
    if !rel.is_finite() || rel > tol || !max_err.is_finite() || max_err > 1.0e-5 {
        return Err("PCG true residual or manufactured solution certification failed".into());
    }
    Ok(Solution {
        x,
        iterations: iters,
        true_rel: rel,
        max_error: max_err,
        solve_ms: world_max(started.elapsed().as_secs_f64() * 1000.0),
        spmv_calls: spmvs,
        dot_reductions: dots,
    })
}

/// Collect each noncontiguous owned solution to root and evaluate the
/// ORIGINAL (pre-scatter) matrix. Unlike distributed residual checks, this
/// reference uses neither scattered values nor Halo numeric output.
fn verify_root_original(
    mpi: &MpiRuntime,
    root: Option<&Csr32Matrix>,
    owners: Option<&PartitionAssignment>,
    rows: &AssignmentOwnedCsrRows,
    x: &[f64],
    tol: f64,
) -> BenchResult<(f64, f64)> {
    let world = mpi::topology::SimpleCommunicator::world();
    let count = rows.owned_globals.len() as u64;
    let mut counts = vec![0u64; mpi.size() as usize];
    world.all_gather_into(&count, &mut counts[..]);
    if mpi.all_reduce_sum_u64(u64::from(
        x.len() != count as usize || x.iter().any(|v| !v.is_finite()),
    )) != 0
    {
        return Err("invalid local solution before root reference gather".into());
    }
    let total = counts.iter().sum::<u64>();
    let expected_n = root.map_or(0, |m| m.nrows() as u64);
    let n = mpi.all_reduce_sum_u64(expected_n);
    if total != n {
        return Err("gathered solution global size mismatch".into());
    }
    let root_result: BenchResult<(f64, f64)> = if mpi.rank() == 0 {
        // Do not abort root before finishing the expected receives.
        let mut packets = vec![(Vec::<u64>::new(), Vec::<f64>::new()); mpi.size() as usize];
        packets[0] = (rows.owned_globals.clone(), x.to_vec());
        for rank in 1..mpi.size() {
            let len = counts[rank as usize] as usize;
            let mut ids = vec![0u64; len];
            let mut vals = vec![0f64; len];
            world.process_at_rank(rank).receive_into(&mut ids[..]);
            world.process_at_rank(rank).receive_into(&mut vals[..]);
            packets[rank as usize] = (ids, vals);
        }
        (|| -> BenchResult<(f64, f64)> {
            let matrix = root.ok_or("root matrix missing")?;
            let selection = owners.ok_or("root assignment missing")?;
            let mut full = vec![0.0f64; matrix.nrows()];
            let mut seen = vec![false; matrix.nrows()];
            for (rank, (ids, values)) in packets.iter().enumerate() {
                if ids.len() != values.len() {
                    return Err("gather row/value counts disagree".into());
                }
                for (&id, &v) in ids.iter().zip(values) {
                    let idx = usize::try_from(id)?;
                    if idx >= seen.len()
                        || seen[idx]
                        || selection.owner_of(id)? != rank as u32
                        || !v.is_finite()
                    {
                        return Err("gather invalid global row ownership".into());
                    }
                    seen[idx] = true;
                    full[idx] = v;
                }
            }
            if seen.iter().any(|&flag| !flag) {
                return Err("root missed a global solution row".into());
            }
            let mut numer = 0.0f64;
            let mut denom = 0.0f64;
            let mut reference_error = 0.0f64;
            for i in 0..matrix.nrows() {
                let mut ax = 0.0;
                let mut b = 0.0;
                for k in matrix.row_ptr()[i] as usize..matrix.row_ptr()[i + 1] as usize {
                    let j = matrix.col_idx()[k] as usize;
                    let aij = matrix.values()[k];
                    ax += aij * full[j];
                    b += aij * x_exact(j as u64);
                }
                numer += (b - ax) * (b - ax);
                denom += b * b;
                reference_error = reference_error.max((full[i] - x_exact(i as u64)).abs());
            }
            if !denom.is_finite() || denom <= 0.0 {
                return Err("root manufactured RHS invalid".into());
            }
            let reference_rel = (numer / denom).sqrt();
            Ok((reference_rel, reference_error))
        })()
    } else {
        world.process_at_rank(0).send(&rows.owned_globals[..]);
        world.process_at_rank(0).send(x);
        Ok((0.0, 0.0))
    };
    if mpi.all_reduce_sum_u64(u64::from(root_result.is_err())) != 0 {
        if let Err(ref err) = root_result {
            eprintln!("K8-A root original residual verification error: {err}");
        }
        return Err("K8-A root original reference check failed on a rank".into());
    }
    let (mut reference_rel, mut reference_error) = root_result?;
    reference_rel = world_max(reference_rel);
    reference_error = world_max(reference_error);
    if reference_rel > tol
        || !reference_rel.is_finite()
        || !reference_error.is_finite()
        || reference_error > 1.0e-5
    {
        return Err("root-original CSR final residual/solution error failed".into());
    }
    Ok((reference_rel, reference_error))
}

fn main() -> BenchResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err("usage: mpi_k8a_sparse_pcg <ranks> <input.mtx> <equal|region|multisource> <warmups> <repeats> <max_iters> <tol> <output.csv> (8 arguments)".into());
    }
    let ranks: i32 = args[0].parse()?;
    let path = Path::new(&args[1]);
    let policy = Policy::parse(&args[2])?;
    let warmups: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    let max_iters: usize = args[5].parse()?;
    let tol: f64 = args[6].parse()?;
    let output = Path::new(&args[7]);
    if !matches!(ranks, 1 | 2 | 4 | 8)
        || warmups > 3
        || !(1..=12).contains(&repeats)
        || !(1..=3000).contains(&max_iters)
        || !(1.0e-12..=1.0e-3).contains(&tol)
    {
        return Err("K8-A guarded parameter bounds violated".into());
    }
    let mpi = MpiRuntime::initialize()?;
    if mpi.size() != ranks {
        return Err("unexpected MPI ranks".into());
    }
    if mpi.all_reduce_sum_u64(u64::from(mpi.rank() == 0 && output.exists())) != 0 {
        return Err("refuse to overwrite output CSV".into());
    }
    mpi.barrier();
    let stamp = Instant::now();
    let loaded = if mpi.rank() == 0 {
        Some((|| -> BenchResult<Csr32Matrix> {
            if std::fs::metadata(path)?.len() > MAX_INPUT_BYTES {
                return Err("input too large".into());
            }
            let (m, _) = read_matrix_market(path)?;
            if m.nrows() == 0
                || m.nrows() != m.ncols()
                || m.nrows() > MAX_DOFS
                || m.nrows() < ranks as usize
                || m.nnz() > MAX_NNZ
            {
                return Err("invalid root-only input matrix dimensions".into());
            }
            certify_strict_spd(&m)?;
            Ok(m)
        })())
    } else {
        None
    };
    if mpi.all_reduce_sum_u64(u64::from(loaded.as_ref().is_some_and(Result::is_err))) != 0 {
        if let Some(Err(err)) = &loaded {
            eprintln!("K8-A SPD input rejected: {err}");
        }
        return Err("root SPD input validation failed".into());
    }
    let matrix = loaded.transpose()?;
    let read_ms = world_max(stamp.elapsed().as_secs_f64() * 1000.0);
    let n = mpi.all_reduce_sum_u64(matrix.as_ref().map_or(0, |m| m.nrows() as u64));
    let nnz = mpi.all_reduce_sum_u64(matrix.as_ref().map_or(0, |m| m.nnz() as u64));
    mpi.barrier();
    let started = Instant::now();
    let proposed = matrix.as_ref().map(|m| partition(m, ranks as u32, policy));
    if mpi.all_reduce_sum_u64(u64::from(proposed.as_ref().is_some_and(Result::is_err))) != 0 {
        return Err("root partition proposal failed".into());
    }
    let selection = proposed.transpose()?;
    let partition_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    mpi.barrier();
    let (directory, rows, stats) =
        scatter_sparse_owners_from_root(&mpi, matrix.as_ref(), selection.as_ref())?;
    let scatter_ms = world_max(stats.distribution.scatter_ns as f64 * 1e-6);
    let verification = verify_sparse_assignment_scatter_exact(
        &mpi,
        matrix.as_ref(),
        selection.as_ref(),
        &directory,
        &rows,
    )?;
    if verification.verified_rows != n
        || verification.verified_nnz != nnz
        || !verification.bitwise_coefficients
    {
        return Err("root original distributed CSR exact check failed".into());
    }
    let verify_ms = world_max(verification.verify_ns as f64 * 1e-6);
    mpi.barrier();
    let started = Instant::now();
    let inverse = inverse_diagonal(&rows);
    if mpi.all_reduce_sum_u64(u64::from(inverse.is_err())) != 0 {
        return Err("Jacobi local diagonal preparation failed".into());
    }
    let inverse = inverse?;
    let (halo, local, hstats) = prepare_assignment_owned_rows_sparse(&mpi, &directory, &rows)?;
    let halo_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    let ghosts = mpi.all_reduce_sum_u64(hstats.ghosts as u64);
    let sparse_owner_bytes = mpi.all_reduce_sum_u64(stats.owner_bytes as u64);
    mpi.barrier();
    let started = Instant::now();
    let known = halo
        .owned_globals
        .iter()
        .map(|&g| x_exact(g))
        .collect::<Vec<_>>();
    let mut rhs = vec![0.0; known.len()];
    spmv_assignment(&mpi, &halo, &local, &known, &mut halo.workspace(), &mut rhs)?;
    let rhs_ms = world_max(started.elapsed().as_secs_f64() * 1000.0);
    let mut solve_times = Vec::new();
    let mut observed_iters = None;
    let mut last_solution = None;
    let mut worst_true_rel = 0.0f64;
    let mut worst_err = 0.0f64;
    let mut calls = 0usize;
    let mut reductions = 0usize;
    for trial in 0..warmups + repeats {
        let result = solve_pcg(&mpi, &halo, &local, &inverse, &rhs, max_iters, tol)?;
        if observed_iters.is_some_and(|k| k != result.iterations) {
            return Err("inconsistent iteration count between repeats".into());
        }
        observed_iters = Some(result.iterations);
        if trial >= warmups {
            solve_times.push(result.solve_ms);
            worst_true_rel = worst_true_rel.max(result.true_rel);
            worst_err = worst_err.max(result.max_error);
            calls = result.spmv_calls;
            reductions = result.dot_reductions;
        }
        last_solution = Some(result.x);
    }
    let final_solution = last_solution.ok_or("missing PCG result")?;
    mpi.barrier();
    let root_started = Instant::now();
    let (root_rel, root_err) = verify_root_original(
        &mpi,
        matrix.as_ref(),
        selection.as_ref(),
        &rows,
        &final_solution,
        tol,
    )?;
    let root_verify_ms = world_max(root_started.elapsed().as_secs_f64() * 1000.0);
    let solve_median = median(&solve_times);
    if mpi.rank() == 0 {
        println!(
            "K8-A ORIGINAL-COEFFICIENT PCG PASS method={} ranks={} n={} nnz={} iterations={} median_solve_ms={:.6} true_rel={:.3e} root_original_rel={:.3e} root_solution_error={:.3e} scatter_ms={:.3} halo_setup_ms={:.3} sparse_owner_bytes_sum={} ghosts={} exact_original=true",
            policy.label(),
            ranks,
            n,
            nnz,
            observed_iters.unwrap_or(0),
            solve_median,
            worst_true_rel,
            root_rel,
            root_err,
            scatter_ms,
            halo_ms,
            sparse_owner_bytes,
            ghosts
        );
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output)?;
        writeln!(
            file,
            "matrix,ranks,method,n,nnz,spd_certificate,preconditioner,warmups,repeats,max_iters,tol,iterations,median_solve_ms,read_ms,partition_ms,scatter_ms,verify_csr_ms,halo_setup_ms,rhs_setup_ms,root_residual_verification_ms,distributed_true_rel,root_original_true_rel,max_solution_error,root_solution_error,sparse_owner_bytes_sum,ghosts,spmv_calls,dot_reductions,exact_original_verified"
        )?;
        writeln!(
            file,
            "{},{},{},{},{},symmetric_strict_diagonal_dominance,jacobi,{},{},{},{:.3e},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{},true",
            path.display(),
            ranks,
            policy.label(),
            n,
            nnz,
            warmups,
            repeats,
            max_iters,
            tol,
            observed_iters.unwrap_or(0),
            solve_median,
            read_ms,
            partition_ms,
            scatter_ms,
            verify_ms,
            halo_ms,
            rhs_ms,
            root_verify_ms,
            worst_true_rel,
            root_rel,
            worst_err,
            root_err,
            sparse_owner_bytes,
            ghosts,
            calls,
            reductions
        )?;
    }
    mpi.barrier();
    Ok(())
}

fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[m - 1] + sorted[m])
    } else {
        sorted[m]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_matrix(d0: f64, d1: f64, a01: f64, a10: f64) -> Csr32Matrix {
        Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![d0, a01, a10, d1],
        )
        .unwrap()
    }
    #[test]
    fn strict_spd_certificate_accepts_diagonally_dominant_matrix() {
        assert!(certify_strict_spd(&make_matrix(3.0, 4.0, -1.0, -1.0)).is_ok());
    }
    #[test]
    fn strict_spd_certificate_rejects_asymmetry() {
        assert!(certify_strict_spd(&make_matrix(3.0, 4.0, -1.0, -0.9)).is_err());
    }
    #[test]
    fn strict_spd_certificate_rejects_insufficient_dominance() {
        assert!(certify_strict_spd(&make_matrix(1.0, 1.0, -1.0, -1.0)).is_err());
    }
}
