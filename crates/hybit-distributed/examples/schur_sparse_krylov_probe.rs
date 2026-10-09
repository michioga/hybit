//! HyBIT 0.9 Schur S3a: sparse rank-local PCG and matrix-free Schur/FGMRES pilot.
//!
//! Experimental example. Synthetic SPD only; does not change production routing.
//! S1 two-sided separator => A_II is exactly block diagonal by owner.
//! Local PCG solves are *inexact*, so outer FGMRES is diagnostic rather than a
//! proof of exact fixed-operator Krylov convergence. Always gate true ||Ax-b||.

use hybit_core::{HybitError, LinearOperator, Preconditioner, SolveStatus, SolverOptions};
use hybit_distributed::{abtm_multilevel_partition, AbtmMultilevelOptions, PartitionAssignment};
use hybit_krylov::{fgmres, pcg, FgmresOptions, KrylovOutcome};
use hybit_matrix::Csr32Matrix;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

type R<T> = Result<T, Box<dyn Error>>;
const INNER_TOL: f64 = 2.0e-13;
const OUTER_TOL: f64 = 1.0e-10;
const TRUE_TOL: f64 = 2.0e-8;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn relative(x: &[f64], y: &[f64]) -> f64 {
    let diff: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
    diff.sqrt() / dot(y, y).sqrt().max(1e-300)
}

#[derive(Debug)]
struct Diag {
    inverse: Vec<f64>,
}

impl Diag {
    fn from_matrix(matrix: &Csr32Matrix) -> Result<Self, HybitError> {
        let diag = matrix.diagonal()?;
        let mut inverse = Vec::with_capacity(diag.len());
        for (row, value) in diag.into_iter().enumerate() {
            if value <= 0.0 || !value.is_finite() {
                return Err(HybitError::ZeroDiagonal { row });
            }
            inverse.push(1.0 / value);
        }
        Ok(Self { inverse })
    }

    fn from_values(values: &[f64]) -> Result<Self, HybitError> {
        let mut inverse = Vec::with_capacity(values.len());
        for (row, &v) in values.iter().enumerate() {
            if v <= 0.0 || !v.is_finite() {
                return Err(HybitError::ZeroDiagonal { row });
            }
            inverse.push(1.0 / v);
        }
        Ok(Self { inverse })
    }
}

impl Preconditioner for Diag {
    fn len(&self) -> usize {
        self.inverse.len()
    }
    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.inverse.len() || z.len() != self.inverse.len() {
            return Err(HybitError::InvalidArgument(
                "Jacobi vector dimension mismatch",
            ));
        }
        for ((zi, &ri), &d) in z.iter_mut().zip(r).zip(&self.inverse) {
            *zi = ri * d;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct SparseLocal {
    // Interior DOFs in ascending global order; independent local CSR.
    dofs: Vec<usize>,
    a: Csr32Matrix,
    diagonal: Diag,
    // Each interior row: (global interface index in compact Gamma, A_{I,Gamma} entry).
    ig: Vec<Vec<(usize, f64)>>,
}

impl SparseLocal {
    fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, HybitError> {
        if rhs.is_empty() {
            return Ok(Vec::new());
        }
        let mut solution = vec![0.0; rhs.len()];
        let outcome = pcg(
            &self.a,
            &self.diagonal,
            rhs,
            &mut solution,
            SolverOptions {
                relative_tolerance: INNER_TOL,
                absolute_tolerance: 1.0e-14,
                max_iterations: 800,
            },
        )?;
        if outcome.status != SolveStatus::Converged {
            return Err(HybitError::NotConverged {
                iterations: outcome.iterations,
                residual: outcome.final_residual,
            });
        }
        Ok(solution)
    }
}

#[derive(Debug)]
struct Prepared {
    n: usize,
    gamma: Vec<usize>,
    locals: Vec<SparseLocal>,
    // compact-Gamma row adjacency of A_{Gamma,Gamma}
    gg: Vec<Vec<(usize, f64)>>,
    // compact-Gamma row adjacency of A_{Gamma,I}, using rank and local index.
    gi: Vec<Vec<(usize, usize, f64)>>,
    diagonal_gamma: Diag,
    interface_count: usize,
    interior_count: usize,
    stored_coupling_nnz: usize,
    stored_bytes_lower_bound: usize,
}

impl Prepared {
    fn new(a: &Csr32Matrix, owners: &PartitionAssignment) -> R<Self> {
        let n = a.nrows();
        if n != a.ncols() || owners.owners().len() != n {
            return Err("S3a expects a square matrix and complete owners".into());
        }
        let ranks = owners.rank_count() as usize;
        let mut on_gamma = vec![false; n];
        for i in 0..n {
            for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                let j = a.col_idx()[p] as usize;
                if owners.owners()[i] != owners.owners()[j] {
                    on_gamma[i] = true;
                    on_gamma[j] = true;
                }
            }
        }
        let gamma: Vec<usize> = (0..n).filter(|&i| on_gamma[i]).collect();
        let mut gamma_of = vec![usize::MAX; n];
        for (k, &global) in gamma.iter().enumerate() {
            gamma_of[global] = k;
        }
        let mut dofs_by_rank = vec![Vec::new(); ranks];
        let mut local_of = vec![usize::MAX; n];
        for i in 0..n {
            if !on_gamma[i] {
                let rank = owners.owners()[i] as usize;
                local_of[i] = dofs_by_rank[rank].len();
                dofs_by_rank[rank].push(i);
            }
        }
        let mut gg = vec![Vec::<(usize, f64)>::new(); gamma.len()];
        let mut gi = vec![Vec::<(usize, usize, f64)>::new(); gamma.len()];
        let mut diag_gamma = vec![0.0; gamma.len()];
        for (k, &i) in gamma.iter().enumerate() {
            for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                let j = a.col_idx()[p] as usize;
                let value = a.values()[p];
                if on_gamma[j] {
                    gg[k].push((gamma_of[j], value));
                    if i == j {
                        diag_gamma[k] += value;
                    }
                } else {
                    gi[k].push((owners.owners()[j] as usize, local_of[j], value));
                }
            }
        }
        let mut locals = Vec::with_capacity(ranks);
        let mut stored_bytes = 0usize;
        let mut stored_coupling_nnz = gi.iter().map(Vec::len).sum::<usize>();
        for (rank, dofs) in dofs_by_rank.into_iter().enumerate() {
            let mut ptr = Vec::with_capacity(dofs.len() + 1);
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            let mut ig = Vec::with_capacity(dofs.len());
            ptr.push(0u32);
            for &i in &dofs {
                let mut row_ig = Vec::new();
                for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                    let j = a.col_idx()[p] as usize;
                    let value = a.values()[p];
                    if on_gamma[j] {
                        row_ig.push((gamma_of[j], value));
                    } else {
                        if owners.owners()[j] as usize != rank {
                            return Err("cross-owner nonzero inside A_II".into());
                        }
                        cols.push(u32::try_from(local_of[j])?);
                        vals.push(value);
                    }
                }
                ig.push(row_ig);
                ptr.push(u32::try_from(cols.len())?);
            }
            stored_coupling_nnz += ig.iter().map(Vec::len).sum::<usize>();
            let local = Csr32Matrix::new(dofs.len(), dofs.len(), ptr, cols, vals)?;
            let diagonal = Diag::from_matrix(&local)?;
            stored_bytes += local.storage_bytes();
            locals.push(SparseLocal {
                dofs,
                a: local,
                diagonal,
                ig,
            });
        }
        stored_bytes += stored_coupling_nnz * (std::mem::size_of::<usize>() * 2 + 8);
        stored_bytes += gg.iter().map(Vec::len).sum::<usize>() * 16;
        let interior_count = locals.iter().map(|b| b.dofs.len()).sum::<usize>();
        if interior_count + gamma.len() != n {
            return Err("S3a partition accounting mismatch".into());
        }
        Ok(Self {
            n,
            interface_count: gamma.len(),
            interior_count,
            diagonal_gamma: Diag::from_values(&diag_gamma)?,
            gamma,
            locals,
            gg,
            gi,
            stored_coupling_nnz,
            stored_bytes_lower_bound: stored_bytes,
        })
    }

    fn eliminate_interior(&self, rhs: &[f64]) -> Result<Vec<Vec<f64>>, HybitError> {
        let mut outputs = Vec::with_capacity(self.locals.len());
        for local in &self.locals {
            let local_rhs: Vec<f64> = local.dofs.iter().map(|&global| rhs[global]).collect();
            outputs.push(local.solve(&local_rhs)?);
        }
        Ok(outputs)
    }

    fn condensed_rhs(&self, b: &[f64]) -> Result<Vec<f64>, HybitError> {
        let solved = self.eliminate_interior(b)?;
        let mut result: Vec<f64> = self.gamma.iter().map(|&global| b[global]).collect();
        for (k, row) in self.gi.iter().enumerate() {
            for &(rank, j, coefficient) in row {
                result[k] -= coefficient * solved[rank][j];
            }
        }
        Ok(result)
    }

    fn reconstruct(&self, b: &[f64], xg: &[f64]) -> Result<Vec<f64>, HybitError> {
        let mut x = vec![0.0; self.n];
        for (&global, &value) in self.gamma.iter().zip(xg) {
            x[global] = value;
        }
        for local in &self.locals {
            let mut rhs: Vec<f64> = local.dofs.iter().map(|&global| b[global]).collect();
            for (i, row) in local.ig.iter().enumerate() {
                for &(j, coefficient) in row {
                    rhs[i] -= coefficient * xg[j];
                }
            }
            let solution = local.solve(&rhs)?;
            for (&global, &value) in local.dofs.iter().zip(&solution) {
                x[global] = value;
            }
        }
        Ok(x)
    }
}

impl LinearOperator for Prepared {
    fn rows(&self) -> usize {
        self.interface_count
    }
    fn cols(&self) -> usize {
        self.interface_count
    }
    fn apply(&self, v: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
        if v.len() != self.interface_count || y.len() != self.interface_count {
            return Err(HybitError::InvalidArgument(
                "Schur vector dimension mismatch",
            ));
        }
        // Apply A_GammaGamma without building any explicit Schur matrix.
        for (i, row) in self.gg.iter().enumerate() {
            y[i] = row.iter().map(|&(j, coefficient)| coefficient * v[j]).sum();
        }
        let mut eliminated = Vec::with_capacity(self.locals.len());
        for local in &self.locals {
            let rhs: Vec<f64> = local
                .ig
                .iter()
                .map(|row| row.iter().map(|&(j, coefficient)| coefficient * v[j]).sum())
                .collect();
            eliminated.push(local.solve(&rhs)?);
        }
        for (i, row) in self.gi.iter().enumerate() {
            for &(rank, j, coefficient) in row {
                y[i] -= coefficient * eliminated[rank][j];
            }
        }
        Ok(())
    }
}

fn make_grid(nx: usize, ny: usize, vertical: f64) -> Csr32Matrix {
    let mut ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let mut row = vec![(y * nx + x, 2.25 + 2.0 * vertical)];
            if x > 0 {
                row.push((y * nx + x - 1, -1.0));
            }
            if x + 1 < nx {
                row.push((y * nx + x + 1, -1.0));
            }
            if y > 0 {
                row.push(((y - 1) * nx + x, -vertical));
            }
            if y + 1 < ny {
                row.push(((y + 1) * nx + x, -vertical));
            }
            row.sort_unstable_by_key(|v| v.0);
            for (j, a) in row {
                cols.push(j as u32);
                vals.push(a);
            }
            ptr.push(cols.len() as u32);
        }
    }
    Csr32Matrix::new(nx * ny, nx * ny, ptr, cols, vals).expect("grid CSR")
}

fn make_disconnected() -> Csr32Matrix {
    let mut ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..24usize {
        if i > 0 && (i - 1) / 12 == i / 12 {
            cols.push((i - 1) as u32);
            vals.push(-1.0);
        }
        cols.push(i as u32);
        vals.push(3.0);
        if i + 1 < 24 && (i + 1) / 12 == i / 12 {
            cols.push((i + 1) as u32);
            vals.push(-1.0);
        }
        ptr.push(cols.len() as u32);
    }
    Csr32Matrix::new(24, 24, ptr, cols, vals).expect("disconnected CSR")
}

#[derive(Clone, Copy)]
enum Owners {
    Columns,
    Checkerboard,
    Abtm,
    Disconnected,
}

struct Case {
    name: &'static str,
    nx: usize,
    ny: usize,
    ranks: u32,
    vertical: f64,
    owners: Owners,
}

const CASES: &[Case] = &[
    Case {
        name: "grid_12x12_columns4",
        nx: 12,
        ny: 12,
        ranks: 4,
        vertical: 1.0,
        owners: Owners::Columns,
    },
    Case {
        name: "grid_24x24_columns4",
        nx: 24,
        ny: 24,
        ranks: 4,
        vertical: 1.0,
        owners: Owners::Columns,
    },
    Case {
        name: "grid_40x40_columns4",
        nx: 40,
        ny: 40,
        ranks: 4,
        vertical: 0.2,
        owners: Owners::Columns,
    },
    Case {
        name: "grid_24x24_checkerboard4",
        nx: 24,
        ny: 24,
        ranks: 4,
        vertical: 1.0,
        owners: Owners::Checkerboard,
    },
    Case {
        name: "grid_32x32_abtm4",
        nx: 32,
        ny: 32,
        ranks: 4,
        vertical: 1.0,
        owners: Owners::Abtm,
    },
    Case {
        name: "disconnected_two_blocks",
        nx: 0,
        ny: 0,
        ranks: 2,
        vertical: 1.0,
        owners: Owners::Disconnected,
    },
];

fn make_case(case: &Case) -> R<(Csr32Matrix, PartitionAssignment, f64)> {
    if matches!(case.owners, Owners::Disconnected) {
        let a = make_disconnected();
        let owners =
            PartitionAssignment::from_owners(2, (0..24).map(|i| (i / 12) as u32).collect())?;
        return Ok((a, owners, 0.0));
    }
    let a = make_grid(case.nx, case.ny, case.vertical);
    let begin = Instant::now();
    let owners = match case.owners {
        Owners::Abtm => {
            abtm_multilevel_partition(
                &a,
                case.ranks,
                AbtmMultilevelOptions {
                    max_levels: 5,
                    refinement_passes: 2,
                    ..AbtmMultilevelOptions::default()
                },
            )?
            .0
        }
        Owners::Columns | Owners::Checkerboard => {
            let ids: Vec<u32> = (0..a.nrows())
                .map(|i| {
                    let x = i % case.nx;
                    let y = i / case.nx;
                    match case.owners {
                        Owners::Columns => (x * case.ranks as usize / case.nx) as u32,
                        Owners::Checkerboard => ((x + y) % case.ranks as usize) as u32,
                        _ => unreachable!(),
                    }
                })
                .collect();
            PartitionAssignment::from_owners(case.ranks, ids)?
        }
        Owners::Disconnected => unreachable!(),
    };
    Ok((a, owners, begin.elapsed().as_secs_f64() * 1e3))
}

#[derive(Debug)]
struct Row {
    name: &'static str,
    n: usize,
    nnz: usize,
    interior: usize,
    interface: usize,
    partition_ms: f64,
    prepare_ms: f64,
    sparse_bytes: usize,
    coupling_nnz: usize,
    schur_iterations: usize,
    global_iterations: usize,
    schur_solve_ms: f64,
    global_solve_ms: f64,
    schur_residual: f64,
    global_residual: f64,
    solution_rel_error: f64,
}

fn eval(case: &Case) -> R<Row> {
    let (a, owners, partition_ms) = make_case(case)?;
    let n = a.nrows();
    let x_ref: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.13).sin()).collect();
    let b = a.spmv(&x_ref)?;
    let t0 = Instant::now();
    let prepared = Prepared::new(&a, &owners)?;
    let prepare_ms = t0.elapsed().as_secs_f64() * 1e3;
    let opts = SolverOptions {
        relative_tolerance: OUTER_TOL,
        absolute_tolerance: 1.0e-13,
        max_iterations: 300,
    };
    let global_diag = Diag::from_matrix(&a)?;
    let t1 = Instant::now();
    let mut x_global = vec![0.0; n];
    let global = pcg(&a, &global_diag, &b, &mut x_global, opts)?;
    let global_solve_ms = t1.elapsed().as_secs_f64() * 1e3;
    if global.status != SolveStatus::Converged {
        return Err(format!("global PCG did not converge in {} steps", global.iterations).into());
    }
    let t2 = Instant::now();
    let condensed = prepared.condensed_rhs(&b)?;
    let (gamma_solution, schur): (Vec<f64>, Option<KrylovOutcome>) =
        if prepared.interface_count == 0 {
            (Vec::new(), None)
        } else {
            let mut x_gamma = vec![0.0; prepared.interface_count];
            let mut diag = Diag {
                inverse: prepared.diagonal_gamma.inverse.clone(),
            };
            let answer = fgmres(
                &prepared,
                &mut diag,
                &condensed,
                &mut x_gamma,
                FgmresOptions {
                    solver: opts,
                    restart: 32,
                },
            )?;
            if answer.status != SolveStatus::Converged {
                return Err(format!(
                    "Schur FGMRES did not converge in {} steps",
                    answer.iterations
                )
                .into());
            }
            (x_gamma, Some(answer))
        };
    let x_schur = prepared.reconstruct(&b, &gamma_solution)?;
    let schur_solve_ms = t2.elapsed().as_secs_f64() * 1e3;
    let schur_residual = relative(&a.spmv(&x_schur)?, &b);
    let global_residual = relative(&a.spmv(&x_global)?, &b);
    let solution_rel_error = relative(&x_schur, &x_ref);
    if schur_residual > TRUE_TOL || global_residual > TRUE_TOL || solution_rel_error > TRUE_TOL {
        return Err(format!(
            "true-residual gate failed for {}: Schur {:.3e}, global {:.3e}, solution {:.3e}",
            case.name, schur_residual, global_residual, solution_rel_error
        )
        .into());
    }
    Ok(Row {
        name: case.name,
        n,
        nnz: a.nnz(),
        interior: prepared.interior_count,
        interface: prepared.interface_count,
        partition_ms,
        prepare_ms,
        sparse_bytes: prepared.stored_bytes_lower_bound,
        coupling_nnz: prepared.stored_coupling_nnz,
        schur_iterations: schur.map_or(0, |result| result.iterations),
        global_iterations: global.iterations,
        schur_solve_ms,
        global_solve_ms,
        schur_residual,
        global_residual,
        solution_rel_error,
    })
}

fn main() -> R<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: schur_sparse_krylov_probe <results.csv>");
        std::process::exit(2);
    }
    let mut csv = BufWriter::new(File::create(&args[1])?);
    writeln!(csv, "case,n,nnz,interior,interface,partition_ms,prepare_ms,sparse_bytes_lower_bound,coupling_nnz,schur_iterations,global_iterations,schur_solve_ms,global_solve_ms,schur_true_residual,global_true_residual,schur_solution_error")?;
    for case in CASES {
        let r = eval(case)?;
        writeln!(
            csv,
            "{},{},{},{},{},{:.6},{:.6},{},{},{},{},{:.6},{:.6},{:.12e},{:.12e},{:.12e}",
            r.name,
            r.n,
            r.nnz,
            r.interior,
            r.interface,
            r.partition_ms,
            r.prepare_ms,
            r.sparse_bytes,
            r.coupling_nnz,
            r.schur_iterations,
            r.global_iterations,
            r.schur_solve_ms,
            r.global_solve_ms,
            r.schur_residual,
            r.global_residual,
            r.solution_rel_error,
        )?;
        println!(
            "PASS {} n={} I={} Gamma={} Schur/FGMRES={} global/PCG={} solve={:.3}/{:.3}ms true_res={:.3e}/{:.3e}",
            r.name, r.n, r.interior, r.interface, r.schur_iterations,
            r.global_iterations, r.schur_solve_ms, r.global_solve_ms,
            r.schur_residual, r.global_residual
        );
    }
    csv.flush()?;
    println!("=== HYBIT 0.9 SCHUR S3a SPARSE LOCAL / KRYLOV PILOT PASS ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3a_sparse_local_blocks_are_rank_separated() {
        let c = &CASES[0];
        let (a, owners, _) = make_case(c).unwrap();
        let plan = Prepared::new(&a, &owners).unwrap();
        assert_eq!(plan.interior_count + plan.interface_count, a.nrows());
        assert_eq!(plan.interface_count, 72);
        assert_eq!(plan.locals.len(), 4);
        assert!(plan
            .locals
            .iter()
            .all(|loc| loc.a.nrows() == loc.dofs.len()));
    }

    #[test]
    fn s3a_schur_numeric_grid_matches_original_spd_system() {
        eval(&CASES[0]).unwrap();
        eval(&CASES[1]).unwrap();
    }

    #[test]
    fn s3a_zero_interface_and_zero_interior() {
        let (a, owners, _) = make_case(&CASES[5]).unwrap();
        assert_eq!(Prepared::new(&a, &owners).unwrap().interface_count, 0);
        eval(&CASES[5]).unwrap();
        let (b, labels, _) = make_case(&CASES[3]).unwrap();
        assert_eq!(Prepared::new(&b, &labels).unwrap().interior_count, 0);
        eval(&CASES[3]).unwrap();
    }
}
