//! HyBIT 0.9 / Schur S2: small SPD reference and matrix-free cross-check.
//!
//! Experimental example only; no production solver, MPI, or preconditioner changes.
//! Interior/interface classification matches Schur S1: both endpoints of every
//! cross-owner structural nonzero are interface DOFs. Thus A_II is rank-block
//! diagonal. Local Cholesky is deliberately dense and bounded to n <= 256.

use hybit_distributed::{ContiguousPartition, PartitionAssignment};
use hybit_matrix::Csr32Matrix;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

type ResultT<T> = Result<T, Box<dyn Error>>;
const MAX_REFERENCE_DOF: usize = 256;
const TOL: f64 = 2.0e-11;

#[derive(Debug, Clone)]
struct DenseChol {
    n: usize,
    lower: Vec<f64>,
}

impl DenseChol {
    fn new(a: &[f64], n: usize) -> ResultT<Self> {
        if a.len() != n * n {
            return Err("dense Cholesky dimension mismatch".into());
        }
        let mut lower = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut entry = a[i * n + j];
                for k in 0..j {
                    entry -= lower[i * n + k] * lower[j * n + k];
                }
                if i == j {
                    if !entry.is_finite() || entry <= 0.0 {
                        return Err(format!("dense Cholesky: nonpositive pivot at {i}").into());
                    }
                    lower[i * n + i] = entry.sqrt();
                } else {
                    lower[i * n + j] = entry / lower[j * n + j];
                }
            }
        }
        Ok(Self { n, lower })
    }

    fn solve(&self, rhs: &[f64]) -> ResultT<Vec<f64>> {
        if rhs.len() != self.n {
            return Err("dense Cholesky RHS length mismatch".into());
        }
        let mut x = rhs.to_vec();
        for i in 0..self.n {
            for j in 0..i {
                let correction = self.lower[i * self.n + j] * x[j];
                x[i] -= correction;
            }
            x[i] /= self.lower[i * self.n + i];
        }
        for i in (0..self.n).rev() {
            for j in i + 1..self.n {
                let correction = self.lower[j * self.n + i] * x[j];
                x[i] -= correction;
            }
            x[i] /= self.lower[i * self.n + i];
        }
        Ok(x)
    }
}

fn to_dense(a: &Csr32Matrix) -> ResultT<Vec<f64>> {
    if a.nrows() != a.ncols() || a.nrows() > MAX_REFERENCE_DOF {
        return Err("S2 requires a square SPD matrix with n <= 256".into());
    }
    let n = a.nrows();
    let mut dense = vec![0.0; n * n];
    for i in 0..n {
        for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
            dense[i * n + a.col_idx()[p] as usize] += a.values()[p];
        }
    }
    for i in 0..n {
        for j in 0..i {
            let aij = dense[i * n + j];
            let aji = dense[j * n + i];
            if (aij - aji).abs() > 1e-12 * aij.abs().max(aji.abs()).max(1.0) {
                return Err("Schur S2 supports symmetric matrices only".into());
            }
        }
    }
    Ok(dense)
}

fn spmv(a: &Csr32Matrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows()];
    for (i, yi) in y.iter_mut().enumerate() {
        for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
            *yi += a.values()[p] * x[a.col_idx()[p] as usize];
        }
    }
    y
}

#[derive(Debug)]
struct Local {
    dofs: Vec<usize>,
    factor: DenseChol,
}

#[derive(Debug)]
struct Prepared<'a> {
    a: &'a Csr32Matrix,
    dense: Vec<f64>,
    gamma: Vec<usize>,
    locals: Vec<Local>,
    n_interior: usize,
}

impl<'a> Prepared<'a> {
    fn new(a: &'a Csr32Matrix, owners: &PartitionAssignment) -> ResultT<Self> {
        let dense = to_dense(a)?;
        let n = a.nrows();
        if owners.owners().len() != n {
            return Err("owner length does not equal n".into());
        }
        let mut on_interface = vec![false; n];
        for i in 0..n {
            for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                let j = a.col_idx()[p] as usize;
                if owners.owners()[i] != owners.owners()[j] {
                    on_interface[i] = true;
                    on_interface[j] = true;
                }
            }
        }
        let gamma: Vec<usize> = (0..n).filter(|&i| on_interface[i]).collect();
        let mut locals = Vec::new();
        let mut n_interior = 0usize;
        for rank in 0..owners.rank_count() {
            let dofs: Vec<usize> = (0..n)
                .filter(|&i| !on_interface[i] && owners.owners()[i] == rank)
                .collect();
            let m = dofs.len();
            n_interior += m;
            let mut block = vec![0.0; m * m];
            for (r, &i) in dofs.iter().enumerate() {
                for (c, &j) in dofs.iter().enumerate() {
                    block[r * m + c] = dense[i * n + j];
                }
            }
            locals.push(Local {
                dofs,
                factor: DenseChol::new(&block, m)?,
            });
        }
        if n_interior + gamma.len() != n {
            return Err("Schur interior/interface accounting mismatch".into());
        }
        // S1's two-sided interface rule forbids cross-rank A_II coupling.
        for i in 0..n {
            if on_interface[i] {
                continue;
            }
            for j in 0..n {
                if !on_interface[j]
                    && owners.owners()[i] != owners.owners()[j]
                    && dense[i * n + j] != 0.0
                {
                    return Err("unexpected cross-rank A_II coupling".into());
                }
            }
        }
        Ok(Self {
            a,
            dense,
            gamma,
            locals,
            n_interior,
        })
    }

    // Independent dense block formula: S = A_gg - sum_r A_gIr A_IrIr^-1 A_Irg.
    fn form_exact(&self) -> ResultT<Vec<f64>> {
        let ng = self.gamma.len();
        let n = self.a.nrows();
        let mut s = vec![0.0; ng * ng];
        for (i, &gi) in self.gamma.iter().enumerate() {
            for (j, &gj) in self.gamma.iter().enumerate() {
                s[i * ng + j] = self.dense[gi * n + gj];
            }
        }
        for local in &self.locals {
            for (j, &gj) in self.gamma.iter().enumerate() {
                let coupling: Vec<f64> =
                    local.dofs.iter().map(|&i| self.dense[i * n + gj]).collect();
                let solved = local.factor.solve(&coupling)?;
                for (i, &gi) in self.gamma.iter().enumerate() {
                    for (k, &ik) in local.dofs.iter().enumerate() {
                        s[i * ng + j] -= self.dense[gi * n + ik] * solved[k];
                    }
                }
            }
        }
        Ok(s)
    }

    // Matrix-free Schur operator: two CSR SpMVs + independent local solves.
    // No global Schur matrix is stored or read.
    fn apply_free(&self, v: &[f64]) -> ResultT<Vec<f64>> {
        if v.len() != self.gamma.len() {
            return Err("Schur vector length mismatch".into());
        }
        let n = self.a.nrows();
        let mut embedded = vec![0.0; n];
        for (k, &gi) in self.gamma.iter().enumerate() {
            embedded[gi] = v[k];
        }
        let av = spmv(self.a, &embedded);
        let mut eliminated = vec![0.0; n];
        for local in &self.locals {
            let rhs: Vec<f64> = local.dofs.iter().map(|&i| av[i]).collect();
            let solution = local.factor.solve(&rhs)?;
            for (&i, &value) in local.dofs.iter().zip(&solution) {
                eliminated[i] = value;
            }
        }
        let correction = spmv(self.a, &eliminated);
        Ok(self
            .gamma
            .iter()
            .map(|&gi| av[gi] - correction[gi])
            .collect())
    }

    fn solve_full_via_schur(&self, b: &[f64], s: &[f64]) -> ResultT<Vec<f64>> {
        let n = self.a.nrows();
        if b.len() != n || s.len() != self.gamma.len() * self.gamma.len() {
            return Err("Schur solve dimension mismatch".into());
        }
        let mut rhs_gamma: Vec<f64> = self.gamma.iter().map(|&i| b[i]).collect();
        for local in &self.locals {
            let rhs_i: Vec<f64> = local.dofs.iter().map(|&i| b[i]).collect();
            let solved = local.factor.solve(&rhs_i)?;
            for (k, &g) in self.gamma.iter().enumerate() {
                for (&i, &value) in local.dofs.iter().zip(&solved) {
                    rhs_gamma[k] -= self.dense[g * n + i] * value;
                }
            }
        }
        let gamma_solution = DenseChol::new(s, self.gamma.len())?.solve(&rhs_gamma)?;
        let mut x = vec![0.0; n];
        for (&i, &value) in self.gamma.iter().zip(&gamma_solution) {
            x[i] = value;
        }
        for local in &self.locals {
            let rhs_i: Vec<f64> = local
                .dofs
                .iter()
                .map(|&i| {
                    let mut value = b[i];
                    for (&g, &xg) in self.gamma.iter().zip(&gamma_solution) {
                        value -= self.dense[i * n + g] * xg;
                    }
                    value
                })
                .collect();
            let interior_solution = local.factor.solve(&rhs_i)?;
            for (&i, &value) in local.dofs.iter().zip(&interior_solution) {
                x[i] = value;
            }
        }
        Ok(x)
    }
}

fn rel_error(x: &[f64], reference: &[f64]) -> f64 {
    let num: f64 = x
        .iter()
        .zip(reference)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    let den: f64 = reference.iter().map(|&v| v * v).sum();
    num.sqrt() / den.sqrt().max(1e-30)
}

fn make_grid(nx: usize, ny: usize, vertical_weight: f64) -> Csr32Matrix {
    let n = nx * ny;
    let mut ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let mut row = vec![(y * nx + x, 2.25 + 2.0 * vertical_weight)];
            if x > 0 {
                row.push((y * nx + x - 1, -1.0));
            }
            if x + 1 < nx {
                row.push((y * nx + x + 1, -1.0));
            }
            if y > 0 {
                row.push(((y - 1) * nx + x, -vertical_weight));
            }
            if y + 1 < ny {
                row.push(((y + 1) * nx + x, -vertical_weight));
            }
            row.sort_unstable_by_key(|entry| entry.0);
            for (j, value) in row {
                cols.push(j as u32);
                vals.push(value);
            }
            ptr.push(cols.len() as u32);
        }
    }
    Csr32Matrix::new(n, n, ptr, cols, vals).expect("valid grid CSR")
}

fn make_disconnected() -> Csr32Matrix {
    let mut ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..12usize {
        if i > 0 && (i - 1) / 6 == i / 6 {
            cols.push((i - 1) as u32);
            vals.push(-1.0);
        }
        cols.push(i as u32);
        vals.push(3.0);
        if i + 1 < 12 && (i + 1) / 6 == i / 6 {
            cols.push((i + 1) as u32);
            vals.push(-1.0);
        }
        ptr.push(cols.len() as u32);
    }
    Csr32Matrix::new(12, 12, ptr, cols, vals).expect("valid disconnected CSR")
}

#[derive(Clone, Copy)]
enum PartitionStyle {
    Contiguous,
    Columns,
    Rows,
    Checkerboard,
    Disconnected,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    nx: usize,
    ny: usize,
    ranks: u32,
    style: PartitionStyle,
    vertical_weight: f64,
}

impl Case {
    fn build(self) -> ResultT<(Csr32Matrix, PartitionAssignment)> {
        if matches!(self.style, PartitionStyle::Disconnected) {
            return Ok((
                make_disconnected(),
                PartitionAssignment::from_owners(2, vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])?,
            ));
        }
        let matrix = make_grid(self.nx, self.ny, self.vertical_weight);
        let assignment = match self.style {
            PartitionStyle::Contiguous => {
                let contiguous = ContiguousPartition::balanced(matrix.nrows() as u64, self.ranks)?;
                PartitionAssignment::from_contiguous(&contiguous)?
            }
            _ => {
                let owners = (0..matrix.nrows())
                    .map(|i| {
                        let x = i % self.nx;
                        let y = i / self.nx;
                        match self.style {
                            PartitionStyle::Columns => (x * self.ranks as usize / self.nx) as u32,
                            PartitionStyle::Rows => (y * self.ranks as usize / self.ny) as u32,
                            PartitionStyle::Checkerboard => ((x + y) % self.ranks as usize) as u32,
                            PartitionStyle::Contiguous | PartitionStyle::Disconnected => {
                                unreachable!()
                            }
                        }
                    })
                    .collect();
                PartitionAssignment::from_owners(self.ranks, owners)?
            }
        };
        Ok((matrix, assignment))
    }
}

const CASES: &[Case] = &[
    Case {
        name: "grid_5x5_columns2",
        nx: 5,
        ny: 5,
        ranks: 2,
        style: PartitionStyle::Columns,
        vertical_weight: 1.0,
    },
    Case {
        name: "grid_8x8_contiguous4",
        nx: 8,
        ny: 8,
        ranks: 4,
        style: PartitionStyle::Contiguous,
        vertical_weight: 1.0,
    },
    Case {
        name: "grid_9x9_columns3",
        nx: 9,
        ny: 9,
        ranks: 3,
        style: PartitionStyle::Columns,
        vertical_weight: 0.2,
    },
    Case {
        name: "grid_10x10_rows2",
        nx: 10,
        ny: 10,
        ranks: 2,
        style: PartitionStyle::Rows,
        vertical_weight: 2.0,
    },
    Case {
        name: "grid_12x12_columns4",
        nx: 12,
        ny: 12,
        ranks: 4,
        style: PartitionStyle::Columns,
        vertical_weight: 1.0,
    },
    Case {
        name: "grid_8x8_checkerboard4",
        nx: 8,
        ny: 8,
        ranks: 4,
        style: PartitionStyle::Checkerboard,
        vertical_weight: 1.0,
    },
    Case {
        name: "disconnected_two_blocks",
        nx: 0,
        ny: 0,
        ranks: 2,
        style: PartitionStyle::Disconnected,
        vertical_weight: 1.0,
    },
];

#[derive(Debug)]
struct Results {
    n: usize,
    interior: usize,
    gamma: usize,
    ranks: u32,
    exact_build_ms: f64,
    matrixfree_ms: f64,
    solve_ms: f64,
    action_rel_error: f64,
    full_solution_rel_error: f64,
    original_rel_residual: f64,
}

fn evaluate(matrix: &Csr32Matrix, owners: &PartitionAssignment) -> ResultT<Results> {
    let prepared = Prepared::new(matrix, owners)?;
    let n = matrix.nrows();
    let t1 = Instant::now();
    let s = prepared.form_exact()?;
    let exact_build_ms = t1.elapsed().as_secs_f64() * 1e3;
    let ng = prepared.gamma.len();
    let probe: Vec<f64> = (0..ng).map(|i| ((i + 2) as f64 * 0.37).sin()).collect();
    let expected: Vec<f64> = (0..ng)
        .map(|i| (0..ng).map(|j| s[i * ng + j] * probe[j]).sum())
        .collect();
    let t2 = Instant::now();
    let actual = prepared.apply_free(&probe)?;
    let matrixfree_ms = t2.elapsed().as_secs_f64() * 1e3;
    let action_rel_error = rel_error(&actual, &expected);
    let x_true: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.19).cos()).collect();
    let b = spmv(matrix, &x_true);
    let t3 = Instant::now();
    let x_schur = prepared.solve_full_via_schur(&b, &s)?;
    let solve_ms = t3.elapsed().as_secs_f64() * 1e3;
    let full_solution_rel_error = rel_error(&x_schur, &x_true);
    let reconstructed_b = spmv(matrix, &x_schur);
    let original_rel_residual = rel_error(&reconstructed_b, &b);
    // Cross-check against an independent dense solve of the complete A.
    let x_full = DenseChol::new(&prepared.dense, n)?.solve(&b)?;
    let direct_rel_error = rel_error(&x_schur, &x_full);
    if action_rel_error > TOL
        || full_solution_rel_error > TOL
        || original_rel_residual > TOL
        || direct_rel_error > TOL
    {
        return Err(format!(
            "S2 accuracy gate failed: action={action_rel_error:e} solution={full_solution_rel_error:e} residual={original_rel_residual:e} direct={direct_rel_error:e}"
        ).into());
    }
    Ok(Results {
        n,
        interior: prepared.n_interior,
        gamma: ng,
        ranks: owners.rank_count(),
        exact_build_ms,
        matrixfree_ms,
        solve_ms,
        action_rel_error,
        full_solution_rel_error,
        original_rel_residual,
    })
}

fn main() -> ResultT<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: schur_reference_probe <results.csv>");
        std::process::exit(2);
    }
    let mut out = BufWriter::new(File::create(&args[1])?);
    writeln!(out, "case,n,ranks,interior,interface,exact_build_ms,matrixfree_apply_ms,solve_ms,action_rel_error,solution_rel_error,original_rel_residual")?;
    for case in CASES {
        let (matrix, owners) = case.build()?;
        let result = evaluate(&matrix, &owners)?;
        println!(
            "PASS {}: n={} I={} Gamma={} action={:.3e} solution={:.3e} residual={:.3e}",
            case.name,
            result.n,
            result.interior,
            result.gamma,
            result.action_rel_error,
            result.full_solution_rel_error,
            result.original_rel_residual
        );
        writeln!(
            out,
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.12e},{:.12e},{:.12e}",
            case.name,
            result.n,
            result.ranks,
            result.interior,
            result.gamma,
            result.exact_build_ms,
            result.matrixfree_ms,
            result.solve_ms,
            result.action_rel_error,
            result.full_solution_rel_error,
            result.original_rel_residual
        )?;
    }
    out.flush()?;
    println!("=== HYBIT 0.9 SCHUR S2 EXACT / MATRIX-FREE CROSSCHECK PASS ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2_reference_and_matrix_free_all_cases() {
        for case in CASES {
            let (a, owners) = case.build().unwrap();
            evaluate(&a, &owners).unwrap();
        }
    }

    #[test]
    fn s2_s1_boundary_counts() {
        let case = CASES[0];
        let (a, owners) = case.build().unwrap();
        let prepared = Prepared::new(&a, &owners).unwrap();
        assert_eq!(prepared.gamma.len(), 10); // two columns on opposite sides of split, each with 5 rows
        assert_eq!(prepared.n_interior, 15);
    }

    #[test]
    fn s2_rejects_nonsymmetric_input() {
        let matrix =
            Csr32Matrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![3.0, -1.0, 3.0]).unwrap();
        let owners = PartitionAssignment::from_owners(2, vec![0, 1]).unwrap();
        assert!(Prepared::new(&matrix, &owners).is_err());
    }

    #[test]
    fn s2_rejects_indefinite_interior() {
        let matrix = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![-1.0, 2.0]).unwrap();
        let owners = PartitionAssignment::from_owners(2, vec![0, 1]).unwrap();
        assert!(Prepared::new(&matrix, &owners).is_err());
    }

    #[test]
    fn s2_no_interface_returns_exact_local_solution() {
        let (matrix, owners) = CASES[6].build().unwrap();
        let prepared = Prepared::new(&matrix, &owners).unwrap();
        assert!(prepared.gamma.is_empty());
        let b = vec![1.0; matrix.nrows()];
        let x = prepared.solve_full_via_schur(&b, &[]).unwrap();
        assert!(rel_error(&spmv(&matrix, &x), &b) < TOL);
    }
}
