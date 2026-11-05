//! C1-H15-A: research-only ABTM region construction + bounded dense-LU RAS.
//!
//! This small-matrix *reference* experiment does not modify prepared HyBIT,
//! automatic routing, or the existing FEM FGMRES benchmark. In particular,
//! dense LU is NOT a scalable sparse PARDISO replacement.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    Csr32Matrix, FgmresOptions, HybitError, Ilu0Preconditioner, Preconditioner, SolveStatus,
    SolverOptions, fgmres, read_matrix_market,
};
use hybit_matrix::{AbtmDualTopology, AbtmTopologyRow, DofMask};

const MAX_REFERENCE_ROWS: usize = 10_000;
const DEFAULT_FACTOR_BUDGET_BYTES: usize = 128 * 1024 * 1024;
const RELATIVE_PIVOT_FLOOR: f64 = 1.0e-13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Ilu0Abtm,
    RasLu,
}

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: PathBuf,
    arm: Arm,
    tol: f64,
    max_iters: usize,
    restart: usize,
    core_rows: usize,
    overlap: usize,
    max_local_rows: usize,
    max_factor_bytes: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut arm = Arm::RasLu;
        let mut tol: f64 = 1.0e-10;
        let mut max_iters = 360;
        let mut restart = 30;
        let mut core_rows = 64;
        let mut overlap = 0;
        let mut max_local_rows = 512;
        let mut max_factor_bytes = DEFAULT_FACTOR_BUDGET_BYTES;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            let value = match arg.as_str() {
                "--matrix" | "--rhs" | "--precond" | "--tol" | "--max-iters" | "--restart"
                | "--core-rows" | "--overlap" | "--max-local-rows" | "--max-factor-bytes" => {
                    args.next().ok_or(format!("{arg} needs a value"))?
                }
                "--help" | "-h" => {
                    println!("Usage: c1h15_schwarz_probe --matrix FILE.mm --rhs FILE.rhs");
                    println!(
                        "  [--precond ilu0-abtm|ras-lu] [--tol 1e-10] [--max-iters 360] [--restart 30]"
                    );
                    println!(
                        "  [--core-rows 64] [--overlap 0|1] [--max-local-rows 512] [--max-factor-bytes 134217728]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option: {other}").into()),
            };
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(value)),
                "--rhs" => rhs = Some(PathBuf::from(value)),
                "--precond" => {
                    arm = match value.as_str() {
                        "ilu0-abtm" => Arm::Ilu0Abtm,
                        "ras-lu" => Arm::RasLu,
                        _ => return Err(format!("unknown preconditioner: {value}").into()),
                    };
                }
                "--tol" => tol = value.parse()?,
                "--max-iters" => max_iters = value.parse()?,
                "--restart" => restart = value.parse()?,
                "--core-rows" => core_rows = value.parse()?,
                "--overlap" => overlap = value.parse()?,
                "--max-local-rows" => max_local_rows = value.parse()?,
                "--max-factor-bytes" => max_factor_bytes = value.parse()?,
                _ => unreachable!(),
            }
        }
        if !tol.is_finite() || tol <= 0.0 || max_iters == 0 || restart == 0 {
            return Err("invalid tolerance, iterations or restart".into());
        }
        if core_rows == 0
            || max_local_rows == 0
            || core_rows > max_local_rows
            || overlap > 1
            || max_factor_bytes == 0
        {
            return Err("invalid local capacity, overlap (only 0 or 1), or factor budget".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("--matrix is required")?,
            rhs: rhs.ok_or("--rhs is required")?,
            arm,
            tol,
            max_iters,
            restart,
            core_rows,
            overlap,
            max_local_rows,
            max_factor_bytes,
        })
    }
}

fn read_rhs(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let source = fs::read_to_string(path)?;
    let mut result = Vec::with_capacity(n);
    for word in source.split_whitespace() {
        let value: f64 = word.replace('D', "E").replace('d', "e").parse()?;
        if !value.is_finite() || result.len() >= n {
            return Err("invalid or oversized RHS".into());
        }
        result.push(value);
    }
    if result.len() != n {
        return Err(format!("RHS has {} values, expected {n}", result.len()).into());
    }
    Ok(result)
}

fn stable_norm(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0_f64, f64::hypot)
}

fn verified_relative(a: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<f64, HybitError> {
    let ax = a.spmv(x)?;
    let rnorm = stable_norm(b.iter().zip(ax).map(|(&bi, ai)| bi - ai));
    let bnorm = stable_norm(b.iter().copied());
    Ok(if bnorm == 0.0 { rnorm } else { rnorm / bnorm })
}

fn enqueue(
    row: AbtmTopologyRow<'_>,
    queued: &mut DofMask,
    claimed: &DofMask,
    frontier: &mut VecDeque<usize>,
) -> Result<(), HybitError> {
    for word in row.words() {
        let wi = word.word_index() as usize;
        let mut bits = word.mask() & !queued.words()[wi] & !claimed.words()[wi];
        while bits != 0 {
            let neighbor = wi * 64 + bits.trailing_zeros() as usize;
            queued.set(neighbor, true)?;
            frontier.push_back(neighbor);
            bits &= bits - 1;
        }
    }
    Ok(())
}

fn add_overlap(row: AbtmTopologyRow<'_>, selected: &mut DofMask) -> Result<(), HybitError> {
    for word in row.words() {
        let wi = word.word_index() as usize;
        let mut bits = word.mask() & !selected.words()[wi];
        while bits != 0 {
            selected.set(wi * 64 + bits.trailing_zeros() as usize, true)?;
            bits &= bits - 1;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct Region {
    core: Vec<usize>,
    rows: Vec<usize>,
}

fn regions(
    topology: &AbtmDualTopology,
    core_rows: usize,
    overlap: usize,
    max_local_rows: usize,
) -> Result<Vec<Region>, HybitError> {
    let n = topology.nrows();
    if n != topology.ncols() {
        return Err(HybitError::InvalidMatrix(
            "Schwarz requires square topology",
        ));
    }
    let mut claimed = DofMask::new(n);
    let mut result = Vec::new();
    for seed in 0..n {
        if claimed.contains(seed) {
            continue;
        }
        let mut queued = DofMask::new(n);
        let mut frontier = VecDeque::new();
        queued.set(seed, true)?;
        frontier.push_back(seed);
        let mut core = Vec::new();
        while let Some(row) = frontier.pop_front() {
            if claimed.contains(row) {
                continue;
            }
            claimed.set(row, true)?;
            core.push(row);
            if core.len() == core_rows {
                break;
            }
            enqueue(topology.row(row)?, &mut queued, &claimed, &mut frontier)?;
            enqueue(topology.column(row)?, &mut queued, &claimed, &mut frontier)?;
        }
        let mut selected = DofMask::new(n);
        for &row in &core {
            selected.set(row, true)?;
        }
        if overlap == 1 {
            for &row in &core {
                add_overlap(topology.row(row)?, &mut selected)?;
                add_overlap(topology.column(row)?, &mut selected)?;
            }
        }
        let rows = selected.indices();
        if rows.len() > max_local_rows {
            return Err(HybitError::InvalidArgument(
                "Schwarz overlap exceeds maximum local rows",
            ));
        }
        result.push(Region { core, rows });
    }
    if claimed.count_ones() != n {
        return Err(HybitError::InvalidMatrix(
            "Schwarz cores did not cover all rows",
        ));
    }
    Ok(result)
}

/// Dense LU is a deliberately bounded reference implementation for small FEM.
struct LocalLu {
    rows: Vec<usize>,
    core_pairs: Vec<(usize, usize)>,
    values: Vec<f64>,
    pivots: Vec<usize>,
}

impl LocalLu {
    fn factor(
        matrix: &Csr32Matrix,
        region: Region,
        position: &mut [usize],
    ) -> Result<Self, HybitError> {
        let m = region.rows.len();
        let mut values = vec![0.0_f64; m * m];
        for (local, &global) in region.rows.iter().enumerate() {
            position[global] = local;
        }
        for (local, &global) in region.rows.iter().enumerate() {
            let start = matrix.row_ptr()[global] as usize;
            let end = matrix.row_ptr()[global + 1] as usize;
            for idx in start..end {
                let col = matrix.col_idx()[idx] as usize;
                let mapped = position[col];
                if mapped != usize::MAX {
                    values[local * m + mapped] += matrix.values()[idx];
                }
            }
        }
        let mut core_pairs = Vec::with_capacity(region.core.len());
        for &global in &region.core {
            core_pairs.push((global, position[global]));
        }
        for &global in &region.rows {
            position[global] = usize::MAX;
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(HybitError::NumericalBreakdown(
                "Schwarz local matrix contains non-finite values",
            ));
        }
        let reference_scale = values.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let mut pivots = Vec::with_capacity(m);
        for k in 0..m {
            let mut pivot_row = k;
            let mut largest = values[k * m + k].abs();
            for row in (k + 1)..m {
                let value = values[row * m + k].abs();
                if value > largest {
                    largest = value;
                    pivot_row = row;
                }
            }
            if !largest.is_finite()
                || reference_scale == 0.0
                || largest <= RELATIVE_PIVOT_FLOOR * reference_scale
            {
                return Err(HybitError::NumericalBreakdown(
                    "Schwarz local LU singular or invalid pivot",
                ));
            }
            pivots.push(pivot_row);
            if pivot_row != k {
                for col in 0..m {
                    values.swap(k * m + col, pivot_row * m + col);
                }
            }
            let pivot = values[k * m + k];
            for row in (k + 1)..m {
                let multiplier = values[row * m + k] / pivot;
                if !multiplier.is_finite() {
                    return Err(HybitError::NumericalBreakdown("non-finite LU multiplier"));
                }
                values[row * m + k] = multiplier;
                for col in (k + 1)..m {
                    let correction = multiplier * values[k * m + col];
                    values[row * m + col] -= correction;
                }
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(HybitError::NumericalBreakdown(
                "Schwarz LU became non-finite",
            ));
        }
        Ok(Self {
            rows: region.rows,
            core_pairs,
            values,
            pivots,
        })
    }

    fn solve_in_place(&self, work: &mut [f64]) -> Result<(), HybitError> {
        let m = self.rows.len();
        for (k, &pivot) in self.pivots.iter().enumerate() {
            work.swap(k, pivot);
        }
        for row in 0..m {
            for col in 0..row {
                let correction = self.values[row * m + col] * work[col];
                work[row] -= correction;
            }
        }
        for row in (0..m).rev() {
            for col in (row + 1)..m {
                let correction = self.values[row * m + col] * work[col];
                work[row] -= correction;
            }
            work[row] /= self.values[row * m + row];
            if !work[row].is_finite() {
                return Err(HybitError::NumericalBreakdown(
                    "Schwarz local triangular solve became non-finite",
                ));
            }
        }
        Ok(())
    }
}

struct RasLu {
    n: usize,
    factors: Vec<LocalLu>,
    workspace: RefCell<Vec<f64>>,
    factor_bytes: usize,
    max_region_rows: usize,
    dual_metadata_bytes: usize,
    graph_ms: f64,
    region_ms: f64,
    numeric_ms: f64,
}

impl RasLu {
    fn build(matrix: &Csr32Matrix, args: &Args) -> Result<Self, HybitError> {
        let n = matrix.nrows();
        if n != matrix.ncols() || n == 0 || n > MAX_REFERENCE_ROWS {
            return Err(HybitError::InvalidArgument(
                "dense Schwarz reference requires 1..=10000 square rows",
            ));
        }
        let graph_start = Instant::now();
        let topology = AbtmDualTopology::from_csr32(matrix)?;
        let graph_ms = graph_start.elapsed().as_secs_f64() * 1e3;
        let dual_metadata_bytes = topology.stats().total_metadata_bytes();

        let region_start = Instant::now();
        let partitions = regions(&topology, args.core_rows, args.overlap, args.max_local_rows)?;
        let region_ms = region_start.elapsed().as_secs_f64() * 1e3;
        let max_region_rows = partitions.iter().map(|r| r.rows.len()).max().unwrap_or(0);

        // Before any factor allocation, enforce the complete resident dense
        // factor budget. Ancillary topology, indices and scratch are separate.
        let mut factor_bytes = 0usize;
        for region in &partitions {
            let m = region.rows.len();
            let bytes = m
                .checked_mul(m)
                .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
                .ok_or(HybitError::SizeOverflow)?;
            factor_bytes = factor_bytes
                .checked_add(bytes)
                .ok_or(HybitError::SizeOverflow)?;
        }
        if factor_bytes > args.max_factor_bytes {
            return Err(HybitError::InvalidArgument(
                "Schwarz dense LU exceeds explicit factor-byte budget",
            ));
        }

        let numeric_start = Instant::now();
        let mut position = vec![usize::MAX; n];
        let mut factors = Vec::with_capacity(partitions.len());
        for region in partitions {
            factors.push(LocalLu::factor(matrix, region, &mut position)?);
        }
        let numeric_ms = numeric_start.elapsed().as_secs_f64() * 1e3;
        Ok(Self {
            n,
            factors,
            workspace: RefCell::new(vec![0.0; max_region_rows]),
            factor_bytes,
            max_region_rows,
            dual_metadata_bytes,
            graph_ms,
            region_ms,
            numeric_ms,
        })
    }
}

impl Preconditioner for RasLu {
    fn len(&self) -> usize {
        self.n
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n || z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        z.fill(0.0);
        let mut scratch = self.workspace.borrow_mut();
        for local in &self.factors {
            let m = local.rows.len();
            for (i, &row) in local.rows.iter().enumerate() {
                scratch[i] = r[row];
            }
            local.solve_in_place(&mut scratch[..m])?;
            // Restricted additive Schwarz: each global core row is written
            // exactly once even when local factorizations overlap.
            for &(global, index) in &local.core_pairs {
                z[global] = scratch[index];
            }
        }
        Ok(())
    }
}

struct ProbeMetrics {
    name: &'static str,
    setup_ms: f64,
    graph_ms: f64,
    region_ms: f64,
    numeric_ms: f64,
    factor_bytes: usize,
    max_local_rows: usize,
    regions: usize,
    metadata_bytes: usize,
}

fn solve<P: Preconditioner>(
    matrix: &Csr32Matrix,
    rhs: &[f64],
    args: &Args,
    mut precond: P,
    metrics: ProbeMetrics,
) -> Result<(), Box<dyn Error>> {
    let mut x = vec![0.0; matrix.nrows()];
    let started = Instant::now();
    let outcome = fgmres(
        matrix,
        &mut precond,
        rhs,
        &mut x,
        FgmresOptions {
            solver: SolverOptions {
                relative_tolerance: args.tol,
                absolute_tolerance: 0.0,
                max_iterations: args.max_iters,
            },
            restart: args.restart,
        },
    )?;
    let solver_ms = started.elapsed().as_secs_f64() * 1e3;
    let verified = verified_relative(matrix, rhs, &x)?;
    if !verified.is_finite() {
        return Err("FGMRES true relative residual became non-finite".into());
    }
    if outcome.status == SolveStatus::Converged && verified > args.tol * (1.0 + 1.0e-8) {
        return Err("FGMRES convergence claim failed independent true-residual gate".into());
    }
    println!(
        "C1H15_RESULT|arm={}|status={:?}|iterations={}|verified_relative={verified:.12e}|reported_final_norm={:.12e}|setup_ms={:.6}|graph_ms={:.6}|region_ms={:.6}|numeric_ms={:.6}|solver_ms={solver_ms:.6}|factor_bytes={}|dual_metadata_bytes={}|regions={}|max_local_rows={}|restart={}|max_iters={}|tol={:.12e}",
        metrics.name,
        outcome.status,
        outcome.iterations,
        outcome.final_residual,
        metrics.setup_ms,
        metrics.graph_ms,
        metrics.region_ms,
        metrics.numeric_ms,
        metrics.factor_bytes,
        metrics.metadata_bytes,
        metrics.regions,
        metrics.max_local_rows,
        args.restart,
        args.max_iters,
        args.tol
    );
    // The process may return zero on MaxIterations. Supervising scripts must
    // inspect status and independently verified residual, not only exit code.
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let load_started = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err("matrix must be complete and square, not MPI rank-local".into());
    }
    let rhs = read_rhs(&args.rhs, matrix.nrows())?;
    println!(
        "C1H15_INPUT|rows={}|input_nnz={}|csr_nnz={}|load_ms={:.6}|arm={:?}",
        matrix.nrows(),
        mm.input_entries,
        mm.csr_nnz,
        load_started.elapsed().as_secs_f64() * 1e3,
        args.arm
    );
    let started = Instant::now();
    match args.arm {
        Arm::Ilu0Abtm => {
            let factor = Ilu0Preconditioner::from_csr32_general_abtm(&matrix)?;
            let setup_ms = started.elapsed().as_secs_f64() * 1e3;
            let bytes = factor.factor_bytes();
            solve(
                &matrix,
                &rhs,
                &args,
                factor,
                ProbeMetrics {
                    name: "ilu0-abtm",
                    setup_ms,
                    graph_ms: 0.0,
                    region_ms: 0.0,
                    numeric_ms: setup_ms,
                    factor_bytes: bytes,
                    max_local_rows: 0,
                    regions: 0,
                    metadata_bytes: 0,
                },
            )?;
        }
        Arm::RasLu => {
            let factor = RasLu::build(&matrix, &args)?;
            let setup_ms = started.elapsed().as_secs_f64() * 1e3;
            let graph_ms = factor.graph_ms;
            let region_ms = factor.region_ms;
            let numeric_ms = factor.numeric_ms;
            let factor_bytes = factor.factor_bytes;
            let max_region_rows = factor.max_region_rows;
            let regions = factor.factors.len();
            let metadata_bytes = factor.dual_metadata_bytes;
            solve(
                &matrix,
                &rhs,
                &args,
                factor,
                ProbeMetrics {
                    name: "ras-lu",
                    setup_ms,
                    graph_ms,
                    region_ms,
                    numeric_ms,
                    factor_bytes,
                    max_local_rows: max_region_rows,
                    regions,
                    metadata_bytes,
                },
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Args {
        Args {
            matrix: PathBuf::new(),
            rhs: PathBuf::new(),
            arm: Arm::RasLu,
            tol: 1.0e-10,
            max_iters: 60,
            restart: 8,
            core_rows: 2,
            overlap: 0,
            max_local_rows: 4,
            max_factor_bytes: 1024,
        }
    }

    #[test]
    fn pivoted_local_lu_recovers_nonsymmetric_exact_solution() {
        let a = Csr32Matrix::new(2, 2, vec![0, 1, 3], vec![1, 0, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let prec = RasLu::build(&a, &defaults()).unwrap();
        let mut x = vec![0.0; 2];
        prec.apply(&[2.0, 8.0], &mut x).unwrap();
        assert!((x[0] - 1.0).abs() < 1.0e-12);
        assert!((x[1] - 2.0).abs() < 1.0e-12);
        assert_eq!(prec.factors.len(), 1);
    }

    #[test]
    fn disconnected_blocks_are_covered_exactly_once() {
        let a = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 0, 1, 2, 3, 2, 3],
            vec![2.0, 1.0, -1.0, 3.0, 4.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let prec = RasLu::build(&a, &defaults()).unwrap();
        assert_eq!(prec.factors.len(), 2);
        let mut x = vec![0.0; 4];
        prec.apply(&[4.0, 5.0, 20.0, 10.0], &mut x).unwrap();
        for (&got, want) in x.iter().zip([1.0, 2.0, 5.0, 0.0]) {
            assert!((got - want).abs() < 1.0e-12);
        }
    }

    #[test]
    fn fgmres_with_ras_converges_and_true_residual_is_verified() {
        let a = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 0, 1, 2, 3, 2, 3],
            vec![2.0, 1.0, -1.0, 3.0, 4.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let b = [4.0, 5.0, 20.0, 10.0];
        let mut ras = RasLu::build(&a, &defaults()).unwrap();
        let mut x = [0.0; 4];
        let outcome = fgmres(
            &a,
            &mut ras,
            &b,
            &mut x,
            FgmresOptions {
                solver: SolverOptions {
                    relative_tolerance: 1.0e-12,
                    absolute_tolerance: 0.0,
                    max_iterations: 8,
                },
                restart: 4,
            },
        )
        .unwrap();
        assert_eq!(outcome.status, SolveStatus::Converged);
        assert!(verified_relative(&a, &b, &x).unwrap() < 1.0e-12);
    }

    #[test]
    fn overlap_uses_only_core_for_restricted_injection() {
        let a = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
        )
        .unwrap();
        let mut args = defaults();
        args.core_rows = 1;
        args.overlap = 1;
        let prec = RasLu::build(&a, &args).unwrap();
        assert_eq!(prec.factors.len(), 3);
        assert!(prec.factors.iter().any(|f| f.rows.len() > 1));
        let mut output = vec![0.0; 3];
        prec.apply(&[1.0, 2.0, 3.0], &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn singular_factor_and_budget_excess_fail_closed() {
        let singular = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![1, 1], vec![1.0, 0.0]).unwrap();
        assert!(RasLu::build(&singular, &defaults()).is_err());

        let invertible = Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, 1.0, 1.0, 2.0],
        )
        .unwrap();
        let mut args = defaults();
        args.max_factor_bytes = 8;
        assert!(RasLu::build(&invertible, &args).is_err());
    }
}
