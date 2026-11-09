//! C1-H12-A10B: multiple-RHS hierarchy reuse with the A8 ABTM Galerkin kernel.
//! Isolates hierarchy lifecycle reuse, not a claim of ABTM-only speedup.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    AbtmTopology, Csr32Matrix, HybitError, LinearOperator, Preconditioner, SolveStatus,
    SolverOptions, TwoLevelAggregation, TwoLevelBasis, TwoLevelBlockJacobiPreconditioner,
    TwoLevelCoarseApplyPolicy, TwoLevelTransferApplyPolicy, TwoLevelTransferOptions,
    TwoLevelTransferStoragePolicy, TwoLevelTransferValueStoragePolicy, pcg, read_matrix_market,
};

const TARGET_COARSE: usize = 512;
const AGGREGATE_SIZE: usize = 4;
const MAX_LEVELS: usize = 8;
const STRENGTH_THRESHOLD: f64 = 0.25;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Policy {
    TwoLevel,
    Multilevel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReuseMode {
    Fresh,
    Reuse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RhsMode {
    Ones,
    IndexSmooth,
    SeededRandom,
    PointLoad,
}

/// The manufactured cases retain an exact x for an additional error check.
/// PointLoad uses a physical-looking sparse load without manufacturing x.
fn make_rhs(
    matrix: &Csr32Matrix,
    mode: RhsMode,
) -> Result<(Vec<f64>, Option<Vec<f64>>), HybitError> {
    let n = matrix.nrows();
    if n == 0 {
        return Err(HybitError::InvalidArgument(
            "RHS diversity needs nonempty matrix",
        ));
    }
    if mode == RhsMode::PointLoad {
        let mut rhs = vec![0.0; n];
        rhs[n / 2] = 1.0;
        return Ok((rhs, None));
    }
    let exact = match mode {
        RhsMode::Ones => vec![1.0; n],
        RhsMode::IndexSmooth => (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                0.5 + 0.3 * t.sin() + 0.2 * (3.0 * t).cos()
            })
            .collect(),
        RhsMode::SeededRandom => {
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    // 53 deterministic mantissa bits, uniform in [-1, 1).
                    (state >> 11) as f64 * (2.0 / (1u64 << 53) as f64) - 1.0
                })
                .collect()
        }
        RhsMode::PointLoad => unreachable!(),
    };
    Ok((matrix.spmv(&exact)?, Some(exact)))
}

fn rhs_hash(rhs: &[f64]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for v in rhs {
        for byte in v.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

struct Args {
    matrix: PathBuf,
    case_name: String,
    aggregation: TwoLevelAggregation,
    basis: TwoLevelBasis,
    policy: Policy,
    tolerance: f64,
    reuse_mode: ReuseMode,
    repeats: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut case_name = None;
        let mut aggregation = None;
        let mut basis = None;
        let mut policy = None;
        let mut tolerance = None;
        let mut reuse_mode = None;
        let mut repeats = None;
        let mut arguments = env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or(format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(value)),
                "--case" => case_name = Some(value),
                "--aggregation" => {
                    aggregation = Some(match value.as_str() {
                        "contiguous" => TwoLevelAggregation::Contiguous,
                        "graph" => TwoLevelAggregation::Graph,
                        "strong-graph" => TwoLevelAggregation::StrongGraph,
                        _ => return Err(format!("unknown aggregation {value}").into()),
                    });
                }
                "--basis" => {
                    basis = Some(match value.as_str() {
                        "piecewise" => TwoLevelBasis::PiecewiseConstant,
                        "smoothed" => TwoLevelBasis::JacobiSmoothed,
                        _ => return Err(format!("unknown basis {value}").into()),
                    });
                }
                "--policy" => {
                    policy = Some(match value.as_str() {
                        "two-level" => Policy::TwoLevel,
                        "multilevel" => Policy::Multilevel,
                        _ => return Err(format!("unknown policy {value}").into()),
                    });
                }
                "--reuse-mode" => {
                    reuse_mode = Some(match value.as_str() {
                        "fresh" => ReuseMode::Fresh,
                        "reuse" => ReuseMode::Reuse,
                        _ => return Err(format!("unknown reuse mode {value}").into()),
                    });
                }
                "--repeats" => {
                    let count: usize = value.parse()?;
                    if !(2..=9).contains(&count) {
                        return Err("repeats must be between 2 and 9".into());
                    }
                    repeats = Some(count);
                }
                "--tol" => {
                    tolerance = Some(match value.as_str() {
                        "1e-12" => 1e-12,
                        _ => return Err("A10B tolerance must be 1e-12".into()),
                    });
                }
                _ => return Err(format!("unknown argument {flag}").into()),
            }
        }
        let case_name = case_name.ok_or("missing --case")?;
        if !matches!(
            case_name.as_str(),
            "island_shell" | "multi_patch" | "rotated_anisotropy"
        ) {
            return Err("unknown FEM case".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("missing --matrix")?,
            case_name,
            aggregation: aggregation.ok_or("missing --aggregation")?,
            basis: basis.ok_or("missing --basis")?,
            policy: policy.ok_or("missing --policy")?,
            tolerance: tolerance.ok_or("missing --tol")?,
            reuse_mode: reuse_mode.ok_or("missing --reuse-mode")?,
            repeats: repeats.ok_or("missing --repeats")?,
        })
    }
}

fn checked_u32(value: usize) -> Result<u32, HybitError> {
    u32::try_from(value).map_err(|_| HybitError::SizeOverflow)
}

fn diag_and_omega(matrix: &Csr32Matrix) -> Result<(Vec<f64>, f64), HybitError> {
    let n = matrix.nrows();
    let mut inv = vec![0.0; n];
    let mut max_row_ratio = 0.0f64;
    for (row, inverse) in inv.iter_mut().enumerate() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let mut diagonal = 0.0;
        let mut abs_sum = 0.0;
        for k in start..end {
            let value = matrix.values()[k];
            abs_sum += value.abs();
            if matrix.col_idx()[k] as usize == row {
                diagonal += value;
            }
        }
        if !diagonal.is_finite() || diagonal <= 0.0 || !abs_sum.is_finite() {
            return Err(HybitError::InvalidMatrix(
                "AMG requires positive finite diagonal",
            ));
        }
        *inverse = diagonal.recip();
        max_row_ratio = max_row_ratio.max(abs_sum / diagonal);
    }
    if !max_row_ratio.is_finite() || max_row_ratio <= 0.0 {
        return Err(HybitError::InvalidMatrix(
            "invalid AMG Jacobi spectral upper bound",
        ));
    }
    // Gershgorin: lambda_max(D^-1 A) <= max_i sum_j |a_ij|/a_ii.
    // Hence omega < 2 / bound for the symmetric Richardson smoother.
    let omega = 0.67f64.min(1.6 / max_row_ratio);
    Ok((inv, omega))
}

fn check_symmetric(matrix: &Csr32Matrix) -> Result<(), HybitError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err(HybitError::InvalidMatrix(
            "AMG requires nonempty square SPD candidate",
        ));
    }
    for row in 0..matrix.nrows() {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for k in start..end {
            let col = matrix.col_idx()[k] as usize;
            let start_col = matrix.row_ptr()[col] as usize;
            let end_col = matrix.row_ptr()[col + 1] as usize;
            let mirrored = matrix.col_idx()[start_col..end_col]
                .binary_search(&checked_u32(row)?)
                .map_err(|_| HybitError::InvalidMatrix("AMG requires symmetric CSR structure"))?;
            let other = matrix.values()[start_col + mirrored];
            let value = matrix.values()[k];
            if (value - other).abs() > 1e-10 * (value.abs() + other.abs() + 1.0) {
                return Err(HybitError::InvalidMatrix(
                    "AMG requires symmetric matrix values",
                ));
            }
        }
    }
    Ok(())
}

fn aggregate(
    matrix: &Csr32Matrix,
    aggregation: TwoLevelAggregation,
    size: usize,
) -> Result<(Vec<usize>, usize), HybitError> {
    let n = matrix.nrows();
    if size < 2 || n < 2 {
        return Err(HybitError::InvalidArgument(
            "AMG aggregate size and dimension must exceed one",
        ));
    }
    if aggregation == TwoLevelAggregation::Contiguous {
        return Ok(((0..n).map(|row| row / size).collect(), n.div_ceil(size)));
    }
    let mut adjacency = vec![Vec::<usize>::new(); n];
    for row in 0..n {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let mut strongest = 0.0f64;
        for k in start..end {
            if matrix.col_idx()[k] as usize != row {
                strongest = strongest.max(matrix.values()[k].abs());
            }
        }
        for k in start..end {
            let col = matrix.col_idx()[k] as usize;
            if col == row {
                continue;
            }
            if aggregation == TwoLevelAggregation::Graph
                || matrix.values()[k].abs() >= STRENGTH_THRESHOLD * strongest
            {
                adjacency[row].push(col);
                adjacency[col].push(row);
            }
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut result = vec![usize::MAX; n];
    let mut count = 0usize;
    let mut queue = VecDeque::<usize>::new();
    for seed in 0..n {
        if result[seed] != usize::MAX {
            continue;
        }
        result[seed] = count;
        queue.push_back(seed);
        let mut members = 1;
        while members < size {
            let Some(row) = queue.pop_front() else {
                break;
            };
            for &neighbor in &adjacency[row] {
                if result[neighbor] == usize::MAX {
                    result[neighbor] = count;
                    members += 1;
                    if members < size {
                        queue.push_back(neighbor);
                    }
                    if members == size {
                        break;
                    }
                }
            }
        }
        queue.clear();
        count += 1;
    }
    Ok((result, count))
}

fn rows_to_csr(rows: Vec<Vec<(u32, f64)>>, ncols: usize) -> Result<Csr32Matrix, HybitError> {
    let nrows = rows.len();
    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut cols = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in rows {
        for (column, value) in row {
            if !value.is_finite() {
                return Err(HybitError::InvalidMatrix(
                    "AMG numeric product is nonfinite",
                ));
            }
            cols.push(column);
            values.push(value);
        }
        row_ptr.push(checked_u32(cols.len())?);
    }
    Csr32Matrix::new(nrows, ncols, row_ptr, cols, values)
}

fn prolongation(
    matrix: &Csr32Matrix,
    owners: &[usize],
    coarse_n: usize,
    basis: TwoLevelBasis,
    diag_inv: &[f64],
    omega: f64,
) -> Result<Csr32Matrix, HybitError> {
    let mut rows = Vec::with_capacity(matrix.nrows());
    for row in 0..matrix.nrows() {
        let mut weights = BTreeMap::<u32, f64>::new();
        weights.insert(checked_u32(owners[row])?, 1.0);
        if basis == TwoLevelBasis::JacobiSmoothed {
            let start = matrix.row_ptr()[row] as usize;
            let end = matrix.row_ptr()[row + 1] as usize;
            for k in start..end {
                let column = matrix.col_idx()[k] as usize;
                let owner = checked_u32(owners[column])?;
                *weights.entry(owner).or_default() -= omega * diag_inv[row] * matrix.values()[k];
            }
        }
        let sparse: Vec<_> = weights
            .into_iter()
            .filter(|(_, value)| *value != 0.0)
            .collect();
        if sparse.is_empty() {
            return Err(HybitError::InvalidMatrix(
                "AMG prolongation contains empty fine row",
            ));
        }
        rows.push(sparse);
    }
    rows_to_csr(rows, coarse_n)
}

fn multiply_ap(a: &Csr32Matrix, p: &Csr32Matrix) -> Result<Csr32Matrix, HybitError> {
    if a.ncols() != p.nrows() {
        return Err(HybitError::DimensionMismatch {
            expected: a.ncols(),
            actual: p.nrows(),
        });
    }
    let mut rows = Vec::with_capacity(a.nrows());
    for row in 0..a.nrows() {
        let mut sum = BTreeMap::<u32, f64>::new();
        for k in a.row_ptr()[row] as usize..a.row_ptr()[row + 1] as usize {
            let fine_col = a.col_idx()[k] as usize;
            let aij = a.values()[k];
            for t in p.row_ptr()[fine_col] as usize..p.row_ptr()[fine_col + 1] as usize {
                *sum.entry(p.col_idx()[t]).or_default() += aij * p.values()[t];
            }
        }
        rows.push(sum.into_iter().collect());
    }
    rows_to_csr(rows, p.ncols())
}

#[derive(Default)]
struct GalerkinStats {
    candidate_nnz: usize,
    exact_zero_candidates: usize,
    abtm_words: usize,
}

/// ABTM exact Boolean support for P^T (A P), followed by numeric accumulation.
/// Retains exact zero candidates: structural support is not a convergence gate.
fn galerkin(a: &Csr32Matrix, p: &Csr32Matrix) -> Result<(Csr32Matrix, GalerkinStats), HybitError> {
    let ap = multiply_ap(a, p)?;
    let topology = AbtmTopology::from_csr32(&ap)?;
    let coarse_n = p.ncols();
    let mut word_rows = vec![BTreeMap::<u32, u64>::new(); coarse_n];
    for fine in 0..p.nrows() {
        let source = topology.row(fine)?;
        for t in p.row_ptr()[fine] as usize..p.row_ptr()[fine + 1] as usize {
            let coarse_row = p.col_idx()[t] as usize;
            for word in source.words() {
                *word_rows[coarse_row].entry(word.word_index()).or_default() |= word.mask();
            }
        }
    }
    let mut columns = Vec::with_capacity(coarse_n);
    let mut numeric = Vec::with_capacity(coarse_n);
    let mut stats = GalerkinStats {
        abtm_words: topology.nonempty_words(),
        ..Default::default()
    };
    for row in word_rows {
        let mut cols = Vec::new();
        for (word_index, mut mask) in row {
            while mask != 0 {
                cols.push(checked_u32(
                    word_index as usize * 64 + mask.trailing_zeros() as usize,
                )?);
                mask &= mask - 1;
            }
        }
        stats.candidate_nnz += cols.len();
        numeric.push(vec![0.0; cols.len()]);
        columns.push(cols);
    }
    for fine in 0..p.nrows() {
        for t in p.row_ptr()[fine] as usize..p.row_ptr()[fine + 1] as usize {
            let coarse_row = p.col_idx()[t] as usize;
            let factor = p.values()[t];
            for k in ap.row_ptr()[fine] as usize..ap.row_ptr()[fine + 1] as usize {
                let coarse_col = ap.col_idx()[k];
                let pos = columns[coarse_row]
                    .binary_search(&coarse_col)
                    .map_err(|_| {
                        HybitError::InvalidMatrix("ABTM Galerkin candidate missing product")
                    })?;
                numeric[coarse_row][pos] += factor * ap.values()[k];
            }
        }
    }
    let mut row_ptr = vec![0_u32];
    let mut col_idx = Vec::with_capacity(stats.candidate_nnz);
    let mut values = Vec::with_capacity(stats.candidate_nnz);
    for (cols, vals) in columns.into_iter().zip(numeric) {
        col_idx.extend(cols);
        values.extend(vals);
        row_ptr.push(checked_u32(col_idx.len())?);
    }
    for row in 0..coarse_n {
        for k in row_ptr[row] as usize..row_ptr[row + 1] as usize {
            let col = col_idx[k] as usize;
            if col > row {
                let mate_base = row_ptr[col] as usize;
                let mate_end = row_ptr[col + 1] as usize;
                let mate = col_idx[mate_base..mate_end]
                    .binary_search(&checked_u32(row)?)
                    .map_err(|_| {
                        HybitError::InvalidMatrix("Galerkin symbolic symmetry mismatch")
                    })?;
                let mate = mate_base + mate;
                if (values[k] - values[mate]).abs()
                    > 1e-9 * (1.0 + values[k].abs() + values[mate].abs())
                {
                    return Err(HybitError::InvalidMatrix("Galerkin numerical asymmetry"));
                }
                let average = 0.5 * (values[k] + values[mate]);
                values[k] = average;
                values[mate] = average;
            }
        }
    }
    stats.exact_zero_candidates = values.iter().filter(|&&value| value == 0.0).count();
    let coarse = Csr32Matrix::new(coarse_n, coarse_n, row_ptr, col_idx, values)?;
    let _ = diag_and_omega(&coarse)?;
    Ok((coarse, stats))
}

struct DenseCholesky {
    n: usize,
    lower: Vec<f64>,
}

impl DenseCholesky {
    fn new(matrix: &Csr32Matrix) -> Result<Self, HybitError> {
        let n = matrix.nrows();
        let mut lower = vec![0.0; n * n];
        let mut dense = vec![0.0; n * n];
        for row in 0..n {
            for k in matrix.row_ptr()[row] as usize..matrix.row_ptr()[row + 1] as usize {
                dense[row * n + matrix.col_idx()[k] as usize] += matrix.values()[k];
            }
        }
        for i in 0..n {
            for j in 0..=i {
                let mut value = dense[i * n + j];
                for k in 0..j {
                    value -= lower[i * n + k] * lower[j * n + k];
                }
                lower[i * n + j] = if i == j {
                    if value <= 0.0 || !value.is_finite() {
                        return Err(HybitError::InvalidMatrix(
                            "AMG coarsest Cholesky is not SPD",
                        ));
                    }
                    value.sqrt()
                } else {
                    value / lower[j * n + j]
                };
            }
        }
        Ok(Self { n, lower })
    }

    fn solve(&self, rhs: &[f64], out: &mut [f64]) -> Result<(), HybitError> {
        if rhs.len() != self.n || out.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: rhs.len().min(out.len()),
            });
        }
        for i in 0..self.n {
            let mut sum = rhs[i];
            for (j, &value) in out.iter().take(i).enumerate() {
                sum -= self.lower[i * self.n + j] * value;
            }
            out[i] = sum / self.lower[i * self.n + i];
        }
        for i in (0..self.n).rev() {
            let mut sum = out[i];
            for (j, &value) in out.iter().enumerate().skip(i + 1) {
                sum -= self.lower[j * self.n + i] * value;
            }
            out[i] = sum / self.lower[i * self.n + i];
        }
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.lower.len() * std::mem::size_of::<f64>()
    }
}

struct Level {
    a: Csr32Matrix,
    p: Option<Csr32Matrix>,
    diag_inv: Vec<f64>,
    omega: f64,
}

struct CycleScratch {
    ax: Vec<f64>,
    coarse_rhs: Vec<f64>,
    coarse_x: Vec<f64>,
}

#[derive(Default)]
struct HierarchyStats {
    symbolic_nnz: usize,
    exact_zero_candidates: usize,
    abtm_words: usize,
}

struct MultiLevelAmg {
    levels: Vec<Level>,
    bottom: DenseCholesky,
    scratch: RefCell<Vec<CycleScratch>>,
    calls: Cell<usize>,
    stats: HierarchyStats,
}

impl MultiLevelAmg {
    fn new(
        matrix: Csr32Matrix,
        aggregation: TwoLevelAggregation,
        basis: TwoLevelBasis,
        coarse_stop: usize,
        aggregate_size: usize,
        max_levels: usize,
    ) -> Result<Self, HybitError> {
        if coarse_stop < 2 || aggregate_size < 2 || max_levels < 3 {
            return Err(HybitError::InvalidArgument(
                "invalid AMG hierarchy controls",
            ));
        }
        check_symmetric(&matrix)?;
        let mut current = matrix;
        let mut levels = Vec::new();
        let mut stats = HierarchyStats::default();
        while current.nrows() > coarse_stop && levels.len() + 1 < max_levels {
            let (diag_inv, omega) = diag_and_omega(&current)?;
            let (owners, coarse_n) = aggregate(&current, aggregation, aggregate_size)?;
            if coarse_n >= current.nrows() {
                return Err(HybitError::InvalidMatrix(
                    "AMG aggregation failed to reduce dimension",
                ));
            }
            let p = prolongation(&current, &owners, coarse_n, basis, &diag_inv, omega)?;
            let (coarse, gs) = galerkin(&current, &p)?;
            check_symmetric(&coarse)?;
            stats.symbolic_nnz += gs.candidate_nnz;
            stats.exact_zero_candidates += gs.exact_zero_candidates;
            stats.abtm_words += gs.abtm_words;
            levels.push(Level {
                a: current,
                p: Some(p),
                diag_inv,
                omega,
            });
            current = coarse;
        }
        if current.nrows() > coarse_stop || levels.len() < 2 {
            return Err(HybitError::InvalidMatrix(
                "AMG requires at least three valid levels and a bounded coarse factor",
            ));
        }
        let (last_diag, last_omega) = diag_and_omega(&current)?;
        let bottom = DenseCholesky::new(&current)?;
        levels.push(Level {
            a: current,
            p: None,
            diag_inv: last_diag,
            omega: last_omega,
        });
        let mut scratch = Vec::with_capacity(levels.len());
        for (i, level) in levels.iter().enumerate() {
            let next_n = levels.get(i + 1).map_or(0, |next| next.a.nrows());
            scratch.push(CycleScratch {
                ax: vec![0.0; level.a.nrows()],
                coarse_rhs: vec![0.0; next_n],
                coarse_x: vec![0.0; next_n],
            });
        }
        Ok(Self {
            levels,
            bottom,
            scratch: RefCell::new(scratch),
            calls: Cell::new(0),
            stats,
        })
    }

    fn level_dims(&self) -> Vec<usize> {
        self.levels.iter().map(|level| level.a.nrows()).collect()
    }

    fn level_nnz(&self) -> Vec<usize> {
        self.levels.iter().map(|level| level.a.nnz()).collect()
    }

    fn scratch_bytes(&self) -> usize {
        self.scratch
            .borrow()
            .iter()
            .map(|item| item.ax.len() + item.coarse_rhs.len() + item.coarse_x.len())
            .sum::<usize>()
            * std::mem::size_of::<f64>()
    }

    fn cycle(
        &self,
        depth: usize,
        rhs: &[f64],
        x: &mut [f64],
        work: &mut [CycleScratch],
    ) -> Result<(), HybitError> {
        let level = &self.levels[depth];
        if rhs.len() != level.a.nrows() || x.len() != level.a.nrows() {
            return Err(HybitError::DimensionMismatch {
                expected: level.a.nrows(),
                actual: rhs.len().min(x.len()),
            });
        }
        if depth + 1 == self.levels.len() {
            return self.bottom.solve(rhs, x);
        }
        let (current, remaining) = work
            .split_first_mut()
            .ok_or(HybitError::InvalidMatrix("missing AMG recursive scratch"))?;
        let p = level
            .p
            .as_ref()
            .ok_or(HybitError::InvalidMatrix("missing AMG prolongation"))?;
        for i in 0..rhs.len() {
            x[i] = level.omega * level.diag_inv[i] * rhs[i];
        }
        level.a.apply(x, &mut current.ax)?;
        current.coarse_rhs.fill(0.0);
        for (fine, &rhs_value) in rhs.iter().enumerate() {
            let residual = rhs_value - current.ax[fine];
            for t in p.row_ptr()[fine] as usize..p.row_ptr()[fine + 1] as usize {
                current.coarse_rhs[p.col_idx()[t] as usize] += p.values()[t] * residual;
            }
        }
        current.coarse_x.fill(0.0);
        self.cycle(
            depth + 1,
            &current.coarse_rhs,
            &mut current.coarse_x,
            remaining,
        )?;
        for (fine, value) in x.iter_mut().enumerate() {
            for t in p.row_ptr()[fine] as usize..p.row_ptr()[fine + 1] as usize {
                *value += p.values()[t] * current.coarse_x[p.col_idx()[t] as usize];
            }
        }
        level.a.apply(x, &mut current.ax)?;
        for (i, value) in x.iter_mut().enumerate() {
            *value += level.omega * level.diag_inv[i] * (rhs[i] - current.ax[i]);
        }
        Ok(())
    }
}

impl Preconditioner for MultiLevelAmg {
    fn len(&self) -> usize {
        self.levels[0].a.nrows()
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        let mut scratch = self.scratch.borrow_mut();
        self.cycle(0, r, z, &mut scratch[..])?;
        self.calls.set(self.calls.get() + 1);
        Ok(())
    }
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn true_relative(matrix: &Csr32Matrix, rhs: &[f64], x: &[f64]) -> Result<f64, HybitError> {
    let ax = matrix.spmv(x)?;
    let residual = rhs
        .iter()
        .zip(&ax)
        .map(|(b, value)| (b - value).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(residual / norm2(rhs).max(f64::MIN_POSITIVE))
}

/// Independent sequential CSR accumulation with Kahan correction per row.
/// This checks the residual measurement, not the exact-solution error.
fn compensated_true_relative(matrix: &Csr32Matrix, rhs: &[f64], x: &[f64]) -> f64 {
    let mut sum_squares = 0.0;
    for (row, &b) in rhs.iter().enumerate() {
        let mut sum = 0.0;
        let mut correction = 0.0;
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for k in start..end {
            let item = matrix.values()[k] * x[matrix.col_idx()[k] as usize];
            let adjusted = item - correction;
            let next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        sum_squares += (b - sum).powi(2);
    }
    sum_squares.sqrt() / norm2(rhs).max(f64::MIN_POSITIVE)
}

struct AccuracyDiagnostics {
    relative_x_error: String,
    error_response_relative: String,
    error_rayleigh: String,
    residual_amplification: String,
    compensated_true_relative: f64,
}

fn accuracy_diagnostics(
    matrix: &Csr32Matrix,
    rhs: &[f64],
    x: &[f64],
    exact: Option<&[f64]>,
    true_residual: f64,
) -> Result<AccuracyDiagnostics, HybitError> {
    let compensated = compensated_true_relative(matrix, rhs, x);
    let Some(exact) = exact else {
        return Ok(AccuracyDiagnostics {
            relative_x_error: "n/a".into(),
            error_response_relative: "n/a".into(),
            error_rayleigh: "n/a".into(),
            residual_amplification: "n/a".into(),
            compensated_true_relative: compensated,
        });
    };
    let difference: Vec<f64> = x.iter().zip(exact).map(|(a, b)| a - b).collect();
    let error_norm = norm2(&difference);
    let relative_error = error_norm / norm2(exact).max(f64::MIN_POSITIVE);
    let ae = matrix.spmv(&difference)?;
    let response_relative = norm2(&ae) / norm2(rhs).max(f64::MIN_POSITIVE);
    let rayleigh = if error_norm > 0.0 {
        difference
            .iter()
            .zip(&ae)
            .map(|(v, av)| v * av)
            .sum::<f64>()
            / error_norm.powi(2)
    } else {
        0.0
    };
    let amplification = relative_error / true_residual.max(f64::MIN_POSITIVE);
    Ok(AccuracyDiagnostics {
        relative_x_error: format!("{relative_error:.12e}"),
        error_response_relative: format!("{response_relative:.12e}"),
        error_rayleigh: format!("{rayleigh:.12e}"),
        residual_amplification: format!("{amplification:.12e}"),
        compensated_true_relative: compensated,
    })
}

#[derive(Clone, Copy)]
struct Certified {
    status: SolveStatus,
    iterations: usize,
    reported_relative: f64,
    verified_true_relative: f64,
    true_checks: usize,
    restarts: usize,
    rejected: usize,
}

fn solve_certified(
    matrix: &Csr32Matrix,
    preconditioner: &dyn Preconditioner,
    rhs: &[f64],
    x: &mut [f64],
    options: SolverOptions,
) -> Result<Certified, Box<dyn Error>> {
    let mut iterations = 0;
    let mut checks = 0;
    let mut restarts = 0;
    let mut rejected = 0;
    let mut reported = norm2(rhs);
    loop {
        let remaining = options.max_iterations - iterations;
        if remaining == 0 {
            return Ok(Certified {
                status: SolveStatus::MaxIterations,
                iterations,
                reported_relative: reported / norm2(rhs).max(f64::MIN_POSITIVE),
                verified_true_relative: true_relative(matrix, rhs, x)?,
                true_checks: checks,
                restarts,
                rejected,
            });
        }
        let mut segment = options;
        segment.max_iterations = remaining;
        let out = pcg(matrix, preconditioner, rhs, x, segment)?;
        iterations += out.iterations;
        reported = out.final_residual;
        checks += 1;
        let verified = true_relative(matrix, rhs, x)?;
        if !verified.is_finite() {
            return Err("nonfinite original-CSR true residual".into());
        }
        if out.status == SolveStatus::Converged && verified <= options.relative_tolerance {
            return Ok(Certified {
                status: SolveStatus::Converged,
                iterations,
                reported_relative: reported / norm2(rhs).max(f64::MIN_POSITIVE),
                verified_true_relative: verified,
                true_checks: checks,
                restarts,
                rejected,
            });
        }
        if out.status == SolveStatus::Converged {
            rejected += 1;
            if iterations < options.max_iterations && out.iterations > 0 {
                restarts += 1;
                continue;
            }
        }
        return Ok(Certified {
            status: if out.status == SolveStatus::Converged {
                if iterations == options.max_iterations {
                    SolveStatus::MaxIterations
                } else {
                    SolveStatus::Breakdown
                }
            } else {
                out.status
            },
            iterations,
            reported_relative: reported / norm2(rhs).max(f64::MIN_POSITIVE),
            verified_true_relative: verified,
            true_checks: checks,
            restarts,
            rejected,
        });
    }
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

enum Prepared {
    Control(Box<TwoLevelBlockJacobiPreconditioner>),
    Amg(MultiLevelAmg),
}

impl Prepared {
    fn preconditioner(&self) -> &dyn Preconditioner {
        match self {
            Self::Control(control) => control.as_ref(),
            Self::Amg(amg) => amg,
        }
    }

    fn level_dims(&self, n: usize) -> Vec<usize> {
        match self {
            Self::Control(control) => vec![n, control.coarse_dimension()],
            Self::Amg(amg) => amg.level_dims(),
        }
    }

    fn level_nnz(&self, fine_nnz: usize) -> Vec<usize> {
        match self {
            Self::Control(_) => vec![fine_nnz, 0],
            Self::Amg(amg) => amg.level_nnz(),
        }
    }

    fn symbolic_nnz(&self) -> usize {
        match self {
            Self::Control(_) => 0,
            Self::Amg(amg) => amg.stats.symbolic_nnz,
        }
    }

    fn exact_zero_candidates(&self) -> usize {
        match self {
            Self::Control(_) => 0,
            Self::Amg(amg) => amg.stats.exact_zero_candidates,
        }
    }

    fn scratch_bytes(&self) -> usize {
        match self {
            Self::Control(_) => 0,
            Self::Amg(amg) => amg.scratch_bytes(),
        }
    }

    fn abtm_words(&self) -> usize {
        match self {
            Self::Control(_) => 0,
            Self::Amg(amg) => amg.stats.abtm_words,
        }
    }

    fn factor_bytes(&self) -> usize {
        match self {
            Self::Control(control) => control.factor_bytes(),
            Self::Amg(amg) => amg.bottom.bytes(),
        }
    }

    fn calls(&self) -> usize {
        match self {
            Self::Control(_) => 0,
            Self::Amg(amg) => amg.calls.get(),
        }
    }
}

fn prepare(args: &Args, matrix: &Csr32Matrix) -> Result<Prepared, HybitError> {
    let n = matrix.nrows();
    match args.policy {
        Policy::TwoLevel => Ok(Prepared::Control(Box::new(
            TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_transfer_options(
                matrix,
                1,
                n.div_ceil(TARGET_COARSE).max(1),
                args.aggregation,
                args.basis,
                TwoLevelCoarseApplyPolicy::Auto,
                TwoLevelTransferOptions {
                    apply_policy: TwoLevelTransferApplyPolicy::Parallel,
                    storage_policy: TwoLevelTransferStoragePolicy::Wide,
                    value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
                },
            )?,
        ))),
        Policy::Multilevel => Ok(Prepared::Amg(MultiLevelAmg::new(
            matrix.clone(),
            args.aggregation,
            args.basis,
            TARGET_COARSE,
            AGGREGATE_SIZE,
            MAX_LEVELS,
        )?)),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let start = Instant::now();
    let (matrix, _) = read_matrix_market(&args.matrix)?;
    check_symmetric(&matrix)?;
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    let n = matrix.nrows();
    let rhs_cases = [
        RhsMode::Ones,
        RhsMode::IndexSmooth,
        RhsMode::SeededRandom,
        RhsMode::PointLoad,
    ]
    .into_iter()
    .map(|mode| -> Result<_, HybitError> {
        let (rhs, exact) = make_rhs(&matrix, mode)?;
        let hash = rhs_hash(&rhs);
        if !norm2(&rhs).is_finite() || norm2(&rhs) <= 0.0 {
            return Err(HybitError::InvalidArgument("invalid RHS for reuse trial"));
        }
        Ok((mode, rhs, exact, hash))
    })
    .collect::<Result<Vec<_>, _>>()?;
    let options = SolverOptions {
        relative_tolerance: args.tolerance,
        absolute_tolerance: 0.0,
        max_iterations: 1200,
    };

    // Reuse is constructed once before the unmeasured warmup; its setup
    // cost is counted exactly once in the first measured sample.
    let reuse_setup_start = Instant::now();
    let reusable = if args.reuse_mode == ReuseMode::Reuse {
        Some(prepare(&args, &matrix)?)
    } else {
        None
    };
    let reuse_setup_ms = if reusable.is_some() {
        reuse_setup_start.elapsed().as_secs_f64() * 1e3
    } else {
        0.0
    };
    println!(
        "C1H12A10B_SESSION|case={}|policy={:?}|aggregation={:?}|basis={:?}|reuse_mode={:?}|rows={}|nnz={}|repeats={}|rhs_count=4|tol={:.12e}|max_iterations={}|load_ms={load_ms:.6}",
        args.case_name,
        args.policy,
        args.aggregation,
        args.basis,
        args.reuse_mode,
        n,
        matrix.nnz(),
        args.repeats,
        args.tolerance,
        options.max_iterations,
    );
    let mut sum_setup = 0.0;
    let mut sum_solve = 0.0;
    let mut samples = 0;
    // A complete warmup batch is excluded from the measured solves.
    for pass in 0..=args.repeats {
        let warmup = pass == 0;
        for (index, (rhs_mode, rhs, exact, hash)) in rhs_cases.iter().enumerate() {
            let setup_start = Instant::now();
            let fresh = if args.reuse_mode == ReuseMode::Fresh {
                Some(prepare(&args, &matrix)?)
            } else {
                None
            };
            let fresh_setup_ms = if fresh.is_some() {
                setup_start.elapsed().as_secs_f64() * 1e3
            } else {
                0.0
            };
            let prepared = fresh
                .as_ref()
                .or(reusable.as_ref())
                .ok_or("missing prepared AMG")?;
            let mut x = vec![0.0; n]; // no warm-start or RHS cross-contamination
            let calls_before = prepared.calls();
            let solve_start = Instant::now();
            let out = solve_certified(&matrix, prepared.preconditioner(), rhs, &mut x, options)?;
            let solve_ms = solve_start.elapsed().as_secs_f64() * 1e3;
            let calls = prepared.calls() - calls_before;
            let diagnostics = accuracy_diagnostics(
                &matrix,
                rhs,
                &x,
                exact.as_deref(),
                out.verified_true_relative,
            )?;
            if !diagnostics.compensated_true_relative.is_finite() {
                return Err("nonfinite compensated true residual".into());
            }
            if warmup {
                continue;
            }
            let setup_ms = if args.reuse_mode == ReuseMode::Fresh {
                fresh_setup_ms
            } else if pass == 1 && index == 0 {
                reuse_setup_ms
            } else {
                0.0
            };
            sum_setup += setup_ms;
            sum_solve += solve_ms;
            samples += 1;
            let dims = prepared.level_dims(n);
            let nnzs = prepared.level_nnz(matrix.nnz());
            println!(
                "C1H12A10B|case={}|policy={:?}|aggregation={:?}|basis={:?}|reuse_mode={:?}|repeat={}|rhs_mode={:?}|rhs_hash={}|rows={}|nnz={}|levels={}|level_dims={}|level_nnz={}|coarse_factor_bytes={}|symbolic_nnz={}|exact_zero_candidates={}|abtm_words={}|scratch_bytes={}|status={:?}|iterations={}|tol={:.12e}|max_iterations={}|reported_relative={:.12e}|verified_true_relative={:.12e}|compensated_true_relative={:.12e}|relative_x_error={}|error_response_relative={}|error_rayleigh={}|residual_amplification={}|setup_ms={setup_ms:.6}|solve_ms={solve_ms:.6}|sample_total_ms={:.6}|true_checks={}|restarts={}|rejected_recursive_convergences={}|precond_calls={}",
                args.case_name,
                args.policy,
                args.aggregation,
                args.basis,
                args.reuse_mode,
                pass - 1,
                rhs_mode,
                hash,
                n,
                matrix.nnz(),
                dims.len(),
                join_usize(&dims),
                join_usize(&nnzs),
                prepared.factor_bytes(),
                prepared.symbolic_nnz(),
                prepared.exact_zero_candidates(),
                prepared.abtm_words(),
                prepared.scratch_bytes(),
                out.status,
                out.iterations,
                options.relative_tolerance,
                options.max_iterations,
                out.reported_relative,
                out.verified_true_relative,
                diagnostics.compensated_true_relative,
                diagnostics.relative_x_error,
                diagnostics.error_response_relative,
                diagnostics.error_rayleigh,
                diagnostics.residual_amplification,
                setup_ms + solve_ms,
                out.true_checks,
                out.restarts,
                out.rejected,
                calls,
            );
        }
    }
    println!(
        "C1H12A10B_SUMMARY|case={}|policy={:?}|aggregation={:?}|basis={:?}|reuse_mode={:?}|samples={samples}|setup_ms_total={sum_setup:.6}|solve_ms_total={sum_solve:.6}|measured_total_ms={:.6}",
        args.case_name,
        args.policy,
        args.aggregation,
        args.basis,
        args.reuse_mode,
        sum_setup + sum_solve,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(n: usize) -> Csr32Matrix {
        let mut rows = Vec::new();
        for row in 0..n {
            let mut cols = Vec::new();
            if row > 0 {
                cols.push((u32::try_from(row - 1).unwrap(), -1.0));
            }
            cols.push((u32::try_from(row).unwrap(), 2.2));
            if row + 1 < n {
                cols.push((u32::try_from(row + 1).unwrap(), -1.0));
            }
            rows.push(cols);
        }
        rows_to_csr(rows, n).unwrap()
    }

    #[test]
    fn reused_amg_matches_fresh_across_rhs_and_repeat() {
        let matrix = chain(144);
        let reused = MultiLevelAmg::new(
            matrix.clone(),
            TwoLevelAggregation::Graph,
            TwoLevelBasis::PiecewiseConstant,
            5,
            4,
            8,
        )
        .unwrap();
        let options = SolverOptions {
            relative_tolerance: 1e-12,
            absolute_tolerance: 0.0,
            max_iterations: 1200,
        };
        for _ in 0..2 {
            for mode in [
                RhsMode::Ones,
                RhsMode::IndexSmooth,
                RhsMode::SeededRandom,
                RhsMode::PointLoad,
            ] {
                let (rhs, _) = make_rhs(&matrix, mode).unwrap();
                let fresh = MultiLevelAmg::new(
                    matrix.clone(),
                    TwoLevelAggregation::Graph,
                    TwoLevelBasis::PiecewiseConstant,
                    5,
                    4,
                    8,
                )
                .unwrap();
                let mut fresh_x = vec![0.0; matrix.nrows()];
                let mut reused_x = vec![0.0; matrix.nrows()];
                let fresh_out =
                    solve_certified(&matrix, &fresh, &rhs, &mut fresh_x, options).unwrap();
                let reused_out =
                    solve_certified(&matrix, &reused, &rhs, &mut reused_x, options).unwrap();
                assert_eq!(fresh_out.status, reused_out.status);
                assert_eq!(fresh_out.iterations, reused_out.iterations);
                assert_eq!(fresh_x, reused_x);
                assert!(fresh_out.verified_true_relative <= 1e-12);
                assert!(reused_out.verified_true_relative <= 1e-12);
            }
        }
    }

    #[test]
    fn compensated_residual_and_error_rayleigh_detect_perturbed_solution() {
        let matrix = chain(72);
        let expected = vec![1.0; 72];
        let rhs = matrix.spmv(&expected).unwrap();
        let exact = accuracy_diagnostics(&matrix, &rhs, &expected, Some(&expected), 0.0).unwrap();
        assert!(exact.compensated_true_relative < 1e-14);
        assert_eq!(exact.relative_x_error.parse::<f64>().unwrap(), 0.0);

        let mut perturbed = expected.clone();
        perturbed[23] += 0.1;
        let true_residual = true_relative(&matrix, &rhs, &perturbed).unwrap();
        let metrics =
            accuracy_diagnostics(&matrix, &rhs, &perturbed, Some(&expected), true_residual)
                .unwrap();
        assert!((metrics.compensated_true_relative - true_residual).abs() < 1e-12);
        assert!(metrics.error_rayleigh.parse::<f64>().unwrap() > 0.0);
        assert!(metrics.error_response_relative.parse::<f64>().unwrap() > 0.0);
    }

    #[test]
    fn rhs_modes_are_deterministic_and_separated() {
        let matrix = chain(129);
        let modes = [
            RhsMode::Ones,
            RhsMode::IndexSmooth,
            RhsMode::SeededRandom,
            RhsMode::PointLoad,
        ];
        let mut fingerprints = std::collections::BTreeSet::new();
        for mode in modes {
            let (b, truth) = make_rhs(&matrix, mode).unwrap();
            let (again, again_truth) = make_rhs(&matrix, mode).unwrap();
            assert_eq!(b, again);
            assert_eq!(truth, again_truth);
            assert!(fingerprints.insert(rhs_hash(&b)));
            assert!(norm2(&b) > 0.0);
            if let Some(exact) = truth {
                assert!(true_relative(&matrix, &b, &exact).unwrap() < 1e-14);
                if mode != RhsMode::Ones {
                    assert!(exact.iter().any(|value| (value - 1.0).abs() > 0.01));
                }
            } else {
                assert_eq!(mode, RhsMode::PointLoad);
                assert_eq!(b.iter().filter(|value| **value != 0.0).count(), 1);
            }
        }
    }

    #[test]
    fn abtm_numeric_galerkin_matches_quadratic_form_across_word_boundary() {
        let a = chain(260);
        let (d, omega) = diag_and_omega(&a).unwrap();
        let (owners, count) = aggregate(&a, TwoLevelAggregation::Contiguous, 4).unwrap();
        assert_eq!(count, 65);
        for basis in [
            TwoLevelBasis::PiecewiseConstant,
            TwoLevelBasis::JacobiSmoothed,
        ] {
            let p = prolongation(&a, &owners, count, basis, &d, omega).unwrap();
            let (ac, stats) = galerkin(&a, &p).unwrap();
            assert!(stats.abtm_words > 0);
            check_symmetric(&ac).unwrap();
            let coarse_vec: Vec<_> = (0..count).map(|i| (i % 7) as f64 - 3.0).collect();
            let fine_vec = p.spmv(&coarse_vec).unwrap();
            let acx = ac.spmv(&coarse_vec).unwrap();
            let afx = a.spmv(&fine_vec).unwrap();
            let coarse_energy: f64 = coarse_vec.iter().zip(&acx).map(|(x, ax)| x * ax).sum();
            let fine_energy: f64 = fine_vec.iter().zip(&afx).map(|(x, ax)| x * ax).sum();
            assert!((coarse_energy - fine_energy).abs() < 1e-10 * fine_energy.abs().max(1.0));
        }
    }

    #[test]
    fn native_multilevel_is_symmetric_positive_and_pcg_certified() {
        let a = chain(144);
        let mg = MultiLevelAmg::new(
            a.clone(),
            TwoLevelAggregation::Graph,
            TwoLevelBasis::JacobiSmoothed,
            5,
            4,
            8,
        )
        .unwrap();
        assert!(mg.levels.len() >= 3);
        let u: Vec<_> = (0..144).map(|i| ((i * 7) % 17) as f64 - 8.0).collect();
        let v: Vec<_> = (0..144).map(|i| ((i * 3) % 13) as f64 - 6.0).collect();
        let mut mu = vec![0.0; 144];
        let mut mv = vec![0.0; 144];
        mg.apply(&u, &mut mu).unwrap();
        mg.apply(&v, &mut mv).unwrap();
        let positivity: f64 = u.iter().zip(&mu).map(|(x, z)| x * z).sum();
        let symmetry1: f64 = u.iter().zip(&mv).map(|(x, z)| x * z).sum();
        let symmetry2: f64 = v.iter().zip(&mu).map(|(x, z)| x * z).sum();
        assert!(positivity > 0.0);
        assert!((symmetry1 - symmetry2).abs() < 1e-10 * symmetry1.abs().max(1.0));
        let b = a.spmv(&vec![1.0; 144]).unwrap();
        let mut x = vec![0.0; 144];
        let outcome = solve_certified(
            &a,
            &mg,
            &b,
            &mut x,
            SolverOptions {
                relative_tolerance: 1e-10,
                absolute_tolerance: 0.0,
                max_iterations: 100,
            },
        )
        .unwrap();
        assert_eq!(outcome.status, SolveStatus::Converged);
        assert!(outcome.verified_true_relative <= 1e-10);
    }

    #[test]
    fn invalid_hierarchy_and_non_spd_are_not_silent_fallbacks() {
        let a = chain(16);
        assert!(
            MultiLevelAmg::new(
                a.clone(),
                TwoLevelAggregation::Graph,
                TwoLevelBasis::PiecewiseConstant,
                2,
                4,
                2,
            )
            .is_err()
        );
        let zero = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![0.0, 1.0]).unwrap();
        assert!(
            MultiLevelAmg::new(
                zero,
                TwoLevelAggregation::Graph,
                TwoLevelBasis::PiecewiseConstant,
                2,
                4,
                4,
            )
            .is_err()
        );
    }
}
