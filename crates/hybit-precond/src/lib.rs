use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use hybit_core::{HybitError, LinearOperator, Preconditioner};
use hybit_matrix::Csr32Matrix;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct IdentityPreconditioner {
    n: usize,
}

impl IdentityPreconditioner {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Preconditioner for IdentityPreconditioner {
    fn len(&self) -> usize {
        self.n
    }
    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        if z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: z.len(),
            });
        }
        z.copy_from_slice(r);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct JacobiPreconditioner {
    inv_diag: Vec<f64>,
}

impl JacobiPreconditioner {
    pub fn from_csr32(matrix: &Csr32Matrix) -> Result<Self, HybitError> {
        let diagonal = matrix.diagonal()?;
        let mut inv_diag = Vec::with_capacity(diagonal.len());
        for (row, d) in diagonal.into_iter().enumerate() {
            if d == 0.0 || !d.is_finite() {
                return Err(HybitError::ZeroDiagonal { row });
            }
            if d < 0.0 {
                return Err(HybitError::InvalidMatrix(
                    "Jacobi-PCG requires a positive diagonal",
                ));
            }
            inv_diag.push(1.0 / d);
        }
        Ok(Self { inv_diag })
    }

    pub fn inv_diagonal(&self) -> &[f64] {
        &self.inv_diag
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn len(&self) -> usize {
        self.inv_diag.len()
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.inv_diag.len() {
            return Err(HybitError::DimensionMismatch {
                expected: self.inv_diag.len(),
                actual: r.len(),
            });
        }
        if z.len() != self.inv_diag.len() {
            return Err(HybitError::DimensionMismatch {
                expected: self.inv_diag.len(),
                actual: z.len(),
            });
        }
        for ((out, &ri), &d) in z.iter_mut().zip(r).zip(&self.inv_diag) {
            *out = d * ri;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct BlockJacobiFactor {
    start: usize,
    size: usize,
    lower: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct BlockJacobiPreconditioner {
    n: usize,
    block_size: usize,
    blocks: Vec<BlockJacobiFactor>,
    factor_bytes: usize,
}

impl BlockJacobiPreconditioner {
    /// Build a contiguous block-Jacobi preconditioner from principal diagonal blocks.
    ///
    /// This is especially useful for vector-valued FEM systems whose DOFs are
    /// stored contiguously per node (for example, x/y/z displacement blocks of 3).
    /// Every block is factorized with a dense Cholesky factorization, preserving
    /// the SPD requirement of PCG. A final short block is allowed when `n` is not
    /// an exact multiple of `block_size`.
    pub fn from_csr32(matrix: &Csr32Matrix, block_size: usize) -> Result<Self, HybitError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(HybitError::InvalidMatrix(
                "block Jacobi requires a square matrix",
            ));
        }
        if block_size == 0 {
            return Err(HybitError::InvalidArgument(
                "block Jacobi block_size must be > 0",
            ));
        }

        let n = matrix.nrows();
        let mut blocks = Vec::with_capacity(n.div_ceil(block_size));
        let mut factor_bytes = 0usize;

        for start in (0..n).step_by(block_size) {
            let size = block_size.min(n - start);
            let end_block = start + size;
            let mut dense = vec![0.0f64; size * size];

            for local_row in 0..size {
                let global_row = start + local_row;
                let rs = matrix.row_ptr()[global_row] as usize;
                let re = matrix.row_ptr()[global_row + 1] as usize;
                for p in rs..re {
                    let global_col = matrix.col_idx()[p] as usize;
                    if global_col >= start && global_col < end_block {
                        dense[local_row * size + (global_col - start)] += matrix.values()[p];
                    }
                }
            }

            let mut scale = 0.0f64;
            for &v in &dense {
                scale = scale.max(v.abs());
            }
            let symmetry_tol = 1.0e-11 * scale.max(1.0);
            for i in 0..size {
                for j in 0..i {
                    if (dense[i * size + j] - dense[j * size + i]).abs() > symmetry_tol {
                        return Err(HybitError::InvalidMatrix(
                            "block Jacobi diagonal block is not symmetric",
                        ));
                    }
                }
            }

            let mut lower = vec![0.0f64; size * size];
            let pivot_tol = 1.0e-14 * scale.max(1.0);
            for i in 0..size {
                for j in 0..=i {
                    let mut sum = dense[i * size + j];
                    for k in 0..j {
                        sum -= lower[i * size + k] * lower[j * size + k];
                    }
                    if i == j {
                        if !sum.is_finite() || sum <= pivot_tol {
                            return Err(HybitError::NumericalBreakdown(
                                "block Jacobi Cholesky encountered a non-positive pivot",
                            ));
                        }
                        lower[i * size + i] = sum.sqrt();
                    } else {
                        lower[i * size + j] = sum / lower[j * size + j];
                    }
                }
            }

            factor_bytes = factor_bytes
                .checked_add(lower.len() * std::mem::size_of::<f64>())
                .ok_or(HybitError::SizeOverflow)?;
            blocks.push(BlockJacobiFactor { start, size, lower });
        }

        Ok(Self {
            n,
            block_size,
            blocks,
            factor_bytes,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }

    /// Apply independent diagonal blocks in parallel. This is kept separate
    /// from the default trait implementation so existing callers retain the
    /// low-overhead serial path for small systems.
    #[doc(hidden)]
    pub fn apply_parallel(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        if z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: z.len(),
            });
        }

        let blocks = &self.blocks;
        z.par_chunks_mut(self.block_size)
            .enumerate()
            .for_each(|(block_index, z_block)| {
                let block = &blocks[block_index];
                let start = block.start;
                let n = block.size;
                debug_assert_eq!(z_block.len(), n);
                for i in 0..n {
                    let mut sum = r[start + i];
                    for (k, &zk) in z_block.iter().take(i).enumerate() {
                        sum -= block.lower[i * n + k] * zk;
                    }
                    z_block[i] = sum / block.lower[i * n + i];
                }
                for i in (0..n).rev() {
                    let mut sum = z_block[i];
                    for (k, &zk) in z_block.iter().enumerate().skip(i + 1) {
                        sum -= block.lower[k * n + i] * zk;
                    }
                    z_block[i] = sum / block.lower[i * n + i];
                }
            });
        Ok(())
    }
}

impl Preconditioner for BlockJacobiPreconditioner {
    fn len(&self) -> usize {
        self.n
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        if z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: z.len(),
            });
        }

        for block in &self.blocks {
            let start = block.start;
            let n = block.size;
            for i in 0..n {
                let mut sum = r[start + i];
                for k in 0..i {
                    sum -= block.lower[i * n + k] * z[start + k];
                }
                z[start + i] = sum / block.lower[i * n + i];
            }
            for i in (0..n).rev() {
                let mut sum = z[start + i];
                for k in (i + 1)..n {
                    sum -= block.lower[k * n + i] * z[start + k];
                }
                z[start + i] = sum / block.lower[i * n + i];
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct CoarseScratch {
    rhs: Vec<f64>,
    sol: Vec<f64>,
    restriction_buffers: Vec<Vec<f64>>,
}

#[inline]
fn packed_lower_len(n: usize) -> Result<usize, HybitError> {
    n.checked_add(1)
        .and_then(|np1| n.checked_mul(np1))
        .map(|v| v / 2)
        .ok_or(HybitError::SizeOverflow)
}

#[inline]
fn packed_lower_index(row: usize, col: usize) -> usize {
    debug_assert!(col <= row);
    row * (row + 1) / 2 + col
}

/// Empirical coarse-dimension crossover used by [`TwoLevelCoarseApplyPolicy::Auto`].
///
/// The L-angle crossover benchmark found packed factor solves slightly faster at
/// coarse dimension 768, while explicit inverse application was clearly faster
/// by coarse dimension 1275 and above. 1024 is therefore a conservative midpoint
/// threshold; explicit `FactorSolve` / `ExplicitInverse` selections remain
/// available for callers with different workloads.
pub const EXPLICIT_INVERSE_AUTO_MIN_COARSE_DIMENSION: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoLevelCoarseApplyPolicy {
    /// Resolve from the actual coarse dimension after aggregation.
    ///
    /// Dimensions below [`EXPLICIT_INVERSE_AUTO_MIN_COARSE_DIMENSION`] use
    /// `FactorSolve`; dimensions at or above it use `ExplicitInverse`.
    Auto,
    /// Apply the coarse inverse with packed forward/backward Cholesky solves.
    FactorSolve,
    /// Form the dense coarse inverse once during setup, then apply it as a
    /// row-major dense matrix-vector product. This increases setup work but
    /// removes triangular dependencies from every Krylov iteration.
    ExplicitInverse,
}

impl TwoLevelCoarseApplyPolicy {
    /// Resolve `Auto` against the actual coarse dimension.
    pub fn resolve(self, coarse_dimension: usize) -> Self {
        match self {
            Self::Auto if coarse_dimension >= EXPLICIT_INVERSE_AUTO_MIN_COARSE_DIMENSION => {
                Self::ExplicitInverse
            }
            Self::Auto => Self::FactorSolve,
            explicit => explicit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoLevelAggregation {
    /// Group consecutive node-major blocks. This is the historical generic path.
    Contiguous,
    /// Build deterministic breadth-first aggregates from the block sparsity graph.
    Graph,
    /// Follow the block sparsity graph but prioritize normalized strong couplings
    /// while growing each deterministic breadth-first aggregate.
    StrongGraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoLevelBasis {
    /// One piecewise-constant coarse mode per aggregate and component.
    PiecewiseConstant,
    /// Apply one damped Jacobi smoothing step to the tentative piecewise-constant
    /// prolongator, then form the true Galerkin operator `P^T A P`.
    JacobiSmoothed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoLevelTransferApplyPolicy {
    /// Apply the sparse smoothed transfer with serial fine-row traversal.
    Serial,
    /// Parallelize smoothed restriction with per-worker coarse buffers and
    /// prolongation with independent fine-row updates.
    Parallel,
}

#[derive(Clone, Debug)]
pub struct TwoLevelBlockJacobiPreconditioner {
    n: usize,
    dofs_per_node: usize,
    aggregate_nodes: usize,
    aggregate_count: usize,
    min_aggregate_nodes: usize,
    max_aggregate_nodes: usize,
    aggregation: TwoLevelAggregation,
    basis: TwoLevelBasis,
    transfer_apply_policy: TwoLevelTransferApplyPolicy,
    aggregate_of_node: Vec<usize>,
    coarse_dimension: usize,
    transfer_row_ptr: Vec<usize>,
    transfer_col_idx: Vec<u32>,
    transfer_values: Vec<f64>,
    smoothing_omega: f64,
    base: BlockJacobiPreconditioner,
    coarse_apply_policy: TwoLevelCoarseApplyPolicy,
    // FactorSolve keeps both triangular orientations in packed row-major form.
    // ExplicitInverse discards these setup factors after building E^-1 and
    // stores only the dense row-major inverse, so persistent memory remains
    // approximately n_coarse^2 f64 values in either mode.
    coarse_lower_packed: Vec<f64>,
    coarse_upper_packed: Vec<f64>,
    coarse_upper_row_start: Vec<usize>,
    coarse_inverse: Vec<f64>,
    scratch: RefCell<CoarseScratch>,
    factor_bytes: usize,
}

impl TwoLevelBlockJacobiPreconditioner {
    /// Build an SPD additive two-level preconditioner
    ///
    /// ```text
    /// M^-1 = B^-1 + P (P^T A P)^-1 P^T
    /// ```
    ///
    /// where `B^-1` is contiguous block Jacobi and `P` is the selected coarse
    /// transfer. `PiecewiseConstant` uses the tentative aggregate modes directly;
    /// `JacobiSmoothed` applies one damped Jacobi step and forms the true
    /// Galerkin operator `P^T A P`. Aggregates remain geometry-free.
    pub fn from_csr32(
        matrix: &Csr32Matrix,
        dofs_per_node: usize,
        aggregate_nodes: usize,
    ) -> Result<Self, HybitError> {
        Self::from_csr32_with_aggregation_basis_and_policy(
            matrix,
            dofs_per_node,
            aggregate_nodes,
            TwoLevelAggregation::Contiguous,
            TwoLevelBasis::PiecewiseConstant,
            TwoLevelCoarseApplyPolicy::FactorSolve,
        )
    }

    pub fn from_csr32_with_policy(
        matrix: &Csr32Matrix,
        dofs_per_node: usize,
        aggregate_nodes: usize,
        coarse_apply_policy: TwoLevelCoarseApplyPolicy,
    ) -> Result<Self, HybitError> {
        Self::from_csr32_with_aggregation_basis_and_policy(
            matrix,
            dofs_per_node,
            aggregate_nodes,
            TwoLevelAggregation::Contiguous,
            TwoLevelBasis::PiecewiseConstant,
            coarse_apply_policy,
        )
    }

    pub fn from_csr32_graph_with_policy(
        matrix: &Csr32Matrix,
        dofs_per_node: usize,
        target_aggregate_nodes: usize,
        coarse_apply_policy: TwoLevelCoarseApplyPolicy,
    ) -> Result<Self, HybitError> {
        Self::from_csr32_with_aggregation_basis_and_policy(
            matrix,
            dofs_per_node,
            target_aggregate_nodes,
            TwoLevelAggregation::Graph,
            TwoLevelBasis::PiecewiseConstant,
            coarse_apply_policy,
        )
    }

    pub fn from_csr32_with_aggregation_and_policy(
        matrix: &Csr32Matrix,
        dofs_per_node: usize,
        aggregate_nodes: usize,
        aggregation: TwoLevelAggregation,
        coarse_apply_policy: TwoLevelCoarseApplyPolicy,
    ) -> Result<Self, HybitError> {
        Self::from_csr32_with_aggregation_basis_and_policy(
            matrix,
            dofs_per_node,
            aggregate_nodes,
            aggregation,
            TwoLevelBasis::PiecewiseConstant,
            coarse_apply_policy,
        )
    }

    pub fn from_csr32_with_aggregation_basis_and_policy(
        matrix: &Csr32Matrix,
        dofs_per_node: usize,
        aggregate_nodes: usize,
        aggregation: TwoLevelAggregation,
        basis: TwoLevelBasis,
        coarse_apply_policy: TwoLevelCoarseApplyPolicy,
    ) -> Result<Self, HybitError> {
        Self::from_csr32_with_aggregation_basis_and_policies(
            matrix,
            dofs_per_node,
            aggregate_nodes,
            aggregation,
            basis,
            coarse_apply_policy,
            TwoLevelTransferApplyPolicy::Serial,
        )
    }

    pub fn from_csr32_with_aggregation_basis_and_policies(
        matrix: &Csr32Matrix,
        dofs_per_node: usize,
        aggregate_nodes: usize,
        aggregation: TwoLevelAggregation,
        basis: TwoLevelBasis,
        coarse_apply_policy: TwoLevelCoarseApplyPolicy,
        transfer_apply_policy: TwoLevelTransferApplyPolicy,
    ) -> Result<Self, HybitError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(HybitError::InvalidMatrix(
                "two-level block Jacobi requires a square matrix",
            ));
        }
        if dofs_per_node == 0 {
            return Err(HybitError::InvalidArgument("dofs_per_node must be > 0"));
        }
        if aggregate_nodes == 0 {
            return Err(HybitError::InvalidArgument("aggregate_nodes must be > 0"));
        }
        let n = matrix.nrows();
        if n % dofs_per_node != 0 {
            return Err(HybitError::InvalidArgument(
                "matrix dimension must be divisible by dofs_per_node for aggregation",
            ));
        }

        let base = BlockJacobiPreconditioner::from_csr32(matrix, dofs_per_node)?;
        let node_count = n / dofs_per_node;
        let (aggregate_of_node, aggregate_count) = match aggregation {
            TwoLevelAggregation::Contiguous => {
                let count = node_count
                    .checked_add(aggregate_nodes - 1)
                    .ok_or(HybitError::SizeOverflow)?
                    / aggregate_nodes;
                (Vec::new(), count)
            }
            TwoLevelAggregation::Graph => {
                let adjacency = build_block_node_adjacency(matrix, node_count, dofs_per_node)?;
                build_piecewise_constant_graph_aggregates(&adjacency, aggregate_nodes)?
            }
            TwoLevelAggregation::StrongGraph => {
                let adjacency =
                    build_strong_block_node_adjacency(matrix, node_count, dofs_per_node)?;
                build_piecewise_constant_graph_aggregates(&adjacency, aggregate_nodes)?
            }
        };
        let mut aggregate_counts = vec![0usize; aggregate_count];
        if aggregate_of_node.is_empty() {
            for node in 0..node_count {
                aggregate_counts[node / aggregate_nodes] += 1;
            }
        } else {
            for &aggregate in &aggregate_of_node {
                aggregate_counts[aggregate] += 1;
            }
        }
        let min_aggregate_nodes = *aggregate_counts.iter().min().unwrap_or(&0);
        let max_aggregate_nodes = *aggregate_counts.iter().max().unwrap_or(&0);
        let coarse_dimension = aggregate_count
            .checked_mul(dofs_per_node)
            .ok_or(HybitError::SizeOverflow)?;
        let coarse_apply_policy = coarse_apply_policy.resolve(coarse_dimension);
        let coarse_len = coarse_dimension
            .checked_mul(coarse_dimension)
            .ok_or(HybitError::SizeOverflow)?;

        let (transfer_row_ptr, transfer_col_idx, transfer_values, smoothing_omega, mut coarse) =
            match basis {
                TwoLevelBasis::PiecewiseConstant => {
                    let mut coarse = vec![0.0f64; coarse_len];
                    // Galerkin coarse operator E = Z^T A Z. Since each fine DOF
                    // belongs to exactly one tentative piecewise-constant mode,
                    // this is a direct sparse accumulation into the dense coarse
                    // matrix.
                    for row in 0..n {
                        let cr = piecewise_coarse_index(
                            row,
                            dofs_per_node,
                            aggregate_nodes,
                            &aggregate_of_node,
                        );
                        let rs = matrix.row_ptr()[row] as usize;
                        let re = matrix.row_ptr()[row + 1] as usize;
                        for p in rs..re {
                            let col = matrix.col_idx()[p] as usize;
                            let cc = piecewise_coarse_index(
                                col,
                                dofs_per_node,
                                aggregate_nodes,
                                &aggregate_of_node,
                            );
                            coarse[cr * coarse_dimension + cc] += matrix.values()[p];
                        }
                    }
                    (Vec::new(), Vec::new(), Vec::new(), 0.0, coarse)
                }
                TwoLevelBasis::JacobiSmoothed => {
                    let transfer = build_jacobi_smoothed_transfer(
                        matrix,
                        dofs_per_node,
                        aggregate_nodes,
                        &aggregate_of_node,
                        coarse_dimension,
                    )?;
                    let coarse = build_galerkin_coarse_from_transfer(
                        matrix,
                        &transfer.row_ptr,
                        &transfer.col_idx,
                        &transfer.values,
                        coarse_dimension,
                    )?;
                    (
                        transfer.row_ptr,
                        transfer.col_idx,
                        transfer.values,
                        transfer.omega,
                        coarse,
                    )
                }
            };

        // Exact input symmetry can accumulate in a different order on the two
        // halves. Symmetrize before Cholesky so roundoff cannot create a false
        // asymmetry in the coarse SPD operator.
        for i in 0..coarse_dimension {
            for j in 0..i {
                let avg =
                    0.5 * (coarse[i * coarse_dimension + j] + coarse[j * coarse_dimension + i]);
                coarse[i * coarse_dimension + j] = avg;
                coarse[j * coarse_dimension + i] = avg;
            }
        }

        let mut scale = 0.0f64;
        for &v in &coarse {
            scale = scale.max(v.abs());
        }
        if !scale.is_finite() || scale == 0.0 {
            return Err(HybitError::NumericalBreakdown(
                "aggregation coarse operator is zero or non-finite",
            ));
        }

        let coarse_factor_len = packed_lower_len(coarse_dimension)?;
        let mut coarse_lower_packed = vec![0.0f64; coarse_factor_len];
        let pivot_tol = 1.0e-14 * scale.max(1.0);
        for i in 0..coarse_dimension {
            let i_base = packed_lower_index(i, 0);
            for j in 0..=i {
                let j_base = packed_lower_index(j, 0);
                let mut sum = coarse[i * coarse_dimension + j];
                for k in 0..j {
                    sum -= coarse_lower_packed[i_base + k] * coarse_lower_packed[j_base + k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= pivot_tol {
                        return Err(HybitError::NumericalBreakdown(
                            "aggregation coarse Cholesky encountered a non-positive pivot",
                        ));
                    }
                    coarse_lower_packed[i_base + i] = sum.sqrt();
                } else {
                    coarse_lower_packed[i_base + j] = sum / coarse_lower_packed[j_base + j];
                }
            }
        }

        // Build a packed row-major copy of L^T.  Row i stores
        // [L(i,i), L(i+1,i), ..., L(n-1,i)].  The backward solve can then
        // stream through a contiguous row instead of reading L with stride n.
        let mut coarse_upper_row_start = Vec::with_capacity(coarse_dimension + 1);
        coarse_upper_row_start.push(0usize);
        for i in 0..coarse_dimension {
            let next = coarse_upper_row_start[i]
                .checked_add(coarse_dimension - i)
                .ok_or(HybitError::SizeOverflow)?;
            coarse_upper_row_start.push(next);
        }
        debug_assert_eq!(coarse_upper_row_start[coarse_dimension], coarse_factor_len);
        let mut coarse_upper_packed = vec![0.0f64; coarse_factor_len];
        for i in 0..coarse_dimension {
            let dst = coarse_upper_row_start[i];
            for k in i..coarse_dimension {
                coarse_upper_packed[dst + (k - i)] = coarse_lower_packed[packed_lower_index(k, i)];
            }
        }

        let mut coarse_inverse = Vec::new();
        if coarse_apply_policy == TwoLevelCoarseApplyPolicy::ExplicitInverse {
            coarse_inverse = vec![0.0f64; coarse_len];
            // Each solve A x = e_i produces column i of E^-1. Since E^-1 is
            // symmetric, write that vector directly as row i; this layout lets
            // independent inverse rows be built in parallel without strided
            // concurrent writes. A final symmetrization removes roundoff skew.
            coarse_inverse
                .par_chunks_mut(coarse_dimension)
                .enumerate()
                .for_each_init(
                    || {
                        (
                            vec![0.0f64; coarse_dimension],
                            vec![0.0f64; coarse_dimension],
                        )
                    },
                    |(rhs, sol), (row_index, inverse_row)| {
                        rhs.fill(0.0);
                        rhs[row_index] = 1.0;
                        Self::solve_packed_coarse(
                            coarse_dimension,
                            &coarse_lower_packed,
                            &coarse_upper_packed,
                            &coarse_upper_row_start,
                            rhs,
                            sol,
                        );
                        inverse_row.copy_from_slice(sol);
                    },
                );

            // The mathematical inverse is symmetric. Average mirrored entries
            // so floating-point triangular-solve roundoff cannot make the
            // explicit apply observably asymmetric to PCG.
            for i in 0..coarse_dimension {
                for j in 0..i {
                    let avg = 0.5
                        * (coarse_inverse[i * coarse_dimension + j]
                            + coarse_inverse[j * coarse_dimension + i]);
                    coarse_inverse[i * coarse_dimension + j] = avg;
                    coarse_inverse[j * coarse_dimension + i] = avg;
                }
            }

            // The inverse replaces the setup factors in persistent state.
            coarse_lower_packed = Vec::new();
            coarse_upper_packed = Vec::new();
            coarse_upper_row_start = Vec::new();
        }

        let restriction_buffer_count = match (basis, transfer_apply_policy) {
            (TwoLevelBasis::JacobiSmoothed, TwoLevelTransferApplyPolicy::Parallel) => {
                rayon::current_num_threads().max(1)
            }
            _ => 0,
        };
        let scratch = RefCell::new(CoarseScratch {
            rhs: vec![0.0; coarse_dimension],
            sol: vec![0.0; coarse_dimension],
            restriction_buffers: (0..restriction_buffer_count)
                .map(|_| vec![0.0; coarse_dimension])
                .collect(),
        });
        let factor_bytes = base
            .factor_bytes()
            .checked_add(coarse_lower_packed.len() * std::mem::size_of::<f64>())
            .and_then(|v| v.checked_add(coarse_upper_packed.len() * std::mem::size_of::<f64>()))
            .and_then(|v| {
                v.checked_add(coarse_upper_row_start.len() * std::mem::size_of::<usize>())
            })
            .and_then(|v| v.checked_add(coarse_inverse.len() * std::mem::size_of::<f64>()))
            .and_then(|v| v.checked_add(aggregate_of_node.len() * std::mem::size_of::<usize>()))
            .and_then(|v| v.checked_add(transfer_row_ptr.len() * std::mem::size_of::<usize>()))
            .and_then(|v| v.checked_add(transfer_col_idx.len() * std::mem::size_of::<u32>()))
            .and_then(|v| v.checked_add(transfer_values.len() * std::mem::size_of::<f64>()))
            .and_then(|v| v.checked_add(2 * coarse_dimension * std::mem::size_of::<f64>()))
            .and_then(|v| {
                let restriction_bytes = restriction_buffer_count
                    .checked_mul(coarse_dimension)?
                    .checked_mul(std::mem::size_of::<f64>())?;
                v.checked_add(restriction_bytes)
            })
            .ok_or(HybitError::SizeOverflow)?;

        Ok(Self {
            n,
            dofs_per_node,
            aggregate_nodes,
            aggregate_count,
            min_aggregate_nodes,
            max_aggregate_nodes,
            aggregation,
            basis,
            transfer_apply_policy,
            aggregate_of_node,
            coarse_dimension,
            transfer_row_ptr,
            transfer_col_idx,
            transfer_values,
            smoothing_omega,
            base,
            coarse_apply_policy,
            coarse_lower_packed,
            coarse_upper_packed,
            coarse_upper_row_start,
            coarse_inverse,
            scratch,
            factor_bytes,
        })
    }

    pub fn dofs_per_node(&self) -> usize {
        self.dofs_per_node
    }
    pub fn aggregate_nodes(&self) -> usize {
        self.aggregate_nodes
    }
    pub fn aggregate_count(&self) -> usize {
        self.aggregate_count
    }
    pub fn min_aggregate_nodes(&self) -> usize {
        self.min_aggregate_nodes
    }
    pub fn max_aggregate_nodes(&self) -> usize {
        self.max_aggregate_nodes
    }
    pub fn aggregation(&self) -> TwoLevelAggregation {
        self.aggregation
    }
    pub fn basis(&self) -> TwoLevelBasis {
        self.basis
    }
    pub fn transfer_apply_policy(&self) -> TwoLevelTransferApplyPolicy {
        self.transfer_apply_policy
    }
    pub fn smoothing_omega(&self) -> f64 {
        self.smoothing_omega
    }
    pub fn transfer_nnz(&self) -> usize {
        self.transfer_values.len()
    }
    pub fn coarse_dimension(&self) -> usize {
        self.coarse_dimension
    }
    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }
    pub fn base_factor_bytes(&self) -> usize {
        self.base.factor_bytes()
    }
    pub fn coarse_apply_policy(&self) -> TwoLevelCoarseApplyPolicy {
        self.coarse_apply_policy
    }
    pub fn coarse_factor_bytes(&self) -> usize {
        (self.coarse_lower_packed.len()
            + self.coarse_upper_packed.len()
            + self.coarse_inverse.len())
            * std::mem::size_of::<f64>()
    }

    #[inline]
    fn coarse_index(&self, dof: usize) -> usize {
        piecewise_coarse_index(
            dof,
            self.dofs_per_node,
            self.aggregate_nodes,
            &self.aggregate_of_node,
        )
    }

    fn solve_packed_coarse(
        n: usize,
        lower: &[f64],
        upper: &[f64],
        upper_row_start: &[usize],
        rhs: &[f64],
        sol: &mut [f64],
    ) {
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(sol.len(), n);

        // Forward substitution reads one contiguous packed-L row at a time.
        for i in 0..n {
            let row_start = packed_lower_index(i, 0);
            let row = &lower[row_start..row_start + i + 1];
            let correction: f64 = row[..i]
                .iter()
                .zip(&sol[..i])
                .map(|(&lik, &sk)| lik * sk)
                .sum();
            sol[i] = (rhs[i] - correction) / row[i];
        }

        // Backward substitution streams through packed rows of L^T.
        for i in (0..n).rev() {
            let start = upper_row_start[i];
            let end = upper_row_start[i + 1];
            let row = &upper[start..end];
            let correction: f64 = row[1..]
                .iter()
                .zip(&sol[i + 1..])
                .map(|(&uki, &sk)| uki * sk)
                .sum();
            sol[i] = (sol[i] - correction) / row[0];
        }
    }

    fn solve_coarse(&self, rhs: &[f64], sol: &mut [f64]) {
        let n = self.coarse_dimension;
        match self.coarse_apply_policy {
            TwoLevelCoarseApplyPolicy::FactorSolve => Self::solve_packed_coarse(
                n,
                &self.coarse_lower_packed,
                &self.coarse_upper_packed,
                &self.coarse_upper_row_start,
                rhs,
                sol,
            ),
            TwoLevelCoarseApplyPolicy::ExplicitInverse => {
                debug_assert_eq!(self.coarse_inverse.len(), n * n);
                let coarse_inverse = self.coarse_inverse.as_slice();
                sol.par_iter_mut().enumerate().for_each(|(i, si)| {
                    let row = &coarse_inverse[i * n..(i + 1) * n];
                    *si = row.iter().zip(rhs).map(|(&a, &b)| a * b).sum();
                });
            }
            TwoLevelCoarseApplyPolicy::Auto => {
                unreachable!("Auto coarse-apply policy must be resolved during construction")
            }
        }
    }
}

impl Preconditioner for TwoLevelBlockJacobiPreconditioner {
    fn len(&self) -> usize {
        self.n
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        if z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: z.len(),
            });
        }

        // Fine/local SPD term.
        self.base.apply(r, z)?;

        // Coarse/global SPD term P E^-1 P^T. PiecewiseConstant keeps the
        // historical implicit one-entry-per-row transfer. JacobiSmoothed stores
        // the one-step smoothed prolongator sparsely by fine row.
        let mut scratch = self.scratch.borrow_mut();
        scratch.rhs.fill(0.0);
        match self.basis {
            TwoLevelBasis::PiecewiseConstant => {
                for (dof, &ri) in r.iter().enumerate() {
                    let ci = self.coarse_index(dof);
                    scratch.rhs[ci] += ri;
                }
            }
            TwoLevelBasis::JacobiSmoothed => match self.transfer_apply_policy {
                TwoLevelTransferApplyPolicy::Serial => {
                    for (row, &ri) in r.iter().enumerate() {
                        let start = self.transfer_row_ptr[row];
                        let end = self.transfer_row_ptr[row + 1];
                        for p in start..end {
                            let ci = self.transfer_col_idx[p] as usize;
                            scratch.rhs[ci] += self.transfer_values[p] * ri;
                        }
                    }
                }
                TwoLevelTransferApplyPolicy::Parallel => {
                    let n = self.n;
                    let row_ptr = self.transfer_row_ptr.as_slice();
                    let col_idx = self.transfer_col_idx.as_slice();
                    let values = self.transfer_values.as_slice();
                    let worker_count = scratch.restriction_buffers.len();
                    debug_assert!(worker_count > 0);
                    scratch
                        .restriction_buffers
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(worker, local)| {
                            local.fill(0.0);
                            let begin = n * worker / worker_count;
                            let end_row = n * (worker + 1) / worker_count;
                            for (offset, &ri) in r[begin..end_row].iter().enumerate() {
                                let row = begin + offset;
                                let start = row_ptr[row];
                                let end = row_ptr[row + 1];
                                for p in start..end {
                                    local[col_idx[p] as usize] += values[p] * ri;
                                }
                            }
                        });
                    let CoarseScratch {
                        rhs,
                        restriction_buffers,
                        ..
                    } = &mut *scratch;
                    rhs.fill(0.0);
                    for local in restriction_buffers.iter() {
                        for (dst, &value) in rhs.iter_mut().zip(local) {
                            *dst += value;
                        }
                    }
                }
            },
        }
        let CoarseScratch { rhs, sol, .. } = &mut *scratch;
        self.solve_coarse(rhs, sol);
        match self.basis {
            TwoLevelBasis::PiecewiseConstant => {
                for (dof, zi) in z.iter_mut().enumerate() {
                    *zi += sol[self.coarse_index(dof)];
                }
            }
            TwoLevelBasis::JacobiSmoothed => match self.transfer_apply_policy {
                TwoLevelTransferApplyPolicy::Serial => {
                    for (row, zi) in z.iter_mut().enumerate() {
                        let start = self.transfer_row_ptr[row];
                        let end = self.transfer_row_ptr[row + 1];
                        let correction = (start..end)
                            .map(|p| {
                                self.transfer_values[p] * sol[self.transfer_col_idx[p] as usize]
                            })
                            .sum::<f64>();
                        *zi += correction;
                    }
                }
                TwoLevelTransferApplyPolicy::Parallel => {
                    let row_ptr = self.transfer_row_ptr.as_slice();
                    let col_idx = self.transfer_col_idx.as_slice();
                    let values = self.transfer_values.as_slice();
                    let coarse_solution = sol.as_slice();
                    z.par_iter_mut().enumerate().for_each(|(row, zi)| {
                        let start = row_ptr[row];
                        let end = row_ptr[row + 1];
                        let correction = (start..end)
                            .map(|p| values[p] * coarse_solution[col_idx[p] as usize])
                            .sum::<f64>();
                        *zi += correction;
                    });
                }
            },
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RigidBodyAggregation {
    /// Let Structural Auto try graph-connected aggregation first and fall back
    /// to contiguous RCM-order aggregation if the graph coarse space is
    /// numerically singular or the graph contains components too small for a
    /// six-mode rigid-body aggregate. This variant is a policy request; prepared
    /// systems report the actual aggregation that was constructed.
    Auto,
    Contiguous,
    Graph,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RigidBodyApplyProfile {
    pub repeats: usize,
    pub base: Duration,
    pub restriction: Duration,
    pub coarse_solve: Duration,
    pub prolongation: Duration,
}

impl RigidBodyApplyProfile {
    pub fn total(self) -> Duration {
        self.base + self.restriction + self.coarse_solve + self.prolongation
    }
}

#[derive(Clone, Debug)]
pub struct RigidBodyTwoLevelBlockJacobiPreconditioner {
    n: usize,
    node_count: usize,
    aggregate_nodes: usize,
    aggregate_count: usize,
    min_aggregate_nodes: usize,
    max_aggregate_nodes: usize,
    aggregation: RigidBodyAggregation,
    aggregate_of_node: Vec<usize>,
    coarse_dimension: usize,
    base: BlockJacobiPreconditioner,
    normalized_offsets: Vec<[f64; 3]>,
    // Dense Cholesky L stored row-wise in packed lower-triangular form.
    // This keeps exactly n(n+1)/2 values instead of an n*n buffer.
    coarse_lower_packed: Vec<f64>,
    coarse_row_start: Vec<usize>,
    scratch: RefCell<CoarseScratch>,
    parallel_index: OnceLock<ParallelRigidBodyIndex>,
    factor_bytes: usize,
}

impl RigidBodyTwoLevelBlockJacobiPreconditioner {
    /// Build the original deterministic contiguous aggregation used by HyBIT 0.6 r4-r9.
    pub fn from_csr32(
        matrix: &Csr32Matrix,
        coordinates: &[[f64; 3]],
        aggregate_nodes: usize,
    ) -> Result<Self, HybitError> {
        let node_count = coordinates.len();
        if aggregate_nodes == 0 {
            return Err(HybitError::InvalidArgument("aggregate_nodes must be > 0"));
        }
        let aggregate_count = node_count
            .checked_add(aggregate_nodes - 1)
            .ok_or(HybitError::SizeOverflow)?
            / aggregate_nodes;
        let mut aggregate_of_node = Vec::with_capacity(node_count);
        for node in 0..node_count {
            aggregate_of_node.push(node / aggregate_nodes);
        }
        Self::from_assignment(
            matrix,
            coordinates,
            aggregate_nodes,
            aggregate_count,
            aggregate_of_node,
            RigidBodyAggregation::Contiguous,
        )
    }

    /// Build a graph-connected rigid-body coarse space.
    ///
    /// Node adjacency is inferred from the 3x3 block sparsity of the structural
    /// CSR matrix. Aggregates are deterministic breadth-first regions capped at
    /// `target_aggregate_nodes`; one/two-node tails are merged into adjacent
    /// aggregates. This keeps each six-mode rigid-body aggregate geometrically
    /// meaningful while no longer assuming that consecutive node numbers alone
    /// define a good coarse region.
    pub fn from_csr32_graph(
        matrix: &Csr32Matrix,
        coordinates: &[[f64; 3]],
        target_aggregate_nodes: usize,
    ) -> Result<Self, HybitError> {
        if target_aggregate_nodes < 3 {
            return Err(HybitError::InvalidArgument(
                "graph rigid-body target_aggregate_nodes must be at least 3",
            ));
        }
        let node_count = coordinates.len();
        let adjacency = build_structural_node_adjacency(matrix, node_count)?;
        let (aggregate_of_node, aggregate_count) =
            build_graph_aggregates(&adjacency, target_aggregate_nodes)?;
        Self::from_assignment(
            matrix,
            coordinates,
            target_aggregate_nodes,
            aggregate_count,
            aggregate_of_node,
            RigidBodyAggregation::Graph,
        )
    }

    fn from_assignment(
        matrix: &Csr32Matrix,
        coordinates: &[[f64; 3]],
        aggregate_nodes: usize,
        aggregate_count: usize,
        aggregate_of_node: Vec<usize>,
        aggregation: RigidBodyAggregation,
    ) -> Result<Self, HybitError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(HybitError::InvalidMatrix(
                "rigid-body two-level preconditioner requires a square matrix",
            ));
        }
        let node_count = coordinates.len();
        let n = node_count.checked_mul(3).ok_or(HybitError::SizeOverflow)?;
        if matrix.nrows() != n {
            return Err(HybitError::DimensionMismatch {
                expected: matrix.nrows(),
                actual: n,
            });
        }
        if coordinates.iter().flatten().any(|v| !v.is_finite()) {
            return Err(HybitError::InvalidArgument(
                "rigid-body coordinates contain NaN or infinity",
            ));
        }
        if aggregate_of_node.len() != node_count || aggregate_count == 0 {
            return Err(HybitError::InvalidArgument(
                "invalid rigid-body aggregate assignment",
            ));
        }
        if aggregate_of_node.iter().any(|&a| a >= aggregate_count) {
            return Err(HybitError::InvalidArgument(
                "rigid-body aggregate id is out of range",
            ));
        }

        let mut counts = vec![0usize; aggregate_count];
        for &aggregate in &aggregate_of_node {
            counts[aggregate] = counts[aggregate]
                .checked_add(1)
                .ok_or(HybitError::SizeOverflow)?;
        }
        if counts.iter().any(|&count| count < 3) {
            return Err(HybitError::InvalidArgument(
                "every rigid-body aggregate must contain at least three nodes",
            ));
        }
        let min_aggregate_nodes = *counts.iter().min().unwrap_or(&0);
        let max_aggregate_nodes = *counts.iter().max().unwrap_or(&0);

        let base = BlockJacobiPreconditioner::from_csr32(matrix, 3)?;
        let coarse_dimension = aggregate_count
            .checked_mul(6)
            .ok_or(HybitError::SizeOverflow)?;
        let coarse_len = coarse_dimension
            .checked_mul(coarse_dimension)
            .ok_or(HybitError::SizeOverflow)?;

        let mut centroids = vec![[0.0f64; 3]; aggregate_count];
        for (node, &aggregate) in aggregate_of_node.iter().enumerate() {
            centroids[aggregate][0] += coordinates[node][0];
            centroids[aggregate][1] += coordinates[node][1];
            centroids[aggregate][2] += coordinates[node][2];
        }
        for aggregate in 0..aggregate_count {
            let inv = 1.0 / counts[aggregate] as f64;
            centroids[aggregate][0] *= inv;
            centroids[aggregate][1] *= inv;
            centroids[aggregate][2] *= inv;
        }

        let mut radius_sq = vec![0.0f64; aggregate_count];
        for (node, &aggregate) in aggregate_of_node.iter().enumerate() {
            let dx = coordinates[node][0] - centroids[aggregate][0];
            let dy = coordinates[node][1] - centroids[aggregate][1];
            let dz = coordinates[node][2] - centroids[aggregate][2];
            radius_sq[aggregate] += dx * dx + dy * dy + dz * dz;
        }
        let mut inv_radius = vec![0.0f64; aggregate_count];
        for aggregate in 0..aggregate_count {
            let radius = (radius_sq[aggregate] / counts[aggregate] as f64).sqrt();
            if !radius.is_finite() || radius <= f64::EPSILON {
                return Err(HybitError::NumericalBreakdown(
                    "rigid-body aggregate has zero geometric radius",
                ));
            }
            inv_radius[aggregate] = 1.0 / radius;
        }

        let mut normalized_offsets = vec![[0.0f64; 3]; node_count];
        for (node, &aggregate) in aggregate_of_node.iter().enumerate() {
            normalized_offsets[node] = [
                (coordinates[node][0] - centroids[aggregate][0]) * inv_radius[aggregate],
                (coordinates[node][1] - centroids[aggregate][1]) * inv_radius[aggregate],
                (coordinates[node][2] - centroids[aggregate][2]) * inv_radius[aggregate],
            ];
        }

        #[inline]
        fn local_active_modes(component: usize, offset: [f64; 3]) -> ([usize; 3], [f64; 3]) {
            let [x, y, z] = offset;
            match component {
                0 => ([0, 4, 5], [1.0, z, -y]),
                1 => ([1, 3, 5], [1.0, -z, x]),
                2 => ([2, 3, 4], [1.0, y, -x]),
                _ => unreachable!(),
            }
        }

        let mut coarse = vec![0.0f64; coarse_len];
        for row in 0..n {
            let row_node = row / 3;
            let row_component = row % 3;
            let row_aggregate = aggregate_of_node[row_node];
            let (row_modes, row_values) =
                local_active_modes(row_component, normalized_offsets[row_node]);
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let col = matrix.col_idx()[p] as usize;
                let col_node = col / 3;
                let col_component = col % 3;
                let col_aggregate = aggregate_of_node[col_node];
                let (col_modes, col_values) =
                    local_active_modes(col_component, normalized_offsets[col_node]);
                let a = matrix.values()[p];
                for i in 0..3 {
                    let cr = row_aggregate * 6 + row_modes[i];
                    let rv = row_values[i];
                    for j in 0..3 {
                        let cc = col_aggregate * 6 + col_modes[j];
                        coarse[cr * coarse_dimension + cc] += a * rv * col_values[j];
                    }
                }
            }
        }

        for i in 0..coarse_dimension {
            for j in 0..i {
                let avg =
                    0.5 * (coarse[i * coarse_dimension + j] + coarse[j * coarse_dimension + i]);
                coarse[i * coarse_dimension + j] = avg;
                coarse[j * coarse_dimension + i] = avg;
            }
        }

        let mut scale = 0.0f64;
        for &v in &coarse {
            scale = scale.max(v.abs());
        }
        if !scale.is_finite() || scale == 0.0 {
            return Err(HybitError::NumericalBreakdown(
                "rigid-body coarse operator is zero or non-finite",
            ));
        }

        let coarse_factor_len = packed_lower_len(coarse_dimension)?;
        let mut coarse_row_start = Vec::with_capacity(coarse_dimension + 1);
        coarse_row_start.push(0usize);
        for row in 0..coarse_dimension {
            let next = coarse_row_start[row]
                .checked_add(row + 1)
                .ok_or(HybitError::SizeOverflow)?;
            coarse_row_start.push(next);
        }
        debug_assert_eq!(coarse_row_start[coarse_dimension], coarse_factor_len);

        let mut coarse_lower_packed = vec![0.0f64; coarse_factor_len];
        let pivot_tol = 1.0e-14 * scale.max(1.0);
        for i in 0..coarse_dimension {
            let i_base = coarse_row_start[i];
            for j in 0..=i {
                let j_base = coarse_row_start[j];
                let mut sum = coarse[i * coarse_dimension + j];
                for k in 0..j {
                    sum -= coarse_lower_packed[i_base + k] * coarse_lower_packed[j_base + k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= pivot_tol {
                        return Err(HybitError::NumericalBreakdown(
                            "rigid-body coarse Cholesky encountered a non-positive pivot",
                        ));
                    }
                    coarse_lower_packed[i_base + i] = sum.sqrt();
                } else {
                    coarse_lower_packed[i_base + j] = sum / coarse_lower_packed[j_base + j];
                }
            }
        }

        let scratch = RefCell::new(CoarseScratch {
            rhs: vec![0.0; coarse_dimension],
            sol: vec![0.0; coarse_dimension],
            restriction_buffers: Vec::new(),
        });
        let factor_bytes = base
            .factor_bytes()
            .checked_add(coarse_lower_packed.len() * std::mem::size_of::<f64>())
            .and_then(|v| v.checked_add(coarse_row_start.len() * std::mem::size_of::<usize>()))
            .and_then(|v| v.checked_add(2 * coarse_dimension * std::mem::size_of::<f64>()))
            .and_then(|v| v.checked_add(normalized_offsets.len() * std::mem::size_of::<[f64; 3]>()))
            .and_then(|v| v.checked_add(aggregate_of_node.len() * std::mem::size_of::<usize>()))
            .ok_or(HybitError::SizeOverflow)?;

        Ok(Self {
            n,
            node_count,
            aggregate_nodes,
            aggregate_count,
            min_aggregate_nodes,
            max_aggregate_nodes,
            aggregation,
            aggregate_of_node,
            coarse_dimension,
            base,
            normalized_offsets,
            coarse_lower_packed,
            coarse_row_start,
            scratch,
            parallel_index: OnceLock::new(),
            factor_bytes,
        })
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }
    pub fn aggregate_nodes(&self) -> usize {
        self.aggregate_nodes
    }
    pub fn aggregate_count(&self) -> usize {
        self.aggregate_count
    }
    pub fn min_aggregate_nodes(&self) -> usize {
        self.min_aggregate_nodes
    }
    pub fn max_aggregate_nodes(&self) -> usize {
        self.max_aggregate_nodes
    }
    pub fn aggregation(&self) -> RigidBodyAggregation {
        self.aggregation
    }
    pub fn coarse_dimension(&self) -> usize {
        self.coarse_dimension
    }
    pub fn modes_per_aggregate(&self) -> usize {
        6
    }
    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }
    pub fn base_factor_bytes(&self) -> usize {
        self.base.factor_bytes()
    }
    pub fn coarse_factor_bytes(&self) -> usize {
        self.coarse_lower_packed.len() * std::mem::size_of::<f64>()
    }
    pub fn geometry_bytes(&self) -> usize {
        self.normalized_offsets.len() * std::mem::size_of::<[f64; 3]>()
            + self.aggregate_of_node.len() * std::mem::size_of::<usize>()
    }

    fn ensure_parallel_index(&self) -> Result<&ParallelRigidBodyIndex, HybitError> {
        if let Some(index) = self.parallel_index.get() {
            return Ok(index);
        }

        let mut counts = vec![0usize; self.aggregate_count];
        for &aggregate in &self.aggregate_of_node {
            counts[aggregate] = counts[aggregate]
                .checked_add(1)
                .ok_or(HybitError::SizeOverflow)?;
        }

        let mut aggregate_node_ptr = Vec::with_capacity(self.aggregate_count + 1);
        aggregate_node_ptr.push(0usize);
        for &count in &counts {
            let next = aggregate_node_ptr
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(count)
                .ok_or(HybitError::SizeOverflow)?;
            aggregate_node_ptr.push(next);
        }
        debug_assert_eq!(aggregate_node_ptr[self.aggregate_count], self.node_count);

        let mut aggregate_nodes_order = vec![0usize; self.node_count];
        let mut next = aggregate_node_ptr[..self.aggregate_count].to_vec();
        for (node, &aggregate) in self.aggregate_of_node.iter().enumerate() {
            let slot = next[aggregate];
            aggregate_nodes_order[slot] = node;
            next[aggregate] += 1;
        }

        let built = ParallelRigidBodyIndex {
            aggregate_node_ptr,
            aggregate_nodes_order,
        };
        let _ = self.parallel_index.set(built);
        Ok(self
            .parallel_index
            .get()
            .expect("parallel rigid-body index initialized"))
    }

    pub fn parallel_index_bytes(&self) -> usize {
        self.parallel_index.get().map_or(0, |index| {
            (index.aggregate_node_ptr.len() + index.aggregate_nodes_order.len())
                * std::mem::size_of::<usize>()
        })
    }

    #[inline]
    fn active_modes(&self, dof: usize) -> ([usize; 3], [f64; 3]) {
        let node = dof / 3;
        let component = dof % 3;
        let [x, y, z] = self.normalized_offsets[node];
        let aggregate = self.aggregate_of_node[node];
        match component {
            0 => (
                [aggregate * 6, aggregate * 6 + 4, aggregate * 6 + 5],
                [1.0, z, -y],
            ),
            1 => (
                [aggregate * 6 + 1, aggregate * 6 + 3, aggregate * 6 + 5],
                [1.0, -z, x],
            ),
            2 => (
                [aggregate * 6 + 2, aggregate * 6 + 3, aggregate * 6 + 4],
                [1.0, y, -x],
            ),
            _ => unreachable!(),
        }
    }

    fn solve_coarse(&self, rhs: &[f64], sol: &mut [f64]) {
        let n = self.coarse_dimension;
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(sol.len(), n);

        // Forward solve L y = rhs. Packed rows make this pass contiguous.
        for i in 0..n {
            let i_base = self.coarse_row_start[i];
            let mut sum = rhs[i];
            for (k, &sk) in sol.iter().take(i).enumerate() {
                sum -= self.coarse_lower_packed[i_base + k] * sk;
            }
            sol[i] = sum / self.coarse_lower_packed[i_base + i];
        }

        // Backward solve L^T x = y.
        for i in (0..n).rev() {
            let i_base = self.coarse_row_start[i];
            let mut sum = sol[i];
            for (k, &sk) in sol.iter().enumerate().skip(i + 1) {
                let k_base = self.coarse_row_start[k];
                sum -= self.coarse_lower_packed[k_base + i] * sk;
            }
            sol[i] = sum / self.coarse_lower_packed[i_base + i];
        }
    }

    /// Apply only the Galerkin coarse correction
    ///
    /// ```text
    /// C r = Z (Z^T A Z)^-1 Z^T r
    /// ```
    ///
    /// This is kept separate from the additive preconditioner so experimental
    /// symmetric balanced two-level compositions can reuse exactly the same
    /// coarse space and Cholesky factor.
    /// Profile the four kernels that make up one additive rigid-body
    /// preconditioner application. This diagnostic path is intentionally
    /// separate from `Preconditioner::apply`, so normal solves pay no timing
    /// overhead.
    #[doc(hidden)]
    pub fn profile_apply_components(
        &self,
        r: &[f64],
        z: &mut [f64],
        repeats: usize,
    ) -> Result<RigidBodyApplyProfile, HybitError> {
        if r.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        if z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: z.len(),
            });
        }
        if repeats == 0 {
            return Err(HybitError::InvalidArgument("profile repeats must be > 0"));
        }

        let mut profile = RigidBodyApplyProfile {
            repeats,
            ..RigidBodyApplyProfile::default()
        };
        for _ in 0..repeats {
            let start = Instant::now();
            self.base.apply(r, z)?;
            profile.base += start.elapsed();

            let mut scratch = self.scratch.borrow_mut();

            let start = Instant::now();
            scratch.rhs.fill(0.0);
            for (dof, &ri) in r.iter().enumerate() {
                let (modes, values) = self.active_modes(dof);
                for k in 0..3 {
                    scratch.rhs[modes[k]] += values[k] * ri;
                }
            }
            profile.restriction += start.elapsed();

            let start = Instant::now();
            {
                let CoarseScratch { rhs, sol, .. } = &mut *scratch;
                self.solve_coarse(rhs, sol);
            }
            profile.coarse_solve += start.elapsed();

            let start = Instant::now();
            for (dof, zi) in z.iter_mut().enumerate() {
                let (modes, values) = self.active_modes(dof);
                let mut correction = 0.0f64;
                for k in 0..3 {
                    correction += values[k] * scratch.sol[modes[k]];
                }
                *zi += correction;
            }
            profile.prolongation += start.elapsed();
        }
        Ok(profile)
    }

    fn apply_coarse_only(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        if z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: z.len(),
            });
        }

        let mut scratch = self.scratch.borrow_mut();
        scratch.rhs.fill(0.0);
        for (dof, &ri) in r.iter().enumerate() {
            let (modes, values) = self.active_modes(dof);
            for k in 0..3 {
                scratch.rhs[modes[k]] += values[k] * ri;
            }
        }
        let CoarseScratch { rhs, sol, .. } = &mut *scratch;
        self.solve_coarse(rhs, sol);
        for (dof, zi) in z.iter_mut().enumerate() {
            let (modes, values) = self.active_modes(dof);
            let mut correction = 0.0f64;
            for k in 0..3 {
                correction += values[k] * sol[modes[k]];
            }
            *zi = correction;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct BalancedRigidScratch {
    aq_or_ay: Vec<f64>,
    projected_or_coarse: Vec<f64>,
    fine: Vec<f64>,
}

/// Experimental symmetric balanced two-level structural preconditioner.
///
/// With
///
/// ```text
/// C = Z (Z^T A Z)^-1 Z^T
/// P = I - C A
/// ```
///
/// this applies
///
/// ```text
/// M_bal^-1 = P B^-1 P^T + C
///          = (I - C A) B^-1 (I - A C) + C
/// ```
///
/// where `B^-1` is the same 3x3 block-Jacobi fine preconditioner used by the
/// additive rigid-body two-level method. For symmetric positive-definite A, B,
/// and a full-rank coarse basis, this composition is symmetric positive
/// definite and is therefore compatible with PCG.
///
/// The tradeoff is deliberate and measurable: each preconditioner application
/// performs two extra sparse matrix-vector products (`A C r` and `A B^-1 ...`).
/// HyBIT 0.6 keeps this as an experimental benchmark path until wall-clock data
/// justifies making it part of Structural Auto.
#[derive(Debug)]
pub struct BalancedRigidBodyTwoLevelBlockJacobiPreconditioner<'a> {
    matrix: &'a Csr32Matrix,
    inner: RigidBodyTwoLevelBlockJacobiPreconditioner,
    scratch: RefCell<BalancedRigidScratch>,
}

impl<'a> BalancedRigidBodyTwoLevelBlockJacobiPreconditioner<'a> {
    pub fn from_csr32(
        matrix: &'a Csr32Matrix,
        coordinates: &[[f64; 3]],
        aggregate_nodes: usize,
    ) -> Result<Self, HybitError> {
        let inner = RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32(
            matrix,
            coordinates,
            aggregate_nodes,
        )?;
        Self::from_inner(matrix, inner)
    }

    pub fn from_csr32_graph(
        matrix: &'a Csr32Matrix,
        coordinates: &[[f64; 3]],
        target_aggregate_nodes: usize,
    ) -> Result<Self, HybitError> {
        let inner = RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32_graph(
            matrix,
            coordinates,
            target_aggregate_nodes,
        )?;
        Self::from_inner(matrix, inner)
    }

    fn from_inner(
        matrix: &'a Csr32Matrix,
        inner: RigidBodyTwoLevelBlockJacobiPreconditioner,
    ) -> Result<Self, HybitError> {
        if matrix.nrows() != inner.n || matrix.ncols() != inner.n {
            return Err(HybitError::DimensionMismatch {
                expected: inner.n,
                actual: matrix.nrows(),
            });
        }
        let n = inner.n;
        Ok(Self {
            matrix,
            inner,
            scratch: RefCell::new(BalancedRigidScratch {
                aq_or_ay: vec![0.0; n],
                projected_or_coarse: vec![0.0; n],
                fine: vec![0.0; n],
            }),
        })
    }

    pub fn aggregation(&self) -> RigidBodyAggregation {
        self.inner.aggregation()
    }
    pub fn aggregate_nodes(&self) -> usize {
        self.inner.aggregate_nodes()
    }
    pub fn aggregate_count(&self) -> usize {
        self.inner.aggregate_count()
    }
    pub fn min_aggregate_nodes(&self) -> usize {
        self.inner.min_aggregate_nodes()
    }
    pub fn max_aggregate_nodes(&self) -> usize {
        self.inner.max_aggregate_nodes()
    }
    pub fn coarse_dimension(&self) -> usize {
        self.inner.coarse_dimension()
    }
    pub fn base_factor_bytes(&self) -> usize {
        self.inner.base_factor_bytes()
    }
    pub fn coarse_factor_bytes(&self) -> usize {
        self.inner.coarse_factor_bytes()
    }
    pub fn geometry_bytes(&self) -> usize {
        self.inner.geometry_bytes()
    }
    pub fn factor_bytes(&self) -> usize {
        self.inner.factor_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        3 * self.inner.n * std::mem::size_of::<f64>()
    }
}

impl Preconditioner for BalancedRigidBodyTwoLevelBlockJacobiPreconditioner<'_> {
    fn len(&self) -> usize {
        self.inner.n
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        let n = self.inner.n;
        if r.len() != n {
            return Err(HybitError::DimensionMismatch {
                expected: n,
                actual: r.len(),
            });
        }
        if z.len() != n {
            return Err(HybitError::DimensionMismatch {
                expected: n,
                actual: z.len(),
            });
        }

        // q = C r. Keep q directly in the output buffer until the final sum.
        self.inner.apply_coarse_only(r, z)?;

        let mut scratch = self.scratch.borrow_mut();
        let BalancedRigidScratch {
            aq_or_ay,
            projected_or_coarse,
            fine,
        } = &mut *scratch;

        // t = P^T r = (I - A C) r.
        self.matrix.apply(z, aq_or_ay)?;
        for i in 0..n {
            projected_or_coarse[i] = r[i] - aq_or_ay[i];
        }

        // y = B^-1 t.
        self.inner.base.apply(projected_or_coarse, fine)?;

        // cAy = C A y. Reuse the two temporary vectors now that t is no
        // longer needed.
        self.matrix.apply(fine, aq_or_ay)?;
        self.inner
            .apply_coarse_only(aq_or_ay, projected_or_coarse)?;

        // z = C r + (I - C A) y.
        for i in 0..n {
            z[i] += fine[i] - projected_or_coarse[i];
        }
        Ok(())
    }
}

#[inline]
fn piecewise_coarse_index(
    dof: usize,
    dofs_per_node: usize,
    aggregate_nodes: usize,
    aggregate_of_node: &[usize],
) -> usize {
    let node = dof / dofs_per_node;
    let component = dof % dofs_per_node;
    let aggregate = if aggregate_of_node.is_empty() {
        node / aggregate_nodes
    } else {
        aggregate_of_node[node]
    };
    aggregate * dofs_per_node + component
}

fn estimate_jacobi_spectral_radius(
    matrix: &Csr32Matrix,
    diagonal: &[f64],
) -> Result<f64, HybitError> {
    let n = matrix.nrows();
    if diagonal.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: diagonal.len(),
        });
    }
    let mut inv_sqrt = Vec::with_capacity(n);
    for (row, &d) in diagonal.iter().enumerate() {
        if !d.is_finite() || d == 0.0 {
            return Err(HybitError::ZeroDiagonal { row });
        }
        if d < 0.0 {
            return Err(HybitError::InvalidMatrix(
                "Jacobi-smoothed coarse basis requires a positive diagonal",
            ));
        }
        inv_sqrt.push(1.0 / d.sqrt());
    }

    let mut x: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((i.wrapping_mul(17).wrapping_add(11)) % 101) as f64)
        .collect();
    let mut y = vec![0.0f64; n];
    let mut norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(HybitError::NumericalBreakdown(
            "Jacobi smoothing spectral estimate has zero initial norm",
        ));
    }
    for xi in &mut x {
        *xi /= norm;
    }

    // Power iteration on D^-1/2 A D^-1/2, which is symmetric positive
    // definite whenever A is SPD with a positive diagonal. Ten iterations are
    // enough for this setup-time damping estimate and keep it deterministic.
    for _ in 0..10 {
        for (row, yi) in y.iter_mut().enumerate() {
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            let mut sum = 0.0f64;
            for p in rs..re {
                let col = matrix.col_idx()[p] as usize;
                sum += matrix.values()[p] * inv_sqrt[col] * x[col];
            }
            *yi = inv_sqrt[row] * sum;
        }
        norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(HybitError::NumericalBreakdown(
                "Jacobi smoothing spectral estimate broke down",
            ));
        }
        for (xi, &yi) in x.iter_mut().zip(&y) {
            *xi = yi / norm;
        }
    }

    for (row, yi) in y.iter_mut().enumerate() {
        let rs = matrix.row_ptr()[row] as usize;
        let re = matrix.row_ptr()[row + 1] as usize;
        let mut sum = 0.0f64;
        for p in rs..re {
            let col = matrix.col_idx()[p] as usize;
            sum += matrix.values()[p] * inv_sqrt[col] * x[col];
        }
        *yi = inv_sqrt[row] * sum;
    }
    let rho = x.iter().zip(&y).map(|(&xi, &yi)| xi * yi).sum::<f64>();
    if !rho.is_finite() || rho <= 0.0 {
        return Err(HybitError::NumericalBreakdown(
            "Jacobi smoothing spectral estimate is non-positive",
        ));
    }
    Ok(rho)
}

struct JacobiSmoothedTransfer {
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    values: Vec<f64>,
    omega: f64,
}

fn build_jacobi_smoothed_transfer(
    matrix: &Csr32Matrix,
    dofs_per_node: usize,
    aggregate_nodes: usize,
    aggregate_of_node: &[usize],
    coarse_dimension: usize,
) -> Result<JacobiSmoothedTransfer, HybitError> {
    let n = matrix.nrows();
    let diagonal = matrix.diagonal()?;
    let rho = estimate_jacobi_spectral_radius(matrix, &diagonal)?;
    let omega = 4.0 / (3.0 * rho);
    if !omega.is_finite() || omega <= 0.0 {
        return Err(HybitError::NumericalBreakdown(
            "Jacobi smoothing produced an invalid damping factor",
        ));
    }

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::<u32>::new();
    let mut values = Vec::<f64>::new();
    row_ptr.push(0usize);

    let mut stamp = vec![usize::MAX; coarse_dimension];
    let mut accum = vec![0.0f64; coarse_dimension];
    let mut touched = Vec::<usize>::new();
    for (row, &diag) in diagonal.iter().enumerate() {
        touched.clear();
        let token = row;
        let own = piecewise_coarse_index(
            row,
            dofs_per_node,
            aggregate_nodes,
            aggregate_of_node,
        );
        stamp[own] = token;
        accum[own] = 1.0;
        touched.push(own);

        let inv_diag = 1.0 / diag;
        let rs = matrix.row_ptr()[row] as usize;
        let re = matrix.row_ptr()[row + 1] as usize;
        for p in rs..re {
            let col = matrix.col_idx()[p] as usize;
            let coarse_col = piecewise_coarse_index(
                col,
                dofs_per_node,
                aggregate_nodes,
                aggregate_of_node,
            );
            if stamp[coarse_col] != token {
                stamp[coarse_col] = token;
                accum[coarse_col] = 0.0;
                touched.push(coarse_col);
            }
            accum[coarse_col] -= omega * matrix.values()[p] * inv_diag;
        }

        touched.sort_unstable();
        for &coarse_col in &touched {
            let value = accum[coarse_col];
            if value != 0.0 {
                col_idx.push(u32::try_from(coarse_col).map_err(|_| HybitError::SizeOverflow)?);
                values.push(value);
            }
        }
        row_ptr.push(col_idx.len());
    }

    Ok(JacobiSmoothedTransfer {
        row_ptr,
        col_idx,
        values,
        omega,
    })
}

fn build_galerkin_coarse_from_transfer(
    matrix: &Csr32Matrix,
    transfer_row_ptr: &[usize],
    transfer_col_idx: &[u32],
    transfer_values: &[f64],
    coarse_dimension: usize,
) -> Result<Vec<f64>, HybitError> {
    let n = matrix.nrows();
    if transfer_row_ptr.len() != n + 1 || transfer_col_idx.len() != transfer_values.len() {
        return Err(HybitError::InvalidArgument(
            "smoothed coarse transfer has inconsistent CSR storage",
        ));
    }
    let coarse_len = coarse_dimension
        .checked_mul(coarse_dimension)
        .ok_or(HybitError::SizeOverflow)?;
    let mut coarse = vec![0.0f64; coarse_len];

    // Form E = P^T A P without materializing A P.  Each fine row uses a dense
    // coarse accumulator with stamps; the touched set stays small because one
    // Jacobi smoothing step only reaches neighboring aggregates.
    let mut stamp = vec![usize::MAX; coarse_dimension];
    let mut ap = vec![0.0f64; coarse_dimension];
    let mut touched = Vec::<usize>::new();
    for row in 0..n {
        touched.clear();
        let token = row;
        let rs = matrix.row_ptr()[row] as usize;
        let re = matrix.row_ptr()[row + 1] as usize;
        for p in rs..re {
            let fine_col = matrix.col_idx()[p] as usize;
            let a = matrix.values()[p];
            let ps = transfer_row_ptr[fine_col];
            let pe = transfer_row_ptr[fine_col + 1];
            for q in ps..pe {
                let coarse_col = transfer_col_idx[q] as usize;
                if coarse_col >= coarse_dimension {
                    return Err(HybitError::InvalidArgument(
                        "smoothed coarse transfer column is out of range",
                    ));
                }
                if stamp[coarse_col] != token {
                    stamp[coarse_col] = token;
                    ap[coarse_col] = 0.0;
                    touched.push(coarse_col);
                }
                ap[coarse_col] += a * transfer_values[q];
            }
        }

        let ps = transfer_row_ptr[row];
        let pe = transfer_row_ptr[row + 1];
        for q in ps..pe {
            let coarse_row = transfer_col_idx[q] as usize;
            let weight = transfer_values[q];
            let dst = coarse_row
                .checked_mul(coarse_dimension)
                .ok_or(HybitError::SizeOverflow)?;
            for &coarse_col in &touched {
                coarse[dst + coarse_col] += weight * ap[coarse_col];
            }
        }
    }
    Ok(coarse)
}

fn build_block_node_adjacency(
    matrix: &Csr32Matrix,
    node_count: usize,
    dofs_per_node: usize,
) -> Result<Vec<Vec<usize>>, HybitError> {
    let expected = node_count
        .checked_mul(dofs_per_node)
        .ok_or(HybitError::SizeOverflow)?;
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(HybitError::DimensionMismatch {
            expected: matrix.nrows(),
            actual: expected,
        });
    }
    let mut adjacency = Vec::with_capacity(node_count);
    for node in 0..node_count {
        let mut neighbors = Vec::<usize>::new();
        for component in 0..dofs_per_node {
            let row = node * dofs_per_node + component;
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let other = matrix.col_idx()[p] as usize / dofs_per_node;
                if other != node {
                    neighbors.push(other);
                }
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        adjacency.push(neighbors);
    }
    Ok(adjacency)
}

fn build_strong_block_node_adjacency(
    matrix: &Csr32Matrix,
    node_count: usize,
    dofs_per_node: usize,
) -> Result<Vec<Vec<usize>>, HybitError> {
    let expected = node_count
        .checked_mul(dofs_per_node)
        .ok_or(HybitError::SizeOverflow)?;
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(HybitError::DimensionMismatch {
            expected: matrix.nrows(),
            actual: expected,
        });
    }

    // Accumulate Frobenius-norm squares for every node block.  Normalizing
    // off-diagonal block norms by the two diagonal-block norms keeps the
    // ordering meaningful when local stiffness scales differ strongly.
    let mut diagonal_norm_sq = vec![0.0f64; node_count];
    let mut edge_norm_sq = vec![HashMap::<usize, f64>::new(); node_count];
    for (node, edges) in edge_norm_sq.iter_mut().enumerate() {
        for component in 0..dofs_per_node {
            let row = node * dofs_per_node + component;
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let other = matrix.col_idx()[p] as usize / dofs_per_node;
                let value = matrix.values()[p];
                let square = value * value;
                if other == node {
                    diagonal_norm_sq[node] += square;
                } else {
                    *edges.entry(other).or_insert(0.0) += square;
                }
            }
        }
    }

    let diagonal_norm: Vec<f64> = diagonal_norm_sq.into_iter().map(f64::sqrt).collect();
    let mut adjacency = Vec::with_capacity(node_count);
    for (node, edges) in edge_norm_sq.iter().enumerate() {
        let mut weighted = Vec::<(usize, f64)>::with_capacity(edges.len());
        for (&other, &norm_sq) in edges {
            let edge_norm = norm_sq.sqrt();
            let denom = (diagonal_norm[node] * diagonal_norm[other]).sqrt();
            let strength = if denom.is_finite() && denom > 0.0 {
                edge_norm / denom
            } else {
                edge_norm
            };
            weighted.push((other, strength));
        }
        weighted.sort_unstable_by(|(a_node, a_strength), (b_node, b_strength)| {
            b_strength
                .total_cmp(a_strength)
                .then_with(|| a_node.cmp(b_node))
        });
        adjacency.push(weighted.into_iter().map(|(other, _)| other).collect());
    }
    Ok(adjacency)
}

fn build_piecewise_constant_graph_aggregates(
    adjacency: &[Vec<usize>],
    target_size: usize,
) -> Result<(Vec<usize>, usize), HybitError> {
    if target_size == 0 {
        return Err(HybitError::InvalidArgument(
            "graph two-level target aggregate size must be > 0",
        ));
    }
    let n = adjacency.len();
    if n == 0 {
        return Err(HybitError::InvalidArgument(
            "graph two-level aggregation requires at least one node",
        ));
    }

    let unassigned = usize::MAX;
    let mut aggregate_of_node = vec![unassigned; n];
    let mut aggregate_count = 0usize;
    let mut queue = std::collections::VecDeque::<usize>::new();
    let mut members = Vec::<usize>::with_capacity(target_size);

    for seed in 0..n {
        if aggregate_of_node[seed] != unassigned {
            continue;
        }
        queue.clear();
        members.clear();
        queue.push_back(seed);
        while members.len() < target_size {
            let Some(node) = queue.pop_front() else {
                break;
            };
            if aggregate_of_node[node] != unassigned {
                continue;
            }
            aggregate_of_node[node] = aggregate_count;
            members.push(node);
            for &neighbor in &adjacency[node] {
                if aggregate_of_node[neighbor] == unassigned {
                    queue.push_back(neighbor);
                }
            }
        }
        if members.is_empty() {
            return Err(HybitError::InvalidArgument(
                "graph two-level aggregation produced an empty region",
            ));
        }
        aggregate_count = aggregate_count
            .checked_add(1)
            .ok_or(HybitError::SizeOverflow)?;
    }

    // Absorb very small graph fragments into the neighboring aggregate with
    // the strongest edge connection. Track member lists so merging costs scale
    // with the fragment boundary rather than rescanning every node per region.
    // Piecewise-constant modes remain valid on singleton disconnected
    // components, so fragments with no external edge are intentionally kept.
    let merge_floor = (target_size / 4).max(1);
    let mut aggregate_members = vec![Vec::<usize>::new(); aggregate_count];
    for (node, &aggregate) in aggregate_of_node.iter().enumerate() {
        aggregate_members[aggregate].push(node);
    }

    loop {
        let mut merged_any = false;
        for small in 0..aggregate_count {
            if aggregate_members[small].is_empty() || aggregate_members[small].len() >= merge_floor
            {
                continue;
            }

            let mut edge_counts = HashMap::<usize, usize>::new();
            for &node in &aggregate_members[small] {
                for &neighbor in &adjacency[node] {
                    let other = aggregate_of_node[neighbor];
                    if other != small && !aggregate_members[other].is_empty() {
                        *edge_counts.entry(other).or_insert(0) += 1;
                    }
                }
            }
            let replacement = edge_counts
                .into_iter()
                .max_by(|(a_id, a_edges), (b_id, b_edges)| {
                    a_edges
                        .cmp(b_edges)
                        .then_with(|| {
                            aggregate_members[*b_id]
                                .len()
                                .cmp(&aggregate_members[*a_id].len())
                        })
                        .then_with(|| b_id.cmp(a_id))
                })
                .map(|(id, _)| id);
            let Some(replacement) = replacement else {
                continue;
            };

            let moved = std::mem::take(&mut aggregate_members[small]);
            for &node in &moved {
                aggregate_of_node[node] = replacement;
            }
            aggregate_members[replacement].extend(moved);
            merged_any = true;
        }
        if !merged_any {
            break;
        }
    }

    let mut remap = vec![usize::MAX; aggregate_count];
    let mut compact_count = 0usize;
    for &aggregate in &aggregate_of_node {
        if remap[aggregate] == usize::MAX {
            remap[aggregate] = compact_count;
            compact_count += 1;
        }
    }
    for aggregate in &mut aggregate_of_node {
        *aggregate = remap[*aggregate];
    }
    Ok((aggregate_of_node, compact_count))
}

fn build_structural_node_adjacency(
    matrix: &Csr32Matrix,
    node_count: usize,
) -> Result<Vec<Vec<usize>>, HybitError> {
    build_block_node_adjacency(matrix, node_count, 3)
}

fn build_graph_aggregates(
    adjacency: &[Vec<usize>],
    target_size: usize,
) -> Result<(Vec<usize>, usize), HybitError> {
    let n = adjacency.len();
    if n < 3 {
        return Err(HybitError::InvalidArgument(
            "graph rigid-body aggregation requires at least three nodes",
        ));
    }
    let unassigned = usize::MAX;
    let mut aggregate_of_node = vec![unassigned; n];
    let mut aggregate_count = 0usize;
    let mut queue = std::collections::VecDeque::<usize>::new();
    let mut members = Vec::<usize>::with_capacity(target_size);

    for seed in 0..n {
        if aggregate_of_node[seed] != unassigned {
            continue;
        }
        queue.clear();
        members.clear();
        queue.push_back(seed);

        while members.len() < target_size {
            let Some(node) = queue.pop_front() else {
                break;
            };
            if aggregate_of_node[node] != unassigned {
                continue;
            }
            aggregate_of_node[node] = aggregate_count;
            members.push(node);
            for &neighbor in &adjacency[node] {
                if aggregate_of_node[neighbor] == unassigned {
                    queue.push_back(neighbor);
                }
            }
        }

        if members.len() >= 3 {
            aggregate_count = aggregate_count
                .checked_add(1)
                .ok_or(HybitError::SizeOverflow)?;
            continue;
        }

        // Merge a one/two-node tail into already-built neighboring aggregates.
        // This can occur when the final graph region is smaller than the target.
        if aggregate_count == 0 {
            return Err(HybitError::InvalidArgument(
                "a structural graph component contains fewer than three nodes",
            ));
        }
        for &node in &members {
            let mut chosen = None::<usize>;
            for &neighbor in &adjacency[node] {
                let a = aggregate_of_node[neighbor];
                if a != unassigned && a != aggregate_count {
                    chosen = Some(a);
                    break;
                }
            }
            aggregate_of_node[node] = chosen.unwrap_or(aggregate_count - 1);
        }
    }

    if aggregate_of_node
        .iter()
        .any(|&a| a == unassigned || a >= aggregate_count)
    {
        return Err(HybitError::InvalidArgument(
            "graph aggregation left unassigned nodes",
        ));
    }

    // Greedy capped BFS can leave small connected islands after neighboring
    // nodes were already claimed by earlier aggregates. Six rigid-body modes
    // are a poor basis on tiny islands (three nearly-collinear nodes are enough
    // to make the local rigid space rank deficient), so absorb undersized
    // islands into a strongly connected neighboring aggregate before building
    // the Galerkin coarse operator. Keep the threshold conservative: at least
    // three nodes and otherwise one quarter of the requested aggregate size.
    let merge_floor = (target_size / 4).max(3);
    let mut counts = vec![0usize; aggregate_count];
    for &a in &aggregate_of_node {
        counts[a] += 1;
    }

    loop {
        let mut merged_any = false;
        for small in 0..aggregate_count {
            if counts[small] == 0 || counts[small] >= merge_floor {
                continue;
            }

            // Prefer the neighboring aggregate with the most graph edges back
            // to this small island. Ties go to the smaller target aggregate and
            // then the lower aggregate id for deterministic behavior.
            let mut edge_counts = HashMap::<usize, usize>::new();
            for node in 0..n {
                if aggregate_of_node[node] != small {
                    continue;
                }
                for &neighbor in &adjacency[node] {
                    let other = aggregate_of_node[neighbor];
                    if other != small && other < aggregate_count && counts[other] > 0 {
                        *edge_counts.entry(other).or_insert(0) += 1;
                    }
                }
            }

            let replacement = edge_counts
                .into_iter()
                .max_by(|(a_id, a_edges), (b_id, b_edges)| {
                    a_edges
                        .cmp(b_edges)
                        .then_with(|| counts[*b_id].cmp(&counts[*a_id]))
                        .then_with(|| b_id.cmp(a_id))
                })
                .map(|(id, _)| id);

            // A disconnected structural component cannot be merged across a
            // nonexistent edge. Leave such a component intact if it already has
            // at least three nodes; from_assignment will still validate its
            // geometry and coarse factorization explicitly.
            let Some(replacement) = replacement else {
                continue;
            };

            for a in &mut aggregate_of_node {
                if *a == small {
                    *a = replacement;
                }
            }
            counts[replacement] += counts[small];
            counts[small] = 0;
            merged_any = true;
        }
        if !merged_any {
            break;
        }
    }

    // Compact aggregate ids after merges.
    let mut remap = vec![usize::MAX; aggregate_count];
    let mut compact_count = 0usize;
    for &a in &aggregate_of_node {
        if remap[a] == usize::MAX {
            remap[a] = compact_count;
            compact_count += 1;
        }
    }
    for a in &mut aggregate_of_node {
        *a = remap[*a];
    }
    Ok((aggregate_of_node, compact_count))
}

impl Preconditioner for RigidBodyTwoLevelBlockJacobiPreconditioner {
    fn len(&self) -> usize {
        self.n
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: r.len(),
            });
        }
        if z.len() != self.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.n,
                actual: z.len(),
            });
        }

        self.base.apply(r, z)?;

        let mut scratch = self.scratch.borrow_mut();
        scratch.rhs.fill(0.0);
        for (dof, &ri) in r.iter().enumerate() {
            let (modes, values) = self.active_modes(dof);
            for k in 0..3 {
                scratch.rhs[modes[k]] += values[k] * ri;
            }
        }
        let CoarseScratch { rhs, sol, .. } = &mut *scratch;
        self.solve_coarse(rhs, sol);
        for (dof, zi) in z.iter_mut().enumerate() {
            let (modes, values) = self.active_modes(dof);
            let mut correction = 0.0f64;
            for k in 0..3 {
                correction += values[k] * sol[modes[k]];
            }
            *zi += correction;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ParallelRigidBodyIndex {
    aggregate_node_ptr: Vec<usize>,
    aggregate_nodes_order: Vec<usize>,
}

/// Experimental view that parallelizes only the embarrassingly parallel
/// fine/coarse transfer and 3x3 block-Jacobi kernels. The dense coarse
/// triangular solve remains serial. It borrows the already prepared rigid-body
/// preconditioner, so no Cholesky factor is duplicated.
#[derive(Debug)]
pub struct ParallelRigidBodyTwoLevelPreconditioner<'a> {
    inner: &'a RigidBodyTwoLevelBlockJacobiPreconditioner,
}

impl<'a> ParallelRigidBodyTwoLevelPreconditioner<'a> {
    pub fn new(inner: &'a RigidBodyTwoLevelBlockJacobiPreconditioner) -> Result<Self, HybitError> {
        inner.ensure_parallel_index()?;
        Ok(Self { inner })
    }

    pub fn rayon_threads(&self) -> usize {
        rayon::current_num_threads()
    }
    pub fn index_storage_bytes(&self) -> usize {
        self.inner.parallel_index_bytes()
    }
    pub fn aggregate_count(&self) -> usize {
        self.inner.aggregate_count()
    }
    pub fn coarse_dimension(&self) -> usize {
        self.inner.coarse_dimension()
    }
    pub fn factor_bytes(&self) -> usize {
        self.inner.factor_bytes()
    }
}

impl Preconditioner for ParallelRigidBodyTwoLevelPreconditioner<'_> {
    fn len(&self) -> usize {
        self.inner.n
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.inner.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.inner.n,
                actual: r.len(),
            });
        }
        if z.len() != self.inner.n {
            return Err(HybitError::DimensionMismatch {
                expected: self.inner.n,
                actual: z.len(),
            });
        }

        self.inner.base.apply_parallel(r, z)?;

        let offsets = self.inner.normalized_offsets.as_slice();
        let aggregate_of_node = self.inner.aggregate_of_node.as_slice();
        let index = self
            .inner
            .parallel_index
            .get()
            .expect("parallel rigid-body index initialized");
        let node_ptr = index.aggregate_node_ptr.as_slice();
        let node_order = index.aggregate_nodes_order.as_slice();
        let mut scratch = self.inner.scratch.borrow_mut();

        {
            let coarse_rhs = scratch.rhs.as_mut_slice();
            coarse_rhs
                .par_chunks_mut(6)
                .enumerate()
                .for_each(|(aggregate, out)| {
                    out.fill(0.0);
                    for &node in &node_order[node_ptr[aggregate]..node_ptr[aggregate + 1]] {
                        let base = node * 3;
                        let rx = r[base];
                        let ry = r[base + 1];
                        let rz = r[base + 2];
                        let [x, y, zc] = offsets[node];
                        out[0] += rx;
                        out[1] += ry;
                        out[2] += rz;
                        out[3] += -zc * ry;
                        out[3] += y * rz;
                        out[4] += zc * rx;
                        out[4] += -x * rz;
                        out[5] += -y * rx;
                        out[5] += x * ry;
                    }
                });
        }

        {
            let CoarseScratch { rhs, sol, .. } = &mut *scratch;
            self.inner.solve_coarse(rhs, sol);
        }

        let coarse_sol = scratch.sol.as_slice();
        z.par_chunks_mut(3).enumerate().for_each(|(node, z_node)| {
            let aggregate = aggregate_of_node[node];
            let c = &coarse_sol[aggregate * 6..aggregate * 6 + 6];
            let [x, y, zc] = offsets[node];
            z_node[0] += c[0] + zc * c[4] - y * c[5];
            z_node[1] += c[1] - zc * c[3] + x * c[5];
            z_node[2] += c[2] + y * c[3] - x * c[4];
        });
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LocalCholeskyRegion {
    indices: Vec<usize>,
    lower: Vec<f64>,
}

impl LocalCholeskyRegion {
    pub fn from_csr32(matrix: &Csr32Matrix, indices: &[usize]) -> Result<Self, HybitError> {
        if indices.is_empty() {
            return Err(HybitError::InvalidArgument(
                "local Cholesky region may not be empty",
            ));
        }

        let mut canonical = indices.to_vec();
        canonical.sort_unstable();
        canonical.dedup();

        if canonical.len() != indices.len() {
            return Err(HybitError::InvalidArgument(
                "local Cholesky region contains duplicate DOFs",
            ));
        }
        if canonical.iter().any(|&i| i >= matrix.nrows()) {
            return Err(HybitError::InvalidArgument(
                "local Cholesky DOF is out of range",
            ));
        }
        if matrix.nrows() != matrix.ncols() {
            return Err(HybitError::InvalidMatrix(
                "local Cholesky requires a square matrix",
            ));
        }

        let n = canonical.len();
        let local_of: HashMap<usize, usize> = canonical
            .iter()
            .copied()
            .enumerate()
            .map(|(i, g)| (g, i))
            .collect();

        let packed_len = packed_lower_len(n)?;

        // Keep the two matrix halves in packed triangular storage while
        // checking symmetry. After validation, `upper` is released and
        // `lower` is factorized in place. This removes the old pair of n*n
        // dense buffers from local-direct setup.
        let mut lower = vec![0.0f64; packed_len];
        let mut upper = vec![0.0f64; packed_len];

        for (local_row, &global_row) in canonical.iter().enumerate() {
            let start = matrix.row_ptr()[global_row] as usize;
            let end = matrix.row_ptr()[global_row + 1] as usize;

            for p in start..end {
                let global_col = matrix.col_idx()[p] as usize;
                if let Some(&local_col) = local_of.get(&global_col) {
                    let value = matrix.values()[p];
                    if local_row >= local_col {
                        lower[packed_lower_index(local_row, local_col)] += value;
                    } else {
                        // Store A(i,j) at the packed slot of its mirrored
                        // lower-triangular position A(j,i).
                        upper[packed_lower_index(local_col, local_row)] += value;
                    }
                }
            }
        }

        // Local direct correction is currently an SPD path. Require the local
        // principal matrix to be numerically symmetric before factorization.
        let mut scale = 0.0f64;
        for (&lower_value, &upper_value) in lower.iter().zip(&upper) {
            scale = scale.max(lower_value.abs()).max(upper_value.abs());
        }

        let symmetry_tol = 1.0e-11 * scale.max(1.0);
        for i in 0..n {
            for j in 0..i {
                let ij = packed_lower_index(i, j);
                if (lower[ij] - upper[ij]).abs() > symmetry_tol {
                    return Err(HybitError::InvalidMatrix(
                        "local Cholesky region is not symmetric",
                    ));
                }
            }
        }

        // The upper half is no longer required. Reuse the packed lower matrix
        // as the Cholesky factor buffer.
        drop(upper);

        let pivot_tol = 1.0e-14 * scale.max(1.0);
        for i in 0..n {
            for j in 0..=i {
                let ij = packed_lower_index(i, j);
                let mut sum = lower[ij];

                for k in 0..j {
                    sum -= lower[packed_lower_index(i, k)] * lower[packed_lower_index(j, k)];
                }

                if i == j {
                    if !sum.is_finite() || sum <= pivot_tol {
                        return Err(HybitError::NumericalBreakdown(
                            "local Cholesky encountered a non-positive pivot",
                        ));
                    }
                    lower[ij] = sum.sqrt();
                } else {
                    lower[ij] = sum / lower[packed_lower_index(j, j)];
                }
            }
        }

        Ok(Self {
            indices: canonical,
            lower,
        })
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn solve_local(&self, rhs: &[f64], out: &mut [f64]) {
        let n = self.indices.len();
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(out.len(), n);

        // Forward substitution: L y = rhs.
        for i in 0..n {
            let mut sum = rhs[i];
            for (k, &yk) in out.iter().take(i).enumerate() {
                sum -= self.lower[packed_lower_index(i, k)] * yk;
            }
            out[i] = sum / self.lower[packed_lower_index(i, i)];
        }

        // Backward substitution: L^T x = y.
        for i in (0..n).rev() {
            let mut sum = out[i];
            for (k, &xk) in out.iter().enumerate().skip(i + 1) {
                sum -= self.lower[packed_lower_index(k, i)] * xk;
            }
            out[i] = sum / self.lower[packed_lower_index(i, i)];
        }
    }

    pub fn factor_bytes(&self) -> usize {
        self.indices.len() * std::mem::size_of::<usize>()
            + self.lower.len() * std::mem::size_of::<f64>()
    }
}

#[derive(Clone, Debug)]
struct RegionScratch {
    rhs: Vec<f64>,
    sol: Vec<f64>,
}

#[derive(Clone, Debug)]
struct WeightedLocalRegion {
    factor: LocalCholeskyRegion,
    weights: Vec<f64>,
    scratch: RefCell<RegionScratch>,
}

#[derive(Clone, Debug)]
pub struct HybridPreconditioner {
    jacobi: JacobiPreconditioner,
    regions: Vec<WeightedLocalRegion>,
    multiplicity: Vec<u16>,
    largest_region: usize,
    unique_local_dofs: usize,
    factor_bytes: usize,
}

impl HybridPreconditioner {
    /// Estimate the persistent bytes owned by the local-direct portion of a
    /// hybrid preconditioner before numerical factorization.
    ///
    /// The estimate matches [`HybridPreconditioner::factor_bytes`] for the
    /// current packed local-Cholesky representation. It includes the global
    /// overlap multiplicity array, per-region indices, packed Cholesky values,
    /// symmetric weights, and the two reusable local scratch vectors.
    pub fn estimated_factor_bytes(
        matrix_rows: usize,
        regions: &[Vec<usize>],
    ) -> Result<usize, HybitError> {
        let mut bytes = matrix_rows
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(HybitError::SizeOverflow)?;

        for region in regions {
            if region.is_empty() {
                continue;
            }

            let mut canonical = region.clone();
            canonical.sort_unstable();
            canonical.dedup();
            if canonical.iter().any(|&dof| dof >= matrix_rows) {
                return Err(HybitError::InvalidArgument(
                    "hybrid preconditioner DOF is out of range",
                ));
            }

            let n = canonical.len();
            let packed = packed_lower_len(n)?;

            let region_bytes = n
                .checked_mul(std::mem::size_of::<usize>())
                .and_then(|v| {
                    packed
                        .checked_mul(std::mem::size_of::<f64>())
                        .and_then(|packed_bytes| v.checked_add(packed_bytes))
                })
                .and_then(|v| {
                    n.checked_mul(std::mem::size_of::<f64>())
                        .and_then(|weights| v.checked_add(weights))
                })
                .and_then(|v| {
                    n.checked_mul(2 * std::mem::size_of::<f64>())
                        .and_then(|scratch| v.checked_add(scratch))
                })
                .ok_or(HybitError::SizeOverflow)?;

            bytes = bytes
                .checked_add(region_bytes)
                .ok_or(HybitError::SizeOverflow)?;
        }

        Ok(bytes)
    }

    /// Build a symmetric weighted overlapping Schwarz preconditioner.
    ///
    /// For a DOF contained in m local regions each local restriction uses
    /// w_i = 1/sqrt(m).  The local term is therefore
    /// R^T W A_H^{-1} W R, which is symmetric positive semidefinite when
    /// A_H is SPD. Jacobi is retained only on DOFs not covered by a local
    /// factor, preventing double counting in the single-region case.
    pub fn from_csr32(matrix: &Csr32Matrix, regions: Vec<Vec<usize>>) -> Result<Self, HybitError> {
        let jacobi = JacobiPreconditioner::from_csr32(matrix)?;
        let mut canonical_regions = Vec::with_capacity(regions.len());
        let mut multiplicity = vec![0u16; matrix.nrows()];
        for mut region in regions {
            if region.is_empty() {
                continue;
            }
            region.sort_unstable();
            region.dedup();
            for &dof in &region {
                if dof >= matrix.nrows() {
                    return Err(HybitError::InvalidArgument(
                        "hybrid preconditioner DOF is out of range",
                    ));
                }
                multiplicity[dof] = multiplicity[dof]
                    .checked_add(1)
                    .ok_or(HybitError::SizeOverflow)?;
            }
            canonical_regions.push(region);
        }
        if canonical_regions.is_empty() {
            return Err(HybitError::InvalidArgument(
                "hybrid preconditioner requires at least one local region",
            ));
        }

        let unique_local_dofs = multiplicity.iter().filter(|&&m| m > 0).count();
        let mut factors = Vec::with_capacity(canonical_regions.len());
        let mut largest_region = 0usize;
        let mut factor_bytes = multiplicity.len() * std::mem::size_of::<u16>();
        for region in canonical_regions {
            let factor = LocalCholeskyRegion::from_csr32(matrix, &region)?;
            let weights: Vec<f64> = factor
                .indices()
                .iter()
                .map(|&dof| 1.0 / (multiplicity[dof] as f64).sqrt())
                .collect();
            largest_region = largest_region.max(factor.len());
            let n = factor.len();
            let scratch = RegionScratch {
                rhs: vec![0.0; n],
                sol: vec![0.0; n],
            };
            factor_bytes += factor.factor_bytes()
                + weights.len() * std::mem::size_of::<f64>()
                + 2 * n * std::mem::size_of::<f64>();
            factors.push(WeightedLocalRegion {
                factor,
                weights,
                scratch: RefCell::new(scratch),
            });
        }

        Ok(Self {
            jacobi,
            regions: factors,
            multiplicity,
            largest_region,
            unique_local_dofs,
            factor_bytes,
        })
    }

    pub fn region_count(&self) -> usize {
        self.regions.len()
    }
    pub fn largest_region(&self) -> usize {
        self.largest_region
    }
    pub fn local_dofs(&self) -> usize {
        self.regions.iter().map(|r| r.factor.len()).sum()
    }
    pub fn unique_local_dofs(&self) -> usize {
        self.unique_local_dofs
    }
    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }
    pub fn regions(&self) -> impl Iterator<Item = &LocalCholeskyRegion> {
        self.regions.iter().map(|r| &r.factor)
    }

    /// Add only the weighted selective-direct correction to an existing
    /// output vector. This excludes the HybridPreconditioner's Jacobi base and
    /// is intended for composing the local-direct term with another SPD base
    /// preconditioner, such as a two-level coarse correction.
    pub fn add_local_correction(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.len() {
            return Err(HybitError::DimensionMismatch {
                expected: self.len(),
                actual: r.len(),
            });
        }
        if z.len() != self.len() {
            return Err(HybitError::DimensionMismatch {
                expected: self.len(),
                actual: z.len(),
            });
        }

        for region in &self.regions {
            let mut scratch = region.scratch.borrow_mut();
            let RegionScratch { rhs, sol } = &mut *scratch;
            for (i, (&gi, &w)) in region
                .factor
                .indices()
                .iter()
                .zip(&region.weights)
                .enumerate()
            {
                rhs[i] = w * r[gi];
            }
            region.factor.solve_local(rhs, sol);
            for (i, (&gi, &w)) in region
                .factor
                .indices()
                .iter()
                .zip(&region.weights)
                .enumerate()
            {
                z[gi] += w * sol[i];
            }
        }
        Ok(())
    }
}

impl Preconditioner for HybridPreconditioner {
    fn len(&self) -> usize {
        self.jacobi.len()
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        if r.len() != self.len() {
            return Err(HybitError::DimensionMismatch {
                expected: self.len(),
                actual: r.len(),
            });
        }
        if z.len() != self.len() {
            return Err(HybitError::DimensionMismatch {
                expected: self.len(),
                actual: z.len(),
            });
        }

        // Base Jacobi acts only outside all selected local factors.
        for i in 0..z.len() {
            z[i] = if self.multiplicity[i] == 0 {
                self.jacobi.inv_diagonal()[i] * r[i]
            } else {
                0.0
            };
        }

        // Symmetrically weighted overlapping local corrections. Scratch storage
        // is allocated once when the preconditioner is built; apply() performs
        // no heap allocation in the Krylov iteration loop.
        self.add_local_correction(r, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poisson_1d(n: usize) -> Csr32Matrix {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..n {
            if i > 0 {
                col_idx.push((i - 1) as u32);
                values.push(-1.0);
            }
            col_idx.push(i as u32);
            values.push(2.0);
            if i + 1 < n {
                col_idx.push((i + 1) as u32);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn local_cholesky_solves_poisson_block() {
        let a = poisson_1d(4);
        let region = LocalCholeskyRegion::from_csr32(&a, &[0, 1, 2, 3]).unwrap();
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0; 4];
        region.solve_local(&r, &mut z);
        let y = a.spmv(&z).unwrap();
        for (yi, ri) in y.iter().zip(&r) {
            assert!((yi - ri).abs() < 1.0e-12);
        }
    }

    #[test]
    fn local_cholesky_uses_packed_lower_storage() {
        let n = 8usize;
        let a = poisson_1d(n);
        let indices: Vec<usize> = (0..n).collect();
        let region = LocalCholeskyRegion::from_csr32(&a, &indices).unwrap();
        let packed_len = n * (n + 1) / 2;

        assert_eq!(region.lower.len(), packed_len);
        assert_eq!(
            region.factor_bytes(),
            n * std::mem::size_of::<usize>() + packed_len * std::mem::size_of::<f64>()
        );
    }

    #[test]
    fn block_jacobi_solves_block_diagonal_spd_system() {
        let a = Csr32Matrix::new(
            6,
            6,
            vec![0, 2, 5, 7, 9, 12, 14],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 4, 5],
            vec![
                4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0,
            ],
        )
        .unwrap();
        let bj = BlockJacobiPreconditioner::from_csr32(&a, 3).unwrap();
        let r = vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0];
        let mut z = vec![0.0; 6];
        bj.apply(&r, &mut z).unwrap();
        let az = a.spmv(&z).unwrap();
        for (lhs, rhs) in az.iter().zip(&r) {
            assert!((lhs - rhs).abs() < 1.0e-12);
        }
        assert_eq!(bj.block_size(), 3);
        assert_eq!(bj.block_count(), 2);
        assert!(bj.factor_bytes() > 0);
    }

    #[test]
    fn two_level_block_jacobi_is_positive() {
        let a = poisson_1d(16);
        let two = TwoLevelBlockJacobiPreconditioner::from_csr32(&a, 1, 4).unwrap();
        let r: Vec<f64> = (0..16).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        let mut z = vec![0.0; 16];
        two.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        assert!(rz > 0.0);
        assert_eq!(two.aggregate_count(), 4);
        assert_eq!(two.coarse_dimension(), 4);
        assert!(two.factor_bytes() > two.base_factor_bytes());
    }

    #[test]
    fn two_level_coarse_factor_uses_bidirectional_packed_rows() {
        let a = poisson_1d(16);
        let two = TwoLevelBlockJacobiPreconditioner::from_csr32(&a, 1, 4).unwrap();
        let n = two.coarse_dimension();
        let packed_len = n * (n + 1) / 2;

        assert_eq!(
            two.coarse_apply_policy(),
            TwoLevelCoarseApplyPolicy::FactorSolve
        );
        assert_eq!(two.coarse_lower_packed.len(), packed_len);
        assert_eq!(two.coarse_upper_packed.len(), packed_len);
        assert_eq!(two.coarse_upper_row_start.len(), n + 1);
        assert_eq!(two.coarse_upper_row_start[n], packed_len);
        assert_eq!(
            two.coarse_factor_bytes(),
            2 * packed_len * std::mem::size_of::<f64>()
        );
    }

    #[test]
    fn two_level_auto_policy_resolves_at_empirical_crossover() {
        assert_eq!(
            TwoLevelCoarseApplyPolicy::Auto.resolve(EXPLICIT_INVERSE_AUTO_MIN_COARSE_DIMENSION - 1),
            TwoLevelCoarseApplyPolicy::FactorSolve
        );
        assert_eq!(
            TwoLevelCoarseApplyPolicy::Auto.resolve(EXPLICIT_INVERSE_AUTO_MIN_COARSE_DIMENSION),
            TwoLevelCoarseApplyPolicy::ExplicitInverse
        );
        assert_eq!(
            TwoLevelCoarseApplyPolicy::FactorSolve.resolve(usize::MAX),
            TwoLevelCoarseApplyPolicy::FactorSolve
        );
        assert_eq!(
            TwoLevelCoarseApplyPolicy::ExplicitInverse.resolve(0),
            TwoLevelCoarseApplyPolicy::ExplicitInverse
        );
    }

    #[test]
    fn two_level_graph_aggregation_builds_connected_piecewise_constant_regions() {
        // Path graph with deliberately scrambled numbering: 0-2-4-1-3-5.
        // Graph aggregation should follow connectivity rather than contiguous ids.
        let order = [0usize, 2, 4, 1, 3, 5];
        let mut rows = vec![Vec::<(usize, f64)>::new(); 6];
        for (i, row) in rows.iter_mut().enumerate() {
            row.push((i, 3.0));
        }
        for pair in order.windows(2) {
            let a = pair[0];
            let b = pair[1];
            rows[a].push((b, -1.0));
            rows[b].push((a, -1.0));
        }
        let mut row_ptr = vec![0u32];
        let mut col_idx = Vec::<u32>::new();
        let mut values = Vec::<f64>::new();
        for row in &mut rows {
            row.sort_unstable_by_key(|(col, _)| *col);
            for &(col, value) in row.iter() {
                col_idx.push(col as u32);
                values.push(value);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let a = Csr32Matrix::new(6, 6, row_ptr, col_idx, values).unwrap();
        let graph = TwoLevelBlockJacobiPreconditioner::from_csr32_graph_with_policy(
            &a,
            1,
            3,
            TwoLevelCoarseApplyPolicy::FactorSolve,
        )
        .unwrap();
        assert_eq!(graph.aggregation(), TwoLevelAggregation::Graph);
        assert_eq!(graph.aggregate_count(), 2);
        assert_eq!(graph.min_aggregate_nodes(), 3);
        assert_eq!(graph.max_aggregate_nodes(), 3);
        assert_eq!(graph.aggregate_of_node[0], graph.aggregate_of_node[2]);
        assert_eq!(graph.aggregate_of_node[2], graph.aggregate_of_node[4]);
        assert_ne!(graph.aggregate_of_node[0], graph.aggregate_of_node[1]);

        let r = vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.75];
        let mut z = vec![0.0; 6];
        graph.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
        assert!(rz.is_finite());
        assert!(rz > 0.0);
    }

    #[test]
    fn two_level_strong_graph_prefers_stronger_block_couplings() {
        // Node 0 is weakly coupled to 1 and strongly coupled to 2.  Plain
        // graph BFS visits node ids in sorted order and groups {0,1}; strong
        // graph aggregation should instead group {0,2}.
        let mut rows = vec![Vec::<(usize, f64)>::new(); 4];
        for (i, row) in rows.iter_mut().enumerate() {
            row.push((i, 10.0));
        }
        for (a, b, value) in [
            (0usize, 1usize, -0.1),
            (0, 2, -5.0),
            (1, 3, -5.0),
            (2, 3, -0.1),
        ] {
            rows[a].push((b, value));
            rows[b].push((a, value));
        }
        let mut row_ptr = vec![0u32];
        let mut col_idx = Vec::<u32>::new();
        let mut values = Vec::<f64>::new();
        for row in &mut rows {
            row.sort_unstable_by_key(|(col, _)| *col);
            for &(col, value) in row.iter() {
                col_idx.push(col as u32);
                values.push(value);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let a = Csr32Matrix::new(4, 4, row_ptr, col_idx, values).unwrap();
        let plain = TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_and_policy(
            &a,
            1,
            2,
            TwoLevelAggregation::Graph,
            TwoLevelCoarseApplyPolicy::FactorSolve,
        )
        .unwrap();
        let strong = TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_and_policy(
            &a,
            1,
            2,
            TwoLevelAggregation::StrongGraph,
            TwoLevelCoarseApplyPolicy::FactorSolve,
        )
        .unwrap();

        assert_eq!(plain.aggregate_of_node[0], plain.aggregate_of_node[1]);
        assert_ne!(plain.aggregate_of_node[0], plain.aggregate_of_node[2]);
        assert_eq!(strong.aggregation(), TwoLevelAggregation::StrongGraph);
        assert_eq!(strong.aggregate_of_node[0], strong.aggregate_of_node[2]);
        assert_ne!(strong.aggregate_of_node[0], strong.aggregate_of_node[1]);

        let r = vec![1.0, -0.5, 0.25, 2.0];
        let mut z = vec![0.0; 4];
        strong.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
        assert!(rz.is_finite());
        assert!(rz > 0.0);
    }

    #[test]
    fn two_level_parallel_smoothed_transfer_matches_serial() {
        let a = poisson_1d(96);
        let serial =
            TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_policies(
                &a,
                1,
                8,
                TwoLevelAggregation::Graph,
                TwoLevelBasis::JacobiSmoothed,
                TwoLevelCoarseApplyPolicy::FactorSolve,
                TwoLevelTransferApplyPolicy::Serial,
            )
            .unwrap();
        let parallel =
            TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_policies(
                &a,
                1,
                8,
                TwoLevelAggregation::Graph,
                TwoLevelBasis::JacobiSmoothed,
                TwoLevelCoarseApplyPolicy::FactorSolve,
                TwoLevelTransferApplyPolicy::Parallel,
            )
            .unwrap();

        assert_eq!(
            parallel.transfer_apply_policy(),
            TwoLevelTransferApplyPolicy::Parallel
        );
        let r: Vec<f64> = (0..a.nrows())
            .map(|i| ((i * 13 + 5) as f64).cos())
            .collect();
        let mut z_serial = vec![0.0; a.nrows()];
        let mut z_parallel = vec![0.0; a.nrows()];
        serial.apply(&r, &mut z_serial).unwrap();
        parallel.apply(&r, &mut z_parallel).unwrap();
        for (&serial_value, &parallel_value) in z_serial.iter().zip(&z_parallel) {
            let scale = serial_value.abs().max(parallel_value.abs()).max(1.0);
            assert!((serial_value - parallel_value).abs() <= 1.0e-12 * scale);
        }
    }

    #[test]
    fn two_level_jacobi_smoothed_basis_is_positive() {
        let a = poisson_1d(24);
        let piecewise =
            TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_policy(
                &a,
                1,
                4,
                TwoLevelAggregation::Graph,
                TwoLevelBasis::PiecewiseConstant,
                TwoLevelCoarseApplyPolicy::FactorSolve,
            )
            .unwrap();
        let smoothed =
            TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_policy(
                &a,
                1,
                4,
                TwoLevelAggregation::Graph,
                TwoLevelBasis::JacobiSmoothed,
                TwoLevelCoarseApplyPolicy::FactorSolve,
            )
            .unwrap();

        assert_eq!(smoothed.basis(), TwoLevelBasis::JacobiSmoothed);
        assert_eq!(smoothed.coarse_dimension(), piecewise.coarse_dimension());
        assert!(smoothed.smoothing_omega().is_finite());
        assert!(smoothed.smoothing_omega() > 0.0);
        assert!(smoothed.transfer_nnz() >= a.nrows());

        let r: Vec<f64> = (0..a.nrows())
            .map(|i| ((i * 7 + 3) as f64).sin())
            .collect();
        let mut z = vec![0.0; a.nrows()];
        smoothed.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
        assert!(rz.is_finite());
        assert!(rz > 0.0);
    }

    #[test]
    fn two_level_explicit_inverse_matches_factor_solve() {
        let a = poisson_1d(24);
        let factor = TwoLevelBlockJacobiPreconditioner::from_csr32(&a, 1, 4).unwrap();
        let inverse = TwoLevelBlockJacobiPreconditioner::from_csr32_with_policy(
            &a,
            1,
            4,
            TwoLevelCoarseApplyPolicy::ExplicitInverse,
        )
        .unwrap();
        let r: Vec<f64> = (0..24).map(|i| ((i * 11 + 5) % 17) as f64 - 8.0).collect();
        let mut z_factor = vec![0.0; 24];
        let mut z_inverse = vec![0.0; 24];
        factor.apply(&r, &mut z_factor).unwrap();
        inverse.apply(&r, &mut z_inverse).unwrap();

        for (a, b) in z_factor.iter().zip(&z_inverse) {
            assert!((a - b).abs() < 1.0e-10);
        }
        let rz: f64 = r.iter().zip(&z_inverse).map(|(ri, zi)| ri * zi).sum();
        assert!(rz.is_finite());
        assert!(rz > 0.0);
        assert_eq!(
            inverse.coarse_apply_policy(),
            TwoLevelCoarseApplyPolicy::ExplicitInverse
        );
        assert!(inverse.coarse_lower_packed.is_empty());
        assert!(inverse.coarse_upper_packed.is_empty());
        assert!(inverse.coarse_upper_row_start.is_empty());
        assert_eq!(
            inverse.coarse_inverse.len(),
            inverse.coarse_dimension() * inverse.coarse_dimension()
        );
    }

    #[test]
    fn single_region_matches_exact_local_solve() {
        let a = poisson_1d(4);
        let hybrid = HybridPreconditioner::from_csr32(&a, vec![vec![0, 1, 2, 3]]).unwrap();
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0; 4];
        hybrid.apply(&r, &mut z).unwrap();
        let y = a.spmv(&z).unwrap();
        for (yi, ri) in y.iter().zip(&r) {
            assert!((yi - ri).abs() < 1.0e-12);
        }
    }

    #[test]
    fn hybrid_factor_byte_estimate_matches_owned_storage() {
        let a = poisson_1d(12);
        let regions = vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]];
        let estimated = HybridPreconditioner::estimated_factor_bytes(a.nrows(), &regions).unwrap();
        let hybrid = HybridPreconditioner::from_csr32(&a, regions).unwrap();

        assert_eq!(estimated, hybrid.factor_bytes());
    }

    #[test]
    fn overlapping_weighted_schwarz_is_positive() {
        let a = poisson_1d(8);
        let hybrid =
            HybridPreconditioner::from_csr32(&a, vec![vec![0, 1, 2, 3, 4], vec![3, 4, 5, 6, 7]])
                .unwrap();
        let r = vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25];
        let mut z = vec![0.0; 8];
        hybrid.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        assert!(rz > 0.0);
        assert_eq!(hybrid.region_count(), 2);
        assert_eq!(hybrid.unique_local_dofs(), 8);
        assert!(hybrid.factor_bytes() > 0);
    }
}

#[cfg(test)]
mod rigid_body_two_level_tests {
    use super::*;

    fn identity(n: usize) -> Csr32Matrix {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        row_ptr.push(0);
        for i in 0..n {
            col_idx.push(i as u32);
            values.push(1.0);
            row_ptr.push((i + 1) as u32);
        }
        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn rigid_body_coarse_space_builds_on_cube() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let a = identity(coords.len() * 3);
        let p = RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32(&a, &coords, 8).unwrap();
        assert_eq!(p.aggregate_count(), 1);
        assert_eq!(p.coarse_dimension(), 6);
        let r = vec![1.0; a.nrows()];
        let mut z = vec![0.0; a.nrows()];
        p.apply(&r, &mut z).unwrap();
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn balanced_rigid_body_identity_is_exact() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let a = identity(coords.len() * 3);
        let p =
            BalancedRigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32(&a, &coords, 8).unwrap();
        let r: Vec<f64> = (0..a.nrows())
            .map(|i| ((i * 11 + 5) as f64).sin())
            .collect();
        let mut z = vec![0.0; a.nrows()];
        p.apply(&r, &mut z).unwrap();
        for (&zi, &ri) in z.iter().zip(&r) {
            assert!((zi - ri).abs() < 1.0e-11);
        }
        let rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
        assert!(rz > 0.0);
    }

    #[test]
    fn graph_aggregation_merges_small_remainder_island() {
        let n = 35usize;
        let mut adjacency = vec![Vec::<usize>::new(); n];
        for i in 0..(n - 1) {
            adjacency[i].push(i + 1);
            adjacency[i + 1].push(i);
        }
        let (assignment, count) = build_graph_aggregates(&adjacency, 16).unwrap();
        let mut counts = vec![0usize; count];
        for a in assignment {
            counts[a] += 1;
        }
        assert_eq!(count, 2);
        assert_eq!(counts.iter().sum::<usize>(), n);
        assert!(*counts.iter().min().unwrap() >= 16);
        assert!(*counts.iter().max().unwrap() <= 19);
    }

    #[test]
    fn graph_rigid_body_aggregation_builds_connected_cube_regions() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let edges = [
            (0usize, 1usize),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ];
        let nodes = coords.len();
        let n = nodes * 3;
        let mut rows = vec![Vec::<(usize, f64)>::new(); n];
        for node in 0..nodes {
            for c in 0..3 {
                rows[node * 3 + c].push((node * 3 + c, 4.0));
            }
        }
        for &(a_node, b_node) in &edges {
            for c in 0..3 {
                rows[a_node * 3 + c].push((b_node * 3 + c, -1.0));
                rows[b_node * 3 + c].push((a_node * 3 + c, -1.0));
            }
        }
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for row in &mut rows {
            row.sort_unstable_by_key(|e| e.0);
            for &(c, v) in row.iter() {
                col_idx.push(c as u32);
                values.push(v);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        let a = Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap();
        let p =
            RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32_graph(&a, &coords, 4).unwrap();
        assert_eq!(p.aggregation(), RigidBodyAggregation::Graph);
        assert_eq!(p.aggregate_count(), 2);
        assert!(p.min_aggregate_nodes() >= 3);
        assert!(p.max_aggregate_nodes() <= 5);
        assert_eq!(p.coarse_dimension(), 12);
        let r: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).sin()).collect();
        let mut z = vec![0.0; n];
        p.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
        assert!(rz > 0.0 && z.iter().all(|v| v.is_finite()));
    }
}

/// Recommend a contiguous aggregate size for the six-mode rigid-body coarse space.
///
/// `target_coarse_dimension` is treated as a soft upper target. Since each
/// aggregate contributes six modes, the selected aggregate size is rounded up
/// to a power of two when practical so that the resulting dense coarse system
/// remains bounded and aggregation stays simple/reproducible. The final tiny
/// aggregate is avoided because fewer than three nodes cannot support the
/// six-mode construction used by `RigidBodyTwoLevelBlockJacobiPreconditioner`.
pub fn recommend_rigid_body_aggregate_nodes(
    node_count: usize,
    target_coarse_dimension: usize,
) -> Result<usize, HybitError> {
    if node_count < 3 {
        return Err(HybitError::InvalidArgument(
            "rigid-body coarse space requires at least three nodes",
        ));
    }
    if target_coarse_dimension < 6 {
        return Err(HybitError::InvalidArgument(
            "target_coarse_dimension must be at least 6",
        ));
    }

    // At least three nodes must remain in every aggregate.  This also prevents
    // an over-large requested coarse space from creating rank-deficient tail
    // aggregates on small problems.
    let max_aggregates = (node_count / 3).max(1);
    let target_aggregates = (target_coarse_dimension / 6).max(1).min(max_aggregates);
    let minimum_nodes = node_count
        .checked_add(target_aggregates - 1)
        .ok_or(HybitError::SizeOverflow)?
        / target_aggregates;

    let mut aggregate_nodes = minimum_nodes
        .checked_next_power_of_two()
        .unwrap_or(node_count)
        .min(node_count)
        .max(3);

    // Keep the fixed-width aggregate mapping used by the current preconditioner
    // while avoiding a final aggregate of only one or two nodes.
    while aggregate_nodes < node_count {
        let tail = node_count % aggregate_nodes;
        if tail == 0 || tail >= 3 {
            break;
        }
        aggregate_nodes = aggregate_nodes
            .checked_add(1)
            .ok_or(HybitError::SizeOverflow)?;
    }

    Ok(aggregate_nodes.min(node_count))
}

#[cfg(test)]
mod structural_auto_tests {
    use super::*;

    #[test]
    fn l_angle_sized_problem_selects_512_nodes_for_1536_target() {
        let aggregate = recommend_rigid_body_aggregate_nodes(119_355, 1_536).unwrap();
        assert_eq!(aggregate, 512);
        let aggregates = 119_355_usize.div_ceil(aggregate);
        assert_eq!(aggregates * 6, 1_404);
    }

    #[test]
    fn recommendation_never_leaves_one_or_two_node_tail() {
        for nodes in 3usize..500 {
            let aggregate = recommend_rigid_body_aggregate_nodes(nodes, 96).unwrap();
            let tail = nodes % aggregate;
            assert!(aggregate >= nodes || tail == 0 || tail >= 3);
        }
    }
}
