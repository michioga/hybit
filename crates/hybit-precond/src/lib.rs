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
        let mut blocks = Vec::with_capacity((n + block_size - 1) / block_size);
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
                    for k in 0..i {
                        sum -= block.lower[i * n + k] * z_block[k];
                    }
                    z_block[i] = sum / block.lower[i * n + i];
                }
                for i in (0..n).rev() {
                    let mut sum = z_block[i];
                    for k in (i + 1)..n {
                        sum -= block.lower[k * n + i] * z_block[k];
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
}

#[inline]
fn packed_lower_len(n: usize) -> Result<usize, HybitError> {
    n.checked_add(1)
        .and_then(|np1| n.checked_mul(np1))
        .map(|v| v / 2)
        .ok_or(HybitError::SizeOverflow)
}

#[derive(Clone, Debug)]
pub struct TwoLevelBlockJacobiPreconditioner {
    n: usize,
    dofs_per_node: usize,
    aggregate_nodes: usize,
    aggregate_count: usize,
    coarse_dimension: usize,
    base: BlockJacobiPreconditioner,
    coarse_lower: Vec<f64>,
    scratch: RefCell<CoarseScratch>,
    factor_bytes: usize,
}

impl TwoLevelBlockJacobiPreconditioner {
    /// Build an SPD additive two-level preconditioner
    ///
    ///     M^-1 = B^-1 + Z (Z^T A Z)^-1 Z^T,
    ///
    /// where `B^-1` is contiguous block Jacobi and `Z` contains piecewise
    /// constant vector-FEM aggregate modes. Each aggregate contains
    /// `aggregate_nodes` consecutive nodes, and every displacement/component
    /// receives its own coarse basis vector. The construction is deliberately
    /// geometry-free: it only requires node-major contiguous DOFs.
    pub fn from_csr32(
        matrix: &Csr32Matrix,
        dofs_per_node: usize,
        aggregate_nodes: usize,
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
        let aggregate_count = node_count
            .checked_add(aggregate_nodes - 1)
            .ok_or(HybitError::SizeOverflow)?
            / aggregate_nodes;
        let coarse_dimension = aggregate_count
            .checked_mul(dofs_per_node)
            .ok_or(HybitError::SizeOverflow)?;
        let coarse_len = coarse_dimension
            .checked_mul(coarse_dimension)
            .ok_or(HybitError::SizeOverflow)?;
        let mut coarse = vec![0.0f64; coarse_len];

        #[inline]
        fn coarse_index(dof: usize, dofs_per_node: usize, aggregate_nodes: usize) -> usize {
            let node = dof / dofs_per_node;
            let component = dof % dofs_per_node;
            (node / aggregate_nodes) * dofs_per_node + component
        }

        // Galerkin coarse operator E = Z^T A Z.  Since each fine DOF belongs
        // to exactly one piecewise-constant coarse mode, this is just a sparse
        // accumulation from fine CSR entries into a small dense matrix.
        for row in 0..n {
            let cr = coarse_index(row, dofs_per_node, aggregate_nodes);
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let col = matrix.col_idx()[p] as usize;
                let cc = coarse_index(col, dofs_per_node, aggregate_nodes);
                coarse[cr * coarse_dimension + cc] += matrix.values()[p];
            }
        }

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

        let mut coarse_lower = vec![0.0f64; coarse_len];
        let pivot_tol = 1.0e-14 * scale.max(1.0);
        for i in 0..coarse_dimension {
            for j in 0..=i {
                let mut sum = coarse[i * coarse_dimension + j];
                for k in 0..j {
                    sum -= coarse_lower[i * coarse_dimension + k]
                        * coarse_lower[j * coarse_dimension + k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= pivot_tol {
                        return Err(HybitError::NumericalBreakdown(
                            "aggregation coarse Cholesky encountered a non-positive pivot",
                        ));
                    }
                    coarse_lower[i * coarse_dimension + i] = sum.sqrt();
                } else {
                    coarse_lower[i * coarse_dimension + j] =
                        sum / coarse_lower[j * coarse_dimension + j];
                }
            }
        }

        let scratch = RefCell::new(CoarseScratch {
            rhs: vec![0.0; coarse_dimension],
            sol: vec![0.0; coarse_dimension],
        });
        let factor_bytes = base
            .factor_bytes()
            .checked_add(coarse_lower.len() * std::mem::size_of::<f64>())
            .and_then(|v| v.checked_add(2 * coarse_dimension * std::mem::size_of::<f64>()))
            .ok_or(HybitError::SizeOverflow)?;

        Ok(Self {
            n,
            dofs_per_node,
            aggregate_nodes,
            aggregate_count,
            coarse_dimension,
            base,
            coarse_lower,
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
    pub fn coarse_dimension(&self) -> usize {
        self.coarse_dimension
    }
    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }
    pub fn base_factor_bytes(&self) -> usize {
        self.base.factor_bytes()
    }
    pub fn coarse_factor_bytes(&self) -> usize {
        self.coarse_lower.len() * std::mem::size_of::<f64>()
    }

    #[inline]
    fn coarse_index(&self, dof: usize) -> usize {
        let node = dof / self.dofs_per_node;
        let component = dof % self.dofs_per_node;
        (node / self.aggregate_nodes) * self.dofs_per_node + component
    }

    fn solve_coarse(&self, rhs: &[f64], sol: &mut [f64]) {
        let n = self.coarse_dimension;
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(sol.len(), n);
        for i in 0..n {
            let mut sum = rhs[i];
            for k in 0..i {
                sum -= self.coarse_lower[i * n + k] * sol[k];
            }
            sol[i] = sum / self.coarse_lower[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = sol[i];
            for k in (i + 1)..n {
                sum -= self.coarse_lower[k * n + i] * sol[k];
            }
            sol[i] = sum / self.coarse_lower[i * n + i];
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

        // Coarse/global SPD term Z E^-1 Z^T.
        let mut scratch = self.scratch.borrow_mut();
        scratch.rhs.fill(0.0);
        for (dof, &ri) in r.iter().enumerate() {
            let ci = self.coarse_index(dof);
            scratch.rhs[ci] += ri;
        }
        let CoarseScratch { rhs, sol } = &mut *scratch;
        self.solve_coarse(rhs, sol);
        for (dof, zi) in z.iter_mut().enumerate() {
            *zi += sol[self.coarse_index(dof)];
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
            for k in 0..i {
                sum -= self.coarse_lower_packed[i_base + k] * sol[k];
            }
            sol[i] = sum / self.coarse_lower_packed[i_base + i];
        }

        // Backward solve L^T x = y.
        for i in (0..n).rev() {
            let i_base = self.coarse_row_start[i];
            let mut sum = sol[i];
            for k in (i + 1)..n {
                let k_base = self.coarse_row_start[k];
                sum -= self.coarse_lower_packed[k_base + i] * sol[k];
            }
            sol[i] = sum / self.coarse_lower_packed[i_base + i];
        }
    }

    /// Apply only the Galerkin coarse correction
    ///
    ///     C r = Z (Z^T A Z)^-1 Z^T r.
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
                let CoarseScratch { rhs, sol } = &mut *scratch;
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
        let CoarseScratch { rhs, sol } = &mut *scratch;
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
///     C = Z (Z^T A Z)^-1 Z^T,
///     P = I - C A,
///
/// this applies
///
///     M_bal^-1 = P B^-1 P^T + C
///              = (I - C A) B^-1 (I - A C) + C,
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

fn build_structural_node_adjacency(
    matrix: &Csr32Matrix,
    node_count: usize,
) -> Result<Vec<Vec<usize>>, HybitError> {
    let expected = node_count.checked_mul(3).ok_or(HybitError::SizeOverflow)?;
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(HybitError::DimensionMismatch {
            expected: matrix.nrows(),
            actual: expected,
        });
    }
    let mut adjacency = Vec::with_capacity(node_count);
    for node in 0..node_count {
        let mut neighbors = Vec::<usize>::new();
        for component in 0..3 {
            let row = node * 3 + component;
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let other = matrix.col_idx()[p] as usize / 3;
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
        let CoarseScratch { rhs, sol } = &mut *scratch;
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
            let CoarseScratch { rhs, sol } = &mut *scratch;
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
        let mut dense = vec![0.0; n * n];
        for (li, &global_row) in canonical.iter().enumerate() {
            let start = matrix.row_ptr()[global_row] as usize;
            let end = matrix.row_ptr()[global_row + 1] as usize;
            for p in start..end {
                let global_col = matrix.col_idx()[p] as usize;
                if let Some(&lj) = local_of.get(&global_col) {
                    dense[li * n + lj] += matrix.values()[p];
                }
            }
        }

        // Local direct correction is currently an SPD path. Require the local
        // principal matrix to be numerically symmetric before factorization.
        let mut scale = 0.0f64;
        for &v in &dense {
            scale = scale.max(v.abs());
        }
        let symmetry_tol = 1.0e-11 * scale.max(1.0);
        for i in 0..n {
            for j in 0..i {
                if (dense[i * n + j] - dense[j * n + i]).abs() > symmetry_tol {
                    return Err(HybitError::InvalidMatrix(
                        "local Cholesky region is not symmetric",
                    ));
                }
            }
        }

        let mut lower = vec![0.0; n * n];
        let pivot_tol = 1.0e-14 * scale.max(1.0);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = dense[i * n + j];
                for k in 0..j {
                    sum -= lower[i * n + k] * lower[j * n + k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= pivot_tol {
                        return Err(HybitError::NumericalBreakdown(
                            "local Cholesky encountered a non-positive pivot",
                        ));
                    }
                    lower[i * n + i] = sum.sqrt();
                } else {
                    lower[i * n + j] = sum / lower[j * n + j];
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
        for i in 0..n {
            let mut sum = rhs[i];
            for k in 0..i {
                sum -= self.lower[i * n + k] * out[k];
            }
            out[i] = sum / self.lower[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = out[i];
            for k in (i + 1)..n {
                sum -= self.lower[k * n + i] * out[k];
            }
            out[i] = sum / self.lower[i * n + i];
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
        let aggregates = (119_355 + aggregate - 1) / aggregate;
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
