//! HyBIT — Autonomous Hybrid Sparse Solver.
//!
//! HyBIT 0.5.0 provides a Rust facade for real SPD sparse systems using
//! CSR32 input, PCG, adaptive selective local Cholesky correction, weighted
//! overlapping Schwarz, and reusable analyze/prepare/solve-many contexts.
//!
//! # Example
//!
//! ```
//! use hybit::{Csr32Matrix, HybitSolver};
//!
//! let a = Csr32Matrix::new(
//!     3,
//!     3,
//!     vec![0, 2, 5, 7],
//!     vec![0, 1, 0, 1, 2, 1, 2],
//!     vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
//! )?;
//! let b = vec![1.0, 0.0, 1.0];
//! let mut x = vec![0.0; 3];
//! let solver = HybitSolver::new();
//! let report = solver.solve_csr32(&a, &b, &mut x)?;
//! assert!(report.relative_residual < 1.0e-8);
//! # Ok::<(), hybit::HybitError>(())
//! ```
//!
//! The automatic path is currently experimental and restricted to real SPD
//! systems with PCG. See the repository README for current limitations.

pub use hybit_auto::{BackendPolicy, HybridOptions, HybitAnalysis, HybitPreparedSystem, HybitSolver};
pub use hybit_core::{
    HybitError, LinearOperator, MatrixBackend, Preconditioner, PreconditionerKind,
    SolveReport, SolveStatus, SolverKind, SolverOptions,
};
pub use hybit_krylov::{pcg, pcg_with_workspace, KrylovOutcome, PcgWorkspace};
pub use hybit_matrix::{
    analyze_csr32, AbtmConfig, AbtmMatrix, AbtmStats, Csr32Matrix, MatrixProfile,
    DofMask, TileDesc, TileKind, TILE_WIDTH,
};
pub use hybit_precond::{HybridPreconditioner, IdentityPreconditioner, JacobiPreconditioner, LocalCholeskyRegion};

pub fn solve(matrix: &Csr32Matrix, b: &[f64]) -> Result<(Vec<f64>, SolveReport), HybitError> {
    let solver = HybitSolver::new();
    let mut x = vec![0.0; matrix.ncols()];
    let report = solver.solve_csr32(matrix, b, &mut x)?;
    Ok((x, report))
}
