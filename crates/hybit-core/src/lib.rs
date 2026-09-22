use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq)]
pub enum HybitError {
    InvalidMatrix(&'static str),
    InvalidArgument(&'static str),
    DimensionMismatch { expected: usize, actual: usize },
    MissingDiagonal { row: usize },
    ZeroDiagonal { row: usize },
    SizeOverflow,
    NumericalBreakdown(&'static str),
    NotConverged { iterations: usize, residual: f64 },
}

impl Display for HybitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMatrix(msg) => write!(f, "invalid matrix: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::MissingDiagonal { row } => write!(f, "missing diagonal entry at row {row}"),
            Self::ZeroDiagonal { row } => write!(f, "zero diagonal entry at row {row}"),
            Self::SizeOverflow => write!(
                f,
                "matrix or index size exceeds the selected representation"
            ),
            Self::NumericalBreakdown(msg) => write!(f, "numerical breakdown: {msg}"),
            Self::NotConverged {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "solver did not converge after {iterations} iterations; residual={residual:e}"
                )
            }
        }
    }
}

impl Error for HybitError {}

pub trait LinearOperator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError>;
}

pub trait Preconditioner {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixBackend {
    Csr32,
    Abtm,
    MatrixFree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    Pcg,
    Minres,
    Gmres,
    Bicgstab,
    Hybrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreconditionerKind {
    None,
    Jacobi,
    BlockJacobi,
    LocalDirect,
    Hybrid,
    RigidBodyTwoLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    Converged,
    MaxIterations,
    Breakdown,
}

#[derive(Clone, Copy, Debug)]
pub struct SolverOptions {
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: usize,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-8,
            absolute_tolerance: 0.0,
            max_iterations: 1000,
        }
    }
}

impl SolverOptions {
    pub fn validate(&self) -> Result<(), HybitError> {
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(HybitError::InvalidArgument(
                "relative_tolerance must be finite and >= 0",
            ));
        }
        if !self.absolute_tolerance.is_finite() || self.absolute_tolerance < 0.0 {
            return Err(HybitError::InvalidArgument(
                "absolute_tolerance must be finite and >= 0",
            ));
        }
        if self.relative_tolerance == 0.0 && self.absolute_tolerance == 0.0 {
            return Err(HybitError::InvalidArgument(
                "at least one tolerance must be > 0",
            ));
        }
        if self.max_iterations == 0 {
            return Err(HybitError::InvalidArgument("max_iterations must be > 0"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct HybridEscalationStageReport {
    /// One-based selective-direct escalation stage number.
    pub stage: usize,
    /// PCG iterations spent with the preconditioner built for this stage.
    pub iterations: usize,
    /// Residual norm at the beginning of the restarted PCG stage.
    pub initial_residual: f64,
    /// Residual norm at the end of the restarted PCG stage.
    pub final_residual: f64,
    /// `final_residual / initial_residual`, or zero when the initial residual is zero.
    pub residual_ratio: f64,
    /// Number of local direct regions active in this stage.
    pub local_direct_regions: usize,
    /// Number of unique global DOFs covered by the stage preconditioner.
    pub unique_local_factor_dofs: usize,
    /// Persistent local-direct bytes owned by the stage preconditioner.
    pub local_factor_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct SolveReport {
    pub status: SolveStatus,
    pub solver: SolverKind,
    pub preconditioner: PreconditionerKind,
    pub backend: MatrixBackend,
    pub iterations: usize,
    pub initial_residual: f64,
    pub final_residual: f64,
    pub relative_residual: f64,
    pub setup_seconds: f64,
    pub solve_seconds: f64,
    /// Structural matrix analysis cost charged to this solve.
    pub analysis_seconds: f64,
    /// Reusable baseline preparation cost (Jacobi, ABTM topology and Krylov workspace).
    pub prepare_seconds: f64,
    /// Initial Jacobi-PCG probe time.
    pub probe_seconds: f64,
    /// Residual/risk analysis and ABTM topology-region construction time.
    pub diagnostics_seconds: f64,
    /// Dense local Cholesky construction time.
    pub local_factor_seconds: f64,
    /// Restarted PCG time after escalation.
    pub restart_seconds: f64,
    pub escalations: usize,
    /// Per-stage telemetry for selective-direct strengthening restarts.
    ///
    /// This vector is empty for non-hybrid solves and for prepared solves that
    /// directly reuse an already-built hybrid preconditioner.
    pub escalation_stages: Vec<HybridEscalationStageReport>,
    pub probe_iterations: usize,
    pub probe_final_residual: f64,
    /// Core DOFs identified as numerically difficult before halo expansion.
    pub hard_dofs: usize,
    /// Number of local direct subdomains.
    pub local_direct_regions: usize,
    /// Largest local factor order after overlap expansion.
    pub largest_local_region: usize,
    /// Sum of local factor orders. Overlapped DOFs are counted once per factor.
    pub local_factor_dofs: usize,
    /// Unique global DOFs covered by at least one local factor.
    pub unique_local_factor_dofs: usize,
    /// Bytes owned by local Cholesky factors, indices, weights, multiplicity,
    /// and reusable local-direct scratch storage.
    pub local_factor_bytes: usize,
    /// Coarse-space dimension for the optional geometry-free algebraic two-level
    /// base used by the generic hybrid path. Zero when disabled/not constructed.
    pub algebraic_coarse_dimension: usize,
    /// Persistent bytes owned by the algebraic two-level base preconditioner.
    pub algebraic_coarse_factor_bytes: usize,
    /// Requested/target nodes per algebraic coarse aggregate. Zero when disabled.
    /// Graph aggregation can produce variable-sized regions around this target.
    pub algebraic_coarse_aggregate_nodes: usize,
    /// Time spent constructing the algebraic two-level base.
    pub algebraic_coarse_seconds: f64,
    /// Configured persistent local-direct memory budget in bytes.
    ///
    /// This is zero for solver paths that do not use the generic hybrid
    /// local-direct controller.
    pub local_factor_budget_bytes: usize,
    /// Number of candidate local-direct regions rejected because admitting
    /// them would exceed `local_factor_budget_bytes`.
    pub local_factor_regions_skipped_for_budget: usize,
    /// True when at least one candidate region was rejected by the local
    /// factor memory budget.
    pub local_factor_budget_limited: bool,
    /// Number of topology halo layers requested for each hard region.
    pub overlap_layers: usize,
    /// True when an already-built local direct preconditioner was reused.
    pub preconditioner_reused: bool,
    /// 1-based solve number within a prepared context.
    pub solve_sequence: usize,
    /// Bytes reserved for reusable PCG work vectors.
    pub krylov_workspace_bytes: usize,
}

impl SolveReport {
    pub fn converged(&self) -> bool {
        self.status == SolveStatus::Converged
    }
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn l2_norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}
