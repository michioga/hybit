use std::collections::VecDeque;
use std::time::Instant;

use hybit_core::{
    l2_norm, HybitError, HybridEscalationStageReport, LinearOperator, MatrixBackend,
    Preconditioner, PreconditionerKind, SolveReport, SolveStatus, SolverKind, SolverOptions,
};
use hybit_krylov::{
    parallel_vector_worker_count, pcg_continue_with_workspace, pcg_start_with_workspace,
    pcg_with_workspace, pcg_with_workspace_parallel_vectors, KrylovOutcome, PcgSession,
    PcgWorkspace,
};
use hybit_matrix::{
    analyze_csr32, AbtmConfig, AbtmMatrix, Csr32Matrix, DofMask, MatrixProfile,
    ParallelCsr32Operator,
};
use hybit_precond::{
    recommend_rigid_body_aggregate_nodes, HybridPreconditioner, JacobiPreconditioner,
    ParallelRigidBodyTwoLevelPreconditioner, RigidBodyTwoLevelBlockJacobiPreconditioner,
    TwoLevelBlockJacobiPreconditioner,
};
pub use hybit_precond::{
    RigidBodyAggregation, TwoLevelAggregation, TwoLevelBasis, TwoLevelCoarseApplyPolicy,
    TwoLevelTransferApplyPolicy, TwoLevelTransferOptions, TwoLevelTransferStoragePolicy,
    TwoLevelTransferValueStoragePolicy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendPolicy {
    Auto,
    Csr32,
    Abtm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalFactorSelectionPolicy {
    /// Preserve the diagnostic candidate order and accept regions greedily
    /// until the persistent local-factor memory budget is exhausted.
    CandidateOrder,
    /// Rank new regions by uncovered residual L2 energy per incremental
    /// persistent factor byte and greedily accept the best remaining candidate.
    BenefitPerByte,
    /// Rank new regions by uncovered Jacobi-preconditioned residual energy
    /// `sum(r_i^2 / A_ii)` per incremental persistent factor byte.
    ///
    /// This is a diagonal approximation to the correction energy `r^T A^-1 r`
    /// and therefore accounts for local stiffness scale that raw residual
    /// magnitude alone cannot distinguish.
    JacobiEnergyPerByte,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlgebraicCoarseOptions {
    /// Enable a geometry-free algebraic two-level base preconditioner from the
    /// first Krylov iteration. Selective-direct local corrections may still be
    /// added later if the coarse-base controller probe shows poor progress.
    pub enabled: bool,
    /// Number of contiguous DOFs per node/component block. Structural solids
    /// normally use 3. The matrix dimension must be divisible by this value.
    pub dofs_per_node: usize,
    /// Soft upper target for the algebraic coarse-space dimension.
    pub target_coarse_dimension: usize,
    /// How node-major blocks are grouped into piecewise-constant aggregates.
    /// `Contiguous` preserves the historical generic path; `Graph` follows
    /// block sparsity with deterministic breadth-first regions; `StrongGraph`
    /// prioritizes normalized block coupling strength while growing regions.
    pub aggregation: TwoLevelAggregation,
    /// Coarse transfer basis built on top of the chosen aggregates.
    /// `PiecewiseConstant` preserves the historical tentative basis;
    /// `JacobiSmoothed` applies one damped Jacobi step and forms `P^T A P`.
    pub basis: TwoLevelBasis,
    /// How the sparse smoothed transfer is applied. This only affects
    /// `JacobiSmoothed`; the piecewise-constant basis uses its implicit mapping.
    pub transfer_apply_policy: TwoLevelTransferApplyPolicy,
    /// Persistent index representation for the sparse smoothed transfer.
    /// `Wide` preserves the historical layout; `Compact` uses `u32` row offsets
    /// and `u16` coarse columns when representable; `Auto` chooses compact when
    /// possible and otherwise falls back to wide storage.
    pub transfer_storage_policy: TwoLevelTransferStoragePolicy,
    /// Persistent floating-point representation for smoothed transfer weights.
    /// `F64` preserves the current numerical path; `F32` stores quantized
    /// weights but promotes them back to `f64` during application.
    pub transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy,
    /// How the dense coarse inverse is applied inside every PCG iteration.
    ///
    /// `Auto` is the generic default and currently uses an empirical
    /// coarse-dimension crossover. Callers with unusually short solve phases
    /// can still force `FactorSolve` to avoid explicit-inverse setup overhead.
    pub apply_policy: TwoLevelCoarseApplyPolicy,
}

impl Default for AlgebraicCoarseOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            dofs_per_node: 3,
            target_coarse_dimension: 1536,
            aggregation: TwoLevelAggregation::Contiguous,
            basis: TwoLevelBasis::PiecewiseConstant,
            transfer_apply_policy: TwoLevelTransferApplyPolicy::Parallel,
            transfer_storage_policy: TwoLevelTransferStoragePolicy::Wide,
            transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy::Auto,
            apply_policy: TwoLevelCoarseApplyPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HybridOptions {
    pub enabled: bool,
    /// Initial controller stage length. Uses algebraic coarse when explicitly
    /// enabled, otherwise Jacobi. The live PCG recurrence is preserved across
    /// this boundary if the preconditioner remains unchanged.
    pub probe_iterations: usize,
    /// Maximum number of local-direct strengthening restarts for one RHS.
    pub max_escalations: usize,
    /// PCG iteration budget for each non-final escalation stage.
    ///
    /// The final allowed escalation receives all remaining iterations.
    pub escalation_stage_iterations: usize,
    pub escalation_residual_ratio: f64,
    pub coupling_risk_threshold: f64,
    pub scale_jump_threshold: f64,
    pub residual_seed_fraction: f64,
    pub max_local_region_size: usize,
    pub max_local_regions: usize,
    /// Policy used to choose newly discovered local-direct regions when the
    /// persistent factor-memory budget cannot hold every candidate.
    pub local_factor_selection: LocalFactorSelectionPolicy,
    /// Persistent memory budget for selective local-direct state.
    ///
    /// This limits the bytes reported by `HybridPreconditioner::factor_bytes()`.
    /// When the budget is constrained, newly discovered regions are selected
    /// according to `local_factor_selection`.
    /// Factorization setup temporaries are not included in this checkpoint.
    pub max_local_factor_bytes: usize,
    /// Optional geometry-free two-level base combined additively with the
    /// selective-direct local corrections. Disabled by default because generic
    /// matrices do not necessarily have node-major blocked DOF ordering.
    pub algebraic_coarse: AlgebraicCoarseOptions,
    pub overlap_layers: usize,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            probe_iterations: 12,
            max_escalations: 3,
            escalation_stage_iterations: 24,
            escalation_residual_ratio: 0.50,
            coupling_risk_threshold: 0.90,
            scale_jump_threshold: 100.0,
            residual_seed_fraction: 0.25,
            max_local_region_size: 128,
            max_local_regions: 8,
            local_factor_selection: LocalFactorSelectionPolicy::JacobiEnergyPerByte,
            max_local_factor_bytes: 64 * 1024 * 1024,
            algebraic_coarse: AlgebraicCoarseOptions::default(),
            overlap_layers: 1,
        }
    }
}

/// Execution policy for the fine-grid CSR SpMV used by structural PCG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralSpmvPolicy {
    /// Use parallel CSR only for sufficiently large CSR systems.
    Auto,
    /// Always use the ordinary serial CSR/selected backend operator.
    Serial,
    /// Force Rayon-parallel CSR SpMV. Requires the CSR32 backend.
    Parallel,
}

/// Auto switches to parallel CSR after this many nonzeros. Small matrices stay
/// serial because Rayon scheduling overhead dominates there.
pub const STRUCTURAL_PARALLEL_SPMV_MIN_NNZ: usize = 1_000_000;

/// Execution policy for the structural rigid-body two-level preconditioner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralPreconditionerPolicy {
    /// Use the parallel fine/coarse transfer kernels only for large structural systems.
    Auto,
    /// Use the serial preconditioner implementation.
    Serial,
    /// Force parallel block-Jacobi, restriction, and prolongation kernels.
    Parallel,
}

/// Auto switches to the parallel structural preconditioner after this many
/// nonzeros. The dense coarse triangular solve itself remains serial.
pub const STRUCTURAL_PARALLEL_PRECONDITIONER_MIN_NNZ: usize = 1_000_000;

/// Execution policy for dense vector kernels inside structural PCG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralPcgVectorPolicy {
    /// Use the parallel/fused vector path only for sufficiently large systems
    /// when the shared Rayon pool has enough workers to amortize reductions.
    Auto,
    /// Use the established serial PCG vector kernels.
    Serial,
    /// Force parallel/fused PCG vector kernels.
    Parallel,
}

/// Auto threshold for the parallel/fused PCG dense-vector path.
pub const STRUCTURAL_PARALLEL_PCG_VECTOR_MIN_N: usize = 131_072;
/// r24 showed a 2-worker pool can be slower than serial vector kernels.
pub const STRUCTURAL_PARALLEL_PCG_VECTOR_MIN_THREADS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct StructuralOptions {
    /// Soft upper target for the dense rigid-body coarse space dimension.
    pub target_coarse_dimension: usize,
    /// How structural nodes are grouped into rigid-body coarse aggregates.
    pub aggregation: RigidBodyAggregation,
    /// Fine-grid SpMV execution policy for structural PCG.
    pub spmv_policy: StructuralSpmvPolicy,
    /// Execution policy for the rigid-body two-level preconditioner.
    pub preconditioner_policy: StructuralPreconditionerPolicy,
    /// Execution policy for dense vector kernels inside structural PCG.
    pub pcg_vector_policy: StructuralPcgVectorPolicy,
}

impl Default for StructuralOptions {
    fn default() -> Self {
        Self {
            target_coarse_dimension: 1536,
            aggregation: RigidBodyAggregation::Auto,
            spmv_policy: StructuralSpmvPolicy::Auto,
            preconditioner_policy: StructuralPreconditionerPolicy::Auto,
            pcg_vector_policy: StructuralPcgVectorPolicy::Auto,
        }
    }
}

impl StructuralOptions {
    pub fn validate(&self) -> Result<(), HybitError> {
        if self.target_coarse_dimension < 6 {
            return Err(HybitError::InvalidArgument(
                "target_coarse_dimension must be at least 6",
            ));
        }
        Ok(())
    }
}

impl HybridOptions {
    pub fn validate(&self) -> Result<(), HybitError> {
        if self.probe_iterations == 0 {
            return Err(HybitError::InvalidArgument("probe_iterations must be > 0"));
        }
        if self.max_escalations == 0 {
            return Err(HybitError::InvalidArgument("max_escalations must be > 0"));
        }
        if self.escalation_stage_iterations == 0 {
            return Err(HybitError::InvalidArgument(
                "escalation_stage_iterations must be > 0",
            ));
        }
        if !self.escalation_residual_ratio.is_finite() || self.escalation_residual_ratio <= 0.0 {
            return Err(HybitError::InvalidArgument(
                "escalation_residual_ratio must be finite and > 0",
            ));
        }
        if !self.coupling_risk_threshold.is_finite() || self.coupling_risk_threshold < 0.0 {
            return Err(HybitError::InvalidArgument(
                "coupling_risk_threshold must be finite and >= 0",
            ));
        }
        if !self.scale_jump_threshold.is_finite() || self.scale_jump_threshold < 1.0 {
            return Err(HybitError::InvalidArgument(
                "scale_jump_threshold must be finite and >= 1",
            ));
        }
        if !self.residual_seed_fraction.is_finite()
            || self.residual_seed_fraction <= 0.0
            || self.residual_seed_fraction > 1.0
        {
            return Err(HybitError::InvalidArgument(
                "residual_seed_fraction must be in (0, 1]",
            ));
        }
        if self.max_local_region_size == 0 || self.max_local_regions == 0 {
            return Err(HybitError::InvalidArgument(
                "local region limits must be > 0",
            ));
        }
        if self.max_local_factor_bytes == 0 {
            return Err(HybitError::InvalidArgument(
                "max_local_factor_bytes must be > 0",
            ));
        }
        if self.algebraic_coarse.enabled {
            if self.algebraic_coarse.dofs_per_node == 0 {
                return Err(HybitError::InvalidArgument(
                    "algebraic coarse dofs_per_node must be > 0",
                ));
            }
            if self.algebraic_coarse.target_coarse_dimension < self.algebraic_coarse.dofs_per_node {
                return Err(HybitError::InvalidArgument(
                    "algebraic coarse target dimension must be >= dofs_per_node",
                ));
            }
        }
        if self.overlap_layers > 8 {
            return Err(HybitError::InvalidArgument("overlap_layers must be <= 8"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct HybitSolver {
    options: SolverOptions,
    backend_policy: BackendPolicy,
    hybrid_options: HybridOptions,
    structural_options: StructuralOptions,
}

impl Default for HybitSolver {
    fn default() -> Self {
        Self {
            options: SolverOptions::default(),
            backend_policy: BackendPolicy::Auto,
            hybrid_options: HybridOptions::default(),
            structural_options: StructuralOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HybitAnalysis {
    profile: MatrixProfile,
    backend: MatrixBackend,
    structure_signature: u64,
    value_signature: u64,
    analysis_seconds: f64,
}

impl HybitAnalysis {
    pub fn profile(&self) -> &MatrixProfile {
        &self.profile
    }
    pub fn backend(&self) -> MatrixBackend {
        self.backend
    }
    pub fn analysis_seconds(&self) -> f64 {
        self.analysis_seconds
    }
}

#[derive(Debug)]
pub struct HybitPreparedSystem {
    options: SolverOptions,
    hybrid_options: HybridOptions,
    backend: MatrixBackend,
    structure_signature: u64,
    value_signature: u64,
    analysis_seconds: f64,
    prepare_seconds: f64,
    jacobi: JacobiPreconditioner,
    abtm: Option<AbtmMatrix>,
    hybrid: Option<HybridPreconditioner>,
    algebraic_coarse: Option<TwoLevelBlockJacobiPreconditioner>,
    algebraic_coarse_aggregate_nodes: usize,
    workspace: PcgWorkspace,
    solve_sequence: usize,
}

struct AlgebraicTwoLevelHybrid<'a> {
    coarse: &'a TwoLevelBlockJacobiPreconditioner,
    local: &'a HybridPreconditioner,
}

impl Preconditioner for AlgebraicTwoLevelHybrid<'_> {
    fn len(&self) -> usize {
        self.coarse.len()
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        self.coarse.apply(r, z)?;
        self.local.add_local_correction(r, z)
    }
}

fn recommend_algebraic_aggregate_nodes(
    matrix_rows: usize,
    options: AlgebraicCoarseOptions,
) -> Result<usize, HybitError> {
    if matrix_rows % options.dofs_per_node != 0 {
        return Err(HybitError::InvalidArgument(
            "matrix dimension must be divisible by algebraic coarse dofs_per_node",
        ));
    }
    let node_count = matrix_rows / options.dofs_per_node;
    let max_aggregates = (options.target_coarse_dimension / options.dofs_per_node).max(1);
    Ok(node_count.div_ceil(max_aggregates).max(1))
}

impl HybitPreparedSystem {
    pub fn backend(&self) -> MatrixBackend {
        self.backend
    }
    pub fn analysis_seconds(&self) -> f64 {
        self.analysis_seconds
    }
    pub fn prepare_seconds(&self) -> f64 {
        self.prepare_seconds
    }
    pub fn solve_count(&self) -> usize {
        self.solve_sequence
    }
    pub fn krylov_workspace_bytes(&self) -> usize {
        self.workspace.bytes()
    }
    pub fn has_cached_hybrid(&self) -> bool {
        self.hybrid.is_some()
    }

    fn ensure_algebraic_coarse(&mut self, matrix: &Csr32Matrix) -> Result<f64, HybitError> {
        if !self.hybrid_options.algebraic_coarse.enabled || self.algebraic_coarse.is_some() {
            return Ok(0.0);
        }

        let aggregate_nodes = recommend_algebraic_aggregate_nodes(
            matrix.nrows(),
            self.hybrid_options.algebraic_coarse,
        )?;
        let start = Instant::now();
        let coarse =
            TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_transfer_options(
                matrix,
                self.hybrid_options.algebraic_coarse.dofs_per_node,
                aggregate_nodes,
                self.hybrid_options.algebraic_coarse.aggregation,
                self.hybrid_options.algebraic_coarse.basis,
                self.hybrid_options.algebraic_coarse.apply_policy,
                TwoLevelTransferOptions {
                    apply_policy: self.hybrid_options.algebraic_coarse.transfer_apply_policy,
                    storage_policy: self.hybrid_options.algebraic_coarse.transfer_storage_policy,
                    value_storage_policy: self
                        .hybrid_options
                        .algebraic_coarse
                        .transfer_value_storage_policy,
                },
            )?;
        let elapsed = start.elapsed().as_secs_f64();
        self.algebraic_coarse_aggregate_nodes = aggregate_nodes;
        self.algebraic_coarse = Some(coarse);
        Ok(elapsed)
    }

    fn validate_matrix(&self, matrix: &Csr32Matrix) -> Result<(), HybitError> {
        let (structure, values) = matrix_signatures(matrix);
        if structure != self.structure_signature {
            return Err(HybitError::InvalidArgument(
                "prepared context matrix structure changed; analyze and prepare again",
            ));
        }
        if values != self.value_signature {
            return Err(HybitError::InvalidArgument("prepared context matrix values changed; prepare again before reusing local factors"));
        }
        Ok(())
    }

    pub fn solve(
        &mut self,
        matrix: &Csr32Matrix,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<SolveReport, HybitError> {
        self.validate_matrix(matrix)?;
        if b.len() != matrix.nrows() {
            return Err(HybitError::DimensionMismatch {
                expected: matrix.nrows(),
                actual: b.len(),
            });
        }
        if x.len() != matrix.ncols() {
            return Err(HybitError::DimensionMismatch {
                expected: matrix.ncols(),
                actual: x.len(),
            });
        }
        self.solve_sequence += 1;
        let sequence = self.solve_sequence;
        let charge_context_setup = sequence == 1;

        // Once a difficult subspace has been learned for this matrix, subsequent
        // RHS vectors reuse the exact local Cholesky factors and skip probe/
        // diagnostics/factorization entirely.
        if let Some(hybrid) = self.hybrid.as_ref() {
            let start = Instant::now();
            let outcome = run_hybrid_pcg(
                operator_for_backend(matrix, self.abtm.as_ref(), self.backend),
                self.algebraic_coarse.as_ref(),
                hybrid,
                b,
                x,
                self.options,
                &mut self.workspace,
            )?;
            let elapsed = start.elapsed().as_secs_f64();
            let metrics = ReportMetrics {
                analysis_seconds: if charge_context_setup {
                    self.analysis_seconds
                } else {
                    0.0
                },
                prepare_seconds: if charge_context_setup {
                    self.prepare_seconds
                } else {
                    0.0
                },
                restart_seconds: elapsed,
                local_direct_regions: hybrid.region_count(),
                largest_local_region: hybrid.largest_region(),
                local_factor_dofs: hybrid.local_dofs(),
                unique_local_factor_dofs: hybrid.unique_local_dofs(),
                local_factor_bytes: hybrid.factor_bytes(),
                algebraic_coarse_dimension: self
                    .algebraic_coarse
                    .as_ref()
                    .map_or(0, TwoLevelBlockJacobiPreconditioner::coarse_dimension),
                algebraic_coarse_factor_bytes: self
                    .algebraic_coarse
                    .as_ref()
                    .map_or(0, TwoLevelBlockJacobiPreconditioner::factor_bytes),
                algebraic_coarse_aggregate_nodes: self.algebraic_coarse_aggregate_nodes,
                local_factor_budget_bytes: self.hybrid_options.max_local_factor_bytes,
                overlap_layers: self.hybrid_options.overlap_layers,
                preconditioner_reused: true,
                solve_sequence: sequence,
                krylov_workspace_bytes: self.workspace.bytes(),
                ..ReportMetrics::default()
            };
            return Ok(report_from_outcome(
                outcome,
                SolverKind::Hybrid,
                PreconditionerKind::Hybrid,
                self.backend,
                b,
                metrics,
            ));
        }

        self.solve_uncached(matrix, b, x, sequence, charge_context_setup)
    }

    fn solve_uncached(
        &mut self,
        matrix: &Csr32Matrix,
        b: &[f64],
        x: &mut [f64],
        sequence: usize,
        charge_context_setup: bool,
    ) -> Result<SolveReport, HybitError> {
        let coarse_requested =
            self.hybrid_options.enabled && self.hybrid_options.algebraic_coarse.enabled;
        let coarse_was_cached = coarse_requested && self.algebraic_coarse.is_some();
        let algebraic_coarse_seconds = if coarse_requested {
            self.ensure_algebraic_coarse(matrix)?
        } else {
            0.0
        };

        let probe_budget = if self.options.max_iterations <= 1 {
            self.options.max_iterations
        } else {
            self.hybrid_options
                .probe_iterations
                .min(self.options.max_iterations - 1)
                .max(1)
        };

        // The controller probe must use the preconditioner that is actually
        // intended as the base solve.  In particular, an explicitly enabled
        // algebraic coarse space starts at iteration zero rather than after a
        // destructive Jacobi probe.  The live PCG session is retained across
        // the controller boundary whenever the preconditioner is unchanged.
        let probe_start = Instant::now();
        let operator = operator_for_backend(matrix, self.abtm.as_ref(), self.backend);
        let mut current_pcg = if let Some(coarse) = self.algebraic_coarse.as_ref() {
            pcg_start_with_workspace(operator, coarse, b, x, self.options, &mut self.workspace)?
        } else {
            pcg_start_with_workspace(
                operator,
                &self.jacobi,
                b,
                x,
                self.options,
                &mut self.workspace,
            )?
        };
        let probe = if let Some(coarse) = self.algebraic_coarse.as_ref() {
            pcg_continue_with_workspace(
                operator,
                coarse,
                b,
                x,
                probe_budget,
                &mut current_pcg,
                &mut self.workspace,
            )?
        } else {
            pcg_continue_with_workspace(
                operator,
                &self.jacobi,
                b,
                x,
                probe_budget,
                &mut current_pcg,
                &mut self.workspace,
            )?
        };
        let probe_seconds = probe_start.elapsed().as_secs_f64();
        let probe_iterations = probe.iterations;
        let probe_final_residual = probe.final_residual;
        let mut base_metrics = ReportMetrics {
            analysis_seconds: if charge_context_setup {
                self.analysis_seconds
            } else {
                0.0
            },
            prepare_seconds: if charge_context_setup {
                self.prepare_seconds
            } else {
                0.0
            },
            probe_seconds,
            probe_iterations,
            probe_final_residual,
            algebraic_coarse_seconds,
            local_factor_budget_bytes: self.hybrid_options.max_local_factor_bytes,
            overlap_layers: self.hybrid_options.overlap_layers,
            preconditioner_reused: coarse_was_cached,
            solve_sequence: sequence,
            krylov_workspace_bytes: self.workspace.bytes(),
            ..ReportMetrics::default()
        };
        if let Some(coarse) = self.algebraic_coarse.as_ref() {
            base_metrics.algebraic_coarse_dimension = coarse.coarse_dimension();
            base_metrics.algebraic_coarse_factor_bytes = coarse.factor_bytes();
            base_metrics.algebraic_coarse_aggregate_nodes = self.algebraic_coarse_aggregate_nodes;
        }

        let base_uses_coarse = self.algebraic_coarse.is_some();
        if probe.status == SolveStatus::Converged || probe.iterations >= self.options.max_iterations
        {
            return Ok(report_from_outcome(
                probe,
                if base_uses_coarse {
                    SolverKind::Hybrid
                } else {
                    SolverKind::Pcg
                },
                if base_uses_coarse {
                    PreconditionerKind::Hybrid
                } else {
                    PreconditionerKind::Jacobi
                },
                self.backend,
                b,
                base_metrics,
            ));
        }

        let poor_progress = self.hybrid_options.enabled
            && probe.status == SolveStatus::MaxIterations
            && probe.initial_residual > 0.0
            && probe.final_residual / probe.initial_residual
                > self.hybrid_options.escalation_residual_ratio;

        let mut remaining = self.options.max_iterations.saturating_sub(probe_iterations);
        if remaining == 0 {
            return Ok(report_from_outcome(
                probe,
                if base_uses_coarse {
                    SolverKind::Hybrid
                } else {
                    SolverKind::Pcg
                },
                if base_uses_coarse {
                    PreconditionerKind::Hybrid
                } else {
                    PreconditionerKind::Jacobi
                },
                self.backend,
                b,
                base_metrics,
            ));
        }

        // Acceptable progress is only a controller boundary, not a Krylov
        // restart. Continue the exact same PCG recurrence with the same base
        // preconditioner for the remaining budget.
        if !poor_progress {
            let continuation_start = Instant::now();
            let operator = operator_for_backend(matrix, self.abtm.as_ref(), self.backend);
            let continuation = if let Some(coarse) = self.algebraic_coarse.as_ref() {
                pcg_continue_with_workspace(
                    operator,
                    coarse,
                    b,
                    x,
                    remaining,
                    &mut current_pcg,
                    &mut self.workspace,
                )?
            } else {
                pcg_continue_with_workspace(
                    operator,
                    &self.jacobi,
                    b,
                    x,
                    remaining,
                    &mut current_pcg,
                    &mut self.workspace,
                )?
            };
            let outcome = combine_outcomes(probe, continuation);
            let mut metrics = base_metrics;
            metrics.restart_seconds = continuation_start.elapsed().as_secs_f64();
            return Ok(report_from_outcome(
                outcome,
                if base_uses_coarse {
                    SolverKind::Hybrid
                } else {
                    SolverKind::Pcg
                },
                if base_uses_coarse {
                    PreconditionerKind::Hybrid
                } else {
                    PreconditionerKind::Jacobi
                },
                self.backend,
                b,
                metrics,
            ));
        }

        let mut outcome = probe;
        let mut metrics = base_metrics;
        let mut active_regions: Vec<Vec<usize>> = Vec::new();
        let mut current_hybrid: Option<HybridPreconditioner> = None;

        while remaining > 0 && metrics.escalations < self.hybrid_options.max_escalations {
            let diagnostics_start = Instant::now();
            let current_residual = residual(matrix, b, x)?;
            let risk = numerical_risk_mask(matrix, self.hybrid_options)?;
            let seeds = residual_seed_mask(
                &current_residual,
                self.hybrid_options.residual_seed_fraction,
            )?;
            let (selected, core_regions) = discover_core_regions(
                matrix,
                &risk,
                &seeds,
                &current_residual,
                self.hybrid_options,
            )?;
            metrics.hard_dofs = metrics.hard_dofs.max(selected.count_ones());

            if self.abtm.is_none() {
                self.abtm = Some(AbtmMatrix::from_csr32(matrix, AbtmConfig::default())?);
            }
            let abtm = self
                .abtm
                .as_ref()
                .expect("ABTM topology initialized for hybrid escalation");
            let candidate_regions = expand_regions_with_overlap(
                abtm,
                &core_regions,
                &current_residual,
                self.hybrid_options,
            )?;
            let budgeted = select_regions_with_factor_budget(
                matrix.nrows(),
                &active_regions,
                candidate_regions,
                &current_residual,
                self.jacobi.inv_diagonal(),
                self.hybrid_options.local_factor_selection,
                self.hybrid_options.max_local_factor_bytes,
            )?;
            metrics.diagnostics_seconds += diagnostics_start.elapsed().as_secs_f64();
            metrics.local_factor_regions_skipped_for_budget = metrics
                .local_factor_regions_skipped_for_budget
                .checked_add(budgeted.skipped_regions)
                .ok_or(HybitError::SizeOverflow)?;
            metrics.local_factor_budget_limited |= budgeted.skipped_regions > 0;

            // A later diagnostic can rediscover only regions that are already
            // active. In that case there is nothing new to factorize; keep the
            // current SPD preconditioner fixed and spend the remaining Krylov
            // budget on it instead of rebuilding an identical factorization.
            if budgeted.regions.is_empty() || budgeted.regions == active_regions {
                break;
            }

            let factor_start = Instant::now();
            let next_hybrid =
                match HybridPreconditioner::from_csr32(matrix, budgeted.regions.clone()) {
                    Ok(hybrid) => hybrid,
                    Err(HybitError::NumericalBreakdown(_)) | Err(HybitError::InvalidMatrix(_)) => {
                        metrics.local_factor_seconds += factor_start.elapsed().as_secs_f64();
                        break;
                    }
                    Err(err) => return Err(err),
                };
            metrics.local_factor_seconds += factor_start.elapsed().as_secs_f64();

            active_regions = budgeted.regions;
            metrics.escalations += 1;
            metrics.local_direct_regions = next_hybrid.region_count();
            metrics.largest_local_region = next_hybrid.largest_region();
            metrics.local_factor_dofs = next_hybrid.local_dofs();
            metrics.unique_local_factor_dofs = next_hybrid.unique_local_dofs();
            metrics.local_factor_bytes = next_hybrid.factor_bytes();

            let stage_budget =
                escalation_stage_budget(remaining, metrics.escalations, self.hybrid_options);
            let mut stage_options = self.options;
            stage_options.max_iterations = stage_budget;

            // The local correction changes the SPD preconditioner, so this is
            // the one place where restarting PCG is mathematically required.
            let restart_start = Instant::now();
            let operator = operator_for_backend(matrix, self.abtm.as_ref(), self.backend);
            let pcg_context = HybridPcgContext {
                operator,
                coarse: self.algebraic_coarse.as_ref(),
                local: &next_hybrid,
                b,
            };
            let mut stage_pcg = pcg_context.start(x, stage_options, &mut self.workspace)?;
            let stage = pcg_context.continue_with_workspace(
                x,
                stage_budget,
                &mut stage_pcg,
                &mut self.workspace,
            )?;
            metrics.restart_seconds += restart_start.elapsed().as_secs_f64();

            let stage_iterations = stage.iterations;
            let stage_status = stage.status;
            let stage_residual_ratio = if stage.initial_residual == 0.0 {
                0.0
            } else {
                stage.final_residual / stage.initial_residual
            };
            metrics.escalation_stages.push(HybridEscalationStageReport {
                stage: metrics.escalations,
                iterations: stage.iterations,
                initial_residual: stage.initial_residual,
                final_residual: stage.final_residual,
                residual_ratio: stage_residual_ratio,
                local_direct_regions: next_hybrid.region_count(),
                unique_local_factor_dofs: next_hybrid.unique_local_dofs(),
                local_factor_bytes: next_hybrid.factor_bytes(),
            });
            let stage_poor_progress = stage.status == SolveStatus::MaxIterations
                && stage.initial_residual > 0.0
                && stage.final_residual / stage.initial_residual
                    > self.hybrid_options.escalation_residual_ratio;

            remaining = remaining.saturating_sub(stage_iterations);
            outcome = combine_outcomes(outcome, stage);
            current_hybrid = Some(next_hybrid);
            current_pcg = stage_pcg;

            if stage_status == SolveStatus::Converged
                || stage_status == SolveStatus::Breakdown
                || remaining == 0
            {
                break;
            }

            // If the strengthened stage is making acceptable progress, do not
            // spend more setup memory/time. Continue below with the same fixed
            // preconditioner for all remaining iterations.
            if !stage_poor_progress {
                break;
            }
        }

        // Finish with the strongest successfully constructed preconditioner.
        // A controller-only boundary preserves the live PCG recurrence; only a
        // genuine local-preconditioner strengthening above replaced the session.
        if remaining > 0
            && outcome.status != SolveStatus::Converged
            && outcome.status != SolveStatus::Breakdown
        {
            let continuation_start = Instant::now();
            let operator = operator_for_backend(matrix, self.abtm.as_ref(), self.backend);
            let continuation = if let Some(hybrid) = current_hybrid.as_ref() {
                let pcg_context = HybridPcgContext {
                    operator,
                    coarse: self.algebraic_coarse.as_ref(),
                    local: hybrid,
                    b,
                };
                pcg_context.continue_with_workspace(
                    x,
                    remaining,
                    &mut current_pcg,
                    &mut self.workspace,
                )?
            } else if let Some(coarse) = self.algebraic_coarse.as_ref() {
                pcg_continue_with_workspace(
                    operator,
                    coarse,
                    b,
                    x,
                    remaining,
                    &mut current_pcg,
                    &mut self.workspace,
                )?
            } else {
                pcg_continue_with_workspace(
                    operator,
                    &self.jacobi,
                    b,
                    x,
                    remaining,
                    &mut current_pcg,
                    &mut self.workspace,
                )?
            };
            metrics.restart_seconds += continuation_start.elapsed().as_secs_f64();
            outcome = combine_outcomes(outcome, continuation);
        }

        let used_hybrid = current_hybrid.is_some() || self.algebraic_coarse.is_some();
        if current_hybrid.is_some() && outcome.status != SolveStatus::Breakdown {
            // Prepared solve-many reuses the strongest successfully learned
            // multi-stage local-direct state for later RHS vectors.
            self.hybrid = current_hybrid;
        }

        Ok(report_from_outcome(
            outcome,
            if used_hybrid {
                SolverKind::Hybrid
            } else {
                SolverKind::Pcg
            },
            if used_hybrid {
                PreconditionerKind::Hybrid
            } else {
                PreconditionerKind::Jacobi
            },
            self.backend,
            b,
            metrics,
        ))
    }
}

#[derive(Debug)]
pub struct HybitPreparedStructuralSystem {
    options: SolverOptions,
    backend: MatrixBackend,
    structure_signature: u64,
    value_signature: u64,
    analysis_seconds: f64,
    prepare_seconds: f64,
    aggregate_nodes: usize,
    preconditioner: RigidBodyTwoLevelBlockJacobiPreconditioner,
    abtm: Option<AbtmMatrix>,
    effective_spmv_policy: StructuralSpmvPolicy,
    effective_preconditioner_policy: StructuralPreconditionerPolicy,
    effective_pcg_vector_policy: StructuralPcgVectorPolicy,
    workspace: PcgWorkspace,
    solve_sequence: usize,
}

impl HybitPreparedStructuralSystem {
    pub fn backend(&self) -> MatrixBackend {
        self.backend
    }
    pub fn analysis_seconds(&self) -> f64 {
        self.analysis_seconds
    }
    pub fn prepare_seconds(&self) -> f64 {
        self.prepare_seconds
    }
    pub fn solve_count(&self) -> usize {
        self.solve_sequence
    }
    pub fn krylov_workspace_bytes(&self) -> usize {
        self.workspace.bytes()
    }
    pub fn aggregate_nodes(&self) -> usize {
        self.aggregate_nodes
    }
    pub fn aggregate_count(&self) -> usize {
        self.preconditioner.aggregate_count()
    }
    pub fn min_aggregate_nodes(&self) -> usize {
        self.preconditioner.min_aggregate_nodes()
    }
    pub fn max_aggregate_nodes(&self) -> usize {
        self.preconditioner.max_aggregate_nodes()
    }
    pub fn aggregation(&self) -> RigidBodyAggregation {
        self.preconditioner.aggregation()
    }
    pub fn coarse_dimension(&self) -> usize {
        self.preconditioner.coarse_dimension()
    }
    pub fn preconditioner_bytes(&self) -> usize {
        self.preconditioner.factor_bytes()
    }
    pub fn base_factor_bytes(&self) -> usize {
        self.preconditioner.base_factor_bytes()
    }
    pub fn coarse_factor_bytes(&self) -> usize {
        self.preconditioner.coarse_factor_bytes()
    }
    pub fn geometry_bytes(&self) -> usize {
        self.preconditioner.geometry_bytes()
    }
    /// Effective execution policy after resolving `StructuralSpmvPolicy::Auto`.
    pub fn spmv_policy(&self) -> StructuralSpmvPolicy {
        self.effective_spmv_policy
    }
    pub fn parallel_spmv_enabled(&self) -> bool {
        self.effective_spmv_policy == StructuralSpmvPolicy::Parallel
    }
    /// Effective execution policy after resolving `StructuralPreconditionerPolicy::Auto`.
    pub fn structural_preconditioner_policy(&self) -> StructuralPreconditionerPolicy {
        self.effective_preconditioner_policy
    }
    pub fn parallel_preconditioner_enabled(&self) -> bool {
        self.effective_preconditioner_policy == StructuralPreconditionerPolicy::Parallel
    }
    pub fn parallel_preconditioner_index_bytes(&self) -> usize {
        self.preconditioner.parallel_index_bytes()
    }
    /// Effective dense-vector policy after resolving `StructuralPcgVectorPolicy::Auto`.
    pub fn pcg_vector_policy(&self) -> StructuralPcgVectorPolicy {
        self.effective_pcg_vector_policy
    }
    pub fn parallel_pcg_vectors_enabled(&self) -> bool {
        self.effective_pcg_vector_policy == StructuralPcgVectorPolicy::Parallel
    }

    fn validate_matrix(&self, matrix: &Csr32Matrix) -> Result<(), HybitError> {
        let (structure, values) = matrix_signatures(matrix);
        if structure != self.structure_signature {
            return Err(HybitError::InvalidArgument(
                "prepared structural context matrix structure changed; analyze and prepare again",
            ));
        }
        if values != self.value_signature {
            return Err(HybitError::InvalidArgument(
                "prepared structural context matrix values changed; prepare again before reusing coarse factors",
            ));
        }
        Ok(())
    }

    pub fn solve(
        &mut self,
        matrix: &Csr32Matrix,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<SolveReport, HybitError> {
        self.validate_matrix(matrix)?;
        if b.len() != matrix.nrows() {
            return Err(HybitError::DimensionMismatch {
                expected: matrix.nrows(),
                actual: b.len(),
            });
        }
        if x.len() != matrix.ncols() {
            return Err(HybitError::DimensionMismatch {
                expected: matrix.ncols(),
                actual: x.len(),
            });
        }

        self.solve_sequence += 1;
        let sequence = self.solve_sequence;
        let charge_context_setup = sequence == 1;
        let start = Instant::now();
        let outcome = match (
            self.effective_spmv_policy,
            self.effective_preconditioner_policy,
        ) {
            (StructuralSpmvPolicy::Parallel, StructuralPreconditionerPolicy::Parallel) => {
                let operator = ParallelCsr32Operator::new(matrix);
                let preconditioner =
                    ParallelRigidBodyTwoLevelPreconditioner::new(&self.preconditioner)?;
                run_structural_pcg(
                    self.effective_pcg_vector_policy,
                    &operator,
                    &preconditioner,
                    b,
                    x,
                    self.options,
                    &mut self.workspace,
                )?
            }
            (StructuralSpmvPolicy::Parallel, StructuralPreconditionerPolicy::Serial) => {
                let operator = ParallelCsr32Operator::new(matrix);
                run_structural_pcg(
                    self.effective_pcg_vector_policy,
                    &operator,
                    &self.preconditioner,
                    b,
                    x,
                    self.options,
                    &mut self.workspace,
                )?
            }
            (StructuralSpmvPolicy::Serial, StructuralPreconditionerPolicy::Parallel) => {
                let preconditioner =
                    ParallelRigidBodyTwoLevelPreconditioner::new(&self.preconditioner)?;
                run_structural_pcg(
                    self.effective_pcg_vector_policy,
                    operator_for_backend(matrix, self.abtm.as_ref(), self.backend),
                    &preconditioner,
                    b,
                    x,
                    self.options,
                    &mut self.workspace,
                )?
            }
            (StructuralSpmvPolicy::Serial, StructuralPreconditionerPolicy::Serial) => {
                run_structural_pcg(
                    self.effective_pcg_vector_policy,
                    operator_for_backend(matrix, self.abtm.as_ref(), self.backend),
                    &self.preconditioner,
                    b,
                    x,
                    self.options,
                    &mut self.workspace,
                )?
            }
            _ => unreachable!("structural execution policies are resolved during prepare"),
        };
        let elapsed = start.elapsed().as_secs_f64();
        let metrics = ReportMetrics {
            analysis_seconds: if charge_context_setup {
                self.analysis_seconds
            } else {
                0.0
            },
            prepare_seconds: if charge_context_setup {
                self.prepare_seconds
            } else {
                0.0
            },
            restart_seconds: elapsed,
            preconditioner_reused: sequence > 1,
            solve_sequence: sequence,
            krylov_workspace_bytes: self.workspace.bytes(),
            ..ReportMetrics::default()
        };
        Ok(report_from_outcome(
            outcome,
            SolverKind::Pcg,
            PreconditionerKind::RigidBodyTwoLevel,
            self.backend,
            b,
            metrics,
        ))
    }
}

#[derive(Clone, Debug, Default)]
struct ReportMetrics {
    analysis_seconds: f64,
    prepare_seconds: f64,
    probe_seconds: f64,
    diagnostics_seconds: f64,
    local_factor_seconds: f64,
    restart_seconds: f64,
    escalations: usize,
    escalation_stages: Vec<HybridEscalationStageReport>,
    probe_iterations: usize,
    probe_final_residual: f64,
    hard_dofs: usize,
    local_direct_regions: usize,
    largest_local_region: usize,
    local_factor_dofs: usize,
    unique_local_factor_dofs: usize,
    local_factor_bytes: usize,
    algebraic_coarse_dimension: usize,
    algebraic_coarse_factor_bytes: usize,
    algebraic_coarse_aggregate_nodes: usize,
    algebraic_coarse_seconds: f64,
    local_factor_budget_bytes: usize,
    local_factor_regions_skipped_for_budget: usize,
    local_factor_budget_limited: bool,
    overlap_layers: usize,
    preconditioner_reused: bool,
    solve_sequence: usize,
    krylov_workspace_bytes: usize,
}

fn structural_graph_auto_can_fallback(error: &HybitError) -> bool {
    match error {
        HybitError::NumericalBreakdown(_) => true,
        HybitError::InvalidArgument(message) => matches!(
            *message,
            "a structural graph component contains fewer than three nodes"
        ),
        _ => false,
    }
}

impl HybitSolver {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn options(&self) -> SolverOptions {
        self.options
    }
    pub fn hybrid_options(&self) -> HybridOptions {
        self.hybrid_options
    }
    pub fn structural_options(&self) -> StructuralOptions {
        self.structural_options
    }

    pub fn set_options(&mut self, options: SolverOptions) -> Result<(), HybitError> {
        options.validate()?;
        self.options = options;
        Ok(())
    }

    pub fn set_hybrid_options(&mut self, options: HybridOptions) -> Result<(), HybitError> {
        options.validate()?;
        self.hybrid_options = options;
        Ok(())
    }

    pub fn set_structural_options(&mut self, options: StructuralOptions) -> Result<(), HybitError> {
        options.validate()?;
        self.structural_options = options;
        Ok(())
    }

    pub fn set_backend_policy(&mut self, policy: BackendPolicy) {
        self.backend_policy = policy;
    }

    pub fn analyze_csr32(&self, matrix: &Csr32Matrix) -> Result<HybitAnalysis, HybitError> {
        self.options.validate()?;
        self.hybrid_options.validate()?;
        let start = Instant::now();
        let profile = analyze_csr32(matrix)?;
        if !profile.square {
            return Err(HybitError::InvalidMatrix(
                "AutoSolver currently supports square SPD systems",
            ));
        }
        if !profile.full_diagonal || !profile.positive_diagonal {
            return Err(HybitError::InvalidMatrix(
                "PCG path requires a complete positive diagonal",
            ));
        }
        let backend = match self.backend_policy {
            BackendPolicy::Auto | BackendPolicy::Csr32 => MatrixBackend::Csr32,
            BackendPolicy::Abtm => MatrixBackend::Abtm,
        };
        let (structure_signature, value_signature) = matrix_signatures(matrix);
        Ok(HybitAnalysis {
            profile,
            backend,
            structure_signature,
            value_signature,
            analysis_seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// Compatibility alias retained from 0.3.
    pub fn analyze(&self, matrix: &Csr32Matrix) -> Result<MatrixProfile, HybitError> {
        Ok(self.analyze_csr32(matrix)?.profile)
    }

    pub fn prepare_csr32(
        &self,
        matrix: &Csr32Matrix,
        analysis: &HybitAnalysis,
    ) -> Result<HybitPreparedSystem, HybitError> {
        self.options.validate()?;
        self.hybrid_options.validate()?;
        let (structure_signature, value_signature) = matrix_signatures(matrix);
        if structure_signature != analysis.structure_signature
            || value_signature != analysis.value_signature
        {
            return Err(HybitError::InvalidArgument(
                "matrix changed between analyze and prepare",
            ));
        }
        let start = Instant::now();
        let jacobi = JacobiPreconditioner::from_csr32(matrix)?;
        // Do not eagerly pay the ABTM conversion cost for an easy CSR32
        // problem. Forced-ABTM backends build it here; the Auto/CSR32 hybrid
        // path builds topology lazily only if escalation is actually needed.
        let abtm = if analysis.backend == MatrixBackend::Abtm {
            Some(AbtmMatrix::from_csr32(matrix, AbtmConfig::default())?)
        } else {
            None
        };
        let workspace = PcgWorkspace::new(matrix.nrows());
        let prepare_seconds = start.elapsed().as_secs_f64();
        Ok(HybitPreparedSystem {
            options: self.options,
            hybrid_options: self.hybrid_options,
            backend: analysis.backend,
            structure_signature,
            value_signature,
            analysis_seconds: analysis.analysis_seconds,
            prepare_seconds,
            jacobi,
            abtm,
            hybrid: None,
            algebraic_coarse: None,
            algebraic_coarse_aggregate_nodes: 0,
            workspace,
            solve_sequence: 0,
        })
    }

    pub fn prepare(&self, matrix: &Csr32Matrix) -> Result<HybitPreparedSystem, HybitError> {
        let analysis = self.analyze_csr32(matrix)?;
        self.prepare_csr32(matrix, &analysis)
    }

    pub fn prepare_structural_csr32(
        &self,
        matrix: &Csr32Matrix,
        analysis: &HybitAnalysis,
        coordinates: &[[f64; 3]],
    ) -> Result<HybitPreparedStructuralSystem, HybitError> {
        self.options.validate()?;
        self.structural_options.validate()?;
        let (structure_signature, value_signature) = matrix_signatures(matrix);
        if structure_signature != analysis.structure_signature
            || value_signature != analysis.value_signature
        {
            return Err(HybitError::InvalidArgument(
                "matrix changed between analyze and structural prepare",
            ));
        }
        let expected = coordinates
            .len()
            .checked_mul(3)
            .ok_or(HybitError::SizeOverflow)?;
        if matrix.nrows() != expected || matrix.ncols() != expected {
            return Err(HybitError::DimensionMismatch {
                expected: matrix.nrows(),
                actual: expected,
            });
        }

        let start = Instant::now();
        let aggregate_nodes = recommend_rigid_body_aggregate_nodes(
            coordinates.len(),
            self.structural_options.target_coarse_dimension,
        )?;
        let preconditioner = match self.structural_options.aggregation {
            RigidBodyAggregation::Auto => {
                match RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32_graph(
                    matrix,
                    coordinates,
                    aggregate_nodes,
                ) {
                    Ok(preconditioner) => preconditioner,
                    Err(err) if structural_graph_auto_can_fallback(&err) => {
                        // Graph aggregation is an optimization policy, not a
                        // correctness requirement. Some valid SPD systems have
                        // disconnected one/two-node graph components, and some
                        // geometries can make a six-mode graph coarse basis rank
                        // deficient. Auto falls back to the deterministic
                        // contiguous baseline in those cases. Explicit Graph
                        // remains strict and still surfaces the original error.
                        RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32(
                            matrix,
                            coordinates,
                            aggregate_nodes,
                        )?
                    }
                    Err(err) => return Err(err),
                }
            }
            RigidBodyAggregation::Contiguous => {
                RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32(
                    matrix,
                    coordinates,
                    aggregate_nodes,
                )?
            }
            RigidBodyAggregation::Graph => {
                RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32_graph(
                    matrix,
                    coordinates,
                    aggregate_nodes,
                )?
            }
        };
        let abtm = if analysis.backend == MatrixBackend::Abtm {
            Some(AbtmMatrix::from_csr32(matrix, AbtmConfig::default())?)
        } else {
            None
        };
        let effective_spmv_policy = match self.structural_options.spmv_policy {
            StructuralSpmvPolicy::Auto => {
                if analysis.backend == MatrixBackend::Csr32
                    && matrix.nnz() >= STRUCTURAL_PARALLEL_SPMV_MIN_NNZ
                {
                    StructuralSpmvPolicy::Parallel
                } else {
                    StructuralSpmvPolicy::Serial
                }
            }
            StructuralSpmvPolicy::Serial => StructuralSpmvPolicy::Serial,
            StructuralSpmvPolicy::Parallel => {
                if analysis.backend != MatrixBackend::Csr32 {
                    return Err(HybitError::InvalidArgument(
                        "parallel structural SpMV requires the CSR32 backend",
                    ));
                }
                StructuralSpmvPolicy::Parallel
            }
        };
        let effective_preconditioner_policy = match self.structural_options.preconditioner_policy {
            StructuralPreconditionerPolicy::Auto => {
                if matrix.nnz() >= STRUCTURAL_PARALLEL_PRECONDITIONER_MIN_NNZ {
                    StructuralPreconditionerPolicy::Parallel
                } else {
                    StructuralPreconditionerPolicy::Serial
                }
            }
            StructuralPreconditionerPolicy::Serial => StructuralPreconditionerPolicy::Serial,
            StructuralPreconditionerPolicy::Parallel => StructuralPreconditionerPolicy::Parallel,
        };
        if effective_preconditioner_policy == StructuralPreconditionerPolicy::Parallel {
            // Build the aggregate->node index during prepare. The view created
            // during each solve then reuses this cached index without allocation.
            let _ = ParallelRigidBodyTwoLevelPreconditioner::new(&preconditioner)?;
        }
        let effective_pcg_vector_policy = match self.structural_options.pcg_vector_policy {
            StructuralPcgVectorPolicy::Auto => {
                if matrix.nrows() >= STRUCTURAL_PARALLEL_PCG_VECTOR_MIN_N
                    && parallel_vector_worker_count() >= STRUCTURAL_PARALLEL_PCG_VECTOR_MIN_THREADS
                {
                    StructuralPcgVectorPolicy::Parallel
                } else {
                    StructuralPcgVectorPolicy::Serial
                }
            }
            StructuralPcgVectorPolicy::Serial => StructuralPcgVectorPolicy::Serial,
            StructuralPcgVectorPolicy::Parallel => StructuralPcgVectorPolicy::Parallel,
        };
        let workspace = PcgWorkspace::new(matrix.nrows());
        let prepare_seconds = start.elapsed().as_secs_f64();

        Ok(HybitPreparedStructuralSystem {
            options: self.options,
            backend: analysis.backend,
            structure_signature,
            value_signature,
            analysis_seconds: analysis.analysis_seconds,
            prepare_seconds,
            aggregate_nodes,
            preconditioner,
            abtm,
            effective_spmv_policy,
            effective_preconditioner_policy,
            effective_pcg_vector_policy,
            workspace,
            solve_sequence: 0,
        })
    }

    pub fn prepare_structural(
        &self,
        matrix: &Csr32Matrix,
        coordinates: &[[f64; 3]],
    ) -> Result<HybitPreparedStructuralSystem, HybitError> {
        let analysis = self.analyze_csr32(matrix)?;
        self.prepare_structural_csr32(matrix, &analysis, coordinates)
    }

    pub fn solve_structural_csr32(
        &self,
        matrix: &Csr32Matrix,
        coordinates: &[[f64; 3]],
        b: &[f64],
        x: &mut [f64],
    ) -> Result<SolveReport, HybitError> {
        let analysis = self.analyze_csr32(matrix)?;
        let mut prepared = self.prepare_structural_csr32(matrix, &analysis, coordinates)?;
        prepared.solve(matrix, b, x)
    }

    pub fn solve_csr32(
        &self,
        matrix: &Csr32Matrix,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<SolveReport, HybitError> {
        let analysis = self.analyze_csr32(matrix)?;
        let mut prepared = self.prepare_csr32(matrix, &analysis)?;
        prepared.solve(matrix, b, x)
    }
}

fn run_structural_pcg(
    vector_policy: StructuralPcgVectorPolicy,
    operator: &dyn LinearOperator,
    preconditioner: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
    workspace: &mut PcgWorkspace,
) -> Result<KrylovOutcome, HybitError> {
    match vector_policy {
        StructuralPcgVectorPolicy::Parallel => {
            pcg_with_workspace_parallel_vectors(operator, preconditioner, b, x, options, workspace)
        }
        StructuralPcgVectorPolicy::Serial => {
            pcg_with_workspace(operator, preconditioner, b, x, options, workspace)
        }
        StructuralPcgVectorPolicy::Auto => {
            unreachable!("structural PCG vector policy is resolved during prepare")
        }
    }
}

fn operator_for_backend<'a>(
    matrix: &'a Csr32Matrix,
    abtm: Option<&'a AbtmMatrix>,
    backend: MatrixBackend,
) -> &'a dyn LinearOperator {
    match backend {
        MatrixBackend::Csr32 => matrix,
        MatrixBackend::Abtm => abtm.expect("ABTM storage initialized"),
        MatrixBackend::MatrixFree => unreachable!(),
    }
}

struct HybridPcgContext<'a> {
    operator: &'a dyn LinearOperator,
    coarse: Option<&'a TwoLevelBlockJacobiPreconditioner>,
    local: &'a HybridPreconditioner,
    b: &'a [f64],
}

impl HybridPcgContext<'_> {
    fn start(
        &self,
        x: &mut [f64],
        options: SolverOptions,
        workspace: &mut PcgWorkspace,
    ) -> Result<PcgSession, HybitError> {
        if let Some(coarse) = self.coarse {
            let combined = AlgebraicTwoLevelHybrid {
                coarse,
                local: self.local,
            };
            pcg_start_with_workspace(self.operator, &combined, self.b, x, options, workspace)
        } else {
            pcg_start_with_workspace(self.operator, self.local, self.b, x, options, workspace)
        }
    }

    fn continue_with_workspace(
        &self,
        x: &mut [f64],
        additional_iterations: usize,
        session: &mut PcgSession,
        workspace: &mut PcgWorkspace,
    ) -> Result<KrylovOutcome, HybitError> {
        if let Some(coarse) = self.coarse {
            let combined = AlgebraicTwoLevelHybrid {
                coarse,
                local: self.local,
            };
            pcg_continue_with_workspace(
                self.operator,
                &combined,
                self.b,
                x,
                additional_iterations,
                session,
                workspace,
            )
        } else {
            pcg_continue_with_workspace(
                self.operator,
                self.local,
                self.b,
                x,
                additional_iterations,
                session,
                workspace,
            )
        }
    }
}

fn run_hybrid_pcg(
    operator: &dyn LinearOperator,
    coarse: Option<&TwoLevelBlockJacobiPreconditioner>,
    local: &HybridPreconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
    workspace: &mut PcgWorkspace,
) -> Result<KrylovOutcome, HybitError> {
    if let Some(coarse) = coarse {
        let combined = AlgebraicTwoLevelHybrid { coarse, local };
        pcg_with_workspace(operator, &combined, b, x, options, workspace)
    } else {
        pcg_with_workspace(operator, local, b, x, options, workspace)
    }
}

fn combine_outcomes(first: KrylovOutcome, second: KrylovOutcome) -> KrylovOutcome {
    KrylovOutcome {
        status: second.status,
        iterations: first.iterations + second.iterations,
        initial_residual: first.initial_residual,
        final_residual: second.final_residual,
    }
}

fn report_from_outcome(
    outcome: KrylovOutcome,
    solver: SolverKind,
    preconditioner: PreconditionerKind,
    backend: MatrixBackend,
    b: &[f64],
    metrics: ReportMetrics,
) -> SolveReport {
    let b_norm = l2_norm(b);
    let relative_residual = if b_norm == 0.0 {
        outcome.final_residual
    } else {
        outcome.final_residual / b_norm
    };
    let setup_seconds = metrics.analysis_seconds
        + metrics.prepare_seconds
        + metrics.diagnostics_seconds
        + metrics.local_factor_seconds
        + metrics.algebraic_coarse_seconds;
    let solve_seconds = metrics.probe_seconds + metrics.restart_seconds;
    SolveReport {
        status: outcome.status,
        solver,
        preconditioner,
        backend,
        iterations: outcome.iterations,
        initial_residual: outcome.initial_residual,
        final_residual: outcome.final_residual,
        relative_residual,
        setup_seconds,
        solve_seconds,
        analysis_seconds: metrics.analysis_seconds,
        prepare_seconds: metrics.prepare_seconds,
        probe_seconds: metrics.probe_seconds,
        diagnostics_seconds: metrics.diagnostics_seconds,
        local_factor_seconds: metrics.local_factor_seconds,
        restart_seconds: metrics.restart_seconds,
        escalations: metrics.escalations,
        escalation_stages: metrics.escalation_stages,
        probe_iterations: metrics.probe_iterations,
        probe_final_residual: metrics.probe_final_residual,
        hard_dofs: metrics.hard_dofs,
        local_direct_regions: metrics.local_direct_regions,
        largest_local_region: metrics.largest_local_region,
        local_factor_dofs: metrics.local_factor_dofs,
        unique_local_factor_dofs: metrics.unique_local_factor_dofs,
        local_factor_bytes: metrics.local_factor_bytes,
        algebraic_coarse_dimension: metrics.algebraic_coarse_dimension,
        algebraic_coarse_factor_bytes: metrics.algebraic_coarse_factor_bytes,
        algebraic_coarse_aggregate_nodes: metrics.algebraic_coarse_aggregate_nodes,
        algebraic_coarse_seconds: metrics.algebraic_coarse_seconds,
        local_factor_budget_bytes: metrics.local_factor_budget_bytes,
        local_factor_regions_skipped_for_budget: metrics.local_factor_regions_skipped_for_budget,
        local_factor_budget_limited: metrics.local_factor_budget_limited,
        overlap_layers: metrics.overlap_layers,
        preconditioner_reused: metrics.preconditioner_reused,
        solve_sequence: metrics.solve_sequence,
        krylov_workspace_bytes: metrics.krylov_workspace_bytes,
    }
}

fn fnv_mix(mut h: u64, value: u64) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    for b in value.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

fn matrix_signatures(matrix: &Csr32Matrix) -> (u64, u64) {
    let mut structure = 0xcbf29ce484222325u64;
    structure = fnv_mix(structure, matrix.nrows() as u64);
    structure = fnv_mix(structure, matrix.ncols() as u64);
    structure = fnv_mix(structure, matrix.nnz() as u64);
    for &v in matrix.row_ptr() {
        structure = fnv_mix(structure, v as u64);
    }
    for &v in matrix.col_idx() {
        structure = fnv_mix(structure, v as u64);
    }

    let mut values = 0xcbf29ce484222325u64;
    values = fnv_mix(values, structure);
    for &v in matrix.values() {
        values = fnv_mix(values, v.to_bits());
    }
    (structure, values)
}
fn residual(matrix: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<Vec<f64>, HybitError> {
    let mut ax = vec![0.0; matrix.nrows()];
    matrix.apply(x, &mut ax)?;
    Ok(b.iter().zip(ax).map(|(&bi, ai)| bi - ai).collect())
}

fn numerical_risk_mask(
    matrix: &Csr32Matrix,
    options: HybridOptions,
) -> Result<DofMask, HybitError> {
    let diagonal = matrix.diagonal()?;
    let n = matrix.nrows();
    let mut risk = DofMask::new(n);
    for row in 0..n {
        let diag = diagonal[row].abs();
        if diag == 0.0 {
            continue;
        }
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let mut offdiag_sum = 0.0;
        let mut max_scale_jump = 1.0f64;
        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            if col == row {
                continue;
            }
            offdiag_sum += matrix.values()[p].abs();
            let neighbor_diag = diagonal[col].abs();
            if neighbor_diag > 0.0 {
                max_scale_jump =
                    max_scale_jump.max((diag / neighbor_diag).max(neighbor_diag / diag));
            }
        }
        let coupling = offdiag_sum / diag;
        if coupling >= options.coupling_risk_threshold
            || max_scale_jump >= options.scale_jump_threshold
        {
            risk.set(row, true)?;
        }
    }
    Ok(risk)
}

fn residual_seed_mask(residual: &[f64], fraction: f64) -> Result<DofMask, HybitError> {
    let max_abs = residual.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut seeds = DofMask::new(residual.len());
    if max_abs == 0.0 {
        return Ok(seeds);
    }
    let threshold = fraction * max_abs;
    for (i, &value) in residual.iter().enumerate() {
        if value.abs() >= threshold {
            seeds.set(i, true)?;
        }
    }
    Ok(seeds)
}

fn discover_core_regions(
    matrix: &Csr32Matrix,
    risk: &DofMask,
    seeds: &DofMask,
    residual: &[f64],
    options: HybridOptions,
) -> Result<(DofMask, Vec<Vec<usize>>), HybitError> {
    let n = matrix.nrows();
    if residual.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: residual.len(),
        });
    }

    let region_cap = options.max_local_region_size.max(1);
    let mut component_visited = vec![false; n];
    let mut claimed = vec![false; n];
    let mut queue_stamp = vec![0u32; n];
    let mut stamp = 0u32;
    let mut candidates: Vec<Vec<usize>> = Vec::new();

    // A large connected risk component used to collapse to one capped local
    // region. Harvest several connected residual-centered chunks instead so
    // the downstream memory-budget selector has a meaningful candidate pool.
    for component_start in risk.indices() {
        if component_visited[component_start] {
            continue;
        }

        let mut component = Vec::new();
        let mut component_queue = VecDeque::new();
        component_queue.push_back(component_start);
        component_visited[component_start] = true;
        let mut touches_seed = false;

        while let Some(row) = component_queue.pop_front() {
            component.push(row);
            touches_seed |= seeds.contains(row);

            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let col = matrix.col_idx()[p] as usize;
                if col < n && risk.contains(col) && !component_visited[col] {
                    component_visited[col] = true;
                    component_queue.push_back(col);
                }
            }
        }

        if !touches_seed {
            continue;
        }

        // High-residual DOFs become roots first. Each root grows one connected
        // chunk through still-unclaimed risk DOFs. Removing an earlier chunk
        // may split the remainder; later roots naturally start new chunks in
        // those residual subcomponents.
        component.sort_by(|&a, &b| {
            residual[b]
                .abs()
                .total_cmp(&residual[a].abs())
                .then_with(|| a.cmp(&b))
        });

        for root in component {
            if claimed[root] {
                continue;
            }

            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                queue_stamp.fill(0);
                stamp = 1;
            }

            let mut region = Vec::with_capacity(region_cap);
            let mut region_queue = VecDeque::new();
            region_queue.push_back(root);
            queue_stamp[root] = stamp;

            while let Some(row) = region_queue.pop_front() {
                if claimed[row] {
                    continue;
                }

                claimed[row] = true;
                region.push(row);
                if region.len() >= region_cap {
                    break;
                }

                let rs = matrix.row_ptr()[row] as usize;
                let re = matrix.row_ptr()[row + 1] as usize;
                for p in rs..re {
                    let col = matrix.col_idx()[p] as usize;
                    if col < n && risk.contains(col) && !claimed[col] && queue_stamp[col] != stamp {
                        queue_stamp[col] = stamp;
                        region_queue.push_back(col);
                    }
                }
            }

            if !region.is_empty() {
                region.sort_unstable();
                candidates.push(region);
            }
        }
    }

    // If matrix diagnostics found no seeded risk component, preserve the old
    // residual-only fallback. Seeds are partitioned into capped regions so one
    // broad residual event can still yield several candidates.
    if candidates.is_empty() {
        let mut seed_visited = vec![false; n];
        for start in seeds.indices() {
            if seed_visited[start] {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            seed_visited[start] = true;

            while let Some(row) = queue.pop_front() {
                component.push(row);
                let rs = matrix.row_ptr()[row] as usize;
                let re = matrix.row_ptr()[row + 1] as usize;
                for p in rs..re {
                    let col = matrix.col_idx()[p] as usize;
                    if col < n && seeds.contains(col) && !seed_visited[col] {
                        seed_visited[col] = true;
                        queue.push_back(col);
                    }
                }
            }

            component.sort_by(|&a, &b| {
                residual[b]
                    .abs()
                    .total_cmp(&residual[a].abs())
                    .then_with(|| a.cmp(&b))
            });
            for chunk in component.chunks(region_cap) {
                if !chunk.is_empty() {
                    let mut region = chunk.to_vec();
                    region.sort_unstable();
                    candidates.push(region);
                }
            }
        }
    }

    // CandidateOrder retains the historical strongest-residual-first
    // diagnostic order. BenefitPerByte is applied later, after overlap growth,
    // using exact persistent factor-byte estimates.
    candidates.sort_by(|a, b| {
        let a_peak = a.iter().fold(0.0f64, |m, &i| m.max(residual[i].abs()));
        let b_peak = b.iter().fold(0.0f64, |m, &i| m.max(residual[i].abs()));
        b_peak
            .total_cmp(&a_peak)
            .then_with(|| a.first().cmp(&b.first()))
    });
    candidates.truncate(options.max_local_regions);

    let mut selected = DofMask::new(n);
    for region in &candidates {
        for &dof in region {
            selected.set(dof, true)?;
        }
    }

    Ok((selected, candidates))
}

#[derive(Debug)]
struct BudgetedRegions {
    regions: Vec<Vec<usize>>,
    skipped_regions: usize,
}

fn select_regions_with_factor_budget(
    matrix_rows: usize,
    existing_regions: &[Vec<usize>],
    candidate_regions: Vec<Vec<usize>>,
    residual: &[f64],
    jacobi_inv_diag: &[f64],
    selection_policy: LocalFactorSelectionPolicy,
    max_factor_bytes: usize,
) -> Result<BudgetedRegions, HybitError> {
    if residual.len() != matrix_rows {
        return Err(HybitError::DimensionMismatch {
            expected: matrix_rows,
            actual: residual.len(),
        });
    }
    if jacobi_inv_diag.len() != matrix_rows {
        return Err(HybitError::DimensionMismatch {
            expected: matrix_rows,
            actual: jacobi_inv_diag.len(),
        });
    }

    // Regions learned by earlier escalation stages are locked in. Re-ranking
    // may choose among newly discovered regions, but it must not evict already
    // paid-for local factors and thereby undo multi-stage learning.
    let mut accepted = merge_unique_regions(&[], existing_regions.to_vec());
    let mut current_bytes = HybridPreconditioner::estimated_factor_bytes(matrix_rows, &accepted)?;
    if current_bytes > max_factor_bytes {
        return Err(HybitError::InvalidArgument(
            "existing local factors exceed configured memory budget",
        ));
    }

    let merged = merge_unique_regions(&accepted, candidate_regions);
    let mut pending = merged[accepted.len()..].to_vec();

    if selection_policy == LocalFactorSelectionPolicy::CandidateOrder {
        let mut skipped_regions = 0usize;
        for region in pending {
            let mut trial = accepted.clone();
            trial.push(region.clone());
            let trial_bytes = HybridPreconditioner::estimated_factor_bytes(matrix_rows, &trial)?;
            if trial_bytes <= max_factor_bytes {
                accepted.push(region);
                current_bytes = trial_bytes;
            } else {
                skipped_regions = skipped_regions
                    .checked_add(1)
                    .ok_or(HybitError::SizeOverflow)?;
            }
        }
        debug_assert!(current_bytes <= max_factor_bytes);
        return Ok(BudgetedRegions {
            regions: accepted,
            skipped_regions,
        });
    }

    let mut covered = vec![false; matrix_rows];
    for region in &accepted {
        for &dof in region {
            covered[dof] = true;
        }
    }

    let mut skipped_regions = 0usize;

    // Greedy benefit/byte selection. Benefit is the residual L2 energy on
    // DOFs not already covered by an accepted local factor. Cost is measured
    // by the exact persistent-byte estimator used by HybridPreconditioner.
    // Scores are recomputed after each acceptance so overlapping candidates
    // are charged only for newly covered residual energy.
    while !pending.is_empty() {
        let mut best: Option<(usize, f64, f64, usize)> = None;

        for (candidate_index, region) in pending.iter().enumerate() {
            let mut trial = accepted.clone();
            trial.push(region.clone());
            let trial_bytes = HybridPreconditioner::estimated_factor_bytes(matrix_rows, &trial)?;
            if trial_bytes > max_factor_bytes {
                continue;
            }

            let marginal_bytes = trial_bytes
                .checked_sub(current_bytes)
                .ok_or(HybitError::SizeOverflow)?;
            if marginal_bytes == 0 {
                continue;
            }

            let benefit = region.iter().fold(0.0f64, |energy, &dof| {
                if covered[dof] {
                    energy
                } else {
                    let residual_energy = residual[dof] * residual[dof];
                    match selection_policy {
                        LocalFactorSelectionPolicy::CandidateOrder
                        | LocalFactorSelectionPolicy::BenefitPerByte => energy + residual_energy,
                        LocalFactorSelectionPolicy::JacobiEnergyPerByte => {
                            energy + residual_energy * jacobi_inv_diag[dof]
                        }
                    }
                }
            });
            let score = benefit / marginal_bytes as f64;

            let better = match best {
                None => true,
                Some((best_index, best_score, best_benefit, best_bytes)) => {
                    score.total_cmp(&best_score).is_gt()
                        || (score.total_cmp(&best_score).is_eq()
                            && benefit.total_cmp(&best_benefit).is_gt())
                        || (score.total_cmp(&best_score).is_eq()
                            && benefit.total_cmp(&best_benefit).is_eq()
                            && marginal_bytes < best_bytes)
                        || (score.total_cmp(&best_score).is_eq()
                            && benefit.total_cmp(&best_benefit).is_eq()
                            && marginal_bytes == best_bytes
                            && region < &pending[best_index])
                }
            };

            if better {
                best = Some((candidate_index, score, benefit, marginal_bytes));
            }
        }

        let Some((best_index, _, _, marginal_bytes)) = best else {
            skipped_regions = skipped_regions
                .checked_add(pending.len())
                .ok_or(HybitError::SizeOverflow)?;
            break;
        };

        let region = pending.remove(best_index);
        current_bytes = current_bytes
            .checked_add(marginal_bytes)
            .ok_or(HybitError::SizeOverflow)?;
        for &dof in &region {
            covered[dof] = true;
        }
        accepted.push(region);
    }

    Ok(BudgetedRegions {
        regions: accepted,
        skipped_regions,
    })
}

fn escalation_stage_budget(
    remaining: usize,
    completed_escalations: usize,
    options: HybridOptions,
) -> usize {
    if completed_escalations >= options.max_escalations {
        remaining
    } else {
        remaining.min(options.escalation_stage_iterations)
    }
}

fn merge_unique_regions(
    existing: &[Vec<usize>],
    candidate_regions: Vec<Vec<usize>>,
) -> Vec<Vec<usize>> {
    let mut merged = existing.to_vec();

    for mut region in candidate_regions {
        region.sort_unstable();
        region.dedup();
        if region.is_empty() || merged.iter().any(|current| current == &region) {
            continue;
        }
        merged.push(region);
    }

    merged
}

fn expand_regions_with_overlap(
    abtm: &AbtmMatrix,
    core_regions: &[Vec<usize>],
    residual: &[f64],
    options: HybridOptions,
) -> Result<Vec<Vec<usize>>, HybitError> {
    let mut result = Vec::with_capacity(core_regions.len());
    for core in core_regions {
        let mut mask = DofMask::from_indices(abtm.rows(), core)?;
        for _ in 0..options.overlap_layers {
            mask = abtm.expand_mask_one_hop(&mask)?;
        }
        let mut expanded = mask.indices();
        if expanded.len() > options.max_local_region_size {
            // Core DOFs are never discarded. Fill remaining slots with the
            // strongest residual halo candidates. This keeps dense factors bounded.
            let mut keep = core.clone();
            keep.sort_unstable();
            keep.dedup();
            if keep.len() > options.max_local_region_size {
                keep.truncate(options.max_local_region_size);
            } else {
                expanded.retain(|dof| keep.binary_search(dof).is_err());
                expanded.sort_by(|&a, &b| residual[b].abs().total_cmp(&residual[a].abs()));
                let room = options.max_local_region_size - keep.len();
                keep.extend(expanded.into_iter().take(room));
            }
            expanded = keep;
        }
        expanded.sort_unstable();
        expanded.dedup();
        if !expanded.is_empty() {
            result.push(expanded);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hybit_core::Preconditioner;
    use hybit_krylov::pcg;

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

    fn block_diagonal(
        easy_before: usize,
        hard_sizes: &[usize],
        easy_between: usize,
    ) -> Csr32Matrix {
        let n = easy_before
            + hard_sizes.iter().sum::<usize>()
            + easy_between * hard_sizes.len().saturating_sub(1);
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        let mut row = 0usize;
        for _ in 0..easy_before {
            col_idx.push(row as u32);
            values.push(1.0);
            row += 1;
            row_ptr.push(col_idx.len() as u32);
        }
        for (bi, &hard) in hard_sizes.iter().enumerate() {
            let base = row;
            for local in 0..hard {
                let i = base + local;
                if local > 0 {
                    col_idx.push((i - 1) as u32);
                    values.push(-1.0);
                }
                col_idx.push(i as u32);
                values.push(2.0);
                if local + 1 < hard {
                    col_idx.push((i + 1) as u32);
                    values.push(-1.0);
                }
                row += 1;
                row_ptr.push(col_idx.len() as u32);
            }
            if bi + 1 < hard_sizes.len() {
                for _ in 0..easy_between {
                    col_idx.push(row as u32);
                    values.push(1.0);
                    row += 1;
                    row_ptr.push(col_idx.len() as u32);
                }
            }
        }
        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    #[test]
    fn auto_solver_converges_on_spd_system() {
        let a = poisson_1d(64);
        let b = vec![1.0; 64];
        let mut x = vec![0.0; 64];
        let solver = HybitSolver::new();
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();
        assert!(report.converged());
        assert!(report.krylov_workspace_bytes > 0);
    }

    #[test]
    fn forced_abtm_converges() {
        let a = poisson_1d(64);
        let b = vec![1.0; 64];
        let mut x = vec![0.0; 64];
        let mut solver = HybitSolver::new();
        solver.set_backend_policy(BackendPolicy::Abtm);
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();
        assert!(report.converged());
        assert_eq!(report.backend, MatrixBackend::Abtm);
    }

    #[test]
    fn selective_direct_escalation_beats_plain_jacobi_pcg() {
        let a = block_diagonal(32, &[64], 0);
        let b = vec![1.0; a.nrows()];
        let options = SolverOptions {
            relative_tolerance: 1.0e-10,
            absolute_tolerance: 0.0,
            max_iterations: 100,
        };
        let jacobi = JacobiPreconditioner::from_csr32(&a).unwrap();
        let mut x_plain = vec![0.0; a.nrows()];
        let plain = pcg(&a, &jacobi, &b, &mut x_plain, options).unwrap();

        let mut solver = HybitSolver::new();
        solver.set_options(options).unwrap();
        let mut x = vec![0.0; a.nrows()];
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();
        assert!(report.converged());
        assert_eq!(report.preconditioner, PreconditionerKind::Hybrid);
        assert!(report.iterations < plain.iterations);
    }

    #[test]
    fn multi_region_overlap_detects_two_hard_blocks() {
        let a = block_diagonal(16, &[48, 48], 8);
        let b = vec![1.0; a.nrows()];
        let options = SolverOptions {
            relative_tolerance: 1.0e-10,
            absolute_tolerance: 0.0,
            max_iterations: 100,
        };
        let mut solver = HybitSolver::new();
        solver.set_options(options).unwrap();
        let mut x = vec![0.0; a.nrows()];
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();
        assert!(report.converged());
        assert!(report.local_direct_regions >= 2);
    }

    #[test]
    fn large_risk_component_harvests_multiple_core_regions() {
        let a = poisson_1d(96);
        let all_dofs: Vec<usize> = (0..a.nrows()).collect();
        let risk = DofMask::from_indices(a.nrows(), &all_dofs).unwrap();
        let seeds = DofMask::from_indices(a.nrows(), &all_dofs).unwrap();
        let residual: Vec<f64> = (0..a.nrows()).map(|i| (i + 1) as f64).collect();
        let options = HybridOptions {
            max_local_region_size: 16,
            max_local_regions: 4,
            ..HybridOptions::default()
        };

        let (selected, regions) =
            discover_core_regions(&a, &risk, &seeds, &residual, options).unwrap();

        assert_eq!(regions.len(), 4);
        assert_eq!(selected.count_ones(), 64);
        assert!(regions.iter().all(|region| region.len() <= 16));
        assert!(regions.iter().all(|region| !region.is_empty()));
    }

    #[test]
    fn harvested_core_regions_prioritize_strong_residual_chunks() {
        let a = poisson_1d(96);
        let all_dofs: Vec<usize> = (0..a.nrows()).collect();
        let risk = DofMask::from_indices(a.nrows(), &all_dofs).unwrap();
        let seeds = DofMask::from_indices(a.nrows(), &all_dofs).unwrap();
        let mut residual = vec![1.0; a.nrows()];
        residual[72..96].fill(100.0);
        let options = HybridOptions {
            max_local_region_size: 16,
            max_local_regions: 2,
            ..HybridOptions::default()
        };

        let (_, regions) = discover_core_regions(&a, &risk, &seeds, &residual, options).unwrap();

        assert_eq!(regions.len(), 2);
        assert!(regions[0].iter().any(|&dof| dof >= 72));
    }

    #[test]
    fn local_factor_budget_ranks_equal_cost_regions_by_residual_energy() {
        let matrix_rows = 128usize;
        let low_energy = (0..32).collect::<Vec<_>>();
        let high_energy = (32..64).collect::<Vec<_>>();
        let regions = vec![low_energy.clone(), high_energy.clone()];
        let mut residual = vec![0.0; matrix_rows];
        residual[..32].fill(1.0);
        residual[32..64].fill(4.0);

        let one_region_budget = HybridPreconditioner::estimated_factor_bytes(
            matrix_rows,
            std::slice::from_ref(&low_energy),
        )
        .unwrap();
        let selected = select_regions_with_factor_budget(
            matrix_rows,
            &[],
            regions,
            &residual,
            &vec![1.0; matrix_rows],
            LocalFactorSelectionPolicy::BenefitPerByte,
            one_region_budget,
        )
        .unwrap();

        assert_eq!(selected.regions, vec![high_energy]);
        assert_eq!(selected.skipped_regions, 1);
    }

    #[test]
    fn local_factor_budget_prefers_higher_benefit_per_byte() {
        let matrix_rows = 128usize;
        let large = (0..48).collect::<Vec<_>>();
        let small = (64..80).collect::<Vec<_>>();
        let regions = vec![large.clone(), small.clone()];
        let mut residual = vec![0.0; matrix_rows];
        residual[..48].fill(1.0);
        residual[64..80].fill(1.0);

        // The budget can hold the large region by itself, but not both. The
        // smaller region has less total residual energy but substantially more
        // residual energy per persistent factor byte.
        let one_large_budget =
            HybridPreconditioner::estimated_factor_bytes(matrix_rows, std::slice::from_ref(&large))
                .unwrap();
        let selected = select_regions_with_factor_budget(
            matrix_rows,
            &[],
            regions,
            &residual,
            &vec![1.0; matrix_rows],
            LocalFactorSelectionPolicy::BenefitPerByte,
            one_large_budget,
        )
        .unwrap();

        assert_eq!(selected.regions, vec![small]);
        assert_eq!(selected.skipped_regions, 1);
    }

    #[test]
    fn local_factor_budget_accepts_all_regions_when_capacity_is_sufficient() {
        let matrix_rows = 128usize;
        let regions = vec![(0..32).collect::<Vec<_>>(), (32..64).collect::<Vec<_>>()];
        let residual = vec![1.0; matrix_rows];

        let full_budget =
            HybridPreconditioner::estimated_factor_bytes(matrix_rows, &regions).unwrap();
        let selected = select_regions_with_factor_budget(
            matrix_rows,
            &[],
            regions,
            &residual,
            &vec![1.0; matrix_rows],
            LocalFactorSelectionPolicy::BenefitPerByte,
            full_budget,
        )
        .unwrap();

        assert_eq!(selected.regions.len(), 2);
        assert_eq!(selected.skipped_regions, 0);
    }

    #[test]
    fn local_factor_budget_preserves_existing_learned_regions() {
        let matrix_rows = 128usize;
        let existing = vec![(0..32).collect::<Vec<_>>()];
        let candidate = (64..96).collect::<Vec<_>>();
        let mut residual = vec![0.0; matrix_rows];
        residual[64..96].fill(100.0);

        let existing_budget =
            HybridPreconditioner::estimated_factor_bytes(matrix_rows, &existing).unwrap();
        let selected = select_regions_with_factor_budget(
            matrix_rows,
            &existing,
            vec![candidate],
            &residual,
            &vec![1.0; matrix_rows],
            LocalFactorSelectionPolicy::BenefitPerByte,
            existing_budget,
        )
        .unwrap();

        assert_eq!(selected.regions, existing);
        assert_eq!(selected.skipped_regions, 1);
    }

    #[test]
    fn local_factor_budget_jacobi_energy_accounts_for_stiffness_scale() {
        let matrix_rows = 128usize;
        let soft = (0..32).collect::<Vec<_>>();
        let stiff = (32..64).collect::<Vec<_>>();
        let regions = vec![stiff.clone(), soft.clone()];
        let mut residual = vec![0.0; matrix_rows];
        residual[..64].fill(1.0);

        // Equal raw residual energy and equal factor cost. The Jacobi-energy
        // selector must prefer the region with the larger inverse diagonal,
        // i.e. the softer local stiffness scale.
        let mut inv_diag = vec![1.0; matrix_rows];
        inv_diag[..32].fill(100.0);
        inv_diag[32..64].fill(0.01);

        let one_region_budget =
            HybridPreconditioner::estimated_factor_bytes(matrix_rows, std::slice::from_ref(&soft))
                .unwrap();
        let selected = select_regions_with_factor_budget(
            matrix_rows,
            &[],
            regions,
            &residual,
            &inv_diag,
            LocalFactorSelectionPolicy::JacobiEnergyPerByte,
            one_region_budget,
        )
        .unwrap();

        assert_eq!(selected.regions, vec![soft]);
        assert_eq!(selected.skipped_regions, 1);
    }

    #[test]
    fn local_factor_candidate_order_policy_preserves_diagnostic_order() {
        let matrix_rows = 128usize;
        let low_energy = (0..32).collect::<Vec<_>>();
        let high_energy = (32..64).collect::<Vec<_>>();
        let regions = vec![low_energy.clone(), high_energy];
        let mut residual = vec![0.0; matrix_rows];
        residual[..32].fill(1.0);
        residual[32..64].fill(100.0);

        let one_region_budget = HybridPreconditioner::estimated_factor_bytes(
            matrix_rows,
            std::slice::from_ref(&low_energy),
        )
        .unwrap();
        let selected = select_regions_with_factor_budget(
            matrix_rows,
            &[],
            regions,
            &residual,
            &vec![1.0; matrix_rows],
            LocalFactorSelectionPolicy::CandidateOrder,
            one_region_budget,
        )
        .unwrap();

        assert_eq!(selected.regions, vec![low_energy]);
        assert_eq!(selected.skipped_regions, 1);
    }

    #[test]
    fn default_algebraic_coarse_apply_policy_is_auto() {
        assert_eq!(
            AlgebraicCoarseOptions::default().apply_policy,
            TwoLevelCoarseApplyPolicy::Auto
        );
    }

    #[test]
    fn default_algebraic_coarse_aggregation_is_contiguous() {
        assert_eq!(
            AlgebraicCoarseOptions::default().aggregation,
            TwoLevelAggregation::Contiguous
        );
    }

    #[test]
    fn default_algebraic_coarse_basis_is_piecewise_constant() {
        assert_eq!(
            AlgebraicCoarseOptions::default().basis,
            TwoLevelBasis::PiecewiseConstant
        );
    }

    #[test]
    fn default_algebraic_coarse_transfer_apply_is_parallel() {
        assert_eq!(
            AlgebraicCoarseOptions::default().transfer_apply_policy,
            TwoLevelTransferApplyPolicy::Parallel
        );
    }

    #[test]
    fn default_algebraic_coarse_transfer_storage_is_wide() {
        assert_eq!(
            AlgebraicCoarseOptions::default().transfer_storage_policy,
            TwoLevelTransferStoragePolicy::Wide
        );
    }

    #[test]
    fn default_algebraic_coarse_transfer_values_are_auto() {
        assert_eq!(
            AlgebraicCoarseOptions::default().transfer_value_storage_policy,
            TwoLevelTransferValueStoragePolicy::Auto
        );
    }

    #[test]
    fn default_local_factor_policy_uses_jacobi_energy_per_byte() {
        assert_eq!(
            HybridOptions::default().local_factor_selection,
            LocalFactorSelectionPolicy::JacobiEnergyPerByte
        );
    }

    #[test]
    fn escalation_stage_budget_gives_final_stage_all_remaining_iterations() {
        let options = HybridOptions {
            max_escalations: 3,
            escalation_stage_iterations: 7,
            ..HybridOptions::default()
        };

        assert_eq!(escalation_stage_budget(50, 1, options), 7);
        assert_eq!(escalation_stage_budget(43, 2, options), 7);
        assert_eq!(escalation_stage_budget(36, 3, options), 36);
    }

    #[test]
    fn merge_unique_regions_preserves_learned_regions_without_duplicates() {
        let existing = vec![vec![0, 1, 2], vec![8, 9]];
        let candidates = vec![vec![2, 1, 0], vec![16, 17], vec![9, 8]];
        let merged = merge_unique_regions(&existing, candidates);

        assert_eq!(merged, vec![vec![0, 1, 2], vec![8, 9], vec![16, 17]]);
    }

    #[test]
    fn multi_stage_escalation_can_accumulate_hard_regions() {
        // Use unequal hard blocks so the 12-iteration Jacobi probe cannot
        // collapse both blocks into the same small Krylov subspace. The first
        // block starts with a much larger RHS and is selected first. After its
        // exact local correction, the residual shifts to the larger second
        // block, forcing a second diagnostic/factorization stage.
        let a = block_diagonal(0, &[48, 64], 4);
        let mut b = vec![0.0; a.nrows()];
        b[..48].fill(100.0);
        b[52..116].fill(1.0);

        let mut solver = HybitSolver::new();
        solver
            .set_options(SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 220,
            })
            .unwrap();
        solver
            .set_hybrid_options(HybridOptions {
                max_escalations: 3,
                escalation_stage_iterations: 6,
                escalation_residual_ratio: 1.0e-6,
                residual_seed_fraction: 0.50,
                max_local_region_size: 64,
                max_local_regions: 1,
                ..HybridOptions::default()
            })
            .unwrap();

        let mut x = vec![0.0; a.nrows()];
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();

        assert!(report.converged());
        assert!(report.escalations >= 2);
        assert!(report.local_direct_regions >= 2);
        assert_eq!(report.escalation_stages.len(), report.escalations);
        for (index, stage) in report.escalation_stages.iter().enumerate() {
            assert_eq!(stage.stage, index + 1);
            assert!(stage.iterations > 0);
            assert!(stage.initial_residual.is_finite());
            assert!(stage.final_residual.is_finite());
            assert!(stage.residual_ratio.is_finite());
            assert!(stage.local_direct_regions > 0);
            assert!(stage.unique_local_factor_dofs > 0);
            assert!(stage.local_factor_bytes > 0);
        }
    }

    #[test]
    fn hybrid_preconditioner_is_positive_on_overlapping_regions() {
        let a = poisson_1d(12);
        let hybrid =
            HybridPreconditioner::from_csr32(&a, vec![(0..8).collect(), (4..12).collect()])
                .unwrap();
        let r = vec![1.0; 12];
        let mut z = vec![0.0; 12];
        hybrid.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        assert!(rz > 0.0);
    }

    #[test]
    fn algebraic_coarse_recommendation_respects_target_dimension() {
        let options = AlgebraicCoarseOptions {
            enabled: true,
            dofs_per_node: 3,
            target_coarse_dimension: 12,
            aggregation: TwoLevelAggregation::Contiguous,
            basis: TwoLevelBasis::PiecewiseConstant,
            transfer_apply_policy: TwoLevelTransferApplyPolicy::Serial,
            transfer_storage_policy: TwoLevelTransferStoragePolicy::Wide,
            transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
            apply_policy: TwoLevelCoarseApplyPolicy::FactorSolve,
        };
        let aggregate_nodes = recommend_algebraic_aggregate_nodes(120, options).unwrap();
        assert_eq!(aggregate_nodes, 10);

        let node_count = 120 / options.dofs_per_node;
        let aggregate_count = node_count.div_ceil(aggregate_nodes);
        let coarse_dimension = aggregate_count * options.dofs_per_node;
        assert!(coarse_dimension <= options.target_coarse_dimension);
    }

    #[test]
    fn algebraic_two_level_plus_local_direct_is_positive() {
        let a = poisson_1d(24);
        let coarse = TwoLevelBlockJacobiPreconditioner::from_csr32(&a, 1, 6).unwrap();
        let local = HybridPreconditioner::from_csr32(&a, vec![(8..16).collect()]).unwrap();
        let combined = AlgebraicTwoLevelHybrid {
            coarse: &coarse,
            local: &local,
        };
        let r: Vec<f64> = (0..24).map(|i| 1.0 + (i % 5) as f64).collect();
        let mut z = vec![0.0; 24];
        combined.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
        assert!(rz.is_finite());
        assert!(rz > 0.0);
    }

    #[test]
    fn prepared_context_reuses_algebraic_coarse_with_hybrid() {
        let a = block_diagonal(32, &[64], 0);
        let mut solver = HybitSolver::new();
        solver
            .set_options(SolverOptions {
                relative_tolerance: 1.0e-10,
                absolute_tolerance: 0.0,
                max_iterations: 100,
            })
            .unwrap();
        solver
            .set_hybrid_options(HybridOptions {
                // Keep this test on the coarse + local reuse path even though
                // r32 starts with coarse PCG instead of Jacobi PCG.
                escalation_residual_ratio: 1.0e-300,
                algebraic_coarse: AlgebraicCoarseOptions {
                    enabled: true,
                    dofs_per_node: 1,
                    target_coarse_dimension: 16,
                    aggregation: TwoLevelAggregation::Contiguous,
                    basis: TwoLevelBasis::PiecewiseConstant,
                    transfer_apply_policy: TwoLevelTransferApplyPolicy::Serial,
                    transfer_storage_policy: TwoLevelTransferStoragePolicy::Wide,
                    transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
                    apply_policy: TwoLevelCoarseApplyPolicy::FactorSolve,
                },
                ..HybridOptions::default()
            })
            .unwrap();

        let analysis = solver.analyze_csr32(&a).unwrap();
        let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();
        let b1 = vec![1.0; a.nrows()];
        let mut x1 = vec![0.0; a.nrows()];
        let first = prepared.solve(&a, &b1, &mut x1).unwrap();
        assert!(first.converged());
        assert!(prepared.has_cached_hybrid());
        assert!(first.algebraic_coarse_dimension > 0);
        assert!(first.algebraic_coarse_factor_bytes > 0);

        let mut b2 = vec![1.0; a.nrows()];
        b2[0] = 2.0;
        let mut x2 = vec![0.0; a.nrows()];
        let second = prepared.solve(&a, &b2, &mut x2).unwrap();
        assert!(second.converged());
        assert!(second.preconditioner_reused);
        assert_eq!(second.algebraic_coarse_seconds, 0.0);
        assert_eq!(
            second.algebraic_coarse_dimension,
            first.algebraic_coarse_dimension
        );
        assert_eq!(
            second.algebraic_coarse_factor_bytes,
            first.algebraic_coarse_factor_bytes
        );
    }

    #[test]
    fn prepared_context_reuses_coarse_without_local_hybrid() {
        let a = poisson_1d(96);
        let mut solver = HybitSolver::new();
        solver
            .set_options(SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 200,
            })
            .unwrap();
        solver
            .set_hybrid_options(HybridOptions {
                probe_iterations: 3,
                escalation_residual_ratio: 1.0e300,
                algebraic_coarse: AlgebraicCoarseOptions {
                    enabled: true,
                    dofs_per_node: 1,
                    target_coarse_dimension: 16,
                    aggregation: TwoLevelAggregation::Contiguous,
                    basis: TwoLevelBasis::PiecewiseConstant,
                    transfer_apply_policy: TwoLevelTransferApplyPolicy::Serial,
                    transfer_storage_policy: TwoLevelTransferStoragePolicy::Wide,
                    transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
                    apply_policy: TwoLevelCoarseApplyPolicy::FactorSolve,
                },
                ..HybridOptions::default()
            })
            .unwrap();

        let analysis = solver.analyze_csr32(&a).unwrap();
        let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();
        let b1 = vec![1.0; a.nrows()];
        let mut x1 = vec![0.0; a.nrows()];
        let first = prepared.solve(&a, &b1, &mut x1).unwrap();
        assert!(first.converged());
        assert!(!prepared.has_cached_hybrid());
        assert!(!first.preconditioner_reused);
        assert!(first.algebraic_coarse_seconds > 0.0);
        assert!(first.algebraic_coarse_dimension > 0);

        let mut b2 = vec![1.0; a.nrows()];
        b2[0] = 2.0;
        let mut x2 = vec![0.0; a.nrows()];
        let second = prepared.solve(&a, &b2, &mut x2).unwrap();
        assert!(second.converged());
        assert!(!prepared.has_cached_hybrid());
        assert!(second.preconditioner_reused);
        assert_eq!(second.algebraic_coarse_seconds, 0.0);
        assert_eq!(
            second.algebraic_coarse_dimension,
            first.algebraic_coarse_dimension
        );
        assert_eq!(
            second.algebraic_coarse_factor_bytes,
            first.algebraic_coarse_factor_bytes
        );
    }

    #[test]
    fn explicitly_enabled_algebraic_coarse_runs_from_initial_probe() {
        let a = poisson_1d(64);
        let mut solver = HybitSolver::new();
        solver
            .set_options(SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 100,
            })
            .unwrap();
        solver
            .set_hybrid_options(HybridOptions {
                probe_iterations: 1,
                // Any finite one-step coarse residual ratio is below this
                // threshold, so local escalation is deliberately not requested.
                escalation_residual_ratio: 1.0e300,
                coupling_risk_threshold: 100.0,
                scale_jump_threshold: 1.0e12,
                algebraic_coarse: AlgebraicCoarseOptions {
                    enabled: true,
                    dofs_per_node: 1,
                    target_coarse_dimension: 16,
                    aggregation: TwoLevelAggregation::Contiguous,
                    basis: TwoLevelBasis::PiecewiseConstant,
                    transfer_apply_policy: TwoLevelTransferApplyPolicy::Serial,
                    transfer_storage_policy: TwoLevelTransferStoragePolicy::Wide,
                    transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
                    apply_policy: TwoLevelCoarseApplyPolicy::FactorSolve,
                },
                ..HybridOptions::default()
            })
            .unwrap();

        let b = vec![1.0; a.nrows()];
        let mut x = vec![0.0; a.nrows()];
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();

        assert!(report.converged());
        assert_eq!(report.escalations, 0);
        assert_eq!(report.hard_dofs, 0);
        assert_eq!(report.local_direct_regions, 0);
        assert!(report.algebraic_coarse_dimension > 0);
        assert!(report.algebraic_coarse_factor_bytes > 0);
        assert_eq!(report.preconditioner, PreconditionerKind::Hybrid);
    }

    #[test]
    fn algebraic_coarse_probe_preserves_direct_pcg_recurrence() {
        let a = poisson_1d(96);
        let options = SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 200,
        };
        let coarse_options = AlgebraicCoarseOptions {
            enabled: true,
            dofs_per_node: 1,
            target_coarse_dimension: 16,
            aggregation: TwoLevelAggregation::Contiguous,
            basis: TwoLevelBasis::PiecewiseConstant,
            transfer_apply_policy: TwoLevelTransferApplyPolicy::Serial,
            transfer_storage_policy: TwoLevelTransferStoragePolicy::Wide,
            transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
            apply_policy: TwoLevelCoarseApplyPolicy::FactorSolve,
        };
        let aggregate_nodes =
            recommend_algebraic_aggregate_nodes(a.nrows(), coarse_options).unwrap();
        let coarse =
            TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_transfer_options(
                &a,
                coarse_options.dofs_per_node,
                aggregate_nodes,
                coarse_options.aggregation,
                coarse_options.basis,
                coarse_options.apply_policy,
                TwoLevelTransferOptions {
                    apply_policy: coarse_options.transfer_apply_policy,
                    storage_policy: coarse_options.transfer_storage_policy,
                    value_storage_policy: coarse_options.transfer_value_storage_policy,
                },
            )
            .unwrap();
        let b = vec![1.0; a.nrows()];
        let mut x_direct = vec![0.0; a.nrows()];
        let direct = pcg(&a, &coarse, &b, &mut x_direct, options).unwrap();

        let mut solver = HybitSolver::new();
        solver.set_options(options).unwrap();
        solver
            .set_hybrid_options(HybridOptions {
                probe_iterations: 3,
                // Keep the preconditioner fixed after the controller boundary.
                escalation_residual_ratio: 1.0e300,
                algebraic_coarse: coarse_options,
                ..HybridOptions::default()
            })
            .unwrap();
        let mut x_segmented = vec![0.0; a.nrows()];
        let report = solver.solve_csr32(&a, &b, &mut x_segmented).unwrap();

        assert!(report.converged());
        assert_eq!(report.probe_iterations, 3);
        assert_eq!(report.escalations, 0);
        assert_eq!(report.iterations, direct.iterations);
        assert_eq!(report.preconditioner, PreconditionerKind::Hybrid);
        for (&segmented, &direct_value) in x_segmented.iter().zip(&x_direct) {
            let scale = segmented.abs().max(direct_value.abs()).max(1.0);
            assert!((segmented - direct_value).abs() <= 1.0e-12 * scale);
        }
    }

    #[test]
    fn explicitly_enabled_algebraic_coarse_survives_local_factor_budget_rejection() {
        let a = poisson_1d(64);
        let mut solver = HybitSolver::new();
        solver
            .set_options(SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 100,
            })
            .unwrap();
        solver
            .set_hybrid_options(HybridOptions {
                probe_iterations: 1,
                // Force the controller into the diagnostic/escalation path.
                escalation_residual_ratio: 1.0e-300,
                // The empty hybrid state itself needs matrix_rows * sizeof(u16)
                // bytes for multiplicity bookkeeping.  Give exactly that much
                // capacity so every non-empty local factor is rejected while
                // the explicitly requested coarse space must still be built.
                max_local_factor_bytes: 64 * std::mem::size_of::<u16>(),
                algebraic_coarse: AlgebraicCoarseOptions {
                    enabled: true,
                    dofs_per_node: 1,
                    target_coarse_dimension: 16,
                    aggregation: TwoLevelAggregation::Contiguous,
                    basis: TwoLevelBasis::PiecewiseConstant,
                    transfer_apply_policy: TwoLevelTransferApplyPolicy::Serial,
                    transfer_storage_policy: TwoLevelTransferStoragePolicy::Wide,
                    transfer_value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
                    apply_policy: TwoLevelCoarseApplyPolicy::FactorSolve,
                },
                ..HybridOptions::default()
            })
            .unwrap();

        let b = vec![1.0; a.nrows()];
        let mut x = vec![0.0; a.nrows()];
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();

        assert!(report.converged());
        assert_eq!(report.escalations, 0);
        assert!(report.hard_dofs > 0);
        assert_eq!(report.local_direct_regions, 0);
        assert!(report.local_factor_budget_limited);
        assert!(report.local_factor_regions_skipped_for_budget > 0);
        assert!(report.algebraic_coarse_dimension > 0);
        assert!(report.algebraic_coarse_factor_bytes > 0);
        assert_eq!(report.preconditioner, PreconditionerKind::Hybrid);
    }

    #[test]
    fn prepared_context_reuses_hybrid_factor_and_workspace() {
        let a = block_diagonal(32, &[64], 0);
        let mut solver = HybitSolver::new();
        solver
            .set_options(SolverOptions {
                relative_tolerance: 1.0e-10,
                absolute_tolerance: 0.0,
                max_iterations: 100,
            })
            .unwrap();
        let analysis = solver.analyze_csr32(&a).unwrap();
        let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();
        let workspace_bytes = prepared.krylov_workspace_bytes();

        let b1 = vec![1.0; a.nrows()];
        let mut x1 = vec![0.0; a.nrows()];
        let first = prepared.solve(&a, &b1, &mut x1).unwrap();
        assert!(first.converged());
        assert!(prepared.has_cached_hybrid());
        assert!(!first.preconditioner_reused);
        assert!(first.local_factor_seconds >= 0.0);

        let mut b2 = vec![1.0; a.nrows()];
        b2[0] = 2.0;
        let mut x2 = vec![0.0; a.nrows()];
        let second = prepared.solve(&a, &b2, &mut x2).unwrap();
        assert!(second.converged());
        assert!(second.preconditioner_reused);
        assert_eq!(second.probe_iterations, 0);
        assert_eq!(second.local_factor_seconds, 0.0);
        assert_eq!(second.analysis_seconds, 0.0);
        assert_eq!(second.prepare_seconds, 0.0);
        assert_eq!(second.solve_sequence, 2);
        assert_eq!(second.krylov_workspace_bytes, workspace_bytes);
    }
}
