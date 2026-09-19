use std::collections::VecDeque;
use std::time::Instant;

use hybit_core::{
    l2_norm, HybitError, LinearOperator, MatrixBackend, PreconditionerKind, SolveReport,
    SolveStatus, SolverKind, SolverOptions,
};
use hybit_krylov::{pcg_with_workspace, KrylovOutcome, PcgWorkspace};
use hybit_matrix::{analyze_csr32, AbtmConfig, AbtmMatrix, Csr32Matrix, DofMask, MatrixProfile, ParallelCsr32Operator};
pub use hybit_precond::RigidBodyAggregation;
use hybit_precond::{
    recommend_rigid_body_aggregate_nodes, HybridPreconditioner, JacobiPreconditioner,
    ParallelRigidBodyTwoLevelPreconditioner, RigidBodyTwoLevelBlockJacobiPreconditioner,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendPolicy {
    Auto,
    Csr32,
    Abtm,
}

#[derive(Clone, Copy, Debug)]
pub struct HybridOptions {
    pub enabled: bool,
    pub probe_iterations: usize,
    pub escalation_residual_ratio: f64,
    pub coupling_risk_threshold: f64,
    pub scale_jump_threshold: f64,
    pub residual_seed_fraction: f64,
    pub max_local_region_size: usize,
    pub max_local_regions: usize,
    pub overlap_layers: usize,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            probe_iterations: 12,
            escalation_residual_ratio: 0.50,
            coupling_risk_threshold: 0.90,
            scale_jump_threshold: 100.0,
            residual_seed_fraction: 0.25,
            max_local_region_size: 128,
            max_local_regions: 8,
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
}

impl Default for StructuralOptions {
    fn default() -> Self {
        Self {
            target_coarse_dimension: 1536,
            aggregation: RigidBodyAggregation::Auto,
            spmv_policy: StructuralSpmvPolicy::Auto,
            preconditioner_policy: StructuralPreconditionerPolicy::Auto,
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
        if !self.escalation_residual_ratio.is_finite() || self.escalation_residual_ratio <= 0.0 {
            return Err(HybitError::InvalidArgument("escalation_residual_ratio must be finite and > 0"));
        }
        if !self.coupling_risk_threshold.is_finite() || self.coupling_risk_threshold < 0.0 {
            return Err(HybitError::InvalidArgument("coupling_risk_threshold must be finite and >= 0"));
        }
        if !self.scale_jump_threshold.is_finite() || self.scale_jump_threshold < 1.0 {
            return Err(HybitError::InvalidArgument("scale_jump_threshold must be finite and >= 1"));
        }
        if !self.residual_seed_fraction.is_finite()
            || self.residual_seed_fraction <= 0.0
            || self.residual_seed_fraction > 1.0
        {
            return Err(HybitError::InvalidArgument("residual_seed_fraction must be in (0, 1]"));
        }
        if self.max_local_region_size == 0 || self.max_local_regions == 0 {
            return Err(HybitError::InvalidArgument("local region limits must be > 0"));
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
    pub fn profile(&self) -> &MatrixProfile { &self.profile }
    pub fn backend(&self) -> MatrixBackend { self.backend }
    pub fn analysis_seconds(&self) -> f64 { self.analysis_seconds }
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
    workspace: PcgWorkspace,
    solve_sequence: usize,
}

impl HybitPreparedSystem {
    pub fn backend(&self) -> MatrixBackend { self.backend }
    pub fn analysis_seconds(&self) -> f64 { self.analysis_seconds }
    pub fn prepare_seconds(&self) -> f64 { self.prepare_seconds }
    pub fn solve_count(&self) -> usize { self.solve_sequence }
    pub fn krylov_workspace_bytes(&self) -> usize { self.workspace.bytes() }
    pub fn has_cached_hybrid(&self) -> bool { self.hybrid.is_some() }

    fn validate_matrix(&self, matrix: &Csr32Matrix) -> Result<(), HybitError> {
        let (structure, values) = matrix_signatures(matrix);
        if structure != self.structure_signature {
            return Err(HybitError::InvalidArgument("prepared context matrix structure changed; analyze and prepare again"));
        }
        if values != self.value_signature {
            return Err(HybitError::InvalidArgument("prepared context matrix values changed; prepare again before reusing local factors"));
        }
        Ok(())
    }

    pub fn solve(&mut self, matrix: &Csr32Matrix, b: &[f64], x: &mut [f64]) -> Result<SolveReport, HybitError> {
        self.validate_matrix(matrix)?;
        if b.len() != matrix.nrows() {
            return Err(HybitError::DimensionMismatch { expected: matrix.nrows(), actual: b.len() });
        }
        if x.len() != matrix.ncols() {
            return Err(HybitError::DimensionMismatch { expected: matrix.ncols(), actual: x.len() });
        }
        self.solve_sequence += 1;
        let sequence = self.solve_sequence;
        let charge_context_setup = sequence == 1;

        // Once a difficult subspace has been learned for this matrix, subsequent
        // RHS vectors reuse the exact local Cholesky factors and skip probe/
        // diagnostics/factorization entirely.
        if let Some(hybrid) = self.hybrid.as_ref() {
            let start = Instant::now();
            let outcome = pcg_with_workspace(
                operator_for_backend(matrix, self.abtm.as_ref(), self.backend),
                hybrid,
                b,
                x,
                self.options,
                &mut self.workspace,
            )?;
            let elapsed = start.elapsed().as_secs_f64();
            let metrics = ReportMetrics {
                analysis_seconds: if charge_context_setup { self.analysis_seconds } else { 0.0 },
                prepare_seconds: if charge_context_setup { self.prepare_seconds } else { 0.0 },
                restart_seconds: elapsed,
                local_direct_regions: hybrid.region_count(),
                largest_local_region: hybrid.largest_region(),
                local_factor_dofs: hybrid.local_dofs(),
                unique_local_factor_dofs: hybrid.unique_local_dofs(),
                local_factor_bytes: hybrid.factor_bytes(),
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
        let probe_budget = if self.options.max_iterations <= 1 {
            self.options.max_iterations
        } else {
            self.hybrid_options.probe_iterations.min(self.options.max_iterations - 1).max(1)
        };
        let mut probe_options = self.options;
        probe_options.max_iterations = probe_budget;

        let probe_start = Instant::now();
        let probe = pcg_with_workspace(
            operator_for_backend(matrix, self.abtm.as_ref(), self.backend),
            &self.jacobi,
            b,
            x,
            probe_options,
            &mut self.workspace,
        )?;
        let probe_seconds = probe_start.elapsed().as_secs_f64();
        let probe_iterations = probe.iterations;
        let probe_final_residual = probe.final_residual;
        let base_metrics = ReportMetrics {
            analysis_seconds: if charge_context_setup { self.analysis_seconds } else { 0.0 },
            prepare_seconds: if charge_context_setup { self.prepare_seconds } else { 0.0 },
            probe_seconds,
            probe_iterations,
            probe_final_residual,
            overlap_layers: self.hybrid_options.overlap_layers,
            solve_sequence: sequence,
            krylov_workspace_bytes: self.workspace.bytes(),
            ..ReportMetrics::default()
        };

        if probe.status == SolveStatus::Converged || probe.iterations >= self.options.max_iterations {
            return Ok(report_from_outcome(
                probe,
                SolverKind::Pcg,
                PreconditionerKind::Jacobi,
                self.backend,
                b,
                base_metrics,
            ));
        }

        let poor_progress = self.hybrid_options.enabled
            && probe.status == SolveStatus::MaxIterations
            && probe.initial_residual > 0.0
            && probe.final_residual / probe.initial_residual > self.hybrid_options.escalation_residual_ratio;

        let remaining = self.options.max_iterations.saturating_sub(probe_iterations);
        if !poor_progress || remaining == 0 {
            let (continuation, continuation_seconds) = run_continuation(
                matrix,
                self.abtm.as_ref(),
                self.backend,
                &self.jacobi,
                b,
                x,
                self.options,
                remaining,
                &mut self.workspace,
            )?;
            let outcome = combine_outcomes(probe, continuation);
            let mut metrics = base_metrics;
            metrics.restart_seconds = continuation_seconds;
            return Ok(report_from_outcome(
                outcome,
                SolverKind::Pcg,
                PreconditionerKind::Jacobi,
                self.backend,
                b,
                metrics,
            ));
        }

        let diagnostics_start = Instant::now();
        let residual = residual(matrix, b, x)?;
        let risk = numerical_risk_mask(matrix, self.hybrid_options)?;
        let seeds = residual_seed_mask(&residual, self.hybrid_options.residual_seed_fraction)?;
        let selected = select_risk_components(matrix, &risk, &seeds, &residual, self.hybrid_options)?;
        let hard_dofs = selected.count_ones();
        let core_regions = extract_core_regions(matrix, &selected, &residual, self.hybrid_options)?;
        if self.abtm.is_none() {
            self.abtm = Some(AbtmMatrix::from_csr32(matrix, AbtmConfig::default())?);
        }
        let abtm = self.abtm.as_ref().expect("ABTM topology initialized for hybrid escalation");
        let regions = expand_regions_with_overlap(abtm, &core_regions, &residual, self.hybrid_options)?;
        let diagnostics_seconds = diagnostics_start.elapsed().as_secs_f64();

        if regions.is_empty() {
            let (continuation, continuation_seconds) = run_continuation(
                matrix,
                self.abtm.as_ref(),
                self.backend,
                &self.jacobi,
                b,
                x,
                self.options,
                remaining,
                &mut self.workspace,
            )?;
            let outcome = combine_outcomes(probe, continuation);
            let mut metrics = base_metrics;
            metrics.diagnostics_seconds = diagnostics_seconds;
            metrics.restart_seconds = continuation_seconds;
            metrics.hard_dofs = hard_dofs;
            return Ok(report_from_outcome(
                outcome,
                SolverKind::Pcg,
                PreconditionerKind::Jacobi,
                self.backend,
                b,
                metrics,
            ));
        }

        let factor_start = Instant::now();
        let hybrid = match HybridPreconditioner::from_csr32(matrix, regions) {
            Ok(hybrid) => hybrid,
            Err(HybitError::NumericalBreakdown(_)) | Err(HybitError::InvalidMatrix(_)) => {
                let local_factor_seconds = factor_start.elapsed().as_secs_f64();
                let (continuation, continuation_seconds) = run_continuation(
                    matrix,
                    self.abtm.as_ref(),
                    self.backend,
                    &self.jacobi,
                    b,
                    x,
                    self.options,
                    remaining,
                    &mut self.workspace,
                )?;
                let outcome = combine_outcomes(probe, continuation);
                let mut metrics = base_metrics;
                metrics.diagnostics_seconds = diagnostics_seconds;
                metrics.local_factor_seconds = local_factor_seconds;
                metrics.restart_seconds = continuation_seconds;
                metrics.hard_dofs = hard_dofs;
                return Ok(report_from_outcome(
                    outcome,
                    SolverKind::Pcg,
                    PreconditionerKind::Jacobi,
                    self.backend,
                    b,
                    metrics,
                ));
            }
            Err(err) => return Err(err),
        };
        let local_factor_seconds = factor_start.elapsed().as_secs_f64();

        let mut stage_options = self.options;
        stage_options.max_iterations = remaining;
        let restart_start = Instant::now();
        let stage = pcg_with_workspace(
            operator_for_backend(matrix, self.abtm.as_ref(), self.backend),
            &hybrid,
            b,
            x,
            stage_options,
            &mut self.workspace,
        )?;
        let restart_seconds = restart_start.elapsed().as_secs_f64();
        let cacheable_hybrid = stage.status != SolveStatus::Breakdown;
        let outcome = combine_outcomes(probe, stage);

        let metrics = ReportMetrics {
            analysis_seconds: base_metrics.analysis_seconds,
            prepare_seconds: base_metrics.prepare_seconds,
            probe_seconds,
            diagnostics_seconds,
            local_factor_seconds,
            restart_seconds,
            escalations: 1,
            probe_iterations,
            probe_final_residual,
            hard_dofs,
            local_direct_regions: hybrid.region_count(),
            largest_local_region: hybrid.largest_region(),
            local_factor_dofs: hybrid.local_dofs(),
            unique_local_factor_dofs: hybrid.unique_local_dofs(),
            local_factor_bytes: hybrid.factor_bytes(),
            overlap_layers: self.hybrid_options.overlap_layers,
            preconditioner_reused: false,
            solve_sequence: sequence,
            krylov_workspace_bytes: self.workspace.bytes(),
        };

        // Cache only a successfully constructed hybrid preconditioner. It is
        // matrix-value dependent, and validate_matrix() forbids accidental reuse
        // after coefficients change.
        if cacheable_hybrid {
            self.hybrid = Some(hybrid);
        }

        Ok(report_from_outcome(
            outcome,
            SolverKind::Hybrid,
            PreconditionerKind::Hybrid,
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
    workspace: PcgWorkspace,
    solve_sequence: usize,
}

impl HybitPreparedStructuralSystem {
    pub fn backend(&self) -> MatrixBackend { self.backend }
    pub fn analysis_seconds(&self) -> f64 { self.analysis_seconds }
    pub fn prepare_seconds(&self) -> f64 { self.prepare_seconds }
    pub fn solve_count(&self) -> usize { self.solve_sequence }
    pub fn krylov_workspace_bytes(&self) -> usize { self.workspace.bytes() }
    pub fn aggregate_nodes(&self) -> usize { self.aggregate_nodes }
    pub fn aggregate_count(&self) -> usize { self.preconditioner.aggregate_count() }
    pub fn min_aggregate_nodes(&self) -> usize { self.preconditioner.min_aggregate_nodes() }
    pub fn max_aggregate_nodes(&self) -> usize { self.preconditioner.max_aggregate_nodes() }
    pub fn aggregation(&self) -> RigidBodyAggregation { self.preconditioner.aggregation() }
    pub fn coarse_dimension(&self) -> usize { self.preconditioner.coarse_dimension() }
    pub fn preconditioner_bytes(&self) -> usize { self.preconditioner.factor_bytes() }
    pub fn base_factor_bytes(&self) -> usize { self.preconditioner.base_factor_bytes() }
    pub fn coarse_factor_bytes(&self) -> usize { self.preconditioner.coarse_factor_bytes() }
    pub fn geometry_bytes(&self) -> usize { self.preconditioner.geometry_bytes() }
    /// Effective execution policy after resolving `StructuralSpmvPolicy::Auto`.
    pub fn spmv_policy(&self) -> StructuralSpmvPolicy { self.effective_spmv_policy }
    pub fn parallel_spmv_enabled(&self) -> bool { self.effective_spmv_policy == StructuralSpmvPolicy::Parallel }
    /// Effective execution policy after resolving `StructuralPreconditionerPolicy::Auto`.
    pub fn structural_preconditioner_policy(&self) -> StructuralPreconditionerPolicy { self.effective_preconditioner_policy }
    pub fn parallel_preconditioner_enabled(&self) -> bool { self.effective_preconditioner_policy == StructuralPreconditionerPolicy::Parallel }
    pub fn parallel_preconditioner_index_bytes(&self) -> usize { self.preconditioner.parallel_index_bytes() }

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
            return Err(HybitError::DimensionMismatch { expected: matrix.nrows(), actual: b.len() });
        }
        if x.len() != matrix.ncols() {
            return Err(HybitError::DimensionMismatch { expected: matrix.ncols(), actual: x.len() });
        }

        self.solve_sequence += 1;
        let sequence = self.solve_sequence;
        let charge_context_setup = sequence == 1;
        let start = Instant::now();
        let outcome = match (self.effective_spmv_policy, self.effective_preconditioner_policy) {
            (StructuralSpmvPolicy::Parallel, StructuralPreconditionerPolicy::Parallel) => {
                let operator = ParallelCsr32Operator::new(matrix);
                let preconditioner = ParallelRigidBodyTwoLevelPreconditioner::new(&self.preconditioner)?;
                pcg_with_workspace(
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
                pcg_with_workspace(
                    &operator,
                    &self.preconditioner,
                    b,
                    x,
                    self.options,
                    &mut self.workspace,
                )?
            }
            (StructuralSpmvPolicy::Serial, StructuralPreconditionerPolicy::Parallel) => {
                let preconditioner = ParallelRigidBodyTwoLevelPreconditioner::new(&self.preconditioner)?;
                pcg_with_workspace(
                    operator_for_backend(matrix, self.abtm.as_ref(), self.backend),
                    &preconditioner,
                    b,
                    x,
                    self.options,
                    &mut self.workspace,
                )?
            }
            (StructuralSpmvPolicy::Serial, StructuralPreconditionerPolicy::Serial) => {
                pcg_with_workspace(
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
            analysis_seconds: if charge_context_setup { self.analysis_seconds } else { 0.0 },
            prepare_seconds: if charge_context_setup { self.prepare_seconds } else { 0.0 },
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

#[derive(Clone, Copy, Debug, Default)]
struct ReportMetrics {
    analysis_seconds: f64,
    prepare_seconds: f64,
    probe_seconds: f64,
    diagnostics_seconds: f64,
    local_factor_seconds: f64,
    restart_seconds: f64,
    escalations: usize,
    probe_iterations: usize,
    probe_final_residual: f64,
    hard_dofs: usize,
    local_direct_regions: usize,
    largest_local_region: usize,
    local_factor_dofs: usize,
    unique_local_factor_dofs: usize,
    local_factor_bytes: usize,
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
    pub fn new() -> Self { Self::default() }
    pub fn options(&self) -> SolverOptions { self.options }
    pub fn hybrid_options(&self) -> HybridOptions { self.hybrid_options }
    pub fn structural_options(&self) -> StructuralOptions { self.structural_options }

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

    pub fn set_backend_policy(&mut self, policy: BackendPolicy) { self.backend_policy = policy; }

    pub fn analyze_csr32(&self, matrix: &Csr32Matrix) -> Result<HybitAnalysis, HybitError> {
        self.options.validate()?;
        self.hybrid_options.validate()?;
        let start = Instant::now();
        let profile = analyze_csr32(matrix)?;
        if !profile.square {
            return Err(HybitError::InvalidMatrix("AutoSolver currently supports square SPD systems"));
        }
        if !profile.full_diagonal || !profile.positive_diagonal {
            return Err(HybitError::InvalidMatrix("PCG path requires a complete positive diagonal"));
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
        if structure_signature != analysis.structure_signature || value_signature != analysis.value_signature {
            return Err(HybitError::InvalidArgument("matrix changed between analyze and prepare"));
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
        if structure_signature != analysis.structure_signature || value_signature != analysis.value_signature {
            return Err(HybitError::InvalidArgument(
                "matrix changed between analyze and structural prepare",
            ));
        }
        let expected = coordinates.len().checked_mul(3).ok_or(HybitError::SizeOverflow)?;
        if matrix.nrows() != expected || matrix.ncols() != expected {
            return Err(HybitError::DimensionMismatch { expected: matrix.nrows(), actual: expected });
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

    pub fn solve_csr32(&self, matrix: &Csr32Matrix, b: &[f64], x: &mut [f64]) -> Result<SolveReport, HybitError> {
        let analysis = self.analyze_csr32(matrix)?;
        let mut prepared = self.prepare_csr32(matrix, &analysis)?;
        prepared.solve(matrix, b, x)
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

fn run_continuation(
    matrix: &Csr32Matrix,
    abtm: Option<&AbtmMatrix>,
    backend: MatrixBackend,
    jacobi: &JacobiPreconditioner,
    b: &[f64],
    x: &mut [f64],
    base_options: SolverOptions,
    remaining: usize,
    workspace: &mut PcgWorkspace,
) -> Result<(KrylovOutcome, f64), HybitError> {
    if remaining == 0 {
        let r = residual(matrix, b, x)?;
        let norm = l2_norm(&r);
        return Ok((KrylovOutcome {
            status: SolveStatus::MaxIterations,
            iterations: 0,
            initial_residual: norm,
            final_residual: norm,
        }, 0.0));
    }
    let mut options = base_options;
    options.max_iterations = remaining;
    let start = Instant::now();
    let outcome = pcg_with_workspace(
        operator_for_backend(matrix, abtm, backend),
        jacobi,
        b,
        x,
        options,
        workspace,
    )?;
    Ok((outcome, start.elapsed().as_secs_f64()))
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
        + metrics.local_factor_seconds;
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
        probe_iterations: metrics.probe_iterations,
        probe_final_residual: metrics.probe_final_residual,
        hard_dofs: metrics.hard_dofs,
        local_direct_regions: metrics.local_direct_regions,
        largest_local_region: metrics.largest_local_region,
        local_factor_dofs: metrics.local_factor_dofs,
        unique_local_factor_dofs: metrics.unique_local_factor_dofs,
        local_factor_bytes: metrics.local_factor_bytes,
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
    for &v in matrix.row_ptr() { structure = fnv_mix(structure, v as u64); }
    for &v in matrix.col_idx() { structure = fnv_mix(structure, v as u64); }

    let mut values = 0xcbf29ce484222325u64;
    values = fnv_mix(values, structure);
    for &v in matrix.values() { values = fnv_mix(values, v.to_bits()); }
    (structure, values)
}
fn residual(matrix: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<Vec<f64>, HybitError> {
    let mut ax = vec![0.0; matrix.nrows()];
    matrix.apply(x, &mut ax)?;
    Ok(b.iter().zip(ax).map(|(&bi, ai)| bi - ai).collect())
}

fn numerical_risk_mask(matrix: &Csr32Matrix, options: HybridOptions) -> Result<DofMask, HybitError> {
    let diagonal = matrix.diagonal()?;
    let n = matrix.nrows();
    let mut risk = DofMask::new(n);
    for row in 0..n {
        let diag = diagonal[row].abs();
        if diag == 0.0 { continue; }
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let mut offdiag_sum = 0.0;
        let mut max_scale_jump = 1.0f64;
        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            if col == row { continue; }
            offdiag_sum += matrix.values()[p].abs();
            let neighbor_diag = diagonal[col].abs();
            if neighbor_diag > 0.0 {
                max_scale_jump = max_scale_jump.max((diag / neighbor_diag).max(neighbor_diag / diag));
            }
        }
        let coupling = offdiag_sum / diag;
        if coupling >= options.coupling_risk_threshold || max_scale_jump >= options.scale_jump_threshold {
            risk.set(row, true)?;
        }
    }
    Ok(risk)
}

fn residual_seed_mask(residual: &[f64], fraction: f64) -> Result<DofMask, HybitError> {
    let max_abs = residual.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut seeds = DofMask::new(residual.len());
    if max_abs == 0.0 { return Ok(seeds); }
    let threshold = fraction * max_abs;
    for (i, &value) in residual.iter().enumerate() {
        if value.abs() >= threshold { seeds.set(i, true)?; }
    }
    Ok(seeds)
}

fn select_risk_components(
    matrix: &Csr32Matrix,
    risk: &DofMask,
    seeds: &DofMask,
    residual: &[f64],
    options: HybridOptions,
) -> Result<DofMask, HybitError> {
    let n = matrix.nrows();
    let mut selected = DofMask::new(n);
    let mut visited = vec![false; n];
    let cap = options.max_local_region_size.max(1);

    for start in risk.indices() {
        if visited[start] { continue; }
        let mut queue = VecDeque::new();
        let mut component = Vec::new();
        queue.push_back(start);
        visited[start] = true;
        let mut touches_seed = false;
        while let Some(row) = queue.pop_front() {
            component.push(row);
            touches_seed |= seeds.contains(row);
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let col = matrix.col_idx()[p] as usize;
                if col < n && risk.contains(col) && !visited[col] {
                    visited[col] = true;
                    queue.push_back(col);
                }
            }
        }
        if !touches_seed { continue; }

        if component.len() <= cap {
            for dof in component { selected.set(dof, true)?; }
        } else {
            let root = *component
                .iter()
                .max_by(|&&a, &&b| residual[a].abs().total_cmp(&residual[b].abs()))
                .expect("component is non-empty");
            let mut local_seen = vec![false; n];
            let mut local_queue = VecDeque::new();
            local_queue.push_back(root);
            local_seen[root] = true;
            let mut count = 0usize;
            while let Some(row) = local_queue.pop_front() {
                if count >= cap { break; }
                selected.set(row, true)?;
                count += 1;
                let rs = matrix.row_ptr()[row] as usize;
                let re = matrix.row_ptr()[row + 1] as usize;
                for p in rs..re {
                    let col = matrix.col_idx()[p] as usize;
                    if col < n && risk.contains(col) && !local_seen[col] {
                        local_seen[col] = true;
                        local_queue.push_back(col);
                    }
                }
            }
        }
    }

    if selected.is_empty() {
        selected.union_assign(seeds)?;
    }
    Ok(selected)
}

fn extract_core_regions(
    matrix: &Csr32Matrix,
    mask: &DofMask,
    residual: &[f64],
    options: HybridOptions,
) -> Result<Vec<Vec<usize>>, HybitError> {
    let n = matrix.nrows();
    let mut visited = vec![false; n];
    let mut regions: Vec<Vec<usize>> = Vec::new();

    for start in mask.indices() {
        if visited[start] { continue; }
        let mut queue = VecDeque::new();
        let mut component = Vec::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(row) = queue.pop_front() {
            component.push(row);
            let rs = matrix.row_ptr()[row] as usize;
            let re = matrix.row_ptr()[row + 1] as usize;
            for p in rs..re {
                let col = matrix.col_idx()[p] as usize;
                if col < n && mask.contains(col) && !visited[col] {
                    visited[col] = true;
                    queue.push_back(col);
                }
            }
        }
        for chunk in component.chunks(options.max_local_region_size) {
            if !chunk.is_empty() { regions.push(chunk.to_vec()); }
        }
    }

    regions.sort_by(|a, b| {
        let sa = a.iter().fold(0.0f64, |m, &i| m.max(residual[i].abs()));
        let sb = b.iter().fold(0.0f64, |m, &i| m.max(residual[i].abs()));
        sb.total_cmp(&sa)
    });
    regions.truncate(options.max_local_regions);
    Ok(regions)
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
        if !expanded.is_empty() { result.push(expanded); }
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
            if i > 0 { col_idx.push((i - 1) as u32); values.push(-1.0); }
            col_idx.push(i as u32); values.push(2.0);
            if i + 1 < n { col_idx.push((i + 1) as u32); values.push(-1.0); }
            row_ptr.push(col_idx.len() as u32);
        }
        Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
    }

    fn block_diagonal(easy_before: usize, hard_sizes: &[usize], easy_between: usize) -> Csr32Matrix {
        let n = easy_before + hard_sizes.iter().sum::<usize>() + easy_between * hard_sizes.len().saturating_sub(1);
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        let mut row = 0usize;
        for _ in 0..easy_before {
            col_idx.push(row as u32); values.push(1.0); row += 1; row_ptr.push(col_idx.len() as u32);
        }
        for (bi, &hard) in hard_sizes.iter().enumerate() {
            let base = row;
            for local in 0..hard {
                let i = base + local;
                if local > 0 { col_idx.push((i - 1) as u32); values.push(-1.0); }
                col_idx.push(i as u32); values.push(2.0);
                if local + 1 < hard { col_idx.push((i + 1) as u32); values.push(-1.0); }
                row += 1;
                row_ptr.push(col_idx.len() as u32);
            }
            if bi + 1 < hard_sizes.len() {
                for _ in 0..easy_between {
                    col_idx.push(row as u32); values.push(1.0); row += 1; row_ptr.push(col_idx.len() as u32);
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
        let options = SolverOptions { relative_tolerance: 1.0e-10, absolute_tolerance: 0.0, max_iterations: 100 };
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
        let options = SolverOptions { relative_tolerance: 1.0e-10, absolute_tolerance: 0.0, max_iterations: 100 };
        let mut solver = HybitSolver::new();
        solver.set_options(options).unwrap();
        let mut x = vec![0.0; a.nrows()];
        let report = solver.solve_csr32(&a, &b, &mut x).unwrap();
        assert!(report.converged());
        assert!(report.local_direct_regions >= 2);
    }

    #[test]
    fn hybrid_preconditioner_is_positive_on_overlapping_regions() {
        let a = poisson_1d(12);
        let hybrid = HybridPreconditioner::from_csr32(&a, vec![(0..8).collect(), (4..12).collect()]).unwrap();
        let r = vec![1.0; 12];
        let mut z = vec![0.0; 12];
        hybrid.apply(&r, &mut z).unwrap();
        let rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        assert!(rz > 0.0);
    }

    #[test]
    fn prepared_context_reuses_hybrid_factor_and_workspace() {
        let a = block_diagonal(32, &[64], 0);
        let mut solver = HybitSolver::new();
        solver.set_options(SolverOptions { relative_tolerance: 1.0e-10, absolute_tolerance: 0.0, max_iterations: 100 }).unwrap();
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
