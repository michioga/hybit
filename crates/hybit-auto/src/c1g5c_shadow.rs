//! C1-G5C: opt-in, library-resident, fail-closed autonomous-selection shadow.
//!
//! This module is deliberately not a trained solver selector.  It consumes
//! runtime CSR/RHS features and a *qualified evidence state* embedded in Rust;
//! scripts, CSV files, Matrix Market files, and reference solutions are never
//! consulted by the library's execution path.

use std::time::Instant;

use hybit_core::{HybitError, MatrixProblemClass, SolveReport};
use hybit_matrix::Csr32Matrix;

use crate::HybitSolver;

/// Frozen, descriptive C1-G4/C1-G5B corpus status; NOT a learned cost model.
/// These values are used only to prevent an unjustified positive prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowEvidenceQualification {
    pub independent_measured_families: usize,
    pub g4_measured_preconditioner_runs: usize,
    pub g5b_jacobi_hybrid_runs: usize,
    pub g5b_hybrid_regressions: usize,
    pub g5b_local_rescues: usize,
    pub independent_heldout_gate_passed: bool,
}

pub const C1G5C_EVIDENCE_20261114: ShadowEvidenceQualification = ShadowEvidenceQualification {
    independent_measured_families: 1,
    g4_measured_preconditioner_runs: 72,
    g5b_jacobi_hybrid_runs: 24,
    g5b_hybrid_regressions: 2,
    g5b_local_rescues: 0,
    independent_heldout_gate_passed: false,
};

/// Only facts available before the solve. No matrix/RHS identity or outcome labels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowRuntimeFeatures {
    pub declared_class: MatrixProblemClass,
    pub nrows: usize,
    pub nnz: usize,
    pub max_nnz_per_row: usize,
    pub complete_positive_diagonal: bool,
    pub diagonal_log10_spread: Option<f64>,
    pub rhs_nonzero_fraction: f64,
    pub rhs_peak_over_norm: f64,
    pub requested_relative_tolerance: f64,
    pub max_iterations: usize,
    pub local_factor_budget_bytes: usize,
    pub expected_rhs_reuse: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowReason {
    /// This pilot is intentionally restricted to caller-declared SPD.
    UnsupportedMatrixClass,
    /// No work is needed for this mathematically trivial RHS.
    ZeroRightHandSide,
    /// Matrix declaration is not enough to establish a usable SPD diagonal.
    IncompleteOrNonpositiveDiagonal,
    /// No trained/held-out validated rule may be executed.
    InsufficientIndependentEvidence,
    /// A separate audited rule/model is mandatory even after family qualification.
    NoApprovedCostModel,
}

/// A shadow *observation*, not a change to the solver's actual preconditioner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowSelectionObservation {
    pub recommended: Option<ShadowSolverCandidate>,
    pub reason: ShadowReason,
    pub features: ShadowRuntimeFeatures,
    pub evidence: ShadowEvidenceQualification,
    pub observation_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowSolverCandidate {
    JacobiPcg,
    HybridLocalCholesky,
    TwoLevel,
    NativeMultilevel,
}

fn capture_features(
    solver: &HybitSolver,
    matrix: &Csr32Matrix,
    rhs: &[f64],
    expected_rhs_reuse: usize,
) -> Result<ShadowRuntimeFeatures, HybitError> {
    if matrix.nrows() != rhs.len() {
        return Err(HybitError::DimensionMismatch {
            expected: matrix.nrows(),
            actual: rhs.len(),
        });
    }
    if expected_rhs_reuse == 0 {
        return Err(HybitError::InvalidArgument(
            "expected_rhs_reuse must be positive",
        ));
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(HybitError::InvalidArgument(
            "shadow RHS contains nonfinite values",
        ));
    }
    let n = matrix.nrows();
    let mut max_nnz_per_row = 0usize;
    let mut min_diagonal = f64::INFINITY;
    let mut max_diagonal = 0.0_f64;
    let mut complete_positive_diagonal = matrix.nrows() == matrix.ncols();
    let rp = matrix.row_ptr();
    let ci = matrix.col_idx();
    let val = matrix.values();
    for row in 0..n {
        let start = rp[row] as usize;
        let end = rp[row + 1] as usize;
        max_nnz_per_row = max_nnz_per_row.max(end - start);
        if !complete_positive_diagonal {
            continue;
        }
        let mut found = false;
        let mut diagonal = 0.0_f64;
        for p in start..end {
            if ci[p] as usize == row {
                diagonal += val[p];
                found = true;
            }
        }
        if !found || !diagonal.is_finite() || diagonal <= 0.0 {
            complete_positive_diagonal = false;
        } else {
            min_diagonal = min_diagonal.min(diagonal);
            max_diagonal = max_diagonal.max(diagonal);
        }
    }
    let rhs_norm = rhs.iter().fold(0.0_f64, |norm, &v| norm.hypot(v));
    let rhs_nonzero = rhs.iter().filter(|&&v| v != 0.0).count();
    let rhs_max = rhs.iter().fold(0.0_f64, |mx, v| mx.max(v.abs()));
    let rhs_peak_over_norm = if rhs_norm == 0.0 {
        0.0
    } else {
        rhs_max / rhs_norm
    };
    let diagonal_log10_spread = if complete_positive_diagonal && min_diagonal.is_finite() {
        Some(max_diagonal.log10() - min_diagonal.log10())
    } else {
        None
    };
    Ok(ShadowRuntimeFeatures {
        declared_class: solver.problem_class(),
        nrows: n,
        nnz: matrix.nnz(),
        max_nnz_per_row,
        complete_positive_diagonal,
        diagonal_log10_spread,
        rhs_nonzero_fraction: if n == 0 {
            0.0
        } else {
            rhs_nonzero as f64 / n as f64
        },
        rhs_peak_over_norm,
        requested_relative_tolerance: solver.options().relative_tolerance,
        max_iterations: solver.options().max_iterations,
        local_factor_budget_bytes: solver.hybrid_options().max_local_factor_bytes,
        expected_rhs_reuse,
    })
}

impl HybitSolver {
    /// Inspect a real input using library code; do not route based on the
    /// unqualified single-family C1-G5 dataset. Does not solve the system.
    #[cfg(feature = "c1g5c-shadow")]
    pub fn shadow_select_csr32(
        &self,
        matrix: &Csr32Matrix,
        rhs: &[f64],
        expected_rhs_reuse: usize,
    ) -> Result<ShadowSelectionObservation, HybitError> {
        self.options().validate()?;
        let start = Instant::now();
        let features = capture_features(self, matrix, rhs, expected_rhs_reuse)?;
        let reason = if features.declared_class != MatrixProblemClass::Spd {
            ShadowReason::UnsupportedMatrixClass
        } else if features.rhs_nonzero_fraction == 0.0 {
            ShadowReason::ZeroRightHandSide
        } else if !features.complete_positive_diagonal {
            ShadowReason::IncompleteOrNonpositiveDiagonal
        } else if !C1G5C_EVIDENCE_20261114.independent_heldout_gate_passed
            || C1G5C_EVIDENCE_20261114.independent_measured_families < 2
        {
            ShadowReason::InsufficientIndependentEvidence
        } else {
            ShadowReason::NoApprovedCostModel
        };
        Ok(ShadowSelectionObservation {
            recommended: None,
            reason,
            features,
            evidence: C1G5C_EVIDENCE_20261114,
            observation_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Opt-in library API which observes the selection inputs and then calls
    /// the existing, unchanged solve path. No new numerical or routing policy.
    #[cfg(feature = "c1g5c-shadow")]
    pub fn solve_csr32_with_selection_shadow(
        &self,
        matrix: &Csr32Matrix,
        rhs: &[f64],
        x: &mut [f64],
        expected_rhs_reuse: usize,
    ) -> Result<(SolveReport, ShadowSelectionObservation), HybitError> {
        let observation = self.shadow_select_csr32(matrix, rhs, expected_rhs_reuse)?;
        let report = self.solve_csr32(matrix, rhs, x)?;
        Ok((report, observation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spd() -> Csr32Matrix {
        Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn embedded_evidence_blocks_unsafe_choice() {
        let solver = HybitSolver::new();
        let a = tiny_spd();
        let decision = solver.shadow_select_csr32(&a, &[1.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(decision.recommended, None);
        assert_eq!(
            decision.reason,
            ShadowReason::InsufficientIndependentEvidence
        );
        assert_eq!(decision.evidence.g5b_hybrid_regressions, 2);
        assert!(!decision.evidence.independent_heldout_gate_passed);
        assert_eq!(decision.features.nnz, 7);
        assert!(decision.observation_ms.is_finite());
    }

    #[test]
    fn not_a_trained_matrix_hash_lookup() {
        let solver = HybitSolver::new();
        let decision = solver
            .shadow_select_csr32(&tiny_spd(), &[0.0, 1.0, 0.0], 3)
            .unwrap();
        assert!(decision.recommended.is_none());
        assert_eq!(decision.features.expected_rhs_reuse, 3);
        assert!((decision.features.rhs_nonzero_fraction - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn unsupported_class_refuses_to_guess() {
        let mut solver = HybitSolver::new();
        solver.set_problem_class(MatrixProblemClass::GeneralSquare);
        let decision = solver
            .shadow_select_csr32(&tiny_spd(), &[1.0, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(decision.reason, ShadowReason::UnsupportedMatrixClass);
    }

    #[test]
    fn invalid_rhs_and_reuse_rejected() {
        let solver = HybitSolver::new();
        let a = tiny_spd();
        assert!(solver.shadow_select_csr32(&a, &[1.0], 1).is_err());
        assert!(solver.shadow_select_csr32(&a, &[1.0, 0.0, 0.0], 0).is_err());
        assert!(
            solver
                .shadow_select_csr32(&a, &[f64::NAN, 0.0, 0.0], 1)
                .is_err()
        );
    }

    #[test]
    fn zero_rhs_abstains() {
        let solver = HybitSolver::new();
        let decision = solver
            .shadow_select_csr32(&tiny_spd(), &[0.0; 3], 1)
            .unwrap();
        assert_eq!(decision.reason, ShadowReason::ZeroRightHandSide);
    }

    #[test]
    fn opt_in_shadow_does_not_change_the_solution() {
        let solver = HybitSolver::new();
        let a = tiny_spd();
        let b = [1.0, 0.0, 1.0];
        let mut x_control = [0.0; 3];
        let mut x_shadow = [0.0; 3];
        let original = solver.solve_csr32(&a, &b, &mut x_control).unwrap();
        let (observed, shadow) = solver
            .solve_csr32_with_selection_shadow(&a, &b, &mut x_shadow, 1)
            .unwrap();
        assert_eq!(x_control, x_shadow);
        assert_eq!(original.status, observed.status);
        assert_eq!(original.iterations, observed.iterations);
        assert!(shadow.recommended.is_none());
    }
}
