//! C1-G5F: runtime structural feature audit inside the Rust library.
//!
//! Research/SHADOW-ONLY. This does not train, certify SPD, route, or change a
//! numerical solve. No corpus paths, labels, outcome times, or CSV are read.

use std::time::Instant;

use hybit_core::HybitError;
use hybit_matrix::Csr32Matrix;

use crate::HybitSolver;
use crate::c1g5d_selection_model::{ModelShadowObservation, SelectionModel};

/// C1-G5F corpus state is descriptive, not a success/admission certificate.
/// It is deliberately separate from the frozen C1-G5C historical record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowCorpusCoverage {
    pub constructed_generator_families: usize,
    pub common_four_arm_measured_families: usize,
    pub externally_sourced_physical_families: usize,
    pub heldout_model_qualified: bool,
    pub production_routing_authorized: bool,
}

pub const C1G5F_CORPUS_COVERAGE_20261114: ShadowCorpusCoverage = ShadowCorpusCoverage {
    constructed_generator_families: 2,
    common_four_arm_measured_families: 1,
    externally_sourced_physical_families: 0,
    heldout_model_qualified: false,
    production_routing_authorized: false,
};

/// Pre-solve facts from CSR numbers and topology. No matrix identity is used.
/// Strong links use |a_ij|/sqrt(a_ii*a_jj) >= 0.25 when all diagonals > 0.
/// These ratios are diagnostics; they are neither SPD proofs nor costs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowCouplingFeatures {
    pub offdiagonal_entries: usize,
    pub positive_offdiagonal_fraction: f64,
    pub diagonal_non_dominant_rows: Option<usize>,
    pub max_offdiagonal_to_diagonal: Option<f64>,
    pub strong_normalized_offdiagonal_fraction: Option<f64>,
    pub nonfinite_ratio_detected: bool,
}

#[derive(Clone, Debug)]
pub struct ShadowFeatureAudit {
    pub model: ModelShadowObservation,
    pub coupling: ShadowCouplingFeatures,
    pub corpus: ShadowCorpusCoverage,
    /// Entire audit including previous shadow/model score, in milliseconds.
    pub audit_ms: f64,
}

fn coupling_features(matrix: &Csr32Matrix) -> ShadowCouplingFeatures {
    let n = matrix.nrows();
    let mut diagonal = vec![0.0_f64; n];
    let mut found = vec![false; n];
    let rp = matrix.row_ptr();
    let ci = matrix.col_idx();
    let val = matrix.values();
    let square = n == matrix.ncols();

    if square {
        for (row, (diag, present)) in diagonal.iter_mut().zip(&mut found).enumerate() {
            for p in rp[row] as usize..rp[row + 1] as usize {
                if ci[p] as usize == row {
                    *diag += val[p];
                    *present = true;
                }
            }
        }
    }
    let usable_diagonal = square
        && n > 0
        && diagonal
            .iter()
            .zip(&found)
            .all(|(&d, &present)| present && d.is_finite() && d > 0.0);

    let mut offdiagonal_entries = 0usize;
    let mut positive_offdiagonal = 0usize;
    let mut strong_links = 0usize;
    let mut non_dominant = 0usize;
    let mut max_ratio = 0.0_f64;
    let mut invalid_ratio = false;

    // Keep each normalized off-diagonal contribution independently bounded;
    // never use an overflowing sum of raw |A_ij| or a_ii * a_jj.
    for (row, &diagonal_row) in diagonal.iter().enumerate() {
        let mut row_ratio = 0.0_f64;
        for p in rp[row] as usize..rp[row + 1] as usize {
            let col = ci[p] as usize;
            if col == row {
                continue;
            }
            offdiagonal_entries += 1;
            if val[p] > 0.0 {
                positive_offdiagonal += 1;
            }
            if !usable_diagonal {
                continue;
            }
            let normalized = val[p].abs() / diagonal_row;
            row_ratio += normalized;
            if !normalized.is_finite() || !row_ratio.is_finite() {
                invalid_ratio = true;
            }
            if val[p] != 0.0 {
                let log_strength =
                    val[p].abs().ln() - 0.5 * (diagonal_row.ln() + diagonal[col].ln());
                if !log_strength.is_finite() {
                    invalid_ratio = true;
                } else if log_strength >= 0.25_f64.ln() {
                    strong_links += 1;
                }
            }
        }
        if usable_diagonal {
            if row_ratio > 1.0 {
                non_dominant += 1;
            }
            max_ratio = max_ratio.max(row_ratio);
        }
    }
    let valid = usable_diagonal && !invalid_ratio;
    ShadowCouplingFeatures {
        offdiagonal_entries,
        positive_offdiagonal_fraction: if offdiagonal_entries == 0 {
            0.0
        } else {
            positive_offdiagonal as f64 / offdiagonal_entries as f64
        },
        diagonal_non_dominant_rows: valid.then_some(non_dominant),
        max_offdiagonal_to_diagonal: valid.then_some(max_ratio),
        strong_normalized_offdiagonal_fraction: valid.then_some(if offdiagonal_entries == 0 {
            0.0
        } else {
            strong_links as f64 / offdiagonal_entries as f64
        }),
        nonfinite_ratio_detected: invalid_ratio,
    }
}

impl HybitSolver {
    /// Opt-in, in-library feature diagnostics. The model may report a research
    /// score, but a recommendation remains blocked by the existing shadow gate.
    /// The function does not solve, allocate factors, inspect filesystem paths,
    /// or enable native Multilevel in the production dispatcher.
    pub fn shadow_audit_csr32_with_model(
        &self,
        matrix: &Csr32Matrix,
        rhs: &[f64],
        expected_rhs_reuse: usize,
        model: &SelectionModel,
    ) -> Result<ShadowFeatureAudit, HybitError> {
        let start = Instant::now();
        let model_observation =
            self.shadow_select_csr32_with_model(matrix, rhs, expected_rhs_reuse, model)?;
        let coupling = coupling_features(matrix);
        Ok(ShadowFeatureAudit {
            model: model_observation,
            coupling,
            corpus: C1G5F_CORPUS_COVERAGE_20261114,
            audit_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c1g5c_shadow::ShadowReason;
    use crate::c1g5d_selection_model::SelectionModel;

    fn chain() -> Csr32Matrix {
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
    fn diagonal_dominance_and_link_strength_are_observed() {
        let observed = HybitSolver::new()
            .shadow_audit_csr32_with_model(
                &chain(),
                &[1.0, 0.0, 1.0],
                3,
                &SelectionModel::embedded_standard(),
            )
            .unwrap();
        assert_eq!(observed.coupling.offdiagonal_entries, 4);
        assert_eq!(observed.coupling.positive_offdiagonal_fraction, 0.0);
        assert_eq!(observed.coupling.diagonal_non_dominant_rows, Some(0));
        assert_eq!(observed.coupling.max_offdiagonal_to_diagonal, Some(1.0));
        assert_eq!(
            observed.coupling.strong_normalized_offdiagonal_fraction,
            Some(1.0)
        );
        assert_eq!(
            observed.model.base.reason,
            ShadowReason::InsufficientIndependentEvidence
        );
        assert!(observed.model.base.recommended.is_none());
        assert!(observed.audit_ms.is_finite());
        assert!(!observed.corpus.production_routing_authorized);
    }

    #[test]
    fn positive_offdiagonal_and_nondominant_rows() {
        let matrix = Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 2.0, 1.0],
        )
        .unwrap();
        // Symmetric positive diagonal but not SPD: never declare it SPD!
        let record = coupling_features(&matrix);
        assert_eq!(record.diagonal_non_dominant_rows, Some(2));
        assert_eq!(record.max_offdiagonal_to_diagonal, Some(2.0));
        assert_eq!(record.positive_offdiagonal_fraction, 1.0);
    }

    #[test]
    fn incomplete_diagonal_has_no_fabricated_ratios() {
        let matrix = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![1, 1], vec![1.0, 2.0]).unwrap();
        let record = coupling_features(&matrix);
        assert_eq!(record.diagonal_non_dominant_rows, None);
        assert_eq!(record.strong_normalized_offdiagonal_fraction, None);
    }

    #[test]
    fn extreme_finite_matrix_does_not_fake_finite_features() {
        let matrix = Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0e-300, 1.0e300, 1.0e300, 1.0e-300],
        )
        .unwrap();
        let record = coupling_features(&matrix);
        assert!(record.nonfinite_ratio_detected);
        assert!(record.max_offdiagonal_to_diagonal.is_none());
    }

    #[test]
    fn invalid_rhs_and_reuse_are_rejected_by_library() {
        let solver = HybitSolver::new();
        let model = SelectionModel::embedded_standard();
        assert!(
            solver
                .shadow_audit_csr32_with_model(&chain(), &[1.0], 1, &model)
                .is_err()
        );
        assert!(
            solver
                .shadow_audit_csr32_with_model(&chain(), &[1.0, 0.0, 0.0], 0, &model)
                .is_err()
        );
    }
}
