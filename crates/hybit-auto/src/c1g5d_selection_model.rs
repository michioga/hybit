//! C1-G5D: versioned *in-library* selection-model representation.
//!
//! Embedded standard policy is an intentionally untrained, abstaining model.
//! Explicit external model loading is research/SHADOW-ONLY; neither changes
//! a HyBIT solve, authorizes SPD, nor enables any production preconditioner.
//! No Python, CSV, environment lookup, automatic file discovery or runtime
//! network dependency is permitted in this module.

use std::fmt;
use std::io::Read;
use std::path::Path;

use crate::c1g5c_shadow::{ShadowRuntimeFeatures, ShadowSolverCandidate};

/// Fixed-size, portable, explicitly versioned model format (little endian).
/// Note: FNV-1a is a *corruption detector*, not an authenticity signature.
pub const SELECTION_MODEL_SCHEMA_V1: u32 = 1;
const MAGIC: &[u8; 8] = b"HYBITSM1";
const CANDIDATES: usize = 4;
const COEFFICIENTS: usize = 5;
const CONTENT_LEN: usize = 8 + 4 + CANDIDATES * COEFFICIENTS * 8;
const FILE_LEN: usize = CONTENT_LEN + 8;
const MAX_COEFFICIENT: f64 = 1.0e6;

/// Model provenance is controlled by the Rust library, NOT a file claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionModelOrigin {
    /// Built-in standard: no trained coefficients, no unsafe recommendation.
    EmbeddedUntrained,
    /// Explicitly supplied research model, always shadow-only.
    ExternalResearch,
}

/// Nonnegative dimensionless research score coefficients, NOT measured ms.
/// Candidate order: Jacobi, local Hybrid, Two-Level, native Multilevel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionCostWeights {
    pub bias: f64,
    pub log_nnz: f64,
    pub diagonal_spread: f64,
    pub rhs_peak_fraction: f64,
    pub inverse_rhs_reuse: f64,
}

impl SelectionCostWeights {
    fn values(self) -> [f64; COEFFICIENTS] {
        [
            self.bias,
            self.log_nnz,
            self.diagonal_spread,
            self.rhs_peak_fraction,
            self.inverse_rhs_reuse,
        ]
    }

    fn from_values(values: [f64; COEFFICIENTS]) -> Self {
        Self {
            bias: values[0],
            log_nnz: values[1],
            diagonal_spread: values[2],
            rhs_peak_fraction: values[3],
            inverse_rhs_reuse: values[4],
        }
    }

    fn validate(self) -> Result<(), SelectionModelError> {
        if self
            .values()
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0 || *v > MAX_COEFFICIENT)
        {
            return Err(SelectionModelError::InvalidFormat(
                "coefficients must be finite, nonnegative and within v1 bounds",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum SelectionModelError {
    Io(std::io::Error),
    InvalidFormat(&'static str),
}

impl fmt::Display for SelectionModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "selection model I/O: {error}"),
            Self::InvalidFormat(message) => write!(f, "invalid selection model: {message}"),
        }
    }
}

impl std::error::Error for SelectionModelError {}

impl From<std::io::Error> for SelectionModelError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// Immutable, versioned model handle. Clone is cheap (four fixed arrays).
/// An application can load it once and pass it to multiple shadow calls.
#[derive(Clone, Debug)]
pub struct SelectionModel {
    origin: SelectionModelOrigin,
    weights: Option<[SelectionCostWeights; CANDIDATES]>,
}

impl Default for SelectionModel {
    fn default() -> Self {
        Self::embedded_standard()
    }
}

impl SelectionModel {
    /// Standard model compiled into HyBIT: conservative abstention until an
    /// independently validated model has been approved through a future gate.
    pub const fn embedded_standard() -> Self {
        Self {
            origin: SelectionModelOrigin::EmbeddedUntrained,
            weights: None,
        }
    }

    /// Explicit research coefficients; never assert trained/held-out status.
    pub fn research_only(
        weights: [SelectionCostWeights; CANDIDATES],
    ) -> Result<Self, SelectionModelError> {
        for weight in weights {
            weight.validate()?;
        }
        Ok(Self {
            origin: SelectionModelOrigin::ExternalResearch,
            weights: Some(weights),
        })
    }

    pub fn origin(&self) -> SelectionModelOrigin {
        self.origin
    }

    pub const fn schema_version(&self) -> u32 {
        SELECTION_MODEL_SCHEMA_V1
    }

    /// Strict fixed-length little-endian V1 binary decoder; no arbitrary code.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SelectionModelError> {
        if data.len() != FILE_LEN {
            return Err(SelectionModelError::InvalidFormat("wrong binary length"));
        }
        if &data[..8] != MAGIC {
            return Err(SelectionModelError::InvalidFormat("wrong model magic"));
        }
        let version = u32::from_le_bytes(data[8..12].try_into().expect("fixed range"));
        if version != SELECTION_MODEL_SCHEMA_V1 {
            return Err(SelectionModelError::InvalidFormat(
                "unsupported schema version",
            ));
        }
        let stored = u64::from_le_bytes(
            data[CONTENT_LEN..FILE_LEN]
                .try_into()
                .expect("fixed checksum range"),
        );
        if stored != fnv64(&data[..CONTENT_LEN]) {
            return Err(SelectionModelError::InvalidFormat("checksum mismatch"));
        }
        let mut weights = [SelectionCostWeights {
            bias: 0.0,
            log_nnz: 0.0,
            diagonal_spread: 0.0,
            rhs_peak_fraction: 0.0,
            inverse_rhs_reuse: 0.0,
        }; CANDIDATES];
        let mut offset = 12;
        for weight in &mut weights {
            let mut fields = [0.0; COEFFICIENTS];
            for value in &mut fields {
                *value = f64::from_le_bytes(
                    data[offset..offset + 8]
                        .try_into()
                        .expect("fixed coefficient range"),
                );
                offset += 8;
            }
            *weight = SelectionCostWeights::from_values(fields);
        }
        Self::research_only(weights)
    }

    /// Explicit, bounded file load. No automatic startup read, paths, env
    /// variables, global cache, or runtime corpus files in `hybit::solve`.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, SelectionModelError> {
        let file = std::fs::File::open(path)?;
        let mut input = file.take((FILE_LEN + 1) as u64);
        let mut data = Vec::with_capacity(FILE_LEN + 1);
        input.read_to_end(&mut data)?;
        Self::from_bytes(&data)
    }

    /// Offline serialization for research fixtures/exporters; the embedded
    /// model has no learned weights to export.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SelectionModelError> {
        let Some(weights) = self.weights else {
            return Err(SelectionModelError::InvalidFormat(
                "embedded untrained model has no exportable coefficients",
            ));
        };
        let mut out = Vec::with_capacity(FILE_LEN);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&SELECTION_MODEL_SCHEMA_V1.to_le_bytes());
        for weight in weights {
            weight.validate()?;
            for value in weight.values() {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        let integrity = fnv64(&out);
        out.extend_from_slice(&integrity.to_le_bytes());
        Ok(out)
    }

    /// Informational research ranking. No model can authorize a production
    /// recommendation, a PCG class contract, or a memory overrun.
    pub(crate) fn diagnostic_rank(
        &self,
        features: &ShadowRuntimeFeatures,
    ) -> Option<(ShadowSolverCandidate, f64)> {
        let weights = self.weights?;
        let values = [
            1.0,
            (features.nnz as f64).ln_1p(),
            features.diagonal_log10_spread.unwrap_or(0.0).max(0.0),
            features.rhs_peak_over_norm,
            1.0 / features.expected_rhs_reuse as f64,
        ];
        if values.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let candidates = [
            ShadowSolverCandidate::JacobiPcg,
            ShadowSolverCandidate::HybridLocalCholesky,
            ShadowSolverCandidate::TwoLevel,
            ShadowSolverCandidate::NativeMultilevel,
        ];
        let mut best: Option<(ShadowSolverCandidate, f64)> = None;
        for (candidate, weight) in candidates.into_iter().zip(weights) {
            let w = weight.values();
            let score = w.iter().zip(values).map(|(a, b)| a * b).sum::<f64>();
            if !score.is_finite() {
                return None;
            }
            if best.is_none_or(|(_, current)| score < current) {
                best = Some((candidate, score));
            }
        }
        best
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c1g5c_shadow::ShadowRuntimeFeatures;
    use hybit_core::MatrixProblemClass;

    fn row(bias: f64) -> SelectionCostWeights {
        SelectionCostWeights {
            bias,
            log_nnz: 0.1,
            diagonal_spread: 0.2,
            rhs_peak_fraction: 0.3,
            inverse_rhs_reuse: 0.4,
        }
    }

    fn features() -> ShadowRuntimeFeatures {
        ShadowRuntimeFeatures {
            declared_class: MatrixProblemClass::Spd,
            nrows: 3,
            nnz: 7,
            max_nnz_per_row: 3,
            complete_positive_diagonal: true,
            diagonal_log10_spread: Some(0.0),
            rhs_nonzero_fraction: 2.0 / 3.0,
            rhs_peak_over_norm: 0.5,
            requested_relative_tolerance: 1.0e-8,
            max_iterations: 100,
            local_factor_budget_bytes: 1024,
            expected_rhs_reuse: 3,
        }
    }

    #[test]
    fn standard_is_embedded_and_abstains() {
        let model = SelectionModel::default();
        assert_eq!(model.origin(), SelectionModelOrigin::EmbeddedUntrained);
        assert_eq!(model.schema_version(), 1);
        assert!(model.diagnostic_rank(&features()).is_none());
        assert!(model.to_bytes().is_err());
    }

    #[test]
    fn strict_roundtrip_and_research_ranking() {
        let model =
            SelectionModel::research_only([row(9.0), row(8.0), row(1.0), row(12.0)]).unwrap();
        let bytes = model.to_bytes().unwrap();
        assert_eq!(bytes.len(), FILE_LEN);
        let parsed = SelectionModel::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.origin(), SelectionModelOrigin::ExternalResearch);
        assert_eq!(
            parsed.diagnostic_rank(&features()).unwrap().0,
            ShadowSolverCandidate::TwoLevel
        );
        assert_eq!(bytes, parsed.to_bytes().unwrap());
    }

    #[test]
    fn reject_corruption_truncation_magic_version_and_excess() {
        let bytes = SelectionModel::research_only([row(1.0); 4])
            .unwrap()
            .to_bytes()
            .unwrap();
        assert!(SelectionModel::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut oversized = bytes.clone();
        oversized.push(0);
        assert!(SelectionModel::from_bytes(&oversized).is_err());
        let mut corrupted = bytes.clone();
        corrupted[16] ^= 1;
        assert!(SelectionModel::from_bytes(&corrupted).is_err());
        let mut wrong_magic = bytes.clone();
        wrong_magic[0] = 0;
        assert!(SelectionModel::from_bytes(&wrong_magic).is_err());
        let mut wrong_schema = bytes.clone();
        wrong_schema[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert!(SelectionModel::from_bytes(&wrong_schema).is_err());
    }

    #[test]
    fn reject_nonfinite_and_negative_even_with_updated_checksum() {
        for bad in [-1.0, f64::NAN, f64::INFINITY, 1.0e7] {
            let mut bytes = SelectionModel::research_only([row(1.0); 4])
                .unwrap()
                .to_bytes()
                .unwrap();
            bytes[12..20].copy_from_slice(&bad.to_le_bytes());
            let checksum = fnv64(&bytes[..CONTENT_LEN]);
            bytes[CONTENT_LEN..FILE_LEN].copy_from_slice(&checksum.to_le_bytes());
            assert!(SelectionModel::from_bytes(&bytes).is_err());
        }
    }

    #[test]
    fn research_cannot_read_implicit_paths() {
        assert!(SelectionModel::load_from_path("/nonexistent-hybit-c1g5d-model.bin").is_err());
    }
}

/// The underlying C1-G5C observation is always returned, unchanged.
/// Diagnostic research ranks are not a permitted execution recommendation.
#[derive(Clone, Debug)]
pub struct ModelShadowObservation {
    pub base: crate::c1g5c_shadow::ShadowSelectionObservation,
    pub model_origin: SelectionModelOrigin,
    pub model_schema_version: u32,
    pub diagnostic_candidate: Option<ShadowSolverCandidate>,
    pub diagnostic_score: Option<f64>,
}

impl crate::HybitSolver {
    /// Evaluate a built-in or explicitly supplied model *within hybit-auto*.
    /// This neither changes the actual solver nor turns research scores into
    /// approved routing. No filesystem access occurs here.
    pub fn shadow_select_csr32_with_model(
        &self,
        matrix: &hybit_matrix::Csr32Matrix,
        rhs: &[f64],
        expected_rhs_reuse: usize,
        model: &SelectionModel,
    ) -> Result<ModelShadowObservation, hybit_core::HybitError> {
        let base = self.shadow_select_csr32(matrix, rhs, expected_rhs_reuse)?;
        let candidate = if matches!(
            base.reason,
            crate::c1g5c_shadow::ShadowReason::InsufficientIndependentEvidence
                | crate::c1g5c_shadow::ShadowReason::NoApprovedCostModel
        ) {
            model.diagnostic_rank(&base.features)
        } else {
            None
        };
        Ok(ModelShadowObservation {
            base,
            model_origin: model.origin(),
            model_schema_version: model.schema_version(),
            diagnostic_candidate: candidate.map(|(candidate, _)| candidate),
            diagnostic_score: candidate.map(|(_, score)| score),
        })
    }

    /// Opt-in research solve wrapper with a model; reports diagnostic ranking
    /// but calls the existing, unchanged `solve_csr32` numerical path.
    pub fn solve_csr32_with_model_shadow(
        &self,
        matrix: &hybit_matrix::Csr32Matrix,
        rhs: &[f64],
        x: &mut [f64],
        expected_rhs_reuse: usize,
        model: &SelectionModel,
    ) -> Result<(hybit_core::SolveReport, ModelShadowObservation), hybit_core::HybitError> {
        let observation =
            self.shadow_select_csr32_with_model(matrix, rhs, expected_rhs_reuse, model)?;
        let report = self.solve_csr32(matrix, rhs, x)?;
        Ok((report, observation))
    }
}
