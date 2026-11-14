#![cfg(feature = "c1g5c-shadow")]

use hybit::{
    Csr32Matrix, HybitSolver, SelectionCostWeights, SelectionModel, SelectionModelOrigin,
    ShadowReason, ShadowSolverCandidate,
};

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

fn weight(bias: f64) -> SelectionCostWeights {
    SelectionCostWeights {
        bias,
        log_nnz: 0.0,
        diagonal_spread: 0.0,
        rhs_peak_fraction: 0.0,
        inverse_rhs_reuse: 0.0,
    }
}

#[test]
fn embedded_model_abstains_and_keeps_numerical_solution() {
    let solver = HybitSolver::new();
    let a = tiny_spd();
    let b = [1.0, 0.0, 1.0];
    let mut x_original = [0.0; 3];
    let mut x_shadow = [0.0; 3];
    let original = solver.solve_csr32(&a, &b, &mut x_original).unwrap();
    let (shadowed, observation) = solver
        .solve_csr32_with_model_shadow(&a, &b, &mut x_shadow, 1, &SelectionModel::default())
        .unwrap();
    assert_eq!(x_original, x_shadow);
    assert_eq!(original.status, shadowed.status);
    assert_eq!(original.iterations, shadowed.iterations);
    assert_eq!(
        observation.base.reason,
        ShadowReason::InsufficientIndependentEvidence
    );
    assert!(observation.base.recommended.is_none());
    assert!(observation.diagnostic_candidate.is_none());
    assert_eq!(
        observation.model_origin,
        SelectionModelOrigin::EmbeddedUntrained
    );
}

#[test]
fn external_research_rank_never_routes_or_overrides_gate() {
    let solver = HybitSolver::new();
    let a = tiny_spd();
    let b = [1.0, 0.0, 1.0];
    let external = SelectionModel::research_only([
        weight(100.0),
        weight(90.0),
        weight(80.0),
        weight(0.0), // arbitrary research rank; do NOT route to this arm
    ])
    .unwrap();
    let observation = solver
        .shadow_select_csr32_with_model(&a, &b, 1, &external)
        .unwrap();
    assert_eq!(
        observation.model_origin,
        SelectionModelOrigin::ExternalResearch
    );
    assert_eq!(
        observation.diagnostic_candidate,
        Some(ShadowSolverCandidate::NativeMultilevel)
    );
    assert_eq!(
        observation.base.reason,
        ShadowReason::InsufficientIndependentEvidence
    );
    assert!(observation.base.recommended.is_none());

    let bytes = external.to_bytes().unwrap();
    let reread = SelectionModel::from_bytes(&bytes).unwrap();
    assert_eq!(reread.origin(), SelectionModelOrigin::ExternalResearch);
    let re_observation = solver
        .shadow_select_csr32_with_model(&a, &b, 1, &reread)
        .unwrap();
    assert_eq!(
        re_observation.diagnostic_candidate,
        observation.diagnostic_candidate
    );
}

#[test]
fn invalid_rhs_and_unsupported_class_still_refuse_research_rank() {
    let mut solver = HybitSolver::new();
    let model = SelectionModel::research_only([weight(1.0); 4]).unwrap();
    assert!(
        solver
            .shadow_select_csr32_with_model(&tiny_spd(), &[1.0], 1, &model)
            .is_err()
    );
    solver.set_problem_class(hybit::MatrixProblemClass::GeneralSquare);
    let observation = solver
        .shadow_select_csr32_with_model(&tiny_spd(), &[1.0, 0.0, 0.0], 1, &model)
        .unwrap();
    assert_eq!(
        observation.base.reason,
        ShadowReason::UnsupportedMatrixClass
    );
    assert!(observation.diagnostic_candidate.is_none());
}

#[test]
fn explicit_file_load_matches_in_memory_without_discovery() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let model = SelectionModel::research_only([weight(8.0), weight(7.0), weight(6.0), weight(5.0)])
        .unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("hybit-c1g5d-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&path, model.to_bytes().unwrap()).unwrap();
    let loaded = SelectionModel::load_from_path(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded.origin(), SelectionModelOrigin::ExternalResearch);
    let solver = HybitSolver::new();
    let observation = solver
        .shadow_select_csr32_with_model(&tiny_spd(), &[1.0, 0.0, 1.0], 1, &loaded)
        .unwrap();
    assert!(observation.base.recommended.is_none());
    assert_eq!(
        observation.diagnostic_candidate,
        Some(ShadowSolverCandidate::NativeMultilevel)
    );
}
