#![cfg(feature = "c1g5c-shadow")]

use hybit::{
    C1G5F_CORPUS_COVERAGE_20261114, Csr32Matrix, HybitSolver, SelectionModel, ShadowReason,
};

#[test]
fn feature_audit_stays_in_library_and_never_routes() {
    let a = Csr32Matrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
    )
    .unwrap();
    let b = [1.0, 0.0, 1.0];
    let solver = HybitSolver::new();
    let model = SelectionModel::embedded_standard();
    let observed = solver
        .shadow_audit_csr32_with_model(&a, &b, 1, &model)
        .unwrap();
    assert_eq!(observed.corpus, C1G5F_CORPUS_COVERAGE_20261114);
    assert_eq!(
        observed.model.base.reason,
        ShadowReason::InsufficientIndependentEvidence
    );
    assert_eq!(observed.model.diagnostic_candidate, None);
    assert_eq!(observed.model.base.recommended, None);

    let mut x_control = [0.0; 3];
    let mut x_shadow = [0.0; 3];
    let control = solver.solve_csr32(&a, &b, &mut x_control).unwrap();
    let (shadow, _old_observation) = solver
        .solve_csr32_with_model_shadow(&a, &b, &mut x_shadow, 1, &model)
        .unwrap();
    assert_eq!(x_control, x_shadow);
    assert_eq!(control.status, shadow.status);
    assert_eq!(control.iterations, shadow.iterations);
}
