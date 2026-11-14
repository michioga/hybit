#![cfg(feature = "c1g5c-shadow")]

use hybit::{Csr32Matrix, HybitSolver, ShadowReason};

#[test]
fn facade_shadow_is_library_only_and_does_not_route() {
    let matrix = Csr32Matrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![3.0, -1.0, -1.0, 3.0],
    )
    .unwrap();
    let solver = HybitSolver::new();
    let mut x = [0.0; 2];
    let (_, observed) = solver
        .solve_csr32_with_selection_shadow(&matrix, &[1.0, 1.0], &mut x, 1)
        .unwrap();
    assert_eq!(
        observed.reason,
        ShadowReason::InsufficientIndependentEvidence
    );
    assert!(observed.recommended.is_none());
    assert!(x.iter().all(|v| v.is_finite()));
}
