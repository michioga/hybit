//! P3-B opt-in numerical contracts for equilibrated local Cholesky.
#![cfg(feature = "p3-equilibrated-local-cholesky")]

use hybit_core::Preconditioner;
use hybit_matrix::Csr32Matrix;
use hybit_precond::{HybridPreconditioner, LocalCholeskyRegion};

fn matrix(scale: f64, dof: [f64; 4], asym: bool) -> Csr32Matrix {
    let mut ptr = vec![0_u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..4 {
        if i > 0 {
            cols.push((i - 1) as u32);
            vals.push(-scale * dof[i] * dof[i - 1]);
        }
        cols.push(i as u32);
        vals.push(2.0 * scale * dof[i] * dof[i]);
        if i < 3 {
            cols.push((i + 1) as u32);
            let original = -scale * dof[i] * dof[i + 1];
            vals.push(if asym && i == 0 {
                1.5 * original
            } else {
                original
            });
        }
        ptr.push(cols.len() as u32);
    }
    Csr32Matrix::new(4, 4, ptr, cols, vals).unwrap()
}

#[test]
fn p3b_accepts_constructed_small_large_and_congruence_spd() {
    for (scale, diag) in [
        (1.0, [1.0; 4]),
        (1.0e-16, [1.0; 4]),
        (1.0e16, [1.0; 4]),
        (1.0, [1.0e-8, 1.0, 1.0, 1.0e8]),
    ] {
        let a = matrix(scale, diag, false);
        let local = LocalCholeskyRegion::from_csr32(&a, &[0, 1, 2, 3]).unwrap();
        assert_eq!(local.factor_bytes(), 4 * 8 + 10 * 8 + 4 * 8);
        let hybrid = HybridPreconditioner::from_csr32(&a, vec![(0..4).collect()]).unwrap();
        assert_eq!(hybrid.factor_bytes(), 248);
        assert_eq!(
            hybrid.factor_bytes(),
            HybridPreconditioner::estimated_factor_bytes(4, &[(0..4).collect()]).unwrap()
        );
        let mut z = [0.0; 4];
        let r = [1.0, 2.0, 3.0, 4.0];
        hybrid.apply(&r, &mut z).unwrap();
        assert!(z.iter().all(|&v| v.is_finite()));
        assert!(r.iter().zip(&z).map(|(&x, &y)| x * y).sum::<f64>() > 0.0);
    }
}

#[test]
fn p3b_rejects_small_scale_nonsymmetry_without_absolute_floor() {
    for scale in [1.0e-12, 1.0] {
        let a = matrix(scale, [1.0; 4], true);
        assert!(LocalCholeskyRegion::from_csr32(&a, &[0, 1, 2, 3]).is_err());
    }
}

#[test]
fn p3b_rejects_indefinite_symmetric_principal_matrix() {
    let a = Csr32Matrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 2.0, 2.0, 1.0],
    )
    .unwrap();
    assert!(LocalCholeskyRegion::from_csr32(&a, &[0, 1]).is_err());
}
