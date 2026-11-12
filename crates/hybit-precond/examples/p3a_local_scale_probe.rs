//! P3-A: observational audit of local Cholesky under matrix scaling.
//! Numerical production code and default policies are deliberately unchanged.
use hybit_core::Preconditioner;
use hybit_matrix::Csr32Matrix;
use hybit_precond::HybridPreconditioner;
use std::error::Error;

const N: usize = 4;
const X_REF: [f64; N] = [1.0, -0.75, 1.25, -0.5];

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    scale: f64,
    diag_congruence: [f64; N],
    nonsymmetric: bool,
    mathematical_spd: bool,
}

fn make_matrix(case: Case) -> Result<Csr32Matrix, Box<dyn Error>> {
    // Symmetric cases: A = scale * D * T * D, with T=tridiag(-1,2,-1).
    // D has strictly positive diagonal entries and scale > 0; T is SPD.
    // This is a constructive SPD proof, not an inference from PCG convergence.
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..N {
        let d = case.diag_congruence[i];
        if i > 0 {
            col_idx.push((i - 1) as u32);
            values.push(-case.scale * d * case.diag_congruence[i - 1]);
        }
        col_idx.push(i as u32);
        values.push(2.0 * case.scale * d * d);
        if i + 1 < N {
            col_idx.push((i + 1) as u32);
            let value = -case.scale * d * case.diag_congruence[i + 1];
            values.push(if case.nonsymmetric && i == 0 {
                1.5 * value // A(0,1) != A(1,0): deliberately NOT symmetric
            } else {
                value
            });
        }
        row_ptr.push(col_idx.len() as u32);
    }
    Ok(Csr32Matrix::new(N, N, row_ptr, col_idx, values)?)
}

// Independent BLAS-style scaled two-norm: avoids premature sum-of-squares overflow.
fn scaled_norm(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for value in values {
        let v = value.abs();
        if !v.is_finite() {
            return f64::NAN;
        }
        if v == 0.0 {
            continue;
        }
        if scale < v {
            let ratio = scale / v;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = v;
        } else {
            let ratio = v / scale;
            ssq += ratio * ratio;
        }
    }
    scale * ssq.sqrt()
}

fn run_case(case: Case) -> Result<bool, Box<dyn Error>> {
    let a = make_matrix(case)?;
    let rhs = a.spmv(&X_REF)?;
    let expected = if case.mathematical_spd {
        "accept"
    } else {
        "reject"
    };
    let (observed, relative_residual, relative_x_error, factor_bytes) =
        match HybridPreconditioner::from_csr32(&a, vec![(0..N).collect()]) {
            Ok(factor) => {
                let mut solution = vec![0.0_f64; N];
                factor.apply(&rhs, &mut solution)?;
                let ax = a.spmv(&solution)?;
                let relative_residual = scaled_norm(rhs.iter().zip(&ax).map(|(&r, &v)| r - v))
                    / scaled_norm(rhs.iter().copied());
                let relative_x_error =
                    scaled_norm(solution.iter().zip(&X_REF).map(|(&x, &xr)| x - xr))
                        / scaled_norm(X_REF.iter().copied());
                (
                    "accept",
                    relative_residual,
                    relative_x_error,
                    factor.factor_bytes(),
                )
            }
            Err(_) => ("reject", 0.0, 0.0, 0),
        };
    println!(
        "P3A_CASE|name={}|expected={expected}|observed={observed}|relative_residual={relative_residual:.17e}|relative_x_error={relative_x_error:.17e}|factor_bytes={factor_bytes}|mathematical_spd={}|scale={:.17e}",
        case.name, case.mathematical_spd, case.scale
    );
    Ok(observed != expected)
}

fn main() -> Result<(), Box<dyn Error>> {
    let unit = [1.0; N];
    let cases = [
        Case {
            name: "spd_unit",
            scale: 1.0,
            diag_congruence: unit,
            nonsymmetric: false,
            mathematical_spd: true,
        },
        Case {
            name: "spd_small_1e12",
            scale: 1.0e-12,
            diag_congruence: unit,
            nonsymmetric: false,
            mathematical_spd: true,
        },
        Case {
            name: "spd_small_1e16",
            scale: 1.0e-16,
            diag_congruence: unit,
            nonsymmetric: false,
            mathematical_spd: true,
        },
        Case {
            name: "spd_large_1e16",
            scale: 1.0e16,
            diag_congruence: unit,
            nonsymmetric: false,
            mathematical_spd: true,
        },
        Case {
            name: "spd_congruence",
            scale: 1.0,
            diag_congruence: [1.0e-8, 1.0, 1.0, 1.0e8],
            nonsymmetric: false,
            mathematical_spd: true,
        },
        Case {
            name: "nonsym_small_1e12",
            scale: 1.0e-12,
            diag_congruence: unit,
            nonsymmetric: true,
            mathematical_spd: false,
        },
        Case {
            name: "nonsym_unit",
            scale: 1.0,
            diag_congruence: unit,
            nonsymmetric: true,
            mathematical_spd: false,
        },
    ];
    let mut discrepancies = 0;
    for case in cases {
        discrepancies += usize::from(run_case(case)?);
    }
    println!("P3A_SUMMARY|cases=7|contract_discrepancies={discrepancies}");
    // Intentional: this is a baseline *audit*; observations are not a regression
    // acceptance gate. The Python consumer reports discrepancies explicitly.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn congruence_matrix_is_symmetric() {
        let a = make_matrix(Case {
            name: "symmetry",
            scale: 1.0,
            diag_congruence: [1.0e-8, 1.0, 1.0, 1.0e8],
            nonsymmetric: false,
            mathematical_spd: true,
        })
        .unwrap();
        let row0_col1 = a.values()[1];
        let row1_col0 = a.values()[2];
        assert_eq!(row0_col1, row1_col0);
    }

    #[test]
    fn scaled_norm_is_finite_for_large_components() {
        assert!(scaled_norm([1.0e200, 1.0e200]).is_finite());
        assert!(scaled_norm([1.0e-200, 1.0e-200]) > 0.0);
    }
}
