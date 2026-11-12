//! P3-C: research-only synthetic 6-DOF/node structural SPD audit.
//!
//! This is a constructed matrix (not an assembled beam/shell FEM mesh).
//! For SPD cases A = scalar * D (K kron S) D, with scalar>0, D>0,
//! K=tridiag(-0.65,2,-0.65) SPD and S=1.375*I+0.025*11^T SPD.
//! Therefore SPD is known algebraically, without using a solver outcome.
//! The reference PCG below is deliberately independent of production PCG;
//! it verifies original-CSR residuals and makes no production rescue claim.
use hybit_core::Preconditioner;
use hybit_matrix::Csr32Matrix;
use hybit_precond::{HybridPreconditioner, JacobiPreconditioner};
use std::error::Error;

const NODES: usize = 4;
const DOFS: usize = 6;
const N: usize = NODES * DOFS;
const MAX_ITERS: usize = 96;
const TOL: f64 = 1.0e-10;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    uniform_scale: f64,
    rotation: [f64; 3],
    defect: Defect,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Defect {
    None,
    Nonsymmetric,
    Indefinite,
}

impl Case {
    fn expected(self) -> &'static str {
        if self.defect == Defect::None {
            "accept"
        } else {
            "reject"
        }
    }
}

fn diagonal_scale(case: Case, global: usize) -> f64 {
    let dof = global % DOFS;
    let node = global / DOFS;
    let node_weight = [1.0, 1.125, 0.875, 1.25][node];
    if dof < 3 {
        node_weight
    } else {
        node_weight * case.rotation[dof - 3]
    }
}

fn matrix(case: Case) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut row_ptr = Vec::with_capacity(N + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..N {
        for j in 0..N {
            let node_distance = (i / DOFS).abs_diff(j / DOFS);
            let k = if node_distance == 0 {
                2.0
            } else if node_distance == 1 {
                -0.65
            } else {
                continue;
            };
            let s = if i % DOFS == j % DOFS { 1.4 } else { 0.025 };
            let mut aij =
                (diagonal_scale(case, i) * diagonal_scale(case, j)) * (k * s) * case.uniform_scale;
            // Asymmetric perturbation is deliberately confined to two
            // rotational DOFs, so an absolute global tolerance can miss it.
            if case.defect == Defect::Nonsymmetric && i == 3 && j == 4 {
                aij *= 1.5;
            }
            if case.defect == Defect::Indefinite && i == 3 && j == 3 {
                aij = -aij;
            }
            col_idx.push(j as u32);
            values.push(aij);
        }
        row_ptr.push(col_idx.len() as u32);
    }
    Ok(Csr32Matrix::new(N, N, row_ptr, col_idx, values)?)
}

fn scaled_norm(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for x in values {
        let value = x.abs();
        if !value.is_finite() {
            return f64::NAN;
        }
        if value == 0.0 {
            continue;
        }
        if scale < value {
            let ratio = scale / value;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = value;
        } else {
            let ratio = value / scale;
            ssq += ratio * ratio;
        }
    }
    scale * ssq.sqrt()
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(&a, &b)| a * b).sum()
}

fn x_reference() -> Vec<f64> {
    (0..N)
        .map(|i| (((i * 7 + 3) % 17) as f64 - 8.0) / 9.0)
        .collect()
}

fn relative_true_residual(a: &Csr32Matrix, rhs: &[f64], x: &[f64]) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let numerator = scaled_norm(rhs.iter().zip(&ax).map(|(&b, &v)| b - v));
    let denominator = scaled_norm(rhs.iter().copied());
    Ok(if denominator == 0.0 {
        numerator
    } else {
        numerator / denominator
    })
}

struct PcgResult {
    status: &'static str,
    iterations: usize,
    true_relative_residual: f64,
}

// Reference-only fixed-SPD PCG. Independent true residual certification on
// the original CSR is evaluated each iteration, with a strict global cap.
fn reference_pcg(
    a: &Csr32Matrix,
    rhs: &[f64],
    precond: &dyn Preconditioner,
) -> Result<PcgResult, Box<dyn Error>> {
    let mut x = vec![0.0; N];
    let mut r = rhs.to_vec();
    let mut z = vec![0.0; N];
    precond.apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut status = "MaxIterations";
    let mut iterations = 0;
    let mut verified = relative_true_residual(a, rhs, &x)?;
    if verified <= TOL {
        return Ok(PcgResult {
            status: "Converged",
            iterations: 0,
            true_relative_residual: verified,
        });
    }
    if !rz.is_finite() || rz <= 0.0 {
        status = "Breakdown";
    } else {
        for k in 1..=MAX_ITERS {
            let ap = a.spmv(&p)?;
            let pap = dot(&p, &ap);
            if !pap.is_finite() || pap <= 0.0 {
                status = "Breakdown";
                break;
            }
            let alpha = rz / pap;
            if !alpha.is_finite() {
                status = "Breakdown";
                break;
            }
            for i in 0..N {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            iterations = k;
            verified = relative_true_residual(a, rhs, &x)?;
            if !verified.is_finite() {
                status = "Breakdown";
                break;
            }
            if verified <= TOL {
                status = "Converged";
                break;
            }
            precond.apply(&r, &mut z)?;
            let next = dot(&r, &z);
            if !next.is_finite() || next <= 0.0 {
                status = "Breakdown";
                break;
            }
            let beta = next / rz;
            if !beta.is_finite() {
                status = "Breakdown";
                break;
            }
            for i in 0..N {
                p[i] = z[i] + beta * p[i];
            }
            rz = next;
        }
    }
    Ok(PcgResult {
        status,
        iterations,
        true_relative_residual: verified,
    })
}

fn run_case(case: Case) -> Result<(), Box<dyn Error>> {
    let a = matrix(case)?;
    let xr = x_reference();
    let rhs = a.spmv(&xr)?;
    let expected = case.expected();
    let result = HybridPreconditioner::from_csr32(&a, vec![(0..N).collect()]);
    match result {
        Err(_) => {
            println!(
                "P3C_CASE|name={}|expected={expected}|observed=reject|full_relres=0|full_xerror=0|factor_bytes=0|estimated_bytes=0|full_positive=na|overlap_positive=na|jacobi_status=NA|jacobi_iters=0|jacobi_true_relres=0|overlap_status=NA|overlap_iters=0|overlap_true_relres=0|full_status=NA|full_iters=0|full_true_relres=0",
                case.name
            );
        }
        Ok(full) => {
            let mut solution = vec![0.0; N];
            full.apply(&rhs, &mut solution)?;
            let residual = relative_true_residual(&a, &rhs, &solution)?;
            let x_error = scaled_norm(solution.iter().zip(&xr).map(|(&x, &y)| x - y))
                / scaled_norm(xr.iter().copied());
            if expected == "reject" {
                // Document the legacy false admission without running PCG
                // on a matrix that violates the SPD contract.
                println!(
                    "P3C_CASE|name={}|expected=reject|observed=accept|full_relres={residual:.17e}|full_xerror=0|factor_bytes={}|estimated_bytes={}|full_positive=na|overlap_positive=na|jacobi_status=NA|jacobi_iters=0|jacobi_true_relres=0|overlap_status=NA|overlap_iters=0|overlap_true_relres=0|full_status=NA|full_iters=0|full_true_relres=0",
                    case.name,
                    full.factor_bytes(),
                    HybridPreconditioner::estimated_factor_bytes(N, &[(0..N).collect()])?,
                );
                return Ok(());
            }
            let estimate = HybridPreconditioner::estimated_factor_bytes(N, &[(0..N).collect()])?;
            let test_vector: Vec<f64> = (0..N).map(|i| 1.0 + (i % 7) as f64 * 0.125).collect();
            let mut full_z = vec![0.0; N];
            full.apply(&test_vector, &mut full_z)?;
            let full_positive =
                dot(&test_vector, &full_z).is_finite() && dot(&test_vector, &full_z) > 0.0;
            let overlap_regions = vec![(0..12).collect(), (6..18).collect(), (12..N).collect()];
            let overlap = HybridPreconditioner::from_csr32(&a, overlap_regions)?;
            let mut overlap_z = vec![0.0; N];
            overlap.apply(&test_vector, &mut overlap_z)?;
            let overlap_positive =
                dot(&test_vector, &overlap_z).is_finite() && dot(&test_vector, &overlap_z) > 0.0;
            let jacobi = JacobiPreconditioner::from_csr32(&a)?;
            let jacobi_pcg = reference_pcg(&a, &rhs, &jacobi)?;
            let overlap_pcg = reference_pcg(&a, &rhs, &overlap)?;
            let full_pcg = reference_pcg(&a, &rhs, &full)?;
            println!(
                "P3C_CASE|name={}|expected={expected}|observed=accept|full_relres={residual:.17e}|full_xerror={x_error:.17e}|factor_bytes={}|estimated_bytes={estimate}|full_positive={full_positive}|overlap_positive={overlap_positive}|jacobi_status={}|jacobi_iters={}|jacobi_true_relres={:.17e}|overlap_status={}|overlap_iters={}|overlap_true_relres={:.17e}|full_status={}|full_iters={}|full_true_relres={:.17e}",
                case.name,
                full.factor_bytes(),
                jacobi_pcg.status,
                jacobi_pcg.iterations,
                jacobi_pcg.true_relative_residual,
                overlap_pcg.status,
                overlap_pcg.iterations,
                overlap_pcg.true_relative_residual,
                full_pcg.status,
                full_pcg.iterations,
                full_pcg.true_relative_residual,
            );
        }
    }
    Ok(())
}

fn cases() -> [Case; 8] {
    [
        Case {
            name: "balanced_6dof",
            uniform_scale: 1.0,
            rotation: [1.0; 3],
            defect: Defect::None,
        },
        Case {
            name: "beam_rot_1e4",
            uniform_scale: 1.0,
            rotation: [1.0e-4; 3],
            defect: Defect::None,
        },
        Case {
            name: "beam_rot_1e8",
            uniform_scale: 1.0,
            rotation: [1.0e-8; 3],
            defect: Defect::None,
        },
        Case {
            name: "shell_rot_staggered",
            uniform_scale: 1.0,
            rotation: [1.0e-3, 1.0e-6, 1.0e-8],
            defect: Defect::None,
        },
        Case {
            name: "beam_rot_large_1e4",
            uniform_scale: 1.0,
            rotation: [1.0e4; 3],
            defect: Defect::None,
        },
        Case {
            name: "beam_uniform_1e16",
            uniform_scale: 1.0e-16,
            rotation: [1.0; 3],
            defect: Defect::None,
        },
        Case {
            name: "nonsym_rot_1e8",
            uniform_scale: 1.0,
            rotation: [1.0e-8; 3],
            defect: Defect::Nonsymmetric,
        },
        Case {
            name: "indefinite_rot",
            uniform_scale: 1.0,
            rotation: [1.0; 3],
            defect: Defect::Indefinite,
        },
    ]
}

fn main() -> Result<(), Box<dyn Error>> {
    for case in cases() {
        run_case(case)?;
    }
    println!(
        "P3C_SUMMARY|cases=8|nodes=4|dofs_per_node=6|reference_pcg_max_iters={MAX_ITERS}|reference_pcg_tol={TOL:.17e}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_cases_have_matching_symmetry() {
        for case in cases().into_iter().filter(|c| c.defect == Defect::None) {
            let a = matrix(case).unwrap();
            let mut d = vec![0.0; N * N];
            for i in 0..N {
                for p in a.row_ptr()[i] as usize..a.row_ptr()[i + 1] as usize {
                    d[i * N + a.col_idx()[p] as usize] += a.values()[p];
                }
            }
            for i in 0..N {
                for j in 0..N {
                    assert_eq!(d[i * N + j], d[j * N + i]);
                }
            }
        }
    }

    #[test]
    fn mixed_dof_spd_construction_has_positive_diagonal() {
        for case in cases().into_iter().filter(|c| c.defect == Defect::None) {
            let a = matrix(case).unwrap();
            assert!(
                a.diagonal()
                    .unwrap()
                    .iter()
                    .all(|&v| v.is_finite() && v > 0.0)
            );
        }
    }
}
