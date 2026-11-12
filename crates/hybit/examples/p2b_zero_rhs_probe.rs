//! P2-B: read-only observation of the production SPD PCG stopping contract.
//! Compiled separately with feature OFF and ON. No expected-status assertions
//! on the known zero-RHS/nonzero-initial-x corner; report it as evidence.
use hybit::{BackendPolicy, Csr32Matrix, HybitSolver, SolveStatus, SolverOptions};
use std::error::Error;

fn laplacian(n: usize) -> Result<Csr32Matrix, Box<dyn Error>> {
    let mut ptr = Vec::with_capacity(n + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    ptr.push(0_u32);
    for row in 0..n {
        if row > 0 {
            cols.push((row - 1) as u32);
            vals.push(-1.0);
        }
        cols.push(row as u32);
        vals.push(2.0);
        if row + 1 < n {
            cols.push((row + 1) as u32);
            vals.push(-1.0);
        }
        ptr.push(cols.len() as u32);
    }
    Ok(Csr32Matrix::new(n, n, ptr, cols, vals)?)
}

// Stable Blue/LAPACK-style norm, independent of the internal report field.
fn scaled_norm(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for value in values {
        let magnitude = value.abs();
        if magnitude == 0.0 {
            continue;
        }
        if !magnitude.is_finite() {
            return f64::NAN;
        }
        if magnitude > scale {
            let ratio = scale / magnitude;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            ssq += ratio * ratio;
        }
    }
    scale * ssq.sqrt()
}

fn run_case(
    name: &str,
    n: usize,
    zero_rhs: bool,
    initial_one: bool,
    absolute_tolerance: f64,
    max_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    let a = laplacian(n)?;
    let x_ref: Vec<f64> = if zero_rhs {
        vec![0.0; n]
    } else {
        (0..n).map(|i| 0.5 + (i as f64 + 1.0) / n as f64).collect()
    };
    let b = a.spmv(&x_ref)?;
    let mut x = if initial_one {
        vec![1.0; n]
    } else {
        vec![0.0; n]
    };
    let ax_initial = a.spmv(&x)?;
    let initial_norm = scaled_norm(b.iter().zip(&ax_initial).map(|(&bi, &ai)| bi - ai));
    let options = SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance,
        max_iterations,
    };
    let rhs_norm = scaled_norm(b.iter().copied());
    let target = options
        .absolute_tolerance
        .max(options.relative_tolerance * rhs_norm.max(f64::MIN_POSITIVE));
    let mut solver = HybitSolver::new();
    solver.set_backend_policy(BackendPolicy::Csr32);
    solver.set_options(options)?;
    let mut hybrid = solver.hybrid_options();
    hybrid.enabled = false; // Isolate fixed Jacobi PCG; no escalation claims.
    solver.set_hybrid_options(hybrid)?;
    let report = solver.solve_csr32(&a, &b, &mut x)?;
    let ax = a.spmv(&x)?;
    let true_norm = scaled_norm(b.iter().zip(&ax).map(|(&bi, &ai)| bi - ai));
    let x_error = scaled_norm(x.iter().zip(&x_ref).map(|(&xi, &ri)| xi - ri));
    if !true_norm.is_finite() || !target.is_finite() || !x_error.is_finite() {
        return Err("P2-B nonfinite numerical result".into());
    }
    if report.iterations > max_iterations {
        return Err("P2-B global iteration budget exceeded".into());
    }
    if report.status == SolveStatus::Converged && true_norm > target {
        return Err("P2-B FALSE CERTIFIED CONVERGENCE".into());
    }
    println!(
        "P2B_CASE|name={name}|feature={}|status={:?}|iterations={}|max_iters={max_iterations}|true_norm={true_norm:.17e}|target={target:.17e}|rhs_norm={rhs_norm:.17e}|initial_norm={initial_norm:.17e}|x_error={x_error:.17e}",
        if cfg!(feature = "p2-certified-restart") {
            "on"
        } else {
            "off"
        },
        report.status,
        report.iterations,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // A zero starting point should be accepted immediately on the homogeneous system.
    run_case("zero_rhs_zero_x", 32, true, false, 0.0, 96)?;
    // Known reviewer case: relative-only threshold ~1e-318 is often unattainable.
    // Failure to converge here is observed and NOT counted as a rescue success.
    run_case("zero_rhs_nonzero_x_rel_only", 128, true, true, 0.0, 96)?;
    // Explicit absolute contract acts as an independent control.
    run_case("zero_rhs_nonzero_x_abs", 128, true, true, 1.0e-10, 96)?;
    run_case("manufactured_nonzero_rhs", 128, false, false, 0.0, 192)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operator_is_symmetric_and_spd_on_two_points() {
        let a = laplacian(2).unwrap();
        assert_eq!(a.spmv(&[1.0, 0.0]).unwrap(), vec![2.0, -1.0]);
        assert_eq!(a.spmv(&[0.0, 1.0]).unwrap(), vec![-1.0, 2.0]);
    }
    #[test]
    fn scaled_norm_retains_extreme_finite_components() {
        assert!(scaled_norm([1e-200, 1e-200]) > 0.0);
        assert_eq!(scaled_norm([0.0, 0.0]), 0.0);
    }
}
