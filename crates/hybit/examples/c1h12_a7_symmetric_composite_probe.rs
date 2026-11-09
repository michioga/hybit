//! C1-H12-A7: opt-in two-level SPD correction probe, not production AMG.
//! The policy adds B^-1 to a congruence transform of the existing additive
//! two-level preconditioner. Both arms use the same certified PCG routine.

use std::cell::RefCell;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    Csr32Matrix, HybitError, JacobiPreconditioner, LinearOperator, Preconditioner, SolveStatus,
    SolverOptions, TwoLevelAggregation, TwoLevelBasis, TwoLevelBlockJacobiPreconditioner,
    TwoLevelCoarseApplyPolicy, TwoLevelTransferApplyPolicy, TwoLevelTransferOptions,
    TwoLevelTransferStoragePolicy, TwoLevelTransferValueStoragePolicy, pcg, read_matrix_market,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    Additive,
    SymmetricComposite,
}

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    case_name: String,
    aggregation: TwoLevelAggregation,
    basis: TwoLevelBasis,
    policy: Policy,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut case_name = None;
        let mut aggregation = None;
        let mut basis = None;
        let mut policy = None;
        let mut words = env::args().skip(1);
        while let Some(flag) = words.next() {
            let value = words.next().ok_or(format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(value)),
                "--case" => case_name = Some(value),
                "--aggregation" => {
                    aggregation = Some(match value.as_str() {
                        "contiguous" => TwoLevelAggregation::Contiguous,
                        "graph" => TwoLevelAggregation::Graph,
                        "strong-graph" => TwoLevelAggregation::StrongGraph,
                        _ => return Err(format!("unknown aggregation {value}").into()),
                    });
                }
                "--basis" => {
                    basis = Some(match value.as_str() {
                        "piecewise" => TwoLevelBasis::PiecewiseConstant,
                        "smoothed" => TwoLevelBasis::JacobiSmoothed,
                        _ => return Err(format!("unknown basis {value}").into()),
                    });
                }
                "--policy" => {
                    policy = Some(match value.as_str() {
                        "additive" => Policy::Additive,
                        "symmetric" => Policy::SymmetricComposite,
                        _ => return Err(format!("unknown policy {value}").into()),
                    });
                }
                _ => return Err(format!("unknown flag {flag}").into()),
            }
        }
        let case_name = case_name.ok_or("missing --case")?;
        if !matches!(
            case_name.as_str(),
            "island_shell" | "multi_patch" | "rotated_anisotropy"
        ) {
            return Err("unrecognized A7 FEM case".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("missing --matrix")?,
            case_name,
            aggregation: aggregation.ok_or("missing --aggregation")?,
            basis: basis.ok_or("missing --basis")?,
            policy: policy.ok_or("missing --policy")?,
        })
    }
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn verified_true_relative(matrix: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<f64, HybitError> {
    let ax = matrix.spmv(x)?;
    let squared = b
        .iter()
        .zip(&ax)
        .map(|(bi, axi)| (bi - axi).powi(2))
        .sum::<f64>();
    Ok(squared.sqrt() / norm2(b).max(f64::MIN_POSITIVE))
}

struct Scratch {
    pre: Vec<f64>,
    ax: Vec<f64>,
    residual: Vec<f64>,
    center: Vec<f64>,
    post: Vec<f64>,
    calls: usize,
}

/// B^-1 + (I - B^-1 A) * M_add^-1 * (I - A B^-1).
/// For SPD A, B and M_add this sum is SPD and symmetric. Unlike an
/// unsymmetrized multiplicative coarse correction, it is PCG-compatible.
struct SymmetricComposite<'a> {
    matrix: &'a Csr32Matrix,
    fine: &'a JacobiPreconditioner,
    additive: &'a TwoLevelBlockJacobiPreconditioner,
    scratch: RefCell<Scratch>,
}

impl<'a> SymmetricComposite<'a> {
    fn new(
        matrix: &'a Csr32Matrix,
        fine: &'a JacobiPreconditioner,
        additive: &'a TwoLevelBlockJacobiPreconditioner,
    ) -> Self {
        let n = matrix.nrows();
        Self {
            matrix,
            fine,
            additive,
            scratch: RefCell::new(Scratch {
                pre: vec![0.0; n],
                ax: vec![0.0; n],
                residual: vec![0.0; n],
                center: vec![0.0; n],
                post: vec![0.0; n],
                calls: 0,
            }),
        }
    }

    fn calls(&self) -> usize {
        self.scratch.borrow().calls
    }

    fn extra_workspace_bytes(&self) -> usize {
        5 * self.matrix.nrows() * std::mem::size_of::<f64>()
    }
}

impl Preconditioner for SymmetricComposite<'_> {
    fn len(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        let n = self.len();
        if r.len() != n {
            return Err(HybitError::DimensionMismatch {
                expected: n,
                actual: r.len(),
            });
        }
        if z.len() != n {
            return Err(HybitError::DimensionMismatch {
                expected: n,
                actual: z.len(),
            });
        }
        let mut scratch = self.scratch.borrow_mut();
        let Scratch {
            pre,
            ax,
            residual,
            center,
            post,
            calls,
        } = &mut *scratch;
        self.fine.apply(r, pre)?;
        self.matrix.apply(pre, ax)?;
        for i in 0..n {
            residual[i] = r[i] - ax[i];
        }
        self.additive.apply(residual, center)?;
        self.matrix.apply(center, ax)?;
        self.fine.apply(ax, post)?;
        for i in 0..n {
            z[i] = pre[i] + center[i] - post[i];
        }
        *calls += 1;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Certified {
    status: SolveStatus,
    iterations: usize,
    reported_final: f64,
    true_relative: f64,
    true_checks: usize,
    restarts: usize,
    rejected: usize,
}

/// Same restart-on-false-recursive-convergence policy as the A5 experiment.
fn solve_certified(
    matrix: &Csr32Matrix,
    preconditioner: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
) -> Result<Certified, Box<dyn Error>> {
    let mut iterations = 0;
    let mut true_checks = 0;
    let mut restarts = 0;
    let mut rejected = 0;
    let mut reported_final = norm2(b);
    loop {
        let remaining = options.max_iterations - iterations;
        if remaining == 0 {
            return Ok(Certified {
                status: SolveStatus::MaxIterations,
                iterations,
                reported_final,
                true_relative: verified_true_relative(matrix, b, x)?,
                true_checks,
                restarts,
                rejected,
            });
        }
        let mut segment = options;
        segment.max_iterations = remaining;
        let out = pcg(matrix, preconditioner, b, x, segment)?;
        iterations += out.iterations;
        reported_final = out.final_residual;
        true_checks += 1;
        let true_relative = verified_true_relative(matrix, b, x)?;
        if !true_relative.is_finite() {
            return Err("nonfinite original-CSR true residual".into());
        }
        if out.status == SolveStatus::Converged && true_relative <= options.relative_tolerance {
            return Ok(Certified {
                status: SolveStatus::Converged,
                iterations,
                reported_final,
                true_relative,
                true_checks,
                restarts,
                rejected,
            });
        }
        if out.status == SolveStatus::Converged {
            rejected += 1;
            if iterations < options.max_iterations && out.iterations > 0 {
                restarts += 1;
                continue;
            }
        }
        return Ok(Certified {
            status: if out.status == SolveStatus::Converged {
                if iterations == options.max_iterations {
                    SolveStatus::MaxIterations
                } else {
                    SolveStatus::Breakdown
                }
            } else {
                out.status
            },
            iterations,
            reported_final,
            true_relative,
            true_checks,
            restarts,
            rejected,
        });
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let load_start = Instant::now();
    let (matrix, _) = read_matrix_market(&args.matrix)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1.0e3;
    if matrix.nrows() != matrix.ncols() || matrix.nrows() == 0 {
        return Err("A7 requires a nonempty square matrix".into());
    }
    let n = matrix.nrows();
    let ones = vec![1.0; n];
    let b = matrix.spmv(&ones)?;

    let target_coarse_dimension = 512usize;
    let aggregate_nodes = n.div_ceil(target_coarse_dimension).max(1);
    let setup_start = Instant::now();
    let additive =
        TwoLevelBlockJacobiPreconditioner::from_csr32_with_aggregation_basis_and_transfer_options(
            &matrix,
            1,
            aggregate_nodes,
            args.aggregation,
            args.basis,
            TwoLevelCoarseApplyPolicy::Auto,
            TwoLevelTransferOptions {
                apply_policy: TwoLevelTransferApplyPolicy::Parallel,
                storage_policy: TwoLevelTransferStoragePolicy::Wide,
                value_storage_policy: TwoLevelTransferValueStoragePolicy::F64,
            },
        )?;
    let fine = if args.policy == Policy::SymmetricComposite {
        Some(JacobiPreconditioner::from_csr32(&matrix)?)
    } else {
        None
    };
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;

    let options = SolverOptions {
        relative_tolerance: 1e-10,
        absolute_tolerance: 0.0,
        max_iterations: 600,
    };
    let mut x = vec![0.0; n];
    let solve_start = Instant::now();
    let (out, calls, scratch_bytes) = match args.policy {
        Policy::Additive => (
            solve_certified(&matrix, &additive, &b, &mut x, options)?,
            0,
            0,
        ),
        Policy::SymmetricComposite => {
            let fine = fine.as_ref().ok_or("missing A7 Jacobi smoother")?;
            let composite = SymmetricComposite::new(&matrix, fine, &additive);
            let out = solve_certified(&matrix, &composite, &b, &mut x, options)?;
            (out, composite.calls(), composite.extra_workspace_bytes())
        }
    };
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1.0e3;

    let relative_x_error =
        x.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>().sqrt() / (n as f64).sqrt();
    let reported_relative = out.reported_final / norm2(&b).max(f64::MIN_POSITIVE);
    println!(
        "C1H12A7|case={}|policy={:?}|aggregation={:?}|basis={:?}|rows={}|nnz={}|coarse_dimension={}|coarse_factor_bytes={}|status={:?}|iterations={}|tol={:.12e}|reported_relative={:.12e}|verified_true_relative={:.12e}|relative_x_error={:.12e}|load_ms={load_ms:.6}|setup_ms={setup_ms:.6}|solve_ms={solve_ms:.6}|total_wall_ms={:.6}|true_checks={}|restarts={}|rejected_recursive_convergences={}|precond_calls={}|extra_spmv_calls={}|scratch_bytes={}",
        args.case_name,
        args.policy,
        args.aggregation,
        args.basis,
        n,
        matrix.nnz(),
        additive.coarse_dimension(),
        additive.factor_bytes(),
        out.status,
        out.iterations,
        options.relative_tolerance,
        reported_relative,
        out.true_relative,
        relative_x_error,
        load_ms + setup_ms + solve_ms,
        out.true_checks,
        out.restarts,
        out.rejected,
        calls,
        2 * calls,
        scratch_bytes,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_spd() -> Csr32Matrix {
        Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.1, -0.2, -0.2, 1.8, -0.3, -0.3, 2.6],
        )
        .unwrap()
    }

    #[test]
    fn symmetric_composite_is_spd_and_symmetric_on_reference() {
        let matrix = small_spd();
        let fine = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let additive = TwoLevelBlockJacobiPreconditioner::from_csr32(&matrix, 1, 2).unwrap();
        let precond = SymmetricComposite::new(&matrix, &fine, &additive);
        let u = [0.3, -1.1, 0.8];
        let v = [0.6, 0.1, -0.2];
        let mut mu = [0.0; 3];
        let mut mv = [0.0; 3];
        precond.apply(&u, &mut mu).unwrap();
        precond.apply(&v, &mut mv).unwrap();
        let positivity = u.iter().zip(&mu).map(|(x, z)| *x * *z).sum::<f64>();
        assert!(positivity > 0.0);
        let uv = u.iter().zip(&mv).map(|(x, z)| *x * *z).sum::<f64>();
        let vu = v.iter().zip(&mu).map(|(x, z)| *x * *z).sum::<f64>();
        assert!((uv - vu).abs() < 1e-12);
        assert_eq!(precond.calls(), 2);
        assert_eq!(precond.extra_workspace_bytes(), 120);
    }

    #[test]
    fn composite_pcg_converges_with_true_residual_certificate() {
        let matrix = small_spd();
        let fine = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let additive = TwoLevelBlockJacobiPreconditioner::from_csr32(&matrix, 1, 2).unwrap();
        let precond = SymmetricComposite::new(&matrix, &fine, &additive);
        let b = matrix.spmv(&[1.0, 1.0, 1.0]).unwrap();
        let mut x = [0.0; 3];
        let out = solve_certified(
            &matrix,
            &precond,
            &b,
            &mut x,
            SolverOptions {
                relative_tolerance: 1e-10,
                absolute_tolerance: 0.0,
                max_iterations: 30,
            },
        )
        .unwrap();
        assert_eq!(out.status, SolveStatus::Converged);
        assert!(out.true_relative <= 1e-10);
        assert!(out.iterations <= 30);
    }

    #[test]
    fn zero_dimensional_input_rejected_by_jacobi() {
        let matrix = small_spd();
        let fine = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let additive = TwoLevelBlockJacobiPreconditioner::from_csr32(&matrix, 1, 2).unwrap();
        let precond = SymmetricComposite::new(&matrix, &fine, &additive);
        assert!(precond.apply(&[0.0; 2], &mut [0.0; 3]).is_err());
        assert!(precond.apply(&[0.0; 3], &mut [0.0; 2]).is_err());
    }
}
