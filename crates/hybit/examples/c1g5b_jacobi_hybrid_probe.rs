//! C1-G5B: independent-process Jacobi/monitored-Hybrid controls for C1-G4.
//! Research-only: no production solver behavior, public API or ABI change.
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{
    BackendPolicy, Csr32Matrix, HybitError, HybitSolver, JacobiPreconditioner, SolverOptions, pcg,
    read_matrix_market,
};

const TOL: f64 = 1e-10;
const MAX_ITERS: usize = 600;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RhsMode {
    Ones,
    IndexSmooth,
    SeededRandom,
    PointLoad,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Jacobi,
    Hybrid,
}

struct Args {
    matrix: PathBuf,
    case: String,
    rhs_mode: RhsMode,
    arm: Arm,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut case = None;
        let mut rhs_mode = None;
        let mut arm = None;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args.next().ok_or(format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(value)),
                "--case" => {
                    if !matches!(
                        value.as_str(),
                        "island_shell" | "multi_patch" | "rotated_anisotropy"
                    ) {
                        return Err("unrecognized C1-G4 matrix case".into());
                    }
                    case = Some(value);
                }
                "--rhs-mode" => {
                    rhs_mode = Some(match value.as_str() {
                        "ones" => RhsMode::Ones,
                        "index-smooth" => RhsMode::IndexSmooth,
                        "seeded-random" => RhsMode::SeededRandom,
                        "point-load" => RhsMode::PointLoad,
                        _ => return Err("unknown RHS mode".into()),
                    });
                }
                "--arm" => {
                    arm = Some(match value.as_str() {
                        "jacobi" => Arm::Jacobi,
                        "hybrid" => Arm::Hybrid,
                        _ => return Err("expected jacobi or hybrid".into()),
                    });
                }
                _ => return Err(format!("unknown flag {flag}").into()),
            }
        }
        Ok(Self {
            matrix: matrix.ok_or("missing --matrix")?,
            case: case.ok_or("missing --case")?,
            rhs_mode: rhs_mode.ok_or("missing --rhs-mode")?,
            arm: arm.ok_or("missing --arm")?,
        })
    }
}

// Copy of A9's manufactured/algebraic RHS contract; hash parity is enforced
// against the frozen C1-G4 results, never inferred from matching mode names.
fn make_rhs(
    matrix: &Csr32Matrix,
    mode: RhsMode,
) -> Result<(Vec<f64>, Option<Vec<f64>>), HybitError> {
    let n = matrix.nrows();
    if n == 0 {
        return Err(HybitError::InvalidArgument("RHS needs a nonempty matrix"));
    }
    if mode == RhsMode::PointLoad {
        let mut rhs = vec![0.0; n];
        rhs[n / 2] = 1.0;
        return Ok((rhs, None));
    }
    let exact = match mode {
        RhsMode::Ones => vec![1.0; n],
        RhsMode::IndexSmooth => (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                0.5 + 0.3 * t.sin() + 0.2 * (3.0 * t).cos()
            })
            .collect(),
        RhsMode::SeededRandom => {
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 * (2.0 / (1u64 << 53) as f64) - 1.0
                })
                .collect()
        }
        RhsMode::PointLoad => unreachable!(),
    };
    Ok((matrix.spmv(&exact)?, Some(exact)))
}

fn rhs_hash(rhs: &[f64]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for v in rhs {
        for byte in v.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn checked_residuals(
    matrix: &Csr32Matrix,
    rhs: &[f64],
    x: &[f64],
) -> Result<(f64, f64), HybitError> {
    let ax = matrix.spmv(x)?;
    let denominator = norm2(rhs).max(f64::MIN_POSITIVE);
    let naive = rhs
        .iter()
        .zip(&ax)
        .map(|(b, a)| (b - a).powi(2))
        .sum::<f64>()
        .sqrt()
        / denominator;
    let mut compensated_sq = 0.0;
    let ptr = matrix.row_ptr();
    let col = matrix.col_idx();
    let values = matrix.values();
    for i in 0..matrix.nrows() {
        let mut sum = 0.0;
        let mut correction = 0.0;
        for k in ptr[i] as usize..ptr[i + 1] as usize {
            let term = values[k] * x[col[k] as usize] - correction;
            let next = sum + term;
            correction = (next - sum) - term;
            sum = next;
        }
        compensated_sq += (rhs[i] - sum).powi(2);
    }
    Ok((naive, compensated_sq.sqrt() / denominator))
}

fn relative_error(x: &[f64], exact: Option<&[f64]>) -> String {
    let Some(exact) = exact else {
        return "n/a".to_owned();
    };
    let numerator = x
        .iter()
        .zip(exact)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    format!("{:.12e}", numerator / norm2(exact).max(f64::MIN_POSITIVE))
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let args = Args::parse()?;
    let load_start = Instant::now();
    let (matrix, _) = read_matrix_market(&args.matrix)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    if matrix.nrows() != 20736 || matrix.ncols() != 20736 {
        return Err("C1-G5B expected a frozen 20736-DOF matrix".into());
    }
    let rhs_start = Instant::now();
    let (rhs, exact) = make_rhs(&matrix, args.rhs_mode)?;
    let rhs_ms = rhs_start.elapsed().as_secs_f64() * 1000.0;
    let rhs_fingerprint = rhs_hash(&rhs);
    if !norm2(&rhs).is_finite() || norm2(&rhs) <= 0.0 {
        return Err("invalid RHS norm".into());
    }
    let options = SolverOptions {
        relative_tolerance: TOL,
        absolute_tolerance: 0.0,
        max_iterations: MAX_ITERS,
    };
    let mut x = vec![0.0; matrix.nrows()];
    let arm_start = Instant::now();
    let (
        status,
        iterations,
        reported,
        escalations,
        regions,
        factor_bytes,
        effective,
        setup_ms,
        solve_ms,
    ) = match args.arm {
        Arm::Jacobi => {
            let t = Instant::now();
            let preconditioner = JacobiPreconditioner::from_csr32(&matrix)?;
            let setup = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let result = pcg(&matrix, &preconditioner, &rhs, &mut x, options)?;
            let solve = t.elapsed().as_secs_f64() * 1000.0;
            (
                result.status,
                result.iterations,
                result.final_residual / norm2(&rhs).max(f64::MIN_POSITIVE),
                0,
                0,
                0,
                "Jacobi",
                setup,
                solve,
            )
        }
        Arm::Hybrid => {
            let mut solver = HybitSolver::new();
            solver.set_options(options)?;
            solver.set_backend_policy(BackendPolicy::Csr32);
            let mut hybrid = solver.hybrid_options();
            hybrid.probe_iterations = 12;
            hybrid.progress_monitor_interval = 64;
            hybrid.progress_monitor_rate_limit = Some(0.99);
            hybrid.max_local_region_size = 128;
            hybrid.max_local_regions = 8;
            hybrid.max_local_factor_bytes = 64 * 1024 * 1024;
            hybrid.max_escalations = 1;
            solver.set_hybrid_options(hybrid)?;
            let report = solver.solve_csr32(&matrix, &rhs, &mut x)?;
            let effective = if report.local_direct_regions > 0 && report.local_factor_bytes > 0 {
                "Hybrid"
            } else {
                "JacobiOnly"
            };
            (
                report.status,
                report.iterations,
                report.relative_residual,
                report.escalations,
                report.local_direct_regions,
                report.local_factor_bytes,
                effective,
                (report.analysis_seconds + report.prepare_seconds) * 1000.0,
                report.solve_seconds * 1000.0,
            )
        }
    };
    let (verified, compensated) = checked_residuals(&matrix, &rhs, &x)?;
    let forward = relative_error(&x, exact.as_deref());
    let work_ms = arm_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    if !(verified.is_finite() && compensated.is_finite() && reported.is_finite()) {
        return Err("nonfinite residual".into());
    }
    println!(
        "C1G5B|case={}|arm={}|rhs_mode={}|rhs_hash={}|status={:?}|iterations={}|tol={:.12e}|max_iterations={}|reported_relative={:.12e}|verified_true_relative={:.12e}|compensated_true_relative={:.12e}|relative_x_error={}|escalations={}|local_regions={}|local_factor_bytes={}|effective_preconditioner={}|rows={}|nnz={}|load_ms={:.6}|rhs_ms={:.6}|setup_ms={:.6}|solve_ms={:.6}|arm_work_ms={:.6}|inprocess_total_wall_ms={:.6}",
        args.case,
        match args.arm {
            Arm::Jacobi => "jacobi_pcg",
            Arm::Hybrid => "hybrid_local_cholesky",
        },
        match args.rhs_mode {
            RhsMode::Ones => "ones",
            RhsMode::IndexSmooth => "index_smooth",
            RhsMode::SeededRandom => "seeded_random",
            RhsMode::PointLoad => "point_load",
        },
        rhs_fingerprint,
        status,
        iterations,
        TOL,
        MAX_ITERS,
        reported,
        verified,
        compensated,
        forward,
        escalations,
        regions,
        factor_bytes,
        effective,
        matrix.nrows(),
        matrix.nnz(),
        load_ms,
        rhs_ms,
        setup_ms,
        solve_ms,
        work_ms,
        total_ms,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rhs_hash_is_deterministic() {
        assert_eq!(rhs_hash(&[0.0, 1.0]), rhs_hash(&[0.0, 1.0]));
        assert_ne!(rhs_hash(&[0.0, 1.0]), rhs_hash(&[1.0, 0.0]));
    }

    #[test]
    fn relative_error_missing_reference_is_not_zero() {
        assert_eq!(relative_error(&[1.0, 2.0], None), "n/a");
        assert_eq!(
            relative_error(&[1.0, 2.0], Some(&[1.0, 2.0])),
            "0.000000000000e0"
        );
    }
}
