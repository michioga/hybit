//! C1-G3-A1: isolated shadow-trial checkpoint reference for SPD PCG.
//!
//! This example deliberately does not modify HyBIT's production controller.
//! It verifies that an unchanged-preconditioner PCG recurrence can continue
//! while a speculative candidate restarts from exactly the same x.
//!
//! The synthetic diagonal candidate below is an exact Jacobi preconditioner,
//! NOT the local Cholesky region selected by HyBIT on real FEM problems.

use hybit_core::{HybitError, LinearOperator, Preconditioner, SolveStatus, SolverOptions, l2_norm};
use hybit_krylov::{
    KrylovOutcome, PcgSession, PcgWorkspace, pcg_continue_with_workspace, pcg_start_with_workspace,
};

#[derive(Debug)]
struct ShadowReport {
    accepted_candidate: bool,
    baseline: KrylovOutcome,
    candidate: Option<KrylovOutcome>,
    baseline_verified_relative: f64,
    candidate_verified_relative: Option<f64>,
    chosen_verified_relative: f64,
    /// Both branches consume real computation even though only one is kept.
    executed_branch_iterations: usize,
    /// The copied solution plus the five copied PCG workspace vectors.
    checkpoint_bytes: usize,
}

fn verified_relative(a: &dyn LinearOperator, b: &[f64], x: &[f64]) -> Result<f64, HybitError> {
    let mut ax = vec![0.0; b.len()];
    a.apply(x, &mut ax)?;
    for (axi, &bi) in ax.iter_mut().zip(b) {
        *axi = bi - *axi;
    }
    let residual_norm = l2_norm(&ax);
    let b_norm = l2_norm(b);
    if !residual_norm.is_finite() || !b_norm.is_finite() {
        return Err(HybitError::NumericalBreakdown(
            "shadow trial produced a non-finite true residual",
        ));
    }
    Ok(residual_norm / b_norm.max(f64::MIN_POSITIVE))
}

/// Compare one unchanged PCG continuation against a restarted candidate.
///
/// The candidate runs on cloned state first. The original live recurrence
/// then advances on the baseline preconditioner. Only after independent
/// residual verification may the candidate overwrite that baseline state.
/// Candidate setup/factorization is caller-owned and is not timed here.
///
/// The caller must bound `trial_iterations` by its *remaining* solver budget.
#[allow(clippy::too_many_arguments)]
fn shadow_trial(
    a: &dyn LinearOperator,
    baseline: &dyn Preconditioner,
    candidate: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    baseline_session: &mut PcgSession,
    baseline_workspace: &mut PcgWorkspace,
    options: SolverOptions,
    trial_iterations: usize,
    candidate_accept_ratio: f64,
) -> Result<ShadowReport, HybitError> {
    if trial_iterations == 0 || trial_iterations > options.max_iterations {
        return Err(HybitError::InvalidArgument(
            "shadow trial iterations must be within the solver budget",
        ));
    }
    if !candidate_accept_ratio.is_finite()
        || candidate_accept_ratio <= 0.0
        || candidate_accept_ratio >= 1.0
    {
        return Err(HybitError::InvalidArgument(
            "candidate acceptance ratio must be finite and in (0,1)",
        ));
    }
    if baseline_session.is_finished() {
        return Err(HybitError::InvalidArgument(
            "shadow trial requires an unfinished baseline session",
        ));
    }

    let checkpoint_bytes = baseline_workspace
        .bytes()
        .checked_add(
            x.len()
                .checked_mul(std::mem::size_of::<f64>())
                .ok_or(HybitError::SizeOverflow)?,
        )
        .ok_or(HybitError::SizeOverflow)?;
    let mut candidate_x = x.to_vec();
    let mut candidate_workspace = baseline_workspace.clone();
    candidate_workspace.enter_hybrid_kernel_phase();

    // Candidate errors are fail-closed: they cannot mutate the live baseline.
    let candidate_result = (|| {
        let mut candidate_session = pcg_start_with_workspace(
            a,
            candidate,
            b,
            &mut candidate_x,
            options,
            &mut candidate_workspace,
        )?;
        let outcome = pcg_continue_with_workspace(
            a,
            candidate,
            b,
            &mut candidate_x,
            trial_iterations,
            &mut candidate_session,
            &mut candidate_workspace,
        )?;
        let verified = verified_relative(a, b, &candidate_x)?;
        Ok::<_, HybitError>((candidate_session, outcome, verified))
    })();

    // No restart here. This continues the original search direction and rz.
    let baseline_outcome = pcg_continue_with_workspace(
        a,
        baseline,
        b,
        x,
        trial_iterations,
        baseline_session,
        baseline_workspace,
    )?;
    let baseline_verified = verified_relative(a, b, x)?;
    let mut report = ShadowReport {
        accepted_candidate: false,
        baseline: baseline_outcome,
        candidate: None,
        baseline_verified_relative: baseline_verified,
        candidate_verified_relative: None,
        chosen_verified_relative: baseline_verified,
        executed_branch_iterations: baseline_outcome.iterations,
        checkpoint_bytes,
    };

    if let Ok((candidate_session, candidate_outcome, candidate_verified)) = candidate_result {
        report.candidate = Some(candidate_outcome);
        report.candidate_verified_relative = Some(candidate_verified);
        report.executed_branch_iterations = report
            .executed_branch_iterations
            .checked_add(candidate_outcome.iterations)
            .ok_or(HybitError::SizeOverflow)?;

        let b_norm = l2_norm(b);
        let absolute_target = options
            .absolute_tolerance
            .max(options.relative_tolerance * b_norm.max(f64::MIN_POSITIVE));
        let baseline_already_accurate = baseline_verified * b_norm <= absolute_target;
        let candidate_false_convergence = candidate_outcome.status == SolveStatus::Converged
            && candidate_verified * b_norm > absolute_target;
        if !baseline_already_accurate
            && !candidate_false_convergence
            && candidate_outcome.status != SolveStatus::Breakdown
            && candidate_verified <= candidate_accept_ratio * baseline_verified
        {
            x.copy_from_slice(&candidate_x);
            *baseline_session = candidate_session;
            *baseline_workspace = candidate_workspace;
            report.accepted_candidate = true;
            report.chosen_verified_relative = candidate_verified;
        }
    }
    Ok(report)
}

struct Diagonal(Vec<f64>);
impl LinearOperator for Diagonal {
    fn rows(&self) -> usize {
        self.0.len()
    }
    fn cols(&self) -> usize {
        self.0.len()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
        for ((yi, &ai), &xi) in y.iter_mut().zip(&self.0).zip(x) {
            *yi = ai * xi;
        }
        Ok(())
    }
}

struct Identity(usize);
impl Preconditioner for Identity {
    fn len(&self) -> usize {
        self.0
    }
    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

struct DiagonalInverse(Vec<f64>);
impl Preconditioner for DiagonalInverse {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
        for ((zi, &inverse), &ri) in z.iter_mut().zip(&self.0).zip(r) {
            *zi = inverse * ri;
        }
        Ok(())
    }
}

fn example_system() -> (Diagonal, Identity, DiagonalInverse, Vec<f64>) {
    let values = (0..12).map(|i| 2.0f64.powi(i)).collect::<Vec<_>>();
    let inverse = values.iter().map(|&value| 1.0 / value).collect();
    let n = values.len();
    (
        Diagonal(values),
        Identity(n),
        DiagonalInverse(inverse),
        vec![1.0; n],
    )
}

fn run() -> Result<(), HybitError> {
    let (a, base, candidate, rhs) = example_system();
    let options = SolverOptions {
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 32,
    };
    let mut x = vec![0.0; rhs.len()];
    let mut workspace = PcgWorkspace::new(rhs.len());
    let mut session = pcg_start_with_workspace(&a, &base, &rhs, &mut x, options, &mut workspace)?;
    let initial =
        pcg_continue_with_workspace(&a, &base, &rhs, &mut x, 2, &mut session, &mut workspace)?;
    if initial.status != SolveStatus::MaxIterations {
        return Err(HybitError::InvalidArgument(
            "reference baseline unexpectedly completed before the checkpoint",
        ));
    }
    let report = shadow_trial(
        &a,
        &base,
        &candidate,
        &rhs,
        &mut x,
        &mut session,
        &mut workspace,
        options,
        3,
        0.8,
    )?;
    println!(
        "C1G3A1_RESULT|accepted={}|baseline_relative={:.12e}|candidate_relative={:.12e}|chosen_relative={:.12e}|baseline_iterations={}|candidate_iterations={}|executed_branch_iterations={}|checkpoint_bytes={}",
        report.accepted_candidate,
        report.baseline_verified_relative,
        report.candidate_verified_relative.unwrap_or(f64::NAN),
        report.chosen_verified_relative,
        report.baseline.iterations,
        report.candidate.map_or(0, |outcome| outcome.iterations),
        report.executed_branch_iterations,
        report.checkpoint_bytes,
    );
    if !report.accepted_candidate || report.chosen_verified_relative > options.relative_tolerance {
        return Err(HybitError::NumericalBreakdown(
            "reference candidate did not achieve verified convergence",
        ));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> SolverOptions {
        SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 32,
        }
    }

    #[test]
    fn checkpoint_clone_continues_without_restarting() {
        let (a, base, _, rhs) = example_system();
        let mut x = vec![0.0; rhs.len()];
        let mut workspace = PcgWorkspace::new(rhs.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &rhs, &mut x, options(), &mut workspace).unwrap();
        let first =
            pcg_continue_with_workspace(&a, &base, &rhs, &mut x, 2, &mut session, &mut workspace)
                .unwrap();
        assert_eq!(first.iterations, 2);

        let mut clone_x = x.clone();
        let mut clone_session = session;
        let mut clone_workspace = workspace.clone();
        let normal =
            pcg_continue_with_workspace(&a, &base, &rhs, &mut x, 3, &mut session, &mut workspace)
                .unwrap();
        let cloned = pcg_continue_with_workspace(
            &a,
            &base,
            &rhs,
            &mut clone_x,
            3,
            &mut clone_session,
            &mut clone_workspace,
        )
        .unwrap();
        assert_eq!(normal.iterations, cloned.iterations);
        assert_eq!(normal.status, cloned.status);
        assert_eq!(normal.final_residual, cloned.final_residual);
        assert_eq!(x, clone_x);
    }

    #[test]
    fn accepts_candidate_with_independently_verified_benefit() {
        let (a, base, candidate, rhs) = example_system();
        let mut x = vec![0.0; rhs.len()];
        let mut workspace = PcgWorkspace::new(rhs.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &rhs, &mut x, options(), &mut workspace).unwrap();
        pcg_continue_with_workspace(&a, &base, &rhs, &mut x, 2, &mut session, &mut workspace)
            .unwrap();
        let report = shadow_trial(
            &a,
            &base,
            &candidate,
            &rhs,
            &mut x,
            &mut session,
            &mut workspace,
            options(),
            3,
            0.8,
        )
        .unwrap();
        assert!(report.accepted_candidate);
        assert!(report.chosen_verified_relative <= 1.0e-12);
        assert_eq!(report.checkpoint_bytes, 6 * rhs.len() * 8);
        assert!(session.is_finished());
    }

    #[test]
    fn rejection_keeps_exact_live_baseline_recurrence() {
        let (a, base, _, rhs) = example_system();
        let mut x = vec![0.0; rhs.len()];
        let mut workspace = PcgWorkspace::new(rhs.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &rhs, &mut x, options(), &mut workspace).unwrap();
        pcg_continue_with_workspace(&a, &base, &rhs, &mut x, 2, &mut session, &mut workspace)
            .unwrap();
        let mut expected_x = x.clone();
        let mut expected_workspace = workspace.clone();
        let mut expected_session = session;
        let expected = pcg_continue_with_workspace(
            &a,
            &base,
            &rhs,
            &mut expected_x,
            2,
            &mut expected_session,
            &mut expected_workspace,
        )
        .unwrap();
        // A restarted copy of the SAME preconditioner must not be promoted.
        let report = shadow_trial(
            &a,
            &base,
            &base,
            &rhs,
            &mut x,
            &mut session,
            &mut workspace,
            options(),
            2,
            1.0e-12,
        )
        .unwrap();
        assert!(!report.accepted_candidate);
        assert_eq!(report.baseline.iterations, expected.iterations);
        assert_eq!(x, expected_x);
        assert_eq!(session.final_residual(), expected_session.final_residual());
    }

    struct BadPreconditioner(usize);
    impl Preconditioner for BadPreconditioner {
        fn len(&self) -> usize {
            self.0
        }
        fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
            for (&ri, zi) in r.iter().zip(z.iter_mut()) {
                *zi = -ri;
            }
            Ok(())
        }
    }

    #[test]
    fn failed_candidate_is_rejected_without_corrupting_baseline() {
        let (a, base, _, rhs) = example_system();
        let mut x = vec![0.0; rhs.len()];
        let mut workspace = PcgWorkspace::new(rhs.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &rhs, &mut x, options(), &mut workspace).unwrap();
        pcg_continue_with_workspace(&a, &base, &rhs, &mut x, 2, &mut session, &mut workspace)
            .unwrap();
        let mut expected_x = x.clone();
        let mut expected_workspace = workspace.clone();
        let mut expected_session = session;
        pcg_continue_with_workspace(
            &a,
            &base,
            &rhs,
            &mut expected_x,
            2,
            &mut expected_session,
            &mut expected_workspace,
        )
        .unwrap();
        let report = shadow_trial(
            &a,
            &base,
            &BadPreconditioner(rhs.len()),
            &rhs,
            &mut x,
            &mut session,
            &mut workspace,
            options(),
            2,
            0.8,
        )
        .unwrap();
        assert!(!report.accepted_candidate);
        assert!(report.candidate.is_none());
        assert_eq!(x, expected_x);
        assert_eq!(session.final_residual(), expected_session.final_residual());
    }

    #[test]
    fn invalid_trial_budget_is_fail_closed() {
        let (a, base, candidate, rhs) = example_system();
        let mut x = vec![0.0; rhs.len()];
        let mut workspace = PcgWorkspace::new(rhs.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &rhs, &mut x, options(), &mut workspace).unwrap();
        let before_x = x.clone();
        assert!(
            shadow_trial(
                &a,
                &base,
                &candidate,
                &rhs,
                &mut x,
                &mut session,
                &mut workspace,
                options(),
                0,
                0.8,
            )
            .is_err()
        );
        assert_eq!(x, before_x);
    }
}
