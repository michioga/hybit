//! C1-G3-A3-A2: isolated post-admission PCG shadow rollback reference.
//!
//! This example is not wired into the production HyBIT controller. A rejected
//! speculative candidate must not destroy the *live* baseline PCG recurrence.
//! The watchdog independently verifies both paths at the same logical Krylov
//! iteration budget, then either retains the candidate or restores the exact
//! baseline x/session/workspace without restarting it.

use hybit_core::{HybitError, LinearOperator, Preconditioner, SolveStatus, SolverOptions, l2_norm};
use hybit_krylov::{
    KrylovOutcome, PcgSession, PcgWorkspace, pcg_continue_with_workspace, pcg_start_with_workspace,
};

struct BaselineCheckpoint {
    x: Vec<f64>,
    session: PcgSession,
    workspace: PcgWorkspace,
    bytes: usize,
}

impl BaselineCheckpoint {
    fn capture(
        x: &[f64],
        session: PcgSession,
        workspace: &PcgWorkspace,
        max_bytes: usize,
    ) -> Result<Self, HybitError> {
        if x.len() != workspace.len() {
            return Err(HybitError::DimensionMismatch {
                expected: workspace.len(),
                actual: x.len(),
            });
        }
        let bytes = x
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|solution| workspace.bytes().checked_add(solution))
            .ok_or(HybitError::SizeOverflow)?;
        if bytes > max_bytes {
            return Err(HybitError::InvalidArgument(
                "rollback checkpoint exceeds the explicit byte budget",
            ));
        }
        if session.is_finished() {
            return Err(HybitError::InvalidArgument(
                "rollback baseline checkpoint must be a live PCG session",
            ));
        }
        Ok(Self {
            x: x.to_vec(),
            session,
            workspace: workspace.clone(),
            bytes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct WatchdogReport {
    rolled_back: bool,
    candidate_watchdog: Option<KrylovOutcome>,
    baseline_replay: KrylovOutcome,
    candidate_true_relative: Option<f64>,
    baseline_true_relative: f64,
    selected_total_iterations: usize,
    extra_replay_iterations: usize,
    checkpoint_bytes: usize,
}

fn verified_relative(a: &dyn LinearOperator, b: &[f64], x: &[f64]) -> Result<f64, HybitError> {
    let mut residual = vec![0.0; b.len()];
    a.apply(x, &mut residual)?;
    for (ri, &bi) in residual.iter_mut().zip(b) {
        *ri = bi - *ri;
    }
    let norm = l2_norm(&residual);
    let b_norm = l2_norm(b);
    if !norm.is_finite() || !b_norm.is_finite() {
        return Err(HybitError::NumericalBreakdown(
            "rollback reference has a non-finite verified norm",
        ));
    }
    let relative = if b_norm == 0.0 { norm } else { norm / b_norm };
    if !relative.is_finite() {
        return Err(HybitError::NumericalBreakdown(
            "rollback reference has a non-finite verified relative residual",
        ));
    }
    Ok(relative)
}

/// Run one experimental watchdog window, then preserve the better valid path.
///
/// `checkpoint_iteration` is the number of selected iterations preceding the
/// speculative trial. `candidate_trial_iterations` is the *actual* count
/// already executed by the newly started candidate, not the configured cap.
/// The baseline is replayed from its true checkpoint through exactly the
/// candidate's selected logical iteration count. If baseline converges early,
/// its selected iteration count is shorter; it is never inflated artificially.
///
/// The extra checkpoint memory must already have passed `capture`'s cap.
/// Candidate errors are fail-closed if the baseline replay remains valid.
#[allow(clippy::too_many_arguments)]
fn watchdog(
    a: &dyn LinearOperator,
    baseline: &dyn Preconditioner,
    candidate: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    session: &mut PcgSession,
    workspace: &mut PcgWorkspace,
    mut checkpoint: BaselineCheckpoint,
    options: SolverOptions,
    checkpoint_iteration: usize,
    candidate_trial_iterations: usize,
    watchdog_budget: usize,
    rollback_ratio: f64,
) -> Result<WatchdogReport, HybitError> {
    if candidate_trial_iterations == 0 || watchdog_budget == 0 {
        return Err(HybitError::InvalidArgument(
            "rollback requires positive trial and watchdog budgets",
        ));
    }
    if !rollback_ratio.is_finite() || rollback_ratio < 1.0 {
        return Err(HybitError::InvalidArgument(
            "rollback ratio must be finite and >= 1",
        ));
    }
    if checkpoint_iteration
        .checked_add(candidate_trial_iterations)
        .and_then(|used| used.checked_add(watchdog_budget))
        .is_none_or(|total| total > options.max_iterations)
    {
        return Err(HybitError::InvalidArgument(
            "rollback watchdog exceeds the total Krylov iteration budget",
        ));
    }
    if x.len() != checkpoint.x.len() || workspace.len() != checkpoint.workspace.len() {
        return Err(HybitError::InvalidArgument(
            "rollback checkpoint dimension has changed",
        ));
    }

    let candidate_result =
        pcg_continue_with_workspace(a, candidate, b, x, watchdog_budget, session, workspace);
    let candidate_progress = candidate_result
        .as_ref()
        .map_or(watchdog_budget, |outcome| outcome.iterations);
    let candidate_relative = candidate_result
        .as_ref()
        .ok()
        .and_then(|_| verified_relative(a, b, x).ok());

    // Crucially, replay the *live* baseline from the checkpoint. Do not call
    // pcg_start_with_workspace: that would discard the saved search direction.
    let replay_budget = candidate_trial_iterations
        .checked_add(candidate_progress)
        .ok_or(HybitError::SizeOverflow)?;
    let baseline_outcome = pcg_continue_with_workspace(
        a,
        baseline,
        b,
        &mut checkpoint.x,
        replay_budget,
        &mut checkpoint.session,
        &mut checkpoint.workspace,
    )?;
    let baseline_relative = verified_relative(a, b, &checkpoint.x)?;
    let b_norm = l2_norm(b);
    let target = options
        .absolute_tolerance
        .max(options.relative_tolerance * b_norm.max(f64::MIN_POSITIVE));
    if !target.is_finite() {
        return Err(HybitError::NumericalBreakdown(
            "rollback reference has a non-finite convergence target",
        ));
    }
    let baseline_false_convergence =
        baseline_outcome.status == SolveStatus::Converged && baseline_relative * b_norm > target;
    let baseline_valid =
        baseline_outcome.status != SolveStatus::Breakdown && !baseline_false_convergence;
    let candidate_false_convergence = candidate_result.as_ref().is_ok_and(|outcome| {
        outcome.status == SolveStatus::Converged
            && candidate_relative.is_some_and(|value| value * b_norm > target)
    });
    let candidate_poor = candidate_result.is_err()
        || candidate_result
            .as_ref()
            .is_ok_and(|outcome| outcome.status == SolveStatus::Breakdown)
        || candidate_false_convergence
        || candidate_relative.is_none_or(|norm| norm > rollback_ratio * baseline_relative);
    let rolled_back = baseline_valid && candidate_poor;
    if !baseline_valid && candidate_poor {
        return Err(HybitError::NumericalBreakdown(
            "neither watchdog branch is eligible for safe continuation",
        ));
    }

    let selected_total_iterations = if rolled_back {
        x.copy_from_slice(&checkpoint.x);
        *session = checkpoint.session;
        *workspace = checkpoint.workspace;
        checkpoint_iteration
            .checked_add(baseline_outcome.iterations)
            .ok_or(HybitError::SizeOverflow)?
    } else {
        checkpoint_iteration
            .checked_add(replay_budget)
            .ok_or(HybitError::SizeOverflow)?
    };
    Ok(WatchdogReport {
        rolled_back,
        candidate_watchdog: candidate_result.ok(),
        baseline_replay: baseline_outcome,
        candidate_true_relative: candidate_relative,
        baseline_true_relative: baseline_relative,
        selected_total_iterations,
        extra_replay_iterations: baseline_outcome.iterations,
        checkpoint_bytes: checkpoint.bytes,
    })
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

struct DiagonalPreconditioner(Vec<f64>);
impl Preconditioner for DiagonalPreconditioner {
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

fn diagonal_system(
    exponent: f64,
) -> (
    Diagonal,
    DiagonalPreconditioner,
    DiagonalPreconditioner,
    Vec<f64>,
) {
    let values = (0..12).map(|i| 2.0f64.powi(i)).collect::<Vec<_>>();
    let candidate = values.iter().map(|v| v.powf(-exponent)).collect::<Vec<_>>();
    let n = values.len();
    (
        Diagonal(values),
        DiagonalPreconditioner(vec![1.0; n]),
        DiagonalPreconditioner(candidate),
        vec![1.0; n],
    )
}

fn options() -> SolverOptions {
    SolverOptions {
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 32,
    }
}

fn trial_and_watchdog(exponent: f64, watchdog_budget: usize) -> Result<WatchdogReport, HybitError> {
    let (a, base, candidate, b) = diagonal_system(exponent);
    let mut x = vec![0.0; b.len()];
    let mut workspace = PcgWorkspace::new(b.len());
    let mut session = pcg_start_with_workspace(&a, &base, &b, &mut x, options(), &mut workspace)?;
    let probe =
        pcg_continue_with_workspace(&a, &base, &b, &mut x, 2, &mut session, &mut workspace)?;
    let checkpoint = BaselineCheckpoint::capture(&x, session, &workspace, 4096)?;
    let mut candidate_session =
        pcg_start_with_workspace(&a, &candidate, &b, &mut x, options(), &mut workspace)?;
    let trial = pcg_continue_with_workspace(
        &a,
        &candidate,
        &b,
        &mut x,
        2,
        &mut candidate_session,
        &mut workspace,
    )?;
    watchdog(
        &a,
        &base,
        &candidate,
        &b,
        &mut x,
        &mut candidate_session,
        &mut workspace,
        checkpoint,
        options(),
        probe.iterations,
        trial.iterations,
        watchdog_budget,
        1.0,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = trial_and_watchdog(0.1, 2)?;
    println!(
        "C1G3A3A2_RESULT|rolled_back={}|candidate_true_relative={:.12e}|baseline_true_relative={:.12e}|selected_iterations={}|extra_replay_iterations={}|checkpoint_bytes={}|candidate_watchdog_iters={}|baseline_replay_status={:?}",
        report.rolled_back,
        report.candidate_true_relative.unwrap_or(f64::NAN),
        report.baseline_true_relative,
        report.selected_total_iterations,
        report.extra_replay_iterations,
        report.checkpoint_bytes,
        report
            .candidate_watchdog
            .map_or(0, |outcome| outcome.iterations),
        report.baseline_replay.status,
    );
    if !report.rolled_back {
        return Err("reference watchdog failed to reject the deteriorating candidate".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delayed_candidate_deterioration_rolls_back_to_exact_baseline() {
        let (a, base, candidate, b) = diagonal_system(0.1);
        let mut x = vec![0.0; b.len()];
        let mut ws = PcgWorkspace::new(b.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &b, &mut x, options(), &mut ws).unwrap();
        let probe =
            pcg_continue_with_workspace(&a, &base, &b, &mut x, 2, &mut session, &mut ws).unwrap();
        let at_checkpoint = verified_relative(&a, &b, &x).unwrap();
        let snapshot = BaselineCheckpoint::capture(&x, session, &ws, 4096).unwrap();

        let mut expected_x = x.clone();
        let mut expected_session = session;
        let mut expected_ws = ws.clone();
        let expected = pcg_continue_with_workspace(
            &a,
            &base,
            &b,
            &mut expected_x,
            4,
            &mut expected_session,
            &mut expected_ws,
        )
        .unwrap();

        let mut candidate_session =
            pcg_start_with_workspace(&a, &candidate, &b, &mut x, options(), &mut ws).unwrap();
        let trial = pcg_continue_with_workspace(
            &a,
            &candidate,
            &b,
            &mut x,
            2,
            &mut candidate_session,
            &mut ws,
        )
        .unwrap();
        let after_trial = verified_relative(&a, &b, &x).unwrap();
        let mut trial_baseline_x = snapshot.x.clone();
        let mut trial_baseline_session = snapshot.session;
        let mut trial_baseline_ws = snapshot.workspace.clone();
        pcg_continue_with_workspace(
            &a,
            &base,
            &b,
            &mut trial_baseline_x,
            2,
            &mut trial_baseline_session,
            &mut trial_baseline_ws,
        )
        .unwrap();
        let baseline_trial_norm = verified_relative(&a, &b, &trial_baseline_x).unwrap();
        // A short 0.95 admission gate genuinely accepts this synthetic arm.
        assert!(after_trial < at_checkpoint);
        assert!(after_trial <= 0.95 * baseline_trial_norm);

        let report = watchdog(
            &a,
            &base,
            &candidate,
            &b,
            &mut x,
            &mut candidate_session,
            &mut ws,
            snapshot,
            options(),
            probe.iterations,
            trial.iterations,
            2,
            1.0,
        )
        .unwrap();
        assert!(report.rolled_back);
        assert_eq!(report.baseline_replay.iterations, expected.iterations);
        assert_eq!(
            report.selected_total_iterations,
            probe.iterations + expected.iterations
        );
        assert!(report.candidate_true_relative.unwrap() > report.baseline_true_relative);
        assert_eq!(report.checkpoint_bytes, b.len() * 6 * 8);
        assert_eq!(x, expected_x);
        assert_eq!(
            candidate_session.final_residual(),
            expected_session.final_residual()
        );

        let continued =
            pcg_continue_with_workspace(&a, &base, &b, &mut x, 2, &mut candidate_session, &mut ws)
                .unwrap();
        let expected_continued = pcg_continue_with_workspace(
            &a,
            &base,
            &b,
            &mut expected_x,
            2,
            &mut expected_session,
            &mut expected_ws,
        )
        .unwrap();
        assert_eq!(continued.status, expected_continued.status);
        assert_eq!(continued.final_residual, expected_continued.final_residual);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn useful_candidate_is_kept_after_watchdog() {
        let report = trial_and_watchdog(0.9, 2).unwrap();
        assert!(!report.rolled_back);
        assert!(report.candidate_true_relative.unwrap() < report.baseline_true_relative);
        assert!(report.candidate_watchdog.unwrap().iterations > 0);
    }

    #[test]
    fn clone_memory_limit_rejects_before_state_mutation() {
        let (a, base, _, b) = diagonal_system(0.1);
        let mut x = vec![0.0; b.len()];
        let mut ws = PcgWorkspace::new(b.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &b, &mut x, options(), &mut ws).unwrap();
        pcg_continue_with_workspace(&a, &base, &b, &mut x, 2, &mut session, &mut ws).unwrap();
        let before = x.clone();
        let norm_before = session.final_residual();
        assert!(BaselineCheckpoint::capture(&x, session, &ws, b.len() * 6 * 8 - 1).is_err());
        assert_eq!(x, before);
        assert_eq!(session.final_residual(), norm_before);
    }

    #[test]
    fn invalid_watchdog_budget_is_fail_closed() {
        let (a, base, candidate, b) = diagonal_system(0.1);
        let mut x = vec![0.0; b.len()];
        let mut ws = PcgWorkspace::new(b.len());
        let mut session =
            pcg_start_with_workspace(&a, &base, &b, &mut x, options(), &mut ws).unwrap();
        pcg_continue_with_workspace(&a, &base, &b, &mut x, 2, &mut session, &mut ws).unwrap();
        let checkpoint = BaselineCheckpoint::capture(&x, session, &ws, 4096).unwrap();
        let mut candidate_session =
            pcg_start_with_workspace(&a, &candidate, &b, &mut x, options(), &mut ws).unwrap();
        pcg_continue_with_workspace(
            &a,
            &candidate,
            &b,
            &mut x,
            2,
            &mut candidate_session,
            &mut ws,
        )
        .unwrap();
        let before = x.clone();
        let bad_options = SolverOptions {
            max_iterations: 5,
            ..options()
        };
        assert!(
            watchdog(
                &a,
                &base,
                &candidate,
                &b,
                &mut x,
                &mut candidate_session,
                &mut ws,
                checkpoint,
                bad_options,
                2,
                2,
                2,
                1.0
            )
            .is_err()
        );
        assert_eq!(x, before);
    }
}
