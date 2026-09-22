use hybit_core::{
    dot, l2_norm, HybitError, LinearOperator, Preconditioner, SolveStatus, SolverOptions,
};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct KrylovOutcome {
    pub status: SolveStatus,
    pub iterations: usize,
    pub initial_residual: f64,
    pub final_residual: f64,
}

/// Stateful PCG recurrence used when a caller wants to divide one solve into
/// telemetry/control segments without restarting the Krylov method.
///
/// The session is only valid while the operator and preconditioner remain
/// unchanged. If either changes, start a new session. The vector recurrence
/// itself lives in [`PcgWorkspace`], while this object stores the scalar state
/// required to resume the next iteration exactly.
#[derive(Clone, Copy, Debug)]
pub struct PcgSession {
    target: f64,
    rz_old: f64,
    final_residual: f64,
    finished: Option<SolveStatus>,
}

impl PcgSession {
    pub fn is_finished(&self) -> bool {
        self.finished.is_some()
    }

    pub fn final_residual(&self) -> f64 {
        self.final_residual
    }
}

/// Reusable PCG scratch storage. A prepared HyBIT context allocates this once
/// and reuses it across every Krylov iteration and every subsequent RHS.
#[derive(Clone, Debug)]
pub struct PcgWorkspace {
    ax: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
}

impl PcgWorkspace {
    pub fn new(n: usize) -> Self {
        Self {
            ax: vec![0.0; n],
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.r.len()
    }
    pub fn is_empty(&self) -> bool {
        self.r.is_empty()
    }
    pub fn bytes(&self) -> usize {
        5 * self.len() * std::mem::size_of::<f64>()
    }

    fn validate_len(&self, n: usize) -> Result<(), HybitError> {
        if self.len() != n {
            return Err(HybitError::DimensionMismatch {
                expected: n,
                actual: self.len(),
            });
        }
        Ok(())
    }
}

pub fn pcg(
    a: &dyn LinearOperator,
    m: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
) -> Result<KrylovOutcome, HybitError> {
    let mut workspace = PcgWorkspace::new(a.rows());
    pcg_with_workspace(a, m, b, x, options, &mut workspace)
}

pub fn pcg_with_workspace(
    a: &dyn LinearOperator,
    m: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
    workspace: &mut PcgWorkspace,
) -> Result<KrylovOutcome, HybitError> {
    let mut session = pcg_start_with_workspace(a, m, b, x, options, workspace)?;
    pcg_continue_with_workspace(a, m, b, x, options.max_iterations, &mut session, workspace)
}

/// Initializes a resumable PCG recurrence without consuming an iteration.
///
/// The returned session may be advanced repeatedly with
/// [`pcg_continue_with_workspace`] as long as `a` and `m` are unchanged.
/// Changing the preconditioner invalidates conjugacy and requires a new
/// session, which is exactly the restart rule used by HyBIT escalation.
pub fn pcg_start_with_workspace(
    a: &dyn LinearOperator,
    m: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
    workspace: &mut PcgWorkspace,
) -> Result<PcgSession, HybitError> {
    options.validate()?;
    if a.rows() != a.cols() {
        return Err(HybitError::InvalidMatrix("PCG requires a square operator"));
    }
    let n = a.rows();
    if b.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }
    if x.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    if m.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: m.len(),
        });
    }
    workspace.validate_len(n)?;

    let PcgWorkspace { ax, r, z, p, .. } = workspace;
    a.apply(x, ax)?;
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }

    let initial_residual = l2_norm(r);
    let b_norm = l2_norm(b);
    let target = options
        .absolute_tolerance
        .max(options.relative_tolerance * b_norm.max(f64::MIN_POSITIVE));
    if initial_residual <= target {
        return Ok(PcgSession {
            target,
            rz_old: 0.0,
            final_residual: initial_residual,
            finished: Some(SolveStatus::Converged),
        });
    }

    m.apply(r, z)?;
    p.copy_from_slice(z);
    let rz_old = dot(r, z);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return Err(HybitError::NumericalBreakdown(
            "non-positive r^T M^-1 r; PCG assumptions may be violated",
        ));
    }

    Ok(PcgSession {
        target,
        rz_old,
        final_residual: initial_residual,
        finished: None,
    })
}

/// Advances an existing PCG recurrence by at most `additional_iterations`.
///
/// Unlike calling [`pcg_with_workspace`] again, this function preserves the
/// search direction and `r^T M^-1 r` scalar from the preceding segment, so no
/// Krylov information is lost at controller/telemetry boundaries.
pub fn pcg_continue_with_workspace(
    a: &dyn LinearOperator,
    m: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    additional_iterations: usize,
    session: &mut PcgSession,
    workspace: &mut PcgWorkspace,
) -> Result<KrylovOutcome, HybitError> {
    if a.rows() != a.cols() {
        return Err(HybitError::InvalidMatrix("PCG requires a square operator"));
    }
    let n = a.rows();
    if b.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }
    if x.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    if m.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: m.len(),
        });
    }
    workspace.validate_len(n)?;

    let initial_residual = session.final_residual;
    if let Some(status) = session.finished {
        return Ok(KrylovOutcome {
            status,
            iterations: 0,
            initial_residual,
            final_residual: session.final_residual,
        });
    }
    if additional_iterations == 0 {
        return Ok(KrylovOutcome {
            status: SolveStatus::MaxIterations,
            iterations: 0,
            initial_residual,
            final_residual: session.final_residual,
        });
    }

    let PcgWorkspace { r, z, p, ap, .. } = workspace;
    for iter in 1..=additional_iterations {
        a.apply(p, ap)?;
        let denom = dot(p, ap);
        if !denom.is_finite() || denom <= 0.0 {
            session.finished = Some(SolveStatus::Breakdown);
            return Ok(KrylovOutcome {
                status: SolveStatus::Breakdown,
                iterations: iter - 1,
                initial_residual,
                final_residual: session.final_residual,
            });
        }
        let alpha = session.rz_old / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        session.final_residual = l2_norm(r);
        if session.final_residual <= session.target {
            session.finished = Some(SolveStatus::Converged);
            return Ok(KrylovOutcome {
                status: SolveStatus::Converged,
                iterations: iter,
                initial_residual,
                final_residual: session.final_residual,
            });
        }
        m.apply(r, z)?;
        let rz_new = dot(r, z);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            session.finished = Some(SolveStatus::Breakdown);
            return Ok(KrylovOutcome {
                status: SolveStatus::Breakdown,
                iterations: iter,
                initial_residual,
                final_residual: session.final_residual,
            });
        }
        let beta = rz_new / session.rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        session.rz_old = rz_new;
    }

    Ok(KrylovOutcome {
        status: SolveStatus::MaxIterations,
        iterations: additional_iterations,
        initial_residual,
        final_residual: session.final_residual,
    })
}

/// Chunk size used by the experimental parallel PCG vector kernels.
///
/// The sparse operator and preconditioner may already use Rayon.  Keeping the
/// vector kernels chunked avoids creating one Rayon task per scalar while
/// still exposing enough work to the shared pool for large FEM systems.
pub const PARALLEL_PCG_VECTOR_CHUNK: usize = 16_384;

/// Number of workers in the shared Rayon pool used by parallel Krylov kernels.
pub fn parallel_vector_worker_count() -> usize {
    rayon::current_num_threads()
}

#[inline]
fn parallel_dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.par_chunks(PARALLEL_PCG_VECTOR_CHUNK)
        .zip(b.par_chunks(PARALLEL_PCG_VECTOR_CHUNK))
        .map(|(aa, bb)| aa.iter().zip(bb).map(|(x, y)| x * y).sum::<f64>())
        .sum()
}

#[inline]
fn parallel_l2_norm(x: &[f64]) -> f64 {
    parallel_dot(x, x).sqrt()
}

#[inline]
fn parallel_initial_residual(b: &[f64], ax: &[f64], r: &mut [f64]) {
    r.par_chunks_mut(PARALLEL_PCG_VECTOR_CHUNK)
        .zip(b.par_chunks(PARALLEL_PCG_VECTOR_CHUNK))
        .zip(ax.par_chunks(PARALLEL_PCG_VECTOR_CHUNK))
        .for_each(|((rr, bb), aa)| {
            for i in 0..rr.len() {
                rr[i] = bb[i] - aa[i];
            }
        });
}

/// Updates `x` and `r` and returns `||r||_2` in the same parallel traversal.
/// This removes the separate residual-norm memory pass used by the scalar PCG
/// path after every Krylov update.
#[inline]
fn parallel_update_x_r_and_norm(
    x: &mut [f64],
    r: &mut [f64],
    p: &[f64],
    ap: &[f64],
    alpha: f64,
) -> f64 {
    let sumsq = x
        .par_chunks_mut(PARALLEL_PCG_VECTOR_CHUNK)
        .zip(r.par_chunks_mut(PARALLEL_PCG_VECTOR_CHUNK))
        .zip(p.par_chunks(PARALLEL_PCG_VECTOR_CHUNK))
        .zip(ap.par_chunks(PARALLEL_PCG_VECTOR_CHUNK))
        .map(|(((xx, rr), pp), aa)| {
            let mut local = 0.0;
            for i in 0..xx.len() {
                xx[i] += alpha * pp[i];
                rr[i] -= alpha * aa[i];
                local += rr[i] * rr[i];
            }
            local
        })
        .sum::<f64>();
    sumsq.sqrt()
}

#[inline]
fn parallel_update_p(p: &mut [f64], z: &[f64], beta: f64) {
    p.par_chunks_mut(PARALLEL_PCG_VECTOR_CHUNK)
        .zip(z.par_chunks(PARALLEL_PCG_VECTOR_CHUNK))
        .for_each(|(pp, zz)| {
            for i in 0..pp.len() {
                pp[i] = zz[i] + beta * pp[i];
            }
        });
}

/// Experimental PCG variant whose dense vector kernels use the shared Rayon
/// pool.  The sparse operator and preconditioner are unchanged.
///
/// This path deliberately keeps the same five-vector [`PcgWorkspace`] and the
/// same PCG recurrence as [`pcg_with_workspace`].  The only mathematical
/// difference is floating-point reduction order in dot products/norms.  The
/// residual update and residual norm are fused into one traversal.
pub fn pcg_with_workspace_parallel_vectors(
    a: &dyn LinearOperator,
    m: &dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
    workspace: &mut PcgWorkspace,
) -> Result<KrylovOutcome, HybitError> {
    options.validate()?;
    if a.rows() != a.cols() {
        return Err(HybitError::InvalidMatrix("PCG requires a square operator"));
    }
    let n = a.rows();
    if b.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }
    if x.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    if m.len() != n {
        return Err(HybitError::DimensionMismatch {
            expected: n,
            actual: m.len(),
        });
    }
    workspace.validate_len(n)?;

    let PcgWorkspace { ax, r, z, p, ap } = workspace;
    a.apply(x, ax)?;
    parallel_initial_residual(b, ax, r);

    let initial_residual = parallel_l2_norm(r);
    let b_norm = parallel_l2_norm(b);
    let target = options
        .absolute_tolerance
        .max(options.relative_tolerance * b_norm.max(f64::MIN_POSITIVE));
    if initial_residual <= target {
        return Ok(KrylovOutcome {
            status: SolveStatus::Converged,
            iterations: 0,
            initial_residual,
            final_residual: initial_residual,
        });
    }

    m.apply(r, z)?;
    p.copy_from_slice(z);
    let mut rz_old = parallel_dot(r, z);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return Err(HybitError::NumericalBreakdown(
            "non-positive r^T M^-1 r; PCG assumptions may be violated",
        ));
    }

    let mut final_residual = initial_residual;
    for iter in 1..=options.max_iterations {
        a.apply(p, ap)?;
        let denom = parallel_dot(p, ap);
        if !denom.is_finite() || denom <= 0.0 {
            return Ok(KrylovOutcome {
                status: SolveStatus::Breakdown,
                iterations: iter - 1,
                initial_residual,
                final_residual,
            });
        }
        let alpha = rz_old / denom;
        final_residual = parallel_update_x_r_and_norm(x, r, p, ap, alpha);
        if final_residual <= target {
            return Ok(KrylovOutcome {
                status: SolveStatus::Converged,
                iterations: iter,
                initial_residual,
                final_residual,
            });
        }
        m.apply(r, z)?;
        let rz_new = parallel_dot(r, z);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            return Ok(KrylovOutcome {
                status: SolveStatus::Breakdown,
                iterations: iter,
                initial_residual,
                final_residual,
            });
        }
        let beta = rz_new / rz_old;
        parallel_update_p(p, z, beta);
        rz_old = rz_new;
    }

    Ok(KrylovOutcome {
        status: SolveStatus::MaxIterations,
        iterations: options.max_iterations,
        initial_residual,
        final_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hybit_core::LinearOperator;

    struct Identity(usize);
    impl LinearOperator for Identity {
        fn rows(&self) -> usize {
            self.0
        }
        fn cols(&self) -> usize {
            self.0
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
            y.copy_from_slice(x);
            Ok(())
        }
    }
    struct IdentityPrecond(usize);
    impl Preconditioner for IdentityPrecond {
        fn len(&self) -> usize {
            self.0
        }
        fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
            z.copy_from_slice(r);
            Ok(())
        }
    }

    #[test]
    fn parallel_vector_pcg_matches_serial() {
        let a = Identity(4096);
        let m = IdentityPrecond(4096);
        let rhs: Vec<f64> = (0..4096).map(|i| 1.0 + (i % 17) as f64 * 0.125).collect();
        let mut serial = vec![0.0; rhs.len()];
        let mut parallel = vec![0.0; rhs.len()];
        let mut ws_serial = PcgWorkspace::new(rhs.len());
        let mut ws_parallel = PcgWorkspace::new(rhs.len());
        let options = SolverOptions::default();
        let out_serial =
            pcg_with_workspace(&a, &m, &rhs, &mut serial, options, &mut ws_serial).unwrap();
        let out_parallel = pcg_with_workspace_parallel_vectors(
            &a,
            &m,
            &rhs,
            &mut parallel,
            options,
            &mut ws_parallel,
        )
        .unwrap();
        assert_eq!(out_serial.status, SolveStatus::Converged);
        assert_eq!(out_parallel.status, SolveStatus::Converged);
        assert_eq!(out_parallel.iterations, 1);
        for (xs, xp) in serial.iter().zip(&parallel) {
            assert!((xs - xp).abs() <= 1.0e-12);
        }
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

    #[test]
    fn segmented_pcg_preserves_recurrence() {
        let a = Diagonal(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        let m = IdentityPrecond(6);
        let rhs = vec![1.0; 6];
        let options = SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 12,
        };

        let mut x_full = vec![0.0; 6];
        let mut ws_full = PcgWorkspace::new(6);
        let full = pcg_with_workspace(&a, &m, &rhs, &mut x_full, options, &mut ws_full).unwrap();

        let mut x_segmented = vec![0.0; 6];
        let mut ws_segmented = PcgWorkspace::new(6);
        let mut session =
            pcg_start_with_workspace(&a, &m, &rhs, &mut x_segmented, options, &mut ws_segmented)
                .unwrap();
        let first = pcg_continue_with_workspace(
            &a,
            &m,
            &rhs,
            &mut x_segmented,
            2,
            &mut session,
            &mut ws_segmented,
        )
        .unwrap();
        assert_eq!(first.status, SolveStatus::MaxIterations);
        assert_eq!(first.iterations, 2);
        let second = pcg_continue_with_workspace(
            &a,
            &m,
            &rhs,
            &mut x_segmented,
            options.max_iterations - first.iterations,
            &mut session,
            &mut ws_segmented,
        )
        .unwrap();

        assert_eq!(second.status, full.status);
        assert_eq!(first.iterations + second.iterations, full.iterations);
        assert!((second.final_residual - full.final_residual).abs() <= 1.0e-14);
        for (segmented, reference) in x_segmented.iter().zip(&x_full) {
            assert!((segmented - reference).abs() <= 1.0e-14);
        }
    }

    #[test]
    fn reusable_workspace_solves_multiple_rhs() {
        let a = Identity(4);
        let m = IdentityPrecond(4);
        let mut ws = PcgWorkspace::new(4);
        for rhs in [vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]] {
            let mut x = vec![0.0; 4];
            let out = pcg_with_workspace(&a, &m, &rhs, &mut x, SolverOptions::default(), &mut ws)
                .unwrap();
            assert_eq!(out.status, SolveStatus::Converged);
            assert_eq!(x, rhs);
        }
        assert_eq!(ws.bytes(), 5 * 4 * 8);
    }
}
