use hybit_core::{
    dot, l2_norm, HybitError, LinearOperator, Preconditioner, SolveStatus, SolverOptions,
};

use crate::KrylovOutcome;

/// Preconditioner contract for flexible Krylov methods.
///
/// Unlike [`Preconditioner`], this method receives the global Krylov iteration
/// and a mutable receiver. Implementations may therefore change internal state
/// or select a different approximate inverse at every iteration.
///
/// Existing fixed preconditioners automatically implement this trait through
/// the blanket implementation below.
pub trait FlexiblePreconditioner {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn apply(
        &mut self,
        iteration: usize,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), HybitError>;
}

impl<T> FlexiblePreconditioner for T
where
    T: Preconditioner + ?Sized,
{
    fn len(&self) -> usize {
        Preconditioner::len(self)
    }

    fn apply(
        &mut self,
        _iteration: usize,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), HybitError> {
        Preconditioner::apply(self, input, output)
    }
}

/// Restarted FGMRES configuration.
///
/// `solver.max_iterations` is the total iteration budget across all restart
/// cycles. `restart` is the maximum Arnoldi basis dimension per cycle.
#[derive(Clone, Copy, Debug)]
pub struct FgmresOptions {
    pub solver: SolverOptions,
    pub restart: usize,
}

impl Default for FgmresOptions {
    fn default() -> Self {
        Self {
            solver: SolverOptions::default(),
            restart: 30,
        }
    }
}

impl FgmresOptions {
    pub fn validate(&self) -> Result<(), HybitError> {
        self.solver.validate()?;
        if self.restart == 0 {
            return Err(HybitError::InvalidArgument("FGMRES restart must be > 0"));
        }
        Ok(())
    }
}

/// Exact restart-boundary telemetry exposed to an FGMRES restart controller.
///
/// Residuals are exact norms from `b - A x`, not Hessenberg estimates. The
/// controller runs only after a completed restart cycle that has not already
/// converged or broken down. Returning the current restart preserves ordinary
/// restarted FGMRES; returning another positive value changes the next cycle's
/// Arnoldi dimension without resetting the global Krylov iteration count.
#[derive(Clone, Copy, Debug)]
pub struct FgmresRestartProgress {
    pub restart: usize,
    pub cycle_iterations: usize,
    pub total_iterations: usize,
    pub initial_residual: f64,
    pub cycle_initial_residual: f64,
    pub final_residual: f64,
    pub remaining_iterations: usize,
}
/// Reusable restarted-FGMRES storage.
///
/// For restart `m`, the workspace owns `m + 1` Arnoldi basis vectors `V`,
/// `m` independently preconditioned basis vectors `Z`, two full-size scratch
/// vectors, and the dense `(m + 1) x m` Hessenberg/Givens state.
///
/// No vector allocation occurs inside the FGMRES iteration loop.
#[derive(Clone, Debug)]
pub struct FgmresWorkspace {
    n: usize,
    restart: usize,
    v: Vec<Vec<f64>>,
    z: Vec<Vec<f64>>,
    r: Vec<f64>,
    w: Vec<f64>,
    h: Vec<f64>,
    cs: Vec<f64>,
    sn: Vec<f64>,
    g: Vec<f64>,
    y: Vec<f64>,
}

impl FgmresWorkspace {
    pub fn new(n: usize, restart: usize) -> Result<Self, HybitError> {
        if restart == 0 {
            return Err(HybitError::InvalidArgument("FGMRES restart must be > 0"));
        }

        let basis_count = restart.checked_add(1).ok_or(HybitError::SizeOverflow)?;
        let h_len = basis_count
            .checked_mul(restart)
            .ok_or(HybitError::SizeOverflow)?;

        let mut v = Vec::with_capacity(basis_count);
        for _ in 0..basis_count {
            v.push(vec![0.0; n]);
        }

        let mut z = Vec::with_capacity(restart);
        for _ in 0..restart {
            z.push(vec![0.0; n]);
        }

        Ok(Self {
            n,
            restart,
            v,
            z,
            r: vec![0.0; n],
            w: vec![0.0; n],
            h: vec![0.0; h_len],
            cs: vec![0.0; restart],
            sn: vec![0.0; restart],
            g: vec![0.0; basis_count],
            y: vec![0.0; restart],
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Maximum restart dimension this workspace can serve.
    ///
    /// A workspace allocated for restart capacity `m` may be reused by solves
    /// whose requested restart is any value in `1..=m`.
    pub fn restart(&self) -> usize {
        self.restart
    }

    /// Persistent numeric payload owned by the workspace.
    pub fn bytes(&self) -> usize {
        let vector_values = self
            .v
            .iter()
            .map(Vec::len)
            .sum::<usize>()
            .saturating_add(self.z.iter().map(Vec::len).sum::<usize>())
            .saturating_add(self.r.len())
            .saturating_add(self.w.len());

        let dense_values = self
            .h
            .len()
            .saturating_add(self.cs.len())
            .saturating_add(self.sn.len())
            .saturating_add(self.g.len())
            .saturating_add(self.y.len());

        vector_values
            .saturating_add(dense_values)
            .saturating_mul(std::mem::size_of::<f64>())
    }

    fn validate(&self, n: usize, restart: usize) -> Result<(), HybitError> {
        if self.n != n {
            return Err(HybitError::DimensionMismatch {
                expected: n,
                actual: self.n,
            });
        }
        if self.restart < restart {
            return Err(HybitError::InvalidArgument(
                "FGMRES workspace restart capacity is smaller than options",
            ));
        }
        Ok(())
    }

    #[inline]
    fn h_index(&self, row: usize, col: usize) -> usize {
        col * (self.restart + 1) + row
    }

    fn reset_cycle(&mut self) {
        self.h.fill(0.0);
        self.cs.fill(0.0);
        self.sn.fill(0.0);
        self.g.fill(0.0);
        self.y.fill(0.0);
    }
}

fn finite_norm(x: &[f64], label: &'static str) -> Result<f64, HybitError> {
    let norm = l2_norm(x);
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(HybitError::NumericalBreakdown(label))
    }
}

fn exact_residual(
    a: &dyn LinearOperator,
    b: &[f64],
    x: &[f64],
    workspace: &mut FgmresWorkspace,
) -> Result<f64, HybitError> {
    a.apply(x, &mut workspace.w)?;
    for ((ri, &bi), &axi) in workspace.r.iter_mut().zip(b).zip(&workspace.w) {
        *ri = bi - axi;
    }
    finite_norm(&workspace.r, "FGMRES residual became non-finite")
}

fn backsolve_and_update(
    workspace: &mut FgmresWorkspace,
    x: &mut [f64],
    basis_count: usize,
) -> Result<(), HybitError> {
    for i in (0..basis_count).rev() {
        let mut rhs = workspace.g[i];
        for j in (i + 1)..basis_count {
            rhs -= workspace.h[workspace.h_index(i, j)] * workspace.y[j];
        }

        let diag = workspace.h[workspace.h_index(i, i)];
        if !diag.is_finite() || diag == 0.0 {
            return Err(HybitError::NumericalBreakdown(
                "FGMRES Hessenberg factor is singular",
            ));
        }
        workspace.y[i] = rhs / diag;
    }

    for j in 0..basis_count {
        let alpha = workspace.y[j];
        if !alpha.is_finite() {
            return Err(HybitError::NumericalBreakdown(
                "FGMRES least-squares solution became non-finite",
            ));
        }
        for (xi, &zji) in x.iter_mut().zip(&workspace.z[j]) {
            *xi += alpha * zji;
        }
    }

    Ok(())
}

pub fn fgmres<M>(
    a: &dyn LinearOperator,
    m: &mut M,
    b: &[f64],
    x: &mut [f64],
    options: FgmresOptions,
) -> Result<KrylovOutcome, HybitError>
where
    M: FlexiblePreconditioner + ?Sized,
{
    let mut workspace = FgmresWorkspace::new(a.rows(), options.restart)?;
    fgmres_with_workspace(a, m, b, x, options, &mut workspace)
}

/// Restarted right-preconditioned flexible GMRES.
///
/// The preconditioned Arnoldi vectors `Z_j = M_j^{-1} V_j` are stored
/// independently from the orthonormal basis `V`, so the preconditioner may
/// change at every Krylov iteration. Modified Gram-Schmidt is followed by a
/// second reorthogonalization pass for a conservative first implementation.
///
/// The residual is recomputed from `b - A x` at every restart and before
/// reporting convergence, avoiding acceptance based only on the small
/// Hessenberg residual estimate.
pub fn fgmres_with_workspace<M>(
    a: &dyn LinearOperator,
    m: &mut M,
    b: &[f64],
    x: &mut [f64],
    options: FgmresOptions,
    workspace: &mut FgmresWorkspace,
) -> Result<KrylovOutcome, HybitError>
where
    M: FlexiblePreconditioner + ?Sized,
{
    let fixed_restart = options.restart;
    fgmres_with_workspace_and_restart_controller(a, m, b, x, options, workspace, move |_| {
        Ok(fixed_restart)
    })
}

/// Restarted right-preconditioned FGMRES with a restart-boundary controller.
///
/// The controller may change the Arnoldi dimension for the next cycle while the
/// solve remains inside one FGMRES invocation. This preserves the global
/// flexible-preconditioner iteration number and avoids an extra initial
/// residual SpMV that would be incurred by re-entering FGMRES at every cycle.
///
/// The returned restart must be positive and no larger than the capacity of
/// `workspace`.
pub fn fgmres_with_workspace_and_restart_controller<M, F>(
    a: &dyn LinearOperator,
    m: &mut M,
    b: &[f64],
    x: &mut [f64],
    options: FgmresOptions,
    workspace: &mut FgmresWorkspace,
    mut controller: F,
) -> Result<KrylovOutcome, HybitError>
where
    M: FlexiblePreconditioner + ?Sized,
    F: FnMut(FgmresRestartProgress) -> Result<usize, HybitError>,
{
    options.validate()?;

    if a.rows() != a.cols() {
        return Err(HybitError::InvalidMatrix(
            "FGMRES requires a square operator",
        ));
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
    workspace.validate(n, options.restart)?;
    let mut current_restart = options.restart;

    let b_norm = finite_norm(b, "FGMRES RHS norm became non-finite")?;
    let target = options
        .solver
        .absolute_tolerance
        .max(options.solver.relative_tolerance * b_norm.max(f64::MIN_POSITIVE));

    let initial_residual = exact_residual(a, b, x, workspace)?;
    if initial_residual <= target {
        return Ok(KrylovOutcome {
            status: SolveStatus::Converged,
            iterations: 0,
            initial_residual,
            final_residual: initial_residual,
        });
    }

    let mut total_iterations = 0usize;
    let mut final_residual = initial_residual;

    while total_iterations < options.solver.max_iterations {
        workspace.reset_cycle();

        let beta = finite_norm(&workspace.r, "FGMRES restart residual became non-finite")?;
        let cycle_initial_residual = beta;
        let cycle_start_iterations = total_iterations;
        if beta <= target {
            return Ok(KrylovOutcome {
                status: SolveStatus::Converged,
                iterations: total_iterations,
                initial_residual,
                final_residual: beta,
            });
        }

        for i in 0..n {
            workspace.v[0][i] = workspace.r[i] / beta;
        }
        workspace.g[0] = beta;

        let remaining = options.solver.max_iterations - total_iterations;
        let cycle_limit = current_restart.min(remaining).min(n.max(1));
        let mut cycle_updated = false;

        for j in 0..cycle_limit {
            m.apply(total_iterations, &workspace.v[j], &mut workspace.z[j])?;

            a.apply(&workspace.z[j], &mut workspace.w)?;
            let operator_norm =
                finite_norm(&workspace.w, "FGMRES Arnoldi vector became non-finite")?;

            // Two-pass modified Gram-Schmidt. The first pass forms H; the
            // second accumulates the correction into the same coefficients.
            for _ in 0..2 {
                for i in 0..=j {
                    let correction = dot(&workspace.w, &workspace.v[i]);
                    if !correction.is_finite() {
                        return Err(HybitError::NumericalBreakdown(
                            "FGMRES orthogonalization became non-finite",
                        ));
                    }

                    let index = workspace.h_index(i, j);
                    workspace.h[index] += correction;

                    for k in 0..n {
                        workspace.w[k] -= correction * workspace.v[i][k];
                    }
                }
            }

            let next_norm = finite_norm(&workspace.w, "FGMRES Arnoldi norm became non-finite")?;
            let happy_tolerance = f64::EPSILON * operator_norm.max(1.0);
            let happy_breakdown = next_norm <= happy_tolerance;

            let next_index = workspace.h_index(j + 1, j);
            workspace.h[next_index] = next_norm;

            if !happy_breakdown {
                for k in 0..n {
                    workspace.v[j + 1][k] = workspace.w[k] / next_norm;
                }
            } else {
                workspace.v[j + 1].fill(0.0);
            }

            // Apply previous Givens rotations to the new Hessenberg column.
            for i in 0..j {
                let upper_index = workspace.h_index(i, j);
                let lower_index = workspace.h_index(i + 1, j);
                let upper = workspace.h[upper_index];
                let lower = workspace.h[lower_index];

                workspace.h[upper_index] = workspace.cs[i] * upper + workspace.sn[i] * lower;
                workspace.h[lower_index] = -workspace.sn[i] * upper + workspace.cs[i] * lower;
            }

            let diagonal_index = workspace.h_index(j, j);
            let subdiagonal_index = workspace.h_index(j + 1, j);
            let diagonal = workspace.h[diagonal_index];
            let subdiagonal = workspace.h[subdiagonal_index];
            let rho = diagonal.hypot(subdiagonal);

            if !rho.is_finite() || rho == 0.0 {
                return Ok(KrylovOutcome {
                    status: SolveStatus::Breakdown,
                    iterations: total_iterations,
                    initial_residual,
                    final_residual,
                });
            }

            workspace.cs[j] = diagonal / rho;
            workspace.sn[j] = subdiagonal / rho;
            workspace.h[diagonal_index] = rho;
            workspace.h[subdiagonal_index] = 0.0;

            let gj = workspace.g[j];
            workspace.g[j] = workspace.cs[j] * gj;
            workspace.g[j + 1] = -workspace.sn[j] * gj;

            total_iterations += 1;
            let basis_count = j + 1;
            let estimated_residual = workspace.g[j + 1].abs();

            if estimated_residual <= target || happy_breakdown {
                backsolve_and_update(workspace, x, basis_count)?;
                final_residual = exact_residual(a, b, x, workspace)?;
                cycle_updated = true;

                if final_residual <= target {
                    return Ok(KrylovOutcome {
                        status: SolveStatus::Converged,
                        iterations: total_iterations,
                        initial_residual,
                        final_residual,
                    });
                }

                if happy_breakdown {
                    return Ok(KrylovOutcome {
                        status: SolveStatus::Breakdown,
                        iterations: total_iterations,
                        initial_residual,
                        final_residual,
                    });
                }

                break;
            }
        }

        if !cycle_updated {
            let basis_count = cycle_limit;
            backsolve_and_update(workspace, x, basis_count)?;
            final_residual = exact_residual(a, b, x, workspace)?;

            if final_residual <= target {
                return Ok(KrylovOutcome {
                    status: SolveStatus::Converged,
                    iterations: total_iterations,
                    initial_residual,
                    final_residual,
                });
            }
        }

        if total_iterations < options.solver.max_iterations {
            let next_restart = controller(FgmresRestartProgress {
                restart: current_restart,
                cycle_iterations: total_iterations - cycle_start_iterations,
                total_iterations,
                initial_residual,
                cycle_initial_residual,
                final_residual,
                remaining_iterations: options.solver.max_iterations - total_iterations,
            })?;

            if next_restart == 0 {
                return Err(HybitError::InvalidArgument(
                    "FGMRES restart controller returned zero",
                ));
            }
            workspace.validate(n, next_restart)?;
            current_restart = next_restart;
        }
    }

    Ok(KrylovOutcome {
        status: SolveStatus::MaxIterations,
        iterations: total_iterations,
        initial_residual,
        final_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DenseOperator {
        n: usize,
        values: Vec<f64>,
    }

    impl DenseOperator {
        fn new(n: usize, values: Vec<f64>) -> Self {
            assert_eq!(values.len(), n * n);
            Self { n, values }
        }
    }

    impl LinearOperator for DenseOperator {
        fn rows(&self) -> usize {
            self.n
        }

        fn cols(&self) -> usize {
            self.n
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
            if x.len() != self.n {
                return Err(HybitError::DimensionMismatch {
                    expected: self.n,
                    actual: x.len(),
                });
            }
            if y.len() != self.n {
                return Err(HybitError::DimensionMismatch {
                    expected: self.n,
                    actual: y.len(),
                });
            }

            for (row, yi) in y.iter_mut().enumerate() {
                let start = row * self.n;
                *yi = self.values[start..start + self.n]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum();
            }
            Ok(())
        }
    }

    struct IdentityPreconditioner(usize);

    impl Preconditioner for IdentityPreconditioner {
        fn len(&self) -> usize {
            self.0
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), HybitError> {
            output.copy_from_slice(input);
            Ok(())
        }
    }

    struct AlternatingScalePreconditioner {
        n: usize,
        calls: Vec<usize>,
    }

    impl FlexiblePreconditioner for AlternatingScalePreconditioner {
        fn len(&self) -> usize {
            self.n
        }

        fn apply(
            &mut self,
            iteration: usize,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), HybitError> {
            self.calls.push(iteration);
            let scale = if iteration % 2 == 0 { 0.5 } else { 1.5 };
            for (out, &value) in output.iter_mut().zip(input) {
                *out = scale * value;
            }
            Ok(())
        }
    }

    fn nonsymmetric_operator() -> DenseOperator {
        DenseOperator::new(
            4,
            vec![
                4.0, 1.0, 0.0, 0.0, //
                -2.0, 3.0, 1.0, 0.0, //
                0.0, -1.0, 2.0, 1.0, //
                1.0, 0.0, 0.0, 2.0,
            ],
        )
    }

    fn rhs_for(a: &DenseOperator, x: &[f64]) -> Vec<f64> {
        let mut b = vec![0.0; a.rows()];
        a.apply(x, &mut b).unwrap();
        b
    }

    fn test_options() -> FgmresOptions {
        FgmresOptions {
            solver: SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 32,
            },
            restart: 4,
        }
    }

    #[test]
    fn fixed_preconditioner_blanket_impl_solves_nonsymmetric_system() {
        let a = nonsymmetric_operator();
        let exact = [1.0, -2.0, 0.5, 3.0];
        let b = rhs_for(&a, &exact);
        let mut x = vec![0.0; 4];
        let mut m = IdentityPreconditioner(4);

        let outcome = fgmres(&a, &mut m, &b, &mut x, test_options()).unwrap();

        assert_eq!(outcome.status, SolveStatus::Converged);
        assert!(outcome.iterations <= 4);
        for (&actual, &expected) in x.iter().zip(&exact) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }
    }

    #[test]
    fn flexible_preconditioner_can_change_each_iteration() {
        let a = nonsymmetric_operator();
        let exact = [-1.0, 0.25, 2.0, -0.5];
        let b = rhs_for(&a, &exact);
        let mut x = vec![0.0; 4];
        let mut m = AlternatingScalePreconditioner {
            n: 4,
            calls: Vec::new(),
        };

        let outcome = fgmres(&a, &mut m, &b, &mut x, test_options()).unwrap();

        assert_eq!(outcome.status, SolveStatus::Converged);
        assert_eq!(m.calls.len(), outcome.iterations);
        assert_eq!(m.calls, (0..outcome.iterations).collect::<Vec<usize>>());
        for (&actual, &expected) in x.iter().zip(&exact) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }
    }

    #[test]
    fn workspace_reuses_allocated_basis_across_rhs_vectors() {
        let a = nonsymmetric_operator();
        let mut m = IdentityPreconditioner(4);
        let options = test_options();
        let mut workspace = FgmresWorkspace::new(4, options.restart).unwrap();
        let bytes = workspace.bytes();

        let exact1 = [1.0, 2.0, -1.0, 0.5];
        let b1 = rhs_for(&a, &exact1);
        let mut x1 = vec![0.0; 4];
        let first =
            fgmres_with_workspace(&a, &mut m, &b1, &mut x1, options, &mut workspace).unwrap();
        assert_eq!(first.status, SolveStatus::Converged);

        let exact2 = [-0.5, 1.0, 3.0, -2.0];
        let b2 = rhs_for(&a, &exact2);
        let mut x2 = vec![0.0; 4];
        let second =
            fgmres_with_workspace(&a, &mut m, &b2, &mut x2, options, &mut workspace).unwrap();
        assert_eq!(second.status, SolveStatus::Converged);

        assert_eq!(workspace.bytes(), bytes);
        for (&actual, &expected) in x2.iter().zip(&exact2) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }
    }

    #[test]
    fn larger_workspace_capacity_serves_smaller_restart() {
        let a = nonsymmetric_operator();
        let exact = [1.0, -2.0, 0.5, 3.0];
        let b = rhs_for(&a, &exact);
        let mut m = IdentityPreconditioner(4);
        let mut workspace = FgmresWorkspace::new(4, 4).unwrap();
        let bytes = workspace.bytes();

        let options = FgmresOptions {
            solver: SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 32,
            },
            restart: 2,
        };

        let mut x = vec![0.0; 4];
        let outcome =
            fgmres_with_workspace(&a, &mut m, &b, &mut x, options, &mut workspace).unwrap();

        assert_eq!(outcome.status, SolveStatus::Converged);
        assert_eq!(workspace.restart(), 4);
        assert_eq!(workspace.bytes(), bytes);
        for (&actual, &expected) in x.iter().zip(&exact) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }

        let oversized = FgmresOptions {
            solver: options.solver,
            restart: 5,
        };
        let error =
            fgmres_with_workspace(&a, &mut m, &b, &mut x, oversized, &mut workspace).unwrap_err();
        assert!(matches!(
            error,
            HybitError::InvalidArgument(
                "FGMRES workspace restart capacity is smaller than options"
            )
        ));
    }
    #[test]
    fn restarted_fgmres_crosses_multiple_cycles() {
        let n = 12;
        let mut values = vec![0.0; n * n];

        for row in 0..n {
            values[row * n + row] = 4.0 + 0.03 * row as f64;

            if row > 0 {
                values[row * n + row - 1] = -1.2;
            }
            if row + 1 < n {
                values[row * n + row + 1] = 0.7;
            }
            if row + 2 < n {
                values[row * n + row + 2] = 0.15;
            }
        }

        let a = DenseOperator::new(n, values);

        let exact: Vec<f64> = (0..n)
            .map(|i| {
                let magnitude = 0.5 + 0.1 * i as f64;
                if i % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();

        let b = rhs_for(&a, &exact);
        let mut x = vec![0.0; n];
        let mut m = IdentityPreconditioner(n);

        let options = FgmresOptions {
            solver: SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 64,
            },
            restart: 3,
        };

        let outcome = fgmres(&a, &mut m, &b, &mut x, options).unwrap();

        assert_eq!(outcome.status, SolveStatus::Converged);
        assert!(
            outcome.iterations > options.restart,
            "test must exercise at least one real FGMRES restart"
        );

        for (&actual, &expected) in x.iter().zip(&exact) {
            assert!(
                (actual - expected).abs() <= 1.0e-10,
                "solution mismatch: actual={actual:e}, expected={expected:e}"
            );
        }
    }

    #[test]
    fn restart_controller_changes_cycle_dimension_without_resetting_global_iteration() {
        let a = nonsymmetric_operator();
        let exact = [1.0, -2.0, 0.5, 3.0];
        let b = rhs_for(&a, &exact);
        let mut x = vec![0.0; 4];
        let mut m = AlternatingScalePreconditioner {
            n: 4,
            calls: Vec::new(),
        };
        let mut workspace = FgmresWorkspace::new(4, 4).unwrap();
        let options = FgmresOptions {
            solver: SolverOptions {
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 0.0,
                max_iterations: 64,
            },
            restart: 1,
        };
        let mut boundaries = Vec::new();

        let outcome = fgmres_with_workspace_and_restart_controller(
            &a,
            &mut m,
            &b,
            &mut x,
            options,
            &mut workspace,
            |progress| {
                boundaries.push(progress);
                Ok((progress.restart * 2).min(4))
            },
        )
        .unwrap();

        assert_eq!(outcome.status, SolveStatus::Converged);
        assert!(!boundaries.is_empty());
        assert_eq!(boundaries[0].restart, 1);
        assert_eq!(boundaries[0].cycle_iterations, 1);
        assert_eq!(m.calls, (0..outcome.iterations).collect::<Vec<usize>>());
        assert_eq!(workspace.restart(), 4);

        for (&actual, &expected) in x.iter().zip(&exact) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }
    }
    #[test]
    fn zero_restart_is_rejected() {
        match FgmresWorkspace::new(4, 0) {
            Err(HybitError::InvalidArgument(message)) => {
                assert_eq!(message, "FGMRES restart must be > 0");
            }
            other => panic!("unexpected result: {other:?}"),
        }

        let options = FgmresOptions {
            solver: SolverOptions::default(),
            restart: 0,
        };
        assert!(matches!(
            options.validate(),
            Err(HybitError::InvalidArgument("FGMRES restart must be > 0"))
        ));
    }
}
