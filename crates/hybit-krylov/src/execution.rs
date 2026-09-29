use crate::KrylovOutcome;
use hybit_core::{
    ExecutionTarget, HybitError, LinearOperator, Preconditioner, SolveStatus, SolverOptions,
};
use rayon::prelude::*;

/// Execution boundary for Krylov methods whose vectors may live outside host memory.
///
/// The associated `Vector` type is intentionally opaque. The CPU reference
/// backend uses `Vec<f64>`; a CUDA or portable GPU backend can instead use a
/// device-resident buffer while keeping the Krylov recurrence independent of
/// the storage location.
///
/// The primitive vector operations are broad enough to support the current PCG
/// path and form the base for future MINRES/FGMRES work. Backends may override
/// the fused update helpers when a device kernel can reduce memory traffic.
pub trait KrylovExecutionBackend {
    type Vector;

    fn target(&self) -> ExecutionTarget;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn preconditioner_len(&self) -> usize;

    fn allocate_vector(&mut self, len: usize) -> Result<Self::Vector, HybitError>;
    fn upload(&mut self, host: &[f64], vector: &mut Self::Vector) -> Result<(), HybitError>;
    fn download(&mut self, vector: &Self::Vector, host: &mut [f64]) -> Result<(), HybitError>;
    fn copy(&mut self, src: &Self::Vector, dst: &mut Self::Vector) -> Result<(), HybitError>;

    fn apply_operator(&mut self, x: &Self::Vector, y: &mut Self::Vector) -> Result<(), HybitError>;

    fn apply_preconditioner(
        &mut self,
        r: &Self::Vector,
        z: &mut Self::Vector,
    ) -> Result<(), HybitError>;

    fn dot(&mut self, a: &Self::Vector, b: &Self::Vector) -> Result<f64, HybitError>;

    fn axpy(
        &mut self,
        alpha: f64,
        x: &Self::Vector,
        y: &mut Self::Vector,
    ) -> Result<(), HybitError>;

    fn scale(&mut self, alpha: f64, x: &mut Self::Vector) -> Result<(), HybitError>;

    fn l2_norm(&mut self, x: &Self::Vector) -> Result<f64, HybitError> {
        let sumsq = self.dot(x, x)?;
        if !sumsq.is_finite() || sumsq < 0.0 {
            return Err(HybitError::NumericalBreakdown(
                "non-finite or negative squared norm in Krylov backend",
            ));
        }
        Ok(sumsq.sqrt())
    }

    /// PCG fused update hook. GPU backends can override this with one kernel.
    fn update_x_r_and_norm(
        &mut self,
        solution: &mut Self::Vector,
        residual: &mut Self::Vector,
        direction: &Self::Vector,
        operator_direction: &Self::Vector,
        alpha: f64,
    ) -> Result<f64, HybitError> {
        self.axpy(alpha, direction, solution)?;
        self.axpy(-alpha, operator_direction, residual)?;
        self.l2_norm(residual)
    }

    /// PCG search-direction hook. GPU backends can override this with one kernel.
    fn update_search_direction(
        &mut self,
        direction: &mut Self::Vector,
        preconditioned_residual: &Self::Vector,
        beta: f64,
    ) -> Result<(), HybitError> {
        self.scale(beta, direction)?;
        self.axpy(1.0, preconditioned_residual, direction)
    }
}

/// Resident PCG state.
///
/// Unlike the legacy host `PcgWorkspace`, this workspace also owns resident
/// solution and RHS buffers. A GPU implementation can therefore upload `b` and
/// the initial `x` once per solve, keep all seven vectors on the device for the
/// full recurrence, and download only the final solution.
#[derive(Clone, Debug)]
pub struct ResidentPcgWorkspace<V> {
    solution: V,
    rhs: V,
    ax: V,
    r: V,
    z: V,
    p: V,
    ap: V,
}

impl<V> ResidentPcgWorkspace<V> {
    pub fn allocate_with<B>(backend: &mut B, n: usize) -> Result<Self, HybitError>
    where
        B: KrylovExecutionBackend<Vector = V>,
    {
        Ok(Self {
            solution: backend.allocate_vector(n)?,
            rhs: backend.allocate_vector(n)?,
            ax: backend.allocate_vector(n)?,
            r: backend.allocate_vector(n)?,
            z: backend.allocate_vector(n)?,
            p: backend.allocate_vector(n)?,
            ap: backend.allocate_vector(n)?,
        })
    }
}

impl ResidentPcgWorkspace<Vec<f64>> {
    /// Persistent bytes owned by the seven host-resident f64 vectors.
    pub fn bytes(&self) -> usize {
        7 * self.r.len() * std::mem::size_of::<f64>()
    }
}

/// CPU reference implementation of the resident execution boundary.
///
/// It deliberately delegates operator and preconditioner application to the
/// existing traits, so this path can be cross-checked against the established
/// PCG implementation before any GPU backend is introduced.
pub struct CpuKrylovExecution<'a> {
    operator: &'a dyn LinearOperator,
    preconditioner: &'a dyn Preconditioner,
}

impl<'a> CpuKrylovExecution<'a> {
    pub fn new(operator: &'a dyn LinearOperator, preconditioner: &'a dyn Preconditioner) -> Self {
        Self {
            operator,
            preconditioner,
        }
    }
}

fn require_len(expected: usize, actual: usize) -> Result<(), HybitError> {
    if expected != actual {
        return Err(HybitError::DimensionMismatch { expected, actual });
    }
    Ok(())
}

impl KrylovExecutionBackend for CpuKrylovExecution<'_> {
    type Vector = Vec<f64>;

    fn target(&self) -> ExecutionTarget {
        ExecutionTarget::Cpu
    }

    fn rows(&self) -> usize {
        self.operator.rows()
    }

    fn cols(&self) -> usize {
        self.operator.cols()
    }

    fn preconditioner_len(&self) -> usize {
        self.preconditioner.len()
    }

    fn allocate_vector(&mut self, len: usize) -> Result<Self::Vector, HybitError> {
        Ok(vec![0.0; len])
    }

    fn upload(&mut self, host: &[f64], vector: &mut Self::Vector) -> Result<(), HybitError> {
        require_len(vector.len(), host.len())?;
        vector.copy_from_slice(host);
        Ok(())
    }

    fn download(&mut self, vector: &Self::Vector, host: &mut [f64]) -> Result<(), HybitError> {
        require_len(vector.len(), host.len())?;
        host.copy_from_slice(vector);
        Ok(())
    }

    fn copy(&mut self, src: &Self::Vector, dst: &mut Self::Vector) -> Result<(), HybitError> {
        require_len(src.len(), dst.len())?;
        dst.copy_from_slice(src);
        Ok(())
    }

    fn apply_operator(&mut self, x: &Self::Vector, y: &mut Self::Vector) -> Result<(), HybitError> {
        require_len(self.cols(), x.len())?;
        require_len(self.rows(), y.len())?;
        self.operator.apply(x, y)
    }

    fn apply_preconditioner(
        &mut self,
        r: &Self::Vector,
        z: &mut Self::Vector,
    ) -> Result<(), HybitError> {
        require_len(self.preconditioner_len(), r.len())?;
        require_len(self.preconditioner_len(), z.len())?;
        self.preconditioner.apply(r, z)
    }

    fn dot(&mut self, a: &Self::Vector, b: &Self::Vector) -> Result<f64, HybitError> {
        require_len(a.len(), b.len())?;
        Ok(hybit_core::dot(a, b))
    }

    fn axpy(
        &mut self,
        alpha: f64,
        x: &Self::Vector,
        y: &mut Self::Vector,
    ) -> Result<(), HybitError> {
        require_len(x.len(), y.len())?;
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi += alpha * *xi;
        }
        Ok(())
    }

    fn scale(&mut self, alpha: f64, x: &mut Self::Vector) -> Result<(), HybitError> {
        for value in x {
            *value *= alpha;
        }
        Ok(())
    }

    fn update_x_r_and_norm(
        &mut self,
        solution: &mut Self::Vector,
        residual: &mut Self::Vector,
        direction: &Self::Vector,
        operator_direction: &Self::Vector,
        alpha: f64,
    ) -> Result<f64, HybitError> {
        let n = solution.len();
        require_len(n, residual.len())?;
        require_len(n, direction.len())?;
        require_len(n, operator_direction.len())?;

        for i in 0..n {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * operator_direction[i];
        }
        self.l2_norm(residual)
    }

    fn update_search_direction(
        &mut self,
        direction: &mut Self::Vector,
        preconditioned_residual: &Self::Vector,
        beta: f64,
    ) -> Result<(), HybitError> {
        require_len(direction.len(), preconditioned_residual.len())?;
        for i in 0..direction.len() {
            direction[i] = preconditioned_residual[i] + beta * direction[i];
        }
        Ok(())
    }
}

/// Minimum vector length for recommending the Rayon resident-vector backend.
///
/// This deliberately matches the validated Structural Auto PCG-vector crossover
/// used by the current CPU FEM path. Operator and preconditioner parallelism
/// remain independent decisions.
pub const RESIDENT_RAYON_MIN_N: usize = 131_072;

/// Minimum shared Rayon worker count for recommending resident vector kernels.
pub const RESIDENT_RAYON_MIN_THREADS: usize = 4;

#[inline]
fn resident_rayon_recommended_for(n: usize, workers: usize) -> bool {
    n >= RESIDENT_RAYON_MIN_N && workers >= RESIDENT_RAYON_MIN_THREADS
}

/// Returns whether large resident dense-vector operations should prefer Rayon
/// on the current shared Rayon pool.
///
/// This is a recommendation for vector kernels only. Sparse operator and
/// preconditioner implementations remain independently selectable.
pub fn resident_rayon_recommended(n: usize) -> bool {
    resident_rayon_recommended_for(n, rayon::current_num_threads())
}

/// Rayon implementation of the resident Krylov execution boundary.
///
/// Matrix/operator and preconditioner application deliberately remain delegated
/// to their existing traits. This backend parallelizes only resident dense
/// vector work, allowing it to compose with either serial or parallel sparse
/// operators/preconditioners without coupling those policies.
pub struct RayonKrylovExecution<'a> {
    operator: &'a dyn LinearOperator,
    preconditioner: &'a dyn Preconditioner,
}

impl<'a> RayonKrylovExecution<'a> {
    pub fn new(operator: &'a dyn LinearOperator, preconditioner: &'a dyn Preconditioner) -> Self {
        Self {
            operator,
            preconditioner,
        }
    }
}

impl KrylovExecutionBackend for RayonKrylovExecution<'_> {
    type Vector = Vec<f64>;

    fn target(&self) -> ExecutionTarget {
        ExecutionTarget::Cpu
    }

    fn rows(&self) -> usize {
        self.operator.rows()
    }

    fn cols(&self) -> usize {
        self.operator.cols()
    }

    fn preconditioner_len(&self) -> usize {
        self.preconditioner.len()
    }

    fn allocate_vector(&mut self, len: usize) -> Result<Self::Vector, HybitError> {
        Ok(vec![0.0; len])
    }

    fn upload(&mut self, host: &[f64], vector: &mut Self::Vector) -> Result<(), HybitError> {
        require_len(vector.len(), host.len())?;
        vector.copy_from_slice(host);
        Ok(())
    }

    fn download(&mut self, vector: &Self::Vector, host: &mut [f64]) -> Result<(), HybitError> {
        require_len(vector.len(), host.len())?;
        host.copy_from_slice(vector);
        Ok(())
    }

    fn copy(&mut self, src: &Self::Vector, dst: &mut Self::Vector) -> Result<(), HybitError> {
        require_len(src.len(), dst.len())?;
        dst.copy_from_slice(src);
        Ok(())
    }

    fn apply_operator(&mut self, x: &Self::Vector, y: &mut Self::Vector) -> Result<(), HybitError> {
        require_len(self.cols(), x.len())?;
        require_len(self.rows(), y.len())?;
        self.operator.apply(x, y)
    }

    fn apply_preconditioner(
        &mut self,
        r: &Self::Vector,
        z: &mut Self::Vector,
    ) -> Result<(), HybitError> {
        require_len(self.preconditioner_len(), r.len())?;
        require_len(self.preconditioner_len(), z.len())?;
        self.preconditioner.apply(r, z)
    }

    fn dot(&mut self, a: &Self::Vector, b: &Self::Vector) -> Result<f64, HybitError> {
        require_len(a.len(), b.len())?;
        Ok(crate::parallel_dot(a, b))
    }

    fn axpy(
        &mut self,
        alpha: f64,
        x: &Self::Vector,
        y: &mut Self::Vector,
    ) -> Result<(), HybitError> {
        require_len(x.len(), y.len())?;
        y.par_chunks_mut(crate::PARALLEL_PCG_VECTOR_CHUNK)
            .zip(x.par_chunks(crate::PARALLEL_PCG_VECTOR_CHUNK))
            .for_each(|(yy, xx)| {
                for (yi, xi) in yy.iter_mut().zip(xx) {
                    *yi += alpha * *xi;
                }
            });
        Ok(())
    }

    fn scale(&mut self, alpha: f64, x: &mut Self::Vector) -> Result<(), HybitError> {
        x.par_chunks_mut(crate::PARALLEL_PCG_VECTOR_CHUNK)
            .for_each(|chunk| {
                for value in chunk {
                    *value *= alpha;
                }
            });
        Ok(())
    }

    fn update_x_r_and_norm(
        &mut self,
        solution: &mut Self::Vector,
        residual: &mut Self::Vector,
        direction: &Self::Vector,
        operator_direction: &Self::Vector,
        alpha: f64,
    ) -> Result<f64, HybitError> {
        let n = solution.len();
        require_len(n, residual.len())?;
        require_len(n, direction.len())?;
        require_len(n, operator_direction.len())?;
        Ok(crate::parallel_update_x_r_and_norm(
            solution,
            residual,
            direction,
            operator_direction,
            alpha,
        ))
    }

    fn update_search_direction(
        &mut self,
        direction: &mut Self::Vector,
        preconditioned_residual: &Self::Vector,
        beta: f64,
    ) -> Result<(), HybitError> {
        require_len(direction.len(), preconditioned_residual.len())?;
        crate::parallel_update_p(direction, preconditioned_residual, beta);
        Ok(())
    }
}
/// PCG recurrence over an execution backend with resident vectors.
///
/// Existing `pcg_with_workspace` remains the production CPU path in 0.8-a1.
/// This parallel implementation exists to validate the execution boundary
/// before `HybitSolver` is wired to it.
pub fn pcg_with_execution<B>(
    backend: &mut B,
    b: &[f64],
    x: &mut [f64],
    options: SolverOptions,
    workspace: &mut ResidentPcgWorkspace<B::Vector>,
) -> Result<KrylovOutcome, HybitError>
where
    B: KrylovExecutionBackend,
{
    options.validate()?;

    if backend.rows() != backend.cols() {
        return Err(HybitError::InvalidMatrix("PCG requires a square operator"));
    }

    let n = backend.rows();
    require_len(n, b.len())?;
    require_len(n, x.len())?;
    require_len(n, backend.preconditioner_len())?;

    backend.upload(x, &mut workspace.solution)?;
    backend.upload(b, &mut workspace.rhs)?;

    backend.apply_operator(&workspace.solution, &mut workspace.ax)?;
    backend.copy(&workspace.rhs, &mut workspace.r)?;
    backend.axpy(-1.0, &workspace.ax, &mut workspace.r)?;

    let initial_residual = backend.l2_norm(&workspace.r)?;
    let b_norm = backend.l2_norm(&workspace.rhs)?;
    let target = options
        .absolute_tolerance
        .max(options.relative_tolerance * b_norm.max(f64::MIN_POSITIVE));

    if initial_residual <= target {
        backend.download(&workspace.solution, x)?;
        return Ok(KrylovOutcome {
            status: SolveStatus::Converged,
            iterations: 0,
            initial_residual,
            final_residual: initial_residual,
        });
    }

    backend.apply_preconditioner(&workspace.r, &mut workspace.z)?;
    backend.copy(&workspace.z, &mut workspace.p)?;

    let mut rz_old = backend.dot(&workspace.r, &workspace.z)?;
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return Err(HybitError::NumericalBreakdown(
            "non-positive r^T M^-1 r; PCG assumptions may be violated",
        ));
    }

    let mut final_residual = initial_residual;

    for iter in 1..=options.max_iterations {
        backend.apply_operator(&workspace.p, &mut workspace.ap)?;
        let denom = backend.dot(&workspace.p, &workspace.ap)?;

        if !denom.is_finite() || denom <= 0.0 {
            backend.download(&workspace.solution, x)?;
            return Ok(KrylovOutcome {
                status: SolveStatus::Breakdown,
                iterations: iter - 1,
                initial_residual,
                final_residual,
            });
        }

        let alpha = rz_old / denom;
        final_residual = backend.update_x_r_and_norm(
            &mut workspace.solution,
            &mut workspace.r,
            &workspace.p,
            &workspace.ap,
            alpha,
        )?;

        if final_residual <= target {
            backend.download(&workspace.solution, x)?;
            return Ok(KrylovOutcome {
                status: SolveStatus::Converged,
                iterations: iter,
                initial_residual,
                final_residual,
            });
        }

        backend.apply_preconditioner(&workspace.r, &mut workspace.z)?;
        let rz_new = backend.dot(&workspace.r, &workspace.z)?;

        if !rz_new.is_finite() || rz_new <= 0.0 {
            backend.download(&workspace.solution, x)?;
            return Ok(KrylovOutcome {
                status: SolveStatus::Breakdown,
                iterations: iter,
                initial_residual,
                final_residual,
            });
        }

        let beta = rz_new / rz_old;
        backend.update_search_direction(&mut workspace.p, &workspace.z, beta)?;
        rz_old = rz_new;
    }

    backend.download(&workspace.solution, x)?;

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

    struct Diagonal(Vec<f64>);

    impl LinearOperator for Diagonal {
        fn rows(&self) -> usize {
            self.0.len()
        }

        fn cols(&self) -> usize {
            self.0.len()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), HybitError> {
            for ((yi, &di), &xi) in y.iter_mut().zip(&self.0).zip(x) {
                *yi = di * xi;
            }
            Ok(())
        }
    }

    struct IdentityPreconditioner(usize);

    impl Preconditioner for IdentityPreconditioner {
        fn len(&self) -> usize {
            self.0
        }

        fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), HybitError> {
            z.copy_from_slice(r);
            Ok(())
        }
    }

    #[test]
    fn resident_cpu_pcg_matches_legacy_workspace_path() {
        let a = Diagonal(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]);
        let m = IdentityPreconditioner(7);
        let rhs = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0];
        let options = SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 32,
        };

        let mut legacy_x = vec![0.0; rhs.len()];
        let mut legacy_workspace = crate::PcgWorkspace::new(rhs.len());
        let legacy =
            crate::pcg_with_workspace(&a, &m, &rhs, &mut legacy_x, options, &mut legacy_workspace)
                .unwrap();

        let mut resident_x = vec![0.0; rhs.len()];
        let mut backend = CpuKrylovExecution::new(&a, &m);
        assert_eq!(backend.target(), ExecutionTarget::Cpu);

        let mut resident_workspace =
            ResidentPcgWorkspace::allocate_with(&mut backend, rhs.len()).unwrap();
        let resident = pcg_with_execution(
            &mut backend,
            &rhs,
            &mut resident_x,
            options,
            &mut resident_workspace,
        )
        .unwrap();

        assert_eq!(resident.status, legacy.status);
        assert_eq!(resident.iterations, legacy.iterations);
        assert!((resident.initial_residual - legacy.initial_residual).abs() <= 1.0e-14);
        assert!((resident.final_residual - legacy.final_residual).abs() <= 1.0e-14);

        for (actual, expected) in resident_x.iter().zip(&legacy_x) {
            assert!((actual - expected).abs() <= 1.0e-14);
        }
    }

    #[test]
    fn resident_rayon_pcg_matches_serial_execution_on_large_vector() {
        let n = RESIDENT_RAYON_MIN_N;
        let a = IdentityOperator(n);
        let m = IdentityPreconditioner(n);
        let rhs: Vec<f64> = (0..n).map(|i| 0.5 + (i % 31) as f64 * 0.03125).collect();
        let options = SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 8,
        };

        let mut serial_x = vec![0.0; n];
        let mut serial_backend = CpuKrylovExecution::new(&a, &m);
        let mut serial_workspace =
            ResidentPcgWorkspace::allocate_with(&mut serial_backend, n).unwrap();
        let serial = pcg_with_execution(
            &mut serial_backend,
            &rhs,
            &mut serial_x,
            options,
            &mut serial_workspace,
        )
        .unwrap();

        let mut rayon_x = vec![0.0; n];
        let mut rayon_backend = RayonKrylovExecution::new(&a, &m);
        let mut rayon_workspace =
            ResidentPcgWorkspace::allocate_with(&mut rayon_backend, n).unwrap();
        let parallel = pcg_with_execution(
            &mut rayon_backend,
            &rhs,
            &mut rayon_x,
            options,
            &mut rayon_workspace,
        )
        .unwrap();

        assert_eq!(serial.status, SolveStatus::Converged);
        assert_eq!(parallel.status, SolveStatus::Converged);
        assert_eq!(serial.iterations, 1);
        assert_eq!(parallel.iterations, 1);
        assert!((serial.initial_residual - parallel.initial_residual).abs() <= 1.0e-9);
        assert!((serial.final_residual - parallel.final_residual).abs() <= 1.0e-12);

        for (serial_value, parallel_value) in serial_x.iter().zip(&rayon_x) {
            assert!((serial_value - parallel_value).abs() <= 1.0e-14);
        }
    }

    #[test]
    fn resident_rayon_recommendation_uses_validated_cpu_thresholds() {
        assert!(!resident_rayon_recommended_for(
            RESIDENT_RAYON_MIN_N - 1,
            RESIDENT_RAYON_MIN_THREADS
        ));
        assert!(!resident_rayon_recommended_for(
            RESIDENT_RAYON_MIN_N,
            RESIDENT_RAYON_MIN_THREADS - 1
        ));
        assert!(resident_rayon_recommended_for(
            RESIDENT_RAYON_MIN_N,
            RESIDENT_RAYON_MIN_THREADS
        ));
    }

    struct IdentityOperator(usize);

    impl LinearOperator for IdentityOperator {
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
}
