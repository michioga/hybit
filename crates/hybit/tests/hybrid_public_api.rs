use hybit::{
    Csr32Matrix, ExecutionPolicy, ExecutionTarget, HybitError, HybitSolver, MatrixProblemClass,
    PreconditionerKind, SolverOptions,
};

fn test_matrix() -> Csr32Matrix {
    let easy = 16usize;
    let hard = 32usize;
    let n = easy + hard;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..easy {
        col_idx.push(i as u32);
        values.push(1.0);
        row_ptr.push(col_idx.len() as u32);
    }
    for local in 0..hard {
        let i = easy + local;
        if local > 0 {
            col_idx.push((i - 1) as u32);
            values.push(-1.0);
        }
        col_idx.push(i as u32);
        values.push(2.0);
        if local + 1 < hard {
            col_idx.push((i + 1) as u32);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len() as u32);
    }
    Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
}

#[test]
fn public_auto_solver_can_escalate() {
    let a = test_matrix();
    let b = vec![1.0; 48];
    let mut x = vec![0.0; 48];
    let mut solver = HybitSolver::new();
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-10,
            absolute_tolerance: 0.0,
            max_iterations: 80,
        })
        .unwrap();
    let report = solver.solve_csr32(&a, &b, &mut x).unwrap();
    assert!(report.converged());
    assert_eq!(report.preconditioner, PreconditionerKind::Hybrid);
    assert_eq!(report.escalations, 1);
    assert!(report.local_factor_bytes > 0);
    assert!(report.setup_seconds >= report.analysis_seconds);
    assert!(report.solve_seconds >= report.probe_seconds);
}

#[test]
fn public_prepared_context_reuses_local_factors() {
    let a = test_matrix();
    let mut solver = HybitSolver::new();
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-10,
            absolute_tolerance: 0.0,
            max_iterations: 80,
        })
        .unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();

    let b1 = vec![1.0; 48];
    let mut x1 = vec![0.0; 48];
    let first = prepared.solve(&a, &b1, &mut x1).unwrap();
    assert!(first.converged());
    assert!(prepared.has_cached_hybrid());

    let mut b2 = vec![1.0; 48];
    b2[0] = 2.0;
    let mut x2 = vec![0.0; 48];
    let second = prepared.solve(&a, &b2, &mut x2).unwrap();
    assert!(second.converged());
    assert!(second.preconditioner_reused);
    assert_eq!(second.probe_iterations, 0);
    assert_eq!(second.local_factor_seconds, 0.0);
    assert_eq!(second.solve_sequence, 2);
}
#[test]
fn default_execution_and_problem_policy_preserve_spd_cpu_path() {
    let a = test_matrix();
    let solver = HybitSolver::new();

    assert_eq!(solver.execution_policy(), ExecutionPolicy::Auto);
    assert_eq!(solver.problem_class(), MatrixProblemClass::Spd);

    let analysis = solver.analyze_csr32(&a).unwrap();
    assert_eq!(analysis.execution_target(), ExecutionTarget::Cpu);
    assert_eq!(analysis.problem_class(), MatrixProblemClass::Spd);

    let prepared = solver.prepare_csr32(&a, &analysis).unwrap();
    assert_eq!(prepared.execution_target(), ExecutionTarget::Cpu);
    assert_eq!(prepared.problem_class(), MatrixProblemClass::Spd);
}

#[test]
fn explicit_cpu_execution_preserves_current_solver_path() {
    let a = test_matrix();
    let b = vec![1.0; 48];

    let mut default_x = vec![0.0; 48];
    let default_report = HybitSolver::new()
        .solve_csr32(&a, &b, &mut default_x)
        .unwrap();

    let mut cpu_solver = HybitSolver::new();
    cpu_solver.set_execution_policy(ExecutionPolicy::Cpu);
    let mut cpu_x = vec![0.0; 48];
    let cpu_report = cpu_solver.solve_csr32(&a, &b, &mut cpu_x).unwrap();

    assert_eq!(cpu_report.status, default_report.status);
    assert_eq!(cpu_report.iterations, default_report.iterations);
    assert_eq!(cpu_report.preconditioner, default_report.preconditioner);
    for (cpu, default) in cpu_x.iter().zip(&default_x) {
        assert!((cpu - default).abs() <= 1.0e-14);
    }
}

#[test]
fn gpu_policy_is_explicitly_rejected_until_backend_exists() {
    let a = test_matrix();
    let mut solver = HybitSolver::new();
    solver.set_execution_policy(ExecutionPolicy::Gpu);

    let error = solver.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "GPU execution is recognized but not implemented in HyBIT 0.8-a3"
        )
    ));
}

#[test]
fn future_square_problem_classes_are_recognized_but_not_silently_routed_to_pcg() {
    let a = test_matrix();

    let mut symmetric_indefinite = HybitSolver::new();
    symmetric_indefinite.set_problem_class(MatrixProblemClass::SymmetricIndefinite);
    let error = symmetric_indefinite.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "symmetric-indefinite systems are recognized but MINRES is not implemented in HyBIT 0.8-a3"
        )
    ));

    let mut general = HybitSolver::new();
    general.set_problem_class(MatrixProblemClass::GeneralSquare);
    let error = general.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "general square systems are recognized but FGMRES/BiCGStab is not implemented in HyBIT 0.8-a3"
        )
    ));
}

#[test]
fn prepare_rejects_policy_changes_after_analysis() {
    let a = test_matrix();
    let mut solver = HybitSolver::new();
    let analysis = solver.analyze_csr32(&a).unwrap();

    solver.set_execution_policy(ExecutionPolicy::Cpu);
    // Auto and Cpu resolve to the same CPU target, so this is intentionally safe.
    solver.prepare_csr32(&a, &analysis).unwrap();

    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    let error = solver.prepare_csr32(&a, &analysis).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "solver execution/problem policy changed between analyze and prepare"
        )
    ));
}
#[test]
fn cpu_resident_requires_fixed_preconditioner_checkpoint() {
    let a = test_matrix();
    let mut solver = HybitSolver::new();
    solver.set_execution_policy(ExecutionPolicy::CpuResident);

    let error = solver.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "CpuResident validation currently requires HybridOptions.enabled = false"
        )
    ));
}

#[test]
fn cpu_resident_fixed_jacobi_matches_legacy_cpu_path() {
    let a = test_matrix();
    let b = vec![1.0; 48];
    let options = SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 160,
    };

    let mut legacy = HybitSolver::new();
    legacy.set_execution_policy(ExecutionPolicy::Cpu);
    legacy.set_options(options).unwrap();
    let mut legacy_hybrid = legacy.hybrid_options();
    legacy_hybrid.enabled = false;
    legacy.set_hybrid_options(legacy_hybrid).unwrap();

    let mut x_legacy = vec![0.0; 48];
    let legacy_report = legacy.solve_csr32(&a, &b, &mut x_legacy).unwrap();
    assert!(legacy_report.converged());

    let mut resident = HybitSolver::new();
    resident.set_execution_policy(ExecutionPolicy::CpuResident);
    resident.set_options(options).unwrap();
    let mut resident_hybrid = resident.hybrid_options();
    resident_hybrid.enabled = false;
    resident.set_hybrid_options(resident_hybrid).unwrap();

    let analysis = resident.analyze_csr32(&a).unwrap();
    assert_eq!(analysis.execution_policy(), ExecutionPolicy::CpuResident);
    assert_eq!(analysis.execution_target(), ExecutionTarget::Cpu);

    let mut prepared = resident.prepare_csr32(&a, &analysis).unwrap();
    assert_eq!(prepared.execution_policy(), ExecutionPolicy::CpuResident);

    let mut x_resident = vec![0.0; 48];
    let resident_report = prepared.solve(&a, &b, &mut x_resident).unwrap();

    assert!(resident_report.converged());
    assert_eq!(resident_report.status, legacy_report.status);
    assert_eq!(resident_report.iterations, legacy_report.iterations);
    assert_eq!(resident_report.preconditioner, PreconditionerKind::Jacobi);
    assert!((resident_report.final_residual - legacy_report.final_residual).abs() <= 1.0e-12);

    for (resident_value, legacy_value) in x_resident.iter().zip(&x_legacy) {
        assert!((resident_value - legacy_value).abs() <= 1.0e-12);
    }

    // 0.8-a3 deliberately keeps the established five-vector workspace beside
    // the seven-vector resident validation workspace. This overhead is removed
    // when the resident path becomes the sole prepared representation.
    assert_eq!(prepared.krylov_workspace_bytes(), 12 * 48 * 8);
}

#[test]
fn cpu_resident_prepared_workspace_reuses_vectors_across_rhs() {
    let a = test_matrix();

    let mut solver = HybitSolver::new();
    solver.set_execution_policy(ExecutionPolicy::CpuResident);
    let mut hybrid = solver.hybrid_options();
    hybrid.enabled = false;
    solver.set_hybrid_options(hybrid).unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();

    let b1 = vec![1.0; 48];
    let mut x1 = vec![0.0; 48];
    let first = prepared.solve(&a, &b1, &mut x1).unwrap();
    assert!(first.converged());

    let mut b2 = vec![1.0; 48];
    b2[7] = 2.0;
    let mut x2 = vec![0.0; 48];
    let second = prepared.solve(&a, &b2, &mut x2).unwrap();
    assert!(second.converged());
    assert_eq!(prepared.solve_count(), 2);
    assert!(second.preconditioner_reused);
    assert_eq!(second.analysis_seconds, 0.0);
    assert_eq!(second.prepare_seconds, 0.0);
}
