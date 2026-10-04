use hybit::{
    Csr32Matrix, ExecutionPolicy, ExecutionTarget, GeneralSquareOptions,
    GeneralSquarePreconditionerPolicy, GeneralSquareRestartPolicy, HybitError, HybitSolver,
    MatrixProblemClass, PreconditionerKind, SolverKind, SolverOptions,
};

fn nonsymmetric_general_matrix() -> Csr32Matrix {
    Csr32Matrix::new(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 0, 3],
        vec![-4.0, 1.0, 2.0, 3.0, 1.0, -1.0, -2.0, 1.0, 1.0, 2.0],
    )
    .unwrap()
}

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
fn symmetric_indefinite_remains_recognized_but_unrouted() {
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
}

#[test]
fn general_square_uses_prepared_fgmres_with_jacobi() {
    let a = nonsymmetric_general_matrix();
    let exact1 = vec![1.0, -2.0, 0.5, 3.0];
    let b1 = a.spmv(&exact1).unwrap();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 64,
        })
        .unwrap();
    solver
        .set_general_square_options(GeneralSquareOptions {
            restart: 2,
            ..GeneralSquareOptions::default()
        })
        .unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    assert_eq!(analysis.problem_class(), MatrixProblemClass::GeneralSquare);

    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();
    assert_eq!(prepared.problem_class(), MatrixProblemClass::GeneralSquare);
    assert!(prepared.krylov_workspace_bytes() > 0);

    let mut x1 = vec![0.0; 4];
    let first = prepared.solve(&a, &b1, &mut x1).unwrap();
    assert!(first.converged());
    assert_eq!(first.solver, SolverKind::Fgmres);
    assert_eq!(first.preconditioner, PreconditionerKind::Jacobi);
    assert!(!first.preconditioner_reused);
    assert_eq!(first.solve_sequence, 1);

    for (&actual, &expected) in x1.iter().zip(&exact1) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }

    let exact2 = vec![-0.5, 1.5, -2.0, 0.25];
    let b2 = a.spmv(&exact2).unwrap();
    let mut x2 = vec![0.0; 4];
    let second = prepared.solve(&a, &b2, &mut x2).unwrap();

    assert!(second.converged());
    assert_eq!(second.solver, SolverKind::Fgmres);
    assert!(second.preconditioner_reused);
    assert_eq!(second.solve_sequence, 2);
    assert_eq!(second.prepare_seconds, 0.0);

    for (&actual, &expected) in x2.iter().zip(&exact2) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
}

#[test]
fn general_square_escalating_restart_reuses_max_capacity_workspace() {
    let a = nonsymmetric_general_matrix();
    let exact = vec![1.0, -2.0, 0.5, 3.0];
    let b = a.spmv(&exact).unwrap();

    let mut fixed = HybitSolver::new();
    fixed.set_problem_class(MatrixProblemClass::GeneralSquare);
    fixed
        .set_general_square_options(GeneralSquareOptions {
            restart: 1,
            ..GeneralSquareOptions::default()
        })
        .unwrap();
    let fixed_analysis = fixed.analyze_csr32(&a).unwrap();
    let fixed_prepared = fixed.prepare_csr32(&a, &fixed_analysis).unwrap();
    let fixed_bytes = fixed_prepared.krylov_workspace_bytes();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 64,
        })
        .unwrap();
    solver
        .set_general_square_options(GeneralSquareOptions {
            restart: 1,
            restart_policy: GeneralSquareRestartPolicy::Escalating,
            max_restart: 4,
            escalation_stage_iterations: 1,
        })
        .unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();

    assert!(
        prepared.krylov_workspace_bytes() > fixed_bytes,
        "escalating prepared state must allocate the max restart capacity"
    );

    let mut x = vec![0.0; 4];
    let report = prepared.solve(&a, &b, &mut x).unwrap();

    assert!(report.converged());
    assert_eq!(report.solver, SolverKind::Fgmres);
    assert_eq!(report.preconditioner, PreconditionerKind::Jacobi);

    for (&actual, &expected) in x.iter().zip(&exact) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
}

#[test]
fn general_square_budget_aware_restart_reuses_max_capacity_workspace() {
    let a = nonsymmetric_general_matrix();
    let exact = vec![1.0, -2.0, 0.5, 3.0];
    let b = a.spmv(&exact).unwrap();

    let mut fixed = HybitSolver::new();
    fixed.set_problem_class(MatrixProblemClass::GeneralSquare);
    fixed
        .set_general_square_options(GeneralSquareOptions {
            restart: 1,
            ..GeneralSquareOptions::default()
        })
        .unwrap();
    let fixed_analysis = fixed.analyze_csr32(&a).unwrap();
    let fixed_prepared = fixed.prepare_csr32(&a, &fixed_analysis).unwrap();
    let fixed_bytes = fixed_prepared.krylov_workspace_bytes();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 64,
        })
        .unwrap();
    solver
        .set_general_square_options(GeneralSquareOptions {
            restart: 1,
            restart_policy: GeneralSquareRestartPolicy::BudgetAware,
            max_restart: 4,
            escalation_stage_iterations: 0,
        })
        .unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();

    assert!(
        prepared.krylov_workspace_bytes() > fixed_bytes,
        "budget-aware prepared state must allocate the max restart capacity"
    );

    let mut x = vec![0.0; 4];
    let report = prepared.solve(&a, &b, &mut x).unwrap();

    assert!(report.converged());
    assert_eq!(report.solver, SolverKind::Fgmres);
    assert_eq!(report.preconditioner, PreconditionerKind::Jacobi);

    for (&actual, &expected) in x.iter().zip(&exact) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
}

#[test]
fn general_square_budget_aware_options_validate_bounds() {
    let options = GeneralSquareOptions {
        restart: 10,
        restart_policy: GeneralSquareRestartPolicy::BudgetAware,
        max_restart: 5,
        escalation_stage_iterations: 0,
    };

    assert!(matches!(
        options.validate(),
        Err(HybitError::InvalidArgument(
            "GeneralSquare budget-aware max_restart must be >= restart"
        ))
    ));
}
#[test]
fn general_square_escalating_options_validate_bounds() {
    let mut options = GeneralSquareOptions {
        restart: 10,
        restart_policy: GeneralSquareRestartPolicy::Escalating,
        max_restart: 5,
        escalation_stage_iterations: 300,
    };
    assert!(matches!(
        options.validate(),
        Err(HybitError::InvalidArgument(
            "GeneralSquare escalating max_restart must be >= restart"
        ))
    ));

    options.max_restart = 20;
    options.escalation_stage_iterations = 0;
    assert!(matches!(
        options.validate(),
        Err(HybitError::InvalidArgument(
            "GeneralSquare escalation_stage_iterations must be > 0"
        ))
    ));
}
#[test]
fn general_square_ilu0_is_opt_in_and_reused() {
    let a = nonsymmetric_general_matrix();
    let exact1 = vec![1.0, -2.0, 0.5, 3.0];
    let b1 = a.spmv(&exact1).unwrap();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 64,
        })
        .unwrap();
    solver.set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0);
    solver
        .set_general_square_options(GeneralSquareOptions {
            restart: 3,
            ..GeneralSquareOptions::default()
        })
        .unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();
    assert!(prepared.general_square_preconditioner_bytes() > 0);
    assert_eq!(prepared.general_square_ilu_adjusted_pivots(), 0);

    let mut x1 = vec![0.0; 4];
    let first = prepared.solve(&a, &b1, &mut x1).unwrap();
    assert!(first.converged());
    assert_eq!(first.solver, SolverKind::Fgmres);
    assert_eq!(first.preconditioner, PreconditionerKind::Ilu0);
    assert!(!first.preconditioner_reused);

    let exact2 = vec![-0.5, 1.5, -2.0, 0.25];
    let b2 = a.spmv(&exact2).unwrap();
    let mut x2 = vec![0.0; 4];
    let second = prepared.solve(&a, &b2, &mut x2).unwrap();
    assert!(second.converged());
    assert_eq!(second.preconditioner, PreconditionerKind::Ilu0);
    assert!(second.preconditioner_reused);

    for (&actual, &expected) in x2.iter().zip(&exact2) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
}

#[test]
fn general_square_ilu0_stabilizes_factor_zero_pivot() {
    let a = Csr32Matrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let exact = vec![1.0, -0.5, 2.0];
    let b = a.spmv(&exact).unwrap();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-10,
            absolute_tolerance: 0.0,
            max_iterations: 32,
        })
        .unwrap();
    solver.set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0);
    solver
        .set_general_square_options(GeneralSquareOptions {
            restart: 3,
            ..GeneralSquareOptions::default()
        })
        .unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();
    assert_eq!(prepared.general_square_ilu_adjusted_pivots(), 1);

    let mut x = vec![0.0; 3];
    let report = prepared.solve(&a, &b, &mut x).unwrap();
    assert!(report.converged());
    assert_eq!(report.preconditioner, PreconditionerKind::Ilu0);
}
#[test]
fn general_square_rejects_resident_pcg_execution_policy() {
    let a = nonsymmetric_general_matrix();
    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver.set_execution_policy(ExecutionPolicy::CpuResident);

    let error = solver.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "GeneralSquare FGMRES currently supports Auto/Cpu execution only"
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
fn cpu_resident_rayon_requires_fixed_preconditioner_checkpoint() {
    let a = test_matrix();
    let mut solver = HybitSolver::new();
    solver.set_execution_policy(ExecutionPolicy::CpuResidentRayon);

    let error = solver.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "CpuResidentRayon validation currently requires HybridOptions.enabled = false"
        )
    ));
}

#[test]
fn cpu_resident_rayon_requires_csr32_backend() {
    let a = test_matrix();
    let mut solver = HybitSolver::new();
    solver.set_execution_policy(ExecutionPolicy::CpuResidentRayon);
    solver.set_backend_policy(hybit::BackendPolicy::Abtm);
    let mut hybrid = solver.hybrid_options();
    hybrid.enabled = false;
    solver.set_hybrid_options(hybrid).unwrap();

    let error = solver.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument("resident Rayon execution requires the CSR32 backend")
    ));
}

#[test]
fn cpu_resident_rayon_jacobi_requires_fixed_preconditioner_checkpoint() {
    let a = test_matrix();
    let mut solver = HybitSolver::new();
    solver.set_execution_policy(ExecutionPolicy::CpuResidentRayonJacobi);

    let error = solver.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidArgument(
            "CpuResidentRayonJacobi validation currently requires HybridOptions.enabled = false"
        )
    ));
}

#[test]
fn cpu_resident_rayon_jacobi_matches_rayon_serial_jacobi_path() {
    let a = test_matrix();
    let b = vec![1.0; 48];
    let options = SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 160,
    };

    let mut b2 = HybitSolver::new();
    b2.set_execution_policy(ExecutionPolicy::CpuResidentRayon);
    b2.set_options(options).unwrap();
    let mut b2_hybrid = b2.hybrid_options();
    b2_hybrid.enabled = false;
    b2.set_hybrid_options(b2_hybrid).unwrap();

    let mut x_b2 = vec![0.0; 48];
    let b2_report = b2.solve_csr32(&a, &b, &mut x_b2).unwrap();
    assert!(b2_report.converged());

    let mut b3 = HybitSolver::new();
    b3.set_execution_policy(ExecutionPolicy::CpuResidentRayonJacobi);
    b3.set_options(options).unwrap();
    let mut b3_hybrid = b3.hybrid_options();
    b3_hybrid.enabled = false;
    b3.set_hybrid_options(b3_hybrid).unwrap();

    let analysis = b3.analyze_csr32(&a).unwrap();
    assert_eq!(
        analysis.execution_policy(),
        ExecutionPolicy::CpuResidentRayonJacobi
    );
    assert_eq!(analysis.execution_target(), ExecutionTarget::Cpu);

    let mut prepared = b3.prepare_csr32(&a, &analysis).unwrap();
    let mut x_b3 = vec![0.0; 48];
    let b3_report = prepared.solve(&a, &b, &mut x_b3).unwrap();

    assert!(b3_report.converged());
    assert_eq!(b3_report.status, b2_report.status);
    assert_eq!(b3_report.iterations, b2_report.iterations);
    assert_eq!(b3_report.preconditioner, PreconditionerKind::Jacobi);
    assert!((b3_report.final_residual - b2_report.final_residual).abs() <= 1.0e-12);

    for (b3_value, b2_value) in x_b3.iter().zip(&x_b2) {
        assert!((b3_value - b2_value).abs() <= 1.0e-12);
    }
}

#[test]
fn cpu_resident_rayon_matches_serial_resident_path() {
    let a = test_matrix();
    let b = vec![1.0; 48];
    let options = SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 160,
    };

    let mut serial = HybitSolver::new();
    serial.set_execution_policy(ExecutionPolicy::CpuResident);
    serial.set_options(options).unwrap();
    let mut serial_hybrid = serial.hybrid_options();
    serial_hybrid.enabled = false;
    serial.set_hybrid_options(serial_hybrid).unwrap();

    let mut x_serial = vec![0.0; 48];
    let serial_report = serial.solve_csr32(&a, &b, &mut x_serial).unwrap();
    assert!(serial_report.converged());

    let mut rayon = HybitSolver::new();
    rayon.set_execution_policy(ExecutionPolicy::CpuResidentRayon);
    rayon.set_options(options).unwrap();
    let mut rayon_hybrid = rayon.hybrid_options();
    rayon_hybrid.enabled = false;
    rayon.set_hybrid_options(rayon_hybrid).unwrap();

    let analysis = rayon.analyze_csr32(&a).unwrap();
    assert_eq!(
        analysis.execution_policy(),
        ExecutionPolicy::CpuResidentRayon
    );
    assert_eq!(analysis.execution_target(), ExecutionTarget::Cpu);

    let mut prepared = rayon.prepare_csr32(&a, &analysis).unwrap();
    assert_eq!(
        prepared.execution_policy(),
        ExecutionPolicy::CpuResidentRayon
    );

    let mut x_rayon = vec![0.0; 48];
    let rayon_report = prepared.solve(&a, &b, &mut x_rayon).unwrap();

    assert!(rayon_report.converged());
    assert_eq!(rayon_report.status, serial_report.status);
    assert_eq!(rayon_report.iterations, serial_report.iterations);
    assert_eq!(rayon_report.preconditioner, PreconditionerKind::Jacobi);
    assert!((rayon_report.final_residual - serial_report.final_residual).abs() <= 1.0e-12);

    for (rayon_value, serial_value) in x_rayon.iter().zip(&x_serial) {
        assert!((rayon_value - serial_value).abs() <= 1.0e-12);
    }

    assert_eq!(prepared.krylov_workspace_bytes(), 12 * 48 * 8);
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

fn missing_diagonal_general_matrix() -> Csr32Matrix {
    // [[0, 1],
    //  [1, 2]]
    // The (0,0) structural diagonal entry is absent, but the matrix is
    // nonsingular (determinant -1), so unpreconditioned FGMRES is a valid
    // fallback validation case.
    Csr32Matrix::new(2, 2, vec![0, 1, 3], vec![1, 0, 1], vec![1.0, 1.0, 2.0]).unwrap()
}

#[test]
fn general_square_strict_ilu0_still_rejects_missing_diagonal() {
    let a = missing_diagonal_general_matrix();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver.set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0);

    let error = solver.analyze_csr32(&a).unwrap_err();
    assert!(matches!(
        error,
        HybitError::InvalidMatrix("GeneralSquare FGMRES path requires a complete diagonal")
    ));
}

#[test]
fn general_square_ilu0_fallback_uses_identity_for_missing_diagonal() {
    let a = missing_diagonal_general_matrix();
    let exact = vec![1.0, 2.0];
    let b = a.spmv(&exact).unwrap();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0Fallback);
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 16,
        })
        .unwrap();
    solver
        .set_general_square_options(GeneralSquareOptions {
            restart: 2,
            ..GeneralSquareOptions::default()
        })
        .unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_csr32(&a, &analysis).unwrap();

    assert_eq!(
        prepared.general_square_preconditioner_kind(),
        Some(PreconditionerKind::None)
    );
    assert!(prepared.general_square_ilu_fallback_used());
    assert_eq!(prepared.general_square_preconditioner_bytes(), 0);
    assert_eq!(prepared.general_square_ilu_adjusted_pivots(), 0);

    let mut x = vec![0.0; 2];
    let report = prepared.solve(&a, &b, &mut x).unwrap();
    assert!(report.converged());
    assert_eq!(report.solver, SolverKind::Fgmres);
    assert_eq!(report.preconditioner, PreconditionerKind::None);

    for (&actual, &expected) in x.iter().zip(&exact) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
}

#[test]
fn general_square_ilu0_fallback_keeps_ilu0_when_supported() {
    let a = nonsymmetric_general_matrix();

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver
        .set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0Fallback);

    let analysis = solver.analyze_csr32(&a).unwrap();
    let prepared = solver.prepare_csr32(&a, &analysis).unwrap();

    assert_eq!(
        prepared.general_square_preconditioner_kind(),
        Some(PreconditionerKind::Ilu0)
    );
    assert!(!prepared.general_square_ilu_fallback_used());
    assert!(prepared.general_square_preconditioner_bytes() > 0);
}
