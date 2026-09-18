use hybit::{Csr32Matrix, HybitSolver, PreconditionerKind, SolverOptions};

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
        if local > 0 { col_idx.push((i - 1) as u32); values.push(-1.0); }
        col_idx.push(i as u32); values.push(2.0);
        if local + 1 < hard { col_idx.push((i + 1) as u32); values.push(-1.0); }
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
    solver.set_options(SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 80,
    }).unwrap();
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
    solver.set_options(SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 80,
    }).unwrap();

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
