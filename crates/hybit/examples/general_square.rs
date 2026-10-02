use hybit::{
    Csr32Matrix, GeneralSquareOptions, GeneralSquarePreconditionerPolicy, HybitSolver,
    MatrixProblemClass, SolverOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Csr32Matrix::new(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 0, 3],
        vec![-4.0, 1.0, 2.0, 3.0, 1.0, -1.0, -2.0, 1.0, 1.0, 2.0],
    )?;
    let exact = vec![1.0, -2.0, 0.5, 3.0];
    let b = a.spmv(&exact)?;

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver.set_options(SolverOptions {
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 64,
    })?;
    solver.set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0);
    solver.set_general_square_options(GeneralSquareOptions {
        restart: 3,
        ..GeneralSquareOptions::default()
    })?;

    let analysis = solver.analyze_csr32(&a)?;
    let mut prepared = solver.prepare_csr32(&a, &analysis)?;
    println!(
        "prepared: preconditioner={:.3} KiB, krylov={:.3} KiB, adjusted_pivots={}",
        prepared.general_square_preconditioner_bytes() as f64 / 1024.0,
        prepared.krylov_workspace_bytes() as f64 / 1024.0,
        prepared.general_square_ilu_adjusted_pivots(),
    );

    let mut x = vec![0.0; a.nrows()];
    let report = prepared.solve(&a, &b, &mut x)?;
    println!(
        "status={:?} solver={:?} preconditioner={:?} iterations={} relative_residual={:.3e}",
        report.status,
        report.solver,
        report.preconditioner,
        report.iterations,
        report.relative_residual
    );
    println!("x={x:?}");
    Ok(())
}
