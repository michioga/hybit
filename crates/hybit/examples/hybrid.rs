use hybit::{
    pcg, Csr32Matrix, HybitSolver, JacobiPreconditioner, PreconditionerKind,
    SolveStatus, SolverOptions,
};

fn easy_plus_hard_block(easy: usize, hard: usize) -> Result<Csr32Matrix, Box<dyn std::error::Error>> {
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
    Ok(Csr32Matrix::new(n, n, row_ptr, col_idx, values)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = easy_plus_hard_block(32, 64)?;
    let b = vec![1.0; 96];
    let options = SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 100,
    };

    let jacobi = JacobiPreconditioner::from_csr32(&matrix)?;
    let mut x_plain = vec![0.0; 96];
    let plain = pcg(&matrix, &jacobi, &b, &mut x_plain, options)?;

    let mut solver = HybitSolver::new();
    solver.set_options(options)?;
    let mut x_hybrid = vec![0.0; 96];
    let report = solver.solve_csr32(&matrix, &b, &mut x_hybrid)?;

    println!("HyBIT {} selective-direct / Schwarz demo", env!("CARGO_PKG_VERSION"));
    println!("plain Jacobi-PCG : status={:?}, iterations={}, residual={:.3e}", plain.status, plain.iterations, plain.final_residual);
    println!("HyBIT auto       : status={:?}, iterations={}, residual={:.3e}", report.status, report.iterations, report.final_residual);
    println!("  escalations    : {}", report.escalations);
    println!("  probe           : {} iterations -> {:.3e}", report.probe_iterations, report.probe_final_residual);
    println!("  hard core DOFs  : {}", report.hard_dofs);
    println!("  local regions   : {}", report.local_direct_regions);
    println!("  factor DOFs     : {} total / {} unique", report.local_factor_dofs, report.unique_local_factor_dofs);
    println!("  largest region  : {}", report.largest_local_region);
    println!("  overlap layers  : {}", report.overlap_layers);
    println!("  factor memory   : {:.3} KiB", report.local_factor_bytes as f64 / 1024.0);
    println!("  timings [ms]    : analysis={:.3}, probe={:.3}, diagnostics={:.3}, factor={:.3}, restart={:.3}",
        report.analysis_seconds * 1.0e3,
        report.probe_seconds * 1.0e3,
        report.diagnostics_seconds * 1.0e3,
        report.local_factor_seconds * 1.0e3,
        report.restart_seconds * 1.0e3,
    );
    println!("  preconditioner  : {:?}", report.preconditioner);

    if plain.status != SolveStatus::Converged || !report.converged() {
        return Err("solver did not converge in the demo".into());
    }
    if report.preconditioner != PreconditionerKind::Hybrid || report.iterations >= plain.iterations {
        return Err("selective-direct escalation did not improve the demo problem".into());
    }
    Ok(())
}
