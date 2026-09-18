use hybit::{Csr32Matrix, HybitSolver, SolverOptions};

fn block_diagonal(easy_before: usize, hard: usize) -> Csr32Matrix {
    let n = easy_before + hard;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for row in 0..easy_before {
        col_idx.push(row as u32);
        values.push(1.0);
        row_ptr.push(col_idx.len() as u32);
    }

    let base = easy_before;
    for local in 0..hard {
        let row = base + local;
        if local > 0 {
            col_idx.push((row - 1) as u32);
            values.push(-1.0);
        }
        col_idx.push(row as u32);
        values.push(2.0);
        if local + 1 < hard {
            col_idx.push((row + 1) as u32);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len() as u32);
    }

    Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = block_diagonal(32, 64);
    let mut solver = HybitSolver::new();
    solver.set_options(SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 100,
    })?;

    let analysis = solver.analyze_csr32(&a)?;
    let mut prepared = solver.prepare_csr32(&a, &analysis)?;

    let b1 = vec![1.0; a.nrows()];
    let mut x1 = vec![0.0; a.nrows()];
    let first = prepared.solve(&a, &b1, &mut x1)?;

    let mut b2 = vec![1.0; a.nrows()];
    b2[0] = 2.0;
    let mut x2 = vec![0.0; a.nrows()];
    let second = prepared.solve(&a, &b2, &mut x2)?;

    println!("HyBIT {} prepared solve-many demo", env!("CARGO_PKG_VERSION"));
    println!("analysis          : {:.3} ms", analysis.analysis_seconds() * 1.0e3);
    println!("prepare           : {:.3} ms", prepared.prepare_seconds() * 1.0e3);
    println!("workspace         : {:.3} KiB", prepared.krylov_workspace_bytes() as f64 / 1024.0);
    println!("first solve       : {} iterations, reused={}, factors={} bytes",
        first.iterations, first.preconditioner_reused, first.local_factor_bytes);
    println!("second solve      : {} iterations, reused={}, probe={}, factor={:.3} ms",
        second.iterations,
        second.preconditioner_reused,
        second.probe_iterations,
        second.local_factor_seconds * 1.0e3);
    println!("solve sequence    : {} -> {}", first.solve_sequence, second.solve_sequence);

    Ok(())
}
