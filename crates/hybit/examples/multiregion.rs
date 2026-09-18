use hybit::{pcg, Csr32Matrix, HybitSolver, JacobiPreconditioner, SolverOptions};

fn two_hard_blocks() -> Result<Csr32Matrix, Box<dyn std::error::Error>> {
    let easy_before = 16usize;
    let hard = 48usize;
    let spacer = 8usize;
    let n = easy_before + hard + spacer + hard;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    let mut row = 0usize;

    for _ in 0..easy_before {
        col_idx.push(row as u32); values.push(1.0); row += 1; row_ptr.push(col_idx.len() as u32);
    }
    for block in 0..2 {
        let base = row;
        for local in 0..hard {
            let i = base + local;
            if local > 0 { col_idx.push((i - 1) as u32); values.push(-1.0); }
            col_idx.push(i as u32); values.push(2.0);
            if local + 1 < hard { col_idx.push((i + 1) as u32); values.push(-1.0); }
            row += 1;
            row_ptr.push(col_idx.len() as u32);
        }
        if block == 0 {
            for _ in 0..spacer {
                col_idx.push(row as u32); values.push(1.0); row += 1; row_ptr.push(col_idx.len() as u32);
            }
        }
    }
    Ok(Csr32Matrix::new(n, n, row_ptr, col_idx, values)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = two_hard_blocks()?;
    let b = vec![1.0; a.nrows()];
    let opts = SolverOptions { relative_tolerance: 1.0e-10, absolute_tolerance: 0.0, max_iterations: 100 };

    let jacobi = JacobiPreconditioner::from_csr32(&a)?;
    let mut xp = vec![0.0; a.nrows()];
    let plain = pcg(&a, &jacobi, &b, &mut xp, opts)?;

    let mut solver = HybitSolver::new();
    solver.set_options(opts)?;
    let mut x = vec![0.0; a.nrows()];
    let r = solver.solve_csr32(&a, &b, &mut x)?;

    println!("HyBIT {} multi-region demo", env!("CARGO_PKG_VERSION"));
    println!("plain iterations  : {}", plain.iterations);
    println!("HyBIT iterations  : {}", r.iterations);
    println!("hard core DOFs    : {}", r.hard_dofs);
    println!("local regions     : {}", r.local_direct_regions);
    println!("local factor DOFs : {} total / {} unique", r.local_factor_dofs, r.unique_local_factor_dofs);
    println!("largest region    : {}", r.largest_local_region);
    println!("factor memory     : {:.3} KiB", r.local_factor_bytes as f64 / 1024.0);
    println!("total setup/solve : {:.3} / {:.3} ms", r.setup_seconds * 1.0e3, r.solve_seconds * 1.0e3);
    Ok(())
}
