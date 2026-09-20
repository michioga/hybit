use hybit::{solve, Csr32Matrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Csr32Matrix::new(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
    )?;
    let b = vec![15.0, 10.0, 10.0, 10.0];
    let (x, report) = solve(&a, &b)?;
    println!("HyBIT {}", env!("CARGO_PKG_VERSION"));
    println!("x = {x:?}");
    println!(
        "status = {:?}, iterations = {}, residual = {:.3e}",
        report.status, report.iterations, report.final_residual
    );
    println!(
        "solver = {:?}, preconditioner = {:?}, escalations = {}",
        report.solver, report.preconditioner, report.escalations
    );
    if report.escalations > 0 {
        println!(
            "hard DOFs = {}, local regions = {}, largest region = {}",
            report.hard_dofs, report.local_direct_regions, report.largest_local_region
        );
    }
    Ok(())
}
