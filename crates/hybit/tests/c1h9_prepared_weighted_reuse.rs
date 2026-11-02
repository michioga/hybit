//! C1-H9: opt-in weighted-ABTM prepared solve-many must re-diagnose every RHS.
//! No factor is applied without a matching current RHS candidate.
use hybit::{
    BackendPolicy, Csr32Matrix, HybitSolver, PreconditionerKind, SolveStatus, SolverOptions,
};
use hybit::{JacobiPreconditioner, pcg};

const N: usize = 160;
const LOCAL: usize = 96;

fn local_system() -> (Csr32Matrix, Vec<f64>) {
    let mut rows = vec![Vec::<(usize, f64)>::new(); N];
    let mut diagonal = vec![2.0; N];
    for i in 0..LOCAL {
        if i + 1 < LOCAL {
            rows[i].push((i + 1, -1.0));
            rows[i + 1].push((i, -1.0));
        }
    }
    diagonal[LOCAL - 1] += 1.0e-4;
    diagonal[LOCAL] += 1.0e-4;
    rows[LOCAL - 1].push((LOCAL, -1.0e-4));
    rows[LOCAL].push((LOCAL - 1, -1.0e-4));
    let mut ptr = vec![0u32];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for (i, row) in rows.iter_mut().enumerate() {
        row.push((i, diagonal[i]));
        row.sort_unstable_by_key(|&(j, _)| j);
        for &(j, v) in row.iter() {
            cols.push(j as u32);
            vals.push(v);
        }
        ptr.push(cols.len() as u32);
    }
    let matrix = Csr32Matrix::new(N, N, ptr, cols, vals).unwrap();
    let reference = (0..N)
        .map(|i| {
            if i < LOCAL {
                (std::f64::consts::PI * (i + 1) as f64 / (LOCAL + 1) as f64).sin()
            } else {
                0.7 + 0.2 * (0.13 * i as f64).sin()
            }
        })
        .collect();
    (matrix, reference)
}

fn options() -> SolverOptions {
    SolverOptions {
        relative_tolerance: 1.0e-10,
        absolute_tolerance: 0.0,
        max_iterations: 30,
    }
}

fn new_prepared(matrix: &Csr32Matrix) -> hybit::HybitPreparedSystem {
    let mut solver = HybitSolver::new();
    solver.set_options(options()).unwrap();
    solver.set_backend_policy(BackendPolicy::Csr32);
    let mut hybrid = solver.hybrid_options();
    hybrid.abtm_weighted_routing_enabled = true;
    hybrid.probe_iterations = 12;
    hybrid.max_local_region_size = 128; // actual strong component is only 96
    hybrid.max_local_regions = 1;
    hybrid.max_escalations = 1;
    hybrid.overlap_layers = 0;
    hybrid.max_local_factor_bytes = 1024 * 1024;
    solver.set_hybrid_options(hybrid).unwrap();
    solver.prepare(matrix).unwrap()
}

fn true_relative(matrix: &Csr32Matrix, b: &[f64], x: &[f64]) -> f64 {
    let ax = matrix.spmv(x).unwrap();
    let nr = b
        .iter()
        .zip(ax)
        .fold(0.0_f64, |v, (&rhs, ax)| v.hypot(rhs - ax));
    let nb = b.iter().fold(0.0_f64, |v, &rhs| v.hypot(rhs));
    nr / nb
}

#[test]
fn opt_in_prepared_reuses_only_after_matching_rhs_diagnostic() {
    let (matrix, reference) = local_system();
    let b1 = matrix.spmv(&reference).unwrap();
    let b2 = b1.iter().map(|&v| -2.0 * v).collect::<Vec<_>>();
    let mut prepared = new_prepared(&matrix);

    let mut x1 = vec![0.0; N];
    let first = prepared.solve(&matrix, &b1, &mut x1).unwrap();
    assert_eq!(first.status, SolveStatus::Converged);
    assert_eq!(first.escalations, 1);
    assert_eq!(first.largest_local_region, LOCAL);
    assert!(first.local_factor_bytes > 0);
    assert!(!first.preconditioner_reused);
    assert!(true_relative(&matrix, &b1, &x1) < 1.0e-10);

    let mut x2 = vec![0.0; N];
    let second = prepared.solve(&matrix, &b2, &mut x2).unwrap();
    assert_eq!(second.status, SolveStatus::Converged);
    assert_eq!(second.escalations, 1);
    assert!(second.preconditioner_reused);
    assert_eq!(second.local_factor_seconds, 0.0);
    assert_eq!(second.local_factor_bytes, first.local_factor_bytes);
    assert_eq!(second.solve_sequence, 2);
    assert!(true_relative(&matrix, &b2, &x2) < 1.0e-10);

    // A different, trivial RHS must not use the previously cached factor.
    let mut outside = vec![0.0; N];
    outside[140] = 1.0;
    let b3 = matrix.spmv(&outside).unwrap();
    let mut x3 = vec![0.0; N];
    let third = prepared.solve(&matrix, &b3, &mut x3).unwrap();
    assert_eq!(third.status, SolveStatus::Converged);
    assert_eq!(third.escalations, 0);
    assert_eq!(third.preconditioner, PreconditionerKind::Jacobi);
    assert_eq!(third.local_factor_bytes, 0);
    assert!(!third.preconditioner_reused);
    let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
    let mut plain_x = vec![0.0; N];
    let plain = pcg(&matrix, &jacobi, &b3, &mut plain_x, options()).unwrap();
    assert_eq!(third.iterations, plain.iterations);
    assert!(
        x3.iter()
            .zip(plain_x)
            .all(|(&a, b)| a.to_bits() == b.to_bits())
    );

    // Invalid numeric values are rejected before cached data can be used.
    let mut changed_values = matrix.values().to_vec();
    changed_values[0] += 0.125;
    let changed = Csr32Matrix::new(
        N,
        N,
        matrix.row_ptr().to_vec(),
        matrix.col_idx().to_vec(),
        changed_values,
    )
    .unwrap();
    let mut bad_x = vec![0.0; N];
    assert!(prepared.solve(&changed, &b2, &mut bad_x).is_err());

    // A rejected RHS and a failed matrix check must not evict a valid cache.
    let mut x4 = vec![0.0; N];
    let fourth = prepared.solve(&matrix, &b2, &mut x4).unwrap();
    assert_eq!(fourth.status, SolveStatus::Converged);
    assert!(fourth.preconditioner_reused);
    assert_eq!(fourth.local_factor_seconds, 0.0);
    assert_eq!(fourth.escalations, 1);
    assert!(true_relative(&matrix, &b2, &x4) < 1.0e-10);

    // A new prepared system must construct its own local factor, and its
    // numerical result must agree with the cached-factor path.
    let mut fresh = new_prepared(&matrix);
    let mut fresh_x = vec![0.0; N];
    let fresh_report = fresh.solve(&matrix, &b2, &mut fresh_x).unwrap();
    assert_eq!(fresh_report.status, SolveStatus::Converged);
    assert!(!fresh_report.preconditioner_reused);
    assert_eq!(fresh_report.escalations, 1);
    assert!(
        x4.iter()
            .zip(fresh_x)
            .all(|(&a, b)| a.to_bits() == b.to_bits())
    );
}
