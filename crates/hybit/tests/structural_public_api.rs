use hybit::{
    Csr32Matrix, HybitError, HybitSolver, ParallelRigidBodyTwoLevelPreconditioner,
    Preconditioner, PreconditionerKind, RigidBodyAggregation,
    RigidBodyTwoLevelBlockJacobiPreconditioner, SolverOptions, StructuralOptions,
    StructuralPcgVectorPolicy, StructuralPreconditionerPolicy, StructuralSpmvPolicy,
};

fn identity_3d_nodes(nodes: usize) -> Csr32Matrix {
    let n = nodes * 3;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    row_ptr.push(0);
    for i in 0..n {
        col_idx.push(i as u32);
        values.push(1.0);
        row_ptr.push((i + 1) as u32);
    }
    Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
}

fn cube_coordinates() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
}

fn connected_cube_matrix() -> Csr32Matrix {
    let coords = cube_coordinates();
    let edges = [
        (0usize,1usize),(0,2),(0,4),(1,3),(1,5),(2,3),
        (2,6),(3,7),(4,5),(4,6),(5,7),(6,7),
    ];
    let nodes = coords.len();
    let n = nodes * 3;
    let mut rows = vec![Vec::<(usize, f64)>::new(); n];
    for node in 0..nodes {
        for c in 0..3 { rows[node * 3 + c].push((node * 3 + c, 4.0)); }
    }
    for &(a_node, b_node) in &edges {
        for c in 0..3 {
            rows[a_node * 3 + c].push((b_node * 3 + c, -1.0));
            rows[b_node * 3 + c].push((a_node * 3 + c, -1.0));
        }
    }
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in &mut rows {
        row.sort_unstable_by_key(|entry| entry.0);
        for &(col, value) in row.iter() {
            col_idx.push(col as u32);
            values.push(value);
        }
        row_ptr.push(col_idx.len() as u32);
    }
    Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap()
}

#[test]
fn structural_solver_uses_rigid_body_two_level() {
    let a = identity_3d_nodes(8);
    let coords = cube_coordinates();
    let b = vec![1.0; 24];
    let mut x = vec![0.0; 24];

    let mut solver = HybitSolver::new();
    solver
        .set_options(SolverOptions {
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 50,
        })
        .unwrap();
    solver
        .set_structural_options(StructuralOptions {
            target_coarse_dimension: 6,
            ..StructuralOptions::default()
        })
        .unwrap();

    let report = solver
        .solve_structural_csr32(&a, &coords, &b, &mut x)
        .unwrap();
    assert!(report.converged());
    assert_eq!(report.preconditioner, PreconditionerKind::RigidBodyTwoLevel);
    assert_eq!(report.iterations, 1);
    assert!(x.iter().all(|&v| (v - 1.0).abs() < 1.0e-12));
}

#[test]
fn prepared_structural_system_reuses_coarse_factors() {
    let a = identity_3d_nodes(8);
    let coords = cube_coordinates();
    let b = vec![1.0; 24];

    let mut solver = HybitSolver::new();
    solver
        .set_structural_options(StructuralOptions {
            target_coarse_dimension: 6,
            ..StructuralOptions::default()
        })
        .unwrap();
    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver
        .prepare_structural_csr32(&a, &analysis, &coords)
        .unwrap();

    assert_eq!(prepared.aggregate_nodes(), 8);
    assert_eq!(prepared.aggregate_count(), 1);
    assert_eq!(prepared.coarse_dimension(), 6);
    // Rigid-body coarse Cholesky uses packed lower-triangular storage:
    // 6 * 7 / 2 = 21 f64 values.
    assert_eq!(prepared.coarse_factor_bytes(), 21 * std::mem::size_of::<f64>());
    // The identity has no inter-node graph edges, so Auto must fall back
    // rather than failing the valid SPD solve.
    assert_eq!(prepared.aggregation(), RigidBodyAggregation::Contiguous);
    assert_eq!(prepared.spmv_policy(), StructuralSpmvPolicy::Serial);
    assert_eq!(prepared.pcg_vector_policy(), StructuralPcgVectorPolicy::Serial);

    let mut x1 = vec![0.0; 24];
    let first = prepared.solve(&a, &b, &mut x1).unwrap();
    assert!(first.converged());
    assert!(!first.preconditioner_reused);
    assert_eq!(first.solve_sequence, 1);

    let mut x2 = vec![0.0; 24];
    let second = prepared.solve(&a, &b, &mut x2).unwrap();
    assert!(second.converged());
    assert!(second.preconditioner_reused);
    assert_eq!(second.solve_sequence, 2);
    assert_eq!(second.prepare_seconds, 0.0);
}


#[test]
fn structural_graph_aggregation_uses_connected_regions() {
    let coords = cube_coordinates();
    let a = connected_cube_matrix();
    let n = a.nrows();
    let b = a.spmv(&vec![1.0; n]).unwrap();
    let mut x = vec![0.0; n];

    let mut solver = HybitSolver::new();
    solver.set_options(SolverOptions {
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 50,
    }).unwrap();
    solver.set_structural_options(StructuralOptions {
        target_coarse_dimension: 12,
        aggregation: RigidBodyAggregation::Graph,
        ..StructuralOptions::default()
    }).unwrap();

    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_structural_csr32(&a, &analysis, &coords).unwrap();
    assert_eq!(prepared.aggregation(), RigidBodyAggregation::Graph);
    assert!(prepared.aggregate_count() >= 1);
    assert!(prepared.min_aggregate_nodes() >= 3);
    let report = prepared.solve(&a, &b, &mut x).unwrap();
    assert!(report.converged());
    assert_eq!(report.preconditioner, PreconditionerKind::RigidBodyTwoLevel);
}


#[test]
fn structural_auto_falls_back_when_graph_components_are_too_small() {
    let a = identity_3d_nodes(8);
    let coords = cube_coordinates();
    let mut solver = HybitSolver::new();
    solver.set_structural_options(StructuralOptions {
        target_coarse_dimension: 6,
        ..StructuralOptions::default()
    }).unwrap();
    let analysis = solver.analyze_csr32(&a).unwrap();
    let prepared = solver.prepare_structural_csr32(&a, &analysis, &coords).unwrap();
    assert_eq!(prepared.aggregation(), RigidBodyAggregation::Contiguous);
}

#[test]
fn structural_auto_defaults_to_graph_when_graph_coarse_is_valid() {
    let a = connected_cube_matrix();
    let coords = cube_coordinates();
    let mut solver = HybitSolver::new();
    solver.set_structural_options(StructuralOptions {
        target_coarse_dimension: 6,
        ..StructuralOptions::default()
    }).unwrap();
    assert_eq!(solver.structural_options().aggregation, RigidBodyAggregation::Auto);
    let analysis = solver.analyze_csr32(&a).unwrap();
    let prepared = solver.prepare_structural_csr32(&a, &analysis, &coords).unwrap();
    assert_eq!(prepared.aggregation(), RigidBodyAggregation::Graph);
}


#[test]
fn structural_explicit_graph_rejects_too_small_components() {
    let a = identity_3d_nodes(8);
    let coords = cube_coordinates();
    let mut solver = HybitSolver::new();
    solver
        .set_structural_options(StructuralOptions {
            target_coarse_dimension: 6,
            aggregation: RigidBodyAggregation::Graph,
            ..StructuralOptions::default()
        })
        .unwrap();
    let analysis = solver.analyze_csr32(&a).unwrap();

    match solver.prepare_structural_csr32(&a, &analysis, &coords) {
        Err(HybitError::InvalidArgument(msg)) => {
            assert_eq!(msg, "a structural graph component contains fewer than three nodes");
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("explicit Graph must reject structural components smaller than three nodes"),
    }
}

#[test]
fn structural_explicit_parallel_spmv_is_available() {
    let a = identity_3d_nodes(8);
    let coords = cube_coordinates();
    let b = vec![1.0; 24];
    let mut x = vec![0.0; 24];

    let mut solver = HybitSolver::new();
    solver
        .set_structural_options(StructuralOptions {
            target_coarse_dimension: 6,
            spmv_policy: StructuralSpmvPolicy::Parallel,
            ..StructuralOptions::default()
        })
        .unwrap();
    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver
        .prepare_structural_csr32(&a, &analysis, &coords)
        .unwrap();
    assert_eq!(prepared.spmv_policy(), StructuralSpmvPolicy::Parallel);
    let report = prepared.solve(&a, &b, &mut x).unwrap();
    assert!(report.converged());
    assert_eq!(report.iterations, 1);
}

#[test]
fn parallel_rigid_body_preconditioner_matches_serial() {
    let a = connected_cube_matrix();
    let coords = cube_coordinates();
    let serial = RigidBodyTwoLevelBlockJacobiPreconditioner::from_csr32_graph(&a, &coords, 8).unwrap();
    let parallel = ParallelRigidBodyTwoLevelPreconditioner::new(&serial).unwrap();
    let r: Vec<f64> = (0..a.nrows()).map(|i| ((i * 17 + 5) % 23) as f64 - 11.0).collect();
    let mut zs = vec![0.0; a.nrows()];
    let mut zp = vec![0.0; a.nrows()];
    serial.apply(&r, &mut zs).unwrap();
    parallel.apply(&r, &mut zp).unwrap();
    let scale = zs.iter().fold(1.0f64, |m, &v| m.max(v.abs()));
    let max_diff = zs.iter().zip(&zp).fold(0.0f64, |m, (&a, &b)| m.max((a - b).abs()));
    assert!(max_diff <= 1.0e-12 * scale, "parallel rigid-body preconditioner mismatch: {max_diff:e}");
}


#[test]
fn structural_auto_keeps_tiny_preconditioner_serial() {
    let a = connected_cube_matrix();
    let coords = cube_coordinates();
    let solver = HybitSolver::new();
    let analysis = solver.analyze_csr32(&a).unwrap();
    let prepared = solver.prepare_structural_csr32(&a, &analysis, &coords).unwrap();
    assert_eq!(prepared.structural_preconditioner_policy(), StructuralPreconditionerPolicy::Serial);
    assert!(!prepared.parallel_preconditioner_enabled());
}

#[test]
fn structural_explicit_parallel_preconditioner_is_available() {
    let a = connected_cube_matrix();
    let coords = cube_coordinates();
    let b = vec![1.0; 24];
    let mut x = vec![0.0; 24];
    let mut solver = HybitSolver::new();
    solver.set_structural_options(StructuralOptions {
        target_coarse_dimension: 6,
        preconditioner_policy: StructuralPreconditionerPolicy::Parallel,
        ..StructuralOptions::default()
    }).unwrap();
    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_structural_csr32(&a, &analysis, &coords).unwrap();
    assert_eq!(prepared.structural_preconditioner_policy(), StructuralPreconditionerPolicy::Parallel);
    assert!(prepared.parallel_preconditioner_enabled());
    assert!(prepared.parallel_preconditioner_index_bytes() > 0);
    let report = prepared.solve(&a, &b, &mut x).unwrap();
    assert!(report.converged());
}

#[test]
fn structural_auto_keeps_tiny_pcg_vectors_serial() {
    let a = connected_cube_matrix();
    let coords = cube_coordinates();
    let solver = HybitSolver::new();
    let analysis = solver.analyze_csr32(&a).unwrap();
    let prepared = solver.prepare_structural_csr32(&a, &analysis, &coords).unwrap();
    assert_eq!(prepared.pcg_vector_policy(), StructuralPcgVectorPolicy::Serial);
    assert!(!prepared.parallel_pcg_vectors_enabled());
}

#[test]
fn structural_explicit_parallel_pcg_vectors_are_available() {
    let a = connected_cube_matrix();
    let coords = cube_coordinates();
    let b = vec![1.0; 24];
    let mut x = vec![0.0; 24];
    let mut solver = HybitSolver::new();
    solver.set_structural_options(StructuralOptions {
        target_coarse_dimension: 6,
        pcg_vector_policy: StructuralPcgVectorPolicy::Parallel,
        ..StructuralOptions::default()
    }).unwrap();
    let analysis = solver.analyze_csr32(&a).unwrap();
    let mut prepared = solver.prepare_structural_csr32(&a, &analysis, &coords).unwrap();
    assert_eq!(prepared.pcg_vector_policy(), StructuralPcgVectorPolicy::Parallel);
    assert!(prepared.parallel_pcg_vectors_enabled());
    let report = prepared.solve(&a, &b, &mut x).unwrap();
    assert!(report.converged());
}

