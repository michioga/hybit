use hybit::{
    AbtmConfig, AbtmMatrix, Csr32Matrix, DofMask, LinearOperator,
    PreparedColumnRestrictedCsrOperator, PreparedLocalCsrOperator,
};

fn assert_close(reference: &[f64], actual: &[f64]) {
    assert_eq!(reference.len(), actual.len());
    for (&r, &a) in reference.iter().zip(actual) {
        let scale = r.abs().max(a.abs()).max(1.0);
        assert!(
            (r - a).abs() / scale <= 1.0e-13,
            "reference={r:e} actual={a:e}"
        );
    }
}

#[test]
fn public_column_restriction_matches_masked_global_operator() {
    let matrix = Csr32Matrix::new(
        4,
        4,
        vec![0, 3, 6, 9, 12],
        vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3],
        vec![
            4.0, -1.0, 0.5, -1.0, 5.0, 0.25, 0.25, 6.0, -0.75, 0.5, -0.75, 7.0,
        ],
    )
    .unwrap();
    let mask = DofMask::from_indices(4, &[0, 2]).unwrap();
    let prepared = PreparedColumnRestrictedCsrOperator::from_csr32(&matrix, &mask).unwrap();

    let x = [1.0, 2.0, 3.0, 4.0];
    let x_masked = [1.0, 0.0, 3.0, 0.0];

    let reference = matrix.spmv(&x_masked).unwrap();
    let mut actual = vec![0.0; 4];
    prepared.apply(&x, &mut actual).unwrap();

    assert_close(&reference, &actual);
    assert_eq!(prepared.rows(), 4);
    assert_eq!(prepared.cols(), 4);
    assert!(prepared.retained_fraction() > 0.0);
    assert!(prepared.retained_fraction() < 1.0);
}

#[test]
fn public_abtm_column_preparation_is_explicit_and_equivalent() {
    let matrix = Csr32Matrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
    )
    .unwrap();
    let mask = DofMask::from_indices(3, &[0, 2]).unwrap();
    let abtm = AbtmMatrix::from_csr32(&matrix, AbtmConfig::default()).unwrap();

    let direct = PreparedColumnRestrictedCsrOperator::from_csr32(&matrix, &mask).unwrap();
    let metadata = PreparedColumnRestrictedCsrOperator::from_abtm(&abtm, &mask).unwrap();

    let x = [0.25, -2.0, 1.5];
    let mut yd = vec![0.0; 3];
    let mut ya = vec![0.0; 3];
    direct.apply(&x, &mut yd).unwrap();
    metadata.apply(&x, &mut ya).unwrap();
    assert_close(&yd, &ya);
}

#[test]
fn public_local_restriction_uses_compact_local_vectors() {
    let matrix = Csr32Matrix::new(
        5,
        5,
        vec![0, 3, 6, 9, 12, 15],
        vec![0, 1, 4, 0, 1, 2, 1, 2, 3, 0, 3, 4, 0, 3, 4],
        vec![
            4.0, -1.0, 0.5, -1.0, 5.0, 0.25, 0.25, 6.0, -0.75, 0.5, 7.0, -1.0, 0.5, -1.0, 8.0,
        ],
    )
    .unwrap();
    let region = DofMask::from_indices(5, &[0, 3, 4]).unwrap();
    let prepared = PreparedLocalCsrOperator::from_csr32(&matrix, &region).unwrap();

    assert_eq!(prepared.global_nodes(), &[0, 3, 4]);
    assert_eq!(prepared.rows(), 3);
    assert_eq!(prepared.cols(), 3);

    let x_global = [1.0, 2.0, 3.0, 4.0, 5.0];
    let x_local = prepared.gather_input_vec(&x_global).unwrap();
    let mut actual = vec![0.0; 3];
    prepared.apply(&x_local, &mut actual).unwrap();

    let mut reference = vec![0.0; 3];
    for (local_row, &global_row) in prepared.global_nodes().iter().enumerate() {
        let row = global_row as usize;
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            if region.contains(col) {
                reference[local_row] += matrix.values()[p] * x_global[col];
            }
        }
    }

    assert_close(&reference, &actual);
    assert!(prepared.storage_bytes() > 0);
}
#[test]
fn public_prepared_restrictions_expose_explicit_rayon_execution() {
    let matrix = Csr32Matrix::new(
        5,
        5,
        vec![0, 3, 6, 9, 12, 15],
        vec![0, 1, 4, 0, 1, 2, 1, 2, 3, 0, 3, 4, 0, 3, 4],
        vec![
            4.0, -1.0, 0.5, -1.0, 5.0, 0.25, 0.25, 6.0, -0.75, 0.5, 7.0, -1.0, 0.5, -1.0, 8.0,
        ],
    )
    .unwrap();

    let x_global = [1.0, 2.0, 3.0, 4.0, 5.0];

    let mask = DofMask::from_indices(5, &[0, 2, 4]).unwrap();
    let column = PreparedColumnRestrictedCsrOperator::from_csr32(&matrix, &mask).unwrap();

    let mut ys = vec![0.0; 5];
    let mut yp = vec![0.0; 5];
    let mut yt = vec![0.0; 5];
    column.apply(&x_global, &mut ys).unwrap();
    column.apply_parallel(&x_global, &mut yp).unwrap();
    column
        .apply_parallel_with_tasks(&x_global, &mut yt, 2)
        .unwrap();
    assert_close(&ys, &yp);
    assert_close(&ys, &yt);

    let region = DofMask::from_indices(5, &[0, 3, 4]).unwrap();
    let local = PreparedLocalCsrOperator::from_csr32(&matrix, &region).unwrap();
    let x_local = local.gather_input_vec(&x_global).unwrap();

    let mut lys = vec![0.0; local.local_nodes()];
    let mut lyp = vec![0.0; local.local_nodes()];
    let mut lyt = vec![0.0; local.local_nodes()];
    local.apply(&x_local, &mut lys).unwrap();
    local.apply_parallel(&x_local, &mut lyp).unwrap();
    local
        .apply_parallel_with_tasks(&x_local, &mut lyt, 2)
        .unwrap();
    assert_close(&lys, &lyp);
    assert_close(&lys, &lyt);
}
