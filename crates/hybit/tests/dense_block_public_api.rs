use hybit::{Csr32Matrix, DenseBlockCsrOperator, DenseBlockSize, LinearOperator};

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
fn public_b3_operator_matches_scalar_csr_with_partial_tail() {
    let matrix = Csr32Matrix::new(
        5,
        5,
        vec![0, 3, 6, 9, 12, 15],
        vec![0, 1, 3, 0, 1, 4, 1, 2, 3, 0, 2, 3, 1, 3, 4],
        vec![
            4.0, -1.0, 0.5, -1.0, 5.0, 0.25, 0.5, 6.0, -0.75, 0.5, -0.75, 7.0, 0.25, 1.0, 8.0,
        ],
    )
    .unwrap();

    let operator = DenseBlockCsrOperator::from_csr32(&matrix, DenseBlockSize::B3).unwrap();

    let x = [1.0, -2.0, 0.5, 3.0, -1.5];
    let reference = matrix.spmv(&x).unwrap();
    let mut actual = vec![0.0; matrix.nrows()];
    operator.apply(&x, &mut actual).unwrap();

    assert_close(&reference, &actual);
    assert_eq!(operator.block_width(), 3);
    assert!(operator.unique_blocks() > 0);
    assert!(operator.fast_block_fraction() >= 0.0);
}

#[test]
fn public_b6_operator_matches_scalar_csr() {
    let n = 12usize;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for row in 0..n {
        col_idx.push(row as u32);
        values.push(10.0 + row as f64);

        if row + 1 < n {
            col_idx.push((row + 1) as u32);
            values.push(-0.5);
        }

        if row >= 1 {
            col_idx.push((row - 1) as u32);
            values.push(0.25);
        }

        let coupled = (row + 6) % n;
        col_idx.push(coupled as u32);
        values.push(0.125);

        row_ptr.push(col_idx.len() as u32);
    }

    let matrix = Csr32Matrix::new(n, n, row_ptr, col_idx, values).unwrap();
    let operator = DenseBlockCsrOperator::from_csr32(&matrix, DenseBlockSize::B6).unwrap();

    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.125).collect();
    let reference = matrix.spmv(&x).unwrap();
    let mut actual = vec![0.0; n];
    operator.apply(&x, &mut actual).unwrap();

    assert_close(&reference, &actual);
    assert_eq!(operator.block_width(), 6);
}

#[test]
fn public_dense_block_operator_reports_dimension_errors() {
    let matrix =
        Csr32Matrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 4.0]).unwrap();

    let operator = DenseBlockCsrOperator::from_csr32(&matrix, DenseBlockSize::B3).unwrap();

    let mut y = vec![0.0; 3];
    assert!(operator.apply(&[1.0, 2.0], &mut y).is_err());

    let mut short_y = vec![0.0; 2];
    assert!(operator.apply(&[1.0, 2.0, 3.0], &mut short_y).is_err());
}
