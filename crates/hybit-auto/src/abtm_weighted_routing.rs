//! C1-H6: conservative opt-in weighted ABTM candidate diagnostic.
//!
//! No mutation of the operator, PCG recurrence, or preconditioner occurs here.
//! This is an intentionally narrow research policy, not an SPD certificate.
use std::collections::VecDeque;

use hybit_core::HybitError;
use hybit_matrix::{AbtmTopology, Csr32Matrix};

use super::HybridOptions;

const STRONG_EDGE_THRESHOLD: f64 = 0.1;

pub(super) fn candidate(
    matrix: &Csr32Matrix,
    residual: &[f64],
    options: HybridOptions,
) -> Result<Option<Vec<usize>>, HybitError> {
    let n = matrix.nrows();
    if n == 0 || n != matrix.ncols() || residual.len() != n {
        return Ok(None);
    }
    let capacity = options.max_local_region_size;
    if capacity < 2 || capacity >= n {
        return Ok(None);
    }
    let diagonal = matrix.diagonal()?;
    if diagonal.iter().any(|&d| !d.is_finite() || d <= 0.0)
        || residual.iter().any(|v| !v.is_finite())
    {
        return Ok(None);
    }
    // The C1-H5 pilot assumed canonical CSR. Fail closed on unsorted or
    // repeated indices rather than silently selecting the wrong edge weight.
    for row in 0..n {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let columns = &matrix.col_idx()[start..end];
        if columns.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Ok(None);
        }
    }

    let root = (0..n)
        .max_by(|&i, &j| {
            let left = residual[i].abs() / diagonal[i].sqrt();
            let right = residual[j].abs() / diagonal[j].sqrt();
            left.total_cmp(&right).then_with(|| j.cmp(&i))
        })
        .expect("nonempty dimensions checked above");
    let topology = AbtmTopology::from_csr32(matrix)?;
    let mut visited = vec![false; n];
    let mut queue = VecDeque::from([root]);
    let mut region = Vec::with_capacity(capacity);
    visited[root] = true;
    while let Some(i) = queue.pop_front() {
        // A queued (capacity + 1)-th strong DOF proves truncation: fail closed.
        // A component of exactly `capacity` DOFs is accepted only if the
        // queue becomes empty after all its strong neighbors are examined.
        if region.len() == capacity {
            return Ok(None);
        }
        region.push(i);
        let start = matrix.row_ptr()[i] as usize;
        let end = matrix.row_ptr()[i + 1] as usize;
        let columns = &matrix.col_idx()[start..end];
        for word in topology.row(i)?.words() {
            let mut bits = word.mask();
            while bits != 0 {
                let j = word.base_col() + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if j == i || j >= n || visited[j] {
                    continue;
                }
                let Ok(offset) = columns.binary_search(&(j as u32)) else {
                    continue;
                };
                let weight =
                    matrix.values()[start + offset].abs() / (diagonal[i] * diagonal[j]).sqrt();
                if weight.is_finite() && weight >= STRONG_EDGE_THRESHOLD {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
    }
    if region.len() < 2 {
        return Ok(None);
    }
    // Charge the actual closed component, not its configured maximum.
    let Some(bytes) = region
        .len()
        .checked_mul(region.len())
        .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
    else {
        return Ok(None);
    };
    if bytes > options.max_local_factor_bytes {
        return Ok(None);
    }
    region.sort_unstable();
    let mut inside = vec![false; n];
    for &row in &region {
        inside[row] = true;
    }
    let mut strong_cut = 0usize;
    let mut weak_cut = 0usize;
    for &row in &region {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for p in start..end {
            let col = matrix.col_idx()[p] as usize;
            if col == row || inside[col] {
                continue;
            }
            let weight = matrix.values()[p].abs() / (diagonal[row] * diagonal[col]).sqrt();
            if !weight.is_finite() {
                return Ok(None);
            }
            if weight >= STRONG_EDGE_THRESHOLD {
                strong_cut += 1;
            } else if weight > 0.0 {
                weak_cut += 1;
            }
        }
    }
    if strong_cut == 0 && weak_cut > 0 {
        Ok(Some(region))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hybit_core::{SolveStatus, SolverOptions};
    use hybit_krylov::pcg;
    use hybit_precond::JacobiPreconditioner;

    fn chain(global: bool) -> (Csr32Matrix, Vec<f64>) {
        const N: usize = 160;
        const LOCAL: usize = 96;
        let mut rows = vec![Vec::<(usize, f64)>::new(); N];
        let mut diagonal = vec![2.0_f64; N];
        for i in 0..N - 1 {
            let coupling = if global || i < LOCAL - 1 {
                1.0
            } else if i == LOCAL - 1 {
                diagonal[i] += 1.0e-4;
                diagonal[i + 1] += 1.0e-4;
                1.0e-4
            } else {
                continue;
            };
            rows[i].push((i + 1, -coupling));
            rows[i + 1].push((i, -coupling));
        }
        for i in 0..N {
            rows[i].push((i, diagonal[i]));
            rows[i].sort_unstable_by_key(|&(j, _)| j);
        }
        let mut ptr = vec![0u32];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for row in rows {
            for (j, v) in row {
                cols.push(j as u32);
                vals.push(v);
            }
            ptr.push(cols.len() as u32);
        }
        let matrix = Csr32Matrix::new(N, N, ptr, cols, vals).unwrap();
        let xref = (0..N)
            .map(|i| {
                if global {
                    (std::f64::consts::PI * (i + 1) as f64 / (N + 1) as f64).sin()
                        + 0.11 * (0.21 * i as f64).sin()
                } else if i < LOCAL {
                    (std::f64::consts::PI * (i + 1) as f64 / (LOCAL + 1) as f64).sin()
                } else {
                    0.7 + 0.2 * (0.13 * i as f64).sin()
                }
            })
            .collect::<Vec<_>>();
        (matrix, xref)
    }

    fn options() -> super::HybridOptions {
        super::HybridOptions {
            abtm_weighted_routing_enabled: true,
            probe_iterations: 12,
            max_local_region_size: 96,
            max_local_regions: 1,
            max_escalations: 1,
            overlap_layers: 0,
            ..super::HybridOptions::default()
        }
    }

    #[test]
    fn local_boundary_admitted_global_truncation_rejected() {
        for global in [false, true] {
            let (matrix, reference) = chain(global);
            let rhs = matrix.spmv(&reference).unwrap();
            let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
            let opts = SolverOptions {
                relative_tolerance: 1.0e-10,
                absolute_tolerance: 0.0,
                max_iterations: 12,
            };
            let mut x = vec![0.0; matrix.nrows()];
            let out = pcg(&matrix, &jacobi, &rhs, &mut x, opts).unwrap();
            assert_eq!(out.status, SolveStatus::MaxIterations);
            let ax = matrix.spmv(&x).unwrap();
            let residual = rhs
                .iter()
                .zip(ax)
                .map(|(&b, ax)| b - ax)
                .collect::<Vec<_>>();
            let proposed = candidate(&matrix, &residual, options()).unwrap();
            if global {
                assert!(proposed.is_none());
            } else {
                assert_eq!(proposed.unwrap(), (0..96).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn opt_in_auto_rescues_local_and_preserves_global_pcg_bits() {
        for global in [false, true] {
            let (matrix, reference) = chain(global);
            let rhs = matrix.spmv(&reference).unwrap();
            let opts = SolverOptions {
                relative_tolerance: 1.0e-10,
                absolute_tolerance: 0.0,
                max_iterations: 30,
            };
            let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
            let mut baseline_x = vec![0.0; matrix.ncols()];
            let baseline = pcg(&matrix, &jacobi, &rhs, &mut baseline_x, opts).unwrap();
            assert_eq!(baseline.status, SolveStatus::MaxIterations);

            let mut solver = super::super::HybitSolver::new();
            solver.set_options(opts).unwrap();
            solver.set_backend_policy(super::super::BackendPolicy::Csr32);
            solver.set_hybrid_options(options()).unwrap();
            let mut x = vec![0.0; matrix.ncols()];
            let report = solver.solve_csr32(&matrix, &rhs, &mut x).unwrap();
            if global {
                assert_eq!(report.escalations, 0);
                assert_eq!(report.status, baseline.status);
                assert_eq!(report.iterations, baseline.iterations);
                assert!(
                    x.iter()
                        .zip(&baseline_x)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            } else {
                assert_eq!(report.status, SolveStatus::Converged);
                assert_eq!(report.iterations, 14);
                assert_eq!(report.escalations, 1);
                let ax = matrix.spmv(&x).unwrap();
                let r = rhs
                    .iter()
                    .zip(ax)
                    .map(|(&b, ax)| b - ax)
                    .collect::<Vec<_>>();
                let nr = r.iter().fold(0.0_f64, |acc, &v| acc.hypot(v));
                let nb = rhs.iter().fold(0.0_f64, |acc, &v| acc.hypot(v));
                assert!(nr / nb <= 1.0e-10);
            }
        }
    }

    // C1-H8: verify that a complete strong component may be *smaller* than
    // the capacity, while a component larger than the cap must fail closed.
    fn sized_local_chain(local: usize) -> (Csr32Matrix, Vec<f64>) {
        let n = 160;
        let mut rows = vec![Vec::<(usize, f64)>::new(); n];
        let mut diagonal = vec![2.0_f64; n];
        for i in 0..local - 1 {
            rows[i].push((i + 1, -1.0));
            rows[i + 1].push((i, -1.0));
        }
        rows[local - 1].push((local, -1.0e-4));
        rows[local].push((local - 1, -1.0e-4));
        diagonal[local - 1] += 1.0e-4;
        diagonal[local] += 1.0e-4;
        for (i, row) in rows.iter_mut().enumerate() {
            row.push((i, diagonal[i]));
            row.sort_unstable_by_key(|&(col, _)| col);
        }
        let mut row_ptr = vec![0u32];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for row in rows {
            for (col, value) in row {
                columns.push(col as u32);
                values.push(value);
            }
            row_ptr.push(columns.len() as u32);
        }
        let matrix = Csr32Matrix::new(n, n, row_ptr, columns, values).unwrap();
        let reference = (0..n)
            .map(|i| {
                if i < local {
                    (std::f64::consts::PI * (i + 1) as f64 / (local + 1) as f64).sin()
                } else {
                    0.7 + 0.2 * (0.13 * i as f64).sin()
                }
            })
            .collect();
        (matrix, reference)
    }

    fn probed_residual(matrix: &Csr32Matrix, reference: &[f64]) -> Vec<f64> {
        let rhs = matrix.spmv(reference).unwrap();
        let jacobi = JacobiPreconditioner::from_csr32(matrix).unwrap();
        let mut x = vec![0.0; matrix.nrows()];
        let _outcome = pcg(
            matrix,
            &jacobi,
            &rhs,
            &mut x,
            SolverOptions {
                relative_tolerance: 1.0e-10,
                absolute_tolerance: 0.0,
                max_iterations: 12,
            },
        )
        .unwrap();
        rhs.iter()
            .zip(matrix.spmv(&x).unwrap())
            .map(|(&bi, ai)| bi - ai)
            .collect()
    }

    #[test]
    fn c1h8_naturally_closed_regions_and_truncated_rejection() {
        for size in [64, 96, 128] {
            let (matrix, reference) = sized_local_chain(size);
            let residual = probed_residual(&matrix, &reference);
            let mut config = options();
            config.max_local_region_size = 128;
            assert_eq!(
                candidate(&matrix, &residual, config).unwrap(),
                Some((0..size).collect::<Vec<_>>()),
                "a closed {size}-DOF strong component must fit cap=128"
            );
            config.max_local_region_size = size - 1;
            assert_eq!(
                candidate(&matrix, &residual, config).unwrap(),
                None,
                "a truncated {size}-DOF component must be rejected"
            );
        }
    }

    #[test]
    fn c1h8_memory_bound_uses_actual_region_size() {
        let (matrix, reference) = sized_local_chain(64);
        let residual = probed_residual(&matrix, &reference);
        let mut config = options();
        config.max_local_region_size = 128;
        config.max_local_factor_bytes = 8 * 64 * 64;
        assert_eq!(
            candidate(&matrix, &residual, config).unwrap(),
            Some((0..64).collect())
        );
        config.max_local_factor_bytes -= 1;
        assert!(candidate(&matrix, &residual, config).unwrap().is_none());
    }

    #[test]
    fn invalid_opt_in_combinations_are_rejected() {
        let mut bad = options();
        bad.overlap_layers = 1;
        assert!(bad.validate().is_err());
        bad.overlap_layers = 0;
        bad.algebraic_coarse.enabled = true;
        assert!(bad.validate().is_err());
        bad.algebraic_coarse.enabled = false;
        bad.progress_monitor_interval = 64;
        assert!(bad.validate().is_err());
    }
}
