//! C1-H5: opt-in experimental ABTM topology-aware routing (NOT production Auto).
//!
//! Pilot automatic *decision*, not integrated with the production HybitSolver or FFI.
use std::collections::VecDeque;
use std::error::Error;
use std::time::Instant;

use hybit_krylov::{PcgWorkspace, pcg_continue_with_workspace, pcg_start_with_workspace};

use hybit::{
    AbtmTopology, Csr32Matrix, HybridPreconditioner, JacobiPreconditioner, SolveStatus,
    SolverOptions, pcg,
};

const N: usize = 160;
const LOCAL: usize = 96;
const BUDGET: usize = 30;
const PROBE: usize = 12;
const TOL: f64 = 1.0e-10;
const WEAK_EDGE: f64 = 1.0e-4;
const STRONG_THRESHOLD: f64 = 0.1;
const MAX_ESTIMATED_FACTOR_BYTES: usize = 128 * 1024;

type AnyResult<T> = Result<T, Box<dyn Error>>;

fn norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.hypot(x))
}

// Dirichlet 1D chain: diag=2, offdiag=-1, SPD by its energy form.
// The local case couples a 96-DOF chain to an uncoupled diagonal remainder
// through one weak PSD Laplacian edge. Both construction families are SPD.
fn build_case(global: bool) -> AnyResult<(Csr32Matrix, Vec<f64>)> {
    let mut rows = vec![Vec::<(usize, f64)>::new(); N];
    let mut diagonal = vec![2.0_f64; N];
    for i in 0..N - 1 {
        let coupling = if global || i < LOCAL - 1 {
            1.0
        } else if i == LOCAL - 1 {
            diagonal[i] += WEAK_EDGE;
            diagonal[i + 1] += WEAK_EDGE;
            WEAK_EDGE
        } else {
            continue;
        };
        rows[i].push((i + 1, -coupling));
        rows[i + 1].push((i, -coupling));
    }
    for i in 0..N {
        rows[i].push((i, diagonal[i]));
        rows[i].sort_unstable_by_key(|&(col, _)| col);
    }
    let mut row_ptr = Vec::<u32>::with_capacity(N + 1);
    let mut col_idx = Vec::<u32>::new();
    let mut values = Vec::<f64>::new();
    row_ptr.push(0);
    for row in rows {
        for (col, value) in row {
            col_idx.push(u32::try_from(col)?);
            values.push(value);
        }
        row_ptr.push(u32::try_from(col_idx.len())?);
    }
    let a = Csr32Matrix::new(N, N, row_ptr, col_idx, values)?;
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
        .collect();
    Ok((a, xref))
}

fn true_relative(a: &Csr32Matrix, b: &[f64], x: &[f64]) -> AnyResult<f64> {
    let ax = a.spmv(x)?;
    let residual: Vec<f64> = b.iter().zip(ax).map(|(&bi, ai)| bi - ai).collect();
    Ok(norm(&residual) / norm(b).max(f64::MIN_POSITIVE))
}

fn forward_relative(x: &[f64], reference: &[f64]) -> f64 {
    let differences: Vec<f64> = x.iter().zip(reference).map(|(&a, &b)| a - b).collect();
    norm(&differences) / norm(reference).max(f64::MIN_POSITIVE)
}

// Strong-edge graph built by intersecting ABTM topology words with numerical
// edge strengths. CSR32 rows in this *synthetic corpus* are canonical/sorted.
fn weighted_region(
    a: &Csr32Matrix,
    topology: &AbtmTopology,
    diagonal: &[f64],
    root: usize,
    capacity: usize,
    threshold: f64,
) -> AnyResult<Vec<usize>> {
    let mut visited = vec![false; a.nrows()];
    let mut queue = VecDeque::<usize>::from([root]);
    let mut selected = Vec::with_capacity(capacity);
    visited[root] = true;
    while let Some(i) = queue.pop_front() {
        selected.push(i);
        if selected.len() == capacity {
            break;
        }
        let start = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        let columns = &a.col_idx()[start..end];
        for word in topology.row(i)?.words() {
            let mut bits = word.mask();
            while bits != 0 {
                let j = word.base_col() + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if j == i || visited[j] {
                    continue;
                }
                let Ok(offset) = columns.binary_search(&(j as u32)) else {
                    continue;
                };
                let weight = a.values()[start + offset].abs() / (diagonal[i] * diagonal[j]).sqrt();
                if weight >= threshold {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
    }
    selected.sort_unstable();
    Ok(selected)
}

fn boundary_cuts(a: &Csr32Matrix, diagonal: &[f64], region: &[usize]) -> (usize, usize) {
    let mut inside = vec![false; a.nrows()];
    for &i in region {
        inside[i] = true;
    }
    let mut strong = 0;
    let mut weak = 0;
    for &i in region {
        let start = a.row_ptr()[i] as usize;
        let end = a.row_ptr()[i + 1] as usize;
        for p in start..end {
            let j = a.col_idx()[p] as usize;
            if i != j && !inside[j] {
                let w = a.values()[p].abs() / (diagonal[i] * diagonal[j]).sqrt();
                if w >= STRONG_THRESHOLD {
                    strong += 1;
                } else if w > 0.0 {
                    weak += 1;
                }
            }
        }
    }
    (strong, weak)
}

// Experimental decision rule: deliberately restrictive and **not trained** on
// physical FEM matrices. The topology boundary is inspected before factoring;
// a region truncated by the budget or with a strong external connection is
// never admitted. A local weak cut alone is NOT a general rescue certificate.
fn local_direct_candidate(region_size: usize, strong_cut: usize, weak_cut: usize) -> bool {
    let conservative_bytes = region_size
        .checked_mul(region_size)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()));
    region_size == LOCAL
        && strong_cut == 0
        && weak_cut > 0
        && conservative_bytes.is_some_and(|bytes| bytes <= MAX_ESTIMATED_FACTOR_BYTES)
}

fn run_case(global: bool) -> AnyResult<bool> {
    let name = if global {
        "global_chain"
    } else {
        "localized_chain"
    };
    let (a, reference) = build_case(global)?;
    let b = a.spmv(&reference)?;
    let diagonal = a.diagonal()?;
    let jacobi = JacobiPreconditioner::from_csr32(&a)?;
    let opts = SolverOptions {
        relative_tolerance: TOL,
        absolute_tolerance: 0.0,
        max_iterations: BUDGET,
    };

    let mut x_plain = vec![0.0; N];
    let plain_start = Instant::now();
    let plain = pcg(&a, &jacobi, &b, &mut x_plain, opts)?;
    let plain_ms = plain_start.elapsed().as_secs_f64() * 1000.0;
    let plain_res = true_relative(&a, &b, &x_plain)?;
    let plain_err = forward_relative(&x_plain, &reference);

    // Preserve the exact Jacobi-PCG recurrence when no promotion is selected.
    let mut x = vec![0.0; N];
    let mut workspace = PcgWorkspace::new(N);
    let mut session = pcg_start_with_workspace(&a, &jacobi, &b, &mut x, opts, &mut workspace)?;
    let probe_start = Instant::now();
    let probe =
        pcg_continue_with_workspace(&a, &jacobi, &b, &mut x, PROBE, &mut session, &mut workspace)?;
    let probe_ms = probe_start.elapsed().as_secs_f64() * 1000.0;
    if probe.iterations != PROBE || probe.status != SolveStatus::MaxIterations {
        return Err("probe ended early: fixed controls require a complete probe".into());
    }

    let ax = a.spmv(&x)?;
    let seed = (0..N)
        .max_by(|&i, &j| {
            let si = (b[i] - ax[i]).abs() / diagonal[i].sqrt();
            let sj = (b[j] - ax[j]).abs() / diagonal[j].sqrt();
            si.total_cmp(&sj).then_with(|| j.cmp(&i))
        })
        .ok_or("empty seed candidates")?;
    let topology_start = Instant::now();
    let topology = AbtmTopology::from_csr32(&a)?;
    let region = weighted_region(&a, &topology, &diagonal, seed, LOCAL, STRONG_THRESHOLD)?;
    let (strong_cut, weak_cut) = boundary_cuts(&a, &diagonal, &region);
    let topology_ms = topology_start.elapsed().as_secs_f64() * 1000.0;
    let promote = local_direct_candidate(region.len(), strong_cut, weak_cut);

    let mut factor_bytes = 0;
    let mut factor_ms = 0.0;
    let (stage, stage_ms) = if promote {
        // A changed preconditioner must restart the PCG recurrence.
        let factor_start = Instant::now();
        let local = HybridPreconditioner::from_csr32(&a, vec![region.clone()])?;
        factor_ms = factor_start.elapsed().as_secs_f64() * 1000.0;
        factor_bytes = local.factor_bytes();
        if factor_bytes > MAX_ESTIMATED_FACTOR_BYTES {
            return Err("actual factor exceeds experimental memory ceiling".into());
        }
        let stage_start = Instant::now();
        let outcome = pcg(
            &a,
            &local,
            &b,
            &mut x,
            SolverOptions {
                max_iterations: BUDGET - PROBE,
                ..opts
            },
        )?;
        (outcome, stage_start.elapsed().as_secs_f64() * 1000.0)
    } else {
        // No restart: retain Krylov basis and preconditioner state.
        let stage_start = Instant::now();
        let outcome = pcg_continue_with_workspace(
            &a,
            &jacobi,
            &b,
            &mut x,
            BUDGET - PROBE,
            &mut session,
            &mut workspace,
        )?;
        (outcome, stage_start.elapsed().as_secs_f64() * 1000.0)
    };
    let verified = true_relative(&a, &b, &x)?;
    let forward = forward_relative(&x, &reference);
    let total_iterations = PROBE + stage.iterations;
    let invariant = x
        .iter()
        .zip(&x_plain)
        .all(|(&a, &b)| a.to_bits() == b.to_bits())
        && stage.status == plain.status
        && total_iterations == plain.iterations;

    println!("=== {name} ===");
    println!("matrix SPD        : constructed Dirichlet / weak positive-semidefinite edge");
    println!("dimension / nnz    : {N} / {}", a.nnz());
    println!(
        "iteration budget   : {BUDGET} (probe {PROBE} + stage {})",
        BUDGET - PROBE
    );
    println!("plain status       : {:?}", plain.status);
    println!("plain iterations   : {}", plain.iterations);
    println!("plain residual     : {plain_res:.9e}");
    println!("plain forward err  : {plain_err:.9e}");
    println!("ABTM seed          : {seed}");
    println!("ABTM region DOFs   : {}", region.len());
    println!("strong/weak cuts   : {strong_cut} / {weak_cut}");
    println!(
        "routing decision   : {}",
        if promote { "LOCAL_DIRECT" } else { "KEEP_PCG" }
    );
    println!("routing status     : {:?}", stage.status);
    println!("routing iterations : {total_iterations}");
    println!("routing residual   : {verified:.9e}");
    println!("routing forward err: {forward:.9e}");
    println!("local factor bytes : {factor_bytes}");
    println!(
        "timing ms          : plain={plain_ms:.3} probe={probe_ms:.3} topology={topology_ms:.3} factor={factor_ms:.3} stage={stage_ms:.3}"
    );
    println!("qualification      : independent routing trial, NOT production HyBIT Auto");

    if global {
        let pass = !promote && factor_bytes == 0 && invariant;
        println!(
            "keep PCG invariant : {}",
            if pass { "PASS" } else { "FAIL" }
        );
        return Ok(pass);
    }
    let rescue = promote
        && plain.status == SolveStatus::MaxIterations
        && plain_res > TOL
        && stage.status == SolveStatus::Converged
        && verified <= TOL
        && forward <= 1.0e-6
        && total_iterations <= BUDGET
        && region == (0..LOCAL).collect::<Vec<_>>()
        && strong_cut == 0
        && weak_cut == 1
        && factor_bytes > 0;
    println!(
        "local rescue gate  : {}",
        if rescue { "PASS" } else { "FAIL" }
    );
    Ok(rescue)
}

fn main() -> AnyResult<()> {
    if std::env::args().len() != 1 {
        return Err("no arguments supported: C1-H5 is a fixed synthetic control".into());
    }
    let positive = run_case(false)?;
    let negative = run_case(true)?;
    if !positive || !negative {
        return Err("C1-H5 experimental routing gate FAILED".into());
    }
    println!("C1-H5 routing control gate PASS (no production Auto claim)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_early_gate_rejects_strong_boundary_and_oversize() {
        assert!(local_direct_candidate(96, 0, 1));
        assert!(!local_direct_candidate(96, 1, 1));
        assert!(!local_direct_candidate(96, 0, 0));
        assert!(!local_direct_candidate(95, 0, 1));
        assert!(!local_direct_candidate(160, 0, 1));
    }

    #[test]
    fn numerical_cut_splits_weak_bridge_but_not_strong_chain() {
        let (a, _) = build_case(false).unwrap();
        let diag = a.diagonal().unwrap();
        let top = AbtmTopology::from_csr32(&a).unwrap();
        let region = weighted_region(&a, &top, &diag, 84, LOCAL, STRONG_THRESHOLD).unwrap();
        assert_eq!(region, (0..LOCAL).collect::<Vec<_>>());
        assert_eq!(boundary_cuts(&a, &diag, &region), (0, 1));
        assert!(top.row(LOCAL - 1).unwrap().contains(LOCAL));
    }

    #[test]
    fn global_chain_has_no_weak_boundary_cut() {
        let (a, _) = build_case(true).unwrap();
        let diag = a.diagonal().unwrap();
        let top = AbtmTopology::from_csr32(&a).unwrap();
        let region = weighted_region(&a, &top, &diag, 147, LOCAL, STRONG_THRESHOLD).unwrap();
        assert_eq!(region.len(), LOCAL);
        assert_eq!(boundary_cuts(&a, &diag, &region), (1, 0));
    }

    #[test]
    fn known_reference_has_small_residual_and_symmetric_matrix() {
        for global in [false, true] {
            let (a, x) = build_case(global).unwrap();
            let b = a.spmv(&x).unwrap();
            assert!(true_relative(&a, &b, &x).unwrap() <= 1.0e-14);
            for i in 0..N {
                let start = a.row_ptr()[i] as usize;
                let end = a.row_ptr()[i + 1] as usize;
                for p in start..end {
                    let j = a.col_idx()[p] as usize;
                    let js = a.row_ptr()[j] as usize;
                    let je = a.row_ptr()[j + 1] as usize;
                    let off = a.col_idx()[js..je].binary_search(&(i as u32)).unwrap();
                    assert_eq!(a.values()[p].to_bits(), a.values()[js + off].to_bits());
                }
            }
        }
    }
}
