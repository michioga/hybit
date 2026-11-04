//! C1-H1: Generic FEM Matrix Market + RHS -> prepared HyBIT FGMRES baseline.
//! Only input/output plumbing; does not alter production routing.
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    GeneralSquareOptions, GeneralSquarePreconditionerPolicy, GeneralSquareRestartPolicy,
    HybitSolver, MatrixProblemClass, SolverOptions, read_matrix_market,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: PathBuf,
    xref: Option<PathBuf>,
    tol: f64,
    max_iters: usize,
    precond: GeneralSquarePreconditionerPolicy,
    restart: usize,
    max_restart: usize,
    policy: GeneralSquareRestartPolicy,
    component_diagnostics: bool,
    hotspot_diagnostics: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut xref = None;
        let mut tol = 1e-8_f64;
        let mut max_iters = 3000_usize;
        let mut precond = GeneralSquarePreconditionerPolicy::Jacobi;
        let mut restart = 30_usize;
        let mut max_restart = 70_usize;
        let mut policy = GeneralSquareRestartPolicy::Fixed;
        let mut component_diagnostics = false;
        let mut hotspot_diagnostics = false;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(required(&mut it, "--matrix")?)),
                "--rhs" => rhs = Some(PathBuf::from(required(&mut it, "--rhs")?)),
                "--xref" => xref = Some(PathBuf::from(required(&mut it, "--xref")?)),
                "--tol" => tol = required(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iters = required(&mut it, "--max-iters")?.parse()?,
                "--restart" => restart = required(&mut it, "--restart")?.parse()?,
                "--max-restart" => max_restart = required(&mut it, "--max-restart")?.parse()?,
                "--precond" => {
                    precond = match required(&mut it, "--precond")?.as_str() {
                        "jacobi" => GeneralSquarePreconditionerPolicy::Jacobi,
                        "ilu0" => GeneralSquarePreconditionerPolicy::Ilu0,
                        "ilu0-fallback" => GeneralSquarePreconditionerPolicy::Ilu0Fallback,
                        "ilu0-abtm" => GeneralSquarePreconditionerPolicy::Ilu0Abtm,
                        other => return Err(format!("unsupported --precond {other}").into()),
                    };
                }
                "--restart-policy" => {
                    policy = match required(&mut it, "--restart-policy")?.as_str() {
                        "fixed" => GeneralSquareRestartPolicy::Fixed,
                        "budget-aware" => GeneralSquareRestartPolicy::BudgetAware,
                        "escalating" => GeneralSquareRestartPolicy::Escalating,
                        other => return Err(format!("unsupported --restart-policy {other}").into()),
                    };
                }
                "--component-diagnostics" => component_diagnostics = true,
                "--hotspot-diagnostics" => hotspot_diagnostics = true,
                "-h" | "--help" => {
                    println!(
                        "Usage: fem_fgmres_bench --matrix FILE.mm --rhs FILE.rhs [--xref known_solution.txt] \\\n  [--tol 1e-8] [--max-iters 3000] [--precond jacobi|ilu0|ilu0-fallback|ilu0-abtm] \\\n  [--restart 30] [--max-restart 70] [--restart-policy fixed|budget-aware|escalating]"
                    );
                    println!("  [--component-diagnostics] (post-solve, experimental)");
                    println!("  [--hotspot-diagnostics] (post-solve, experimental)");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if !tol.is_finite() || tol <= 0.0 || max_iters == 0 || restart == 0 {
            return Err("invalid positive tolerance / iteration / restart setting".into());
        }
        if policy != GeneralSquareRestartPolicy::Fixed && max_restart < restart {
            return Err("max-restart must be >= restart".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("--matrix required")?,
            rhs: rhs.ok_or("--rhs required; explicit physical or manufactured RHS required")?,
            xref,
            tol,
            max_iters,
            precond,
            restart,
            max_restart,
            policy,
            component_diagnostics,
            hotspot_diagnostics,
        })
    }
}

fn required(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn read_vector(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let input = fs::read_to_string(path)?;
    let mut result = Vec::with_capacity(n);
    for token in input.split_whitespace() {
        // FEM Fortran E-format dumps may use E or D in vector files.
        let value: f64 = token.replace('D', "E").replace('d', "e").parse()?;
        if !value.is_finite() {
            return Err(format!("{} has a non-finite value", path.display()).into());
        }
        result.push(value);
        if result.len() > n {
            return Err(format!("{} has more than {n} values", path.display()).into());
        }
    }
    if result.len() != n {
        return Err(format!(
            "{}: expected {n} values, got {}",
            path.display(),
            result.len()
        )
        .into());
    }
    Ok(result)
}

fn stable_norm(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0_f64, f64::hypot)
}

fn relative_residual(a: &hybit::Csr32Matrix, b: &[f64], x: &[f64]) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let residual = stable_norm(b.iter().zip(ax).map(|(&bi, ai)| bi - ai));
    let bnorm = stable_norm(b.iter().copied());
    Ok(if bnorm == 0.0 {
        residual
    } else {
        residual / bnorm
    })
}

// Weak connectivity: a stored A(i,j) connects i and j even if A(j,i) is absent.
// This is only a structural diagnostic; it never asserts SPD or symmetry.
struct ComponentUnionFind {
    parent: Vec<usize>,
    sizes: Vec<usize>,
}

impl ComponentUnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            sizes: vec![1; n],
        }
    }

    fn root(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }

    fn unite(&mut self, a: usize, b: usize) {
        let mut a = self.root(a);
        let mut b = self.root(b);
        if a == b {
            return;
        }
        if self.sizes[a] < self.sizes[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        self.sizes[a] += self.sizes[b];
    }
}

#[derive(Default)]
struct ComponentMetrics {
    first_row: usize,
    rows: usize,
    nnz: usize,
    rhs_norm: f64,
    residual_norm: f64,
}

struct ComponentReport {
    parts: Vec<ComponentMetrics>,
    row_to_part: Vec<usize>,
    residuals: Vec<f64>,
    rhs_norm: f64,
    residual_norm: f64,
}

fn component_report(
    matrix: &hybit::Csr32Matrix,
    rhs: &[f64],
    x: &[f64],
) -> Result<ComponentReport, Box<dyn Error>> {
    let n = matrix.nrows();
    if n != matrix.ncols() || rhs.len() != n || x.len() != n {
        return Err("component diagnostics require a square matrix and matching vectors".into());
    }
    let mut uf = ComponentUnionFind::new(n);
    for row in 0..n {
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        for &col in &matrix.col_idx()[start..end] {
            uf.unite(row, col as usize);
        }
    }

    let mut root_to_part = vec![usize::MAX; n];
    let mut row_to_part = Vec::with_capacity(n);
    let mut parts = Vec::<ComponentMetrics>::new();
    for row in 0..n {
        let root = uf.root(row);
        if root_to_part[root] == usize::MAX {
            root_to_part[root] = parts.len();
            parts.push(ComponentMetrics {
                first_row: row,
                ..ComponentMetrics::default()
            });
        }
        let part = root_to_part[root];
        row_to_part.push(part);
        parts[part].rows += 1;
        parts[part].nnz += matrix.row_ptr()[row + 1] as usize - matrix.row_ptr()[row] as usize;
    }

    let ax = matrix.spmv(x)?;
    let (mut rhs_norm, mut residual_norm) = (0.0_f64, 0.0_f64);
    let mut residuals = Vec::with_capacity(n);
    for row in 0..n {
        let residual = rhs[row] - ax[row];
        if !residual.is_finite() {
            return Err("component diagnostics require finite true residual entries".into());
        }
        residuals.push(residual);
        let part = &mut parts[row_to_part[row]];
        part.rhs_norm = part.rhs_norm.hypot(rhs[row]);
        part.residual_norm = part.residual_norm.hypot(residual);
        rhs_norm = rhs_norm.hypot(rhs[row]);
        residual_norm = residual_norm.hypot(residual);
    }
    Ok(ComponentReport {
        parts,
        row_to_part,
        residuals,
        rhs_norm,
        residual_norm,
    })
}

// This is a diagnostic ranking, not a candidate local-direct region.
// A top-k set does not imply graph connectivity or a numerical improvement.
const HOTSPOT_CAPACITIES: [usize; 5] = [32, 64, 128, 256, 512];

struct HotspotMetric {
    // Sorted by descending magnitude, then ascending 0-based row index.
    rows: Vec<(usize, f64)>,
    norm: f64,
}

impl HotspotMetric {
    fn new(mut rows: Vec<(usize, f64)>) -> Self {
        rows.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let norm = stable_norm(rows.iter().map(|&(_, magnitude)| magnitude));
        Self { rows, norm }
    }

    fn energy_fraction(&self, k: usize) -> f64 {
        if self.norm == 0.0 {
            return 0.0;
        }
        // Squaring the ratio of stable norms avoids overflow from r_i * r_i.
        let top_norm = stable_norm(self.rows.iter().take(k).map(|&(_, value)| value));
        let fraction = top_norm / self.norm;
        (fraction * fraction).min(1.0)
    }
}

struct HotspotReport {
    first_row: usize,
    component_rows: usize,
    component_norm_fraction: f64,
    raw: HotspotMetric,
    row_scaled: Option<HotspotMetric>,
    row_scaled_unavailable: Option<&'static str>,
}

fn hotspot_report(
    matrix: &hybit::Csr32Matrix,
    components: &ComponentReport,
) -> Result<HotspotReport, Box<dyn Error>> {
    let (dominant, part) = components
        .parts
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.residual_norm
                .total_cmp(&b.residual_norm)
                .then(a.rows.cmp(&b.rows))
                .then(b.first_row.cmp(&a.first_row))
        })
        .ok_or("hotspot diagnostics require at least one matrix row")?;

    let mut raw_rows = Vec::with_capacity(part.rows);
    let mut scaled_rows = Vec::with_capacity(part.rows);
    let mut scaled_unavailable = None;
    for (row, &part_id) in components.row_to_part.iter().enumerate() {
        if part_id != dominant {
            continue;
        }
        let magnitude = components.residuals[row].abs();
        if !magnitude.is_finite() {
            return Err("non-finite original-CSR hotspot residual".into());
        }
        raw_rows.push((row, magnitude));

        // Maximum coefficient magnitude is a scale heuristic, NOT an SPD
        // normalization or an independent solver convergence certificate.
        let start = matrix.row_ptr()[row] as usize;
        let end = matrix.row_ptr()[row + 1] as usize;
        let row_max = matrix.values()[start..end]
            .iter()
            .fold(0.0_f64, |maximum, &value| maximum.max(value.abs()));
        let scaled = if row_max > 0.0 && row_max.is_finite() {
            magnitude / row_max
        } else if row_max == 0.0 && magnitude == 0.0 {
            0.0
        } else {
            scaled_unavailable = Some("ZERO_OR_NONFINITE_ROW_SCALE");
            continue;
        };
        if !scaled.is_finite() {
            scaled_unavailable = Some("NONFINITE_SCALED_RESIDUAL");
        } else {
            scaled_rows.push((row, scaled));
        }
    }

    let row_scaled = if scaled_unavailable.is_none() {
        Some(HotspotMetric::new(scaled_rows))
    } else {
        None
    };
    Ok(HotspotReport {
        first_row: part.first_row,
        component_rows: part.rows,
        component_norm_fraction: if components.residual_norm == 0.0 {
            0.0
        } else {
            part.residual_norm / components.residual_norm
        },
        raw: HotspotMetric::new(raw_rows),
        row_scaled,
        row_scaled_unavailable: scaled_unavailable,
    })
}

fn print_hotspot_metric(name: &str, metric: &HotspotMetric) {
    for k in HOTSPOT_CAPACITIES {
        println!(
            "C1H14_TOPK|metric={name}|k={k}|selected_rows={}|component_residual_energy_fraction={:.12e}",
            k.min(metric.rows.len()),
            metric.energy_fraction(k)
        );
    }
    for (rank, &(row, magnitude)) in metric.rows.iter().take(5).enumerate() {
        println!(
            "C1H14_ROW|metric={name}|rank={}|row_1based={}|magnitude={magnitude:.12e}",
            rank + 1,
            row + 1
        );
    }
}

fn print_hotspot_report(report: &HotspotReport) {
    println!(
        "C1H14_SUMMARY|scope=dominant_weak_component|first_row_1based={}|rows={}|component_residual_norm_fraction={:.12e}|raw_residual_norm={:.12e}",
        report.first_row + 1,
        report.component_rows,
        report.component_norm_fraction,
        report.raw.norm
    );
    print_hotspot_metric("raw", &report.raw);
    if let Some(ref metric) = report.row_scaled {
        print_hotspot_metric("row_max_scaled", metric);
    } else {
        println!(
            "C1H14_SCALED|status=UNAVAILABLE|reason={}",
            report.row_scaled_unavailable.unwrap_or("UNKNOWN")
        );
    }
}

fn print_component_report(report: &ComponentReport) {
    let parts = &report.parts;
    let singles = parts.iter().filter(|p| p.rows == 1).count();
    let largest = parts.iter().map(|p| p.rows).max().unwrap_or(0);
    let active_rhs = parts.iter().filter(|p| p.rhs_norm > 0.0).count();
    let relative = if report.rhs_norm == 0.0 {
        report.residual_norm
    } else {
        report.residual_norm / report.rhs_norm
    };
    println!(
        "C1H13_SUMMARY|components={}|singletons={singles}|largest_rows={largest}|active_rhs_components={active_rhs}|rhs_norm={:.12e}|residual_norm={:.12e}|verified_relative={relative:.12e}",
        parts.len(),
        report.rhs_norm,
        report.residual_norm
    );

    // Print both largest residuals and largest blocks; a block can appear twice.
    for order in ["residual", "size"] {
        let mut ids: Vec<_> = (0..parts.len()).collect();
        if order == "residual" {
            ids.sort_unstable_by(|&a, &b| {
                parts[b]
                    .residual_norm
                    .total_cmp(&parts[a].residual_norm)
                    .then(parts[a].first_row.cmp(&parts[b].first_row))
            });
        } else {
            ids.sort_unstable_by(|&a, &b| {
                parts[b]
                    .rows
                    .cmp(&parts[a].rows)
                    .then(parts[a].first_row.cmp(&parts[b].first_row))
            });
        }
        for (rank, id) in ids.into_iter().take(10).enumerate() {
            let p = &parts[id];
            let rhs_fraction = if report.rhs_norm == 0.0 {
                0.0
            } else {
                p.rhs_norm / report.rhs_norm
            };
            let residual_fraction = if report.residual_norm == 0.0 {
                0.0
            } else {
                p.residual_norm / report.residual_norm
            };
            let part_relative = if p.rhs_norm == 0.0 {
                "NA_ZERO_RHS".to_string()
            } else {
                format!("{:.12e}", p.residual_norm / p.rhs_norm)
            };
            println!(
                "C1H13_COMPONENT|order={order}|rank={}|first_row_1based={}|rows={}|nnz={}|rhs_norm_fraction={rhs_fraction:.12e}|residual_norm_fraction={residual_fraction:.12e}|component_relative={part_relative}",
                rank + 1,
                p.first_row + 1,
                p.rows,
                p.nnz
            );
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let load = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    if matrix.nrows() != matrix.ncols() {
        return Err(
            "this example expects a complete square matrix; do not pass unassembled MPI rank dumps"
                .into(),
        );
    }
    let b = read_vector(&args.rhs, matrix.nrows())?;
    let reference = args
        .xref
        .as_ref()
        .map(|path| read_vector(path, matrix.ncols()))
        .transpose()?;
    let load_time = load.elapsed().as_secs_f64();
    let reference_residual = reference
        .as_ref()
        .map(|x| relative_residual(&matrix, &b, x))
        .transpose()?;

    let mut solver = HybitSolver::new();
    solver.set_problem_class(MatrixProblemClass::GeneralSquare);
    solver.set_options(SolverOptions {
        relative_tolerance: args.tol,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iters,
    })?;
    solver.set_general_square_preconditioner_policy(args.precond);
    solver.set_general_square_options(GeneralSquareOptions {
        restart: args.restart,
        restart_policy: args.policy,
        max_restart: args.max_restart,
        escalation_stage_iterations: args.max_iters,
    })?;
    let mut x = vec![0.0; matrix.ncols()];
    let start = Instant::now();
    let report = solver.solve_csr32(&matrix, &b, &mut x)?;
    let wall = start.elapsed().as_secs_f64();
    let verified = relative_residual(&matrix, &b, &x)?;
    if !verified.is_finite() {
        return Err("non-finite verified residual".into());
    }
    println!("HyBIT 0.10.0 FEM FGMRES benchmark (experimental)");
    println!("matrix             : {}", args.matrix.display());
    println!("RHS                : {}", args.rhs.display());
    println!(
        "dimensions         : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("input/csr nnz      : {} / {}", mm.input_entries, mm.csr_nnz);
    println!("matrix class       : GeneralSquare (FGMRES; no SPD assertion)");
    println!("solver             : {:?}", report.solver);
    println!("preconditioner     : {:?}", report.preconditioner);
    println!(
        "restart policy     : {:?}, initial {}, max {}",
        args.policy, args.restart, args.max_restart
    );
    println!("status             : {:?}", report.status);
    println!("iterations         : {}", report.iterations);
    println!("reported residual  : {:.6e}", report.relative_residual);
    println!("verified residual  : {:.6e}", verified);
    if let Some(rr) = reference_residual {
        println!("reference residual : {:.6e}", rr);
    }
    if let Some(ref xref) = reference {
        let err = stable_norm(x.iter().zip(xref).map(|(&a, &r)| a - r));
        let denom = stable_norm(xref.iter().copied());
        println!(
            "relative x error   : {:.6e}",
            if denom == 0.0 { err } else { err / denom }
        );
    }
    println!("load time          : {:.3} ms", load_time * 1e3);
    println!(
        "analysis           : {:.3} ms",
        report.analysis_seconds * 1e3
    );
    println!(
        "prepare            : {:.3} ms",
        report.prepare_seconds * 1e3
    );
    println!("solver time        : {:.3} ms", report.solve_seconds * 1e3);
    println!("total wall         : {:.3} ms", wall * 1e3);
    if args.component_diagnostics || args.hotspot_diagnostics {
        // Intentionally timed after the existing solve wall-clock metric.
        let diagnostics_start = Instant::now();
        let components = component_report(&matrix, &b, &x)?;
        if args.component_diagnostics {
            print_component_report(&components);
            println!(
                "C1H13_DIAGNOSTICS|elapsed_ms={:.3}",
                diagnostics_start.elapsed().as_secs_f64() * 1e3
            );
        }
        if args.hotspot_diagnostics {
            print_hotspot_report(&hotspot_report(&matrix, &components)?);
            println!(
                "C1H14_DIAGNOSTICS|elapsed_ms={:.3}",
                diagnostics_start.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    // Status is semantic; unlike process errors, MaxIterations is returned as a report.
    // Supervising scripts MUST check status AND verified residual.
    Ok(())
}

#[cfg(test)]
mod fem_fgmres_bench_tests {
    use super::{component_report, hotspot_report, stable_norm};
    use hybit::Csr32Matrix;
    #[test]
    fn stable_norm_handles_large_and_small_values() {
        assert!((stable_norm([3.0, 4.0]) - 5.0).abs() < 1e-14);
        assert!(stable_norm([1e200, 1e-200]).is_finite());
    }

    #[test]
    fn weak_components_partition_directed_edges_and_residuals() {
        // Directed 0->1 and 2->3 edges; row 4 remains a singleton.
        let a = Csr32Matrix::new(
            5,
            5,
            vec![0, 2, 3, 5, 6, 7],
            vec![0, 1, 1, 2, 3, 3, 4],
            vec![1.0; 7],
        )
        .unwrap();
        let b = [3.0, 4.0, 0.0, 0.0, 12.0];
        let report = component_report(&a, &b, &[0.0; 5]).unwrap();
        assert_eq!(report.parts.len(), 3);
        assert_eq!(
            report.parts.iter().map(|p| p.rows).collect::<Vec<_>>(),
            [2, 2, 1]
        );
        assert_eq!(
            report.parts.iter().map(|p| p.nnz).collect::<Vec<_>>(),
            [3, 3, 1]
        );
        assert!((report.rhs_norm - 13.0).abs() < 1e-14);
        assert!((report.residual_norm - 13.0).abs() < 1e-14);
        assert!((report.parts[0].residual_norm - 5.0).abs() < 1e-14);
        assert_eq!(report.parts[1].residual_norm, 0.0);
        assert_eq!(report.parts[2].residual_norm, 12.0);
    }

    #[test]
    fn component_report_handles_zero_rhs_and_exact_solution() {
        let a = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 4.0]).unwrap();
        let empty = component_report(&a, &[0.0, 0.0], &[0.0, 0.0]).unwrap();
        assert_eq!(empty.parts.len(), 2);
        assert_eq!(empty.rhs_norm, 0.0);
        assert_eq!(empty.residual_norm, 0.0);
        assert_eq!(
            component_report(&a, &[2.0, 4.0], &[1.0, 1.0])
                .unwrap()
                .residual_norm,
            0.0
        );
    }

    #[test]
    fn hotspots_are_component_local_and_rank_by_exact_energy() {
        // One-sided edge 0->1; singleton row 2 is excluded from the dominant
        // component even though its individual residual is nonzero.
        let a = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 3, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 1.0, 8.0, 1.0],
        )
        .unwrap();
        let components = component_report(&a, &[3.0, 4.0, 1.0], &[0.0; 3]).unwrap();
        let report = hotspot_report(&a, &components).unwrap();
        assert_eq!(report.first_row, 0);
        assert_eq!(report.component_rows, 2);
        assert_eq!(report.raw.rows[0].0, 1);
        assert!((report.raw.energy_fraction(1) - 16.0 / 25.0).abs() < 1e-14);
        assert_eq!(report.raw.energy_fraction(32), 1.0);
        let scaled = report.row_scaled.unwrap();
        assert_eq!(scaled.rows[0].0, 0);
        assert!((scaled.energy_fraction(1) - 9.0 / 9.25).abs() < 1e-14);
        assert_eq!(scaled.energy_fraction(512), 1.0);
    }

    #[test]
    fn hotspots_handle_zero_residuals_without_fabricated_energy() {
        let a = Csr32Matrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 4.0]).unwrap();
        let components = component_report(&a, &[2.0, 4.0], &[1.0, 1.0]).unwrap();
        let report = hotspot_report(&a, &components).unwrap();
        assert_eq!(report.first_row, 0);
        assert_eq!(report.component_norm_fraction, 0.0);
        assert_eq!(report.raw.energy_fraction(32), 0.0);
        assert_eq!(report.row_scaled.unwrap().energy_fraction(32), 0.0);
    }

    #[test]
    fn hotspots_handle_extreme_magnitudes_without_squared_overflow() {
        let a = Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let components = component_report(&a, &[1e200, 1e-200], &[0.0, 0.0]).unwrap();
        let report = hotspot_report(&a, &components).unwrap();
        assert!(report.raw.norm.is_finite());
        assert_eq!(report.raw.energy_fraction(1), 1.0);
    }

    #[test]
    fn nonzero_residual_on_zero_row_disables_scaled_metric_only() {
        let a = Csr32Matrix::new(1, 1, vec![0, 0], vec![], vec![]).unwrap();
        let components = component_report(&a, &[1.0], &[0.0]).unwrap();
        let report = hotspot_report(&a, &components).unwrap();
        assert_eq!(report.raw.energy_fraction(32), 1.0);
        assert_eq!(
            report.row_scaled_unavailable,
            Some("ZERO_OR_NONFINITE_ROW_SCALE")
        );
        assert!(report.row_scaled.is_none());
    }
}
