//! C1-H2: observation-only ABTM residual-topology probe for SPD FEM systems.
//!
//! No automatic escalation, no solver state mutation by the observer.
use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use hybit::{
    AbtmTopology, Csr32Matrix, DofMask, JacobiPreconditioner, SolveStatus, SolverOptions, pcg,
    read_matrix_market,
};
use hybit_krylov::{PcgWorkspace, pcg_continue_with_workspace, pcg_start_with_workspace};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: PathBuf,
    xref: Option<PathBuf>,
    tol: f64,
    max_iters: usize,
    first_check: usize,
    monitor_iters: usize,
    seed_fraction: f64,
}

fn value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut xref = None;
        let mut tol: f64 = 1e-10;
        let mut max_iters = 1200;
        let mut first_check = 12;
        let mut monitor_iters = 64;
        let mut seed_fraction: f64 = 0.2;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(value(&mut it, "--matrix")?)),
                "--rhs" => rhs = Some(PathBuf::from(value(&mut it, "--rhs")?)),
                "--xref" => xref = Some(PathBuf::from(value(&mut it, "--xref")?)),
                "--tol" => tol = value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iters = value(&mut it, "--max-iters")?.parse()?,
                "--first-check" => first_check = value(&mut it, "--first-check")?.parse()?,
                "--monitor-iters" => monitor_iters = value(&mut it, "--monitor-iters")?.parse()?,
                "--seed-fraction" => seed_fraction = value(&mut it, "--seed-fraction")?.parse()?,
                "--help" | "-h" => {
                    println!(
                        "Usage: fem_abtm_residual_observer --matrix FILE.mtx --rhs FILE.rhs [--xref FILE] [--tol 1e-10] [--max-iters 1200] [--first-check 12] [--monitor-iters 64] [--seed-fraction 0.2]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unrecognized argument {arg}").into()),
            }
        }
        if !tol.is_finite() || tol <= 0.0 {
            return Err("--tol must be finite and positive".into());
        }
        if max_iters == 0 || first_check == 0 || monitor_iters == 0 {
            return Err("iteration counts must be positive".into());
        }
        if !seed_fraction.is_finite() || !(0.0 < seed_fraction && seed_fraction <= 1.0) {
            return Err("--seed-fraction must be in (0,1]".into());
        }
        Ok(Self {
            matrix: matrix.ok_or("--matrix is required")?,
            rhs: rhs.ok_or("--rhs is required")?,
            xref,
            tol,
            max_iters,
            first_check,
            monitor_iters,
            seed_fraction,
        })
    }
}

fn read_vector(path: &PathBuf, expected: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let mut numbers = Vec::new();
    for token in contents.split_whitespace() {
        let parsed: f64 = token.parse()?;
        if !parsed.is_finite() {
            return Err(format!("{} has non-finite vector entry", path.display()).into());
        }
        numbers.push(parsed);
    }
    if numbers.len() != expected {
        return Err(format!(
            "{}: expected {} entries, found {}",
            path.display(),
            expected,
            numbers.len()
        )
        .into());
    }
    Ok(numbers)
}

fn norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |n, &v| n.hypot(v))
}

fn true_residual(a: &Csr32Matrix, b: &[f64], x: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let r = b
        .iter()
        .zip(ax)
        .map(|(&bi, ai)| bi - ai)
        .collect::<Vec<_>>();
    if r.iter().any(|x| !x.is_finite()) {
        return Err("non-finite true residual".into());
    }
    Ok(r)
}

fn bit_intersection_size(a: &DofMask, b: &DofMask) -> usize {
    a.words()
        .iter()
        .zip(b.words())
        .map(|(&x, &y)| (x & y).count_ones() as usize)
        .sum()
}

fn jaccard(a: &DofMask, b: &DofMask) -> f64 {
    let both = bit_intersection_size(a, b);
    let union = a.count_ones() + b.count_ones() - both;
    if union == 0 {
        1.0
    } else {
        both as f64 / union as f64
    }
}

fn triple_intersection_size(a: &DofMask, b: &DofMask, c: &DofMask) -> usize {
    a.words()
        .iter()
        .zip(b.words())
        .zip(c.words())
        .map(|((&x, &y), &z)| (x & y & z).count_ones() as usize)
        .sum()
}

fn topology_stats(
    topology: &AbtmTopology,
    seeds: &DofMask,
) -> Result<(usize, usize, usize), Box<dyn Error>> {
    // One-hop expansion is a *word-wise OR*, using ABTM topology rather than
    // reading numerical CSR values or modifying the solver's Krylov recurrence.
    let mut grown = seeds.words().to_vec();
    for row in seeds.indices() {
        for word in topology.row(row)?.words() {
            grown[word.word_index() as usize] |= word.mask();
        }
    }
    let frontier = grown
        .iter()
        .zip(seeds.words())
        .map(|(&grown, &selected)| (grown & !selected).count_ones() as usize)
        .sum();

    let mut visited = DofMask::new(seeds.len());
    let mut components = 0;
    let mut largest_component = 0;
    for root in seeds.indices() {
        if visited.contains(root) {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::from([root]);
        visited.set(root, true)?;
        let mut size = 0;
        while let Some(row) = queue.pop_front() {
            size += 1;
            for word in topology.row(row)?.words() {
                let mut bits = word.mask();
                let base = word.base_col();
                while bits != 0 {
                    let neighbor = base + bits.trailing_zeros() as usize;
                    if seeds.contains(neighbor) && !visited.contains(neighbor) {
                        visited.set(neighbor, true)?;
                        queue.push_back(neighbor);
                    }
                    bits &= bits - 1;
                }
            }
        }
        largest_component = largest_component.max(size);
    }
    Ok((components, largest_component, frontier))
}

struct Snapshot {
    mask: DofMask,
    relative_residual: f64,
    selected_energy_fraction: f64,
    components: usize,
    largest_component: usize,
    frontier: usize,
}

fn snapshot(
    a: &Csr32Matrix,
    topology: &AbtmTopology,
    diagonal: &[f64],
    b: &[f64],
    x: &[f64],
    seed_fraction: f64,
) -> Result<Snapshot, Box<dyn Error>> {
    let r = true_residual(a, b, x)?;
    let relative_residual = norm(&r) / norm(b).max(f64::MIN_POSITIVE);
    // Jacobi-scaled residual is only a hotspot-ranking proxy. It does not
    // estimate global forward error or certify a low-energy eigenmode.
    let scaled = r
        .iter()
        .zip(diagonal)
        .map(|(&ri, &di)| ri.abs() / di.sqrt())
        .collect::<Vec<_>>();
    if scaled.iter().any(|v| !v.is_finite()) {
        return Err("non-finite Jacobi-scaled residual".into());
    }
    let peak = scaled.iter().copied().fold(0.0_f64, f64::max);
    let mut mask = DofMask::new(r.len());
    if peak > 0.0 {
        for (i, &value) in scaled.iter().enumerate() {
            if value >= seed_fraction * peak {
                mask.set(i, true)?;
            }
        }
    }
    let selected_norm = norm(
        &scaled
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| mask.contains(i).then_some(v))
            .collect::<Vec<_>>(),
    );
    let total_norm = norm(&scaled);
    let selected_energy_fraction = if total_norm > 0.0 {
        (selected_norm / total_norm).powi(2)
    } else {
        0.0
    };
    let (components, largest_component, frontier) = topology_stats(topology, &mask)?;
    Ok(Snapshot {
        mask,
        relative_residual,
        selected_energy_fraction,
        components,
        largest_component,
        frontier,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let (a, _) = read_matrix_market(&args.matrix)?;
    if a.nrows() != a.ncols() {
        return Err("PCG observer requires square SPD operator".into());
    }
    let b = read_vector(&args.rhs, a.nrows())?;
    let reference = args
        .xref
        .as_ref()
        .map(|p| read_vector(p, a.ncols()))
        .transpose()?;
    let diagonal = a.diagonal()?;
    if diagonal.iter().any(|&d| !d.is_finite() || d <= 0.0) {
        return Err("SPD observer requires finite strictly positive diagonal".into());
    }
    let topology_start = Instant::now();
    let topology = AbtmTopology::from_csr32(&a)?;
    let topology_seconds = topology_start.elapsed().as_secs_f64();
    let jacobi = JacobiPreconditioner::from_csr32(&a)?;
    let options = SolverOptions {
        relative_tolerance: args.tol,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iters,
    };

    println!("C1-H2 ABTM residual topology observer (observation-only)");
    println!("matrix             : {}", args.matrix.display());
    println!("dimensions         : {}", a.nrows());
    println!("topology words     : {}", topology.nonempty_words());
    println!("topology setup     : {:.3} ms", topology_seconds * 1000.0);
    println!("seed fraction      : {:.5}", args.seed_fraction);
    println!(
        "monitor checkpoints: first={}, interval={}",
        args.first_check, args.monitor_iters
    );
    println!(
        "iteration true_residual active energy_fraction jaccard_prev persist2 persist3 components max_component frontier"
    );

    let mut x = vec![0.0; a.ncols()];
    let mut workspace = PcgWorkspace::new(a.nrows());
    let start = Instant::now();
    let mut session = pcg_start_with_workspace(&a, &jacobi, &b, &mut x, options, &mut workspace)?;
    let mut total = 0;
    let mut status = SolveStatus::MaxIterations;
    let mut history = VecDeque::<DofMask>::new();
    let mut observe_time = Duration::ZERO;
    while total < args.max_iters {
        let step_limit = if total == 0 {
            args.first_check.min(args.max_iters)
        } else {
            args.monitor_iters.min(args.max_iters - total)
        };
        let step = pcg_continue_with_workspace(
            &a,
            &jacobi,
            &b,
            &mut x,
            step_limit,
            &mut session,
            &mut workspace,
        )?;
        total += step.iterations;
        status = step.status;
        let observation_start = Instant::now();
        let sample = snapshot(&a, &topology, &diagonal, &b, &x, args.seed_fraction)?;
        let jaccard_previous = history
            .back()
            .map(|previous| jaccard(&sample.mask, previous));
        let persistent2 = history
            .back()
            .map(|previous| bit_intersection_size(&sample.mask, previous))
            .unwrap_or(0);
        let persistent3 = if history.len() == 2 {
            triple_intersection_size(&sample.mask, &history[0], &history[1])
        } else {
            0
        };
        observe_time += observation_start.elapsed();
        println!(
            "{} {:.9e} {} {:.6} {} {} {} {} {} {}",
            total,
            sample.relative_residual,
            sample.mask.count_ones(),
            sample.selected_energy_fraction,
            jaccard_previous.map_or_else(|| "-".to_owned(), |v| format!("{v:.6}")),
            persistent2,
            persistent3,
            sample.components,
            sample.largest_component,
            sample.frontier,
        );
        if history.len() == 2 {
            history.pop_front();
        }
        history.push_back(sample.mask);
        if status != SolveStatus::MaxIterations || step.iterations == 0 {
            break;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let observer_verified = norm(&true_residual(&a, &b, &x)?) / norm(&b).max(f64::MIN_POSITIVE);

    // Identical operator, initial guess, preconditioner and iteration budget.
    // Checking both the Krylov outcome and complete final iterate protects
    // against observers inadvertently restarting PCG at checkpoints.
    let mut x_plain = vec![0.0; a.ncols()];
    let plain_start = Instant::now();
    let plain = pcg(&a, &jacobi, &b, &mut x_plain, options)?;
    let plain_seconds = plain_start.elapsed().as_secs_f64();
    let plain_verified = norm(&true_residual(&a, &b, &x_plain)?) / norm(&b).max(f64::MIN_POSITIVE);
    let bitwise_equal = x
        .iter()
        .zip(&x_plain)
        .all(|(&l, &r)| l.to_bits() == r.to_bits());
    let invariant = bitwise_equal && status == plain.status && total == plain.iterations;
    println!("observer status    : {:?}", status);
    println!("observer iterations: {}", total);
    println!("observer residual  : {:.9e}", observer_verified);
    println!("observer total     : {:.3} ms", elapsed * 1000.0);
    println!(
        "observer diagnostics: {:.3} ms",
        observe_time.as_secs_f64() * 1000.0
    );
    println!("baseline status    : {:?}", plain.status);
    println!("baseline iterations: {}", plain.iterations);
    println!("baseline residual  : {:.9e}", plain_verified);
    println!("baseline solve     : {:.3} ms", plain_seconds * 1000.0);
    println!("bitwise same x     : {}", bitwise_equal);
    println!(
        "PCG invariant      : {}",
        if invariant { "PASS" } else { "FAIL" }
    );
    if let Some(reference) = reference.as_deref() {
        let error = norm(
            &x.iter()
                .zip(reference)
                .map(|(&actual, &expected)| actual - expected)
                .collect::<Vec<_>>(),
        ) / norm(reference).max(f64::MIN_POSITIVE);
        println!("relative x error   : {:.9e}", error);
        let reference_residual =
            norm(&true_residual(&a, &b, reference)?) / norm(&b).max(f64::MIN_POSITIVE);
        println!("reference residual : {:.9e}", reference_residual);
    }
    println!("classification     : observation only; no rescue claim");
    if !invariant {
        return Err("observational PCG differs from uninterrupted PCG".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_overlap_distinguishes_migration_and_persistence() {
        let first = DofMask::from_indices(130, &[0, 64, 129]).unwrap();
        let second = DofMask::from_indices(130, &[64, 65, 129]).unwrap();
        let third = DofMask::from_indices(130, &[64, 129]).unwrap();
        assert_eq!(bit_intersection_size(&first, &second), 2);
        assert!((jaccard(&first, &second) - 0.5).abs() < 1e-14);
        assert_eq!(triple_intersection_size(&first, &second, &third), 2);
    }

    #[test]
    fn abtm_topology_expansion_and_components() {
        // 0 -- 1 -- 2, with an isolated third degree of freedom.
        let matrix = Csr32Matrix::new(
            4,
            4,
            vec![0, 2, 5, 7, 8],
            vec![0, 1, 0, 1, 2, 1, 2, 3],
            vec![2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0, 1.0],
        )
        .unwrap();
        let topology = AbtmTopology::from_csr32(&matrix).unwrap();
        let seeds = DofMask::from_indices(4, &[0, 2, 3]).unwrap();
        let (components, largest, frontier) = topology_stats(&topology, &seeds).unwrap();
        assert_eq!((components, largest, frontier), (3, 1, 1));
        let connected = DofMask::from_indices(4, &[0, 1, 2]).unwrap();
        let (components, largest, frontier) = topology_stats(&topology, &connected).unwrap();
        assert_eq!((components, largest, frontier), (1, 3, 0));
    }

    #[test]
    fn observation_checkpoints_preserve_pcg_iterate() {
        let matrix = Csr32Matrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0],
        )
        .unwrap();
        let reference = vec![1.0, -0.5, 2.0];
        let b = matrix.spmv(&reference).unwrap();
        let diagonal = matrix.diagonal().unwrap();
        let topology = AbtmTopology::from_csr32(&matrix).unwrap();
        let jacobi = JacobiPreconditioner::from_csr32(&matrix).unwrap();
        let options = SolverOptions {
            relative_tolerance: 1e-12,
            absolute_tolerance: 0.0,
            max_iterations: 20,
        };
        let mut x_plain = vec![0.0; 3];
        let plain = pcg(&matrix, &jacobi, &b, &mut x_plain, options).unwrap();
        let mut x_segmented = vec![0.0; 3];
        let mut workspace = PcgWorkspace::new(3);
        let mut session = pcg_start_with_workspace(
            &matrix,
            &jacobi,
            &b,
            &mut x_segmented,
            options,
            &mut workspace,
        )
        .unwrap();
        let mut count = 0;
        let status = loop {
            let step = pcg_continue_with_workspace(
                &matrix,
                &jacobi,
                &b,
                &mut x_segmented,
                1,
                &mut session,
                &mut workspace,
            )
            .unwrap();
            count += step.iterations;
            let _ = snapshot(&matrix, &topology, &diagonal, &b, &x_segmented, 0.2).unwrap();
            if step.status != SolveStatus::MaxIterations || count == 20 {
                break step.status;
            }
        };
        assert_eq!(status, plain.status);
        assert_eq!(count, plain.iterations);
        assert!(
            x_segmented
                .iter()
                .zip(x_plain.iter())
                .all(|(&a, &b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn snapshot_records_jacobi_scaled_residual_energy() {
        let matrix = Csr32Matrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, -1.0, -1.0, 2.0],
        )
        .unwrap();
        let topology = AbtmTopology::from_csr32(&matrix).unwrap();
        let sample = snapshot(
            &matrix,
            &topology,
            &[2.0, 2.0],
            &[2.0, 0.0],
            &[0.0, 0.0],
            0.2,
        )
        .unwrap();
        assert_eq!(sample.mask.indices(), vec![0]);
        assert_eq!(sample.components, 1);
        assert_eq!(sample.frontier, 1);
        assert!((sample.selected_energy_fraction - 1.0).abs() < 1e-14);
    }
}
