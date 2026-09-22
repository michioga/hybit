use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hybit::{
    analyze_csr32, pcg, read_matrix_market, AlgebraicCoarseOptions, BackendPolicy, Csr32Matrix,
    HybitSolver, JacobiPreconditioner, LocalFactorSelectionPolicy, MatrixBackend, SolverOptions,
    TwoLevelAggregation, TwoLevelBasis, TwoLevelCoarseApplyPolicy, TwoLevelTransferApplyPolicy,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
    rhs: Option<PathBuf>,
    relative_tolerance: f64,
    max_iterations: usize,
    overlap_layers: usize,
    probe_iterations: usize,
    max_local_region_size: usize,
    max_local_regions: usize,
    max_local_factor_bytes: usize,
    max_escalations: usize,
    escalation_stage_iterations: usize,
    local_factor_selection: LocalFactorSelectionPolicy,
    algebraic_coarse_enabled: bool,
    algebraic_coarse_dofs_per_node: usize,
    algebraic_coarse_target_dimension: usize,
    algebraic_coarse_aggregation: TwoLevelAggregation,
    algebraic_coarse_basis: TwoLevelBasis,
    algebraic_coarse_transfer_apply_policy: TwoLevelTransferApplyPolicy,
    algebraic_coarse_apply_policy: TwoLevelCoarseApplyPolicy,
    backend: BackendPolicy,
    skip_plain: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut rhs = None;
        let mut relative_tolerance: f64 = 1.0e-8;
        let mut max_iterations = 1000usize;
        let mut overlap_layers = 1usize;
        let mut probe_iterations = 12usize;
        let mut max_local_region_size = 128usize;
        let mut max_local_regions = 8usize;
        let mut max_local_factor_bytes = 64usize * 1024 * 1024;
        let mut max_escalations = 3usize;
        let mut escalation_stage_iterations = 24usize;
        let mut local_factor_selection = LocalFactorSelectionPolicy::JacobiEnergyPerByte;
        let mut algebraic_coarse_enabled = false;
        let mut algebraic_coarse_dofs_per_node = 3usize;
        let mut algebraic_coarse_target_dimension = 1536usize;
        let mut algebraic_coarse_aggregation = TwoLevelAggregation::Contiguous;
        let mut algebraic_coarse_basis = TwoLevelBasis::PiecewiseConstant;
        let mut algebraic_coarse_transfer_apply_policy = TwoLevelTransferApplyPolicy::Serial;
        let mut algebraic_coarse_apply_policy = TwoLevelCoarseApplyPolicy::Auto;
        let mut backend = BackendPolicy::Auto;
        let mut skip_plain = false;

        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut it, "--matrix")?)),
                "--rhs" => rhs = Some(PathBuf::from(next_value(&mut it, "--rhs")?)),
                "--tol" => relative_tolerance = next_value(&mut it, "--tol")?.parse()?,
                "--max-iters" => max_iterations = next_value(&mut it, "--max-iters")?.parse()?,
                "--overlap" => overlap_layers = next_value(&mut it, "--overlap")?.parse()?,
                "--probe-iters" => {
                    probe_iterations = next_value(&mut it, "--probe-iters")?.parse()?
                }
                "--max-region" => {
                    max_local_region_size = next_value(&mut it, "--max-region")?.parse()?
                }
                "--max-regions" => {
                    max_local_regions = next_value(&mut it, "--max-regions")?.parse()?
                }
                "--factor-budget-mib" => {
                    let mib: usize = next_value(&mut it, "--factor-budget-mib")?.parse()?;
                    max_local_factor_bytes = mib
                        .checked_mul(1024 * 1024)
                        .ok_or("--factor-budget-mib is too large")?;
                }
                "--max-escalations" => {
                    max_escalations = next_value(&mut it, "--max-escalations")?.parse()?
                }
                "--stage-iters" => {
                    escalation_stage_iterations = next_value(&mut it, "--stage-iters")?.parse()?
                }
                "--selector" => {
                    let value = next_value(&mut it, "--selector")?;
                    local_factor_selection = match value.as_str() {
                        "candidate" | "candidate-order" | "legacy" => {
                            LocalFactorSelectionPolicy::CandidateOrder
                        }
                        "benefit-byte" | "benefit-per-byte" | "residual-byte" => {
                            LocalFactorSelectionPolicy::BenefitPerByte
                        }
                        "jacobi-byte" | "jacobi-energy-byte" | "auto" => {
                            LocalFactorSelectionPolicy::JacobiEnergyPerByte
                        }
                        other => {
                            return Err(format!(
                                "unknown selector '{other}'; use candidate-order|benefit-byte|jacobi-byte"
                            )
                            .into())
                        }
                    };
                }
                "--hybrid-coarse" => algebraic_coarse_enabled = true,
                "--coarse-dofs" => {
                    algebraic_coarse_dofs_per_node =
                        next_value(&mut it, "--coarse-dofs")?.parse()?
                }
                "--coarse-target" => {
                    algebraic_coarse_target_dimension =
                        next_value(&mut it, "--coarse-target")?.parse()?
                }
                "--coarse-aggregation" => {
                    let value = next_value(&mut it, "--coarse-aggregation")?;
                    algebraic_coarse_aggregation = match value.as_str() {
                        "contiguous" | "linear" => TwoLevelAggregation::Contiguous,
                        "graph" | "graph-bfs" => TwoLevelAggregation::Graph,
                        "strong-graph" | "graph-strong" | "strength" => {
                            TwoLevelAggregation::StrongGraph
                        }
                        other => {
                            return Err(format!(
                                "unknown coarse aggregation '{other}'; use contiguous|graph|strong-graph"
                            )
                            .into())
                        }
                    };
                }
                "--coarse-basis" => {
                    let value = next_value(&mut it, "--coarse-basis")?;
                    algebraic_coarse_basis = match value.as_str() {
                        "piecewise" | "piecewise-constant" | "tentative" => {
                            TwoLevelBasis::PiecewiseConstant
                        }
                        "smoothed" | "jacobi-smoothed" | "sa" => {
                            TwoLevelBasis::JacobiSmoothed
                        }
                        other => {
                            return Err(format!(
                                "unknown coarse basis '{other}'; use piecewise|smoothed"
                            )
                            .into())
                        }
                    };
                }
                "--coarse-transfer" => {
                    let value = next_value(&mut it, "--coarse-transfer")?;
                    algebraic_coarse_transfer_apply_policy = match value.as_str() {
                        "serial" => TwoLevelTransferApplyPolicy::Serial,
                        "parallel" | "rayon" => TwoLevelTransferApplyPolicy::Parallel,
                        other => {
                            return Err(format!(
                                "unknown coarse transfer policy '{other}'; use serial|parallel"
                            )
                            .into())
                        }
                    };
                }
                "--coarse-apply" => {
                    let value = next_value(&mut it, "--coarse-apply")?;
                    algebraic_coarse_apply_policy = match value.as_str() {
                        "auto" => TwoLevelCoarseApplyPolicy::Auto,
                        "factor" | "factor-solve" => TwoLevelCoarseApplyPolicy::FactorSolve,
                        "inverse" | "explicit-inverse" => {
                            TwoLevelCoarseApplyPolicy::ExplicitInverse
                        }
                        other => {
                            return Err(format!(
                                "unknown coarse apply policy '{other}'; use auto|factor|inverse"
                            )
                            .into())
                        }
                    };
                }
                "--skip-plain" => skip_plain = true,
                "--backend" => {
                    backend = match next_value(&mut it, "--backend")?.as_str() {
                        "auto" => BackendPolicy::Auto,
                        "csr" | "csr32" => BackendPolicy::Csr32,
                        "abtm" => BackendPolicy::Abtm,
                        other => {
                            return Err(
                                format!("unknown backend '{other}'; use auto|csr|abtm").into()
                            )
                        }
                    }
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => {
                    matrix = Some(PathBuf::from(other))
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }

        let matrix = matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?;
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err("--tol must be finite and > 0".into());
        }
        if max_iterations == 0 {
            return Err("--max-iters must be > 0".into());
        }
        if probe_iterations == 0 {
            return Err("--probe-iters must be > 0".into());
        }
        if max_local_region_size == 0 {
            return Err("--max-region must be > 0".into());
        }
        if max_local_regions == 0 {
            return Err("--max-regions must be > 0".into());
        }
        if max_local_factor_bytes == 0 {
            return Err("--factor-budget-mib must be > 0".into());
        }
        if max_escalations == 0 {
            return Err("--max-escalations must be > 0".into());
        }
        if escalation_stage_iterations == 0 {
            return Err("--stage-iters must be > 0".into());
        }
        if algebraic_coarse_dofs_per_node == 0 {
            return Err("--coarse-dofs must be > 0".into());
        }
        if algebraic_coarse_target_dimension < algebraic_coarse_dofs_per_node {
            return Err("--coarse-target must be >= --coarse-dofs".into());
        }

        Ok(Self {
            matrix,
            rhs,
            relative_tolerance,
            max_iterations,
            overlap_layers,
            probe_iterations,
            max_local_region_size,
            max_local_regions,
            max_local_factor_bytes,
            max_escalations,
            escalation_stage_iterations,
            local_factor_selection,
            algebraic_coarse_enabled,
            algebraic_coarse_dofs_per_node,
            algebraic_coarse_target_dimension,
            algebraic_coarse_aggregation,
            algebraic_coarse_basis,
            algebraic_coarse_transfer_apply_policy,
            algebraic_coarse_apply_policy,
            backend,
            skip_plain,
        })
    }
}

fn next_value<I: Iterator<Item = String>>(
    it: &mut I,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| format!("missing value after {flag}").into())
}

fn print_usage() {
    println!("HyBIT FEM / Matrix Market benchmark");
    println!();
    println!("Usage:");
    println!("  cargo run --release -p hybit --example fem_bench -- --matrix K.mtx [options]");
    println!();
    println!("Options:");
    println!("  --rhs FILE          whitespace-separated RHS vector; default: b=A*1");
    println!("  --tol VALUE         relative tolerance (default 1e-8)");
    println!("  --max-iters N       maximum PCG iterations (default 1000)");
    println!("  --backend MODE      auto|csr|abtm (default auto)");
    println!("  --probe-iters N     HyBIT probe length (default 12)");
    println!("  --overlap N         topology halo layers (default 1)");
    println!("  --max-region N      maximum local Cholesky order (default 128)");
    println!("  --max-regions N     maximum number of local regions (default 8)");
    println!("  --factor-budget-mib N  persistent local-factor budget in MiB (default 64)");
    println!("  --max-escalations N maximum selective-direct strengthening stages (default 3)");
    println!("  --stage-iters N     PCG iterations for non-final escalation stages (default 24)");
    println!(
        "  --selector MODE     candidate-order|benefit-byte|jacobi-byte (default jacobi-byte)"
    );
    println!("  --hybrid-coarse     enable algebraic two-level coarse base for Hybrid Auto");
    println!("  --coarse-dofs N     contiguous DOFs per node/block (default 3)");
    println!("  --coarse-target N   coarse-space dimension target (default 1536)");
    println!(
        "  --coarse-aggregation MODE contiguous|graph|strong-graph (default contiguous)"
    );
    println!("  --coarse-basis MODE piecewise|smoothed (default piecewise)");
    println!("  --coarse-transfer MODE serial|parallel (default serial; smoothed only)");
    println!("  --coarse-apply MODE auto|factor|inverse (default auto)");
    println!("  --skip-plain        skip the Plain Jacobi-PCG baseline (for parameter sweeps)");
}

fn load_rhs(path: &Path, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut values = Vec::with_capacity(n);

    for (line_index, raw_line) in text.lines().enumerate() {
        // Be tolerant of an UTF-8 BOM and optional comment lines, while never
        // silently ignoring malformed numeric data.
        let line = raw_line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }
        for (token_index, token) in line.split_whitespace().enumerate() {
            let value = token.parse::<f64>().map_err(|e| {
                format!(
                    "{}: invalid RHS float at line {}, token {}: {:?} ({e})",
                    path.display(),
                    line_index + 1,
                    token_index + 1,
                    token
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "{}: non-finite RHS value at line {}, token {}: {:?}",
                    path.display(),
                    line_index + 1,
                    token_index + 1,
                    token
                )
                .into());
            }
            values.push(value);
        }
    }

    if values.len() != n {
        return Err(format!(
            "{}: RHS length mismatch: expected {n}, got {}",
            path.display(),
            values.len()
        )
        .into());
    }
    Ok(values)
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn verified_relative_residual(
    a: &Csr32Matrix,
    b: &[f64],
    x: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let ax = a.spmv(x)?;
    let mut sum = 0.0;
    for i in 0..b.len() {
        let r = b[i] - ax[i];
        sum += r * r;
    }
    let denom = norm2(b);
    Ok(if denom == 0.0 {
        sum.sqrt()
    } else {
        sum.sqrt() / denom
    })
}

fn relative_error_to_ones(x: &[f64]) -> f64 {
    let mut diff = 0.0;
    for &xi in x {
        let d = xi - 1.0;
        diff += d * d;
    }
    diff.sqrt() / (x.len() as f64).sqrt().max(f64::MIN_POSITIVE)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn backend_name(backend: MatrixBackend) -> &'static str {
    match backend {
        MatrixBackend::Csr32 => "CSR32",
        MatrixBackend::Abtm => "ABTM",
        MatrixBackend::MatrixFree => "MatrixFree",
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    println!("HyBIT {} FEM matrix benchmark", env!("CARGO_PKG_VERSION"));
    println!("matrix             : {}", args.matrix.display());

    let load_start = Instant::now();
    let (matrix, mm) = read_matrix_market(&args.matrix)?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let profile = analyze_csr32(&matrix)?;

    println!(
        "Matrix Market      : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "coalesced / zeros  : {} / {}",
        mm.duplicate_entries_combined, mm.zero_entries_removed
    );
    println!("dimensions         : {} x {}", profile.nrows, profile.ncols);
    println!("nnz                : {}", profile.nnz);
    println!(
        "avg/max nnz/row    : {:.2} / {}",
        profile.avg_nnz_per_row, profile.max_nnz_per_row
    );
    println!(
        "CSR storage        : {:.3} MiB",
        mib(matrix.storage_bytes())
    );
    println!(
        "CSR metadata       : {:.3} MiB",
        mib(profile.csr_metadata_bytes)
    );
    println!(
        "full/+ diagonal    : {} / {}",
        profile.full_diagonal, profile.positive_diagonal
    );
    println!("load time          : {:.3} ms", load_seconds * 1.0e3);

    if !profile.square || !profile.full_diagonal || !profile.positive_diagonal {
        return Err("current HyBIT PCG benchmark requires a square matrix with a complete positive diagonal".into());
    }

    let generated_rhs = args.rhs.is_none();
    let b = if let Some(path) = args.rhs.as_deref() {
        println!("RHS                : {}", path.display());
        load_rhs(path, matrix.nrows())?
    } else {
        println!("RHS                : generated as b=A*1 (known exact solution)");
        matrix.spmv(&vec![1.0; matrix.ncols()])?
    };

    let options = SolverOptions {
        relative_tolerance: args.relative_tolerance,
        absolute_tolerance: 0.0,
        max_iterations: args.max_iterations,
    };

    let plain_metrics = if args.skip_plain {
        println!();
        println!("Plain Jacobi-PCG   : skipped (--skip-plain)");
        None
    } else {
        println!();
        println!("[1/2] Plain Jacobi-PCG");
        let plain_setup_start = Instant::now();
        let jacobi = JacobiPreconditioner::from_csr32(&matrix)?;
        let plain_setup_seconds = plain_setup_start.elapsed().as_secs_f64();
        let mut x_plain = vec![0.0; matrix.ncols()];
        let plain_start = Instant::now();
        let plain = pcg(&matrix, &jacobi, &b, &mut x_plain, options)?;
        let plain_solve_seconds = plain_start.elapsed().as_secs_f64();
        let plain_verified = verified_relative_residual(&matrix, &b, &x_plain)?;
        println!("status             : {:?}", plain.status);
        println!("iterations         : {}", plain.iterations);
        println!(
            "reported residual  : {:.6e}",
            plain.final_residual / norm2(&b).max(f64::MIN_POSITIVE)
        );
        println!("verified residual  : {:.6e}", plain_verified);
        println!(
            "setup / solve      : {:.3} / {:.3} ms",
            plain_setup_seconds * 1.0e3,
            plain_solve_seconds * 1.0e3
        );
        println!(
            "workspace estimate : {:.3} MiB",
            mib(6 * matrix.nrows() * std::mem::size_of::<f64>())
        );
        if generated_rhs {
            println!(
                "relative x error   : {:.6e}",
                relative_error_to_ones(&x_plain)
            );
        }
        Some((
            plain,
            plain_setup_seconds,
            plain_solve_seconds,
            plain_verified,
        ))
    };

    println!();
    if args.skip_plain {
        println!("[1/1] HyBIT Auto");
    } else {
        println!("[2/2] HyBIT Auto");
    }
    println!("selector           : {:?}", args.local_factor_selection);
    println!(
        "factor budget      : {:.3} MiB",
        mib(args.max_local_factor_bytes)
    );
    println!("max escalations    : {}", args.max_escalations);
    println!("stage iterations   : {}", args.escalation_stage_iterations);
    println!("algebraic coarse   : {}", args.algebraic_coarse_enabled);
    if args.algebraic_coarse_enabled {
        println!(
            "coarse DOFs/node   : {}",
            args.algebraic_coarse_dofs_per_node
        );
        println!(
            "coarse target dim  : {}",
            args.algebraic_coarse_target_dimension
        );
        println!(
            "coarse aggregation : {:?}",
            args.algebraic_coarse_aggregation
        );
        println!("coarse basis       : {:?}", args.algebraic_coarse_basis);
        println!(
            "coarse transfer    : {:?}",
            args.algebraic_coarse_transfer_apply_policy
        );
        println!(
            "coarse apply req.  : {:?}",
            args.algebraic_coarse_apply_policy
        );
    }
    let mut solver = HybitSolver::new();
    solver.set_options(options)?;
    solver.set_backend_policy(args.backend);
    let mut hybrid = solver.hybrid_options();
    hybrid.probe_iterations = args.probe_iterations;
    hybrid.overlap_layers = args.overlap_layers;
    hybrid.max_local_region_size = args.max_local_region_size;
    hybrid.max_local_regions = args.max_local_regions;
    hybrid.max_local_factor_bytes = args.max_local_factor_bytes;
    hybrid.max_escalations = args.max_escalations;
    hybrid.escalation_stage_iterations = args.escalation_stage_iterations;
    hybrid.local_factor_selection = args.local_factor_selection;
    hybrid.algebraic_coarse = AlgebraicCoarseOptions {
        enabled: args.algebraic_coarse_enabled,
        dofs_per_node: args.algebraic_coarse_dofs_per_node,
        target_coarse_dimension: args.algebraic_coarse_target_dimension,
        aggregation: args.algebraic_coarse_aggregation,
        basis: args.algebraic_coarse_basis,
        transfer_apply_policy: args.algebraic_coarse_transfer_apply_policy,
        apply_policy: args.algebraic_coarse_apply_policy,
    };
    solver.set_hybrid_options(hybrid)?;

    let mut x_hybit = vec![0.0; matrix.ncols()];
    let hybit_wall_start = Instant::now();
    let report = solver.solve_csr32(&matrix, &b, &mut x_hybit)?;
    let hybit_wall_seconds = hybit_wall_start.elapsed().as_secs_f64();
    let hybit_verified = verified_relative_residual(&matrix, &b, &x_hybit)?;

    println!("status             : {:?}", report.status);
    println!("backend            : {}", backend_name(report.backend));
    println!("preconditioner     : {:?}", report.preconditioner);
    println!("iterations         : {}", report.iterations);
    println!("reported residual  : {:.6e}", report.relative_residual);
    println!("verified residual  : {:.6e}", hybit_verified);
    println!("escalations        : {}", report.escalations);
    for stage in &report.escalation_stages {
        println!(
            "  stage {:>2}         : {:>4} iters, residual ratio {:.6e}, {} regions, {} unique DOFs, {:.3} MiB",
            stage.stage,
            stage.iterations,
            stage.residual_ratio,
            stage.local_direct_regions,
            stage.unique_local_factor_dofs,
            mib(stage.local_factor_bytes)
        );
    }
    println!(
        "probe              : {} iters, {:.3} ms",
        report.probe_iterations,
        report.probe_seconds * 1.0e3
    );
    println!("hard core DOFs     : {}", report.hard_dofs);
    println!("local regions      : {}", report.local_direct_regions);
    println!("largest region     : {}", report.largest_local_region);
    println!(
        "factor DOFs        : {} total / {} unique",
        report.local_factor_dofs, report.unique_local_factor_dofs
    );
    println!(
        "factor memory      : {:.3} MiB",
        mib(report.local_factor_bytes)
    );
    if report.algebraic_coarse_dimension > 0 {
        println!("coarse dimension   : {}", report.algebraic_coarse_dimension);
        println!(
            "coarse apply eff.  : {:?}",
            args.algebraic_coarse_apply_policy
                .resolve(report.algebraic_coarse_dimension)
        );
        println!(
            "coarse aggregate   : {} nodes",
            report.algebraic_coarse_aggregate_nodes
        );
        println!(
            "coarse memory      : {:.3} MiB",
            mib(report.algebraic_coarse_factor_bytes)
        );
        println!(
            "coarse setup       : {:.3} ms",
            report.algebraic_coarse_seconds * 1.0e3
        );
    }
    println!(
        "factor budget      : {:.3} MiB",
        mib(report.local_factor_budget_bytes)
    );
    println!(
        "budget skipped     : {} regions",
        report.local_factor_regions_skipped_for_budget
    );
    println!(
        "budget limited     : {}",
        report.local_factor_budget_limited
    );
    println!(
        "Krylov workspace   : {:.3} MiB",
        mib(report.krylov_workspace_bytes)
    );
    println!(
        "analysis           : {:.3} ms",
        report.analysis_seconds * 1.0e3
    );
    println!(
        "prepare            : {:.3} ms",
        report.prepare_seconds * 1.0e3
    );
    println!(
        "diagnostics        : {:.3} ms",
        report.diagnostics_seconds * 1.0e3
    );
    println!(
        "local factor       : {:.3} ms",
        report.local_factor_seconds * 1.0e3
    );
    println!(
        "solver time        : {:.3} ms",
        report.solve_seconds * 1.0e3
    );
    println!("total wall         : {:.3} ms", hybit_wall_seconds * 1.0e3);
    if generated_rhs {
        println!(
            "relative x error   : {:.6e}",
            relative_error_to_ones(&x_hybit)
        );
    }

    if let Some((plain, plain_setup_seconds, plain_solve_seconds, plain_verified)) = plain_metrics {
        println!();
        println!("Comparison");
        println!(
            "iteration ratio    : {:.3} (HyBIT / plain)",
            report.iterations as f64 / (plain.iterations.max(1) as f64)
        );
        println!(
            "solve-time ratio   : {:.3} (HyBIT wall / plain setup+solve)",
            hybit_wall_seconds / (plain_setup_seconds + plain_solve_seconds).max(f64::MIN_POSITIVE)
        );

        if !plain_verified.is_finite() {
            return Err("non-finite independently verified plain residual".into());
        }
    }

    if !hybit_verified.is_finite() {
        return Err("non-finite independently verified HyBIT residual".into());
    }
    Ok(())
}
