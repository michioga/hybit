use std::env;
use std::error::Error;
use std::path::PathBuf;

use hybit::{
    read_matrix_market, GeneralSquarePreconditionerPolicy, HybitSolver, MatrixProblemClass,
    PreconditionerKind,
};

#[derive(Debug)]
struct Args {
    matrix: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(
                        it.next().ok_or("missing value after --matrix")?,
                    ));
                }
                "-h" | "--help" => {
                    println!("Usage: general_square_ilu_fallback --matrix A.mtx");
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && matrix.is_none() => {
                    matrix = Some(PathBuf::from(other));
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }
        Ok(Self {
            matrix: matrix.ok_or("missing matrix path; use --matrix FILE.mtx")?,
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let (matrix, mm) = read_matrix_market(&args.matrix)?;

    println!(
        "HyBIT {} GeneralSquare ILU(0) fallback preflight",
        env!("CARGO_PKG_VERSION")
    );
    println!("matrix              : {}", args.matrix.display());
    println!(
        "Matrix Market       : {:?}, {} input entries -> {} CSR nnz",
        mm.symmetry, mm.input_entries, mm.csr_nnz
    );
    println!(
        "dimensions          : {} x {}",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("nnz                 : {}", matrix.nnz());

    let mut strict = HybitSolver::new();
    strict.set_problem_class(MatrixProblemClass::GeneralSquare);
    strict.set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0);

    let strict_result = strict
        .analyze_csr32(&matrix)
        .and_then(|analysis| strict.prepare_csr32(&matrix, &analysis));

    match strict_result {
        Ok(prepared) => {
            println!(
                "STRICT_ILU0|status=accepted|kind={:?}|bytes={}|adjusted_pivots={}",
                prepared
                    .general_square_preconditioner_kind()
                    .unwrap_or(PreconditionerKind::None),
                prepared.general_square_preconditioner_bytes(),
                prepared.general_square_ilu_adjusted_pivots()
            );
        }
        Err(error) => {
            println!("STRICT_ILU0|status=rejected|error={error}");
        }
    }

    let mut fallback = HybitSolver::new();
    fallback.set_problem_class(MatrixProblemClass::GeneralSquare);
    fallback
        .set_general_square_preconditioner_policy(GeneralSquarePreconditionerPolicy::Ilu0Fallback);

    let analysis = fallback.analyze_csr32(&matrix)?;
    let prepared = fallback.prepare_csr32(&matrix, &analysis)?;
    let kind = prepared
        .general_square_preconditioner_kind()
        .unwrap_or(PreconditionerKind::None);

    println!(
        "FALLBACK_PREFLIGHT|status=accepted|kind={kind:?}|fallback_used={}|bytes={}|adjusted_pivots={}",
        prepared.general_square_ilu_fallback_used(),
        prepared.general_square_preconditioner_bytes(),
        prepared.general_square_ilu_adjusted_pivots()
    );

    Ok(())
}
