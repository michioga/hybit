//! Research-only G4/C1-H12 ABTM ILU(0) word operation comparison.
//! Run each arm in a separate process to measure peak RSS fairly.

use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hybit::{Csr32Matrix, Ilu0Preconditioner, Preconditioner, read_matrix_market};
use hybit_precond::Ilu0AbtmExperimentalMode;

#[derive(Clone, Copy, Debug)]
enum Arm {
    Csr,
    LegacyLut,
    DirectLut,
    DirectPopcnt,
    DirectRotatePopcnt,
}

impl Arm {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "csr" => Ok(Self::Csr),
            "legacy-lut" => Ok(Self::LegacyLut),
            "direct-lut" => Ok(Self::DirectLut),
            "direct-popcnt" => Ok(Self::DirectPopcnt),
            "direct-rotate-popcnt" => Ok(Self::DirectRotatePopcnt),
            other => Err(format!("unknown arm '{other}'").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Csr => "csr",
            Self::LegacyLut => "legacy-lut",
            Self::DirectLut => "direct-lut",
            Self::DirectPopcnt => "direct-popcnt",
            Self::DirectRotatePopcnt => "direct-rotate-popcnt",
        }
    }

    fn prepare(self, matrix: &Csr32Matrix) -> Result<Ilu0Preconditioner, hybit::HybitError> {
        match self {
            Self::Csr => Ilu0Preconditioner::from_csr32_general(matrix),
            Self::LegacyLut => Ilu0Preconditioner::from_csr32_general_abtm(matrix),
            Self::DirectLut => Ilu0Preconditioner::from_csr32_general_abtm_experimental(
                matrix,
                Ilu0AbtmExperimentalMode::DirectLut,
            ),
            Self::DirectPopcnt => Ilu0Preconditioner::from_csr32_general_abtm_experimental(
                matrix,
                Ilu0AbtmExperimentalMode::DirectPopcnt,
            ),
            Self::DirectRotatePopcnt => Ilu0Preconditioner::from_csr32_general_abtm_experimental(
                matrix,
                Ilu0AbtmExperimentalMode::DirectRotatePopcnt,
            ),
        }
    }
}

struct Args {
    matrix: PathBuf,
    arm: Arm,
    repeats: usize,
    warmups: usize,
    verify: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut matrix = None;
        let mut arm = None;
        let mut repeats = 7usize;
        let mut warmups = 2usize;
        let mut verify = false;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--matrix" => {
                    matrix = Some(PathBuf::from(args.next().ok_or("missing --matrix value")?));
                }
                "--arm" => {
                    arm = Some(Arm::parse(&args.next().ok_or("missing --arm value")?)?);
                }
                "--repeats" => {
                    repeats = args.next().ok_or("missing --repeats value")?.parse()?;
                    if repeats == 0 {
                        return Err("--repeats must be positive".into());
                    }
                }
                "--warmups" => {
                    warmups = args.next().ok_or("missing --warmups value")?.parse()?;
                }
                "--verify" => verify = true,
                "-h" | "--help" => {
                    println!(
                        "Usage: abtm_ilu0_word_ops_experiment --matrix A.mm --arm csr|legacy-lut|direct-lut|direct-popcnt|direct-rotate-popcnt [--warmups 2] [--repeats 7] [--verify]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unexpected argument '{other}'").into()),
            }
        }
        Ok(Self {
            matrix: matrix.ok_or("--matrix is required")?,
            arm: arm.ok_or("--arm is required")?,
            repeats,
            warmups,
            verify,
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let (matrix, info) = read_matrix_market(&args.matrix)?;

    println!(
        "C1H12_INPUT|matrix={}|arm={}|rows={}|cols={}|input_entries={}|csr_nnz={}|warmups={}|repeats={}|verify={}",
        args.matrix.display(),
        args.arm.as_str(),
        matrix.nrows(),
        matrix.ncols(),
        info.input_entries,
        matrix.nnz(),
        args.warmups,
        args.repeats,
        args.verify,
    );

    // Verification is separate from RSS timing; it materializes both factors.
    if args.verify {
        let reference = Arm::Csr.prepare(&matrix)?;
        let candidate = args.arm.prepare(&matrix)?;
        if !reference.experimental_factor_bits_equal(&candidate) {
            return Err("ILU(0) factors differ bitwise".into());
        }
        let rhs: Vec<f64> = (0..matrix.nrows())
            .map(|row| (row as f64 * 0.003_906_25).sin() + 0.25)
            .collect();
        let mut expected = vec![0.0; matrix.nrows()];
        let mut actual = vec![0.0; matrix.nrows()];
        reference.apply(&rhs, &mut expected)?;
        candidate.apply(&rhs, &mut actual)?;
        if !expected
            .iter()
            .zip(&actual)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err("ILU(0) application differs bitwise".into());
        }
        println!(
            "C1H12_VERIFY|arm={}|factor_bitwise=true|apply_bitwise=true|canonical_nnz={}|adjusted_pivots={}|factor_bytes={}",
            args.arm.as_str(),
            candidate.canonical_nnz(),
            candidate.adjusted_pivots(),
            candidate.factor_bytes()
        );
    }

    for _ in 0..args.warmups {
        let factor = args.arm.prepare(&matrix)?;
        black_box(factor.canonical_nnz());
    }

    let mut samples = Vec::with_capacity(args.repeats);
    let mut expected_metadata = None;
    for sample in 0..args.repeats {
        let start = Instant::now();
        let factor = args.arm.prepare(&matrix)?;
        let ms = start.elapsed().as_secs_f64() * 1.0e3;
        let metadata = (
            factor.canonical_nnz(),
            factor.adjusted_pivots(),
            factor.factor_bytes(),
        );
        if let Some(previous) = expected_metadata
            && metadata != previous
        {
            return Err("ILU(0) metadata changed between repetitions".into());
        }
        expected_metadata = Some(metadata);
        black_box(&factor);
        samples.push(ms);
        println!(
            "C1H12_SAMPLE|arm={}|index={}|prepare_ms={ms:.6}",
            args.arm.as_str(),
            sample + 1
        );
    }

    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let (canonical_nnz, adjusted_pivots, factor_bytes) = expected_metadata.unwrap();
    println!(
        "C1H12_RESULT|arm={}|median_prepare_ms={median:.6}|min_prepare_ms={:.6}|max_prepare_ms={:.6}|canonical_nnz={canonical_nnz}|adjusted_pivots={adjusted_pivots}|factor_bytes={factor_bytes}",
        args.arm.as_str(),
        samples[0],
        samples[samples.len() - 1],
    );
    Ok(())
}
