use std::env;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: scotch_map_to_labels <input.map> <output.labels> <vertices> <ranks>");
        std::process::exit(2);
    }

    let map_path = &args[1];
    let output_path = &args[2];
    let vertices: usize = args[3].parse()?;
    let ranks: u32 = args[4].parse()?;

    let text = fs::read_to_string(map_path)?;
    let mut tokens = text.split_whitespace();

    let mapping_lines: usize = tokens
        .next()
        .ok_or("SCOTCH mapping file is empty")?
        .parse()?;

    if mapping_lines != vertices {
        return Err(
            format!("SCOTCH mapping contains {mapping_lines} pairs; expected {vertices}").into(),
        );
    }

    let mut labels = vec![u32::MAX; vertices];

    for _ in 0..mapping_lines {
        let vertex: usize = tokens
            .next()
            .ok_or("SCOTCH mapping ended before all vertex labels")?
            .parse()?;
        let part: u32 = tokens
            .next()
            .ok_or("SCOTCH mapping ended before all partition labels")?
            .parse()?;

        if vertex >= vertices {
            return Err(format!("SCOTCH vertex {vertex} is outside 0..{vertices}").into());
        }
        if part >= ranks {
            return Err(format!("SCOTCH part {part} is outside 0..{ranks}").into());
        }
        if labels[vertex] != u32::MAX {
            return Err(format!("SCOTCH vertex {vertex} appears more than once").into());
        }
        labels[vertex] = part;
    }

    if labels.contains(&u32::MAX) {
        return Err("SCOTCH mapping does not cover every vertex".into());
    }

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    for label in labels {
        writeln!(writer, "{label}")?;
    }

    println!("SCOTCH mapping converted");
    println!("input    : {map_path}");
    println!("output   : {output_path}");
    println!("vertices : {vertices}");
    println!("ranks    : {ranks}");

    Ok(())
}
