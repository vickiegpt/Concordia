use clap::Parser;
use ptx::sass::fuzz::{run_sass_lifter_fuzzer, SassLifterFuzzConfig};

#[derive(Debug, Parser)]
#[command(
    name = "sass_lifter_fuzz",
    about = "Deterministic Rust fuzzer for the SASS lifter supported subset"
)]
struct Args {
    #[arg(long, default_value_t = 0x5a55_1200)]
    seed: u64,

    #[arg(long, default_value_t = 1024)]
    cases: usize,

    #[arg(long, default_value_t = 32)]
    max_instructions: usize,

    #[arg(long, default_value_t = 120)]
    sm_version: u32,

    #[arg(long)]
    no_parse: bool,
}

fn main() {
    let args = Args::parse();
    let config = SassLifterFuzzConfig {
        seed: args.seed,
        cases: args.cases,
        max_instructions: args.max_instructions,
        sm_version: args.sm_version,
        parse_lifted_ptx: !args.no_parse,
    };

    match run_sass_lifter_fuzzer(config) {
        Ok(summary) => {
            println!(
                "sass_lifter_fuzz pass seed={} cases={} instructions={} diagnostics={} parse_failures={}",
                summary.seed,
                summary.cases,
                summary.instructions,
                summary.lift_diagnostics,
                summary.parse_failures
            );
        }
        Err(err) => {
            eprintln!("sass_lifter_fuzz fail: {}", err);
            std::process::exit(1);
        }
    }
}
