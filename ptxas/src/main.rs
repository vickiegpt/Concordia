use bpaf::{any, choice, doc::Style, literal, Bpaf, Parser};

#[derive(Debug, Clone, Bpaf)]
#[allow(dead_code)]
#[bpaf(options, version("Open-source ptxas (hetGPU) SM120"))]
pub struct Options {
    #[bpaf(short, long)]
    output: String,
    warn_on_spills: bool,
    #[bpaf(short, long)]
    verbose: bool,
    #[bpaf(external)]
    lineinfo: bool,
    #[bpaf(external)]
    gpu_name: String,
    #[bpaf(long, short('O'), fallback(3))]
    opt_level: usize,
    #[bpaf(positional)]
    input: String,
}

fn lineinfo() -> impl Parser<bool> {
    choice(["-lineinfo", "--lineinfo"].into_iter().map(|s| {
        literal(s)
            .anywhere()
            .optional()
            .map(|_| true)
            .fallback(false)
            .boxed()
    }))
}

// #[bpaf(long, long("gpu_name"), fallback_with(default_arch))]
fn gpu_name() -> impl Parser<String> {
    any("", move |s: String| {
        Some(
            s.strip_prefix("-arch=")
                .or_else(|| s.strip_prefix("--gpu-name="))?
                .to_owned(),
        )
    })
    .metavar(&[("--gpu-name=", Style::Literal), ("SM", Style::Metavar)])
    .anywhere()
    .fallback_with(|| Ok::<String, &'static str>("sm_120".to_string()))
}

fn parse_sm_version(gpu_name: &str) -> u32 {
    gpu_name
        .strip_prefix("sm_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(120)
}

fn main() {
    let options = options().run();
    let sm_version = parse_sm_version(&options.gpu_name);

    if options.verbose {
        eprintln!(
            "hetGPU ptxas: compiling {} -> {} for sm_{}",
            options.input, options.output, sm_version
        );
    }

    // Read PTX source
    let ptx_source = match std::fs::read_to_string(&options.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", options.input, e);
            std::process::exit(1);
        }
    };

    let kernel_name = infer_entry_name(&ptx_source).unwrap_or("kernel");
    match nvidia_sass::pipeline::compile_ptx_to_cubin(&ptx_source, kernel_name, sm_version) {
        Ok(result) => {
            if let Err(e) = std::fs::write(&options.output, &result.cubin) {
                eprintln!("error: cannot write {}: {}", options.output, e);
                std::process::exit(1);
            }
            if options.verbose {
                eprintln!(
                    "hetGPU ptxas: wrote {} ({} bytes) via passes: {}",
                    options.output,
                    result.cubin.len(),
                    result.pass_names.join(",")
                );
            }
        }
        Err(e) => {
            eprintln!("error: PTX to SASS compilation failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn infer_entry_name(ptx_source: &str) -> Option<&str> {
    for line in ptx_source.lines() {
        let line = line.split_once("//").map(|(line, _)| line).unwrap_or(line);
        let marker = ".entry ";
        let Some(start) = line.find(marker).map(|idx| idx + marker.len()) else {
            continue;
        };
        let rest = line[start..].trim_start();
        let end = rest
            .find(|ch: char| ch == '(' || ch.is_whitespace())
            .unwrap_or(rest.len());
        if end > 0 {
            return Some(&rest[..end]);
        }
    }
    None
}
