use std::{env, fs, path::Path};

fn main() {
    let mut args = env::args().skip(1).peekable();
    if args.peek().is_none() {
        eprintln!("usage: ptx_parse_check <file.ptx> [...]");
        std::process::exit(2);
    }

    let mut failures = 0usize;
    for path in args {
        let path_ref = Path::new(&path);
        let text = match fs::read_to_string(path_ref) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("ptx_parse_check fail {}: {}", path_ref.display(), err);
                failures += 1;
                continue;
            }
        };

        match ptx_parser::parse_module_checked(&text) {
            Ok(_) => {
                println!("ptx_parse_check pass {}", path_ref.display());
            }
            Err(errors) => {
                eprintln!(
                    "ptx_parse_check fail {}: {} parse error(s)",
                    path_ref.display(),
                    errors.len()
                );
                for error in errors.iter().take(8) {
                    eprintln!("  {:?}", error);
                }
                if errors.len() > 8 {
                    eprintln!("  ... {} additional parse errors", errors.len() - 8);
                }
                failures += 1;
            }
        }
    }

    if failures != 0 {
        std::process::exit(1);
    }
}
