fn main() {
    if let Err(err) = concordia_all::run_from_env() {
        eprintln!("concordia_all=fail error={err}");
        std::process::exit(1);
    }
}
