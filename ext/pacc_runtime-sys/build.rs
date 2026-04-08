use std::env;
use std::process::Command;

fn main() {
    // Detect SiFive spike (riscv-isa-sim-zvfbfa-plus) installation
    let spike_path = env::var("SPIKE_PATH")
        .ok()
        .or_else(|| {
            Command::new("which")
                .arg("spike")
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        });

    if let Some(path) = &spike_path {
        println!("cargo:rustc-cfg=has_spike");
        println!("cargo:warning=PACC: Found Spike at: {}", path);

        // Check if this spike fork supports Zvbdot extensions
        if let Ok(output) = Command::new(path).arg("--help").output() {
            let help_text = String::from_utf8_lossy(&output.stderr);
            if help_text.contains("--isa") {
                println!("cargo:rustc-cfg=has_spike_isa_flag");
            }
        }
    } else {
        println!("cargo:warning=PACC: Spike not found. Set SPIKE_PATH or install spike.");
    }

    // Detect RISC-V toolchain
    let riscv_gcc = env::var("RISCV")
        .map(|r| format!("{}/bin/riscv64-unknown-elf-gcc", r))
        .ok()
        .or_else(|| {
            Command::new("which")
                .arg("riscv64-unknown-elf-gcc")
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        });

    if riscv_gcc.is_some() {
        println!("cargo:rustc-cfg=has_riscv_toolchain");
    }

    // Detect LLC
    if Command::new("llc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        println!("cargo:rustc-cfg=has_llc");
    }

    // PACC configuration constants
    println!("cargo:rustc-env=PACC_VLEN=512");
    println!("cargo:rustc-env=PACC_DEFAULT_SEW=8");
}
