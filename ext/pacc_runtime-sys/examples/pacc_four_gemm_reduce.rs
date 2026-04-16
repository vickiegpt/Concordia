use pacc_runtime_sys::{
    pacc_CreateDevice, pacc_CreateKernelOnDevice, pacc_CreateProgram, pacc_DestroyDevice,
    pacc_DestroyKernel, pacc_LoadProgram, pacc_LaunchKernel, PaccComm, PaccReduceOp,
};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let real_io = std::env::var("HETGPU_PACC_REAL_IO").ok().as_deref() == Some("1");
    if !real_io {
        std::env::set_var("HETGPU_PACC_DRY_RUN", "1");
        std::env::set_var("HETGPU_PACC_SKIP_HW_REDUCE", "1");
        println!("PACC smoke using dry-run io; set HETGPU_PACC_REAL_IO=1 for driver submission");
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("examples/pacc_gemm_kernel.c");
    let elf = manifest_dir.join("target/pacc_gemm_kernel.elf");

    compile_kernel(&source, &elf)?;
    let elf_bytes = std::fs::read(&elf)?;
    println!("compiled custom PACC ELF: {} bytes", elf_bytes.len());

    let mut launched = 0usize;
    for device_id in 0..4u32 {
        match launch_on_device(device_id, &elf_bytes) {
            Ok(()) => {
                println!("pacc{} launch submitted", device_id);
                launched += 1;
            }
            Err(e) => {
                eprintln!("pacc{} launch failed: {}", device_id, e);
                if strict() {
                    return Err(e);
                }
            }
        }
    }

    if launched != 4 && strict() {
        return Err(format!("strict mode expected 4 launches, got {}", launched).into());
    }

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 4];
    match PaccComm::init_all().and_then(|comm| comm.all_reduce(&input, &mut output, PaccReduceOp::Sum)) {
        Ok(()) => println!("PACC communicator reduce smoke: {:?}", output),
        Err(e) => {
            eprintln!("PACC communicator reduce failed: {}", e);
            if strict() {
                return Err(format!("reduce failed: {}", e).into());
            }
        }
    }

    println!("four-PACC GEMM/reduce smoke complete: {} launch attempts succeeded", launched);
    Ok(())
}

fn compile_kernel(source: &Path, elf: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = elf.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let compiler = std::env::var("PACC_CC").unwrap_or_else(|_| "riscv64-linux-gnu-gcc".to_string());
    let status = Command::new(&compiler)
        .arg("-nostdlib")
        .arg("-static")
        .arg("-march=rv64gcv")
        .arg("-mabi=lp64d")
        .arg("-Wl,-Ttext=0x30080000")
        .arg("-Wl,-e,_start")
        .arg("-o")
        .arg(elf)
        .arg(source)
        .status()?;

    if !status.success() {
        return Err(format!("{} failed with {}", compiler, status).into());
    }

    Ok(())
}

fn launch_on_device(device_id: u32, elf_bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        let dev = pacc_CreateDevice(device_id);
        if dev.is_null() {
            return Err(format!("pacc_CreateDevice({}) returned null", device_id).into());
        }

        let program = pacc_CreateProgram();
        if program.is_null() {
            pacc_DestroyDevice(dev);
            return Err("pacc_CreateProgram returned null".into());
        }

        let load_result = pacc_LoadProgram(
            program,
            elf_bytes.as_ptr().cast(),
            elf_bytes.len() as u64,
        );
        if load_result != 0 {
            pacc_DestroyDevice(dev);
            return Err(format!("pacc_LoadProgram failed: {}", load_result).into());
        }

        let kernel_name = CString::new("pacc_gemm_smoke")?;
        let kernel = pacc_CreateKernelOnDevice(program, dev, kernel_name.as_ptr());
        if kernel.is_null() {
            pacc_DestroyDevice(dev);
            return Err("pacc_CreateKernelOnDevice returned null".into());
        }

        let launch_result = pacc_LaunchKernel(kernel, 1, 1, 1, 1, 1, 1);

        if launch_result != 0 {
            return Err(format!("pacc_LaunchKernel failed: {}", launch_result).into());
        }

        pacc_DestroyKernel(kernel);
        pacc_DestroyDevice(dev);
    }

    Ok(())
}

fn strict() -> bool {
    std::env::var("HETGPU_PACC_STRICT").ok().as_deref() == Some("1")
}
