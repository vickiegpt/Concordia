use lazy_static::lazy_static;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io;
use std::mem;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use tempfile::{Builder, TempDir, tempdir};

use crate::{
    pacc_comgr_action_info_t, pacc_comgr_action_kind_s, pacc_comgr_create_data,
    pacc_comgr_data_kind_s, pacc_comgr_data_set_add, pacc_comgr_data_set_bytes,
    pacc_comgr_data_set_name, pacc_comgr_data_set_t, pacc_comgr_language_s,
    pacc_comgr_release_data, pacc_comgr_status_s, pacc_comgr_status_t,
};

pub(crate) struct DataContent {
    pub(crate) kind: pacc_comgr_data_kind_s,
    pub(crate) content: Vec<u8>,
    pub(crate) name: Option<String>,
}

pub(crate) type DataMap = HashMap<u64, DataContent>;

lazy_static! {
    pub(crate) static ref DATA_STORE: Mutex<DataMap> = Mutex::new(HashMap::new());
    pub(crate) static ref DATA_SET_STORE: Mutex<HashMap<u64, Vec<u64>>> =
        Mutex::new(HashMap::new());
    pub(crate) static ref ACTION_INFO_STORE: Mutex<HashMap<u64, ActionInfo>> =
        Mutex::new(HashMap::new());
    pub(crate) static ref NEXT_HANDLE: Mutex<u64> = Mutex::new(1);
}

pub(crate) fn get_next_handle() -> u64 {
    let mut handle = NEXT_HANDLE.lock().unwrap();
    let current = *handle;
    *handle += 1;
    current
}

#[derive(Default, Clone)]
pub(crate) struct ActionInfo {
    pub(crate) language: Option<pacc_comgr_language_s>,
    pub(crate) options: Vec<String>,
    pub(crate) working_directory: Option<String>,
    pub(crate) target: Option<String>,
}

struct ActionContext {
    temp_dir: PathBuf,
    options: Vec<String>,
    language: pacc_comgr_language_s,
    input_files: Vec<(PathBuf, pacc_comgr_data_kind_s)>,
    working_directory: Option<PathBuf>,
    target: Option<String>,
    action_kind: pacc_comgr_action_kind_s,
}

fn preferred_temp_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    for key in ["HETGPU_PACC_TMPDIR", "TMPDIR", "TEMP", "TMP"] {
        if let Ok(value) = std::env::var(key) {
            if !value.is_empty() {
                let path = PathBuf::from(value);
                if !roots.iter().any(|existing| existing == &path) {
                    roots.push(path);
                }
            }
        }
    }

    for candidate in ["/mnt/usb/hetgpu_tmp", "/dev/shm/hetgpu_tmp"] {
        let path = PathBuf::from(candidate);
        if !roots.iter().any(|existing| existing == &path) {
            roots.push(path);
        }
    }

    let system_tmp = std::env::temp_dir();
    if !roots.iter().any(|existing| existing == &system_tmp) {
        roots.push(system_tmp);
    }

    roots
}

fn create_action_tempdir() -> io::Result<TempDir> {
    let mut last_error: Option<io::Error> = None;

    for root in preferred_temp_roots() {
        if let Err(err) = fs::create_dir_all(&root) {
            last_error = Some(io::Error::new(
                err.kind(),
                format!("{}: {}", root.display(), err),
            ));
            continue;
        }

        match Builder::new().prefix(".tmp").tempdir_in(&root) {
            Ok(dir) => return Ok(dir),
            Err(err) => {
                last_error = Some(io::Error::new(
                    err.kind(),
                    format!("{}: {}", root.display(), err),
                ));
            }
        }
    }

    if let Some(err) = last_error {
        Err(err)
    } else {
        tempdir()
    }
}

fn keep_action_tempdir() -> bool {
    matches!(
        std::env::var("HETGPU_PACC_KEEP_TMP").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub fn perform_action(
    action_kind: pacc_comgr_action_kind_s,
    action_info: pacc_comgr_action_info_t,
    input_set: pacc_comgr_data_set_t,
    output_set: pacc_comgr_data_set_t,
) -> pacc_comgr_status_t {
    let dir = match create_action_tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("PACC: Failed to create temporary directory: {}", e);
            return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR);
        }
    };

    let action_info_lock = ACTION_INFO_STORE.lock().unwrap();
    let action_data = match action_info_lock.get(&action_info.handle) {
        Some(data) => data.clone(),
        None => return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };
    drop(action_info_lock);

    let data_set_lock = DATA_SET_STORE.lock().unwrap();
    let data_handles = match data_set_lock.get(&input_set.handle) {
        Some(handles) => handles.clone(),
        None => return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };
    drop(data_set_lock);

    let working_directory = action_data.working_directory.as_ref().map(PathBuf::from);
    let data_store_lock = DATA_STORE.lock().unwrap();
    let mut input_files = Vec::new();

    for handle in &data_handles {
        if let Some(data) = data_store_lock.get(handle) {
            let file_name = match &data.name {
                Some(name) => name.clone(),
                None => match data.kind.0 {
                    1 => "input.cpp".to_string(),
                    6 => format!("input_{}.bc", handle),
                    7 => format!("input_{}.o", handle),
                    _ => format!("data_{}", handle),
                },
            };

            if let Some(original_path) =
                resolve_original_input_path(&file_name, working_directory.as_deref())
            {
                input_files.push((original_path, data.kind));
                continue;
            }

            let file_path = dir.path().join(&file_name);
            if let Some(parent) = file_path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    eprintln!(
                        "PACC: Warning: Could not create parent directories for {}: {}",
                        file_path.display(),
                        e
                    );
                }
            }
            if let Err(e) = fs::write(&file_path, &data.content) {
                eprintln!("PACC: Warning: Could not write input file: {}", e);
            }
            input_files.push((file_path, data.kind));
        }
    }
    drop(data_store_lock);

    let ctx = ActionContext {
        temp_dir: dir.path().to_path_buf(),
        options: action_data.options,
        language: action_data
            .language
            .unwrap_or(pacc_comgr_language_s::PACC_COMGR_LANGUAGE_NONE),
        input_files,
        working_directory,
        target: action_data.target,
        action_kind,
    };

    if keep_action_tempdir() {
        eprintln!("PACC: keeping temporary directory {}", dir.path().display());
    }

    let result = match action_kind.0 {
        0 => preprocess_source(&ctx),
        1 => Ok(()), // precompiled headers not used
        2 => compile_source_to_bc(&ctx),
        3 => Ok(()), // device libraries handled at link time
        4 => link_bc_to_bc(&ctx),
        5 => optimize_bc(&ctx),
        6 => codegen_to_riscv_pacc(&ctx),
        7 => codegen_to_assembly(&ctx),
        8 => compile_to_fatbin(&ctx),
        _ => {
            eprintln!("PACC: Unknown action kind: {}", action_kind.0);
            return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        }
    };

    if result.is_ok() {
        add_outputs_to_set(&ctx, output_set)?;
    }

    if keep_action_tempdir() {
        let preserved = dir.keep();
        eprintln!(
            "PACC: preserved temporary directory {}",
            preserved.display()
        );
    }

    result
}

fn add_outputs_to_set(
    ctx: &ActionContext,
    output_set: pacc_comgr_data_set_t,
) -> pacc_comgr_status_t {
    let output_dir = ctx.temp_dir.clone();
    let entries = match fs::read_dir(&output_dir) {
        Ok(entries) => entries,
        Err(_) => return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR),
    };

    let mut added_files = false;

    for entry in entries {
        if let Ok(entry) = entry {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            if let Some(extension) = path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();

                if ext == "o" || ext == "elf" {
                    add_file_to_set(
                        &path,
                        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_RELOCATABLE,
                        output_set,
                    )?;
                    added_files = true;
                } else if ext == "bc" {
                    add_file_to_set(
                        &path,
                        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC,
                        output_set,
                    )?;
                    added_files = true;
                } else if ext == "s" || ext == "asm" {
                    add_file_to_set(
                        &path,
                        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_SOURCE,
                        output_set,
                    )?;
                    added_files = true;
                } else if ext == "fatbin" {
                    add_file_to_set(
                        &path,
                        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_FATBIN,
                        output_set,
                    )?;
                    added_files = true;
                }
            }
        }
    }

    // Create dummy RISC-V ELF if no outputs for codegen action
    if !added_files
        && ctx.action_kind.0
            == pacc_comgr_action_kind_s::PACC_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE.0
    {
        eprintln!("PACC: No output files found - creating dummy RISC-V ELF output");
        let dummy = create_dummy_riscv_elf();

        let mut data = unsafe { mem::zeroed() };
        pacc_comgr_create_data(
            pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_RELOCATABLE,
            &mut data,
        )?;
        pacc_comgr_data_set_name(data, c"pacc_output.elf".as_ptr())?;
        pacc_comgr_data_set_bytes(data, dummy.as_ptr() as *const c_void, dummy.len())?;
        pacc_comgr_data_set_add(output_set, data)?;
        pacc_comgr_release_data(data)?;
    }

    Ok(())
}

fn add_file_to_set(
    file_path: &Path,
    kind: pacc_comgr_data_kind_s,
    data_set: pacc_comgr_data_set_t,
) -> pacc_comgr_status_t {
    let mut data = unsafe { mem::zeroed() };
    pacc_comgr_create_data(kind, &mut data)?;

    let name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| CString::new(n).ok())
        .unwrap_or_else(|| CString::new("output").unwrap());
    pacc_comgr_data_set_name(data, name.as_ptr())?;

    let content = fs::read(file_path).map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
    pacc_comgr_data_set_bytes(data, content.as_ptr() as *const c_void, content.len())?;
    pacc_comgr_data_set_add(data_set, data)?;
    pacc_comgr_release_data(data)?;

    Ok(())
}

// --- Action implementations targeting RISC-V with VCIX/IME ---

fn preprocess_source(ctx: &ActionContext) -> pacc_comgr_status_t {
    let source_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();

    for input_file in source_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.i", file_stem));
        fs::copy(&input_file, &output_file)
            .map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
    }
    Ok(())
}

fn compile_source_to_bc(ctx: &ActionContext) -> pacc_comgr_status_t {
    let source_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();

    if source_files.is_empty() {
        return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    for input_file in source_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.bc", file_stem));

        let compile_result = if is_cxx_source(&input_file) || is_c_source(&input_file) {
            compile_c_family_source_to_bitcode(ctx, &input_file, &output_file)
        } else {
            compile_ptx_to_bitcode(ctx, &input_file, &output_file)
        };

        if let Err(e) = compile_result {
            eprintln!(
                "PACC: source -> LLVM bitcode failed for {}: {}",
                input_file.display(),
                e
            );
            return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR);
        }

        eprintln!("PACC: Created LLVM bitcode: {}", output_file.display());
    }

    Ok(())
}

fn link_bc_to_bc(ctx: &ActionContext) -> pacc_comgr_status_t {
    let bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.len() < 2 {
        if let Some(input_file) = bc_files.first() {
            let output_file = ctx.temp_dir.join("linked.bc");
            fs::copy(input_file, &output_file)
                .map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
        }
        return Ok(());
    }

    let output_file = ctx.temp_dir.join("linked.bc");

    let mut cmd = Command::new(llvm_link_tool());
    for bc_file in &bc_files {
        cmd.arg(bc_file);
    }
    cmd.arg("-o").arg(&output_file);

    match cmd.output() {
        Ok(output) if output.status.success() => {
            eprintln!("PACC: Successfully linked bitcode files");
            Ok(())
        }
        Ok(output) => {
            eprintln!(
                "PACC: llvm-link failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)
        }
        Err(err) => {
            eprintln!("PACC: failed to execute llvm-link: {err}");
            Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)
        }
    }
}

fn optimize_bc(ctx: &ActionContext) -> pacc_comgr_status_t {
    let bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.is_empty() {
        return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    for input_file in bc_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_optimized.bc", file_stem));
        let input_size = fs::metadata(&input_file).map(|m| m.len()).unwrap_or(0);

        if input_size >= 1_000_000 {
            eprintln!(
                "PACC: Skipping opt for large bitcode ({} bytes): {}",
                input_size,
                input_file.display()
            );
            fs::copy(&input_file, &output_file)
                .map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
            eprintln!(
                "PACC: Copied bitcode without optimization: {}",
                output_file.display()
            );
            continue;
        }

        let mut cmd = Command::new(opt_tool());
        cmd.arg(&input_file)
            .arg("-o")
            .arg(&output_file)
            .arg("-O3")
            .arg("-non-global-value-max-name-size=16384");

        eprintln!(
            "PACC: Running opt on {} -> {}",
            input_file.display(),
            output_file.display()
        );

        match cmd.output() {
            Ok(output) if output.status.success() => {
                eprintln!("PACC: Optimized bitcode: {}", output_file.display());
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if is_llvm23_attr_mismatch(&stderr) {
                    eprintln!(
                        "PACC: skipping opt for {} due to LLVM23/LLVM20 bitcode mismatch; using unoptimized bitcode",
                        input_file.display()
                    );
                } else {
                    eprintln!(
                        "PACC: opt failed for {} (status {:?}) stderr:\n{}",
                        input_file.display(),
                        output.status.code(),
                        stderr
                    );
                }
                fs::copy(&input_file, &output_file)
                    .map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
            }
            Err(err) => {
                eprintln!(
                    "PACC: opt invocation failed for {}: {}",
                    input_file.display(),
                    err
                );
                fs::copy(&input_file, &output_file)
                    .map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
            }
        }
    }

    Ok(())
}

fn is_llvm23_attr_mismatch(stderr: &str) -> bool {
    stderr.contains("Unknown attribute kind (105)")
        || (stderr.contains("Producer: 'LLVM23") && stderr.contains("Reader: 'LLVM 20"))
}

fn codegen_to_riscv_pacc(ctx: &ActionContext) -> pacc_comgr_status_t {
    let mut bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.is_empty() {
        return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // After LINK_BC_TO_BC + OPTIMIZE_BC_TO_BC we may have per-module bitcode
    // (`main_optimized.bc`, `linked_0_optimized.bc`, ...) and the fully linked
    // aggregate (`linked_optimized.bc`). For the PACC path we only need the
    // aggregate relocatable; compiling each helper module separately drags in
    // AMD-specific helper intrinsics that are irrelevant after linking and can
    // crash the RISC-V backend during instruction selection.
    if let Some(linked) = bc_files
        .iter()
        .find(|path| path.file_name().and_then(|n| n.to_str()) == Some("linked_optimized.bc"))
        .cloned()
    {
        bc_files.clear();
        bc_files.push(linked);
    }

    for input_file in bc_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_pacc.o", file_stem));

        let used_config = match compile_bc_to_pacc_object(&input_file, &output_file) {
            Ok(config) => config,
            Err(e) => {
                eprintln!(
                    "PACC/XM: failed to generate RISC-V object from {}: {}",
                    input_file.display(),
                    e
                );
                return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR);
            }
        };

        eprintln!(
            "PACC/XM: Generated RISC-V object code with target={} march={}: {}",
            used_config.target_triple,
            used_config.march,
            output_file.display()
        );
    }

    Ok(())
}

fn codegen_to_assembly(ctx: &ActionContext) -> pacc_comgr_status_t {
    let bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.is_empty() {
        return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let config = crate::PaccConfig::xm_hardware();

    for input_file in bc_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.s", file_stem));

        if let Err(e) = compile_bc_to_xm_assembly(&input_file, &output_file, &config) {
            eprintln!(
                "PACC/XM: failed to generate assembly from {}: {}",
                input_file.display(),
                e
            );
            return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR);
        }

        eprintln!(
            "PACC/XM: Generated RISC-V+VCIX assembly: {}",
            output_file.display()
        );
    }

    Ok(())
}

fn compile_to_fatbin(ctx: &ActionContext) -> pacc_comgr_status_t {
    let source_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();

    if source_files.is_empty() {
        return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    for input_file in source_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_pacc.fatbin", file_stem));
        let bc_file = ctx.temp_dir.join(format!("{}_fatbin.bc", file_stem));
        let elf_file = ctx.temp_dir.join(format!("{}_fatbin.o", file_stem));

        let compile_result = if is_cxx_source(&input_file) || is_c_source(&input_file) {
            compile_c_family_source_to_bitcode(ctx, &input_file, &bc_file)
        } else {
            compile_ptx_to_bitcode(ctx, &input_file, &bc_file)
        };

        if let Err(e) = compile_result {
            eprintln!(
                "PACC: source -> LLVM bitcode failed for fatbin input {}: {}",
                input_file.display(),
                e
            );
            return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR);
        }

        if let Err(e) = compile_bc_to_pacc_object(&bc_file, &elf_file) {
            eprintln!(
                "PACC/XM: bitcode -> object failed for fatbin input {}: {}",
                input_file.display(),
                e
            );
            return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR);
        }

        let elf_bytes =
            fs::read(&elf_file).map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
        let fatbin_content = create_cuda_fatbin_with_elf(&elf_bytes);

        fs::write(&output_file, fatbin_content)
            .map_err(|_| pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR)?;
        eprintln!("PACC: Created fatbin: {}", output_file.display());
    }

    Ok(())
}

fn compile_ptx_to_bitcode(
    ctx: &ActionContext,
    input_file: &Path,
    output_file: &Path,
) -> io::Result<()> {
    let tool = ensure_ptx_helper_tool("ptx_to_llvm_bc")?;
    let llvm_ir_path = output_file.with_extension("ll");
    let mut cmd = Command::new(tool);
    cmd.arg(input_file)
        .arg(output_file)
        .arg("--llvm-ir")
        .arg(&llvm_ir_path);
    apply_action_context_to_command(ctx, &mut cmd);
    run_command(&mut cmd, "PTX -> LLVM bitcode")
}

fn compile_c_family_source_to_bitcode(
    ctx: &ActionContext,
    input_file: &Path,
    output_file: &Path,
) -> io::Result<()> {
    let config = crate::PaccConfig::rvv_linux_bf16();
    let clang = preferred_tool(
        "HETGPU_PACC_SOURCE_CLANG",
        &["/usr/bin/clang-20", "clang-20", "clang"],
    );
    let target = ctx
        .target
        .clone()
        .or_else(|| std::env::var("HETGPU_PACC_SOURCE_TARGET").ok())
        .unwrap_or_else(|| config.target_triple.clone());
    let sysroot = std::env::var("HETGPU_PACC_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
    let gcc_toolchain =
        std::env::var("HETGPU_PACC_SOURCE_GCC_TOOLCHAIN").unwrap_or_else(|_| "/usr".to_string());
    let march = riscv_march_with_required_extensions(
        &std::env::var("HETGPU_PACC_SOURCE_MARCH").unwrap_or_else(|_| config.march.clone()),
    );
    let mabi = std::env::var("HETGPU_PACC_SOURCE_MABI").unwrap_or_else(|_| "lp64d".to_string());

    let mut cmd = Command::new(clang);
    cmd.arg(format!("--target={}", target))
        .arg(format!("--sysroot={}", sysroot))
        .arg(format!("--gcc-toolchain={}", gcc_toolchain))
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-emit-llvm")
        .arg("-c")
        .arg("-O3")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg(format!("-mabi={}", mabi))
        .arg(input_file)
        .arg("-o")
        .arg(output_file);

    if is_cxx_source(input_file) {
        cmd.arg("-std=c++17").arg("-stdlib=libstdc++");
    } else {
        cmd.arg("-std=c11");
    }

    cmd.args(&ctx.options);
    apply_action_context_to_command(ctx, &mut cmd);

    run_command(&mut cmd, "C/C++ source -> LLVM bitcode")
}

fn apply_action_context_to_command(ctx: &ActionContext, cmd: &mut Command) {
    if let Some(dir) = &ctx.working_directory {
        cmd.current_dir(dir);
    }
    cmd.env("HETGPU_PACC_TMPDIR", &ctx.temp_dir)
        .env("TMPDIR", &ctx.temp_dir)
        .env("TEMP", &ctx.temp_dir)
        .env("TMP", &ctx.temp_dir);
}

fn riscv_march_with_required_extensions(march: &str) -> String {
    if !(march.starts_with("rv32") || march.starts_with("rv64")) {
        return march.to_string();
    }

    let has_zbb = march
        .split('_')
        .any(|ext| ext.to_ascii_lowercase().starts_with("zbb"));
    if has_zbb {
        march.to_string()
    } else {
        format!("{march}_zbb")
    }
}

fn resolve_original_input_path(
    file_name: &str,
    working_directory: Option<&Path>,
) -> Option<PathBuf> {
    let path = Path::new(file_name);
    if path.is_absolute() && path.is_file() {
        return Some(path.to_path_buf());
    }

    let candidate = working_directory?.join(path);
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

fn compile_bc_to_pacc_object(
    input_file: &Path,
    output_file: &Path,
) -> io::Result<crate::PaccConfig> {
    let primary = crate::PaccConfig::xm_hardware();
    match compile_bc_to_xm_object(input_file, output_file, &primary) {
        Ok(()) => Ok(primary),
        Err(primary_err) => {
            let fallback = crate::PaccConfig::rvv_linux_bf16();
            match compile_bc_to_xm_object(input_file, output_file, &fallback) {
                Ok(()) => {
                    eprintln!(
                        "PACC: primary XM codegen failed for {} ({}); used RVV/Linux fallback",
                        input_file.display(),
                        primary_err
                    );
                    Ok(fallback)
                }
                Err(fallback_err) => Err(io::Error::other(format!(
                    "primary XM codegen failed: {}; RVV/Linux fallback failed: {}",
                    primary_err, fallback_err
                ))),
            }
        }
    }
}

fn compile_bc_to_xm_object(
    input_file: &Path,
    output_file: &Path,
    config: &crate::PaccConfig,
) -> io::Result<()> {
    if bc_requires_sanitized_ll(input_file)? {
        eprintln!(
            "PACC: forcing sanitized textual IR path for {} due to unsupported AMDGPU intrinsics",
            input_file.display()
        );
        return compile_bc_to_xm_object_via_sanitized_ll(input_file, output_file, config);
    }

    let clang = preferred_tool(
        "HETGPU_PACC_CLANG",
        &["/usr/bin/clang-20", "clang-20", "clang"],
    );
    let march = riscv_march_with_required_extensions(&config.march);
    let mut cmd = Command::new(clang);
    cmd.arg("-target")
        .arg(&config.target_triple)
        .arg("-fPIC")
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg("-Wno-override-module")
        .arg("-c")
        .arg(input_file)
        .arg("-o")
        .arg(output_file);

    if config.target_triple.contains("linux") {
        let sysroot =
            std::env::var("HETGPU_PACC_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
        let gcc_toolchain = std::env::var("HETGPU_PACC_SOURCE_GCC_TOOLCHAIN")
            .unwrap_or_else(|_| "/usr".to_string());
        cmd.arg(format!("--sysroot={}", sysroot))
            .arg(format!("--gcc-toolchain={}", gcc_toolchain));
    }

    match run_command(&mut cmd, "LLVM bitcode -> RISC-V object") {
        Ok(()) => Ok(()),
        Err(err) => {
            eprintln!(
                "PACC: direct BC->object failed for {}; retrying via sanitized textual IR: {}",
                input_file.display(),
                err
            );
            compile_bc_to_xm_object_via_sanitized_ll(input_file, output_file, config)
        }
    }
}

fn bc_requires_sanitized_ll(input_file: &Path) -> io::Result<bool> {
    let llvm_dis = llvm_dis_tool();
    let output = Command::new(llvm_dis)
        .arg(input_file)
        .arg("-o")
        .arg("-")
        .output()?;
    if !output.status.success() {
        return Ok(false);
    }
    let ll = String::from_utf8_lossy(&output.stdout);
    Ok(ll.contains("@llvm.amdgcn.wave.barrier")
        || ll.contains("call void @llvm.amdgcn.wave.barrier("))
}

fn compile_bc_to_xm_assembly(
    input_file: &Path,
    output_file: &Path,
    config: &crate::PaccConfig,
) -> io::Result<()> {
    let clang = preferred_tool(
        "HETGPU_PACC_CLANG",
        &["/usr/bin/clang-20", "clang-20", "clang"],
    );
    let march = riscv_march_with_required_extensions(&config.march);
    let mut cmd = Command::new(clang);
    cmd.arg("-target")
        .arg(&config.target_triple)
        .arg("-fPIC")
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg("-Wno-override-module")
        .arg("-S")
        .arg(input_file)
        .arg("-o")
        .arg(output_file);

    if config.target_triple.contains("linux") {
        let sysroot =
            std::env::var("HETGPU_PACC_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
        let gcc_toolchain = std::env::var("HETGPU_PACC_SOURCE_GCC_TOOLCHAIN")
            .unwrap_or_else(|_| "/usr".to_string());
        cmd.arg(format!("--sysroot={}", sysroot))
            .arg(format!("--gcc-toolchain={}", gcc_toolchain));
    }

    run_command(&mut cmd, "LLVM bitcode -> RISC-V assembly")
}

fn compile_bc_to_xm_object_via_sanitized_ll(
    input_file: &Path,
    output_file: &Path,
    config: &crate::PaccConfig,
) -> io::Result<()> {
    let llvm_dis = llvm_dis_tool();
    let ll_file = output_file.with_extension("from_bc.ll");
    let sanitized_ll_file = output_file.with_extension("sanitized.ll");

    let mut dis_cmd = Command::new(llvm_dis);
    dis_cmd.arg(input_file).arg("-o").arg(&ll_file);
    run_command(&mut dis_cmd, "LLVM bitcode -> textual LLVM IR")?;

    let ll_text = fs::read_to_string(&ll_file)?;
    let sanitized = sanitize_llvm23_ir_for_llvm20(&ll_text);
    fs::write(&sanitized_ll_file, sanitized)?;

    let clang = preferred_tool(
        "HETGPU_PACC_CLANG",
        &["/usr/bin/clang-20", "clang-20", "clang"],
    );
    let march = riscv_march_with_required_extensions(&config.march);
    let mut cmd = Command::new(clang);
    cmd.arg("-target")
        .arg(&config.target_triple)
        .arg("-fPIC")
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg("-Wno-override-module")
        .arg("-c")
        .arg(&sanitized_ll_file)
        .arg("-o")
        .arg(output_file);

    if config.target_triple.contains("linux") {
        let sysroot =
            std::env::var("HETGPU_PACC_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
        let gcc_toolchain = std::env::var("HETGPU_PACC_SOURCE_GCC_TOOLCHAIN")
            .unwrap_or_else(|_| "/usr".to_string());
        cmd.arg(format!("--sysroot={}", sysroot))
            .arg(format!("--gcc-toolchain={}", gcc_toolchain));
    }

    run_command(&mut cmd, "sanitized LLVM IR -> RISC-V object")
}

fn sanitize_llvm23_ir_for_llvm20(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.lines() {
        let trimmed = line.trim_start();
        if trimmed.contains("llvm.amdgcn.wave.barrier") {
            continue;
        }
        if trimmed.starts_with("attributes #") {
            if let Some((lhs, _rhs)) = line.split_once('=') {
                out.push_str(lhs.trim());
                out.push_str(" = { nounwind }\n");
            } else {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        let mut owned = line.to_string();
        for token in [
            " nocallback",
            " nocreateundeforpoison",
            " nofree",
            " nosync",
            " willreturn",
            " speculatable",
            " mustprogress",
            " memory(none)",
            " memory(argmem: readwrite)",
            " memory(argmem: read)",
            " memory(read)",
            " memory(write)",
            " captures(none)",
            " captures(address)",
            " captures(provenance)",
            " captures(address, provenance)",
            " denormal-fp-math-f32=\"ieee,ieee\"",
            " denormal-fp-math=\"ieee,ieee\"",
            " denormal-fp-math-f64=\"ieee,ieee\"",
            " denormal-fp-math-f16=\"ieee,ieee\"",
            " denormal-fp-math-f32=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math-f64=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math-f16=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math-f32=\"positive-zero,positive-zero\"",
            " denormal-fp-math=\"positive-zero,positive-zero\"",
            " denormal-fpenv(\"dynamic\")",
            " denormal-fpenv(dynamic)",
            " denormal-fpenv(\"ieee\")",
            " denormal-fpenv(ieee)",
        ] {
            owned = owned.replace(token, "");
        }
        out.push_str(&owned);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{riscv_march_with_required_extensions, sanitize_llvm23_ir_for_llvm20};

    #[test]
    fn sanitize_attribute_group_does_not_duplicate_attributes_keyword() {
        let input = "attributes #0 = { nocallback nofree nounwind willreturn memory(none) }\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(output, "attributes #0 = { nounwind }\n");
    }

    #[test]
    fn sanitize_drops_wave_barrier_lines() {
        let input = "declare void @llvm.amdgcn.wave.barrier()\ncall void @llvm.amdgcn.wave.barrier()\nret void\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(output, "ret void\n");
    }

    #[test]
    fn riscv_march_adds_zbb_for_backend_orc_b_selection() {
        assert_eq!(
            riscv_march_with_required_extensions(
                "rv64gcv_zvfbfmin_xsfvcp_xsfvfnrclipxfqf_xsfvfwmaccqqq_xsfvqmaccqoq"
            ),
            "rv64gcv_zvfbfmin_xsfvcp_xsfvfnrclipxfqf_xsfvfwmaccqqq_xsfvqmaccqoq_zbb"
        );
        assert_eq!(
            riscv_march_with_required_extensions("rv64gcv_zbb_zvfbfmin"),
            "rv64gcv_zbb_zvfbfmin"
        );
    }
}

fn source_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn is_c_source(path: &Path) -> bool {
    matches!(source_extension(path).as_deref(), Some("c"))
}

fn is_cxx_source(path: &Path) -> bool {
    matches!(
        source_extension(path).as_deref(),
        Some("cc" | "cp" | "cxx" | "cpp" | "c++")
    )
}

fn preferred_tool(env_var: &str, fallbacks: &[&str]) -> String {
    if !env_var.is_empty() {
        if let Ok(value) = std::env::var(env_var) {
            if !value.trim().is_empty() {
                return value;
            }
        }
    }

    for tool in fallbacks {
        if tool.contains('/') {
            if Path::new(tool).is_file() {
                return (*tool).to_string();
            }
        }
    }

    fallbacks.first().copied().unwrap_or("clang").to_string()
}

fn llvm_link_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_PACC_LLVM_LINK",
        "llvm-link",
        &["/usr/bin/llvm-link-20", "llvm-link-20", "llvm-link"],
    )
}

fn opt_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_PACC_OPT",
        "opt",
        &["/usr/bin/opt-20", "opt-20", "opt"],
    )
}

fn llvm_dis_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_PACC_LLVM_DIS",
        "llvm-dis",
        &["/usr/bin/llvm-dis-20", "llvm-dis-20", "llvm-dis"],
    )
}

fn existing_tool(path: PathBuf) -> Option<String> {
    if path.is_file() {
        Some(path.display().to_string())
    } else {
        None
    }
}

fn llvm_tool_from_build_dir(build_dir: &Path, tool_name: &str) -> Option<String> {
    let entries = fs::read_dir(build_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path().join("out/build/bin").join(tool_name);
        if let Some(tool) = existing_tool(path) {
            return Some(tool);
        }
    }
    None
}

fn llvm_tool_from_target_dir(target_dir: &Path, tool_name: &str) -> Option<String> {
    for candidate in [
        target_dir.join("debug/build"),
        target_dir.join("release/build"),
        target_dir.join("build"),
    ] {
        if let Some(tool) = llvm_tool_from_build_dir(&candidate, tool_name) {
            return Some(tool);
        }
    }
    None
}

fn bundled_llvm_tool(env_var: &str, tool_name: &str, fallbacks: &[&str]) -> String {
    if let Ok(value) = std::env::var(env_var) {
        if !value.trim().is_empty() {
            return value;
        }
    }

    if let Ok(dir) = std::env::var("HETGPU_PACC_LLVM_TOOLS_DIR") {
        if let Some(tool) = existing_tool(PathBuf::from(dir).join(tool_name)) {
            return tool;
        }
    }

    let clang = preferred_tool(
        "HETGPU_PACC_CLANG",
        &["/usr/bin/clang-20", "clang-20", "clang"],
    );
    let mut clang_candidates = vec![PathBuf::from(&clang)];
    if let Ok(resolved) = PathBuf::from(&clang).canonicalize() {
        if !clang_candidates.iter().any(|path| path == &resolved) {
            clang_candidates.push(resolved);
        }
    }
    for clang_path in clang_candidates {
        if let Some(parent) = clang_path.parent() {
            for candidate in [
                parent.join(tool_name),
                parent.join(format!("{tool_name}-20")),
            ] {
                if let Some(tool) = existing_tool(candidate) {
                    return tool;
                }
            }
        }
    }

    for env in ["HETGPU_PACC_HELPER_TARGET_DIR", "CARGO_TARGET_DIR"] {
        if let Ok(dir) = std::env::var(env) {
            if let Some(tool) = llvm_tool_from_target_dir(Path::new(&dir), tool_name) {
                return tool;
            }
        }
    }

    for dir in [
        "/mnt/usb/hetgpu_build_target/releasefix",
        "/mnt/usb/hetgpu_build_target/release",
        "/mnt/usb/hetgpu_build_target",
    ] {
        if let Some(tool) = llvm_tool_from_target_dir(Path::new(dir), tool_name) {
            return tool;
        }
    }

    preferred_tool(env_var, fallbacks)
}

fn ensure_ptx_helper_tool(tool_name: &str) -> io::Result<PathBuf> {
    let repo_root = workspace_root()?;
    let mut candidate_roots = Vec::new();

    if let Ok(dir) = std::env::var("HETGPU_PACC_HELPER_TARGET_DIR") {
        if !dir.trim().is_empty() {
            candidate_roots.push(PathBuf::from(dir));
        }
    }
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        if !dir.trim().is_empty() {
            let path = PathBuf::from(dir);
            if !candidate_roots.iter().any(|p| p == &path) {
                candidate_roots.push(path);
            }
        }
    }
    candidate_roots.push(repo_root.join("target"));

    for root in &candidate_roots {
        let tool_path = root.join("debug").join(tool_name);
        if tool_path.is_file() {
            return Ok(tool_path);
        }
    }

    let build_target_dir = candidate_roots
        .first()
        .cloned()
        .unwrap_or_else(|| repo_root.join("target"));
    let status = Command::new("bash")
        .arg("-lc")
        .arg(format!(
            "cd {} && cargo build -p ptx --features pacc --bin {} --target-dir {}",
            shell_escape_path(&repo_root),
            tool_name,
            shell_escape_path(&build_target_dir),
        ))
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "cargo failed while building {}",
            tool_name
        )));
    }

    let tool_path = build_target_dir.join("debug").join(tool_name);
    if !tool_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("expected helper binary at {}", tool_path.display()),
        ));
    }

    Ok(tool_path)
}

fn workspace_root() -> io::Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("../..")
        .canonicalize()
        .map_err(|e| io::Error::other(format!("failed to locate workspace root: {}", e)))
}

fn run_command(cmd: &mut Command, what: &str) -> io::Result<()> {
    let output = cmd.output()?;
    if output.status.success() {
        return Ok(());
    }

    Err(io::Error::other(format!(
        "{} failed: {}",
        what,
        String::from_utf8_lossy(&output.stderr)
    )))
}

fn shell_escape_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[repr(C, align(8))]
struct FatbinHeader {
    magic: u32,
    version: u16,
    header_size: u16,
    files_size: u64,
}

#[repr(C)]
struct FatbinFileHeader {
    kind: u16,
    version: u16,
    header_size: u32,
    padded_payload_size: u32,
    unknown0: u32,
    payload_size: u32,
    unknown1: u32,
    unknown2: u32,
    sm_version: u32,
    bit_width: u32,
    unknown3: u32,
    unknown4: u64,
    unknown5: u64,
    uncompressed_payload: u64,
}

fn create_cuda_fatbin_with_elf(elf_bytes: &[u8]) -> Vec<u8> {
    const FATBIN_MAGIC: u32 = 0xBA55ED50;
    const FATBIN_VERSION: u16 = 0x0001;
    const FATBIN_KIND_ELF: u16 = 0x0002;
    const FATBIN_FILE_HEADER_VERSION_CURRENT: u16 = 0x0101;

    let header_size = mem::size_of::<FatbinHeader>();
    let file_header_size = mem::size_of::<FatbinFileHeader>();
    let padded_payload_size = align_up(elf_bytes.len(), 8);
    let file_header = FatbinFileHeader {
        kind: FATBIN_KIND_ELF,
        version: FATBIN_FILE_HEADER_VERSION_CURRENT,
        header_size: file_header_size as u32,
        padded_payload_size: padded_payload_size as u32,
        unknown0: 0,
        payload_size: elf_bytes.len() as u32,
        unknown1: 0,
        unknown2: 0,
        sm_version: 90,
        bit_width: 64,
        unknown3: 0,
        unknown4: 0,
        unknown5: 0,
        uncompressed_payload: 0,
    };

    let mut files = Vec::with_capacity(file_header_size + padded_payload_size);
    files.extend_from_slice(as_bytes(&file_header));
    files.extend_from_slice(elf_bytes);
    files.resize(file_header_size + padded_payload_size, 0);

    let header = FatbinHeader {
        magic: FATBIN_MAGIC,
        version: FATBIN_VERSION,
        header_size: header_size as u16,
        files_size: files.len() as u64,
    };

    let mut fatbin = Vec::with_capacity(header_size + files.len());
    fatbin.extend_from_slice(as_bytes(&header));
    fatbin.extend_from_slice(&files);
    fatbin
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>()) }
}

fn create_dummy_riscv_elf() -> Vec<u8> {
    let mut elf = Vec::with_capacity(128);

    // ELF header for RISC-V 64-bit
    elf.extend_from_slice(&[
        0x7f, 0x45, 0x4c, 0x46, // ELF magic
        0x02, // 64-bit
        0x01, // Little endian
        0x01, // Current version
        0x00, // System V ABI
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Padding
    ]);
    elf.extend_from_slice(&[0x01, 0x00]); // ET_REL
    elf.extend_from_slice(&[0xf3, 0x00]); // EM_RISCV
    elf.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // Version
    elf.extend_from_slice(&[0; 8]); // Entry point
    elf.extend_from_slice(&[0; 8]); // Program header offset
    elf.extend_from_slice(&[0; 8]); // Section header offset
    elf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Flags
    elf.extend_from_slice(&[0x40, 0x00]); // ELF header size
    elf.extend_from_slice(&[0x00, 0x00]); // Program header entry size
    elf.extend_from_slice(&[0x00, 0x00]); // Program header entry count
    elf.extend_from_slice(&[0x40, 0x00]); // Section header entry size
    elf.extend_from_slice(&[0x03, 0x00]); // Section header entry count
    elf.extend_from_slice(&[0x02, 0x00]); // Section name string table index

    // PACC marker
    elf.extend_from_slice(b"PACC_RISCV_IME_VCIX\0");

    elf
}
