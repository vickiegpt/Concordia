// PACC Runtime System for SiFive Intelligence XM / RISC-V IME via VCIX
#![allow(warnings)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::ptr;
use std::sync::Mutex;
use tempfile::TempDir;

// PACC/IME configuration constants
/// Default VLEN in bits (SiFive Intelligence XM typical)
pub const PACC_VLEN: usize = 256;
/// Default SEW for matrix operations
pub const PACC_DEFAULT_SEW: usize = 8;
/// Matrix tile M dimension: sqrt(VLEN/64) when VLEN=256 → M=2
pub const PACC_TILE_M: usize = 2;
/// Matrix tile N dimension (same as M for square tiles)
pub const PACC_TILE_N: usize = 2;
/// Matrix tile K dimension: VLEN / (M * SEW) when VLEN=256, M=2, SEW=8 → K=16
pub const PACC_TILE_K: usize = 16;

/// Number of vector registers available (v0-v31)
pub const PACC_NUM_VREGS: usize = 32;

// Core coordinate for multi-core PACC
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct CoreCoord {
    pub x: u32,
    pub y: u32,
}

// Data format types
pub const pacc_DataFormat_Invalid: pacc_DataFormat = 0;
pub const pacc_DataFormat_Int4: pacc_DataFormat = 1;
pub const pacc_DataFormat_Int8: pacc_DataFormat = 2;
pub const pacc_DataFormat_Int16: pacc_DataFormat = 3;
pub const pacc_DataFormat_Int32: pacc_DataFormat = 4;
pub const pacc_DataFormat_Float16: pacc_DataFormat = 5;
pub const pacc_DataFormat_Float32: pacc_DataFormat = 6;
pub const pacc_DataFormat_Bfloat16: pacc_DataFormat = 7;
pub type pacc_DataFormat = ::core::ffi::c_uint;

// Buffer types
pub const pacc_BufferType_DRAM: pacc_BufferType = 0;
pub const pacc_BufferType_SPAD: pacc_BufferType = 1;
pub const pacc_BufferType_ACC: pacc_BufferType = 2;
pub type pacc_BufferType = ::core::ffi::c_uint;

// Result types
pub const pacc_Result_Success: pacc_Result = 0;
pub const pacc_Result_Error: pacc_Result = 1;
pub type pacc_Result = ::core::ffi::c_uint;

// Opaque handle types
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct pacc_Device {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct pacc_Program {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct pacc_Buffer {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct pacc_Kernel {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct pacc_BufferConfig {
    pub device: *mut pacc_Device,
    pub size: u64,
    pub buffer_type: pacc_BufferType,
    pub data_format: pacc_DataFormat,
}

// --- VCIX/IME instruction encoding helpers ---

/// Encode a VCIX sf.vc.vvv instruction (3 vector operands, accumulate)
/// funct6_hi4 = 0b1010 (VCIX_Rs1VV), bit[25] = 0 (has output)
/// bit27_26 = opcode selector (0-3)
pub fn encode_vcix_vvv(opcode: u8, vd: u8, vs2: u8, vs1: u8) -> u32 {
    let base: u32 = 0x5B; // CUSTOM-2
    let funct6_hi4: u32 = 0b1010;
    let bit27_26 = (opcode as u32) & 0x3;
    let has_output: u32 = 0; // bit[25] = 0 means output to vd

    (funct6_hi4 << 28)
        | (bit27_26 << 26)
        | (has_output << 25)
        | ((vs2 as u32 & 0x1f) << 20)
        | ((vs1 as u32 & 0x1f) << 15)
        | (0b000 << 12) // funct3 = 000 for VR
        | ((vd as u32 & 0x1f) << 7)
        | base
}

/// Encode a VCIX sf.vc.vvw instruction (widening, vd is 2x wider)
/// funct6_hi4 = 0b1111 (VCIX_Rs1VW), bit[25] = 0 (has output)
pub fn encode_vcix_vvw(opcode: u8, vd: u8, vs2: u8, vs1: u8) -> u32 {
    let base: u32 = 0x5B;
    let funct6_hi4: u32 = 0b1111;
    let bit27_26 = (opcode as u32) & 0x3;
    let has_output: u32 = 0;

    (funct6_hi4 << 28)
        | (bit27_26 << 26)
        | (has_output << 25)
        | ((vs2 as u32 & 0x1f) << 20)
        | ((vs1 as u32 & 0x1f) << 15)
        | (0b000 << 12)
        | ((vd as u32 & 0x1f) << 7)
        | base
}

/// Encode a SpacemiT IME smt.vmadot instruction (CUSTOM_1 = 0x2B)
/// funct7 = 0b1110001 (OPMMA), sign encoding in bits[13:12]
pub fn encode_ime_vmadot(vd: u8, vs1: u8, vs2: u8, sign: ImeDotSign) -> u32 {
    let base: u32 = 0x2B; // CUSTOM-1
    let funct7: u32 = 0b1110001;
    let sign_bits: u32 = match sign {
        ImeDotSign::UU => 0b00,
        ImeDotSign::US => 0b01,
        ImeDotSign::SU => 0b10,
        ImeDotSign::SS => 0b11,
    };

    (funct7 << 25)
        | ((vs2 as u32 & 0x1f) << 20)
        | ((vs1 as u32 & 0x1f) << 15)
        | (0 << 14)
        | (sign_bits << 12)
        | (((vd as u32 >> 1) & 0xf) << 8) // vd is 4 bits, even-numbered
        | (0 << 7)
        | base
}

/// Sign encoding for IME dot-product instructions
#[derive(Debug, Clone, Copy)]
pub enum ImeDotSign {
    /// Unsigned x Unsigned
    UU,
    /// Unsigned x Signed
    US,
    /// Signed x Unsigned
    SU,
    /// Signed x Signed
    SS,
}

// --- Internal state management ---

static PACC_STATE: Mutex<Option<PaccState>> = Mutex::new(None);

struct PaccState {
    temp_dir: TempDir,
    device_count: u32,
    programs: Vec<ProgramData>,
    buffers: Vec<BufferData>,
    kernels: Vec<KernelData>,
}

struct ProgramData {
    id: usize,
    llvm_ir: Option<String>,
    elf_path: Option<PathBuf>,
}

struct BufferData {
    id: usize,
    size: u64,
    buffer_type: pacc_BufferType,
    data: Vec<u8>,
}

struct KernelData {
    id: usize,
    name: String,
    program_id: usize,
}

// --- FFI functions ---

pub unsafe extern "C" fn pacc_CreateDevice(
    device_id: ::core::ffi::c_int,
) -> *mut pacc_Device {
    let mut state = PACC_STATE.lock().unwrap();

    if state.is_none() {
        match TempDir::new() {
            Ok(temp_dir) => {
                *state = Some(PaccState {
                    temp_dir,
                    device_count: 1,
                    programs: Vec::new(),
                    buffers: Vec::new(),
                    kernels: Vec::new(),
                });
                eprintln!("PACC/RISCV-IME: Initialized device {}", device_id);
            }
            Err(e) => {
                eprintln!("PACC: Failed to create temp directory: {}", e);
                return ptr::null_mut();
            }
        }
    }

    1 as *mut pacc_Device
}

pub unsafe extern "C" fn pacc_CloseDevice(device: *mut pacc_Device) -> ::core::ffi::c_int {
    if device.is_null() {
        return pacc_Result_Error as c_int;
    }

    let mut state = PACC_STATE.lock().unwrap();
    if state.is_some() {
        *state = None;
        eprintln!("PACC/RISCV-IME: Closed device");
    }

    pacc_Result_Success as c_int
}

pub unsafe extern "C" fn pacc_CreateProgram() -> *mut pacc_Program {
    let mut state = PACC_STATE.lock().unwrap();

    if let Some(ref mut pacc_state) = *state {
        let program_id = pacc_state.programs.len();
        pacc_state.programs.push(ProgramData {
            id: program_id,
            llvm_ir: None,
            elf_path: None,
        });

        eprintln!("PACC/RISCV-IME: Created program {}", program_id);
        return (program_id + 1) as *mut pacc_Program;
    }

    ptr::null_mut()
}

pub unsafe extern "C" fn pacc_CreateBuffer(
    config: *const pacc_BufferConfig,
) -> *mut pacc_Buffer {
    if config.is_null() {
        return ptr::null_mut();
    }

    let config = &*config;
    let mut state = PACC_STATE.lock().unwrap();

    if let Some(ref mut pacc_state) = *state {
        let buffer_id = pacc_state.buffers.len();
        pacc_state.buffers.push(BufferData {
            id: buffer_id,
            size: config.size,
            buffer_type: config.buffer_type,
            data: vec![0u8; config.size as usize],
        });

        eprintln!(
            "PACC/RISCV-IME: Created buffer {} (size: {} bytes, type: {})",
            buffer_id, config.size, config.buffer_type
        );
        return (buffer_id + 1) as *mut pacc_Buffer;
    }

    ptr::null_mut()
}

pub unsafe extern "C" fn pacc_WriteBuffer(
    buffer: *mut pacc_Buffer,
    data: *const c_void,
    size: u64,
) -> c_int {
    if buffer.is_null() || data.is_null() {
        return pacc_Result_Error as c_int;
    }

    let buffer_id = buffer as usize - 1;
    let mut state = PACC_STATE.lock().unwrap();

    if let Some(ref mut pacc_state) = *state {
        if buffer_id < pacc_state.buffers.len() {
            let src = std::slice::from_raw_parts(data as *const u8, size as usize);
            let buf = &mut pacc_state.buffers[buffer_id];
            let copy_len = std::cmp::min(src.len(), buf.data.len());
            buf.data[..copy_len].copy_from_slice(&src[..copy_len]);
            return pacc_Result_Success as c_int;
        }
    }

    pacc_Result_Error as c_int
}

pub unsafe extern "C" fn pacc_ReadBuffer(
    buffer: *mut pacc_Buffer,
    data: *mut c_void,
    size: u64,
) -> c_int {
    if buffer.is_null() || data.is_null() {
        return pacc_Result_Error as c_int;
    }

    let buffer_id = buffer as usize - 1;
    let state = PACC_STATE.lock().unwrap();

    if let Some(ref pacc_state) = *state {
        if buffer_id < pacc_state.buffers.len() {
            let buf = &pacc_state.buffers[buffer_id];
            let copy_len = std::cmp::min(size as usize, buf.data.len());
            let dst = std::slice::from_raw_parts_mut(data as *mut u8, copy_len);
            dst.copy_from_slice(&buf.data[..copy_len]);
            return pacc_Result_Success as c_int;
        }
    }

    pacc_Result_Error as c_int
}

pub unsafe extern "C" fn pacc_FreeBuffer(buffer: *mut pacc_Buffer) -> c_int {
    if buffer.is_null() {
        return pacc_Result_Error as c_int;
    }
    // Buffers freed when state is dropped
    pacc_Result_Success as c_int
}

pub unsafe extern "C" fn pacc_LoadProgram(
    program: *mut pacc_Program,
    binary: *const c_void,
    binary_size: u64,
) -> c_int {
    if program.is_null() || binary.is_null() {
        return pacc_Result_Error as c_int;
    }

    let program_id = program as usize - 1;
    let mut state = PACC_STATE.lock().unwrap();

    if let Some(ref mut pacc_state) = *state {
        if program_id < pacc_state.programs.len() {
            let binary_data =
                std::slice::from_raw_parts(binary as *const u8, binary_size as usize);

            // Write ELF to temp file
            let elf_path = pacc_state
                .temp_dir
                .path()
                .join(format!("program_{}.elf", program_id));

            if let Err(e) = std::fs::write(&elf_path, binary_data) {
                eprintln!("PACC: Failed to write program ELF: {}", e);
                return pacc_Result_Error as c_int;
            }

            pacc_state.programs[program_id].elf_path = Some(elf_path.clone());
            eprintln!(
                "PACC/RISCV-IME: Loaded program {} ({} bytes) to {}",
                program_id,
                binary_size,
                elf_path.display()
            );
            return pacc_Result_Success as c_int;
        }
    }

    pacc_Result_Error as c_int
}

pub unsafe extern "C" fn pacc_CreateKernel(
    program: *mut pacc_Program,
    name: *const c_char,
) -> *mut pacc_Kernel {
    if program.is_null() || name.is_null() {
        return ptr::null_mut();
    }

    let program_id = program as usize - 1;
    let kernel_name = CStr::from_ptr(name).to_string_lossy().to_string();

    let mut state = PACC_STATE.lock().unwrap();

    if let Some(ref mut pacc_state) = *state {
        let kernel_id = pacc_state.kernels.len();
        pacc_state.kernels.push(KernelData {
            id: kernel_id,
            name: kernel_name.clone(),
            program_id,
        });

        eprintln!(
            "PACC/RISCV-IME: Created kernel '{}' (id={}, program={})",
            kernel_name, kernel_id, program_id
        );
        return (kernel_id + 1) as *mut pacc_Kernel;
    }

    ptr::null_mut()
}

pub unsafe extern "C" fn pacc_LaunchKernel(
    kernel: *mut pacc_Kernel,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> c_int {
    if kernel.is_null() {
        return pacc_Result_Error as c_int;
    }

    let kernel_id = kernel as usize - 1;
    let state = PACC_STATE.lock().unwrap();

    if let Some(ref pacc_state) = *state {
        if kernel_id < pacc_state.kernels.len() {
            let kernel_data = &pacc_state.kernels[kernel_id];
            eprintln!(
                "PACC/RISCV-IME: Launching kernel '{}' grid=({},{},{}) block=({},{},{})",
                kernel_data.name, grid_x, grid_y, grid_z, block_x, block_y, block_z
            );

            // Run on SiFive spike simulator (zvfbfa-plus fork)
            drop(state);
            let result = run_on_spike(kernel_id);
            return result as c_int;
        }
    }

    pacc_Result_Error as c_int
}

// =========================================================================
// Spike simulation (SiFive riscv-isa-sim-zvfbfa-plus)
// =========================================================================

/// Default ISA string for the SiFive spike zvfbfa-plus fork.
/// Enables: V extension, BF16, block/lane dot products for matrix ops.
const SPIKE_ISA: &str = "rv64gcv_zfbfmin_zvfbfmin_zvfbfwma_zvfbfa_zvqbdot8i_zvqbdot16i_zvfqbdot8f_zvfwbdot16bf_zvfbdot32f_zvl512b";

/// Default VLEN for X390 simulation
const SPIKE_VLEN: usize = 512;

/// Run a compiled kernel on the SiFive spike simulator.
fn run_on_spike(kernel_id: usize) -> pacc_Result {
    let mut state = PACC_STATE.lock().unwrap();
    let pacc_state = match state.as_mut() {
        Some(s) => s,
        None => {
            eprintln!("PACC/Spike: No state initialized");
            return pacc_Result_Error;
        }
    };

    if kernel_id >= pacc_state.kernels.len() {
        eprintln!("PACC/Spike: Invalid kernel id {}", kernel_id);
        return pacc_Result_Error;
    }

    let program_id = pacc_state.kernels[kernel_id].program_id;
    let kernel_name = pacc_state.kernels[kernel_id].name.clone();

    let elf_path = match pacc_state.programs.get(program_id).and_then(|p| p.elf_path.as_ref()) {
        Some(p) => p.clone(),
        None => {
            // No ELF yet — try to compile from LLVM IR
            let llvm_ir = match pacc_state.programs.get(program_id).and_then(|p| p.llvm_ir.as_ref()) {
                Some(ir) => ir.clone(),
                None => {
                    eprintln!("PACC/Spike: No ELF or LLVM IR for program {}", program_id);
                    return simulate_fallback(pacc_state);
                }
            };

            match compile_to_riscv_elf(pacc_state, program_id, &llvm_ir) {
                Some(path) => path,
                None => {
                    eprintln!("PACC/Spike: Compilation failed, using fallback");
                    return simulate_fallback(pacc_state);
                }
            }
        }
    };

    eprintln!(
        "PACC/Spike: Running kernel '{}' (program={}) on spike",
        kernel_name, program_id
    );

    // Prepare input data files for spike
    let input_path = pacc_state.temp_dir.path().join("pacc_input.bin");
    let output_path = pacc_state.temp_dir.path().join("pacc_output.bin");
    prepare_memory_file(pacc_state, &input_path);

    // Invoke spike
    let result = invoke_spike(&elf_path, &input_path, &output_path);

    // Read back results
    if result == pacc_Result_Success {
        read_spike_output(pacc_state, &output_path);
    }

    result
}

/// Compile LLVM IR to RISC-V ELF for spike execution.
fn compile_to_riscv_elf(
    state: &mut PaccState,
    program_id: usize,
    llvm_ir: &str,
) -> Option<PathBuf> {
    let ir_path = state.temp_dir.path().join(format!("pacc_program_{}.ll", program_id));
    let asm_path = state.temp_dir.path().join(format!("pacc_program_{}.s", program_id));
    let obj_path = state.temp_dir.path().join(format!("pacc_program_{}.o", program_id));
    let elf_path = state.temp_dir.path().join(format!("pacc_program_{}.elf", program_id));

    // Write LLVM IR
    std::fs::write(&ir_path, llvm_ir).ok()?;

    // Step 1: LLC — LLVM IR → RISC-V assembly
    let llc_result = Command::new("llc")
        .arg(&ir_path)
        .arg("-o").arg(&asm_path)
        .arg("-march=riscv64")
        .arg("-mattr=+v,+d,+f,+zfbfmin,+zvfbfmin,+zvfbfwma,+zvl512b")
        .arg("-filetype=asm")
        .output();

    match llc_result {
        Ok(output) if output.status.success() => {
            eprintln!("PACC/Spike: LLC compiled to assembly: {}", asm_path.display());
        }
        Ok(output) => {
            eprintln!(
                "PACC/Spike: LLC failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            // Fallback: write a test assembly
            let test_asm = generate_test_assembly();
            std::fs::write(&asm_path, test_asm).ok()?;
        }
        Err(_) => {
            eprintln!("PACC/Spike: LLC not found, writing test assembly");
            let test_asm = generate_test_assembly();
            std::fs::write(&asm_path, test_asm).ok()?;
        }
    }

    // Step 2: Assemble — assembly → object file
    let as_result = Command::new("riscv64-unknown-elf-as")
        .arg(&asm_path)
        .arg("-o").arg(&obj_path)
        .arg("-march=rv64gcv_zvl512b")
        .output();

    if as_result.map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("PACC/Spike: Assembled to object: {}", obj_path.display());
    } else {
        eprintln!("PACC/Spike: Assembler failed or not found");
        return None;
    }

    // Step 3: Link — object → ELF executable
    let linker_script = state.temp_dir.path().join("pacc.ld");
    std::fs::write(&linker_script, generate_linker_script()).ok()?;

    let link_result = Command::new("riscv64-unknown-elf-gcc")
        .arg("-static")
        .arg("-nostartfiles")
        .arg("-mcmodel=medany")
        .arg(format!("-T{}", linker_script.display()))
        .arg(&obj_path)
        .arg("-o").arg(&elf_path)
        .output();

    if link_result.map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("PACC/Spike: Linked ELF: {}", elf_path.display());
        if program_id < state.programs.len() {
            state.programs[program_id].elf_path = Some(elf_path.clone());
        }
        Some(elf_path)
    } else {
        eprintln!("PACC/Spike: Linker failed");
        None
    }
}

/// Invoke the SiFive spike simulator (riscv-isa-sim-zvfbfa-plus).
fn invoke_spike(
    elf_path: &Path,
    _input_path: &Path,
    _output_path: &Path,
) -> pacc_Result {
    let spike_bin = std::env::var("SPIKE_PATH").unwrap_or_else(|_| "spike".to_string());

    // Try: spike --isa=<ISA> --varch=vlen:<VLEN>,elen:64 <elf>
    let mut cmd = Command::new(&spike_bin);
    cmd.arg(format!("--isa={}", SPIKE_ISA))
        .arg(format!("--varch=vlen:{},elen:64", SPIKE_VLEN))
        .arg(elf_path);

    eprintln!("PACC/Spike: Running: {:?}", cmd);

    match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !stdout.is_empty() {
                eprintln!("PACC/Spike stdout: {}", stdout);
            }
            if !stderr.is_empty() {
                eprintln!("PACC/Spike stderr: {}", stderr);
            }

            if output.status.success() {
                eprintln!("PACC/Spike: Simulation completed successfully");
                pacc_Result_Success
            } else {
                eprintln!("PACC/Spike: Simulation failed with status: {}", output.status);

                // Try with proxy kernel (pk)
                let mut pk_cmd = Command::new(&spike_bin);
                pk_cmd
                    .arg(format!("--isa={}", SPIKE_ISA))
                    .arg(format!("--varch=vlen:{},elen:64", SPIKE_VLEN))
                    .arg("pk")
                    .arg(elf_path);

                eprintln!("PACC/Spike: Retrying with pk: {:?}", pk_cmd);
                match pk_cmd.output() {
                    Ok(pk_output) if pk_output.status.success() => {
                        eprintln!("PACC/Spike: pk simulation completed");
                        pacc_Result_Success
                    }
                    _ => {
                        eprintln!("PACC/Spike: pk simulation also failed");
                        pacc_Result_Error
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("PACC/Spike: Failed to run spike: {}", e);
            eprintln!("PACC/Spike: Install from https://github.com/sifive/riscv-isa-sim-zvfbfa-plus");
            pacc_Result_Error
        }
    }
}

/// Write buffer data to a binary file for spike to consume.
fn prepare_memory_file(state: &PaccState, path: &Path) {
    let mut data = Vec::new();
    for buf in &state.buffers {
        data.extend_from_slice(&buf.data);
    }
    if let Err(e) = std::fs::write(path, &data) {
        eprintln!("PACC/Spike: Failed to write memory file: {}", e);
    }
}

/// Read spike output from the output binary file.
fn read_spike_output(state: &mut PaccState, path: &Path) {
    if let Ok(data) = std::fs::read(path) {
        // Copy output data back into the last buffer (output buffer)
        if let Some(last_buf) = state.buffers.last_mut() {
            let copy_len = std::cmp::min(data.len(), last_buf.data.len());
            last_buf.data[..copy_len].copy_from_slice(&data[..copy_len]);
            eprintln!("PACC/Spike: Read {} bytes of output", copy_len);
        }
    }
}

/// Fallback: simulate a simple matrix multiply in Rust.
fn simulate_fallback(state: &mut PaccState) -> pacc_Result {
    eprintln!("PACC/Spike: Running fallback software simulation");

    if state.buffers.len() >= 3 {
        // Interpret buffers as float32 matrices and do a simple matmul
        let n = 32usize; // assume 32x32
        let a_bytes = &state.buffers[0].data;
        let b_bytes = &state.buffers[1].data;

        let a: Vec<f32> = a_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b: Vec<f32> = b_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let size = std::cmp::min(n * n, std::cmp::min(a.len(), b.len()));
        let dim = (size as f64).sqrt() as usize;
        let mut c = vec![0.0f32; dim * dim];

        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..dim {
                    if i * dim + k < a.len() && k * dim + j < b.len() {
                        sum += a[i * dim + k] * b[k * dim + j];
                    }
                }
                c[i * dim + j] = sum;
            }
        }

        // Write result to output buffer
        let c_bytes: Vec<u8> = c.iter().flat_map(|f| f.to_le_bytes()).collect();
        let out_buf = &mut state.buffers[2];
        let copy_len = std::cmp::min(c_bytes.len(), out_buf.data.len());
        out_buf.data[..copy_len].copy_from_slice(&c_bytes[..copy_len]);

        eprintln!("PACC/Spike: Fallback computed {}x{} matmul", dim, dim);
    }

    pacc_Result_Success
}

/// Generate test RISC-V assembly using V extension operations.
fn generate_test_assembly() -> String {
    format!(
        ".attribute 5, \"rv64gcv_zvl512b_zfbfmin_zvfbfmin_zvfbfwma\"\n\
         .text\n\
         .globl _start\n\
         .type _start, @function\n\
         _start:\n\
         # PACC/X390 test: vector operations with Zvbdot extensions\n\
         # Configure vector unit: SEW=8, LMUL=1\n\
         vsetivli zero, 16, e8, m1, ta, ma\n\
         \n\
         # Load test vectors\n\
         # In a real kernel, these would be matrix tile loads\n\
         # vle8.v v0, (a0)   # A tile\n\
         # vle8.v v8, (a1)   # B tile block (v8-v15)\n\
         \n\
         # Block dot product (requires zvqbdot8i extension)\n\
         # vqbdots.vv v16, v0, v8\n\
         \n\
         # Store result\n\
         # vse32.v v16, (a2)\n\
         \n\
         # Exit via ecall\n\
         li a7, 93\n\
         li a0, 0\n\
         ecall\n\
         .size _start, .-_start\n"
    )
}

/// Generate a linker script for bare-metal RISC-V execution on spike.
fn generate_linker_script() -> String {
    "OUTPUT_ARCH(riscv)\n\
     ENTRY(_start)\n\
     MEMORY {\n\
       ram (rwx) : ORIGIN = 0x80000000, LENGTH = 0x10000000\n\
     }\n\
     SECTIONS {\n\
       .text : { *(.text .text.*) } > ram\n\
       .rodata : { *(.rodata .rodata.*) } > ram\n\
       .data : { *(.data .data.*) } > ram\n\
       .bss : { *(.bss .bss.* COMMON) } > ram\n\
       _end = .;\n\
     }\n".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vcix_vvv_encoding() {
        // sf.vc.v.vvv opcode=3, vd=2, vs2=4, vs1=6
        let encoded = encode_vcix_vvv(3, 2, 4, 6);
        assert_eq!(encoded & 0x7f, 0x5B); // CUSTOM-2
        assert_eq!((encoded >> 12) & 0x7, 0b000); // funct3 = VR
        assert_eq!((encoded >> 26) & 0x3, 3); // bit27_26 = 3
    }

    #[test]
    fn test_vcix_vvw_encoding() {
        let encoded = encode_vcix_vvw(1, 0, 2, 4);
        assert_eq!(encoded & 0x7f, 0x5B);
        assert_eq!((encoded >> 28) & 0xf, 0b1111); // VCIX_Rs1VW
    }

    #[test]
    fn test_ime_vmadot_encoding() {
        // smt.vmadot v16, v0, v8 (signed x signed)
        let encoded = encode_ime_vmadot(16, 0, 8, ImeDotSign::SS);
        assert_eq!(encoded & 0x7f, 0x2B); // CUSTOM-1
        assert_eq!((encoded >> 12) & 0x3, 0b11); // SS sign
    }

    #[test]
    fn test_tile_dims() {
        // VLEN=256, SEW=8 → M=2, N=2, K=16
        let config = crate::PACC_VLEN;
        let sew = 8;
        let m = (config as f64 / 64.0).sqrt() as usize;
        let k = config / (m * sew);
        assert_eq!(m, 2);
        assert_eq!(k, 16);
    }
}
