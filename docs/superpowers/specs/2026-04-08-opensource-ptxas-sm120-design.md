# Open-Source ptxas for SM120 (Blackwell)

**Date:** 2026-04-08
**Status:** Approved
**Scope:** Build an open-source PTX assembler targeting NVIDIA SM120 Blackwell GPUs, producing native CUBIN (SASS) binaries without NVIDIA's proprietary ptxas toolchain.

## Goals

1. Replace NVIDIA's proprietary `ptxas` with an open-source implementation for SM120
2. Produce valid CUBIN ELF binaries that `cuModuleLoadData` can load on Blackwell hardware
3. Integrate as both a standalone CLI tool and a library backend in the existing comgr architecture
4. Validate correctness via disassemble round-trip (assemble -> disassemble -> compare)

## Non-Goals

- Supporting SM architectures older than SM120 (can be added later)
- Achieving identical binary output to NVIDIA's ptxas (functional equivalence is sufficient)
- Replacing the NVIDIA driver's JIT compiler

## Architecture

### Three Crates

```
nvidia_sass/              -- Core library (new crate)
  src/
    lib.rs
    encoding/
      mod.rs              -- Encoder trait + dispatch
      sm120.rs            -- SM120 instruction encoding tables
      sm120_tensor.rs     -- TCGen05, HMMA, IMMA encoding
      sm120_memory.rs     -- LDG, STG, LDS, STS, atomics encoding
      sm120_special.rs    -- TEX, TMA, async copy encoding
      control_codes.rs    -- Stall count, barriers, reuse flag encoding
    isel/
      mod.rs              -- Instruction selection framework
      patterns.rs         -- LLVM IR -> SASS pattern matching
      lowering.rs         -- Complex op lowering (e.g., div -> rcp+mul)
    regalloc/
      mod.rs              -- Register allocator
      linear_scan.rs      -- Linear scan allocator (MVP)
      liveness.rs         -- Liveness analysis
    scheduler/
      mod.rs              -- Instruction scheduler
      latency.rs          -- SM120 instruction latency tables
      scoreboard.rs       -- Dependency scoreboard for barrier assignment
    cubin_builder/
      mod.rs              -- CUBIN ELF builder
      elf.rs              -- ELF64 generation (machine type 190)
      nv_info.rs          -- .nv.info section generation
      nv_constant.rs      -- .nv.constant section generation
      sections.rs         -- .text, symbol table, section headers
    roundtrip/
      mod.rs              -- Assemble -> disassemble -> compare validation

ptxas/                    -- CLI tool (replace existing stub)
  src/
    main.rs               -- ptxas CLI: parse PTX, compile, write CUBIN

comgr/
  src/
    lib.rs                -- Add #[cfg(feature = "nvidia")] compile_bitcode()
```

### Compilation Pipeline

```
PTX source text
  |
  v
ptx_parser::parse_module_checked()          [existing]
  |
  v
PTX AST (ptx_parser types)
  |
  v
ptx::to_llvm_module() with NVPTX variant    [modify existing]
  - NVPTX address spaces (0=generic, 1=global, 3=shared, 4=const, 5=local)
  - PTX_Kernel calling convention (71) for kernels
  - NVPTX-compatible intrinsics for special registers
  |
  v
LLVM IR Module (NVPTX target triple)
  |
  v
nvidia_sass::isel::select()                 [new]
  - Pattern-match LLVM IR instructions to SASS opcodes
  - Lower complex operations (div, rem, sqrt)
  - Handle address space casts
  - Map LLVM intrinsics to special instructions (barrier, shfl, etc.)
  |
  v
Vec<SassInstruction> (virtual registers)
  |
  v
nvidia_sass::regalloc::allocate()           [new]
  - Liveness analysis on basic blocks
  - Linear scan register allocation: R0-R255, P0-P6
  - Spill to local memory if needed
  - Track register count for .nv.info
  |
  v
Vec<SassInstruction> (physical registers)
  |
  v
nvidia_sass::scheduler::schedule()          [new]
  - Compute instruction latencies
  - Assign stall counts (control code bits [3:0])
  - Assign read/write barriers (control code bits [13:5])
  - Set wait barrier masks (control code bits [16:11])
  - Set register reuse flags (control code bits [20:17])
  - Set yield hints (control code bit [4])
  |
  v
Vec<ScheduledSassInstruction>
  |
  v
nvidia_sass::encoding::encode()             [new]
  - Map opcode + operands + modifiers -> 128-bit encoding
  - Lower 64 bits: opcode, registers, immediates, flags
  - Upper 64 bits: control codes from scheduler
  |
  v
Vec<[u8; 16]> (raw 128-bit encoded instructions)
  |
  v
nvidia_sass::cubin_builder::build()         [new]
  - ELF64 header (machine=190, little-endian)
  - .text.<kernel> sections with encoded SASS
  - .nv.info.<kernel> sections (reg count, shared mem, max threads)
  - .nv.constant0.<kernel> sections (kernel parameters)
  - .nv.constant2 section (global constants)
  - Symbol table (kernel entries as STT_FUNC, global)
  - Section headers and string tables
  |
  v
CUBIN ELF binary (Vec<u8>)
```

## SM120 Instruction Encoding

### 128-bit Instruction Format (SM70+ / SM120)

```
Bits [127:64] - Control codes:
  [127:121] - Reserved / extended control
  [120:117] - Register reuse flags (4 bits, one per source)
  [116:111] - Wait barrier mask (6 bits, which barriers to wait on)
  [110:108] - Read barrier (3 bits, barrier index for read dep)
  [107:105] - Write barrier (3 bits, barrier index for write dep)
  [104]     - Yield flag
  [103:100] - Stall count (4 bits, 0-15 cycles)
  [99:64]   - Extended opcode / modifier bits

Bits [63:0] - Instruction:
  [63:52]   - Primary opcode (12 bits)
  [51:0]    - Operand encoding (varies by opcode class)
```

Operand field layout varies per instruction class. Common patterns:

```
ALU (3-operand):
  [7:0]    = Destination register (R0-R255, RZ=255)
  [15:8]   = Source 1 register
  [27:20]  = Source 2 register or immediate[7:0]
  [35:28]  = Source 3 register (for FMA-like ops)
  [19:16]  = Predicate: [16]=negation, [19:17]=register (PT=7)
  [51:36]  = Modifiers, immediate extension, flags

Memory (load/store):
  [7:0]    = Data register
  [15:8]   = Address register
  [51:20]  = Offset (32-bit immediate)
  [19:16]  = Predicate
```

### Encoding Table Structure

Each opcode has an encoding descriptor:

```rust
struct OpcodeEncoding {
    opcode_bits: u16,           // bits [63:52]
    format: InstructionFormat,  // ALU3, ALU2, Memory, Control, Tensor, etc.
    modifiers: &[ModifierField], // bit positions for .E, .U32, .STRONG, etc.
    operand_fields: OperandLayout,
}
```

### Reverse Engineering Strategy

Build encoding tables by:
1. Write minimal PTX kernels exercising each instruction
2. Compile with NVIDIA ptxas for sm_120
3. Disassemble with nvdisasm to get text + binary
4. Extract bit patterns, correlate with operand values
5. Encode as Rust data tables in `encoding/sm120.rs`

For instructions shared with SM90 (most ALU/memory), start from known SM90 encodings. Focus reverse engineering effort on SM120-new instructions: TCGen05, FP6/FP4, new TMA ops.

## LLVM IR Emission (NVPTX Variant)

The existing `ptx/src/pass/llvm/emit.rs` uses AMDGPU conventions. For NVIDIA, we need:

### Address Spaces
```
NVPTX:                    AMDGPU (current):
0 = Generic               0 = Generic (flat)
1 = Global                1 = Global
3 = Shared                3 = Shared (LDS)
4 = Constant              4 = Constant
5 = Local (private)       5 = Private
```

The numeric values happen to match, but the calling conventions and intrinsics differ.

### Calling Convention
- Kernels: `PTX_Kernel` (LLVM CC 71) instead of `AMDGPUKernelCallConv`
- Functions: C calling convention (same)

### Special Registers
Map PTX special registers to NVPTX intrinsics:
- `%tid.x` -> `llvm.nvvm.read.ptx.sreg.tid.x`
- `%ctaid.x` -> `llvm.nvvm.read.ptx.sreg.ctaid.x`
- `%nctaid.x` -> `llvm.nvvm.read.ptx.sreg.nctaid.x`
- `%laneid` -> `llvm.nvvm.read.ptx.sreg.laneid`
- etc.

### Implementation
Add a new feature flag `nvidia` and conditionally compile the NVPTX variant in emit.rs:

```rust
#[cfg(feature = "nvidia")]
const GENERIC_ADDRESS_SPACE: u32 = 0;
#[cfg(feature = "nvidia")]
const GLOBAL_ADDRESS_SPACE: u32 = 1;
// ... same values, but different kernel CC and intrinsics
```

## CUBIN ELF Format

### ELF Header
```
e_ident: 7f 45 4c 46 02 01 01 (ELF64, little-endian, ELFOSABI_NONE)
e_type: ET_EXEC (2)
e_machine: 190 (EM_CUDA)
e_flags: encode SM version + PTX version
```

### Required Sections

| Section | Type | Content |
|---------|------|---------|
| `.text.<kernel>` | SHT_PROGBITS | Encoded SASS instructions |
| `.nv.info` | 0x70000000 | Global CUDA info attributes |
| `.nv.info.<kernel>` | 0x70000000 | Per-kernel attributes (regcount, smem, maxthreads) |
| `.nv.constant0.<kernel>` | SHT_PROGBITS | Kernel parameters (constant bank 0) |
| `.nv.constant2` | SHT_PROGBITS | Module-level constants |
| `.symtab` | SHT_SYMTAB | Symbol table |
| `.strtab` | SHT_STRTAB | String table |
| `.shstrtab` | SHT_STRTAB | Section header string table |

### .nv.info Attribute Format
```
struct NvInfoAttribute {
    attr_type: u16,    // e.g., 0x0401 = REGCOUNT
    attr_size: u16,    // payload size in bytes
    attr_data: [u8],   // payload (padded to 4-byte alignment)
}
```

Key attributes:
- `0x0401` EIATTR_REGCOUNT: Number of registers used
- `0x0808` EIATTR_SMEM_SIZE: Shared memory bytes
- `0x0A04` EIATTR_LMEM_SIZE: Local memory per thread
- `0x0504` EIATTR_MAX_THREADS: Max threads per block
- `0x1704` EIATTR_EXIT_INSTR_OFFSETS: Offsets of EXIT instructions
- `0x1D04` EIATTR_CRS_STACK_SIZE: Call/return stack size

## Instruction Coverage Tiers

### Tier 1 - Core Compute (MVP)

| Category | Opcodes |
|----------|---------|
| Integer ALU | IADD3, IMAD, IMAD.WIDE, IABS, INEG, IMNMX |
| Integer Logic | LOP3, SHL, SHR, BFE, BFI, FLO, POPC |
| Integer Compare | ISETP, ICMP |
| Float ALU | FADD, FMUL, FFMA, FABS, FNEG, FMNMX |
| Float Compare | FSETP, FCMP |
| Float Special | MUFU (rcp, rsq, sin, cos, ex2, lg2) |
| Double ALU | DADD, DMUL, DFMA |
| Conversion | I2I, I2F, F2I, F2F |
| Global Memory | LDG, LDG.E, STG, STG.E (U8/U16/U32/U64/U128) |
| Shared Memory | LDS, STS |
| Local Memory | LDL, STL |
| Constant Memory | LDC |
| Data Movement | MOV, MOV32I, SEL, PRMT, SHFL |
| Special Registers | S2R, CS2R |
| Predicates | PSETP, PLOP3, P2R, R2P |
| Control Flow | BRA, JMP, CAL, RET, EXIT |
| Synchronization | BAR, DEPBAR, MEMBAR |
| No-op | NOP |

Estimated instruction count: ~60 opcodes

### Tier 2 - Tensor Core + Half Precision

| Category | Opcodes |
|----------|---------|
| Half Precision | HADD2, HMUL2, HFMA2, HSETP2 |
| Tensor Core (legacy) | HMMA.16816, IMMA.8816 |
| Tensor Core (SM120) | TCGen05 variants (FP16, BF16, FP8, INT8) |
| Global Atomics | ATOMG (ADD, MIN, MAX, CAS, EXCH) |
| Shared Atomics | ATOMS (ADD, MIN, MAX, CAS, EXCH) |
| Reductions | RED (ADD, MIN, MAX) |

Estimated additional: ~30 opcodes

### Tier 3 - Special + Async

| Category | Opcodes |
|----------|---------|
| Texture | TEX, TLD, TLD4, TXQ |
| Surface | SULD, SUST, SUATOM |
| Async Copy | CP.ASYNC, LDGSTS |
| TMA | TMA.LOAD, TMA.STORE, TMA.PREFETCH |
| FP8/FP6/FP4 | Specialized conversion and compute |
| Warp-level | MATCH, VOTE, REDUX |
| Uniform ops | UIADD, UFLO, ULDC, ULEA |

Estimated additional: ~40 opcodes

## Integration Points

### CLI (ptxas replacement)

Replace the current no-op stub in `ptxas/src/main.rs`:

```rust
fn main() {
    let options = options().run();
    let ptx_source = std::fs::read_to_string(&options.input).unwrap();
    let sm_version = parse_sm_version(&options.gpu_name); // e.g., 120
    let cubin = nvidia_sass::compile_ptx_to_cubin(&ptx_source, sm_version, options.opt_level);
    std::fs::write(&options.output, cubin).unwrap();
}
```

### comgr Backend

Add `#[cfg(feature = "nvidia")]` variant to `comgr/src/lib.rs`:

```rust
#[cfg(feature = "nvidia")]
pub fn compile_bitcode(
    sm_arch: &CStr,       // "sm_120"
    main_buffer: &[u8],   // LLVM IR bitcode
    ptx_impl: &[u8],      // PTX impl library bitcode
) -> Result<Vec<u8>, NvidiaComgrError> {
    nvidia_sass::compile_bitcode_to_cubin(sm_arch, main_buffer, ptx_impl)
}
```

### Module Loading

The existing `module.rs` load_data path for AMD:
```
PTX -> ptx_to_llvm_to_ptx_with_sass_mapping -> comgr::compile_bitcode -> hipModuleLoadData
```

The NVIDIA path would be:
```
PTX -> ptx::to_llvm_module (NVPTX) -> comgr::compile_bitcode (nvidia) -> cuModuleLoadData
```

## Validation: Disassemble Round-Trip

### Strategy

For every encoded instruction, verify by disassembling and comparing:

```rust
fn validate_roundtrip(instruction: &SassInstruction) -> Result<(), ValidationError> {
    let encoded = encode_instruction(instruction);
    let disassembler = SassDisassembler::new(120)?;
    let decoded = disassembler.decode_128bit(&encoded, 0);
    assert_eq!(decoded.opcode, instruction.opcode);
    assert_eq!(decoded.dest_operands, instruction.dest_operands);
    assert_eq!(decoded.src_operands, instruction.src_operands);
    assert_eq!(decoded.control_codes, instruction.control_codes);
    Ok(())
}
```

### Test Suite Structure

```
nvidia_sass/tests/
  encoding_roundtrip.rs    -- Per-opcode encode/decode round-trip
  cubin_format.rs          -- CUBIN ELF validity (parse with cubin_parser)
  regalloc_basic.rs        -- Register allocation correctness
  scheduler_basic.rs       -- Control code validity
  integration/
    vector_add.ptx         -- Simple kernel end-to-end
    matmul.ptx             -- Matrix multiply with shared memory
    reduce.ptx             -- Reduction with atomics
    tensor_core.ptx        -- HMMA/TCGen05 kernel
```

### CI Pipeline

1. Encode each opcode -> decode with existing disassembler -> compare
2. Build complete CUBIN -> parse with existing cubin_parser -> verify structure
3. (Optional, when hardware available) Load CUBIN on Blackwell GPU -> run -> compare outputs

## Dependencies

### New Dependencies

- `object` crate (already used by cubin_parser) for ELF writing
- No new external dependencies needed

### Internal Dependencies

```
nvidia_sass depends on:
  - ptx/src/sass/instruction.rs (SassInstruction types, opcode classification)
  - ptx/src/sass/disassembler.rs (for round-trip validation)
  - ptx/src/sass/cubin_parser.rs (for CUBIN validation)

ptxas depends on:
  - nvidia_sass
  - ptx_parser
  - ptx (for normalization passes + LLVM emission)
  - bpaf (already used, CLI argument parsing)

comgr (nvidia feature) depends on:
  - nvidia_sass
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SM120 encoding differs significantly from SM90 | Major rework of encoding tables | Start with known-shared opcodes (ALU, memory), isolate SM120-specific in separate modules |
| Register allocator produces suboptimal code | Performance regression vs NVIDIA ptxas | Linear scan is sufficient for correctness; optimize later |
| Instruction scheduler stall counts wrong | GPU hangs or incorrect results | Conservative scheduling (high stall counts) first, optimize down |
| CUBIN ELF format has undocumented fields | Driver rejects CUBIN | Diff a minimal NVIDIA-produced CUBIN byte-by-byte against our output |
| TCGen05 encoding is complex | Tensor core ops broken | Tier 2, defer until core works |

## Milestones

1. **M1: Encoding + CUBIN builder** - Encode Tier 1 instructions, generate valid CUBIN ELF, round-trip validates
2. **M2: Instruction selection** - LLVM IR -> SASS for simple kernels (vector_add)
3. **M3: Register allocation + scheduling** - Complete pipeline for Tier 1
4. **M4: CLI + comgr integration** - ptxas CLI works, comgr nvidia feature compiles
5. **M5: Tier 2 (tensor core)** - HMMA, TCGen05, atomics
6. **M6: Tier 3 (special)** - Texture, TMA, async copy
