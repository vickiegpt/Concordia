# aie_comgr-sys

TOSA-MLIR → AMD AIE XCLBIN compilation driver. Shells out to the Xilinx
`mlir-aie` toolchain (`aie-opt`, `aie-translate`).

## Prerequisites

- Xilinx mlir-aie built and installed; `aie-opt` and `aie-translate` on `$PATH`
  (or set `AIE_TOOLCHAIN_DIR` to the `bin/` directory).

## Tests

- `cargo test -p aie_comgr_sys --lib` — unit tests (no toolchain needed).
- `cargo test -p aie_comgr_sys --test pipeline -- --ignored` — integration
  test that runs a trivial TOSA module through the full toolchain pipeline.
  Requires mlir-aie.
