# aie_runtime-sys

Raw Rust bindings (bindgen) for the Xilinx XRT C API, used to load AIE
XCLBIN files and launch kernels on AMD Strix NPU.

## Prerequisites

- XRT installed (typically at `/opt/xilinx/xrt/`). The build script uses
  `pkg-config` first, falls back to `$XRT_PATH` / `/opt/xilinx/xrt`.
- `amdxdna` kernel driver loaded for hardware tests.
- `libclang-dev` for bindgen.

## Tests

- `cargo test -p aie_runtime_sys --no-run` — verify bindings compile.
- `cargo test -p aie_runtime_sys --features hw-test` — hardware smoke test
  requiring `/dev/accel/accel0` to be present.
