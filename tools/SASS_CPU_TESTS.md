# Isolated SASS CPU tests

The normal project check remains `cargo test --workspace`. To test the lifter
without building unrelated LLVM/GPU backends, prepare a standalone harness:

```sh
python3 tools/prepare_sass_cpu_tests.py --output /tmp/concordia-sass-tests
cargo test --manifest-path /tmp/concordia-sass-tests/Cargo.toml
```

The output directory must be new. Paths derive from this checkout; `--source`
can select another checkout. `--lockfile` optionally seeds dependency versions
from an existing Cargo.lock without modifying it. For a prepared offline cache,
use `--offline`; after the harness lock is resolved, use `--locked`. Preserve that
lock and the source revision with results. No private lock or server path is
required by the repository. The setup script itself does not download or run
anything. Cargo outputs stay in the isolated harness unless the caller overrides
Cargo's target directory.

The harness imports six actual source modules through `#[path]`, the actual
`ptx_parser` and its real macros, and both complete CPU integration files
`sass_lifter_fuzz.rs` and `sass_translation_pipeline.rs`. It neither copies or
rewrites tests nor supplies parser stubs. The object dependency enables only
read/ELF features consumed by this subset. Tests in source modules can additionally
invoke an explicitly configured offline assembler; consult their environment
contract and report whether that check actually ran.

This is not all repository integration: LLVM inliner CLI, GPU record/replay and
tmatmul/backend tests are outside the subset. Report passed/failed/ignored counts
and conventional workspace build failures independently; do not label this a
workspace or GPU PASS. A same-revision source checkout must remain available
because this harness imports it directly.
