# Concordia Persistent-Kernel Overhead Ablation

This benchmark measures the runtime cost of keeping a Concordia-style resident
worker alive while normal CUDA work runs in another stream. It is implemented
with NVlabs `cuda-oxide`, matching the delta-checkpoint benchmark in
`bench/concordia_delta_checkpoint`.

Outputs are CSV rows with this schema:

```text
worker_blocks,theoretical_sm_pct,copy_ms,copy_gbs,copy_overhead_pct,compute_ms,compute_gops,compute_overhead_pct,worker_active_threads,worker_counter_xor
```

Run a static check and chart smoke:

```bash
CONCORDIA_PERSISTENT_OVERHEAD_STATIC_ONLY=1 bash bench/concordia_persistent_overhead/test_persistent_overhead.sh
```

Run the live GPU self-test:

```bash
cd bench/concordia_persistent_overhead
RUSTUP_TOOLCHAIN=nightly-2026-04-03 CONCORDIA_PERSISTENT_OVERHEAD_SELF_TEST=1 cargo oxide run --arch sm_120
```

Run the full ablation:

```bash
cd bench/concordia_persistent_overhead
RUSTUP_TOOLCHAIN=nightly-2026-04-03 \
CONCORDIA_PERSISTENT_OVERHEAD_WORKER_BLOCKS=0,1,2,4,8 \
CONCORDIA_PERSISTENT_OVERHEAD_WARMUP=3 \
CONCORDIA_PERSISTENT_OVERHEAD_ITERS=10 \
cargo oxide run --arch sm_120 \
  > /tmp/concordia-persistent-overhead.csv
```

Generate paper figures:

```bash
python3 bench/concordia_persistent_overhead/plot_persistent_overhead.py \
  --overhead-csv /tmp/concordia-persistent-overhead.csv \
  --delta-log /tmp/concordia-eval-claims-20260704-full2/extra/delta_10iter.log \
  --out-dir 69b5f215f5c67f0702bd8f65/img
```
