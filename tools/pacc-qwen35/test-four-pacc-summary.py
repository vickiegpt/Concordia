#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("four_smoke", Path(__file__).with_name("run-four-pacc-smoke.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
requests = [dict(all_correct=True, ngl=99, generated_tokens=[17, 220], requested_tokens=2,
                 started_monotonic_us=1_000_000 + i * 100_000,
                 decode_started_monotonic_us=11_000_000 + i * 100_000,
                 finished_monotonic_us=21_000_000 + i * 100_000) for i in range(4)]
profiles = [dict(capture_valid=True, build_matches=True, device=i) for i in range(4)]
report = module.aggregate(requests, profiles)
assert report["all_correct"]
assert abs(report["aggregate_output_tps"] - 8 / 20.3) < 1e-12
assert abs(report["aggregate_decode_tps"] - 4 / 10.3) < 1e-12
for case in range(6):
    r, p = copy.deepcopy(requests), copy.deepcopy(profiles)
    if case == 0: r[0]["all_correct"] = False
    if case == 1: r[0]["trace_only"] = True
    if case == 2: p[0]["device"] = 1
    if case == 3: p[0]["capture_valid"] = False
    if case == 4: r[0]["generated_tokens"] = [17]
    if case == 5: r.pop()
    report = module.aggregate(r, p)
    assert not report["all_correct"] and report["aggregate_output_tps"] is None
    assert report["aggregate_decode_tps"] is None
print("PASS aggregate windows, wrong tokens/count, diagnostic, missing worker, duplicate device and failed capture gates")
