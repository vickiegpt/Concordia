#!/usr/bin/env python3
"""Exercise the real exported allocator/memset ABI without accelerator submission."""
import ctypes as c
import json
import os
from pathlib import Path
import time

root = Path('/srv/lanxin/qwen35-a3b')
os.environ.update(HETGPU_SIFIVE_ALLOW_HOST_DEVICE_MEM='1', HETGPU_SIFIVE_REAL_DEVICE_MEM='0',
                  HETGPU_SIFIVE_SHARED_DEVICE_MEM='0', HETGPU_SIFIVE_LAZY_ZERO='1')
lib = c.CDLL(str(Path(os.environ.get('CONCORDIA_RELEASE', str(root / 'runtime-memory-v20'))) / 'libnvcuda.so'))
lib.cuMemAlloc_v2.argtypes = [c.POINTER(c.c_uint64), c.c_size_t]
lib.cuMemFree_v2.argtypes = [c.c_uint64]
lib.cuMemsetD8_v2.argtypes = [c.c_uint64, c.c_ubyte, c.c_size_t]
lib.cuMemsetD32_v2.argtypes = [c.c_uint64, c.c_uint, c.c_size_t]
libc = c.CDLL(None, use_errno=True)
libc.mincore.argtypes = [c.c_void_p, c.c_size_t, c.c_void_p]
page = os.sysconf('SC_PAGE_SIZE')
size = 32 * 1024 * 1024 + 123
ptr = c.c_uint64()
assert lib.hetgpu_sifive_host_backing_mode_v1() == 1
os.environ['HETGPU_SIFIVE_REAL_DEVICE_MEM'] = '1'
assert lib.hetgpu_sifive_host_backing_mode_v1() == 0
os.environ['HETGPU_SIFIVE_REAL_DEVICE_MEM'] = '0'
assert lib.cuMemAlloc_v2(c.byref(ptr), size) == 0
assert ptr.value % page == 0
def resident():
    v = (c.c_ubyte * ((size + page - 1) // page))()
    assert libc.mincore(ptr.value, size, v) == 0
    return sum(x & 1 for x in v)
def data(offset, count): return c.string_at(ptr.value + offset, count)
fresh = resident()
assert fresh == 0, fresh
assert lib.cuMemsetD8_v2(ptr.value, 0x5a, size) == 0
filled = resident()
start = time.monotonic_ns()
assert lib.cuMemsetD8_v2(ptr.value + 17, 0, size - 60) == 0
elapsed = (time.monotonic_ns() - start) / 1000
discarded = resident()
assert discarded <= 2, discarded
assert data(0, 17) == b'Z' * 17 and data(size - 43, 43) == b'Z' * 43
for offset in (17, page - 8, page, size // 2, size - 51):
    assert data(offset, 8) == b'\0' * 8
assert lib.cuMemsetD8_v2(ptr.value + size - 2, 0, 3) != 0
assert data(size - 43, 43) == b'Z' * 43
assert lib.cuMemsetD32_v2(ptr.value, 0x12345678, size // 4) == 0
assert lib.cuMemsetD32_v2(ptr.value + 20, 0, (size - 80) // 4) == 0
assert data(0, 20) == bytes.fromhex('78563412') * 5
assert data(20, 16) == b'\0' * 16
assert lib.cuMemsetD8_v2(ptr.value, 0x5a, size) == 0
child = os.fork()
if child == 0:
    ok = lib.cuMemsetD8_v2(ptr.value, 0, size) == 0 and data(0, 64) == b'\0' * 64
    os._exit(0 if ok else 1)
assert os.waitpid(child, 0)[1] == 0
assert data(0, 64) == b'Z' * 64
assert lib.cuMemFree_v2(ptr.value) == 0
v = (c.c_ubyte * 1)()
assert libc.mincore(ptr.value, page, v) != 0
print(json.dumps(dict(passed=True, bytes=size, fresh_resident_pages=fresh,
    filled_resident_pages=filled, after_unaligned_zero_pages=discarded,
    zero_call_us=elapsed, bounds_rejected=True, fork_isolation=True), indent=2))
