#!/usr/bin/env python3
"""CUDA ABI data-copy regression. This is not an XM compute or TPS benchmark."""
import array
import ctypes as c
import os

lib = c.CDLL(os.environ["CONCORDIA_RELEASE"] + "/libnvcuda.so")
P, U = c.c_void_p, c.c_uint64

def invoke(name, types, *args, success=True):
    f = getattr(lib, name)
    f.argtypes, f.restype = types, c.c_int
    result = f(*args)
    assert (result == 0) == success, (name, result, success)
    return result

def alloc(data):
    p = U()
    invoke("cuMemAlloc_v2", [c.POINTER(U), c.c_size_t], c.byref(p), len(data))
    invoke("cuMemcpyHtoD_v2", [U, c.c_char_p, c.c_size_t], p, data, len(data))
    return p

def read(p, size):
    data = c.create_string_buffer(size)
    invoke("cuMemcpyDtoH_v2", [P, U, c.c_size_t], data, p, size)
    return data.raw

invoke("cuInit", [c.c_uint], 0)
context = P()
invoke("cuCtxCreate_v2", [c.POINTER(P), c.c_uint, c.c_int], c.byref(context), 0, 0)

def run(cols, vector, bad=None):
    ne10, ne11, ne12, src_rows = (1, 1, 1, 1) if cols > 1024 or bad else (2, 2, 2, 3)
    row_stride = cols + 8
    channel_stride = row_stride * src_rows + 8
    sample_stride = channel_stride * ne11 + 8
    source = array.array("f", ((i % 113 - 56) * 0.125 for i in range(sample_stride * ne12)))
    ids = array.array("i", ((i + 1) % src_rows for i in range(ne10 * ne11 * ne12)))
    if bad == "negative":
        ids[0] = -1
    d1, d2, d3 = cols + 12, ne10 * (cols + 12) + 8, ne11 * (ne10 * (cols + 12) + 8) + 8
    dest = array.array("f", [12345.0]) * (d3 * ne12 + 8)
    expected = dest[:]
    if bad == "short-source":
        source = source[:cols // 4]
    if bad == "short-dest":
        dest = dest[:cols // 4]
    if not bad:
        for z in range(ne12):
            for y in range(ne11):
                for x in range(ne10):
                    index = ids[(z * ne11 + y) * ne10 + x]
                    so = z * sample_stride + y * channel_stride + index * row_stride
                    do = z * d3 + y * d2 + x * d1
                    expected[do:do + cols] = source[so:so + cols]
    pointers = [alloc(source.tobytes()), alloc(ids.tobytes()), alloc(dest.tobytes())]
    params = [*pointers, c.c_int64(cols // 4 if vector else cols), c.c_int64(ne11),
              (c.c_uint32 * 3)(0, 0, ne12),
              *map(U, [d1, d2, d3, row_stride * 4, channel_stride * 4, sample_stride * 4,
                       1, ne10, ne10 * ne11])]
    name = ("_ZL20k_get_rows_float_vecIfEvPKT_PKiPS0_" if vector else
            "_ZL16k_get_rows_floatIffEvPKT_PKiPT0_") + "ll5uint3mmmmmmmmm"
    decl = [f".param .align 4 .b8 p{i}[{c.sizeof(v)}]" if isinstance(v, c.Array)
            else f".param .b{c.sizeof(v)*8} p{i}" for i, v in enumerate(params)]
    ptx = (".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry " + name +
           "(" + ",".join(decl) + "){ret;}").encode()
    module, function = P(), P()
    invoke("cuModuleLoadData", [c.POINTER(P), c.c_char_p], c.byref(module), ptx)
    invoke("cuModuleGetFunction", [c.POINTER(P), P, c.c_char_p], c.byref(function), module, name.encode())
    argv = (P * len(params))(*(c.addressof(v) for v in params))
    invoke("cuLaunchKernel", [P] + [c.c_uint] * 7 + [P, c.POINTER(P), P], function,
           ne10, max(1, (cols + 1023) // 1024), ne11 * ne12, 256, 1, 1, 0, None, argv, None,
           success=bad is None)
    actual = read(pointers[2], len(dest) * 4)
    assert actual == (dest if bad else expected).tobytes(), (cols, vector, bad, "copy/guard mismatch")
    print(f"PASS data-copy ABI vector={vector} cols={cols} bad={bad} full_bytes_and_guards=verified", flush=True)
    invoke("cuModuleUnload", [P], module)
    for p in pointers:
        invoke("cuMemFree_v2", [U], p)

run(17, False)
run(64, True)
run(524288, True)
for failure in ("negative", "short-source", "short-dest"):
    run(64, True, failure)
invoke("cuCtxDestroy_v2", [P], context)
