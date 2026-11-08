#!/usr/bin/env python3
"""Real PACC copy kernel with the nine-argument GGML concat_cont ABI."""
import ctypes as c
import math
import os
import struct

lib = c.CDLL(os.environ['CONCORDIA_RELEASE'] + '/libnvcuda.so')
P, U = c.c_void_p, c.c_uint64
def call(name, types, *args):
    f = getattr(lib, name)
    f.argtypes, f.restype = types, c.c_int
    rc = f(*args)
    assert rc == 0, (name, rc)

call('cuInit', [c.c_uint], 0)
ctx = P()
call('cuCtxCreate_v2', [c.POINTER(P), c.c_uint, c.c_int], c.byref(ctx), 0, 0)
def alloc(data):
    p = U()
    call('cuMemAlloc_v2', [c.POINTER(U), c.c_size_t], c.byref(p), len(data))
    buf = c.create_string_buffer(data)
    call('cuMemcpyHtoD_v2', [U, P, c.c_size_t], p, c.cast(buf, P), len(data))
    return p

for dim in range(3):
    a, d = [3, 2, 2], [3, 2, 2]
    d[dim] += 2
    b = d.copy()
    b[dim] -= a[dim]
    x = list(range(100, 100 + math.prod(a)))
    y = list(range(900, 900 + math.prod(b)))
    total = math.prod(d)
    expected = []
    for k in range(d[2]):
        for j in range(d[1]):
            for i in range(d[0]):
                at = [i, j, k]
                src, shape = x, a
                if at[dim] >= a[dim]:
                    at[dim] -= a[dim]
                    src, shape = y, b
                expected.append(src[(at[2]*shape[1] + at[1])*shape[0] + at[0]])
    xp = alloc(struct.pack('<%dI' % len(x), *x))
    yp = alloc(struct.pack('<%dI' % len(y), *y))
    out = alloc(bytes([0xa5]) * (4*total + 32))
    name = f'_ZL11concat_contIjLi{dim}EEvPKT_S2_PS0_llllll'
    # Segment concatenation is equivalent for all three contiguous dimensions.
    segment = ['mov.u64 %a, %s0; mov.u64 %d, %d0;',
               'mul.lo.u64 %a, %d0, %s1; mul.lo.u64 %d, %d0, %d1;',
               'mul.lo.u64 %a, %d0, %d1; mul.lo.u64 %a, %a, %s2; mov.u64 %d, %n;'][dim]
    ptx = f'''.version 7.0
.target sm_70
.address_size 64
.visible .entry {name}(
 .param .u64 p0, .param .u64 p1, .param .u64 p2,
 .param .u64 p3, .param .u64 p4, .param .u64 p5,
 .param .u64 p6, .param .u64 p7, .param .u64 p8) {{
 .reg .pred %p;
 .reg .u32 %r0, %v;
 .reg .u64 %x,%y,%z,%s0,%s1,%s2,%d0,%d1,%d2,%n,%i,%a,%d,%b,%row,%off,%addr;
 ld.param.u64 %x,[p0]; ld.param.u64 %y,[p1]; ld.param.u64 %z,[p2];
 ld.param.u64 %s0,[p3]; ld.param.u64 %s1,[p4]; ld.param.u64 %s2,[p5];
 ld.param.u64 %d0,[p6]; ld.param.u64 %d1,[p7]; ld.param.u64 %d2,[p8];
 mov.u32 %r0,%tid.x; cvt.u64.u32 %i,%r0;
 mul.lo.u64 %n,%d0,%d1; mul.lo.u64 %n,%n,%d2;
 {segment}
 sub.u64 %b,%d,%a;
 LOOP:
 setp.ge.u64 %p,%i,%n; @%p bra DONE;
 div.u64 %row,%i,%d; rem.u64 %off,%i,%d;
 setp.ge.u64 %p,%off,%a; @%p bra FROM_Y;
 mul.lo.u64 %addr,%row,%a; add.u64 %addr,%addr,%off;
 mul.lo.u64 %addr,%addr,4; add.u64 %addr,%x,%addr;
 bra LOAD;
 FROM_Y:
 sub.u64 %off,%off,%a; mul.lo.u64 %addr,%row,%b; add.u64 %addr,%addr,%off;
 mul.lo.u64 %addr,%addr,4; add.u64 %addr,%y,%addr;
 LOAD:
 ld.global.u32 %v,[%addr]; mul.lo.u64 %addr,%i,4; add.u64 %addr,%z,%addr;
 st.global.u32 [%addr],%v; add.u64 %i,%i,32; bra LOOP;
 DONE: ret;
 }}'''.encode()
    module, fn = P(), P()
    call('cuModuleLoadData', [c.POINTER(P), c.c_char_p], c.byref(module), ptx)
    call('cuModuleGetFunction', [c.POINTER(P), P, c.c_char_p], c.byref(fn), module, name.encode())
    params = [xp, yp, out, *map(U, a+d)]
    argv = (P*len(params))(*[c.addressof(v) for v in params])
    call('cuLaunchKernel', [P]+[c.c_uint]*7+[P,c.POINTER(P),P], fn,1,1,1,32,1,1,0,None,argv,None)
    call('cuCtxSynchronize', [])
    raw = c.create_string_buffer(total*4 + 32)
    call('cuMemcpyDtoH_v2', [P,U,c.c_size_t], c.cast(raw,P),out,len(raw))
    assert list(struct.unpack_from('<%dI'%total,raw)) == expected, dim
    assert raw.raw[total*4:] == bytes([0xa5])*32, 'output guard changed'
    print(f'PASS PACC concat_cont dim={dim} elements={total} output_guard=unchanged',flush=True)
    call('cuModuleUnload',[P],module)
    for p in (xp,yp,out):
        call('cuMemFree_v2',[U],p)
call('cuCtxDestroy_v2',[P],ctx)
