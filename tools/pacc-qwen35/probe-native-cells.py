#!/usr/bin/env python3
"""Diagnostic for the native-source parameter-cell transport on PACC0."""
import ctypes as c
import math
import os
from pathlib import Path
import sys
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
P=c.c_void_p; U=c.c_uint64
def call(name,ret,types,*args):
    f=getattr(lib,name); f.argtypes=types; f.restype=ret
    return f(*args)
source=b'''typedef unsigned long u64;
extern unsigned f___zluda_ptx_impl_sreg_tid(unsigned char);
static unsigned counter;
void pacc_probe_cells_v1(u64 a,u64 b,u64 d,u64 e,u64 f,u64 g,u64 h,u64 j) {
 if(f___zluda_ptx_impl_sreg_tid(0)) return;
 float *out=(float*)*(u64*)b;
 out[0]=*(float*)*(u64*)a;
 out[1]=(float)*(u64*)d; out[2]=(float)*(u64*)e;
 out[3]=(float)*(u64*)f; out[4]=(float)*(u64*)g;
 out[5]=(float)*(u64*)h; out[6]=(float)*(u64*)j;
 out[7]=(float)++counter;
}'''
dev=call('sifive_CreateDevice',P,[c.c_uint],0)
program=call('sifive_CreateProgram',P,[])
gdn='--gdn' in sys.argv
symbol=b'pacc_gdn_head_v1' if gdn else b'pacc_probe_cells_v1'
if gdn: source=Path('/home/ubuntu/Concordia/ext/sifive_runtime-sys/src/gated_delta.c').read_bytes()
if '--xm-entry' in sys.argv:
    source=Path('/home/ubuntu/Concordia/ext/sifive_runtime-sys/src/xsfmm_native_bf16.c').read_bytes()+b'\n'+source
rc=call('sifive_LoadProgramSource',c.c_int,[P,c.c_char_p,c.c_char_p,c.c_char_p,U,c.c_char_p,P,U,P,U],
        program,None,symbol+b'.c',source,len(source),None,None,0,None,0)
assert rc==0,rc
kernel=call('sifive_CreateKernelOnDevice',P,[P,P,c.c_char_p],program,dev,symbol)
class Record(c.Structure): _fields_=[('kind',c.c_uint),('size',c.c_uint),('flags',c.c_uint),('reserved',c.c_uint),('value',U),('hi',U)]
class Binding(c.Structure): _fields_=[('index',c.c_uint),('flags',c.c_uint),('addr',U),('size',U)]
inp=(c.c_float*4)(1234.5,1,2,3); out=(c.c_float*8)(*([math.nan]*8))
if gdn:
    inp=(c.c_float*306)(*([1.]*16+[.1]*16+[.2]*16+[0.,.1]+[0.]*256))
    out=(c.c_float*272)(*([math.nan]*272))
for i,v in enumerate([c.addressof(inp),c.addressof(out),16,1,0,0,1,1048576000]):
    r=Record(1 if i<2 else 0,8,0,0,v,0)
    assert call('sifive_KernelPushArgRecord',c.c_int,[P,c.POINTER(Record)],kernel,c.byref(r))==0
    if i<2:
        b=Binding(i,256 if i==0 else 768,v,c.sizeof(inp) if i==0 else c.sizeof(out))
        assert call('sifive_KernelAddBufferBinding',c.c_int,[P,c.POINTER(Binding)],kernel,c.byref(b))==0
for iteration in range(int(os.environ.get('CELL_PROBE_REPEATS','1'))):
    for i in range(len(out)): out[i]=math.nan
    rc=call('sifive_LaunchKernel',c.c_int,[P]+[c.c_uint]*6,kernel,1,1,1,16 if gdn else 2,1,1)
    if iteration%100==0 or rc!=0:
        print('cell probe iteration=',iteration,'rc=',rc,'output=',list(out)[:24],flush=True)
    if gdn:
        assert rc==0 and all(abs(v-.008)<1e-6 for v in list(out)[:16]),iteration
    else:
        assert rc==0 and list(out)[:6]==[1234.5,16.,1.,0.,0.,1.] and out[7]==1.,iteration
print('PASS cell probe repeated submissions', iteration+1, flush=True)
