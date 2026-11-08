#!/usr/bin/env python3
import ctypes as c
import math
import os
from pathlib import Path
import runpy
import struct

ref=runpy.run_path(str(Path(__file__).with_name('test-q4k-xm.py')))
qtype=ref['qtype'];block_bytes=ref['block_bytes'];small=qtype==5 or os.environ.get('QK_TEST_SMALL')=='1'
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so');P=c.c_void_p;U=c.c_uint64
def call(name,types,*args):
    fn=getattr(lib,name);fn.argtypes=types;fn.restype=c.c_int
    rc=fn(*args);assert rc==0,(name,rc)
def alloc(data):
    p=U();call('cuMemAlloc_v2',[c.POINTER(U),c.c_size_t],c.byref(p),len(data))
    call('cuMemcpyHtoD_v2',[U,c.c_char_p,c.c_size_t],p,data,len(data));return p
def read(p,sz):
    out=c.create_string_buffer(sz);call('cuMemcpyDtoH_v2',[P,U,c.c_size_t],out,p,sz);return out.raw
call('cuInit',[c.c_uint],0);ctx=P();call('cuCtxCreate_v2',[c.POINTER(P),c.c_uint,c.c_int],c.byref(ctx),0,0)
bias_enabled=os.environ.get('QK_TEST_BIAS')=='1'
for gate in ([1] if qtype==8 and not small and not bias_enabled else [0,1]):
    m,n,nb,experts=(68 if small else 37),1,2,3
    x,y=ref['generate'](m*experts,n,nb,gate);xs=m*nb*block_bytes
    no_ids=qtype==8
    order=[0] if no_ids else [2,0];stride=m+3
    xbvals=[.0078125*(i%17-8) for i in range(experts*stride)]
    gbvals=[.015625*(i%13-6) for i in range(experts*stride)]
    expected=[]
    for expert in order:
        a=x[expert*xs:(expert+1)*xs]
        if gate:a+=x[(experts+expert)*xs:(experts+expert+1)*xs]
        if bias_enabled:
            up=ref['reference'](a[:xs],y,m,1,nb,0)
            g=ref['reference'](a[xs:],y,m,1,nb,0) if gate else [0.]*m
            vals=[]
            for row in range(m):
                u=up[row]+xbvals[expert*stride+row];v=g[row]+gbvals[expert*stride+row]
                vals.append(u*v/(1+math.exp(-v)) if gate else u)
        else:vals=ref['reference'](a,y,m,1,nb,gate)
        expected+=vals+[12345.]*3
    expected+=[12345.]*8
    xp=alloc(x[:experts*xs]);gp=alloc(x[experts*xs:]) if gate else U(0)
    yp=alloc(y);ip=U(0) if no_ids else alloc(struct.pack('<%di'%len(order),*order));out=alloc(struct.pack('<%df'%len(expected),*([12345.]*len(expected))))
    class Fusion(c.Structure):_fields_=[('ptrs',U*5),('op',c.c_int)]
    bp=alloc(struct.pack('<%df'%len(xbvals),*xbvals)) if bias_enabled else U(0)
    gbp=alloc(struct.pack('<%df'%len(gbvals),*gbvals)) if bias_enabled and gate else U(0)
    fusion=Fusion((U*5)(bp.value,gp.value,gbp.value,0,0),2)
    one=(c.c_uint32*3)(0,0,1)
    unused_ratio=(c.c_uint32*3)(0,0,0)
    blocks=nb*(8 if qtype==8 else 1)
    params=[xp,yp,ip,fusion,out,c.c_uint(nb*256),unused_ratio if no_ids else one,c.c_uint(blocks),c.c_uint(nb*8),c.c_uint(stride*len(order)),one if no_ids else unused_ratio,
            c.c_uint(m*blocks),c.c_uint(nb*8),c.c_uint(stride),one,c.c_uint(0),c.c_uint(0),c.c_uint(0),c.c_uint(0)]
    dtype=8 if qtype==8 else qtype+8
    name=f'_ZL13mul_mat_vec_qIL9ggml_type{dtype}ELi1ELb{int(bool(gate or bias_enabled))}ELb{int(small)}ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'
    decl=[f'.param .align 8 .b8 p{i}[{c.sizeof(v)}]' if isinstance(v,(c.Array,c.Structure)) else f'.param .b{c.sizeof(v)*8} p{i}' for i,v in enumerate(params)]
    ptx=('.version 7.0\n.target sm_70\n.address_size 64\n.visible .entry '+name+'('+','.join(decl)+'){ret;}').encode()
    mod=P();fun=P();call('cuModuleLoadData',[c.POINTER(P),c.c_char_p],c.byref(mod),ptx)
    call('cuModuleGetFunction',[c.POINTER(P),P,c.c_char_p],c.byref(fun),mod,name.encode())
    argv=(P*19)(*[c.addressof(v) for v in params])
    call('cuLaunchKernel',[P]+[c.c_uint]*7+[P,c.POINTER(P),P],fun,m//4 if small else m,len(order),1,32,4,1,0,None,argv,None)
    got=struct.unpack('<%df'%len(expected),read(out,len(expected)*4))
    assert all(math.isfinite(a) and abs(a-b)<5e-5+5e-5*abs(b) for a,b in zip(got,expected)),(gate,got,expected)
    print(f'PASS Q{qtype}_K CUDA ABI gate={gate} bias={bias_enabled} small_k={small} expert_order={order} output_stride={stride} guards=unchanged',flush=True)
    call('cuModuleUnload',[P],mod)
    for p in [xp,yp,out]+([] if no_ids else [ip])+([gp] if gate else []):call('cuMemFree_v2',[U],p)
    for p in (bp,gbp):
        if p.value:call('cuMemFree_v2',[U],p)
call('cuCtxDestroy_v2',[P],ctx)
