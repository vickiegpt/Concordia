#!/usr/bin/env python3
"""Current GGML 25-argument ABI, padded tensors, head/sequence mapping and state slots."""
import ctypes as c
import math
import os
from pathlib import Path
import runpy
import struct

ref=runpy.run_path(str(Path(__file__).with_name('test-gdn-hardware.py')))['reference']
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
P,U=c.c_void_p,c.c_uint64
def call(name,types,*args):
    f=getattr(lib,name); f.argtypes=types; f.restype=c.c_int
    rc=f(*args); assert rc==0,(name,rc)
def alloc(values):
    data=struct.pack('<%df'%len(values),*values); p=U()
    call('cuMemAlloc_v2',[c.POINTER(U),c.c_size_t],c.byref(p),len(data))
    call('cuMemcpyHtoD_v2',[U,c.c_char_p,c.c_size_t],p,data,len(data))
    return p
def read(p,count):
    buf=c.create_string_buffer(count*4)
    call('cuMemcpyDtoH_v2',[P,U,c.c_size_t],c.cast(buf,P),p,count*4)
    return list(struct.unpack('<%df'%count,buf.raw))

call('cuInit',[c.c_uint],0); ctx=P()
call('cuCtxCreate_v2',[c.POINTER(P),c.c_uint,c.c_int],c.byref(ctx),0,0)
for kda,keep in [(0,0),(0,1),(1,1)]:
    s,n,h,seqs,qh,ratio=16,2,3,2,2,2
    qs=[s+2,qh*(s+2)+3,n*(qh*(s+2)+3)+5]
    vs=[s+1,h*(s+1)+3,n*(h*(s+1)+3)+5]
    bs=[2,h*2+1,n*(h*2+1)+3]
    span=lambda heads,seq,strides,w:(seq-1)*strides[2]+(n-1)*strides[1]+(heads-1)*strides[0]+w
    lengths=[span(qh,1,qs,s)]*2+[span(h,seqs,vs,s),span(h,seqs,bs,1)*(s if kda else 1),span(h,seqs,bs,1)]
    inputs=[[c.c_float(((i*7+j*3)%29-14)*.004).value for i in range(sz)] for j,sz in enumerate(lengths)]
    state_size=seqs*h*s*s; slot_stride=state_size+7; cap=4
    initial=[c.c_float(((i*3)%37-18)*.002).value for i in range(state_size)]
    states=initial+[12345.]*(slot_stride*cap+8-state_size)
    scale=c.c_float(1/math.sqrt(s)).value
    expected=[math.nan]*(seqs*n*h*s); expected_states=states.copy()
    for seq in range(seqs):
        for head in range(h):
            packed=[]
            for idx,st,sq,hq,width in [(0,qs,seq//ratio,head%qh,s),(1,qs,seq//ratio,head%qh,s),
                                      (2,vs,seq,head,s),(3,bs,seq,head,s if kda else 1),(4,bs,seq,head,1)]:
                for t in range(n):
                    off=sq*st[2]+t*st[1]+hq*st[0]
                    if idx==3 and kda: off*=s
                    packed.extend(inputs[idx][off:off+width])
            base=(seq*h+head)*s*s
            packed+=initial[base:base+s*s]
            slots=n if keep else 1
            result=ref(packed,s,n,kda,keep,slots,scale)
            for t in range(n):
                off=((seq*n+t)*h+head)*s; expected[off:off+s]=result[t*s:(t+1)*s]
            for slot in range(slots):
                off=base+slot*slot_stride
                expected_states[off:off+s*s]=result[n*s+slot*s*s:n*s+(slot+1)*s*s]
    ptrs=[alloc(v) for v in inputs]; state=alloc(states); out=alloc([12345.]*(len(expected)+8))
    params=[*ptrs,state,out,state,*map(c.c_int64,[h,n,seqs,*qs,*vs,*bs]),
            (c.c_uint32*3)(0,0,qh),(c.c_uint32*3)(0,0,ratio),c.c_float(scale),c.c_int64(slot_stride),c.c_int(cap)]
    assert len(params)==25
    name=f'_Z20gated_delta_net_cudaILi16ELb{kda}ELb{keep}EEvPKfS1_S1_S1_S1_S1_PfS2_llllllllllll5uint3S3_fli'
    decl=[f'.param .align 4 .b8 p{i}[{c.sizeof(v)}]' if isinstance(v,c.Array) else f'.param .b{c.sizeof(v)*8} p{i}' for i,v in enumerate(params)]
    ptx=('.version 7.0\n.target sm_70\n.address_size 64\n.visible .entry '+name+'('+','.join(decl)+'){ret;}').encode()
    module,fn=P(),P()
    call('cuModuleLoadData',[c.POINTER(P),c.c_char_p],c.byref(module),ptx)
    call('cuModuleGetFunction',[c.POINTER(P),P,c.c_char_p],c.byref(fn),module,name.encode())
    argv=(P*25)(*[c.addressof(v) for v in params])
    call('cuLaunchKernel',[P]+[c.c_uint]*7+[P,c.POINTER(P),P],fn,h,seqs,s//4,s,4,1,0,None,argv,None)
    for actual,wanted in [(read(out,len(expected)+8),expected+[12345.]*8),(read(state,len(states)),expected_states)]:
        assert all(math.isfinite(a) and abs(a-b)<3e-5+3e-5*abs(b) for a,b in zip(actual,wanted)),(kda,keep)
    print(f'PASS GDN CUDA ABI kda={kda} keep={keep} heads={h} sequences={seqs} alias_state=true guards=unchanged',flush=True)
    call('cuModuleUnload',[P],module)
    for p in [*ptrs,state,out]: call('cuMemFree_v2',[U],p)
call('cuCtxDestroy_v2',[P],ctx)
