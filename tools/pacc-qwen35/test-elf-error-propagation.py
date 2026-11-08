#!/usr/bin/env python3
"""A deliberately unresolved device symbol must return an error, never success."""
import ctypes as c
import os

lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so');P=c.c_void_p
def bind(name,args,result=c.c_int):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
dev=bind('sifive_CreateDevice',[c.c_uint],P)(int(os.environ['HETGPU_SIFIVE_FORCE_DEVICE']))
program=bind('sifive_CreateProgram',[],P)();assert dev and program
source=b'extern void hetgpu_intentionally_missing_v1(void); void pacc_error_probe_v1(void) { hetgpu_intentionally_missing_v1(); }'
load=bind('sifive_LoadProgramSource',[P,P,c.c_char_p,c.c_char_p,c.c_uint64,P,P,c.c_uint,P,c.c_uint])
assert load(program,None,b'error_probe.c',source,len(source),None,None,0,None,0)==0
kernel=bind('sifive_CreateKernelOnDevice',[P,P,c.c_char_p],P)(program,dev,b'pacc_error_probe_v1');assert kernel
rc=bind('sifive_LaunchKernel',[P]+[c.c_uint]*6)(kernel,1,1,1,1,1,1)
assert rc!=0, 'device ELF load failure was falsely reported as success'
print(f'PASS unresolved device ELF rejected rc={rc}',flush=True)
