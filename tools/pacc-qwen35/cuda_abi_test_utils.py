import ctypes as c
import os

P=c.c_void_p
U=c.c_uint64
class Cuda:
    def __init__(self):
        self.lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
        self.buffers=[];self.modules=[]
        self.call('cuInit',[c.c_uint],0)
        self.ctx=P();self.call('cuCtxCreate_v2',[c.POINTER(P),c.c_uint,c.c_int],c.byref(self.ctx),0,0)
    def call(self,name,types,*args):
        fn=getattr(self.lib,name);fn.argtypes=types;fn.restype=c.c_int
        rc=fn(*args);assert rc==0,(name,rc)
    def alloc(self,data):
        p=U();self.call('cuMemAlloc_v2',[c.POINTER(U),c.c_size_t],c.byref(p),len(data))
        self.call('cuMemcpyHtoD_v2',[U,c.c_char_p,c.c_size_t],p,data,len(data));self.buffers.append(p);return p
    def read(self,p,size):
        out=c.create_string_buffer(size);self.call('cuMemcpyDtoH_v2',[P,U,c.c_size_t],out,p,size);return out.raw
    def launch(self,name,params,grid,block,ptx=None):
        if ptx is None:
            decl=[f'.param .align 8 .b8 p{i}[{c.sizeof(v)}]' if isinstance(v,(c.Array,c.Structure)) else f'.param .b{c.sizeof(v)*8} p{i}' for i,v in enumerate(params)]
            ptx=('.version 7.0\n.target sm_70\n.address_size 64\n.visible .entry '+name+'('+','.join(decl)+'){ret;}').encode()
        mod=P();fun=P();self.call('cuModuleLoadData',[c.POINTER(P),c.c_char_p],c.byref(mod),ptx);self.modules.append(mod)
        self.call('cuModuleGetFunction',[c.POINTER(P),P,c.c_char_p],c.byref(fun),mod,name.encode())
        argv=(P*len(params))(*[c.addressof(v) for v in params])
        self.call('cuLaunchKernel',[P]+[c.c_uint]*7+[P,c.POINTER(P),P],fun,*grid,*block,0,None,argv,None)
    def close(self):
        for p in self.buffers:self.call('cuMemFree_v2',[U],p)
        for m in self.modules:self.call('cuModuleUnload',[P],m)
        self.call('cuCtxDestroy_v2',[P],self.ctx)
