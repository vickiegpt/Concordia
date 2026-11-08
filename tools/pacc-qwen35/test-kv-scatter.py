#!/usr/bin/env python3
import ctypes as c
from pathlib import Path
import struct
from cuda_abi_test_utils import Cuda

cuda=Cuda();x=[i*.137-1 for i in range(16)];ids=[7,3,5,1]
xp=cuda.alloc(struct.pack('<16f',*x));ip=cuda.alloc(struct.pack('<4q',*ids))
expected=[123.]*88;dp=cuda.alloc(struct.pack('<88e',*expected))
for head in range(2):
    for row in range(2):
        for col in range(4):expected[head*40+ids[head*2+row]*4+col]=struct.unpack('<e',struct.pack('<e',x[head*8+row*4+col]))[0]
fd=lambda d:(c.c_uint32*3)(1,d.bit_length()-1,d)
params=[xp,ip,dp]+[c.c_int64(v) for v in [16,2,2,1,1,4,8,16,1,2,4,4,40,80]]+[fd(v) for v in [4,2,2,2,1]]
name='_ZL10k_set_rowsIfl6__halfEvPKT_PKT0_PT1_llllllllllllll5uint3S9_S9_S9_S9_'
ptx=Path('/srv/lanxin/qwen35-a3b/ptx-cache/libggml_cuda_so_0/set-rows.cu.ptx').read_bytes()
cuda.launch(name,params,(1,1,1),(256,1,1),ptx)
got=struct.unpack('<88e',cuda.read(dp,176))
assert list(got)==expected,(got,expected)
print('PASS KV scatter real PTX noncontiguous rows=[7,3,5,1] FP16 old rows and guards unchanged',flush=True)
cuda.close()
