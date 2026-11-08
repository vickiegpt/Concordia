#!/usr/bin/env python3
"""Q4_K exact integer XM dot validation; deliberately nonzero minima/scales."""
import ctypes as c
import math
import os
import struct

lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
qtype=int(os.environ.get('QK_TEST_TYPE','4'))
assert qtype in (4,5,8)
block_bytes=272 if qtype==8 else (144 if qtype==4 else 176)
fn=getattr(lib,'hetgpu_sifive_submit_q8_tile' if qtype==8 else f'hetgpu_sifive_submit_q{qtype}k_tile')
fn.argtypes=[c.c_int,c.c_void_p,c.c_void_p,c.c_void_p]+[c.c_uint]*4
fn.restype=c.c_int
def half(b):return struct.unpack('<e',b)[0]
def generate(m,n,nb,gate):
    x=bytearray();y=bytearray()
    for r in range(m*(1+gate)):
        for block in range(nb):
            if qtype==8:
                for group in range(8):
                    x+=struct.pack('<e',.00073*(1+(r+group)%3))
                    x+=bytes(((i*7+r*11+block*3+group*5)%255-127)&255 for i in range(32))
                continue
            x+=struct.pack('<ee',.00073*(1+r%3),.00021*(1+block%3))
            x+=bytes((i*17+r*3+block*5)%256 for i in range(12))
            if qtype==5:x+=bytes((i*23+r*5+block*7)%256 for i in range(32))
            x+=bytes((i*7+r*11+block*3)%256 for i in range(128))
    for col in range(n):
        for block in range(nb*8):
            y+=struct.pack('<ee',.00313*(1+col%3),999.)
            y+=bytes(((i*13+block*7+col*11)%255-127)&255 for i in range(32))
    return bytes(x),bytes(y)
def reference(x,y,m,n,nb,gate):
    result=[]
    for r in range(m):
        for col in range(n):
            totals=[]
            for t in range(1+gate):
                s=0.
                for block in range(nb):
                    v=x[((t*m+r)*nb+block)*block_bytes:][:block_bytes]
                    dx=half(v[:2]);dm=half(v[2:4]);sc=v[4:16]
                    for g in range(8):
                        scale=sc[g]&63 if g<4 else (sc[g+4]&15)|((sc[g-4]>>6)<<4)
                        minimum=sc[g+4]&63 if g<4 else (sc[g+4]>>4)|((sc[g]>>6)<<4)
                        if qtype==8:
                            dx=half(v[g*34:g*34+2]);dm=0;scale=1;minimum=0
                        b=y[((col*nb*8+block*8+g)*36):][:36];dy=half(b[:2])
                        for lane in range(32):
                            q=(v[(16 if qtype==4 else 48)+(g//2)*32+lane]>>(g%2*4))&15
                            if qtype==5:q|=((v[16+lane]>>g)&1)<<4
                            if qtype==8:
                                q=v[g*34+2+lane];q=q if q<128 else q-256
                            u=b[4+lane];u=u if u<128 else u-256
                            s+=(dx*scale*q-dm*minimum)*dy*u
                totals.append(s)
            result.append(totals[0]*(totals[1]/(1+math.exp(-totals[1]))) if gate else totals[0])
    return result

for m,n,nb,gate in [(3,1,1,0),(32,4,2,0),(7,3,3,1),(32,1,8,1),(129,3,2,1),(384 if qtype==8 else 512,1,8,1)]:
    x,y=generate(m,n,nb,gate);ref=reference(x,y,m,n,nb,gate)
    out=(c.c_float*(m*n+8))(*([math.nan]*(m*n)+[12345.]*8))
    assert fn(0,c.c_char_p(x),c.c_char_p(y),out,m,n,nb,gate)==0,(m,n,nb,gate)
    err=max(abs(out[i]-v) for i,v in enumerate(ref))
    assert all(math.isfinite(out[i]) and abs(out[i]-v)<5e-5+5e-5*abs(v) for i,v in enumerate(ref)),(m,n,nb,gate,err,list(out)[:8],ref[:8])
    assert list(out)[m*n:]==[12345.]*8
    print(f'PASS Q{qtype}_K XM m={m} n={n} k={nb*256} gate={gate} max_abs_error={err:.9g}',flush=True)
