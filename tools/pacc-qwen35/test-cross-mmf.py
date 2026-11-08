#!/usr/bin/env python3
import ctypes as c
import math
import struct
from cuda_abi_test_utils import Cuda, U

cuda=Cuda()
for half in (False,True):
  for n in (8,16):
    m,k,heads,samples=32,64,4,2;cr,sr=2,2
    sx=k+4;sy=k+8;sd=m+4;cx=m*sx+8;cy=n*sy+8;cd=n*sd+8
    px=2*cx+8;py=4*cy+8;pd=4*cd+8;vals=2 if half else 1
    x=[((i*7)%53-26)*.003137 for i in range(px)]
    xb=struct.pack('<%d%s'%(len(x),'e' if half else 'f'),*x)
    x=struct.unpack('<%d%s'%(len(x),'e' if half else 'f'),xb)
    y=[c.c_float(((i*17)%71-35)*.004319).value for i in range(samples*py)]
    expected=[12345.]*(samples*pd)
    for sample in range(samples):
      for head in range(heads):
        for col in range(n):
          for row in range(m):
            expected[sample*pd+head*cd+col*sd+row]=sum(x[(head//cr)*cx+row*sx+j]*y[sample*py+head*cy+col*sy+j] for j in range(k))
    xp=cuda.alloc(xb);yp=cuda.alloc(struct.pack('<%df'%len(y),*y))
    out=cuda.alloc(struct.pack('<%df'%len(expected),*([12345.]*len(expected))))
    params=[xp,yp,U(0),out]+[c.c_int(i) for i in (k//vals,n,heads,sx//vals,sy//vals,sd,0,0,cr,cx//vals,cy,cd,sr,px//vals,py,pd)]
    name=f'_ZL9mul_mat_fI{"7__half2" if half else "f"}Li32ELi{n}ELi8ELb0EEvPKT_PKfPKiPfiiiiiiiiiiiiiiii'
    cuda.launch(name,params,(m//32,heads,samples),(32,8,1))
    actual=struct.unpack('<%df'%len(expected),cuda.read(out,len(expected)*4))
    assert all(math.isfinite(a) and abs(a-b)<2e-5 for a,b in zip(actual,expected)),(half,n)
    print(f'PASS MMF device math half={half} n={n} broadcast_heads=4 samples=2 guards=unchanged',flush=True)
cuda.close()
