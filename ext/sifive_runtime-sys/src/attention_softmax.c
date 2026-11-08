#include <stdint.h>
#include <math.h>
extern unsigned f___zluda_ptx_impl_sreg_tid(unsigned char);
void pacc_attention_softmax_v1(uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3){
    if(f___zluda_ptx_impl_sreg_tid(0))return;
    const float *input=(const float *)*(uint64_t *)a0;
    volatile float *out=(float *)*(uint64_t *)a1;
    unsigned rows=*(uint64_t *)a2,n=*(uint64_t *)a3;
    out[rows*n]=1;
    for(unsigned row=0;row<rows;row++){
        const float *p=input+row*(2*n+4),*x=p+4,*mask=x+n;
        float maxv=p[3]?p[2]:-INFINITY;
        for(unsigned col=0;col<n;col++){
            float v=x[col]*p[0]+mask[col]*p[1];
            out[row*n+col]=v;
            if(isnan(v)||v==INFINITY)return;
            if(v>maxv)maxv=v;
        }
        if(!isfinite(maxv))return;
        float sum=0;
        for(unsigned col=0;col<n;col++){
            float v=exp2f((out[row*n+col]-maxv)*1.4426950408889634f);
            out[row*n+col]=v;sum+=v;
        }
        if(p[3])sum+=exp2f((p[2]-maxv)*1.4426950408889634f);
        if(!isfinite(sum)||sum<=0)return;
        for(unsigned col=0;col<n;col++)out[row*n+col]/=sum;
    }
    out[rows*n]=0;
}
