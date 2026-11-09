#include <stdint.h>
extern uint32_t f___zluda_ptx_impl_sreg_tid(uint32_t);

void pacc_rms_exact_v1(uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3,uint64_t a4,uint64_t a5) {
    if(f___zluda_ptx_impl_sreg_tid(0))return;
    const float *x=(const float *)*(uint64_t *)a0;
    const float *w=(const float *)*(uint64_t *)a1;
    volatile float *y=(float *)*(uint64_t *)a2;
    uint64_t rows=*(uint64_t *)a3,hidden=*(uint64_t *)a4;
    union {uint32_t u;float f;} eps={.u=(uint32_t)*(uint64_t *)a5};
    for(uint64_t row=0;row<rows;++row) {
        // Match the CPU mathematical contract, without reduction reassociation/FMA.
        volatile double sum=0;
        for(uint64_t j=0;j<hidden;++j) {
            float value=x[row*hidden+j];
            volatile float square=value*value;
            sum+=(double)square;
        }
        float mean=(float)(sum/(double)hidden),root;
        float argument=mean+eps.f;
        __asm__ volatile("fsqrt.s %0, %1" : "=f"(root) : "f"(argument));
        float scale=1.0f/root;
        for(uint64_t j=0;j<hidden;++j) {
            volatile float normalized=x[row*hidden+j]*scale;
            y[row*hidden+j]=normalized*w[j];
        }
    }
    y[rows*hidden]=0;
}
