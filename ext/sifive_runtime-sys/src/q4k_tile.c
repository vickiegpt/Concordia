#include <stdint.h>
#include <math.h>
extern unsigned f___zluda_ptx_impl_sreg_tid(unsigned char);

static float q4_half(uint16_t h) {
    unsigned sign=(unsigned)(h&32768)<<16,e=(h>>10)&31,m=h&1023,b;
    if(!e){if(!m)b=sign;else{int ex=-14;while(!(m&1024)){m<<=1;ex--;}b=sign|((unsigned)(ex+127)<<23)|((m&1023)<<13);}}
    else b=sign|((e==31?255:e+112)<<23)|(m<<13);
    union {unsigned u;float f;} v={.u=b};return v.f;
}
static uint16_t q4_bf(int q){union{float f;uint32_t u;}v={.f=(float)q};return v.u>>16;}
static unsigned q4_scale(const uint8_t *s,unsigned g){return g<4?s[g]&63:(s[g+4]&15)|((s[g-4]>>6)<<4);}
static unsigned q4_min(const uint8_t *s,unsigned g){return g<4?s[g+4]&63:(s[g+4]>>4)|((s[g]>>6)<<4);}

// Integer dot products are exact BF16/FP32 XM operations. Scaling and the
// zero-point correction stay outside XM, preserving the Q4_K representation.
void pacc_q4k_tile_v1(uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3,uint64_t a4,uint64_t a5,uint64_t a6,uint64_t a7,uint64_t a8) {
    if(f___zluda_ptx_impl_sreg_tid(0))return;
    const uint8_t *x=(const uint8_t *)*(uint64_t *)a0,*y=(const uint8_t *)*(uint64_t *)a1;
    volatile float *out=(float *)*(uint64_t *)a2;
    unsigned total_m=*(uint64_t *)a3,n=*(uint64_t *)a4,nb=*(uint64_t *)a5;
    unsigned gate=*(uint64_t *)a6;
    unsigned qtype=*(uint64_t *)a7,block_bytes=qtype==8?272:(qtype==5?176:144);
    const float *bias=(const float *)*(uint64_t *)a8;
    volatile float *diag=out+total_m*n;
    for(unsigned i=0;i<4;i++)diag[i]=0;
    volatile float sums[2][256];
    volatile uint16_t a[1024],b[256];float dots[256];
    for(unsigned row0=0;row0<total_m;row0+=32){
    unsigned m=total_m-row0<32?total_m-row0:32;
    for(unsigned t=0;t<2;t++)for(unsigned i=0;i<256;i++)sums[t][i]=0;
    for(unsigned block=0;block<nb;block++)for(unsigned group=0;group<8;group++){
        float dy[8];int sy[8];
        for(unsigned col=0;col<n;col++){
            const uint8_t *v=y+(col*nb*8+block*8+group)*36;
            dy[col]=q4_half(v[0]|(v[1]<<8));sy[col]=0;
            for(unsigned lane=0;lane<32;lane++){
                int q=(int8_t)v[4+lane];sy[col]+=q;b[lane*n+col]=q4_bf(q);
            }
        }
        for(unsigned t=0;t<=gate;t++){
            float dx[32],dm[32];unsigned sc[32],mi[32];
            for(unsigned r=0;r<m;r++){
                const uint8_t *v=x+((t*total_m+row0+r)*nb+block)*block_bytes;
                if(qtype==8)v+=group*34;
                dx[r]=q4_half(v[0]|(v[1]<<8));dm[r]=q4_half(v[2]|(v[3]<<8));
                sc[r]=q4_scale(v+4,group);mi[r]=q4_min(v+4,group);
                if(qtype==8){dm[r]=0;sc[r]=1;mi[r]=0;}
                for(unsigned lane=0;lane<32;lane++){
                    int q;
                    if(qtype==8)q=(int8_t)v[2+lane];
                    else{
                        q=(v[(qtype==5?48:16)+(group/2)*32+lane]>>((group%2)*4))&15;
                        if(qtype==5)q|=((v[16+lane]>>group)&1)<<4;
                    }
                    a[lane*m+r]=q4_bf(q);
                }
            }
            if(xsfmm_native_bf16((const uint16_t *)a,(const uint16_t *)b,dots,m,n,32)){diag[0]=-2;return;}
            for(unsigned r=0;r<m;r++)for(unsigned col=0;col<n;col++){
                float dot=dots[r*n+col];
                if(!isfinite(dot)||fabsf(dot)>524288.f||dot!=(float)(int)dot){diag[0]=-1;diag[1]=dot;diag[2]=group;diag[3]=(row0+r)*n+col;return;}
                sums[t][r*n+col]+=dy[col]*(dx[r]*sc[r]*dot-dm[r]*mi[r]*sy[col]);
            }
        }
    }
    for(unsigned i=0;i<m*n;i++){
        float up=sums[0][i]+bias[row0+i/n];
        float g=sums[1][i]+bias[total_m+row0+i/n];
        out[row0*n+i]=gate?up*(g/(1.f+exp2f(-g*1.4426950408889634f))):up;
    }
    }
}
