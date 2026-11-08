#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>
extern unsigned f___zluda_ptx_impl_sreg_tid(unsigned char);
extern long syscall(long, ...);
static uint64_t qk_now_ns(void) {
    struct { long sec, ns; } ts;
    if (syscall(113L, 1L, &ts)) return 0;
    return (uint64_t)ts.sec*1000000000ULL+(uint64_t)ts.ns;
}
// Payload only: the caller owns the input span until completion. No control MMIO.
static __attribute__((noinline)) void qk_payload_copy(uint8_t *dst, const uint8_t *src, size_t bytes) {
    while(bytes>=8) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %3, e64, m4, ta, ma\n"
                         "vle64.v v8, (%1)\n"
                         "vse64.v v8, (%2)\n"
                         : "=&r"(vl) : "r"(src), "r"(dst), "r"(bytes/8)
                         : "v8", "v9", "v10", "v11", "memory");
        src+=vl*8;dst+=vl*8;bytes-=vl*8;
    }
    while(bytes--)*dst++=*src++;
}

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
void pacc_q4k_tile_v1(uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3,uint64_t a4,uint64_t a5,uint64_t a6,uint64_t a7,uint64_t a8,uint64_t a9) {
    if(f___zluda_ptx_impl_sreg_tid(0))return;
    const uint8_t *x=(const uint8_t *)*(uint64_t *)a0,*y=(const uint8_t *)*(uint64_t *)a1;
    volatile float *out=(float *)*(uint64_t *)a2;
    unsigned total_m=*(uint64_t *)a3,n=*(uint64_t *)a4,nb=*(uint64_t *)a5;
    unsigned gate=*(uint64_t *)a6;
    unsigned qtype=*(uint64_t *)a7,block_bytes=qtype==8?272:(qtype==5?176:144);
    const float *bias=(const float *)*(uint64_t *)a8;
    unsigned options=*(uint64_t *)a9,profile=options&1;
    uint64_t started=profile?qk_now_ns():0, activation_ns=0, weights_ns=0, xm_ns=0, reduce_ns=0, calls=0, copy_ns=0;
    volatile float *diag=out+total_m*n;
    for(unsigned i=0;i<16;i++)diag[i]=0;
    if(profile && !started){diag[0]=-3;return;}
    if(options&2) {
        size_t xb=(size_t)total_m*nb*block_bytes*(gate+1),yb=(size_t)n*nb*288,bb=(size_t)total_m*8;
        size_t yo=(xb+63)&~(size_t)63,bo=yo+((yb+63)&~(size_t)63),bytes=bo+bb;
        if(bytes>512*1024){diag[0]=-4;return;}
        uint8_t *cache=(uint8_t *)(((uintptr_t)__builtin_alloca(bytes+63)+63)&~(uintptr_t)63);
        uint64_t before=profile?qk_now_ns():0;
        qk_payload_copy(cache,x,xb);qk_payload_copy(cache+yo,y,yb);
        qk_payload_copy(cache+bo,(const uint8_t *)bias,bb);
        x=cache;y=cache+yo;bias=(const float *)(cache+bo);
        if(profile)copy_ns=qk_now_ns()-before;
    }
    volatile float sums[2][256];
    volatile uint16_t a[1024],b[256];float dots[256];
    for(unsigned row0=0;row0<total_m;row0+=32){
    unsigned m=total_m-row0<32?total_m-row0:32;
    for(unsigned t=0;t<2;t++)for(unsigned i=0;i<256;i++)sums[t][i]=0;
    for(unsigned block=0;block<nb;block++)for(unsigned group=0;group<8;group++){
        uint64_t phase=profile?qk_now_ns():0;
        float dy[8];int sy[8];
        for(unsigned col=0;col<n;col++){
            const uint8_t *v=y+(col*nb*8+block*8+group)*36;
            dy[col]=q4_half(v[0]|(v[1]<<8));sy[col]=0;
            for(unsigned lane=0;lane<32;lane++){
                int q=(int8_t)v[4+lane];sy[col]+=q;b[lane*n+col]=q4_bf(q);
            }
        }
        if(profile)activation_ns+=qk_now_ns()-phase;
        for(unsigned t=0;t<=gate;t++){
            phase=profile?qk_now_ns():0;
            float dx[32],dm[32];unsigned sc[32],mi[32];
            for(unsigned r=0;r<m;r++){
                const uint8_t *v=x+((t*total_m+row0+r)*nb+block)*block_bytes;
                if(qtype==8)v+=group*34;
                dx[r]=q4_half(v[0]|(v[1]<<8));dm[r]=q4_half(v[2]|(v[3]<<8));
                sc[r]=q4_scale(v+4,group);mi[r]=q4_min(v+4,group);
                if(qtype==8){dm[r]=0;sc[r]=1;mi[r]=0;}
                if(options&4) {
                    for(unsigned lane=0;lane<32;) {
                        size_t vl=__riscv_vsetvl_e8m1(32-lane);
                        vint16m2_t qi;
                        if(qtype==8) {
                            qi=__riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1((const int8_t *)v+2+lane,vl),vl);
                        } else {
                            vuint8m1_t q=__riscv_vle8_v_u8m1(v+(qtype==5?48:16)+(group/2)*32+lane,vl);
                            q=__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q,(group%2)*4,vl),15,vl);
                            if(qtype==5) {
                                vuint8m1_t hi=__riscv_vle8_v_u8m1(v+16+lane,vl);
                                hi=__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(hi,group,vl),1,vl);
                                q=__riscv_vor_vv_u8m1(q,__riscv_vsll_vx_u8m1(hi,4,vl),vl);
                            }
                            qi=__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(q,vl));
                        }
                        vfloat32m4_t f=__riscv_vfwcvt_f_x_v_f32m4(qi,vl);
                        vuint16m2_t bf=__riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_f32m4_u32m4(f),16,vl);
                        __riscv_vsse16_v_u16m2((uint16_t *)a+lane*m+r,m*2,bf,vl);
                        lane+=vl;
                    }
                } else for(unsigned lane=0;lane<32;lane++){
                    int q;
                    if(qtype==8)q=(int8_t)v[2+lane];
                    else{
                        q=(v[(qtype==5?48:16)+(group/2)*32+lane]>>((group%2)*4))&15;
                        if(qtype==5)q|=((v[16+lane]>>group)&1)<<4;
                    }
                    a[lane*m+r]=q4_bf(q);
                }
            }
            if(profile){weights_ns+=qk_now_ns()-phase;phase=qk_now_ns();}
            if(xsfmm_native_bf16((const uint16_t *)a,(const uint16_t *)b,dots,m,n,32)){diag[0]=-2;return;}
            calls++;
            if(profile){xm_ns+=qk_now_ns()-phase;phase=qk_now_ns();}
            for(unsigned r=0;r<m;r++)for(unsigned col=0;col<n;col++){
                float dot=dots[r*n+col];
                if(!isfinite(dot)||fabsf(dot)>524288.f||dot!=(float)(int)dot){diag[0]=-1;diag[1]=dot;diag[2]=group;diag[3]=(row0+r)*n+col;return;}
                sums[t][r*n+col]+=dy[col]*(dx[r]*sc[r]*dot-dm[r]*mi[r]*sy[col]);
            }
            if(profile)reduce_ns+=qk_now_ns()-phase;
        }
    }
    for(unsigned i=0;i<m*n;i++){
        float up=sums[0][i]+bias[row0+i/n];
        float g=sums[1][i]+bias[total_m+row0+i/n];
        out[row0*n+i]=gate?up*(g/(1.f+exp2f(-g*1.4426950408889634f))):up;
    }
    }
    if(profile){
        diag[4]=(qk_now_ns()-started)/1000.0;
        diag[5]=activation_ns/1000.0;diag[6]=weights_ns/1000.0;
        diag[7]=xm_ns/1000.0;diag[8]=reduce_ns/1000.0;diag[9]=calls;diag[10]=copy_ns/1000.0;
    }
}
