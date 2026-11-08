#include <stdint.h>
#include <stddef.h>
extern unsigned f___zluda_ptx_impl_sreg_tid(unsigned char);
static void copy_bytes(uint8_t *d,const uint8_t *s,size_t n) {
    while(n) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %3, e8, m4, ta, ma\n"
                         "vle8.v v8, (%1)\n" "vse8.v v8, (%2)\n"
                         : "=&r"(vl):"r"(s),"r"(d),"r"(n)
                         :"v8","v9","v10","v11","memory");
        s+=vl;d+=vl;n-=vl;
    }
}
static uint16_t half_rne(uint32_t u) {
    unsigned sign=(u>>16)&0x8000,exp=(u>>23)&255,m=u&0x7fffff;
    if(exp==255)return sign|0x7c00|(m?0x200:0);
    int e=(int)exp-112;
    if(e>=31)return sign|0x7c00;
    if(e<=0) {
        if(e < -10)return sign;
        m|=0x800000;unsigned shift=14-e,h=m>>shift,rem=m&((1u<<shift)-1),mid=1u<<(shift-1);
        return sign|(h+(rem>mid||(rem==mid&&(h&1))));
    }
    unsigned h=m>>13,rem=m&8191;h+=(rem>4096||(rem==4096&&(h&1)));
    if(h==1024){h=0;e++;}
    return sign|((unsigned)e<<10)|h;
}
void pacc_concat_tile_v1(uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3,uint64_t a4,uint64_t a5,uint64_t a6) {
    if(f___zluda_ptx_impl_sreg_tid(0))return;
    const uint8_t *x=(const uint8_t *)*(uint64_t *)a0,*y=(const uint8_t *)*(uint64_t *)a1;
    uint8_t *out=(uint8_t *)*(uint64_t *)a2;
    size_t xb=*(uint64_t *)a3,yb=*(uint64_t *)a4,rows=*(uint64_t *)a5;
    unsigned mode=*(uint64_t *)a6;
    size_t bytes=rows*(xb+yb);
    if((mode&255)==2 || (mode&255)==3) {
        unsigned threads=mode>>8,n=xb/4;
        volatile float partial[512],warp[32];
        for(size_t r=0;r<rows;r++) {
            const float *p=(const float *)x+r*n;
            for(unsigned t=0;t<threads;t++) {
                volatile float sums[8]={0,0,0,0,0,0,0,0};
                for(unsigned i=t;i<n;)for(unsigned j=0;j<8;j++,i+=threads)
                    if(i<n)sums[j]+=p[i];
                float sum=0;for(unsigned j=0;j<8;j++)sum+=sums[j];partial[t]=sum;
            }
            // Equivalent lane-zero sum for CUDA's warp/block reduction tree.
            for(unsigned base=0;base<threads;base+=32) {
                for(unsigned gap=16;gap;gap/=2)for(unsigned t=0;t<gap;t++)
                    partial[base+t]+=partial[base+t+gap];
                warp[base/32]=partial[base];
            }
            for(unsigned t=threads/32;t<32;t++)warp[t]=0;
            for(unsigned gap=16;gap;gap/=2)for(unsigned t=0;t<gap;t++)warp[t]+=warp[t+gap];
            ((float *)out)[r]=(mode&255)==3?warp[0]/n:warp[0];
        }
        bytes=rows*4;
    } else if(mode==1) {
        for(size_t i=0;i<xb/4;i++)((uint16_t *)out)[i]=half_rne(((const uint32_t *)x)[i]);
        bytes=xb/2;
    } else {
        for(size_t r=0;r<rows;r++) {
            copy_bytes(out+r*(xb+yb),x+r*xb,xb);
            copy_bytes(out+r*(xb+yb)+xb,y+r*yb,yb);
        }
    }
    for(unsigned i=0;i<8;i++)out[bytes+i]=(uint8_t)(0xa0+i);
}
