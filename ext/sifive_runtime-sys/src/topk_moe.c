#include <stdint.h>
#include <math.h>
#include <float.h>

void pacc_topk_row_v1(uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3,uint64_t a4,uint64_t a5,uint64_t a6) {
    const float *x=(const float *)*(uint64_t *)a0;
    float *out=(float *)*(uint64_t *)a1;
    unsigned n=*(uint64_t *)a2, k=*(uint64_t *)a3;
    unsigned flags=*(uint64_t *)a4;
    union {uint32_t u; float f;} clamp={.u=*(uint64_t *)a5},scale={.u=*(uint64_t *)a6};
    if(flags&64) {
        // Scalar accesses avoid a no-output failure observed with automatic
        // vectorization in the current PACC compilation/execution pipeline.
        volatile unsigned ids[1024]; unsigned pad=1;
        volatile float values[576];
        while(pad<n)pad*=2;
        for(unsigned i=0;i<pad;i++)ids[i]=i;
        for(unsigned i=0;i<n;i++)values[i]=x[i];
        // Match GGML's bitonic compare/swap network, including tie ordering.
        for(unsigned width=2;width<=pad;width*=2)for(unsigned gap=width/2;gap;gap/=2)
            for(unsigned col=0;col<pad;col++) {
                unsigned other=col^gap;if(other<=col)continue;
                unsigned a=ids[col],b=ids[other];int swap;
                if(!(col&width))swap=a>=n||(b<n&&((flags&128)?values[a]>values[b]:values[a]<values[b]));
                else swap=b>=n||(a<n&&((flags&128)?values[a]<values[b]:values[a]>values[b]));
                if(swap){ids[col]=b;ids[other]=a;}
            }
        for(unsigned i=0;i<n;i++){out[i]=values[i];out[n+i]=(float)ids[i];}
        return;
    }
    volatile float w[576], rank[576];
    float max=-FLT_MAX, sum=0;
    for (unsigned i=0;i<n;i++) if(x[i]>max) max=x[i];
    for (unsigned i=0;i<n;i++) {
        float v=x[i];
        if (!(flags&8)) v=(flags&1)?1.f/(1.f+exp2f(-v*1.4426950408889634f)):exp2f((v-max)*1.4426950408889634f);
        w[i]=v; sum+=v;
    }
    for (unsigned i=0;i<n;i++) {
        float v=w[i];
        if (!(flags&9)) v/=sum;
        if (v!=v) v=-FLT_MAX;
        w[i]=v; rank[i]=v+((flags&16)?x[n+i]:0.f);
    }
    sum=0;
    for (unsigned j=0;j<k;j++) {
        unsigned best=0;
        for (unsigned i=1;i<n;i++) if(rank[i]>rank[best]) best=i;
        out[j]=w[best]; out[k+j]=(float)best;
        sum+=w[best]; rank[best]=-INFINITY;
    }
    if (flags&4) {
        if(sum<clamp.f) sum=clamp.f;
        for(unsigned j=0;j<k;j++) out[j]/=sum;
    }
    if (flags&8) {
        max=-FLT_MAX; sum=0;
        for(unsigned j=0;j<k;j++) if(out[j]>max) max=out[j];
        for(unsigned j=0;j<k;j++) {out[j]=exp2f((out[j]-max)*1.4426950408889634f);sum+=out[j];}
        for(unsigned j=0;j<k;j++) out[j]/=sum;
    }
    for(unsigned j=0;j<k;j++) out[j]*=scale.f;
}
