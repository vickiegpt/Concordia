#include <assert.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Layout { uint64_t hidden, dims[3], xs[3], ws[4], wt[3]; float eps; };
_Static_assert(sizeof(struct Layout)==120, "RMSNorm layout ABI mismatch");
typedef int (*Submit)(int,const float *,const float *,float *,const struct Layout *);
int main(void) {
    const char *path=getenv("RMS_TEST_LIBRARY"); assert(path);
    void *lib=dlopen(path,RTLD_NOW); if (!lib) { puts(dlerror()); return 1; }
    Submit submit=(Submit)dlsym(lib,"hetgpu_sifive_submit_rmsnorm_strided_f32"); assert(submit);
    struct Layout cases[]={
        {256,{16,2,1},{512,8192,0},{256,1,1,1},{0,0,0},1e-5f},
        {256,{16,2,1},{512,8192,0},{128,2,1,1},{140,0,0},1e-5f},
        {7,{3,2,2},{12,44,100},{3,2,2,1},{5,20,0},1e-5f},
        {2048,{4,1,1},{2048,0,0},{2048,1,1,1},{0,0,0},1e-5f},
    };
    for (unsigned t=0;t<4;++t) {
        struct Layout l=cases[t];
        size_t count=l.hidden*l.dims[0]*l.dims[1]*l.dims[2], xs=l.hidden, ws=l.ws[0];
        for (unsigned i=0;i<3;++i) { xs+=(l.dims[i]-1)*l.xs[i]; ws+=(l.ws[i+1]-1)*l.wt[i]; }
        float *x=malloc(xs*4), *w=malloc(ws*4), *out=malloc((count+8)*4);
        assert(x&&w&&out);
        for (size_t i=0;i<xs;++i) x[i]=(float)((int)(i%71)-35)*0.03125f;
        for (size_t i=0;i<ws;++i) w[i]=(float)((int)(i%11)-5)*0.125f;
        for (size_t i=0;i<count+8;++i) out[i]=NAN;
        assert(submit(0,x,w,out,&l)==0);
        double worst=0; size_t row=0;
        for (size_t s=0;s<l.dims[2];++s) for (size_t c=0;c<l.dims[1];++c) for (size_t r=0;r<l.dims[0];++r,++row) {
            size_t at=r*l.xs[0]+c*l.xs[1]+s*l.xs[2];
            size_t wt=(r%l.ws[1])*l.wt[0]+(c%l.ws[2])*l.wt[1]+(s%l.ws[3])*l.wt[2];
            double sum=0; for (size_t j=0;j<l.hidden;++j) sum+=(double)x[at+j]*x[at+j];
            double scale=1.0/sqrt(sum/l.hidden+l.eps);
            for (size_t j=0;j<l.hidden;++j) {
                double ref=x[at+j]*scale*w[wt+j%l.ws[0]], value=out[row*l.hidden+j], error=fabs(value-ref);
                if (!isfinite(value) || error>2e-5+2e-5*fabs(ref)) {
                    fprintf(stderr,"FAIL RMS case=%u row=%zu col=%zu got=%g expected=%g\n",t,row,j,value,ref); return 1;
                }
                if (error>worst) worst=error;
            }
        }
        for(size_t i=count;i<count+8;++i) assert(isnan(out[i]));
        l.hidden=0; assert(submit(0,x,w,out,&l)!=0);
        l=cases[t]; l.ws[0]=0; assert(submit(0,x,w,out,&l)!=0);
        printf("PASS RMS hardware case=%u rows=%zu hidden=%llu max_abs_error=%.9g\n",t,count/cases[t].hidden,(unsigned long long)cases[t].hidden,worst);
        free(x);free(w);free(out);
    }
    float x[4]={1,2,3,4}, out[4]={99,99,99,99};
    struct Layout l={4,{1,1,1},{4,0,0},{1,1,1,1},{0,0,0},1e-5f};
    const char *unsafe[]={"HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK", "HETGPU_SIFIVE_SKIP_JOB_WAIT",
                         "HETGPU_SIFIVE_RMSNORM_ASSUME_SUCCESS_ON_OUTPUT_TIMEOUT",
                         "HETGPU_SIFIVE_RMSNORM_STAGE_SHARED_DDR"};
    for (unsigned i=0;i<4;++i) {
        const char *old=getenv(unsafe[i]); char *saved=old?strdup(old):NULL;
        setenv(unsafe[i],i==3?"0":"1",1);
        assert(submit(0,x,NULL,out,&l)!=0);
        for (unsigned j=0;j<4;++j) assert(out[j]==99);
        if (saved) { setenv(unsafe[i],saved,1); free(saved); } else unsetenv(unsafe[i]);
    }
    puts("PASS RMS unsafe completion/host policy rejected without output writes");
    return 0;
}
