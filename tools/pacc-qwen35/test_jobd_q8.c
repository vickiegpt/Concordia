#define main jobd_program_main
#include "hetgpu_pacc_jobd.c"
#undef main
#include <assert.h>

static uint32_t seed = 0x531880;
static uint8_t random_byte(void) {
    seed = seed*1664525u+1013904223u;
    return seed >> 24;
}

int main(int argc, char **argv) {
    bool hardware = argc == 2 && !strcmp(argv[1], "--hardware");
    void *ggml = dlopen("/srv/lanxin/qwen35-a3b/llama.cpp/build-pacc-cuda-gcc14/bin/libggml-base.so", RTLD_NOW);
    assert(ggml);
    void (*dequant)(const void *, float *, int64_t) = dlsym(ggml, "dequantize_row_q8_0");
    assert(dequant);
    int (*submit)(int, const void *, const void *, float *, uint32_t, uint32_t,
                  uint32_t, uint32_t, uint32_t, uint32_t) = NULL;
    if (hardware) {
        const char *library = getenv("Q8_TEST_LIBRARY");
        assert(library);
        void *lib = dlopen(library, RTLD_NOW);
        if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        submit = dlsym(lib, "hetgpu_sifive_submit_q8_0_q8_1_tiled");
        assert(submit);
    }
    const unsigned shapes[][3] = {{33,1,32}, {1031,8,2048}, {3,3,32768}, {8192,2,2048}};
    for (unsigned t = 0; t < 4; ++t) {
        unsigned m=shapes[t][0], n=shapes[t][1], k=shapes[t][2];
        size_t xb=k/32*34, yb=k/32*36, sx=xb+16, sy=yb+16, sd=m+5;
        uint8_t *x=calloc(m,sx), *y=calloc(n,sy);
        float *out=malloc(n*sd*4), *xf=malloc(k*4);
        assert(x && y && out && xf);
        for (size_t i=0; i<n*sd; ++i) out[i]=NAN;
        for (unsigned row=0; row<m; ++row)
            for (unsigned b=0; b<k/32; ++b) {
                uint8_t *p=x+row*sx+b*34;
                uint16_t h=0x2000+(random_byte()<<2);
                if (row & 1) h |= 0x8000;
                memcpy(p,&h,2);
                for (unsigned j=0; j<32; ++j) p[2+j]=random_byte();
                p[2]=128; p[3]=127;
            }
        for (unsigned col=0; col<n; ++col)
            for (unsigned b=0; b<k/32; ++b) {
                uint8_t *p=y+col*sy+b*36;
                uint16_t h=0x2400+(random_byte()<<2);
                memcpy(p,&h,2);
                for (unsigned j=0; j<32; ++j) p[4+j]=random_byte();
                p[4]=128; p[5]=127;
            }
        struct MmvfJob job={0};
        job.x_type=4; job.reserved0=6; job.ncols2=k/2;
        job.grid_x=m; job.grid_y=job.grid_z=1; job.ncols_dst=n;
        job.stride_row=sx; job.stride_col_y2=sy; job.stride_col_dst=sd;
        job.x_bytes=m*sx; job.y_bytes=n*sy; job.dst_bytes=n*sd*4;
        uint64_t start=monotonic_us();
        int rc=hardware ? submit(0,x,y,out,m,n,k,sx,sy,sd) : run_q8_0_q8_1(&job,x,y,out);
        if (rc) { fprintf(stderr,"FAIL Q8 submit rc=%d\n",rc); return 1; }
        double worst=0;
        for (unsigned row=0; row<m; ++row) {
            dequant(x+row*sx,xf,k);
            for (unsigned col=0; col<n; ++col) {
                double ref=0;
                for (unsigned j=0; j<k; ++j) {
                    const uint8_t *p=y+col*sy+(j/32)*36;
                    uint16_t h; memcpy(&h,p,2);
                    ref += (double)xf[j]*pacc_f16_bits_to_f32(h)*(int8_t)p[4+j%32];
                }
                double value=out[col*sd+row], error=fabs(value-ref);
                if (!isfinite(value) || error > 0.0005+0.0001*fabs(ref)) {
                    fprintf(stderr,"FAIL Q8 row=%u col=%u value=%g ref=%g error=%g\n",row,col,value,ref,error);
                    return 1;
                }
                if (error>worst) worst=error;
            }
        }
        for (unsigned col=0; col<n; ++col)
            for (unsigned row=m; row<sd; ++row) assert(isnan(out[col*sd+row]));
        if (!hardware) {
            job.x_bytes=(m-1)*sx+xb-1;
            assert(run_q8_0_q8_1(&job,x,y,out));
            job.x_bytes=m*sx; job.grid_x=0;
            assert(run_q8_0_q8_1(&job,x,y,out));
            job.grid_x=m; job.reserved0=1;
            assert(run_q8_0_q8_1(&job,x,y,out));
        } else {
            assert(submit(0,x,y,out,m,n,k,sx-17,sy,sd));
            assert(submit(0,x,y,out,m,n,k+1,sx,sy,sd));
        }
        printf("PASS Q8 mode=%s m=%u n=%u k=%u max_abs_error=%.9g elapsed_us=%" PRIu64 "\n",
               hardware?"PACC":"unit",m,n,k,worst,monotonic_us()-start);
        free(x); free(y); free(out); free(xf);
    }
    return 0;
}
