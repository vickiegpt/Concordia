#define main jobd_program_main
#ifdef TEST_SIFIVE_JOBD
#include "hetgpu_sifive_jobd.c"
#define pacc_f16_bits_to_f32 sifive_f16_bits_to_f32
#else
#include "hetgpu_pacc_jobd.c"
#endif
#undef main
#include <assert.h>

static uint32_t seed = 0x12345678;
static uint32_t next_random(void) {
    seed = seed * 1664525u + 1013904223u;
    return seed;
}

int main(int argc, char **argv) {
    bool large = argc == 2 && strcmp(argv[1], "--hardware-large") == 0;
    bool hardware = large || (argc == 2 && strcmp(argv[1], "--hardware") == 0);
    const char *ggml_path = getenv("GGML_BASE_LIBRARY");
    if (!ggml_path) ggml_path = "/srv/lanxin/qwen35-a3b/llama.cpp/build-pacc-cuda-gcc14/bin/libggml-base.so";
    void *ggml = dlopen(ggml_path, RTLD_NOW);
    if (!ggml) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    void (*dequant)(const void *, float *, int64_t) = dlsym(ggml, "dequantize_row_q6_K");
    assert(dequant);
    int (*submit)(int, const void *, const void *, float *, uint32_t, uint32_t,
                  uint32_t, uint32_t, uint32_t, uint32_t) = NULL;
    if (hardware) {
        void *lib = dlopen("/home/ubuntu/Concordia/target-pacc/release/libnvcuda.so", RTLD_NOW);
        if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        submit = dlsym(lib, "hetgpu_sifive_submit_q6k_q8_1_tiled");
        assert(submit);
    }
    const unsigned shapes[][3] = {{17, 1, 256}, {1031, 2, 2048}, {3, 1, 8192}, {1031, 8, 2048}, {248320, 1, 2048}};
    for (unsigned test = 0; test < (large ? 5u : 4u); ++test) {
        unsigned m = shapes[test][0], n = shapes[test][1], k = shapes[test][2];
        size_t xb = k / 256 * 210, yb = k / 32 * 36;
        size_t sx = xb + 16, sy = yb + 16, sd = m + 5;
        uint8_t *x = calloc(m, sx), *y = calloc(n, sy);
        float *out = malloc(n * sd * sizeof(float)), *xf = malloc(k * sizeof(float));
        assert(x && y && out && xf);
        for (size_t i = 0; i < n * sd; ++i) out[i] = NAN;
        for (unsigned row = 0; row < m; ++row) {
            for (unsigned b = 0; b < k / 256; ++b) {
                uint8_t *p = x + row * sx + b * 210;
                for (unsigned j = 0; j < 192; ++j) p[j] = next_random() >> 24;
                for (unsigned j = 192; j < 208; ++j) p[j] = (int8_t)((next_random() >> 24) % 31 - 15);
                uint16_t dh = 0x2400 + (next_random() & 0x3ff);
                memcpy(p + 208, &dh, 2);
            }
        }
        for (unsigned col = 0; col < n; ++col) {
            for (unsigned b = 0; b < k / 32; ++b) {
                uint8_t *p = y + col * sy + b * 36;
                uint16_t dh = 0x2800 + (next_random() & 0x3ff);
                memcpy(p, &dh, 2);
                for (unsigned j = 0; j < 32; ++j) p[4+j] = next_random() >> 24;
            }
        }
        struct MmvfJob job = {0};
        job.x_type=5; job.reserved0=6; job.ncols2=k/2;
        job.grid_x=m; job.grid_y=job.grid_z=1; job.ncols_dst=n;
        job.stride_row=sx; job.stride_col_y2=sy; job.stride_col_dst=sd;
        job.x_bytes=m*sx; job.y_bytes=n*sy; job.dst_bytes=n*sd*4;
        uint64_t start = monotonic_us();
        int rc = hardware ? submit(0, x, y, out, m, n, k, sx, sy, sd)
                          : run_q6k_q8_1(&job, x, y, out);
        if (rc) { fprintf(stderr, "FAIL submit rc=%d\n", rc); return 1; }
        double worst = 0;
        for (unsigned row = 0; row < m; ++row) {
            dequant(x + row*sx, xf, k);
            for (unsigned col = 0; col < n; ++col) {
                double expected = 0;
                for (unsigned j = 0; j < k; ++j) {
                    const uint8_t *p = y + col*sy + (j/32)*36;
                    uint16_t dh; memcpy(&dh, p, 2);
                    expected += (double)xf[j] * pacc_f16_bits_to_f32(dh) * (int8_t)p[4+j%32];
                }
                double actual = out[col*sd+row], error = fabs(actual-expected);
                if (!isfinite(actual) || error > 0.0005 + 0.0001*fabs(expected)) {
                    fprintf(stderr, "FAIL row=%u col=%u actual=%g reference=%g error=%g\n", row,col,actual,expected,error);
                    return 1;
                }
                if (error > worst) worst=error;
            }
        }
        for (unsigned col=0; col<n; ++col)
            for (unsigned row=m; row<sd; ++row) assert(isnan(out[col*sd+row]));
        printf("PASS mode=%s m=%u n=%u k=%u max_abs_error=%.9g elapsed_us=%" PRIu64 "\n",
               hardware ? "PACC" : "unit",m,n,k,worst,monotonic_us()-start);
        if (!hardware) {
            --job.x_bytes; job.x_bytes = (m-1)*sx + xb - 1;
            assert(run_q6k_q8_1(&job,x,y,out) != 0);
            job.x_bytes=m*sx; job.reserved0=1;
            assert(run_q6k_q8_1(&job,x,y,out) != 0);
        }
        free(x); free(y); free(out); free(xf);
    }
    return 0;
}
