#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern int xsfmm_native_bf16(const uint16_t *, const uint16_t *, float *,
                            size_t, size_t, size_t);

static float half_float(uint16_t h) {
    unsigned sign = (unsigned)(h & 0x8000) << 16;
    unsigned e = (h >> 10) & 31, mant = h & 1023, bits;
    if (!e) {
        if (!mant) bits = sign;
        else {
            int exp = -14;
            while (!(mant & 1024)) { mant <<= 1; --exp; }
            bits = sign | (unsigned)(exp+127) << 23 | (mant & 1023) << 13;
        }
    } else bits = sign | (e == 31 ? 255 : e+112) << 23 | mant << 13;
    float value;
    memcpy(&value, &bits, 4);
    return value;
}

static uint16_t integer_bf16(int value) {
    float f = (float)value;
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return bits >> 16;
}

/* Integers [-128,127] are exact BF16. Apply Q6 scales after two K=16
 * XM dots so no rounded BF16 weight*scale product enters the computation. */
int hetgpu_q6k_q8_1_batch_xm(const uint8_t *restrict x, const uint8_t *restrict y,
                           float *restrict dst, size_t m, size_t n, size_t nb,
                           size_t sx, size_t sy, size_t sd) {
    if (!m || !n || n > 8 || !nb || nb > 128 || sx < nb*210 || sy < nb*288 || sd < m)
        return -1;
    uint16_t *yb = malloc(nb*256*n*sizeof(*yb));
    float *dy = malloc(nb*8*n*sizeof(*dy));
    if (!yb || !dy) { free(yb); free(dy); return -1; }
    for (size_t block = 0; block < nb*8; ++block)
        for (size_t col = 0; col < n; ++col) {
            const uint8_t *src = y+col*sy+block*36;
            uint16_t h;
            memcpy(&h, src, 2);
            dy[block*n+col] = half_float(h);
            for (unsigned lane = 0; lane < 32; ++lane)
                yb[(block*32+lane)*n+col] = integer_bf16((int8_t)src[4+lane]);
        }
    for (size_t row0 = 0; row0 < m; row0 += 32) {
        const size_t rows = m-row0 < 32 ? m-row0 : 32;
        float sums[32*8] = {0}, dx[32], dots[2][32*8];
        int8_t scales[16][32];
        uint16_t xb[256*32];
        for (size_t block = 0; block < nb; ++block) {
            for (size_t row = 0; row < rows; ++row) {
                const uint8_t *src = x+(row0+row)*sx+block*210;
                uint16_t h;
                memcpy(&h, src+208, 2);
                dx[row] = half_float(h);
                for (unsigned s = 0; s < 16; ++s) scales[s][row] = (int8_t)src[192+s];
                for (unsigned half = 0; half < 2; ++half)
                    for (unsigned group = 0; group < 4; ++group)
                        for (unsigned lane = 0; lane < 32; ++lane) {
                            unsigned ql = src[half*64+(group%2)*32+lane];
                            unsigned qh = src[128+half*32+lane];
                            int q = (int)(((ql >> ((group/2)*4)) & 15) |
                                          (((qh >> (group*2)) & 3) << 4))-32;
                            xb[((half*4+group)*32+lane)*rows+row] = integer_bf16(q);
                        }
            }
            for (unsigned group = 0; group < 8; ++group) {
                for (unsigned half = 0; half < 2; ++half) {
                    if (xsfmm_native_bf16(xb+(group*32+half*16)*rows,
                        yb+((block*8+group)*32+half*16)*n, dots[half], rows, n, 16)) {
                        free(yb); free(dy); return -1;
                    }
                }
                for (size_t row = 0; row < rows; ++row)
                    for (size_t col = 0; col < n; ++col) {
                        size_t at = row*n+col;
                        float f0 = dots[0][at], f1 = dots[1][at];
                        if (!isfinite(f0) || !isfinite(f1) || fabsf(f0) > 65536 || fabsf(f1) > 65536 ||
                            f0 != (float)(int32_t)f0 || f1 != (float)(int32_t)f1) {
                            free(yb); free(dy); return -1;
                        }
                        int32_t dot = (int32_t)f0*scales[group*2][row] +
                                      (int32_t)f1*scales[group*2+1][row];
                        sums[at] += (dx[row]*dy[(block*8+group)*n+col])*(float)dot;
                    }
            }
        }
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < n; ++col) {
                if (!isfinite(sums[row*n+col])) { free(yb); free(dy); return -1; }
                dst[col*sd+row0+row] = sums[row*n+col];
            }
    }
    free(yb); free(dy);
    return 0;
}
