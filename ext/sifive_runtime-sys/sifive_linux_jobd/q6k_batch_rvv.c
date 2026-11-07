#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>

static float half_to_float(uint16_t h) {
    unsigned sign = (unsigned)(h & 0x8000) << 16;
    unsigned exponent = (h >> 10) & 31, mantissa = h & 1023, bits;
    if (exponent == 0) {
        if (!mantissa) bits = sign;
        else {
            int e = -14;
            while (!(mantissa & 1024)) { mantissa <<= 1; --e; }
            bits = sign | (unsigned)(e+127) << 23 | (mantissa & 1023) << 13;
        }
    } else bits = sign | (exponent == 31 ? 255 : exponent+112) << 23 | mantissa << 13;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* RVV integer dot products, not XM. Packed weights are reused across columns. */
int hetgpu_q6k_q8_1_batch_rvv(const uint8_t *restrict x, const uint8_t *restrict y,
                             float *restrict dst, size_t m, size_t n, size_t nb,
                             size_t sx, size_t sy, size_t sd) {
    if (!n || n > 8 || !nb || nb > 128) return -1;
    float *dy = malloc(n*nb*8*sizeof(*dy));
    if (!dy) return -1;
    for (size_t col = 0; col < n; ++col)
        for (size_t block = 0; block < nb*8; ++block) {
            uint16_t h;
            memcpy(&h, y+col*sy+block*36, 2);
            dy[col*nb*8+block] = half_to_float(h);
        }
    for (size_t row = 0; row < m; ++row) {
        float sums[8] = {0};
        for (size_t block = 0; block < nb; ++block) {
            const uint8_t *xb = x+row*sx+block*210;
            uint16_t h;
            memcpy(&h, xb+208, 2);
            float dx = half_to_float(h);
            for (unsigned half = 0; half < 2; ++half)
                for (unsigned group = 0; group < 4; ++group) {
                    int16_t values[32];
                    for (unsigned lane = 0; lane < 32; ++lane) {
                        unsigned ql = xb[half*64+(group%2)*32+lane];
                        unsigned qh = xb[128+half*32+lane];
                        int q = (int)(((ql >> ((group/2)*4)) & 15) |
                                      (((qh >> (group*2)) & 3) << 4))-32;
                        values[lane] = q*(int8_t)xb[192+half*8+group*2+lane/16];
                    }
                    size_t yblock = block*8+half*4+group;
                    for (size_t col = 0; col < n; ++col) {
                        const int8_t *qs = (const int8_t *)(y+col*sy+yblock*36+4);
                        int32_t dot = 0;
                        for (size_t lane = 0; lane < 32;) {
                            size_t vl = __riscv_vsetvl_e16m1(32-lane);
                            vint16m1_t vx = __riscv_vle16_v_i16m1(values+lane, vl);
                            vint8mf2_t vy8 = __riscv_vle8_v_i8mf2(qs+lane, vl);
                            vint16m1_t vy = __riscv_vsext_vf2_i16m1(vy8, vl);
                            vint32m2_t product = __riscv_vwmul_vv_i32m2(vx, vy, vl);
                            vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
                            vint32m1_t sum = __riscv_vredsum_vs_i32m2_i32m1(product, zero, vl);
                            dot += __riscv_vmv_x_s_i32m1_i32(sum);
                            lane += vl;
                        }
                        sums[col] += (dx*dy[col*nb*8+yblock])*(float)dot;
                    }
                }
        }
        for (size_t col = 0; col < n; ++col) {
            if (!isfinite(sums[col])) { free(dy); return -1; }
            dst[col*sd+row] = sums[col];
        }
    }
    free(dy);
    return 0;
}
