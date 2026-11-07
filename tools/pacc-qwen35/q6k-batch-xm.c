#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "q6k-xm-profile.h"
#ifdef HETGPU_PACC_Q6K_VECTOR_PACK
#include <riscv_vector.h>
#endif

_Thread_local struct Q6kXmProfile q6k_xm_profile;
static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec*1000000000u+t.tv_nsec;
}

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

/* Q8 integers are exact BF16 and a 32-term dot is exact FP32. Keep block
 * scales outside XM instead of rounding dequantized operands to BF16. */
int hetgpu_q8_0_q8_1_xm(const uint8_t *x, const uint8_t *y, float *dst,
                       size_t m, size_t n, size_t nb, size_t sx, size_t sy, size_t sd) {
    memset(&q6k_xm_profile, 0, sizeof(q6k_xm_profile));
    q6k_xm_profile.workers = 1;
    if (!x || !y || !dst || !m || !n || n > 8 || !nb || nb > 1024 ||
        sx < nb*34 || sy < nb*36 || sd < m) return -1;
    uint64_t begin = now_ns();
    uint16_t *yb = malloc(nb*32*n*sizeof(*yb));
    float *dy = malloc(nb*n*sizeof(*dy));
    if (!yb || !dy) { free(yb); free(dy); return -1; }
    for (size_t b = 0; b < nb; ++b)
        for (size_t col = 0; col < n; ++col) {
            const uint8_t *src = y+col*sy+b*36;
            uint16_t h; memcpy(&h, src, 2);
            dy[b*n+col] = half_float(h);
            if (!isfinite(dy[b*n+col])) { free(yb); free(dy); return -1; }
            for (size_t lane = 0; lane < 32; ++lane)
                yb[(b*32+lane)*n+col] = integer_bf16((int8_t)src[4+lane]);
        }
    q6k_xm_profile.activation += now_ns()-begin;
    for (size_t row0 = 0; row0 < m; row0 += 32) {
        size_t rows = m-row0 < 32 ? m-row0 : 32;
        float sums[32*8] = {0}, dots[32*8], dx[32];
        uint16_t xb[32*32];
        for (size_t b = 0; b < nb; ++b) {
            begin = now_ns();
            for (size_t row = 0; row < rows; ++row) {
                const uint8_t *src = x+(row0+row)*sx+b*34;
                uint16_t h; memcpy(&h, src, 2);
                dx[row] = half_float(h);
                if (!isfinite(dx[row])) { free(yb); free(dy); return -1; }
                for (size_t lane = 0; lane < 32; ++lane)
                    xb[lane*rows+row] = integer_bf16((int8_t)src[2+lane]);
            }
            q6k_xm_profile.weights += now_ns()-begin;
            begin = now_ns();
            if (xsfmm_native_bf16(xb, yb+b*32*n, dots, rows, n, 32)) {
                free(yb); free(dy); return -1;
            }
            ++q6k_xm_profile.calls;
            q6k_xm_profile.matrix += now_ns()-begin;
            begin = now_ns();
            for (size_t row = 0; row < rows; ++row)
                for (size_t col = 0; col < n; ++col) {
                    float dot = dots[row*n+col];
                    if (!isfinite(dot) || fabsf(dot) > 524288 || dot != (float)(int32_t)dot) {
                        free(yb); free(dy); return -1;
                    }
                    sums[row*n+col] += (dx[row]*dy[b*n+col])*dot;
                }
            q6k_xm_profile.reduce += now_ns()-begin;
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

/* Integers [-128,127] are exact BF16. Apply Q6 scales after two K=16
 * XM dots so no rounded BF16 weight*scale product enters the computation. */
int hetgpu_q6k_q8_1_xm_serial(const uint8_t *restrict x, const uint8_t *restrict y,
                           float *restrict dst, size_t m, size_t n, size_t nb,
                           size_t sx, size_t sy, size_t sd) {
    memset(&q6k_xm_profile, 0, sizeof(q6k_xm_profile));
    q6k_xm_profile.workers = 1;
    uint64_t begin = now_ns();
#ifdef HETGPU_PACC_Q6K_VECTOR_PACK
    uint16_t lookup[64];
    for (unsigned i = 0; i < 64; ++i) lookup[i] = integer_bf16((int)i-32);
#endif
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
    q6k_xm_profile.activation += now_ns()-begin;
    for (size_t row0 = 0; row0 < m; row0 += 32) {
        const size_t rows = m-row0 < 32 ? m-row0 : 32;
        float sums[32*8] = {0}, dx[32], dots[2][32*8];
        int8_t scales[16][32];
        uint16_t xb[256*32];
        for (size_t block = 0; block < nb; ++block) {
            begin = now_ns();
            for (size_t row = 0; row < rows; ++row) {
                const uint8_t *src = x+(row0+row)*sx+block*210;
                uint16_t h;
                memcpy(&h, src+208, 2);
                dx[row] = half_float(h);
                for (unsigned s = 0; s < 16; ++s) scales[s][row] = (int8_t)src[192+s];
                for (unsigned half = 0; half < 2; ++half)
                    for (unsigned group = 0; group < 4; ++group) {
#ifdef HETGPU_PACC_Q6K_VECTOR_PACK
                        for (unsigned lane = 0; lane < 32;) {
                            size_t vl = __riscv_vsetvl_e8m1(32-lane);
                            vuint8m1_t lo = __riscv_vle8_v_u8m1(src+half*64+(group%2)*32+lane, vl);
                            vuint8m1_t hi = __riscv_vle8_v_u8m1(src+128+half*32+lane, vl);
                            lo = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(lo, (group/2)*4, vl), 15, vl);
                            hi = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(hi, group*2, vl), 3, vl);
                            vuint8m1_t q = __riscv_vor_vv_u8m1(lo, __riscv_vsll_vx_u8m1(hi, 4, vl), vl);
#ifdef HETGPU_PACC_Q6K_VECTOR_FLOAT_PACK
                            vuint16m2_t centered = __riscv_vsub_vx_u16m2(__riscv_vzext_vf2_u16m2(q, vl), 32, vl);
                            vint16m2_t signed_q = __riscv_vreinterpret_v_u16m2_i16m2(centered);
                            vfloat32m4_t exact = __riscv_vfwcvt_f_x_v_f32m4(signed_q, vl);
                            vuint16m2_t packed = __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_f32m4_u32m4(exact), 16, vl);
#else
                            vuint16m2_t index = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(q, vl), 1, vl);
                            vuint16m2_t packed = __riscv_vluxei16_v_u16m2(lookup, index, vl);
#endif
                            __riscv_vsse16_v_u16m2(xb+((half*4+group)*32+lane)*rows+row,
                                                   rows*sizeof(uint16_t), packed, vl);
                            lane += vl;
                        }
#else
                        for (unsigned lane = 0; lane < 32; ++lane) {
                            unsigned ql = src[half*64+(group%2)*32+lane];
                            unsigned qh = src[128+half*32+lane];
                            int q = (int)(((ql >> ((group/2)*4)) & 15) |
                                          (((qh >> (group*2)) & 3) << 4))-32;
                            xb[((half*4+group)*32+lane)*rows+row] = integer_bf16(q);
                        }
#endif
                    }
            }
            q6k_xm_profile.weights += now_ns()-begin;
            for (unsigned group = 0; group < 8; ++group) {
                begin = now_ns();
                for (unsigned half = 0; half < 2; ++half) {
                    if (xsfmm_native_bf16(xb+(group*32+half*16)*rows,
                        yb+((block*8+group)*32+half*16)*n, dots[half], rows, n, 16)) {
                        free(yb); free(dy); return -1;
                    }
                    ++q6k_xm_profile.calls;
                }
                q6k_xm_profile.matrix += now_ns()-begin;
                begin = now_ns();
#ifdef HETGPU_PACC_Q6K_VECTOR_REDUCE
                for (size_t col = 0; col < n; ++col) {
                    for (size_t row = 0; row < rows;) {
                        size_t vl = __riscv_vsetvl_e32m1(rows-row);
                        size_t at = row*n+col;
                        ptrdiff_t stride = n*sizeof(float);
                        vfloat32m1_t f0 = __riscv_vlse32_v_f32m1(dots[0]+at, stride, vl);
                        vfloat32m1_t f1 = __riscv_vlse32_v_f32m1(dots[1]+at, stride, vl);
                        vbool32_t valid = __riscv_vmand_mm_b32(
                            __riscv_vmfle_vf_f32m1_b32(__riscv_vfabs_v_f32m1(f0, vl), 65536, vl),
                            __riscv_vmfle_vf_f32m1_b32(__riscv_vfabs_v_f32m1(f1, vl), 65536, vl), vl);
                        if (__riscv_vcpop_m_b32(valid, vl) != vl) {
                            free(yb); free(dy); return -1;
                        }
                        vint32m1_t i0 = __riscv_vfcvt_rtz_x_f_v_i32m1(f0, vl);
                        vint32m1_t i1 = __riscv_vfcvt_rtz_x_f_v_i32m1(f1, vl);
                        valid = __riscv_vmand_mm_b32(
                            __riscv_vmfeq_vv_f32m1_b32(f0, __riscv_vfcvt_f_x_v_f32m1(i0, vl), vl),
                            __riscv_vmfeq_vv_f32m1_b32(f1, __riscv_vfcvt_f_x_v_f32m1(i1, vl), vl), vl);
                        if (__riscv_vcpop_m_b32(valid, vl) != vl) {
                            free(yb); free(dy); return -1;
                        }
                        vint32m1_t s0 = __riscv_vsext_vf4_i32m1(
                            __riscv_vle8_v_i8mf4(scales[group*2]+row, vl), vl);
                        vint32m1_t s1 = __riscv_vsext_vf4_i32m1(
                            __riscv_vle8_v_i8mf4(scales[group*2+1]+row, vl), vl);
                        vint32m1_t dot = __riscv_vadd_vv_i32m1(
                            __riscv_vmul_vv_i32m1(i0, s0, vl),
                            __riscv_vmul_vv_i32m1(i1, s1, vl), vl);
                        vfloat32m1_t scale = __riscv_vfmul_vf_f32m1(
                            __riscv_vle32_v_f32m1(dx+row, vl), dy[(block*8+group)*n+col], vl);
                        vfloat32m1_t value = __riscv_vfmul_vv_f32m1(
                            scale, __riscv_vfcvt_f_x_v_f32m1(dot, vl), vl);
                        value = __riscv_vfadd_vv_f32m1(
                            __riscv_vlse32_v_f32m1(sums+at, stride, vl), value, vl);
                        __riscv_vsse32_v_f32m1(sums+at, stride, value, vl);
                        row += vl;
                    }
                }
#else
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
#endif
                q6k_xm_profile.reduce += now_ns()-begin;
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
