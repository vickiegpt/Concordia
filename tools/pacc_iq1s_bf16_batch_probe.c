#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int (*submit_gemm_fn)(
    int, int, int, int, int, const void *,
    const void *, int, int, long long,
    const void *, int, int, long long,
    const void *, void *, int, int, long long, int, int);

static uint32_t rng_state = 1;

static uint32_t next_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static int write_full_at(int fd, const void * buf, size_t len, off_t off) {
    const uint8_t * src = (const uint8_t *) buf;
    size_t done = 0;
    while (done < len) {
        const ssize_t n = pwrite(fd, src + done, len - done, off + (off_t) done);
        if (n <= 0) {
            return -1;
        }
        done += (size_t) n;
    }
    return 0;
}

static int read_full_at(int fd, void * buf, size_t len, off_t off) {
    uint8_t * dst = (uint8_t *) buf;
    size_t done = 0;
    while (done < len) {
        const ssize_t n = pread(fd, dst + done, len - done, off + (off_t) done);
        if (n <= 0) {
            return -1;
        }
        done += (size_t) n;
    }
    return 0;
}

int main(void) {
    enum {
        M = 32,
        N = 1,
        K = 6144,
        BATCHES = 4,
        ATYPE_BF16 = 5,
        TYPE_F32 = 4,
    };
    const uint64_t shared_base = UINT64_C(0x20110600000);
    const uint64_t user_off = UINT64_C(0x100000);
    const uint64_t data_off = UINT64_C(0xd0000000);
    const size_t quant_row_bytes =
        (size_t) (K / QK_K) * sizeof(block_iq1_s);
    const size_t row_bytes = (size_t) K * sizeof(ggml_bf16_t);
    const size_t a_bytes = (size_t) M * BATCHES * row_bytes;
    const size_t b_bytes = (size_t) K * N * sizeof(ggml_bf16_t);
    const size_t c_bytes = (size_t) M * N * BATCHES * sizeof(float);
    const uint64_t a_off = data_off;
    const uint64_t b_off = (a_off + a_bytes + 63) & ~UINT64_C(63);
    const uint64_t c_off = (b_off + b_bytes + 63) & ~UINT64_C(63);
    const size_t quant_bytes = (size_t) M * BATCHES * quant_row_bytes;
    block_iq1_s * quant = calloc(1, quant_bytes);
    ggml_bf16_t * a = malloc(a_bytes);
    ggml_bf16_t * b = malloc(b_bytes);
    float * c = calloc(1, c_bytes);
    float * row = malloc((size_t) K * sizeof(float));
    float * reference = malloc((size_t) M * BATCHES * sizeof(float));
    if (!quant || !a || !b || !c || !row || !reference) {
        return 2;
    }

    const size_t blocks = quant_bytes / sizeof(block_iq1_s);
    for (size_t i = 0; i < blocks; ++i) {
        quant[i].d = ggml_fp32_to_fp16(0.0005f * (float) (1 + i % 7));
        for (size_t j = 0; j < sizeof(quant[i].qs); ++j) {
            quant[i].qs[j] = (uint8_t) next_u32();
        }
        for (size_t j = 0; j < sizeof(quant[i].qh) / sizeof(quant[i].qh[0]); ++j) {
            quant[i].qh[j] = (uint16_t) next_u32();
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int col = 0; col < N; ++col) {
            b[(size_t) k * N + col] =
                ggml_fp32_to_bf16(0.01f * (float) (((k + col) % 23) - 11));
        }
    }
    for (int r = 0; r < M * BATCHES; ++r) {
        dequantize_row_iq1_s(
            (const block_iq1_s *)
                ((const uint8_t *) quant + (size_t) r * quant_row_bytes),
            row, K);
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            a[(size_t) r * K + k] = ggml_fp32_to_bf16(row[k]);
            sum += (double) ggml_bf16_to_fp32(a[(size_t) r * K + k]) *
                ggml_bf16_to_fp32(b[(size_t) k * N]);
        }
        reference[r] = (float) sum;
    }

    submit_gemm_fn submit =
        (submit_gemm_fn) dlsym(RTLD_DEFAULT, "hetgpu_pacc_submit_gemm");
    if (!submit) {
        fprintf(stderr, "missing hetgpu_pacc_submit_gemm: %s\n", dlerror());
        return 3;
    }
    const int fd = open("/dev/hetgpu_pacc_mbox_ddr_coh0", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open shared DDR");
        return 4;
    }
    if (write_full_at(fd, a, a_bytes, (off_t) (user_off + a_off)) ||
        write_full_at(fd, b, b_bytes, (off_t) (user_off + b_off)) ||
        write_full_at(fd, c, c_bytes, (off_t) (user_off + c_off))) {
        perror("stage shared DDR");
        return 5;
    }

    const uintptr_t source_id = (uintptr_t) a;
    const uint32_t cache_tag =
        (uint32_t) (((source_id >> 4) ^ (source_id >> 32)) & 0x007fffffU);
    const int compute_type = (int) (TYPE_F32 | (cache_tag << 8));
    const int rc = submit(
        0, 0, M, N, K, NULL,
        (const void *) (uintptr_t) (shared_base + a_off),
        ATYPE_BF16, K, (long long) M * K,
        (const void *) (uintptr_t) (shared_base + b_off),
        ATYPE_BF16, N, -1,
        NULL,
        (void *) (uintptr_t) (shared_base + c_off),
        TYPE_F32, N, (long long) M * N,
        BATCHES, compute_type);
    if (rc != 0) {
        fprintf(stderr, "submit failed rc=%d\n", rc);
        return 6;
    }
    if (read_full_at(fd, c, c_bytes, (off_t) (user_off + c_off))) {
        perror("read shared DDR");
        return 7;
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < M * BATCHES; ++i) {
        const int batch = i / M;
        const int row_index = i % M;
        const int c_index = batch * M * N + row_index * N;
        const float abs_error = fabsf(c[c_index] - reference[i]);
        const float rel_error = abs_error / fmaxf(fabsf(reference[i]), 1.0e-6f);
        max_abs = fmaxf(max_abs, abs_error);
        max_rel = fmaxf(max_rel, rel_error);
        if (!isfinite(c[c_index]) ||
            abs_error > 2.0e-3f + 2.0e-2f * fabsf(reference[i])) {
            if (mismatches < 12) {
                fprintf(stderr,
                        "mismatch row=%d got=%g expected=%g abs=%g rel=%g\n",
                        i, c[c_index], reference[i], abs_error, rel_error);
            }
            ++mismatches;
        }
    }
    printf("pacc_iq1s_bf16_batch shape=%dx%dx%d batches=%d rc=%d "
           "mismatches=%d max_abs=%g max_rel=%g c0=%g ref0=%g\n",
           M, N, K, BATCHES, rc, mismatches, max_abs, max_rel, c[0], reference[0]);

    close(fd);
    free(reference);
    free(row);
    free(c);
    free(b);
    free(a);
    free(quant);
    return mismatches == 0 ? 0 : 20;
}
