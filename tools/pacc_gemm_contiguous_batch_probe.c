#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int (*submit_gemm_fn)(
    int, int, int, int, int, const void *,
    const void *, int, int, long long,
    const void *, int, int, long long,
    const void *, void *, int, int, long long, int, int);

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1.0e3 + (double) ts.tv_nsec * 1.0e-6;
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *) lhs;
    const double b = *(const double *) rhs;
    return (a > b) - (a < b);
}

static int write_full_at(int fd, const void *buf, size_t len, off_t off) {
    const uint8_t *src = (const uint8_t *) buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = pwrite(fd, src + done, len - done, off + (off_t) done);
        if (n <= 0) return -1;
        done += (size_t) n;
    }
    return 0;
}

static int read_full_at(int fd, void *buf, size_t len, off_t off) {
    uint8_t *dst = (uint8_t *) buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, dst + done, len - done, off + (off_t) done);
        if (n <= 0) return -1;
        done += (size_t) n;
    }
    return 0;
}

int main(void) {
    enum {
        M = 32,
        N = 32,
        BATCHES = 5,
        ATYPE_BF16 = 5,
        CTYPE_F32 = 4,
        COMPUTE_F32_STATIC_A = 4 | (1 << 8),
    };
    const uint64_t shared_base = UINT64_C(0x20110600000);
    const uint64_t user_off = UINT64_C(0x100000);
    const uint64_t data_off = UINT64_C(0xe0000000);
    const int iterations = getenv("PACC_BATCH_ITERS") ?
        atoi(getenv("PACC_BATCH_ITERS")) : 64;
    const int k = getenv("PACC_BATCH_K") ?
        atoi(getenv("PACC_BATCH_K")) : 2048;
    const size_t a_elems = (size_t) M * k * BATCHES;
    const size_t b_elems = (size_t) k * N;
    const size_t c_elems = (size_t) M * N * BATCHES;
    const size_t a_bytes = a_elems * sizeof(uint16_t);
    const size_t b_bytes = b_elems * sizeof(uint16_t);
    const size_t c_bytes = c_elems * sizeof(float);
    const uint64_t a_off = data_off;
    const uint64_t b_off = (a_off + a_bytes + 63) & ~UINT64_C(63);
    const uint64_t c_off = (b_off + b_bytes + 63) & ~UINT64_C(63);
    uint16_t *a = NULL;
    uint16_t *b = NULL;
    float *c = NULL;
    double *samples = NULL;
    int mismatches = 0;

    if (iterations <= 0 || k <= 0 || (k & 1) != 0) return 2;
    submit_gemm_fn submit = (submit_gemm_fn)
        dlsym(RTLD_DEFAULT, "hetgpu_pacc_submit_gemm");
    if (!submit) {
        fprintf(stderr, "missing hetgpu_pacc_submit_gemm: %s\n", dlerror());
        return 3;
    }
    int fd = open("/dev/hetgpu_pacc_mbox_ddr_coh0", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open shared DDR");
        return 4;
    }
    a = (uint16_t *) malloc(a_bytes);
    b = (uint16_t *) malloc(b_bytes);
    c = (float *) malloc(c_bytes);
    samples = (double *) malloc((size_t) iterations * sizeof(double));
    if (!a || !b || !c || !samples) return 5;
    for (size_t i = 0; i < a_elems; ++i) a[i] = UINT16_C(0x3f80);
    for (size_t i = 0; i < b_elems; ++i) b[i] = UINT16_C(0x3f80);
    memset(c, 0, c_bytes);
    if (write_full_at(fd, a, a_bytes, (off_t) (user_off + a_off)) ||
        write_full_at(fd, b, b_bytes, (off_t) (user_off + b_off)) ||
        write_full_at(fd, c, c_bytes, (off_t) (user_off + c_off))) {
        perror("stage shared DDR");
        return 6;
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const double start_ms = monotonic_ms();
        int rc = submit(
            0, 0, M, N, k, NULL,
            (const void *) (uintptr_t) (shared_base + a_off),
            ATYPE_BF16, k, (long long) M * k,
            (const void *) (uintptr_t) (shared_base + b_off),
            ATYPE_BF16, N, -1,
            NULL,
            (void *) (uintptr_t) (shared_base + c_off),
            CTYPE_F32, N, (long long) M * N,
            BATCHES, COMPUTE_F32_STATIC_A);
        samples[iteration] = monotonic_ms() - start_ms;
        if (rc != 0) {
            fprintf(stderr, "batch submit failed iteration=%d rc=%d\n",
                    iteration, rc);
            return 7;
        }
    }
    if (read_full_at(fd, c, c_bytes, (off_t) (user_off + c_off))) {
        perror("read shared DDR");
        return 8;
    }
    for (size_t i = 0; i < c_elems; ++i) {
        if (fabsf(c[i] - (float) k) > 1.0f) {
            if (mismatches < 8) {
                fprintf(stderr, "mismatch index=%zu got=%g expected=%d\n",
                        i, c[i], k);
            }
            ++mismatches;
        }
    }
    qsort(samples, (size_t) iterations, sizeof(double), compare_double);
    double sum = 0.0;
    for (int i = 0; i < iterations; ++i) sum += samples[i];
    const int p50_index = (iterations - 1) * 50 / 100;
    const int p99_index = (iterations - 1) * 99 / 100;
    printf("pacc_contiguous_batch iterations=%d batches=%d shape=%dx%dx%d "
           "avg_ms=%.3f p50_ms=%.3f p99_ms=%.3f min_ms=%.3f max_ms=%.3f "
           "mismatches=%d c0=%g clast=%g\n",
           iterations, BATCHES, M, N, k,
           sum / iterations, samples[p50_index], samples[p99_index],
           samples[0], samples[iterations - 1],
           mismatches, c[0], c[c_elems - 1]);

    close(fd);
    free(a);
    free(b);
    free(c);
    free(samples);
    return mismatches == 0 ? 0 : 20;
}
