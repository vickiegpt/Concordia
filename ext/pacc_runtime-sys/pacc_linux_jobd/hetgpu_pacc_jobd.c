#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<riscv_vector.h>)
#include <riscv_vector.h>
#endif
#endif

#define HETGPU_PACC_JOB_MAGIC 0x4847505550414343ULL
#define HETGPU_PACC_JOB_VERSION 1U

#define HETGPU_PACC_JOB_GEMM 1U
#define HETGPU_PACC_JOB_SOFTMAX 2U
#define HETGPU_PACC_JOB_RMSNORM 3U
#define HETGPU_PACC_JOB_ALLREDUCE 4U

#define HETGPU_PACC_ARG_SLOT_BYTES 0x400UL
#define HETGPU_PACC_CONTROL_BYTES 0x2000UL
#define HETGPU_PACC_ARG_BASE_OFF 0x100UL
#define HETGPU_PACC_RUNTIME_TABLE_OFF 0x1400UL
#define HETGPU_PACC_RUNTIME_TABLE_MAGIC 0x4847505554424c31ULL
#define HETGPU_PACC_RUNTIME_TABLE_VERSION 1U
#define AP2PACC_MBOX_PHYS 0x20000000ULL
#define PACC2AP_MBOX_PHYS 0x20002000ULL
#define PACC_DTYPE_INT8 0U
#define PACC_DTYPE_UINT8 1U
#define PACC_DTYPE_INT32 2U
#define PACC_DTYPE_F32 4U
#define PACC_DTYPE_BF16 5U
#define PACC_GEMM_THREADS 4U

#if defined(__riscv_vector)
#define PACC_RVV_UNUSED __attribute__((unused))
#else
#define PACC_RVV_UNUSED
#endif

struct Doorbell {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint32_t flags;
    uint32_t status;
    uint64_t seq;
};

struct HostStatus {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint32_t status;
    uint64_t seq;
};

struct ArgSlotHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint64_t seq;
    uint64_t arg_len;
};

struct GemmJob {
    uint32_t transa;
    uint32_t transb;
    uint32_t atype;
    uint32_t btype;
    uint32_t ctype;
    uint32_t compute_type;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
    uint64_t alpha_addr;
    uint64_t beta_addr;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_c;
    uint64_t batch_count;
};

struct SoftmaxJob {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t rows;
    uint64_t cols;
    uint64_t stride;
    uint32_t dtype;
    uint32_t reserved;
};

struct RmsNormJob {
    uint64_t x_addr;
    uint64_t weight_addr;
    uint64_t y_addr;
    uint64_t rows;
    uint64_t hidden;
    float eps;
    uint32_t dtype;
};

struct AllReduceJob {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t count;
    uint32_t nranks;
    uint32_t reduce_op;
    uint32_t dtype;
    uint32_t reserved;
};

struct PreloadedJobs {
    bool have_gemm;
    bool have_softmax;
    bool have_rmsnorm;
    bool have_allreduce;
    struct GemmJob gemm;
    struct SoftmaxJob softmax;
    struct RmsNormJob rmsnorm;
    struct AllReduceJob allreduce;
};

struct RuntimeJobTable {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t seq;
    uint32_t have_gemm;
    uint32_t have_softmax;
    uint32_t have_rmsnorm;
    uint32_t have_allreduce;
    struct GemmJob gemm;
    struct SoftmaxJob softmax;
    struct RmsNormJob rmsnorm;
    struct AllReduceJob allreduce;
};

struct Map {
    void *base;
    size_t map_len;
    void *ptr;
};

struct ScanRange {
    uint64_t start;
    uint64_t end;
};

static long g_page_size = 4096;

static void log_msg(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "hetgpu_pacc_jobd: %s\n", buf);
    int kmsg = open("/dev/kmsg", O_WRONLY | O_CLOEXEC);
    if (kmsg >= 0) {
        dprintf(kmsg, "hetgpu_pacc_jobd: %s\n", buf);
        close(kmsg);
    }
}

static uint64_t parse_u64(const char *s) {
    return strtoull(s, NULL, 0);
}

static int map_phys(int fd, uint64_t phys, size_t len, struct Map *out) {
    uint64_t page = (uint64_t)g_page_size;
    uint64_t base = phys & ~(page - 1);
    size_t off = (size_t)(phys - base);
    size_t map_len = ((off + len + page - 1) / page) * page;
    void *p = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    if (p == MAP_FAILED) {
        return -1;
    }
    out->base = p;
    out->map_len = map_len;
    out->ptr = (char *)p + off;
    return 0;
}

static void unmap_phys(struct Map *m) {
    if (m->base && m->base != MAP_FAILED) {
        munmap(m->base, m->map_len);
    }
    memset(m, 0, sizeof(*m));
}

static int load_jobs_config(const char *path, struct PreloadedJobs *jobs) {
    FILE *f = fopen(path, "r");
    if (!f) {
        log_msg("no preloaded job config at %s: %s", path, strerror(errno));
        return -1;
    }

    char line[1024];
    unsigned lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == 0) continue;

        char op[32] = {0};
        char *tok[24] = {0};
        unsigned ntok = 0;
        char *save = NULL;
        for (char *t = strtok_r(p, " \t\r\n", &save); t && ntok < 24;
             t = strtok_r(NULL, " \t\r\n", &save)) {
            tok[ntok++] = t;
        }
        if (ntok == 0) continue;
        snprintf(op, sizeof(op), "%s", tok[0]);

        if (!strcmp(op, "gemm")) {
            if (ntok < 13) {
                log_msg("%s:%u bad gemm line", path, lineno);
                continue;
            }
            struct GemmJob *j = &jobs->gemm;
            memset(j, 0, sizeof(*j));
            j->atype = PACC_DTYPE_F32;
            j->btype = PACC_DTYPE_F32;
            j->ctype = PACC_DTYPE_F32;
            j->m = parse_u64(tok[1]);
            j->n = parse_u64(tok[2]);
            j->k = parse_u64(tok[3]);
            j->a_addr = parse_u64(tok[4]);
            j->b_addr = parse_u64(tok[5]);
            j->c_addr = parse_u64(tok[6]);
            j->lda = (int64_t)parse_u64(tok[7]);
            j->ldb = (int64_t)parse_u64(tok[8]);
            j->ldc = (int64_t)parse_u64(tok[9]);
            j->alpha_addr = parse_u64(tok[10]);
            j->beta_addr = parse_u64(tok[11]);
            j->batch_count = parse_u64(tok[12]);
            if (!j->batch_count) j->batch_count = 1;
            jobs->have_gemm = true;
        } else if (!strcmp(op, "softmax")) {
            if (ntok < 6) {
                log_msg("%s:%u bad softmax line", path, lineno);
                continue;
            }
            struct SoftmaxJob *j = &jobs->softmax;
            memset(j, 0, sizeof(*j));
            j->src_addr = parse_u64(tok[1]);
            j->dst_addr = parse_u64(tok[2]);
            j->rows = parse_u64(tok[3]);
            j->cols = parse_u64(tok[4]);
            j->stride = parse_u64(tok[5]);
            j->dtype = PACC_DTYPE_F32;
            jobs->have_softmax = true;
        } else if (!strcmp(op, "rmsnorm")) {
            if (ntok < 7) {
                log_msg("%s:%u bad rmsnorm line", path, lineno);
                continue;
            }
            struct RmsNormJob *j = &jobs->rmsnorm;
            memset(j, 0, sizeof(*j));
            j->x_addr = parse_u64(tok[1]);
            j->weight_addr = parse_u64(tok[2]);
            j->y_addr = parse_u64(tok[3]);
            j->rows = parse_u64(tok[4]);
            j->hidden = parse_u64(tok[5]);
            j->eps = strtof(tok[6], NULL);
            j->dtype = PACC_DTYPE_F32;
            jobs->have_rmsnorm = true;
        } else {
            log_msg("%s:%u unknown job op %s", path, lineno, op);
        }
    }
    fclose(f);
    return 0;
}

static int arg_slot_for_job(uint32_t job_id) {
    switch (job_id) {
    case HETGPU_PACC_JOB_GEMM: return 0;
    case HETGPU_PACC_JOB_SOFTMAX: return 1;
    case HETGPU_PACC_JOB_RMSNORM: return 2;
    case HETGPU_PACC_JOB_ALLREDUCE: return 3;
    default: return -1;
    }
}

static float expf_fast(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) x = 20.0f;
    float term = 1.0f;
    float sum = 1.0f;
    for (int i = 1; i <= 8; i++) {
        term *= x / (float)i;
        sum += term;
    }
    return sum > 0.0f ? sum : 0.0f;
}

static float rsqrtf_newton(float x) {
    if (x <= 0.0f) return 0.0f;
    float y = 1.0f;
    while (x * y * y > 4.0f) y *= 0.5f;
    while (x * y * y < 0.25f) y *= 2.0f;
    for (int i = 0; i < 6; i++) {
        y = y * (1.5f - 0.5f * x * y * y);
    }
    return y;
}

struct GemmWorker {
    const struct GemmJob *job;
    const void *a;
    const void *b;
    void *c;
    uint64_t row_begin;
    uint64_t row_end;
    float alpha;
    float beta;
};

static size_t gemm_span(uint64_t rows, uint64_t cols, int64_t ld) {
    if (!rows || !cols) return 0;
    uint64_t lead = ld > 0 ? (uint64_t)ld : cols;
    return (size_t)((rows - 1) * lead + cols);
}

static size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case PACC_DTYPE_INT8:
        return sizeof(int8_t);
    case PACC_DTYPE_UINT8:
        return sizeof(uint8_t);
    case PACC_DTYPE_INT32:
        return sizeof(int32_t);
    case PACC_DTYPE_F32:
        return sizeof(float);
    case PACC_DTYPE_BF16:
        return sizeof(uint16_t);
    default:
        return 0;
    }
}

static float bf16_to_f32(uint16_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)x << 16;
    return v.f;
}

static uint16_t f32_to_bf16(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    uint32_t lsb = (v.u >> 16) & 1U;
    uint32_t rounding_bias = 0x7fffU + lsb;
    return (uint16_t)((v.u + rounding_bias) >> 16);
}

static int32_t round_to_i32(float x) {
    if (x >= 2147483647.0f) return 2147483647;
    if (x <= -2147483648.0f) return (-2147483647 - 1);
    return x >= 0.0f ? (int32_t)(x + 0.5f) : (int32_t)(x - 0.5f);
}

static int8_t round_to_i8(float x) {
    int32_t v = round_to_i32(x);
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

static uint8_t round_to_u8(float x) {
    int32_t v = round_to_i32(x);
    if (v > 255) v = 255;
    if (v < 0) v = 0;
    return (uint8_t)v;
}

static float load_typed(const void *base, size_t idx, uint32_t dtype) {
    if (dtype == PACC_DTYPE_INT8) {
        return (float)((const int8_t *)base)[idx];
    }
    if (dtype == PACC_DTYPE_UINT8) {
        return (float)((const uint8_t *)base)[idx];
    }
    if (dtype == PACC_DTYPE_INT32) {
        return (float)((const int32_t *)base)[idx];
    }
    if (dtype == PACC_DTYPE_F32) {
        return ((const float *)base)[idx];
    }
    if (dtype == PACC_DTYPE_BF16) {
        return bf16_to_f32(((const uint16_t *)base)[idx]);
    }
    return 0.0f;
}

static void store_typed(void *base, size_t idx, uint32_t dtype, float value) {
    if (dtype == PACC_DTYPE_INT8) {
        ((int8_t *)base)[idx] = round_to_i8(value);
    } else if (dtype == PACC_DTYPE_UINT8) {
        ((uint8_t *)base)[idx] = round_to_u8(value);
    } else if (dtype == PACC_DTYPE_INT32) {
        ((int32_t *)base)[idx] = round_to_i32(value);
    } else if (dtype == PACC_DTYPE_F32) {
        ((float *)base)[idx] = value;
    } else if (dtype == PACC_DTYPE_BF16) {
        ((uint16_t *)base)[idx] = f32_to_bf16(value);
    }
}

static float PACC_RVV_UNUSED gemm_dot_f32_scalar(const float *a, ptrdiff_t a_stride,
                                                 const float *b, ptrdiff_t b_stride,
                                                 uint64_t k) {
    float acc = 0.0f;
    for (uint64_t kk = 0; kk < k; kk++) {
        acc += a[kk * a_stride] * b[kk * b_stride];
    }
    return acc;
}

static float gemm_dot_f32(const float *a, ptrdiff_t a_stride,
                          const float *b, ptrdiff_t b_stride,
                          uint64_t k) {
#if defined(__riscv_vector)
    float acc = 0.0f;
    for (uint64_t kk = 0; kk < k;) {
        size_t vl = __riscv_vsetvl_e32m1(k - kk);
        vfloat32m1_t va = a_stride == 1
            ? __riscv_vle32_v_f32m1(a + kk, vl)
            : __riscv_vlse32_v_f32m1(a + kk * a_stride, a_stride * (ptrdiff_t)sizeof(float), vl);
        vfloat32m1_t vb = b_stride == 1
            ? __riscv_vle32_v_f32m1(b + kk, vl)
            : __riscv_vlse32_v_f32m1(b + kk * b_stride, b_stride * (ptrdiff_t)sizeof(float), vl);
        vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(prod, zero, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(sum);
        kk += vl;
    }
    return acc;
#else
    return gemm_dot_f32_scalar(a, a_stride, b, b_stride, k);
#endif
}

static float gemm_dot_typed(const void *a, uint32_t atype, ptrdiff_t a_stride,
                            const void *b, uint32_t btype, ptrdiff_t b_stride,
                            uint64_t k) {
    if (atype == PACC_DTYPE_F32 && btype == PACC_DTYPE_F32) {
        return gemm_dot_f32((const float *)a, a_stride, (const float *)b, b_stride, k);
    }

    float acc = 0.0f;
    for (uint64_t kk = 0; kk < k; kk++) {
        acc += load_typed(a, (size_t)(kk * a_stride), atype) *
               load_typed(b, (size_t)(kk * b_stride), btype);
    }
    return acc;
}

static void *gemm_worker_main(void *arg) {
    struct GemmWorker *w = (struct GemmWorker *)arg;
    const struct GemmJob *job = w->job;
    for (uint64_t row = w->row_begin; row < w->row_end; row++) {
        for (uint64_t col = 0; col < job->n; col++) {
            size_t a_base = job->transa ? (size_t)row : (size_t)(row * job->lda);
            ptrdiff_t a_stride = job->transa ? (ptrdiff_t)job->lda : 1;
            size_t b_base = job->transb ? (size_t)(col * job->ldb) : (size_t)col;
            ptrdiff_t b_stride = job->transb ? 1 : (ptrdiff_t)job->ldb;
            const void *ap = (const char *)w->a + a_base * dtype_size(job->atype);
            const void *bp = (const char *)w->b + b_base * dtype_size(job->btype);
            size_t c_idx = (size_t)(row * job->ldc + col);
            float acc = gemm_dot_typed(ap, job->atype, a_stride, bp, job->btype, b_stride, job->k);
            float old = w->beta != 0.0f ? load_typed(w->c, c_idx, job->ctype) : 0.0f;
            store_typed(w->c, c_idx, job->ctype, w->alpha * acc + w->beta * old);
        }
    }
    return NULL;
}

static int run_gemm_matrix_threads(const struct GemmJob *job, const void *a,
                                   const void *b, void *c, float alpha, float beta) {
    pthread_t threads[PACC_GEMM_THREADS];
    struct GemmWorker workers[PACC_GEMM_THREADS];
    unsigned nthreads = PACC_GEMM_THREADS;
    int started = 0;

    for (unsigned tid = 0; tid < nthreads; tid++) {
        uint64_t row_begin = (job->m * tid) / nthreads;
        uint64_t row_end = (job->m * (tid + 1)) / nthreads;
        workers[tid] = (struct GemmWorker){
            .job = job,
            .a = a,
            .b = b,
            .c = c,
            .row_begin = row_begin,
            .row_end = row_end,
            .alpha = alpha,
            .beta = beta,
        };
        if (pthread_create(&threads[tid], NULL, gemm_worker_main, &workers[tid]) != 0) {
            for (int i = 0; i < started; i++) pthread_join(threads[i], NULL);
            return -1;
        }
        started++;
    }
    for (int i = 0; i < started; i++) pthread_join(threads[i], NULL);
    return 0;
}

static int run_gemm(int fd, const struct GemmJob *job) {
    if (!job->m || !job->n || !job->k || !job->a_addr || !job->b_addr || !job->c_addr) {
        return 0xffff1001;
    }
    size_t a_dtype_size = dtype_size(job->atype);
    size_t b_dtype_size = dtype_size(job->btype);
    size_t c_dtype_size = dtype_size(job->ctype);
    if (!a_dtype_size || !b_dtype_size || !c_dtype_size) {
        return 0xffff1002;
    }

    struct GemmJob norm = *job;
    if (norm.lda <= 0) norm.lda = norm.transa ? (int64_t)norm.m : (int64_t)norm.k;
    if (norm.ldb <= 0) norm.ldb = norm.transb ? (int64_t)norm.k : (int64_t)norm.n;
    if (norm.ldc <= 0) norm.ldc = (int64_t)norm.n;
    job = &norm;

    uint64_t batch_count = job->batch_count ? job->batch_count : 1;
    size_t a_matrix_elems = job->transa
        ? gemm_span(job->k, job->m, job->lda)
        : gemm_span(job->m, job->k, job->lda);
    size_t b_matrix_elems = job->transb
        ? gemm_span(job->n, job->k, job->ldb)
        : gemm_span(job->k, job->n, job->ldb);
    size_t c_matrix_elems = gemm_span(job->m, job->n, job->ldc);
    uint64_t a_batch_stride = job->stride_a > 0 ? (uint64_t)job->stride_a : (uint64_t)a_matrix_elems;
    uint64_t b_batch_stride = job->stride_b > 0 ? (uint64_t)job->stride_b : (uint64_t)b_matrix_elems;
    uint64_t c_batch_stride = job->stride_c > 0 ? (uint64_t)job->stride_c : (uint64_t)c_matrix_elems;
    size_t a_elems = (size_t)(a_batch_stride * (batch_count - 1) + a_matrix_elems);
    size_t b_elems = (size_t)(b_batch_stride * (batch_count - 1) + b_matrix_elems);
    size_t c_elems = (size_t)(c_batch_stride * (batch_count - 1) + c_matrix_elems);
    struct Map ma = {0}, mb = {0}, mc = {0}, malpha = {0}, mbeta = {0};
    if (map_phys(fd, job->a_addr, a_elems * a_dtype_size, &ma) ||
        map_phys(fd, job->b_addr, b_elems * b_dtype_size, &mb) ||
        map_phys(fd, job->c_addr, c_elems * c_dtype_size, &mc)) {
        unmap_phys(&ma); unmap_phys(&mb); unmap_phys(&mc);
        return 0xffff1003;
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    if (job->alpha_addr && !map_phys(fd, job->alpha_addr, sizeof(float), &malpha)) {
        alpha = *(float *)malpha.ptr;
    }
    if (job->beta_addr && !map_phys(fd, job->beta_addr, sizeof(float), &mbeta)) {
        beta = *(float *)mbeta.ptr;
    }

    const char *a0 = (const char *)ma.ptr;
    const char *b0 = (const char *)mb.ptr;
    char *c0 = (char *)mc.ptr;
    for (uint64_t batch = 0; batch < batch_count; batch++) {
        const void *a = a0 + a_batch_stride * batch * a_dtype_size;
        const void *b = b0 + b_batch_stride * batch * b_dtype_size;
        void *c = c0 + c_batch_stride * batch * c_dtype_size;
        if (run_gemm_matrix_threads(job, a, b, c, alpha, beta) != 0) {
            unmap_phys(&malpha); unmap_phys(&mbeta); unmap_phys(&ma); unmap_phys(&mb); unmap_phys(&mc);
            return 0xffff1004;
        }
    }
    msync(mc.base, mc.map_len, MS_SYNC);
    unmap_phys(&malpha); unmap_phys(&mbeta); unmap_phys(&ma); unmap_phys(&mb); unmap_phys(&mc);
    return 0;
}

static int run_softmax(int fd, const struct SoftmaxJob *job) {
    if (!job->src_addr || !job->dst_addr || !job->rows || !job->cols) return 0xffff2001;
    size_t elem_size = dtype_size(job->dtype);
    if (!elem_size) return 0xffff2002;
    uint64_t stride = job->stride ? job->stride : job->cols;
    size_t elems = (size_t)(job->rows * stride);
    struct Map ms = {0}, md = {0};
    if (map_phys(fd, job->src_addr, elems * elem_size, &ms) ||
        map_phys(fd, job->dst_addr, elems * elem_size, &md)) {
        unmap_phys(&ms); unmap_phys(&md);
        return 0xffff2003;
    }
    const void *src = ms.ptr;
    void *dst = md.ptr;
    for (uint64_t row = 0; row < job->rows; row++) {
        uint64_t base = row * stride;
        float max_v = load_typed(src, base, job->dtype);
        for (uint64_t col = 1; col < job->cols; col++) {
            float v = load_typed(src, base + col, job->dtype);
            if (v > max_v) max_v = v;
        }
        float sum = 0.0f;
        for (uint64_t col = 0; col < job->cols; col++) {
            sum += expf_fast(load_typed(src, base + col, job->dtype) - max_v);
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint64_t col = 0; col < job->cols; col++) {
            float e = expf_fast(load_typed(src, base + col, job->dtype) - max_v);
            store_typed(dst, base + col, job->dtype, e * inv);
        }
    }
    msync(md.base, md.map_len, MS_SYNC);
    unmap_phys(&ms); unmap_phys(&md);
    return 0;
}

static int run_rmsnorm(int fd, const struct RmsNormJob *job) {
    if (!job->x_addr || !job->y_addr || !job->rows || !job->hidden) return 0xffff3001;
    size_t elem_size = dtype_size(job->dtype);
    if (!elem_size) return 0xffff3002;
    size_t elems = (size_t)(job->rows * job->hidden);
    struct Map mx = {0}, mw = {0}, my = {0};
    if (map_phys(fd, job->x_addr, elems * elem_size, &mx) ||
        map_phys(fd, job->y_addr, elems * elem_size, &my)) {
        unmap_phys(&mx); unmap_phys(&my);
        return 0xffff3003;
    }
    const void *x = mx.ptr;
    void *y = my.ptr;
    const void *w = NULL;
    if (job->weight_addr && !map_phys(fd, job->weight_addr, job->hidden * elem_size, &mw)) {
        w = mw.ptr;
    }
    for (uint64_t row = 0; row < job->rows; row++) {
        uint64_t base = row * job->hidden;
        float sumsq = 0.0f;
        for (uint64_t i = 0; i < job->hidden; i++) {
            float v = load_typed(x, base + i, job->dtype);
            sumsq += v * v;
        }
        float scale = rsqrtf_newton(sumsq / (float)job->hidden + job->eps);
        for (uint64_t i = 0; i < job->hidden; i++) {
            float weight = w ? load_typed(w, i, job->dtype) : 1.0f;
            store_typed(y, base + i, job->dtype, load_typed(x, base + i, job->dtype) * scale * weight);
        }
    }
    msync(my.base, my.map_len, MS_SYNC);
    unmap_phys(&mx); unmap_phys(&mw); unmap_phys(&my);
    return 0;
}

static int run_allreduce(int fd, const struct AllReduceJob *job) {
    if (!job->src_addr || !job->dst_addr || !job->count || !job->nranks) return 0xffff4001;
    if (job->dtype != PACC_DTYPE_F32 || job->reduce_op != 0) return 0xffff4002;
    size_t per_rank = (size_t)job->count;
    size_t nranks = (size_t)job->nranks;
    if (per_rank > ((size_t)-1 / sizeof(float)) ||
        nranks > ((size_t)-1 / per_rank)) {
        return 0xffff4004;
    }
    size_t total = per_rank * nranks;
    struct Map ms = {0}, md = {0};
    if (map_phys(fd, job->src_addr, total * sizeof(float), &ms) ||
        map_phys(fd, job->dst_addr, per_rank * sizeof(float), &md)) {
        unmap_phys(&ms); unmap_phys(&md);
        return 0xffff4003;
    }
    const float *src = (const float *)ms.ptr;
    float *dst = (float *)md.ptr;
    for (size_t i = 0; i < per_rank; i++) {
        float acc = 0.0f;
        for (size_t r = 0; r < nranks; r++) {
            acc += src[r * per_rank + i];
        }
        dst[i] = acc;
    }
    msync(md.base, md.map_len, MS_SYNC);
    unmap_phys(&ms); unmap_phys(&md);
    return 0;
}

static void *arg_payload(volatile struct Doorbell *ctl, uint32_t job_id, uint64_t seq, size_t want) {
    int slot = arg_slot_for_job(job_id);
    if (slot < 0) return NULL;
    char *base = (char *)ctl;
    volatile struct ArgSlotHeader *h =
        (volatile struct ArgSlotHeader *)(base + HETGPU_PACC_ARG_BASE_OFF +
                                          (size_t)slot * HETGPU_PACC_ARG_SLOT_BYTES);
    if (h->magic != HETGPU_PACC_JOB_MAGIC || h->version != HETGPU_PACC_JOB_VERSION ||
        h->job_id != job_id || h->seq != seq || h->arg_len < want) {
        return NULL;
    }
    return (void *)((char *)h + sizeof(*h));
}

static bool refresh_runtime_table(volatile struct Doorbell *ctl, struct PreloadedJobs *jobs, uint64_t *last_table_seq) {
    volatile struct RuntimeJobTable *table =
        (volatile struct RuntimeJobTable *)((volatile char *)ctl + HETGPU_PACC_RUNTIME_TABLE_OFF);
    if (table->magic != HETGPU_PACC_RUNTIME_TABLE_MAGIC ||
        table->version != HETGPU_PACC_RUNTIME_TABLE_VERSION ||
        table->seq == 0 ||
        table->seq == *last_table_seq) {
        return false;
    }

    struct RuntimeJobTable local;
    memcpy(&local, (const void *)table, sizeof(local));
    __sync_synchronize();
    if (local.magic != HETGPU_PACC_RUNTIME_TABLE_MAGIC ||
        local.version != HETGPU_PACC_RUNTIME_TABLE_VERSION ||
        local.seq == 0 ||
        local.seq == *last_table_seq) {
        return false;
    }

    if (local.have_gemm) {
        jobs->gemm = local.gemm;
        jobs->have_gemm = true;
    }
    if (local.have_softmax) {
        jobs->softmax = local.softmax;
        jobs->have_softmax = true;
    }
    if (local.have_rmsnorm) {
        jobs->rmsnorm = local.rmsnorm;
        jobs->have_rmsnorm = true;
    }
    if (local.have_allreduce) {
        jobs->allreduce = local.allreduce;
        jobs->have_allreduce = true;
    }
    *last_table_seq = local.seq;
    log_msg("runtime table seq=%" PRIu64 " have_gemm=%u have_softmax=%u have_rmsnorm=%u have_allreduce=%u",
            local.seq, local.have_gemm, local.have_softmax, local.have_rmsnorm, local.have_allreduce);
    return true;
}

static int dispatch_job(int fd, volatile struct Doorbell *ctl, const struct PreloadedJobs *jobs, bool strict) {
    uint32_t job_id = ctl->job_id;
    uint64_t seq = ctl->seq;
    if (job_id == HETGPU_PACC_JOB_GEMM) {
        const struct GemmJob *job = jobs->have_gemm ? &jobs->gemm : NULL;
        if (!strict) {
            const struct GemmJob *dyn = arg_payload(ctl, job_id, seq, sizeof(struct GemmJob));
            if (dyn) job = dyn;
        }
        return job ? run_gemm(fd, job) : (int)0xffff0101U;
    }
    if (job_id == HETGPU_PACC_JOB_SOFTMAX) {
        const struct SoftmaxJob *job = jobs->have_softmax ? &jobs->softmax : NULL;
        if (!strict) {
            const struct SoftmaxJob *dyn = arg_payload(ctl, job_id, seq, sizeof(struct SoftmaxJob));
            if (dyn) job = dyn;
        }
        return job ? run_softmax(fd, job) : (int)0xffff0201U;
    }
    if (job_id == HETGPU_PACC_JOB_RMSNORM) {
        const struct RmsNormJob *job = jobs->have_rmsnorm ? &jobs->rmsnorm : NULL;
        if (!strict) {
            const struct RmsNormJob *dyn = arg_payload(ctl, job_id, seq, sizeof(struct RmsNormJob));
            if (dyn) job = dyn;
        }
        return job ? run_rmsnorm(fd, job) : (int)0xffff0301U;
    }
    if (job_id == HETGPU_PACC_JOB_ALLREDUCE) {
        const struct AllReduceJob *job = jobs->have_allreduce ? &jobs->allreduce : NULL;
        if (!strict) {
            const struct AllReduceJob *dyn = arg_payload(ctl, job_id, seq, sizeof(struct AllReduceJob));
            if (dyn) job = dyn;
        }
        return job ? run_allreduce(fd, job) : (int)0xffff0401U;
    }
    return 0xffff00ff;
}

static volatile struct Doorbell *scan_for_control(int fd, const struct ScanRange *ranges, size_t nranges) {
    const size_t chunk = 1UL << 20;
    for (size_t r = 0; r < nranges; r++) {
        for (uint64_t base = ranges[r].start; base < ranges[r].end; base += chunk) {
            size_t len = chunk;
            if (base + len > ranges[r].end) len = (size_t)(ranges[r].end - base);
            struct Map m = {0};
            if (map_phys(fd, base, len, &m)) continue;
            char *p = (char *)m.ptr;
            for (size_t off = 0; off + sizeof(struct Doorbell) <= len; off += (size_t)g_page_size) {
                volatile struct Doorbell *d = (volatile struct Doorbell *)(p + off);
                if (d->magic == HETGPU_PACC_JOB_MAGIC && d->version == HETGPU_PACC_JOB_VERSION) {
                    log_msg("found control page at phys 0x%" PRIx64, base + off);
                    return d;
                }
            }
            unmap_phys(&m);
        }
    }
    return NULL;
}

static volatile struct Doorbell *map_mailbox_control(int fd, struct Map *map) {
    if (map_phys(fd, AP2PACC_MBOX_PHYS, HETGPU_PACC_CONTROL_BYTES, map)) {
        log_msg("map AP2PACC mailbox 0x%" PRIx64 " failed: %s",
                (uint64_t)AP2PACC_MBOX_PHYS, strerror(errno));
        return NULL;
    }
    log_msg("polling AP2PACC mailbox at phys 0x%" PRIx64, (uint64_t)AP2PACC_MBOX_PHYS);
    return (volatile struct Doorbell *)map->ptr;
}

static void pid1_bootstrap_devices(void) {
    mkdir("/proc", 0555);
    mkdir("/sys", 0555);
    mkdir("/dev", 0755);
    mkdir("/tmp", 01777);
    mount("proc", "/proc", "proc", 0, "");
    mount("sysfs", "/sys", "sysfs", 0, "");
    mount("devtmpfs", "/dev", "devtmpfs", 0, "mode=0755");
    mknod("/dev/null", S_IFCHR | 0666, makedev(1, 3));
    mknod("/dev/console", S_IFCHR | 0600, makedev(5, 1));
    mknod("/dev/mem", S_IFCHR | 0600, makedev(1, 1));
}

static void mirror_host_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status) {
    struct Map map = {0};
    if (map_phys(fd, PACC2AP_MBOX_PHYS, sizeof(struct HostStatus), &map)) {
        log_msg("map PACC2AP mailbox 0x%" PRIx64 " failed: %s",
                (uint64_t)PACC2AP_MBOX_PHYS, strerror(errno));
        return;
    }
    volatile struct HostStatus *host = (volatile struct HostStatus *)map.ptr;
    host->magic = HETGPU_PACC_JOB_MAGIC;
    host->version = HETGPU_PACC_JOB_VERSION;
    host->job_id = job_id;
    host->status = status;
    host->seq = seq;
    __sync_synchronize();
    msync(map.base, map.map_len, MS_SYNC);
    unmap_phys(&map);
}

int main(int argc, char **argv) {
    const char *devmem = "/dev/mem";
    const char *config = "/etc/hetgpu_pacc_jobs.conf";
    bool strict = false;
    bool scan_ddr_control = false;
    unsigned poll_us = 1000;
    struct ScanRange ranges[4] = {
        {0x80100000ULL, 0xc0000000ULL},
        {0xe0000000ULL, 0xffc00000ULL},
        {0x20100000000ULL, 0x20110000000ULL},
        {0, 0},
    };
    size_t nranges = 3;

    if (getpid() == 1) {
        pid1_bootstrap_devices();
        strict = true;
        config = "/etc/skel/.bashrc";
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--strict-job-id-only")) {
            strict = true;
        } else if (!strcmp(argv[i], "--devmem") && i + 1 < argc) {
            devmem = argv[++i];
        } else if (!strcmp(argv[i], "--config") && i + 1 < argc) {
            config = argv[++i];
        } else if (!strcmp(argv[i], "--poll-us") && i + 1 < argc) {
            poll_us = (unsigned)strtoul(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--scan-ddr-control")) {
            scan_ddr_control = true;
        }
    }

    g_page_size = sysconf(_SC_PAGESIZE);
    if (g_page_size <= 0) g_page_size = 4096;

    struct PreloadedJobs jobs;
    memset(&jobs, 0, sizeof(jobs));
    load_jobs_config(config, &jobs);

    int fd = open(devmem, O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        log_msg("open %s failed: %s", devmem, strerror(errno));
        return 1;
    }

    log_msg("started strict=%d scan_ddr_control=%d config=%s",
            strict ? 1 : 0, scan_ddr_control ? 1 : 0, config);
    mirror_host_status(fd, 0, 0, 0x600d);
    struct Map mailbox_map = {0};
    volatile struct Doorbell *ctl = NULL;
    uint64_t last_seq = 0;
    uint64_t last_table_seq = 0;
    if (!scan_ddr_control) {
        ctl = map_mailbox_control(fd, &mailbox_map);
        if (!ctl) {
            close(fd);
            return 1;
        }
    }
    for (;;) {
        if (!ctl) {
            ctl = scan_for_control(fd, ranges, nranges);
            if (!ctl) {
                usleep(100000);
                continue;
            }
        }
        if (ctl->magic != HETGPU_PACC_JOB_MAGIC || ctl->version != HETGPU_PACC_JOB_VERSION) {
            if (scan_ddr_control) {
                ctl = NULL;
            }
            usleep(poll_us);
            continue;
        }
        if (ctl->seq != last_seq) {
            last_seq = ctl->seq;
            uint32_t job_id = ctl->job_id;
            ctl->status = 1;
            __sync_synchronize();
            refresh_runtime_table(ctl, &jobs, &last_table_seq);
            int status = dispatch_job(fd, ctl, &jobs, strict);
            __sync_synchronize();
            ctl->status = (uint32_t)status;
            mirror_host_status(fd, job_id, last_seq, (uint32_t)status);
            log_msg("job_id=%u seq=%" PRIu64 " status=0x%x", job_id, last_seq, (uint32_t)status);
        }
        usleep(poll_us);
    }
}
