typedef unsigned long u64;
typedef unsigned int u32;

#define AP2PACC_SEC_MBSRAM 0x20000000UL
#define PACC2AP_SEC_MBSRAM 0x20002000UL

#define HETGPU_PACC_JOB_MAGIC 0x4847505550414343UL
#define HETGPU_PACC_JOB_VERSION 1U

#define HETGPU_PACC_JOB_GEMM 1U
#define HETGPU_PACC_JOB_SOFTMAX 2U
#define HETGPU_PACC_JOB_RMSNORM 3U
#define HETGPU_PACC_JOB_ALLREDUCE 4U

#define HETGPU_PACC_ARG_BASE (AP2PACC_SEC_MBSRAM + 0x100UL)
#define HETGPU_PACC_ARG_SLOT_BYTES 0x400UL

#define PACC_DTYPE_F32 4U

struct Doorbell {
    u64 magic;
    u32 version;
    u32 job_id;
    u32 flags;
    u32 status;
    u64 seq;
};

struct ArgSlotHeader {
    u64 magic;
    u32 version;
    u32 job_id;
    u64 seq;
    u64 arg_len;
};

struct GemmJob {
    u32 transa;
    u32 transb;
    u32 atype;
    u32 btype;
    u32 ctype;
    u32 compute_type;
    u64 m;
    u64 n;
    u64 k;
    u64 a_addr;
    u64 b_addr;
    u64 c_addr;
    u64 alpha_addr;
    u64 beta_addr;
    long lda;
    long ldb;
    long ldc;
    long stride_a;
    long stride_b;
    long stride_c;
    u64 batch_count;
};

struct SoftmaxJob {
    u64 src_addr;
    u64 dst_addr;
    u64 rows;
    u64 cols;
    u64 stride;
    u32 dtype;
    u32 reserved;
};

struct RmsNormJob {
    u64 x_addr;
    u64 weight_addr;
    u64 y_addr;
    u64 rows;
    u64 hidden;
    float eps;
    u32 dtype;
};

static inline void fence_all(void) {
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline float fabsf_local(float x) {
    return x < 0.0f ? -x : x;
}

static float expf_fast(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) x = 20.0f;
    float term = 1.0f;
    float sum = 1.0f;
    for (int i = 1; i <= 8; ++i) {
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
    for (int i = 0; i < 6; ++i) {
        y = y * (1.5f - 0.5f * x * y * y);
    }
    return y;
}

static void gemm_f32(const struct GemmJob *job) {
    const float alpha = job->alpha_addr ? *(const float *)job->alpha_addr : 1.0f;
    const float beta = job->beta_addr ? *(const float *)job->beta_addr : 0.0f;

    for (u64 batch = 0; batch < job->batch_count; ++batch) {
        const float *a = (const float *)(job->a_addr + (u64)(job->stride_a > 0 ? job->stride_a : 0) * batch * sizeof(float));
        const float *b = (const float *)(job->b_addr + (u64)(job->stride_b > 0 ? job->stride_b : 0) * batch * sizeof(float));
        float *c = (float *)(job->c_addr + (u64)(job->stride_c > 0 ? job->stride_c : 0) * batch * sizeof(float));

        for (u64 row = 0; row < job->m; ++row) {
            for (u64 col = 0; col < job->n; ++col) {
                float acc = 0.0f;
                for (u64 kk = 0; kk < job->k; ++kk) {
                    float av = job->transa ? a[kk * job->lda + row] : a[row * job->lda + kk];
                    float bv = job->transb ? b[col * job->ldb + kk] : b[kk * job->ldb + col];
                    acc += av * bv;
                }
                c[row * job->ldc + col] = alpha * acc + beta * c[row * job->ldc + col];
            }
        }
    }
}

static void softmax_f32(const struct SoftmaxJob *job) {
    const float *src = (const float *)job->src_addr;
    float *dst = (float *)job->dst_addr;
    const u64 stride = job->stride ? job->stride : job->cols;

    for (u64 row = 0; row < job->rows; ++row) {
        const u64 base = row * stride;
        float max_v = src[base];
        for (u64 col = 1; col < job->cols; ++col) {
            float v = src[base + col];
            max_v = v > max_v ? v : max_v;
        }
        float sum = 0.0f;
        for (u64 col = 0; col < job->cols; ++col) {
            float e = expf_fast(src[base + col] - max_v);
            dst[base + col] = e;
            sum += e;
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (u64 col = 0; col < job->cols; ++col) {
            dst[base + col] *= inv;
        }
    }
}

static void rmsnorm_f32_rvv(const struct RmsNormJob *job) {
    const float *x = (const float *)job->x_addr;
    const float *w = (const float *)job->weight_addr;
    float *y = (float *)job->y_addr;

    for (u64 row = 0; row < job->rows; ++row) {
        const u64 base = row * job->hidden;
        float sumsq = 0.0f;
        for (u64 i = 0; i < job->hidden; ++i) {
            float v = x[base + i];
            sumsq += v * v;
        }
        __asm__ volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"(job->hidden));
        float scale = rsqrtf_newton(sumsq / (float)job->hidden + job->eps);
        for (u64 i = 0; i < job->hidden; ++i) {
            float weight = w ? w[i] : 1.0f;
            y[base + i] = x[base + i] * scale * weight;
        }
    }
}

static volatile struct ArgSlotHeader *arg_slot(u32 job_id) {
    u64 slot = 0;
    if (job_id == HETGPU_PACC_JOB_GEMM) {
        slot = 0;
    } else if (job_id == HETGPU_PACC_JOB_SOFTMAX) {
        slot = 1;
    } else if (job_id == HETGPU_PACC_JOB_RMSNORM) {
        slot = 2;
    } else if (job_id == HETGPU_PACC_JOB_ALLREDUCE) {
        slot = 3;
    } else {
        slot = 0xff;
    }
    if (slot == 0xff) {
        return (volatile struct ArgSlotHeader *)0;
    }
    return (volatile struct ArgSlotHeader *)(HETGPU_PACC_ARG_BASE + slot * HETGPU_PACC_ARG_SLOT_BYTES);
}

static void *arg_payload(volatile struct ArgSlotHeader *slot) {
    return (void *)((u64)slot + sizeof(struct ArgSlotHeader));
}

static void run_preloaded_job(volatile struct Doorbell *doorbell) {
    volatile struct ArgSlotHeader *slot = arg_slot(doorbell->job_id);
    doorbell->status = 1;
    fence_all();

    if (doorbell->version != HETGPU_PACC_JOB_VERSION) {
        doorbell->status = 0xffff0001U;
    } else if (!slot || slot->magic != HETGPU_PACC_JOB_MAGIC ||
               slot->version != HETGPU_PACC_JOB_VERSION ||
               slot->job_id != doorbell->job_id ||
               slot->seq != doorbell->seq) {
        doorbell->status = 0xffff0005U;
    } else if (doorbell->job_id == HETGPU_PACC_JOB_GEMM) {
        const struct GemmJob *job = (const struct GemmJob *)arg_payload(slot);
        if (job->atype == PACC_DTYPE_F32 && job->btype == PACC_DTYPE_F32 && job->ctype == PACC_DTYPE_F32) {
            gemm_f32(job);
            doorbell->status = 0;
        } else {
            doorbell->status = 0xffff0002U;
        }
    } else if (doorbell->job_id == HETGPU_PACC_JOB_SOFTMAX) {
        const struct SoftmaxJob *job = (const struct SoftmaxJob *)arg_payload(slot);
        if (job->dtype == PACC_DTYPE_F32) {
            softmax_f32(job);
            doorbell->status = 0;
        } else {
            doorbell->status = 0xffff0003U;
        }
    } else if (doorbell->job_id == HETGPU_PACC_JOB_RMSNORM) {
        const struct RmsNormJob *job = (const struct RmsNormJob *)arg_payload(slot);
        if (job->dtype == PACC_DTYPE_F32) {
            rmsnorm_f32_rvv(job);
            doorbell->status = 0;
        } else {
            doorbell->status = 0xffff0004U;
        }
    } else {
        doorbell->status = 0xffff00ffU;
    }
    fence_all();
}

__attribute__((section(".text.start")))
void _start(void) {
    volatile struct Doorbell *doorbell = (volatile struct Doorbell *)AP2PACC_SEC_MBSRAM;
    volatile u64 *heartbeat = (volatile u64 *)PACC2AP_SEC_MBSRAM;
    u64 last_seq = 0;

    for (;;) {
        if (doorbell->magic == HETGPU_PACC_JOB_MAGIC && doorbell->seq != last_seq) {
            last_seq = doorbell->seq;
            run_preloaded_job(doorbell);
            *heartbeat = last_seq;
        }
        __asm__ volatile("wfi");
    }
}
