#if defined(__clang__)
#include <riscv_vector.h>
#include <sifive_vector.h>
#endif

typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed char s8;
typedef signed int s32;

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

#define PACC_DTYPE_INT8 0U
#define PACC_DTYPE_UINT8 1U
#define PACC_DTYPE_INT32 2U
#define PACC_DTYPE_F32 4U
#define PACC_DTYPE_BF16 5U

#define XM_INT8_TILE_M 4U
#define XM_INT8_TILE_N 4U
#define XM_INT8_TILE_K 8U
#define XM_BF16_TILE_M 4U
#define XM_BF16_TILE_N 4U
#define XM_BF16_TILE_K 4U

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

static float bf16_to_f32(u16 x) {
    union {
        u32 u;
        float f;
    } v;
    v.u = (u32)x << 16;
    return v.f;
}

static u16 f32_to_bf16(float x) {
    union {
        float f;
        u32 u;
    } v;
    v.f = x;
    u32 lsb = (v.u >> 16) & 1U;
    u32 rounding_bias = 0x7fffU + lsb;
    return (u16)((v.u + rounding_bias) >> 16);
}

static s32 round_to_i32(float x) {
    if (x >= 2147483647.0f) return 2147483647;
    if (x <= -2147483648.0f) return (-2147483647 - 1);
    return x >= 0.0f ? (s32)(x + 0.5f) : (s32)(x - 0.5f);
}

static s8 round_to_i8(float x) {
    s32 v = round_to_i32(x);
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return (s8)v;
}

static u8 round_to_u8(float x) {
    s32 v = round_to_i32(x);
    if (v > 255) v = 255;
    if (v < 0) v = 0;
    return (u8)v;
}

static float load_typed(const void *base, u64 idx, u32 dtype) {
    if (dtype == PACC_DTYPE_INT8) {
        return (float)((const s8 *)base)[idx];
    }
    if (dtype == PACC_DTYPE_UINT8) {
        return (float)((const u8 *)base)[idx];
    }
    if (dtype == PACC_DTYPE_INT32) {
        return (float)((const s32 *)base)[idx];
    }
    if (dtype == PACC_DTYPE_F32) {
        return ((const float *)base)[idx];
    }
    if (dtype == PACC_DTYPE_BF16) {
        return bf16_to_f32(((const u16 *)base)[idx]);
    }
    return 0.0f;
}

static void store_typed(void *base, u64 idx, u32 dtype, float value) {
    if (dtype == PACC_DTYPE_INT8) {
        ((s8 *)base)[idx] = round_to_i8(value);
    } else if (dtype == PACC_DTYPE_UINT8) {
        ((u8 *)base)[idx] = round_to_u8(value);
    } else if (dtype == PACC_DTYPE_INT32) {
        ((s32 *)base)[idx] = round_to_i32(value);
    } else if (dtype == PACC_DTYPE_F32) {
        ((float *)base)[idx] = value;
    } else if (dtype == PACC_DTYPE_BF16) {
        ((u16 *)base)[idx] = f32_to_bf16(value);
    }
}

static int dtype_supported(u32 dtype) {
    return dtype == PACC_DTYPE_INT8 || dtype == PACC_DTYPE_UINT8 ||
           dtype == PACC_DTYPE_INT32 || dtype == PACC_DTYPE_F32 ||
           dtype == PACC_DTYPE_BF16;
}

static u64 dtype_size(u32 dtype) {
    if (dtype == PACC_DTYPE_INT8 || dtype == PACC_DTYPE_UINT8) {
        return 1;
    }
    if (dtype == PACC_DTYPE_INT32 || dtype == PACC_DTYPE_F32) {
        return 4;
    }
    if (dtype == PACC_DTYPE_BF16) {
        return 2;
    }
    return 0;
}

static float read_scale(const void *ptr, float default_value) {
    return ptr ? *(const float *)ptr : default_value;
}

static void gemm_scalar_block(const struct GemmJob *job,
                              const void *a,
                              const void *b,
                              void *c,
                              float alpha,
                              float beta,
                              u64 row0,
                              u64 row1,
                              u64 col0,
                              u64 col1) {
    for (u64 row = row0; row < row1; ++row) {
        for (u64 col = col0; col < col1; ++col) {
            float acc = 0.0f;
            for (u64 kk = 0; kk < job->k; ++kk) {
                u64 a_idx = job->transa ? kk + row * (u64)job->lda : row + kk * (u64)job->lda;
                u64 b_idx = job->transb ? col + kk * (u64)job->ldb : kk + col * (u64)job->ldb;
                acc += load_typed(a, a_idx, job->atype) * load_typed(b, b_idx, job->btype);
            }
            u64 c_idx = row + col * (u64)job->ldc;
            float old = beta != 0.0f ? load_typed(c, c_idx, job->ctype) : 0.0f;
            store_typed(c, c_idx, job->ctype, alpha * acc + beta * old);
        }
    }
}

#if defined(__clang__) && defined(__riscv_xsfvcp)
static void xm_gemm_qoq_4x8x4_tile(const struct GemmJob *job,
                                   const void *a,
                                   const void *b,
                                   u64 row0,
                                   u64 col0,
                                   u64 k0,
                                   s32 *acc) {
    s8 a_pack[XM_INT8_TILE_M * XM_INT8_TILE_K] = {0};
    u8 b_pack_u[XM_INT8_TILE_K * XM_INT8_TILE_N] = {0};
    s8 b_pack_s[XM_INT8_TILE_K * XM_INT8_TILE_N] = {0};

    for (u64 row = 0; row < XM_INT8_TILE_M; ++row) {
        for (u64 kk = 0; kk < XM_INT8_TILE_K; ++kk) {
            u64 a_idx = (row0 + row) + (k0 + kk) * (u64)job->lda;
            a_pack[row * XM_INT8_TILE_K + kk] = (s8)load_typed(a, a_idx, job->atype);
        }
    }

    for (u64 kk = 0; kk < XM_INT8_TILE_K; ++kk) {
        for (u64 col = 0; col < XM_INT8_TILE_N; ++col) {
            u64 b_idx = (k0 + kk) + (col0 + col) * (u64)job->ldb;
            float value = load_typed(b, b_idx, job->btype);
            b_pack_s[kk * XM_INT8_TILE_N + col] = (s8)value;
            b_pack_u[kk * XM_INT8_TILE_N + col] = (u8)value;
        }
    }

    vint32m1_t vacc = __riscv_vle32_v_i32m1((const s32 *)acc, XM_INT8_TILE_M * XM_INT8_TILE_N);

#if defined(__riscv_xsfvqmaccqoq)
    if (job->atype == PACC_DTYPE_INT8 && job->btype == PACC_DTYPE_INT8) {
        vint8m1_t va = __riscv_vle8_v_i8m1((const s8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vint8mf2_t vb = __riscv_vle8_v_i8mf2((const s8 *)b_pack_s, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmacc_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    } else if (job->atype == PACC_DTYPE_UINT8 && job->btype == PACC_DTYPE_UINT8) {
        vuint8m1_t va = __riscv_vle8_v_u8m1((const u8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vuint8mf2_t vb = __riscv_vle8_v_u8mf2((const u8 *)b_pack_u, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmaccu_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    } else if (job->atype == PACC_DTYPE_INT8 && job->btype == PACC_DTYPE_UINT8) {
        vint8m1_t va = __riscv_vle8_v_i8m1((const s8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vuint8mf2_t vb = __riscv_vle8_v_u8mf2((const u8 *)b_pack_u, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmaccsu_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    } else if (job->atype == PACC_DTYPE_UINT8 && job->btype == PACC_DTYPE_INT8) {
        vuint8m1_t va = __riscv_vle8_v_u8m1((const u8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vint8mf2_t vb = __riscv_vle8_v_i8mf2((const s8 *)b_pack_s, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmaccus_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    }
#endif

    __riscv_vse32_v_i32m1((s32 *)acc, vacc, XM_INT8_TILE_M * XM_INT8_TILE_N);
}

static void xm_gemm_qqq_4x4x4_tile(const struct GemmJob *job,
                                   const void *a,
                                   const void *b,
                                   u64 row0,
                                   u64 col0,
                                   u64 k0,
                                   float *acc) {
    u16 a_pack[XM_BF16_TILE_M * XM_BF16_TILE_K] = {0};
    u16 b_pack[XM_BF16_TILE_K * XM_BF16_TILE_N] = {0};

    for (u64 row = 0; row < XM_BF16_TILE_M; ++row) {
        for (u64 kk = 0; kk < XM_BF16_TILE_K; ++kk) {
            u64 a_idx = (row0 + row) * (u64)job->lda + (k0 + kk);
            a_pack[row * XM_BF16_TILE_K + kk] = f32_to_bf16(load_typed(a, a_idx, job->atype));
        }
    }

    for (u64 kk = 0; kk < XM_BF16_TILE_K; ++kk) {
        for (u64 col = 0; col < XM_BF16_TILE_N; ++col) {
            u64 b_idx = (k0 + kk) + (col0 + col) * (u64)job->ldb;
            b_pack[kk * XM_BF16_TILE_N + col] = f32_to_bf16(load_typed(b, b_idx, job->btype));
        }
    }

#if defined(__riscv_xsfvfwmaccqqq)
    vfloat32m1_t vacc = __riscv_vle32_v_f32m1((const float *)acc, XM_BF16_TILE_M * XM_BF16_TILE_N);
    vbfloat16m1_t va = __riscv_vle16_v_bf16m1((const __bf16 *)(const void *)a_pack, XM_BF16_TILE_M * XM_BF16_TILE_K);
    vbfloat16mf2_t vb = __riscv_vle16_v_bf16mf2((const __bf16 *)(const void *)b_pack, XM_BF16_TILE_K * XM_BF16_TILE_N);
    vacc = __riscv_sf_vfwmacc_4x4x4_f32m1(vacc, va, vb, XM_BF16_TILE_K * XM_BF16_TILE_N);
    __riscv_vse32_v_f32m1((float *)acc, vacc, XM_BF16_TILE_M * XM_BF16_TILE_N);
#endif
}

static int gemm_try_xm_native(const struct GemmJob *job,
                              const void *a,
                              const void *b,
                              void *c,
                              float alpha,
                              float beta) {
    if (job->transa || job->transb) {
        return 0;
    }

    if ((job->atype == PACC_DTYPE_INT8 || job->atype == PACC_DTYPE_UINT8) &&
        (job->btype == PACC_DTYPE_INT8 || job->btype == PACC_DTYPE_UINT8)) {
        for (u64 row0 = 0; row0 < job->m; row0 += XM_INT8_TILE_M) {
            for (u64 col0 = 0; col0 < job->n; col0 += XM_INT8_TILE_N) {
                u64 row1 = row0 + XM_INT8_TILE_M <= job->m ? row0 + XM_INT8_TILE_M : job->m;
                u64 col1 = col0 + XM_INT8_TILE_N <= job->n ? col0 + XM_INT8_TILE_N : job->n;
                if (row1 - row0 != XM_INT8_TILE_M || col1 - col0 != XM_INT8_TILE_N) {
                    gemm_scalar_block(job, a, b, c, alpha, beta, row0, row1, col0, col1);
                    continue;
                }

                s32 acc[XM_INT8_TILE_M * XM_INT8_TILE_N] = {0};
                u64 k_native = job->k - (job->k % XM_INT8_TILE_K);
                for (u64 k0 = 0; k0 < k_native; k0 += XM_INT8_TILE_K) {
                    xm_gemm_qoq_4x8x4_tile(job, a, b, row0, col0, k0, acc);
                }

                for (u64 row = row0; row < row1; ++row) {
                    for (u64 col = col0; col < col1; ++col) {
                        float value = (float)acc[(row - row0) * XM_INT8_TILE_N + (col - col0)];
                        for (u64 kk = k_native; kk < job->k; ++kk) {
                            u64 a_idx = row + kk * (u64)job->lda;
                            u64 b_idx = kk + col * (u64)job->ldb;
                            value += load_typed(a, a_idx, job->atype) * load_typed(b, b_idx, job->btype);
                        }
                        u64 c_idx = row + col * (u64)job->ldc;
                        float old = beta != 0.0f ? load_typed(c, c_idx, job->ctype) : 0.0f;
                        store_typed(c, c_idx, job->ctype, alpha * value + beta * old);
                    }
                }
            }
        }
        return 1;
    }

    if (job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16) {
        for (u64 row0 = 0; row0 < job->m; row0 += XM_BF16_TILE_M) {
            for (u64 col0 = 0; col0 < job->n; col0 += XM_BF16_TILE_N) {
                u64 row1 = row0 + XM_BF16_TILE_M <= job->m ? row0 + XM_BF16_TILE_M : job->m;
                u64 col1 = col0 + XM_BF16_TILE_N <= job->n ? col0 + XM_BF16_TILE_N : job->n;
                if (row1 - row0 != XM_BF16_TILE_M || col1 - col0 != XM_BF16_TILE_N) {
                    gemm_scalar_block(job, a, b, c, alpha, beta, row0, row1, col0, col1);
                    continue;
                }

                float acc[XM_BF16_TILE_M * XM_BF16_TILE_N] = {0.0f};
                u64 k_native = job->k - (job->k % XM_BF16_TILE_K);
                for (u64 k0 = 0; k0 < k_native; k0 += XM_BF16_TILE_K) {
                    xm_gemm_qqq_4x4x4_tile(job, a, b, row0, col0, k0, acc);
                }

                for (u64 row = row0; row < row1; ++row) {
                    for (u64 col = col0; col < col1; ++col) {
                        float value = acc[(row - row0) * XM_BF16_TILE_N + (col - col0)];
                        for (u64 kk = k_native; kk < job->k; ++kk) {
                            u64 a_idx = row + kk * (u64)job->lda;
                            u64 b_idx = kk + col * (u64)job->ldb;
                            value += load_typed(a, a_idx, job->atype) * load_typed(b, b_idx, job->btype);
                        }
                        u64 c_idx = row + col * (u64)job->ldc;
                        float old = beta != 0.0f ? load_typed(c, c_idx, job->ctype) : 0.0f;
                        store_typed(c, c_idx, job->ctype, alpha * value + beta * old);
                    }
                }
            }
        }
        return 1;
    }

    return 0;
}
#endif

static void gemm_typed(const struct GemmJob *job) {
    const float alpha = read_scale((const void *)job->alpha_addr, 1.0f);
    const float beta = read_scale((const void *)job->beta_addr, 0.0f);

    for (u64 batch = 0; batch < job->batch_count; ++batch) {
        const void *a = (const void *)(job->a_addr + (u64)(job->stride_a > 0 ? job->stride_a : 0) * batch * dtype_size(job->atype));
        const void *b = (const void *)(job->b_addr + (u64)(job->stride_b > 0 ? job->stride_b : 0) * batch * dtype_size(job->btype));
        void *c = (void *)(job->c_addr + (u64)(job->stride_c > 0 ? job->stride_c : 0) * batch * dtype_size(job->ctype));

#if defined(__clang__) && defined(__riscv_xsfvcp)
        if (gemm_try_xm_native(job, a, b, c, alpha, beta)) {
            continue;
        }
#endif
        gemm_scalar_block(job, a, b, c, alpha, beta, 0, job->m, 0, job->n);
    }
}

static void softmax_typed(const struct SoftmaxJob *job) {
    const void *src = (const void *)job->src_addr;
    void *dst = (void *)job->dst_addr;
    const u64 stride = job->stride ? job->stride : job->cols;

    for (u64 row = 0; row < job->rows; ++row) {
        const u64 base = row * stride;
        float max_v = load_typed(src, base, job->dtype);
        for (u64 col = 1; col < job->cols; ++col) {
            float v = load_typed(src, base + col, job->dtype);
            max_v = v > max_v ? v : max_v;
        }
        float sum = 0.0f;
        for (u64 col = 0; col < job->cols; ++col) {
            sum += expf_fast(load_typed(src, base + col, job->dtype) - max_v);
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (u64 col = 0; col < job->cols; ++col) {
            float e = expf_fast(load_typed(src, base + col, job->dtype) - max_v);
            store_typed(dst, base + col, job->dtype, e * inv);
        }
    }
}

static void rmsnorm_typed(const struct RmsNormJob *job) {
    const void *x = (const void *)job->x_addr;
    const void *w = (const void *)job->weight_addr;
    void *y = (void *)job->y_addr;

    for (u64 row = 0; row < job->rows; ++row) {
        const u64 base = row * job->hidden;
        float sumsq = 0.0f;
        for (u64 i = 0; i < job->hidden; ++i) {
            float v = load_typed(x, base + i, job->dtype);
            sumsq += v * v;
        }
        float scale = rsqrtf_newton(sumsq / (float)job->hidden + job->eps);
        for (u64 i = 0; i < job->hidden; ++i) {
            float weight = w ? load_typed(w, i, job->dtype) : 1.0f;
            store_typed(y, base + i, job->dtype, load_typed(x, base + i, job->dtype) * scale * weight);
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
        if (dtype_supported(job->atype) && dtype_supported(job->btype) && dtype_supported(job->ctype)) {
            gemm_typed(job);
            doorbell->status = 0;
        } else {
            doorbell->status = 0xffff0002U;
        }
    } else if (doorbell->job_id == HETGPU_PACC_JOB_SOFTMAX) {
        const struct SoftmaxJob *job = (const struct SoftmaxJob *)arg_payload(slot);
        if (dtype_supported(job->dtype)) {
            softmax_typed(job);
            doorbell->status = 0;
        } else {
            doorbell->status = 0xffff0003U;
        }
    } else if (doorbell->job_id == HETGPU_PACC_JOB_RMSNORM) {
        const struct RmsNormJob *job = (const struct RmsNormJob *)arg_payload(slot);
        if (dtype_supported(job->dtype)) {
            rmsnorm_typed(job);
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
