#if defined(__riscv) && defined(__has_include)
#if __has_include(<riscv_vector.h>)
#include <riscv_vector.h>
#define HETGPU_HAVE_RISCV_VECTOR_H 1
#endif
#if defined(HETGPU_HAVE_RISCV_VECTOR_H) && __has_include(<sifive_vector.h>)
#include <sifive_vector.h>
#endif
#endif

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed char s8;
typedef signed int s32;

#define HETGPU_SIFIVE_JOB_MAGIC 0x4847505550414343UL
#define HETGPU_SIFIVE_JOB_VERSION 1U

#define HETGPU_SIFIVE_JOB_GEMM 1U
#define HETGPU_SIFIVE_JOB_SOFTMAX 2U
#define HETGPU_SIFIVE_JOB_RMSNORM 3U
#define HETGPU_SIFIVE_JOB_ALLREDUCE 4U
#define HETGPU_SIFIVE_JOB_MMVF 5U
#define HETGPU_SIFIVE_JOB_KERNEL 0U
#define SIFIVE_JOB_MAGIC 0x504143434a4f4231UL

#define HETGPU_SIFIVE_CONTROL_BYTES 0x2000UL
#define HETGPU_SIFIVE_DOORBELL_OFF 0x0UL
#define HETGPU_SIFIVE_ARG_BASE_OFF 0x100UL
#define HETGPU_SIFIVE_ARG_SLOT_BYTES 0x400UL
#define HETGPU_SIFIVE_RUNTIME_TABLE_OFF 0x1400UL
#define HETGPU_SIFIVE_RUNTIME_TABLE_MAGIC 0x4847505554424c31UL
#define HETGPU_SIFIVE_RUNTIME_TABLE_VERSION 1U
#define HETGPU_SIFIVE_COMPLETION_OFF 0x1f20UL
#define HETGPU_SIFIVE_CORE_COUNT 4U
#define HETGPU_SIFIVE_WORKER_THREADS 4U
#define HETGPU_SIFIVE_MAX_WORKER_THREADS 64U

#define SIFIVE_IOC_MAGIC 'p'
#define SIFIVE_IOC_ZLUDA_IRQ _IO(SIFIVE_IOC_MAGIC, 5)
#define SIFIVE_IOC_ZLUDA_GET_DDR_BASE _IOR(SIFIVE_IOC_MAGIC, 6, struct sifive_zluda_ddr_info)
#define SIFIVE_IOC_GET_SIFIVE_ID _IOR(SIFIVE_IOC_MAGIC, 7, unsigned long)

#define SIFIVE_DTYPE_INT8 0U
#define SIFIVE_DTYPE_UINT8 1U
#define SIFIVE_DTYPE_INT32 2U
#define SIFIVE_DTYPE_F16 3U
#define SIFIVE_DTYPE_F32 4U
#define SIFIVE_DTYPE_BF16 5U

#define XM_INT8_TILE_M 4U
#define XM_INT8_TILE_N 4U
#define XM_INT8_TILE_K 8U
#define XSFMM16_TILE_M 32U
#define XSFMM16_TILE_N 32U
#define XSFMM16_TILE_K 2U
#define F32_TILE_M 32U
#define F32_TILE_N 32U
#define F32_TILE_K 32U

struct RuntimeConfig {
    u32 sifive_id;
    int mbox_fd;
    u64 shared_ddr_base;
    u64 shared_ddr_size;
    u64 map_len;
    u8 *shared_ddr;
    void *map_ptr;
};

static struct RuntimeConfig g_runtime;

struct sifive_zluda_ddr_info {
    u64 ddr_base;
    u64 ddr_size;
};

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

struct AllReduceJob {
    u64 src_addr;
    u64 dst_addr;
    u64 count;
    u32 nranks;
    u32 reduce_op;
    u32 dtype;
    u32 reserved;
};

struct RuntimeJobTable {
    u64 magic;
    u32 version;
    u32 flags;
    u64 seq;
    u32 have_gemm;
    u32 have_softmax;
    u32 have_rmsnorm;
    u32 have_allreduce;
    u32 have_mmvf;
    u32 reserved0;
    struct GemmJob gemm;
    struct SoftmaxJob softmax;
    struct RmsNormJob rmsnorm;
    struct AllReduceJob allreduce;
};

struct HostStatus {
    u64 magic;
    u32 version;
    u32 job_id;
    u32 status;
    u64 seq;
};

struct SifiveJobDesc {
    u64 addr;
    u64 len;
    u64 rsvd;
    u64 buf_info;
};

static inline void fence_all(void) {
#if defined(__riscv)
    __asm__ volatile("fence iorw, iorw" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static int parse_u64_arg(const char *s, u64 *out) {
    char *end = 0;
    unsigned long long value;
    if (!s || !*s) return 0;
    errno = 0;
    value = strtoull(s, &end, 0);
    if (errno || end == s || (end && *end)) return 0;
    *out = (u64)value;
    return 1;
}

static unsigned runtime_worker_threads(u64 work_items) {
    u64 requested = 0;
    const char *value = getenv("HETGPU_SIFIVE_JOBD_KERNEL_THREADS");
    if (!parse_u64_arg(value, &requested) || requested == 0) {
        requested = HETGPU_SIFIVE_WORKER_THREADS;
    }
    if (requested > HETGPU_SIFIVE_MAX_WORKER_THREADS) {
        requested = HETGPU_SIFIVE_MAX_WORKER_THREADS;
    }
    if (work_items > 0 && requested > work_items) {
        requested = work_items;
    }
    return requested ? (unsigned)requested : 1U;
}

static int parse_prefixed_arg(const char *arg, const char *prefix, u64 *out) {
    size_t n;
    if (!arg || !prefix) return 0;
    n = strlen(prefix);
    if (strncmp(arg, prefix, n) != 0) return 0;
    return parse_u64_arg(arg + n, out);
}

static const char *parse_prefixed_string_arg(const char *arg, const char *prefix) {
    size_t n;
    if (!arg || !prefix) return 0;
    n = strlen(prefix);
    if (strncmp(arg, prefix, n) != 0) return 0;
    return arg + n;
}

static int read_runtime_args(int argc, char **argv, u64 *sifive_id, const char **mbox_path) {
    const char *value;
    int have_sifive_id = 0;
    value = getenv("HETGPU_SIFIVE_ID");
    if (!parse_u64_arg(value, sifive_id)) {
        value = getenv("HETGPU_SIFIVE_DEVICE_ID");
        have_sifive_id = parse_u64_arg(value, sifive_id);
    } else {
        have_sifive_id = 1;
    }
    value = getenv("HETGPU_SIFIVE_MBOX_PATH");
    if (value && *value) *mbox_path = value;

    for (int i = 1; i < argc; ++i) {
        if (parse_prefixed_arg(argv[i], "--sifive-id=", sifive_id)) {
            have_sifive_id = 1;
        } else if (!strcmp(argv[i], "--sifive-id") && i + 1 < argc) {
            if (parse_u64_arg(argv[++i], sifive_id)) have_sifive_id = 1;
        }
        value = parse_prefixed_string_arg(argv[i], "--mbox=");
        if (value && *value) *mbox_path = value;
        if (!strcmp(argv[i], "--mbox") && i + 1 < argc) {
            *mbox_path = argv[++i];
        }
    }
    return have_sifive_id;
}

static int open_runtime_mbox(u64 sifive_id, const char *mbox_path) {
    char path[64];
    int fd;

    if (mbox_path && *mbox_path) {
        fd = open(mbox_path, O_RDWR | O_SYNC | O_CLOEXEC);
        if (fd < 0) {
            fprintf(stderr, "hetgpu_sifive_runtime: open %s failed: %s\n", mbox_path, strerror(errno));
        }
        return fd;
    }

    fd = open("/dev/mbox", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd >= 0) return fd;

    snprintf(path, sizeof(path), "/dev/hetgpu_sifive_mbox%lu", (unsigned long)sifive_id);
    fd = open(path, O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd >= 0) return fd;

    fd = open("/dev/hetgpu_sifive_mbox", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr,
                "hetgpu_sifive_runtime: open /dev/mbox, /dev/hetgpu_sifive_mbox%lu, or /dev/hetgpu_sifive_mbox failed: %s\n",
                (unsigned long)sifive_id, strerror(errno));
    }
    return fd;
}

static int wait_for_control_irq(void) {
    struct pollfd pfd;
    int ret;
    if (g_runtime.mbox_fd < 0) return -1;

    for (;;) {
        fence_all();
        pfd.fd = g_runtime.mbox_fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        ret = poll(&pfd, 1, -1);
        fence_all();
        if (ret > 0) return 0;
        if (ret == 0) continue;
        if (errno == EINTR || errno == EAGAIN) continue;
        fprintf(stderr, "hetgpu_sifive_runtime: poll mailbox failed: %s\n", strerror(errno));
        return -1;
    }
}

static int ioctl_get_shared_ddr(struct sifive_zluda_ddr_info *info) {
    if (g_runtime.mbox_fd < 0 || !info) return -1;
    if (ioctl(g_runtime.mbox_fd, SIFIVE_IOC_ZLUDA_GET_DDR_BASE, info) < 0) {
        fprintf(stderr, "hetgpu_sifive_runtime: ioctl GET_DDR_BASE failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static int ioctl_get_sifive_id(u64 *sifive_id) {
    if (g_runtime.mbox_fd < 0 || !sifive_id) return -1;
    if (ioctl(g_runtime.mbox_fd, SIFIVE_IOC_GET_SIFIVE_ID, sifive_id) < 0) {
        fprintf(stderr, "hetgpu_sifive_runtime: ioctl GET_SIFIVE_ID failed: %s\n", strerror(errno));
        return -1;
    }
    if (*sifive_id >= HETGPU_SIFIVE_CORE_COUNT) {
        fprintf(stderr, "hetgpu_sifive_runtime: ioctl GET_SIFIVE_ID returned invalid sifive_id=%lu\n",
                (unsigned long)*sifive_id);
        return -1;
    }
    return 0;
}

static void signal_host_irq(void) {
    fence_all();
    if (g_runtime.mbox_fd >= 0 && ioctl(g_runtime.mbox_fd, SIFIVE_IOC_ZLUDA_IRQ) < 0) {
        fprintf(stderr, "hetgpu_sifive_runtime: ioctl ZLUDA_IRQ failed: %s\n", strerror(errno));
    }
}

static int remap_shared_ddr(u64 base, u64 size) {
    void *map;
    u64 map_base;
    u64 map_delta;
    u64 map_size;
    long page_size;
    int mem_fd;

    if (!base || size < HETGPU_SIFIVE_CONTROL_BYTES) {
        fprintf(stderr,
                "hetgpu_sifive_runtime: need shared DDR base/size, got base=0x%lx size=0x%lx\n",
                (unsigned long)base, (unsigned long)size);
        return -1;
    }
    if (g_runtime.shared_ddr &&
        g_runtime.shared_ddr_base == base &&
        g_runtime.shared_ddr_size == size) {
        return 0;
    }
    if (g_runtime.map_ptr && g_runtime.map_len) {
        munmap(g_runtime.map_ptr, (size_t)g_runtime.map_len);
        g_runtime.map_ptr = 0;
        g_runtime.map_len = 0;
        g_runtime.shared_ddr = 0;
    }

    g_runtime.shared_ddr_base = base;
    g_runtime.shared_ddr_size = size;

    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 4096;
    map_base = base & ~((u64)page_size - 1UL);
    map_delta = base - map_base;
    map_size = size + map_delta;
    g_runtime.map_len = map_size;

    map = mmap(0, (size_t)g_runtime.map_len, PROT_READ | PROT_WRITE, MAP_SHARED,
               g_runtime.mbox_fd, 0);
    if (map == MAP_FAILED) {
        int mbox_errno = errno;
        mem_fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (mem_fd < 0) {
            fprintf(stderr,
                    "hetgpu_sifive_runtime: mmap /dev/mbox shared DDR base=0x%lx size=0x%lx failed: %s; open /dev/mem failed: %s\n",
                    (unsigned long)base, (unsigned long)size,
                    strerror(mbox_errno), strerror(errno));
            return -1;
        }
        map = mmap(0, (size_t)g_runtime.map_len, PROT_READ | PROT_WRITE, MAP_SHARED,
                   mem_fd, (off_t)map_base);
        close(mem_fd);
        if (map == MAP_FAILED) {
            fprintf(stderr,
                    "hetgpu_sifive_runtime: mmap /dev/mbox shared DDR base=0x%lx size=0x%lx failed: %s; mmap /dev/mem off=0x%lx len=0x%lx failed: %s\n",
                    (unsigned long)base, (unsigned long)size,
                    strerror(mbox_errno), (unsigned long)map_base,
                    (unsigned long)g_runtime.map_len, strerror(errno));
            return -1;
        }
    }

    g_runtime.map_ptr = map;
    g_runtime.shared_ddr = (u8 *)map + map_delta;
    return 0;
}

static int refresh_shared_ddr_from_ioctl(void) {
    struct sifive_zluda_ddr_info info = {0, 0};
    if (ioctl_get_shared_ddr(&info) != 0) return -1;
    return remap_shared_ddr(info.ddr_base, info.ddr_size);
}

static int init_runtime_io(int argc, char **argv) {
    u64 sifive_id = 0;
    int sifive_id_known;
    const char *mbox_path = 0;

    g_runtime.mbox_fd = -1;
    sifive_id_known = read_runtime_args(argc, argv, &sifive_id, &mbox_path);
    if (sifive_id_known && sifive_id >= HETGPU_SIFIVE_CORE_COUNT) {
        fprintf(stderr, "hetgpu_sifive_runtime: invalid sifive_id=%lu\n", (unsigned long)sifive_id);
        return -1;
    }

    g_runtime.mbox_fd = open_runtime_mbox(sifive_id, mbox_path);
    if (g_runtime.mbox_fd < 0) return -1;

    if (ioctl_get_sifive_id(&sifive_id) != 0 && !sifive_id_known) {
        return -1;
    }
    if (sifive_id >= HETGPU_SIFIVE_CORE_COUNT) {
        fprintf(stderr, "hetgpu_sifive_runtime: invalid sifive_id=%lu\n", (unsigned long)sifive_id);
        return -1;
    }
    g_runtime.sifive_id = (u32)sifive_id;

    if (wait_for_control_irq() != 0) return -1;
    return refresh_shared_ddr_from_ioctl();
}

static void *shared_ddr_ptr(u64 phys, u64 bytes) {
    u64 off;
    if (!phys || !g_runtime.shared_ddr) return 0;
    if (phys < g_runtime.shared_ddr_base) return 0;
    off = phys - g_runtime.shared_ddr_base;
    if (bytes == 0) bytes = 1;
    if (off > g_runtime.shared_ddr_size || bytes > g_runtime.shared_ddr_size - off) {
        return 0;
    }
    return g_runtime.shared_ddr + off;
}

static int checked_mul_u64(u64 a, u64 b, u64 *out) {
    if (a != 0 && b > (~0UL / a)) return 0;
    *out = a * b;
    return 1;
}

static int checked_add_u64(u64 a, u64 b, u64 *out) {
    if (b > ~0UL - a) return 0;
    *out = a + b;
    return 1;
}

static u64 min_u64(u64 a, u64 b) {
    return a < b ? a : b;
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

static float f16_to_f32(u16 x) {
    u32 sign = ((u32)x & 0x8000U) << 16;
    u32 exp = ((u32)x >> 10) & 0x1fU;
    u32 frac = (u32)x & 0x03ffU;
    u32 bits;
    union {
        u32 u;
        float f;
    } v;

    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            exp = 127U - 15U + 1U;
            while ((frac & 0x0400U) == 0) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x03ffU;
            bits = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1fU) {
        bits = sign | 0x7f800000U | (frac << 13);
    } else {
        bits = sign | ((exp + (127U - 15U)) << 23) | (frac << 13);
    }

    v.u = bits;
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

static u16 f32_to_f16(float x) {
    union {
        float f;
        u32 u;
    } v;
    u32 sign;
    u32 mant;
    int exp;

    v.f = x;
    sign = (v.u >> 16) & 0x8000U;
    mant = v.u & 0x007fffffU;
    exp = (int)((v.u >> 23) & 0xffU) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return (u16)sign;
        mant |= 0x00800000U;
        return (u16)(sign | (((mant >> (1 - exp)) + 0x1000U) >> 13));
    }
    if (exp >= 0x1f) {
        return (u16)(sign | 0x7c00U);
    }
    return (u16)(sign | ((u32)exp << 10) | ((mant + 0x1000U) >> 13));
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
    if (dtype == SIFIVE_DTYPE_INT8) {
        return (float)((const s8 *)base)[idx];
    }
    if (dtype == SIFIVE_DTYPE_UINT8) {
        return (float)((const u8 *)base)[idx];
    }
    if (dtype == SIFIVE_DTYPE_INT32) {
        return (float)((const s32 *)base)[idx];
    }
    if (dtype == SIFIVE_DTYPE_F16) {
        return f16_to_f32(((const u16 *)base)[idx]);
    }
    if (dtype == SIFIVE_DTYPE_F32) {
        return ((const float *)base)[idx];
    }
    if (dtype == SIFIVE_DTYPE_BF16) {
        return bf16_to_f32(((const u16 *)base)[idx]);
    }
    return 0.0f;
}

static void store_typed(void *base, u64 idx, u32 dtype, float value) {
    if (dtype == SIFIVE_DTYPE_INT8) {
        ((s8 *)base)[idx] = round_to_i8(value);
    } else if (dtype == SIFIVE_DTYPE_UINT8) {
        ((u8 *)base)[idx] = round_to_u8(value);
    } else if (dtype == SIFIVE_DTYPE_INT32) {
        ((s32 *)base)[idx] = round_to_i32(value);
    } else if (dtype == SIFIVE_DTYPE_F16) {
        ((u16 *)base)[idx] = f32_to_f16(value);
    } else if (dtype == SIFIVE_DTYPE_F32) {
        ((float *)base)[idx] = value;
    } else if (dtype == SIFIVE_DTYPE_BF16) {
        ((u16 *)base)[idx] = f32_to_bf16(value);
    }
}

static int dtype_supported(u32 dtype) {
    return dtype == SIFIVE_DTYPE_INT8 || dtype == SIFIVE_DTYPE_UINT8 ||
           dtype == SIFIVE_DTYPE_INT32 || dtype == SIFIVE_DTYPE_F16 ||
           dtype == SIFIVE_DTYPE_F32 || dtype == SIFIVE_DTYPE_BF16;
}

static u64 dtype_size(u32 dtype) {
    if (dtype == SIFIVE_DTYPE_INT8 || dtype == SIFIVE_DTYPE_UINT8) {
        return 1;
    }
    if (dtype == SIFIVE_DTYPE_F16 || dtype == SIFIVE_DTYPE_BF16) {
        return 2;
    }
    if (dtype == SIFIVE_DTYPE_INT32 || dtype == SIFIVE_DTYPE_F32) {
        return 4;
    }
    return 0;
}

static float read_scale(const void *ptr, float default_value) {
    return ptr ? *(const float *)ptr : default_value;
}

static int span_bytes(u64 elems, u32 dtype, u64 *bytes) {
    return checked_mul_u64(elems, dtype_size(dtype), bytes);
}

static int matrix_span_bytes(u64 rows, u64 cols, u64 ld, u32 dtype, int transposed, u64 *bytes) {
    u64 major;
    u64 minor;
    u64 ld_part;
    u64 max_idx;
    if (rows == 0 || cols == 0) {
        *bytes = 0;
        return 1;
    }
    major = transposed ? rows : cols;
    minor = transposed ? cols : rows;
    if (!checked_mul_u64(major - 1UL, ld, &ld_part)) return 0;
    if (!checked_add_u64(ld_part, minor - 1UL, &max_idx)) return 0;
    if (max_idx == ~0UL) return 0;
    return span_bytes(max_idx + 1UL, dtype, bytes);
}

static int strided_batch_phys(u64 base, long stride, u64 batch, u32 dtype, u64 *phys) {
    u64 stride_elems;
    u64 stride_bytes;
    if (stride <= 0) {
        *phys = base;
        return 1;
    }
    if (!checked_mul_u64((u64)stride, batch, &stride_elems)) return 0;
    if (!span_bytes(stride_elems, dtype, &stride_bytes)) return 0;
    return checked_add_u64(base, stride_bytes, phys);
}

static void *shared_ddr_ptr_or_null(u64 phys, u64 bytes) {
    if (!phys) return 0;
    return shared_ddr_ptr(phys, bytes);
}

static u32 gemm_scalar_block(const struct GemmJob *job,
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
    return 0;
}

static void gemm_accumulate_notrans_tile(const struct GemmJob *job,
                                         const void *a,
                                         const void *b,
                                         u64 row0,
                                         u64 row1,
                                         u64 col0,
                                         u64 col1,
                                         u64 k0,
                                         u64 k1,
                                         float *acc,
                                         u64 acc_stride) {
    for (u64 kk = k0; kk < k1; ++kk) {
        for (u64 row = row0; row < row1; ++row) {
            float av = load_typed(a, row + kk * (u64)job->lda, job->atype);
            float *acc_row = acc + (row - row0) * acc_stride;
            for (u64 col = col0; col < col1; ++col) {
                float bv = load_typed(b, kk + col * (u64)job->ldb, job->btype);
                acc_row[col - col0] += av * bv;
            }
        }
    }
}

static void gemm_store_notrans_tile(const struct GemmJob *job,
                                    void *c,
                                    float alpha,
                                    float beta,
                                    u64 row0,
                                    u64 row1,
                                    u64 col0,
                                    u64 col1,
                                    const float *acc,
                                    u64 acc_stride) {
    for (u64 col = col0; col < col1; ++col) {
        for (u64 row = row0; row < row1; ++row) {
            u64 c_idx = row + col * (u64)job->ldc;
            float old = beta != 0.0f ? load_typed(c, c_idx, job->ctype) : 0.0f;
            float value = acc[(row - row0) * acc_stride + (col - col0)];
            store_typed(c, c_idx, job->ctype, alpha * value + beta * old);
        }
    }
}

static int gemm_try_xsfmm16_tiled(const struct GemmJob *job,
                                  const void *a,
                                  const void *b,
                                  void *c,
                                  float alpha,
                                  float beta,
                                  u64 row_begin,
                                  u64 row_end) {
    int fp16 = job->atype == SIFIVE_DTYPE_F16 && job->btype == SIFIVE_DTYPE_F16;
    int bf16 = job->atype == SIFIVE_DTYPE_BF16 && job->btype == SIFIVE_DTYPE_BF16;
    if (job->transa || job->transb || (!fp16 && !bf16)) {
        return 0;
    }
    if (job->ctype != SIFIVE_DTYPE_F32 &&
        job->ctype != SIFIVE_DTYPE_F16 &&
        job->ctype != SIFIVE_DTYPE_BF16) {
        return 0;
    }
    if (row_end > job->m) row_end = job->m;
    if (row_begin >= row_end) return 1;

    for (u64 row0 = row_begin; row0 < row_end; row0 += XSFMM16_TILE_M) {
        u64 row1 = min_u64(row0 + XSFMM16_TILE_M, row_end);
        for (u64 col0 = 0; col0 < job->n; col0 += XSFMM16_TILE_N) {
            u64 col1 = min_u64(col0 + XSFMM16_TILE_N, job->n);
            float acc[XSFMM16_TILE_M * XSFMM16_TILE_N];
            memset(acc, 0, sizeof(acc));

            for (u64 k0 = 0; k0 < job->k; k0 += XSFMM16_TILE_K) {
                u64 k1 = min_u64(k0 + XSFMM16_TILE_K, job->k);
                /*
                 * Xsfmm32a16f shape: C[tm,tn] += A[tk,tm]^T * B[tk,tn].
                 * With VLEN=1024b the natural TEW=32 tile is tm<=32, tn<=32,
                 * and BF16/FP16 has KMAX=2. This scalar body is the fake-XM
                 * version of one sf.mm.f.f tile and can be replaced in-place.
                 */
                gemm_accumulate_notrans_tile(job, a, b, row0, row1, col0, col1,
                                             k0, k1, acc, XSFMM16_TILE_N);
            }
            gemm_store_notrans_tile(job, c, alpha, beta, row0, row1, col0, col1,
                                    acc, XSFMM16_TILE_N);
        }
    }
    return 1;
}

static int gemm_try_f32_tiled(const struct GemmJob *job,
                              const void *a,
                              const void *b,
                              void *c,
                              float alpha,
                              float beta,
                              u64 row_begin,
                              u64 row_end) {
    if (job->transa || job->transb ||
        job->atype != SIFIVE_DTYPE_F32 ||
        job->btype != SIFIVE_DTYPE_F32 ||
        job->ctype != SIFIVE_DTYPE_F32) {
        return 0;
    }
    if (row_end > job->m) row_end = job->m;
    if (row_begin >= row_end) return 1;

    for (u64 row0 = row_begin; row0 < row_end; row0 += F32_TILE_M) {
        u64 row1 = min_u64(row0 + F32_TILE_M, row_end);
        for (u64 col0 = 0; col0 < job->n; col0 += F32_TILE_N) {
            u64 col1 = min_u64(col0 + F32_TILE_N, job->n);
            float acc[F32_TILE_M * F32_TILE_N];
            memset(acc, 0, sizeof(acc));
            for (u64 k0 = 0; k0 < job->k; k0 += F32_TILE_K) {
                u64 k1 = min_u64(k0 + F32_TILE_K, job->k);
                gemm_accumulate_notrans_tile(job, a, b, row0, row1, col0, col1,
                                             k0, k1, acc, F32_TILE_N);
            }
            gemm_store_notrans_tile(job, c, alpha, beta, row0, row1, col0, col1,
                                    acc, F32_TILE_N);
        }
    }
    return 1;
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
    if (job->atype == SIFIVE_DTYPE_INT8 && job->btype == SIFIVE_DTYPE_INT8) {
        vint8m1_t va = __riscv_vle8_v_i8m1((const s8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vint8mf2_t vb = __riscv_vle8_v_i8mf2((const s8 *)b_pack_s, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmacc_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    } else if (job->atype == SIFIVE_DTYPE_UINT8 && job->btype == SIFIVE_DTYPE_UINT8) {
        vuint8m1_t va = __riscv_vle8_v_u8m1((const u8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vuint8mf2_t vb = __riscv_vle8_v_u8mf2((const u8 *)b_pack_u, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmaccu_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    } else if (job->atype == SIFIVE_DTYPE_INT8 && job->btype == SIFIVE_DTYPE_UINT8) {
        vint8m1_t va = __riscv_vle8_v_i8m1((const s8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vuint8mf2_t vb = __riscv_vle8_v_u8mf2((const u8 *)b_pack_u, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmaccsu_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    } else if (job->atype == SIFIVE_DTYPE_UINT8 && job->btype == SIFIVE_DTYPE_INT8) {
        vuint8m1_t va = __riscv_vle8_v_u8m1((const u8 *)a_pack, XM_INT8_TILE_M * XM_INT8_TILE_K);
        vint8mf2_t vb = __riscv_vle8_v_i8mf2((const s8 *)b_pack_s, XM_INT8_TILE_K * XM_INT8_TILE_N);
        vacc = __riscv_sf_vqmaccus_4x8x4_i32m1(vacc, va, vb, XM_INT8_TILE_K * XM_INT8_TILE_N);
    }
#endif

    __riscv_vse32_v_i32m1((s32 *)acc, vacc, XM_INT8_TILE_M * XM_INT8_TILE_N);
}

static int gemm_try_xm_native(const struct GemmJob *job,
                              const void *a,
                              const void *b,
                              void *c,
                              float alpha,
                              float beta,
                              u64 row_begin,
                              u64 row_end) {
    if (job->transa || job->transb) {
        return 0;
    }
    if (row_end > job->m) row_end = job->m;
    if (row_begin >= row_end) return 1;

    if ((job->atype == SIFIVE_DTYPE_INT8 || job->atype == SIFIVE_DTYPE_UINT8) &&
        (job->btype == SIFIVE_DTYPE_INT8 || job->btype == SIFIVE_DTYPE_UINT8)) {
        for (u64 row0 = row_begin; row0 < row_end; row0 += XM_INT8_TILE_M) {
            for (u64 col0 = 0; col0 < job->n; col0 += XM_INT8_TILE_N) {
                u64 row1 = row0 + XM_INT8_TILE_M <= row_end ? row0 + XM_INT8_TILE_M : row_end;
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

    return 0;
}
#endif

struct GemmWorker {
    const struct GemmJob *job;
    const void *a;
    const void *b;
    void *c;
    float alpha;
    float beta;
    u64 row0;
    u64 row1;
    u32 status;
};

static void *gemm_worker_main(void *opaque) {
    struct GemmWorker *worker = (struct GemmWorker *)opaque;
    worker->status = 0;
    if (worker->row0 >= worker->row1) {
        return 0;
    }
    if (gemm_try_xsfmm16_tiled(worker->job, worker->a, worker->b, worker->c,
                               worker->alpha, worker->beta,
                               worker->row0, worker->row1)) {
        return 0;
    }
    if (gemm_try_f32_tiled(worker->job, worker->a, worker->b, worker->c,
                           worker->alpha, worker->beta,
                           worker->row0, worker->row1)) {
        return 0;
    }
#if defined(__clang__) && defined(__riscv_xsfvcp)
    if (gemm_try_xm_native(worker->job, worker->a, worker->b, worker->c,
                           worker->alpha, worker->beta,
                           worker->row0, worker->row1)) {
        return 0;
    }
#endif
    worker->status = gemm_scalar_block(worker->job, worker->a, worker->b, worker->c,
                                       worker->alpha, worker->beta,
                                       worker->row0, worker->row1, 0, worker->job->n);
    return 0;
}

static u32 gemm_run_parallel(const struct GemmJob *job,
                             const void *a,
                             const void *b,
                             void *c,
                             float alpha,
                             float beta) {
    unsigned nthreads = runtime_worker_threads(job->m);
    pthread_t threads[HETGPU_SIFIVE_MAX_WORKER_THREADS];
    struct GemmWorker workers[HETGPU_SIFIVE_MAX_WORKER_THREADS];
    unsigned created = 0;
    u32 status = 0;

    if (nthreads <= 1) {
        if (gemm_try_xsfmm16_tiled(job, a, b, c, alpha, beta, 0, job->m)) {
            return 0;
        }
        if (gemm_try_f32_tiled(job, a, b, c, alpha, beta, 0, job->m)) {
            return 0;
        }
#if defined(__clang__) && defined(__riscv_xsfvcp)
        if (gemm_try_xm_native(job, a, b, c, alpha, beta, 0, job->m)) {
            return 0;
        }
#endif
        return gemm_scalar_block(job, a, b, c, alpha, beta, 0, job->m, 0, job->n);
    }

    memset(workers, 0, sizeof(workers));
    for (unsigned i = 0; i < nthreads; ++i) {
        u64 row0 = job->m * (u64)i / (u64)nthreads;
        u64 row1 = job->m * (u64)(i + 1U) / (u64)nthreads;
        if (row0 >= row1) continue;
        workers[i].job = job;
        workers[i].a = a;
        workers[i].b = b;
        workers[i].c = c;
        workers[i].alpha = alpha;
        workers[i].beta = beta;
        workers[i].row0 = row0;
        workers[i].row1 = row1;
        workers[i].status = 0;
        if (pthread_create(&threads[i], 0, gemm_worker_main, &workers[i]) != 0) {
            status = 0xffff0102U;
            break;
        }
        created++;
    }

    for (unsigned i = 0; i < created; ++i) {
        pthread_join(threads[i], 0);
        if (workers[i].status && !status) {
            status = workers[i].status;
        }
    }
    return status;
}

static u32 gemm_typed(const struct GemmJob *job) {
    const float *alpha_ptr = (const float *)shared_ddr_ptr_or_null(job->alpha_addr, sizeof(float));
    const float *beta_ptr = (const float *)shared_ddr_ptr_or_null(job->beta_addr, sizeof(float));
    const float alpha = read_scale(alpha_ptr, 1.0f);
    const float beta = read_scale(beta_ptr, 0.0f);
    u64 a_bytes;
    u64 b_bytes;
    u64 c_bytes;

    if ((job->alpha_addr && !alpha_ptr) || (job->beta_addr && !beta_ptr)) return 0xffff0101U;
    if (job->lda < 0 || job->ldb < 0 || job->ldc < 0) return 0xffff0101U;
    if (!matrix_span_bytes(job->m, job->k, (u64)job->lda, job->atype, job->transa, &a_bytes)) return 0xffff0101U;
    if (!matrix_span_bytes(job->k, job->n, (u64)job->ldb, job->btype, job->transb, &b_bytes)) return 0xffff0101U;
    if (!matrix_span_bytes(job->m, job->n, (u64)job->ldc, job->ctype, 0, &c_bytes)) return 0xffff0101U;

    for (u64 batch = 0; batch < job->batch_count; ++batch) {
        u64 a_phys;
        u64 b_phys;
        u64 c_phys;
        const void *a;
        const void *b;
        void *c;
        if (!strided_batch_phys(job->a_addr, job->stride_a, batch, job->atype, &a_phys) ||
            !strided_batch_phys(job->b_addr, job->stride_b, batch, job->btype, &b_phys) ||
            !strided_batch_phys(job->c_addr, job->stride_c, batch, job->ctype, &c_phys)) {
            return 0xffff0101U;
        }
        a = shared_ddr_ptr(a_phys, a_bytes);
        b = shared_ddr_ptr(b_phys, b_bytes);
        c = shared_ddr_ptr(c_phys, c_bytes);
        if (!a || !b || !c) return 0xffff0101U;

        u32 status = gemm_run_parallel(job, a, b, c, alpha, beta);
        if (status) return status;
    }
    return 0;
}

struct SoftmaxWorker {
    const struct SoftmaxJob *job;
    const void *src;
    void *dst;
    u64 stride;
    u64 row0;
    u64 row1;
    u32 status;
};

static u32 softmax_rows(const struct SoftmaxJob *job,
                        const void *src,
                        void *dst,
                        u64 stride,
                        u64 row0,
                        u64 row1) {
    for (u64 row = row0; row < row1; ++row) {
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
    return 0;
}

static void *softmax_worker_main(void *opaque) {
    struct SoftmaxWorker *worker = (struct SoftmaxWorker *)opaque;
    worker->status = softmax_rows(worker->job, worker->src, worker->dst,
                                  worker->stride, worker->row0, worker->row1);
    return 0;
}

static u32 softmax_run_parallel(const struct SoftmaxJob *job,
                                const void *src,
                                void *dst,
                                u64 stride) {
    unsigned nthreads = runtime_worker_threads(job->rows);
    pthread_t threads[HETGPU_SIFIVE_MAX_WORKER_THREADS];
    struct SoftmaxWorker workers[HETGPU_SIFIVE_MAX_WORKER_THREADS];
    unsigned created = 0;
    u32 status = 0;

    if (nthreads <= 1) {
        return softmax_rows(job, src, dst, stride, 0, job->rows);
    }

    memset(workers, 0, sizeof(workers));
    for (unsigned i = 0; i < nthreads; ++i) {
        u64 row0 = job->rows * (u64)i / (u64)nthreads;
        u64 row1 = job->rows * (u64)(i + 1U) / (u64)nthreads;
        if (row0 >= row1) continue;
        workers[i].job = job;
        workers[i].src = src;
        workers[i].dst = dst;
        workers[i].stride = stride;
        workers[i].row0 = row0;
        workers[i].row1 = row1;
        if (pthread_create(&threads[i], 0, softmax_worker_main, &workers[i]) != 0) {
            status = 0xffff0202U;
            break;
        }
        created++;
    }

    for (unsigned i = 0; i < created; ++i) {
        pthread_join(threads[i], 0);
        if (workers[i].status && !status) {
            status = workers[i].status;
        }
    }
    return status;
}

static u32 softmax_typed(const struct SoftmaxJob *job) {
    const u64 stride = job->stride ? job->stride : job->cols;
    const void *src;
    void *dst;
    u64 last_row;
    u64 elems;
    u64 bytes;

    if (job->rows == 0 || job->cols == 0) return 0;
    if (!checked_mul_u64(job->rows - 1UL, stride, &last_row)) return 0xffff0101U;
    if (!checked_add_u64(last_row, job->cols, &elems)) return 0xffff0101U;
    if (!span_bytes(elems, job->dtype, &bytes)) return 0xffff0101U;
    src = shared_ddr_ptr(job->src_addr, bytes);
    dst = shared_ddr_ptr(job->dst_addr, bytes);
    if (!src || !dst) return 0xffff0101U;

    return softmax_run_parallel(job, src, dst, stride);
}

struct RmsNormWorker {
    const struct RmsNormJob *job;
    const void *x;
    const void *w;
    void *y;
    u64 row0;
    u64 row1;
    u32 status;
};

static u32 rmsnorm_rows(const struct RmsNormJob *job,
                        const void *x,
                        const void *w,
                        void *y,
                        u64 row0,
                        u64 row1) {
    for (u64 row = row0; row < row1; ++row) {
        const u64 base = row * job->hidden;
        float sumsq = 0.0f;
        for (u64 i = 0; i < job->hidden; ++i) {
            float v = load_typed(x, base + i, job->dtype);
            sumsq += v * v;
        }
        float scale = rsqrtf_newton(sumsq / (float)job->hidden + job->eps);
        for (u64 i = 0; i < job->hidden; ++i) {
            float weight = w ? load_typed(w, i, job->dtype) : 1.0f;
            store_typed(y, base + i, job->dtype,
                        load_typed(x, base + i, job->dtype) * scale * weight);
        }
    }
    return 0;
}

static void *rmsnorm_worker_main(void *opaque) {
    struct RmsNormWorker *worker = (struct RmsNormWorker *)opaque;
    worker->status = rmsnorm_rows(worker->job, worker->x, worker->w, worker->y,
                                  worker->row0, worker->row1);
    return 0;
}

static u32 rmsnorm_run_parallel(const struct RmsNormJob *job,
                                const void *x,
                                const void *w,
                                void *y) {
    unsigned nthreads = runtime_worker_threads(job->rows);
    pthread_t threads[HETGPU_SIFIVE_MAX_WORKER_THREADS];
    struct RmsNormWorker workers[HETGPU_SIFIVE_MAX_WORKER_THREADS];
    unsigned created = 0;
    u32 status = 0;

    if (nthreads <= 1) {
        return rmsnorm_rows(job, x, w, y, 0, job->rows);
    }

    memset(workers, 0, sizeof(workers));
    for (unsigned i = 0; i < nthreads; ++i) {
        u64 row0 = job->rows * (u64)i / (u64)nthreads;
        u64 row1 = job->rows * (u64)(i + 1U) / (u64)nthreads;
        if (row0 >= row1) continue;
        workers[i].job = job;
        workers[i].x = x;
        workers[i].w = w;
        workers[i].y = y;
        workers[i].row0 = row0;
        workers[i].row1 = row1;
        if (pthread_create(&threads[i], 0, rmsnorm_worker_main, &workers[i]) != 0) {
            status = 0xffff0302U;
            break;
        }
        created++;
    }

    for (unsigned i = 0; i < created; ++i) {
        pthread_join(threads[i], 0);
        if (workers[i].status && !status) {
            status = workers[i].status;
        }
    }
    return status;
}

static u32 rmsnorm_typed(const struct RmsNormJob *job) {
    const void *x;
    const void *w = 0;
    void *y;
    u64 elems;
    u64 bytes;
    u64 weight_bytes;

    if (job->rows == 0 || job->hidden == 0) return 0;
    if (!checked_mul_u64(job->rows, job->hidden, &elems)) return 0xffff0101U;
    if (!span_bytes(elems, job->dtype, &bytes)) return 0xffff0101U;
    if (!span_bytes(job->hidden, job->dtype, &weight_bytes)) return 0xffff0101U;
    x = shared_ddr_ptr(job->x_addr, bytes);
    y = shared_ddr_ptr(job->y_addr, bytes);
    if (job->weight_addr) {
        w = shared_ddr_ptr(job->weight_addr, weight_bytes);
        if (!w) return 0xffff0101U;
    }
    if (!x || !y) return 0xffff0101U;

    return rmsnorm_run_parallel(job, x, w, y);
}

static volatile struct ArgSlotHeader *arg_slot(volatile struct Doorbell *control, u32 job_id) {
    u64 slot = 0;
    if (job_id == HETGPU_SIFIVE_JOB_GEMM) {
        slot = 0;
    } else if (job_id == HETGPU_SIFIVE_JOB_SOFTMAX) {
        slot = 1;
    } else if (job_id == HETGPU_SIFIVE_JOB_RMSNORM) {
        slot = 2;
    } else if (job_id == HETGPU_SIFIVE_JOB_ALLREDUCE) {
        slot = 3;
    } else {
        slot = 0xff;
    }
    if (slot == 0xff) {
        return (volatile struct ArgSlotHeader *)0;
    }
    return (volatile struct ArgSlotHeader *)((u64)control + HETGPU_SIFIVE_ARG_BASE_OFF +
                                             slot * HETGPU_SIFIVE_ARG_SLOT_BYTES);
}

static void *arg_payload(volatile struct ArgSlotHeader *slot) {
    return (void *)((u64)slot + sizeof(struct ArgSlotHeader));
}

static void mirror_host_status(volatile void *completion_base, u32 job_id, u64 seq, u32 status) {
    volatile struct HostStatus *host =
        (volatile struct HostStatus *)((u64)completion_base + HETGPU_SIFIVE_COMPLETION_OFF);
    host->magic = HETGPU_SIFIVE_JOB_MAGIC;
    host->version = HETGPU_SIFIVE_JOB_VERSION;
    host->job_id = job_id;
    host->status = status;
    host->seq = seq;
    fence_all();
}

static u32 run_job_from_table(volatile struct Doorbell *control, volatile struct Doorbell *doorbell) {
    volatile struct RuntimeJobTable *table =
        (volatile struct RuntimeJobTable *)((u64)control + HETGPU_SIFIVE_RUNTIME_TABLE_OFF);
    if (table->magic != HETGPU_SIFIVE_RUNTIME_TABLE_MAGIC ||
        table->version != HETGPU_SIFIVE_RUNTIME_TABLE_VERSION ||
        table->seq != doorbell->seq) {
        return 0xffffffffU;
    }

    if (doorbell->job_id == HETGPU_SIFIVE_JOB_GEMM) {
        const struct GemmJob *job = (const struct GemmJob *)&table->gemm;
        if (!table->have_gemm) return 0xffff0005U;
        if (!dtype_supported(job->atype) || !dtype_supported(job->btype) || !dtype_supported(job->ctype)) {
            return 0xffff0002U;
        }
        return gemm_typed(job);
    }
    if (doorbell->job_id == HETGPU_SIFIVE_JOB_SOFTMAX) {
        const struct SoftmaxJob *job = (const struct SoftmaxJob *)&table->softmax;
        if (!table->have_softmax) return 0xffff0005U;
        if (!dtype_supported(job->dtype)) return 0xffff0003U;
        return softmax_typed(job);
    }
    if (doorbell->job_id == HETGPU_SIFIVE_JOB_RMSNORM) {
        const struct RmsNormJob *job = (const struct RmsNormJob *)&table->rmsnorm;
        if (!table->have_rmsnorm) return 0xffff0005U;
        if (!dtype_supported(job->dtype)) return 0xffff0004U;
        return rmsnorm_typed(job);
    }
    if (doorbell->job_id == HETGPU_SIFIVE_JOB_ALLREDUCE) {
        return table->have_allreduce ? 0xffff0006U : 0xffff0005U;
    }
    if (doorbell->job_id == HETGPU_SIFIVE_JOB_MMVF) {
        return table->have_mmvf ? 0xffff0007U : 0xffff0005U;
    }
    return 0xffff00ffU;
}

static u32 run_job_from_arg_slot(volatile struct Doorbell *control, volatile struct Doorbell *doorbell) {
    volatile struct ArgSlotHeader *slot = arg_slot(control, doorbell->job_id);
    if (!slot || slot->magic != HETGPU_SIFIVE_JOB_MAGIC ||
        slot->version != HETGPU_SIFIVE_JOB_VERSION ||
        slot->job_id != doorbell->job_id ||
        slot->seq != doorbell->seq) {
        return 0xffff0005U;
    }

    if (doorbell->job_id == HETGPU_SIFIVE_JOB_GEMM) {
        const struct GemmJob *job = (const struct GemmJob *)arg_payload(slot);
        if (!dtype_supported(job->atype) || !dtype_supported(job->btype) || !dtype_supported(job->ctype)) {
            return 0xffff0002U;
        }
        return gemm_typed(job);
    }
    if (doorbell->job_id == HETGPU_SIFIVE_JOB_SOFTMAX) {
        const struct SoftmaxJob *job = (const struct SoftmaxJob *)arg_payload(slot);
        if (!dtype_supported(job->dtype)) return 0xffff0003U;
        return softmax_typed(job);
    }
    if (doorbell->job_id == HETGPU_SIFIVE_JOB_RMSNORM) {
        const struct RmsNormJob *job = (const struct RmsNormJob *)arg_payload(slot);
        if (!dtype_supported(job->dtype)) return 0xffff0004U;
        return rmsnorm_typed(job);
    }
    return 0xffff00ffU;
}

static void run_preloaded_job(volatile struct Doorbell *control, volatile void *completion_base) {
    volatile struct Doorbell *doorbell = control;
    u32 status = 0xffff00ffU;
    doorbell->status = 1;
    mirror_host_status(completion_base, doorbell->job_id, doorbell->seq, 1);
    fence_all();

    if (doorbell->version != HETGPU_SIFIVE_JOB_VERSION) {
        status = 0xffff0001U;
    } else {
        status = run_job_from_table(control, doorbell);
        if (status == 0xffffffffU) {
            status = run_job_from_arg_slot(control, doorbell);
        }
    }
    doorbell->status = status;
    mirror_host_status(completion_base, doorbell->job_id, doorbell->seq, status);
    fence_all();
}

static void hetgpu_sifive_runtime_loop(void) {
    u64 last_seq = 0;
    u64 last_kernel_seq = 0;
    int have_pending_irq = 1;

    for (;;) {
        u64 control_base;
        volatile struct Doorbell *doorbell;
        volatile void *completion_base;
        volatile u64 *heartbeat;
        volatile struct SifiveJobDesc *kernel_desc;
        int did_work = 0;

        if (!have_pending_irq && wait_for_control_irq() != 0) {
            return;
        }
        fence_all();
        have_pending_irq = 0;
        if (refresh_shared_ddr_from_ioctl() != 0) {
            continue;
        }
        if (g_runtime.shared_ddr_size <
            ((u64)g_runtime.sifive_id + 1UL) * HETGPU_SIFIVE_CONTROL_BYTES) {
            fprintf(stderr,
                    "hetgpu_sifive_runtime: shared DDR control window missing sifive_id=%u size=0x%lx\n",
                    g_runtime.sifive_id, (unsigned long)g_runtime.shared_ddr_size);
            continue;
        }

        control_base = (u64)g_runtime.shared_ddr + (u64)g_runtime.sifive_id * HETGPU_SIFIVE_CONTROL_BYTES;
        doorbell = (volatile struct Doorbell *)(control_base + HETGPU_SIFIVE_DOORBELL_OFF);
        completion_base = (volatile void *)control_base;
        heartbeat =
            (volatile u64 *)(control_base + HETGPU_SIFIVE_COMPLETION_OFF + sizeof(struct HostStatus));
        kernel_desc = (volatile struct SifiveJobDesc *)doorbell;
        fence_all();
        if (kernel_desc->buf_info == SIFIVE_JOB_MAGIC &&
            kernel_desc->rsvd != 0 &&
            kernel_desc->rsvd != last_kernel_seq) {
            last_kernel_seq = kernel_desc->rsvd;
            mirror_host_status(completion_base, HETGPU_SIFIVE_JOB_KERNEL, last_kernel_seq, 0xffff0e1fU);
            *heartbeat = last_kernel_seq;
            did_work = 1;
            fence_all();
        }
        if (doorbell->magic == HETGPU_SIFIVE_JOB_MAGIC && doorbell->seq != last_seq) {
            last_seq = doorbell->seq;
            run_preloaded_job(doorbell, completion_base);
            *heartbeat = last_seq;
            did_work = 1;
            fence_all();
        }
        if (did_work) {
            fence_all();
            signal_host_irq();
        }
    }
}

int main(int argc, char **argv) {
    if (init_runtime_io(argc, argv) != 0) {
        return 1;
    }
    hetgpu_sifive_runtime_loop();
    return 0;
}
