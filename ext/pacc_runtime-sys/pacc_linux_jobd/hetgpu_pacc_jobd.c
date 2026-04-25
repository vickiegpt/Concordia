#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<riscv_vector.h>)
#include <riscv_vector.h>
#endif
#endif

#define HETGPU_PACC_JOB_MAGIC 0x4847505550414343ULL
#define HETGPU_PACC_JOB_VERSION 1U
#define PACC_JOB_MAGIC 0x504143434a4f4231ULL
#define PACC_JOB_VERSION 1U
#define PACC_JOB_FLAG_HAS_LAUNCH_ABI (1U << 0)
#define PACC_KERNEL_LAUNCH_ABI_MAGIC 0x5041434341524731ULL
#define PACC_KERNEL_LAUNCH_ABI_VERSION 1U
#define PACC_KERNEL_JOB_ID 0U

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
#define HETGPU_PACC_DEFAULT_SHARED_DDR_BYTES 0x01000000ULL
#define HETGPU_PACC_COMPLETION_OFF 0x1f20ULL
#define PACC_DTYPE_INT8 0U
#define PACC_DTYPE_UINT8 1U
#define PACC_DTYPE_INT32 2U
#define PACC_DTYPE_F32 4U
#define PACC_DTYPE_BF16 5U
#define PACC_GEMM_THREADS 4U
#define PACC_MAX_KERNEL_ARGS 16U
#define PACC_MAX_KERNEL_BINDINGS 16U

#define PACC_IOC_MAGIC 'p'
#define PACC_IOC_ZLUDA_IRQ _IOW(PACC_IOC_MAGIC, 5, struct pacc_zluda_ddr_info)
#define PACC_IOC_ZLUDA_GET_DDR_BASE _IOR(PACC_IOC_MAGIC, 6, struct pacc_zluda_ddr_info)

#define PACC_ELF_ET_REL 1U
#define PACC_ELF_ET_EXEC 2U
#define PACC_ELF_ET_DYN 3U
#define PACC_ELF_SHT_SYMTAB 2U
#define PACC_ELF_SHT_STRTAB 3U
#define PACC_ELF_SHT_DYNSYM 11U
#define PACC_ELF_STT_NOTYPE 0U
#define PACC_ELF_STT_FUNC 2U

#define PACC_FNV64_OFFSET 0xcbf29ce484222325ULL
#define PACC_FNV64_PRIME 0x100000001b3ULL

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

struct pacc_zluda_ddr_info {
    uint64_t ddr_base;
    uint64_t ddr_size;
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

struct PaccJobDesc {
    uint64_t addr;
    uint64_t len;
    uint64_t seq;
    uint64_t buf_info;
};

struct PaccJobImageHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t entry_offset;
    uint64_t image_size;
    uint64_t kernel_name_hash;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t reserved;
};

struct PaccKernelLaunchAbiHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t arg_records_offset;
    uint32_t arg_record_count;
    uint32_t bindings_offset;
    uint32_t binding_count;
    uint32_t raw_param_offset;
    uint32_t raw_param_size;
    uint64_t reserved;
};

struct PaccKernelArgRecord {
    uint32_t kind;
    uint32_t size;
    uint32_t flags;
    uint32_t reserved;
    uint64_t value;
};

struct PaccKernelBufferBinding {
    uint32_t arg_index;
    uint32_t flags;
    uint64_t addr;
    uint64_t size;
};

struct PaccJobImage {
    struct PaccJobImageHeader header;
    const uint8_t *elf;
    size_t elf_len;
    const struct PaccKernelLaunchAbiHeader *abi;
    const struct PaccKernelArgRecord *arg_records;
    size_t arg_count;
    const struct PaccKernelBufferBinding *bindings;
    size_t binding_count;
    const uint8_t *raw_params;
    size_t raw_param_size;
};

struct KernelBindingMap {
    struct Map map;
    uint32_t arg_index;
    uint32_t flags;
};

enum DispatchPollResult {
    DISPATCH_INVALID = 0,
    DISPATCH_IDLE = 1,
    DISPATCH_HANDLED = 2,
};

struct Map {
    void *base;
    size_t map_len;
    void *ptr;
};

static long g_page_size = 4096;
static struct pacc_zluda_ddr_info g_ddr_info;

static void mirror_host_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status);

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

static int control_poll_timeout_ms(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_POLL_TIMEOUT_MS");
    char *end = NULL;
    long parsed;
    if (!value || !*value) {
        return -1;
    }
    errno = 0;
    parsed = strtol(value, &end, 0);
    if (errno || end == value) {
        return 10;
    }
    if (parsed < -1) {
        return 10;
    }
    if (parsed > INT_MAX) {
        return INT_MAX;
    }
    return (int)parsed;
}

static bool parse_u64_checked(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long value;
    if (!s || !*s || !out) {
        return false;
    }
    errno = 0;
    value = strtoull(s, &end, 0);
    if (errno || end == s) {
        return false;
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
        end++;
    }
    if (*end) {
        return false;
    }
    *out = (uint64_t)value;
    return true;
}

static bool read_u64_file(const char *path, uint64_t *out) {
    char buf[64];
    int fd;
    ssize_t n;
    if (!path || !out) {
        return false;
    }
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) {
        return false;
    }
    buf[n] = '\0';
    return parse_u64_checked(buf, out);
}

static bool read_shared_ddr_info_from_env_or_debugfs(struct pacc_zluda_ddr_info *info) {
    uint64_t base = 0;
    uint64_t size = 0;
    if (!info) {
        return false;
    }
    parse_u64_checked(getenv("HETGPU_PACC_SHARED_DDR_BASE"), &base);
    parse_u64_checked(getenv("HETGPU_PACC_SHARED_DDR_BYTES"), &size);
    if (!size) {
        parse_u64_checked(getenv("HETGPU_PACC_SHARED_DDR_SIZE"), &size);
    }
    if (!base) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base", &base);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_bytes", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_size", &size);
    }
    if (!size && base) {
        size = HETGPU_PACC_DEFAULT_SHARED_DDR_BYTES;
    }
    if (!base || size < HETGPU_PACC_CONTROL_BYTES) {
        return false;
    }
    info->ddr_base = base;
    info->ddr_size = size;
    return true;
}

static void wait_for_control(int mbox_fd) {
    for (;;) {
        struct pollfd pfd;
        int ret;
        memset(&pfd, 0, sizeof(pfd));
        pfd.fd = mbox_fd;
        pfd.events = POLLIN;
        ret = poll(&pfd, 1, control_poll_timeout_ms());
        if (ret > 0) {
            if (pfd.revents & (POLLIN | POLLPRI)) {
                return;
            }
            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                log_msg("buggy: poll revents 0x%x", pfd.revents);
                exit(EIO);
            }
            continue;
        }
        if (ret == 0) {
            return;
        }
        if (errno == EINTR || errno == EAGAIN) {
            continue;
        }
        log_msg("buggy: poll return %d", errno);
        exit(errno ? errno : EIO);
    }
}

static void read_shared_ddr_info_from_mbox(int mbox_fd) {
    struct pacc_zluda_ddr_info info;
    int ret;
    memset(&info, 0, sizeof(info));
    if (read_shared_ddr_info_from_env_or_debugfs(&info)) {
        g_ddr_info = info;
        log_msg("shared ddr base 0x%" PRIx64 " size 0x%" PRIx64 " from env/debugfs",
                g_ddr_info.ddr_base, g_ddr_info.ddr_size);
        return;
    }
    ret = ioctl(mbox_fd, PACC_IOC_ZLUDA_GET_DDR_BASE, &info);
    if (ret < 0) {
        log_msg("failed to read shared ddr base/size: %d", errno);
        exit(errno ? errno : EIO);
    }
    if (!info.ddr_base || info.ddr_size < HETGPU_PACC_CONTROL_BYTES) {
        log_msg("invalid shared ddr base 0x%" PRIx64 " size 0x%" PRIx64,
                info.ddr_base, info.ddr_size);
        exit(EINVAL);
    }
    g_ddr_info = info;
    log_msg("shared ddr base 0x%" PRIx64 " size 0x%" PRIx64,
            g_ddr_info.ddr_base, g_ddr_info.ddr_size);
}

static uint64_t shared_ddr_control_phys(uint64_t off, size_t len) {
    if (g_ddr_info.ddr_base &&
        off <= g_ddr_info.ddr_size &&
        (uint64_t)len <= g_ddr_info.ddr_size - off) {
        return g_ddr_info.ddr_base + off;
    }
    return PACC2AP_MBOX_PHYS + off;
}

static uint64_t parse_u64(const char *s) {
    return strtoull(s, NULL, 0);
}

static const char *job_name(uint32_t job_id) {
    switch (job_id) {
    case PACC_KERNEL_JOB_ID:
        return "KERNEL_ELF";
    case HETGPU_PACC_JOB_GEMM:
        return "GEMM";
    case HETGPU_PACC_JOB_SOFTMAX:
        return "SOFTMAX";
    case HETGPU_PACC_JOB_RMSNORM:
        return "RMSNORM";
    case HETGPU_PACC_JOB_ALLREDUCE:
        return "ALLREDUCE";
    default:
        return "UNKNOWN";
    }
}

static uint16_t read_u16_le(const void *ptr) {
    const uint8_t *p = (const uint8_t *)ptr;
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32_le(const void *ptr) {
    const uint8_t *p = (const uint8_t *)ptr;
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t read_u64_le(const void *ptr) {
    const uint8_t *p = (const uint8_t *)ptr;
    return (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

static uint64_t hash_kernel_name_bytes(const char *name) {
    uint64_t hash = PACC_FNV64_OFFSET;
    if (!name) return hash;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        hash ^= (uint64_t)*p;
        hash *= PACC_FNV64_PRIME;
    }
    return hash;
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

static int write_file_all(const char *path, const uint8_t *data, size_t len) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0700);
    size_t written = 0;
    if (fd < 0) return -1;
    while (written < len) {
        ssize_t rc = write(fd, data + written, len - written);
        if (rc < 0) {
            close(fd);
            return -1;
        }
        written += (size_t)rc;
    }
    if (close(fd) != 0) return -1;
    return 0;
}

static const char *find_program_on_path(const char *name) {
    static char resolved[8][512];
    static unsigned next_slot;
    const char *path = getenv("PATH");
    if (!name || !*name) return NULL;
    if (strchr(name, '/')) return access(name, X_OK) == 0 ? name : NULL;
    if (!path || !*path) path = "/usr/bin:/bin:/usr/local/bin";

    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", path);
    char *save = NULL;
    for (char *dir = strtok_r(buf, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
        unsigned slot = next_slot++ % (sizeof(resolved) / sizeof(resolved[0]));
        snprintf(resolved[slot], sizeof(resolved[slot]), "%s/%s", dir, name);
        if (access(resolved[slot], X_OK) == 0) {
            return resolved[slot];
        }
    }
    return NULL;
}

static const char kernel_host_stubs_c[] =
"#include <stdint.h>\n"
"#define WEAK __attribute__((weak))\n"
"struct ShflSyncResult { uint32_t x; uint32_t pred; };\n"
"struct DivF32Part1Result { float fma_4; float fma_1; float fma_3; uint8_t numerator_scaled_flag; };\n"
"static uint32_t lane_u8(uint32_t x, unsigned lane) { return (x >> (lane * 8)) & 0xffu; }\n"
"static int32_t lane_s8(uint32_t x, unsigned lane) { return (int8_t)lane_u8(x, lane); }\n"
"static uint32_t pack_lane_u8(uint32_t base, unsigned lane, uint32_t value) {\n"
"    uint32_t shift = lane * 8;\n"
"    return (base & ~(0xffu << shift)) | ((value & 0xffu) << shift);\n"
"}\n"
"static uint32_t sat_u8(int32_t v) { return v < 0 ? 0u : (v > 255 ? 255u : (uint32_t)v); }\n"
"static int32_t sat_s8(int32_t v) { return v < -128 ? -128 : (v > 127 ? 127 : v); }\n"
"WEAK uint32_t f___zluda_ptx_impl_vsub4_u32_u32_u32(uint32_t a, uint32_t b, uint32_t c) {\n"
"    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, lane_u8(a, i) - lane_u8(b, i)); return r;\n"
"}\n"
"WEAK uint32_t f___zluda_ptx_impl_vsub4_u32_u32_u32_sat(uint32_t a, uint32_t b, uint32_t c) {\n"
"    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, sat_u8((int32_t)lane_u8(a, i) - (int32_t)lane_u8(b, i))); return r;\n"
"}\n"
"WEAK uint32_t f___zluda_ptx_impl_vsub4_s32_s32_s32(uint32_t a, uint32_t b, uint32_t c) {\n"
"    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, (uint8_t)(lane_s8(a, i) - lane_s8(b, i))); return r;\n"
"}\n"
"WEAK uint32_t f___zluda_ptx_impl_vsub4_s32_s32_s32_sat(uint32_t a, uint32_t b, uint32_t c) {\n"
"    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, (uint8_t)sat_s8(lane_s8(a, i) - lane_s8(b, i))); return r;\n"
"}\n"
"static uint32_t vset_cmp(uint32_t a, uint32_t b, int op) {\n"
"    uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) { uint32_t x = lane_u8(a, i), y = lane_u8(b, i); int p = 0;\n"
"    switch (op) { case 0: p = x == y; break; case 1: p = x != y; break; case 2: p = x < y; break; case 3: p = x <= y; break; case 4: p = x > y; break; default: p = x >= y; break; }\n"
"    r = pack_lane_u8(r, i, p ? 1u : 0u); } return r;\n"
"}\n"
"WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_eq(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 0); }\n"
"WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_ne(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 1); }\n"
"WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_lt(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 2); }\n"
"WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_le(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 3); }\n"
"WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_gt(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 4); }\n"
"WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_ge(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 5); }\n"
"WEAK void f___zluda_ptx_impl_bar_sync(uint32_t barrier_id) { (void)barrier_id; __sync_synchronize(); }\n"
"WEAK uint32_t f___zluda_ptx_impl_activemask(void) { return 1u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_tid(uint8_t member) { (void)member; return 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_ntid(uint8_t member) { (void)member; return 1u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_ctaid(uint8_t member) { (void)member; return 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_nctaid(uint8_t member) { (void)member; return 1u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_laneid(void) { return 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_lt(void) { return 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_ge(void) { return ~0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_clock(void) { return 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_bfe_u32(uint32_t base, uint32_t pos_32, uint32_t len_32) {\n"
"    uint32_t pos = pos_32 & 0xffu, len = len_32 & 0xffu; if (pos >= 32u || len == 0u) return 0u; if (len >= 32u) return base >> pos; if (len > 31u) len = 31u; return (base >> pos) & ((1u << len) - 1u);\n"
"}\n"
"WEAK int32_t f___zluda_ptx_impl_bfe_s32(int32_t base, uint32_t pos_32, uint32_t len_32) {\n"
"    uint32_t pos = pos_32 & 0xffu, len = len_32 & 0xffu; if (len == 0u) return 0; if (pos >= 32u) return base >> 31; if (len >= 32u || pos + len >= 32u) return base >> pos; return (base << (32u - pos - len)) >> (32u - len);\n"
"}\n"
"WEAK uint64_t f___zluda_ptx_impl_bfe_u64(uint64_t base, uint32_t pos, uint32_t len) { if (pos >= 64u || len == 0u) return 0u; if (len >= 64u) return base >> pos; return (base >> pos) & ((1ull << len) - 1ull); }\n"
"WEAK int64_t f___zluda_ptx_impl_bfe_s64(int64_t base, uint32_t pos, uint32_t len) { if (len == 0u) return 0; if (pos >= 64u) return base >> 63; if (len >= 64u || pos + len >= 64u) return base >> pos; return (base << (64u - pos - len)) >> (64u - len); }\n"
"WEAK uint32_t f___zluda_ptx_impl_bfi_b32(uint32_t insert, uint32_t base, uint32_t pos_32, uint32_t len_32) { uint32_t pos = pos_32 & 0xffu, len = len_32 & 0xffu; if (pos >= 32u || len == 0u) return base; uint32_t mask = (len >= 32u || pos + len >= 32u) ? (~0u << pos) : (((1u << len) - 1u) << pos); return (base & ~mask) | ((insert << pos) & mask); }\n"
"WEAK uint64_t f___zluda_ptx_impl_bfi_b64(uint64_t insert, uint64_t base, uint32_t pos, uint32_t len) { if (pos >= 64u || len == 0u) return base; uint64_t mask = (len >= 64u || pos + len >= 64u) ? (~0ull << pos) : (((1ull << len) - 1ull) << pos); return (base & ~mask) | ((insert << pos) & mask); }\n"
"WEAK uint32_t f___zluda_ptx_impl_prmt_b32(uint32_t a, uint32_t b, uint32_t c) { uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) { uint32_t sel = (c >> (4 * i)) & 0xfu; uint32_t src = (sel & 4u) ? b : a; uint32_t val = (src >> (8 * (sel & 3u))) & 0xffu; if (sel & 8u) val = (val & 0x80u) ? 0xffu : 0u; r |= val << (8 * i); } return r; }\n"
"WEAK struct DivF32Part1Result f___zluda_ptx_impl_div_f32_part1(float lhs, float rhs) { (void)lhs; (void)rhs; return (struct DivF32Part1Result){ 0.0f, 0.0f, 0.0f, 0u }; }\n"
"WEAK float f___zluda_ptx_impl_div_f32_part2(float x, float y, float fma_4, float fma_1, float fma_3, uint8_t numerator_scaled_flag) { (void)fma_4; (void)fma_1; (void)fma_3; (void)numerator_scaled_flag; return x / y; }\n"
"WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_bfly_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }\n"
"WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_up_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }\n"
"WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_down_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }\n"
"WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_idx_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }\n"
"WEAK uint32_t f___zluda_ptx_impl_shfl_sync_bfly_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }\n"
"WEAK uint32_t f___zluda_ptx_impl_shfl_sync_up_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }\n"
"WEAK uint32_t f___zluda_ptx_impl_shfl_sync_down_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }\n"
"WEAK uint32_t f___zluda_ptx_impl_shfl_sync_idx_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }\n";

static int run_command(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        execvp(argv[0], argv);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return -1;
    }
    return 0;
}

static int compile_kernel_c_object(const char *src, const char *obj) {
    const char *env_cc = getenv("HETGPU_PACC_DEVICE_CC");
    const char *candidates[] = {
        "riscv64-linux-gnu-gcc",
        "gcc",
        "cc",
        "clang",
        NULL,
    };

    if (env_cc && *env_cc) {
        const char *tool = find_program_on_path(env_cc);
        if (tool) {
            char *const argv[] = {
                (char *)tool, (char *)"-O2", (char *)"-fPIC", (char *)"-c",
                (char *)"-o", (char *)obj, (char *)src, NULL,
            };
            if (run_command(argv) == 0) return 0;
        }
    }

    for (size_t i = 0; candidates[i]; i++) {
        const char *tool = find_program_on_path(candidates[i]);
        if (!tool) continue;
        char *const argv[] = {
            (char *)tool, (char *)"-O2", (char *)"-fPIC", (char *)"-c",
            (char *)"-o", (char *)obj, (char *)src, NULL,
        };
        if (run_command(argv) == 0) return 0;
    }

    return -1;
}

static int build_kernel_host_stubs(const char *stub_src, const char *stub_obj) {
    if (write_file_all(stub_src, (const uint8_t *)kernel_host_stubs_c,
                       sizeof(kernel_host_stubs_c) - 1) != 0) {
        return -1;
    }
    return compile_kernel_c_object(stub_src, stub_obj);
}

static bool is_c_symbol_char(char c, bool first) {
    if (c == '_') return true;
    if (c >= 'A' && c <= 'Z') return true;
    if (c >= 'a' && c <= 'z') return true;
    if (!first && c >= '0' && c <= '9') return true;
    return false;
}

static bool is_valid_c_symbol_name(const char *s) {
    if (!s || !*s || !is_c_symbol_char(*s, true)) return false;
    for (const char *p = s + 1; *p; p++) {
        if (!is_c_symbol_char(*p, false)) return false;
    }
    return true;
}

static bool symbol_name_seen(char symbols[][1024], size_t count, const char *name) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(symbols[i], name) == 0) return true;
    }
    return false;
}

static size_t kernel_tmp_shared_stub_bytes(void) {
    const char *env = getenv("HETGPU_PACC_KERNEL_TMP_SHARED_BYTES");
    if (env && *env) {
        char *end = NULL;
        unsigned long long value = strtoull(env, &end, 0);
        if (end != env && value >= 4096ULL && value <= (16ULL << 20)) {
            return (size_t)value;
        }
    }
    return 64UL << 10;
}

static int build_kernel_tmp_shared_stubs(const char *input_obj,
                                         const char *stub_src,
                                         const char *stub_obj) {
    enum { MAX_TMP_SHARED_SYMBOLS = 512 };
    char symbols[MAX_TMP_SHARED_SYMBOLS][1024];
    size_t symbol_count = 0;
    const char *nm_candidates[] = {
        "riscv64-linux-gnu-nm",
        "nm",
        NULL,
    };
    const char *nm_tool = NULL;
    for (size_t i = 0; nm_candidates[i]; i++) {
        nm_tool = find_program_on_path(nm_candidates[i]);
        if (nm_tool) break;
    }

    if (nm_tool) {
        char cmd[PATH_MAX * 2 + 64];
        snprintf(cmd, sizeof(cmd), "%s -u %s", nm_tool, input_obj);
        FILE *pipe = popen(cmd, "r");
        if (pipe) {
            char line[2048];
            while (fgets(line, sizeof(line), pipe)) {
                char *save = NULL;
                char *last = NULL;
                for (char *tok = strtok_r(line, " \t\r\n", &save); tok;
                     tok = strtok_r(NULL, " \t\r\n", &save)) {
                    last = tok;
                }
                if (!last || !strstr(last, "tmp_shared")) continue;
                if (!is_valid_c_symbol_name(last)) continue;
                if (symbol_name_seen(symbols, symbol_count, last)) continue;
                if (symbol_count >= MAX_TMP_SHARED_SYMBOLS) {
                    log_msg("device-link: too many tmp_shared symbols, truncating at %u",
                            MAX_TMP_SHARED_SYMBOLS);
                    break;
                }
                snprintf(symbols[symbol_count], sizeof(symbols[symbol_count]), "%s", last);
                symbol_count++;
            }
            int status = pclose(pipe);
            if (status != 0) {
                log_msg("device-link: nm returned non-zero while scanning tmp_shared stubs");
            }
        }
    }

    FILE *out = fopen(stub_src, "w");
    if (!out) return -1;
    fprintf(out, "#include <stdint.h>\n");
    fprintf(out, "__attribute__((used)) static unsigned char hetgpu_pacc_tmp_shared_anchor;\n");
    size_t bytes = kernel_tmp_shared_stub_bytes();
    for (size_t i = 0; i < symbol_count; i++) {
        fprintf(out,
                "__attribute__((weak, aligned(16))) unsigned char %s[%zu];\n",
                symbols[i], bytes);
    }
    if (fclose(out) != 0) return -1;

    if (symbol_count > 0) {
        log_msg("device-link: adding %zu tmp_shared BSS stubs (%zu bytes each)",
                symbol_count, bytes);
    }
    return compile_kernel_c_object(stub_src, stub_obj);
}

static int device_link_kernel_object(const char *input_obj, const char *output_so) {
    const char *env_linker = getenv("HETGPU_PACC_DEVICE_LINKER");
    char stub_src[PATH_MAX];
    char stub_obj[PATH_MAX];
    char tmp_shared_src[PATH_MAX];
    char tmp_shared_obj[PATH_MAX];
    const char *candidates[] = {
        "riscv64-linux-gnu-gcc",
        "gcc",
        "cc",
        "clang",
        NULL,
    };

    snprintf(stub_src, sizeof(stub_src), "%s.host_stubs.c", output_so);
    snprintf(stub_obj, sizeof(stub_obj), "%s.host_stubs.o", output_so);
    if (build_kernel_host_stubs(stub_src, stub_obj) != 0) {
        log_msg("device-link failed to build host PTX helper stubs");
        return -1;
    }
    snprintf(tmp_shared_src, sizeof(tmp_shared_src), "%s.tmp_shared_stubs.c", output_so);
    snprintf(tmp_shared_obj, sizeof(tmp_shared_obj), "%s.tmp_shared_stubs.o", output_so);
    if (build_kernel_tmp_shared_stubs(input_obj, tmp_shared_src, tmp_shared_obj) != 0) {
        log_msg("device-link failed to build tmp_shared data stubs");
        return -1;
    }

    if (env_linker && *env_linker) {
        const char *tool = find_program_on_path(env_linker);
        if (tool) {
            char *const env_argv[] = {
                (char *)tool,
                (char *)"-fuse-ld=bfd",
                (char *)"-shared",
                (char *)"-fPIC",
                (char *)"-o",
                (char *)output_so,
                (char *)input_obj,
                (char *)stub_obj,
                (char *)tmp_shared_obj,
                (char *)"-lm",
                (char *)"-ldl",
                NULL,
            };
            if (run_command(env_argv) == 0) {
                log_msg("device-link ok: %s -> %s via %s", input_obj, output_so, tool);
                return 0;
            }
        }
    }

    for (size_t i = 0; candidates[i]; i++) {
        const char *tool = find_program_on_path(candidates[i]);
        if (!tool) continue;

        char *const cc_argv[] = {
            (char *)tool,
            (char *)"-fuse-ld=bfd",
            (char *)"-shared",
            (char *)"-fPIC",
            (char *)"-o",
            (char *)output_so,
            (char *)input_obj,
            (char *)stub_obj,
            (char *)tmp_shared_obj,
            (char *)"-lm",
            (char *)"-ldl",
            NULL,
        };
        if (run_command(cc_argv) == 0) {
            log_msg("device-link ok: %s -> %s via %s", input_obj, output_so, tool);
            return 0;
        }
    }

    return -1;
}

static bool elf64_bounds_ok(size_t off, size_t size, size_t total) {
    return off <= total && size <= total - off;
}

static bool elf64_locate_symbol_by_hash(const uint8_t *elf, size_t elf_len,
                                        uint64_t want_hash,
                                        char *name_out, size_t name_out_len) {
    if (!elf || elf_len < 64 || !name_out || name_out_len == 0) return false;
    if (!(elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F')) return false;
    if (elf[4] != 2 || elf[5] != 1) return false;

    size_t shoff = (size_t)read_u64_le(elf + 0x28);
    uint16_t shentsize = read_u16_le(elf + 0x3a);
    uint16_t shnum = read_u16_le(elf + 0x3c);
    if (!shoff || !shentsize || !shnum) return false;
    if (!elf64_bounds_ok(shoff, (size_t)shentsize * shnum, elf_len)) return false;

    const char *fallback = NULL;
    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t *sh = elf + shoff + (size_t)i * shentsize;
        uint32_t shtype = read_u32_le(sh + 0x04);
        if (shtype != PACC_ELF_SHT_SYMTAB && shtype != PACC_ELF_SHT_DYNSYM) continue;

        size_t sym_off = (size_t)read_u64_le(sh + 0x18);
        size_t sym_size = (size_t)read_u64_le(sh + 0x20);
        size_t sym_entsize = (size_t)read_u64_le(sh + 0x38);
        uint32_t strtab_index = read_u32_le(sh + 0x28);
        if (sym_entsize < 24 || strtab_index >= shnum) continue;
        if (!elf64_bounds_ok(sym_off, sym_size, elf_len)) continue;

        const uint8_t *str_sh = elf + shoff + (size_t)strtab_index * shentsize;
        if (read_u32_le(str_sh + 0x04) != PACC_ELF_SHT_STRTAB) continue;
        size_t str_off = (size_t)read_u64_le(str_sh + 0x18);
        size_t str_size = (size_t)read_u64_le(str_sh + 0x20);
        if (!elf64_bounds_ok(str_off, str_size, elf_len)) continue;
        const char *strtab = (const char *)(elf + str_off);

        size_t sym_count = sym_size / sym_entsize;
        for (size_t sym_idx = 0; sym_idx < sym_count; sym_idx++) {
            const uint8_t *sym = elf + sym_off + sym_idx * sym_entsize;
            uint32_t st_name = read_u32_le(sym + 0x00);
            unsigned st_type = sym[4] & 0x0f;
            uint16_t st_shndx = read_u16_le(sym + 0x06);
            if (st_name >= str_size || st_shndx == 0) continue;
            const char *name = strtab + st_name;
            if (!*name) continue;
            if (st_type != PACC_ELF_STT_FUNC && st_type != PACC_ELF_STT_NOTYPE) continue;
            if (!fallback) fallback = name;
            if (hash_kernel_name_bytes(name) == want_hash) {
                snprintf(name_out, name_out_len, "%s", name);
                return true;
            }
        }
    }

    if (fallback) {
        snprintf(name_out, name_out_len, "%s", fallback);
        return true;
    }
    return false;
}

static uint16_t elf64_type(const uint8_t *elf, size_t elf_len) {
    if (!elf || elf_len < 0x12 || !(elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F')) {
        return 0;
    }
    return read_u16_le(elf + 0x10);
}

static int load_kernel_image(const uint8_t *elf, size_t elf_len,
                             uint64_t kernel_hash,
                             char *symbol_name, size_t symbol_name_len,
                             char *artifact_path, size_t artifact_path_len,
                             void **handle_out) {
    char tmpdir[] = "/tmp/hetgpu_pacc_kernelXXXXXX";
    char obj_path[PATH_MAX];
    char so_path[PATH_MAX];
    uint16_t e_type;
    const char *load_path = NULL;

    if (!elf || !elf_len || !handle_out) return -1;
    *handle_out = NULL;
    if (!mkdtemp(tmpdir)) {
        return -1;
    }

    e_type = elf64_type(elf, elf_len);
    if (!elf64_locate_symbol_by_hash(elf, elf_len, kernel_hash, symbol_name, symbol_name_len)) {
        log_msg("kernel image: no symbol matched hash=0x%" PRIx64, kernel_hash);
        return -1;
    }

    if (e_type == PACC_ELF_ET_REL) {
        snprintf(obj_path, sizeof(obj_path), "%s/kernel.o", tmpdir);
        snprintf(so_path, sizeof(so_path), "%s/kernel.so", tmpdir);
        if (write_file_all(obj_path, elf, elf_len) != 0) {
            return -1;
        }
        if (device_link_kernel_object(obj_path, so_path) != 0) {
            log_msg("device-link failed for ET_REL kernel object");
            return -1;
        }
        load_path = so_path;
        snprintf(artifact_path, artifact_path_len, "%s", so_path);
    } else if (e_type == PACC_ELF_ET_DYN) {
        snprintf(so_path, sizeof(so_path), "%s/kernel.so", tmpdir);
        if (write_file_all(so_path, elf, elf_len) != 0) {
            return -1;
        }
        load_path = so_path;
        snprintf(artifact_path, artifact_path_len, "%s", so_path);
    } else {
        log_msg("unsupported kernel ELF type=%u", (unsigned)e_type);
        return -1;
    }

    *handle_out = dlopen(load_path, RTLD_NOW | RTLD_LOCAL);
    if (!*handle_out) {
        log_msg("dlopen(%s) failed: %s", load_path, dlerror());
        return -1;
    }
    return 0;
}

static int invoke_kernel_symbol(void *fn, const uint64_t *args, size_t argc) {
    switch (argc) {
    case 0: ((void (*)(void))fn)(); return 0;
    case 1: ((void (*)(uint64_t))fn)(args[0]); return 0;
    case 2: ((void (*)(uint64_t,uint64_t))fn)(args[0], args[1]); return 0;
    case 3: ((void (*)(uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2]); return 0;
    case 4: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3]); return 0;
    case 5: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4]); return 0;
    case 6: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5]); return 0;
    case 7: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6]); return 0;
    case 8: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]); return 0;
    case 9: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]); return 0;
    case 10: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]); return 0;
    case 11: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]); return 0;
    case 12: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]); return 0;
    default:
        return -1;
    }
}

static void release_kernel_binding_maps(struct KernelBindingMap *maps, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (maps[i].map.base) {
            msync(maps[i].map.base, maps[i].map.map_len, MS_SYNC);
            unmap_phys(&maps[i].map);
        }
    }
}

static struct KernelBindingMap *find_binding_map(struct KernelBindingMap *maps, size_t count, uint32_t arg_index) {
    for (size_t i = 0; i < count; i++) {
        if (maps[i].arg_index == arg_index) return &maps[i];
    }
    return NULL;
}

static int build_kernel_launch_args(
    int fd,
    const struct PaccJobImage *job,
    uint64_t *argv_out,
    size_t *argc_out,
    struct KernelBindingMap *maps,
    size_t *map_count_out) {
    size_t argc = 0;
    size_t map_count = 0;
    size_t default_bind_bytes = (size_t)g_page_size;

    if (!job || !argv_out || !argc_out || !maps || !map_count_out) return -1;
    if (job->arg_count > PACC_MAX_KERNEL_ARGS || job->binding_count > PACC_MAX_KERNEL_BINDINGS) return -1;

    for (size_t i = 0; i < job->binding_count; i++) {
        const struct PaccKernelBufferBinding *binding = &job->bindings[i];
        size_t bind_bytes = binding->size ? (size_t)binding->size : default_bind_bytes;
        if (map_count >= PACC_MAX_KERNEL_BINDINGS) return -1;
        if (binding->addr == 0) continue;
        if (map_phys(fd, binding->addr, bind_bytes, &maps[map_count].map) != 0) {
            log_msg("kernel binding map failed: arg=%u phys=0x%" PRIx64 " size=%zu",
                    binding->arg_index, binding->addr, bind_bytes);
            release_kernel_binding_maps(maps, map_count);
            return -1;
        }
        maps[map_count].arg_index = binding->arg_index;
        maps[map_count].flags = binding->flags;
        map_count++;
    }

    if (job->arg_count) {
        for (size_t i = 0; i < job->arg_count; i++) {
            const struct PaccKernelArgRecord *rec = &job->arg_records[i];
            uint64_t value = rec->value;
            if (rec->kind == 1U) {
                struct KernelBindingMap *binding = find_binding_map(maps, map_count, (uint32_t)i);
                if (binding) {
                    value = (uint64_t)(uintptr_t)binding->map.ptr;
                }
            }
            argv_out[argc++] = value;
        }
    } else if (job->raw_params && job->raw_param_size) {
        size_t words = job->raw_param_size / sizeof(uint64_t);
        if (words > PACC_MAX_KERNEL_ARGS) {
            release_kernel_binding_maps(maps, map_count);
            return -1;
        }
        for (size_t i = 0; i < words; i++) {
            argv_out[argc++] = read_u64_le(job->raw_params + i * sizeof(uint64_t));
        }
    }

    *argc_out = argc;
    *map_count_out = map_count;
    return 0;
}

static int parse_kernel_job_image(const uint8_t *image, size_t image_len, struct PaccJobImage *out) {
    size_t abi_len = sizeof(struct PaccKernelLaunchAbiHeader);

    if (!image || !out || image_len < sizeof(struct PaccJobImageHeader)) return -1;
    memset(out, 0, sizeof(*out));
    memcpy(&out->header, image, sizeof(out->header));
    if (out->header.magic != PACC_JOB_MAGIC || out->header.version != PACC_JOB_VERSION) return -1;
    if (out->header.entry_offset > image_len || out->header.image_size > image_len - out->header.entry_offset) return -1;
    out->elf = image + out->header.entry_offset;
    out->elf_len = (size_t)out->header.image_size;

    if (out->header.flags & PACC_JOB_FLAG_HAS_LAUNCH_ABI) {
        if (image_len < sizeof(struct PaccJobImageHeader) + abi_len) return -1;
        out->abi = (const struct PaccKernelLaunchAbiHeader *)(image + sizeof(struct PaccJobImageHeader));
        if (out->abi->magic != PACC_KERNEL_LAUNCH_ABI_MAGIC ||
            out->abi->version != PACC_KERNEL_LAUNCH_ABI_VERSION) {
            return -1;
        }
        if (out->abi->arg_record_count) {
            size_t bytes = (size_t)out->abi->arg_record_count * sizeof(struct PaccKernelArgRecord);
            if (!elf64_bounds_ok(out->abi->arg_records_offset, bytes, image_len)) return -1;
            out->arg_records = (const struct PaccKernelArgRecord *)(image + out->abi->arg_records_offset);
            out->arg_count = out->abi->arg_record_count;
        }
        if (out->abi->binding_count) {
            size_t bytes = (size_t)out->abi->binding_count * sizeof(struct PaccKernelBufferBinding);
            if (!elf64_bounds_ok(out->abi->bindings_offset, bytes, image_len)) return -1;
            out->bindings = (const struct PaccKernelBufferBinding *)(image + out->abi->bindings_offset);
            out->binding_count = out->abi->binding_count;
        }
        if (out->abi->raw_param_size) {
            if (!elf64_bounds_ok(out->abi->raw_param_offset, out->abi->raw_param_size, image_len)) return -1;
            out->raw_params = image + out->abi->raw_param_offset;
            out->raw_param_size = out->abi->raw_param_size;
        }
    }
    return 0;
}

static int dispatch_kernel_job(int fd, const struct PaccJobDesc *desc) {
    struct Map map = {0};
    struct PaccJobImage job;
    uint64_t argv[PACC_MAX_KERNEL_ARGS];
    size_t argc = 0;
    struct KernelBindingMap binding_maps[PACC_MAX_KERNEL_BINDINGS];
    size_t binding_map_count = 0;
    char symbol[256] = {0};
    char artifact[PATH_MAX] = {0};
    void *handle = NULL;
    void *fn = NULL;
    int status = 0;

    memset(binding_maps, 0, sizeof(binding_maps));
    if (!desc || desc->buf_info != PACC_JOB_MAGIC || desc->len < sizeof(struct PaccJobImageHeader)) {
        return 0xffff5001;
    }
    if (map_phys(fd, desc->addr, (size_t)desc->len, &map) != 0) {
        return 0xffff5002;
    }
    if (parse_kernel_job_image((const uint8_t *)map.ptr, (size_t)desc->len, &job) != 0) {
        unmap_phys(&map);
        return 0xffff5003;
    }

    log_msg("kernel job seq=%" PRIu64 " hash=0x%" PRIx64 " elf=%zu args=%zu bindings=%zu grid=%ux%ux%u block=%ux%ux%u",
            desc->seq, job.header.kernel_name_hash, job.elf_len, job.arg_count, job.binding_count,
            job.header.grid_x, job.header.grid_y, job.header.grid_z,
            job.header.block_x, job.header.block_y, job.header.block_z);

    if (load_kernel_image(job.elf, job.elf_len, job.header.kernel_name_hash,
                          symbol, sizeof(symbol), artifact, sizeof(artifact), &handle) != 0) {
        unmap_phys(&map);
        return 0xffff5004;
    }

    if (build_kernel_launch_args(fd, &job, argv, &argc, binding_maps, &binding_map_count) != 0) {
        dlclose(handle);
        unmap_phys(&map);
        return 0xffff5005;
    }

    fn = dlsym(handle, symbol);
    if (!fn) {
        log_msg("dlsym(%s from %s) failed: %s", symbol, artifact, dlerror());
        release_kernel_binding_maps(binding_maps, binding_map_count);
        dlclose(handle);
        unmap_phys(&map);
        return 0xffff5006;
    }

    log_msg("kernel dispatch: seq=%" PRIu64 " symbol=%s argc=%zu artifact=%s",
            desc->seq, symbol, argc, artifact);
    status = invoke_kernel_symbol(fn, argv, argc);

    release_kernel_binding_maps(binding_maps, binding_map_count);
    dlclose(handle);
    unmap_phys(&map);
    if (status != 0) {
        return 0xffff5007;
    }
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
        log_msg("arg_payload mismatch: job_id=%u/%s seq=%" PRIu64 " slot=%d magic=0x%" PRIx64 " ver=%u hdr_job=%u hdr_seq=%" PRIu64 " arg_len=%" PRIu64 " want=%zu",
                job_id, job_name(job_id), seq, slot, h->magic, h->version,
                h->job_id, h->seq, h->arg_len, want);
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
    if (local.have_gemm) {
        log_msg("runtime table GEMM: m=%" PRIu64 " n=%" PRIu64 " k=%" PRIu64 " atype=%u btype=%u ctype=%u a=0x%" PRIx64 " b=0x%" PRIx64 " c=0x%" PRIx64,
                local.gemm.m, local.gemm.n, local.gemm.k,
                local.gemm.atype, local.gemm.btype, local.gemm.ctype,
                local.gemm.a_addr, local.gemm.b_addr, local.gemm.c_addr);
    }
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
        if (job) {
            log_msg("dispatch GEMM: seq=%" PRIu64 " m=%" PRIu64 " n=%" PRIu64 " k=%" PRIu64 " atype=%u btype=%u ctype=%u",
                    seq, job->m, job->n, job->k, job->atype, job->btype, job->ctype);
        }
        return job ? run_gemm(fd, job) : (int)0xffff0101U;
    }
    if (job_id == HETGPU_PACC_JOB_SOFTMAX) {
        const struct SoftmaxJob *job = jobs->have_softmax ? &jobs->softmax : NULL;
        if (!strict) {
            const struct SoftmaxJob *dyn = arg_payload(ctl, job_id, seq, sizeof(struct SoftmaxJob));
            if (dyn) job = dyn;
        }
        if (job) {
            log_msg("dispatch SOFTMAX: seq=%" PRIu64 " rows=%" PRIu64 " cols=%" PRIu64 " stride=%" PRIu64 " dtype=%u",
                    seq, job->rows, job->cols, job->stride, job->dtype);
        }
        return job ? run_softmax(fd, job) : (int)0xffff0201U;
    }
    if (job_id == HETGPU_PACC_JOB_RMSNORM) {
        const struct RmsNormJob *job = jobs->have_rmsnorm ? &jobs->rmsnorm : NULL;
        if (!strict) {
            const struct RmsNormJob *dyn = arg_payload(ctl, job_id, seq, sizeof(struct RmsNormJob));
            if (dyn) job = dyn;
        }
        if (job) {
            log_msg("dispatch RMSNORM: seq=%" PRIu64 " rows=%" PRIu64 " hidden=%" PRIu64 " dtype=%u eps=%g",
                    seq, job->rows, job->hidden, job->dtype, job->eps);
        }
        return job ? run_rmsnorm(fd, job) : (int)0xffff0301U;
    }
    if (job_id == HETGPU_PACC_JOB_ALLREDUCE) {
        const struct AllReduceJob *job = jobs->have_allreduce ? &jobs->allreduce : NULL;
        if (!strict) {
            const struct AllReduceJob *dyn = arg_payload(ctl, job_id, seq, sizeof(struct AllReduceJob));
            if (dyn) job = dyn;
        }
        if (job) {
            log_msg("dispatch ALLREDUCE: seq=%" PRIu64 " count=%" PRIu64 " nranks=%u dtype=%u op=%u",
                    seq, job->count, job->nranks, job->dtype, job->reduce_op);
        }
        return job ? run_allreduce(fd, job) : (int)0xffff0401U;
    }
    return 0xffff00ff;
}

static enum DispatchPollResult maybe_dispatch_kernel_job(
    int fd,
    volatile struct Doorbell *ctl,
    uint64_t *last_kernel_seq) {
    const struct PaccJobDesc *kernel_desc = (const struct PaccJobDesc *)(const void *)ctl;
    int status;

    if (!kernel_desc || kernel_desc->buf_info != PACC_JOB_MAGIC ||
        kernel_desc->len < sizeof(struct PaccJobImageHeader)) {
        return DISPATCH_INVALID;
    }
    if (kernel_desc->seq == 0 || kernel_desc->seq == *last_kernel_seq) {
        return DISPATCH_IDLE;
    }

    *last_kernel_seq = kernel_desc->seq;
    log_msg("new kernel doorbell: seq=%" PRIu64 " addr=0x%" PRIx64 " len=%" PRIu64,
            kernel_desc->seq, kernel_desc->addr, kernel_desc->len);
    status = dispatch_kernel_job(fd, kernel_desc);
    mirror_host_status(fd, PACC_KERNEL_JOB_ID, kernel_desc->seq, (uint32_t)status);
    log_msg("kernel dispatch done: seq=%" PRIu64 " status=0x%x",
            kernel_desc->seq, (uint32_t)status);
    return DISPATCH_HANDLED;
}

static enum DispatchPollResult maybe_dispatch_preloaded_job(
    int fd,
    volatile struct Doorbell *ctl,
    struct PreloadedJobs *jobs,
    bool strict,
    uint64_t *last_seq,
    uint64_t *last_table_seq) {
    uint32_t job_id;
    int status;

    if (ctl->magic != HETGPU_PACC_JOB_MAGIC || ctl->version != HETGPU_PACC_JOB_VERSION) {
        return DISPATCH_INVALID;
    }
    if (ctl->seq == *last_seq) {
        return DISPATCH_IDLE;
    }

    *last_seq = ctl->seq;
    job_id = ctl->job_id;
    ctl->status = 1;
    __sync_synchronize();
    log_msg("new doorbell: job_id=%u/%s seq=%" PRIu64,
            job_id, job_name(job_id), *last_seq);
    refresh_runtime_table(ctl, jobs, last_table_seq);
    log_msg("dispatch enter: job_id=%u/%s seq=%" PRIu64,
            job_id, job_name(job_id), *last_seq);
    status = dispatch_job(fd, ctl, jobs, strict);
    __sync_synchronize();
    ctl->status = (uint32_t)status;
    mirror_host_status(fd, job_id, *last_seq, (uint32_t)status);
    log_msg("dispatch done: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
            job_id, job_name(job_id), *last_seq, (uint32_t)status);
    return DISPATCH_HANDLED;
}

static enum DispatchPollResult dispatch_any_job(
    int fd,
    volatile struct Doorbell *ctl,
    struct PreloadedJobs *jobs,
    bool strict,
    uint64_t *last_seq,
    uint64_t *last_table_seq,
    uint64_t *last_kernel_seq) {
    enum DispatchPollResult result;

    result = maybe_dispatch_preloaded_job(fd, ctl, jobs, strict, last_seq, last_table_seq);
    if (result != DISPATCH_INVALID) {
        return result;
    }

    result = maybe_dispatch_kernel_job(fd, ctl, last_kernel_seq);
    if (result != DISPATCH_INVALID) {
        return result;
    }

    return DISPATCH_INVALID;
}

static volatile struct Doorbell *scan_for_control(int fd, const struct pacc_zluda_ddr_info *ddr_info, struct Map *map) {
    if (!ddr_info || !ddr_info->ddr_base || ddr_info->ddr_size < HETGPU_PACC_CONTROL_BYTES) {
        log_msg("invalid shared ddr control window: base=0x%" PRIx64 " size=0x%" PRIx64,
                ddr_info ? ddr_info->ddr_base : 0, ddr_info ? ddr_info->ddr_size : 0);
        return NULL;
    }
    if (map_phys(fd, ddr_info->ddr_base, HETGPU_PACC_CONTROL_BYTES, map)) {
        log_msg("map shared DDR control 0x%" PRIx64 " len 0x%x failed: %s",
                ddr_info->ddr_base, (unsigned)HETGPU_PACC_CONTROL_BYTES, strerror(errno));
        return NULL;
    }
    log_msg("mapped shared DDR control at phys 0x%" PRIx64 " len 0x%x",
            ddr_info->ddr_base, (unsigned)HETGPU_PACC_CONTROL_BYTES);
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
    uint64_t phys = shared_ddr_control_phys(HETGPU_PACC_COMPLETION_OFF, sizeof(struct HostStatus));
    if (map_phys(fd, phys, sizeof(struct HostStatus), &map)) {
        log_msg("map host status 0x%" PRIx64 " failed: %s", phys, strerror(errno));
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
    log_msg("mirror_host_status: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
            job_id, job_name(job_id), seq, status);
    unmap_phys(&map);
}

int main(int argc, char **argv) {
    const char *devmem = "/dev/mem";
    const char *mbox_path = "/dev/pacc0";
    const char *config = "/etc/hetgpu_pacc_jobs.conf";
    bool strict = false;

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
        } else if ((!strcmp(argv[i], "--mbox") || !strcmp(argv[i], "--pacc-dev")) && i + 1 < argc) {
            mbox_path = argv[++i];
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
    int mbox_fd = open(mbox_path, O_RDWR | O_SYNC | O_CLOEXEC);
    if (mbox_fd < 0) {
        log_msg("open %s failed: %s", mbox_path, strerror(errno));
        close(fd);
        return 1;
    }

    log_msg("started strict=%d config=%s mbox=%s",
            strict ? 1 : 0, config, mbox_path);
    wait_for_control(mbox_fd);
    read_shared_ddr_info_from_mbox(mbox_fd);
    mirror_host_status(fd, 0, 0, 0x600d);
    struct Map control_map = {0};
    volatile struct Doorbell *ctl = NULL;
    uint64_t last_seq = 0;
    uint64_t last_table_seq = 0;
    uint64_t last_kernel_seq = 0;
    ctl = scan_for_control(fd, &g_ddr_info, &control_map);
    if (!ctl) {
        close(mbox_fd);
        close(fd);
        return 1;
    }
    for (;;) {
        enum DispatchPollResult poll_result;
        poll_result = dispatch_any_job(
            fd,
            ctl,
            &jobs,
            strict,
            &last_seq,
            &last_table_seq,
            &last_kernel_seq);
        if (poll_result == DISPATCH_HANDLED) {
            struct pacc_zluda_ddr_info irq_info = g_ddr_info;
            int ret = ioctl(mbox_fd, PACC_IOC_ZLUDA_IRQ, &irq_info);
            if (ret < 0) {
                log_msg("failed to response: %d", errno);
                return errno ? errno : EIO;
            }
        }
        wait_for_control(mbox_fd);
    }
}
