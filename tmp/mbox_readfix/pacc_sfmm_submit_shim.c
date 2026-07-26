#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

enum {
    PACC_DTYPE_F32 = 4,
    PACC_DTYPE_BF16 = 5,
    PACC_JOB_GEMM = 1,
};

static const uint64_t HETGPU_PACC_JOB_MAGIC = 0x4847505550414343ULL;
static const uint64_t HETGPU_PACC_RUNTIME_TABLE_MAGIC = 0x4847505554424c31ULL;
static const uint32_t HETGPU_PACC_JOB_VERSION = 1;
static const uint32_t HETGPU_PACC_RUNTIME_TABLE_VERSION = 1;
static const uint64_t HETGPU_PACC_SHARED_DDR_HELPER_OFF = 0x00100000ULL;
static const uint64_t HETGPU_PACC_SHARED_DDR_CONTROL_BASE_OFF = 0x00100000ULL;
static const uint64_t HETGPU_PACC_CONTROL_BYTES = 0x2000ULL;
static const uint64_t HETGPU_PACC_SHARED_DDR_PACC_ALIAS_BASE = 0x80000000ULL;
static const uint64_t HETGPU_PACC_DOORBELL_OFF = 0x0ULL;
static const uint64_t HETGPU_PACC_ARG_BASE_OFF = 0x100ULL;
static const uint64_t HETGPU_PACC_ARG_SLOT_BYTES = 0x400ULL;
static const uint64_t HETGPU_PACC_RUNTIME_TABLE_OFF = 0x1400ULL;
static const uint64_t HETGPU_PACC_COMPLETION_OFF = 0x1f20ULL;
static const uint64_t HETGPU_PACC_BEACON_OFF = 0x1f40ULL;
static const uint64_t HETGPU_PACC_COMPLETION_TELEMETRY_OFF = 0x1f80ULL;
static const uint64_t HETGPU_PACC_COMPLETION_TELEMETRY_MAGIC =
    0x48475055544c4d31ULL;
static const uint32_t HETGPU_PACC_COMPLETION_TELEMETRY_VERSION = 1;
static const uint64_t HETGPU_PACC_AP2PACC_READBACK_HELPER_OFF = 0x02000000ULL;
static const uint64_t HETGPU_PACC_PACC2AP_RW_HELPER_OFF = 0x02002000ULL;
static const uint64_t HETGPU_PACC_TOP_MBOX_RAM_OFF = 0x00010000ULL;
static const uint64_t HETGPU_PACC_TOP_MBOX_DB_OFF = 0x00014000ULL;
#define PACC_IOC_MAGIC 'p'
#define PACC_IOC_ZLUDA_IRQ _IO(PACC_IOC_MAGIC, 5)

struct pacc_control_write {
    uint64_t pacc_id;
    uint64_t off;
    uint64_t len;
    uint64_t user_ptr;
    uint64_t flags;
};

#define PACC_IOC_CONTROL_WRITE _IOW(PACC_IOC_MAGIC, 9, struct pacc_control_write)

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint32_t flags;
    uint32_t status;
    uint64_t seq;
} HetgpuPaccDoorbell;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t record_bytes;
    uint32_t job_id;
    uint32_t status;
    uint32_t flags;
    uint32_t reserved;
    uint64_t seq;
    uint64_t compute_start_ns;
    uint64_t compute_end_ns;
    uint64_t publish_start_ns;
    uint64_t publish_end_ns;
    uint64_t xsfmm_cycles;
    uint64_t xsfmm_repeats;
} HetgpuPaccCompletionTelemetry;

_Static_assert(sizeof(HetgpuPaccCompletionTelemetry) == 88,
               "completion telemetry ABI must remain 88 bytes");

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint64_t seq;
    uint64_t arg_len;
} HetgpuPaccArgSlotHeader;

typedef struct {
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
} HetgpuPaccGemmJob;

typedef struct {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t rows;
    uint64_t cols;
    uint64_t stride;
    uint32_t dtype;
    uint32_t reserved;
} HetgpuPaccSoftmaxJob;

typedef struct {
    uint64_t x_addr;
    uint64_t weight_addr;
    uint64_t y_addr;
    uint64_t rows;
    uint64_t hidden;
    float eps;
    uint32_t dtype;
} HetgpuPaccRmsNormJob;

typedef struct {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t count;
    uint32_t nranks;
    uint32_t reduce_op;
    uint32_t dtype;
    uint32_t reserved;
} HetgpuPaccAllReduceJob;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} HetgpuPaccUint3;

typedef struct {
    uint64_t x_addr;
    uint64_t y_addr;
    uint64_t ids_addr;
    uint64_t dst_addr;
    uint64_t x_bytes;
    uint64_t y_bytes;
    uint64_t dst_bytes;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t ncols_dst;
    uint32_t x_type;
    uint32_t reserved0;
    int32_t ncols2;
    HetgpuPaccUint3 nchannels_y;
    int32_t stride_row;
    int32_t stride_col_y2;
    int32_t stride_col_dst;
    HetgpuPaccUint3 channel_ratio;
    int32_t stride_channel_x;
    int32_t stride_channel_y;
    int32_t stride_channel_dst;
    HetgpuPaccUint3 sample_ratio;
    int32_t stride_sample_x;
    int32_t stride_sample_y;
    int32_t stride_sample_dst;
    int32_t ids_stride;
} HetgpuPaccMmvfJob;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t seq;
    uint32_t have_gemm;
    uint32_t have_softmax;
    uint32_t have_rmsnorm;
    uint32_t have_allreduce;
    uint32_t have_mmvf;
    uint32_t reserved0;
    HetgpuPaccGemmJob gemm;
    HetgpuPaccSoftmaxJob softmax;
    HetgpuPaccRmsNormJob rmsnorm;
    HetgpuPaccAllReduceJob allreduce;
    HetgpuPaccMmvfJob mmvf;
} HetgpuPaccRuntimeJobTable;

typedef int (*gemm_staged_fn)(
    int, int, int, int, int,
    const void *,
    const void *, int, int, long long,
    const void *, int, int, long long,
    const void *,
    void *, int, int, long long,
    int, int);

typedef int (*gemm_staged_on_fn)(
    int, int,
    int, int, int, int, int,
    const void *,
    const void *, int, int, long long,
    const void *, int, int, long long,
    const void *,
    void *, int, int, long long,
    int, int);

static bool env_is_1(const char *name) {
    const char *v = getenv(name);
    return v && (!strcmp(v, "1") || !strcasecmp(v, "true") || !strcasecmp(v, "yes") || !strcasecmp(v, "force"));
}

static bool trace_enabled(void) {
    return env_is_1("HETGPU_PACC_SFMM_SUBMIT_SHIM_TRACE") || env_is_1("HETGPU_PACC_GEMM_TRACE");
}

static bool raw_output_completion_enabled(void) {
    return env_is_1("HETGPU_PACC_SFMM_RAW_OUTPUT_COMPLETION");
}

static bool notify_enabled(void) {
    const char *v = getenv("HETGPU_PACC_SFMM_NOTIFY_IRQ");
    if (!v || !*v) return true;
    return strcmp(v, "0") && strcasecmp(v, "false") &&
           strcasecmp(v, "no") && strcasecmp(v, "off");
}

static uint64_t parse_u64_env(const char *name, uint64_t def) {
    const char *v = getenv(name);
    if (!v || !*v) {
        return def;
    }
    char *end = NULL;
    errno = 0;
    uint64_t out = strtoull(v, &end, 0);
    return errno == 0 && end && *end == '\0' ? out : def;
}

static const char *mailbox_path(void) {
    const char *p = getenv("HETGPU_PACC_MAILBOX_DEVICE");
    if (p && *p) return p;
    p = getenv("HETGPU_PACC_MBOX_DEVICE");
    if (p && *p) return p;
    return "/dev/hetgpu_pacc_mbox_live0";
}

static const char *shared_ddr_path(void) {
    const char *p = getenv("HETGPU_PACC_SHARED_DDR_DEVICE");
    if (p && *p) return p;
    if (access("/dev/hetgpu_pacc_mbox_ddr_coh0", R_OK | W_OK) == 0) {
        return "/dev/hetgpu_pacc_mbox_ddr_coh0";
    }
    return mailbox_path();
}

static int open_rw(const char *path) {
    int fd = open(path, O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0 && trace_enabled()) {
        fprintf(stderr, "PACC SFMM shim: open %s failed: %s\n", path, strerror(errno));
    }
    return fd;
}

static bool retryable_io_errno(int err) {
    return err == EINTR || err == EAGAIN || err == EBUSY;
}

static void sleep_io_retry(unsigned attempt) {
    unsigned capped = attempt < 200U ? attempt : 200U;
    usleep(50U + capped * 5U);
}

static int write_full_at(int fd, uint64_t off, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    unsigned retries = 0;
    unsigned max_retries = (unsigned)parse_u64_env("HETGPU_PACC_SFMM_IO_RETRIES", 2000);
    while (len != 0) {
        ssize_t n = pwrite(fd, p, len, (off_t)off);
        if (n < 0) {
            int err = errno;
            if (retryable_io_errno(err) && retries++ < max_retries) {
                sleep_io_retry(retries);
                continue;
            }
            return -err;
        }
        if (n == 0) return -EIO;
        retries = 0;
        p += (size_t)n;
        off += (uint64_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int mmap_write_at(int fd, uint64_t off, const void *buf, size_t len) {
    long page_l = sysconf(_SC_PAGESIZE);
    uint64_t page = page_l > 0 ? (uint64_t)page_l : 4096ULL;
    uint64_t base = off & ~(page - 1ULL);
    size_t delta = (size_t)(off - base);
    size_t map_len = ((delta + len + (size_t)page - 1) / (size_t)page) * (size_t)page;
    void *p;
    if (fd < 0 || !buf || len == 0) return -EINVAL;
    p = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    if (p == MAP_FAILED) return -errno;
    memcpy((uint8_t *)p + delta, buf, len);
    __sync_synchronize();
    msync(p, map_len, MS_SYNC);
    munmap(p, map_len);
    return 0;
}

static int read_full_at(int fd, uint64_t off, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    unsigned retries = 0;
    unsigned max_retries = (unsigned)parse_u64_env("HETGPU_PACC_SFMM_IO_RETRIES", 2000);
    while (len != 0) {
        ssize_t n = pread(fd, p, len, (off_t)off);
        if (n < 0) {
            int err = errno;
            if (retryable_io_errno(err) && retries++ < max_retries) {
                sleep_io_retry(retries);
                continue;
            }
            return -err;
        }
        if (n == 0) return -EIO;
        retries = 0;
        p += (size_t)n;
        off += (uint64_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static bool alias_mirror_enabled(void) {
    const char *v = getenv("HETGPU_PACC_SFMM_ALIAS_MIRROR");
    if (!v || !*v) {
        return true;
    }
    return strcmp(v, "0") && strcasecmp(v, "false") && strcasecmp(v, "no") && strcasecmp(v, "off");
}

static bool control_legacy_mirror_enabled(void) {
    return env_is_1("HETGPU_PACC_CONTROL_LEGACY_MIRROR");
}

static bool top_mbox_ring_enabled(void) {
    return env_is_1("HETGPU_PACC_TOP_MBOX_RING") ||
           control_legacy_mirror_enabled();
}

static uint64_t pacc_alias_off(uint64_t off) {
    return parse_u64_env("HETGPU_PACC_SHARED_DDR_PACC_BASE",
                         HETGPU_PACC_SHARED_DDR_PACC_ALIAS_BASE) + off;
}

static int write_ddr(int fd, uint64_t off, const void *buf, size_t len) {
    int helper_rc = write_full_at(fd, HETGPU_PACC_SHARED_DDR_HELPER_OFF + off, buf, len);
    int alias_rc = 0;
    if (alias_mirror_enabled()) {
        alias_rc = write_full_at(fd, pacc_alias_off(off), buf, len);
        if (alias_rc != 0 && trace_enabled()) {
            fprintf(stderr,
                    "PACC SFMM shim: alias write failed off=0x%" PRIx64
                    " alias=0x%" PRIx64 " rc=%d\n",
                    off, pacc_alias_off(off), alias_rc);
        }
    }
    if (helper_rc == 0 && alias_rc == 0) {
        return 0;
    }
    return alias_rc != 0 ? alias_rc : helper_rc;
}

static int read_ddr(int fd, uint64_t off, void *buf, size_t len) {
    if (alias_mirror_enabled() &&
        read_full_at(fd, pacc_alias_off(off), buf, len) == 0) {
        return 0;
    }
    return read_full_at(fd, HETGPU_PACC_SHARED_DDR_HELPER_OFF + off, buf, len);
}

static int read_ddr_control(int pacc_id, int fd, uint64_t off, void *buf, size_t len) {
    uint64_t slot_off = pacc_id >= 0 ? (uint64_t)pacc_id * HETGPU_PACC_CONTROL_BYTES : 0;
    return read_full_at(fd,
                        HETGPU_PACC_SHARED_DDR_CONTROL_BASE_OFF + slot_off + off,
                        buf,
                        len);
}

static int write_control_ioctl(int fd, int pacc_id, uint64_t off, const void *buf, size_t len) {
    unsigned retries = 0;
    unsigned max_retries = (unsigned)parse_u64_env("HETGPU_PACC_SFMM_IO_RETRIES", 2000);
    if (fd < 0) return -ENODEV;
    if (len == 0) return 0;
    struct pacc_control_write req;
    memset(&req, 0, sizeof(req));
    req.pacc_id = pacc_id >= 0 ? (uint64_t)pacc_id : UINT64_MAX;
    req.off = off;
    req.len = len;
    req.user_ptr = (uint64_t)(uintptr_t)buf;
    for (;;) {
        if (ioctl(fd, PACC_IOC_CONTROL_WRITE, &req) == 0) {
            return 0;
        }
        int err = errno ? errno : EIO;
        if (retryable_io_errno(err) && retries++ < max_retries) {
            sleep_io_retry(retries);
            continue;
        }
        return -err;
    }
}

static bool buffer_all_byte(const void *buf, size_t len, uint8_t value) {
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != value) return false;
    }
    return true;
}

static uint64_t align_up_u64(uint64_t value, uint64_t align) {
    return (value + align - 1) / align * align;
}

static float read_f32_arg(const void *ptr, float def) {
    float v = def;
    if (ptr) memcpy(&v, ptr, sizeof(v));
    return v;
}

static uint64_t next_seq(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t seq = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid() << 16);
    return seq ? seq : 1;
}

static int pack_bf16_a(uint16_t *dst, const uint16_t *src, int m, int k, int lda, bool transa) {
    if (!transa) {
        for (int row = 0; row < m; ++row) {
            for (int kk = 0; kk < k; ++kk) {
                dst[(size_t)row * (size_t)k + (size_t)kk] = src[(size_t)row + (size_t)kk * (size_t)lda];
            }
        }
        return 0;
    }
    for (int row = 0; row < m; ++row) {
        memcpy(dst + (size_t)row * (size_t)k, src + (size_t)row * (size_t)lda, (size_t)k * sizeof(uint16_t));
    }
    return 0;
}

static int pack_bf16_b(uint16_t *dst, const uint16_t *src, int k, int n, int ldb, bool transb) {
    if (!transb) {
        for (int kk = 0; kk < k; ++kk) {
            for (int col = 0; col < n; ++col) {
                dst[(size_t)kk * (size_t)n + (size_t)col] = src[(size_t)kk + (size_t)col * (size_t)ldb];
            }
        }
        return 0;
    }
    for (int kk = 0; kk < k; ++kk) {
        memcpy(dst + (size_t)kk * (size_t)n, src + (size_t)kk * (size_t)ldb, (size_t)n * sizeof(uint16_t));
    }
    return 0;
}

static void unpack_bf16_c(uint16_t *dst, const uint16_t *src, int m, int n, int ldc) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            dst[(size_t)row + (size_t)col * (size_t)ldc] = src[(size_t)row * (size_t)n + (size_t)col];
        }
    }
}

static int write_control_both(int pacc_id, int ctl_fd, int ddr_fd, uint64_t off, const void *buf, size_t len) {
    int ctl_ioctl_rc = ctl_fd >= 0 ? write_control_ioctl(ctl_fd, pacc_id, off, buf, len) : -ENODEV;
    int ddr_ioctl_rc = ddr_fd >= 0 ? write_control_ioctl(ddr_fd, pacc_id, off, buf, len) : -ENODEV;
    int ctl_rc = -ENODEV;
    int ctl_mmap_rc = -ENODEV;
    int ctl_helper_rc = -ENODEV;
    int ctl_top_rc = -ENODEV;
    int ddr_rc = -ENODEV;
    int alias_rc = 0;

    if (ctl_ioctl_rc == 0 || ddr_ioctl_rc == 0) return 0;

    if (trace_enabled()) {
        fprintf(stderr,
                "PACC SFMM shim: control ioctl write failed off=0x%" PRIx64
                " len=%zu ctl_rc=%d ddr_rc=%d\n",
                off, len, ctl_ioctl_rc, ddr_ioctl_rc);
    }

    if (!control_legacy_mirror_enabled()) {
        return ctl_ioctl_rc != -ENODEV ? ctl_ioctl_rc : ddr_ioctl_rc;
    }

    ctl_rc = ctl_fd >= 0 ? write_full_at(ctl_fd, off, buf, len) : -ENODEV;
    ctl_mmap_rc = ctl_fd >= 0 ? mmap_write_at(ctl_fd, off, buf, len) : -ENODEV;
    ctl_helper_rc = ctl_fd >= 0 ?
        write_full_at(ctl_fd, HETGPU_PACC_AP2PACC_READBACK_HELPER_OFF + off, buf, len) :
        -ENODEV;
    ctl_top_rc = ctl_fd >= 0 ?
        write_full_at(ctl_fd, HETGPU_PACC_TOP_MBOX_RAM_OFF + off, buf, len) :
        -ENODEV;
    ddr_rc = ddr_fd >= 0 ? write_ddr(ddr_fd, off, buf, len) : -ENODEV;
    if (ctl_helper_rc != 0 && trace_enabled()) {
        fprintf(stderr,
                "PACC SFMM shim: control helper write failed off=0x%" PRIx64
                " helper=0x%" PRIx64 " rc=%d\n",
                off, HETGPU_PACC_AP2PACC_READBACK_HELPER_OFF + off,
                ctl_helper_rc);
    }
    if (ctl_mmap_rc != 0 && trace_enabled()) {
        fprintf(stderr,
                "PACC SFMM shim: control mmap write failed off=0x%" PRIx64
                " rc=%d\n",
                off, ctl_mmap_rc);
    }
    if (ctl_top_rc != 0 && trace_enabled()) {
        fprintf(stderr,
                "PACC SFMM shim: control top-mbox write failed off=0x%" PRIx64
                " top=0x%" PRIx64 " rc=%d\n",
                off, HETGPU_PACC_TOP_MBOX_RAM_OFF + off,
                ctl_top_rc);
    }
    if (ddr_fd >= 0 && alias_mirror_enabled()) {
        uint64_t alias = pacc_alias_off(HETGPU_PACC_SHARED_DDR_HELPER_OFF + off);
        alias_rc = write_full_at(ddr_fd, alias, buf, len);
        if (alias_rc != 0 && trace_enabled()) {
            fprintf(stderr,
                    "PACC SFMM shim: control alias write failed off=0x%" PRIx64
                    " alias=0x%" PRIx64 " rc=%d\n",
                    off, alias, alias_rc);
            }
    }
    if (alias_rc != 0) return alias_rc;
    if (ddr_rc == 0) return 0;
    if (ctl_mmap_rc == 0) return 0;
    if (ctl_helper_rc == 0) return 0;
    if (ctl_top_rc == 0) return 0;
    return ctl_rc == 0 ? 0 : ddr_rc;
}

static void ring_top_mbox(int ctl_fd, int dev_id, uint64_t seq) {
    if (ctl_fd < 0) return;
    if (!top_mbox_ring_enabled()) return;
    uint32_t bit = 1u;
    if (dev_id > 0 && dev_id < 32) bit = 1u << (uint32_t)dev_id;
    int rc = write_full_at(ctl_fd, HETGPU_PACC_TOP_MBOX_DB_OFF, &bit, sizeof(bit));
    if (trace_enabled()) {
        fprintf(stderr,
                "PACC SFMM shim: top mbox doorbell rc=%d bit=0x%x seq=%" PRIu64 "\n",
                rc, bit, seq);
    }
}

static int write_runtime_submit(int pacc_id, int ctl_fd, int ddr_fd, uint64_t seq, const HetgpuPaccGemmJob *job) {
    uint8_t zero32[32] = {0};
    (void)write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_COMPLETION_OFF, zero32, sizeof(zero32));
    (void)write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_BEACON_OFF, zero32, sizeof(zero32));

    HetgpuPaccRuntimeJobTable table;
    memset(&table, 0, sizeof(table));
    table.magic = HETGPU_PACC_RUNTIME_TABLE_MAGIC;
    table.version = HETGPU_PACC_RUNTIME_TABLE_VERSION;
    table.seq = seq;
    table.have_gemm = 1;
    table.gemm = *job;

    const uint8_t *table_bytes = (const uint8_t *)&table;
    size_t commit = 64;
    if (commit > sizeof(table)) commit = sizeof(table);
    int rc = 0;
    if (sizeof(table) > commit) {
        rc = write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_RUNTIME_TABLE_OFF + commit, table_bytes + commit, sizeof(table) - commit);
        if (rc) return rc;
    }
    rc = write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_RUNTIME_TABLE_OFF, table_bytes, commit);
    if (rc) return rc;

    uint8_t slot[HETGPU_PACC_ARG_SLOT_BYTES];
    memset(slot, 0, sizeof(slot));
    HetgpuPaccArgSlotHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = HETGPU_PACC_JOB_MAGIC;
    hdr.version = HETGPU_PACC_JOB_VERSION;
    hdr.job_id = PACC_JOB_GEMM;
    hdr.seq = seq;
    hdr.arg_len = sizeof(*job);
    memcpy(slot + sizeof(hdr), job, sizeof(*job));
    rc = write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_ARG_BASE_OFF, zero32, sizeof(zero32));
    if (rc) return rc;
    rc = write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_ARG_BASE_OFF + sizeof(hdr), slot + sizeof(hdr), sizeof(*job));
    if (rc) return rc;
    memcpy(slot, &hdr, sizeof(hdr));
    rc = write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_ARG_BASE_OFF, slot, sizeof(hdr));
    if (rc) return rc;

    HetgpuPaccDoorbell doorbell;
    memset(&doorbell, 0, sizeof(doorbell));
    doorbell.magic = HETGPU_PACC_JOB_MAGIC;
    doorbell.version = HETGPU_PACC_JOB_VERSION;
    doorbell.job_id = PACC_JOB_GEMM;
    doorbell.seq = seq;
    return write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_DOORBELL_OFF, &doorbell, sizeof(doorbell));
}

static int completion_matches(const uint8_t st[32], uint64_t seq, uint32_t *status_out) {
    uint64_t magic = 0, got_seq = 0;
    uint32_t version = 0, job_id = 0, status = 0;
    memcpy(&magic, st, 8);
    memcpy(&version, st + 8, 4);
    memcpy(&job_id, st + 12, 4);
    memcpy(&status, st + 16, 4);
    memcpy(&got_seq, st + 24, 8);
    if (status_out) *status_out = status;
    return magic == HETGPU_PACC_JOB_MAGIC && version == HETGPU_PACC_JOB_VERSION &&
           job_id == PACC_JOB_GEMM && got_seq == seq;
}

static uint64_t elapsed_ns_since(const struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)(now.tv_sec - start->tv_sec) * 1000000000ULL +
           (uint64_t)((int64_t)now.tv_nsec - (int64_t)start->tv_nsec);
}

static int finish_completion(int pacc_id,
                             int ddr_fd,
                             uint64_t seq,
                             uint32_t status,
                             const char *source,
                             const struct timespec *wait_start) {
    const bool required =
        env_is_1("HETGPU_PACC_COMPLETION_TELEMETRY_REQUIRED");
    const uint64_t timeout_us = parse_u64_env(
        "HETGPU_PACC_COMPLETION_TELEMETRY_TIMEOUT_US",
        required ? 10000ULL : 0ULL);
    const uint64_t deadline_ns = elapsed_ns_since(wait_start) +
                                 timeout_us * 1000ULL;
    HetgpuPaccCompletionTelemetry telemetry;
    bool matched = false;

    if (trace_enabled() || required) {
        fprintf(stderr,
                "PACC SFMM bench: completion_wait_ns=%" PRIu64
                " source=%s seq=%" PRIu64 "\n",
                elapsed_ns_since(wait_start), source, seq);
    }
    if (ddr_fd >= 0) {
        do {
            memset(&telemetry, 0, sizeof(telemetry));
            if (read_ddr_control(pacc_id, ddr_fd,
                                 HETGPU_PACC_COMPLETION_TELEMETRY_OFF,
                                 &telemetry, sizeof(telemetry)) == 0 &&
                telemetry.magic == HETGPU_PACC_COMPLETION_TELEMETRY_MAGIC &&
                telemetry.version ==
                    HETGPU_PACC_COMPLETION_TELEMETRY_VERSION &&
                telemetry.record_bytes == sizeof(telemetry) &&
                telemetry.job_id == PACC_JOB_GEMM &&
                telemetry.seq == seq &&
                (telemetry.flags & 1U) != 0) {
                matched = true;
                break;
            }
            if (timeout_us == 0 ||
                elapsed_ns_since(wait_start) >= deadline_ns) {
                break;
            }
            usleep(10);
        } while (true);
    }
    if (matched) {
        uint64_t compute_ns =
            telemetry.compute_end_ns >= telemetry.compute_start_ns
                ? telemetry.compute_end_ns - telemetry.compute_start_ns
                : 0;
        uint64_t publish_ns =
            telemetry.publish_end_ns >= telemetry.publish_start_ns
                ? telemetry.publish_end_ns - telemetry.publish_start_ns
                : 0;
        fprintf(stderr,
                "PACC SFMM telemetry: seq=%" PRIu64
                " status=0x%x compute_ns=%" PRIu64
                " completion_publish_ns=%" PRIu64
                " xsfmm_cycles=%" PRIu64
                " xsfmm_repeats=%" PRIu64 "\n",
                telemetry.seq, telemetry.status, compute_ns, publish_ns,
                telemetry.xsfmm_cycles, telemetry.xsfmm_repeats);
    } else if (required) {
        fprintf(stderr,
                "PACC SFMM telemetry: missing seq=%" PRIu64
                " timeout_us=%" PRIu64 "\n",
                seq, timeout_us);
        return -EPROTO;
    }
    return status == 0 ? 0 : -EIO;
}

static int synthesize_completion(int pacc_id, int ctl_fd, int ddr_fd, uint64_t seq, uint32_t status) {
    HetgpuPaccDoorbell done;
    memset(&done, 0, sizeof(done));
    done.magic = HETGPU_PACC_JOB_MAGIC;
    done.version = HETGPU_PACC_JOB_VERSION;
    done.job_id = PACC_JOB_GEMM;
    done.status = status;
    done.seq = seq;
    return write_control_both(pacc_id, ctl_fd, ddr_fd, HETGPU_PACC_COMPLETION_OFF,
                              &done, sizeof(done));
}

static int wait_completion(
    int pacc_id, int ctl_fd, int ddr_fd, uint64_t seq, int timeout_ms,
    uint64_t raw_result_off, void *result_buf, size_t result_len) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    uint8_t st[32];
    const uint64_t offs[] = {
        HETGPU_PACC_COMPLETION_OFF,
        HETGPU_PACC_AP2PACC_READBACK_HELPER_OFF + HETGPU_PACC_COMPLETION_OFF,
        HETGPU_PACC_PACC2AP_RW_HELPER_OFF + HETGPU_PACC_COMPLETION_OFF,
    };
    for (;;) {
        for (size_t i = 0; i < sizeof(offs) / sizeof(offs[0]); ++i) {
            memset(st, 0, sizeof(st));
            if (read_full_at(ctl_fd, offs[i], st, sizeof(st)) == 0) {
                uint32_t status = 0;
                if (completion_matches(st, seq, &status)) {
                    if (trace_enabled()) {
                        fprintf(stderr, "PACC SFMM shim: completion source=%zu status=0x%x seq=%" PRIu64 "\n", i, status, seq);
                    }
                    return finish_completion(pacc_id, ddr_fd, seq, status,
                                             "control", &start);
                }
            }
        }
        memset(st, 0, sizeof(st));
        if (ddr_fd >= 0 && read_ddr_control(pacc_id, ddr_fd, HETGPU_PACC_COMPLETION_OFF, st, sizeof(st)) == 0) {
            uint32_t status = 0;
            if (completion_matches(st, seq, &status)) {
                if (trace_enabled()) {
                    fprintf(stderr, "PACC SFMM shim: completion source=shared-ddr-pacc-control status=0x%x seq=%" PRIu64 "\n", status, seq);
                }
                return finish_completion(pacc_id, ddr_fd, seq, status,
                                         "shared-ddr-pacc-control", &start);
            }
        }
        memset(st, 0, sizeof(st));
        if (ddr_fd >= 0 && read_ddr(ddr_fd, HETGPU_PACC_COMPLETION_OFF, st, sizeof(st)) == 0) {
            uint32_t status = 0;
            if (completion_matches(st, seq, &status)) {
                if (trace_enabled()) {
                    fprintf(stderr, "PACC SFMM shim: completion source=shared-ddr-control status=0x%x seq=%" PRIu64 "\n", status, seq);
                }
                return finish_completion(pacc_id, ddr_fd, seq, status,
                                         "shared-ddr-control", &start);
            }
        }
        memset(st, 0, sizeof(st));
        if (ddr_fd >= 0 &&
            read_full_at(ddr_fd,
                         pacc_alias_off(HETGPU_PACC_SHARED_DDR_HELPER_OFF +
                                        HETGPU_PACC_COMPLETION_OFF),
                         st,
                         sizeof(st)) == 0) {
            uint32_t status = 0;
            if (completion_matches(st, seq, &status)) {
                if (trace_enabled()) {
                    fprintf(stderr,
                            "PACC SFMM shim: completion source=shared-ddr-control-alias"
                            " status=0x%x seq=%" PRIu64 "\n",
                            status, seq);
                }
                return finish_completion(pacc_id, ddr_fd, seq, status,
                                         "shared-ddr-control-alias", &start);
            }
        }
        if (raw_output_completion_enabled() &&
            ddr_fd >= 0 && result_buf && result_len != 0) {
            uint8_t sample[128];
            size_t sample_len = result_len < sizeof(sample) ? result_len : sizeof(sample);
            memset(sample, 0, sizeof(sample));
            if (read_full_at(ddr_fd, raw_result_off, sample, sample_len) == 0 &&
                !buffer_all_byte(sample, sample_len, 0xa5)) {
                int raw_rc = read_full_at(ddr_fd, raw_result_off, result_buf, result_len);
                if (raw_rc == 0 && !buffer_all_byte(result_buf, result_len, 0xa5)) {
                    int mirror_rc = write_ddr(ddr_fd, raw_result_off, result_buf, result_len);
                    int done_rc = synthesize_completion(pacc_id, ctl_fd, ddr_fd, seq, 0);
                    if (trace_enabled()) {
                        fprintf(stderr,
                                "PACC SFMM shim: completion source=raw-ddr-output"
                                " off=0x%" PRIx64 " bytes=%zu seq=%" PRIu64
                                " mirror_rc=%d completion_rc=%d\n",
                                raw_result_off, result_len, seq, mirror_rc, done_rc);
                    }
                    return finish_completion(pacc_id, ddr_fd, seq, 0,
                                             "raw-ddr-output", &start);
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        time_t sec = now.tv_sec - start.tv_sec;
        long nsec = now.tv_nsec - start.tv_nsec;
        if (nsec < 0) {
            sec -= 1;
            nsec += 1000000000L;
        }
        uint64_t elapsed_ms = sec > 0 ? (uint64_t)sec * 1000ULL : 0ULL;
        elapsed_ms += (uint64_t)nsec / 1000000ULL;
        if (elapsed_ms >= (uint64_t)timeout_ms) {
            if (trace_enabled()) {
                fprintf(stderr, "PACC SFMM shim: completion timeout seq=%" PRIu64 "\n", seq);
            }
            return -ETIMEDOUT;
        }
        usleep(100);
    }
}

static int call_next_staged(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    gemm_staged_fn next = (gemm_staged_fn)dlsym(RTLD_NEXT, "hetgpu_pacc_submit_gemm_staged");
    if (!next) return -127;
    return next(transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType);
}

static int shim_submit_on(
    int dev_id, int slot_id,
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    if (!env_is_1("HETGPU_PACC_SFMM_SUBMIT_SHIM")) return -ENOSYS;
    if (!A || !B || !C || m <= 0 || n <= 0 || k <= 0 || batchCount != 1 ||
        Atype != PACC_DTYPE_BF16 || Btype != PACC_DTYPE_BF16 || Ctype != PACC_DTYPE_BF16 ||
        strideA != 0 || strideB != 0 || strideC != 0) {
        return -ENOSYS;
    }

    float alpha_v = read_f32_arg(alpha, 1.0f);
    float beta_v = read_f32_arg(beta, 0.0f);
    if (alpha_v != 1.0f || beta_v != 0.0f) {
        return -ENOSYS;
    }

    uint64_t shared_base = parse_u64_env("HETGPU_PACC_SHARED_DDR_BASE", 0x20110600000ULL);
    uint64_t shared_bytes = parse_u64_env("HETGPU_PACC_SHARED_DDR_BYTES", 0x100000000ULL);
    uint64_t payload_base = parse_u64_env("HETGPU_PACC_SHARED_DDR_PAYLOAD_BASE_OFF", 0x00200000ULL);
    int slot_count = (int)parse_u64_env("HETGPU_PACC_GEMM_SHARED_SLOTS", 4);
    if (slot_count <= 0) slot_count = 4;
    if (dev_id < 0) dev_id = (int)parse_u64_env("HETGPU_PACC_GEMM_DEVICE", 0);
    if (slot_id < 0) slot_id = dev_id;
    if (slot_id < 0) slot_id = 0;
    if (slot_id >= slot_count) slot_id = slot_count - 1;

    uint64_t payload_bytes = shared_bytes > payload_base ? shared_bytes - payload_base : 0;
    uint64_t slot_bytes = parse_u64_env("HETGPU_PACC_GEMM_SLOT_BYTES", payload_bytes / (uint64_t)slot_count);
    uint64_t slot_off = payload_base + (uint64_t)slot_id * slot_bytes;

    uint64_t a_bytes = (uint64_t)m * (uint64_t)k * 2ULL;
    uint64_t b_bytes = (uint64_t)k * (uint64_t)n * 2ULL;
    uint64_t c_bytes = (uint64_t)m * (uint64_t)n * 2ULL;
    uint64_t a_off = 0;
    uint64_t b_off = align_up_u64(a_bytes, 64);
    uint64_t c_off = align_up_u64(b_off + b_bytes, 64);
    uint64_t alpha_off = align_up_u64(c_off + c_bytes, 64);
    uint64_t beta_off = alpha_off + 4;
    uint64_t total = beta_off + 4;
    if (payload_bytes == 0 || total > slot_bytes) {
        if (trace_enabled()) fprintf(stderr, "PACC SFMM shim: bad shared DDR window total=%" PRIu64 " slot=%" PRIu64 "\n", total, slot_bytes);
        return -ENOMEM;
    }

    uint16_t *a_stage = (uint16_t *)malloc((size_t)a_bytes);
    uint16_t *b_stage = (uint16_t *)malloc((size_t)b_bytes);
    uint8_t *c_stage = (uint8_t *)malloc((size_t)c_bytes);
    if (!a_stage || !b_stage || !c_stage) {
        free(a_stage); free(b_stage); free(c_stage);
        return -ENOMEM;
    }
    pack_bf16_a(a_stage, (const uint16_t *)A, m, k, lda, transa != 0);
    pack_bf16_b(b_stage, (const uint16_t *)B, k, n, ldb, transb != 0);
    memset(c_stage, 0xa5, (size_t)c_bytes);

    int ddr_fd = open_rw(shared_ddr_path());
    int ctl_fd = open_rw(mailbox_path());
    if (ddr_fd < 0 || ctl_fd < 0) {
        if (ddr_fd >= 0) close(ddr_fd);
        if (ctl_fd >= 0) close(ctl_fd);
        free(a_stage); free(b_stage); free(c_stage);
        return -ENODEV;
    }

    int rc = 0;
    rc = write_ddr(ddr_fd, slot_off + a_off, a_stage, (size_t)a_bytes);
    if (!rc) rc = write_ddr(ddr_fd, slot_off + b_off, b_stage, (size_t)b_bytes);
    if (!rc) rc = write_ddr(ddr_fd, slot_off + c_off, c_stage, (size_t)c_bytes);
    if (!rc) rc = write_ddr(ddr_fd, slot_off + alpha_off, &alpha_v, sizeof(alpha_v));
    if (!rc) rc = write_ddr(ddr_fd, slot_off + beta_off, &beta_v, sizeof(beta_v));
    if (rc) {
        if (trace_enabled()) fprintf(stderr, "PACC SFMM shim: shared DDR write failed rc=%d\n", rc);
        close(ddr_fd); close(ctl_fd); free(a_stage); free(b_stage); free(c_stage);
        return rc;
    }
    HetgpuPaccGemmJob job;
    memset(&job, 0, sizeof(job));
    job.transa = 0;
    job.transb = 0;
    job.atype = PACC_DTYPE_BF16;
    job.btype = PACC_DTYPE_BF16;
    job.ctype = PACC_DTYPE_BF16;
    job.compute_type = (uint32_t)computeType;
    job.m = (uint64_t)m;
    job.n = (uint64_t)n;
    job.k = (uint64_t)k;
    job.a_addr = shared_base + slot_off + a_off;
    job.b_addr = shared_base + slot_off + b_off;
    job.c_addr = shared_base + slot_off + c_off;
    job.alpha_addr = shared_base + slot_off + alpha_off;
    job.beta_addr = shared_base + slot_off + beta_off;
    job.lda = k;
    job.ldb = n;
    job.ldc = n;
    job.batch_count = 1;

    uint64_t seq = next_seq();
    if (trace_enabled()) {
        fprintf(stderr,
                "PACC SFMM shim: submit dev=%d slot=%d slot_off=0x%" PRIx64 " a=0x%" PRIx64 " b=0x%" PRIx64 " c=0x%" PRIx64 " m=%d n=%d k=%d seq=%" PRIu64 "\n",
                dev_id, slot_id, slot_off, job.a_addr, job.b_addr, job.c_addr, m, n, k, seq);
    }
    rc = write_runtime_submit(dev_id, ctl_fd, ddr_fd, seq, &job);
    if (!rc) {
        ring_top_mbox(ctl_fd, dev_id, seq);
    }
    if (!rc && notify_enabled()) {
        int notify_rc = ioctl(ctl_fd, PACC_IOC_ZLUDA_IRQ);
        if (trace_enabled()) {
            fprintf(stderr,
                    "PACC SFMM shim: notify ioctl rc=%d errno=%d seq=%" PRIu64 "\n",
                    notify_rc, notify_rc < 0 ? errno : 0, seq);
        }
    }
    if (!rc) {
        int timeout_ms = (int)parse_u64_env("HETGPU_PACC_JOB_TIMEOUT_MS", 30000);
        rc = wait_completion(dev_id, ctl_fd, ddr_fd, seq, timeout_ms,
                             slot_off + c_off, c_stage, (size_t)c_bytes);
    }
    if (!rc) {
        rc = read_ddr(ddr_fd, slot_off + c_off, c_stage, (size_t)c_bytes);
        if (!rc) {
            unpack_bf16_c((uint16_t *)C, (const uint16_t *)c_stage, m, n, ldc);
        }
    }
    if (trace_enabled() && rc) {
        fprintf(stderr, "PACC SFMM shim: submit failed rc=%d errno_text=%s\n", rc, strerror(-rc));
    }
    close(ddr_fd);
    close(ctl_fd);
    free(a_stage);
    free(b_stage);
    free(c_stage);
    return rc;
}

int hetgpu_pacc_submit_gemm_staged(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    int rc = shim_submit_on(-1, -1, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType);
    if (rc != -ENOSYS) return rc;
    return call_next_staged(transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType);
}

int hetgpu_pacc_submit_gemm_staged_on(
    int dev_id, int slot_id,
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    int rc = shim_submit_on(dev_id, slot_id, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType);
    if (rc != -ENOSYS) return rc;
    gemm_staged_on_fn next = (gemm_staged_on_fn)dlsym(RTLD_NEXT, "hetgpu_pacc_submit_gemm_staged_on");
    if (!next) return -127;
    return next(dev_id, slot_id, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType);
}
