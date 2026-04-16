#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
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
#define PACC_DTYPE_F32 4U

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

struct PreloadedJobs {
    bool have_gemm;
    bool have_softmax;
    bool have_rmsnorm;
    struct GemmJob gemm;
    struct SoftmaxJob softmax;
    struct RmsNormJob rmsnorm;
};

struct RuntimeJobTable {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t seq;
    uint32_t have_gemm;
    uint32_t have_softmax;
    uint32_t have_rmsnorm;
    uint32_t reserved;
    struct GemmJob gemm;
    struct SoftmaxJob softmax;
    struct RmsNormJob rmsnorm;
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

static int run_gemm(int fd, const struct GemmJob *job) {
    if (!job->m || !job->n || !job->k || !job->a_addr || !job->b_addr || !job->c_addr) {
        return 0xffff1001;
    }
    if (job->atype != PACC_DTYPE_F32 || job->btype != PACC_DTYPE_F32 || job->ctype != PACC_DTYPE_F32) {
        return 0xffff1002;
    }

    uint64_t batch_count = job->batch_count ? job->batch_count : 1;
    size_t a_elems = (size_t)(job->m * job->k * batch_count);
    size_t b_elems = (size_t)(job->k * job->n * batch_count);
    size_t c_elems = (size_t)(job->m * job->n * batch_count);
    struct Map ma = {0}, mb = {0}, mc = {0}, malpha = {0}, mbeta = {0};
    if (map_phys(fd, job->a_addr, a_elems * sizeof(float), &ma) ||
        map_phys(fd, job->b_addr, b_elems * sizeof(float), &mb) ||
        map_phys(fd, job->c_addr, c_elems * sizeof(float), &mc)) {
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

    const float *a0 = (const float *)ma.ptr;
    const float *b0 = (const float *)mb.ptr;
    float *c0 = (float *)mc.ptr;
    for (uint64_t batch = 0; batch < batch_count; batch++) {
        const float *a = a0 + (job->stride_a > 0 ? (uint64_t)job->stride_a * batch : job->m * job->k * batch);
        const float *b = b0 + (job->stride_b > 0 ? (uint64_t)job->stride_b * batch : job->k * job->n * batch);
        float *c = c0 + (job->stride_c > 0 ? (uint64_t)job->stride_c * batch : job->m * job->n * batch);
        for (uint64_t row = 0; row < job->m; row++) {
            for (uint64_t col = 0; col < job->n; col++) {
                float acc = 0.0f;
                for (uint64_t kk = 0; kk < job->k; kk++) {
                    float av = job->transa ? a[kk * job->lda + row] : a[row * job->lda + kk];
                    float bv = job->transb ? b[col * job->ldb + kk] : b[kk * job->ldb + col];
                    acc += av * bv;
                }
                c[row * job->ldc + col] = alpha * acc + beta * c[row * job->ldc + col];
            }
        }
    }
    msync(mc.base, mc.map_len, MS_SYNC);
    unmap_phys(&malpha); unmap_phys(&mbeta); unmap_phys(&ma); unmap_phys(&mb); unmap_phys(&mc);
    return 0;
}

static int run_softmax(int fd, const struct SoftmaxJob *job) {
    if (!job->src_addr || !job->dst_addr || !job->rows || !job->cols) return 0xffff2001;
    if (job->dtype != PACC_DTYPE_F32) return 0xffff2002;
    uint64_t stride = job->stride ? job->stride : job->cols;
    size_t elems = (size_t)(job->rows * stride);
    struct Map ms = {0}, md = {0};
    if (map_phys(fd, job->src_addr, elems * sizeof(float), &ms) ||
        map_phys(fd, job->dst_addr, elems * sizeof(float), &md)) {
        unmap_phys(&ms); unmap_phys(&md);
        return 0xffff2003;
    }
    const float *src = (const float *)ms.ptr;
    float *dst = (float *)md.ptr;
    for (uint64_t row = 0; row < job->rows; row++) {
        uint64_t base = row * stride;
        float max_v = src[base];
        for (uint64_t col = 1; col < job->cols; col++) {
            if (src[base + col] > max_v) max_v = src[base + col];
        }
        float sum = 0.0f;
        for (uint64_t col = 0; col < job->cols; col++) {
            float e = expf_fast(src[base + col] - max_v);
            dst[base + col] = e;
            sum += e;
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint64_t col = 0; col < job->cols; col++) dst[base + col] *= inv;
    }
    msync(md.base, md.map_len, MS_SYNC);
    unmap_phys(&ms); unmap_phys(&md);
    return 0;
}

static int run_rmsnorm(int fd, const struct RmsNormJob *job) {
    if (!job->x_addr || !job->y_addr || !job->rows || !job->hidden) return 0xffff3001;
    if (job->dtype != PACC_DTYPE_F32) return 0xffff3002;
    size_t elems = (size_t)(job->rows * job->hidden);
    struct Map mx = {0}, mw = {0}, my = {0};
    if (map_phys(fd, job->x_addr, elems * sizeof(float), &mx) ||
        map_phys(fd, job->y_addr, elems * sizeof(float), &my)) {
        unmap_phys(&mx); unmap_phys(&my);
        return 0xffff3003;
    }
    const float *x = (const float *)mx.ptr;
    float *y = (float *)my.ptr;
    const float *w = NULL;
    if (job->weight_addr && !map_phys(fd, job->weight_addr, job->hidden * sizeof(float), &mw)) {
        w = (const float *)mw.ptr;
    }
    for (uint64_t row = 0; row < job->rows; row++) {
        uint64_t base = row * job->hidden;
        float sumsq = 0.0f;
        for (uint64_t i = 0; i < job->hidden; i++) {
            float v = x[base + i];
            sumsq += v * v;
        }
        float scale = rsqrtf_newton(sumsq / (float)job->hidden + job->eps);
        for (uint64_t i = 0; i < job->hidden; i++) {
            y[base + i] = x[base + i] * scale * (w ? w[i] : 1.0f);
        }
    }
    msync(my.base, my.map_len, MS_SYNC);
    unmap_phys(&mx); unmap_phys(&mw); unmap_phys(&my);
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
    *last_table_seq = local.seq;
    log_msg("runtime table seq=%" PRIu64 " have_gemm=%u have_softmax=%u have_rmsnorm=%u",
            local.seq, local.have_gemm, local.have_softmax, local.have_rmsnorm);
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
