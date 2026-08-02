#define _GNU_SOURCE
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
#if defined(__has_include)
#if defined(__riscv_vector) && __has_include(<riscv_vector.h>)
#include <riscv_vector.h>
#endif
#if defined(__riscv_vector) && __has_include(<sifive_vector.h>)
#include <sifive_vector.h>
#define HETGPU_PACC_HAVE_SIFIVE_VECTOR 1
#endif
#endif

#if defined(__riscv) && defined(__riscv_vector) && \
    defined(HETGPU_PACC_HAVE_XSFMM32A16F) && defined(__riscv_zvfbfmin)
#define HETGPU_PACC_HAVE_XSFMM_BF16 1
#endif

#define HETGPU_PACC_JOB_MAGIC 0x4847505550414343ULL
#define HETGPU_PACC_BEACON_MAGIC 0x4847505542434e31ULL
#define HETGPU_PACC_JOB_VERSION 1U
#define PACC_JOB_MAGIC 0x504143434a4f4231ULL
#define PACC_JOB_VERSION 1U
#define PACC_JOB_FLAG_HAS_LAUNCH_ABI (1U << 0)
#define PACC_KERNEL_LAUNCH_ABI_MAGIC 0x5041434341524731ULL
#define PACC_KERNEL_LAUNCH_ABI_VERSION 1U
#define PACC_JOB_IMAGE_HEADER_WIRE_BYTES 72U
#define PACC_KERNEL_LAUNCH_ABI_WIRE_BYTES 48U
#define PACC_KERNEL_ARG_RECORD_WIRE_BYTES 32U
#define PACC_KERNEL_BUFFER_BINDING_WIRE_BYTES 24U
#define PACC_KERNEL_ARG_FLAG_INLINE_BLOB (1U << 16)
#define PACC_KERNEL_ARG_FLAG_BUFFER_INPUT (1U << 8)
#define PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT (1U << 9)
#define PACC_KERNEL_JOB_ID 0U

#define HETGPU_PACC_JOB_GEMM 1U
#define HETGPU_PACC_JOB_SOFTMAX 2U
#define HETGPU_PACC_JOB_RMSNORM 3U
#define HETGPU_PACC_JOB_ALLREDUCE 4U
#define HETGPU_PACC_JOB_MMVF 5U
#define HETGPU_PACC_MAX_JOB_ID 16U

#define HETGPU_PACC_ARG_SLOT_BYTES 0x400UL
#define HETGPU_PACC_CONTROL_BYTES 0x2000UL
#define HETGPU_PACC_ARG_BASE_OFF 0x100UL
#define HETGPU_PACC_RUNTIME_TABLE_OFF 0x1400UL
#define HETGPU_PACC_RUNTIME_TABLE_MAGIC 0x4847505554424c31ULL
#define HETGPU_PACC_RUNTIME_TABLE_VERSION 1U
#define HETGPU_PACC_COUNT 4U
#define HETGPU_PACC_ID_CLAIM_OFF 0x1ee0UL
#define HETGPU_PACC_ID_CLAIM_MAGIC 0x4847505550494400ULL
#define AP2PACC_MBOX_PHYS 0x20000000ULL
#define PACC2AP_MBOX_PHYS 0x20002000ULL
#define HETGPU_PACC_DEFAULT_SHARED_DDR_BYTES 0x100000000ULL
#define HETGPU_PACC_SHARED_DDR_USER_OFF 0x00100000ULL
#define HETGPU_PACC_SHARED_DDR_FD_USER_OFF HETGPU_PACC_SHARED_DDR_USER_OFF
#define HETGPU_PACC_AP2PACC_READ_HELPER_OFF 0x02000000ULL
#define HETGPU_PACC_PACC2AP_RW_HELPER_OFF 0x02002000ULL
#define HETGPU_PACC_COMPLETION_OFF 0x1f20ULL
#define HETGPU_PACC_COMPLETION_MIRROR_DEFAULT_OFF 0x120e0ULL
#define HETGPU_PACC_BEACON_OFF 0x1f40ULL
#define HETGPU_PACC_COMPLETION_TELEMETRY_OFF 0x1f80ULL
#define HETGPU_PACC_COMPLETION_TELEMETRY_MAGIC 0x48475055544c4d31ULL
#define HETGPU_PACC_COMPLETION_TELEMETRY_VERSION 1U
#define HETGPU_PACC_DIAG_RING_SLOT 8U
#define HETGPU_PACC_DIAG_RING_OFF 0x0000ULL
#define HETGPU_PACC_DIAG_RING_RECORDS 192U
#define HETGPU_PACC_DIAG_MAGIC 0x4847505544494147ULL
#define HETGPU_PACC_RMS_DEBUG_SLOT 9U
#define HETGPU_PACC_RMS_DEBUG_RECORD_BYTES 0x100ULL
#define HETGPU_PACC_RMS_DEBUG_MAGIC 0x48475055524d5344ULL
#define PACC_DTYPE_INT8 0U
#define PACC_DTYPE_UINT8 1U
#define PACC_DTYPE_INT32 2U
#define PACC_DTYPE_F16 3U
#define PACC_DTYPE_F32 4U
#define PACC_DTYPE_BF16 5U
#define PACC_GEMM_THREADS 32U
#define PACC_XSFMM16_TILE_M 32U
#define PACC_XSFMM16_TILE_N 32U

#ifndef HETGPU_PACC_COMPLETION_TELEMETRY
#define HETGPU_PACC_COMPLETION_TELEMETRY 0
#endif

#ifndef HETGPU_PACC_COMPLETION_SETTLE_NS
#define HETGPU_PACC_COMPLETION_SETTLE_NS 0ULL
#endif

#ifndef HETGPU_PACC_COMPLETION_FAST_PATH
#define HETGPU_PACC_COMPLETION_FAST_PATH 0
#endif
#define PACC_XSFMM16_TILE_K 4U
#define PACC_RVV_F32_TILE_M 32U
#define PACC_RVV_F32_TILE_N 32U
#define PACC_RVV_F32_TILE_K 32U
#define PACC_GEMM_TILE_M_MAX PACC_XSFMM16_TILE_M
#define PACC_GEMM_TILE_N_MAX PACC_XSFMM16_TILE_N
#define PACC_KERNEL_DEFAULT_THREADS 32U
#define PACC_KERNEL_MAX_THREADS 64U
#define PACC_MAX_KERNEL_ARGS 64U
#define PACC_MAX_KERNEL_BINDINGS 256U

static bool g_xsfmm_bf16_checked = false;
static bool g_xsfmm_bf16_usable = false;
static bool g_xsfmm_b_transposed_pack = false;
static bool g_xsfmm_c_transposed_pack = false;
static bool g_xsfmm_context_ready = false;
static int g_xsfmm_context_error = 0;
static int g_xsfmm_run_fd = -1;

#define HETGPU_XSFMM_REQUEST_MAGIC UINT64_C(0x5853464d4d524551)

struct XsfmmRequest {
    uint64_t magic;
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t repeats;
    uint64_t cycles;
    uint64_t completed_repeats;
    int32_t status;
    uint32_t reserved;
    uint64_t batch_count;
    uint64_t a_batch_stride;
    uint64_t b_batch_stride;
    uint64_t c_batch_stride;
};

struct GemmTiming {
    uint64_t seq;
    uint64_t compute_start_ns;
    uint64_t compute_end_ns;
    uint64_t xsfmm_cycles;
    uint64_t xsfmm_repeats;
};

static struct GemmTiming g_last_gemm_timing;
static uint64_t g_xsfmm_estimated_commands;

extern int xsfmm_native_bf16(const uint16_t *a_km,
                             const uint16_t *b_kn,
                             float *c_mn,
                             size_t m,
                             size_t n,
                             size_t k);

#define PACC_IOC_MAGIC 'p'
#define PACC_IOC_ZLUDA_IRQ _IO(PACC_IOC_MAGIC, 5)
#define PACC_IOC_ZLUDA_IRQ_WITH_DDR _IOW(PACC_IOC_MAGIC, 5, struct pacc_zluda_ddr_info)
#define PACC_IOC_ZLUDA_GET_DDR_BASE _IOR(PACC_IOC_MAGIC, 6, struct pacc_zluda_ddr_info)
#define PACC_IOC_GET_PACC_ID _IOR(PACC_IOC_MAGIC, 7, unsigned long)
/*
 * Keep /dev/mbox as an interrupt-only transport.  Older bring-up builds tried
 * a factory nr=2 control path here, but that path is not validated on the
 * current firmware and can hide broken shared-DDR visibility.
 */

#define PACC_ELF_ET_REL 1U
#define PACC_ELF_ET_EXEC 2U
#define PACC_ELF_ET_DYN 3U
#define PACC_ELF_PT_LOAD 1U
#define PACC_ELF_SHT_SYMTAB 2U
#define PACC_ELF_SHT_STRTAB 3U
#define PACC_ELF_SHT_RELA 4U
#define PACC_ELF_SHT_DYNSYM 11U
#define PACC_ELF_STT_NOTYPE 0U
#define PACC_ELF_STT_FUNC 2U
#define PACC_R_RISCV_NONE 0U
#define PACC_R_RISCV_64 2U
#define PACC_R_RISCV_RELATIVE 3U
#define PACC_R_RISCV_JUMP_SLOT 5U

#define PACC_FNV64_OFFSET 0xcbf29ce484222325ULL
#define PACC_FNV64_PRIME 0x100000001b3ULL

#if defined(__riscv_vector)
#define PACC_RVV_UNUSED __attribute__((unused))
#else
#define PACC_RVV_UNUSED
#endif
#if defined(__GNUC__) || defined(__clang__)
#define PACC_UNUSED __attribute__((unused))
#else
#define PACC_UNUSED
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

struct JobdBeacon {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint32_t phase;
    uint32_t detail;
    uint64_t seq;
};

struct JobdDiagEvent {
    uint64_t magic;
    uint32_t index;
    uint32_t status;
    uint32_t job_id;
    uint32_t aux;
    uint64_t seq;
};

struct CompletionTelemetry {
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
};

_Static_assert(sizeof(struct CompletionTelemetry) == 88,
               "completion telemetry ABI must remain 88 bytes");

struct RmsNormDebugRecord {
    uint64_t magic;
    uint32_t version;
    uint32_t pacc_id;
    uint32_t phase;
    uint32_t dtype;
    uint64_t seq;
    uint64_t row;
    uint64_t rows;
    uint64_t hidden;
    uint64_t x_addr;
    uint64_t weight_addr;
    uint64_t y_addr;
    float eps;
    float sumsq;
    float mean;
    float scale;
    float x0;
    float w0;
    float y0;
    float x_last;
    float w_last;
    float y_last;
    uint32_t flags;
    uint32_t reserved;
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

struct PaccUint3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct MmvfJob {
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
    struct PaccUint3 nchannels_y;
    int32_t stride_row;
    int32_t stride_col_y2;
    int32_t stride_col_dst;
    struct PaccUint3 channel_ratio;
    int32_t stride_channel_x;
    int32_t stride_channel_y;
    int32_t stride_channel_dst;
    struct PaccUint3 sample_ratio;
    int32_t stride_sample_x;
    int32_t stride_sample_y;
    int32_t stride_sample_dst;
    int32_t ids_stride;
};

struct PreloadedJobs {
    uint64_t runtime_seq;
    bool have_gemm;
    bool have_softmax;
    bool have_rmsnorm;
    bool have_allreduce;
    bool have_mmvf;
    struct GemmJob gemm;
    struct SoftmaxJob softmax;
    struct RmsNormJob rmsnorm;
    struct AllReduceJob allreduce;
    struct MmvfJob mmvf;
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
    uint32_t have_mmvf;
    uint32_t reserved0;
    struct GemmJob gemm;
    struct SoftmaxJob softmax;
    struct RmsNormJob rmsnorm;
    struct AllReduceJob allreduce;
    struct MmvfJob mmvf;
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
    uint32_t kernel_name_offset;
    uint32_t kernel_name_size;
};

struct PaccKernelArgRecord {
    uint32_t kind;
    uint32_t size;
    uint32_t flags;
    uint32_t reserved;
    uint64_t value;
    uint64_t value_hi;
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
    struct PaccKernelLaunchAbiHeader abi_storage;
    const struct PaccKernelLaunchAbiHeader *abi;
    struct PaccKernelArgRecord arg_records_storage[PACC_MAX_KERNEL_ARGS];
    const struct PaccKernelArgRecord *arg_records;
    size_t arg_count;
    struct PaccKernelBufferBinding bindings_storage[PACC_MAX_KERNEL_BINDINGS];
    const struct PaccKernelBufferBinding *bindings;
    size_t binding_count;
    const uint8_t *raw_params;
    size_t raw_param_size;
    const char *kernel_name;
    size_t kernel_name_size;
};

struct Map {
    void *base;
    size_t map_len;
    void *ptr;
    bool copied;
    bool borrowed;
    int fd;
    uint64_t phys;
    size_t len;
};

typedef void (*PaccSetLaunchFn)(uint32_t, uint32_t, uint32_t,
                                uint32_t, uint32_t, uint32_t,
                                uint32_t, uint32_t, uint32_t,
                                uint32_t, uint32_t, uint32_t);

struct KernelBindingMap {
    struct Map map;
    uint32_t arg_index;
    uint32_t flags;
};

struct KernelParamCell {
    uint64_t lo;
    uint64_t hi;
};

struct LoadedKernelImage {
    bool direct;
    void *handle;
    void *mapping;
    size_t map_len;
    void *fn;
    PaccSetLaunchFn set_launch;
};

enum DispatchPollResult {
    DISPATCH_INVALID = 0,
    DISPATCH_IDLE = 1,
    DISPATCH_HANDLED = 2,
};

static long g_page_size = 4096;
static struct pacc_zluda_ddr_info g_ddr_info;
static uint64_t g_shared_ddr_pacc_base;
static uint64_t g_pacc_id;
static int g_mbox_fd = -1;
static int g_shared_ddr_data_fd = -1;
static bool g_map_uses_shared_ddr_offsets;
static uint64_t g_shared_ddr_mmap_user_off;
static uint64_t g_shared_ddr_fd_user_off;
static uint64_t g_shared_ddr_control_base_off;
static struct Map g_shared_ddr_full_map;
static bool g_shared_ddr_full_map_valid;
static struct Map g_kernel_slot_map;
static bool g_kernel_slot_map_valid;
static volatile uint8_t *g_control_window;
static void *g_control_map_base;
static size_t g_control_map_len;
static uint32_t g_last_arg_header_candidate;
static uint32_t g_pending_arg_header_candidate;
static uint32_t g_pending_arg_header_pacc_id;
static const char *g_current_kernel_symbol = NULL;
static uint64_t g_current_kernel_seq = 0;
static uint32_t g_kernel_load_error = 0;
static uint32_t g_kernel_arg_error = 0;
static uint32_t g_kernel_parse_error = 0;
static uint8_t g_control_snapshot[HETGPU_PACC_CONTROL_BYTES];
static uint8_t g_arg_slot_synthetic_control[HETGPU_PACC_CONTROL_BYTES];
static uint64_t g_last_preloaded_seq_by_job[HETGPU_PACC_MAX_JOB_ID];
static uint32_t g_diag_ring_index;
static bool g_response_irq_pending;
static bool g_kernel_completion_beacon_sticky;
static uint64_t g_kernel_completion_beacon_seq;

static bool phys_is_shared_ddr(uint64_t phys, size_t len);
static uint64_t shared_ddr_pacc_phys(uint64_t phys, size_t len);
static uint32_t g_kernel_completion_beacon_status;
static bool g_preloaded_completion_sticky;
static uint32_t g_preloaded_completion_job_id;
static uint64_t g_preloaded_completion_seq;
static uint32_t g_preloaded_completion_status;

static void mirror_host_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status);
static void mirror_job_completion(int fd, uint32_t job_id, uint64_t seq,
                                  uint32_t status);
static bool mirror_boot_marker_all_slots_mbox_mmap(int fd, uint32_t status);
static bool mirror_host_status_mbox_mmap(int fd, uint32_t job_id, uint64_t seq, uint32_t status);
static void mirror_diag_event(int fd, uint32_t job_id, uint64_t seq, uint32_t status, uint32_t aux);
static void mirror_rmsnorm_debug_record(int fd, const struct RmsNormDebugRecord *record);
static void mirror_aligned_completion_record(int fd, uint32_t job_id, uint64_t seq, uint32_t status);
static void mirror_diag_progress_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status);
static void mirror_progress_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status);
static void write_jobd_beacon(int fd, uint32_t job_id, uint64_t seq, uint32_t phase, uint32_t detail);
static bool mirror_host_status_control_window_direct(uint32_t job_id, uint64_t seq, uint32_t status);
static int write_phys_copy_pwrite_only(int fd, uint64_t phys, const void *src, size_t len);
static int write_phys_copy(int fd, uint64_t phys, const void *src, size_t len);
static bool write_shared_ddr_devmem_direct(uint64_t phys, const void *src, size_t len);
static bool write_shared_ddr_fd_mmap_direct(int fd, uint64_t phys, const void *src, size_t len);
static bool read_shared_ddr_devmem_direct(uint64_t phys, void *dst, size_t len);
static bool read_shared_ddr_fd_mmap_direct(int fd, uint64_t phys, void *dst, size_t len);
static bool read_shared_ddr_host_devmem_direct(uint64_t phys, void *dst, size_t len);
static int read_shared_ddr_control_copy_pread(int fd, uint64_t off, size_t len, uint8_t **out);
static bool arg_slot_fast_peek_direct(uint32_t job_id);
static enum DispatchPollResult maybe_dispatch_gemm_arg_slot_direct(int fd,
                                                                   uint64_t *last_seq);
static const char *job_name(uint32_t job_id);
static void jobd_io_fence(void);
static int map_phys(int fd, uint64_t phys, size_t len, struct Map *out);
static void unmap_phys(struct Map *m);
static void sleep_us(uint64_t usec);
static uint64_t parse_env_u64_default(const char *name, uint64_t fallback);
static bool aligned_completion_record_enabled(void);

static bool env_flag_true(const char *value) {
    return value && *value && strcmp(value, "0") != 0 &&
           strcasecmp(value, "false") != 0 &&
           strcasecmp(value, "off") != 0 &&
           strcasecmp(value, "no") != 0;
}

static bool env_flag_default_true(const char *name) {
    const char *value = getenv(name);
    if (!value || !*value) {
        return true;
    }
    return env_flag_true(value);
}

static bool jobd_trace_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_TRACE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_kmsg_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_KMSG");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_log_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_LOG");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_mbox_poll_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_MBOX_POLL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_initial_mbox_poll_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_INITIAL_MBOX_POLL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_mbox_control_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_MBOX_CONTROL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return jobd_mbox_poll_enabled();
}

static bool jobd_ddr_ioctl_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_DDR_IOCTL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_force_elf_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_FORCE_ELF")) ||
           env_flag_true(getenv("HETGPU_PACC_JOBD_DISABLE_NATIVE"));
}

static bool jobd_disable_native_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_DISABLE_NATIVE"));
}

static bool jobd_force_elf_for_symbol(const char *symbol) {
    if (jobd_disable_native_enabled()) {
        return true;
    }
    if (!env_flag_true(getenv("HETGPU_PACC_JOBD_FORCE_ELF"))) {
        return false;
    }
    return symbol && strstr(symbol, "k_bin_bcast");
}

static bool jobd_elf_fallback_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_ELF_FALLBACK");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_generic_noop_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_GENERIC_NOOP");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_preloaded_noop_enabled(uint32_t job_id) {
    const char *all = getenv("HETGPU_PACC_JOBD_PRELOADED_NOOP");
    const char *gemm = getenv("HETGPU_PACC_JOBD_GEMM_NOOP");

    if (all && *all && env_flag_true(all)) {
        return true;
    }
    return job_id == HETGPU_PACC_JOB_GEMM && gemm && *gemm && env_flag_true(gemm);
}

static bool jobd_gemm_tiled_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_GEMM_TILED");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_gemm_copy_io_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_GEMM_COPY_IO");
    if (!value || !*value) {
        value = getenv("PACC_GCI");
    }
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_shared_ddr_payload_pread_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_SHARED_DDR_PAYLOAD_PREAD");
    if (value && *value) {
        return env_flag_true(value);
    }
    /*
     * AP-published shared-DDR payloads can be stale through the PACC-side
     * /dev/mem alias until the platform cache path is nailed down.  Prefer the
     * fd-backed fresh read by default so strict GEMM/MMVF sees the same bytes the
     * AP mmap probe sees.  This is a correctness default; hot paths can opt out
     * once cache invalidation is verified.
     */
    return true;
}

static bool jobd_shared_ddr_payload_pwrite_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_SHARED_DDR_PAYLOAD_PWRITE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_shared_ddr_payload_sync_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_SHARED_DDR_PAYLOAD_SYNC");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_clear_stale_control_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CLEAR_STALE_CONTROL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_full_ddr_map_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_FULL_DDR_MAP");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static uint64_t jobd_full_ddr_map_bytes(void) {
    uint64_t requested = parse_env_u64_default("HETGPU_PACC_JOBD_FULL_DDR_MAP_BYTES", 0);
    if (requested != 0) {
        return requested;
    }
    if (g_ddr_info.ddr_size != 0) {
        return g_ddr_info.ddr_size;
    }
    return HETGPU_PACC_DEFAULT_SHARED_DDR_BYTES;
}

static uint64_t jobd_kernel_slot_map_bytes(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_SLOT_MAP_BYTES",
                                 0x04000000ULL);
}

static uint64_t jobd_kernel_slot_map_off(uint64_t map_bytes) {
    uint64_t requested = parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_SLOT_MAP_OFF",
                                               UINT64_MAX);
    if (requested != UINT64_MAX) {
        return requested;
    }
    if (g_ddr_info.ddr_size > map_bytes) {
        return g_ddr_info.ddr_size - map_bytes;
    }
    return 0;
}

static bool jobd_kernel_slot_map_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_KERNEL_SLOT_MAP");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_force_pread_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_FORCE_PREAD");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_status_pwrite_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_STATUS_PWRITE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_status_control_window_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_STATUS_CONTROL_WINDOW");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_status_mmap_fallback_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_STATUS_MMAP_FALLBACK");
    if (value && *value) {
        return env_flag_true(value);
    }
#if HETGPU_PACC_COMPLETION_TELEMETRY
    return false;
#else
    return true;
#endif
}

static bool jobd_mbox_status_mmap_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_MBOX_STATUS_MMAP");
    return value && *value && env_flag_true(value);
}

static bool jobd_control_window_read_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CONTROL_WINDOW_READ");
    if (value && *value) {
        return env_flag_true(value);
    }
    /*
     * The host owns the shared-DDR control page updates.  Reusing the
     * long-lived PACC-side mapping can lag behind the IRQ that announced the
     * update, which leaves runtime-table and arg-slot reads spinning on stale
     * bytes.  Prefer a fresh synchronized read unless explicitly requested.
     */
    return true;
}

static bool jobd_control_pread_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CONTROL_PREAD");
    if (value && *value) {
        return env_flag_true(value);
    }
    /*
     * The host publishes arg-slots in shared DDR.  On the current LX500 path,
     * the PACC-side /dev/mem alias can keep a stale cacheline even while AP
     * /dev/pacc mmap shows the new header.  Use the fd-backed read as the
     * default control read and keep /dev/mem as fallback.
     */
    return true;
}

static bool jobd_kernel_metadata_first_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_KERNEL_METADATA_FIRST");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_arg_slot_scan_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_ARG_SLOT_SCAN");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_arg_slot_scan_all_pacc_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_ARG_SLOT_SCAN_ALL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_full_control_snapshot_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_FULL_CONTROL_SNAPSHOT"));
}

static bool jobd_redispatch_seen_arg_slot_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_REDISPATCH_SEEN_ARG_SLOT");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_runtime_table_refresh_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_REFRESH_RUNTIME_TABLE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_progress_status_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_PROGRESS_STATUS"));
}

static bool jobd_progress_completion_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_PROGRESS_COMPLETION"));
}

static bool jobd_loop_trace_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_LOOP_TRACE"));
}

static bool jobd_skip_irq_poll_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_SKIP_IRQ_POLL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_wait_for_control_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_WAIT_FOR_CONTROL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_sticky_kernel_completion_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_STICKY_KERNEL_COMPLETION"));
}

static bool jobd_sticky_preloaded_completion_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_STICKY_PRELOADED_COMPLETION"));
}

static bool jobd_require_completion_visible_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_REQUIRE_COMPLETION_VISIBLE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_diag_ring_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_DIAG_RING"));
}

static bool jobd_rms_debug_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_RMS_DEBUG");
    if (value && *value) {
        return env_flag_true(value);
    }
    return jobd_diag_ring_enabled();
}

static bool jobd_rms_output_pwrite_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_RMS_OUTPUT_PWRITE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_beacon_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_BEACON");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_static_config_fallback_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_STATIC_CONFIG_FALLBACK"));
}

static uint64_t jobd_arg_wait_us(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_ARG_WAIT_US", 1000000ULL);
}

static bool jobd_force_devmem_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_FORCE_DEVMEM");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_cbo_inval_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CBO_INVAL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_cbo_flush_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CBO_FLUSH");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static size_t jobd_cbo_block_bytes(void) {
    uint64_t value = parse_env_u64_default("HETGPU_PACC_JOBD_CBO_BLOCK_BYTES", 64);
    if (value < 16) {
        value = 16;
    }
    if (value > 4096) {
        value = 4096;
    }
    return (size_t)value;
}

static bool jobd_cbo_flush_opcode_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CBO_OP");
    return value && (!strcmp(value, "flush") || !strcmp(value, "2"));
}

static bool jobd_xthead_dcache_cva_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CBO_OP");
    return value && (!strcmp(value, "xthead-cva") ||
                     !strcmp(value, "thead-cva") ||
                     !strcmp(value, "cva") ||
                     !strcmp(value, "3"));
}

static bool jobd_xthead_dcache_civa_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CBO_OP");
    return value && (!strcmp(value, "xthead-civa") ||
                     !strcmp(value, "thead-civa") ||
                     !strcmp(value, "civa") ||
                     !strcmp(value, "4"));
}

static bool jobd_msync_enabled(void) {
    return false;
}

static bool jobd_drain_after_write_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_DRAIN_AFTER_WRITE");
    return value && env_flag_true(value);
}

static size_t jobd_drain_after_write_stride(void) {
    uint64_t value = parse_env_u64_default("HETGPU_PACC_JOBD_DRAIN_AFTER_WRITE_STRIDE", 64);
    if (value == 0) value = 1;
    if (value > 4096) value = 4096;
    return (size_t)value;
}

static volatile uint8_t g_drain_after_write_sink;

static void jobd_drain_after_write(const void *ptr, size_t len) {
    if (!jobd_drain_after_write_enabled() || !ptr || !len) {
        return;
    }
    const volatile uint8_t *p = (const volatile uint8_t *)ptr;
    size_t stride = jobd_drain_after_write_stride();
    uint8_t acc = 0;
    jobd_io_fence();
    for (size_t off = 0; off < len; off += stride) {
        acc ^= p[off];
    }
    acc ^= p[len - 1];
    g_drain_after_write_sink ^= acc;
    jobd_io_fence();
}

static void jobd_evict_after_payload_write(void) {
    static uint8_t *evict_buf;
    static size_t evict_len;
    uint64_t requested = parse_env_u64_default("HETGPU_PACC_JOBD_EVICT_AFTER_WRITE_BYTES", 0);

    if (requested == 0) {
        return;
    }
    if (requested > (uint64_t)SIZE_MAX) {
        requested = (uint64_t)SIZE_MAX;
    }
    if (!evict_buf || evict_len < (size_t)requested) {
        free(evict_buf);
        evict_len = (size_t)requested;
        evict_buf = (uint8_t *)malloc(evict_len);
        if (!evict_buf) {
            evict_len = 0;
            return;
        }
        memset(evict_buf, 0x5a, evict_len);
    }

    for (size_t i = 0; i < evict_len; i += 64) {
        evict_buf[i] = (uint8_t)(evict_buf[i] + 1u);
    }
#if defined(__riscv)
    __asm__ volatile("fence iorw, iorw" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static bool jobd_status_msync_enabled(void) {
    return false;
}

static bool jobd_rms_local_copy_enabled(void) {
    const char *value = getenv("PACC_JOBD_RMS_LOCAL_COPY");
    if (value && *value) {
        return env_flag_true(value);
    }
    value = getenv("HETGPU_PACC_JOBD_RMS_LOCAL_COPY");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_softmax_local_copy_enabled(void) {
    const char *value = getenv("PACC_JOBD_SOFTMAX_LOCAL_COPY");
    if (value && *value) {
        return env_flag_true(value);
    }
    value = getenv("HETGPU_PACC_JOBD_SOFTMAX_LOCAL_COPY");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_rms_rvv_enabled(void) {
    if (!env_flag_true(getenv("HETGPU_PACC_JOBD_RMS_RVV_FORCE"))) {
        return false;
    }
    const char *value = getenv("HETGPU_PACC_JOBD_RMS_RVV");
    if (value && *value) {
        return env_flag_true(value);
    }
    value = getenv("PACC_JOBD_RMS_RVV");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_mmvf_copy_io_enabled(void) {
    const char *value = getenv("PACC_JOBD_MMVF_COPY_IO");
    if (value && *value) {
        return env_flag_true(value);
    }
    value = getenv("HETGPU_PACC_JOBD_MMVF_COPY_IO");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_mmvf_compute_enabled(void) {
    const char *value = getenv("PACC_JOBD_MMVF_COMPUTE");
    if (value && *value) {
        return env_flag_true(value);
    }
    value = getenv("HETGPU_PACC_JOBD_MMVF_COMPUTE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_mmvf_clear_dst_enabled(void) {
    const char *value = getenv("PACC_JOBD_MMVF_CLEAR_DST");
    if (value && *value) {
        return env_flag_true(value);
    }
    value = getenv("HETGPU_PACC_JOBD_MMVF_CLEAR_DST");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static unsigned jobd_mmvf_worker_threads(uint64_t work_items) {
    uint64_t requested = parse_env_u64_default("PACC_JOBD_MMVF_THREADS", 0);
    if (requested == 0) {
        requested = parse_env_u64_default("HETGPU_PACC_JOBD_MMVF_THREADS", 0);
    }
    if (requested == 0) {
        requested = parse_env_u64_default("PACC_JOBD_KERNEL_THREADS", 0);
    }
    if (requested == 0) {
        requested = parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_THREADS", 0);
    }
    if (requested == 0) {
        requested = PACC_KERNEL_DEFAULT_THREADS;
    }
    if (requested > PACC_KERNEL_MAX_THREADS) {
        requested = PACC_KERNEL_MAX_THREADS;
    }
    if (requested > work_items && work_items != 0) {
        requested = work_items;
    }
    return requested == 0 ? 1 : (unsigned)requested;
}

static bool jobd_heartbeat_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_HEARTBEAT");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_boot_marker_enabled(void) {
    return env_flag_true(getenv("HETGPU_PACC_JOBD_BOOT_MARKER"));
}

static size_t jobd_helper_io_chunk_bytes(void) {
    uint64_t value = parse_env_u64_default("HETGPU_PACC_JOBD_HELPER_IO_CHUNK_BYTES", 4096);
    if (value == 0) {
        return 1;
    }
    if (value > (uint64_t)SIZE_MAX) {
        return SIZE_MAX;
    }
    return (size_t)value;
}

static void jobd_cbo_inval_line(const void *ptr) {
#if defined(__riscv)
    if (jobd_xthead_dcache_civa_enabled()) {
        __asm__ volatile(".insn r 0x0b, 0, 0x01, x0, %0, x7" :: "r"(ptr) : "memory");
        return;
    }
    __asm__ volatile(".insn i 15, 2, x0, %0, 0" :: "r"(ptr) : "memory");
#else
    (void)ptr;
#endif
}

static void jobd_cbo_flush_line(const void *ptr) {
#if defined(__riscv)
    if (jobd_xthead_dcache_civa_enabled()) {
        __asm__ volatile(".insn r 0x0b, 0, 0x01, x0, %0, x7" :: "r"(ptr) : "memory");
        return;
    }
    if (jobd_xthead_dcache_cva_enabled()) {
        __asm__ volatile(".insn r 0x0b, 0, 0x01, x0, %0, x5" :: "r"(ptr) : "memory");
        return;
    }
    if (jobd_cbo_flush_opcode_enabled()) {
        __asm__ volatile(".insn i 15, 2, x0, %0, 2" :: "r"(ptr) : "memory");
    } else {
        __asm__ volatile(".insn i 15, 2, x0, %0, 1" :: "r"(ptr) : "memory");
    }
#else
    (void)ptr;
#endif
}

static void jobd_xthead_sync_s(void) {
#if defined(__riscv)
    if (jobd_xthead_dcache_cva_enabled() || jobd_xthead_dcache_civa_enabled()) {
        __asm__ volatile(".insn r 0x0b, 0, 0x00, x0, x0, x25" ::: "memory");
    }
#endif
}

static void jobd_io_fence(void) {
#if defined(__riscv)
    __asm__ volatile("fence iorw, iorw" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static void jobd_invalidate_for_cpu(const void *ptr, size_t len) {
    uintptr_t start;
    uintptr_t end;

    if (!jobd_cbo_inval_enabled() || !ptr || !len) {
        return;
    }

    size_t block = jobd_cbo_block_bytes();
    start = (uintptr_t)ptr & ~((uintptr_t)block - 1u);
    end = ((uintptr_t)ptr + len + block - 1u) & ~((uintptr_t)block - 1u);
    __sync_synchronize();
    for (uintptr_t p = start; p < end; p += block) {
        jobd_cbo_inval_line((const void *)p);
    }
    jobd_xthead_sync_s();
    __sync_synchronize();
}

static void jobd_flush_for_device(const void *ptr, size_t len) {
    uintptr_t start;
    uintptr_t end;

    if (!jobd_cbo_flush_enabled() || !ptr || !len) {
        return;
    }

    size_t block = jobd_cbo_block_bytes();
    start = (uintptr_t)ptr & ~((uintptr_t)block - 1u);
    end = ((uintptr_t)ptr + len + block - 1u) & ~((uintptr_t)block - 1u);
    __sync_synchronize();
    for (uintptr_t p = start; p < end; p += block) {
        jobd_cbo_flush_line((const void *)p);
    }
    jobd_xthead_sync_s();
    __sync_synchronize();
}

static bool jobd_notify_irq_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_NOTIFY_IRQ");
    if (value && *value) {
        return env_flag_true(value);
    }
    return jobd_mbox_poll_enabled();
}

static bool jobd_seed_current_jobs_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_SEED_CURRENT_JOBS");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_claim_pacc_id_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_CLAIM_ID");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_pacc_id_ioctl_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_PACC_ID_IOCTL");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static bool jobd_xsfmm_smoke_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_XSFMM_SMOKE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static uint64_t jobd_xsfmm_smoke_timeout_ms(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_XSFMM_SMOKE_TIMEOUT_MS", 200ULL);
}

static bool jobd_startup_xsfmm_smoke_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_STARTUP_XSFMM_SMOKE");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static bool jobd_xsfmm_gemm_requested(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_XSFMM_GEMM");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static uint64_t jobd_xsfmm_gemm_max_n(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_XSFMM_MAX_N", 32ULL);
}

static uint64_t jobd_xsfmm_gemm_max_m(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_XSFMM_MAX_M", 32ULL);
}

static uint64_t jobd_xsfmm_repeats(void) {
    uint64_t repeats =
        parse_env_u64_default("HETGPU_PACC_JOBD_XSFMM_REPEATS", 1ULL);
    return repeats > 4096ULL ? 4096ULL : repeats;
}

static bool jobd_bf16_skinny_copy_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_BF16_SKINNY_COPY");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static void jobd_apply_xsfmm_layout_env(void) {
    const char *b_pack = getenv("HETGPU_PACC_JOBD_XSFMM_B_TRANSPOSED_PACK");
    const char *c_pack = getenv("HETGPU_PACC_JOBD_XSFMM_C_TRANSPOSED_PACK");

    if (b_pack && *b_pack) {
        g_xsfmm_b_transposed_pack = env_flag_true(b_pack);
    }
    if (c_pack && *c_pack) {
        g_xsfmm_c_transposed_pack = env_flag_true(c_pack);
    }
}

static bool jobd_xsfmm_gemm_enabled(void) {
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    return jobd_xsfmm_gemm_requested() && g_xsfmm_context_ready &&
           g_xsfmm_bf16_checked && g_xsfmm_bf16_usable;
#else
    return false;
#endif
}

static void emit_msg(const char *fmt, va_list ap) {
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    fprintf(stderr, "hetgpu_pacc_jobd: %s\n", buf);
    if (jobd_kmsg_enabled()) {
        int kmsg = open("/dev/kmsg", O_WRONLY | O_CLOEXEC);
        if (kmsg >= 0) {
            dprintf(kmsg, "hetgpu_pacc_jobd: %s\n", buf);
            close(kmsg);
        }
    }
}

static void log_msg(const char *fmt, ...) {
    va_list ap;
    if (!jobd_log_enabled()) {
        return;
    }
    va_start(ap, fmt);
    emit_msg(fmt, ap);
    va_end(ap);
}

static int jobd_enable_xsfmm_context(void) {
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    cpu_set_t set;
    int cpu = sched_getcpu();
    int module_fd;

    if (g_xsfmm_context_ready) {
        return 0;
    }
    if (cpu < 0) {
        int saved_errno = errno;
        log_msg("xsfmm hardware-only init failed: sched_getcpu: %s", strerror(errno));
        return -saved_errno;
    }
    CPU_ZERO(&set);
    CPU_SET((unsigned int)cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        int saved_errno = errno;
        log_msg("xsfmm hardware-only init failed: pin cpu%d: %s", cpu, strerror(errno));
        return -saved_errno;
    }

    module_fd = open("/home/root/xsfmm_ctx.ko", O_RDONLY | O_CLOEXEC);
    if (module_fd < 0) {
        log_msg("xsfmm hardware-only init failed: open module: %s", strerror(errno));
        return -0x41;
    }
    if (syscall(SYS_finit_module, module_fd, "", 0) != 0 && errno != EEXIST) {
        int saved_errno = errno;
        close(module_fd);
        log_msg("xsfmm hardware-only init failed: finit_module: %s",
                strerror(saved_errno));
        return -0x42;
    }
    close(module_fd);
    g_xsfmm_run_fd =
        open("/sys/module/xsfmm_ctx/parameters/run", O_WRONLY | O_CLOEXEC);
    if (g_xsfmm_run_fd < 0) {
        log_msg("xsfmm hardware-only init failed: open run control: %s",
                strerror(errno));
        return -0x45;
    }
    g_xsfmm_context_ready = true;
    return 0;
#else
    log_msg("xsfmm hardware-only init failed: binary lacks Xsfmm32a16f support");
    return -1;
#endif
}

static int jobd_run_xsfmm_request(struct XsfmmRequest *request) {
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    char address[32];
    int address_len;

    if (!request || !g_xsfmm_context_ready || g_xsfmm_run_fd < 0) {
        return -0x45;
    }
    address_len = snprintf(address, sizeof(address), "0x%" PRIxPTR,
                           (uintptr_t)request);
    if (address_len <= 0 || (size_t)address_len >= sizeof(address)) {
        return -0x48;
    }
    if (lseek(g_xsfmm_run_fd, 0, SEEK_SET) < 0) {
        return -0x49;
    }
    request->status = -1;
    request->cycles = 0;
    request->completed_repeats = 0;
    __sync_synchronize();
    if (write(g_xsfmm_run_fd, address, (size_t)address_len) != address_len) {
        return -0x4a;
    }
    __sync_synchronize();
    if (request->status != 0 ||
        request->completed_repeats != request->repeats) {
        return -0x4b;
    }
    if (request->repeats > 1) {
        log_msg("xsfmm batch repeats=%" PRIu64 " cycles=%" PRIu64,
                request->completed_repeats, request->cycles);
    }
    return 0;
#else
    return -1;
#endif
}

static void trace_msg(const char *fmt, ...) {
    if (!jobd_trace_enabled()) return;
    va_list ap;
    va_start(ap, fmt);
    emit_msg(fmt, ap);
    va_end(ap);
}

static void jobd_crash_handler(int sig, siginfo_t *info, void *uctx) {
    void *frames[64];
    int nframes;
    char buf[512];
    uintptr_t pc = 0;
#if defined(__riscv) && defined(REG_PC)
    if (uctx) {
        ucontext_t *uc = (ucontext_t *)uctx;
        pc = (uintptr_t)uc->uc_mcontext.__gregs[REG_PC];
    }
#endif
    int len = snprintf(buf, sizeof(buf),
                       "hetgpu_pacc_jobd: fatal signal %d addr=%p pc=0x%" PRIxPTR
                       " while running seq=%" PRIu64 " symbol=%s\n",
                       sig, info ? info->si_addr : NULL, pc, g_current_kernel_seq,
                       g_current_kernel_symbol ? g_current_kernel_symbol : "<none>");
    if (len > 0) {
        write(STDERR_FILENO, buf, (size_t)len);
    }
    nframes = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
    backtrace_symbols_fd(frames, nframes, STDERR_FILENO);
    _exit(128 + sig);
}

static void install_crash_handlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = jobd_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_NODEFER;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
}

static bool jobd_fork_elf_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_FORK_ELF");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static uint64_t jobd_fork_elf_timeout_ms(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_FORK_ELF_TIMEOUT_MS", 5000ULL);
}

static int control_poll_timeout_ms(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_POLL_TIMEOUT_MS");
    char *end = NULL;
    long parsed;
    if (!value || !*value) {
        return 1;
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

static unsigned poll_irq_settle_us(void) {
    uint64_t value = parse_env_u64_default("HETGPU_PACC_JOBD_POLL_IRQ_SETTLE_US", 1000);
    if (value > UINT_MAX) {
        return UINT_MAX;
    }
    return (unsigned)value;
}

static int idle_sleep_us(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_IDLE_SLEEP_US");
    char *end = NULL;
    long parsed;
    if (!value || !*value) {
        return 1000;
    }
    errno = 0;
    parsed = strtol(value, &end, 0);
    if (errno || end == value) {
        return 1000;
    }
    if (parsed < 0) {
        return 0;
    }
    if (parsed > 1000000) {
        return 1000000;
    }
    return (int)parsed;
}

static void sleep_when_idle(void) {
    int usec = idle_sleep_us();
    struct timespec ts;
    if (usec <= 0) {
        return;
    }
    ts.tv_sec = usec / 1000000;
    ts.tv_nsec = (long)(usec % 1000000) * 1000L;
    while (nanosleep(&ts, &ts) != 0) {
        if (errno != EINTR) {
            return;
        }
    }
}

static void sleep_us(uint64_t usec) {
    struct timespec ts;
    ts.tv_sec = (time_t)(usec / 1000000ULL);
    ts.tv_nsec = (long)(usec % 1000000ULL) * 1000L;
    while (nanosleep(&ts, &ts) != 0) {
        if (errno != EINTR) {
            return;
        }
    }
}

static uint64_t monotonic_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
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
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_base", &base);
    }
    if (!base) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_base", &base);
    }
    if (!base) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_base", &base);
    }
    if (!base) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base", &base);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_bytes", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_size", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_bytes", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_size", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_bytes", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_size", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_bytes", &size);
    }
    if (!size) {
        read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_size", &size);
    }
    if (!base) {
        base = 0x20100600000ULL;
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

static int notify_zluda_irq(int mbox_fd) {
    static bool warned_unsupported;
    struct pacc_zluda_ddr_info irq_info = g_ddr_info;
    jobd_io_fence();
    int ret = ioctl(mbox_fd, PACC_IOC_ZLUDA_IRQ);
    int saved_errno;
    if (ret == 0) {
        jobd_io_fence();
        return 0;
    }

    saved_errno = errno;
    if (saved_errno == ENOTTY || saved_errno == EINVAL) {
        ret = ioctl(mbox_fd, PACC_IOC_ZLUDA_IRQ_WITH_DDR, &irq_info);
        if (ret == 0) {
            jobd_io_fence();
            return 0;
        }
        saved_errno = errno;
    }

    if (saved_errno == ENOTTY || saved_errno == ENOSYS || saved_errno == EOPNOTSUPP) {
        if (!warned_unsupported) {
            log_msg("ZLUDA IRQ ioctl unsupported on mbox fd; continuing with shared-DDR polling mock");
            warned_unsupported = true;
        }
        return 0;
    }

    errno = saved_errno;
    return -1;
}

static void submit_mbox_payload_sync(int mbox_fd,
                                     uint32_t job_id,
                                     uint64_t seq,
                                     uint32_t status,
                                     const char *where) {
    if (!where ||
        strstr(where, "completion") == NULL ||
        strstr(where, "before-completion") != NULL) {
        jobd_io_fence();
        return;
    }
    jobd_io_fence();
    if (mbox_fd < 0 || !jobd_notify_irq_enabled()) {
        return;
    }
    if (notify_zluda_irq(mbox_fd) != 0) {
        trace_msg("notify IRQ failed after %s: job_id=%u/%s seq=%" PRIu64
                  " status=0x%x errno=%d",
                  where ? where : "completion",
                  job_id, job_name(job_id), seq, status, errno);
    } else {
        trace_msg("notify IRQ after %s: job_id=%u/%s seq=%" PRIu64
                  " status=0x%x",
                  where ? where : "completion",
                  job_id, job_name(job_id), seq, status);
    }
    jobd_io_fence();
}

static bool wait_for_control_impl(int mbox_fd, bool required) {
    if (!required && !jobd_mbox_poll_enabled()) {
        return false;
    }

    for (;;) {
        struct pollfd pfd;
        int ret;
        memset(&pfd, 0, sizeof(pfd));
        pfd.fd = mbox_fd;
        pfd.events = POLLIN;
        jobd_io_fence();
        ret = poll(&pfd, 1, control_poll_timeout_ms());
        jobd_io_fence();
        if (ret > 0) {
            if (pfd.revents & (POLLIN | POLLPRI)) {
                unsigned settle_us = poll_irq_settle_us();
                jobd_io_fence();
                if (settle_us != 0) {
                    usleep(settle_us);
                }
                jobd_io_fence();
                return true;
            }
            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                log_msg("buggy: poll revents 0x%x", pfd.revents);
                exit(EIO);
            }
            continue;
        }
        if (ret == 0) {
            if (required) {
                continue;
            }
            return false;
        }
        if (errno == EINTR || errno == EAGAIN) {
            continue;
        }
        log_msg("buggy: poll return %d", errno);
        exit(errno ? errno : EIO);
    }
}

static bool wait_for_control(int mbox_fd) {
    return wait_for_control_impl(mbox_fd, false);
}

static void wait_for_initial_control(int mbox_fd) {
    if (!jobd_initial_mbox_poll_enabled()) {
        return;
    }
    (void)wait_for_control_impl(mbox_fd, true);
}

static void read_shared_ddr_info_from_mbox(int mbox_fd) {
    struct pacc_zluda_ddr_info fallback_info;
    bool have_fallback;
    (void)mbox_fd;
    memset(&fallback_info, 0, sizeof(fallback_info));
    have_fallback = read_shared_ddr_info_from_env_or_debugfs(&fallback_info);
    if (have_fallback) {
        g_ddr_info = fallback_info;
        log_msg("shared ddr base 0x%" PRIx64 " size 0x%" PRIx64
                " from env/debugfs/default; skipped /dev/mbox DDR ioctl",
                g_ddr_info.ddr_base, g_ddr_info.ddr_size);
        return;
    }

    log_msg("invalid shared ddr info from env/debugfs/default; /dev/mbox DDR ioctl disabled");
    exit(EINVAL);
}

static bool set_pacc_id_checked(uint64_t id, const char *source) {
    if (id >= HETGPU_PACC_COUNT) {
        log_msg("invalid pacc id %" PRIu64 " from %s; expected 0..%u",
                id, source ? source : "unknown", HETGPU_PACC_COUNT - 1U);
        return false;
    }
    g_pacc_id = id;
    log_msg("pacc id %" PRIu64 " from %s", g_pacc_id,
            source ? source : "unknown");
    return true;
}

static void read_pacc_id_from_mbox(int mbox_fd) {
    const char *env = getenv("HETGPU_PACC_ID");
    const char *legacy_env = getenv("PACC_JOBD_PACC_ID");
    uint64_t parsed_id;
    unsigned long ioctl_id = 0;

    if (parse_u64_checked(env, &parsed_id) &&
        set_pacc_id_checked(parsed_id, "HETGPU_PACC_ID")) {
        return;
    }
    if (parse_u64_checked(legacy_env, &parsed_id) &&
        set_pacc_id_checked(parsed_id, "PACC_JOBD_PACC_ID")) {
        return;
    }

    if (mbox_fd >= 0 && jobd_pacc_id_ioctl_enabled()) {
        if (ioctl(mbox_fd, PACC_IOC_GET_PACC_ID, &ioctl_id) == 0 &&
            set_pacc_id_checked((uint64_t)ioctl_id,
                                "/dev/mbox PACC_IOC_GET_PACC_ID")) {
            return;
        }
        trace_msg("pacc id ioctl failed or invalid: errno=%d id=%lu",
                  errno, ioctl_id);
    }

    g_pacc_id = 0;
    log_msg("pacc id defaulting to pacc0");
}

static uint64_t shared_ddr_control_rel(uint64_t pacc_id, uint64_t off) {
    return g_shared_ddr_control_base_off +
           pacc_id * HETGPU_PACC_CONTROL_BYTES + off;
}

static bool claim_pacc_id_from_shared_ddr(int fd) {
    struct timespec ts = {0};
    uint64_t claim_token;

    if (!g_ddr_info.ddr_base ||
        g_ddr_info.ddr_size < HETGPU_PACC_COUNT * HETGPU_PACC_CONTROL_BYTES) {
        log_msg("cannot claim pacc id: shared DDR too small base=0x%" PRIx64
                " size=0x%" PRIx64,
                g_ddr_info.ddr_base, g_ddr_info.ddr_size);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts);
    claim_token = HETGPU_PACC_ID_CLAIM_MAGIC |
                  0x100ULL |
                  (((uint64_t)ts.tv_nsec ^ (uint64_t)(uintptr_t)&ts) & 0xffULL);

    for (uint32_t slot = 0; slot < HETGPU_PACC_COUNT; slot++) {
        struct Map map = {0};
        uint64_t phys = g_ddr_info.ddr_base +
            shared_ddr_control_rel(slot, HETGPU_PACC_ID_CLAIM_OFF);
        if (map_phys(fd, phys, sizeof(uint64_t), &map) != 0) {
            log_msg("claim pacc id: map slot %u phys=0x%" PRIx64 " failed: %s",
                    slot, phys, strerror(errno));
            continue;
        }
        volatile uint64_t *claim = (volatile uint64_t *)map.ptr;
        __sync_synchronize();
        if (*claim == 0ULL) {
            *claim = claim_token;
            __sync_synchronize();
            if (jobd_msync_enabled()) {
                msync(map.base, map.map_len, MS_SYNC);
            }
            jobd_io_fence();
            sleep_us(1000);
            if (*claim == claim_token) {
                unmap_phys(&map);
                g_pacc_id = slot;
                log_msg("pacc id %" PRIu64 " claimed from shared DDR slot %u via store",
                        g_pacc_id, slot);
                return true;
            }
        }
        trace_msg("pacc id claim slot %u already owned by 0x%" PRIx64,
                  slot, *claim);
        unmap_phys(&map);
    }

    log_msg("failed to claim pacc id from shared DDR; keeping pacc id %" PRIu64,
            g_pacc_id);
    return false;
}

static uint64_t shared_ddr_control_phys(uint64_t off, size_t len) {
    uint64_t control_off = shared_ddr_control_rel(g_pacc_id, off);
    if (g_ddr_info.ddr_base &&
        control_off <= g_ddr_info.ddr_size &&
        (uint64_t)len <= g_ddr_info.ddr_size - control_off) {
        return g_ddr_info.ddr_base + control_off;
    }
    log_msg("shared DDR control access out of range: pacc_id=%" PRIu64 " off=0x%" PRIx64 " len=%zu base=0x%" PRIx64 " size=0x%" PRIx64,
            g_pacc_id, off, len, g_ddr_info.ddr_base, g_ddr_info.ddr_size);
    exit(EINVAL);
}

static uint64_t parse_u64(const char *s) {
    return strtoull(s, NULL, 0);
}

static uint64_t parse_env_u64_default(const char *name, uint64_t fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long long parsed;
    if (!value || !*value) {
        return fallback;
    }
    errno = 0;
    parsed = strtoull(value, &end, 0);
    if (errno || end == value) {
        return fallback;
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
        end++;
    }
    if (*end) {
        return fallback;
    }
    return (uint64_t)parsed;
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
    case HETGPU_PACC_JOB_MMVF:
        return "MMVF";
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

static uint64_t hash_bytes_fnv64(const uint8_t *data, size_t len) {
    uint64_t hash = PACC_FNV64_OFFSET;
    if (!data) return hash;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)data[i];
        hash *= PACC_FNV64_PRIME;
    }
    return hash;
}

static int map_phys(int fd, uint64_t phys, size_t len, struct Map *out) {
    uint64_t page = (uint64_t)g_page_size;
    uint64_t mmap_addr = phys;
    if (phys_is_shared_ddr(phys, len) && g_shared_ddr_data_fd >= 0) {
        fd = g_shared_ddr_data_fd;
    }
    if (g_kernel_slot_map_valid && g_kernel_slot_map.phys &&
        phys >= g_kernel_slot_map.phys &&
        (uint64_t)len <= (uint64_t)g_kernel_slot_map.len &&
        phys - g_kernel_slot_map.phys <= (uint64_t)g_kernel_slot_map.len - (uint64_t)len) {
        uintptr_t ptr = (uintptr_t)((char *)g_kernel_slot_map.ptr +
                                    (phys - g_kernel_slot_map.phys));
        uintptr_t sync_base = ptr & ~((uintptr_t)page - 1u);
        size_t sync_off = (size_t)(ptr - sync_base);
        out->base = (void *)sync_base;
        out->map_len = ((sync_off + len + page - 1) / page) * page;
        out->ptr = (void *)ptr;
        out->borrowed = true;
        out->fd = fd;
        out->phys = phys;
        out->len = len;
        return 0;
    }
    if (g_shared_ddr_full_map_valid && g_ddr_info.ddr_base &&
        phys >= g_ddr_info.ddr_base &&
        (uint64_t)len <= (uint64_t)g_shared_ddr_full_map.len &&
        phys - g_ddr_info.ddr_base <= (uint64_t)g_shared_ddr_full_map.len - (uint64_t)len) {
        uintptr_t ptr = (uintptr_t)((char *)g_shared_ddr_full_map.ptr +
                                    (phys - g_ddr_info.ddr_base));
        uintptr_t sync_base = ptr & ~((uintptr_t)page - 1u);
        size_t sync_off = (size_t)(ptr - sync_base);
        out->base = (void *)sync_base;
        out->map_len = ((sync_off + len + page - 1) / page) * page;
        out->ptr = (void *)ptr;
        out->borrowed = true;
        out->fd = fd;
        out->phys = phys;
        out->len = len;
        return 0;
    }

    if (g_map_uses_shared_ddr_offsets && g_ddr_info.ddr_base && phys >= g_ddr_info.ddr_base) {
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        if ((uint64_t)len <= g_ddr_info.ddr_size &&
            ddr_off <= g_ddr_info.ddr_size - (uint64_t)len) {
            mmap_addr = g_shared_ddr_mmap_user_off + ddr_off;
            uint64_t base = mmap_addr & ~(page - 1);
            size_t off = (size_t)(mmap_addr - base);
            size_t map_len = ((off + len + page - 1) / page) * page;
            void *p = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
            if (p != MAP_FAILED) {
                out->base = p;
                out->map_len = map_len;
                out->ptr = (char *)p + off;
                out->fd = fd;
                out->phys = phys;
                out->len = len;
                return 0;
            }
        } else {
            return -1;
        }
    }

    mmap_addr = shared_ddr_pacc_phys(phys, len);
    uint64_t base = mmap_addr & ~(page - 1);
    size_t off = (size_t)(mmap_addr - base);
    size_t map_len = ((off + len + page - 1) / page) * page;
    void *p = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    if (p == MAP_FAILED) {
        return (int)0xffff5e01u;
    }
    out->base = p;
    out->map_len = map_len;
    out->ptr = (char *)p + off;
    out->fd = fd;
    out->phys = phys;
    out->len = len;
    return 0;
}

static void unmap_phys(struct Map *m);
static void sync_map_for_cpu(struct Map *m);

static int read_phys_mmap_copy(int fd, uint64_t phys, size_t len, uint8_t **out) {
    struct Map map = {0};
    uint8_t *buf;

    if (!out || len == 0) {
        return (int)0xffff5e02u;
    }
    if (map_phys(fd, phys, len, &map) != 0) {
        return (int)0xffff5e03u;
    }
    sync_map_for_cpu(&map);
    jobd_io_fence();
    buf = (uint8_t *)malloc(len);
    if (!buf) {
        unmap_phys(&map);
        return (int)0xffff5e04u;
    }
    memcpy(buf, map.ptr, len);
    jobd_io_fence();
    unmap_phys(&map);
    *out = buf;
    return 0;
}

static bool phys_to_fd_offset(uint64_t phys, size_t len, uint64_t *out) {
    if (!out) {
        return false;
    }
    if (g_ddr_info.ddr_base && phys >= g_ddr_info.ddr_base) {
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        if ((uint64_t)len <= g_ddr_info.ddr_size &&
            ddr_off <= g_ddr_info.ddr_size - (uint64_t)len) {
            if (jobd_force_devmem_enabled() &&
                !g_map_uses_shared_ddr_offsets &&
                g_shared_ddr_data_fd < 0) {
                *out = shared_ddr_pacc_phys(phys, len);
                return true;
            }
            *out = g_shared_ddr_fd_user_off + ddr_off;
            return true;
        }
        return false;
    }
    *out = phys;
    return true;
}

static bool phys_is_shared_ddr(uint64_t phys, size_t len) {
    if (!g_ddr_info.ddr_base || phys < g_ddr_info.ddr_base) {
        return false;
    }
    uint64_t ddr_off = phys - g_ddr_info.ddr_base;
    return (uint64_t)len <= g_ddr_info.ddr_size &&
           ddr_off <= g_ddr_info.ddr_size - (uint64_t)len;
}

static uint64_t shared_ddr_pacc_phys(uint64_t phys, size_t len) {
    if (g_shared_ddr_pacc_base && g_ddr_info.ddr_base &&
        phys_is_shared_ddr(phys, len)) {
        return g_shared_ddr_pacc_base + (phys - g_ddr_info.ddr_base);
    }
    return phys;
}

static bool phys_is_shared_ddr_control(uint64_t phys, size_t len) {
    if (!phys_is_shared_ddr(phys, len)) {
        return false;
    }
    uint64_t control_start = shared_ddr_control_rel(g_pacc_id, 0);
    uint64_t ddr_off = phys - g_ddr_info.ddr_base;
    return ddr_off >= control_start &&
           ddr_off - control_start <= HETGPU_PACC_CONTROL_BYTES - (uint64_t)len;
}

static int read_control_window_copy(uint64_t phys, size_t len, uint8_t **out) {
    uint64_t control_phys;
    uint64_t off;
    uint8_t *buf;

    if (!out || !g_control_window || !phys_is_shared_ddr_control(phys, len)) {
        return (int)0xffff5e05u;
    }
    control_phys = g_ddr_info.ddr_base + shared_ddr_control_rel(g_pacc_id, 0);
    off = phys - control_phys;
    if (off > HETGPU_PACC_CONTROL_BYTES ||
        (uint64_t)len > HETGPU_PACC_CONTROL_BYTES - off) {
        return (int)0xffff5e06u;
    }
    buf = (uint8_t *)malloc(len);
    if (!buf) {
        return (int)0xffff5e07u;
    }
    jobd_io_fence();
    if (jobd_msync_enabled() && g_control_map_base && g_control_map_len) {
        (void)msync(g_control_map_base, g_control_map_len, MS_SYNC | MS_INVALIDATE);
    }
    jobd_invalidate_for_cpu((const void *)(g_control_window + off), len);
    jobd_io_fence();
    for (size_t i = 0; i < len; i++) {
        buf[i] = g_control_window[off + i];
    }
    jobd_io_fence();
    *out = buf;
    return 0;
}

static bool read_current_control_window_bytes(uint32_t pacc_id,
                                              uint64_t off,
                                              void *dst,
                                              size_t len) {
    volatile uint8_t *src;
    uint8_t *out = (uint8_t *)dst;

    if (!jobd_control_window_read_enabled() ||
        !dst || len == 0 || !g_control_window ||
        pacc_id != (uint32_t)g_pacc_id ||
        off > HETGPU_PACC_CONTROL_BYTES ||
        (uint64_t)len > HETGPU_PACC_CONTROL_BYTES - off) {
        return false;
    }
    src = g_control_window + off;
    jobd_io_fence();
    if (jobd_msync_enabled() && g_control_map_base && g_control_map_len) {
        (void)msync(g_control_map_base, g_control_map_len, MS_SYNC | MS_INVALIDATE);
    }
    jobd_invalidate_for_cpu((const void *)src, len);
    jobd_io_fence();
    for (size_t i = 0; i < len; i++) {
        out[i] = src[i];
    }
    jobd_io_fence();
    return true;
}

static int fd_for_shared_ddr_io(int fallback_fd, uint64_t phys, size_t len) {
    if (phys_is_shared_ddr(phys, len) && g_shared_ddr_data_fd >= 0) {
        return g_shared_ddr_data_fd;
    }
    return fallback_fd;
}

static int read_phys_copy_pread_only(int fd, uint64_t phys, size_t len, uint8_t **out) {
    uint64_t fd_off = 0;
    uint64_t alt_fd_off = 0;
    bool have_alt_fd_off = false;
    bool tried_alt_fd_off = false;
    int io_fd;
    uint8_t *buf;
    size_t done = 0;
    size_t chunk = jobd_helper_io_chunk_bytes();

    if (!out || len == 0 || !phys_to_fd_offset(phys, len, &fd_off)) {
        return (int)0xffff5e08u;
    }
    io_fd = fd_for_shared_ddr_io(fd, phys, len);
    if (phys_is_shared_ddr(phys, len)) {
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        alt_fd_off = ddr_off;
        /*
         * LX500 /dev/mbox is split-brained in practice: fresh reads need the
         * DDR-relative offset, while some non-control writes only become host
         * visible through the historical user offset window.  Completion lives
         * in the control page and already works with the selected fd offset, so
         * only mirror payload/debug writes through the legacy window.
         */
        if (!phys_is_shared_ddr_control(phys, len) &&
            ddr_off <= UINT64_MAX - HETGPU_PACC_SHARED_DDR_FD_USER_OFF) {
            uint64_t legacy_fd_off = HETGPU_PACC_SHARED_DDR_FD_USER_OFF + ddr_off;
            if (legacy_fd_off != fd_off) {
                alt_fd_off = legacy_fd_off;
            }
        }
        have_alt_fd_off = alt_fd_off != fd_off;
    }
    buf = (uint8_t *)malloc(len);
    if (!buf) {
        return -1;
    }
    while (done < len) {
        size_t want = len - done;
        if (want > chunk) want = chunk;
        ssize_t got = pread(io_fd, buf + done, want, (off_t)(fd_off + done));
        if (got <= 0) {
            if (have_alt_fd_off && !tried_alt_fd_off) {
                tried_alt_fd_off = true;
                fd_off = alt_fd_off;
                done = 0;
                memset(buf, 0, len);
                continue;
            }
            free(buf);
            return (int)0xffff5e09u;
        }
        done += (size_t)got;
    }
    __sync_synchronize();
    *out = buf;
    return 0;
}

static int read_phys_copy(int fd, uint64_t phys, size_t len, uint8_t **out) {
    uint64_t fd_off = 0;
    uint64_t alt_fd_off = 0;
    bool have_alt_fd_off = false;
    bool tried_alt_fd_off = false;
    int io_fd;
    uint8_t *buf;
    size_t done = 0;
    size_t chunk = jobd_helper_io_chunk_bytes();

    if (!out || len == 0) {
        return -1;
    }

    if (!jobd_force_pread_enabled()) {
        if (phys_is_shared_ddr_control(phys, len) && jobd_control_pread_enabled()) {
            uint64_t control_base = g_ddr_info.ddr_base +
                                    shared_ddr_control_rel(g_pacc_id, 0);
            uint64_t control_off = phys >= control_base ? phys - control_base : 0;
            if (phys >= control_base &&
                control_off <= HETGPU_PACC_CONTROL_BYTES &&
                (uint64_t)len <= HETGPU_PACC_CONTROL_BYTES - control_off &&
                read_shared_ddr_control_copy_pread(fd, control_off, len, out) == 0) {
                return 0;
            }
        }
        if ((!phys_is_shared_ddr_control(phys, len) || jobd_control_window_read_enabled()) &&
            read_control_window_copy(phys, len, out) == 0) {
            return 0;
        }
    }

    if (g_ddr_info.ddr_base && phys >= g_ddr_info.ddr_base &&
        (uint64_t)len <= g_ddr_info.ddr_size &&
        phys - g_ddr_info.ddr_base <= g_ddr_info.ddr_size - (uint64_t)len) {
        if (phys_is_shared_ddr(phys, len) &&
            !phys_is_shared_ddr_control(phys, len) &&
            jobd_shared_ddr_payload_pread_enabled()) {
            goto pread_fallback;
        }
        if (jobd_force_pread_enabled() &&
            !(jobd_force_devmem_enabled() &&
              phys_is_shared_ddr(phys, len) &&
              !phys_is_shared_ddr_control(phys, len))) {
            goto pread_fallback;
        }
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        if (g_shared_ddr_full_map_valid &&
            ddr_off <= (uint64_t)g_shared_ddr_full_map.len &&
            (uint64_t)len <= (uint64_t)g_shared_ddr_full_map.len - ddr_off) {
            jobd_io_fence();
            return read_phys_mmap_copy(fd, phys, len, out);
        }
        if (g_map_uses_shared_ddr_offsets) {
            uint64_t page = (uint64_t)g_page_size;
            uint64_t mmap_addr = g_shared_ddr_mmap_user_off + ddr_off;
            uint64_t base = mmap_addr & ~(page - 1);
            size_t off = (size_t)(mmap_addr - base);
            size_t map_len = ((off + len + page - 1) / page) * page;
            void *p = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
            if (p == MAP_FAILED) {
                goto pread_fallback;
            }
            buf = (uint8_t *)malloc(len);
            if (!buf) {
                munmap(p, map_len);
                return -1;
            }
            if (jobd_msync_enabled() &&
                msync(p, map_len, MS_SYNC | MS_INVALIDATE) != 0 && errno != EINVAL) {
                trace_msg("shared DDR read invalidate failed: %s", strerror(errno));
            }
            jobd_io_fence();
            jobd_invalidate_for_cpu((char *)p + off, len);
            jobd_io_fence();
            memcpy(buf, (char *)p + off, len);
            jobd_io_fence();
            munmap(p, map_len);
            *out = buf;
            return 0;
        }
        if (read_phys_mmap_copy(fd, phys, len, out) == 0) {
            jobd_io_fence();
            return 0;
        }
    }

pread_fallback:
    if (!phys_to_fd_offset(phys, len, &fd_off)) {
        return -1;
    }
    io_fd = fd_for_shared_ddr_io(fd, phys, len);
    if (phys_is_shared_ddr(phys, len)) {
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        alt_fd_off = ddr_off;
        if (ddr_off <= UINT64_MAX - HETGPU_PACC_SHARED_DDR_FD_USER_OFF) {
            uint64_t legacy_fd_off = HETGPU_PACC_SHARED_DDR_FD_USER_OFF + ddr_off;
            if (legacy_fd_off != fd_off) {
                alt_fd_off = legacy_fd_off;
            }
        }
        have_alt_fd_off = alt_fd_off != fd_off;
    }

    buf = (uint8_t *)malloc(len);
    if (!buf) {
        return -1;
    }
    while (done < len) {
        size_t want = len - done;
        if (want > chunk) want = chunk;
        ssize_t got = pread(io_fd, buf + done, want, (off_t)(fd_off + done));
        if (got <= 0) {
            if (have_alt_fd_off && !tried_alt_fd_off) {
                tried_alt_fd_off = true;
                fd_off = alt_fd_off;
                done = 0;
                memset(buf, 0, len);
                continue;
            }
            free(buf);
            return read_phys_mmap_copy(fd, phys, len, out);
        }
        done += (size_t)got;
    }
    __sync_synchronize();
    *out = buf;
    return 0;
}

static int pread_fd_copy_exact(int io_fd, uint64_t fd_off, size_t len, uint8_t **out);

static int read_mbox_control_copy(int fd, uint64_t off, size_t len, uint8_t **out) {
    uint64_t phys;

    if (!jobd_mbox_control_enabled() || !out || len == 0 ||
        off > HETGPU_PACC_CONTROL_BYTES ||
        (uint64_t)len > HETGPU_PACC_CONTROL_BYTES - off) {
        return -1;
    }
    if (g_mbox_fd >= 0) {
        uint8_t *helper_copy = NULL;
        int rc = pread_fd_copy_exact(g_mbox_fd,
                                     HETGPU_PACC_AP2PACC_READ_HELPER_OFF + off,
                                     len,
                                     &helper_copy);
        if (rc == 0 && helper_copy) {
            *out = helper_copy;
            return 0;
        }
        if (jobd_trace_enabled()) {
            static unsigned helper_fail_trace_count;
            if (helper_fail_trace_count < 64) {
                trace_msg("mbox control helper pread failed off=0x%" PRIx64
                          " len=%zu rc=%d errno=%d",
                          off, len, rc, errno);
                helper_fail_trace_count++;
            }
        }
    }
    phys = AP2PACC_MBOX_PHYS + off;
    return read_phys_mmap_copy(fd, phys, len, out);
}

static void add_control_fd_candidate(uint64_t *candidates,
                                     size_t *candidate_count,
                                     size_t candidate_max,
                                     uint64_t value) {
    if (!candidates || !candidate_count || *candidate_count >= candidate_max) {
        return;
    }
    for (size_t i = 0; i < *candidate_count; i++) {
        if (candidates[i] == value) {
            return;
        }
    }
    candidates[(*candidate_count)++] = value;
}

static size_t shared_ddr_control_fd_candidates(uint64_t off,
                                               size_t len,
                                               uint64_t *candidates,
                                               size_t candidate_max) {
    uint64_t rel;
    size_t candidate_count = 0;

    if (!candidates || candidate_max == 0 || len == 0 ||
        !g_ddr_info.ddr_base || g_pacc_id >= HETGPU_PACC_COUNT) {
        return 0;
    }
    rel = shared_ddr_control_rel(g_pacc_id, off);
    if (rel > g_ddr_info.ddr_size || (uint64_t)len > g_ddr_info.ddr_size - rel) {
        return 0;
    }

    /*
     * LX500 /dev/mbox control reads are DDR-relative.  The host helper exposes
     * the same bytes at SHARED_DDR_USER_OFF+rel, so keep that as a fallback for
     * images whose mbox driver still expects the legacy user window.
     */
    add_control_fd_candidate(candidates, &candidate_count, candidate_max, rel);
    if (g_shared_ddr_fd_user_off <= UINT64_MAX - rel) {
        add_control_fd_candidate(candidates, &candidate_count, candidate_max,
                                 g_shared_ddr_fd_user_off + rel);
    }
    if (HETGPU_PACC_SHARED_DDR_USER_OFF <= UINT64_MAX - rel &&
        HETGPU_PACC_SHARED_DDR_USER_OFF != g_shared_ddr_fd_user_off) {
        add_control_fd_candidate(candidates, &candidate_count, candidate_max,
                                 HETGPU_PACC_SHARED_DDR_USER_OFF + rel);
    }
    return candidate_count;
}

static int pread_fd_copy_exact(int io_fd, uint64_t fd_off, size_t len, uint8_t **out) {
    uint8_t *buf;
    size_t done = 0;
    size_t chunk = jobd_helper_io_chunk_bytes();

    if (!out || len == 0 || io_fd < 0) {
        return -1;
    }
    buf = (uint8_t *)malloc(len);
    if (!buf) {
        return -1;
    }
    while (done < len) {
        size_t want = len - done;
        ssize_t got;
        if (want > chunk) want = chunk;
        got = pread(io_fd, buf + done, want, (off_t)(fd_off + done));
        if (got <= 0) {
            free(buf);
            return (int)0xffff5e0au;
        }
        done += (size_t)got;
    }
    jobd_io_fence();
    *out = buf;
    return 0;
}

static int read_shared_ddr_control_copy_pread_candidate(int fd,
                                                        uint64_t off,
                                                        size_t len,
                                                        size_t candidate_index,
                                                        uint8_t **out) {
    uint64_t candidates[4];
    size_t candidate_count;
    int io_fd;

    if (!jobd_control_pread_enabled()) {
        return -1;
    }
    candidate_count = shared_ddr_control_fd_candidates(off, len, candidates,
                                                       sizeof(candidates) / sizeof(candidates[0]));
    if (candidate_index >= candidate_count) {
        return -1;
    }
    io_fd = g_mbox_fd >= 0 ? g_mbox_fd : fd;
    return pread_fd_copy_exact(io_fd, candidates[candidate_index], len, out);
}

static int read_shared_ddr_control_copy_pread_for_pacc(int fd,
                                                       uint32_t pacc_id,
                                                       uint64_t off,
                                                       size_t len,
                                                       uint8_t **out) {
    uint64_t rel;
    uint64_t candidates[3];
    size_t candidate_count = 0;
    int io_fd;

    if (!jobd_control_pread_enabled() ||
        !out || len == 0 || !g_ddr_info.ddr_base ||
        pacc_id >= HETGPU_PACC_COUNT) {
        return -1;
    }
    rel = shared_ddr_control_rel(pacc_id, off);
    if (rel > g_ddr_info.ddr_size || (uint64_t)len > g_ddr_info.ddr_size - rel) {
        return -1;
    }

    add_control_fd_candidate(candidates, &candidate_count,
                             sizeof(candidates) / sizeof(candidates[0]), rel);
    if (g_shared_ddr_fd_user_off <= UINT64_MAX - rel) {
        add_control_fd_candidate(candidates, &candidate_count,
                                 sizeof(candidates) / sizeof(candidates[0]),
                                 g_shared_ddr_fd_user_off + rel);
    }
    if (HETGPU_PACC_SHARED_DDR_USER_OFF <= UINT64_MAX - rel &&
        HETGPU_PACC_SHARED_DDR_USER_OFF != g_shared_ddr_fd_user_off) {
        add_control_fd_candidate(candidates, &candidate_count,
                                 sizeof(candidates) / sizeof(candidates[0]),
                                 HETGPU_PACC_SHARED_DDR_USER_OFF + rel);
    }

    io_fd = g_mbox_fd >= 0 ? g_mbox_fd : fd;
    for (size_t c = 0; c < candidate_count; c++) {
        uint8_t *buf = NULL;
        int rc = pread_fd_copy_exact(io_fd, candidates[c], len, &buf);
        if (jobd_trace_enabled()) {
            static unsigned trace_count;
            if (trace_count < 64 &&
                off < HETGPU_PACC_ARG_BASE_OFF + HETGPU_PACC_ARG_SLOT_BYTES) {
                uint64_t head = 0;
                if (buf) {
                    memcpy(&head, buf, len < sizeof(head) ? len : sizeof(head));
                }
                trace_msg("control pread pacc=%u off=0x%" PRIx64
                          " cand=%zu fd_off=0x%" PRIx64
                          " rc=%d head=0x%" PRIx64,
                          pacc_id, off, c, candidates[c], rc, head);
                trace_count++;
            }
        }
        if (rc == 0) {
            uint64_t head = 0;
            memcpy(&head, buf, len < sizeof(head) ? len : sizeof(head));
            if (head == 0 && c + 1 < candidate_count) {
                free(buf);
                continue;
            }
            *out = buf;
            g_last_arg_header_candidate = (uint32_t)c;
            return 0;
        }
    }
    return read_phys_copy(fd, g_ddr_info.ddr_base + rel, len, out);
}

static int read_shared_ddr_control_copy_pread(int fd, uint64_t off, size_t len, uint8_t **out) {
    uint64_t candidates[4];
    size_t candidate_count;
    int io_fd;

    if (!jobd_control_pread_enabled() || !out || len == 0) {
        return -1;
    }
    candidate_count = shared_ddr_control_fd_candidates(off, len, candidates,
                                                       sizeof(candidates) / sizeof(candidates[0]));
    if (candidate_count == 0) {
        return -1;
    }
    io_fd = g_mbox_fd >= 0 ? g_mbox_fd : fd;
    for (size_t c = 0; c < candidate_count; c++) {
        uint8_t *buf = NULL;
        if (pread_fd_copy_exact(io_fd, candidates[c], len, &buf) == 0) {
            uint64_t head = 0;
            memcpy(&head, buf, len < sizeof(head) ? len : sizeof(head));
            if (head == 0 && c + 1 < candidate_count) {
                free(buf);
                continue;
            }
            *out = buf;
            return 0;
        }
    }
    return -1;
}

static int write_phys_copy_pwrite_only(int fd, uint64_t phys, const void *src, size_t len) {
    uint64_t fd_off = 0;
    uint64_t alt_fd_off = 0;
    bool have_alt_fd_off = false;
    bool tried_alt_fd_off = false;
    int io_fd;
    const uint8_t *buf = (const uint8_t *)src;
    size_t done = 0;
    size_t chunk = jobd_helper_io_chunk_bytes();

    if (!src || len == 0 || !phys_to_fd_offset(phys, len, &fd_off)) {
        return -1;
    }
    io_fd = fd_for_shared_ddr_io(fd, phys, len);
    if (phys_is_shared_ddr(phys, len)) {
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        alt_fd_off = ddr_off;
        if (ddr_off <= UINT64_MAX - HETGPU_PACC_SHARED_DDR_FD_USER_OFF) {
            uint64_t legacy_fd_off = HETGPU_PACC_SHARED_DDR_FD_USER_OFF + ddr_off;
            if (legacy_fd_off != fd_off) {
                alt_fd_off = legacy_fd_off;
            }
        }
        have_alt_fd_off = alt_fd_off != fd_off;
    }

    jobd_flush_for_device(src, len);
    retry_write:
    jobd_io_fence();
    while (done < len) {
        size_t want = len - done;
        if (want > chunk) want = chunk;
        ssize_t put = pwrite(io_fd, buf + done, want, (off_t)(fd_off + done));
        if (put <= 0) {
            if (have_alt_fd_off && !tried_alt_fd_off) {
                tried_alt_fd_off = true;
                fd_off = alt_fd_off;
                done = 0;
                continue;
            }
            return -1;
        }
        done += (size_t)put;
    }
    jobd_io_fence();
    const char *dual_offset = getenv("HETGPU_PACC_JOBD_DUAL_OFFSET_WRITE");
    if ((!dual_offset || !*dual_offset || env_flag_true(dual_offset)) &&
        have_alt_fd_off && !tried_alt_fd_off &&
        phys_is_shared_ddr(phys, len)) {
        tried_alt_fd_off = true;
        fd_off = alt_fd_off;
        done = 0;
        goto retry_write;
    }
    return 0;
}

static void write_jobd_beacon(int fd, uint32_t job_id, uint64_t seq, uint32_t phase, uint32_t detail) {
    struct JobdBeacon beacon = {
        .magic = HETGPU_PACC_BEACON_MAGIC,
        .version = HETGPU_PACC_JOB_VERSION,
        .job_id = job_id,
        .phase = phase,
        .detail = detail,
        .seq = seq,
    };
    uint64_t phys;

    if (!jobd_beacon_enabled()) {
        return;
    }
    phys = shared_ddr_control_phys(HETGPU_PACC_BEACON_OFF, sizeof(beacon));
    jobd_io_fence();
    if (write_phys_copy(fd, phys, &beacon, sizeof(beacon)) == 0) {
        jobd_io_fence();
        return;
    }
    if (!jobd_status_pwrite_enabled() ||
        write_phys_copy_pwrite_only(fd, phys, &beacon, sizeof(beacon)) != 0) {
        if (write_shared_ddr_devmem_direct(phys, &beacon, sizeof(beacon))) {
            jobd_io_fence();
            return;
        }
        int mem_fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (mem_fd >= 0) {
            int ret = write_phys_copy(mem_fd, phys, &beacon, sizeof(beacon));
            int saved_errno = errno;
            close(mem_fd);
            if (ret == 0) {
                jobd_io_fence();
                return;
            }
            errno = saved_errno;
        }
        trace_msg("write beacon failed: job_id=%u/%s seq=%" PRIu64
                  " phase=0x%x detail=0x%x: %s",
                  job_id, job_name(job_id), seq, phase, detail, strerror(errno));
        return;
    }
    jobd_io_fence();
}

static int write_phys_copy(int fd, uint64_t phys, const void *src, size_t len) {
    uint64_t fd_off = 0;
    int io_fd;
    const uint8_t *buf = (const uint8_t *)src;
    size_t done = 0;
    size_t chunk = jobd_helper_io_chunk_bytes();

    if (!src || len == 0) {
        return -1;
    }

    if (g_ddr_info.ddr_base && phys >= g_ddr_info.ddr_base &&
        (uint64_t)len <= g_ddr_info.ddr_size &&
        phys - g_ddr_info.ddr_base <= g_ddr_info.ddr_size - (uint64_t)len) {
        struct Map map = {0};
        if (phys_is_shared_ddr(phys, len) &&
            !phys_is_shared_ddr_control(phys, len) &&
            jobd_shared_ddr_payload_pwrite_enabled()) {
            if (write_phys_copy_pwrite_only(fd, phys, src, len) == 0) {
                return 0;
            }
            trace_msg("shared DDR payload pwrite failed; falling back to mmap write");
        }
        if (jobd_force_pread_enabled() && phys_is_shared_ddr_control(phys, len)) {
            goto pwrite_fallback;
        }
        if (map_phys(fd, phys, len, &map) == 0) {
            memcpy(map.ptr, src, len);
            __sync_synchronize();
            jobd_drain_after_write(map.ptr, map.len);
            if (map.base && map.base != MAP_FAILED && map.map_len) {
                if (jobd_msync_enabled() &&
                    msync(map.base, map.map_len, MS_SYNC) != 0 && errno != EINVAL) {
                    trace_msg("shared DDR msync write failed: %s", strerror(errno));
                }
            }
            jobd_flush_for_device(map.ptr, map.len);
            unmap_phys(&map);
            if (phys_is_shared_ddr(phys, len) &&
                !phys_is_shared_ddr_control(phys, len) &&
                jobd_shared_ddr_payload_sync_enabled()) {
                submit_mbox_payload_sync(g_mbox_fd, 0, 0, 0, "shared-ddr-payload-write");
            }
            return 0;
        }
    }

pwrite_fallback:
    if (!phys_to_fd_offset(phys, len, &fd_off)) {
        return -1;
    }
    io_fd = fd_for_shared_ddr_io(fd, phys, len);

    __sync_synchronize();
    while (done < len) {
        size_t want = len - done;
        if (want > chunk) want = chunk;
        ssize_t put = pwrite(io_fd, buf + done, want, (off_t)(fd_off + done));
        if (put <= 0) {
            return -1;
        }
        done += (size_t)put;
    }
    __sync_synchronize();
    if (phys_is_shared_ddr(phys, len) &&
        !phys_is_shared_ddr_control(phys, len) &&
        jobd_shared_ddr_payload_sync_enabled()) {
        submit_mbox_payload_sync(g_mbox_fd, 0, 0, 0, "shared-ddr-payload-pwrite");
    }
    return 0;
}

static int write_phys_copy_chunked(int fd,
                                   uint64_t phys,
                                   const void *src,
                                   size_t len,
                                   const char *chunk_env) {
    uint64_t requested;
    size_t chunk;
    const uint8_t *buf = (const uint8_t *)src;
    size_t done = 0;

    if (!src || len == 0) {
        return -1;
    }
    requested = parse_env_u64_default(chunk_env, 0);
    if (requested == 0 || requested >= (uint64_t)len) {
        return write_phys_copy(fd, phys, src, len);
    }
    chunk = (size_t)requested;
    if (chunk == 0) {
        return write_phys_copy(fd, phys, src, len);
    }
    while (done < len) {
        size_t want = len - done;
        if (want > chunk) want = chunk;
        if (write_phys_copy(fd, phys + done, buf + done, want) != 0) {
            return -1;
        }
        if (env_flag_true(getenv("HETGPU_PACC_JOBD_SYNC_WRITE_CHUNKS"))) {
            submit_mbox_payload_sync(g_mbox_fd, 0, 0, 0, "write-chunk");
        }
        done += want;
    }
    return 0;
}

static int write_phys_copy_chunked_pwrite_only(int fd,
                                               uint64_t phys,
                                               const void *src,
                                               size_t len,
                                               const char *chunk_env) {
    uint64_t requested;
    size_t chunk;
    const uint8_t *buf = (const uint8_t *)src;
    size_t done = 0;

    if (!src || len == 0) {
        return -1;
    }
    requested = parse_env_u64_default(chunk_env, 0);
    if (requested == 0 || requested >= (uint64_t)len) {
        return write_phys_copy_pwrite_only(fd, phys, src, len);
    }
    chunk = (size_t)requested;
    if (chunk == 0) {
        return write_phys_copy_pwrite_only(fd, phys, src, len);
    }
    while (done < len) {
        size_t want = len - done;
        if (want > chunk) want = chunk;
        if (write_phys_copy_pwrite_only(fd, phys + done, buf + done, want) != 0) {
            return -1;
        }
        if (env_flag_true(getenv("HETGPU_PACC_JOBD_SYNC_WRITE_CHUNKS"))) {
            submit_mbox_payload_sync(g_mbox_fd, 0, 0, 0, "pwrite-chunk");
        }
        done += want;
    }
    return 0;
}

static bool jobd_shared_ddr_payload_publish_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_SHARED_DDR_PAYLOAD_PUBLISH");
    if (value && *value) {
        return env_flag_true(value);
    }
    return true;
}

static void add_unique_u64(uint64_t *values,
                           size_t *count,
                           size_t max_count,
                           uint64_t value) {
    if (!values || !count || *count >= max_count) {
        return;
    }
    for (size_t i = 0; i < *count; i++) {
        if (values[i] == value) {
            return;
        }
    }
    values[(*count)++] = value;
}

static void add_unique_fd(int *fds, size_t *count, size_t max_count, int fd) {
    if (!fds || !count || *count >= max_count || fd < 0) {
        return;
    }
    for (size_t i = 0; i < *count; i++) {
        if (fds[i] == fd) {
            return;
        }
    }
    fds[(*count)++] = fd;
}

static bool pwrite_fd_exact(int io_fd, uint64_t fd_off, const void *src, size_t len) {
    const uint8_t *buf = (const uint8_t *)src;
    size_t done = 0;
    size_t chunk = jobd_helper_io_chunk_bytes();

    if (io_fd < 0 || !src || len == 0) {
        return false;
    }
    jobd_flush_for_device(src, len);
    jobd_io_fence();
    while (done < len) {
        size_t want = len - done;
        ssize_t put;
        if (want > chunk) want = chunk;
        put = pwrite(io_fd, buf + done, want, (off_t)(fd_off + done));
        if (put <= 0) {
            return false;
        }
        done += (size_t)put;
    }
    jobd_io_fence();
    return true;
}

static bool pread_fd_matches(int io_fd, uint64_t fd_off, const void *expected, size_t len) {
    const uint8_t *want_buf = (const uint8_t *)expected;
    uint8_t stack_buf[256];
    uint8_t *buf = stack_buf;
    size_t done = 0;
    size_t chunk = jobd_helper_io_chunk_bytes();
    bool match = false;

    if (io_fd < 0 || !expected || len == 0) {
        return false;
    }
    if (len > sizeof(stack_buf)) {
        buf = (uint8_t *)malloc(len);
        if (!buf) {
            return false;
        }
    }
    while (done < len) {
        size_t want = len - done;
        ssize_t got;
        if (want > chunk) want = chunk;
        got = pread(io_fd, buf + done, want, (off_t)(fd_off + done));
        if (got <= 0) {
            goto out;
        }
        done += (size_t)got;
    }
    jobd_io_fence();
    match = memcmp(buf, want_buf, len) == 0;
out:
    if (buf != stack_buf) {
        free(buf);
    }
    return match;
}

static bool publish_shared_ddr_payload_visible(int fd,
                                               uint64_t phys,
                                               const void *src,
                                               size_t len,
                                               const char *where) {
    uint64_t offsets[8];
    int fds[4];
    size_t offset_count = 0;
    size_t fd_count = 0;
    uint64_t ddr_off;
    bool wrote_any = false;
    bool visible_any = false;

    if (!jobd_shared_ddr_payload_publish_enabled() ||
        !src || len == 0 || !phys_is_shared_ddr(phys, len)) {
        return false;
    }

    ddr_off = phys - g_ddr_info.ddr_base;
    add_unique_u64(offsets, &offset_count, sizeof(offsets) / sizeof(offsets[0]), ddr_off);
    if (g_shared_ddr_fd_user_off <= UINT64_MAX - ddr_off) {
        add_unique_u64(offsets, &offset_count, sizeof(offsets) / sizeof(offsets[0]),
                       g_shared_ddr_fd_user_off + ddr_off);
    }
    if (HETGPU_PACC_SHARED_DDR_FD_USER_OFF <= UINT64_MAX - ddr_off) {
        add_unique_u64(offsets, &offset_count, sizeof(offsets) / sizeof(offsets[0]),
                       HETGPU_PACC_SHARED_DDR_FD_USER_OFF + ddr_off);
    }
    /*
     * Keep publish offsets in the AP/helper-visible DDR windows.  PACC alias or
     * host physical offsets are mmap/devmem concepts, not safe pwrite targets.
     */

    /*
     * Never use the caller fd here: run_gemm() may pass /dev/mem, and pwrite()
     * to DDR-relative offsets on /dev/mem can fault inside write_mem.  Payload
     * publication must use the mailbox/helper character devices only.
     */
    add_unique_fd(fds, &fd_count, sizeof(fds) / sizeof(fds[0]), g_shared_ddr_data_fd);
    add_unique_fd(fds, &fd_count, sizeof(fds) / sizeof(fds[0]), g_mbox_fd);

    for (size_t f = 0; f < fd_count; f++) {
        for (size_t o = 0; o < offset_count; o++) {
            int saved_errno = 0;
            bool wrote = pwrite_fd_exact(fds[f], offsets[o], src, len);
            if (!wrote) {
                saved_errno = errno;
                if (jobd_trace_enabled()) {
                    trace_msg("payload publish pwrite failed fd=%d off=0x%" PRIx64
                              " len=0x%zx errno=%d where=%s",
                              fds[f], offsets[o], len, saved_errno,
                              where ? where : "payload");
                }
                continue;
            }
            wrote_any = true;
            if (jobd_shared_ddr_payload_sync_enabled()) {
                submit_mbox_payload_sync(g_mbox_fd, 0, 0, 0, "payload-publish");
            }
            if (pread_fd_matches(fds[f], offsets[o], src, len)) {
                visible_any = true;
                trace_msg("payload publish visible fd=%d off=0x%" PRIx64
                          " phys=0x%" PRIx64 " len=0x%zx where=%s",
                          fds[f], offsets[o], phys, len,
                          where ? where : "payload");
            } else if (jobd_trace_enabled()) {
                trace_msg("payload publish readback mismatch fd=%d off=0x%" PRIx64
                          " phys=0x%" PRIx64 " len=0x%zx where=%s",
                          fds[f], offsets[o], phys, len,
                          where ? where : "payload");
            }
        }
    }

    if (!visible_any) {
        trace_msg("payload publish not visible: phys=0x%" PRIx64
                  " len=0x%zx wrote_any=%d where=%s",
                  phys, len, wrote_any ? 1 : 0, where ? where : "payload");
    }
    return visible_any;
}

static void repair_shared_ddr_writeback(int fd,
                                        uint64_t phys,
                                        const void *src,
                                        size_t len,
                                        const char *where) {
    const uint8_t *expected = (const uint8_t *)src;
    uint64_t attempts;
    uint64_t requested_chunk;
    uint64_t sleep_us;
    size_t chunk;

    const char *enabled = getenv("HETGPU_PACC_JOBD_REPAIR_WRITEBACK");
    if (!src || len == 0 || !phys_is_shared_ddr(phys, len)) {
        return;
    }
    if (!enabled || !*enabled || !env_flag_true(enabled)) {
        return;
    }
    attempts = parse_env_u64_default("HETGPU_PACC_JOBD_REPAIR_WRITEBACK_ATTEMPTS", 32);
    if (attempts > 256) attempts = 256;
    sleep_us = parse_env_u64_default("HETGPU_PACC_JOBD_REPAIR_WRITEBACK_SLEEP_US", 0);
    if (sleep_us > 1000000) sleep_us = 1000000;
    requested_chunk = parse_env_u64_default("HETGPU_PACC_JOBD_REPAIR_WRITEBACK_CHUNK_BYTES", 4096);
    if (requested_chunk < 16) requested_chunk = 16;
    if (requested_chunk > 4096) requested_chunk = 4096;
    chunk = (size_t)requested_chunk;

    for (uint64_t attempt = 0; attempt < attempts; attempt++) {
        uint8_t *visible = NULL;
        size_t repaired = 0;
        size_t first_bad = SIZE_MAX;
        if (read_phys_copy_pread_only(fd_for_shared_ddr_io(fd, phys, len),
                                      phys, len, &visible) != 0 || !visible) {
            trace_msg("repair writeback pread failed: phys=0x%" PRIx64
                      " len=0x%zx where=%s",
                      phys, len, where ? where : "unknown");
            free(visible);
            return;
        }
        for (size_t off = 0; off < len;) {
            if (visible[off] == expected[off]) {
                off++;
                continue;
            }
            size_t start = (off / chunk) * chunk;
            size_t want = chunk;
            if (start + want > len) want = len - start;
            if (first_bad == SIZE_MAX) first_bad = start;
            if (write_phys_copy_pwrite_only(fd, phys + start, expected + start, want) == 0) {
                repaired++;
            }
            off = start + want;
        }
        if (repaired != 0 && jobd_shared_ddr_payload_sync_enabled()) {
            submit_mbox_payload_sync(g_mbox_fd, 0, 0, 0, "repair-writeback");
        }
        if (repaired == 0) {
            bool visible_match = memcmp(visible, expected, len) == 0;
            free(visible);
            if (visible_match) {
                trace_msg("repair writeback visible: phys=0x%" PRIx64
                          " len=0x%zx attempt=%" PRIu64 " where=%s",
                          phys, len, attempt, where ? where : "unknown");
                return;
            }
            trace_msg("repair writeback made no progress: phys=0x%" PRIx64
                      " len=0x%zx attempt=%" PRIu64 " where=%s",
                      phys, len, attempt, where ? where : "unknown");
        } else {
            free(visible);
        }
        trace_msg("repair writeback chunks=%zu first=0x%zx phys=0x%" PRIx64
                  " len=0x%zx attempt=%" PRIu64 " where=%s",
                  repaired, first_bad == SIZE_MAX ? 0 : first_bad,
                  phys, len, attempt, where ? where : "unknown");
        if (sleep_us != 0) {
            usleep((useconds_t)sleep_us);
        }
    }
}

static bool native_stage_read(uint64_t phys, size_t len, void **out) {
    struct Map map = {0};
    uint8_t *copy = NULL;
    void *buf = NULL;

    if (!out || len == 0) {
        return false;
    }
    *out = NULL;

    if (phys_is_shared_ddr(phys, len) && g_mbox_fd >= 0) {
        if (map_phys(g_mbox_fd, phys, len, &map) == 0) {
            sync_map_for_cpu(&map);
            jobd_io_fence();
            buf = malloc(len);
            if (buf) {
                memcpy(buf, map.ptr, len);
            }
            jobd_io_fence();
            unmap_phys(&map);
            if (buf) {
                *out = buf;
                return true;
            }
        }
        if (read_phys_copy_pread_only(g_mbox_fd, phys, len, &copy) == 0 && copy) {
            *out = copy;
            return true;
        }
    }

    buf = malloc(len);
    if (!buf) {
        return false;
    }
    memcpy(buf, (const void *)(uintptr_t)phys, len);
    *out = buf;
    return true;
}

static bool native_stage_write(uint64_t phys, const void *src, size_t len) {
    struct Map map = {0};

    if (!src) {
        return false;
    }
    if (len == 0) {
        return true;
    }
    if (phys_is_shared_ddr(phys, len) && g_mbox_fd >= 0) {
        if (map_phys(g_mbox_fd, phys, len, &map) == 0) {
            memcpy(map.ptr, src, len);
            jobd_io_fence();
            jobd_drain_after_write(map.ptr, map.len);
            if (map.base && map.base != MAP_FAILED && map.map_len) {
                if (jobd_msync_enabled() &&
                    msync(map.base, map.map_len, MS_SYNC) != 0 && errno != EINVAL) {
                    trace_msg("native stage shared DDR msync write failed: %s", strerror(errno));
                }
            }
            jobd_flush_for_device(map.ptr, map.len);
            unmap_phys(&map);
            return true;
        }
        if (write_phys_copy_pwrite_only(g_mbox_fd, phys, src, len) == 0) {
            return true;
        }
    }
    memcpy((void *)(uintptr_t)phys, src, len);
    return true;
}

static bool native_stage_read_pread(uint64_t phys, size_t len, void **out) {
    uint8_t *copy = NULL;

    if (!out || len == 0) {
        return false;
    }
    *out = NULL;

    if (phys_is_shared_ddr(phys, len) && !phys_is_shared_ddr_control(phys, len) &&
        g_mbox_fd >= 0 && g_ddr_info.ddr_base && phys >= g_ddr_info.ddr_base) {
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        if ((uint64_t)len <= g_ddr_info.ddr_size &&
            ddr_off <= g_ddr_info.ddr_size - (uint64_t)len) {
            uint64_t fd_off = g_shared_ddr_fd_user_off + ddr_off;
            size_t done = 0;
            size_t chunk = jobd_helper_io_chunk_bytes();
            copy = (uint8_t *)malloc(len);
            if (copy) {
                while (done < len) {
                    size_t want = len - done;
                    if (want > chunk) want = chunk;
                    ssize_t got = pread(g_mbox_fd, copy + done, want, (off_t)(fd_off + done));
                    if (got <= 0) {
                        free(copy);
                        copy = NULL;
                        break;
                    }
                    done += (size_t)got;
                }
                if (copy) {
                    *out = copy;
                    return true;
                }
            }
        }
    }

    if (phys_is_shared_ddr(phys, len) && g_mbox_fd >= 0 &&
        read_phys_copy_pread_only(g_mbox_fd, phys, len, &copy) == 0 && copy) {
        *out = copy;
        return true;
    }
    return native_stage_read(phys, len, out);
}

static bool native_stage_write_pwrite(uint64_t phys, const void *src, size_t len) {
    if (!src) {
        return false;
    }
    if (len == 0) {
        return true;
    }
    if (phys_is_shared_ddr(phys, len) && !phys_is_shared_ddr_control(phys, len) &&
        g_mbox_fd >= 0 && g_ddr_info.ddr_base && phys >= g_ddr_info.ddr_base) {
        uint64_t ddr_off = phys - g_ddr_info.ddr_base;
        if ((uint64_t)len <= g_ddr_info.ddr_size &&
            ddr_off <= g_ddr_info.ddr_size - (uint64_t)len) {
            uint64_t fd_off = g_shared_ddr_fd_user_off + ddr_off;
            const uint8_t *buf = (const uint8_t *)src;
            size_t done = 0;
            size_t chunk = jobd_helper_io_chunk_bytes();
            while (done < len) {
                size_t want = len - done;
                if (want > chunk) want = chunk;
                ssize_t put = pwrite(g_mbox_fd, buf + done, want, (off_t)(fd_off + done));
                if (put <= 0) {
                    break;
                }
                done += (size_t)put;
            }
            if (done == len) {
                jobd_io_fence();
                return true;
            }
        }
    }
    if (phys_is_shared_ddr(phys, len) && g_mbox_fd >= 0 &&
        write_phys_copy_pwrite_only(g_mbox_fd, phys, src, len) == 0) {
        return true;
    }
    return native_stage_write(phys, src, len);
}

static int map_phys_copy_fallback(int fd, uint64_t phys, size_t len, struct Map *out) {
    uint8_t *copy = NULL;
    if (!out || !g_ddr_info.ddr_base || phys < g_ddr_info.ddr_base ||
        (uint64_t)len > g_ddr_info.ddr_size ||
        phys - g_ddr_info.ddr_base > g_ddr_info.ddr_size - (uint64_t)len) {
        return -1;
    }
    if (read_phys_copy(fd, phys, len, &copy) != 0) {
        return -1;
    }
    out->base = NULL;
    out->map_len = 0;
    out->ptr = copy;
    out->copied = true;
    out->fd = fd;
    out->phys = phys;
    out->len = len;
    return 0;
}

static int flush_map_to_phys(struct Map *m) {
    if (!m || !m->copied) {
        return 0;
    }
    if (write_phys_copy_pwrite_only(m->fd, m->phys, m->ptr, m->len) == 0) {
        return 0;
    }
    return write_phys_copy(m->fd, m->phys, m->ptr, m->len);
}

static int map_phys_for_mmvf(int fd, uint64_t phys, size_t len, struct Map *out, bool prefer_copy) {
    if (prefer_copy && map_phys_copy_fallback(fd, phys, len, out) == 0) {
        return 0;
    }
    if (map_phys(fd, phys, len, out) != 0) {
        return -1;
    }
    sync_map_for_cpu(out);
    return 0;
}

static void clear_stale_control_region(int fd, struct Map *control_map) {
    uint8_t zeros[HETGPU_PACC_CONTROL_BYTES];
    uint64_t phys;

    if (!jobd_clear_stale_control_enabled() || !g_ddr_info.ddr_base) {
        return;
    }

    memset(zeros, 0, sizeof(zeros));
    phys = shared_ddr_control_phys(0, sizeof(zeros));
    jobd_io_fence();
    if (write_phys_copy_pwrite_only(fd, phys, zeros, sizeof(zeros)) != 0 &&
        write_phys_copy(fd, phys, zeros, sizeof(zeros)) != 0) {
        log_msg("clear stale control failed: phys=0x%" PRIx64 " len=0x%zx: %s",
                phys, sizeof(zeros), strerror(errno));
        return;
    }
    if (control_map && control_map->ptr && control_map->len >= sizeof(zeros)) {
        memset(control_map->ptr, 0, sizeof(zeros));
        if (control_map->base && control_map->base != MAP_FAILED && control_map->map_len) {
            if (jobd_msync_enabled() &&
                msync(control_map->base, control_map->map_len, MS_SYNC | MS_INVALIDATE) != 0 &&
                errno != EINVAL) {
                trace_msg("clear stale control msync failed: %s", strerror(errno));
            }
        }
        jobd_flush_for_device(control_map->ptr, sizeof(zeros));
    }
    jobd_io_fence();
    log_msg("cleared stale shared-DDR control slot: phys=0x%" PRIx64
            " len=0x%zx", phys, sizeof(zeros));
}

static bool control_copy_has_job_like_header(const void *buf, size_t len) {
    const volatile struct Doorbell *head;
    const struct PaccJobDesc *kernel_head;

    if (!buf) {
        return false;
    }
    if (len >= sizeof(struct Doorbell)) {
        head = (const volatile struct Doorbell *)buf;
        if (head->magic == HETGPU_PACC_JOB_MAGIC &&
            head->version == HETGPU_PACC_JOB_VERSION &&
            head->seq != 0) {
            return true;
        }
    }
    if (len >= sizeof(struct PaccJobDesc)) {
        kernel_head = (const struct PaccJobDesc *)buf;
        if (kernel_head->buf_info == PACC_JOB_MAGIC &&
            kernel_head->seq != 0) {
            return true;
        }
    }
    return false;
}

static void trace_empty_mbox_control_snapshot(const char *source, size_t len) {
    if (jobd_trace_enabled()) {
        static unsigned trace_count;
        if (trace_count < 64) {
            trace_msg("mbox control %s returned non-job snapshot len=%zu; trying shared-DDR control",
                      source ? source : "read", len);
            trace_count++;
        }
    }
}

static int read_control_snapshot(int fd, void *out, size_t len) {
    uint8_t *copy = NULL;
    uint64_t phys;
    if (!out || len == 0 || len > HETGPU_PACC_CONTROL_BYTES) {
        return -1;
    }
    if (jobd_mbox_control_enabled()) {
        /*
         * /dev/mbox is interrupt-oriented on the PACC Linux image; offset
         * pread() returns EINVAL on current builds. Prefer an mmap of the
         * mailbox device and only fall back to /dev/mem for AP2PACC reads.
         * PACC2AP /dev/mem writes are not safe on this kernel.
         */
        if (g_mbox_fd >= 0) {
            int helper_rc = pread_fd_copy_exact(g_mbox_fd,
                                                HETGPU_PACC_AP2PACC_READ_HELPER_OFF,
                                                len,
                                                &copy);
            if (helper_rc == 0 && copy) {
                if (control_copy_has_job_like_header(copy, len)) {
                    memcpy(out, copy, len);
                    free(copy);
                    jobd_io_fence();
                    return 0;
                }
                trace_empty_mbox_control_snapshot("helper pread", len);
                free(copy);
                copy = NULL;
            }
            if (jobd_trace_enabled()) {
                static unsigned helper_fail_trace_count;
                if (helper_fail_trace_count < 64) {
                    trace_msg("mbox control snapshot helper pread failed len=%zu rc=%d errno=%d",
                              len, helper_rc, errno);
                    helper_fail_trace_count++;
                }
            }

            size_t page = (size_t)(g_page_size ? g_page_size : 4096);
            size_t map_len = (len + page - 1) & ~(page - 1);
            void *base = mmap(NULL,
                              map_len,
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED,
                              g_mbox_fd,
                              0);
            if (base != MAP_FAILED) {
                copy = (uint8_t *)malloc(len);
                if (copy) {
                    jobd_io_fence();
                    memcpy(copy, base, len);
                    jobd_io_fence();
                }
                munmap(base, map_len);
                if (copy) {
                    if (control_copy_has_job_like_header(copy, len)) {
                        memcpy(out, copy, len);
                        free(copy);
                        jobd_io_fence();
                        return 0;
                    }
                    trace_empty_mbox_control_snapshot("mmap fallback", len);
                    free(copy);
                    copy = NULL;
                }
            } else {
                trace_msg("mbox control snapshot mmap fallback failed len=%zu errno=%d",
                          len, errno);
            }
        }
        if (read_phys_mmap_copy(fd, AP2PACC_MBOX_PHYS, len, &copy) != 0) {
            trace_msg("mbox control snapshot phys read failed phys=0x%" PRIx64
                      " len=%zu errno=%d",
                      (uint64_t)AP2PACC_MBOX_PHYS, len, errno);
        } else {
            if (control_copy_has_job_like_header(copy, len)) {
                memcpy(out, copy, len);
                free(copy);
                jobd_io_fence();
                return 0;
            }
            trace_empty_mbox_control_snapshot("phys mmap", len);
            free(copy);
            copy = NULL;
        }
    }
    phys = shared_ddr_control_phys(0, len);
    if (jobd_control_pread_enabled() &&
        phys_is_shared_ddr_control(phys, len) && g_mbox_fd >= 0) {
        if (read_shared_ddr_control_copy_pread(fd, 0, len, &copy) == 0) {
            if (control_copy_has_job_like_header(copy, len)) {
                memcpy(out, copy, len);
                free(copy);
                return 0;
            }
            trace_empty_mbox_control_snapshot("shared-DDR pread", len);
            free(copy);
            copy = NULL;
        }
        if (read_phys_copy_pread_only(fd, phys, len, &copy) == 0) {
            if (control_copy_has_job_like_header(copy, len)) {
                memcpy(out, copy, len);
                free(copy);
                return 0;
            }
            trace_empty_mbox_control_snapshot("shared-DDR phys pread", len);
            free(copy);
            copy = NULL;
        }
    }
    if (jobd_control_pread_enabled() &&
        phys_is_shared_ddr_control(phys, len) && g_mbox_fd >= 0) {
        if (read_phys_mmap_copy(fd, phys, len, &copy) == 0) {
            memcpy(out, copy, len);
            free(copy);
            return 0;
        }
    }
    /*
     * The doorbell is producer-owned by the host.  Avoid the persistent
     * control mmap here: on current firmware it can keep seeing an old seq
     * after an IRQ.  A fresh shared-DDR read is a little heavier, but it makes
     * poll+IRQ semantics correct and prevents dispatching stale jobs.
     *
     * When FORCE_PREAD is requested, do not touch the mmap path first.  On the
     * current PACC Linux image /dev/mem mappings can lag one IRQ behind, so a
     * "fresh mmap" is still stale.  The /dev/mbox pread path is the only
     * remaining way to ask the mailbox driver for the latest 32-byte chunks.
     */
    if (jobd_force_pread_enabled()) {
        if (read_phys_copy_pread_only(fd, phys, len, &copy) != 0 &&
            read_phys_mmap_copy(fd, phys, len, &copy) != 0 &&
            read_phys_copy(fd, phys, len, &copy) != 0) {
            return -1;
        }
    } else if (read_phys_mmap_copy(fd, phys, len, &copy) != 0 &&
               read_phys_copy(fd, phys, len, &copy) != 0) {
        return -1;
    }
    memcpy(out, copy, len);
    free(copy);
    return 0;
}

static bool read_mbox_kernel_desc(int mbox_fd, struct PaccJobDesc *desc) {
    struct PaccJobDesc tmp;
    ssize_t got;

    if (mbox_fd < 0 || !desc) {
        return false;
    }
    memset(&tmp, 0, sizeof(tmp));
    jobd_io_fence();
    got = pread(mbox_fd, &tmp, sizeof(tmp), 0);
    jobd_io_fence();
    if (got != (ssize_t)sizeof(tmp)) {
        return false;
    }
    if (tmp.buf_info != PACC_JOB_MAGIC || tmp.seq == 0 || tmp.addr == 0 || tmp.len == 0) {
        return false;
    }
    *desc = tmp;
    return true;
}

static bool probe_shared_ddr_mmap_at(int fd, uint64_t user_off) {
    size_t len = (size_t)g_page_size;
    void *p;
    if (!g_ddr_info.ddr_base || !g_ddr_info.ddr_size) {
        return false;
    }
    if (g_ddr_info.ddr_size < len) {
        len = (size_t)g_ddr_info.ddr_size;
    }
    p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)user_off);
    if (p == MAP_FAILED) {
        log_msg("/dev/mbox shared-DDR mmap probe off=0x%" PRIx64 " failed: %s",
                user_off, strerror(errno));
        return false;
    }
    munmap(p, len);
    g_shared_ddr_mmap_user_off = user_off;
    return true;
}

static bool probe_shared_ddr_mmap(int fd) {
    const char *configured = getenv("HETGPU_PACC_SHARED_DDR_MMAP_USER_OFF");
    uint64_t configured_off = 0;

    if (configured && *configured &&
        parse_u64_checked(configured, &configured_off) &&
        probe_shared_ddr_mmap_at(fd, configured_off)) {
        return true;
    }

    /*
     * /dev/pacc shared-DDR mmap is DDR-relative in the AP-visible driver.  Probe
     * offset 0 before the historical user window; merely mapping 0x100000 can
     * succeed while pointing at unrelated payload bytes.
     */
    if ((!configured || !*configured) && probe_shared_ddr_mmap_at(fd, 0)) {
        return true;
    }
    if ((!configured || !*configured) &&
        probe_shared_ddr_mmap_at(fd, HETGPU_PACC_SHARED_DDR_USER_OFF)) {
        return true;
    }
    if (configured_off != HETGPU_PACC_SHARED_DDR_USER_OFF &&
        probe_shared_ddr_mmap_at(fd, HETGPU_PACC_SHARED_DDR_USER_OFF)) {
        return true;
    }
    if (configured_off != 0 && probe_shared_ddr_mmap_at(fd, 0)) {
        return true;
    }
    return false;
}

static void unmap_phys(struct Map *m) {
    if (m->copied) {
        free(m->ptr);
        memset(m, 0, sizeof(*m));
        return;
    }
    if (m->borrowed) {
        memset(m, 0, sizeof(*m));
        return;
    }
    if (m->base && m->base != MAP_FAILED) {
        munmap(m->base, m->map_len);
    }
    memset(m, 0, sizeof(*m));
}

static void sync_map_for_cpu(struct Map *m) {
    if (!m || !m->base || m->base == MAP_FAILED || !m->map_len) {
        return;
    }
    if (jobd_msync_enabled() &&
        msync(m->base, m->map_len, MS_SYNC | MS_INVALIDATE) != 0 && errno != EINVAL) {
        trace_msg("shared DDR msync invalidate failed: %s", strerror(errno));
    }
    if (m->ptr && m->len) {
        jobd_invalidate_for_cpu(m->ptr, m->len);
    } else if (m->ptr && m->base && m->map_len) {
        uintptr_t base = (uintptr_t)m->base;
        uintptr_t ptr = (uintptr_t)m->ptr;
        size_t off = ptr >= base ? (size_t)(ptr - base) : 0;
        size_t len = off < m->map_len ? m->map_len - off : m->map_len;
        jobd_invalidate_for_cpu(m->ptr, len);
    }
    __sync_synchronize();
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
    case HETGPU_PACC_JOB_MMVF: return 4;
    default: return -1;
    }
}

static bool preloaded_job_seen(uint32_t job_id, uint64_t seq) {
    return job_id < HETGPU_PACC_MAX_JOB_ID &&
           seq != 0 &&
           seq == g_last_preloaded_seq_by_job[job_id];
}

static void mark_preloaded_job_seen(uint32_t job_id, uint64_t seq) {
    if (job_id < HETGPU_PACC_MAX_JOB_ID &&
        seq > g_last_preloaded_seq_by_job[job_id]) {
        g_last_preloaded_seq_by_job[job_id] = seq;
    }
}

static bool arg_slot_header_valid(uint32_t job_id, const struct ArgSlotHeader *header) {
    return header &&
           header->magic == HETGPU_PACC_JOB_MAGIC &&
           header->version == HETGPU_PACC_JOB_VERSION &&
           (header->job_id == job_id || header->job_id == 0) &&
           header->seq != 0 &&
           header->arg_len <= HETGPU_PACC_ARG_SLOT_BYTES - sizeof(*header);
}

static uint32_t arg_slot_header_valid_bits(uint32_t job_id, const struct ArgSlotHeader *header) {
    if (!header) {
        return 0;
    }
    return (header->magic == HETGPU_PACC_JOB_MAGIC ? 0x1u : 0u) |
           (header->version == HETGPU_PACC_JOB_VERSION ? 0x2u : 0u) |
           ((header->job_id == job_id || header->job_id == 0) ? 0x4u : 0u) |
           (header->arg_len <= HETGPU_PACC_ARG_SLOT_BYTES - sizeof(*header) ? 0x8u : 0u) |
           (header->seq != 0 ? 0x10u : 0u);
}

static bool arg_slot_header_copy(int fd, uint32_t job_id, struct ArgSlotHeader *out) {
    int slot = arg_slot_for_job(job_id);
    uint64_t slot_off;
    struct ArgSlotHeader best_bad;
    uint32_t best_bad_bits = 0;
    uint32_t best_bad_candidate = 0;
    bool have_bad = false;

    if (slot < 0 || !out) {
        return false;
    }
    slot_off = HETGPU_PACC_ARG_BASE_OFF + (uint64_t)slot * HETGPU_PACC_ARG_SLOT_BYTES;
    g_last_arg_header_candidate = 0xffffffffu;
    for (size_t c = 0; c < 4; c++) {
        uint8_t *slot_bytes = NULL;
        struct ArgSlotHeader header;
        uint32_t bits;

        if (read_shared_ddr_control_copy_pread_candidate(fd, slot_off, sizeof(*out), c, &slot_bytes) != 0) {
            continue;
        }
        memcpy(&header, slot_bytes, sizeof(header));
        free(slot_bytes);
        bits = arg_slot_header_valid_bits(job_id, &header);
        if (arg_slot_header_valid(job_id, &header)) {
            *out = header;
            g_last_arg_header_candidate = (uint32_t)c;
            return true;
        }
        if (!have_bad || bits > best_bad_bits) {
            best_bad = header;
            best_bad_bits = bits;
            best_bad_candidate = (uint32_t)c;
            have_bad = true;
        }
    }
    if (have_bad) {
        *out = best_bad;
        g_last_arg_header_candidate = best_bad_candidate;
        return true;
    }
    {
        uint8_t *slot_bytes = NULL;
        if (read_phys_copy(fd,
                           shared_ddr_control_phys(slot_off, sizeof(*out)),
                           sizeof(*out),
                           &slot_bytes) == 0) {
            memcpy(out, slot_bytes, sizeof(*out));
            free(slot_bytes);
            g_last_arg_header_candidate = 0xfffffffeu;
            return true;
        }
    }
    return false;
}

static void clear_arg_slot_header(int fd, uint32_t job_id) {
    int slot = arg_slot_for_job(job_id);
    struct ArgSlotHeader zero;
    uint64_t slot_off;

    if (slot < 0) {
        return;
    }
    memset(&zero, 0, sizeof(zero));
    slot_off = HETGPU_PACC_ARG_BASE_OFF + (uint64_t)slot * HETGPU_PACC_ARG_SLOT_BYTES;
    (void)write_phys_copy(fd,
                          shared_ddr_control_phys(slot_off, sizeof(zero)),
                          &zero,
                          sizeof(zero));
}

static bool find_pending_arg_slot_job(int fd, struct ArgSlotHeader *best_out) {
    static const uint32_t job_ids[] = {
        HETGPU_PACC_JOB_GEMM,
        HETGPU_PACC_JOB_SOFTMAX,
        HETGPU_PACC_JOB_RMSNORM,
        HETGPU_PACC_JOB_ALLREDUCE,
        HETGPU_PACC_JOB_MMVF,
    };
    struct ArgSlotHeader best;
    uint32_t best_candidate = 0xffffffffu;
    uint32_t best_pacc_id = UINT32_MAX;
    bool have_best = false;

    if (!jobd_arg_slot_scan_enabled()) {
        mirror_diag_event(fd, 0, 0, 0x520f, 0);
        return false;
    }

    mirror_diag_event(fd, 0, 0, 0x5200, 0);
    memset(&best, 0, sizeof(best));
    {
    uint32_t preferred = g_pacc_id < HETGPU_PACC_COUNT ? (uint32_t)g_pacc_id : 0U;
    uint32_t scan_count = jobd_arg_slot_scan_all_pacc_enabled() ? HETGPU_PACC_COUNT : 1U;

    for (uint32_t pacc_iter = 0; pacc_iter < scan_count; pacc_iter++) {
        uint32_t pacc = (preferred + pacc_iter) % HETGPU_PACC_COUNT;
        for (size_t i = 0; i < sizeof(job_ids) / sizeof(job_ids[0]); i++) {
            struct ArgSlotHeader header;
            uint32_t job_id = job_ids[i];
            int slot = arg_slot_for_job(job_id);
            uint64_t slot_off;
            uint8_t *slot_bytes = NULL;

            if (slot < 0) {
                continue;
            }
            slot_off = HETGPU_PACC_ARG_BASE_OFF + (uint64_t)slot * HETGPU_PACC_ARG_SLOT_BYTES;
            memset(&header, 0, sizeof(header));
            g_last_arg_header_candidate = 0xffffffffu;
            mirror_diag_event(fd, job_id, 0, 0x52010000u | job_id,
                              (pacc << 8) | (uint32_t)i);
            {
                uint64_t phys = g_ddr_info.ddr_base +
                    shared_ddr_control_rel(pacc, slot_off);
                if (read_shared_ddr_control_copy_pread_for_pacc(fd, pacc, slot_off,
                                                                sizeof(header),
                                                                &slot_bytes) != 0 &&
                    read_phys_copy(fd, phys, sizeof(header), &slot_bytes) != 0 &&
                    !read_current_control_window_bytes(pacc, slot_off,
                                                       &header, sizeof(header))) {
                    mirror_diag_event(fd, job_id, 0, 0x52020000u | job_id,
                                      (pacc << 8) | (uint32_t)i);
                    continue;
                }
            }
            if (slot_bytes) {
                memcpy(&header, slot_bytes, sizeof(header));
            }
            free(slot_bytes);
            if (job_id == HETGPU_PACC_JOB_GEMM) {
                trace_msg("arg scan GEMM pacc=%u slot_off=0x%" PRIx64
                          " magic=0x%" PRIx64 " version=%u job_id=%u seq=%" PRIu64
                          " arg_len=%" PRIu64 " cand=0x%x",
                          pacc, slot_off, header.magic, header.version, header.job_id,
                          header.seq, header.arg_len, g_last_arg_header_candidate);
            }
            if (header.job_id == 0 && arg_slot_header_valid(job_id, &header)) {
                header.job_id = job_id;
            }
            if (!arg_slot_header_valid(job_id, &header)) {
                uint32_t aux = arg_slot_header_valid_bits(job_id, &header);
                if (g_last_arg_header_candidate != 0xffffffffu) {
                    aux |= (g_last_arg_header_candidate & 0xffu) << 16;
                }
                aux |= (pacc & 0xffu) << 24;
                mirror_diag_event(fd, job_id, header.seq, 0x52030000u | job_id, aux);
                continue;
            }
            if (preloaded_job_seen(job_id, header.seq)) {
                mirror_diag_event(fd, job_id, header.seq, 0x52040000u | job_id,
                                  (pacc << 8) | (uint32_t)i);
                if (!jobd_redispatch_seen_arg_slot_enabled()) {
                    continue;
                }
            }
            mirror_diag_event(fd, job_id, header.seq, 0x52050000u | job_id,
                              ((pacc & 0xffu) << 24) |
                                  (uint32_t)(header.arg_len & 0x00ffffffu));
            if (!have_best || header.seq > best.seq) {
                best = header;
                best_candidate = g_last_arg_header_candidate;
                best_pacc_id = pacc;
                have_best = true;
            }
        }
    }
    }

    if (!have_best) {
        mirror_diag_event(fd, 0, 0, 0x5206, 0);
        return false;
    }
    if (best_out) {
        *best_out = best;
    }
    g_pending_arg_header_candidate = best_candidate;
    g_pending_arg_header_pacc_id = best_pacc_id;
    g_last_arg_header_candidate = best_candidate;
    mirror_diag_event(fd, best.job_id, best.seq, 0x52070000u | best.job_id,
                      (uint32_t)(best.arg_len & 0xffffffffu));
    return true;
}

static void select_pacc_id_from_arg_slot_candidate(uint32_t candidate,
                                                   uint32_t job_id,
                                                   uint64_t seq) {
    if (candidate >= HETGPU_PACC_COUNT) {
        return;
    }
    if (g_pacc_id == candidate) {
        return;
    }
    if (getenv("HETGPU_PACC_ID") || getenv("PACC_JOBD_PACC_ID")) {
        trace_msg("ignoring arg-slot pacc id switch to %u while HETGPU_PACC_ID is fixed", candidate);
        return;
    }

    log_msg("switching pacc id from %" PRIu64 " to %u from arg-slot job_id=%u/%s seq=%" PRIu64,
            g_pacc_id, candidate, job_id, job_name(job_id), seq);
    g_pacc_id = candidate;
    /*
     * The direct control-window mmap was created for the old g_pacc_id.  Once
     * the active slot is learned from a pending arg-slot job, force completion
     * publication through the shared-DDR physical/pwrite path so the host sees
     * the record in the same slot it submitted.
     */
    g_control_window = NULL;
    g_control_map_base = NULL;
    g_control_map_len = 0;
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

struct GemmTileConfig {
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
    const char *name;
};

static size_t gemm_span(uint64_t rows, uint64_t cols, int64_t ld) {
    if (!rows || !cols) return 0;
    uint64_t lead = ld > 0 ? (uint64_t)ld : cols;
    return (size_t)((cols - 1) * lead + rows);
}

static size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case PACC_DTYPE_INT8:
        return sizeof(int8_t);
    case PACC_DTYPE_UINT8:
        return sizeof(uint8_t);
    case PACC_DTYPE_INT32:
        return sizeof(int32_t);
    case PACC_DTYPE_F16:
        return sizeof(uint16_t);
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

static float f16_to_f32(uint16_t x) {
    uint32_t sign = ((uint32_t)x & 0x8000U) << 16;
    uint32_t exp = ((uint32_t)x >> 10) & 0x1fU;
    uint32_t frac = (uint32_t)x & 0x03ffU;
    uint32_t bits;
    union {
        uint32_t u;
        float f;
    } v;

    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            exp = 127U - 15U + 1U;
            while ((frac & 0x0400U) == 0) {
                frac <<= 1;
                exp--;
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

__attribute__((noinline))
#if defined(HETGPU_PACC_HAVE_XSFVFWMACCQQQ)
static int xsfmm_smoke_bf16_kernel(uint16_t *a_bf16,
                                   uint16_t *b_bf16,
                                   float *c_f32) {
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    const __bf16 *a = (const __bf16 *)a_bf16;
    const __bf16 *b = (const __bf16 *)b_bf16;
    size_t vl = __riscv_vsetvl_e32m2(16);
    vfloat32m2_t acc = __riscv_vle32_v_f32m2(c_f32, vl);
    vbfloat16m1_t va = __riscv_vle16_v_bf16m1(a, vl);
    vbfloat16m1_t vb = __riscv_vle16_v_bf16m1(b, vl);
    acc = __riscv_sf_vfwmacc_4x4x4_f32m2(acc, va, vb, vl);
    __riscv_vse32_v_f32m2(c_f32, acc, vl);
    return 0;
#else
    (void)a_bf16;
    (void)b_bf16;
    (void)c_f32;
    return 95;
#endif
}
#else
static int xsfmm_smoke_bf16_kernel(uint16_t *a_bf16,
                                   uint16_t *b_bf16,
                                   float *c_f32) {
    (void)a_bf16;
    (void)b_bf16;
    (void)c_f32;
    return 77;
}
#endif

static void xsfmm_pack_b_tile_4x4(const uint16_t *b_rowmajor,
                                  uint16_t *b_packed,
                                  bool transposed_pack) {
    for (uint32_t k = 0; k < 4; k++) {
        for (uint32_t col = 0; col < 4; col++) {
            uint32_t src = k * 4 + col;
            uint32_t dst = transposed_pack ? col * 4 + k : k * 4 + col;
            b_packed[dst] = b_rowmajor[src];
        }
    }
}

static size_t xsfmm_c_idx_4x4(uint32_t row, uint32_t col, bool transposed_pack) {
    return transposed_pack ? (size_t)col * 4U + row : (size_t)row * 4U + col;
}

static void xsfmm_reference_bf16_4x4(const uint16_t *a,
                                     const uint16_t *b_rowmajor,
                                     const float *c_initial,
                                     float *ref) {
    for (uint32_t row = 0; row < 4; row++) {
        for (uint32_t col = 0; col < 4; col++) {
            float acc = c_initial[row * 4 + col];
            for (uint32_t k = 0; k < 4; k++) {
                acc += bf16_to_f32(a[row * 4 + k]) *
                       bf16_to_f32(b_rowmajor[k * 4 + col]);
            }
            ref[row * 4 + col] = acc;
        }
    }
}

static int xsfmm_try_layout_bf16_4x4(const uint16_t *a,
                                     const uint16_t *b_rowmajor,
                                     const float *c_initial,
                                     bool b_transposed_pack,
                                     bool c_transposed_pack,
                                     float *max_abs_out,
                                     float *checksum_out) {
    uint16_t b_packed[16];
    float ref[16];
    float c_packed[16];
    float max_abs = 0.0f;
    float checksum = 0.0f;
    int ret;

    xsfmm_pack_b_tile_4x4(b_rowmajor, b_packed, b_transposed_pack);
    xsfmm_reference_bf16_4x4(a, b_rowmajor, c_initial, ref);
    for (uint32_t row = 0; row < 4; row++) {
        for (uint32_t col = 0; col < 4; col++) {
            c_packed[xsfmm_c_idx_4x4(row, col, c_transposed_pack)] =
                c_initial[row * 4 + col];
        }
    }

    ret = xsfmm_smoke_bf16_kernel((uint16_t *)a, b_packed, c_packed);
    if (ret != 0) {
        return ret;
    }

    for (uint32_t row = 0; row < 4; row++) {
        for (uint32_t col = 0; col < 4; col++) {
            float got = c_packed[xsfmm_c_idx_4x4(row, col, c_transposed_pack)];
            float diff = fabsf(got - ref[row * 4 + col]);
            if (diff > max_abs) {
                max_abs = diff;
            }
            checksum += got;
        }
    }
    if (max_abs_out) {
        *max_abs_out = max_abs;
    }
    if (checksum_out) {
        *checksum_out = checksum;
    }
    return max_abs <= 0.25f ? 0 : 93;
}

static int run_xsfmm_smoke_bf16_once(float *checksum_out,
                                     float *max_abs_out) {
    uint16_t a[16];
    uint16_t b[16];
    float c_initial[16];
    float best_checksum = 0.0f;
    float best_err = INFINITY;
    int best_layout = -1;

    for (uint32_t row = 0; row < 4; row++) {
        for (uint32_t k = 0; k < 4; k++) {
            a[row * 4 + k] = f32_to_bf16((float)((int32_t)row * 3 + (int32_t)k + 1));
        }
    }
    for (uint32_t k = 0; k < 4; k++) {
        for (uint32_t col = 0; col < 4; col++) {
            b[k * 4 + col] = f32_to_bf16((float)((int32_t)k - (int32_t)col * 2 + 3));
        }
    }
    for (uint32_t i = 0; i < 16; i++) {
        c_initial[i] = (float)((int32_t)(i % 5) - 2);
    }

    for (int layout = 0; layout < 4; layout++) {
        bool b_transposed = (layout & 0x1) != 0;
        bool c_transposed = (layout & 0x2) != 0;
        float err = INFINITY;
        float checksum = 0.0f;
        int ret = xsfmm_try_layout_bf16_4x4(a, b, c_initial,
                                            b_transposed, c_transposed,
                                            &err, &checksum);
        if (ret == 0 && err < best_err) {
            best_err = err;
            best_checksum = checksum;
            best_layout = layout;
        }
    }

    if (best_layout < 0) {
        return 94;
    }
    if (checksum_out) {
        *checksum_out = best_checksum;
    }
    if (max_abs_out) {
        *max_abs_out = best_err;
    }
    return 32 + best_layout;
}

static void run_xsfmm_smoke_if_requested(void) {
    bool requested = jobd_xsfmm_gemm_requested();
    bool verbose = jobd_xsfmm_smoke_enabled();
    if ((!requested && !verbose) || !jobd_startup_xsfmm_smoke_enabled()) {
        g_xsfmm_bf16_checked = true;
        g_xsfmm_bf16_usable = g_xsfmm_context_ready;
        jobd_apply_xsfmm_layout_env();
        if (verbose || requested) {
            log_msg("xsfmm startup smoke skipped usable=%d context_ready=%d "
                    "b_transposed=%d c_transposed=%d",
                    g_xsfmm_bf16_usable ? 1 : 0,
                    g_xsfmm_context_ready ? 1 : 0,
                    g_xsfmm_b_transposed_pack ? 1 : 0,
                    g_xsfmm_c_transposed_pack ? 1 : 0);
        }
        return;
    }

    g_xsfmm_bf16_checked = true;
    g_xsfmm_bf16_usable = false;
    jobd_apply_xsfmm_layout_env();

#if !defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    if (verbose || requested) {
        log_msg("xsfmm unavailable in this jobd image; BF16 GEMM is fail-closed");
    }
    return;
#else
    pid_t pid = fork();
    if (pid == 0) {
        float checksum = 0.0f;
        float max_abs = INFINITY;
        int ret = run_xsfmm_smoke_bf16_once(&checksum, &max_abs);
        if (ret >= 32 && ret < 36) {
            int layout = ret - 32;
            log_msg("xsfmm smoke child ok layout=%d b_transposed=%d c_transposed=%d checksum=%g max_abs=%g",
                    layout, (layout & 0x1) != 0, (layout & 0x2) != 0,
                    checksum, max_abs);
        }
        _exit(ret);
    }
    if (pid < 0) {
        log_msg("xsfmm smoke fork failed: %s", strerror(errno));
        return;
    }

    uint64_t timeout_ms = jobd_xsfmm_smoke_timeout_ms();
    uint64_t deadline = monotonic_us() + timeout_ms * 1000ULL;
    for (;;) {
        int status = 0;
        pid_t got = waitpid(pid, &status, WNOHANG);
        if (got == pid) {
            if (WIFEXITED(status)) {
                int code = WEXITSTATUS(status);
                if (code >= 32 && code < 36) {
                    int layout = code - 32;
                    g_xsfmm_bf16_usable = true;
                    g_xsfmm_b_transposed_pack = (layout & 0x1) != 0;
                    g_xsfmm_c_transposed_pack = (layout & 0x2) != 0;
                } else {
                    g_xsfmm_bf16_usable = false;
                    g_xsfmm_b_transposed_pack = false;
                    g_xsfmm_c_transposed_pack = false;
                }
                if (verbose || !g_xsfmm_bf16_usable) {
                    log_msg("xsfmm smoke exit=%d usable=%d b_transposed=%d c_transposed=%d",
                            code, g_xsfmm_bf16_usable ? 1 : 0,
                            g_xsfmm_b_transposed_pack ? 1 : 0,
                            g_xsfmm_c_transposed_pack ? 1 : 0);
                }
            } else if (WIFSIGNALED(status)) {
                log_msg("xsfmm smoke signal=%d; BF16 GEMM is fail-closed",
                        WTERMSIG(status));
            }
            return;
        }
        if (got < 0) {
            log_msg("xsfmm smoke wait failed: %s", strerror(errno));
            return;
        }
        if (timeout_ms != 0 && monotonic_us() >= deadline) {
            kill(pid, SIGKILL);
            (void)waitpid(pid, &status, 0);
            log_msg("xsfmm smoke timed out after %" PRIu64
                    " ms; BF16 GEMM is fail-closed", timeout_ms);
            return;
        }
        sleep_us(1000);
    }
#endif
}

static uint16_t f32_to_f16(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign;
    uint32_t mant;
    int exp;

    v.f = x;
    sign = (v.u >> 16) & 0x8000U;
    mant = v.u & 0x007fffffU;
    exp = (int)((v.u >> 23) & 0xffU) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000U;
        return (uint16_t)(sign | (((mant >> (1 - exp)) + 0x1000U) >> 13));
    }
    if (exp >= 0x1f) {
        return (uint16_t)(sign | 0x7c00U);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000U) >> 13));
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
    if (dtype == PACC_DTYPE_F16) {
        return f16_to_f32(((const uint16_t *)base)[idx]);
    }
    if (dtype == PACC_DTYPE_F32) {
        return ((const volatile float *)base)[idx];
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
    } else if (dtype == PACC_DTYPE_F16) {
        ((uint16_t *)base)[idx] = f32_to_f16(value);
    } else if (dtype == PACC_DTYPE_F32) {
        ((volatile float *)base)[idx] = value;
    } else if (dtype == PACC_DTYPE_BF16) {
        ((uint16_t *)base)[idx] = f32_to_bf16(value);
    }
}

static void mirror_rmsnorm_debug_sample(int fd,
                                        const struct RmsNormJob *job,
                                        uint64_t seq,
                                        const void *x,
                                        const void *weight,
                                        const void *y,
                                        uint32_t phase,
                                        uint32_t flags) {
    if (!jobd_rms_debug_enabled() || !job || !x || !y || !job->hidden ||
        job->hidden > (uint64_t)SIZE_MAX) {
        return;
    }

    float sumsq = 0.0f;
    for (uint64_t i = 0; i < job->hidden; i++) {
        float v = load_typed(x, (size_t)i, job->dtype);
        sumsq += v * v;
    }

    struct RmsNormDebugRecord dbg;
    memset(&dbg, 0, sizeof(dbg));
    dbg.magic = HETGPU_PACC_RMS_DEBUG_MAGIC;
    dbg.version = HETGPU_PACC_JOB_VERSION;
    dbg.pacc_id = (uint32_t)g_pacc_id;
    dbg.phase = phase;
    dbg.dtype = job->dtype;
    dbg.seq = seq;
    dbg.row = 0;
    dbg.rows = job->rows;
    dbg.hidden = job->hidden;
    dbg.x_addr = job->x_addr;
    dbg.weight_addr = job->weight_addr;
    dbg.y_addr = job->y_addr;
    dbg.eps = job->eps;
    dbg.sumsq = sumsq;
    dbg.mean = sumsq / (float)job->hidden;
    dbg.scale = rsqrtf_newton(dbg.mean + job->eps);
    dbg.x0 = load_typed(x, 0, job->dtype);
    dbg.w0 = weight ? load_typed(weight, 0, job->dtype) : 1.0f;
    dbg.y0 = load_typed(y, 0, job->dtype);
    dbg.x_last = load_typed(x, (size_t)(job->hidden - 1), job->dtype);
    dbg.w_last = weight ? load_typed(weight, (size_t)(job->hidden - 1), job->dtype) : 1.0f;
    dbg.y_last = load_typed(y, (size_t)(job->hidden - 1), job->dtype);
    dbg.flags = flags;
    mirror_rmsnorm_debug_record(fd, &dbg);
}

static void mirror_rmsnorm_phase_record(int fd,
                                        const struct RmsNormJob *job,
                                        uint64_t seq,
                                        uint32_t phase,
                                        uint32_t flags,
                                        uint32_t status) {
    if (!jobd_rms_debug_enabled()) {
        return;
    }

    struct RmsNormDebugRecord dbg;
    memset(&dbg, 0, sizeof(dbg));
    dbg.magic = HETGPU_PACC_RMS_DEBUG_MAGIC;
    dbg.version = HETGPU_PACC_JOB_VERSION;
    dbg.pacc_id = (uint32_t)g_pacc_id;
    dbg.phase = phase;
    dbg.dtype = job ? job->dtype : 0;
    dbg.seq = seq;
    dbg.rows = job ? job->rows : 0;
    dbg.hidden = job ? job->hidden : 0;
    dbg.x_addr = job ? job->x_addr : 0;
    dbg.weight_addr = job ? job->weight_addr : 0;
    dbg.y_addr = job ? job->y_addr : 0;
    dbg.eps = job ? job->eps : 0.0f;
    dbg.flags = flags;
    dbg.reserved = status;
    mirror_rmsnorm_debug_record(fd, &dbg);
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

static uint64_t min_u64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

static bool gemm_dtype_uses_xsfmm32a16f(uint32_t dtype) {
    return dtype == PACC_DTYPE_BF16 || dtype == PACC_DTYPE_F16;
}

static bool gemm_job_is_compact_rowmajor(const struct GemmJob *job) {
    return job && !job->transa && !job->transb &&
           job->lda == (int64_t)job->k &&
           job->ldb == (int64_t)job->n &&
           job->ldc == (int64_t)job->n;
}

static int gemm_select_notrans_tile_config(const struct GemmJob *job,
                                           struct GemmTileConfig *cfg) {
    if (!jobd_gemm_tiled_enabled() || !gemm_job_is_compact_rowmajor(job) || !cfg) {
        return 0;
    }

    if (jobd_bf16_skinny_copy_enabled() &&
        job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16 &&
        job->ctype == PACC_DTYPE_BF16 && job->n <= 4) {
        return 0;
    }

    if (gemm_dtype_uses_xsfmm32a16f(job->atype) &&
        job->atype == job->btype &&
        (job->ctype == PACC_DTYPE_F32 ||
         job->ctype == PACC_DTYPE_F16 ||
         job->ctype == PACC_DTYPE_BF16)) {
        *cfg = (struct GemmTileConfig){
            .tile_m = PACC_XSFMM16_TILE_M,
            .tile_n = PACC_XSFMM16_TILE_N,
            .tile_k = PACC_XSFMM16_TILE_K,
            .name = job->atype == PACC_DTYPE_BF16
                ? "xsfmm32a16f-bf16"
                : "xsfmm32a16f-f16-soft",
        };
        return 1;
    }

    if (job->atype == PACC_DTYPE_F32 &&
        job->btype == PACC_DTYPE_F32 &&
        job->ctype == PACC_DTYPE_F32) {
        *cfg = (struct GemmTileConfig){
            .tile_m = PACC_RVV_F32_TILE_M,
            .tile_n = PACC_RVV_F32_TILE_N,
            .tile_k = PACC_RVV_F32_TILE_K,
            .name = "rvv-f32-soft",
        };
        return 1;
    }

    return 0;
}

static bool gemm_xsfmm_bf16_4x4_accumulate(const struct GemmJob *job,
                                           const void *a,
                                           const void *b,
                                           uint64_t row0,
                                           uint64_t col0,
                                           uint64_t k0,
                                           float *acc,
                                           uint64_t acc_stride) {
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    if (job->atype != PACC_DTYPE_BF16 || job->btype != PACC_DTYPE_BF16) {
        return false;
    }
    uint16_t atile[16];
    uint16_t btile[16];
    float ctile[16];

    for (uint64_t kk = 0; kk < 4; kk++) {
        for (uint64_t row = 0; row < 4; row++) {
            atile[kk * 4 + row] =
                ((const uint16_t *)a)[(row0 + row) * (uint64_t)job->lda + (k0 + kk)];
        }
    }
    for (uint64_t kk = 0; kk < 4; kk++) {
        for (uint64_t col = 0; col < 4; col++) {
            uint64_t dst = kk * 4 + col;
            btile[dst] =
                ((const uint16_t *)b)[(k0 + kk) * (uint64_t)job->ldb + (col0 + col)];
        }
    }
    for (uint64_t row = 0; row < 4; row++) {
        for (uint64_t col = 0; col < 4; col++) {
            ctile[row * 4 + col] = 0.0f;
        }
    }

    if (xsfmm_native_bf16(atile, btile, ctile, 4, 4, 4) != 0) {
        return false;
    }

    for (uint64_t row = 0; row < 4; row++) {
        for (uint64_t col = 0; col < 4; col++) {
            acc[row * acc_stride + col] =
                ctile[row * 4 + col];
        }
    }
    return true;
#else
    (void)job;
    (void)a;
    (void)b;
    (void)row0;
    (void)col0;
    (void)k0;
    (void)acc;
    (void)acc_stride;
    return false;
#endif
}

static void gemm_accumulate_rowmajor_outer_tile(const struct GemmJob *job,
                                                const void *a,
                                                const void *b,
                                                uint64_t row0,
                                                uint64_t row1,
                                                uint64_t col0,
                                                uint64_t col1,
                                                uint64_t k0,
                                                uint64_t k1,
                                                float *acc,
                                                uint64_t acc_stride) {
    float bvals[PACC_GEMM_TILE_N_MAX];
    uint64_t cols = col1 - col0;
    uint64_t fast_row1 = row0;
    uint64_t fast_col1 = col0;
    bool use_xsfmm = false;

    if (cols > PACC_GEMM_TILE_N_MAX) {
        return;
    }

    /*
     * Fast BF16 4x4x4 body for compact row-major staging.  The remainder path
     * below handles non-BF16 and tail rows/columns without changing semantics.
     */
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    if (jobd_xsfmm_gemm_enabled() &&
        job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16 &&
        job->n <= jobd_xsfmm_gemm_max_n() &&
        k1 - k0 == 4) {
        fast_row1 = row0 + ((row1 - row0) / 4) * 4;
        fast_col1 = col0 + ((col1 - col0) / 4) * 4;
        use_xsfmm = fast_row1 > row0 && fast_col1 > col0;
        if (use_xsfmm) {
            for (uint64_t rb = row0; rb < fast_row1; rb += 4) {
                for (uint64_t cb = col0; cb < fast_col1; cb += 4) {
                    (void)gemm_xsfmm_bf16_4x4_accumulate(
                        job, a, b, rb, cb, k0,
                        acc + (rb - row0) * acc_stride + (cb - col0),
                        acc_stride);
                }
            }
        }
    }
#endif

    for (uint64_t kk = k0; kk < k1; kk++) {
        for (uint64_t col = col0; col < col1; col++) {
            bvals[col - col0] =
                load_typed(b, (size_t)(kk * (uint64_t)job->ldb + col), job->btype);
        }
        for (uint64_t row = row0; row < row1; row++) {
            if (use_xsfmm && row < fast_row1) {
                bool has_fast_cols = fast_col1 > col0;
                if (has_fast_cols && fast_col1 == col1) {
                    continue;
                }
            }
            float av = load_typed(a, (size_t)(row * (uint64_t)job->lda + kk), job->atype);
            float *acc_row = acc + (row - row0) * acc_stride;
#if defined(__riscv_vector)
            uint64_t off = 0;
            while (off < cols) {
                if (use_xsfmm && row < fast_row1 && col0 + off < fast_col1) {
                    off = fast_col1 - col0;
                    continue;
                }
                size_t vl = __riscv_vsetvl_e32m4(cols - off);
                vfloat32m4_t vacc = __riscv_vle32_v_f32m4(acc_row + off, vl);
                vfloat32m4_t vb = __riscv_vle32_v_f32m4(bvals + off, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, av, vb, vl);
                __riscv_vse32_v_f32m4(acc_row + off, vacc, vl);
                off += vl;
            }
#else
            for (uint64_t col = col0; col < col1; col++) {
                if (use_xsfmm && row < fast_row1 && col < fast_col1) {
                    continue;
                }
                acc_row[col - col0] += av * bvals[col - col0];
            }
#endif
        }
    }
}

static void gemm_store_rowmajor_tile(const struct GemmJob *job,
                                     void *c,
                                     float alpha,
                                     float beta,
                                     uint64_t row0,
                                     uint64_t row1,
                                     uint64_t col0,
                                     uint64_t col1,
                                     const float *acc,
                                     uint64_t acc_stride) {
    for (uint64_t row = row0; row < row1; row++) {
        for (uint64_t col = col0; col < col1; col++) {
            size_t c_idx = (size_t)(row * (uint64_t)job->ldc + col);
            float old = beta != 0.0f ? load_typed(c, c_idx, job->ctype) : 0.0f;
            float value = acc[(row - row0) * acc_stride + (col - col0)];
            store_typed(c, c_idx, job->ctype, alpha * value + beta * old);
        }
    }
}

static int gemm_try_notrans_tiled_worker(const struct GemmWorker *w) {
    const struct GemmJob *job = w->job;
    struct GemmTileConfig cfg;
    uint64_t row_end;

    if (!gemm_select_notrans_tile_config(job, &cfg)) {
        return 0;
    }
    if (cfg.tile_m > PACC_GEMM_TILE_M_MAX || cfg.tile_n > PACC_GEMM_TILE_N_MAX ||
        cfg.tile_m == 0 || cfg.tile_n == 0 || cfg.tile_k == 0) {
        return 0;
    }

    row_end = min_u64(w->row_end, job->m);
    trace_msg("GEMM tiled %s m=%" PRIu64 " n=%" PRIu64 " k=%" PRIu64
              " rows=%" PRIu64 "..%" PRIu64 " tile=%ux%ux%u",
              cfg.name,
              job->m, job->n, job->k,
              w->row_begin, row_end,
              cfg.tile_m, cfg.tile_n, cfg.tile_k);
    for (uint64_t row0 = w->row_begin; row0 < row_end; row0 += cfg.tile_m) {
        uint64_t row1 = min_u64(row0 + cfg.tile_m, row_end);
        for (uint64_t col0 = 0; col0 < job->n; col0 += cfg.tile_n) {
            uint64_t col1 = min_u64(col0 + cfg.tile_n, job->n);
            float acc[PACC_GEMM_TILE_M_MAX * PACC_GEMM_TILE_N_MAX];
            memset(acc, 0, sizeof(acc));
            for (uint64_t k0 = 0; k0 < job->k; k0 += cfg.tile_k) {
                uint64_t k1 = min_u64(k0 + cfg.tile_k, job->k);
                gemm_accumulate_rowmajor_outer_tile(job, w->a, w->b, row0, row1,
                                                    col0, col1, k0, k1,
                                                    acc, cfg.tile_n);
            }
            gemm_store_rowmajor_tile(job, w->c, w->alpha, w->beta, row0, row1,
                                     col0, col1, acc, cfg.tile_n);
        }
    }
    return 1;
}

static void *gemm_worker_main(void *arg) {
    struct GemmWorker *w = (struct GemmWorker *)arg;
    const struct GemmJob *job = w->job;
    if (gemm_try_notrans_tiled_worker(w)) {
        return NULL;
    }
    bool compact_rowmajor = gemm_job_is_compact_rowmajor(job);
    for (uint64_t row = w->row_begin; row < w->row_end; row++) {
        for (uint64_t col = 0; col < job->n; col++) {
            size_t a_base = compact_rowmajor
                ? (size_t)(row * (uint64_t)job->lda)
                : (job->transa ? (size_t)(row * (uint64_t)job->lda) : (size_t)row);
            ptrdiff_t a_stride = compact_rowmajor
                ? 1
                : (job->transa ? 1 : (ptrdiff_t)job->lda);
            size_t b_base = compact_rowmajor
                ? (size_t)col
                : (job->transb ? (size_t)col : (size_t)(col * (uint64_t)job->ldb));
            ptrdiff_t b_stride = compact_rowmajor
                ? (ptrdiff_t)job->ldb
                : (job->transb ? (ptrdiff_t)job->ldb : 1);
            const void *ap = (const char *)w->a + a_base * dtype_size(job->atype);
            const void *bp = (const char *)w->b + b_base * dtype_size(job->btype);
            size_t c_idx = compact_rowmajor
                ? (size_t)(row * (uint64_t)job->ldc + col)
                : (size_t)(row + col * (uint64_t)job->ldc);
            float acc = gemm_dot_typed(ap, job->atype, a_stride, bp, job->btype, b_stride, job->k);
            float old = w->beta != 0.0f ? load_typed(w->c, c_idx, job->ctype) : 0.0f;
            store_typed(w->c, c_idx, job->ctype, w->alpha * acc + w->beta * old);
        }
    }
    return NULL;
}

static uint64_t jobd_gemm_single_thread_ops_threshold(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_GEMM_SINGLE_THREAD_OPS", 0);
}

static unsigned jobd_gemm_worker_threads(void) {
    uint64_t requested = parse_env_u64_default("HETGPU_PACC_JOBD_GEMM_THREADS", 0);
    if (requested == 0) {
        requested = parse_env_u64_default("PACC_JOBD_GEMM_THREADS", 0);
    }
    if (requested == 0) {
        requested = parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_THREADS", 0);
    }
    if (requested == 0) {
        requested = parse_env_u64_default("PACC_JOBD_KERNEL_THREADS", 0);
    }
    if (requested == 0) {
        requested = PACC_GEMM_THREADS;
    }
    if (requested > PACC_GEMM_THREADS) {
        requested = PACC_GEMM_THREADS;
    }
    if (requested < 1) {
        requested = 1;
    }
    return (unsigned)requested;
}

static int run_gemm_matrix_threads(const struct GemmJob *job, const void *a,
                                   const void *b, void *c, float alpha, float beta) {
    pthread_t threads[PACC_GEMM_THREADS];
    struct GemmWorker workers[PACC_GEMM_THREADS];
    unsigned nthreads = jobd_gemm_worker_threads();
    int started = 0;
    uint64_t threshold = jobd_gemm_single_thread_ops_threshold();
    uint64_t ops = UINT64_MAX;
    bool xsfmm_row_threading = false;

    if (job->m != 0 && job->n <= UINT64_MAX / job->m) {
        uint64_t mn = job->m * job->n;
        if (job->k != 0 && mn <= UINT64_MAX / job->k) {
            ops = mn * job->k;
        }
    }

#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    if (jobd_xsfmm_gemm_enabled() &&
        job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16 &&
        !job->transa && !job->transb) {
        unsigned row_blocks = (unsigned)(job->m / 4U);
        if (row_blocks == 0) row_blocks = 1;
        if (nthreads > row_blocks) nthreads = row_blocks;
        xsfmm_row_threading = true;
    }
#endif

    if ((threshold != 0 && ops <= threshold) ||
        nthreads <= 1 ||
        (!xsfmm_row_threading && job->m < PACC_GEMM_THREADS)) {
        struct GemmWorker worker = {
            .job = job,
            .a = a,
            .b = b,
            .c = c,
            .row_begin = 0,
            .row_end = job->m,
            .alpha = alpha,
            .beta = beta,
        };
        gemm_worker_main(&worker);
        return 0;
    }

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
    if (job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16 &&
        beta == 0.0f && started > 1) {
        /*
         * LX500 pthread scheduling/cache visibility occasionally leaves the
         * final row worker's stores invisible for staged BF16 GEMM.  Replaying
         * that small tail on the dispatcher thread is idempotent for the
         * staged path (beta is forced to zero per chunk) and preserves the
         * parallel work for the rest of the tile.
         */
        gemm_worker_main(&workers[started - 1]);
    }
    return 0;
}

#define XSFMM_PACKED_A_CACHE_ENTRIES 64U

struct XsfmmPackedACacheEntry {
    bool valid;
    uint64_t a_addr;
    uint64_t m;
    uint64_t k;
    uint64_t batch_count;
    uint64_t a_batch_stride;
    int64_t lda;
    uint32_t compute_type;
    uint64_t age;
    uint16_t *data;
    size_t bytes;
};

static struct XsfmmPackedACacheEntry
    g_xsfmm_packed_a_cache[XSFMM_PACKED_A_CACHE_ENTRIES];
static uint64_t g_xsfmm_packed_a_cache_age;

static uint16_t *xsfmm_packed_a_cache_lookup(
    const struct GemmJob *job,
    uint64_t batch_count,
    uint64_t a_batch_stride) {
    if (!job) return NULL;
    /*
     * The upper compute_type bits carry a host-provided immutable-A
     * generation tag.  Untagged GEMMs may reuse an address with new data.
     */
    if ((job->compute_type >> 8) == 0) return NULL;
    for (size_t i = 0; i < XSFMM_PACKED_A_CACHE_ENTRIES; i++) {
        struct XsfmmPackedACacheEntry *entry = &g_xsfmm_packed_a_cache[i];
        if (entry->valid &&
            entry->a_addr == job->a_addr &&
            entry->m == job->m &&
            entry->k == job->k &&
            entry->batch_count == batch_count &&
            entry->a_batch_stride == a_batch_stride &&
            entry->lda == job->lda &&
            entry->compute_type == job->compute_type) {
            entry->age = ++g_xsfmm_packed_a_cache_age;
            return entry->data;
        }
    }
    return NULL;
}

static uint16_t *xsfmm_packed_a_cache_insert(
    const struct GemmJob *job,
    uint64_t batch_count,
    uint64_t a_batch_stride,
    uint16_t *data,
    size_t bytes) {
    struct XsfmmPackedACacheEntry *victim = NULL;

    if (!job || !data || !bytes) return NULL;
    for (size_t i = 0; i < XSFMM_PACKED_A_CACHE_ENTRIES; i++) {
        struct XsfmmPackedACacheEntry *entry = &g_xsfmm_packed_a_cache[i];
        if (!entry->valid) {
            victim = entry;
            break;
        }
        if (!victim || entry->age < victim->age) {
            victim = entry;
        }
    }
    if (!victim) return NULL;
    free(victim->data);
    *victim = (struct XsfmmPackedACacheEntry) {
        .valid = true,
        .a_addr = job->a_addr,
        .m = job->m,
        .k = job->k,
        .batch_count = batch_count,
        .a_batch_stride = a_batch_stride,
        .lda = job->lda,
        .compute_type = job->compute_type,
        .age = ++g_xsfmm_packed_a_cache_age,
        .data = data,
        .bytes = bytes,
    };
    return victim->data;
}

static int run_gemm_xsfmm_hardware_only(const struct GemmJob *job,
                                        const void *a,
                                        const void *b,
                                        void *c,
                                        float alpha,
                                        float beta,
                                        uint64_t batch_count,
                                        uint64_t a_batch_stride,
                                        uint64_t b_batch_stride,
                                        uint64_t c_batch_stride) {
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
    uint16_t *a_pack = NULL;
    const uint16_t *b_pack = (const uint16_t *)b;
    float *c_tile = NULL;
    uint64_t packed_a_stride;
    uint64_t packed_b_stride;
    uint64_t packed_c_stride;
    bool a_pack_owned = false;
    int status = 0;

    if (!job || !b || !c || job->atype != PACC_DTYPE_BF16 ||
        job->btype != PACC_DTYPE_BF16 || job->transa || job->transb ||
        !gemm_job_is_compact_rowmajor(job) ||
        job->m > jobd_xsfmm_gemm_max_m() ||
        job->n > PACC_XSFMM16_TILE_N || job->n > jobd_xsfmm_gemm_max_n() ||
        (job->k & 1u) != 0 ||
        (job->ctype != PACC_DTYPE_F32 && job->ctype != PACC_DTYPE_F16 &&
         job->ctype != PACC_DTYPE_BF16) ||
        !batch_count || batch_count > 64) {
        return 0xffff1f20;
    }

    packed_a_stride = job->k * job->m;
    packed_b_stride = b_batch_stride;
    packed_c_stride = job->m * job->n;
    a_pack = xsfmm_packed_a_cache_lookup(
        job, batch_count, a_batch_stride);
    if (!a_pack) {
        if (!a) {
            return 0xffff1f24;
        }
        a_pack = (uint16_t *)malloc(
            (size_t)(packed_a_stride * batch_count) * sizeof(*a_pack));
        a_pack_owned = true;
    }
    c_tile = (float *)malloc(
        (size_t)(packed_c_stride * batch_count) * sizeof(*c_tile));
    if (!a_pack || !c_tile) {
        status = 0xffff1f21;
        goto out;
    }

    if (a_pack_owned) {
        for (uint64_t batch = 0; batch < batch_count; batch++) {
            const uint16_t *batch_a =
                (const uint16_t *)a + batch * a_batch_stride;
            uint16_t *packed_a = a_pack + batch * packed_a_stride;

            for (uint64_t kk = 0; kk < job->k; kk++) {
                for (uint64_t row = 0; row < job->m; row++) {
                    packed_a[kk * job->m + row] =
                        batch_a[row * (uint64_t)job->lda + kk];
                }
            }
        }
        uint16_t *cached = xsfmm_packed_a_cache_insert(
            job, batch_count, a_batch_stride, a_pack,
            (size_t)(packed_a_stride * batch_count) * sizeof(*a_pack));
        if (cached) {
            a_pack = cached;
            a_pack_owned = false;
        }
    }
    memset(c_tile, 0,
           (size_t)(packed_c_stride * batch_count) * sizeof(*c_tile));
    struct XsfmmRequest request = {
        .magic = HETGPU_XSFMM_REQUEST_MAGIC,
        .m = job->m,
        .n = job->n,
        .k = job->k,
        .repeats = jobd_xsfmm_repeats(),
        .status = -1,
        .a_batch_stride = packed_a_stride,
        .b_batch_stride = b_batch_stride == 0 ? 0 : packed_b_stride,
        .c_batch_stride = packed_c_stride,
    };
    uint64_t xsfmm_start_ns = monotonic_ns();
    uint64_t context_batch = parse_env_u64_default(
        "HETGPU_PACC_JOBD_XSFMM_CONTEXT_BATCH", 1);
    if (context_batch == 0 || context_batch > batch_count) {
        context_batch = batch_count;
    }
    for (uint64_t batch = 0; batch < batch_count; batch += context_batch) {
        uint64_t count = min_u64(context_batch, batch_count - batch);
        uint64_t command_budget = parse_env_u64_default(
            "HETGPU_PACC_JOBD_XSFMM_COMMAND_BUDGET", 480000);
        uint64_t commands_per_tile = job->k / 2 + job->m + 8;
        uint64_t requested_commands;

        if (commands_per_tile > UINT64_MAX / count ||
            commands_per_tile * count > UINT64_MAX / request.repeats) {
            status = 0xffff1f25;
            goto out;
        }
        requested_commands = commands_per_tile * count * request.repeats;

        if (command_budget != 0 &&
            (requested_commands > command_budget ||
             g_xsfmm_estimated_commands > command_budget - requested_commands)) {
            status = 0xffff1f25;
            goto out;
        }

        request.a = (uintptr_t)(a_pack + batch * packed_a_stride);
        request.b = (uintptr_t)(b_pack +
            (b_batch_stride == 0 ? 0 : batch * packed_b_stride));
        request.c = (uintptr_t)(c_tile + batch * packed_c_stride);
        request.batch_count = count;
        if (jobd_run_xsfmm_request(&request) != 0) {
            status = 0xffff1f23;
            goto out;
        }
        g_xsfmm_estimated_commands += requested_commands;
        g_last_gemm_timing.xsfmm_cycles += request.cycles;
        g_last_gemm_timing.xsfmm_repeats += request.completed_repeats;
    }
    uint64_t xsfmm_elapsed_ns = monotonic_ns() - xsfmm_start_ns;
    if (g_last_gemm_timing.xsfmm_cycles == 0) {
        g_last_gemm_timing.xsfmm_cycles = xsfmm_elapsed_ns;
    }

    for (uint64_t batch = 0; batch < batch_count; batch++) {
        void *batch_c = (uint8_t *)c +
            batch * c_batch_stride * dtype_size(job->ctype);
        const float *packed_c = c_tile + batch * packed_c_stride;

        for (uint64_t row = 0; row < job->m; row++) {
            for (uint64_t col = 0; col < job->n; col++) {
                size_t index = (size_t)(row * (uint64_t)job->ldc + col);
                float old = beta != 0.0f ?
                    load_typed(batch_c, index, job->ctype) : 0.0f;
                store_typed(batch_c, index, job->ctype,
                            alpha * packed_c[row * job->n + col] + beta * old);
            }
        }
    }

out:
    if (a_pack_owned) free(a_pack);
    free(c_tile);
    return status;
#else
    (void)job;
    (void)a;
    (void)b;
    (void)c;
    (void)alpha;
    (void)beta;
    (void)batch_count;
    (void)a_batch_stride;
    (void)b_batch_stride;
    (void)c_batch_stride;
    return 0xffff1f01;
#endif
}

static int run_gemm_impl(int fd, const struct GemmJob *job, uint64_t seq) {
    mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5120);
    if (!job->m || !job->n || !job->k || !job->a_addr || !job->b_addr || !job->c_addr) {
        return 0xffff1001;
    }
    size_t a_dtype_size = dtype_size(job->atype);
    size_t b_dtype_size = dtype_size(job->btype);
    size_t c_dtype_size = dtype_size(job->ctype);
    if (!a_dtype_size || !b_dtype_size || !c_dtype_size) {
        return 0xffff1002;
    }
    if (job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16 &&
        !jobd_xsfmm_gemm_enabled()) {
        uint32_t status = 0xffff1f00u | ((uint32_t)g_xsfmm_context_error & 0xffu);
        mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, status);
        log_msg("BF16 GEMM rejected: Xsfmm hardware context error=%d",
                g_xsfmm_context_error);
        return (int)status;
    }

    struct GemmJob norm = *job;
    if (norm.lda <= 0) norm.lda = norm.transa ? (int64_t)norm.m : (int64_t)norm.k;
    if (norm.ldb <= 0) norm.ldb = norm.transb ? (int64_t)norm.k : (int64_t)norm.n;
    if (norm.ldc <= 0) norm.ldc = (int64_t)norm.n;
    job = &norm;

    uint64_t batch_count = job->batch_count ? job->batch_count : 1;
    bool compact_rowmajor = gemm_job_is_compact_rowmajor(job);
    size_t a_matrix_elems = compact_rowmajor
        ? (size_t)(job->m * (uint64_t)job->lda)
        : (job->transa
            ? gemm_span(job->k, job->m, job->lda)
            : gemm_span(job->m, job->k, job->lda));
    size_t b_matrix_elems = compact_rowmajor
        ? (size_t)(job->k * (uint64_t)job->ldb)
        : (job->transb
            ? gemm_span(job->n, job->k, job->ldb)
            : gemm_span(job->k, job->n, job->ldb));
    size_t c_matrix_elems = compact_rowmajor
        ? (size_t)(job->m * (uint64_t)job->ldc)
        : gemm_span(job->m, job->n, job->ldc);
    /*
     * A negative stride means broadcast one matrix across the batch.  Zero
     * retains the cuBLAS-compatible packed-matrix default.
     */
    uint64_t a_batch_stride = job->stride_a < 0 ? 0 :
        (job->stride_a > 0 ? (uint64_t)job->stride_a : (uint64_t)a_matrix_elems);
    uint64_t b_batch_stride = job->stride_b < 0 ? 0 :
        (job->stride_b > 0 ? (uint64_t)job->stride_b : (uint64_t)b_matrix_elems);
    uint64_t c_batch_stride = job->stride_c < 0 ? 0 :
        (job->stride_c > 0 ? (uint64_t)job->stride_c : (uint64_t)c_matrix_elems);
    size_t a_elems = (size_t)(a_batch_stride * (batch_count - 1) + a_matrix_elems);
    size_t b_elems = (size_t)(b_batch_stride * (batch_count - 1) + b_matrix_elems);
    size_t c_elems = (size_t)(c_batch_stride * (batch_count - 1) + c_matrix_elems);
    size_t a_bytes = a_elems * a_dtype_size;
    size_t b_bytes = b_elems * b_dtype_size;
    size_t c_bytes = c_elems * c_dtype_size;
    struct Map ma = {0}, mb = {0}, mc = {0}, malpha = {0}, mbeta = {0};
    mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5121);

    if (jobd_gemm_copy_io_enabled()) {
        uint8_t *a_copy = NULL;
        uint8_t *b_copy = NULL;
        uint8_t *c_copy = NULL;
        uint8_t *scalar_copy = NULL;
        float alpha = 1.0f;
        float beta = 0.0f;
        int status = 0;

        bool xsfmm_bf16 =
            job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16;
        bool xsfmm_a_cached = xsfmm_bf16 &&
            xsfmm_packed_a_cache_lookup(
                job, batch_count, a_batch_stride) != NULL;
        if ((!xsfmm_a_cached &&
             read_phys_copy(fd, job->a_addr, a_bytes, &a_copy) != 0) ||
            read_phys_copy(fd, job->b_addr, b_bytes, &b_copy) != 0) {
            free(a_copy);
            free(b_copy);
            return 0xffff1003;
        }
        if (job->alpha_addr && read_phys_copy(fd, job->alpha_addr, sizeof(float), &scalar_copy) == 0) {
            memcpy(&alpha, scalar_copy, sizeof(alpha));
            free(scalar_copy);
            scalar_copy = NULL;
        }
        if (job->beta_addr && read_phys_copy(fd, job->beta_addr, sizeof(float), &scalar_copy) == 0) {
            memcpy(&beta, scalar_copy, sizeof(beta));
            free(scalar_copy);
            scalar_copy = NULL;
        }
        c_copy = (uint8_t *)malloc(c_bytes);
        if (!c_copy) {
            free(a_copy);
            free(b_copy);
            return 0xffff1004;
        }
        if (beta != 0.0f) {
            if (read_phys_copy(fd, job->c_addr, c_bytes, &scalar_copy) == 0) {
                memcpy(c_copy, scalar_copy, c_bytes);
                free(scalar_copy);
                scalar_copy = NULL;
            } else {
                memset(c_copy, 0, c_bytes);
            }
        } else {
            memset(c_copy, 0, c_bytes);
        }

        mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5122);
        mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5123);
        if (job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16) {
            status = run_gemm_xsfmm_hardware_only(
                job, a_copy, b_copy, c_copy, alpha, beta, batch_count,
                a_batch_stride, b_batch_stride, c_batch_stride);
        } else {
            for (uint64_t batch = 0; batch < batch_count; batch++) {
                const void *a = a_copy + a_batch_stride * batch * a_dtype_size;
                const void *b = b_copy + b_batch_stride * batch * b_dtype_size;
                void *c = c_copy + c_batch_stride * batch * c_dtype_size;
                if (run_gemm_matrix_threads(job, a, b, c, alpha, beta) != 0) {
                    status = 0xffff1004;
                    break;
                }
            }
        }
        mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5124);
        if (status == 0) {
            bool mapped_visible;
            bool publish_visible;
            mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x512401);
            mapped_visible = write_shared_ddr_devmem_direct(job->c_addr, c_copy, c_bytes);
            jobd_evict_after_payload_write();
            mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq,
                                   mapped_visible ? 0x512402 : 0xff512402u);

	            publish_visible = publish_shared_ddr_payload_visible(fd, job->c_addr,
	                                                                 c_copy, c_bytes,
	                                                                 "gemm-output");
	            if (!publish_visible && mapped_visible &&
	                env_flag_default_true("HETGPU_PACC_JOBD_GEMM_DEVMEM_VISIBLE_OK")) {
	                publish_visible = true;
	                mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x512426);
	            }
	            if (!publish_visible && env_flag_true(getenv("HETGPU_PACC_JOBD_GEMM_LINE_PWRITE"))) {
	                size_t line = jobd_cbo_block_bytes();
                if (line < 16) line = 16;
                if (line > 256) line = 256;
                for (size_t off = 0; off < c_bytes; off += line) {
                    size_t want = c_bytes - off;
                    if (want > line) want = line;
                    (void)write_phys_copy_pwrite_only(fd, job->c_addr + off, c_copy + off, want);
                }
                publish_visible = publish_shared_ddr_payload_visible(fd, job->c_addr,
                                                                     c_copy, c_bytes,
                                                                     "gemm-output-line");
            }
            repair_shared_ddr_writeback(fd, job->c_addr, c_copy, c_bytes, "gemm-output");
            if (!publish_visible) {
                uint64_t wait_us = parse_env_u64_default(
                    "HETGPU_PACC_JOBD_GEMM_OUTPUT_VISIBLE_WAIT_US", 0ULL);
                uint64_t sleep_step = parse_env_u64_default(
                    "HETGPU_PACC_JOBD_GEMM_OUTPUT_VISIBLE_SLEEP_US", 100ULL);
                uint64_t deadline = monotonic_us() + wait_us;
                mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x512407);
                while (wait_us != 0 && monotonic_us() < deadline) {
                    (void)write_shared_ddr_devmem_direct(job->c_addr, c_copy, c_bytes);
                    jobd_evict_after_payload_write();
                    publish_visible = publish_shared_ddr_payload_visible(fd, job->c_addr,
                                                                         c_copy, c_bytes,
                                                                         "gemm-output-visible");
                    if (publish_visible) {
                        break;
                    }
                    repair_shared_ddr_writeback(fd, job->c_addr, c_copy, c_bytes,
                                                "gemm-output-visible");
                    if (sleep_step != 0) {
                        sleep_us(sleep_step);
                    }
                }
            }
            mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq,
                                   publish_visible ? 0x5125 : 0xffff5125u);
            if (!publish_visible && env_flag_default_true("HETGPU_PACC_JOBD_GEMM_STRICT_VISIBLE")) {
                trace_msg("GEMM output not AP-visible: c=0x%" PRIx64
                          " bytes=0x%zx seq=%" PRIu64,
                          job->c_addr, c_bytes, seq);
                status = 0xffff1005;
            }
        }
        free(a_copy);
        free(b_copy);
        free(c_copy);
        free(scalar_copy);
        return status;
    }

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

    mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5122);
    const char *a0 = (const char *)ma.ptr;
    const char *b0 = (const char *)mb.ptr;
    char *c0 = (char *)mc.ptr;
    mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5123);
    if (job->atype == PACC_DTYPE_BF16 && job->btype == PACC_DTYPE_BF16) {
        int batch_status = run_gemm_xsfmm_hardware_only(
            job, a0, b0, c0, alpha, beta, batch_count,
            a_batch_stride, b_batch_stride, c_batch_stride);
        if (batch_status != 0) {
            unmap_phys(&malpha); unmap_phys(&mbeta); unmap_phys(&ma); unmap_phys(&mb); unmap_phys(&mc);
            return batch_status;
        }
    } else {
        for (uint64_t batch = 0; batch < batch_count; batch++) {
            const void *a = a0 + a_batch_stride * batch * a_dtype_size;
            const void *b = b0 + b_batch_stride * batch * b_dtype_size;
            void *c = c0 + c_batch_stride * batch * c_dtype_size;
            if (run_gemm_matrix_threads(job, a, b, c, alpha, beta) != 0) {
                unmap_phys(&malpha); unmap_phys(&mbeta); unmap_phys(&ma); unmap_phys(&mb); unmap_phys(&mc);
                return 0xffff1004;
            }
        }
    }
    mirror_progress_status(fd, HETGPU_PACC_JOB_GEMM, seq, 0x5124);
    if (jobd_msync_enabled()) {
        msync(mc.base, mc.map_len, MS_SYNC);
    }
    jobd_flush_for_device(mc.ptr, mc.len);
    repair_shared_ddr_writeback(fd, job->c_addr, mc.ptr, c_elems * c_dtype_size,
                                "gemm-output-mmap");
    unmap_phys(&malpha); unmap_phys(&mbeta); unmap_phys(&ma); unmap_phys(&mb); unmap_phys(&mc);
    return 0;
}

static int run_gemm(int fd, const struct GemmJob *job, uint64_t seq) {
    int status;

    memset(&g_last_gemm_timing, 0, sizeof(g_last_gemm_timing));
    g_last_gemm_timing.seq = seq;
    g_last_gemm_timing.compute_start_ns = monotonic_ns();
    status = run_gemm_impl(fd, job, seq);
    g_last_gemm_timing.compute_end_ns = monotonic_ns();
    return status;
}

static int run_softmax(int fd, const struct SoftmaxJob *job, uint64_t seq) {
    mirror_progress_status(fd, HETGPU_PACC_JOB_SOFTMAX, seq, 0x5220);
    if (!job->src_addr || !job->dst_addr || !job->rows || !job->cols) return 0xffff2001;
    size_t elem_size = dtype_size(job->dtype);
    if (!elem_size) return 0xffff2002;
    uint64_t stride = job->stride ? job->stride : job->cols;
    size_t elems = (size_t)(job->rows * stride);
    size_t bytes = elems * elem_size;
    if (jobd_softmax_local_copy_enabled()) {
        uint8_t *src_copy = NULL;
        uint8_t *dst_copy = NULL;
        mirror_progress_status(fd, HETGPU_PACC_JOB_SOFTMAX, seq, 0x5221);
        if (read_phys_copy(fd, job->src_addr, bytes, &src_copy) != 0 || !src_copy) {
            free(src_copy);
            return 0xffff2101;
        }
        mirror_progress_status(fd, HETGPU_PACC_JOB_SOFTMAX, seq, 0x5222);
        dst_copy = calloc(1, bytes);
        if (!dst_copy) {
            free(src_copy);
            return 0xffff2102;
        }
        for (uint64_t row = 0; row < job->rows; row++) {
            uint64_t base = row * stride;
            float max_v = load_typed(src_copy, base, job->dtype);
            for (uint64_t col = 1; col < job->cols; col++) {
                float v = load_typed(src_copy, base + col, job->dtype);
                if (v > max_v) max_v = v;
            }
            float sum = 0.0f;
            for (uint64_t col = 0; col < job->cols; col++) {
                sum += expf_fast(load_typed(src_copy, base + col, job->dtype) - max_v);
            }
            float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (uint64_t col = 0; col < job->cols; col++) {
                float e = expf_fast(load_typed(src_copy, base + col, job->dtype) - max_v);
                store_typed(dst_copy, base + col, job->dtype, e * inv);
            }
        }
        mirror_progress_status(fd, HETGPU_PACC_JOB_SOFTMAX, seq, 0x5223);
        int ret = write_phys_copy(fd, job->dst_addr, dst_copy, bytes);
        free(src_copy);
        free(dst_copy);
        if (ret != 0) {
            return 0xffff2103;
        }
        mirror_progress_status(fd, HETGPU_PACC_JOB_SOFTMAX, seq, 0x5224);
        return 0;
    }
    struct Map ms = {0}, md = {0};
    mirror_progress_status(fd, HETGPU_PACC_JOB_SOFTMAX, seq, 0x5228);
    if (map_phys(fd, job->src_addr, bytes, &ms) ||
        map_phys(fd, job->dst_addr, bytes, &md)) {
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
    if (jobd_msync_enabled()) {
        msync(md.base, md.map_len, MS_SYNC);
    }
    jobd_flush_for_device(md.ptr, md.len);
    unmap_phys(&ms); unmap_phys(&md);
    return 0;
}

static bool rmsnorm_f32_rvv(const float *x,
                            const float *weight,
                            float *y,
                            uint64_t rows,
                            uint64_t hidden,
                            float eps);

static int run_rmsnorm(int fd, const struct RmsNormJob *job, uint64_t seq) {
    uint64_t t0 = monotonic_us();
    if (!job || !job->x_addr || !job->y_addr || !job->rows || !job->hidden) {
        log_msg("RMSNorm invalid job x=0x%" PRIx64 " w=0x%" PRIx64
                " y=0x%" PRIx64 " rows=%" PRIu64 " hidden=%" PRIu64
                " dtype=%u eps=%g",
                job ? job->x_addr : 0, job ? job->weight_addr : 0,
                job ? job->y_addr : 0, job ? job->rows : 0,
                job ? job->hidden : 0, job ? job->dtype : 0,
                job ? job->eps : 0.0f);
        return 0xffff3001;
    }
    struct RmsNormJob job_copy = *job;
    job = &job_copy;
    size_t elem_size = dtype_size(job->dtype);
    if (!elem_size) return 0xffff3002;
    size_t elems = (size_t)(job->rows * job->hidden);
    size_t data_bytes = elems * elem_size;
    size_t weight_bytes = (size_t)job->hidden * elem_size;
    if (jobd_rms_local_copy_enabled()) {
        uint8_t *x_local = NULL;
        uint8_t *w_local = NULL;
        uint8_t *y_local = NULL;
        bool have_weight = false;
        mirror_progress_status(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0x5120);
        if (read_phys_copy(fd, job->x_addr, data_bytes, &x_local) == 0) {
            mirror_progress_status(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0x5121);
            if (job->weight_addr) {
                if (read_phys_copy(fd, job->weight_addr, weight_bytes, &w_local) == 0) {
                    have_weight = true;
                    mirror_progress_status(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0x5122);
                } else {
                    free(x_local);
                    x_local = NULL;
                }
            } else {
                mirror_progress_status(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0x5122);
            }
        }
        if (x_local) {
            y_local = (uint8_t *)malloc(data_bytes);
            if (!y_local) {
                free(x_local);
                free(w_local);
                return 0xffff3006;
            }
            bool used_rvv = false;
            if (job->dtype == PACC_DTYPE_F32 && jobd_rms_rvv_enabled()) {
                used_rvv = rmsnorm_f32_rvv((const float *)x_local,
                                           have_weight ? (const float *)w_local : NULL,
                                           (float *)y_local,
                                           job->rows,
                                           job->hidden,
                                           job->eps);
            }
            if (!used_rvv) {
                for (uint64_t row = 0; row < job->rows; row++) {
                    uint64_t base = row * job->hidden;
                    float sumsq = 0.0f;
                    for (uint64_t i = 0; i < job->hidden; i++) {
                        float v = load_typed(x_local, base + i, job->dtype);
                        sumsq += v * v;
                    }
                    float scale = rsqrtf_newton(sumsq / (float)job->hidden + job->eps);
                    for (uint64_t i = 0; i < job->hidden; i++) {
                        float weight = have_weight ? load_typed(w_local, i, job->dtype) : 1.0f;
                        store_typed(y_local, base + i, job->dtype,
                                    load_typed(x_local, base + i, job->dtype) * scale * weight);
                    }
                }
            }
            mirror_rmsnorm_debug_sample(fd, job, seq, x_local,
                                        have_weight ? w_local : NULL,
                                        y_local,
                                        used_rvv ? 0x5135u : 0x5131u,
                                        (have_weight ? 1u : 0u) |
                                        4u |
                                        (used_rvv ? 8u : 0u));
            mirror_progress_status(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0x5123);
            bool output_pwrite = jobd_rms_output_pwrite_enabled();
            int write_status = 0;
            unsigned write_attempts =
                (unsigned)parse_env_u64_default("HETGPU_PACC_JOBD_RMS_WRITE_ATTEMPTS", 1);
            if (write_attempts == 0) write_attempts = 1;
            for (unsigned attempt = 0; attempt < write_attempts; attempt++) {
                write_status = output_pwrite
                    ? write_phys_copy_chunked_pwrite_only(
                          fd,
                          job->y_addr,
                          y_local,
                          data_bytes,
                          "HETGPU_PACC_JOBD_RMS_WRITE_CHUNK_BYTES")
                    : write_phys_copy_chunked(
                          fd,
                          job->y_addr,
                          y_local,
                          data_bytes,
                          "HETGPU_PACC_JOBD_RMS_WRITE_CHUNK_BYTES");
                if (write_status != 0) {
                    break;
                }
            }
            if (write_status != 0 && output_pwrite) {
                for (unsigned attempt = 0; attempt < 8; attempt++) {
                    write_status = write_phys_copy_chunked(
                        fd,
                        job->y_addr,
                        y_local,
                        data_bytes,
                        "HETGPU_PACC_JOBD_RMS_WRITE_CHUNK_BYTES");
                    if (write_status != 0) {
                        break;
                    }
                }
            }
            if (write_status == 0) {
                uint64_t sync_attempts = parse_env_u64_default(
                    "HETGPU_PACC_JOBD_RMS_OUTPUT_SYNC_ATTEMPTS", 0);
                if (sync_attempts > 16) sync_attempts = 16;
                for (uint64_t sync_attempt = 0; sync_attempt < sync_attempts; sync_attempt++) {
                    submit_mbox_payload_sync(g_mbox_fd,
                                             HETGPU_PACC_JOB_RMSNORM,
                                             seq,
                                             0,
                                             "rmsnorm-output");
                }
                jobd_evict_after_payload_write();
                repair_shared_ddr_writeback(fd, job->y_addr, y_local, data_bytes,
                                            "rmsnorm-output");
                mirror_rmsnorm_debug_sample(fd, job, seq, x_local,
                                            have_weight ? w_local : NULL,
                                            y_local,
                                            0x5139u,
                                            (have_weight ? 1u : 0u) |
                                            4u |
                                            (used_rvv ? 8u : 0u));
            }
            mirror_rmsnorm_phase_record(fd, job, seq, 0x513au,
                                        (have_weight ? 1u : 0u) |
                                        4u |
                                        (used_rvv ? 8u : 0u),
                                        (uint32_t)write_status);
            mirror_progress_status(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0x5124);
            mirror_rmsnorm_phase_record(fd, job, seq, 0x513bu,
                                        (have_weight ? 1u : 0u) |
                                        4u |
                                        (used_rvv ? 8u : 0u),
                                        (uint32_t)write_status);
            free(x_local);
            free(w_local);
            free(y_local);
            mirror_rmsnorm_phase_record(fd, job, seq, 0x513cu,
                                        (have_weight ? 1u : 0u) |
                                        4u |
                                        (used_rvv ? 8u : 0u),
                                        (uint32_t)write_status);
            trace_msg("RMSNorm local-copy done: rows=%" PRIu64 " hidden=%" PRIu64
                      " dtype=%u elapsed_us=%" PRIu64,
                      job->rows, job->hidden, job->dtype, monotonic_us() - t0);
            return write_status == 0 ? 0 : 0xffff3005;
        }
        free(w_local);
    }
    struct Map mx = {0}, mw = {0}, my = {0};
    mirror_progress_status(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0x5128);
    if (map_phys(fd, job->x_addr, data_bytes, &mx) ||
        map_phys(fd, job->y_addr, data_bytes, &my)) {
        unmap_phys(&mx); unmap_phys(&my);
        return 0xffff3003;
    }
    const void *x = mx.ptr;
    void *y = my.ptr;
    const void *w = NULL;
    if (job->weight_addr && !map_phys(fd, job->weight_addr, weight_bytes, &mw)) {
        w = mw.ptr;
    }
    sync_map_for_cpu(&mx);
    if (w) {
        sync_map_for_cpu(&mw);
    }
    bool used_rvv = false;
    if (job->dtype == PACC_DTYPE_F32 && jobd_rms_rvv_enabled()) {
        used_rvv = rmsnorm_f32_rvv((const float *)x,
                                   (const float *)w,
                                   (float *)y,
                                   job->rows,
                                   job->hidden,
                                   job->eps);
    }
    if (!used_rvv) {
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
    }
    mirror_rmsnorm_debug_sample(fd, job, seq, x, w, y,
                                used_rvv ? 0x5136u : 0x5132u,
                                (w ? 1u : 0u) |
                                2u |
                                (used_rvv ? 8u : 0u));
    if (jobd_msync_enabled()) {
        msync(my.base, my.map_len, MS_SYNC);
    }
    jobd_flush_for_device(my.ptr, my.len);
    if (jobd_rms_output_pwrite_enabled()) {
        (void)write_phys_copy_pwrite_only(fd, job->y_addr, y, data_bytes);
    }
    unmap_phys(&mx); unmap_phys(&mw); unmap_phys(&my);
    trace_msg("RMSNorm done: rows=%" PRIu64 " hidden=%" PRIu64
              " dtype=%u elapsed_us=%" PRIu64,
              job->rows, job->hidden, job->dtype, monotonic_us() - t0);
    return 0;
}

static bool rmsnorm_f32_rvv(const float *x,
                            const float *weight,
                            float *y,
                            uint64_t rows,
                            uint64_t hidden,
                            float eps) {
#if defined(__riscv_vector)
    if (!x || !y || rows == 0 || hidden == 0) {
        return false;
    }
    if (hidden > (uint64_t)SIZE_MAX) {
        return false;
    }

    size_t cols = (size_t)hidden;
    for (uint64_t row = 0; row < rows; row++) {
        const float *xr = x + (size_t)row * cols;
        float *yr = y + (size_t)row * cols;
        float sumsq = 0.0f;

        for (size_t i = 0; i < cols;) {
            size_t vl = __riscv_vsetvl_e32m4(cols - i);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(xr + i, vl);
            vfloat32m4_t vsq = __riscv_vfmul_vv_f32m4(vx, vx, vl);
            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m4_f32m1(vsq, zero, vl);
            sumsq += __riscv_vfmv_f_s_f32m1_f32(reduced);
            i += vl;
        }

        float scale = rsqrtf_newton(sumsq / (float)cols + eps);
        for (size_t i = 0; i < cols;) {
            size_t vl = __riscv_vsetvl_e32m4(cols - i);
            vfloat32m4_t vy = __riscv_vle32_v_f32m4(xr + i, vl);
            vy = __riscv_vfmul_vf_f32m4(vy, scale, vl);
            if (weight) {
                vfloat32m4_t vw = __riscv_vle32_v_f32m4(weight + i, vl);
                vy = __riscv_vfmul_vv_f32m4(vy, vw, vl);
            }
            __riscv_vse32_v_f32m4(yr + i, vy, vl);
            i += vl;
        }
    }
    return true;
#else
    (void)x;
    (void)weight;
    (void)y;
    (void)rows;
    (void)hidden;
    (void)eps;
    return false;
#endif
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
    if (jobd_msync_enabled()) {
        msync(md.base, md.map_len, MS_SYNC);
    }
    jobd_flush_for_device(md.ptr, md.len);
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

static int copy_file_all(const char *dst, const char *src) {
    int in_fd = open(src, O_RDONLY | O_CLOEXEC);
    int out_fd = -1;
    char buf[65536];
    if (in_fd < 0) return -1;
    out_fd = open(dst, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0700);
    if (out_fd < 0) {
        close(in_fd);
        return -1;
    }
    for (;;) {
        ssize_t nread = read(in_fd, buf, sizeof(buf));
        if (nread < 0) {
            close(in_fd);
            close(out_fd);
            return -1;
        }
        if (nread == 0) break;
        ssize_t written = 0;
        while (written < nread) {
            ssize_t nw = write(out_fd, buf + written, (size_t)(nread - written));
            if (nw < 0) {
                close(in_fd);
                close(out_fd);
                return -1;
            }
            written += nw;
        }
    }
    if (close(in_fd) != 0) {
        close(out_fd);
        return -1;
    }
    if (close(out_fd) != 0) return -1;
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
"#include <stdbool.h>\n"
"#include <sys/syscall.h>\n"
"#include <unistd.h>\n"
"#include <math.h>\n"
"#define WEAK __attribute__((weak, visibility(\"hidden\")))\n"
"#define WEAK_EXPORT __attribute__((weak, visibility(\"default\")))\n"
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
"WEAK bool f___zluda_ptx_impl_bar_red_and_pred(uint32_t barrier_id, bool predicate, bool invert_predicate) { (void)barrier_id; __sync_synchronize(); return predicate ^ invert_predicate; }\n"
"WEAK bool f___zluda_ptx_impl_bar_red_or_pred(uint32_t barrier_id, bool predicate, bool invert_predicate) { (void)barrier_id; __sync_synchronize(); return predicate ^ invert_predicate; }\n"
"WEAK uint32_t f___zluda_ptx_impl_activemask(void) { return 1u; }\n"
"struct HetgpuLaunchState {\n"
"    uint32_t tid[3];\n"
"    uint32_t ntid[3];\n"
"    uint32_t ctaid[3];\n"
"    uint32_t nctaid[3];\n"
"};\n"
"static struct HetgpuLaunchState hetgpu_launch_states[64];\n"
"static unsigned hetgpu_launch_slot(void) {\n"
"#if defined(__riscv)\n"
"    uintptr_t id = 0;\n"
"    __asm__ volatile(\"mv %0, tp\" : \"=r\"(id));\n"
"#else\n"
"    uintptr_t id = (uintptr_t)&id;\n"
"#endif\n"
"    return (unsigned)((id ^ (id >> 6)) & 63u);\n"
"}\n"
"static struct HetgpuLaunchState *hetgpu_launch_state(void) {\n"
"    return &hetgpu_launch_states[hetgpu_launch_slot()];\n"
"}\n"
"WEAK_EXPORT void f___zluda_ptx_impl_set_launch(uint32_t tid_x, uint32_t tid_y, uint32_t tid_z, uint32_t ntid_x, uint32_t ntid_y, uint32_t ntid_z, uint32_t ctaid_x, uint32_t ctaid_y, uint32_t ctaid_z, uint32_t nctaid_x, uint32_t nctaid_y, uint32_t nctaid_z) {\n"
"    struct HetgpuLaunchState *s = hetgpu_launch_state();\n"
"    s->tid[0] = tid_x; s->tid[1] = tid_y; s->tid[2] = tid_z;\n"
"    s->ntid[0] = ntid_x ? ntid_x : 1u; s->ntid[1] = ntid_y ? ntid_y : 1u; s->ntid[2] = ntid_z ? ntid_z : 1u;\n"
"    s->ctaid[0] = ctaid_x; s->ctaid[1] = ctaid_y; s->ctaid[2] = ctaid_z;\n"
"    s->nctaid[0] = nctaid_x ? nctaid_x : 1u; s->nctaid[1] = nctaid_y ? nctaid_y : 1u; s->nctaid[2] = nctaid_z ? nctaid_z : 1u;\n"
"}\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_tid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u ? s->tid[member] : 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_ntid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u && s->ntid[member] ? s->ntid[member] : 1u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_ctaid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u ? s->ctaid[member] : 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_nctaid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u && s->nctaid[member] ? s->nctaid[member] : 1u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_laneid(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return s->tid[0] & 31u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_eq(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return 1u << lane; }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_lt(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return lane == 0u ? 0u : ((1u << lane) - 1u); }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_le(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return lane == 31u ? ~0u : ((1u << (lane + 1u)) - 1u); }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_ge(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return ~((lane == 0u ? 0u : ((1u << lane) - 1u))); }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_gt(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return lane == 31u ? 0u : (~0u << (lane + 1u)); }\n"
"WEAK uint32_t f___zluda_ptx_impl_sreg_clock(void) { return 0u; }\n"
"WEAK float f___zluda_ptx_impl_sqrt_approx_f32(float x) { return sqrtf(x); }\n"
"WEAK float f___zluda_ptx_impl_rsqrt_approx_f32(float x) { return 1.0f / sqrtf(x); }\n"
"WEAK float f___zluda_ptx_impl_ex2_approx_f32(float x) { return exp2f(x); }\n"
"WEAK float f___zluda_ptx_impl_lg2_approx_f32(float x) { return log2f(x); }\n"
"WEAK float f___zluda_ptx_impl_rcp_approx_f32(float x) { return 1.0f / x; }\n"
"WEAK void f___zluda_ptx_impl_nanosleep_u32(uint32_t nanoseconds) { (void)nanoseconds; }\n"
"WEAK bool f___zluda_ptx_impl_vote_sync_any_pred(bool value, uint32_t membermask) { (void)membermask; return value; }\n"
"WEAK bool f___zluda_ptx_impl_vote_sync_any_pred_negate(bool value, uint32_t membermask) { (void)membermask; return !value; }\n"
"WEAK bool f___zluda_ptx_impl_vote_sync_all_pred(bool value, uint32_t membermask) { (void)membermask; return value; }\n"
"WEAK bool f___zluda_ptx_impl_vote_sync_all_pred_negate(bool value, uint32_t membermask) { (void)membermask; return !value; }\n"
"WEAK uint32_t f___zluda_ptx_impl_vote_sync_ballot_b32(bool value, uint32_t membermask) { return value ? (membermask ? membermask : 1u) : 0u; }\n"
"WEAK uint32_t f___zluda_ptx_impl_vote_sync_ballot_b32_negate(bool value, uint32_t membermask) { return !value ? (membermask ? membermask : 1u) : 0u; }\n"
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
                "__attribute__((weak, visibility(\"hidden\"), aligned(16))) unsigned char %s[%zu];\n",
                symbols[i], bytes);
    }
    if (fclose(out) != 0) return -1;

    if (symbol_count > 0) {
        trace_msg("device-link: adding %zu tmp_shared BSS stubs (%zu bytes each)",
                  symbol_count, bytes);
    }
    return compile_kernel_c_object(stub_src, stub_obj);
}

static const char *find_riscv_builtins_archive(void) {
    const char *env = getenv("HETGPU_PACC_DEVICE_BUILTINS");
    if (env && *env && access(env, R_OK) == 0) return env;

    const char *candidates[] = {
        "/usr/lib/llvm-23/lib/clang/23/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-22/lib/clang/22/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-21/lib/clang/21/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-20/lib/clang/20/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-19/lib/clang/19/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.builtins-riscv64.a",
        NULL,
    };

    for (size_t i = 0; candidates[i]; i++) {
        if (access(candidates[i], R_OK) == 0) return candidates[i];
    }
    return NULL;
}

static int run_device_link_tool(const char *tool,
                                const char *input_obj,
                                const char *stub_obj,
                                const char *tmp_shared_obj,
                                const char *output_so) {
    const char *builtins = find_riscv_builtins_archive();
    char *argv[24];
    size_t n = 0;

    argv[n++] = (char *)tool;
    argv[n++] = (char *)"-fuse-ld=bfd";
    argv[n++] = (char *)"-shared";
    argv[n++] = (char *)"-fPIC";
    argv[n++] = (char *)"-o";
    argv[n++] = (char *)output_so;
    argv[n++] = (char *)input_obj;
    argv[n++] = (char *)stub_obj;
    argv[n++] = (char *)tmp_shared_obj;
    if (builtins && *builtins) {
        argv[n++] = (char *)builtins;
    }
    argv[n++] = (char *)"-lm";
    argv[n++] = (char *)"-ldl";
    argv[n++] = NULL;

    int rc = run_command(argv);
    if (rc == 0 && builtins && *builtins) {
        trace_msg("device-link: linked compiler builtins %s", builtins);
    }
    return rc;
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
            if (run_device_link_tool(tool, input_obj, stub_obj, tmp_shared_obj, output_so) == 0) {
                trace_msg("device-link ok: %s -> %s via %s", input_obj, output_so, tool);
                return 0;
            }
        }
    }

    for (size_t i = 0; candidates[i]; i++) {
        const char *tool = find_program_on_path(candidates[i]);
        if (!tool) continue;

        if (run_device_link_tool(tool, input_obj, stub_obj, tmp_shared_obj, output_so) == 0) {
            trace_msg("device-link ok: %s -> %s via %s", input_obj, output_so, tool);
            return 0;
        }
    }

    return -1;
}

static bool elf64_bounds_ok(size_t off, size_t size, size_t total) {
    return off <= total && size <= total - off;
}

static bool symbol_matches_kernel_name(const char *symbol,
                                       const char *kernel_name,
                                       size_t kernel_name_size) {
    if (!symbol || !kernel_name || kernel_name_size == 0) return false;
    if (strlen(symbol) == kernel_name_size &&
        memcmp(symbol, kernel_name, kernel_name_size) == 0) {
        return true;
    }
    if (symbol[0] == 'f' && symbol[1] == '_' &&
        strlen(symbol + 2) == kernel_name_size &&
        memcmp(symbol + 2, kernel_name, kernel_name_size) == 0) {
        return true;
    }
    return false;
}

static bool symbol_name_from_kernel_abi(const char *kernel_name,
                                        size_t kernel_name_size,
                                        char *name_out,
                                        size_t name_out_len) {
    int written;
    if (!kernel_name || kernel_name_size == 0 || !name_out || name_out_len == 0) {
        return false;
    }
    if (kernel_name_size > (size_t)INT_MAX) {
        return false;
    }
    if (kernel_name_size >= 2 && kernel_name[0] == 'f' && kernel_name[1] == '_') {
        written = snprintf(name_out, name_out_len, "%.*s",
                           (int)kernel_name_size, kernel_name);
    } else {
        written = snprintf(name_out, name_out_len, "f_%.*s",
                           (int)kernel_name_size, kernel_name);
    }
    return written > 0 && (size_t)written < name_out_len;
}

static bool elf64_locate_symbol_by_hash(const uint8_t *elf, size_t elf_len,
                                        uint64_t want_hash,
                                        const char *kernel_name,
                                        size_t kernel_name_size,
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
            if (!*name || !is_valid_c_symbol_name(name)) continue;
            if (st_type != PACC_ELF_STT_FUNC && st_type != PACC_ELF_STT_NOTYPE) continue;
            if (symbol_matches_kernel_name(name, kernel_name, kernel_name_size) ||
                hash_kernel_name_bytes(name) == want_hash ||
                (name[0] == 'f' && name[1] == '_' &&
                 hash_kernel_name_bytes(name + 2) == want_hash)) {
                snprintf(name_out, name_out_len, "%s", name);
                return true;
            }
            if (st_type == PACC_ELF_STT_FUNC && !fallback) fallback = name;
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

static bool elf64_find_symbol_value(const uint8_t *elf, size_t elf_len,
                                    const char *want_name,
                                    uint64_t *value_out) {
    if (!elf || !want_name || !value_out || elf_len < 64) return false;
    size_t shoff = (size_t)read_u64_le(elf + 0x28);
    uint16_t shentsize = read_u16_le(elf + 0x3a);
    uint16_t shnum = read_u16_le(elf + 0x3c);
    if (!shoff || !shentsize || !shnum) return false;
    if (!elf64_bounds_ok(shoff, (size_t)shentsize * shnum, elf_len)) return false;

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
            uint16_t st_shndx = read_u16_le(sym + 0x06);
            if (st_name >= str_size || st_shndx == 0) continue;
            const char *name = strtab + st_name;
            if (!strcmp(name, want_name)) {
                *value_out = read_u64_le(sym + 0x08);
                return true;
            }
        }
    }
    return false;
}

static bool kernel_symbol_name_eq(const char *name, const char *want) {
    if (!name || !want) return false;
    size_t n = strcspn(name, "@");
    return strlen(want) == n && !memcmp(name, want, n);
}

static void *resolve_kernel_external_symbol(const char *name) {
    if (!name || !*name) return NULL;
    if (kernel_symbol_name_eq(name, "syscall")) return (void *)(uintptr_t)&syscall;
    if (kernel_symbol_name_eq(name, "exp2f")) return (void *)(uintptr_t)&exp2f;
    if (kernel_symbol_name_eq(name, "sqrtf")) return (void *)(uintptr_t)&sqrtf;
    if (kernel_symbol_name_eq(name, "log2f")) return (void *)(uintptr_t)&log2f;
    if (kernel_symbol_name_eq(name, "__cxa_finalize")) return NULL;
    if (!strncmp(name, "_ITM_", 5)) return NULL;
    return dlsym(RTLD_DEFAULT, name);
}

static uint8_t *mmap_kernel_exec_region(size_t map_len) {
    static const uintptr_t hints[] = {
        0x40000000ULL,
        0x50000000ULL,
        0x60000000ULL,
        0x70000000ULL,
    };
    const int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
    const int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    for (size_t i = 0; i < sizeof(hints) / sizeof(hints[0]); i++) {
#ifdef MAP_FIXED_NOREPLACE
        void *fixed = mmap((void *)hints[i], map_len, prot,
                           flags | MAP_FIXED_NOREPLACE, -1, 0);
        if (fixed != MAP_FAILED) {
            return (uint8_t *)fixed;
        }
#endif
        void *hinted = mmap((void *)hints[i], map_len, prot, flags, -1, 0);
        if (hinted != MAP_FAILED) {
            if ((uintptr_t)hinted < 0x80000000ULL) {
                return (uint8_t *)hinted;
            }
            munmap(hinted, map_len);
        }
    }

    void *fallback = mmap(NULL, map_len, prot, flags, -1, 0);
    return fallback == MAP_FAILED ? MAP_FAILED : (uint8_t *)fallback;
}

static int load_kernel_elf_direct(const uint8_t *elf, size_t elf_len,
                                  const char *symbol_name,
                                  struct LoadedKernelImage *out) {
    if (!elf || !out || elf64_type(elf, elf_len) != PACC_ELF_ET_DYN) {
        g_kernel_load_error = 0x101;
        log_msg("direct ELF rejected early: elf=%p out=%p type=%u symbol=%s",
                (const void *)elf, (void *)out, (unsigned)elf64_type(elf, elf_len),
                symbol_name ? symbol_name : "<null>");
        return -1;
    }

    uint64_t phoff = read_u64_le(elf + 0x20);
    uint16_t phentsize = read_u16_le(elf + 0x36);
    uint16_t phnum = read_u16_le(elf + 0x38);
    uint64_t min_vaddr = UINT64_MAX;
    uint64_t max_vaddr = 0;
    uint64_t page = (uint64_t)g_page_size;
    if (!phoff || phentsize < 56 || !phnum ||
        !elf64_bounds_ok((size_t)phoff, (size_t)phentsize * phnum, elf_len)) {
        g_kernel_load_error = 0x102;
        log_msg("direct ELF rejected: bad phdr phoff=0x%" PRIx64
                " entsize=%u num=%u len=0x%zx",
                phoff, phentsize, phnum, elf_len);
        return -1;
    }

    for (uint16_t i = 0; i < phnum; i++) {
        const uint8_t *ph = elf + phoff + (uint64_t)i * phentsize;
        uint32_t type = read_u32_le(ph + 0x00);
        if (type != PACC_ELF_PT_LOAD) continue;
        uint64_t vaddr = read_u64_le(ph + 0x10);
        uint64_t memsz = read_u64_le(ph + 0x28);
        if (!memsz) continue;
        uint64_t seg_min = vaddr & ~(page - 1);
        uint64_t seg_max = (vaddr + memsz + page - 1) & ~(page - 1);
        if (seg_min < min_vaddr) min_vaddr = seg_min;
        if (seg_max > max_vaddr) max_vaddr = seg_max;
    }
    if (min_vaddr == UINT64_MAX) {
        g_kernel_load_error = 0x131;
        log_msg("direct ELF rejected: no PT_LOAD phdr phoff=0x%" PRIx64
                " entsize=%u num=%u len=0x%zx",
                phoff, phentsize, phnum, elf_len);
        return -1;
    }
    if (max_vaddr <= min_vaddr) {
        g_kernel_load_error = 0x132;
        log_msg("direct ELF rejected: empty load span min=0x%" PRIx64
                " max=0x%" PRIx64 " phnum=%u len=0x%zx",
                min_vaddr, max_vaddr, phnum, elf_len);
        return -1;
    }
    if (max_vaddr - min_vaddr > (256ULL << 20)) {
        g_kernel_load_error = 0x133;
        log_msg("direct ELF rejected: load span min=0x%" PRIx64
                " max=0x%" PRIx64 " len=0x%zx",
                min_vaddr, max_vaddr, elf_len);
        return -1;
    }

    size_t map_len = (size_t)(max_vaddr - min_vaddr);
    uint8_t *mapping = mmap_kernel_exec_region(map_len);
    if (mapping == MAP_FAILED) {
        g_kernel_load_error = 0x104;
        log_msg("direct ELF rejected: mmap len=0x%zx failed: %s",
                map_len, strerror(errno));
        return -1;
    }
    memset(mapping, 0, map_len);

    for (uint16_t i = 0; i < phnum; i++) {
        const uint8_t *ph = elf + phoff + (uint64_t)i * phentsize;
        uint32_t type = read_u32_le(ph + 0x00);
        if (type != PACC_ELF_PT_LOAD) continue;
        uint64_t off = read_u64_le(ph + 0x08);
        uint64_t vaddr = read_u64_le(ph + 0x10);
        uint64_t filesz = read_u64_le(ph + 0x20);
        uint64_t memsz = read_u64_le(ph + 0x28);
        if (filesz > memsz || !elf64_bounds_ok((size_t)off, (size_t)filesz, elf_len) ||
            vaddr < min_vaddr || vaddr - min_vaddr > map_len ||
            memsz > map_len - (size_t)(vaddr - min_vaddr)) {
            g_kernel_load_error = 0x105;
            log_msg("direct ELF rejected: bad PT_LOAD off=0x%" PRIx64
                    " vaddr=0x%" PRIx64 " filesz=0x%" PRIx64
                    " memsz=0x%" PRIx64 " min=0x%" PRIx64
                    " map_len=0x%zx elf_len=0x%zx",
                    off, vaddr, filesz, memsz, min_vaddr, map_len, elf_len);
            munmap(mapping, map_len);
            return -1;
        }
        memcpy(mapping + (vaddr - min_vaddr), elf + off, (size_t)filesz);
    }

    uintptr_t load_bias = (uintptr_t)mapping - (uintptr_t)min_vaddr;
    size_t shoff = (size_t)read_u64_le(elf + 0x28);
    uint16_t shentsize = read_u16_le(elf + 0x3a);
    uint16_t shnum = read_u16_le(elf + 0x3c);
    if (!shoff || !shentsize || !shnum ||
        !elf64_bounds_ok(shoff, (size_t)shentsize * shnum, elf_len)) {
        g_kernel_load_error = 0x106;
        log_msg("direct ELF rejected: bad shdr shoff=0x%zx entsize=%u num=%u len=0x%zx",
                shoff, shentsize, shnum, elf_len);
        munmap(mapping, map_len);
        return -1;
    }

    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t *rela_sh = elf + shoff + (size_t)i * shentsize;
        if (read_u32_le(rela_sh + 0x04) != PACC_ELF_SHT_RELA) continue;
        size_t rela_off = (size_t)read_u64_le(rela_sh + 0x18);
        size_t rela_size = (size_t)read_u64_le(rela_sh + 0x20);
        size_t rela_entsize = (size_t)read_u64_le(rela_sh + 0x38);
        uint32_t symtab_index = read_u32_le(rela_sh + 0x28);
        const uint8_t *symtab = NULL;
        const char *strtab = NULL;
        size_t sym_count = 0;
        size_t sym_entsize = 0;
        size_t str_size = 0;
        if (rela_entsize == 0) {
            rela_entsize = 24;
        }
        if (rela_entsize < 24 || !elf64_bounds_ok(rela_off, rela_size, elf_len)) {
            g_kernel_load_error = 0x107;
            log_msg("direct ELF rejected: bad rela section off=0x%zx size=0x%zx entsize=0x%zx len=0x%zx",
                    rela_off, rela_size, rela_entsize, elf_len);
            munmap(mapping, map_len);
            return -1;
        }
        if (symtab_index < shnum) {
            const uint8_t *sym_sh = elf + shoff + (size_t)symtab_index * shentsize;
            uint32_t strtab_index = read_u32_le(sym_sh + 0x28);
            size_t sym_off = (size_t)read_u64_le(sym_sh + 0x18);
            size_t sym_size = (size_t)read_u64_le(sym_sh + 0x20);
            sym_entsize = (size_t)read_u64_le(sym_sh + 0x38);
            if (sym_entsize >= 24 && elf64_bounds_ok(sym_off, sym_size, elf_len)) {
                symtab = elf + sym_off;
                sym_count = sym_size / sym_entsize;
            }
            if (strtab_index < shnum) {
                const uint8_t *str_sh = elf + shoff + (size_t)strtab_index * shentsize;
                size_t str_off = (size_t)read_u64_le(str_sh + 0x18);
                str_size = (size_t)read_u64_le(str_sh + 0x20);
                if (read_u32_le(str_sh + 0x04) == PACC_ELF_SHT_STRTAB &&
                    elf64_bounds_ok(str_off, str_size, elf_len)) {
                    strtab = (const char *)(elf + str_off);
                }
            }
        }
        size_t rela_count = rela_size / rela_entsize;
        for (size_t r = 0; r < rela_count; r++) {
            const uint8_t *rela = elf + rela_off + r * rela_entsize;
            uint64_t r_offset = read_u64_le(rela + 0x00);
            uint64_t r_info = read_u64_le(rela + 0x08);
            int64_t r_addend = (int64_t)read_u64_le(rela + 0x10);
            uint32_t r_type = (uint32_t)(r_info & 0xffffffffu);
            uint32_t r_sym = (uint32_t)(r_info >> 32);
            if (r_type == PACC_R_RISCV_NONE) {
                continue;
            }
            if (r_offset < min_vaddr || r_offset - min_vaddr > map_len - sizeof(uint64_t)) {
                g_kernel_load_error = 0x108;
                log_msg("direct ELF rejected: reloc offset out of range off=0x%" PRIx64
                        " min=0x%" PRIx64 " map_len=0x%zx type=%u",
                        r_offset, min_vaddr, map_len, r_type);
                munmap(mapping, map_len);
                return -1;
            }
            uint64_t *where = (uint64_t *)(void *)(mapping + (r_offset - min_vaddr));
            uint64_t value = 0;
            if (r_type == PACC_R_RISCV_RELATIVE) {
                value = (uint64_t)(load_bias + (uintptr_t)r_addend);
            } else if (r_type == PACC_R_RISCV_64 || r_type == PACC_R_RISCV_JUMP_SLOT) {
                const char *name = NULL;
                uint16_t st_shndx = 0;
                uint64_t st_value = 0;
                if (symtab && r_sym < sym_count) {
                    const uint8_t *sym = symtab + (size_t)r_sym * sym_entsize;
                    uint32_t st_name = read_u32_le(sym + 0x00);
                    st_shndx = read_u16_le(sym + 0x06);
                    st_value = read_u64_le(sym + 0x08);
                    if (strtab && st_name < str_size) name = strtab + st_name;
                }
                if (st_shndx != 0) {
                    value = (uint64_t)(load_bias + (uintptr_t)st_value + (uintptr_t)r_addend);
                } else {
                    void *ext = resolve_kernel_external_symbol(name);
                    if (!ext && name && !kernel_symbol_name_eq(name, "__cxa_finalize") &&
                        strncmp(name, "_ITM_", 5)) {
                        g_kernel_load_error = 0x109;
                        log_msg("direct ELF unresolved external symbol: %s", name);
                        munmap(mapping, map_len);
                        return -1;
                    }
                    value = (uint64_t)(uintptr_t)ext + (uint64_t)r_addend;
                }
            } else {
                g_kernel_load_error = 0x10a;
                log_msg("direct ELF unsupported relocation type=%u", r_type);
                munmap(mapping, map_len);
                return -1;
            }
            *where = value;
        }
    }

    __builtin___clear_cache((char *)mapping, (char *)mapping + map_len);

    uint64_t sym_value = 0;
    if (!elf64_find_symbol_value(elf, elf_len, symbol_name, &sym_value)) {
        g_kernel_load_error = 0x10b;
        log_msg("direct ELF rejected: symbol not found: %s", symbol_name ? symbol_name : "<null>");
        munmap(mapping, map_len);
        return -1;
    }
    __builtin___clear_cache((char *)mapping, (char *)mapping + map_len);
    out->direct = true;
    out->handle = NULL;
    out->mapping = mapping;
    out->map_len = map_len;
    out->fn = (void *)(uintptr_t)(load_bias + (uintptr_t)sym_value);
    if (elf64_find_symbol_value(elf, elf_len, "f___zluda_ptx_impl_set_launch", &sym_value)) {
        out->set_launch = (PaccSetLaunchFn)(uintptr_t)(load_bias + (uintptr_t)sym_value);
    }
    log_msg("direct ELF mapped symbol=%s base=%p len=0x%zx min=0x%" PRIx64
            " bias=0x%" PRIxPTR " fn=%p set_launch=%p",
            symbol_name ? symbol_name : "<null>", (void *)mapping, map_len,
            min_vaddr, (uintptr_t)load_bias, out->fn, (void *)out->set_launch);
    return 0;
}

static void unload_kernel_image(struct LoadedKernelImage *loaded) {
    if (!loaded) return;
    if (loaded->direct && loaded->mapping && loaded->map_len) {
        /*
         * Keep direct ELF mappings resident.  On the PACC-side Linux jobd this
         * path was observed to reach 0x5109 and then never publish the final
         * completion status, which leaves the AP polling until timeout.  These
         * per-kernel ET_DYN images are small and process-scoped; correctness is
         * more important than reclaiming them while we bring up direct ELF.
         */
        trace_msg("direct ELF mapping retained: base=%p len=0x%zx",
                  loaded->mapping, loaded->map_len);
    } else if (loaded->handle) {
        dlclose(loaded->handle);
    }
    memset(loaded, 0, sizeof(*loaded));
}

static bool kernel_cache_enabled(void) {
    const char *value = getenv("HETGPU_PACC_KERNEL_CACHE");
    return !(value && (!strcmp(value, "0") || !strcasecmp(value, "false") ||
                      !strcasecmp(value, "off") || !strcasecmp(value, "no")));
}

static const char *kernel_cache_dir(void) {
    const char *value = getenv("HETGPU_PACC_KERNEL_CACHE_DIR");
    return (value && *value) ? value : "/tmp/hetgpu_pacc_kernel_cache";
}

static int ensure_kernel_cache_dir(const char *dir) {
    if (!dir || !*dir) return -1;
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) return -1;
    return 0;
}

static bool make_kernel_cache_path(char *out, size_t out_len,
                                   const uint8_t *elf, size_t elf_len,
                                   uint64_t kernel_hash, uint16_t e_type) {
    const char *dir = kernel_cache_dir();
    uint64_t elf_hash = hash_bytes_fnv64(elf, elf_len);
    if (!out || out_len == 0 || !kernel_cache_enabled()) return false;
    if (ensure_kernel_cache_dir(dir) != 0) return false;
    return snprintf(out, out_len,
                    "%s/kernel-sreg4-t%u-%016" PRIx64 "-%016" PRIx64 "-%zu.so",
                    dir, (unsigned)e_type, kernel_hash, elf_hash, elf_len) < (int)out_len;
}

static bool dlopen_cached_kernel(const char *path, char *artifact_path,
                                 size_t artifact_path_len, void **handle_out) {
    if (!path || !*path || !handle_out || access(path, R_OK) != 0) return false;
    log_msg("kernel cache hit candidate: dlopen %s", path);
    g_current_kernel_symbol = "dlopen-cache";
    *handle_out = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    g_current_kernel_symbol = NULL;
    if (*handle_out) {
        snprintf(artifact_path, artifact_path_len, "%s", path);
        trace_msg("kernel cache hit: %s", path);
        return true;
    }
    log_msg("kernel cache stale: dlopen(%s) failed: %s", path, dlerror());
    unlink(path);
    return false;
}

static void install_kernel_cache_artifact(const char *cache_path,
                                          const char *built_so,
                                          char *artifact_path,
                                          size_t artifact_path_len,
                                          const char **load_path) {
    char tmp_path[PATH_MAX];
    if (!cache_path || !*cache_path || !built_so || !load_path) return;
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld", cache_path, (long)getpid());
    if (copy_file_all(tmp_path, built_so) == 0 && rename(tmp_path, cache_path) == 0) {
        trace_msg("kernel cache store: %s", cache_path);
        *load_path = cache_path;
        snprintf(artifact_path, artifact_path_len, "%s", cache_path);
        return;
    }
    unlink(tmp_path);
}

static int load_kernel_image(const uint8_t *elf, size_t elf_len,
                             uint64_t kernel_hash,
                             const char *kernel_name, size_t kernel_name_size,
                             char *symbol_name, size_t symbol_name_len,
                             char *artifact_path, size_t artifact_path_len,
                             struct LoadedKernelImage *loaded_out) {
    char tmpdir[] = "/tmp/hetgpu_pacc_kernelXXXXXX";
    char obj_path[PATH_MAX];
    char so_path[PATH_MAX];
    char cache_path[PATH_MAX] = {0};
    uint16_t e_type;
    const char *load_path = NULL;

    if (!elf || !elf_len || !loaded_out) return -1;
    memset(loaded_out, 0, sizeof(*loaded_out));
    g_kernel_load_error = 0;

    e_type = elf64_type(elf, elf_len);
    if (!elf64_locate_symbol_by_hash(elf, elf_len, kernel_hash,
                                     kernel_name, kernel_name_size,
                                     symbol_name, symbol_name_len)) {
        if (!symbol_name_from_kernel_abi(kernel_name, kernel_name_size,
                                         symbol_name, symbol_name_len)) {
            g_kernel_load_error = 0x201;
            log_msg("kernel image: no symbol matched hash=0x%" PRIx64
                    " and ABI kernel name is unavailable/too long",
                    kernel_hash);
            return -1;
        }
        log_msg("kernel image: no symbol table match hash=0x%" PRIx64
                " kernel=%.*s; trying ABI-derived symbol %s",
                kernel_hash, (int)kernel_name_size, kernel_name, symbol_name);
    }

    if (e_type == PACC_ELF_ET_DYN &&
        load_kernel_elf_direct(elf, elf_len, symbol_name, loaded_out) == 0) {
        snprintf(artifact_path, artifact_path_len, "<direct-elf>");
        return 0;
    }
    if (e_type == PACC_ELF_ET_DYN) {
        if (!g_kernel_load_error) g_kernel_load_error = 0x202;
        log_msg("direct ELF load failed for symbol %s; refusing dlopen fallback", symbol_name);
        return -1;
    }

    if (make_kernel_cache_path(cache_path, sizeof(cache_path), elf, elf_len, kernel_hash, e_type) &&
        dlopen_cached_kernel(cache_path, artifact_path, artifact_path_len, &loaded_out->handle)) {
        return 0;
    }

    if (!mkdtemp(tmpdir)) {
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
        install_kernel_cache_artifact(cache_path, so_path, artifact_path,
                                      artifact_path_len, &load_path);
    } else if (e_type == PACC_ELF_ET_DYN) {
        snprintf(so_path, sizeof(so_path), "%s/kernel.so", tmpdir);
        if (write_file_all(so_path, elf, elf_len) != 0) {
            return -1;
        }
        load_path = so_path;
        snprintf(artifact_path, artifact_path_len, "%s", so_path);
        install_kernel_cache_artifact(cache_path, so_path, artifact_path,
                                      artifact_path_len, &load_path);
    } else {
        g_kernel_load_error = 0x203;
        log_msg("unsupported kernel ELF type=%u", (unsigned)e_type);
        return -1;
    }

    log_msg("kernel artifact built: dlopen %s", load_path);
    g_current_kernel_symbol = "dlopen-built";
    loaded_out->handle = dlopen(load_path, RTLD_LAZY | RTLD_LOCAL);
    g_current_kernel_symbol = NULL;
    if (!loaded_out->handle) {
        log_msg("dlopen(%s) failed: %s", load_path, dlerror());
        return -1;
    }
    return 0;
}

static struct PaccUint3 kernel_arg_uint3(const uint64_t *args,
                                         const struct PaccJobImage *job,
                                         size_t index) {
    struct PaccUint3 v;
    v.x = (uint32_t)args[index];
    v.y = (uint32_t)(args[index] >> 32);
    v.z = (job && job->arg_records && index < job->arg_count) ? (uint32_t)job->arg_records[index].value_hi : 0;
    return v;
}

static int PACC_UNUSED invoke_kernel_bin_bcast23(void *fn,
                                                 const uint64_t *args,
                                                 const struct PaccJobImage *job) {
    struct PaccUint3 u3_6 = kernel_arg_uint3(args, job, 6);
    struct PaccUint3 u3_7 = kernel_arg_uint3(args, job, 7);
    struct PaccUint3 u3_8 = kernel_arg_uint3(args, job, 8);
    struct PaccUint3 u3_9 = kernel_arg_uint3(args, job, 9);
    struct PaccUint3 u3_10 = kernel_arg_uint3(args, job, 10);
    typedef void (*BinBcast23Fn)(
        const void *, const void *, void *,
        int32_t, int32_t, int32_t,
        struct PaccUint3, struct PaccUint3, struct PaccUint3,
        struct PaccUint3, struct PaccUint3,
        int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
        int32_t, int32_t, int32_t, int32_t, int32_t,
        const void *);
    trace_msg("bin_bcast23 call: ptrs=%p,%p,%p,%p dims=%d,%d,%d "
              "u3={%u,%u,%u}/{%u,%u,%u}/{%u,%u,%u}/{%u,%u,%u}/{%u,%u,%u} "
              "tail=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
              (const void *)(uintptr_t)args[0],
              (const void *)(uintptr_t)args[1],
              (void *)(uintptr_t)args[2],
              (const void *)(uintptr_t)args[22],
              (int32_t)args[3], (int32_t)args[4], (int32_t)args[5],
              u3_6.x, u3_6.y, u3_6.z,
              u3_7.x, u3_7.y, u3_7.z,
              u3_8.x, u3_8.y, u3_8.z,
              u3_9.x, u3_9.y, u3_9.z,
              u3_10.x, u3_10.y, u3_10.z,
              (int32_t)args[11], (int32_t)args[12], (int32_t)args[13],
              (int32_t)args[14], (int32_t)args[15], (int32_t)args[16],
              (int32_t)args[17], (int32_t)args[18], (int32_t)args[19],
              (int32_t)args[20], (int32_t)args[21]);
    ((BinBcast23Fn)fn)(
        (const void *)(uintptr_t)args[0],
        (const void *)(uintptr_t)args[1],
        (void *)(uintptr_t)args[2],
        (int32_t)args[3],
        (int32_t)args[4],
        (int32_t)args[5],
        u3_6,
        u3_7,
        u3_8,
        u3_9,
        u3_10,
        (int32_t)args[11],
        (int32_t)args[12],
        (int32_t)args[13],
        (int32_t)args[14],
        (int32_t)args[15],
        (int32_t)args[16],
        (int32_t)args[17],
        (int32_t)args[18],
        (int32_t)args[19],
        (int32_t)args[20],
        (int32_t)args[21],
        (const void *)(uintptr_t)args[22]);
    return 0;
}

static int invoke_kernel_symbol(const char *symbol, void *fn,
                                const uint64_t *args,
                                const struct PaccJobImage *job,
                                size_t argc) {
    (void)symbol;
    (void)job;
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
    case 13: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]); return 0;
    case 14: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]); return 0;
    case 15: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]); return 0;
    case 16: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15]); return 0;
    case 17: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16]); return 0;
    case 18: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17]); return 0;
    case 19: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18]); return 0;
    case 20: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19]); return 0;
    case 21: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20]); return 0;
    case 22: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21]); return 0;
    case 23: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22]); return 0;
    case 24: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23]); return 0;
    case 25: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24]); return 0;
    case 26: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25]); return 0;
    case 27: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26]); return 0;
    case 28: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27]); return 0;
    case 29: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28]); return 0;
    case 30: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29]); return 0;
    case 31: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30]); return 0;
    case 32: ((void (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31]); return 0;
    default:
        return -1;
    }
}

static uint32_t pacc_nonzero_dim(uint32_t value) {
    return value ? value : 1u;
}

struct KernelGridWorker {
    const char *symbol;
    void *fn;
    const uint64_t *args;
    const struct PaccJobImage *job;
    size_t argc;
    PaccSetLaunchFn set_launch;
    uint32_t gx;
    uint32_t gy;
    uint32_t gz;
    uint32_t bx;
    uint32_t by;
    uint32_t bz;
    uint64_t begin;
    uint64_t end;
    int status;
};

struct BinBcastElfRowWorker {
    const char *symbol;
    void *fn;
    const uint64_t *args;
    const struct PaccJobImage *job;
    size_t argc;
    PaccSetLaunchFn set_launch;
    uint32_t ne1;
    uint32_t z_rows;
    uint64_t begin_row;
    uint64_t end_row;
    int status;
};

static unsigned kernel_worker_threads(uint64_t total_threads) {
    uint64_t requested = parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_THREADS", 0);
    unsigned default_threads = PACC_KERNEL_DEFAULT_THREADS;

    if (requested == 0) {
        requested = default_threads;
    }
    if (requested < 1) {
        requested = 1;
    }
    if (requested > PACC_KERNEL_MAX_THREADS) {
        requested = PACC_KERNEL_MAX_THREADS;
    }
    if (requested > total_threads && total_threads > 0) {
        requested = total_threads;
    }
    return (unsigned)requested;
}

static uint64_t scale_f32_single_thread_max_elements(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_SCALE_F32_SINGLE_THREAD_MAX_ELEMS",
                                 8192ULL);
}

enum PaccBinBcastOp {
    PACC_BIN_BCAST_UNSUPPORTED = 0,
    PACC_BIN_BCAST_REPEAT,
    PACC_BIN_BCAST_ADD,
    PACC_BIN_BCAST_SUB,
    PACC_BIN_BCAST_MUL,
    PACC_BIN_BCAST_DIV,
};

struct BinBcastNativeCtx {
    const float *src0;
    const float *src1;
    float *dst;
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    struct PaccUint3 ne3;
    struct PaccUint3 ne10;
    struct PaccUint3 ne11;
    struct PaccUint3 ne12;
    struct PaccUint3 ne13;
    int32_t s1;
    int32_t s2;
    int32_t s3;
    int32_t s00;
    int32_t s01;
    int32_t s02;
    int32_t s03;
    int32_t s10;
    int32_t s11;
    int32_t s12;
    int32_t s13;
    const float *src1s[PACC_MAX_KERNEL_ARGS];
    size_t src1s_count;
    enum PaccBinBcastOp op;
};

struct BinBcastNativeWorker {
    const struct BinBcastNativeCtx *ctx;
    uint64_t begin_row;
    uint64_t end_row;
    int status;
};

static uint64_t kernel_binding_size_for_arg(const struct PaccJobImage *job,
                                            uint32_t arg_index);

static enum PaccBinBcastOp bin_bcast_op_from_symbol(const char *symbol) {
    if (!symbol) return PACC_BIN_BCAST_UNSUPPORTED;
    if (strstr(symbol, "op_repeatff")) return PACC_BIN_BCAST_REPEAT;
    if (strstr(symbol, "op_addff")) return PACC_BIN_BCAST_ADD;
    if (strstr(symbol, "op_subff")) return PACC_BIN_BCAST_SUB;
    if (strstr(symbol, "op_mulff")) return PACC_BIN_BCAST_MUL;
    if (strstr(symbol, "op_divff")) return PACC_BIN_BCAST_DIV;
    return PACC_BIN_BCAST_UNSUPPORTED;
}

static const struct KernelParamCell *kernel_argv_cell(const uint64_t *argv,
                                                      size_t argc,
                                                      size_t index) {
    if (!argv || index >= argc || !argv[index]) return NULL;
    return (const struct KernelParamCell *)(uintptr_t)argv[index];
}

static uint64_t kernel_cell_u64(const uint64_t *argv, size_t argc, size_t index) {
    const struct KernelParamCell *cell = kernel_argv_cell(argv, argc, index);
    return cell ? cell->lo : 0;
}

static float kernel_cell_f32(const uint64_t *argv, size_t argc, size_t index) {
    uint32_t bits = (uint32_t)kernel_cell_u64(argv, argc, index);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int32_t kernel_cell_i32(const uint64_t *argv, size_t argc, size_t index) {
    return (int32_t)kernel_cell_u64(argv, argc, index);
}

static int64_t kernel_cell_i64(const uint64_t *argv, size_t argc, size_t index) {
    return (int64_t)kernel_cell_u64(argv, argc, index);
}

static void kernel_cell_i32x4(const uint64_t *argv, size_t argc, size_t index,
                              int32_t out[4]) {
    const struct KernelParamCell *cell = kernel_argv_cell(argv, argc, index);
    if (!out) return;
    out[0] = out[1] = out[2] = out[3] = 0;
    if (!cell) return;
    out[0] = (int32_t)(uint32_t)cell->lo;
    out[1] = (int32_t)(uint32_t)(cell->lo >> 32);
    out[2] = (int32_t)(uint32_t)cell->hi;
    out[3] = (int32_t)(uint32_t)(cell->hi >> 32);
}

static struct PaccUint3 kernel_cell_u3(const uint64_t *argv, size_t argc, size_t index) {
    const struct KernelParamCell *cell = kernel_argv_cell(argv, argc, index);
    struct PaccUint3 v = {0, 0, 0};
    if (!cell) return v;
    v.x = (uint32_t)cell->lo;
    v.y = (uint32_t)(cell->lo >> 32);
    v.z = (uint32_t)cell->hi;
    return v;
}

static int invoke_kernel_pytorch_fill_native(const char *symbol,
                                             const uint64_t *args,
                                             const struct PaccJobImage *job,
                                             size_t argc) {
    if (!symbol || strcmp(symbol, "pacc_pytorch_fill_pattern") != 0) {
        return 1;
    }
    if (!job || argc < 4) {
        return -1;
    }

    uint8_t *dst = (uint8_t *)(uintptr_t)kernel_cell_u64(args, argc, 0);
    uint64_t n = kernel_cell_u64(args, argc, 1);
    uint64_t pattern = kernel_cell_u64(args, argc, 2);
    uint64_t elem_size = kernel_cell_u64(args, argc, 3);
    uint64_t stride = argc > 4 ? kernel_cell_u64(args, argc, 4) : 1;
    if (stride == 0) stride = 1;
    if (!dst || n == 0) {
        return 0;
    }
    if (elem_size != 1 && elem_size != 2 && elem_size != 4 && elem_size != 8) {
        log_msg("native pytorch fill invalid elem_size=%" PRIu64 " argc=%zu",
                elem_size, argc);
        return -1;
    }
    if (n > UINT64_MAX / elem_size ||
        n - 1 > (UINT64_MAX - 1) / stride) {
        return -1;
    }
    uint64_t span_elems = (n - 1) * stride + 1;
    if (span_elems > UINT64_MAX / elem_size) {
        return -1;
    }
    uint64_t bytes64 = span_elems * elem_size;
    uint64_t bound = kernel_binding_size_for_arg(job, 0);
    if (bound && bytes64 > bound) {
        log_msg("native pytorch fill exceeds binding: bytes=%" PRIu64
                " bound=%" PRIu64,
                bytes64, bound);
        return -1;
    }

    if (elem_size == 1) {
        uint8_t value = (uint8_t)pattern;
        for (uint64_t i = 0; i < n; i++) dst[i * stride] = value;
    } else if (elem_size == 2) {
        uint16_t value = (uint16_t)pattern;
        uint16_t *out = (uint16_t *)dst;
        for (uint64_t i = 0; i < n; i++) out[i * stride] = value;
    } else if (elem_size == 4) {
        uint32_t value = (uint32_t)pattern;
        uint32_t *out = (uint32_t *)dst;
        for (uint64_t i = 0; i < n; i++) out[i * stride] = value;
    } else {
        uint64_t *out = (uint64_t *)dst;
        for (uint64_t i = 0; i < n; i++) out[i * stride] = pattern;
    }
    jobd_io_fence();
    return 0;
}

static uint32_t bin_bcast_fast_dim(struct PaccUint3 v) {
    return v.z ? v.z : 1u;
}

static float bin_bcast_apply(enum PaccBinBcastOp op, float a, float b) {
    switch (op) {
    case PACC_BIN_BCAST_REPEAT: return b;
    case PACC_BIN_BCAST_ADD: return a + b;
    case PACC_BIN_BCAST_SUB: return a - b;
    case PACC_BIN_BCAST_MUL: return a * b;
    case PACC_BIN_BCAST_DIV: return a / b;
    default: return a;
    }
}

static bool bin_bcast_add_extent(uint64_t dim, int32_t stride, uint64_t *acc) {
    if (!acc || stride < 0) return false;
    if (dim == 0) return false;
    uint64_t ustride = (uint64_t)(uint32_t)stride;
    uint64_t count = dim - 1u;
    if (ustride != 0 && count > UINT64_MAX / ustride) return false;
    uint64_t add = count * ustride;
    if (*acc > UINT64_MAX - add) return false;
    *acc += add;
    return true;
}

static bool bin_bcast_required_elems(uint64_t d0, uint64_t d1,
                                     uint64_t d2, uint64_t d3,
                                     int32_t s0, int32_t s1,
                                     int32_t s2, int32_t s3,
                                     uint64_t *out) {
    uint64_t max_elem = 0;
    if (!out) return false;
    if (!bin_bcast_add_extent(d0, s0, &max_elem) ||
        !bin_bcast_add_extent(d1, s1, &max_elem) ||
        !bin_bcast_add_extent(d2, s2, &max_elem) ||
        !bin_bcast_add_extent(d3, s3, &max_elem)) {
        return false;
    }
    if (max_elem == UINT64_MAX) return false;
    *out = max_elem + 1u;
    return true;
}

static bool bin_bcast_binding_covers(const char *symbol,
                                     const char *label,
                                     uint32_t arg_index,
                                     uint64_t binding_bytes,
                                     uint64_t required_elems) {
    if (required_elems > UINT64_MAX / sizeof(float)) return false;
    uint64_t required_bytes = required_elems * sizeof(float);
    if (binding_bytes != 0 && required_bytes > binding_bytes) {
        log_msg("native bin_bcast range rejected %s arg=%u need=%" PRIu64
                "B binding=%" PRIu64 "B kernel=%s",
                label ? label : "buffer", arg_index, required_bytes,
                binding_bytes, symbol ? symbol : "<unknown>");
        return false;
    }
    return true;
}

#if defined(__riscv_vector)
static inline vfloat32m4_t bin_bcast_apply_vec_vv(enum PaccBinBcastOp op,
                                                  vfloat32m4_t a,
                                                  vfloat32m4_t b,
                                                  size_t vl) {
    switch (op) {
    case PACC_BIN_BCAST_REPEAT: return b;
    case PACC_BIN_BCAST_ADD: return __riscv_vfadd_vv_f32m4(a, b, vl);
    case PACC_BIN_BCAST_SUB: return __riscv_vfsub_vv_f32m4(a, b, vl);
    case PACC_BIN_BCAST_MUL: return __riscv_vfmul_vv_f32m4(a, b, vl);
    case PACC_BIN_BCAST_DIV: return __riscv_vfdiv_vv_f32m4(a, b, vl);
    default: return a;
    }
}

static inline vfloat32m4_t bin_bcast_apply_vec_vf(enum PaccBinBcastOp op,
                                                  vfloat32m4_t a,
                                                  float b,
                                                  size_t vl) {
    switch (op) {
    case PACC_BIN_BCAST_REPEAT: return __riscv_vfmv_v_f_f32m4(b, vl);
    case PACC_BIN_BCAST_ADD: return __riscv_vfadd_vf_f32m4(a, b, vl);
    case PACC_BIN_BCAST_SUB: return __riscv_vfsub_vf_f32m4(a, b, vl);
    case PACC_BIN_BCAST_MUL: return __riscv_vfmul_vf_f32m4(a, b, vl);
    case PACC_BIN_BCAST_DIV: return __riscv_vfdiv_vf_f32m4(a, b, vl);
    default: return a;
    }
}

static bool bin_bcast_native_row_rvv(const struct BinBcastNativeCtx *ctx,
                                     const float *src0_row,
                                     int64_t i_src1,
                                     float *dst_row) {
    if (!ctx || !dst_row || ctx->ne0 <= 0 || ctx->s00 != 1) return false;
    if (ctx->op == PACC_BIN_BCAST_UNSUPPORTED) return false;

    const uint32_t ne10 = bin_bcast_fast_dim(ctx->ne10);
    bool src1_scalar = ne10 == 1u;
    bool src1_contiguous = ctx->s10 == 1 && ne10 >= (uint32_t)ctx->ne0;
    if (!src1_scalar && !src1_contiguous) return false;
    if (ctx->op != PACC_BIN_BCAST_REPEAT && !src0_row) return false;

    for (int32_t i0 = 0; i0 < ctx->ne0;) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(ctx->ne0 - i0));
        vfloat32m4_t acc = src0_row
            ? __riscv_vle32_v_f32m4(src0_row + i0, vl)
            : __riscv_vfmv_v_f_f32m4(0.0f, vl);

        if (ctx->src1s_count != 0) {
            for (size_t j = 0; j < ctx->src1s_count; j++) {
                const float *src = ctx->src1s[j];
                if (src1_scalar) {
                    acc = bin_bcast_apply_vec_vf(ctx->op, acc, src[i_src1], vl);
                } else {
                    vfloat32m4_t b = __riscv_vle32_v_f32m4(src + i_src1 + i0, vl);
                    acc = bin_bcast_apply_vec_vv(ctx->op, acc, b, vl);
                }
            }
        } else {
            if (!ctx->src1) return false;
            if (src1_scalar) {
                acc = bin_bcast_apply_vec_vf(ctx->op, acc, ctx->src1[i_src1], vl);
            } else {
                vfloat32m4_t b = __riscv_vle32_v_f32m4(ctx->src1 + i_src1 + i0, vl);
                acc = bin_bcast_apply_vec_vv(ctx->op, acc, b, vl);
            }
        }

        __riscv_vse32_v_f32m4(dst_row + i0, acc, vl);
        i0 += (int32_t)vl;
    }
    return true;
}
#endif

static void *bin_bcast_native_worker_main(void *opaque) {
    struct BinBcastNativeWorker *worker = (struct BinBcastNativeWorker *)opaque;
    const struct BinBcastNativeCtx *ctx = worker->ctx;
    const uint32_t ne10 = bin_bcast_fast_dim(ctx->ne10);
    const uint32_t ne11 = bin_bcast_fast_dim(ctx->ne11);
    const uint32_t ne12 = bin_bcast_fast_dim(ctx->ne12);
    const uint32_t ne13 = bin_bcast_fast_dim(ctx->ne13);
    const uint32_t ne1 = (uint32_t)ctx->ne1;
    const uint32_t ne2 = (uint32_t)ctx->ne2;

    for (uint64_t row = worker->begin_row; row < worker->end_row; row++) {
        uint64_t t = row;
        uint32_t i1 = (uint32_t)(t % ne1);
        t /= ne1;
        uint32_t i2 = (uint32_t)(t % ne2);
        uint32_t i3 = (uint32_t)(t / ne2);
        uint32_t i11 = i1 % ne11;
        uint32_t i12 = i2 % ne12;
        uint32_t i13 = i3 % ne13;
        int64_t i_src0 = (int64_t)i3 * ctx->s03 + (int64_t)i2 * ctx->s02 + (int64_t)i1 * ctx->s01;
        int64_t i_src1 = (int64_t)i13 * ctx->s13 + (int64_t)i12 * ctx->s12 + (int64_t)i11 * ctx->s11;
        int64_t i_dst = (int64_t)i3 * ctx->s3 + (int64_t)i2 * ctx->s2 + (int64_t)i1 * ctx->s1;
        const float *src0_row = ctx->src0 ? ctx->src0 + i_src0 : NULL;
        float *dst_row = ctx->dst + i_dst;

#if defined(__riscv_vector)
        if (bin_bcast_native_row_rvv(ctx, src0_row, i_src1, dst_row)) {
            continue;
        }
#endif
        for (int32_t i0 = 0; i0 < ctx->ne0; i0++) {
            uint32_t i10 = (uint32_t)i0 % ne10;
            int64_t src1_off = i_src1 + (int64_t)i10 * ctx->s10;
            float result = src0_row ? src0_row[(int64_t)i0 * ctx->s00] : 0.0f;
            if (ctx->src1s_count != 0) {
                for (size_t j = 0; j < ctx->src1s_count; j++) {
                    result = bin_bcast_apply(ctx->op, result, ctx->src1s[j][src1_off]);
                }
            } else {
                result = bin_bcast_apply(ctx->op, result, ctx->src1[src1_off]);
            }
            dst_row[i0] = result;
        }
    }

    worker->status = 0;
    return NULL;
}

static int invoke_kernel_bin_bcast_native(const char *symbol,
                                          const uint64_t *argv,
                                          const struct PaccJobImage *job,
                                          size_t argc) {
    uint64_t t0 = monotonic_us();
    (void)job;
    if (!symbol || !strstr(symbol, "_ZL11k_bin_bcast")) return 1;
    if (!env_flag_default_true("HETGPU_PACC_ENABLE_NATIVE_BIN_BCAST")) return 1;
    if (strstr(symbol, "k_bin_bcast_unravel")) return 1;
    if (!strstr(symbol, "EEfffJ")) return 1;
    if (argc < 22 || argc > PACC_MAX_KERNEL_ARGS) return (int)0xffffb001u;

    struct BinBcastNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.op = bin_bcast_op_from_symbol(symbol);
    if (ctx.op == PACC_BIN_BCAST_UNSUPPORTED) return 1;

    ctx.src0 = (const float *)(uintptr_t)kernel_cell_u64(argv, argc, 0);
    ctx.src1 = (const float *)(uintptr_t)kernel_cell_u64(argv, argc, 1);
    ctx.dst = (float *)(uintptr_t)kernel_cell_u64(argv, argc, 2);
    ctx.ne0 = kernel_cell_i32(argv, argc, 3);
    ctx.ne1 = kernel_cell_i32(argv, argc, 4);
    ctx.ne2 = kernel_cell_i32(argv, argc, 5);
    ctx.ne3 = kernel_cell_u3(argv, argc, 6);
    ctx.ne10 = kernel_cell_u3(argv, argc, 7);
    ctx.ne11 = kernel_cell_u3(argv, argc, 8);
    ctx.ne12 = kernel_cell_u3(argv, argc, 9);
    ctx.ne13 = kernel_cell_u3(argv, argc, 10);
    ctx.s1 = kernel_cell_i32(argv, argc, 11);
    ctx.s2 = kernel_cell_i32(argv, argc, 12);
    ctx.s3 = kernel_cell_i32(argv, argc, 13);
    ctx.s00 = kernel_cell_i32(argv, argc, 14);
    ctx.s01 = kernel_cell_i32(argv, argc, 15);
    ctx.s02 = kernel_cell_i32(argv, argc, 16);
    ctx.s03 = kernel_cell_i32(argv, argc, 17);
    ctx.s10 = kernel_cell_i32(argv, argc, 18);
    ctx.s11 = kernel_cell_i32(argv, argc, 19);
    ctx.s12 = kernel_cell_i32(argv, argc, 20);
    ctx.s13 = kernel_cell_i32(argv, argc, 21);
    ctx.src1s_count = argc > 22 ? argc - 22 : 0;

    if (!ctx.dst || ctx.ne0 <= 0 || ctx.ne1 <= 0 || ctx.ne2 <= 0 || bin_bcast_fast_dim(ctx.ne3) == 0) {
        return 0;
    }
    if (ctx.src1s_count == 0 && !ctx.src1) return (int)0xffffb002u;
    for (size_t i = 0; i < ctx.src1s_count; i++) {
        ctx.src1s[i] = (const float *)(uintptr_t)kernel_cell_u64(argv, argc, 22 + i);
        if (!ctx.src1s[i]) return (int)(0xffffb010u | (uint32_t)(i & 0x0fu));
    }

    uint64_t ne3 = bin_bcast_fast_dim(ctx.ne3);
    uint64_t src0_required = 0;
    uint64_t src1_required = 0;
    uint64_t dst_required = 0;
    uint64_t src1_ne0 = (uint64_t)ctx.ne0;
    uint64_t src1_ne1 = (uint64_t)ctx.ne1;
    uint64_t src1_ne2 = (uint64_t)ctx.ne2;
    uint64_t src1_ne3 = ne3;
    uint32_t ne10 = bin_bcast_fast_dim(ctx.ne10);
    uint32_t ne11 = bin_bcast_fast_dim(ctx.ne11);
    uint32_t ne12 = bin_bcast_fast_dim(ctx.ne12);
    uint32_t ne13 = bin_bcast_fast_dim(ctx.ne13);
    if (ne10 != 0 && src1_ne0 > ne10) src1_ne0 = ne10;
    if (ne11 != 0 && src1_ne1 > ne11) src1_ne1 = ne11;
    if (ne12 != 0 && src1_ne2 > ne12) src1_ne2 = ne12;
    if (ne13 != 0 && src1_ne3 > ne13) src1_ne3 = ne13;
    if ((ctx.src0 &&
         !bin_bcast_required_elems((uint64_t)ctx.ne0, (uint64_t)ctx.ne1,
                                   (uint64_t)ctx.ne2, ne3,
                                   ctx.s00, ctx.s01, ctx.s02, ctx.s03,
                                   &src0_required)) ||
        !bin_bcast_required_elems(src1_ne0, src1_ne1, src1_ne2, src1_ne3,
                                  ctx.s10, ctx.s11, ctx.s12, ctx.s13,
                                  &src1_required) ||
        !bin_bcast_required_elems((uint64_t)ctx.ne0, (uint64_t)ctx.ne1,
                                  (uint64_t)ctx.ne2, ne3,
                                  1, ctx.s1, ctx.s2, ctx.s3,
                                  &dst_required)) {
        log_msg("native bin_bcast invalid extent args: ne=(%d,%d,%d,%" PRIu64
                ") s=(%d,%d,%d,%d) src=(%d,%d,%d,%d) kernel=%s",
                ctx.ne0, ctx.ne1, ctx.ne2, ne3,
                ctx.s1, ctx.s2, ctx.s3, ctx.s00,
                ctx.s01, ctx.s02, ctx.s03, ctx.s10,
                symbol ? symbol : "<unknown>");
        return (int)0xffffb003u;
    }
    bool src0_aliases_dst =
        ctx.src0 == (const float *)ctx.dst &&
        ctx.s00 == 1 &&
        ctx.s01 == ctx.s1 &&
        ctx.s02 == ctx.s2 &&
        ctx.s03 == ctx.s3;
    if (ctx.src0 && !src0_aliases_dst &&
        !bin_bcast_binding_covers(symbol, "src0", 0,
                                  kernel_binding_size_for_arg(job, 0),
                                  src0_required)) {
        return (int)0xffffb004u;
    }
    if (ctx.src1s_count == 0 &&
        !bin_bcast_binding_covers(symbol, "src1", 1,
                                  kernel_binding_size_for_arg(job, 1),
                                  src1_required)) {
        return (int)0xffffb005u;
    }
    for (size_t i = 0; i < ctx.src1s_count; i++) {
        uint32_t arg_index = (uint32_t)(22 + i);
        if (!bin_bcast_binding_covers(symbol, "src1s", arg_index,
                                      kernel_binding_size_for_arg(job, arg_index),
                                      src1_required)) {
            return (int)(0xffffb020u | (uint32_t)(i & 0x0fu));
        }
    }
    if (!bin_bcast_binding_covers(symbol, "dst", 2,
                                  kernel_binding_size_for_arg(job, 2),
                                  dst_required)) {
        return (int)0xffffb006u;
    }

    uint64_t src0_bytes = kernel_binding_size_for_arg(job, 0);
    uint64_t src1_bytes = kernel_binding_size_for_arg(job, 1);
    uint64_t dst_bytes = kernel_binding_size_for_arg(job, 2);
    const float *src0_writeback = ctx.src0;
    const float *src1_writeback = ctx.src1;
    float *dst_writeback = ctx.dst;
    void *local_src0 = NULL;
    void *local_src1 = NULL;
    void *local_dst = NULL;
    void *local_src1s[PACC_MAX_KERNEL_ARGS];
    memset(local_src1s, 0, sizeof(local_src1s));

    uint64_t local_max = parse_env_u64_default("PACC_JOBD_BIN_BCAST_LOCAL_MAX_BYTES",
                                               1u << 20);
    if (local_max != 0) {
        if (dst_bytes != 0 && dst_bytes <= local_max) {
            if (native_stage_read((uint64_t)(uintptr_t)ctx.dst,
                                  (size_t)dst_bytes, &local_dst)) {
                ctx.dst = (float *)local_dst;
            }
        }
        if (src0_aliases_dst && local_dst) {
            ctx.src0 = (const float *)local_dst;
        } else if (ctx.src0 && src0_bytes != 0 && src0_bytes <= local_max) {
            if (native_stage_read((uint64_t)(uintptr_t)ctx.src0,
                                  (size_t)src0_bytes, &local_src0)) {
                ctx.src0 = (const float *)local_src0;
            }
        }
        if (ctx.src1s_count == 0) {
            if (ctx.src1 && src1_bytes != 0 && src1_bytes <= local_max) {
                if (native_stage_read((uint64_t)(uintptr_t)ctx.src1,
                                      (size_t)src1_bytes, &local_src1)) {
                    ctx.src1 = (const float *)local_src1;
                }
            }
        } else {
            for (size_t i = 0; i < ctx.src1s_count; i++) {
                uint32_t arg_index = (uint32_t)(22 + i);
                uint64_t bytes = kernel_binding_size_for_arg(job, arg_index);
                if (bytes != 0 && bytes <= local_max) {
                    if (native_stage_read((uint64_t)(uintptr_t)ctx.src1s[i],
                                          (size_t)bytes, &local_src1s[i])) {
                        ctx.src1s[i] = (const float *)local_src1s[i];
                    }
                }
            }
        }
    }
    (void)src0_writeback;
    (void)src1_writeback;

    uint64_t rows = (uint64_t)(uint32_t)ctx.ne1 *
                    (uint64_t)(uint32_t)ctx.ne2 *
                    (uint64_t)bin_bcast_fast_dim(ctx.ne3);
    unsigned workers = kernel_worker_threads(rows);
    trace_msg("native bin_bcast %s rows=%" PRIu64 " ne0=%d workers=%u fused=%zu",
              symbol, rows, ctx.ne0, workers, ctx.src1s_count);

    int status = 0;
    if (workers <= 1 || rows <= 1) {
        struct BinBcastNativeWorker worker = {
            .ctx = &ctx,
            .begin_row = 0,
            .end_row = rows,
            .status = 0,
        };
        bin_bcast_native_worker_main(&worker);
        status = worker.status;
    } else {
        pthread_t threads[PACC_KERNEL_MAX_THREADS];
        struct BinBcastNativeWorker worker[PACC_KERNEL_MAX_THREADS];
        unsigned created = 0;
        uint64_t chunk = (rows + workers - 1u) / workers;
        memset(worker, 0, sizeof(worker));
        for (unsigned i = 0; i < workers; i++) {
            uint64_t begin = (uint64_t)i * chunk;
            uint64_t end = begin + chunk;
            if (begin >= rows) break;
            if (end > rows) end = rows;
            worker[i].ctx = &ctx;
            worker[i].begin_row = begin;
            worker[i].end_row = end;
            worker[i].status = -1;
            if (pthread_create(&threads[i], NULL, bin_bcast_native_worker_main, &worker[i]) != 0) {
                log_msg("native bin_bcast failed to create worker %u", i);
                for (unsigned j = 0; j < created; j++) {
                    pthread_join(threads[j], NULL);
                }
                status = (int)0xffffb007u;
                goto bin_bcast_native_done;
            }
            created++;
        }

        for (unsigned i = 0; i < created; i++) {
            pthread_join(threads[i], NULL);
            if (worker[i].status != 0 && status == 0) {
                status = worker[i].status;
            }
        }
    }
bin_bcast_native_done:
    if (status == 0 && local_dst) {
        (void)native_stage_write((uint64_t)(uintptr_t)dst_writeback,
                                 local_dst, (size_t)dst_bytes);
    }
    trace_msg("native bin_bcast done %s status=%d elapsed_us=%" PRIu64,
              symbol, status, monotonic_us() - t0);
    free(local_src0);
    free(local_src1);
    free(local_dst);
    for (size_t i = 0; i < ctx.src1s_count; i++) {
        free(local_src1s[i]);
    }
    return status;
}

enum PaccMmvfXType {
    PACC_MMVF_UNSUPPORTED = 0,
    PACC_MMVF_F32,
    PACC_MMVF_F16,
    PACC_MMVF_BF16,
    PACC_MMVF_Q8_0,
};

struct MmvfNativeCtx {
    const void *x;
    const void *y;
    const int32_t *ids;
    float *dst;
    int32_t ncols2;
    struct PaccUint3 nchannels_y;
    int32_t stride_row;
    int32_t stride_col_y2;
    int32_t stride_col_dst;
    struct PaccUint3 channel_ratio;
    int32_t stride_channel_x;
    int32_t stride_channel_y;
    int32_t stride_channel_dst;
    struct PaccUint3 sample_ratio;
    int32_t stride_sample_x;
    int32_t stride_sample_y;
    int32_t stride_sample_dst;
    int32_t ids_stride;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t ncols_dst;
    enum PaccMmvfXType x_type;
    enum PaccMmvfXType y_type;
};

struct MmvfNativeWorker {
    const struct MmvfNativeCtx *ctx;
    uint64_t begin;
    uint64_t end;
    int status;
};

static pthread_once_t PACC_UNUSED g_f16_table_once = PTHREAD_ONCE_INIT;
static float g_f16_to_f32[65536];
static volatile int g_f16_table_ready;

static float pacc_f16_bits_to_f32(uint16_t h) {
#if defined(__riscv_zfh) || defined(__riscv_zfhmin)
    _Float16 value;
    memcpy(&value, &h, sizeof(value));
    return (float)value;
#else
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
#endif
}

static void PACC_UNUSED pacc_f16_table_init_once(void) {
    for (uint32_t i = 0; i <= UINT16_MAX; i++) {
        g_f16_to_f32[i] = pacc_f16_bits_to_f32((uint16_t)i);
    }
    __sync_synchronize();
    g_f16_table_ready = 1;
}

static void pacc_prepare_f16_table(void) {
#if defined(__riscv_zfh) || defined(__riscv_zfhmin)
    g_f16_table_ready = 1;
#else
    pthread_once(&g_f16_table_once, pacc_f16_table_init_once);
#endif
}

static float pacc_f16_to_f32(uint16_t h) {
#if defined(__riscv_zfh) || defined(__riscv_zfhmin)
    return pacc_f16_bits_to_f32(h);
#else
    if (__builtin_expect(g_f16_table_ready, 1)) {
        return g_f16_to_f32[h];
    }
    return pacc_f16_bits_to_f32(h);
#endif
}

static float pacc_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static enum PaccMmvfXType mmvf_type_from_symbol(const char *symbol) {
    if (!symbol || !strstr(symbol, "mul_mat_vec_f")) return PACC_MMVF_UNSUPPORTED;
    if (strstr(symbol, "mul_mat_vec_fI6__half")) return PACC_MMVF_F16;
    if (strstr(symbol, "mul_mat_vec_fIff")) return PACC_MMVF_F32;
    return PACC_MMVF_UNSUPPORTED;
}

static enum PaccMmvfXType mmvf_type_from_job_field(uint32_t value) {
    switch (value) {
    case 2:
        return PACC_MMVF_F16;
    case 3:
        return PACC_MMVF_BF16;
    case 0:
    case 1:
    default:
        return PACC_MMVF_F32;
    }
}

static size_t mmvf_type_size(enum PaccMmvfXType type) {
    if (type == PACC_MMVF_F16 || type == PACC_MMVF_BF16) {
        return sizeof(uint16_t);
    }
    if (type == PACC_MMVF_Q8_0) {
        return 1;
    }
    return sizeof(float);
}

static const struct PaccKernelBufferBinding *kernel_binding_for_arg(const struct PaccJobImage *job,
                                                                    uint32_t arg_index) {
    if (!job || !job->bindings) return 0;
    for (size_t i = 0; i < job->binding_count; i++) {
        if (job->bindings[i].arg_index == arg_index) {
            return &job->bindings[i];
        }
    }
    return 0;
}

static uint64_t kernel_binding_size_for_arg(const struct PaccJobImage *job, uint32_t arg_index) {
    const struct PaccKernelBufferBinding *binding = kernel_binding_for_arg(job, arg_index);
    return binding ? binding->size : 0;
}

static bool kernel_binding_phys_size_for_arg(const struct PaccJobImage *job,
                                             uint32_t arg_index,
                                             uint64_t *phys,
                                             size_t *size) {
    const struct PaccKernelBufferBinding *binding = kernel_binding_for_arg(job, arg_index);
    if (!binding || !binding->addr || !binding->size || binding->size > (uint64_t)SIZE_MAX) {
        return false;
    }
    if (phys) {
        *phys = binding->addr;
    }
    if (size) {
        *size = (size_t)binding->size;
    }
    return true;
}

static size_t kernel_binding_map_bytes(const struct PaccKernelBufferBinding *binding,
                                       size_t default_bind_bytes) {
    uint64_t want = binding && binding->size ? binding->size : (uint64_t)default_bind_bytes;
    uint64_t min_bytes = parse_env_u64_default("PACC_JOBD_KERNEL_BINDING_MIN_BYTES",
                                               parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_BINDING_MIN_BYTES", 0));

    if (min_bytes > want) {
        want = min_bytes;
    }
    if (binding && g_ddr_info.ddr_base && binding->addr >= g_ddr_info.ddr_base) {
        uint64_t ddr_off = binding->addr - g_ddr_info.ddr_base;
        if (ddr_off < g_ddr_info.ddr_size) {
            uint64_t remaining = g_ddr_info.ddr_size - ddr_off;
            if (want > remaining) {
                want = remaining;
            }
        }
    }
    if (want > (uint64_t)SIZE_MAX) {
        want = (uint64_t)SIZE_MAX;
    }
    return (size_t)want;
}

static bool mmvf_parse_template(const char *symbol,
                                uint32_t *ncols_dst_out,
                                bool *has_fusion_out,
                                bool *is_multi_token_id_out) {
    const char *p = symbol ? strstr(symbol, "mul_mat_vec_f") : NULL;
    if (!p) return false;
    p = strstr(p, "Li");
    if (!p) return false;
    p += 2;
    char *end = NULL;
    unsigned long ncols_dst = strtoul(p, &end, 10);
    if (!end || end == p || ncols_dst == 0 || ncols_dst > 64) return false;

    const char *b0 = strstr(end, "ELb");
    if (!b0 || (b0[3] != '0' && b0[3] != '1')) return false;
    const char *b1 = strstr(b0 + 4, "ELb");
    if (!b1 || (b1[3] != '0' && b1[3] != '1')) return false;

    *ncols_dst_out = (uint32_t)ncols_dst;
    *has_fusion_out = b0[3] == '1';
    *is_multi_token_id_out = b1[3] == '1';
    return true;
}

static float mmvf_load_value(enum PaccMmvfXType type, const void *base, int64_t index) {
    if (type == PACC_MMVF_F16) {
        return pacc_f16_to_f32(((const uint16_t *)base)[index]);
    } else if (type == PACC_MMVF_BF16) {
        return pacc_bf16_to_f32(((const uint16_t *)base)[index]);
    }
    return ((const float *)base)[index];
}

static float mmvf_load_x(const struct MmvfNativeCtx *ctx, const void *base, int64_t index) {
    if (ctx && ctx->x_type == PACC_MMVF_Q8_0) {
        const uint8_t *row = (const uint8_t *)base;
        int64_t block = index >> 5;
        int lane = (int)(index & 31);
        const uint8_t *qblock = row + block * 34;
        uint16_t scale_bits;
        memcpy(&scale_bits, qblock, sizeof(scale_bits));
        return pacc_f16_to_f32(scale_bits) * (float)((const int8_t *)(qblock + 2))[lane];
    }
    return mmvf_load_value(ctx->x_type, base, index);
}

#if defined(__riscv_vector) && (defined(__riscv_zfh) || defined(__riscv_zfhmin))
static float pacc_dot_f16_f32_rvv(const uint16_t *x, const float *y, int32_t n) {
    if (n <= 0) return 0.0f;

    const uint16_t *xp = x;
    const float *yp = y;
    size_t remaining = (size_t)n;
    size_t vl = 0;
    float sum;

    /*
     * Keep a wide vector accumulator for the whole dot product and reduce once.
     * The previous loop reduced every VL chunk, which dominated the 2048-wide
     * MMVF inner product.
     */
    asm volatile(
        "vsetvli %[vl], %[remaining], e16, m4, ta, ma\n\t"
        "vsetvli zero, %[vl], e32, m8, ta, ma\n\t"
        "fmv.w.x ft0, zero\n\t"
        "vfmv.v.f v24, ft0\n\t"
        "1:\n\t"
        "bltu %[remaining], %[vl], 2f\n\t"
        "vsetvli zero, %[vl], e16, m4, ta, ma\n\t"
        "vle16.v v0, (%[xp])\n\t"
        "vfwcvt.f.f.v v8, v0\n\t"
        "slli t0, %[vl], 1\n\t"
        "add %[xp], %[xp], t0\n\t"
        "vsetvli zero, %[vl], e32, m8, ta, ma\n\t"
        "vle32.v v16, (%[yp])\n\t"
        "slli t0, %[vl], 2\n\t"
        "add %[yp], %[yp], t0\n\t"
        "vfmacc.vv v24, v8, v16\n\t"
        "sub %[remaining], %[remaining], %[vl]\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "vsetivli zero, 1, e32, m1, ta, ma\n\t"
        "vfmv.s.f v0, ft0\n\t"
        "vsetvli zero, %[vl], e32, m8, ta, ma\n\t"
        "vfredusum.vs v24, v24, v0\n\t"
        "vfmv.f.s %[sum], v24\n\t"
        : [xp] "+r"(xp),
          [yp] "+r"(yp),
          [remaining] "+r"(remaining),
          [vl] "=&r"(vl),
          [sum] "=f"(sum)
        :
        : "t0", "ft0", "memory", "v0", "v8", "v16", "v24");

    for (size_t i = 0; i < remaining; i++) {
        sum += pacc_f16_to_f32(xp[i]) * yp[i];
    }
    return sum;
}
#endif

static bool mmvf_xsfmm_bf16_rows4(const struct MmvfNativeCtx *ctx,
                                  const void *x_base,
                                  const void *y_base,
                                  float *dst_base,
                                  uint32_t row_base) {
#if defined(HETGPU_PACC_HAVE_XSFVFWMACCQQQ)
    if (!ctx || ctx->x_type != PACC_MMVF_BF16 ||
        ctx->y_type != PACC_MMVF_BF16 ||
        ctx->ncols_dst != 1 || ctx->stride_row <= 0 ||
        ctx->ncols2 <= 0 || !x_base || !y_base || !dst_base) {
        return false;
    }

    const uint16_t *xh = (const uint16_t *)x_base;
    const uint16_t *yh = (const uint16_t *)y_base;
    const int32_t total = ctx->ncols2 * 2;
    uint16_t atile[16];
    uint16_t btile[16];
    float ctile[16];
    memset(ctile, 0, sizeof(ctile));

    int32_t kk = 0;
    for (; kk + 3 < total; kk += 4) {
        for (uint32_t r = 0; r < 4; r++) {
            const uint16_t *xr = xh + (int64_t)r * ctx->stride_row + kk;
            for (uint32_t k = 0; k < 4; k++) {
                atile[r * 4 + k] = xr[k];
            }
        }
        for (uint32_t k = 0; k < 4; k++) {
            uint16_t v = yh[kk + k];
            for (uint32_t c = 0; c < 4; c++) {
                btile[k * 4 + c] = v;
            }
        }

        size_t vl = __riscv_vsetvl_e32m2(16);
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(ctile, vl);
        vbfloat16m1_t va = __riscv_vle16_v_bf16m1((const __bf16 *)atile, vl);
        vbfloat16m1_t vb = __riscv_vle16_v_bf16m1((const __bf16 *)btile, vl);
        acc = __riscv_sf_vfwmacc_4x4x4_f32m2(acc, va, vb, vl);
        __riscv_vse32_v_f32m2(ctile, acc, vl);
    }

    float out0 = ctile[0];
    float out1 = ctile[4];
    float out2 = ctile[8];
    float out3 = ctile[12];
    for (; kk < total; kk++) {
        float y = pacc_bf16_to_f32(yh[kk]);
        out0 += pacc_bf16_to_f32(xh[(int64_t)0 * ctx->stride_row + kk]) * y;
        out1 += pacc_bf16_to_f32(xh[(int64_t)1 * ctx->stride_row + kk]) * y;
        out2 += pacc_bf16_to_f32(xh[(int64_t)2 * ctx->stride_row + kk]) * y;
        out3 += pacc_bf16_to_f32(xh[(int64_t)3 * ctx->stride_row + kk]) * y;
    }
    dst_base[row_base + 0] = out0;
    dst_base[row_base + 1] = out1;
    dst_base[row_base + 2] = out2;
    dst_base[row_base + 3] = out3;
    return true;
#else
    (void)ctx;
    (void)x_base;
    (void)y_base;
    (void)dst_base;
    (void)row_base;
    return false;
#endif
}

static bool mmvf_xsfmm_q8_0_bf16_rows4(const struct MmvfNativeCtx *ctx,
                                       const void *x_base,
                                       const void *y_base,
                                       float *dst_base,
                                       uint32_t row_base) {
#if defined(HETGPU_PACC_HAVE_XSFVFWMACCQQQ)
    if (!ctx || ctx->x_type != PACC_MMVF_Q8_0 ||
        ctx->y_type != PACC_MMVF_BF16 ||
        ctx->ncols_dst != 1 || ctx->stride_row <= 0 ||
        ctx->ncols2 <= 0 || !x_base || !y_base || !dst_base) {
        return false;
    }

    const uint8_t *xq = (const uint8_t *)x_base;
    const uint16_t *yh = (const uint16_t *)y_base;
    const int32_t total = ctx->ncols2 * 2;
    uint16_t atile[16];
    uint16_t btile[16];
    float ctile[16];
    memset(ctile, 0, sizeof(ctile));

    int32_t kk = 0;
    for (; kk + 3 < total; kk += 4) {
        for (uint32_t r = 0; r < 4; r++) {
            const uint8_t *row = xq + (int64_t)r * ctx->stride_row;
            for (uint32_t k = 0; k < 4; k++) {
                int32_t idx = kk + (int32_t)k;
                const uint8_t *qblock = row + (idx >> 5) * 34;
                uint16_t scale_bits;
                memcpy(&scale_bits, qblock, sizeof(scale_bits));
                float x = pacc_f16_to_f32(scale_bits) *
                          (float)((const int8_t *)(qblock + 2))[idx & 31];
                atile[r * 4 + k] = f32_to_bf16(x);
            }
        }
        for (uint32_t k = 0; k < 4; k++) {
            uint16_t v = yh[kk + (int32_t)k];
            for (uint32_t c = 0; c < 4; c++) {
                btile[k * 4 + c] = v;
            }
        }

        size_t vl = __riscv_vsetvl_e32m2(16);
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(ctile, vl);
        vbfloat16m1_t va = __riscv_vle16_v_bf16m1((const __bf16 *)atile, vl);
        vbfloat16m1_t vb = __riscv_vle16_v_bf16m1((const __bf16 *)btile, vl);
        acc = __riscv_sf_vfwmacc_4x4x4_f32m2(acc, va, vb, vl);
        __riscv_vse32_v_f32m2(ctile, acc, vl);
    }

    float out0 = ctile[0];
    float out1 = ctile[4];
    float out2 = ctile[8];
    float out3 = ctile[12];
    for (; kk < total; kk++) {
        float y = pacc_bf16_to_f32(yh[kk]);
        out0 += mmvf_load_x(ctx, xq + (int64_t)0 * ctx->stride_row, kk) * y;
        out1 += mmvf_load_x(ctx, xq + (int64_t)1 * ctx->stride_row, kk) * y;
        out2 += mmvf_load_x(ctx, xq + (int64_t)2 * ctx->stride_row, kk) * y;
        out3 += mmvf_load_x(ctx, xq + (int64_t)3 * ctx->stride_row, kk) * y;
    }
    dst_base[row_base + 0] = out0;
    dst_base[row_base + 1] = out1;
    dst_base[row_base + 2] = out2;
    dst_base[row_base + 3] = out3;
    return true;
#else
    (void)ctx;
    (void)x_base;
    (void)y_base;
    (void)dst_base;
    (void)row_base;
    return false;
#endif
}

static void *mmvf_native_worker_main(void *opaque) {
    struct MmvfNativeWorker *worker = (struct MmvfNativeWorker *)opaque;
    const struct MmvfNativeCtx *ctx = worker->ctx;
    const uint32_t channel_ratio = bin_bcast_fast_dim(ctx->channel_ratio);
    const uint32_t sample_ratio = bin_bcast_fast_dim(ctx->sample_ratio);

    for (uint64_t idx = worker->begin; idx < worker->end; idx++) {
        uint64_t t = idx;
        uint32_t row = (uint32_t)(t % ctx->grid_x);
        t /= ctx->grid_x;
        uint32_t channel_dst = (uint32_t)(t % ctx->grid_y);
        uint32_t sample_dst = (uint32_t)(t / ctx->grid_y);

        uint32_t channel_x = channel_dst / channel_ratio;
        uint32_t channel_y = channel_dst;
        uint32_t sample_x = sample_dst / sample_ratio;
        uint32_t sample_y = sample_dst;

        const size_t x_elem = mmvf_type_size(ctx->x_type);
        const size_t y_elem = mmvf_type_size(ctx->y_type);
        const void *x_base = (const char *)ctx->x +
            ((int64_t)sample_x * ctx->stride_sample_x +
             (int64_t)channel_x * ctx->stride_channel_x +
             (int64_t)row * ctx->stride_row) * (int64_t)x_elem;
        const void *y_base = (const char *)ctx->y +
            ((int64_t)sample_y * ctx->stride_sample_y +
             (int64_t)channel_y * ctx->stride_channel_y) * (int64_t)y_elem;
        float *dst_base = ctx->dst +
            (int64_t)sample_dst * ctx->stride_sample_dst +
            (int64_t)channel_dst * ctx->stride_channel_dst;

        if (ctx->x_type == PACC_MMVF_BF16 && ctx->y_type == PACC_MMVF_BF16 &&
            ctx->ncols_dst == 1) {
            uint32_t row_group = row & ~3u;
            uint64_t group_idx = idx - (uint64_t)(row - row_group);
            bool group_in_worker = row_group + 3u < ctx->grid_x &&
                                   group_idx >= worker->begin &&
                                   group_idx + 3u < worker->end;
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
            if (jobd_xsfmm_gemm_enabled() && group_in_worker) {
                if (row != row_group) {
                    continue;
                }
                const void *x_group_base = (const char *)x_base -
                    (int64_t)(row - row_group) * ctx->stride_row *
                    (int64_t)x_elem;
                if (mmvf_xsfmm_bf16_rows4(ctx, x_group_base, y_base,
                                          dst_base, row_group)) {
                    continue;
                }
            }
#else
            (void)group_in_worker;
#endif
        } else if (ctx->x_type == PACC_MMVF_Q8_0 && ctx->y_type == PACC_MMVF_BF16 &&
                   ctx->ncols_dst == 1) {
            uint32_t row_group = row & ~3u;
            uint64_t group_idx = idx - (uint64_t)(row - row_group);
            bool group_in_worker = row_group + 3u < ctx->grid_x &&
                                   group_idx >= worker->begin &&
                                   group_idx + 3u < worker->end;
#if defined(HETGPU_PACC_HAVE_XSFMM_BF16)
            if (jobd_xsfmm_gemm_enabled() && group_in_worker) {
                if (row != row_group) {
                    continue;
                }
                const void *x_group_base = (const char *)x_base -
                    (int64_t)(row - row_group) * ctx->stride_row *
                    (int64_t)x_elem;
                if (mmvf_xsfmm_q8_0_bf16_rows4(ctx, x_group_base, y_base,
                                               dst_base, row_group)) {
                    continue;
                }
            }
#else
            (void)group_in_worker;
#endif
        }

        if (ctx->x_type == PACC_MMVF_F16 && ctx->y_type == PACC_MMVF_F32 &&
            ctx->ncols_dst == 1) {
            const uint16_t *xh = (const uint16_t *)x_base;
            const float *yf = (const float *)y_base;
            int32_t total = ctx->ncols2 * 2;
#if defined(__riscv_vector) && (defined(__riscv_zfh) || defined(__riscv_zfhmin))
            dst_base[row] = pacc_dot_f16_f32_rvv(xh, yf, total);
            continue;
#else
            int32_t i = 0;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (; i + 7 < total; i += 8) {
                s0 += pacc_f16_to_f32(xh[i + 0]) * yf[i + 0];
                s1 += pacc_f16_to_f32(xh[i + 1]) * yf[i + 1];
                s2 += pacc_f16_to_f32(xh[i + 2]) * yf[i + 2];
                s3 += pacc_f16_to_f32(xh[i + 3]) * yf[i + 3];
                s0 += pacc_f16_to_f32(xh[i + 4]) * yf[i + 4];
                s1 += pacc_f16_to_f32(xh[i + 5]) * yf[i + 5];
                s2 += pacc_f16_to_f32(xh[i + 6]) * yf[i + 6];
                s3 += pacc_f16_to_f32(xh[i + 7]) * yf[i + 7];
            }
            float sum = (s0 + s1) + (s2 + s3);
            for (; i < total; i++) {
                sum += pacc_f16_to_f32(xh[i]) * yf[i];
            }
            dst_base[row] = sum;
            continue;
#endif
        }

        for (uint32_t j = 0; j < ctx->ncols_dst; j++) {
            float sum = 0.0f;
            for (int32_t col2 = 0; col2 < ctx->ncols2; col2++) {
                float x0 = mmvf_load_x(ctx, x_base, (int64_t)col2 * 2);
                float x1 = mmvf_load_x(ctx, x_base, (int64_t)col2 * 2 + 1);
                const void *y2 = (const char *)y_base +
                    (((int64_t)j * ctx->stride_col_y2 + col2) * 2) *
                    (int64_t)y_elem;
                sum += x0 * mmvf_load_value(ctx->y_type, y2, 0) +
                       x1 * mmvf_load_value(ctx->y_type, y2, 1);
            }
            dst_base[(int64_t)j * ctx->stride_col_dst + row] = sum;
        }
    }

    worker->status = 0;
    return NULL;
}

static int invoke_kernel_mmvf_native(const char *symbol,
                                     const uint64_t *argv,
                                     const struct PaccJobImage *job,
                                     size_t argc) {
    uint64_t t0 = monotonic_us();
    if (!symbol || !strstr(symbol, "mul_mat_vec_f")) return 1;
    if (!job || argc < 19) return -1;

    struct MmvfNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    bool has_fusion = false;
    bool is_multi_token_id = false;
    ctx.x_type = mmvf_type_from_symbol(symbol);
    ctx.y_type = PACC_MMVF_F32;
    if (ctx.x_type == PACC_MMVF_UNSUPPORTED) return 1;
    if (!mmvf_parse_template(symbol, &ctx.ncols_dst, &has_fusion, &is_multi_token_id)) return 1;
    if (has_fusion || is_multi_token_id) return 1;
    if (ctx.ncols_dst == 0 || ctx.ncols_dst > 16) return 1;
    if (ctx.x_type == PACC_MMVF_F16) {
        pacc_prepare_f16_table();
    }

    ctx.x = (const void *)(uintptr_t)kernel_cell_u64(argv, argc, 0);
    ctx.y = (const void *)(uintptr_t)kernel_cell_u64(argv, argc, 1);
    ctx.ids = (const int32_t *)(uintptr_t)kernel_cell_u64(argv, argc, 2);
    ctx.dst = (float *)(uintptr_t)kernel_cell_u64(argv, argc, 4);
    if (ctx.ids) return 1;
    if (!ctx.x || !ctx.y || !ctx.dst) return -1;

    ctx.ncols2 = kernel_cell_i32(argv, argc, 5);
    ctx.nchannels_y = kernel_cell_u3(argv, argc, 6);
    ctx.stride_row = kernel_cell_i32(argv, argc, 7);
    ctx.stride_col_y2 = kernel_cell_i32(argv, argc, 8);
    ctx.stride_col_dst = kernel_cell_i32(argv, argc, 9);
    ctx.channel_ratio = kernel_cell_u3(argv, argc, 10);
    ctx.stride_channel_x = kernel_cell_i32(argv, argc, 11);
    ctx.stride_channel_y = kernel_cell_i32(argv, argc, 12);
    ctx.stride_channel_dst = kernel_cell_i32(argv, argc, 13);
    ctx.sample_ratio = kernel_cell_u3(argv, argc, 14);
    ctx.stride_sample_x = kernel_cell_i32(argv, argc, 15);
    ctx.stride_sample_y = kernel_cell_i32(argv, argc, 16);
    ctx.stride_sample_dst = kernel_cell_i32(argv, argc, 17);
    ctx.ids_stride = kernel_cell_i32(argv, argc, 18);
    ctx.grid_x = pacc_nonzero_dim(job->header.grid_x);
    ctx.grid_y = pacc_nonzero_dim(job->header.grid_y);
    ctx.grid_z = pacc_nonzero_dim(job->header.grid_z);

    if (ctx.ncols2 <= 0) return 0;
    void *local_x = NULL;
    void *local_y = NULL;
    uint64_t local_x_max = parse_env_u64_default("PACC_JOBD_MMVF_LOCAL_X_MAX_BYTES", 0);
    uint64_t local_y_max = parse_env_u64_default("PACC_JOBD_MMVF_LOCAL_Y_MAX_BYTES", 0);
    if ((ctx.x_type == PACC_MMVF_F16 || ctx.x_type == PACC_MMVF_F32) &&
        ctx.ncols_dst == 1 && local_x_max != 0) {
        uint64_t x_bytes = kernel_binding_size_for_arg(job, 0);
        if (x_bytes > 0 && x_bytes <= local_x_max) {
            local_x = malloc((size_t)x_bytes);
            if (local_x) {
                memcpy(local_x, ctx.x, (size_t)x_bytes);
                ctx.x = local_x;
            }
        }
    }
    if (ctx.stride_col_y2 > 0 && local_y_max != 0) {
        uint64_t y_elems = 2u * ((uint64_t)(ctx.ncols_dst - 1u) *
                                 (uint64_t)ctx.stride_col_y2 +
                                 (uint64_t)ctx.ncols2);
        if (y_elems > 0 && y_elems <= local_y_max / sizeof(float)) {
            size_t y_bytes = (size_t)y_elems * sizeof(float);
            local_y = malloc(y_bytes);
            if (local_y) {
                memcpy(local_y, ctx.y, y_bytes);
                ctx.y = local_y;
            }
        }
    }

    uint64_t work_items = (uint64_t)ctx.grid_x * ctx.grid_y * ctx.grid_z;
    unsigned workers = jobd_mmvf_worker_threads(work_items);
    trace_msg("native mmvf %s work=%" PRIu64 " ncols2=%d ncols_dst=%u workers=%u",
              symbol, work_items, ctx.ncols2, ctx.ncols_dst, workers);

    if (workers <= 1 || work_items <= 1) {
        struct MmvfNativeWorker worker = {
            .ctx = &ctx,
            .begin = 0,
            .end = work_items,
            .status = 0,
        };
        mmvf_native_worker_main(&worker);
        int worker_status = worker.status;
        trace_msg("native mmvf done %s status=%d elapsed_us=%" PRIu64,
                  symbol, worker_status, monotonic_us() - t0);
        free(local_x);
        free(local_y);
        return worker_status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct MmvfNativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (work_items + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= work_items) break;
        if (end > work_items) end = work_items;
        worker[i].ctx = &ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, mmvf_native_worker_main, &worker[i]) != 0) {
            log_msg("native mmvf failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            free(local_x);
            free(local_y);
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    trace_msg("native mmvf done %s status=%d elapsed_us=%" PRIu64,
              symbol, status, monotonic_us() - t0);
    free(local_x);
    free(local_y);
    return status;
}

static int run_mmvf_ctx(struct MmvfNativeCtx *ctx,
                        uint64_t x_bytes,
                        uint64_t y_bytes,
                        const char *label) {
    uint64_t t0 = monotonic_us();
    void *local_x = NULL;
    void *local_y = NULL;

    if (!ctx || !ctx->x || !ctx->y || !ctx->dst || ctx->ids) return 0xffff5001;
    if (ctx->ncols2 <= 0) return 0;
    if (ctx->ncols_dst == 0 || ctx->ncols_dst > 16) return 0xffff5002;
    if (ctx->x_type != PACC_MMVF_F32 && ctx->x_type != PACC_MMVF_F16 &&
        ctx->x_type != PACC_MMVF_BF16 && ctx->x_type != PACC_MMVF_Q8_0) {
        return 0xffff5003;
    }
    if (ctx->y_type != PACC_MMVF_F32 && ctx->y_type != PACC_MMVF_F16 &&
        ctx->y_type != PACC_MMVF_BF16) {
        return 0xffff5003;
    }
    if (ctx->x_type == PACC_MMVF_F16 || ctx->x_type == PACC_MMVF_Q8_0 ||
        ctx->y_type == PACC_MMVF_F16) {
        pacc_prepare_f16_table();
    }

    uint64_t local_x_max = parse_env_u64_default("PACC_JOBD_MMVF_LOCAL_X_MAX_BYTES", 0);
    uint64_t local_y_max = parse_env_u64_default("PACC_JOBD_MMVF_LOCAL_Y_MAX_BYTES", 0);

    if ((ctx->x_type == PACC_MMVF_F16 || ctx->x_type == PACC_MMVF_F32 ||
         ctx->x_type == PACC_MMVF_BF16 || ctx->x_type == PACC_MMVF_Q8_0) &&
        ctx->ncols_dst == 1 &&
        local_x_max != 0 && x_bytes > 0 && x_bytes <= local_x_max) {
        local_x = malloc((size_t)x_bytes);
        if (local_x) {
            memcpy(local_x, ctx->x, (size_t)x_bytes);
            ctx->x = local_x;
        }
    }
    if (local_y_max != 0 && y_bytes > 0 && y_bytes <= local_y_max) {
        local_y = malloc((size_t)y_bytes);
        if (local_y) {
            memcpy(local_y, ctx->y, (size_t)y_bytes);
            ctx->y = local_y;
        }
    }

    uint64_t work_items = (uint64_t)pacc_nonzero_dim(ctx->grid_x) *
                          pacc_nonzero_dim(ctx->grid_y) *
                          pacc_nonzero_dim(ctx->grid_z);
    unsigned workers = jobd_mmvf_worker_threads(work_items);
    trace_msg("native mmvf direct %s work=%" PRIu64 " ncols2=%d ncols_dst=%u workers=%u",
              label ? label : "MMVF", work_items, ctx->ncols2, ctx->ncols_dst, workers);

    if (workers <= 1 || work_items <= 1) {
        struct MmvfNativeWorker worker = {
            .ctx = ctx,
            .begin = 0,
            .end = work_items,
            .status = 0,
        };
        mmvf_native_worker_main(&worker);
        int worker_status = worker.status;
        trace_msg("native mmvf direct done %s status=%d elapsed_us=%" PRIu64,
                  label ? label : "MMVF", worker_status, monotonic_us() - t0);
        free(local_x);
        free(local_y);
        return worker_status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct MmvfNativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (work_items + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= work_items) break;
        if (end > work_items) end = work_items;
        worker[i].ctx = ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, mmvf_native_worker_main, &worker[i]) != 0) {
            log_msg("native mmvf direct failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            free(local_x);
            free(local_y);
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    trace_msg("native mmvf direct done %s status=%d elapsed_us=%" PRIu64,
              label ? label : "MMVF", status, monotonic_us() - t0);
    free(local_x);
    free(local_y);
    return status;
}

static int run_mmvf(int fd, const struct MmvfJob *job, uint64_t seq) {
    mirror_progress_status(fd, HETGPU_PACC_JOB_MMVF, seq, 0x5140);
    if (!job || !job->x_addr || !job->y_addr || !job->dst_addr ||
        !job->grid_x || !job->grid_y || !job->grid_z ||
        !job->x_bytes || !job->y_bytes || !job->dst_bytes) {
        return 0xffff5001;
    }
    if (job->ids_addr) {
        return 0xffff5004;
    }

    struct Map mx = {0}, my = {0}, md = {0};
    bool copy_io = jobd_mmvf_copy_io_enabled();
    bool compute = jobd_mmvf_compute_enabled();
    mirror_progress_status(fd, HETGPU_PACC_JOB_MMVF, seq, 0x5141);
    if (map_phys_for_mmvf(fd, job->dst_addr, (size_t)job->dst_bytes, &md, copy_io)) {
        unmap_phys(&mx);
        unmap_phys(&my);
        unmap_phys(&md);
        return 0xffff5005;
    }

    if (jobd_mmvf_clear_dst_enabled()) {
        memset(md.ptr, 0, (size_t)job->dst_bytes);
        if (flush_map_to_phys(&md) != 0) {
            unmap_phys(&md);
            return 0xffff5006;
        }
        if (!md.copied) {
            if (jobd_msync_enabled()) {
                msync(md.base, md.map_len, MS_SYNC);
            }
            jobd_flush_for_device(md.ptr, md.len);
        }
    }
    if (!compute) {
        mirror_progress_status(fd, HETGPU_PACC_JOB_MMVF, seq, 0x5144);
        unmap_phys(&md);
        return 0;
    }

    if (map_phys_for_mmvf(fd, job->x_addr, (size_t)job->x_bytes, &mx, copy_io) ||
        map_phys_for_mmvf(fd, job->y_addr, (size_t)job->y_bytes, &my, copy_io)) {
        unmap_phys(&mx);
        unmap_phys(&my);
        unmap_phys(&md);
        return 0xffff5005;
    }

    struct MmvfNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.x = mx.ptr;
    ctx.y = my.ptr;
    ctx.ids = NULL;
    ctx.dst = (float *)md.ptr;
    ctx.ncols2 = job->ncols2;
    ctx.nchannels_y = job->nchannels_y;
    ctx.stride_row = job->stride_row;
    ctx.stride_col_y2 = job->stride_col_y2;
    ctx.stride_col_dst = job->stride_col_dst;
    ctx.channel_ratio = job->channel_ratio;
    ctx.stride_channel_x = job->stride_channel_x;
    ctx.stride_channel_y = job->stride_channel_y;
    ctx.stride_channel_dst = job->stride_channel_dst;
    ctx.sample_ratio = job->sample_ratio;
    ctx.stride_sample_x = job->stride_sample_x;
    ctx.stride_sample_y = job->stride_sample_y;
    ctx.stride_sample_dst = job->stride_sample_dst;
    ctx.ids_stride = job->ids_stride;
    ctx.grid_x = job->grid_x;
    ctx.grid_y = job->grid_y;
    ctx.grid_z = job->grid_z;
    ctx.ncols_dst = job->ncols_dst;
    ctx.x_type = (enum PaccMmvfXType)job->x_type;
    ctx.y_type = mmvf_type_from_job_field(job->reserved0);

    mirror_progress_status(fd, HETGPU_PACC_JOB_MMVF, seq, 0x5142);
    int status = run_mmvf_ctx(&ctx, job->x_bytes, job->y_bytes, "MMVF");
    mirror_progress_status(fd, HETGPU_PACC_JOB_MMVF, seq, status == 0 ? 0x5143 : (uint32_t)status);
    if (status == 0) {
        if (flush_map_to_phys(&md) != 0) {
            status = 0xffff5006;
            mirror_progress_status(fd, HETGPU_PACC_JOB_MMVF, seq, (uint32_t)status);
        }
    }
    if (status == 0 && !md.copied) {
        if (jobd_msync_enabled()) {
            msync(md.base, md.map_len, MS_SYNC);
        }
        jobd_flush_for_device(md.ptr, md.len);
    }
    if (status == 0) {
        mirror_progress_status(fd, HETGPU_PACC_JOB_MMVF, seq, 0x5144);
    }
    unmap_phys(&mx);
    unmap_phys(&my);
    unmap_phys(&md);
    return status;
}

enum PaccRopeScalarType {
    PACC_ROPE_SCALAR_UNSUPPORTED = 0,
    PACC_ROPE_SCALAR_F32,
    PACC_ROPE_SCALAR_F16,
};

struct RopeNormNativeCtx {
    const void *x;
    void *dst;
    uint64_t x_elems;
    uint64_t dst_elems;
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t s01;
    int32_t s02;
    int32_t s03;
    int32_t s1;
    int32_t s2;
    int32_t s3;
    int32_t n_dims;
    const int32_t *pos;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float corr_low;
    float corr_high;
    float theta_scale;
    const float *freq_factors;
    const int64_t *row_indices;
    uint64_t pos_elems;
    uint64_t freq_elems;
    uint64_t row_indices_elems;
    int32_t set_rows_stride;
    int32_t sections[4];
    uint32_t row_count;
    uint32_t pair_count;
    bool forward;
    bool has_ff;
    bool is_imrope;
    enum PaccRopeScalarType x_type;
    enum PaccRopeScalarType dst_type;
};

struct RopeNormNativeWorker {
    const struct RopeNormNativeCtx *ctx;
    uint64_t begin;
    uint64_t end;
    int status;
};

static uint16_t pacc_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t rounded = (mant + (1u << (shift - 1))) >> shift;
        return (uint16_t)(sign | rounded);
    }
    if (exp >= 0x1f) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }

    mant += 0x00001000u;
    if (mant & 0x00800000u) {
        mant = 0;
        exp++;
        if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

enum PaccConvertScalarType {
    PACC_CONVERT_SCALAR_UNSUPPORTED = 0,
    PACC_CONVERT_SCALAR_F16,
    PACC_CONVERT_SCALAR_F32,
};

struct ConvertUnaryNativeCtx {
    const void *src;
    void *dst;
    int64_t ne00;
    int64_t ne01;
    int64_t ne0203;
    int64_t ne02;
    int64_t s01;
    int64_t s02;
    int64_t s03;
    enum PaccConvertScalarType src_type;
    enum PaccConvertScalarType dst_type;
};

struct ConvertUnaryNativeWorker {
    const struct ConvertUnaryNativeCtx *ctx;
    uint64_t begin;
    uint64_t end;
    int status;
};

struct ScaleF32NativeCtx {
    const float *src;
    float *dst;
    float scale;
    float bias;
    uint64_t nelements;
};

struct ScaleF32NativeWorker {
    const struct ScaleF32NativeCtx *ctx;
    uint64_t begin;
    uint64_t end;
    int status;
};

struct CpyScalarF32NativeCtx {
    const char *src;
    char *dst;
    int64_t ne;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t nb00;
    int64_t nb01;
    int64_t nb02;
    int64_t nb03;
    int64_t ne10;
    int64_t ne11;
    int64_t ne12;
    int64_t nb10;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;
    bool contiguous;
};

struct CpyScalarF32NativeWorker {
    const struct CpyScalarF32NativeCtx *ctx;
    uint64_t begin;
    uint64_t end;
    int status;
};

enum PaccGetRowsScalarType {
    PACC_GET_ROWS_SCALAR_UNSUPPORTED = 0,
    PACC_GET_ROWS_SCALAR_F16,
    PACC_GET_ROWS_SCALAR_F32,
    PACC_GET_ROWS_SCALAR_I32,
    PACC_GET_ROWS_SCALAR_BF16,
};

struct GetRowsFloatNativeCtx {
    const void *src0;
    const int32_t *src1;
    void *dst;
    int64_t ne00;
    int64_t ne10;
    int64_t ne11;
    int64_t ne12;
    int64_t s1;
    int64_t s2;
    int64_t s3;
    size_t nb01;
    size_t nb02;
    size_t nb03;
    int64_t s10;
    int64_t s11;
    int64_t s12;
    enum PaccGetRowsScalarType src_type;
    enum PaccGetRowsScalarType dst_type;
};

struct GetRowsFloatNativeWorker {
    const struct GetRowsFloatNativeCtx *ctx;
    uint64_t begin_row;
    uint64_t end_row;
    int status;
};

static size_t get_rows_scalar_size(enum PaccGetRowsScalarType ty) {
    switch (ty) {
    case PACC_GET_ROWS_SCALAR_F16:
    case PACC_GET_ROWS_SCALAR_BF16:
        return sizeof(uint16_t);
    case PACC_GET_ROWS_SCALAR_F32:
        return sizeof(float);
    case PACC_GET_ROWS_SCALAR_I32:
        return sizeof(int32_t);
    default:
        return 0;
    }
}

static bool get_rows_parse_scalar(const char **p,
                                  enum PaccGetRowsScalarType *out) {
    if (!p || !*p || !out) return false;
    if (strncmp(*p, "6__half", 7) == 0) {
        *out = PACC_GET_ROWS_SCALAR_F16;
        *p += 7;
        return true;
    }
    if (strncmp(*p, "13__nv_bfloat16", 15) == 0) {
        *out = PACC_GET_ROWS_SCALAR_BF16;
        *p += 15;
        return true;
    }
    if (**p == 'f') {
        *out = PACC_GET_ROWS_SCALAR_F32;
        *p += 1;
        return true;
    }
    if (**p == 'i') {
        *out = PACC_GET_ROWS_SCALAR_I32;
        *p += 1;
        return true;
    }
    return false;
}

static bool get_rows_float_parse_template(const char *symbol,
                                          enum PaccGetRowsScalarType *src_type,
                                          enum PaccGetRowsScalarType *dst_type) {
    const char *p = symbol ? strstr(symbol, "k_get_rows_floatI") : NULL;
    if (!p) return false;
    p += strlen("k_get_rows_floatI");
    if (!get_rows_parse_scalar(&p, src_type)) return false;
    if (strncmp(p, "S0_", 3) == 0) {
        *dst_type = *src_type;
        return true;
    }
    return get_rows_parse_scalar(&p, dst_type);
}

static float get_rows_load_scalar(const void *base,
                                  uint64_t index,
                                  enum PaccGetRowsScalarType ty) {
    switch (ty) {
    case PACC_GET_ROWS_SCALAR_F16:
        return pacc_f16_to_f32(((const uint16_t *)base)[index]);
    case PACC_GET_ROWS_SCALAR_F32:
        return ((const float *)base)[index];
    case PACC_GET_ROWS_SCALAR_I32:
        return (float)((const int32_t *)base)[index];
    case PACC_GET_ROWS_SCALAR_BF16:
        return bf16_to_f32(((const uint16_t *)base)[index]);
    default:
        return 0.0f;
    }
}

static void get_rows_store_scalar(void *base,
                                  uint64_t index,
                                  enum PaccGetRowsScalarType ty,
                                  float value) {
    switch (ty) {
    case PACC_GET_ROWS_SCALAR_F16:
        ((uint16_t *)base)[index] = pacc_f32_to_f16(value);
        break;
    case PACC_GET_ROWS_SCALAR_F32:
        ((float *)base)[index] = value;
        break;
    case PACC_GET_ROWS_SCALAR_I32:
        ((int32_t *)base)[index] = round_to_i32(value);
        break;
    case PACC_GET_ROWS_SCALAR_BF16:
        ((uint16_t *)base)[index] = f32_to_bf16(value);
        break;
    default:
        break;
    }
}

static void get_rows_copy_row(const struct GetRowsFloatNativeCtx *ctx,
                              const void *src_row,
                              void *dst_row) {
    if (ctx->src_type == PACC_GET_ROWS_SCALAR_F32 &&
        ctx->dst_type == PACC_GET_ROWS_SCALAR_F32) {
#if defined(__riscv_vector)
        uint64_t i = 0;
        while (i < (uint64_t)ctx->ne00) {
            size_t remaining = (size_t)((uint64_t)ctx->ne00 - i);
            size_t vl = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v = __riscv_vle32_v_f32m4((const float *)src_row + i, vl);
            __riscv_vse32_v_f32m4((float *)dst_row + i, v, vl);
            i += vl;
        }
#else
        memcpy(dst_row, src_row, (size_t)ctx->ne00 * sizeof(float));
#endif
        return;
    }

    for (uint64_t i = 0; i < (uint64_t)ctx->ne00; i++) {
        float value = get_rows_load_scalar(src_row, i, ctx->src_type);
        get_rows_store_scalar(dst_row, i, ctx->dst_type, value);
    }
}

static void *get_rows_float_native_worker_main(void *opaque) {
    struct GetRowsFloatNativeWorker *worker =
        (struct GetRowsFloatNativeWorker *)opaque;
    const struct GetRowsFloatNativeCtx *ctx = worker->ctx;
    size_t dst_elem = get_rows_scalar_size(ctx->dst_type);

    for (uint64_t row = worker->begin_row; row < worker->end_row; row++) {
        int64_t i10 = (int64_t)(row % (uint64_t)ctx->ne10);
        uint64_t z = row / (uint64_t)ctx->ne10;
        int64_t i11 = (int64_t)(z / (uint64_t)ctx->ne12);
        int64_t i12 = (int64_t)(z % (uint64_t)ctx->ne12);
        int64_t src1_off = i10 * ctx->s10 + i11 * ctx->s11 + i12 * ctx->s12;
        int32_t i01 = ctx->src1[src1_off];
        if (i01 < 0) {
            worker->status = -1;
            return NULL;
        }

        const void *src_row = (const char *)ctx->src0 +
            (size_t)i01 * ctx->nb01 +
            (size_t)i11 * ctx->nb02 +
            (size_t)i12 * ctx->nb03;
        int64_t dst_off = i10 * ctx->s1 + i11 * ctx->s2 + i12 * ctx->s3;
        void *dst_row = (char *)ctx->dst + (uint64_t)dst_off * dst_elem;
        get_rows_copy_row(ctx, src_row, dst_row);
    }

    worker->status = 0;
    return NULL;
}

static int invoke_kernel_get_rows_float_native(const char *symbol,
                                               const uint64_t *args,
                                               const struct PaccJobImage *job,
                                               size_t argc) {
    uint64_t t0 = monotonic_us();
    if (!symbol || !strstr(symbol, "k_get_rows_float")) return 1;
    if (!env_flag_default_true("HETGPU_PACC_ENABLE_NATIVE_GET_ROWS_FLOAT")) return 1;
    if (!job || argc < 15) return -1;

    struct GetRowsFloatNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (!get_rows_float_parse_template(symbol, &ctx.src_type, &ctx.dst_type) ||
        ctx.src_type == PACC_GET_ROWS_SCALAR_UNSUPPORTED ||
        ctx.dst_type == PACC_GET_ROWS_SCALAR_UNSUPPORTED) {
        return 1;
    }

    ctx.src0 = (const void *)(uintptr_t)kernel_cell_u64(args, argc, 0);
    ctx.src1 = (const int32_t *)(uintptr_t)kernel_cell_u64(args, argc, 1);
    ctx.dst = (void *)(uintptr_t)kernel_cell_u64(args, argc, 2);
    ctx.ne00 = kernel_cell_i64(args, argc, 3);
    ctx.ne11 = kernel_cell_i64(args, argc, 4);
    ctx.ne12 = kernel_cell_i64(args, argc, 5);
    ctx.s1 = kernel_cell_i64(args, argc, 6);
    ctx.s2 = kernel_cell_i64(args, argc, 7);
    ctx.s3 = kernel_cell_i64(args, argc, 8);
    ctx.nb01 = (size_t)kernel_cell_u64(args, argc, 9);
    ctx.nb02 = (size_t)kernel_cell_u64(args, argc, 10);
    ctx.nb03 = (size_t)kernel_cell_u64(args, argc, 11);
    ctx.s10 = kernel_cell_i64(args, argc, 12);
    ctx.s11 = kernel_cell_i64(args, argc, 13);
    ctx.s12 = kernel_cell_i64(args, argc, 14);
    ctx.ne10 = (int64_t)pacc_nonzero_dim(job->header.grid_x);

    if (!ctx.src0 || !ctx.src1 || !ctx.dst || ctx.ne00 <= 0 ||
        ctx.ne10 <= 0 || ctx.ne11 <= 0 || ctx.ne12 <= 0 ||
        ctx.s1 < 0 || ctx.s2 < 0 || ctx.s3 < 0 ||
        ctx.s10 < 0 || ctx.s11 < 0 || ctx.s12 < 0 ||
        get_rows_scalar_size(ctx.src_type) == 0 ||
        get_rows_scalar_size(ctx.dst_type) == 0) {
        log_msg("native get_rows_float invalid: symbol=%s src0=%p src1=%p dst=%p "
                "ne=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ")",
                symbol, ctx.src0, (const void *)ctx.src1, ctx.dst,
                ctx.ne00, ctx.ne10, ctx.ne11, ctx.ne12);
        return -1;
    }

    uint64_t rows = (uint64_t)ctx.ne10 * (uint64_t)ctx.ne11 * (uint64_t)ctx.ne12;
    unsigned workers = kernel_worker_threads(rows);
    trace_msg("native get_rows_float %s rows=%" PRIu64 " ne00=%" PRId64
              " ne10=%" PRId64 " ne11=%" PRId64 " ne12=%" PRId64
              " src_type=%d dst_type=%d workers=%u",
              symbol, rows, ctx.ne00, ctx.ne10, ctx.ne11, ctx.ne12,
              ctx.src_type, ctx.dst_type, workers);

    if (workers <= 1 || rows <= 1) {
        struct GetRowsFloatNativeWorker worker = {
            .ctx = &ctx,
            .begin_row = 0,
            .end_row = rows,
            .status = 0,
        };
        get_rows_float_native_worker_main(&worker);
        trace_msg("native get_rows_float done %s status=%d elapsed_us=%" PRIu64,
                  symbol, worker.status, monotonic_us() - t0);
        return worker.status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct GetRowsFloatNativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (rows + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= rows) break;
        if (end > rows) end = rows;
        worker[i].ctx = &ctx;
        worker[i].begin_row = begin;
        worker[i].end_row = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, get_rows_float_native_worker_main, &worker[i]) != 0) {
            log_msg("native get_rows_float failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    trace_msg("native get_rows_float done %s status=%d elapsed_us=%" PRIu64,
              symbol, status, monotonic_us() - t0);
    return status;
}

static void *scale_f32_native_worker_main(void *opaque) {
    struct ScaleF32NativeWorker *worker = (struct ScaleF32NativeWorker *)opaque;
    const struct ScaleF32NativeCtx *ctx = worker->ctx;

#if defined(__riscv_vector)
    uint64_t i = worker->begin;
    while (i < worker->end) {
        size_t remaining = (size_t)(worker->end - i);
        size_t vl = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t x = __riscv_vle32_v_f32m4(ctx->src + i, vl);
        vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(ctx->bias, vl);
        y = __riscv_vfmacc_vf_f32m4(y, ctx->scale, x, vl);
        __riscv_vse32_v_f32m4(ctx->dst + i, y, vl);
        i += vl;
    }
#else
    for (uint64_t i = worker->begin; i < worker->end; i++) {
        ctx->dst[i] = ctx->src[i] * ctx->scale + ctx->bias;
    }
#endif
    worker->status = 0;
    return NULL;
}

static int invoke_kernel_scale_f32_native(const char *symbol,
                                          const uint64_t *args,
                                          const struct PaccJobImage *job,
                                          size_t argc) {
    uint64_t t0 = monotonic_us();
    (void)job;
    if (!symbol || !strstr(symbol, "scale_f32")) return 1;
    if (!env_flag_default_true("HETGPU_PACC_ENABLE_NATIVE_SCALE_F32")) return 1;
    if (argc < 5) return -1;

    struct ScaleF32NativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src = (const float *)(uintptr_t)kernel_cell_u64(args, argc, 0);
    ctx.dst = (float *)(uintptr_t)kernel_cell_u64(args, argc, 1);
    ctx.scale = kernel_cell_f32(args, argc, 2);
    ctx.bias = kernel_cell_f32(args, argc, 3);
    int64_t nelements = kernel_cell_i64(args, argc, 4);
    if (nelements <= 0) return 0;
    ctx.nelements = (uint64_t)nelements;
    if (!ctx.src || !ctx.dst) return -1;

    unsigned workers = kernel_worker_threads(ctx.nelements);
    uint64_t single_thread_max = scale_f32_single_thread_max_elements();
    if (workers > 1 && single_thread_max != 0 && ctx.nelements <= single_thread_max) {
        workers = 1;
    }
    trace_msg("native scale_f32 %s elems=%" PRIu64
              " scale=%g bias=%g workers=%u",
              symbol, ctx.nelements, ctx.scale, ctx.bias, workers);

    if (workers <= 1 || ctx.nelements <= 1) {
        struct ScaleF32NativeWorker worker = {
            .ctx = &ctx,
            .begin = 0,
            .end = ctx.nelements,
            .status = 0,
        };
        scale_f32_native_worker_main(&worker);
        trace_msg("native scale_f32 done %s status=%d elapsed_us=%" PRIu64,
                  symbol, worker.status, monotonic_us() - t0);
        return worker.status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct ScaleF32NativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (ctx.nelements + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= ctx.nelements) break;
        if (end > ctx.nelements) end = ctx.nelements;
        worker[i].ctx = &ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, scale_f32_native_worker_main, &worker[i]) != 0) {
            log_msg("native scale_f32 failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    trace_msg("native scale_f32 done %s status=%d elapsed_us=%" PRIu64,
              symbol, status, monotonic_us() - t0);
    return status;
}

static void *cpy_scalar_f32_native_worker_main(void *opaque) {
    struct CpyScalarF32NativeWorker *worker =
        (struct CpyScalarF32NativeWorker *)opaque;
    const struct CpyScalarF32NativeCtx *ctx = worker->ctx;

    if (ctx->contiguous) {
        const float *src = (const float *)(const void *)ctx->src;
        float *dst = (float *)(void *)ctx->dst;
        for (uint64_t i = worker->begin; i < worker->end; i++) {
            dst[i] = src[i];
        }
        worker->status = 0;
        return NULL;
    }

    for (uint64_t u = worker->begin; u < worker->end; u++) {
        int64_t i = (int64_t)u;
        int64_t i03 = i / (ctx->ne00 * ctx->ne01 * ctx->ne02);
        int64_t i02 = (i - i03 * ctx->ne00 * ctx->ne01 * ctx->ne02) /
                      (ctx->ne00 * ctx->ne01);
        int64_t i01 = (i - i03 * ctx->ne00 * ctx->ne01 * ctx->ne02 -
                       i02 * ctx->ne01 * ctx->ne00) / ctx->ne00;
        int64_t i00 = i - i03 * ctx->ne00 * ctx->ne01 * ctx->ne02 -
                      i02 * ctx->ne01 * ctx->ne00 - i01 * ctx->ne00;
        int64_t x_offset = i00 * ctx->nb00 + i01 * ctx->nb01 +
                           i02 * ctx->nb02 + i03 * ctx->nb03;

        int64_t i13 = i / (ctx->ne10 * ctx->ne11 * ctx->ne12);
        int64_t i12 = (i - i13 * ctx->ne10 * ctx->ne11 * ctx->ne12) /
                      (ctx->ne10 * ctx->ne11);
        int64_t i11 = (i - i13 * ctx->ne10 * ctx->ne11 * ctx->ne12 -
                       i12 * ctx->ne10 * ctx->ne11) / ctx->ne10;
        int64_t i10 = i - i13 * ctx->ne10 * ctx->ne11 * ctx->ne12 -
                      i12 * ctx->ne10 * ctx->ne11 - i11 * ctx->ne10;
        int64_t dst_offset = i10 * ctx->nb10 + i11 * ctx->nb11 +
                             i12 * ctx->nb12 + i13 * ctx->nb13;
        memcpy(ctx->dst + dst_offset, ctx->src + x_offset, sizeof(float));
    }
    worker->status = 0;
    return NULL;
}

static int invoke_kernel_cpy_scalar_f32_native(const char *symbol,
                                               const uint64_t *args,
                                               const struct PaccJobImage *job,
                                               size_t argc) {
    uint64_t t0 = monotonic_us();
    (void)job;
    if (!symbol || !strstr(symbol, "cpy_scalar")) return 1;
    if (!env_flag_default_true("HETGPU_PACC_ENABLE_NATIVE_CPY_SCALAR")) return 1;
    if (!strstr(symbol, "cpy_1_scalarIff") &&
        !strstr(symbol, "cpy_scalar_contiguousIff")) {
        return 1;
    }

    struct CpyScalarF32NativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.contiguous = strstr(symbol, "cpy_scalar_contiguousIff") != NULL;
    ctx.src = (const char *)(uintptr_t)kernel_cell_u64(args, argc, 0);
    ctx.dst = (char *)(uintptr_t)kernel_cell_u64(args, argc, 1);
    if (!ctx.src || !ctx.dst) return -1;

    if (ctx.contiguous) {
        if (argc < 3) return -1;
        ctx.ne = kernel_cell_i64(args, argc, 2);
    } else {
        if (argc < 17) return -1;
        ctx.ne = kernel_cell_i64(args, argc, 2);
        ctx.ne00 = kernel_cell_i64(args, argc, 3);
        ctx.ne01 = kernel_cell_i64(args, argc, 4);
        ctx.ne02 = kernel_cell_i64(args, argc, 5);
        ctx.nb00 = kernel_cell_i64(args, argc, 6);
        ctx.nb01 = kernel_cell_i64(args, argc, 7);
        ctx.nb02 = kernel_cell_i64(args, argc, 8);
        ctx.nb03 = kernel_cell_i64(args, argc, 9);
        ctx.ne10 = kernel_cell_i64(args, argc, 10);
        ctx.ne11 = kernel_cell_i64(args, argc, 11);
        ctx.ne12 = kernel_cell_i64(args, argc, 12);
        ctx.nb10 = kernel_cell_i64(args, argc, 13);
        ctx.nb11 = kernel_cell_i64(args, argc, 14);
        ctx.nb12 = kernel_cell_i64(args, argc, 15);
        ctx.nb13 = kernel_cell_i64(args, argc, 16);
        if (ctx.ne00 <= 0 || ctx.ne01 <= 0 || ctx.ne02 <= 0 ||
            ctx.ne10 <= 0 || ctx.ne11 <= 0 || ctx.ne12 <= 0 ||
            ctx.nb00 < 0 || ctx.nb01 < 0 || ctx.nb02 < 0 || ctx.nb03 < 0 ||
            ctx.nb10 < 0 || ctx.nb11 < 0 || ctx.nb12 < 0 || ctx.nb13 < 0) {
            return -1;
        }
    }
    if (ctx.ne <= 0) return 0;

    uint64_t total = (uint64_t)ctx.ne;
    unsigned workers = kernel_worker_threads(total);
    if (workers > 1 && total <= 8192u) {
        workers = 1;
    }
    trace_msg("native cpy_scalar_f32 %s elems=%" PRIu64
              " contiguous=%u workers=%u",
              symbol, total, ctx.contiguous ? 1u : 0u, workers);

    if (workers <= 1 || total <= 1) {
        struct CpyScalarF32NativeWorker worker = {
            .ctx = &ctx,
            .begin = 0,
            .end = total,
            .status = 0,
        };
        cpy_scalar_f32_native_worker_main(&worker);
        trace_msg("native cpy_scalar_f32 done %s status=%d elapsed_us=%" PRIu64,
                  symbol, worker.status, monotonic_us() - t0);
        return worker.status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct CpyScalarF32NativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (total + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= total) break;
        if (end > total) end = total;
        worker[i].ctx = &ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL,
                           cpy_scalar_f32_native_worker_main, &worker[i]) != 0) {
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    trace_msg("native cpy_scalar_f32 done %s status=%d elapsed_us=%" PRIu64,
              symbol, status, monotonic_us() - t0);
    return status;
}

static bool convert_unary_parse_scalar(const char **p,
                                       enum PaccConvertScalarType *out) {
    if (!p || !*p || !out) return false;
    if (strncmp(*p, "6__half", 7) == 0) {
        *out = PACC_CONVERT_SCALAR_F16;
        *p += 7;
        return true;
    }
    if (**p == 'f') {
        *out = PACC_CONVERT_SCALAR_F32;
        *p += 1;
        return true;
    }
    return false;
}

static bool convert_unary_parse_types(const char *symbol,
                                      enum PaccConvertScalarType *src_type,
                                      enum PaccConvertScalarType *dst_type) {
    const char *p = symbol ? strstr(symbol, "convert_unaryI") : NULL;
    if (!p) return false;
    p += strlen("convert_unaryI");
    if (!convert_unary_parse_scalar(&p, src_type)) return false;
    if (strncmp(p, "S0_", 3) == 0) {
        *dst_type = *src_type;
        return true;
    }
    return convert_unary_parse_scalar(&p, dst_type);
}

static float convert_unary_load_scalar(const void *base,
                                       uint64_t index,
                                       enum PaccConvertScalarType ty) {
    switch (ty) {
    case PACC_CONVERT_SCALAR_F16:
        return pacc_f16_to_f32(((const uint16_t *)base)[index]);
    case PACC_CONVERT_SCALAR_F32:
        return ((const float *)base)[index];
    default:
        return 0.0f;
    }
}

static void convert_unary_store_scalar(void *base,
                                       uint64_t index,
                                       enum PaccConvertScalarType ty,
                                       float value) {
    switch (ty) {
    case PACC_CONVERT_SCALAR_F16:
        ((uint16_t *)base)[index] = pacc_f32_to_f16(value);
        break;
    case PACC_CONVERT_SCALAR_F32:
        ((float *)base)[index] = value;
        break;
    default:
        break;
    }
}

static void *convert_unary_native_worker_main(void *opaque) {
    struct ConvertUnaryNativeWorker *worker =
        (struct ConvertUnaryNativeWorker *)opaque;
    const struct ConvertUnaryNativeCtx *ctx = worker->ctx;
    const uint64_t ne00 = (uint64_t)ctx->ne00;
    const uint64_t ne01 = (uint64_t)ctx->ne01;
    const uint64_t ne02 = (uint64_t)ctx->ne02;

    for (uint64_t t = worker->begin; t < worker->end; t++) {
        uint64_t rem = t;
        uint64_t i0 = ne00 ? rem % ne00 : 0;
        rem = ne00 ? rem / ne00 : 0;
        uint64_t i1 = ne01 ? rem % ne01 : 0;
        rem = ne01 ? rem / ne01 : 0;
        uint64_t i2 = ne02 ? rem % ne02 : 0;
        uint64_t i3 = ne02 ? rem / ne02 : 0;
        uint64_t src_index = i0 +
            i1 * (uint64_t)ctx->s01 +
            i2 * (uint64_t)ctx->s02 +
            i3 * (uint64_t)ctx->s03;
        float value = convert_unary_load_scalar(ctx->src, src_index, ctx->src_type);
        convert_unary_store_scalar(ctx->dst, t, ctx->dst_type, value);
    }
    worker->status = 0;
    return NULL;
}

static int invoke_kernel_convert_unary_native(const char *symbol,
                                              const uint64_t *args,
                                              const struct PaccJobImage *job,
                                              size_t argc) {
    (void)job;
    if (!symbol || !strstr(symbol, "convert_unary")) return 1;
    if (!env_flag_default_true("HETGPU_PACC_ENABLE_NATIVE_CONVERT_UNARY")) return 1;
    if (argc < 9) return -1;

    struct ConvertUnaryNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (!convert_unary_parse_types(symbol, &ctx.src_type, &ctx.dst_type) ||
        ctx.src_type == PACC_CONVERT_SCALAR_UNSUPPORTED ||
        ctx.dst_type == PACC_CONVERT_SCALAR_UNSUPPORTED) {
        return 1;
    }
    ctx.src = (const void *)(uintptr_t)kernel_cell_u64(args, argc, 0);
    ctx.dst = (void *)(uintptr_t)kernel_cell_u64(args, argc, 1);
    ctx.ne00 = kernel_cell_i64(args, argc, 2);
    ctx.ne01 = kernel_cell_i64(args, argc, 3);
    ctx.ne0203 = kernel_cell_i64(args, argc, 4);
    struct PaccUint3 block_nums = kernel_cell_u3(args, argc, 5);
    ctx.ne02 = block_nums.z ? (int64_t)block_nums.z : 1;
    ctx.s01 = kernel_cell_i64(args, argc, 6);
    ctx.s02 = kernel_cell_i64(args, argc, 7);
    ctx.s03 = kernel_cell_i64(args, argc, 8);

    if (!ctx.src || !ctx.dst || ctx.ne00 < 0 || ctx.ne01 < 0 || ctx.ne0203 < 0 ||
        ctx.ne02 <= 0 || ctx.s01 < 0 || ctx.s02 < 0 || ctx.s03 < 0) {
        return -1;
    }
    uint64_t total = (uint64_t)ctx.ne00 * (uint64_t)ctx.ne01 * (uint64_t)ctx.ne0203;
    if (total == 0) return 0;

    unsigned workers = kernel_worker_threads(total);
    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct ConvertUnaryNativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (total + workers - 1u) / workers;

    trace_msg("native convert_unary: symbol=%s total=%" PRIu64
              " ne=(%" PRId64 ",%" PRId64 ",%" PRId64 ") ne02=%" PRId64
              " strides=(%" PRId64 ",%" PRId64 ",%" PRId64 ") workers=%u",
              symbol, total, ctx.ne00, ctx.ne01, ctx.ne0203, ctx.ne02,
              ctx.s01, ctx.s02, ctx.s03, workers);
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= total) break;
        if (end > total) end = total;
        worker[i].ctx = &ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, convert_unary_native_worker_main, &worker[i]) != 0) {
            log_msg("native convert_unary failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    return status;
}

static float rope_native_load(const struct RopeNormNativeCtx *ctx, const void *base, int64_t index) {
    if (ctx->x_type == PACC_ROPE_SCALAR_F16) {
        return pacc_f16_to_f32(((const uint16_t *)base)[index]);
    }
    return ((const float *)base)[index];
}

static void rope_native_store_pair(const struct RopeNormNativeCtx *ctx,
                                   int64_t index, float x0, float x1) {
    if (ctx->dst_type == PACC_ROPE_SCALAR_F16) {
        uint16_t *dst = (uint16_t *)ctx->dst;
        dst[index + 0] = pacc_f32_to_f16(x0);
        dst[index + 1] = pacc_f32_to_f16(x1);
    } else {
        float *dst = (float *)ctx->dst;
        dst[index + 0] = x0;
        dst[index + 1] = x1;
    }
}

static void rope_native_store_one(const struct RopeNormNativeCtx *ctx,
                                  int64_t index, float value) {
    if (ctx->dst_type == PACC_ROPE_SCALAR_F16) {
        ((uint16_t *)ctx->dst)[index] = pacc_f32_to_f16(value);
    } else {
        ((float *)ctx->dst)[index] = value;
    }
}

static float rope_native_ramp(float low, float high, int32_t i0) {
    float denom = high - low;
    if (denom < 0.001f) denom = 0.001f;
    float y = ((float)(i0 / 2) - low) / denom;
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return 1.0f - y;
}

static void rope_native_yarn(const struct RopeNormNativeCtx *ctx,
                             float theta_extrap, int32_t i0,
                             float *cos_theta, float *sin_theta) {
    float theta_interp = ctx->freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = ctx->attn_factor;
    if (ctx->ext_factor != 0.0f) {
        float ramp_mix = rope_native_ramp(ctx->corr_low, ctx->corr_high, i0) * ctx->ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / ctx->freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
    if (!ctx->forward) {
        *sin_theta *= -1.0f;
    }
}

static bool rope_norm_parse_template(const char *symbol,
                                     bool *forward_out,
                                     bool *has_ff_out,
                                     enum PaccRopeScalarType *x_type_out,
                                     enum PaccRopeScalarType *dst_type_out) {
    const char *p = symbol ? strstr(symbol, "rope_norm") : NULL;
    if (!p) return false;
    p = strstr(p, "ILb");
    if (!p || (p[3] != '0' && p[3] != '1')) return false;
    const char *q = strstr(p + 4, "ELb");
    if (!q || (q[3] != '0' && q[3] != '1')) return false;

    const char *types = q + 4;
    if (*types == 'E') types++;

    enum PaccRopeScalarType x_type = PACC_ROPE_SCALAR_UNSUPPORTED;
    enum PaccRopeScalarType dst_type = PACC_ROPE_SCALAR_UNSUPPORTED;
    if (types[0] == 'f') {
        x_type = PACC_ROPE_SCALAR_F32;
        types += 1;
    } else if (strncmp(types, "6__half", 7) == 0) {
        x_type = PACC_ROPE_SCALAR_F16;
        types += 7;
    } else {
        return false;
    }

    if (types[0] == 'f') {
        dst_type = PACC_ROPE_SCALAR_F32;
    } else if (strncmp(types, "6__half", 7) == 0 || strncmp(types, "S0_", 3) == 0) {
        dst_type = PACC_ROPE_SCALAR_F16;
    } else {
        return false;
    }

    *forward_out = p[3] == '1';
    *has_ff_out = q[3] == '1';
    *x_type_out = x_type;
    *dst_type_out = dst_type;
    return true;
}

static bool rope_multi_parse_template(const char *symbol,
                                      bool *forward_out,
                                      bool *has_ff_out,
                                      enum PaccRopeScalarType *type_out) {
    const char *p = symbol ? strstr(symbol, "rope_multi") : NULL;
    if (!p) return false;
    p = strstr(p, "ILb");
    if (!p || (p[3] != '0' && p[3] != '1')) return false;
    const char *q = strstr(p + 4, "ELb");
    if (!q || (q[3] != '0' && q[3] != '1')) return false;

    const char *types = q + 4;
    if (*types == 'E') types++;

    enum PaccRopeScalarType type = PACC_ROPE_SCALAR_UNSUPPORTED;
    if (types[0] == 'f') {
        type = PACC_ROPE_SCALAR_F32;
    } else if (strncmp(types, "6__half", 7) == 0) {
        type = PACC_ROPE_SCALAR_F16;
    } else {
        return false;
    }

    *forward_out = p[3] == '1';
    *has_ff_out = q[3] == '1';
    *type_out = type;
    return true;
}

static void *rope_multi_native_worker_main(void *opaque) {
    struct RopeNormNativeWorker *worker = (struct RopeNormNativeWorker *)opaque;
    const struct RopeNormNativeCtx *ctx = worker->ctx;
    const int32_t sect_dims =
        ctx->sections[0] + ctx->sections[1] + ctx->sections[2] + ctx->sections[3];
    const int32_t sec_w = ctx->sections[0] + ctx->sections[1];

    if (sect_dims <= 0 || ctx->n_dims <= 0) {
        worker->status = -1;
        return NULL;
    }

    for (uint64_t idx = worker->begin; idx < worker->end; idx++) {
        uint32_t pair = (uint32_t)(idx % ctx->pair_count);
        uint32_t row_dst = (uint32_t)(idx / ctx->pair_count);
        int32_t i0 = (int32_t)(pair * 2u);
        if (i0 >= ctx->ne00) continue;

        uint32_t ne01 = (uint32_t)ctx->ne01;
        uint32_t ne02 = (uint32_t)ctx->ne02;
        uint32_t i3 = row_dst / (ne01 * ne02);
        uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
        uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

        int64_t idst = (int64_t)(i0 / 2) +
                       (int64_t)i1 * ctx->s1 +
                       (int64_t)i2 * ctx->s2 +
                       (int64_t)i3 * ctx->s3;
        int64_t ix = (int64_t)(i0 / 2) +
                     (int64_t)i1 * ctx->s01 +
                     (int64_t)i2 * ctx->s02 +
                     (int64_t)i3 * ctx->s03;

        if (i0 >= ctx->n_dims) {
            int64_t src0 = ix + i0 / 2;
            int64_t src1 = src0 + 1;
            int64_t dst0 = idst + i0 / 2;
            int64_t dst1 = dst0 + 1;
            if (src0 < 0 || dst0 < 0 ||
                (ctx->x_elems != 0 && (uint64_t)src1 >= ctx->x_elems) ||
                (ctx->dst_elems != 0 && (uint64_t)dst1 >= ctx->dst_elems)) {
                worker->status = -1;
                return NULL;
            }
            rope_native_store_one(ctx, dst0, rope_native_load(ctx, ctx->x, src0));
            rope_native_store_one(ctx, dst1, rope_native_load(ctx, ctx->x, src1));
            continue;
        }

        if (!ctx->pos || (ctx->has_ff && !ctx->freq_factors)) {
            worker->status = -1;
            return NULL;
        }

        int32_t sector = (i0 / 2) % sect_dims;
        int32_t axis = 0;
        if (ctx->is_imrope) {
            if (sector % 3 == 1 && sector < 3 * ctx->sections[1]) {
                axis = 1;
            } else if (sector % 3 == 2 && sector < 3 * ctx->sections[2]) {
                axis = 2;
            } else if (sector % 3 == 0 && sector < 3 * ctx->sections[0]) {
                axis = 0;
            } else {
                axis = 3;
            }
        } else {
            if (sector < ctx->sections[0]) {
                axis = 0;
            } else if (sector < sec_w) {
                axis = 1;
            } else if (sector < sec_w + ctx->sections[2]) {
                axis = 2;
            } else {
                axis = 3;
            }
        }

        uint64_t pos_index = (uint64_t)i2 + (uint64_t)ne02 * (uint64_t)axis;
        if (ctx->pos_elems != 0 && pos_index >= ctx->pos_elems) {
            worker->status = -1;
            return NULL;
        }
        if (ctx->has_ff && ctx->freq_elems != 0 &&
            (uint64_t)(i0 / 2) >= ctx->freq_elems) {
            worker->status = -1;
            return NULL;
        }

        int64_t src0 = ix;
        int64_t src1 = ix + ctx->n_dims / 2;
        int64_t dst0 = idst;
        int64_t dst1 = idst + ctx->n_dims / 2;
        if (src0 < 0 || dst0 < 0 ||
            (ctx->x_elems != 0 && (uint64_t)src1 >= ctx->x_elems) ||
            (ctx->dst_elems != 0 && (uint64_t)dst1 >= ctx->dst_elems)) {
            worker->status = -1;
            return NULL;
        }

        float theta_base =
            (float)ctx->pos[pos_index] * powf(ctx->theta_scale, (float)i0 * 0.5f);
        float freq_factor = ctx->has_ff ? ctx->freq_factors[i0 / 2] : 1.0f;
        float cos_theta = 1.0f;
        float sin_theta = 0.0f;
        rope_native_yarn(ctx, theta_base / freq_factor, i0, &cos_theta, &sin_theta);

        float x0 = rope_native_load(ctx, ctx->x, src0);
        float x1 = rope_native_load(ctx, ctx->x, src1);
        rope_native_store_one(ctx, dst0, x0 * cos_theta - x1 * sin_theta);
        rope_native_store_one(ctx, dst1, x0 * sin_theta + x1 * cos_theta);
    }

    worker->status = 0;
    return NULL;
}

static void *rope_norm_native_worker_main(void *opaque) {
    struct RopeNormNativeWorker *worker = (struct RopeNormNativeWorker *)opaque;
    const struct RopeNormNativeCtx *ctx = worker->ctx;

    for (uint64_t idx = worker->begin; idx < worker->end; idx++) {
        uint32_t pair = (uint32_t)(idx % ctx->pair_count);
        uint32_t row_dst = (uint32_t)(idx / ctx->pair_count);
        int32_t i0 = (int32_t)(pair * 2u);
        if (i0 >= ctx->ne00) continue;

        uint32_t ne01 = (uint32_t)ctx->ne01;
        uint32_t ne02 = (uint32_t)ctx->ne02;
        uint32_t i3 = row_dst / (ne01 * ne02);
        uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
        uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

        int64_t idst = (int64_t)i0 +
                       (int64_t)i1 * ctx->s1 +
                       (int64_t)i2 * ctx->s2 +
                       (int64_t)i3 * ctx->s3;
        int64_t ix = (int64_t)i0 +
                     (int64_t)i1 * ctx->s01 +
                     (int64_t)i2 * ctx->s02 +
                     (int64_t)i3 * ctx->s03;

        if (ctx->set_rows_stride != 0) {
            if (!ctx->row_indices) {
                worker->status = -1;
                return NULL;
            }
            if (ctx->row_indices_elems != 0 && (uint64_t)i2 >= ctx->row_indices_elems) {
                worker->status = -1;
                return NULL;
            }
            idst = (int64_t)i1 * ctx->s1 + i0;
            idst += ctx->row_indices[i2] * (int64_t)ctx->set_rows_stride;
        }

        if (ix < 0 || idst < 0 ||
            (ctx->x_elems != 0 && (uint64_t)(ix + 1) >= ctx->x_elems) ||
            (ctx->dst_elems != 0 && (uint64_t)(idst + 1) >= ctx->dst_elems)) {
            worker->status = -1;
            return NULL;
        }

        float x0 = rope_native_load(ctx, ctx->x, ix + 0);
        float x1 = rope_native_load(ctx, ctx->x, ix + 1);
        if (i0 >= ctx->n_dims) {
            rope_native_store_pair(ctx, idst, x0, x1);
            continue;
        }

        if (!ctx->pos || (ctx->has_ff && !ctx->freq_factors)) {
            worker->status = -1;
            return NULL;
        }
        if (ctx->pos_elems != 0 && (uint64_t)i2 >= ctx->pos_elems) {
            worker->status = -1;
            return NULL;
        }
        float theta_base = (float)ctx->pos[i2] * powf(ctx->theta_scale, (float)i0 * 0.5f);
        if (ctx->has_ff && ctx->freq_elems != 0 && (uint64_t)(i0 / 2) >= ctx->freq_elems) {
            worker->status = -1;
            return NULL;
        }
        float freq_factor = ctx->has_ff ? ctx->freq_factors[i0 / 2] : 1.0f;
        float cos_theta = 1.0f;
        float sin_theta = 0.0f;
        rope_native_yarn(ctx, theta_base / freq_factor, i0, &cos_theta, &sin_theta);
        rope_native_store_pair(ctx, idst,
                               x0 * cos_theta - x1 * sin_theta,
                               x0 * sin_theta + x1 * cos_theta);
    }

    worker->status = 0;
    return NULL;
}

static int invoke_kernel_rope_norm_native(const char *symbol,
                                          const uint64_t *argv,
                                          const struct PaccJobImage *job,
                                          size_t argc) {
    uint64_t t0 = monotonic_us();
    if (!symbol || !strstr(symbol, "rope_norm")) return 1;
    if (!job || argc < 21) return -1;

    struct RopeNormNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (!rope_norm_parse_template(symbol, &ctx.forward, &ctx.has_ff,
                                  &ctx.x_type, &ctx.dst_type)) {
        return 1;
    }

    ctx.x = (const void *)(uintptr_t)kernel_cell_u64(argv, argc, 0);
    ctx.dst = (void *)(uintptr_t)kernel_cell_u64(argv, argc, 1);
    ctx.ne00 = kernel_cell_i32(argv, argc, 2);
    ctx.ne01 = kernel_cell_i32(argv, argc, 3);
    ctx.ne02 = kernel_cell_i32(argv, argc, 4);
    ctx.s01 = kernel_cell_i32(argv, argc, 5);
    ctx.s02 = kernel_cell_i32(argv, argc, 6);
    ctx.s03 = kernel_cell_i32(argv, argc, 7);
    ctx.s1 = kernel_cell_i32(argv, argc, 8);
    ctx.s2 = kernel_cell_i32(argv, argc, 9);
    ctx.s3 = kernel_cell_i32(argv, argc, 10);
    ctx.n_dims = kernel_cell_i32(argv, argc, 11);
    ctx.pos = (const int32_t *)(uintptr_t)kernel_cell_u64(argv, argc, 12);
    ctx.freq_scale = kernel_cell_f32(argv, argc, 13);
    ctx.ext_factor = kernel_cell_f32(argv, argc, 14);
    ctx.attn_factor = kernel_cell_f32(argv, argc, 15);
    {
        uint64_t corr = kernel_cell_u64(argv, argc, 16);
        uint32_t lo = (uint32_t)corr;
        uint32_t hi = (uint32_t)(corr >> 32);
        memcpy(&ctx.corr_low, &lo, sizeof(ctx.corr_low));
        memcpy(&ctx.corr_high, &hi, sizeof(ctx.corr_high));
    }
    ctx.theta_scale = kernel_cell_f32(argv, argc, 17);
    ctx.freq_factors = (const float *)(uintptr_t)kernel_cell_u64(argv, argc, 18);
    ctx.row_indices = (const int64_t *)(uintptr_t)kernel_cell_u64(argv, argc, 19);
    ctx.set_rows_stride = kernel_cell_i32(argv, argc, 20);
    ctx.row_count = pacc_nonzero_dim(job->header.grid_x) *
                    pacc_nonzero_dim(job->header.block_x);
    ctx.pair_count = pacc_nonzero_dim(job->header.grid_y) *
                     pacc_nonzero_dim(job->header.block_y);

    uint64_t x_bytes = kernel_binding_size_for_arg(job, 0);
    uint64_t dst_bytes = kernel_binding_size_for_arg(job, 1);
    uint64_t pos_bytes = kernel_binding_size_for_arg(job, 12);
    uint64_t freq_bytes = kernel_binding_size_for_arg(job, 18);
    uint64_t rows_bytes = kernel_binding_size_for_arg(job, 19);
    uint64_t x_elem_bytes = ctx.x_type == PACC_ROPE_SCALAR_F16 ? 2u : 4u;
    uint64_t dst_elem_bytes = ctx.dst_type == PACC_ROPE_SCALAR_F16 ? 2u : 4u;
    if (x_bytes != 0) ctx.x_elems = x_bytes / x_elem_bytes;
    if (dst_bytes != 0) ctx.dst_elems = dst_bytes / dst_elem_bytes;
    if (pos_bytes != 0) ctx.pos_elems = pos_bytes / sizeof(int32_t);
    if (freq_bytes != 0) ctx.freq_elems = freq_bytes / sizeof(float);
    if (rows_bytes != 0) ctx.row_indices_elems = rows_bytes / sizeof(int64_t);

    if (!ctx.x || !ctx.dst || ctx.ne00 <= 0 || ctx.ne01 <= 0 ||
        ctx.ne02 <= 0 || ctx.pair_count == 0 || ctx.row_count == 0 ||
        ctx.x_elems == 0 || ctx.dst_elems == 0) {
        log_msg("native rope_norm invalid args: x=%p/%" PRIu64 "B dst=%p/%" PRIu64
                "B ne=(%d,%d,%d) rows=%u pairs=%u",
                ctx.x, x_bytes, ctx.dst, dst_bytes,
                ctx.ne00, ctx.ne01, ctx.ne02, ctx.row_count, ctx.pair_count);
        return -1;
    }

    void *local_x = NULL;
    void *local_dst = NULL;
    int32_t *local_pos = NULL;
    float *local_freq = NULL;
    int64_t *local_rows = NULL;
    void *dst_writeback = ctx.dst;
    uint64_t local_max = parse_env_u64_default("PACC_JOBD_ROPE_LOCAL_MAX_BYTES",
                                               1u << 20);
    if (local_max != 0) {
        if (x_bytes != 0 && x_bytes <= local_max) {
            if (native_stage_read((uint64_t)(uintptr_t)ctx.x,
                                  (size_t)x_bytes, &local_x)) {
                ctx.x = local_x;
            }
        }
        if (dst_bytes != 0 && dst_bytes <= local_max) {
            if (native_stage_read((uint64_t)(uintptr_t)ctx.dst,
                                  (size_t)dst_bytes, &local_dst)) {
                ctx.dst = local_dst;
            }
        }
        if (ctx.pos && pos_bytes != 0 && pos_bytes <= local_max) {
            void *stage = NULL;
            if (native_stage_read((uint64_t)(uintptr_t)ctx.pos,
                                  (size_t)pos_bytes, &stage)) {
                local_pos = (int32_t *)stage;
                ctx.pos = local_pos;
            }
        }
        if (ctx.freq_factors && freq_bytes != 0 && freq_bytes <= local_max) {
            void *stage = NULL;
            if (native_stage_read((uint64_t)(uintptr_t)ctx.freq_factors,
                                  (size_t)freq_bytes, &stage)) {
                local_freq = (float *)stage;
                ctx.freq_factors = local_freq;
            }
        }
        if (ctx.row_indices && rows_bytes != 0 && rows_bytes <= local_max) {
            void *stage = NULL;
            if (native_stage_read((uint64_t)(uintptr_t)ctx.row_indices,
                                  (size_t)rows_bytes, &stage)) {
                local_rows = (int64_t *)stage;
                ctx.row_indices = local_rows;
            }
        }
    }

    uint64_t work_items = (uint64_t)ctx.row_count * ctx.pair_count;
    unsigned workers = kernel_worker_threads(work_items);
    trace_msg("native rope_norm %s rows=%u pairs=%u work=%" PRIu64
              " workers=%u fwd=%u ff=%u xtype=%d dtype=%d x_elems=%" PRIu64
              " dst_elems=%" PRIu64,
              symbol, ctx.row_count, ctx.pair_count, work_items, workers,
              ctx.forward ? 1U : 0U, ctx.has_ff ? 1U : 0U,
              ctx.x_type, ctx.dst_type, ctx.x_elems, ctx.dst_elems);

    if (workers <= 1 || work_items <= 1) {
        struct RopeNormNativeWorker worker = {
            .ctx = &ctx,
            .begin = 0,
            .end = work_items,
            .status = 0,
        };
        rope_norm_native_worker_main(&worker);
        int worker_status = worker.status;
        if (worker_status == 0 && local_dst) {
            uint64_t dst_bytes = kernel_binding_size_for_arg(job, 1);
            (void)native_stage_write((uint64_t)(uintptr_t)dst_writeback,
                                     local_dst, (size_t)dst_bytes);
        }
        trace_msg("native rope_norm done %s status=%d elapsed_us=%" PRIu64,
                  symbol, worker_status, monotonic_us() - t0);
        free(local_x);
        free(local_dst);
        free(local_pos);
        free(local_freq);
        free(local_rows);
        return worker_status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct RopeNormNativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (work_items + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= work_items) break;
        if (end > work_items) end = work_items;
        worker[i].ctx = &ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, rope_norm_native_worker_main, &worker[i]) != 0) {
            log_msg("native rope_norm failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            free(local_x);
            free(local_dst);
            free(local_pos);
            free(local_freq);
            free(local_rows);
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    if (status == 0 && local_dst) {
        uint64_t dst_bytes = kernel_binding_size_for_arg(job, 1);
        (void)native_stage_write((uint64_t)(uintptr_t)dst_writeback,
                                 local_dst, (size_t)dst_bytes);
    }
    trace_msg("native rope_norm done %s status=%d elapsed_us=%" PRIu64,
              symbol, status, monotonic_us() - t0);
    free(local_x);
    free(local_dst);
    free(local_pos);
    free(local_freq);
    free(local_rows);
    return status;
}

static int invoke_kernel_rope_multi_native(const char *symbol,
                                           const uint64_t *argv,
                                           const struct PaccJobImage *job,
                                           size_t argc) {
    uint64_t t0 = monotonic_us();
    if (!symbol || !strstr(symbol, "rope_multi")) return 1;
    if (!job || argc < 21) return -1;

    struct RopeNormNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (!rope_multi_parse_template(symbol, &ctx.forward, &ctx.has_ff,
                                   &ctx.x_type)) {
        return 1;
    }
    ctx.dst_type = ctx.x_type;

    ctx.x = (const void *)(uintptr_t)kernel_cell_u64(argv, argc, 0);
    ctx.dst = (void *)(uintptr_t)kernel_cell_u64(argv, argc, 1);
    ctx.ne00 = kernel_cell_i32(argv, argc, 2);
    ctx.ne01 = kernel_cell_i32(argv, argc, 3);
    ctx.ne02 = kernel_cell_i32(argv, argc, 4);
    ctx.s01 = kernel_cell_i32(argv, argc, 5);
    ctx.s02 = kernel_cell_i32(argv, argc, 6);
    ctx.s03 = kernel_cell_i32(argv, argc, 7);
    ctx.s1 = kernel_cell_i32(argv, argc, 8);
    ctx.s2 = kernel_cell_i32(argv, argc, 9);
    ctx.s3 = kernel_cell_i32(argv, argc, 10);
    ctx.n_dims = kernel_cell_i32(argv, argc, 11);
    ctx.pos = (const int32_t *)(uintptr_t)kernel_cell_u64(argv, argc, 12);
    ctx.freq_scale = kernel_cell_f32(argv, argc, 13);
    ctx.ext_factor = kernel_cell_f32(argv, argc, 14);
    ctx.attn_factor = kernel_cell_f32(argv, argc, 15);
    {
        uint64_t corr = kernel_cell_u64(argv, argc, 16);
        uint32_t lo = (uint32_t)corr;
        uint32_t hi = (uint32_t)(corr >> 32);
        memcpy(&ctx.corr_low, &lo, sizeof(ctx.corr_low));
        memcpy(&ctx.corr_high, &hi, sizeof(ctx.corr_high));
    }
    ctx.theta_scale = kernel_cell_f32(argv, argc, 17);
    ctx.freq_factors = (const float *)(uintptr_t)kernel_cell_u64(argv, argc, 18);
    kernel_cell_i32x4(argv, argc, 19, ctx.sections);
    ctx.is_imrope = kernel_cell_i32(argv, argc, 20) != 0;
    ctx.row_count = pacc_nonzero_dim(job->header.grid_x) *
                    pacc_nonzero_dim(job->header.block_x);
    ctx.pair_count = pacc_nonzero_dim(job->header.grid_y) *
                     pacc_nonzero_dim(job->header.block_y);

    uint64_t x_bytes = kernel_binding_size_for_arg(job, 0);
    uint64_t dst_bytes = kernel_binding_size_for_arg(job, 1);
    uint64_t pos_bytes = kernel_binding_size_for_arg(job, 12);
    uint64_t freq_bytes = kernel_binding_size_for_arg(job, 18);
    uint64_t elem_bytes = ctx.x_type == PACC_ROPE_SCALAR_F16 ? 2u : 4u;
    if (x_bytes != 0) ctx.x_elems = x_bytes / elem_bytes;
    if (dst_bytes != 0) ctx.dst_elems = dst_bytes / elem_bytes;
    if (pos_bytes != 0) ctx.pos_elems = pos_bytes / sizeof(int32_t);
    if (freq_bytes != 0) ctx.freq_elems = freq_bytes / sizeof(float);

    int32_t sect_dims = ctx.sections[0] + ctx.sections[1] +
                        ctx.sections[2] + ctx.sections[3];
    if (!ctx.x || !ctx.dst || !ctx.pos || ctx.ne00 <= 0 ||
        ctx.ne01 <= 0 || ctx.ne02 <= 0 || ctx.n_dims <= 0 ||
        ctx.pair_count == 0 || ctx.row_count == 0 ||
        ctx.x_elems == 0 || ctx.dst_elems == 0 || sect_dims <= 0) {
        log_msg("native rope_multi invalid args: x=%p/%" PRIu64
                "B dst=%p/%" PRIu64 "B pos=%p/%" PRIu64
                "B ne=(%d,%d,%d) n_dims=%d sections=(%d,%d,%d,%d)"
                " rows=%u pairs=%u",
                ctx.x, x_bytes, ctx.dst, dst_bytes, ctx.pos, pos_bytes,
                ctx.ne00, ctx.ne01, ctx.ne02, ctx.n_dims,
                ctx.sections[0], ctx.sections[1],
                ctx.sections[2], ctx.sections[3],
                ctx.row_count, ctx.pair_count);
        return -1;
    }

    void *local_x = NULL;
    void *local_dst = NULL;
    int32_t *local_pos = NULL;
    float *local_freq = NULL;
    void *dst_writeback = ctx.dst;
    uint64_t local_max = parse_env_u64_default("PACC_JOBD_ROPE_LOCAL_MAX_BYTES",
                                               1u << 20);
    if (local_max != 0) {
        if (x_bytes != 0 && x_bytes <= local_max) {
            if (native_stage_read((uint64_t)(uintptr_t)ctx.x,
                                  (size_t)x_bytes, &local_x)) {
                ctx.x = local_x;
            }
        }
        if (dst_bytes != 0 && dst_bytes <= local_max) {
            if (native_stage_read((uint64_t)(uintptr_t)ctx.dst,
                                  (size_t)dst_bytes, &local_dst)) {
                ctx.dst = local_dst;
            }
        }
        if (pos_bytes != 0 && pos_bytes <= local_max) {
            void *stage = NULL;
            if (native_stage_read((uint64_t)(uintptr_t)ctx.pos,
                                  (size_t)pos_bytes, &stage)) {
                local_pos = (int32_t *)stage;
                ctx.pos = local_pos;
            }
        }
        if (ctx.freq_factors && freq_bytes != 0 && freq_bytes <= local_max) {
            void *stage = NULL;
            if (native_stage_read((uint64_t)(uintptr_t)ctx.freq_factors,
                                  (size_t)freq_bytes, &stage)) {
                local_freq = (float *)stage;
                ctx.freq_factors = local_freq;
            }
        }
    }

    uint64_t work_items = (uint64_t)ctx.row_count * ctx.pair_count;
    unsigned workers = kernel_worker_threads(work_items);
    trace_msg("native rope_multi %s rows=%u pairs=%u work=%" PRIu64
              " workers=%u fwd=%u ff=%u type=%d imrope=%u"
              " sections=(%d,%d,%d,%d)",
              symbol, ctx.row_count, ctx.pair_count, work_items, workers,
              ctx.forward ? 1U : 0U, ctx.has_ff ? 1U : 0U,
              ctx.x_type, ctx.is_imrope ? 1U : 0U,
              ctx.sections[0], ctx.sections[1],
              ctx.sections[2], ctx.sections[3]);

    if (workers <= 1 || work_items <= 1) {
        struct RopeNormNativeWorker worker = {
            .ctx = &ctx,
            .begin = 0,
            .end = work_items,
            .status = 0,
        };
        rope_multi_native_worker_main(&worker);
        int worker_status = worker.status;
        if (worker_status == 0 && local_dst) {
            (void)native_stage_write((uint64_t)(uintptr_t)dst_writeback,
                                     local_dst, (size_t)dst_bytes);
        }
        trace_msg("native rope_multi done %s status=%d elapsed_us=%" PRIu64,
                  symbol, worker_status, monotonic_us() - t0);
        free(local_x);
        free(local_dst);
        free(local_pos);
        free(local_freq);
        return worker_status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct RopeNormNativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (work_items + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= work_items) break;
        if (end > work_items) end = work_items;
        worker[i].ctx = &ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, rope_multi_native_worker_main, &worker[i]) != 0) {
            log_msg("native rope_multi failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            free(local_x);
            free(local_dst);
            free(local_pos);
            free(local_freq);
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    if (status == 0 && local_dst) {
        (void)native_stage_write((uint64_t)(uintptr_t)dst_writeback,
                                 local_dst, (size_t)dst_bytes);
    }
    trace_msg("native rope_multi done %s status=%d elapsed_us=%" PRIu64,
              symbol, status, monotonic_us() - t0);
    free(local_x);
    free(local_dst);
    free(local_pos);
    free(local_freq);
    return status;
}

enum PaccSetRowsDstType {
    PACC_SET_ROWS_DST_UNSUPPORTED = 0,
    PACC_SET_ROWS_DST_F32,
    PACC_SET_ROWS_DST_F16,
    PACC_SET_ROWS_DST_BF16,
};

struct SetRowsNativeCtx {
    const float *src0;
    const void *src1;
    void *dst;
    int64_t ne_total;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne11;
    int64_t ne12;
    int64_t s01;
    int64_t s02;
    int64_t s03;
    int64_t s10;
    int64_t s11;
    int64_t s12;
    int64_t s1;
    int64_t s2;
    int64_t s3;
    bool idx_i64;
    enum PaccSetRowsDstType dst_type;
};

struct SetRowsNativeWorker {
    const struct SetRowsNativeCtx *ctx;
    uint64_t begin;
    uint64_t end;
    int status;
};

static uint32_t set_rows_fastdiv_dim(struct PaccUint3 v) {
    return v.z;
}

static bool set_rows_parse_template(const char *symbol,
                                    bool *idx_i64_out,
                                    enum PaccSetRowsDstType *dst_type_out) {
    const char *p = symbol ? strstr(symbol, "k_set_rows") : NULL;
    if (!p) return false;
    p = strchr(p, 'I');
    if (!p || p[1] != 'f') return false;
    if (p[2] == 'l') {
        *idx_i64_out = true;
    } else if (p[2] == 'i') {
        *idx_i64_out = false;
    } else {
        return false;
    }

    const char *dst = p + 3;
    if (dst[0] == 'f') {
        *dst_type_out = PACC_SET_ROWS_DST_F32;
    } else if (strncmp(dst, "6__half", 7) == 0) {
        *dst_type_out = PACC_SET_ROWS_DST_F16;
    } else if (strstr(dst, "bfloat16")) {
        *dst_type_out = PACC_SET_ROWS_DST_BF16;
    } else {
        return false;
    }
    return true;
}

static int64_t set_rows_load_index(const struct SetRowsNativeCtx *ctx, int64_t offset) {
    if (ctx->idx_i64) {
        return ((const int64_t *)ctx->src1)[offset];
    }
    return ((const int32_t *)ctx->src1)[offset];
}

static void set_rows_store(const struct SetRowsNativeCtx *ctx, int64_t offset, float value) {
    switch (ctx->dst_type) {
    case PACC_SET_ROWS_DST_F32:
        ((float *)ctx->dst)[offset] = value;
        break;
    case PACC_SET_ROWS_DST_F16:
        ((uint16_t *)ctx->dst)[offset] = pacc_f32_to_f16(value);
        break;
    case PACC_SET_ROWS_DST_BF16:
        ((uint16_t *)ctx->dst)[offset] = f32_to_bf16(value);
        break;
    default:
        break;
    }
}

static void *set_rows_native_worker_main(void *opaque) {
    struct SetRowsNativeWorker *worker = (struct SetRowsNativeWorker *)opaque;
    const struct SetRowsNativeCtx *ctx = worker->ctx;

    for (uint64_t idx = worker->begin; idx < worker->end; idx++) {
        uint64_t tmp = idx;
        int64_t i00 = (int64_t)(tmp % (uint64_t)ctx->ne00);
        tmp /= (uint64_t)ctx->ne00;
        int64_t i01 = (int64_t)(tmp % (uint64_t)ctx->ne01);
        tmp /= (uint64_t)ctx->ne01;
        int64_t i02 = (int64_t)(tmp % (uint64_t)ctx->ne02);
        int64_t i03 = (int64_t)(tmp / (uint64_t)ctx->ne02);
        int64_t i12 = i03 % ctx->ne12;
        int64_t i11 = i02 % ctx->ne11;
        int64_t i10 = i01;

        int64_t dst_row = set_rows_load_index(
            ctx, i10 * ctx->s10 + i11 * ctx->s11 + i12 * ctx->s12);
        int64_t src_off = i01 * ctx->s01 + i02 * ctx->s02 + i03 * ctx->s03 + i00;
        int64_t dst_off = dst_row * ctx->s1 + i02 * ctx->s2 + i03 * ctx->s3 + i00;
        set_rows_store(ctx, dst_off, ctx->src0[src_off]);
    }

    worker->status = 0;
    return NULL;
}

static int invoke_kernel_set_rows_native(const char *symbol,
                                         const uint64_t *argv,
                                         const struct PaccJobImage *job,
                                         size_t argc) {
    (void)job;
    if (!symbol || !strstr(symbol, "k_set_rows")) return 1;
    if (argc < 22) return -1;

    struct SetRowsNativeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (!set_rows_parse_template(symbol, &ctx.idx_i64, &ctx.dst_type)) {
        log_msg("native set_rows parse miss: symbol=%s argc=%zu", symbol, argc);
        return 1;
    }

    ctx.src0 = (const float *)(uintptr_t)kernel_cell_u64(argv, argc, 0);
    ctx.src1 = (const void *)(uintptr_t)kernel_cell_u64(argv, argc, 1);
    ctx.dst = (void *)(uintptr_t)kernel_cell_u64(argv, argc, 2);
    ctx.ne_total = kernel_cell_i64(argv, argc, 3);
    ctx.ne11 = kernel_cell_i64(argv, argc, 5);
    ctx.ne12 = kernel_cell_i64(argv, argc, 6);
    ctx.s01 = kernel_cell_i64(argv, argc, 8);
    ctx.s02 = kernel_cell_i64(argv, argc, 9);
    ctx.s03 = kernel_cell_i64(argv, argc, 10);
    ctx.s10 = kernel_cell_i64(argv, argc, 11);
    ctx.s11 = kernel_cell_i64(argv, argc, 12);
    ctx.s12 = kernel_cell_i64(argv, argc, 13);
    ctx.s1 = kernel_cell_i64(argv, argc, 14);
    ctx.s2 = kernel_cell_i64(argv, argc, 15);
    ctx.s3 = kernel_cell_i64(argv, argc, 16);
    ctx.ne00 = (int64_t)set_rows_fastdiv_dim(kernel_cell_u3(argv, argc, 17));
    ctx.ne01 = (int64_t)set_rows_fastdiv_dim(kernel_cell_u3(argv, argc, 18));
    ctx.ne02 = (int64_t)set_rows_fastdiv_dim(kernel_cell_u3(argv, argc, 19));
    {
        int64_t ne11_fd = (int64_t)set_rows_fastdiv_dim(kernel_cell_u3(argv, argc, 20));
        int64_t ne12_fd = (int64_t)set_rows_fastdiv_dim(kernel_cell_u3(argv, argc, 21));
        if (ne11_fd > 0) ctx.ne11 = ne11_fd;
        if (ne12_fd > 0) ctx.ne12 = ne12_fd;
    }

    if (!ctx.src0 || !ctx.src1 || !ctx.dst || ctx.ne_total <= 0 ||
        ctx.ne00 <= 0 || ctx.ne01 <= 0 || ctx.ne02 <= 0 ||
        ctx.ne11 <= 0 || ctx.ne12 <= 0) {
        log_msg("native set_rows invalid: symbol=%s argc=%zu src0=%p src1=%p dst=%p total=%" PRId64
                " ne00=%" PRId64 " ne01=%" PRId64 " ne02=%" PRId64
                " ne11=%" PRId64 " ne12=%" PRId64,
                symbol, argc, (const void *)ctx.src0, ctx.src1, ctx.dst,
                ctx.ne_total, ctx.ne00, ctx.ne01, ctx.ne02, ctx.ne11, ctx.ne12);
        return -1;
    }

    uint64_t work_items = (uint64_t)ctx.ne_total;
    unsigned workers = kernel_worker_threads(work_items);
    log_msg("native set_rows enter: symbol=%s total=%" PRIu64
            " src0=%p src1=%p dst=%p ne=(%" PRId64 ",%" PRId64 ",%" PRId64 ")"
            " idx64=%u dst_type=%d stride src=(%" PRId64 ",%" PRId64 ",%" PRId64 ")"
            " idx=(%" PRId64 ",%" PRId64 ",%" PRId64 ") dst=(%" PRId64 ",%" PRId64 ",%" PRId64 ") workers=%u",
            symbol, work_items, (const void *)ctx.src0, ctx.src1, ctx.dst,
            ctx.ne00, ctx.ne01, ctx.ne02, ctx.idx_i64 ? 1U : 0U, ctx.dst_type,
            ctx.s01, ctx.s02, ctx.s03, ctx.s10, ctx.s11, ctx.s12,
            ctx.s1, ctx.s2, ctx.s3, workers);
    trace_msg("native set_rows %s total=%" PRIu64
              " ne=(%" PRId64 ",%" PRId64 ",%" PRId64 ") workers=%u idx64=%u dst=%d",
              symbol, work_items, ctx.ne00, ctx.ne01, ctx.ne02, workers,
              ctx.idx_i64 ? 1U : 0U, ctx.dst_type);

    if (workers <= 1 || work_items <= 1) {
        struct SetRowsNativeWorker worker = {
            .ctx = &ctx,
            .begin = 0,
            .end = work_items,
            .status = 0,
        };
        set_rows_native_worker_main(&worker);
        return worker.status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct SetRowsNativeWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (work_items + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= work_items) break;
        if (end > work_items) end = work_items;
        worker[i].ctx = &ctx;
        worker[i].begin = begin;
        worker[i].end = end;
        worker[i].status = -1;
        if (pthread_create(&threads[i], NULL, set_rows_native_worker_main, &worker[i]) != 0) {
            log_msg("native set_rows failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    return status;
}

static int invoke_kernel_deepep_layout_native(const char *symbol,
                                              const uint64_t *argv,
                                              const struct PaccJobImage *job,
                                              size_t argc) {
    if (!symbol || !strstr(symbol, "deep_ep") || !strstr(symbol, "get_dispatch_layout")) {
        return 1;
    }
    if (argc < 9) {
        log_msg("native deepep layout invalid argc=%zu symbol=%s", argc, symbol);
        return -1;
    }

    int num_tokens = kernel_cell_i32(argv, argc, 5);
    int num_topk = kernel_cell_i32(argv, argc, 6);
    int num_ranks = kernel_cell_i32(argv, argc, 7);
    int num_experts = kernel_cell_i32(argv, argc, 8);

    if (num_tokens < 0 || num_topk <= 0 || num_ranks <= 0 || num_experts <= 0) {
        log_msg("native deepep layout invalid scalar args: tokens=%d topk_n=%d ranks=%d experts=%d",
                num_tokens, num_topk, num_ranks, num_experts);
        return -1;
    }

    int num_expert_per_rank = num_experts / num_ranks;
    if (num_expert_per_rank <= 0) {
        return -1;
    }

    size_t nt = (size_t)num_tokens;
    size_t nk = (size_t)num_topk;
    size_t nr = (size_t)num_ranks;
    size_t ne = (size_t)num_experts;
    if (nt && nk > SIZE_MAX / nt) {
        log_msg("native deepep layout topk element count overflow");
        return -1;
    }
    size_t topk_elems = nt * nk;
    if (topk_elems > SIZE_MAX / sizeof(int64_t) ||
        nr > SIZE_MAX / sizeof(int) ||
        ne > SIZE_MAX / sizeof(int) ||
        (nt && nr > SIZE_MAX / nt)) {
        log_msg("native deepep layout byte count overflow");
        return -1;
    }
    size_t topk_bytes = topk_elems * sizeof(int64_t);
    size_t rank_bytes = nr * sizeof(int);
    size_t expert_bytes = ne * sizeof(int);
    size_t token_rank_elems = nt * nr;
    size_t in_rank_bytes = token_rank_elems * sizeof(uint8_t);

    int num_rdma_ranks = 0;
    size_t rdma_bytes = 0;
    if (num_ranks > 8 && (num_ranks % 8) == 0) {
        num_rdma_ranks = num_ranks / 8;
        rdma_bytes = (size_t)num_rdma_ranks * sizeof(int);
    }

    uint64_t topk_phys = 0, rank_phys = 0, rdma_phys = 0, expert_phys = 0, in_rank_phys = 0;
    size_t topk_bind_bytes = 0, rank_bind_bytes = 0, rdma_bind_bytes = 0;
    size_t expert_bind_bytes = 0, in_rank_bind_bytes = 0;
    if (!kernel_binding_phys_size_for_arg(job, 0, &topk_phys, &topk_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 1, &rank_phys, &rank_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 3, &expert_phys, &expert_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 4, &in_rank_phys, &in_rank_bind_bytes)) {
        log_msg("native deepep layout missing required binding metadata");
        return -1;
    }
    bool have_rdma = kernel_binding_phys_size_for_arg(job, 2, &rdma_phys, &rdma_bind_bytes);
    if (topk_bind_bytes < topk_bytes ||
        rank_bind_bytes < rank_bytes ||
        expert_bind_bytes < expert_bytes ||
        in_rank_bind_bytes < in_rank_bytes ||
        (rdma_bytes && (!have_rdma || rdma_bind_bytes < rdma_bytes))) {
        log_msg("native deepep layout binding too small: topk=%zu/%zu rank=%zu/%zu "
                "rdma=%zu/%zu expert=%zu/%zu in_rank=%zu/%zu",
                topk_bind_bytes, topk_bytes, rank_bind_bytes, rank_bytes,
                rdma_bind_bytes, rdma_bytes, expert_bind_bytes, expert_bytes,
                in_rank_bind_bytes, in_rank_bytes);
        return -1;
    }

    if (num_ranks == 1 && env_flag_default_true("HETGPU_PACC_DEEPEP_LAYOUT_FAST_RANK1")) {
        int rank_one = num_tokens;
        uint8_t *in_rank_one = token_rank_elems ? (uint8_t *)calloc(token_rank_elems, sizeof(uint8_t)) : NULL;
        int *expert_zero = ne ? (int *)calloc(ne, sizeof(int)) : NULL;
        if ((token_rank_elems && !in_rank_one) || (ne && !expert_zero)) {
            free(in_rank_one);
            free(expert_zero);
            return -1;
        }
        if (in_rank_one) {
            memset(in_rank_one, 1, token_rank_elems);
        }
        bool ok =
            native_stage_write_pwrite(rank_phys, &rank_one, sizeof(rank_one)) &&
            native_stage_write_pwrite(expert_phys, expert_zero, expert_bytes) &&
            (!in_rank_bytes || native_stage_write_pwrite(in_rank_phys, in_rank_one, in_rank_bytes));
        free(in_rank_one);
        free(expert_zero);
        return ok ? 0 : -1;
    }

    void *topk_copy = NULL;
    if (topk_bytes && !native_stage_read_pread(topk_phys, topk_bytes, &topk_copy)) {
        log_msg("native deepep layout failed to read topk phys=0x%llx bytes=%zu",
                (unsigned long long)topk_phys, topk_bytes);
        return -1;
    }
    const int64_t *topk_idx = (const int64_t *)topk_copy;
    int *num_tokens_per_rank = (int *)calloc(nr, sizeof(int));
    int *num_tokens_per_rdma_rank = rdma_bytes ? (int *)calloc((size_t)num_rdma_ranks, sizeof(int)) : NULL;
    int *num_tokens_per_expert = (int *)calloc(ne, sizeof(int));
    uint8_t *is_token_in_rank = token_rank_elems ? (uint8_t *)calloc(token_rank_elems, sizeof(uint8_t)) : NULL;
    uint8_t *seen_rank = (uint8_t *)calloc((size_t)num_ranks, 1);
    uint8_t *seen_rdma = NULL;
    int status = -1;

    if (!num_tokens_per_rank || (rdma_bytes && !num_tokens_per_rdma_rank) ||
        !num_tokens_per_expert || (token_rank_elems && !is_token_in_rank) || !seen_rank) {
        log_msg("native deepep layout allocation failed");
        goto out;
    }
    if (rdma_bytes) {
        seen_rdma = (uint8_t *)calloc((size_t)num_rdma_ranks, 1);
        if (!seen_rdma) {
            log_msg("native deepep layout rdma allocation failed");
            goto out;
        }
    }

    for (int token = 0; token < num_tokens; token++) {
        memset(seen_rank, 0, (size_t)num_ranks);
        if (seen_rdma) {
            memset(seen_rdma, 0, (size_t)num_rdma_ranks);
        }

        const int64_t *row = topk_idx + (size_t)token * (size_t)num_topk;
        for (int j = 0; j < num_topk; j++) {
            int expert = (int)row[j];
            if (expert < 0 || expert >= num_experts) {
                continue;
            }
            num_tokens_per_expert[expert]++;

            int rank = expert / num_expert_per_rank;
            if (rank < 0) {
                rank = 0;
            } else if (rank >= num_ranks) {
                rank = num_ranks - 1;
            }
            seen_rank[rank] = 1;
            if (seen_rdma) {
                seen_rdma[rank / 8] = 1;
            }
        }

        for (int rank = 0; rank < num_ranks; rank++) {
            if (seen_rank[rank]) {
                num_tokens_per_rank[rank]++;
                is_token_in_rank[(size_t)token * (size_t)num_ranks + (size_t)rank] = 1;
            }
        }
        if (seen_rdma) {
            for (int rank = 0; rank < num_rdma_ranks; rank++) {
                if (seen_rdma[rank]) {
                    num_tokens_per_rdma_rank[rank]++;
                }
            }
        }
    }

    void *rank_arg_ptr = (void *)(uintptr_t)kernel_cell_u64(argv, argc, 1);
    void *rdma_arg_ptr = have_rdma ? (void *)(uintptr_t)kernel_cell_u64(argv, argc, 2) : NULL;
    void *expert_arg_ptr = (void *)(uintptr_t)kernel_cell_u64(argv, argc, 3);
    void *in_rank_arg_ptr = (void *)(uintptr_t)kernel_cell_u64(argv, argc, 4);
    if (rank_arg_ptr && rank_bytes) {
        memcpy(rank_arg_ptr, num_tokens_per_rank, rank_bytes);
    }
    if (rdma_arg_ptr && rdma_bytes) {
        memcpy(rdma_arg_ptr, num_tokens_per_rdma_rank, rdma_bytes);
    }
    if (expert_arg_ptr && expert_bytes) {
        memcpy(expert_arg_ptr, num_tokens_per_expert, expert_bytes);
    }
    if (in_rank_arg_ptr && in_rank_bytes) {
        memcpy(in_rank_arg_ptr, is_token_in_rank, in_rank_bytes);
    }
    jobd_io_fence();

    if (!native_stage_write_pwrite(rank_phys, num_tokens_per_rank, rank_bytes) ||
        (rdma_bytes && !native_stage_write_pwrite(rdma_phys, num_tokens_per_rdma_rank, rdma_bytes)) ||
        !native_stage_write_pwrite(expert_phys, num_tokens_per_expert, expert_bytes) ||
        (in_rank_bytes && !native_stage_write_pwrite(in_rank_phys, is_token_in_rank, in_rank_bytes))) {
        log_msg("native deepep layout failed to write output buffers");
        goto out;
    }

    trace_msg("native deepep layout done: tokens=%d topk=%d ranks=%d experts=%d",
              num_tokens, num_topk, num_ranks, num_experts);
    status = 0;

out:
    free(is_token_in_rank);
    free(num_tokens_per_expert);
    free(num_tokens_per_rdma_rank);
    free(num_tokens_per_rank);
    free(topk_copy);
    free(seen_rdma);
    free(seen_rank);
    return status;
}

static int32_t kernel_arg_record_i32_for_arg(const struct PaccJobImage *job, uint32_t arg_index) {
    if (!job || !job->arg_records || arg_index >= job->arg_count) {
        return 0;
    }
    return (int32_t)job->arg_records[arg_index].value;
}

static void deepep_direct_marker(uint64_t seq, uint32_t status, uint32_t aux) {
    if (seq) {
        mirror_host_status(g_mbox_fd, PACC_KERNEL_JOB_ID, seq, status);
        write_jobd_beacon(g_mbox_fd, PACC_KERNEL_JOB_ID, seq, status, aux);
    }
}

static int invoke_kernel_deepep_layout_native_direct(const char *symbol,
                                                     const struct PaccJobImage *job,
                                                     uint64_t seq) {
    if (!symbol || !strstr(symbol, "deep_ep") || !strstr(symbol, "get_dispatch_layout")) {
        return 1;
    }
    deepep_direct_marker(seq, 0x5d010000u, 0);
    if (!job || job->arg_count < 9) {
        log_msg("native deepep layout direct invalid argc=%zu symbol=%s",
                job ? job->arg_count : 0, symbol);
        return -1;
    }

    int num_tokens = kernel_arg_record_i32_for_arg(job, 5);
    int num_topk = kernel_arg_record_i32_for_arg(job, 6);
    int num_ranks = kernel_arg_record_i32_for_arg(job, 7);
    int num_experts = kernel_arg_record_i32_for_arg(job, 8);

    if (num_tokens < 0 || num_topk <= 0 || num_ranks <= 0 || num_experts <= 0) {
        log_msg("native deepep layout direct invalid scalar args: tokens=%d topk_n=%d ranks=%d experts=%d",
                num_tokens, num_topk, num_ranks, num_experts);
        return -1;
    }

    int num_expert_per_rank = num_experts / num_ranks;
    if (num_expert_per_rank <= 0) {
        return -1;
    }

    size_t nt = (size_t)num_tokens;
    size_t nk = (size_t)num_topk;
    size_t nr = (size_t)num_ranks;
    size_t ne = (size_t)num_experts;
    if (nt && nk > SIZE_MAX / nt) {
        return -1;
    }
    size_t topk_elems = nt * nk;
    if (topk_elems > SIZE_MAX / sizeof(int64_t) ||
        nr > SIZE_MAX / sizeof(int) ||
        ne > SIZE_MAX / sizeof(int) ||
        (nt && nr > SIZE_MAX / nt)) {
        return -1;
    }
    size_t topk_bytes = topk_elems * sizeof(int64_t);
    size_t rank_bytes = nr * sizeof(int);
    size_t expert_bytes = ne * sizeof(int);
    size_t token_rank_elems = nt * nr;
    size_t in_rank_bytes = token_rank_elems * sizeof(uint8_t);

    int num_rdma_ranks = 0;
    size_t rdma_bytes = 0;
    if (num_ranks > 8 && (num_ranks % 8) == 0) {
        num_rdma_ranks = num_ranks / 8;
        rdma_bytes = (size_t)num_rdma_ranks * sizeof(int);
    }

    uint64_t topk_phys = 0, rank_phys = 0, rdma_phys = 0, expert_phys = 0, in_rank_phys = 0;
    size_t topk_bind_bytes = 0, rank_bind_bytes = 0, rdma_bind_bytes = 0;
    size_t expert_bind_bytes = 0, in_rank_bind_bytes = 0;
    if (!kernel_binding_phys_size_for_arg(job, 0, &topk_phys, &topk_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 1, &rank_phys, &rank_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 3, &expert_phys, &expert_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 4, &in_rank_phys, &in_rank_bind_bytes)) {
        log_msg("native deepep layout direct missing required binding metadata");
        return -1;
    }
    deepep_direct_marker(seq, 0x5d020000u, (uint32_t)(topk_bind_bytes & 0xffffffffu));
    bool have_rdma = kernel_binding_phys_size_for_arg(job, 2, &rdma_phys, &rdma_bind_bytes);
    if (topk_bind_bytes < topk_bytes ||
        rank_bind_bytes < rank_bytes ||
        expert_bind_bytes < expert_bytes ||
        in_rank_bind_bytes < in_rank_bytes ||
        (rdma_bytes && (!have_rdma || rdma_bind_bytes < rdma_bytes))) {
        log_msg("native deepep layout direct binding too small: topk=%zu/%zu rank=%zu/%zu "
                "rdma=%zu/%zu expert=%zu/%zu in_rank=%zu/%zu",
                topk_bind_bytes, topk_bytes, rank_bind_bytes, rank_bytes,
                rdma_bind_bytes, rdma_bytes, expert_bind_bytes, expert_bytes,
                in_rank_bind_bytes, in_rank_bytes);
        return -1;
    }

    void *topk_copy = NULL;
    if (topk_bytes && !native_stage_read_pread(topk_phys, topk_bytes, &topk_copy)) {
        log_msg("native deepep layout direct failed to read topk phys=0x%llx bytes=%zu",
                (unsigned long long)topk_phys, topk_bytes);
        return -1;
    }
    deepep_direct_marker(seq, 0x5d030000u, topk_copy ? (uint32_t)((const int64_t *)topk_copy)[0] : 0);

    const int64_t *topk_idx = (const int64_t *)topk_copy;
    int *num_tokens_per_rank = (int *)calloc(nr, sizeof(int));
    int *num_tokens_per_rdma_rank = rdma_bytes ? (int *)calloc((size_t)num_rdma_ranks, sizeof(int)) : NULL;
    int *num_tokens_per_expert = (int *)calloc(ne, sizeof(int));
    uint8_t *is_token_in_rank = token_rank_elems ? (uint8_t *)calloc(token_rank_elems, sizeof(uint8_t)) : NULL;
    uint8_t *seen_rank = (uint8_t *)calloc((size_t)num_ranks, 1);
    uint8_t *seen_rdma = NULL;
    int status = -1;

    if (!num_tokens_per_rank || (rdma_bytes && !num_tokens_per_rdma_rank) ||
        !num_tokens_per_expert || (token_rank_elems && !is_token_in_rank) || !seen_rank) {
        goto direct_out;
    }
    if (rdma_bytes) {
        seen_rdma = (uint8_t *)calloc((size_t)num_rdma_ranks, 1);
        if (!seen_rdma) {
            goto direct_out;
        }
    }

    for (int token = 0; token < num_tokens; token++) {
        memset(seen_rank, 0, (size_t)num_ranks);
        if (seen_rdma) {
            memset(seen_rdma, 0, (size_t)num_rdma_ranks);
        }

        const int64_t *row = topk_idx + (size_t)token * (size_t)num_topk;
        for (int j = 0; j < num_topk; j++) {
            int expert = (int)row[j];
            if (expert < 0 || expert >= num_experts) {
                continue;
            }
            num_tokens_per_expert[expert]++;

            int rank = expert / num_expert_per_rank;
            if (rank < 0) {
                rank = 0;
            } else if (rank >= num_ranks) {
                rank = num_ranks - 1;
            }
            seen_rank[rank] = 1;
            if (seen_rdma) {
                seen_rdma[rank / 8] = 1;
            }
        }

        for (int rank = 0; rank < num_ranks; rank++) {
            if (seen_rank[rank]) {
                num_tokens_per_rank[rank]++;
                is_token_in_rank[(size_t)token * (size_t)num_ranks + (size_t)rank] = 1;
            }
        }
        if (seen_rdma) {
            for (int rank = 0; rank < num_rdma_ranks; rank++) {
                if (seen_rdma[rank]) {
                    num_tokens_per_rdma_rank[rank]++;
                }
            }
        }
    }

    deepep_direct_marker(seq, 0x5d040000u, nr ? (uint32_t)num_tokens_per_rank[0] : 0);
    if (!native_stage_write_pwrite(rank_phys, num_tokens_per_rank, rank_bytes) ||
        (rdma_bytes && !native_stage_write_pwrite(rdma_phys, num_tokens_per_rdma_rank, rdma_bytes)) ||
        !native_stage_write_pwrite(expert_phys, num_tokens_per_expert, expert_bytes) ||
        (in_rank_bytes && !native_stage_write_pwrite(in_rank_phys, is_token_in_rank, in_rank_bytes))) {
        log_msg("native deepep layout direct failed to write output buffers");
        goto direct_out;
    }
    deepep_direct_marker(seq, 0x5d050000u, nr ? (uint32_t)num_tokens_per_rank[0] : 0);

    status = 0;

direct_out:
    free(is_token_in_rank);
    free(num_tokens_per_expert);
    free(num_tokens_per_rdma_rank);
    free(num_tokens_per_rank);
    free(topk_copy);
    free(seen_rdma);
    free(seen_rank);
    return status;
}

static int invoke_kernel_index_select_embedding_native(const char *symbol,
                                                       const uint64_t *args,
                                                       const struct PaccJobImage *job,
                                                       size_t argc) {
    uint64_t out_phys = 0, weight_phys = 0, indices_phys = 0;
    size_t out_bind_bytes = 0, weight_bind_bytes = 0, indices_bind_bytes = 0;
    uint8_t *dst;
    const uint8_t *weight;
    const uint8_t *indices;
    uint64_t rows, inner, elem_size, index_elem_size;
    uint64_t row_bytes64, out_bytes64, indices_bytes64;
    size_t row_bytes, out_bytes, indices_bytes;

    #define INDEX_EMBED_BEACON(phase_, detail_) \
        do { \
            if (g_mbox_fd >= 0 && g_current_kernel_seq) { \
                write_jobd_beacon(g_mbox_fd, PACC_KERNEL_JOB_ID, g_current_kernel_seq, \
                                  (uint32_t)(phase_), (uint32_t)(detail_)); \
            } \
        } while (0)

    if (!symbol || (!strstr(symbol, "indexSelectSmallIndex") &&
                    !strstr(symbol, "pacc_pytorch_index_select_embedding"))) {
        return 1;
    }
    INDEX_EMBED_BEACON(0x5e10, (uint32_t)argc);
    if (argc < 7 || !job) {
        log_msg("native indexSelect embedding invalid argc=%zu symbol=%s",
                argc, symbol ? symbol : "<unknown>");
        INDEX_EMBED_BEACON(0xffff5e01, (uint32_t)argc);
        return (int)0xffff5e01u;
    }
    if (!kernel_binding_phys_size_for_arg(job, 0, &out_phys, &out_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 1, &weight_phys, &weight_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 2, &indices_phys, &indices_bind_bytes)) {
        log_msg("native indexSelect embedding missing binding metadata");
        INDEX_EMBED_BEACON(0xffff5e02, (uint32_t)job->binding_count);
        return (int)0xffff5e02u;
    }

    rows = kernel_cell_u64(args, argc, 3);
    inner = kernel_cell_u64(args, argc, 4);
    elem_size = kernel_cell_u64(args, argc, 5);
    index_elem_size = kernel_cell_u64(args, argc, 6);
    if (!rows || !inner ||
        (elem_size != 1 && elem_size != 2 && elem_size != 4 && elem_size != 8) ||
        (index_elem_size != 4 && index_elem_size != 8)) {
        log_msg("native indexSelect embedding bad shape rows=%" PRIu64
                " inner=%" PRIu64 " elem=%" PRIu64 " index_elem=%" PRIu64,
                rows, inner, elem_size, index_elem_size);
        INDEX_EMBED_BEACON(0xffff5e03, (uint32_t)rows);
        return (int)0xffff5e03u;
    }
    if (inner > UINT64_MAX / elem_size) {
        INDEX_EMBED_BEACON(0xffff5e04, 0);
        return (int)0xffff5e04u;
    }
    row_bytes64 = inner * elem_size;
    if (rows > UINT64_MAX / row_bytes64 ||
        rows > UINT64_MAX / index_elem_size ||
        row_bytes64 > (uint64_t)SIZE_MAX) {
        INDEX_EMBED_BEACON(0xffff5e05, (uint32_t)(row_bytes64 & 0xffffffffu));
        return (int)0xffff5e05u;
    }
    out_bytes64 = rows * row_bytes64;
    indices_bytes64 = rows * index_elem_size;
    if (out_bytes64 > (uint64_t)SIZE_MAX || indices_bytes64 > (uint64_t)SIZE_MAX) {
        INDEX_EMBED_BEACON(0xffff5e06, (uint32_t)(out_bytes64 & 0xffffffffu));
        return (int)0xffff5e06u;
    }
    row_bytes = (size_t)row_bytes64;
    out_bytes = (size_t)out_bytes64;
    indices_bytes = (size_t)indices_bytes64;
    if (out_bind_bytes < out_bytes || indices_bind_bytes < indices_bytes) {
        log_msg("native indexSelect embedding binding too small out=%zu/%zu indices=%zu/%zu",
                out_bind_bytes, out_bytes, indices_bind_bytes, indices_bytes);
        INDEX_EMBED_BEACON(0xffff5e07, (uint32_t)out_bind_bytes);
        return (int)0xffff5e07u;
    }
    INDEX_EMBED_BEACON(0x5e11, (uint32_t)rows);

    dst = (uint8_t *)(uintptr_t)kernel_cell_u64(args, argc, 0);
    weight = (const uint8_t *)(uintptr_t)kernel_cell_u64(args, argc, 1);
    indices = (const uint8_t *)(uintptr_t)kernel_cell_u64(args, argc, 2);
    INDEX_EMBED_BEACON(0x5e12, (uint32_t)(row_bytes & 0xffffffffu));

    (void)out_phys;
    (void)weight_phys;
    (void)indices_phys;
    dst = (uint8_t *)(uintptr_t)kernel_cell_u64(args, argc, 0);
    weight = (const uint8_t *)(uintptr_t)kernel_cell_u64(args, argc, 1);
    indices = (const uint8_t *)(uintptr_t)kernel_cell_u64(args, argc, 2);
    if (!dst || !weight || (!indices && weight_bind_bytes != out_bytes)) {
        log_msg("native indexSelect embedding mapped pointers missing dst=%p weight=%p indices=%p",
                (void *)dst, (const void *)weight, (const void *)indices);
        INDEX_EMBED_BEACON(0xffff5e08,
                           (dst ? 1u : 0u) | (weight ? 2u : 0u) | (indices ? 4u : 0u));
        return (int)0xffff5e08u;
    }

    if (weight_bind_bytes == out_bytes) {
        /*
         * Host-side sparse embedding staging already packs exactly the requested
         * rows into a compact weight buffer and rewrites indices to 0..rows-1.
         * On LX500 the compact indices window can still observe stale cachelines
         * for later rows, so use the binding size to recognize the packed path
         * and copy it sequentially.
         */
        INDEX_EMBED_BEACON(0x5e14, (uint32_t)(out_bytes & 0xffffffffu));
        jobd_invalidate_for_cpu(weight, out_bytes);
        memcpy(dst, weight, out_bytes);
        jobd_io_fence();
        jobd_flush_for_device(dst, out_bytes);
        INDEX_EMBED_BEACON(0x5e19, (uint32_t)out_bytes);
        return 0;
    }

    jobd_io_fence();
    jobd_invalidate_for_cpu(indices, indices_bytes);

    for (uint64_t row = 0; row < rows; row++) {
        uint64_t idx = 0;
        if (index_elem_size == 8) {
            memcpy(&idx, indices + row * 8, sizeof(idx));
        } else {
            uint32_t idx32 = 0;
            memcpy(&idx32, indices + row * 4, sizeof(idx32));
            idx = idx32;
        }
        if (row == 0) {
            INDEX_EMBED_BEACON(0x5e13, (uint32_t)(idx & 0xffffffffu));
        }
        if (idx > UINT64_MAX / row_bytes64) {
            log_msg("native indexSelect embedding index overflow idx=%" PRIu64, idx);
            INDEX_EMBED_BEACON(0xffff5e09, (uint32_t)(idx & 0xffffffffu));
            return (int)0xffff5e09u;
        }
        uint64_t src_off = idx * row_bytes64;
        uint64_t dst_off = row * row_bytes64;
        if (src_off > (uint64_t)weight_bind_bytes ||
            row_bytes64 > (uint64_t)weight_bind_bytes - src_off) {
            log_msg("native indexSelect embedding index out of range idx=%" PRIu64
                    " src_off=0x%" PRIx64 " row_bytes=%zu weight_bytes=%zu",
                    idx, src_off, row_bytes, weight_bind_bytes);
            INDEX_EMBED_BEACON(0xffff5e0a, (uint32_t)(idx & 0xffffffffu));
            return (int)0xffff5e0au;
        }
        jobd_invalidate_for_cpu(weight + src_off, row_bytes);
        memcpy(dst + dst_off, weight + src_off, row_bytes);
    }

    jobd_io_fence();
    jobd_flush_for_device(dst, out_bytes);
    INDEX_EMBED_BEACON(0x5e19, (uint32_t)out_bytes);
    #undef INDEX_EMBED_BEACON
    return 0;
}

static bool kernel_arg_record_u64(const struct PaccJobImage *job,
                                  size_t index,
                                  uint64_t *out) {
    if (!job || !job->arg_records || index >= job->arg_count || !out) {
        return false;
    }
    const struct PaccKernelArgRecord *rec = &job->arg_records[index];
    if ((rec->flags & PACC_KERNEL_ARG_FLAG_INLINE_BLOB) && rec->size > sizeof(uint64_t)) {
        return false;
    }
    if (rec->size == 0 || rec->size > sizeof(uint64_t)) {
        return false;
    }
    *out = rec->value;
    return true;
}

static int invoke_kernel_index_select_embedding_packed_direct(int fd,
                                                              const char *symbol,
                                                              const struct PaccJobImage *job,
                                                              uint64_t seq) {
    uint64_t out_phys = 0, weight_phys = 0;
    size_t out_bind_bytes = 0, weight_bind_bytes = 0;
    uint64_t rows = 0, inner = 0, elem_size = 0, index_elem_size = 0;
    uint64_t row_bytes64 = 0, out_bytes64 = 0;
    size_t out_bytes = 0;
    struct Map mout = {0}, mw = {0};

    if (!symbol || (!strstr(symbol, "indexSelectSmallIndex") &&
                    !strstr(symbol, "pacc_pytorch_index_select_embedding"))) {
        return 1;
    }
    if (!kernel_binding_phys_size_for_arg(job, 0, &out_phys, &out_bind_bytes) ||
        !kernel_binding_phys_size_for_arg(job, 1, &weight_phys, &weight_bind_bytes)) {
        return 1;
    }

    if (out_bind_bytes != 0 && weight_bind_bytes == out_bind_bytes) {
        out_bytes = out_bind_bytes;
        goto copy_packed_embedding;
    }

    if (!kernel_arg_record_u64(job, 3, &rows) ||
        !kernel_arg_record_u64(job, 4, &inner) ||
        !kernel_arg_record_u64(job, 5, &elem_size) ||
        !kernel_arg_record_u64(job, 6, &index_elem_size)) {
        return 1;
    }
    if (!rows || !inner ||
        (elem_size != 1 && elem_size != 2 && elem_size != 4 && elem_size != 8) ||
        (index_elem_size != 4 && index_elem_size != 8) ||
        inner > UINT64_MAX / elem_size) {
        return 1;
    }
    row_bytes64 = inner * elem_size;
    if (rows > UINT64_MAX / row_bytes64 ||
        rows * row_bytes64 > (uint64_t)SIZE_MAX) {
        return 1;
    }
    out_bytes64 = rows * row_bytes64;
    out_bytes = (size_t)out_bytes64;
    if (out_bind_bytes < out_bytes || weight_bind_bytes != out_bytes) {
        return 1;
    }

copy_packed_embedding:
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, seq,
                      0x5e14u, (uint32_t)(out_bytes & 0xffffffffu));
    if ((map_phys(fd, out_phys, out_bytes, &mout) != 0 &&
         map_phys_copy_fallback(fd, out_phys, out_bytes, &mout) != 0) ||
        (map_phys(fd, weight_phys, out_bytes, &mw) != 0 &&
         map_phys_copy_fallback(fd, weight_phys, out_bytes, &mw) != 0)) {
        unmap_phys(&mout);
        unmap_phys(&mw);
        return (int)0xffff5e81u;
    }
    sync_map_for_cpu(&mw);
    jobd_invalidate_for_cpu(mw.ptr, out_bytes);
    memcpy(mout.ptr, mw.ptr, out_bytes);
    jobd_io_fence();
    if (flush_map_to_phys(&mout) != 0) {
        unmap_phys(&mout);
        unmap_phys(&mw);
        return (int)0xffff5e82u;
    }
    jobd_flush_for_device(mout.ptr, out_bytes);
    unmap_phys(&mout);
    unmap_phys(&mw);
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, seq,
                      0x5e19u, (uint32_t)(out_bytes & 0xffffffffu));
    return 0;
}

static int invoke_kernel_native(const char *symbol,
                                const uint64_t *args,
                                const struct PaccJobImage *job,
                                size_t argc) {
    if (jobd_force_elf_for_symbol(symbol)) {
        trace_msg("native shortcut skipped for %s; forcing direct ELF",
                  symbol ? symbol : "<unknown>");
        return 1;
    }

    int native_status = invoke_kernel_pytorch_fill_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }

    native_status = invoke_kernel_bin_bcast_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_mmvf_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_get_rows_float_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_set_rows_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_deepep_layout_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_index_select_embedding_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_cpy_scalar_f32_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_scale_f32_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_convert_unary_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_rope_norm_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    native_status = invoke_kernel_rope_multi_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }
    if (jobd_force_elf_enabled()) {
        trace_msg("no native shortcut for %s, forcing direct ELF",
                  symbol ? symbol : "<unknown>");
    }
    return 1;
}

static void *bin_bcast_elf_row_worker_main(void *opaque) {
    struct BinBcastElfRowWorker *worker = (struct BinBcastElfRowWorker *)opaque;
    uint32_t ne1 = worker->ne1 ? worker->ne1 : 1u;
    uint32_t z_rows = worker->z_rows ? worker->z_rows : 1u;

    for (uint64_t row = worker->begin_row; row < worker->end_row; row++) {
        uint32_t y = (uint32_t)(row % ne1);
        uint32_t z = (uint32_t)(row / ne1);
        if (z >= z_rows) {
            break;
        }
        worker->set_launch(0, 0, 0,
                           1, 1, 1,
                           0, y, z,
                           1, ne1, z_rows);
        if (jobd_trace_enabled() &&
            (row == worker->begin_row || row + 1 == worker->end_row)) {
            trace_msg("bin_bcast ELF row enter: symbol=%s row=%" PRIu64
                      " y=%u z=%u range=[%" PRIu64 ",%" PRIu64 ")",
                      worker->symbol ? worker->symbol : "<unknown>",
                      row, y, z, worker->begin_row, worker->end_row);
        }
        worker->status = invoke_kernel_symbol(worker->symbol, worker->fn,
                                              worker->args, worker->job,
                                              worker->argc);
        if (jobd_trace_enabled() &&
            (row == worker->begin_row || row + 1 == worker->end_row)) {
            trace_msg("bin_bcast ELF row exit: symbol=%s row=%" PRIu64
                      " status=%d",
                      worker->symbol ? worker->symbol : "<unknown>",
                      row, worker->status);
        }
        if (worker->status != 0) {
            return NULL;
        }
    }
    worker->status = 0;
    return NULL;
}

static int invoke_kernel_bin_bcast_elf_rows(const char *symbol,
                                            void *fn,
                                            const uint64_t *args,
                                            const struct PaccJobImage *job,
                                            size_t argc,
                                            PaccSetLaunchFn set_launch) {
    if (!symbol || !strstr(symbol, "k_bin_bcastIX") || !set_launch || argc < 11) {
        return 1;
    }

    int32_t ne0 = kernel_cell_i32(args, argc, 3);
    int32_t ne1_signed = kernel_cell_i32(args, argc, 4);
    int32_t ne2_signed = kernel_cell_i32(args, argc, 5);
    struct PaccUint3 ne3 = kernel_cell_u3(args, argc, 6);
    if (ne0 <= 0 || ne1_signed <= 0 || ne2_signed <= 0) {
        return 0;
    }

    uint32_t ne1 = (uint32_t)ne1_signed;
    uint32_t ne2 = (uint32_t)ne2_signed;
    uint32_t ne3_dim = ne3.z ? ne3.z : 1u;
    uint64_t z_rows64 = (uint64_t)ne2 * ne3_dim;
    uint64_t total_rows = z_rows64 * ne1;
    if (!total_rows || z_rows64 > UINT32_MAX) {
        return total_rows ? -1 : 0;
    }

    unsigned workers = kernel_worker_threads(total_rows);
    trace_msg("bin_bcast ELF row replay: symbol=%s ne0=%d ne1=%u ne2=%u ne3=%u rows=%" PRIu64 " workers=%u",
              symbol, ne0, ne1, ne2, ne3_dim, total_rows, workers);
    if (workers <= 1) {
        struct BinBcastElfRowWorker worker = {
            .symbol = symbol,
            .fn = fn,
            .args = args,
            .job = job,
            .argc = argc,
            .set_launch = set_launch,
            .ne1 = ne1,
            .z_rows = (uint32_t)z_rows64,
            .begin_row = 0,
            .end_row = total_rows,
            .status = 0,
        };
        bin_bcast_elf_row_worker_main(&worker);
        return worker.status;
    }

    pthread_t threads[PACC_KERNEL_MAX_THREADS];
    struct BinBcastElfRowWorker worker[PACC_KERNEL_MAX_THREADS];
    unsigned created = 0;
    uint64_t chunk = (total_rows + workers - 1u) / workers;
    memset(worker, 0, sizeof(worker));
    for (unsigned i = 0; i < workers; i++) {
        uint64_t begin = (uint64_t)i * chunk;
        uint64_t end = begin + chunk;
        if (begin >= total_rows) break;
        if (end > total_rows) end = total_rows;
        worker[i].symbol = symbol;
        worker[i].fn = fn;
        worker[i].args = args;
        worker[i].job = job;
        worker[i].argc = argc;
        worker[i].set_launch = set_launch;
        worker[i].ne1 = ne1;
        worker[i].z_rows = (uint32_t)z_rows64;
        worker[i].begin_row = begin;
        worker[i].end_row = end;
        worker[i].status = 0;
        if (pthread_create(&threads[i], NULL, bin_bcast_elf_row_worker_main, &worker[i]) != 0) {
            log_msg("bin_bcast ELF row replay failed to create worker %u", i);
            for (unsigned j = 0; j < created; j++) {
                pthread_join(threads[j], NULL);
            }
            return -1;
        }
        created++;
    }

    int status = 0;
    for (unsigned i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
        if (worker[i].status != 0 && status == 0) {
            status = worker[i].status;
        }
    }
    return status;
}

static void kernel_decode_flat_index(uint64_t idx,
                                     uint32_t gx, uint32_t gy, uint32_t gz,
                                     uint32_t bx, uint32_t by, uint32_t bz,
                                     uint32_t *tid_x, uint32_t *tid_y, uint32_t *tid_z,
                                     uint32_t *cta_x, uint32_t *cta_y, uint32_t *cta_z) {
    (void)gz;
    *tid_x = (uint32_t)(idx % bx);
    idx /= bx;
    *tid_y = (uint32_t)(idx % by);
    idx /= by;
    *tid_z = (uint32_t)(idx % bz);
    idx /= bz;
    *cta_x = (uint32_t)(idx % gx);
    idx /= gx;
    *cta_y = (uint32_t)(idx % gy);
    idx /= gy;
    *cta_z = (uint32_t)idx;
}

static void *kernel_grid_worker_main(void *opaque) {
    struct KernelGridWorker *worker = (struct KernelGridWorker *)opaque;
    for (uint64_t idx = worker->begin; idx < worker->end; idx++) {
        uint32_t tid_x, tid_y, tid_z;
        uint32_t cta_x, cta_y, cta_z;
        kernel_decode_flat_index(idx,
                                 worker->gx, worker->gy, worker->gz,
                                 worker->bx, worker->by, worker->bz,
                                 &tid_x, &tid_y, &tid_z,
                                 &cta_x, &cta_y, &cta_z);
        if (worker->set_launch) {
            worker->set_launch(tid_x, tid_y, tid_z,
                               worker->bx, worker->by, worker->bz,
                               cta_x, cta_y, cta_z,
                               worker->gx, worker->gy, worker->gz);
        }
        if (jobd_trace_enabled() && (idx == worker->begin || idx + 1 == worker->end)) {
            trace_msg("kernel grid worker enter: symbol=%s idx=%" PRIu64
                      " range=[%" PRIu64 ",%" PRIu64 ") tid=(%u,%u,%u) cta=(%u,%u,%u)",
                      worker->symbol ? worker->symbol : "<unknown>", idx,
                      worker->begin, worker->end,
                      tid_x, tid_y, tid_z, cta_x, cta_y, cta_z);
        }
        worker->status = invoke_kernel_symbol(worker->symbol, worker->fn,
                                              worker->args, worker->job,
                                              worker->argc);
        if (jobd_trace_enabled() && (idx == worker->begin || idx + 1 == worker->end)) {
            trace_msg("kernel grid worker exit: symbol=%s idx=%" PRIu64
                      " status=%d",
                      worker->symbol ? worker->symbol : "<unknown>",
                      idx, worker->status);
        }
        if (worker->status != 0) {
            return NULL;
        }
    }
    worker->status = 0;
    return NULL;
}

static int invoke_kernel_symbol_grid(const char *symbol, void *fn,
                                     const uint64_t *args,
                                     const struct PaccJobImage *job,
                                     size_t argc,
                                     PaccSetLaunchFn set_launch) {
    uint32_t gx = pacc_nonzero_dim(job ? job->header.grid_x : 1u);
    uint32_t gy = pacc_nonzero_dim(job ? job->header.grid_y : 1u);
    uint32_t gz = pacc_nonzero_dim(job ? job->header.grid_z : 1u);
    uint32_t bx = pacc_nonzero_dim(job ? job->header.block_x : 1u);
    uint32_t by = pacc_nonzero_dim(job ? job->header.block_y : 1u);
    uint32_t bz = pacc_nonzero_dim(job ? job->header.block_z : 1u);
    uint64_t total_threads = (uint64_t)gx * gy * gz * bx * by * bz;

    int native_status = invoke_kernel_native(symbol, args, job, argc);
    if (native_status <= 0) {
        return native_status;
    }

    if (total_threads > 1u && !set_launch) {
        log_msg("kernel dispatch %s needs launch sregs for %" PRIu64
                " logical threads, but helper is missing; clear stale kernel cache",
                symbol ? symbol : "<unknown>", total_threads);
        return -1;
    }

    int bin_bcast_rows = invoke_kernel_bin_bcast_elf_rows(symbol, fn, args, job, argc, set_launch);
    if (bin_bcast_rows <= 0) {
        return bin_bcast_rows;
    }

    if (jobd_fork_elf_enabled() && total_threads == 1u) {
        pid_t pid = fork();
        if (pid < 0) {
            log_msg("kernel fork dispatch %s failed: errno=%d",
                    symbol ? symbol : "<unknown>", errno);
            return -1;
        }
        if (pid == 0) {
            if (set_launch) {
                set_launch(0, 0, 0, bx, by, bz, 0, 0, 0, gx, gy, gz);
            }
            int child_status = invoke_kernel_symbol(symbol, fn, args, job, argc);
            _exit(child_status == 0 ? 0 : 125);
        }

        uint64_t waited_ms = 0;
        uint64_t timeout_ms = jobd_fork_elf_timeout_ms();
        int wstatus = 0;
        for (;;) {
            pid_t got = waitpid(pid, &wstatus, WNOHANG);
            if (got == pid) break;
            if (got < 0) {
                log_msg("kernel fork dispatch %s waitpid failed: errno=%d",
                        symbol ? symbol : "<unknown>", errno);
                return -1;
            }
            if (timeout_ms != 0 && waited_ms >= timeout_ms) {
                kill(pid, SIGKILL);
                waitpid(pid, &wstatus, 0);
                log_msg("kernel fork dispatch %s timed out after %" PRIu64 " ms",
                        symbol ? symbol : "<unknown>", timeout_ms);
                return (int)0xffff51feu;
            }
            usleep(1000);
            waited_ms++;
        }
        if (WIFEXITED(wstatus)) {
            int code = WEXITSTATUS(wstatus);
            if (code == 0) return 0;
            log_msg("kernel fork dispatch %s exited code=%d",
                    symbol ? symbol : "<unknown>", code);
            return (int)(0xffff5100u | (uint32_t)(code & 0xff));
        }
        if (WIFSIGNALED(wstatus)) {
            int sig = WTERMSIG(wstatus);
            log_msg("kernel fork dispatch %s signaled sig=%d",
                    symbol ? symbol : "<unknown>", sig);
            return (int)(0xffff5100u | (uint32_t)(sig & 0xff));
        }
        return -1;
    }

    unsigned workers = kernel_worker_threads(total_threads);
    if (workers > 1) {
        pthread_t threads[PACC_KERNEL_MAX_THREADS];
        struct KernelGridWorker worker[PACC_KERNEL_MAX_THREADS];
        unsigned created = 0;
        uint64_t chunk = (total_threads + workers - 1u) / workers;

        trace_msg("kernel dispatch %s replaying %" PRIu64
                  " logical threads with %u workers",
                  symbol ? symbol : "<unknown>", total_threads, workers);
        memset(worker, 0, sizeof(worker));
        for (unsigned i = 0; i < workers; i++) {
            uint64_t begin = (uint64_t)i * chunk;
            uint64_t end = begin + chunk;
            if (begin >= total_threads) break;
            if (end > total_threads) end = total_threads;
            worker[i].symbol = symbol;
            worker[i].fn = fn;
            worker[i].args = args;
            worker[i].job = job;
            worker[i].argc = argc;
            worker[i].set_launch = set_launch;
            worker[i].gx = gx;
            worker[i].gy = gy;
            worker[i].gz = gz;
            worker[i].bx = bx;
            worker[i].by = by;
            worker[i].bz = bz;
            worker[i].begin = begin;
            worker[i].end = end;
            worker[i].status = 0;
            if (pthread_create(&threads[i], NULL, kernel_grid_worker_main, &worker[i]) != 0) {
                log_msg("kernel dispatch %s failed to create worker %u",
                        symbol ? symbol : "<unknown>", i);
                for (unsigned j = 0; j < created; j++) {
                    pthread_join(threads[j], NULL);
                }
                return -1;
            }
            created++;
        }

        int status = 0;
        for (unsigned i = 0; i < created; i++) {
            pthread_join(threads[i], NULL);
            if (worker[i].status != 0 && status == 0) {
                status = worker[i].status;
            }
        }
        return status;
    }

    for (uint32_t cta_z = 0; cta_z < gz; cta_z++) {
        for (uint32_t cta_y = 0; cta_y < gy; cta_y++) {
            for (uint32_t cta_x = 0; cta_x < gx; cta_x++) {
                for (uint32_t tid_z = 0; tid_z < bz; tid_z++) {
                    for (uint32_t tid_y = 0; tid_y < by; tid_y++) {
                        for (uint32_t tid_x = 0; tid_x < bx; tid_x++) {
                            if (set_launch) {
                                set_launch(tid_x, tid_y, tid_z,
                                           bx, by, bz,
                                           cta_x, cta_y, cta_z,
                                           gx, gy, gz);
                            }
                            if (jobd_trace_enabled() &&
                                ((tid_x == 0 && tid_y == 0 && tid_z == 0 &&
                                  cta_x == 0 && cta_y == 0 && cta_z == 0) ||
                                 (tid_x + 1 == bx && tid_y + 1 == by && tid_z + 1 == bz &&
                                  cta_x + 1 == gx && cta_y + 1 == gy && cta_z + 1 == gz))) {
                                trace_msg("kernel grid serial enter: symbol=%s tid=(%u,%u,%u) cta=(%u,%u,%u)",
                                          symbol ? symbol : "<unknown>",
                                          tid_x, tid_y, tid_z, cta_x, cta_y, cta_z);
                            }
                            int status = invoke_kernel_symbol(symbol, fn, args, job, argc);
                            if (jobd_trace_enabled() &&
                                ((tid_x == 0 && tid_y == 0 && tid_z == 0 &&
                                  cta_x == 0 && cta_y == 0 && cta_z == 0) ||
                                 (tid_x + 1 == bx && tid_y + 1 == by && tid_z + 1 == bz &&
                                  cta_x + 1 == gx && cta_y + 1 == gy && cta_z + 1 == gz))) {
                                trace_msg("kernel grid serial exit: symbol=%s tid=(%u,%u,%u) cta=(%u,%u,%u) status=%d",
                                          symbol ? symbol : "<unknown>",
                                          tid_x, tid_y, tid_z, cta_x, cta_y, cta_z, status);
                            }
                            if (status != 0) return status;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static void release_kernel_binding_maps(struct KernelBindingMap *maps, size_t count) {
    for (size_t i = 0; i < count; i++) {
        bool needs_copyback = (maps[i].flags & PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT) != 0;
        if (needs_copyback) {
            if (flush_map_to_phys(&maps[i].map) != 0) {
                log_msg("kernel binding copy-back failed: arg=%u phys=0x%" PRIx64
                        " len=%zu",
                        maps[i].arg_index, maps[i].map.phys, maps[i].map.len);
            }
            if (!maps[i].map.copied && maps[i].map.ptr && maps[i].map.len) {
                /*
                 * Some shared-DDR mappings are visible to the PACC Linux process
                 * through a cached full-DDR/slot mmap.  A cache maintenance op is
                 * not always enough for the host helper path to observe tiny ELF
                 * kernel outputs, so mirror output mappings back through the
                 * mbox/shared-DDR fd before AP-side readback.
                 */
                if (write_phys_copy_pwrite_only(maps[i].map.fd,
                                                maps[i].map.phys,
                                                maps[i].map.ptr,
                                                maps[i].map.len) != 0) {
                    log_msg("kernel binding pwrite mirror failed: arg=%u phys=0x%" PRIx64
                            " len=%zu",
                            maps[i].arg_index, maps[i].map.phys, maps[i].map.len);
                }
            }
            if (jobd_msync_enabled() && maps[i].map.base) {
                msync(maps[i].map.base, maps[i].map.map_len, MS_SYNC);
            }
            jobd_flush_for_device(maps[i].map.ptr, maps[i].map.len);
        }
        unmap_phys(&maps[i].map);
    }
}

static void release_kernel_binding_maps_native_fast(struct KernelBindingMap *maps,
                                                    size_t count,
                                                    bool copyback_outputs) {
    for (size_t i = 0; i < count; i++) {
        bool needs_copyback =
            copyback_outputs &&
            ((maps[i].flags & PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT) != 0);
        if (needs_copyback) {
            if (flush_map_to_phys(&maps[i].map) != 0) {
                log_msg("native kernel binding copy-back failed: arg=%u phys=0x%" PRIx64
                        " len=%zu",
                        maps[i].arg_index, maps[i].map.phys, maps[i].map.len);
            }
            if (!maps[i].map.copied && maps[i].map.ptr && maps[i].map.len) {
                if (write_phys_copy_pwrite_only(maps[i].map.fd,
                                                maps[i].map.phys,
                                                maps[i].map.ptr,
                                                maps[i].map.len) != 0) {
                    log_msg("native kernel binding pwrite mirror failed: arg=%u phys=0x%" PRIx64
                            " len=%zu",
                            maps[i].arg_index, maps[i].map.phys, maps[i].map.len);
                }
            }
            jobd_flush_for_device(maps[i].map.ptr, maps[i].map.len);
        }
        unmap_phys(&maps[i].map);
    }
}

static struct KernelBindingMap *find_binding_map(struct KernelBindingMap *maps, size_t count, uint32_t arg_index) {
    for (size_t i = 0; i < count; i++) {
        if (maps[i].arg_index == arg_index) return &maps[i];
    }
    return NULL;
}

static bool kernel_symbol_is_index_select_embedding(const char *symbol) {
    return symbol && (strstr(symbol, "indexSelectSmallIndex") ||
                      strstr(symbol, "pacc_pytorch_index_select_embedding"));
}

static int build_kernel_launch_args(
    int fd,
    const struct PaccJobImage *job,
    const char *symbol,
    uint64_t *argv_out,
    size_t *argc_out,
    struct KernelParamCell arg_storage[PACC_MAX_KERNEL_ARGS],
    struct KernelBindingMap *maps,
    size_t *map_count_out) {
    size_t argc = 0;
    size_t map_count = 0;
    size_t default_bind_bytes = (size_t)g_page_size;

    g_kernel_arg_error = 0;
    if (!job || !argv_out || !argc_out || !arg_storage || !maps || !map_count_out) {
        g_kernel_arg_error = 0xe1;
        return -1;
    }
    if (job->arg_count > PACC_MAX_KERNEL_ARGS || job->binding_count > PACC_MAX_KERNEL_BINDINGS) {
        log_msg("kernel launch ABI too large: args=%zu/%u bindings=%zu/%u",
                job->arg_count, PACC_MAX_KERNEL_ARGS,
                job->binding_count, PACC_MAX_KERNEL_BINDINGS);
        g_kernel_arg_error = 0xe2;
        return -1;
    }

    for (size_t i = 0; i < job->binding_count; i++) {
        const struct PaccKernelBufferBinding *binding = &job->bindings[i];
        size_t bind_bytes = kernel_binding_map_bytes(binding, default_bind_bytes);
        if (map_count >= PACC_MAX_KERNEL_BINDINGS) {
            log_msg("kernel binding map limit reached: bindings=%zu map_count=%zu max=%u",
                    job->binding_count, map_count, PACC_MAX_KERNEL_BINDINGS);
            g_kernel_arg_error = 0xa0u | (uint32_t)(i & 0x0fu);
            return -1;
        }
        if (binding->addr == 0) continue;
        if (map_phys(fd, binding->addr, bind_bytes, &maps[map_count].map) != 0 &&
            map_phys_copy_fallback(fd, binding->addr, bind_bytes, &maps[map_count].map) != 0) {
            if (kernel_symbol_is_index_select_embedding(symbol) &&
                binding->arg_index == 2U) {
                trace_msg("kernel binding map skipped optional indexSelect indices arg=%u phys=0x%" PRIx64
                          " size=%zu flags=0x%x",
                          binding->arg_index, binding->addr, bind_bytes, binding->flags);
                continue;
            }
            log_msg("kernel binding map failed: arg=%u phys=0x%" PRIx64
                    " size=%zu flags=0x%x",
                    binding->arg_index, binding->addr, bind_bytes, binding->flags);
            release_kernel_binding_maps(maps, map_count);
            g_kernel_arg_error = 0xb0u | (uint32_t)(i & 0x0fu);
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
            if ((rec->flags & PACC_KERNEL_ARG_FLAG_INLINE_BLOB) && rec->size > 16U) {
                if (!job->raw_params || value > job->raw_param_size ||
                    rec->size > job->raw_param_size - value) {
                    log_msg("kernel inline arg %zu out of raw-param bounds off=%" PRIu64
                            " size=%u raw=%zu",
                            i, value, rec->size, job->raw_param_size);
                    release_kernel_binding_maps(maps, map_count);
                    g_kernel_arg_error = 0xc0u | (uint32_t)(i & 0x0fu);
                    return -1;
                }
                argv_out[argc++] = (uint64_t)(uintptr_t)(job->raw_params + value);
                continue;
            }
            if ((rec->flags & PACC_KERNEL_ARG_FLAG_INLINE_BLOB) && rec->size <= 16U) {
                trace_msg("kernel arg %zu ignoring inline flag for small by-value arg: size=%u flags=0x%x",
                          i, rec->size, rec->flags);
            }
            if (rec->kind == 1U) {
                struct KernelBindingMap *binding = find_binding_map(maps, map_count, (uint32_t)i);
                if (binding) {
                    value = (uint64_t)(uintptr_t)binding->map.ptr;
                }
            }
            if (rec->size > 16U) {
                log_msg("kernel arg %zu too large for byref slot: size=%u", i, rec->size);
                release_kernel_binding_maps(maps, map_count);
                g_kernel_arg_error = 0xd0u | (uint32_t)(i & 0x0fu);
                return -1;
            }
            size_t slot = argc;
            arg_storage[slot].lo = 0;
            arg_storage[slot].hi = 0;
            if (rec->size <= sizeof(value)) {
                memcpy(&arg_storage[slot].lo, &value, rec->size);
            } else {
                size_t hi_len = rec->size - sizeof(value);
                if (hi_len > sizeof(rec->value_hi)) hi_len = sizeof(rec->value_hi);
                memcpy(&arg_storage[slot].lo, &value, sizeof(value));
                memcpy(&arg_storage[slot].hi, &rec->value_hi, hi_len);
            }
            argv_out[argc++] = (uint64_t)(uintptr_t)&arg_storage[slot];
        }
    } else if (job->raw_params && job->raw_param_size) {
        size_t words = job->raw_param_size / sizeof(uint64_t);
        if (words > PACC_MAX_KERNEL_ARGS) {
            release_kernel_binding_maps(maps, map_count);
            g_kernel_arg_error = 0xe7;
            return -1;
        }
        for (size_t i = 0; i < words; i++) {
            uint64_t value = read_u64_le(job->raw_params + i * sizeof(uint64_t));
            size_t slot = argc;
            arg_storage[slot].lo = value;
            arg_storage[slot].hi = 0;
            argv_out[argc++] = (uint64_t)(uintptr_t)&arg_storage[slot];
        }
    }

    *argc_out = argc;
    *map_count_out = map_count;
    return 0;
}

static void trace_kernel_param_cells(const char *symbol,
                                     const struct PaccJobImage *job,
                                     const struct KernelParamCell *cells,
                                     const uint64_t *argv,
                                     size_t argc) {
    if (!symbol || !strstr(symbol, "k_bin_bcast")) return;
    if (!jobd_trace_enabled()) return;
    for (size_t i = 0; i < argc; i++) {
        uint32_t size = (job && job->arg_records && i < job->arg_count) ?
            job->arg_records[i].size : 0;
        uint32_t kind = (job && job->arg_records && i < job->arg_count) ?
            job->arg_records[i].kind : 0;
        uint32_t flags = (job && job->arg_records && i < job->arg_count) ?
            job->arg_records[i].flags : 0;
        const struct KernelParamCell *cell = NULL;
        if (cells && argv && argv[i] >= (uint64_t)(uintptr_t)&cells[0] &&
            argv[i] < (uint64_t)(uintptr_t)&cells[PACC_MAX_KERNEL_ARGS]) {
            cell = (const struct KernelParamCell *)(uintptr_t)argv[i];
        }
        trace_msg("kernel argcell[%zu]: argv=0x%" PRIx64
                  " cell=%p lo=0x%" PRIx64 " hi=0x%" PRIx64
                  " size=%u kind=%u flags=0x%x",
                  i, argv ? argv[i] : 0,
                  (const void *)cell,
                  cell ? cell->lo : 0,
                  cell ? cell->hi : 0,
                  size, kind, flags);
    }
}

static int parse_kernel_job_image_with_total(const uint8_t *image,
                                             size_t image_len,
                                             size_t total_len,
                                             struct PaccJobImage *out) {
    size_t abi_len = PACC_KERNEL_LAUNCH_ABI_WIRE_BYTES;
    g_kernel_parse_error = 0;
    if (total_len < image_len) {
        total_len = image_len;
    }

    if (!image || !out || image_len < PACC_JOB_IMAGE_HEADER_WIRE_BYTES) {
        log_msg("kernel image parse failed: image=%p out=%p len=%zu header=%zu",
                (const void *)image, (void *)out, image_len, (size_t)PACC_JOB_IMAGE_HEADER_WIRE_BYTES);
        g_kernel_parse_error = 0x01;
        return -1;
    }
    memset(out, 0, sizeof(*out));
    /*
     * The submit image is a wire format shared between Rust host code and the
     * PACC-side jobd. Decode it explicitly so compiler-specific struct padding
     * or uint64_t alignment cannot shift the following launch ABI header.
     */
    out->header.magic = read_u64_le(image + 0);
    out->header.version = read_u32_le(image + 8);
    out->header.flags = read_u32_le(image + 12);
    out->header.entry_offset = read_u64_le(image + 16);
    out->header.image_size = read_u64_le(image + 24);
    out->header.kernel_name_hash = read_u64_le(image + 32);
    out->header.grid_x = read_u32_le(image + 40);
    out->header.grid_y = read_u32_le(image + 44);
    out->header.grid_z = read_u32_le(image + 48);
    out->header.block_x = read_u32_le(image + 52);
    out->header.block_y = read_u32_le(image + 56);
    out->header.block_z = read_u32_le(image + 60);
    out->header.reserved = read_u32_le(image + 64);
    if (out->header.magic != PACC_JOB_MAGIC || out->header.version != PACC_JOB_VERSION) {
        log_msg("kernel image parse failed: bad header magic=0x%" PRIx64 " version=%u len=%zu",
                out->header.magic, out->header.version, image_len);
        g_kernel_parse_error = 0x02;
        return -1;
    }
    if (out->header.entry_offset > total_len ||
        out->header.image_size > total_len - out->header.entry_offset) {
        log_msg("kernel image parse failed: ELF bounds entry=0x%" PRIx64
                " image=0x%" PRIx64 " len=0x%zx total=0x%zx flags=0x%x hash=0x%" PRIx64,
                out->header.entry_offset, out->header.image_size, image_len, total_len,
                out->header.flags, out->header.kernel_name_hash);
        g_kernel_parse_error = 0x03;
        return -1;
    }
    out->elf_len = (size_t)out->header.image_size;
    if (out->header.entry_offset <= image_len &&
        out->header.image_size <= image_len - out->header.entry_offset) {
        out->elf = image + out->header.entry_offset;
    }

    if (out->header.flags & PACC_JOB_FLAG_HAS_LAUNCH_ABI) {
        if (image_len < PACC_JOB_IMAGE_HEADER_WIRE_BYTES + abi_len) {
            log_msg("kernel image parse failed: ABI header truncated len=%zu need=%zu",
                    image_len, (size_t)PACC_JOB_IMAGE_HEADER_WIRE_BYTES + abi_len);
            g_kernel_parse_error = 0x04;
            return -1;
        }
        const uint8_t *abi = image + PACC_JOB_IMAGE_HEADER_WIRE_BYTES;
        out->abi_storage.magic = read_u64_le(abi + 0);
        out->abi_storage.version = read_u32_le(abi + 8);
        out->abi_storage.flags = read_u32_le(abi + 12);
        out->abi_storage.arg_records_offset = read_u32_le(abi + 16);
        out->abi_storage.arg_record_count = read_u32_le(abi + 20);
        out->abi_storage.bindings_offset = read_u32_le(abi + 24);
        out->abi_storage.binding_count = read_u32_le(abi + 28);
        out->abi_storage.raw_param_offset = read_u32_le(abi + 32);
        out->abi_storage.raw_param_size = read_u32_le(abi + 36);
        out->abi_storage.kernel_name_offset = read_u32_le(abi + 40);
        out->abi_storage.kernel_name_size = read_u32_le(abi + 44);
        out->abi = &out->abi_storage;
        if (out->abi->magic != PACC_KERNEL_LAUNCH_ABI_MAGIC ||
            out->abi->version != PACC_KERNEL_LAUNCH_ABI_VERSION) {
            log_msg("kernel image parse failed: bad ABI magic=0x%" PRIx64
                    " version=%u arg_off=0x%x arg_count=%u bind_off=0x%x bind_count=%u"
                    " raw_off=0x%x raw_size=0x%x name_off=0x%x name_size=0x%x len=0x%zx",
                    out->abi->magic, out->abi->version,
                    out->abi->arg_records_offset, out->abi->arg_record_count,
                    out->abi->bindings_offset, out->abi->binding_count,
                    out->abi->raw_param_offset, out->abi->raw_param_size,
                    out->abi->kernel_name_offset, out->abi->kernel_name_size,
                    image_len);
            g_kernel_parse_error = 0x05;
            return -1;
        }
        if (out->abi->arg_record_count) {
            size_t bytes = (size_t)out->abi->arg_record_count * PACC_KERNEL_ARG_RECORD_WIRE_BYTES;
            if (out->abi->arg_record_count > PACC_MAX_KERNEL_ARGS) {
                log_msg("kernel image parse failed: too many arg records count=%u max=%u",
                        out->abi->arg_record_count, PACC_MAX_KERNEL_ARGS);
                g_kernel_parse_error = 0x0a;
                return -1;
            }
            if (!elf64_bounds_ok(out->abi->arg_records_offset, bytes, image_len)) {
                log_msg("kernel image parse failed: arg record bounds off=0x%x bytes=0x%zx len=0x%zx count=%u",
                        out->abi->arg_records_offset, bytes, image_len,
                        out->abi->arg_record_count);
                g_kernel_parse_error = 0x06;
                return -1;
            }
            for (uint32_t i = 0; i < out->abi->arg_record_count; i++) {
                const uint8_t *rec = image + out->abi->arg_records_offset +
                                     (size_t)i * PACC_KERNEL_ARG_RECORD_WIRE_BYTES;
                out->arg_records_storage[i].kind = read_u32_le(rec + 0);
                out->arg_records_storage[i].size = read_u32_le(rec + 4);
                out->arg_records_storage[i].flags = read_u32_le(rec + 8);
                out->arg_records_storage[i].reserved = read_u32_le(rec + 12);
                out->arg_records_storage[i].value = read_u64_le(rec + 16);
                out->arg_records_storage[i].value_hi = read_u64_le(rec + 24);
            }
            out->arg_records = out->arg_records_storage;
            out->arg_count = out->abi->arg_record_count;
        }
        if (out->abi->binding_count) {
            size_t bytes = (size_t)out->abi->binding_count * PACC_KERNEL_BUFFER_BINDING_WIRE_BYTES;
            if (out->abi->binding_count > PACC_MAX_KERNEL_BINDINGS) {
                log_msg("kernel image parse failed: too many bindings count=%u max=%u",
                        out->abi->binding_count, PACC_MAX_KERNEL_BINDINGS);
                g_kernel_parse_error = 0x0b;
                return -1;
            }
            if (!elf64_bounds_ok(out->abi->bindings_offset, bytes, image_len)) {
                log_msg("kernel image parse failed: binding bounds off=0x%x bytes=0x%zx len=0x%zx count=%u",
                        out->abi->bindings_offset, bytes, image_len,
                        out->abi->binding_count);
                g_kernel_parse_error = 0x07;
                return -1;
            }
            for (uint32_t i = 0; i < out->abi->binding_count; i++) {
                const uint8_t *binding = image + out->abi->bindings_offset +
                                         (size_t)i * PACC_KERNEL_BUFFER_BINDING_WIRE_BYTES;
                out->bindings_storage[i].arg_index = read_u32_le(binding + 0);
                out->bindings_storage[i].flags = read_u32_le(binding + 4);
                out->bindings_storage[i].addr = read_u64_le(binding + 8);
                out->bindings_storage[i].size = read_u64_le(binding + 16);
            }
            out->bindings = out->bindings_storage;
            out->binding_count = out->abi->binding_count;
        }
        if (out->abi->raw_param_size) {
            if (!elf64_bounds_ok(out->abi->raw_param_offset, out->abi->raw_param_size, image_len)) {
                log_msg("kernel image parse failed: raw param bounds off=0x%x size=0x%x len=0x%zx",
                        out->abi->raw_param_offset, out->abi->raw_param_size, image_len);
                g_kernel_parse_error = 0x08;
                return -1;
            }
            out->raw_params = image + out->abi->raw_param_offset;
            out->raw_param_size = out->abi->raw_param_size;
        }
        if (out->abi->kernel_name_size) {
            if (!elf64_bounds_ok(out->abi->kernel_name_offset, out->abi->kernel_name_size, image_len)) {
                log_msg("kernel image parse failed: kernel name bounds off=0x%x size=0x%x len=0x%zx",
                        out->abi->kernel_name_offset, out->abi->kernel_name_size, image_len);
                g_kernel_parse_error = 0x09;
                return -1;
            }
            out->kernel_name = (const char *)(image + out->abi->kernel_name_offset);
            out->kernel_name_size = out->abi->kernel_name_size;
        }
    }
    return 0;
}

static int parse_kernel_job_image(const uint8_t *image, size_t image_len, struct PaccJobImage *out) {
    return parse_kernel_job_image_with_total(image, image_len, image_len, out);
}

static void reset_kernel_job_image_view(struct Map *map, uint8_t **image_copy) {
    if (image_copy && *image_copy) {
        free(*image_copy);
        *image_copy = NULL;
    }
    if (map) {
        unmap_phys(map);
    }
}

static int load_kernel_job_image_view(int fd,
                                      const struct PaccJobDesc *desc,
                                      struct Map *map,
                                      uint8_t **image_copy,
                                      const uint8_t **image) {
    if (!desc || !map || !image_copy || !image) {
        return -1;
    }

    memset(map, 0, sizeof(*map));
    *image_copy = NULL;
    *image = NULL;

    if (read_phys_copy(fd, desc->addr, (size_t)desc->len, image_copy) == 0) {
        jobd_io_fence();
        *image = (const uint8_t *)*image_copy;
        return 0;
    }

    if (map_phys(fd, desc->addr, (size_t)desc->len, map) != 0) {
        return -1;
    }
    sync_map_for_cpu(map);
    jobd_io_fence();
    *image = (const uint8_t *)map->ptr;
    return 0;
}

static bool kernel_metadata_section_need(size_t *need,
                                         size_t entry_offset,
                                         uint32_t offset,
                                         uint32_t count,
                                         uint32_t wire_bytes) {
    size_t bytes;
    size_t end;

    if (!need || count == 0) {
        return true;
    }
    if (count > SIZE_MAX / wire_bytes) {
        return false;
    }
    bytes = (size_t)count * (size_t)wire_bytes;
    if ((size_t)offset > SIZE_MAX - bytes) {
        return false;
    }
    end = (size_t)offset + bytes;
    if (end > entry_offset) {
        return false;
    }
    if (end > *need) {
        *need = end;
    }
    return true;
}

static bool kernel_metadata_blob_need(size_t *need,
                                      size_t entry_offset,
                                      uint32_t offset,
                                      uint32_t size) {
    size_t end;

    if (!need || size == 0) {
        return true;
    }
    if ((size_t)offset > SIZE_MAX - (size_t)size) {
        return false;
    }
    end = (size_t)offset + (size_t)size;
    if (end > entry_offset) {
        return false;
    }
    if (end > *need) {
        *need = end;
    }
    return true;
}

static bool kernel_metadata_prefix_len_from_head(const uint8_t *head,
                                                 size_t head_len,
                                                 size_t total_len,
                                                 size_t *prefix_out) {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t entry_offset_u64;
    uint64_t image_size;
    size_t entry_offset;
    size_t need = PACC_JOB_IMAGE_HEADER_WIRE_BYTES + PACC_KERNEL_LAUNCH_ABI_WIRE_BYTES;
    const uint8_t *abi;
    uint64_t abi_magic;
    uint32_t abi_version;
    uint32_t arg_count;
    uint32_t binding_count;
    uint32_t raw_size;
    uint32_t name_size;
    uint32_t arg_off;
    uint32_t binding_off;
    uint32_t raw_off;
    uint32_t name_off;

    if (!head || !prefix_out || head_len < need) {
        return false;
    }
    magic = read_u64_le(head + 0);
    version = read_u32_le(head + 8);
    flags = read_u32_le(head + 12);
    entry_offset_u64 = read_u64_le(head + 16);
    image_size = read_u64_le(head + 24);
    if (magic != PACC_JOB_MAGIC || version != PACC_JOB_VERSION ||
        !(flags & PACC_JOB_FLAG_HAS_LAUNCH_ABI)) {
        return false;
    }
    if (entry_offset_u64 > (uint64_t)SIZE_MAX ||
        entry_offset_u64 > (uint64_t)total_len ||
        image_size > (uint64_t)total_len - entry_offset_u64) {
        return false;
    }
    entry_offset = (size_t)entry_offset_u64;
    if (entry_offset < need) {
        return false;
    }

    abi = head + PACC_JOB_IMAGE_HEADER_WIRE_BYTES;
    abi_magic = read_u64_le(abi + 0);
    abi_version = read_u32_le(abi + 8);
    if (abi_magic != PACC_KERNEL_LAUNCH_ABI_MAGIC ||
        abi_version != PACC_KERNEL_LAUNCH_ABI_VERSION) {
        return false;
    }

    arg_off = read_u32_le(abi + 16);
    arg_count = read_u32_le(abi + 20);
    binding_off = read_u32_le(abi + 24);
    binding_count = read_u32_le(abi + 28);
    raw_off = read_u32_le(abi + 32);
    raw_size = read_u32_le(abi + 36);
    name_off = read_u32_le(abi + 40);
    name_size = read_u32_le(abi + 44);

    if (arg_count > PACC_MAX_KERNEL_ARGS ||
        binding_count > PACC_MAX_KERNEL_BINDINGS) {
        return false;
    }
    if (!kernel_metadata_section_need(&need, entry_offset, arg_off, arg_count,
                                      PACC_KERNEL_ARG_RECORD_WIRE_BYTES) ||
        !kernel_metadata_section_need(&need, entry_offset, binding_off, binding_count,
                                      PACC_KERNEL_BUFFER_BINDING_WIRE_BYTES) ||
        !kernel_metadata_blob_need(&need, entry_offset, raw_off, raw_size) ||
        !kernel_metadata_blob_need(&need, entry_offset, name_off, name_size)) {
        return false;
    }

    *prefix_out = need;
    return true;
}

static int load_kernel_job_metadata_view(int fd,
                                         const struct PaccJobDesc *desc,
                                         uint8_t **image_copy,
                                         const uint8_t **image,
                                         size_t *image_len) {
    size_t total_len;
    size_t initial_len;
    size_t prefix_len = 0;
    uint8_t *head = NULL;
    uint8_t *prefix = NULL;

    if (!desc || !image_copy || !image || !image_len ||
        desc->len < PACC_JOB_IMAGE_HEADER_WIRE_BYTES + PACC_KERNEL_LAUNCH_ABI_WIRE_BYTES ||
        desc->len > (uint64_t)SIZE_MAX) {
        return -1;
    }
    *image_copy = NULL;
    *image = NULL;
    *image_len = 0;

    total_len = (size_t)desc->len;
    initial_len = total_len < 4096u ? total_len : 4096u;
    if (read_phys_copy(fd, desc->addr, initial_len, &head) != 0 || !head) {
        return -1;
    }
    if (!kernel_metadata_prefix_len_from_head(head, initial_len, total_len, &prefix_len)) {
        free(head);
        return -1;
    }
    if (prefix_len <= initial_len) {
        *image_copy = head;
        *image = head;
        *image_len = prefix_len;
        return 0;
    }

    free(head);
    if (read_phys_copy(fd, desc->addr, prefix_len, &prefix) != 0 || !prefix) {
        return -1;
    }
    *image_copy = prefix;
    *image = prefix;
    *image_len = prefix_len;
    return 0;
}

static bool kernel_job_parse_retryable(uint32_t detail) {
    switch (detail) {
    case 0x02: /* header magic/version not visible yet */
    case 0x03: /* ELF bounds/header fields not visible yet */
    case 0x05: /* launch ABI header not visible yet */
        return true;
    default:
        return false;
    }
}

static int dispatch_kernel_job_metadata_fast(int fd, const struct PaccJobDesc *desc) {
    struct PaccJobImage job;
    uint8_t *image_copy = NULL;
    const uint8_t *image = NULL;
    size_t image_len = 0;
    char symbol[256] = {0};

    if (!jobd_kernel_metadata_first_enabled()) {
        return 1;
    }
    if (!desc || desc->buf_info != PACC_JOB_MAGIC ||
        desc->len < PACC_JOB_IMAGE_HEADER_WIRE_BYTES + PACC_KERNEL_LAUNCH_ABI_WIRE_BYTES) {
        return 1;
    }

    if (load_kernel_job_metadata_view(fd, desc, &image_copy, &image, &image_len) != 0) {
        return 1;
    }
    if (parse_kernel_job_image_with_total(image, image_len, (size_t)desc->len, &job) != 0) {
        free(image_copy);
        return 1;
    }

    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x510e,
                      (uint32_t)(image_len & 0xffffffffu));
    if (job.kernel_name && job.kernel_name_size) {
        size_t copy_len = job.kernel_name_size;
        if (copy_len >= sizeof(symbol)) copy_len = sizeof(symbol) - 1u;
        memcpy(symbol, job.kernel_name, copy_len);
        symbol[copy_len] = '\0';
    }

    if (symbol[0] && jobd_generic_noop_enabled()) {
        trace_msg("generic kernel noop metadata-fast: seq=%" PRIu64 " symbol=%s",
                  desc->seq, symbol);
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x510f, 0);
        free(image_copy);
        return 0;
    }

    if (symbol[0]) {
        int direct_native_status =
            invoke_kernel_deepep_layout_native_direct(symbol, &job, desc->seq);
        if (direct_native_status <= 0) {
            uint32_t native_u = (uint32_t)direct_native_status;
            uint32_t error_status = (native_u & 0xffff0000u) == 0xffff0000u
                ? native_u
                : 0xffff5d00u;
            write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq,
                              direct_native_status == 0 ? 0x5d11u : error_status,
                              native_u);
            free(image_copy);
            return direct_native_status == 0 ? 0 : (int)error_status;
        }
    }

    if (symbol[0] && (strstr(symbol, "scale_f32") ||
                      strstr(symbol, "cpy_scalar"))) {
        uint64_t argv[PACC_MAX_KERNEL_ARGS];
        struct KernelParamCell arg_storage[PACC_MAX_KERNEL_ARGS];
        struct KernelBindingMap binding_maps[PACC_MAX_KERNEL_BINDINGS];
        size_t argc = 0;
        size_t binding_map_count = 0;
        bool is_cpy_scalar = strstr(symbol, "cpy_scalar") != NULL;
        memset(argv, 0, sizeof(argv));
        memset(arg_storage, 0, sizeof(arg_storage));
        memset(binding_maps, 0, sizeof(binding_maps));

        if (build_kernel_launch_args(fd, &job, symbol, argv, &argc, arg_storage,
                                     binding_maps, &binding_map_count) == 0) {
            int direct_native_status = is_cpy_scalar
                ? invoke_kernel_cpy_scalar_f32_native(symbol, argv, &job, argc)
                : invoke_kernel_scale_f32_native(symbol, argv, &job, argc);
            if (direct_native_status <= 0) {
                uint32_t native_u = (uint32_t)direct_native_status;
                uint32_t error_status = (native_u & 0xffff0000u) == 0xffff0000u
                    ? native_u
                    : 0xffff5c00u;
                write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq,
                                  direct_native_status == 0
                                      ? (is_cpy_scalar ? 0x5c20u : 0x5c10u)
                                      : error_status,
                                  native_u);
                release_kernel_binding_maps_native_fast(binding_maps,
                                                        binding_map_count,
                                                        direct_native_status == 0);
                free(image_copy);
                return direct_native_status == 0 ? 0 : (int)error_status;
            }
            release_kernel_binding_maps_native_fast(binding_maps,
                                                    binding_map_count,
                                                    false);
        } else {
            uint32_t detail = g_kernel_arg_error & 0x00ffu;
            trace_msg("metadata-fast native arg build miss: seq=%" PRIu64
                      " symbol=%s detail=0x%x; falling back to full image path",
                      desc->seq, symbol, detail);
        }
    }

    free(image_copy);
    return 1;
}

static int dispatch_kernel_job(int fd, const struct PaccJobDesc *desc) {
    struct Map map = {0};
    struct PaccJobImage job;
    uint8_t *image_copy = NULL;
    const uint8_t *image = NULL;
    uint64_t argv[PACC_MAX_KERNEL_ARGS];
    struct KernelParamCell arg_storage[PACC_MAX_KERNEL_ARGS];
    size_t argc = 0;
    struct KernelBindingMap binding_maps[PACC_MAX_KERNEL_BINDINGS];
    size_t binding_map_count = 0;
    char symbol[256] = {0};
    char artifact[PATH_MAX] = {0};
    struct LoadedKernelImage loaded;
    void *fn = NULL;
    PaccSetLaunchFn set_launch = NULL;
    int status = 0;
    uint64_t parse_retries = parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_PARSE_RETRIES", 4096);
    uint64_t parse_retry_us = parse_env_u64_default("HETGPU_PACC_JOBD_KERNEL_PARSE_RETRY_US", 1000);

    memset(binding_maps, 0, sizeof(binding_maps));
    memset(&loaded, 0, sizeof(loaded));
    if (!desc || desc->buf_info != PACC_JOB_MAGIC || desc->len < sizeof(struct PaccJobImageHeader)) {
        return 0xffff5001;
    }
    status = dispatch_kernel_job_metadata_fast(fd, desc);
    if (status <= 0) {
        return status;
    }
    status = 0;
    if (load_kernel_job_image_view(fd, desc, &map, &image_copy, &image) != 0) {
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5002u, errno ? (uint32_t)errno : 0);
        return 0xffff5002;
    }
    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5102);
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5102, image_copy ? 1U : 0U);
    for (uint64_t parse_attempt = 0;; parse_attempt++) {
        uint64_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
        if (desc->len >= 32) {
            q0 = read_u64_le(image + 0);
            q1 = read_u64_le(image + 8);
            q2 = read_u64_le(image + 16);
            q3 = read_u64_le(image + 24);
        }
        trace_msg("kernel image inspect: desc_addr=0x%" PRIx64 " len=0x%" PRIx64
                  " image_ptr=%p copied=%u q0=0x%" PRIx64 " q1=0x%" PRIx64
                  " q2=0x%" PRIx64 " q3=0x%" PRIx64,
                  desc->addr, desc->len, (const void *)image,
                  image_copy ? 1U : 0U, q0, q1, q2, q3);
        if (parse_kernel_job_image(image, (size_t)desc->len, &job) == 0) {
            break;
        }
        uint32_t detail = g_kernel_parse_error & 0x00ffu;
        if (!kernel_job_parse_retryable(detail) || parse_attempt >= parse_retries) {
            write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5300u | (detail ? detail : 0xffu), detail);
            reset_kernel_job_image_view(&map, &image_copy);
            return 0xffff5300u | (detail ? detail : 0xffu);
        }
        trace_msg("kernel image parse retry: seq=%" PRIu64 " detail=0x%x attempt=%" PRIu64 "/%" PRIu64
                  " retry_us=%" PRIu64,
                  desc->seq, detail, parse_attempt + 1, parse_retries, parse_retry_us);
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x510a, detail);
        reset_kernel_job_image_view(&map, &image_copy);
        if (parse_retry_us) {
            sleep_us(parse_retry_us);
        }
        if (load_kernel_job_image_view(fd, desc, &map, &image_copy, &image) != 0) {
            write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5002u, errno ? (uint32_t)errno : 0);
            return 0xffff5002;
        }
    }
    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5103);
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5103, (uint32_t)job.arg_count);

    trace_msg("kernel job seq=%" PRIu64 " hash=0x%" PRIx64 " name=%.*s elf=%zu args=%zu bindings=%zu grid=%ux%ux%u block=%ux%ux%u",
              desc->seq, job.header.kernel_name_hash,
              (int)job.kernel_name_size, job.kernel_name ? job.kernel_name : "",
              job.elf_len, job.arg_count, job.binding_count,
              job.header.grid_x, job.header.grid_y, job.header.grid_z,
              job.header.block_x, job.header.block_y, job.header.block_z);

    if (job.kernel_name && job.kernel_name_size) {
        size_t copy_len = job.kernel_name_size;
        if (copy_len >= sizeof(symbol)) copy_len = sizeof(symbol) - 1u;
        memcpy(symbol, job.kernel_name, copy_len);
        symbol[copy_len] = '\0';
    }

    if (symbol[0] && jobd_generic_noop_enabled()) {
        trace_msg("generic kernel noop: seq=%" PRIu64 " symbol=%s",
                  desc->seq, symbol);
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x510f, 0);
        reset_kernel_job_image_view(&map, &image_copy);
        return 0;
    }

    if (symbol[0]) {
        int direct_native_status = invoke_kernel_deepep_layout_native_direct(symbol, &job, desc->seq);
        if (direct_native_status <= 0) {
            uint32_t native_u = (uint32_t)direct_native_status;
            uint32_t error_status = (native_u & 0xffff0000u) == 0xffff0000u
                ? native_u
                : 0xffff5d00u;
            write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq,
                              direct_native_status == 0 ? 0x5d10u : error_status,
                              native_u);
            reset_kernel_job_image_view(&map, &image_copy);
            return direct_native_status == 0 ? 0 : (int)error_status;
        }
    }

    if (symbol[0]) {
        int embedding_status =
            invoke_kernel_index_select_embedding_packed_direct(fd, symbol, &job, desc->seq);
        if (embedding_status <= 0) {
            uint32_t native_u = (uint32_t)embedding_status;
            uint32_t error_status = (native_u & 0xffff0000u) == 0xffff0000u
                ? native_u
                : 0xffff5e80u;
            write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq,
                              embedding_status == 0 ? 0x5e20u : error_status,
                              native_u);
            reset_kernel_job_image_view(&map, &image_copy);
            return embedding_status == 0 ? 0 : (int)error_status;
        }
    }

    if (build_kernel_launch_args(fd, &job, symbol, argv, &argc, arg_storage, binding_maps, &binding_map_count) != 0) {
        uint32_t detail = g_kernel_arg_error & 0x00ffu;
        log_msg("kernel args build failed: seq=%" PRIu64 " symbol=%s detail=0x%x args=%zu bindings=%zu",
                desc->seq, symbol[0] ? symbol : "<unnamed>", detail,
                job.arg_count, job.binding_count);
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5200u | (detail ? detail : 0x05u), detail);
        free(image_copy);
        unmap_phys(&map);
        return 0xffff5200u | (detail ? detail : 0x05u);
    }
    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5104);
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5104, (uint32_t)argc);
    trace_kernel_param_cells(symbol, &job, arg_storage, argv, argc);

    if (symbol[0]) {
        g_current_kernel_seq = desc->seq;
        g_current_kernel_symbol = symbol;
        int native_status = invoke_kernel_native(symbol, argv, &job, argc);
        g_current_kernel_symbol = NULL;
        g_current_kernel_seq = 0;
        if (native_status <= 0) {
            uint32_t native_u = (uint32_t)native_status;
            uint32_t error_status = (native_u & 0xffff0000u) == 0xffff0000u
                ? native_u
                : 0xffff5007u;
            write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, error_status, native_u);
            release_kernel_binding_maps_native_fast(binding_maps,
                                                    binding_map_count,
                                                    native_status == 0);
            free(image_copy);
            unmap_phys(&map);
            return native_status == 0 ? 0 : (int)error_status;
        }
    }

    if (!jobd_elf_fallback_enabled()) {
        log_msg("kernel native miss: seq=%" PRIu64 " symbol=%s; direct ELF fallback disabled",
                desc->seq, symbol[0] ? symbol : "<unnamed>");
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5010u, 0);
        release_kernel_binding_maps(binding_maps, binding_map_count);
        free(image_copy);
        unmap_phys(&map);
        return 0xffff5010u;
    }

    if (load_kernel_image(job.elf, job.elf_len, job.header.kernel_name_hash,
                          job.kernel_name, job.kernel_name_size,
                          symbol, sizeof(symbol), artifact, sizeof(artifact), &loaded) != 0) {
        uint32_t detail = g_kernel_load_error & 0x0fffu;
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5000u | (detail ? detail : 0x004u), detail);
        release_kernel_binding_maps(binding_maps, binding_map_count);
        free(image_copy);
        unmap_phys(&map);
        return 0xffff5000u | (detail ? detail : 0x004u);
    }
    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5105);
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5105, (uint32_t)(job.elf_len & 0xffffffffu));

    fn = loaded.direct ? loaded.fn : dlsym(loaded.handle, symbol);
    if (!fn) {
        log_msg("dlsym(%s from %s) failed: %s", symbol, artifact,
                loaded.direct ? "direct ELF symbol missing" : dlerror());
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5006u, 0);
        release_kernel_binding_maps(binding_maps, binding_map_count);
        unload_kernel_image(&loaded);
        free(image_copy);
        unmap_phys(&map);
        return 0xffff5006;
    }
    set_launch = loaded.direct
                     ? loaded.set_launch
                     : (PaccSetLaunchFn)dlsym(loaded.handle, "f___zluda_ptx_impl_set_launch");

    trace_msg("kernel dispatch: seq=%" PRIu64 " symbol=%s argc=%zu artifact=%s logical_threads=%" PRIu64,
              desc->seq, symbol, argc, artifact,
              (uint64_t)pacc_nonzero_dim(job.header.grid_x) *
              pacc_nonzero_dim(job.header.grid_y) *
              pacc_nonzero_dim(job.header.grid_z) *
              pacc_nonzero_dim(job.header.block_x) *
              pacc_nonzero_dim(job.header.block_y) *
              pacc_nonzero_dim(job.header.block_z));
    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5106);
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5106, (uint32_t)argc);
    g_current_kernel_seq = desc->seq;
    g_current_kernel_symbol = symbol;
    status = invoke_kernel_symbol_grid(symbol, fn, argv, &job, argc, set_launch);
    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5107);
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5107, (uint32_t)status);
    trace_msg("kernel dispatch returned: seq=%" PRIu64 " symbol=%s status=%d",
              desc->seq, symbol, status);
    g_current_kernel_symbol = NULL;
    g_current_kernel_seq = 0;

    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5108);
    release_kernel_binding_maps(binding_maps, binding_map_count);
    mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5109);
    unload_kernel_image(&loaded);
    free(image_copy);
    unmap_phys(&map);
    if (status != 0) {
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0xffff5007u, (uint32_t)status);
        return 0xffff5007;
    }
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc->seq, 0x5109, 0);
    return 0;
}

static void *arg_payload_inner(volatile struct Doorbell *ctl, uint32_t job_id, uint64_t seq, size_t want, bool noisy) {
    int slot = arg_slot_for_job(job_id);
    if (slot < 0) return NULL;
    char *base = (char *)ctl;
    volatile struct ArgSlotHeader *h =
        (volatile struct ArgSlotHeader *)(base + HETGPU_PACC_ARG_BASE_OFF +
                                          (size_t)slot * HETGPU_PACC_ARG_SLOT_BYTES);
    __sync_synchronize();
    if (h->magic != HETGPU_PACC_JOB_MAGIC || h->version != HETGPU_PACC_JOB_VERSION ||
        h->job_id != job_id || h->seq != seq || h->arg_len < want) {
        if (noisy) {
            trace_msg("arg_payload mismatch: job_id=%u/%s seq=%" PRIu64 " slot=%d magic=0x%" PRIx64 " ver=%u hdr_job=%u hdr_seq=%" PRIu64 " arg_len=%" PRIu64 " want=%zu",
                      job_id, job_name(job_id), seq, slot, h->magic, h->version,
                      h->job_id, h->seq, h->arg_len, want);
        }
        return NULL;
    }
    return (void *)((char *)h + sizeof(*h));
}

static PACC_UNUSED void *arg_payload(volatile struct Doorbell *ctl, uint32_t job_id, uint64_t seq, size_t want) {
    return arg_payload_inner(ctl, job_id, seq, want, true);
}

static bool arg_payload_from_control(volatile struct Doorbell *ctl, uint32_t job_id, uint64_t seq, size_t want, void *out) {
    const void *dyn;
    if (!ctl || !out) {
        return false;
    }
    dyn = arg_payload_inner(ctl, job_id, seq, want, false);
    if (!dyn) {
        return false;
    }
    __sync_synchronize();
    memcpy(out, dyn, want);
    __sync_synchronize();
    return true;
}

static bool gemm_job_ready(const struct GemmJob *job) {
    return job && job->a_addr && job->b_addr && job->c_addr && job->m && job->n && job->k;
}

static bool softmax_job_ready(const struct SoftmaxJob *job) {
    return job && job->src_addr && job->dst_addr && job->rows && job->cols && job->stride;
}

static bool rmsnorm_job_ready(const struct RmsNormJob *job) {
    return job && job->x_addr && job->y_addr && job->rows && job->hidden &&
           dtype_size(job->dtype) != 0;
}

static bool allreduce_job_ready(const struct AllReduceJob *job) {
    return job && job->src_addr && job->dst_addr && job->count && job->nranks;
}

static bool mmvf_job_ready(const struct MmvfJob *job) {
    return job && job->x_addr && job->y_addr && job->dst_addr &&
           job->x_bytes && job->y_bytes && job->dst_bytes &&
           job->grid_x && job->grid_y && job->grid_z;
}

static int arg_payload_copy_inner(int fd, uint32_t job_id, uint64_t seq, size_t want, void *out, bool noisy) {
    int slot = arg_slot_for_job(job_id);
    uint64_t slot_off;
    uint8_t *slot_bytes = NULL;
    struct ArgSlotHeader header;

    if (slot < 0 || !out || want > HETGPU_PACC_ARG_SLOT_BYTES - sizeof(header)) {
        return -1;
    }

    slot_off = HETGPU_PACC_ARG_BASE_OFF + (uint64_t)slot * HETGPU_PACC_ARG_SLOT_BYTES;
    mirror_diag_progress_status(fd, job_id, seq, 0x5134);
    if (jobd_mbox_control_enabled()) {
        if (read_mbox_control_copy(fd,
                                   slot_off,
                                   HETGPU_PACC_ARG_SLOT_BYTES,
                                   &slot_bytes) != 0) {
            mirror_diag_progress_status(fd, job_id, seq, 0x51350001u);
            if (noisy) {
                trace_msg("arg_payload_copy mbox-control read failed: job_id=%u/%s"
                          " seq=%" PRIu64 " slot=%d off=0x%" PRIx64,
                          job_id, job_name(job_id), seq, slot, slot_off);
            }
        }
        if (slot_bytes) {
            memcpy(&header, slot_bytes, sizeof(header));
            if (header.magic != HETGPU_PACC_JOB_MAGIC ||
                header.version != HETGPU_PACC_JOB_VERSION ||
                (header.job_id != job_id && header.job_id != 0) ||
                header.seq != seq ||
                header.arg_len < want) {
                if (noisy && jobd_trace_enabled()) {
                    trace_msg("arg_payload_copy mbox-control stale slot: job_id=%u/%s"
                              " seq=%" PRIu64 " slot=%d magic=0x%" PRIx64
                              " ver=%u hdr_job=%u hdr_seq=%" PRIu64
                              " arg_len=%" PRIu64 " want=%zu; trying shared-DDR slot",
                              job_id, job_name(job_id), seq, slot, header.magic,
                              header.version, header.job_id, header.seq,
                              header.arg_len, want);
                }
                free(slot_bytes);
                slot_bytes = NULL;
            }
        }
    }
    if (!slot_bytes && g_control_window) {
        const uint8_t *mapped_slot = (const uint8_t *)g_control_window + slot_off;
        memcpy(&header, mapped_slot, sizeof(header));
        if (header.magic == HETGPU_PACC_JOB_MAGIC &&
            header.version == HETGPU_PACC_JOB_VERSION &&
            (header.job_id == job_id || header.job_id == 0) &&
            header.seq == seq &&
            header.arg_len >= want) {
            slot_bytes = malloc(HETGPU_PACC_ARG_SLOT_BYTES);
            if (slot_bytes) {
                memcpy(slot_bytes, mapped_slot, HETGPU_PACC_ARG_SLOT_BYTES);
                mirror_diag_progress_status(fd, job_id, seq, 0x513d);
            }
        }
    }
    for (size_t c = 0; !slot_bytes && c < 4; c++) {
        uint8_t *candidate = NULL;
        if (read_shared_ddr_control_copy_pread_candidate(fd, slot_off, HETGPU_PACC_ARG_SLOT_BYTES, c, &candidate) != 0) {
            continue;
        }
        memcpy(&header, candidate, sizeof(header));
        if (header.magic == HETGPU_PACC_JOB_MAGIC &&
            header.version == HETGPU_PACC_JOB_VERSION &&
            (header.job_id == job_id || header.job_id == 0) &&
            header.seq == seq &&
            header.arg_len >= want) {
            slot_bytes = candidate;
            mirror_diag_progress_status(fd, job_id, seq, 0x51390000u | (uint32_t)c);
            break;
        }
        free(candidate);
    }
    if (!slot_bytes &&
        read_phys_copy(fd,
                       shared_ddr_control_phys(slot_off, HETGPU_PACC_ARG_SLOT_BYTES),
                       HETGPU_PACC_ARG_SLOT_BYTES,
                       &slot_bytes) != 0) {
        mirror_diag_progress_status(fd, job_id, seq, 0x5135);
        if (noisy) {
            trace_msg("arg_payload_copy read failed: job_id=%u/%s seq=%" PRIu64
                      " slot=%d off=0x%" PRIx64,
                      job_id, job_name(job_id), seq, slot, slot_off);
        }
        return -1;
    }

    memcpy(&header, slot_bytes, sizeof(header));
    mirror_diag_progress_status(fd, job_id, seq, 0x5136);
    if (header.magic != HETGPU_PACC_JOB_MAGIC ||
        header.version != HETGPU_PACC_JOB_VERSION ||
        (header.job_id != job_id && header.job_id != 0) ||
        header.seq != seq ||
        header.arg_len < want) {
        mirror_diag_progress_status(fd, job_id, seq, 0x51370000u |
                                    (uint32_t)(header.arg_len & 0xffffu));
        if (noisy) {
            trace_msg("arg_payload_copy mismatch: job_id=%u/%s seq=%" PRIu64
                      " slot=%d magic=0x%" PRIx64 " ver=%u hdr_job=%u"
                      " hdr_seq=%" PRIu64 " arg_len=%" PRIu64 " want=%zu",
                      job_id, job_name(job_id), seq, slot, header.magic,
                      header.version, header.job_id, header.seq, header.arg_len, want);
        }
        free(slot_bytes);
        return -1;
    }

    memcpy(out, slot_bytes + sizeof(header), want);
    free(slot_bytes);
    mirror_diag_progress_status(fd, job_id, seq, 0x5138);
    return 0;
}

static int arg_payload_copy(int fd, uint32_t job_id, uint64_t seq, size_t want, void *out) {
    return arg_payload_copy_inner(fd, job_id, seq, want, out, true);
}

static int arg_payload_copy_wait(int fd, volatile struct Doorbell *ctl, uint32_t job_id, uint64_t seq, size_t want, void *out) {
    uint64_t start = monotonic_us();
    uint64_t timeout = jobd_arg_wait_us();
    uint64_t attempts = 0;

    mirror_diag_progress_status(fd, job_id, seq, 0x5130);
    while (timeout == 0 || monotonic_us() - start < timeout) {
        jobd_io_fence();

        /*
         * Prefer an explicit shared-DDR slot read over the mapped control
         * window.  The mapped view can still contain a matching header while
         * the payload cache line is stale after a previous kernel job.
         */
        if (arg_payload_copy_inner(fd, job_id, seq, want, out, false) == 0) {
            jobd_io_fence();
            mirror_diag_progress_status(fd, job_id, seq, 0x5132);
            if (attempts != 0) {
                trace_msg("arg_payload_copy_wait matched: job_id=%u/%s seq=%" PRIu64
                          " attempts=%" PRIu64 " elapsed_us=%" PRIu64,
                          job_id, job_name(job_id), seq, attempts, monotonic_us() - start);
            }
            return 0;
        }

        if (ctl) {
            const void *dyn = arg_payload_inner(ctl, job_id, seq, want, false);
            if (dyn) {
                memcpy(out, dyn, want);
                jobd_io_fence();
                if (attempts != 0) {
                    trace_msg("arg_payload_copy_wait matched mapped control: job_id=%u/%s seq=%" PRIu64
                              " attempts=%" PRIu64 " elapsed_us=%" PRIu64,
                              job_id, job_name(job_id), seq, attempts, monotonic_us() - start);
                }
                return 0;
            }
        }

        attempts++;
        jobd_io_fence();
        sleep_us(1000);
    }

    trace_msg("arg_payload_copy_wait timed out: job_id=%u/%s seq=%" PRIu64
              " attempts=%" PRIu64 " elapsed_us=%" PRIu64,
              job_id, job_name(job_id), seq, attempts, monotonic_us() - start);
    mirror_diag_progress_status(fd, job_id, seq, 0x5133);
    return arg_payload_copy(fd, job_id, seq, want, out);
}

static bool refresh_runtime_table(int fd, volatile struct Doorbell *ctl, struct PreloadedJobs *jobs, uint64_t *last_table_seq) {
    struct RuntimeJobTable local;
    uint8_t *table_bytes = NULL;
    bool have_copy = false;
    uint32_t table_source = 0xffffffffu;

    memset(&local, 0, sizeof(local));
    if (g_control_window) {
        volatile struct RuntimeJobTable *table =
            (volatile struct RuntimeJobTable *)(g_control_window + HETGPU_PACC_RUNTIME_TABLE_OFF);
        memcpy(&local, (const void *)table, sizeof(local));
        __sync_synchronize();
        have_copy =
            local.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC &&
            local.version == HETGPU_PACC_RUNTIME_TABLE_VERSION &&
            local.seq != 0 &&
            local.seq != *last_table_seq;
        table_source = 0xfffffffdu;
    }
    for (size_t c = 0; !have_copy && c < 4; c++) {
        uint8_t *candidate = NULL;
        if (read_shared_ddr_control_copy_pread_candidate(fd, HETGPU_PACC_RUNTIME_TABLE_OFF,
                                                         sizeof(local), c, &candidate) != 0) {
            continue;
        }
        memcpy(&local, candidate, sizeof(local));
        free(candidate);
        if (local.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC &&
            local.version == HETGPU_PACC_RUNTIME_TABLE_VERSION &&
            local.seq != 0 &&
            local.seq != *last_table_seq) {
            have_copy = true;
            table_source = (uint32_t)c;
            break;
        }
        memset(&local, 0, sizeof(local));
    }
    if (!have_copy &&
        read_phys_copy(fd,
                       shared_ddr_control_phys(HETGPU_PACC_RUNTIME_TABLE_OFF, sizeof(local)),
                       sizeof(local),
                       &table_bytes) == 0) {
        memcpy(&local, table_bytes, sizeof(local));
        free(table_bytes);
        table_source = 0xfffffffeu;
        have_copy =
            local.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC &&
            local.version == HETGPU_PACC_RUNTIME_TABLE_VERSION &&
            local.seq != 0 &&
            local.seq != *last_table_seq;
    } else {
        if (table_bytes) {
            free(table_bytes);
            table_bytes = NULL;
        }
    }
    if (!have_copy && g_control_window) {
        volatile struct RuntimeJobTable *table =
            (volatile struct RuntimeJobTable *)(g_control_window + HETGPU_PACC_RUNTIME_TABLE_OFF);
        memcpy(&local, (const void *)table, sizeof(local));
        __sync_synchronize();
        trace_msg("runtime table helper read failed; using mapped control window");
    } else if (!have_copy) {
        (void)ctl;
        trace_msg("runtime table helper read failed and mapped control window is unavailable");
    }

    if (local.magic != HETGPU_PACC_RUNTIME_TABLE_MAGIC ||
        local.version != HETGPU_PACC_RUNTIME_TABLE_VERSION ||
        local.seq == 0 ||
        local.seq == *last_table_seq) {
        return false;
    }

    jobs->have_gemm = local.have_gemm != 0;
    jobs->have_softmax = local.have_softmax != 0;
    jobs->have_rmsnorm = local.have_rmsnorm != 0;
    jobs->have_allreduce = local.have_allreduce != 0;
    jobs->have_mmvf = local.have_mmvf != 0;
    jobs->runtime_seq = local.seq;
    if (local.have_gemm) {
        jobs->gemm = local.gemm;
    }
    if (local.have_softmax) {
        jobs->softmax = local.softmax;
    }
    if (local.have_rmsnorm) {
        jobs->rmsnorm = local.rmsnorm;
    }
    if (local.have_allreduce) {
        jobs->allreduce = local.allreduce;
    }
    if (local.have_mmvf) {
        jobs->mmvf = local.mmvf;
    }
    *last_table_seq = local.seq;
	    trace_msg("runtime table seq=%" PRIu64 " have_gemm=%u have_softmax=%u have_rmsnorm=%u have_allreduce=%u have_mmvf=%u source=%s",
	              local.seq, local.have_gemm, local.have_softmax, local.have_rmsnorm,
	              local.have_allreduce, local.have_mmvf, have_copy ? "helper" : "mapped");
	    mirror_diag_event(fd, 0, local.seq, 0x52400000u | (uint32_t)(table_source & 0xffu),
	                      (uint32_t)(local.have_gemm |
	                                 (local.have_softmax << 1) |
	                                 (local.have_rmsnorm << 2) |
	                                 (local.have_allreduce << 3) |
	                                 (local.have_mmvf << 4)));
    if (local.have_gemm) {
        trace_msg("runtime table GEMM: m=%" PRIu64 " n=%" PRIu64 " k=%" PRIu64 " atype=%u btype=%u ctype=%u a=0x%" PRIx64 " b=0x%" PRIx64 " c=0x%" PRIx64,
                  local.gemm.m, local.gemm.n, local.gemm.k,
                  local.gemm.atype, local.gemm.btype, local.gemm.ctype,
                  local.gemm.a_addr, local.gemm.b_addr, local.gemm.c_addr);
    }
    if (local.have_mmvf) {
        trace_msg("runtime table MMVF: grid=%ux%ux%u ncols2=%d ncols_dst=%u x_type=%u x=0x%" PRIx64 " y=0x%" PRIx64 " dst=0x%" PRIx64,
                  local.mmvf.grid_x, local.mmvf.grid_y, local.mmvf.grid_z,
                  local.mmvf.ncols2, local.mmvf.ncols_dst, local.mmvf.x_type,
                  local.mmvf.x_addr, local.mmvf.y_addr, local.mmvf.dst_addr);
    }
    return true;
}

static int dispatch_job(int fd, volatile struct Doorbell *ctl, const struct PreloadedJobs *jobs, bool strict) {
    (void)strict;
    uint32_t job_id = ctl->job_id;
    uint64_t seq = ctl->seq;
    if (job_id == HETGPU_PACC_JOB_GEMM) {
        struct GemmJob copied;
        const struct GemmJob *job = NULL;
        mirror_progress_status(fd, job_id, seq, 0x5134);
        mirror_diag_progress_status(fd, job_id, seq, 0x51340000u |
                                    ((uint32_t)job_id & 0xffffu));
        if (jobs->have_gemm &&
            (jobs->runtime_seq == seq || jobd_static_config_fallback_enabled())) {
            job = &jobs->gemm;
            mirror_progress_status(fd, job_id, seq, 0x5135);
            mirror_diag_progress_status(fd, job_id, seq, 0x51350000u);
        } else if (arg_payload_from_control(ctl, job_id, seq, sizeof(copied), &copied)) {
            job = &copied;
            mirror_progress_status(fd, job_id, seq, 0x5136);
            mirror_diag_progress_status(fd, job_id, seq, 0x51360000u |
                                        ((uint32_t)copied.n & 0xffffu));
        } else if (arg_payload_copy_wait(fd, ctl, job_id, seq, sizeof(copied), &copied) == 0) {
            job = &copied;
            mirror_progress_status(fd, job_id, seq, 0x5137);
            mirror_diag_progress_status(fd, job_id, seq, 0x51370000u |
                                        ((uint32_t)copied.n & 0xffffu));
        }
        mirror_progress_status(fd, job_id, seq, 0x513701);
        if (!gemm_job_ready(job)) {
            mirror_progress_status(fd, job_id, seq, 0x5138);
            if (job) {
                mirror_diag_progress_status(fd, job_id, seq, 0x51380000u |
                                            ((uint32_t)job->atype & 0xffu) |
                                            (((uint32_t)job->btype & 0xffu) << 8) |
                                            (((uint32_t)job->ctype & 0xffu) << 16));
            }
            return -EAGAIN;
        }
        mirror_progress_status(fd, job_id, seq, 0x513702);
        if (job) {
            mirror_diag_progress_status(fd, job_id, seq, 0x51390000u |
                                        (((uint32_t)job->m & 0xffu) << 16) |
                                        (((uint32_t)job->n & 0xffu) << 8) |
                                        ((uint32_t)job->k & 0xffu));
        }
        mirror_progress_status(fd, job_id, seq, 0x5139);
        int gemm_status = run_gemm(fd, job, seq);
        mirror_progress_status(fd, job_id, seq, 0x513a);
        mirror_diag_progress_status(fd, job_id, seq, 0x513a0000u |
                                    ((uint32_t)gemm_status & 0xffffu));
        return gemm_status;
    }
    if (job_id == HETGPU_PACC_JOB_SOFTMAX) {
        struct SoftmaxJob copied;
        const struct SoftmaxJob *job = NULL;
        if (arg_payload_from_control(ctl, job_id, seq, sizeof(copied), &copied)) {
            job = &copied;
        } else if (arg_payload_copy_wait(fd, ctl, job_id, seq, sizeof(copied), &copied) == 0) {
            job = &copied;
        }
        if (!job && jobs->have_softmax &&
            (jobs->runtime_seq == seq || jobd_static_config_fallback_enabled())) {
            job = &jobs->softmax;
        }
        if (!softmax_job_ready(job)) {
            return -EAGAIN;
        }
        if (job) {
            trace_msg("dispatch SOFTMAX: seq=%" PRIu64 " rows=%" PRIu64 " cols=%" PRIu64 " stride=%" PRIu64 " dtype=%u",
                      seq, job->rows, job->cols, job->stride, job->dtype);
        }
        return run_softmax(fd, job, seq);
    }
    if (job_id == HETGPU_PACC_JOB_RMSNORM) {
        struct RmsNormJob copied;
        const struct RmsNormJob *job = NULL;
        if (arg_payload_copy_wait(fd, ctl, job_id, seq, sizeof(copied), &copied) == 0 &&
            rmsnorm_job_ready(&copied)) {
            job = &copied;
        } else if (arg_payload_from_control(ctl, job_id, seq, sizeof(copied), &copied) &&
                   rmsnorm_job_ready(&copied)) {
            job = &copied;
        }
        if (!job && jobs->have_rmsnorm &&
            (jobs->runtime_seq == seq || jobd_static_config_fallback_enabled())) {
            job = &jobs->rmsnorm;
        }
        if (!rmsnorm_job_ready(job)) {
            return -EAGAIN;
        }
        if (job) {
            trace_msg("dispatch RMSNORM: seq=%" PRIu64 " rows=%" PRIu64 " hidden=%" PRIu64 " dtype=%u eps=%g",
                      seq, job->rows, job->hidden, job->dtype, job->eps);
        }
        int rms_status = run_rmsnorm(fd, job, seq);
        mirror_rmsnorm_phase_record(fd, job, seq, 0x5140u, 0, (uint32_t)rms_status);
        if (rms_status == 0) {
            mirror_aligned_completion_record(fd, HETGPU_PACC_JOB_RMSNORM, seq, 0);
            submit_mbox_payload_sync(g_mbox_fd,
                                     HETGPU_PACC_JOB_RMSNORM,
                                     seq,
                                     0,
                                     "rmsnorm-final-completion");
        }
        return rms_status;
    }
    if (job_id == HETGPU_PACC_JOB_ALLREDUCE) {
        struct AllReduceJob copied;
        const struct AllReduceJob *job = NULL;
        if (arg_payload_from_control(ctl, job_id, seq, sizeof(copied), &copied)) {
            job = &copied;
        } else if (arg_payload_copy_wait(fd, ctl, job_id, seq, sizeof(copied), &copied) == 0) {
            job = &copied;
        }
        if (!job && jobs->have_allreduce &&
            (jobs->runtime_seq == seq || jobd_static_config_fallback_enabled())) {
            job = &jobs->allreduce;
        }
        if (!allreduce_job_ready(job)) {
            return -EAGAIN;
        }
        if (job) {
            trace_msg("dispatch ALLREDUCE: seq=%" PRIu64 " count=%" PRIu64 " nranks=%u dtype=%u op=%u",
                      seq, job->count, job->nranks, job->dtype, job->reduce_op);
        }
        return run_allreduce(fd, job);
    }
    if (job_id == HETGPU_PACC_JOB_MMVF) {
        struct MmvfJob copied;
        const struct MmvfJob *job = NULL;
        if (arg_payload_from_control(ctl, job_id, seq, sizeof(copied), &copied)) {
            job = &copied;
        } else if (arg_payload_copy_wait(fd, ctl, job_id, seq, sizeof(copied), &copied) == 0) {
            job = &copied;
        }
        if (!job && jobs->have_mmvf &&
            (jobs->runtime_seq == seq || jobd_static_config_fallback_enabled())) {
            job = &jobs->mmvf;
        }
        if (!mmvf_job_ready(job)) {
            return -EAGAIN;
        }
        if (job) {
            trace_msg("dispatch MMVF: seq=%" PRIu64 " grid=%ux%ux%u ncols2=%d ncols_dst=%u x_type=%u",
                      seq, job->grid_x, job->grid_y, job->grid_z,
                      job->ncols2, job->ncols_dst, job->x_type);
        }
        return run_mmvf(fd, job, seq);
    }
    return 0xffff00ff;
}

static bool host_completion_visible(int fd, uint32_t job_id, uint64_t seq, uint32_t status) {
    struct HostStatus seen;
    uint8_t *copy = NULL;
    uint64_t phys = shared_ddr_control_phys(HETGPU_PACC_COMPLETION_OFF, sizeof(seen));
    bool ok = false;

    memset(&seen, 0, sizeof(seen));
    if (read_phys_copy(fd, phys, sizeof(seen), &copy) == 0 && copy) {
        memcpy(&seen, copy, sizeof(seen));
        free(copy);
        ok = seen.magic == HETGPU_PACC_JOB_MAGIC &&
             seen.version == HETGPU_PACC_JOB_VERSION &&
             seen.job_id == job_id &&
             seen.seq == seq &&
             seen.status == status;
    }
    return ok;
}

static bool host_completion_seq_visible(int fd, uint32_t job_id, uint64_t seq) {
    struct HostStatus seen;
    uint8_t *copy = NULL;
    uint64_t phys = shared_ddr_control_phys(HETGPU_PACC_COMPLETION_OFF, sizeof(seen));
    bool ok = false;

    memset(&seen, 0, sizeof(seen));
    if (read_phys_copy(fd, phys, sizeof(seen), &copy) != 0 || !copy) {
        return false;
    }
    memcpy(&seen, copy, sizeof(seen));
    free(copy);
    ok = seen.magic == HETGPU_PACC_JOB_MAGIC &&
         seen.version == HETGPU_PACC_JOB_VERSION &&
         seen.job_id == job_id &&
         seen.seq == seq;
    return ok;
}

static bool kernel_completion_visible(int fd, uint64_t seq, uint32_t status) {
    return host_completion_visible(fd, PACC_KERNEL_JOB_ID, seq, status);
}

static bool kernel_completion_seq_visible(int fd, uint64_t seq) {
    return host_completion_seq_visible(fd, PACC_KERNEL_JOB_ID, seq);
}

static enum DispatchPollResult maybe_dispatch_kernel_job(
    int fd,
    volatile struct Doorbell *ctl,
    uint64_t *last_kernel_seq) {
    static uint64_t last_idle_report_seq;
    static uint64_t last_invalid_report_seq;
    static uint64_t pending_completion_seq;
    static uint32_t pending_completion_status;
    static bool pending_completion_valid;
    const struct PaccJobDesc *kernel_desc = (const struct PaccJobDesc *)(const void *)ctl;
    struct PaccJobDesc desc;
    int status;

    if (!kernel_desc) {
        return DISPATCH_INVALID;
    }

    desc.addr = kernel_desc->addr;
    desc.len = kernel_desc->len;
    desc.seq = kernel_desc->seq;
    desc.buf_info = kernel_desc->buf_info;
    __sync_synchronize();

    if (desc.buf_info != PACC_JOB_MAGIC ||
        desc.len < sizeof(struct PaccJobImageHeader)) {
        if (jobd_trace_enabled() && desc.seq != 0 && desc.seq != last_invalid_report_seq) {
            last_invalid_report_seq = desc.seq;
            trace_msg("kernel doorbell invalid: seq=%" PRIu64
                      " addr=0x%" PRIx64 " len=%" PRIu64
                      " buf_info=0x%" PRIx64 " last=%" PRIu64,
                      desc.seq, desc.addr, desc.len, desc.buf_info,
                      last_kernel_seq ? *last_kernel_seq : 0);
        }
        return DISPATCH_INVALID;
    }
    if (desc.seq == 0) {
        if (jobd_trace_enabled() && desc.seq != 0 && desc.seq != last_idle_report_seq) {
            last_idle_report_seq = desc.seq;
            trace_msg("kernel doorbell idle: seq=%" PRIu64
                      " addr=0x%" PRIx64 " len=%" PRIu64
                      " last=%" PRIu64,
                      desc.seq, desc.addr, desc.len,
                      last_kernel_seq ? *last_kernel_seq : 0);
        }
        return DISPATCH_IDLE;
    }
    if (desc.seq == *last_kernel_seq &&
        kernel_completion_seq_visible(fd, desc.seq)) {
        if (jobd_trace_enabled() && desc.seq != 0 && desc.seq != last_idle_report_seq) {
            last_idle_report_seq = desc.seq;
            trace_msg("kernel doorbell idle: seq=%" PRIu64
                      " addr=0x%" PRIx64 " len=%" PRIu64
                      " last=%" PRIu64,
                      desc.seq, desc.addr, desc.len,
                      last_kernel_seq ? *last_kernel_seq : 0);
        }
        return DISPATCH_IDLE;
    }
    if (desc.seq == *last_kernel_seq) {
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc.seq, 0x510d, 0);
    }

    if (pending_completion_valid && pending_completion_seq == desc.seq) {
        status = (int)pending_completion_status;
        trace_msg("retry kernel completion publish: seq=%" PRIu64 " status=0x%x",
                  desc.seq, (uint32_t)status);
    } else {
        if (pending_completion_valid && pending_completion_seq != desc.seq) {
            trace_msg("dropping stale pending kernel completion: old_seq=%" PRIu64
                      " new_seq=%" PRIu64,
                      pending_completion_seq, desc.seq);
            pending_completion_valid = false;
        }
        g_kernel_completion_beacon_sticky = false;
        trace_msg("new kernel doorbell: seq=%" PRIu64 " addr=0x%" PRIx64 " len=%" PRIu64,
                  desc.seq, desc.addr, desc.len);
        mirror_progress_status(fd, PACC_KERNEL_JOB_ID, desc.seq, 0x5101);
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc.seq, 0x5101, (uint32_t)(desc.len & 0xffffffffu));
        status = dispatch_kernel_job(fd, &desc);
        pending_completion_seq = desc.seq;
        pending_completion_status = (uint32_t)status;
        pending_completion_valid = true;
    }
    submit_mbox_payload_sync(g_mbox_fd, PACC_KERNEL_JOB_ID, desc.seq,
                             (uint32_t)status, "kernel-before-completion");
    write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc.seq, (uint32_t)status, 0);
    mirror_host_status(fd, PACC_KERNEL_JOB_ID, desc.seq, (uint32_t)status);
    submit_mbox_payload_sync(g_mbox_fd, PACC_KERNEL_JOB_ID, desc.seq,
                             (uint32_t)status, "kernel-completion");
    if (!jobd_require_completion_visible_enabled() ||
        kernel_completion_visible(fd, desc.seq, (uint32_t)status)) {
        if (last_kernel_seq) {
            *last_kernel_seq = desc.seq;
        }
        pending_completion_valid = false;
        g_kernel_completion_beacon_sticky = jobd_sticky_kernel_completion_enabled();
        if (g_kernel_completion_beacon_sticky) {
            g_kernel_completion_beacon_seq = desc.seq;
            g_kernel_completion_beacon_status = (uint32_t)status;
        }
        if ((uint32_t)status == 0) {
            for (unsigned i = 0; i < 3; i++) {
                write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc.seq, 0x511b, 0);
                sleep_us(1000);
            }
        } else {
            write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc.seq, 0x511b, (uint32_t)status);
        }
    } else {
        write_jobd_beacon(fd, PACC_KERNEL_JOB_ID, desc.seq, 0x511a, (uint32_t)status);
        trace_msg("kernel completion not visible after mirror: seq=%" PRIu64 " status=0x%x; will retry",
                  desc.seq, (uint32_t)status);
        sleep_us(1000);
    }
    g_response_irq_pending = true;
    trace_msg("kernel dispatch done: seq=%" PRIu64 " status=0x%x",
              desc.seq, (uint32_t)status);
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
    if (ctl->seq == *last_seq || preloaded_job_seen(ctl->job_id, ctl->seq)) {
        return DISPATCH_IDLE;
    }

    uint64_t seq = ctl->seq;
    job_id = ctl->job_id;
    if (g_preloaded_completion_sticky && g_preloaded_completion_seq != seq) {
        g_preloaded_completion_sticky = false;
    }
    ctl->status = 1;
    __sync_synchronize();
    mirror_diag_progress_status(fd, job_id, seq, 0x510e);
    log_msg("new doorbell: job_id=%u/%s seq=%" PRIu64,
            job_id, job_name(job_id), seq);
    trace_msg("new doorbell: job_id=%u/%s seq=%" PRIu64,
              job_id, job_name(job_id), seq);
    mirror_diag_progress_status(fd, job_id, seq, 0x510f);
    mirror_progress_status(fd, job_id, seq, 0x5110);
    if (jobd_runtime_table_refresh_enabled()) {
        refresh_runtime_table(fd, ctl, jobs, last_table_seq);
    }
    mirror_progress_status(fd, job_id, seq, 0x5111);
    if (jobd_preloaded_noop_enabled(job_id)) {
        ctl->status = 0;
        __sync_synchronize();
        mirror_host_status(fd, job_id, seq, 0);
        submit_mbox_payload_sync(g_mbox_fd, job_id, seq, 0,
                                 "preloaded-noop-completion");
        if (!jobd_require_completion_visible_enabled() ||
            host_completion_visible(fd, job_id, seq, 0)) {
            *last_seq = seq;
            mark_preloaded_job_seen(job_id, seq);
            g_preloaded_completion_sticky = jobd_sticky_preloaded_completion_enabled();
            g_preloaded_completion_job_id = job_id;
            g_preloaded_completion_seq = seq;
            g_preloaded_completion_status = 0;
            write_jobd_beacon(fd, job_id, seq, 0x511f, 0);
            g_response_irq_pending = true;
            log_msg("preloaded noop complete: job_id=%u/%s seq=%" PRIu64,
                    job_id, job_name(job_id), seq);
            trace_msg("preloaded noop complete: job_id=%u/%s seq=%" PRIu64,
                      job_id, job_name(job_id), seq);
            return DISPATCH_HANDLED;
        }
        write_jobd_beacon(fd, job_id, seq, 0x511a, 0);
        trace_msg("preloaded noop completion not visible after mirror: job_id=%u/%s seq=%" PRIu64,
                  job_id, job_name(job_id), seq);
        sleep_us(1000);
        return DISPATCH_IDLE;
    }
    trace_msg("dispatch enter: job_id=%u/%s seq=%" PRIu64,
              job_id, job_name(job_id), seq);
    mirror_progress_status(fd, job_id, seq, 0x5112);
    status = dispatch_job(fd, ctl, jobs, strict);
    if (status == -EAGAIN) {
        log_msg("dispatch pending args: job_id=%u/%s seq=%" PRIu64,
                job_id, job_name(job_id), seq);
        trace_msg("dispatch pending args: job_id=%u/%s seq=%" PRIu64,
                  job_id, job_name(job_id), seq);
        return DISPATCH_IDLE;
    }
    __sync_synchronize();
    ctl->status = (uint32_t)status;
    if (env_flag_true(getenv("HETGPU_PACC_JOBD_COMPLETION_SYNC"))) {
        submit_mbox_payload_sync(g_mbox_fd, job_id, seq, (uint32_t)status,
                                 "preloaded-before-completion");
    }
    mirror_aligned_completion_record(fd, job_id, seq, (uint32_t)status);
    mirror_job_completion(fd, job_id, seq, (uint32_t)status);
    submit_mbox_payload_sync(g_mbox_fd, job_id, seq, (uint32_t)status,
                             "preloaded-completion");
    if (!jobd_require_completion_visible_enabled() ||
        host_completion_visible(fd, job_id, seq, (uint32_t)status)) {
        *last_seq = seq;
        mark_preloaded_job_seen(job_id, seq);
        g_preloaded_completion_sticky = jobd_sticky_preloaded_completion_enabled();
        g_preloaded_completion_job_id = job_id;
        g_preloaded_completion_seq = seq;
        g_preloaded_completion_status = (uint32_t)status;
        g_response_irq_pending = true;
        log_msg("dispatch done: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                job_id, job_name(job_id), seq, (uint32_t)status);
        trace_msg("dispatch done: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                  job_id, job_name(job_id), seq, (uint32_t)status);
        return DISPATCH_HANDLED;
    }
    write_jobd_beacon(fd, job_id, seq, 0x511a, (uint32_t)status);
    trace_msg("dispatch completion not visible after mirror: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
              job_id, job_name(job_id), seq, (uint32_t)status);
    sleep_us(1000);
    return DISPATCH_IDLE;
}

static enum DispatchPollResult maybe_dispatch_gemm_arg_slot_direct(int fd,
                                                                   uint64_t *last_seq) {
    const uint32_t job_id = HETGPU_PACC_JOB_GEMM;
    int slot = arg_slot_for_job(job_id);
    uint64_t slot_rel_off;
    uint64_t slot_off;
    uint64_t payload_off;
    struct ArgSlotHeader header;
    struct GemmJob job;
    int status;

    if (slot < 0 || !g_ddr_info.ddr_base || !g_ddr_info.ddr_size ||
        g_pacc_id >= HETGPU_PACC_COUNT) {
        return DISPATCH_IDLE;
    }
    slot_rel_off = HETGPU_PACC_ARG_BASE_OFF +
                   (uint64_t)slot * HETGPU_PACC_ARG_SLOT_BYTES;
    slot_off = shared_ddr_control_rel(g_pacc_id, slot_rel_off);
    if (slot_off > g_ddr_info.ddr_size ||
        HETGPU_PACC_ARG_SLOT_BYTES > g_ddr_info.ddr_size - slot_off ||
        sizeof(job) > HETGPU_PACC_ARG_SLOT_BYTES) {
        return DISPATCH_IDLE;
    }

    memset(&header, 0, sizeof(header));
    {
        uint8_t *header_copy = NULL;
        uint64_t header_phys = g_ddr_info.ddr_base + slot_off;
        if (read_shared_ddr_control_copy_pread_for_pacc(fd,
                                                        (uint32_t)g_pacc_id,
                                                        slot_rel_off,
                                                        sizeof(header),
                                                        &header_copy) == 0 &&
            header_copy) {
            memcpy(&header, header_copy, sizeof(header));
        } else {
            free(header_copy);
            header_copy = NULL;
            if (read_phys_copy(fd, header_phys, sizeof(header), &header_copy) == 0 &&
                header_copy) {
                memcpy(&header, header_copy, sizeof(header));
            } else if (!read_current_control_window_bytes((uint32_t)g_pacc_id,
                                                          slot_rel_off,
                                                          &header,
                                                          sizeof(header))) {
                trace_msg("gemm arg-slot direct header read failed pacc=%" PRIu64 " off=0x%" PRIx64,
                          g_pacc_id, slot_rel_off);
                free(header_copy);
                return DISPATCH_IDLE;
            }
        }
        free(header_copy);
    }
    if (!((header.magic == HETGPU_PACC_JOB_MAGIC ||
           header.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC) &&
          header.version == HETGPU_PACC_JOB_VERSION &&
          header.seq != 0)) {
        uint8_t *header_copy = NULL;
        uint64_t header_phys = g_ddr_info.ddr_base + slot_off;
        if (read_phys_copy(fd, header_phys, sizeof(header), &header_copy) == 0 &&
            header_copy) {
            memcpy(&header, header_copy, sizeof(header));
        }
        free(header_copy);
    }
    if (!((header.magic == HETGPU_PACC_JOB_MAGIC ||
           header.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC) &&
          header.version == HETGPU_PACC_JOB_VERSION &&
          header.seq != 0)) {
            trace_msg("gemm arg-slot direct header read failed pacc=%" PRIu64 " off=0x%" PRIx64,
                      g_pacc_id, slot_rel_off);
            return DISPATCH_IDLE;
    }
    if (!((header.magic == HETGPU_PACC_JOB_MAGIC ||
           header.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC) &&
          header.version == HETGPU_PACC_JOB_VERSION &&
          header.seq != 0)) {
        trace_msg("gemm arg-slot direct header invalid magic=0x%" PRIx64
                  " version=%u job_id=%u seq=%" PRIu64 " arg_len=%" PRIu64,
                  header.magic, header.version, header.job_id, header.seq, header.arg_len);
        return DISPATCH_IDLE;
    }
    mirror_progress_status(fd, job_id, header.seq, 0x5300);
    if (preloaded_job_seen(job_id, header.seq)) {
        return DISPATCH_IDLE;
    }

    payload_off = header.magic == HETGPU_PACC_JOB_MAGIC ? 0x20ULL : 0x38ULL;
    if (payload_off > HETGPU_PACC_ARG_SLOT_BYTES ||
        sizeof(job) > HETGPU_PACC_ARG_SLOT_BYTES - payload_off) {
        return DISPATCH_IDLE;
    }
    memset(&job, 0, sizeof(job));
    mirror_progress_status(fd, job_id, header.seq, 0x530001);
    {
        uint8_t *job_copy = NULL;
        uint64_t job_phys = g_ddr_info.ddr_base + slot_off + payload_off;
        if (read_shared_ddr_control_copy_pread_for_pacc(fd,
                                                        (uint32_t)g_pacc_id,
                                                        slot_rel_off + payload_off,
                                                        sizeof(job),
                                                        &job_copy) == 0 &&
            job_copy) {
            memcpy(&job, job_copy, sizeof(job));
        } else {
            free(job_copy);
            job_copy = NULL;
            if (read_phys_copy(fd, job_phys, sizeof(job), &job_copy) == 0 &&
                job_copy) {
                memcpy(&job, job_copy, sizeof(job));
            } else if (!read_current_control_window_bytes((uint32_t)g_pacc_id,
                                                          slot_rel_off + payload_off,
                                                          &job,
                                                          sizeof(job))) {
                trace_msg("gemm arg-slot direct payload read failed pacc=%" PRIu64
                          " off=0x%" PRIx64 " seq=%" PRIu64,
                          g_pacc_id, slot_rel_off + payload_off, header.seq);
                free(job_copy);
                mirror_progress_status(fd, job_id, header.seq, 0xffff5301u);
                return DISPATCH_IDLE;
            }
        }
        free(job_copy);
    }
    if (!gemm_job_ready(&job)) {
        uint8_t *job_copy = NULL;
        uint64_t job_phys = g_ddr_info.ddr_base + slot_off + payload_off;
        if (read_phys_copy(fd, job_phys, sizeof(job), &job_copy) == 0 &&
            job_copy) {
            memcpy(&job, job_copy, sizeof(job));
        }
        free(job_copy);
    }
    if (!gemm_job_ready(&job)) {
            trace_msg("gemm arg-slot direct payload read failed pacc=%" PRIu64
                      " off=0x%" PRIx64 " seq=%" PRIu64,
                      g_pacc_id, slot_rel_off + payload_off, header.seq);
            mirror_progress_status(fd, job_id, header.seq, 0xffff5301u);
            return DISPATCH_IDLE;
    }
    mirror_progress_status(fd, job_id, header.seq, 0x5301);
    if (!gemm_job_ready(&job)) {
        trace_msg("gemm arg-slot direct job not ready seq=%" PRIu64
                  " m=%" PRIu64 " n=%" PRIu64 " k=%" PRIu64
                  " a=0x%" PRIx64 " b=0x%" PRIx64 " c=0x%" PRIx64
                  " dtype=%u/%u/%u compute=%u lda=%" PRId64 " ldb=%" PRId64 " ldc=%" PRId64,
                  header.seq, job.m, job.n, job.k, job.a_addr, job.b_addr, job.c_addr,
                  job.atype, job.btype, job.ctype, job.compute_type, job.lda, job.ldb, job.ldc);
        mirror_progress_status(fd, job_id, header.seq, 0xffff5302u);
        return DISPATCH_IDLE;
    }

    trace_msg("gemm arg-slot direct dispatch seq=%" PRIu64
              " m=%" PRIu64 " n=%" PRIu64 " k=%" PRIu64
              " a=0x%" PRIx64 " b=0x%" PRIx64 " c=0x%" PRIx64,
              header.seq, job.m, job.n, job.k, job.a_addr, job.b_addr, job.c_addr);
    mirror_progress_status(fd, job_id, header.seq, 0x5302);
    status = run_gemm(fd, &job, header.seq);
    mirror_progress_status(fd, job_id, header.seq, 0x5303);
    mirror_aligned_completion_record(fd, job_id, header.seq, (uint32_t)status);
    mirror_job_completion(fd, job_id, header.seq, (uint32_t)status);
    submit_mbox_payload_sync(g_mbox_fd, job_id, header.seq, (uint32_t)status,
                             "arg-slot-direct-completion");
    mark_preloaded_job_seen(job_id, header.seq);
    if (last_seq) {
        *last_seq = header.seq;
    }
    g_preloaded_completion_sticky = false;
    g_response_irq_pending = true;
    write_jobd_beacon(fd, job_id, header.seq, status == 0 ? 0x5304 : (uint32_t)status,
                      (uint32_t)status);
    return DISPATCH_HANDLED;
}

static enum DispatchPollResult maybe_dispatch_arg_slot_job(
    int fd,
    struct PreloadedJobs *jobs,
    bool strict,
    uint64_t *last_seq,
    uint64_t *last_table_seq) {
    struct ArgSlotHeader header;
    volatile struct Doorbell *ctl;
    int status;

    memset(&header, 0, sizeof(header));
    if (!find_pending_arg_slot_job(fd, &header)) {
        return DISPATCH_IDLE;
    }
    select_pacc_id_from_arg_slot_candidate(g_pending_arg_header_pacc_id,
                                           header.job_id,
                                           header.seq);

    memset(g_arg_slot_synthetic_control, 0, sizeof(g_arg_slot_synthetic_control));
    ctl = (volatile struct Doorbell *)(void *)g_arg_slot_synthetic_control;
    ctl->magic = HETGPU_PACC_JOB_MAGIC;
    ctl->version = HETGPU_PACC_JOB_VERSION;
    ctl->job_id = header.job_id;
    ctl->flags = 0;
    ctl->status = 0;
    ctl->seq = header.seq;
    jobd_io_fence();

    log_msg("arg-slot dispatch recovery: job_id=%u/%s seq=%" PRIu64
            " arg_len=%" PRIu64,
            header.job_id, job_name(header.job_id), header.seq, header.arg_len);
    trace_msg("arg-slot dispatch recovery: job_id=%u/%s seq=%" PRIu64
              " arg_len=%" PRIu64,
              header.job_id, job_name(header.job_id), header.seq, header.arg_len);
    mirror_progress_status(fd, header.job_id, header.seq, 0x5113);
    if (jobd_runtime_table_refresh_enabled()) {
        refresh_runtime_table(fd, ctl, jobs, last_table_seq);
    }
    mirror_progress_status(fd, header.job_id, header.seq, 0x5114);

    status = dispatch_job(fd, ctl, jobs, strict);
    if (status == -EAGAIN) {
        log_msg("arg-slot dispatch pending args: job_id=%u/%s seq=%" PRIu64,
                header.job_id, job_name(header.job_id), header.seq);
        trace_msg("arg-slot dispatch pending args: job_id=%u/%s seq=%" PRIu64,
                  header.job_id, job_name(header.job_id), header.seq);
        return DISPATCH_IDLE;
    }

    ctl->status = (uint32_t)status;
    if (env_flag_true(getenv("HETGPU_PACC_JOBD_COMPLETION_SYNC"))) {
        submit_mbox_payload_sync(g_mbox_fd, header.job_id, header.seq,
                                 (uint32_t)status, "arg-slot-before-completion");
    }
    mirror_aligned_completion_record(fd, header.job_id, header.seq, (uint32_t)status);
    mirror_job_completion(fd, header.job_id, header.seq, (uint32_t)status);
    submit_mbox_payload_sync(g_mbox_fd, header.job_id, header.seq, (uint32_t)status,
                             "arg-slot-completion");
    if (!jobd_require_completion_visible_enabled() ||
        host_completion_visible(fd, header.job_id, header.seq, (uint32_t)status)) {
        if (last_seq) {
            *last_seq = header.seq;
        }
        mark_preloaded_job_seen(header.job_id, header.seq);
        g_preloaded_completion_sticky = jobd_sticky_preloaded_completion_enabled();
        g_preloaded_completion_job_id = header.job_id;
        g_preloaded_completion_seq = header.seq;
        g_preloaded_completion_status = (uint32_t)status;
        g_response_irq_pending = true;
        log_msg("arg-slot dispatch done: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                header.job_id, job_name(header.job_id), header.seq, (uint32_t)status);
        trace_msg("arg-slot dispatch done: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                  header.job_id, job_name(header.job_id), header.seq, (uint32_t)status);
        return DISPATCH_HANDLED;
    }
    write_jobd_beacon(fd, header.job_id, header.seq, 0x511a, (uint32_t)status);
    trace_msg("arg-slot completion not visible after mirror: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
              header.job_id, job_name(header.job_id), header.seq, (uint32_t)status);
    sleep_us(1000);
    return DISPATCH_IDLE;
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
    trace_msg("dispatch_any_job preloaded result=%d last_seq=%" PRIu64
              " last_table_seq=%" PRIu64 " last_kernel_seq=%" PRIu64,
              result,
              last_seq ? *last_seq : 0,
              last_table_seq ? *last_table_seq : 0,
              last_kernel_seq ? *last_kernel_seq : 0);
    if (result == DISPATCH_HANDLED) {
        return result;
    }
    if (result == DISPATCH_IDLE) {
        result = maybe_dispatch_arg_slot_job(fd, jobs, strict, last_seq, last_table_seq);
        return result == DISPATCH_HANDLED ? result : DISPATCH_IDLE;
    }

    result = maybe_dispatch_kernel_job(fd, ctl, last_kernel_seq);
    trace_msg("dispatch_any_job kernel result=%d last_seq=%" PRIu64
              " last_table_seq=%" PRIu64 " last_kernel_seq=%" PRIu64,
              result,
              last_seq ? *last_seq : 0,
              last_table_seq ? *last_table_seq : 0,
              last_kernel_seq ? *last_kernel_seq : 0);
    if (result == DISPATCH_HANDLED) {
        return result;
    }
    if (result == DISPATCH_IDLE) {
        result = maybe_dispatch_arg_slot_job(fd, jobs, strict, last_seq, last_table_seq);
        return result == DISPATCH_HANDLED ? result : DISPATCH_IDLE;
    }

    result = maybe_dispatch_arg_slot_job(fd, jobs, strict, last_seq, last_table_seq);
    trace_msg("dispatch_any_job arg_slot result=%d last_seq=%" PRIu64
              " last_table_seq=%" PRIu64 " last_kernel_seq=%" PRIu64,
              result,
              last_seq ? *last_seq : 0,
              last_table_seq ? *last_table_seq : 0,
              last_kernel_seq ? *last_kernel_seq : 0);
    return result == DISPATCH_HANDLED ? result : DISPATCH_INVALID;
}

static void maybe_heartbeat_control(int fd,
                                    const uint8_t *snapshot,
                                    uint64_t tick,
                                    uint64_t last_kernel_seq) {
    const volatile struct Doorbell *head;
    const struct PaccJobDesc *kernel_head;
    uint32_t status = 0x7000;
    uint32_t job_id = 0;
    uint64_t seq = 0;

    if (!jobd_heartbeat_enabled() || !snapshot || (tick % 16) != 0) {
        return;
    }

    head = (const volatile struct Doorbell *)(const void *)snapshot;
    kernel_head = (const struct PaccJobDesc *)(const void *)snapshot;
    if (head->magic == HETGPU_PACC_JOB_MAGIC &&
        head->version == HETGPU_PACC_JOB_VERSION) {
        status = 0x7001;
        job_id = head->job_id;
        seq = head->seq;
    } else if (kernel_head->buf_info == PACC_JOB_MAGIC) {
        status = kernel_head->seq == last_kernel_seq ? 0x7102 : 0x7202;
        job_id = PACC_KERNEL_JOB_ID;
        seq = kernel_head->seq;
    } else {
        job_id = PACC_KERNEL_JOB_ID;
        seq = kernel_head->seq;
        status = 0x7000;
        if (kernel_head->addr != 0) status |= 0x1;
        if (kernel_head->len != 0) status |= 0x2;
        if (kernel_head->seq != 0) status |= 0x4;
        if (kernel_head->buf_info != 0) status |= 0x8;
    }
    mirror_progress_status(fd, job_id, seq, status);
}

static bool control_has_pending_job(int fd,
                                    uint64_t last_seq,
                                    uint64_t last_kernel_seq) {
    uint8_t header[sizeof(struct PaccJobDesc)];
    volatile struct Doorbell *head;
    const struct PaccJobDesc *kernel_head;

    memset(header, 0, sizeof(header));
    jobd_io_fence();
    if (read_control_snapshot(fd, header, sizeof(header)) != 0) {
        return false;
    }
    jobd_io_fence();

    head = (volatile struct Doorbell *)(void *)header;
    kernel_head = (const struct PaccJobDesc *)(const void *)header;
    if (head->magic == HETGPU_PACC_JOB_MAGIC &&
        head->version == HETGPU_PACC_JOB_VERSION &&
        head->seq != 0 &&
        head->seq != last_seq &&
        !preloaded_job_seen(head->job_id, head->seq)) {
        return true;
    }
    if (kernel_head->buf_info == PACC_JOB_MAGIC &&
        kernel_head->seq != 0 &&
        kernel_head->seq != last_kernel_seq) {
        return true;
    }
    if (find_pending_arg_slot_job(fd, NULL)) {
        return true;
    }
    return false;
}

static uint32_t control_snapshot_detail(const uint8_t *snapshot,
                                        uint64_t last_seq,
                                        uint64_t last_kernel_seq,
                                        uint64_t *seq_out) {
    const volatile struct Doorbell *head;
    const struct PaccJobDesc *kernel_head;
    bool snapshot_preloaded;
    bool snapshot_kernel;
    uint32_t detail;

    if (seq_out) {
        *seq_out = 0;
    }
    if (!snapshot) {
        return 0xffffu;
    }

    head = (const volatile struct Doorbell *)(const void *)snapshot;
    kernel_head = (const struct PaccJobDesc *)(const void *)snapshot;
    snapshot_preloaded =
        head->magic == HETGPU_PACC_JOB_MAGIC &&
        head->version == HETGPU_PACC_JOB_VERSION &&
        head->seq != 0 &&
        head->seq != last_seq;
    snapshot_kernel =
        kernel_head->buf_info == PACC_JOB_MAGIC &&
        kernel_head->seq != 0 &&
        kernel_head->seq != last_kernel_seq;
    if (seq_out) {
        *seq_out = kernel_head->seq ? kernel_head->seq : head->seq;
    }

    detail = (snapshot_preloaded ? 0x0001u : 0u) |
             (snapshot_kernel ? 0x0002u : 0u) |
             (head->magic == HETGPU_PACC_JOB_MAGIC ? 0x0004u : 0u) |
             (head->version == HETGPU_PACC_JOB_VERSION ? 0x0008u : 0u) |
             (head->seq != 0 ? 0x0010u : 0u) |
             (kernel_head->seq != 0 ? 0x0020u : 0u) |
             (kernel_head->buf_info == PACC_JOB_MAGIC ? 0x0040u : 0u) |
             (kernel_head->addr != 0 ? 0x0080u : 0u) |
             (kernel_head->len != 0 ? 0x0100u : 0u);
    return detail;
}

static uint64_t post_irq_scan_us(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_POST_IRQ_SCAN_US", 2000000ULL);
}

static uint64_t post_irq_scan_sleep_us(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_POST_IRQ_SCAN_SLEEP_US", 50ULL);
}

static uint64_t pre_poll_scan_us(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_PRE_POLL_SCAN_US", 50000ULL);
}

static uint64_t pre_poll_scan_sleep_us(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_PRE_POLL_SCAN_SLEEP_US", 50ULL);
}

static uint32_t load_dispatch_snapshot(int fd,
                                       struct Map *control_map,
                                       uint8_t *snapshot,
                                       uint64_t last_seq,
                                       uint64_t last_kernel_seq,
                                       volatile struct Doorbell **dispatch_ctl) {
    uint32_t detail = 0xffffu;

    if (dispatch_ctl) {
        *dispatch_ctl = NULL;
    }
    if (!snapshot) {
        return detail;
    }

    mirror_diag_progress_status(fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7020);
    memset(snapshot, 0, HETGPU_PACC_CONTROL_BYTES);
    sync_map_for_cpu(control_map);
    mirror_diag_progress_status(fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7021);
    if (read_control_snapshot(fd, snapshot, sizeof(struct PaccJobDesc)) != 0) {
        mirror_diag_progress_status(fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7022);
        sync_map_for_cpu(control_map);
        if (dispatch_ctl && g_control_window) {
            *dispatch_ctl = (volatile struct Doorbell *)g_control_window;
        }
        return detail;
    }

    detail = control_snapshot_detail(snapshot, last_seq, last_kernel_seq, NULL);
    mirror_diag_progress_status(fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                                0x70230000u | detail);
    if ((detail & 0x2u) != 0 ||
        (((detail & 0x1u) != 0) && jobd_full_control_snapshot_enabled())) {
        mirror_diag_progress_status(fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                                    0x70240000u | detail);
        if (read_control_snapshot(fd, snapshot, HETGPU_PACC_CONTROL_BYTES) == 0) {
            jobd_io_fence();
            if (dispatch_ctl) {
                *dispatch_ctl = (volatile struct Doorbell *)(void *)snapshot;
            }
            mirror_diag_progress_status(fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                                        0x70250000u | detail);
        } else {
            mirror_diag_progress_status(fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                                        0x70260000u | detail);
            sync_map_for_cpu(control_map);
            if (dispatch_ctl && g_control_window) {
                *dispatch_ctl = (volatile struct Doorbell *)g_control_window;
            }
        }
    } else if (dispatch_ctl) {
        *dispatch_ctl = (volatile struct Doorbell *)(void *)snapshot;
    }

    return detail;
}

static uint64_t initial_scan_us(void) {
    return parse_env_u64_default("HETGPU_PACC_JOBD_INITIAL_SCAN_US", 200000ULL);
}

static volatile struct Doorbell *scan_for_control(int fd, const struct pacc_zluda_ddr_info *ddr_info, struct Map *map) {
    uint64_t control_off = shared_ddr_control_rel(g_pacc_id, 0);
    if (!ddr_info || !ddr_info->ddr_base ||
        control_off > ddr_info->ddr_size ||
        HETGPU_PACC_CONTROL_BYTES > ddr_info->ddr_size - control_off) {
        log_msg("invalid shared ddr control window: pacc_id=%" PRIu64 " base=0x%" PRIx64 " size=0x%" PRIx64,
                g_pacc_id, ddr_info ? ddr_info->ddr_base : 0, ddr_info ? ddr_info->ddr_size : 0);
        return NULL;
    }
    if (map_phys(fd, ddr_info->ddr_base + control_off, HETGPU_PACC_CONTROL_BYTES, map)) {
        log_msg("map shared DDR control pacc_id=%" PRIu64 " phys=0x%" PRIx64 " len 0x%x failed: %s",
                g_pacc_id, ddr_info->ddr_base + control_off,
                (unsigned)HETGPU_PACC_CONTROL_BYTES, strerror(errno));
        return NULL;
    }
    trace_msg("mapped shared DDR control pacc_id=%" PRIu64 " at phys 0x%" PRIx64 " len 0x%x",
              g_pacc_id, ddr_info->ddr_base + control_off,
              (unsigned)HETGPU_PACC_CONTROL_BYTES);
    return (volatile struct Doorbell *)map->ptr;
}

static void seed_last_seen_sequences(volatile struct Doorbell *ctl,
                                     uint64_t *last_seq,
                                     uint64_t *last_kernel_seq) {
    const struct PaccJobDesc *kernel_desc = (const struct PaccJobDesc *)(const void *)ctl;
    if (!ctl || !last_seq || !last_kernel_seq) {
        return;
    }

    if (!jobd_seed_current_jobs_enabled()) {
        trace_msg("current doorbell will be treated as pending");
        return;
    }

    __sync_synchronize();
    if (ctl->magic == HETGPU_PACC_JOB_MAGIC && ctl->version == HETGPU_PACC_JOB_VERSION) {
        *last_seq = ctl->seq;
        trace_msg("seed normal doorbell seq=%" PRIu64, *last_seq);
    } else if (kernel_desc->buf_info == PACC_JOB_MAGIC) {
        *last_kernel_seq = kernel_desc->seq;
        trace_msg("seed kernel doorbell seq=%" PRIu64, *last_kernel_seq);
    }
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

static bool mirror_host_status_mbox_dual_pwrite(const struct HostStatus *status_msg) {
    uint64_t ddr_off;
    uint64_t offsets[3];
    bool wrote = false;

    if (!status_msg || g_mbox_fd < 0 || !g_ddr_info.ddr_base || !g_ddr_info.ddr_size ||
        g_pacc_id >= HETGPU_PACC_COUNT) {
        return false;
    }

    ddr_off = shared_ddr_control_rel(g_pacc_id, HETGPU_PACC_COMPLETION_OFF);
    if (ddr_off > g_ddr_info.ddr_size ||
        sizeof(*status_msg) > g_ddr_info.ddr_size - ddr_off) {
        return false;
    }

    offsets[0] = g_shared_ddr_fd_user_off + ddr_off;
    offsets[1] = ddr_off;
    offsets[2] = HETGPU_PACC_SHARED_DDR_FD_USER_OFF + ddr_off;

    jobd_io_fence();
    for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        bool duplicate = false;
        for (size_t j = 0; j < i; j++) {
            if (offsets[i] == offsets[j]) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        ssize_t put = pwrite(g_mbox_fd, status_msg, sizeof(*status_msg),
                             (off_t)offsets[i]);
        if (put == (ssize_t)sizeof(*status_msg)) {
            wrote = true;
        } else {
            trace_msg("mirror_host_status dual pwrite off=0x%" PRIx64
                      " failed put=%zd errno=%d",
                      offsets[i], put, errno);
        }
    }
    jobd_io_fence();
    return wrote;
}

static bool mirror_host_status_payload_mirror(const struct HostStatus *status_msg) {
    uint64_t mirror_off;
    uint64_t control_off;
    uint64_t phys;
    uint64_t offsets[3];
    uint8_t padded[128];
    bool wrote = false;

    if (!status_msg || g_mbox_fd < 0 || !g_ddr_info.ddr_base || !g_ddr_info.ddr_size ||
        g_pacc_id >= HETGPU_PACC_COUNT) {
        return false;
    }
    mirror_off = parse_env_u64_default("HETGPU_PACC_JOBD_COMPLETION_MIRROR_OFF",
                                       0);
    if (mirror_off == 0) {
        return false;
    }
    control_off = shared_ddr_control_rel(g_pacc_id, HETGPU_PACC_COMPLETION_OFF);
    if (control_off > g_ddr_info.ddr_size ||
        sizeof(padded) > g_ddr_info.ddr_size - control_off ||
        mirror_off > g_ddr_info.ddr_size - control_off ||
        sizeof(padded) > g_ddr_info.ddr_size - mirror_off - control_off) {
        return false;
    }
    phys = g_ddr_info.ddr_base + mirror_off + control_off;
    memset(padded, 0, sizeof(padded));
    memcpy(padded, status_msg, sizeof(*status_msg));
    if (write_phys_copy_pwrite_only(g_mbox_fd, phys, padded, sizeof(padded)) == 0) {
        wrote = true;
    }
    offsets[0] = g_shared_ddr_fd_user_off + mirror_off + control_off;
    offsets[1] = mirror_off + control_off;
    offsets[2] = HETGPU_PACC_SHARED_DDR_FD_USER_OFF + mirror_off + control_off;
    jobd_io_fence();
    for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        bool duplicate = false;
        for (size_t j = 0; j < i; j++) {
            if (offsets[i] == offsets[j]) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        ssize_t put = pwrite(g_mbox_fd, padded, sizeof(padded),
                             (off_t)offsets[i]);
        if (put == (ssize_t)sizeof(padded)) {
            wrote = true;
            trace_msg("mirror_host_status payload mirror off=0x%" PRIx64
                      " job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                      offsets[i], status_msg->job_id, job_name(status_msg->job_id),
                      status_msg->seq, status_msg->status);
        } else {
            trace_msg("mirror_host_status payload mirror off=0x%" PRIx64
                      " failed put=%zd errno=%d",
                      offsets[i], put, errno);
        }
    }
    jobd_io_fence();
    if (wrote) {
        submit_mbox_payload_sync(g_mbox_fd, status_msg->job_id, status_msg->seq,
                                 status_msg->status, "completion-mirror");
    }
    return wrote;
}

static bool mirror_host_status_control_window_direct(uint32_t job_id, uint64_t seq, uint32_t status) {
    if (!g_control_window ||
        HETGPU_PACC_COMPLETION_OFF > HETGPU_PACC_CONTROL_BYTES ||
        sizeof(struct HostStatus) > HETGPU_PACC_CONTROL_BYTES - HETGPU_PACC_COMPLETION_OFF) {
        return false;
    }

    volatile struct HostStatus *host =
        (volatile struct HostStatus *)(g_control_window + HETGPU_PACC_COMPLETION_OFF);
    jobd_io_fence();
    host->magic = HETGPU_PACC_JOB_MAGIC;
    host->version = HETGPU_PACC_JOB_VERSION;
    host->job_id = job_id;
    host->status = status;
    host->seq = seq;
    jobd_io_fence();
    jobd_flush_for_device((const void *)host, sizeof(*host));
    if (jobd_status_msync_enabled() && g_control_map_base && g_control_map_len) {
        (void)msync(g_control_map_base, g_control_map_len, MS_SYNC);
    }
    jobd_io_fence();
    return true;
}

static bool mirror_host_status_pacc2ap_mmap_only(int fd, const struct HostStatus *status_msg) {
    uint64_t page = (uint64_t)(g_page_size ? g_page_size : 4096);
    uint64_t phys = PACC2AP_MBOX_PHYS + HETGPU_PACC_COMPLETION_OFF;
    uint64_t map_base = phys & ~(page - 1u);
    size_t map_off = (size_t)(phys - map_base);
    size_t map_len = (size_t)(((map_off + sizeof(*status_msg) + page - 1u) / page) * page);
    int io_fd = fd;
    bool close_io_fd = false;
    void *base;
    volatile uint8_t *dst;
    const uint8_t *src;

    if (!status_msg) {
        return false;
    }
    if (!env_flag_true(getenv("HETGPU_PACC_JOBD_ENABLE_PACC2AP_MMAP_STATUS"))) {
        return false;
    }
    if (io_fd < 0 || io_fd == g_mbox_fd) {
        io_fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        close_io_fd = io_fd >= 0;
    }
    if (io_fd < 0) {
        trace_msg("mirror_host_status pacc2ap mmap open failed errno=%d", errno);
        return false;
    }

    base = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, io_fd, (off_t)map_base);
    if (base == MAP_FAILED) {
        int saved_errno = errno;
        if (close_io_fd) {
            close(io_fd);
        }
        errno = saved_errno;
        trace_msg("mirror_host_status pacc2ap mmap failed phys=0x%" PRIx64
                  " errno=%d",
                  phys, errno);
        return false;
    }

    dst = (volatile uint8_t *)base + map_off;
    src = (const uint8_t *)status_msg;
    jobd_io_fence();
    for (size_t i = 0; i < sizeof(*status_msg); i++) {
        dst[i] = src[i];
    }
    jobd_io_fence();
    if (jobd_msync_enabled()) {
        (void)msync(base, map_len, MS_SYNC);
    }
    munmap(base, map_len);
    if (close_io_fd) {
        close(io_fd);
    }
    jobd_io_fence();
    trace_msg("mirror_host_status pacc2ap mmap: job_id=%u/%s seq=%" PRIu64
              " status=0x%x phys=0x%" PRIx64,
              status_msg->job_id, job_name(status_msg->job_id),
              status_msg->seq, status_msg->status, phys);
    return true;
}

static void mirror_host_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status) {
    struct Map map = {0};
    uint64_t phys = shared_ddr_control_phys(HETGPU_PACC_COMPLETION_OFF, sizeof(struct HostStatus));
    bool fd_is_mbox = fd >= 0 && fd == g_mbox_fd;
    struct HostStatus status_msg = {
        .magic = HETGPU_PACC_JOB_MAGIC,
        .version = HETGPU_PACC_JOB_VERSION,
        .job_id = job_id,
        .status = status,
        .seq = seq,
    };
    bool wrote = false;
    bool payload_mirror_ok = false;
    bool dual_pwrite_ok = false;
    bool primary_pwrite_ok = false;

#if HETGPU_PACC_COMPLETION_TELEMETRY && HETGPU_PACC_COMPLETION_FAST_PATH
    bool first_direct =
        mirror_host_status_control_window_direct(job_id, seq, status);
    if (HETGPU_PACC_COMPLETION_SETTLE_NS != 0) {
        uint64_t settle_deadline_ns =
            monotonic_ns() + HETGPU_PACC_COMPLETION_SETTLE_NS;
        while (monotonic_ns() < settle_deadline_ns) {
            __asm__ volatile("" ::: "memory");
        }
    }
    bool second_direct =
        mirror_host_status_control_window_direct(job_id, seq, status);
    if (first_direct && second_direct) {
        return;
    }
#endif

    if (jobd_mbox_control_enabled()) {
        /*
         * Current PACC Linux does not tolerate /dev/mem pwrite() to the
         * PACC2AP mailbox MMIO window from userspace.  Completion in
         * mailbox-control mode must therefore go through the mailbox device
         * itself: first try the driver's pwrite ABI, then its mmap window.
         */
        jobd_io_fence();
        if (!wrote && mirror_host_status_pacc2ap_mmap_only(fd, &status_msg)) {
            wrote = true;
        }
        if (!wrote && g_mbox_fd >= 0) {
            ssize_t put;
            put = pwrite(g_mbox_fd, &status_msg, sizeof(status_msg),
                         (off_t)HETGPU_PACC_COMPLETION_OFF);
            if (put == (ssize_t)sizeof(status_msg)) {
                trace_msg("mirror_host_status mbox-control pwrite: job_id=%u/%s seq=%" PRIu64
                          " status=0x%x off=0x%" PRIx64,
                          job_id, job_name(job_id), seq, status,
                          (uint64_t)HETGPU_PACC_COMPLETION_OFF);
                wrote = true;
            } else {
                trace_msg("mirror_host_status mbox-control pwrite failed put=%zd errno=%d",
                          put, errno);
            }
        }
        if (!wrote && g_mbox_fd >= 0) {
            size_t page = (size_t)(g_page_size ? g_page_size : 4096);
            size_t min_len = (size_t)(HETGPU_PACC_CONTROL_BYTES +
                                      HETGPU_PACC_COMPLETION_OFF +
                                      sizeof(status_msg));
            size_t map_len = (min_len + page - 1) & ~(page - 1);
            void *base = mmap(NULL,
                              map_len,
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED,
                              g_mbox_fd,
                              0);
            if (base != MAP_FAILED) {
                volatile struct HostStatus *host0 =
                    (volatile struct HostStatus *)((uint8_t *)base +
                                                   HETGPU_PACC_COMPLETION_OFF);
                volatile struct HostStatus *host1 =
                    (volatile struct HostStatus *)((uint8_t *)base +
                                                   HETGPU_PACC_CONTROL_BYTES +
                                                   HETGPU_PACC_COMPLETION_OFF);
                jobd_io_fence();
                *host0 = status_msg;
                *host1 = status_msg;
                jobd_io_fence();
                jobd_flush_for_device((const void *)host0, sizeof(*host0));
                jobd_flush_for_device((const void *)host1, sizeof(*host1));
                if (jobd_msync_enabled()) {
                    (void)msync(base, map_len, MS_SYNC);
                }
                munmap(base, map_len);
                trace_msg("mirror_host_status mbox-control mmap dual: job_id=%u/%s seq=%" PRIu64
                          " status=0x%x",
                          job_id, job_name(job_id), seq, status);
                wrote = true;
            } else {
                trace_msg("mirror_host_status mbox-control mmap failed errno=%d", errno);
            }
        }
        jobd_io_fence();
        if (wrote) {
            return;
        }
    }

    if (!jobd_status_pwrite_enabled() &&
        mirror_host_status_control_window_direct(job_id, seq, status)) {
        trace_msg("mirror_host_status control-window: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                  job_id, job_name(job_id), seq, status);
        wrote = true;
    }

    if (env_flag_true(getenv("HETGPU_PACC_JOBD_DEVMEM_DIRECT_STATUS")) &&
        write_shared_ddr_devmem_direct(phys, &status_msg, sizeof(status_msg))) {
        wrote = true;
    }
    if (write_shared_ddr_fd_mmap_direct(fd, phys, &status_msg, sizeof(status_msg))) {
        wrote = true;
    }

    if (jobd_mbox_status_mmap_enabled() &&
        mirror_host_status_mbox_mmap(g_mbox_fd, job_id, seq, status)) {
        wrote = true;
    }

    if (jobd_status_pwrite_enabled()) {
        payload_mirror_ok = mirror_host_status_payload_mirror(&status_msg);
        if (payload_mirror_ok) {
            wrote = true;
        }
        if (!payload_mirror_ok) {
            dual_pwrite_ok = mirror_host_status_mbox_dual_pwrite(&status_msg);
            if (dual_pwrite_ok) {
                wrote = true;
            }
            primary_pwrite_ok = write_phys_copy_pwrite_only(fd, phys, &status_msg, sizeof(status_msg)) == 0;
            if (primary_pwrite_ok) {
                wrote = true;
            } else {
                trace_msg("mirror_host_status pwrite failed for phys=0x%" PRIx64 ": %s; falling back to mmap",
                          phys, strerror(errno));
            }
        }
        if (wrote) {
            submit_mbox_payload_sync(g_mbox_fd, job_id, seq, status, "completion-status");
        }
    }

    if (jobd_status_control_window_enabled() &&
        g_control_window &&
        HETGPU_PACC_COMPLETION_OFF <= HETGPU_PACC_CONTROL_BYTES &&
        sizeof(struct HostStatus) <= HETGPU_PACC_CONTROL_BYTES - HETGPU_PACC_COMPLETION_OFF) {
        volatile struct HostStatus *host =
            (volatile struct HostStatus *)(g_control_window + HETGPU_PACC_COMPLETION_OFF);
        jobd_io_fence();
        host->magic = HETGPU_PACC_JOB_MAGIC;
        host->version = HETGPU_PACC_JOB_VERSION;
        host->job_id = job_id;
        host->status = status;
        host->seq = seq;
        jobd_io_fence();
        jobd_flush_for_device((const void *)host, sizeof(*host));
        if (jobd_msync_enabled() && g_control_map_base && g_control_map_len) {
            (void)msync(g_control_map_base, g_control_map_len, MS_SYNC);
        }
        jobd_io_fence();
        trace_msg("mirror_host_status: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                  job_id, job_name(job_id), seq, status);
        wrote = true;
    }

    if (wrote && !jobd_status_mmap_fallback_enabled()) {
        jobd_io_fence();
        trace_msg("mirror_host_status direct-only: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                  job_id, job_name(job_id), seq, status);
        return;
    }

    /*
     * If dispatch is using /dev/mem as its data path, the completion store must
     * follow that path too.  A successful /dev/mbox mmap probe is not enough to
     * prove host visibility on all firmware builds.
     */
    if ((!fd_is_mbox || jobd_status_mmap_fallback_enabled() || !jobd_status_pwrite_enabled()) &&
        (!wrote || !fd_is_mbox || !jobd_status_pwrite_enabled())) {
        if (map_phys(fd, phys, sizeof(struct HostStatus), &map) == 0) {
            volatile struct HostStatus *host = (volatile struct HostStatus *)map.ptr;
            jobd_io_fence();
            host->magic = HETGPU_PACC_JOB_MAGIC;
            host->version = HETGPU_PACC_JOB_VERSION;
            host->job_id = job_id;
            host->status = status;
            host->seq = seq;
            jobd_io_fence();
            jobd_flush_for_device((const void *)host, sizeof(*host));
            if (jobd_msync_enabled()) {
                (void)msync(map.base, map.map_len, MS_SYNC);
            }
            jobd_io_fence();
            unmap_phys(&map);
            wrote = true;
        } else {
            log_msg("map host status 0x%" PRIx64 " failed: %s", phys, strerror(errno));
        }
    }

    if (env_flag_true(getenv("HETGPU_PACC_JOBD_DEVMEM_DIRECT_STATUS")) &&
        write_shared_ddr_devmem_direct(phys, &status_msg, sizeof(status_msg))) {
        wrote = true;
    }

    jobd_io_fence();
    if (wrote) {
        trace_msg("mirror_host_status: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
                  job_id, job_name(job_id), seq, status);
    }
}

static void mirror_completion_telemetry(int fd,
                                        uint32_t job_id,
                                        uint64_t seq,
                                        uint32_t status,
                                        uint64_t publish_start_ns,
                                        uint64_t publish_end_ns) {
#if HETGPU_PACC_COMPLETION_TELEMETRY
    volatile struct CompletionTelemetry *dst;
    struct CompletionTelemetry record;

    (void)fd;
    if (job_id != HETGPU_PACC_JOB_GEMM ||
        !g_control_window ||
        HETGPU_PACC_COMPLETION_TELEMETRY_OFF > HETGPU_PACC_CONTROL_BYTES ||
        sizeof(record) > HETGPU_PACC_CONTROL_BYTES -
                             HETGPU_PACC_COMPLETION_TELEMETRY_OFF) {
        return;
    }

    memset(&record, 0, sizeof(record));
    record.version = HETGPU_PACC_COMPLETION_TELEMETRY_VERSION;
    record.record_bytes = sizeof(record);
    record.job_id = job_id;
    record.status = status;
    record.flags = 1U;
    record.seq = seq;
    record.publish_start_ns = publish_start_ns;
    record.publish_end_ns = publish_end_ns;
    if (g_last_gemm_timing.seq == seq) {
        record.compute_start_ns = g_last_gemm_timing.compute_start_ns;
        record.compute_end_ns = g_last_gemm_timing.compute_end_ns;
        record.xsfmm_cycles = g_last_gemm_timing.xsfmm_cycles;
        record.xsfmm_repeats = g_last_gemm_timing.xsfmm_repeats;
    }

    dst = (volatile struct CompletionTelemetry *)(
        g_control_window + HETGPU_PACC_COMPLETION_TELEMETRY_OFF);
    dst->magic = 0;
    jobd_io_fence();
    memcpy((void *)dst, &record, sizeof(record));
    jobd_io_fence();
    dst->magic = HETGPU_PACC_COMPLETION_TELEMETRY_MAGIC;
    jobd_io_fence();
    jobd_flush_for_device((const void *)dst, sizeof(*dst));
    jobd_io_fence();
#else
    (void)fd;
    (void)job_id;
    (void)seq;
    (void)status;
    (void)publish_start_ns;
    (void)publish_end_ns;
#endif
}

static void mirror_job_completion(int fd, uint32_t job_id, uint64_t seq,
                                  uint32_t status) {
    uint64_t publish_start_ns = monotonic_ns();
    mirror_host_status(fd, job_id, seq, status);
    uint64_t publish_end_ns = monotonic_ns();
    mirror_completion_telemetry(fd, job_id, seq, status,
                                publish_start_ns, publish_end_ns);
}

static bool mirror_host_status_mbox_mmap(int fd, uint32_t job_id, uint64_t seq, uint32_t status) {
    uint64_t control_off;
    uint64_t end_off;
    size_t map_len;
    uint64_t page;
    void *base;

    if (fd < 0 || fd != g_mbox_fd ||
        !g_ddr_info.ddr_base || !g_ddr_info.ddr_size ||
        g_pacc_id >= HETGPU_PACC_COUNT) {
        return false;
    }
    control_off = g_pacc_id * HETGPU_PACC_CONTROL_BYTES + HETGPU_PACC_COMPLETION_OFF;
    end_off = control_off + sizeof(struct HostStatus);
    if (end_off > g_ddr_info.ddr_size || end_off < control_off) {
        return false;
    }
    page = (uint64_t)g_page_size;
    if (page == 0) {
        page = 4096;
    }
    map_len = (size_t)((end_off + page - 1) & ~(page - 1));
    base = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        trace_msg("mirror_host_status mbox mmap failed: %s", strerror(errno));
        return false;
    }

    volatile struct HostStatus *host =
        (volatile struct HostStatus *)((uint8_t *)base + control_off);
    jobd_io_fence();
    host->magic = HETGPU_PACC_JOB_MAGIC;
    host->version = HETGPU_PACC_JOB_VERSION;
    host->job_id = job_id;
    host->status = status;
    host->seq = seq;
    jobd_io_fence();
    jobd_flush_for_device((const void *)host, sizeof(*host));
    if (jobd_msync_enabled()) {
        (void)msync(base, map_len, MS_SYNC);
    }
    jobd_io_fence();
    munmap(base, map_len);
    trace_msg("mirror_host_status mmap: job_id=%u/%s seq=%" PRIu64 " status=0x%x",
              job_id, job_name(job_id), seq, status);
    return true;
}

static bool write_shared_ddr_devmem_direct(uint64_t phys, const void *src, size_t len) {
    uint64_t host_phys = phys;
    uint64_t page;
    uint64_t mmap_phys;
    uint64_t map_base;
    size_t map_off;
    size_t map_len;
    void *map;
    int fd;

    if (!src || len == 0) {
        return false;
    }
    if (!phys_is_shared_ddr(host_phys, len)) {
        if (!g_ddr_info.ddr_base ||
            (uint64_t)len > g_ddr_info.ddr_size ||
            host_phys > g_ddr_info.ddr_size - (uint64_t)len) {
            return false;
        }
        host_phys = g_ddr_info.ddr_base + host_phys;
    }
    page = g_page_size > 0 ? (uint64_t)g_page_size : 4096ULL;
    mmap_phys = shared_ddr_pacc_phys(host_phys, len);
    map_base = mmap_phys & ~(page - 1u);
    map_off = (size_t)(mmap_phys - map_base);
    map_len = ((map_off + len + page - 1u) / page) * page;
    fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    map = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)map_base);
    if (map == MAP_FAILED) {
        close(fd);
        return false;
    }
    jobd_io_fence();
    memcpy((uint8_t *)map + map_off, src, len);
    jobd_io_fence();
    jobd_flush_for_device((uint8_t *)map + map_off, len);
    if (jobd_msync_enabled()) {
        (void)msync(map, map_len, MS_SYNC);
    }
    jobd_io_fence();
    munmap(map, map_len);
    close(fd);
    return true;
}

static bool write_shared_ddr_fd_mmap_direct(int fd, uint64_t phys, const void *src, size_t len) {
    if (!env_flag_true(getenv("HETGPU_PACC_JOBD_FD_MMAP_DIRECT_STATUS"))) {
        return false;
    }

    uint64_t rel;
    uint64_t page;
    uint64_t map_off64;
    size_t map_off;
    size_t map_len;
    void *map;

    if (fd < 0 || !src || len == 0 || !g_ddr_info.ddr_base ||
        !g_ddr_info.ddr_size || !phys_is_shared_ddr(phys, len)) {
        return false;
    }
    if (fd != g_mbox_fd && (g_shared_ddr_data_fd < 0 || fd != g_shared_ddr_data_fd)) {
        return false;
    }
    rel = phys - g_ddr_info.ddr_base;
    if ((uint64_t)len > g_ddr_info.ddr_size ||
        rel > g_ddr_info.ddr_size - (uint64_t)len) {
        return false;
    }
    page = g_page_size > 0 ? (uint64_t)g_page_size : 4096ULL;
    map_off64 = rel & ~(page - 1u);
    map_off = (size_t)(rel - map_off64);
    map_len = ((map_off + len + page - 1u) / page) * page;
    map = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)map_off64);
    if (map == MAP_FAILED) {
        return false;
    }
    jobd_io_fence();
    memcpy((uint8_t *)map + map_off, src, len);
    jobd_io_fence();
    jobd_flush_for_device((uint8_t *)map + map_off, len);
    jobd_io_fence();
    munmap(map, map_len);
    return true;
}

static bool read_shared_ddr_devmem_direct(uint64_t phys, void *dst, size_t len) {
    uint64_t host_phys = phys;
    uint64_t page;
    uint64_t mmap_phys;
    uint64_t map_base;
    size_t map_off;
    size_t map_len;
    void *map;
    int fd;

    if (!dst || len == 0) {
        return false;
    }
    if (!phys_is_shared_ddr(host_phys, len)) {
        if (!g_ddr_info.ddr_base ||
            (uint64_t)len > g_ddr_info.ddr_size ||
            host_phys > g_ddr_info.ddr_size - (uint64_t)len) {
            return false;
        }
        host_phys = g_ddr_info.ddr_base + host_phys;
    }
    page = g_page_size > 0 ? (uint64_t)g_page_size : 4096ULL;
    mmap_phys = shared_ddr_pacc_phys(host_phys, len);
    map_base = mmap_phys & ~(page - 1u);
    map_off = (size_t)(mmap_phys - map_base);
    map_len = ((map_off + len + page - 1u) / page) * page;
    fd = open("/dev/mem", O_RDONLY | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    map = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, (off_t)map_base);
    if (map == MAP_FAILED) {
        close(fd);
        return false;
    }
    jobd_io_fence();
    jobd_invalidate_for_cpu((const uint8_t *)map + map_off, len);
    memcpy(dst, (const uint8_t *)map + map_off, len);
    jobd_io_fence();
    munmap(map, map_len);
    close(fd);
    return true;
}

static bool read_shared_ddr_fd_mmap_direct(int fd, uint64_t phys, void *dst, size_t len) {
    uint64_t rel;
    uint64_t page;
    uint64_t map_off64;
    size_t map_off;
    size_t map_len;
    void *map;

    if (fd < 0 || !dst || len == 0 || !g_ddr_info.ddr_base ||
        !g_ddr_info.ddr_size || !phys_is_shared_ddr(phys, len)) {
        return false;
    }
    rel = phys - g_ddr_info.ddr_base;
    if ((uint64_t)len > g_ddr_info.ddr_size ||
        rel > g_ddr_info.ddr_size - (uint64_t)len) {
        return false;
    }
    page = g_page_size > 0 ? (uint64_t)g_page_size : 4096ULL;
    map_off64 = rel & ~(page - 1u);
    map_off = (size_t)(rel - map_off64);
    map_len = ((map_off + len + page - 1u) / page) * page;
    map = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, (off_t)map_off64);
    if (map == MAP_FAILED) {
        return false;
    }
    jobd_io_fence();
    jobd_invalidate_for_cpu((const uint8_t *)map + map_off, len);
    memcpy(dst, (const uint8_t *)map + map_off, len);
    jobd_io_fence();
    munmap(map, map_len);
    return true;
}

static bool read_shared_ddr_host_devmem_direct(uint64_t phys, void *dst, size_t len) {
    uint64_t host_phys = phys;
    uint64_t page;
    uint64_t map_base;
    size_t map_off;
    size_t map_len;
    void *map;
    int fd;

    if (!dst || len == 0) {
        return false;
    }
    if (!phys_is_shared_ddr(host_phys, len)) {
        if (!g_ddr_info.ddr_base ||
            (uint64_t)len > g_ddr_info.ddr_size ||
            host_phys > g_ddr_info.ddr_size - (uint64_t)len) {
            return false;
        }
        host_phys = g_ddr_info.ddr_base + host_phys;
    }
    page = g_page_size > 0 ? (uint64_t)g_page_size : 4096ULL;
    map_base = host_phys & ~(page - 1u);
    map_off = (size_t)(host_phys - map_base);
    map_len = ((map_off + len + page - 1u) / page) * page;
    fd = open("/dev/mem", O_RDONLY | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    map = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, (off_t)map_base);
    if (map == MAP_FAILED) {
        close(fd);
        return false;
    }
    jobd_io_fence();
    jobd_invalidate_for_cpu((const uint8_t *)map + map_off, len);
    memcpy(dst, (const uint8_t *)map + map_off, len);
    jobd_io_fence();
    munmap(map, map_len);
    close(fd);
    return true;
}

static bool arg_slot_fast_peek_direct(uint32_t job_id) {
    int slot = arg_slot_for_job(job_id);
    uint64_t slot_off;
    uint64_t phys;
    struct ArgSlotHeader header;

    if (slot < 0 || !g_ddr_info.ddr_base || !g_ddr_info.ddr_size ||
        g_pacc_id >= HETGPU_PACC_COUNT) {
        return false;
    }
    slot_off = shared_ddr_control_rel(g_pacc_id, HETGPU_PACC_ARG_BASE_OFF +
               (uint64_t)slot * HETGPU_PACC_ARG_SLOT_BYTES);
    if (slot_off > g_ddr_info.ddr_size ||
        sizeof(header) > g_ddr_info.ddr_size - slot_off) {
        return false;
    }
    phys = g_ddr_info.ddr_base + slot_off;
    memset(&header, 0, sizeof(header));
    if (!read_shared_ddr_devmem_direct(phys, &header, sizeof(header))) {
        return false;
    }
    return (header.magic == HETGPU_PACC_JOB_MAGIC ||
            header.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC) &&
           header.version == HETGPU_PACC_JOB_VERSION &&
           header.seq != 0;
}

static bool mirror_boot_marker_all_slots_mbox_mmap(int fd, uint32_t status) {
    const size_t map_len = HETGPU_PACC_COUNT * HETGPU_PACC_CONTROL_BYTES;
    void *base;

    if (fd < 0) {
        return false;
    }
    base = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        trace_msg("boot marker all-slot mmap failed: %s", strerror(errno));
        return false;
    }
    for (uint32_t slot = 0; slot < HETGPU_PACC_COUNT; slot++) {
        volatile struct HostStatus *host =
            (volatile struct HostStatus *)((uint8_t *)base +
                                           (uint64_t)slot * HETGPU_PACC_CONTROL_BYTES +
                                           HETGPU_PACC_COMPLETION_OFF);
        host->magic = HETGPU_PACC_JOB_MAGIC;
        host->version = HETGPU_PACC_JOB_VERSION;
        host->job_id = PACC_KERNEL_JOB_ID;
        host->status = status | slot;
        host->seq = 0;
    }
    jobd_io_fence();
    jobd_flush_for_device(base, map_len);
    if (jobd_msync_enabled()) {
        (void)msync(base, map_len, MS_SYNC);
    }
    jobd_io_fence();
    munmap(base, map_len);
    trace_msg("boot marker all-slot status=0x%x", status);
    return true;
}

static void mirror_diag_event(int fd, uint32_t job_id, uint64_t seq, uint32_t status, uint32_t aux) {
    struct JobdDiagEvent event;
    uint32_t index;
    uint32_t diag_slot;
    uint64_t phys;

    if (!jobd_diag_ring_enabled()) {
        return;
    }
    diag_slot = HETGPU_PACC_DIAG_RING_SLOT +
                (g_pacc_id < HETGPU_PACC_COUNT ? (uint32_t)g_pacc_id : 0U);
    if (!g_ddr_info.ddr_base ||
        g_ddr_info.ddr_size < (uint64_t)(diag_slot + 1) * HETGPU_PACC_CONTROL_BYTES) {
        return;
    }

    index = g_diag_ring_index++;
    event.magic = HETGPU_PACC_DIAG_MAGIC;
    event.index = index;
    event.status = status;
    event.job_id = job_id;
    event.aux = aux;
    event.seq = seq;

    phys = g_ddr_info.ddr_base +
           (uint64_t)diag_slot * HETGPU_PACC_CONTROL_BYTES +
           HETGPU_PACC_DIAG_RING_OFF +
           (uint64_t)(index % HETGPU_PACC_DIAG_RING_RECORDS) * sizeof(event);
    if (write_phys_copy_pwrite_only(fd, phys, &event, sizeof(event)) != 0) {
        (void)write_phys_copy(fd, phys, &event, sizeof(event));
    }
}

static void mirror_rmsnorm_debug_record(int fd, const struct RmsNormDebugRecord *record) {
    uint64_t rel;
    uint64_t phys;

    if (!jobd_rms_debug_enabled() || !record ||
        !g_ddr_info.ddr_base || !g_ddr_info.ddr_size ||
        g_pacc_id >= HETGPU_PACC_COUNT) {
        return;
    }

    rel = (uint64_t)HETGPU_PACC_RMS_DEBUG_SLOT * HETGPU_PACC_CONTROL_BYTES +
          (uint64_t)g_pacc_id * HETGPU_PACC_RMS_DEBUG_RECORD_BYTES;
    if (rel > g_ddr_info.ddr_size || sizeof(*record) > g_ddr_info.ddr_size - rel) {
        return;
    }

    phys = g_ddr_info.ddr_base + rel;
    if (write_phys_copy_pwrite_only(fd, phys, record, sizeof(*record)) != 0) {
        (void)write_phys_copy(fd, phys, record, sizeof(*record));
    }
}

static bool aligned_completion_record_enabled(void) {
    const char *value = getenv("HETGPU_PACC_JOBD_ALIGNED_COMPLETION_RECORD");
    if (value && *value) {
        return env_flag_true(value);
    }
    return false;
}

static void mirror_aligned_completion_record(int fd,
                                             uint32_t job_id,
                                             uint64_t seq,
                                             uint32_t status) {
    uint64_t mirror_off;
    uint64_t control_off;
    uint64_t rel;
    uint64_t phys;
    struct RmsNormDebugRecord record;

    if (!aligned_completion_record_enabled() ||
        !g_ddr_info.ddr_base || !g_ddr_info.ddr_size ||
        g_pacc_id >= HETGPU_PACC_COUNT) {
        return;
    }
    mirror_off = parse_env_u64_default("HETGPU_PACC_JOBD_COMPLETION_MIRROR_OFF",
                                       HETGPU_PACC_COMPLETION_MIRROR_DEFAULT_OFF);
    if (mirror_off == 0) {
        return;
    }
    control_off = shared_ddr_control_rel(g_pacc_id, HETGPU_PACC_COMPLETION_OFF);
    if (mirror_off > UINT64_MAX - control_off) {
        return;
    }
    rel = mirror_off + control_off;
    if (rel > g_ddr_info.ddr_size || sizeof(record) > g_ddr_info.ddr_size - rel) {
        return;
    }

    memset(&record, 0, sizeof(record));
    record.magic = HETGPU_PACC_RMS_DEBUG_MAGIC;
    record.version = HETGPU_PACC_JOB_VERSION;
    record.pacc_id = (uint32_t)g_pacc_id;
    record.phase = 0x5151u;
    record.dtype = job_id;
    record.seq = seq;
    record.row = status;
    record.flags = job_id;
    record.reserved = status;

    phys = g_ddr_info.ddr_base + rel;
    if (write_phys_copy_pwrite_only(fd, phys, &record, sizeof(record)) != 0) {
        (void)write_phys_copy(fd, phys, &record, sizeof(record));
    }
}

static void mirror_diag_progress_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status) {
    const uint32_t slot = g_pacc_id < HETGPU_PACC_COUNT ? (uint32_t)g_pacc_id : 0U;
    struct HostStatus msg = {
        .magic = HETGPU_PACC_JOB_MAGIC,
        .version = HETGPU_PACC_JOB_VERSION,
        .job_id = job_id,
        .status = status,
        .seq = seq,
    };
    uint64_t phys;

    mirror_diag_event(fd, job_id, seq, status, 0);
    if (!jobd_progress_completion_enabled()) {
        return;
    }
    if (!g_ddr_info.ddr_base ||
        g_ddr_info.ddr_size < (uint64_t)(slot + 1) * HETGPU_PACC_CONTROL_BYTES) {
        return;
    }
    phys = g_ddr_info.ddr_base +
           (uint64_t)slot * HETGPU_PACC_CONTROL_BYTES +
           HETGPU_PACC_COMPLETION_OFF;
    jobd_io_fence();
    if (write_phys_copy_pwrite_only(fd, phys, &msg, sizeof(msg)) != 0 &&
        write_phys_copy(fd, phys, &msg, sizeof(msg)) != 0) {
        int mem_fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (mem_fd >= 0) {
            (void)write_phys_copy(mem_fd, phys, &msg, sizeof(msg));
            close(mem_fd);
        }
    }
    jobd_io_fence();
}

static void mirror_progress_status(int fd, uint32_t job_id, uint64_t seq, uint32_t status) {
    mirror_diag_progress_status(fd, job_id, seq, status);
    if (jobd_progress_status_enabled()) {
        write_jobd_beacon(fd, job_id, seq, status, 0);
    }
}

static void early_devmem_diag_marker(const char *devmem, uint32_t status, uint64_t seq) {
    uint64_t ddr_base;
    uint64_t page;
    int fd;

    if (!env_flag_true(getenv("HETGPU_PACC_JOBD_EARLY_DEVMEM_MARKER"))) {
        return;
    }

    ddr_base = parse_env_u64_default("HETGPU_PACC_SHARED_DDR_PACC_BASE", 0x80000000ULL);
    if (!ddr_base) {
        return;
    }
    if (g_page_size <= 0) {
        g_page_size = 4096;
    }
    page = (uint64_t)g_page_size;

    mkdir("/dev", 0755);
    mknod("/dev/mem", S_IFCHR | 0600, makedev(1, 1));
    fd = open(devmem ? devmem : "/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        log_msg("early devmem marker open failed: %s", strerror(errno));
        return;
    }

    for (uint32_t slot = 0; slot < HETGPU_PACC_COUNT; slot++) {
        uint64_t phys = ddr_base + (uint64_t)slot * HETGPU_PACC_CONTROL_BYTES +
                        HETGPU_PACC_COMPLETION_OFF;
        uint64_t map_base = phys & ~(page - 1);
        size_t map_off = (size_t)(phys - map_base);
        size_t map_len = ((map_off + sizeof(struct HostStatus) + page - 1) / page) * page;
        void *map = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)map_base);
        if (map == MAP_FAILED) {
            log_msg("early devmem marker mmap phys=0x%" PRIx64 " failed: %s",
                    phys, strerror(errno));
            continue;
        }

        volatile struct HostStatus *host =
            (volatile struct HostStatus *)((uint8_t *)map + map_off);
        jobd_io_fence();
        host->magic = HETGPU_PACC_JOB_MAGIC;
        host->version = HETGPU_PACC_JOB_VERSION;
        host->job_id = PACC_KERNEL_JOB_ID;
        host->status = status | slot;
        host->seq = seq;
        jobd_io_fence();
        jobd_flush_for_device((const void *)host, sizeof(*host));
        if (jobd_msync_enabled()) {
            (void)msync(map, map_len, MS_SYNC);
        }
        jobd_io_fence();
        munmap(map, map_len);
    }

    close(fd);
    trace_msg("early devmem marker status=0x%x seq=%" PRIu64 " base=0x%" PRIx64,
              status, seq, ddr_base);
}

int main(int argc, char **argv) {
    const char *devmem = "/dev/mem";
    const char *mbox_path = "/dev/mbox";
    const char *config = "/etc/hetgpu_pacc_jobs.conf";
    bool strict = false;

    install_crash_handlers();

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
        } else if (!strncmp(argv[i], "--devmem=", 9)) {
            devmem = argv[i] + 9;
        } else if (!strcmp(argv[i], "--config") && i + 1 < argc) {
            config = argv[++i];
        } else if (!strncmp(argv[i], "--config=", 9)) {
            config = argv[i] + 9;
        } else if ((!strcmp(argv[i], "--mbox") || !strcmp(argv[i], "--pacc-dev")) && i + 1 < argc) {
            mbox_path = argv[++i];
        } else if (!strncmp(argv[i], "--mbox=", 7)) {
            mbox_path = argv[i] + 7;
        } else if (!strncmp(argv[i], "--pacc-dev=", 11)) {
            mbox_path = argv[i] + 11;
        }
    }

    g_page_size = sysconf(_SC_PAGESIZE);
    if (g_page_size <= 0) g_page_size = 4096;
    early_devmem_diag_marker(devmem, 0x6d100000u, 1);
    if (jobd_xsfmm_gemm_requested()) {
        int xsfmm_init = jobd_enable_xsfmm_context();
        if (xsfmm_init != 0) {
            g_xsfmm_context_error = -xsfmm_init;
            early_devmem_diag_marker(devmem,
                                     0x6d10d000u | ((uint32_t)(-xsfmm_init) & 0xffu),
                                     2);
            log_msg("continuing in hardware-only fail-closed mode");
        } else {
            early_devmem_diag_marker(devmem, 0x6d10a000u, 2);
        }
    }

    int mbox_fd = open(mbox_path, O_RDWR | O_SYNC | O_CLOEXEC);
    if (mbox_fd < 0) {
        early_devmem_diag_marker(devmem, 0x6d10e000u, (uint64_t)errno);
        log_msg("open %s failed: %s", mbox_path, strerror(errno));
        return 1;
    }
    g_mbox_fd = mbox_fd;
    early_devmem_diag_marker(devmem, 0x6d110000u, 2);

    log_msg("started strict=%d config=%s mbox=%s",
            strict ? 1 : 0, config, mbox_path);
    struct PreloadedJobs jobs;
    memset(&jobs, 0, sizeof(jobs));
    load_jobs_config(config, &jobs);

    wait_for_initial_control(mbox_fd);

    read_shared_ddr_info_from_mbox(mbox_fd);
    read_pacc_id_from_mbox(mbox_fd);
    g_shared_ddr_pacc_base =
        parse_env_u64_default("HETGPU_PACC_SHARED_DDR_PACC_BASE",
                              g_ddr_info.ddr_base ? g_ddr_info.ddr_base : 0x80000000ULL);
    if (g_shared_ddr_pacc_base && g_ddr_info.ddr_base &&
        g_shared_ddr_pacc_base != g_ddr_info.ddr_base) {
        log_msg("shared DDR host base=0x%" PRIx64 " pacc mmap base=0x%" PRIx64
                " size=0x%" PRIx64,
                g_ddr_info.ddr_base, g_shared_ddr_pacc_base,
                g_ddr_info.ddr_size);
    }
    early_devmem_diag_marker(devmem, 0x6d120000u, 3);
    g_shared_ddr_fd_user_off =
        parse_env_u64_default("HETGPU_PACC_SHARED_DDR_FD_USER_OFF",
                              parse_env_u64_default("HETGPU_PACC_SHARED_DDR_USER_OFF",
                                                    HETGPU_PACC_SHARED_DDR_FD_USER_OFF));
    g_shared_ddr_control_base_off =
        parse_env_u64_default("HETGPU_PACC_SHARED_DDR_CONTROL_BASE_OFF",
                              parse_env_u64_default("HETGPU_PACC_JOBD_CONTROL_BASE_OFF",
                                                    jobd_mbox_control_enabled()
                                                        ? g_shared_ddr_fd_user_off
                                                        : 0));
    log_msg("shared DDR control base off=0x%" PRIx64 " fd_user_off=0x%" PRIx64,
            g_shared_ddr_control_base_off, g_shared_ddr_fd_user_off);
    const char *shared_ddr_dev = getenv("HETGPU_PACC_JOBD_SHARED_DDR_DEV");
    if (shared_ddr_dev && *shared_ddr_dev) {
        g_shared_ddr_data_fd = open(shared_ddr_dev, O_RDWR | O_SYNC | O_CLOEXEC);
        if (g_shared_ddr_data_fd >= 0) {
            log_msg("using shared DDR data fd %s while keeping control mbox %s",
                    shared_ddr_dev, mbox_path);
        } else {
            log_msg("open shared DDR data fd %s failed: %s; falling back to %s",
                    shared_ddr_dev, strerror(errno), mbox_path);
        }
    }
    int boot_status_fd = g_shared_ddr_data_fd >= 0 ? g_shared_ddr_data_fd : mbox_fd;
    if (jobd_boot_marker_enabled()) {
        if (!mirror_boot_marker_all_slots_mbox_mmap(boot_status_fd, 0x6a010000u) &&
            jobd_mbox_status_mmap_enabled()) {
            mirror_boot_marker_all_slots_mbox_mmap(mbox_fd, 0x6a010000u);
        }
    }
    if (jobd_boot_marker_enabled()) {
        if (jobd_mbox_status_mmap_enabled() &&
            !mirror_host_status_mbox_mmap(boot_status_fd, PACC_KERNEL_JOB_ID, 0, 0x6a11)) {
            mirror_host_status_mbox_mmap(mbox_fd, PACC_KERNEL_JOB_ID, 0, 0x6a11);
        }
    }
    int map_fd = mbox_fd;
    bool close_map_fd = false;
    int shared_probe_fd = g_shared_ddr_data_fd >= 0 ? g_shared_ddr_data_fd : mbox_fd;
    const char *shared_probe_name = g_shared_ddr_data_fd >= 0 ? shared_ddr_dev : mbox_path;
    if (!jobd_force_devmem_enabled() && probe_shared_ddr_mmap(shared_probe_fd)) {
        map_fd = shared_probe_fd;
        g_map_uses_shared_ddr_offsets = true;
        log_msg("using %s mmap with shared-DDR-relative offsets mmap_user_off=0x%" PRIx64
                " fd_user_off=0x%" PRIx64,
                shared_probe_name, g_shared_ddr_mmap_user_off, g_shared_ddr_fd_user_off);
    } else {
        map_fd = open(devmem, O_RDWR | O_SYNC | O_CLOEXEC);
        if (map_fd < 0) {
            log_msg("open %s failed after mbox mmap probe: %s", devmem, strerror(errno));
            close(mbox_fd);
            return 1;
        }
        close_map_fd = true;
        g_map_uses_shared_ddr_offsets = false;
        log_msg("using %s physical mmap for shared DDR; doorbell/IRQ stay on %s",
                devmem, mbox_path);
    }
    early_devmem_diag_marker(devmem, 0x6d130000u, 4);
    if (jobd_claim_pacc_id_enabled()) {
        if (claim_pacc_id_from_shared_ddr(map_fd)) {
            early_devmem_diag_marker(devmem, 0x6d140000u | ((uint32_t)g_pacc_id << 8), 5);
        } else {
            early_devmem_diag_marker(devmem, 0x6d14e000u | ((uint32_t)g_pacc_id << 8), 5);
        }
    } else {
        early_devmem_diag_marker(devmem, 0x6d14f000u | ((uint32_t)g_pacc_id << 8), 5);
    }
    if (jobd_full_ddr_map_enabled() && g_ddr_info.ddr_base && g_ddr_info.ddr_size) {
        uint64_t map_bytes = jobd_full_ddr_map_bytes();
        if (map_bytes > g_ddr_info.ddr_size) {
            map_bytes = g_ddr_info.ddr_size;
        }
        if (map_bytes != 0 &&
            map_phys(map_fd, g_ddr_info.ddr_base, (size_t)map_bytes,
                     &g_shared_ddr_full_map) == 0) {
            g_shared_ddr_full_map_valid = true;
            log_msg("mapped shared DDR window once: base=0x%" PRIx64
                    " size=0x%" PRIx64 " total=0x%" PRIx64 " ptr=%p",
                    g_ddr_info.ddr_base, map_bytes, g_ddr_info.ddr_size,
                    g_shared_ddr_full_map.ptr);
        } else {
            log_msg("shared DDR window mmap unavailable; falling back to per-access mmap");
        }
    } else {
        log_msg("full shared DDR mmap disabled; using per-access mmap");
    }
    if (jobd_kernel_slot_map_enabled() && g_ddr_info.ddr_base && g_ddr_info.ddr_size) {
        uint64_t slot_bytes = jobd_kernel_slot_map_bytes();
        uint64_t slot_off = jobd_kernel_slot_map_off(slot_bytes);
        if (slot_bytes > g_ddr_info.ddr_size) {
            slot_bytes = g_ddr_info.ddr_size;
            slot_off = 0;
        }
        if (slot_bytes != 0 &&
            slot_off <= g_ddr_info.ddr_size - slot_bytes &&
            map_phys(map_fd, g_ddr_info.ddr_base + slot_off, (size_t)slot_bytes,
                     &g_kernel_slot_map) == 0) {
            g_kernel_slot_map_valid = true;
            log_msg("mapped kernel slot once: phys=0x%" PRIx64
                    " off=0x%" PRIx64 " size=0x%" PRIx64 " ptr=%p",
                    g_ddr_info.ddr_base + slot_off, slot_off, slot_bytes,
                    g_kernel_slot_map.ptr);
        } else {
            log_msg("kernel slot mmap unavailable; using per-kernel mmap");
        }
    }
    struct Map control_map = {0};
    volatile struct Doorbell *ctl = NULL;
    uint64_t last_seq = 0;
    uint64_t last_table_seq = 0;
    uint64_t last_kernel_seq = 0;
    uint64_t heartbeat_tick = 0;
    early_devmem_diag_marker(devmem, 0x6d150000u | ((uint32_t)g_pacc_id << 8), 6);
    ctl = scan_for_control(map_fd, &g_ddr_info, &control_map);
    if (!ctl) {
        early_devmem_diag_marker(devmem, 0x6d15e000u | ((uint32_t)g_pacc_id << 8), 6);
        if (close_map_fd) close(map_fd);
        close(mbox_fd);
        return 1;
    }
    early_devmem_diag_marker(devmem, 0x6d160000u | ((uint32_t)g_pacc_id << 8), 7);
    g_control_window = (volatile uint8_t *)ctl;
    g_control_map_base = control_map.base;
    g_control_map_len = control_map.map_len;
    clear_stale_control_region(map_fd, &control_map);
    run_xsfmm_smoke_if_requested();
    early_devmem_diag_marker(devmem, 0x6d170000u | ((uint32_t)g_pacc_id << 8), 8);
    if (jobd_boot_marker_enabled()) {
        if (!mirror_boot_marker_all_slots_mbox_mmap(map_fd, 0x6a020000u) &&
            jobd_mbox_status_mmap_enabled()) {
            mirror_boot_marker_all_slots_mbox_mmap(mbox_fd, 0x6a020000u);
        }
        mirror_host_status(map_fd, PACC_KERNEL_JOB_ID, 0, 0x6a21);
    }
    early_devmem_diag_marker(devmem, 0x6d180000u | ((uint32_t)g_pacc_id << 8), 9);
    if (jobd_seed_current_jobs_enabled()) {
        seed_last_seen_sequences(ctl, &last_seq, &last_kernel_seq);
    }
    memset(g_control_snapshot, 0, sizeof(g_control_snapshot));
    uint64_t initial_deadline = monotonic_us() + initial_scan_us();
    uint32_t initial_detail = 0;
    uint64_t initial_seq = 0;
    bool initial_dispatched = false;
    do {
        memset(g_control_snapshot, 0, sizeof(g_control_snapshot));
        sync_map_for_cpu(&control_map);
        if (read_control_snapshot(map_fd, g_control_snapshot, sizeof(g_control_snapshot)) != 0) {
            initial_detail = 0xffffu;
            initial_seq = 0;
            sleep_us(1000);
            continue;
        }
        initial_detail = control_snapshot_detail(g_control_snapshot,
                                                 last_seq,
                                                 last_kernel_seq,
                                                 &initial_seq);
        enum DispatchPollResult first_result =
            dispatch_any_job(map_fd,
                             (volatile struct Doorbell *)(void *)g_control_snapshot,
                             &jobs,
                             strict,
                             &last_seq,
                             &last_table_seq,
                             &last_kernel_seq);
        if (first_result == DISPATCH_HANDLED) {
            initial_dispatched = true;
            break;
        } else {
            sleep_us(1000);
        }
    } while (initial_scan_us() != 0 && monotonic_us() < initial_deadline);
    if (!initial_dispatched) {
        uint32_t status = 0x6bc00000u | (initial_detail & 0xffffu);
        mirror_host_status(map_fd, PACC_KERNEL_JOB_ID, initial_seq, status);
        if (jobd_loop_trace_enabled()) {
            mirror_host_status_control_window_direct(PACC_KERNEL_JOB_ID, initial_seq, 0x6bd00000u);
        }
        write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, initial_seq, 0x6ac0, initial_detail);
    }
    for (;;) {
        enum DispatchPollResult poll_result;
        volatile struct Doorbell *dispatch_ctl = ctl;
        bool pending_job = false;
        bool woke_from_poll = false;
        uint32_t snapshot_detail = 0;

        poll_result = maybe_dispatch_gemm_arg_slot_direct(map_fd, &last_seq);
        if (poll_result == DISPATCH_HANDLED) {
            write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                              0x7007, (uint32_t)poll_result);
            continue;
        }

        poll_result = maybe_dispatch_arg_slot_job(map_fd,
                                                  &jobs,
                                                  strict,
                                                  &last_seq,
                                                  &last_table_seq);
        if (poll_result == DISPATCH_HANDLED) {
            write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                              0x7007, (uint32_t)poll_result);
            continue;
        }
        if (jobd_skip_irq_poll_enabled()) {
            write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                              0x70a0, 0);
            sleep_when_idle();
            continue;
        }

        if (jobd_loop_trace_enabled()) {
            mirror_host_status_control_window_direct(PACC_KERNEL_JOB_ID, heartbeat_tick + 1, 0x7b100000u);
        }
        if (jobd_boot_marker_enabled()) {
            mirror_host_status(map_fd,
                               PACC_KERNEL_JOB_ID,
                               heartbeat_tick + 1,
                               0x7a100000u);
        }
        if (jobd_loop_trace_enabled()) {
            mirror_host_status_control_window_direct(PACC_KERNEL_JOB_ID, heartbeat_tick + 1, 0x7b110000u);
        }
        pending_job = control_has_pending_job(map_fd, last_seq, last_kernel_seq);
        if (jobd_loop_trace_enabled()) {
            mirror_host_status_control_window_direct(PACC_KERNEL_JOB_ID, heartbeat_tick + 1,
                                                     pending_job ? 0x7b120001u : 0x7b120000u);
        }
        if (g_preloaded_completion_sticky && !pending_job) {
            mirror_host_status(map_fd,
                               g_preloaded_completion_job_id,
                               g_preloaded_completion_seq,
                               g_preloaded_completion_status);
            write_jobd_beacon(map_fd,
                              g_preloaded_completion_job_id,
                              g_preloaded_completion_seq,
                              g_preloaded_completion_status == 0 ? 0x511b : g_preloaded_completion_status,
                              g_preloaded_completion_status);
            sleep_when_idle();
            continue;
        }
        if (g_kernel_completion_beacon_sticky && !pending_job) {
            mirror_host_status(map_fd,
                               PACC_KERNEL_JOB_ID,
                               g_kernel_completion_beacon_seq,
                               g_kernel_completion_beacon_status);
            write_jobd_beacon(map_fd,
                              PACC_KERNEL_JOB_ID,
                              g_kernel_completion_beacon_seq,
                              g_kernel_completion_beacon_status == 0 ? 0x511b : g_kernel_completion_beacon_status,
                              g_kernel_completion_beacon_status == 0 ? 0 : g_kernel_completion_beacon_status);
            sleep_when_idle();
            continue;
        }
        mirror_progress_status(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7001);
        write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7001, 0);
        mirror_progress_status(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                               pending_job ? 0x7005 : 0x7002);
        write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                          pending_job ? 0x7005 : 0x7002, 0);
        if (!pending_job && jobd_notify_irq_enabled() && g_response_irq_pending) {
            int ret;
            jobd_io_fence();
            ret = notify_zluda_irq(mbox_fd);
            jobd_io_fence();
            if (ret < 0) {
                log_msg("failed to response before poll: %d", errno);
                g_response_irq_pending = false;
                pending_job = control_has_pending_job(map_fd, last_seq, last_kernel_seq);
                if (pending_job) {
                    mirror_progress_status(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7006);
                    write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7006, 0);
                }
                goto after_response_irq;
            }
            g_response_irq_pending = false;
            /*
             * The host may publish the next shared-DDR doorbell while jobd is
             * raising the previous completion IRQ.  If we enter poll without
             * a fresh scan, that IRQ can be consumed by the host as a stale
             * response and the PACC-side poll can sleep past the new job.
             */
            pending_job = control_has_pending_job(map_fd, last_seq, last_kernel_seq);
            if (pending_job) {
                mirror_progress_status(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7006);
                write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7006, 0);
            }
        }
after_response_irq:
        if (!pending_job && pre_poll_scan_us() != 0) {
            uint64_t deadline = monotonic_us() + pre_poll_scan_us();
            uint64_t sleep_step = pre_poll_scan_sleep_us();
            uint64_t scans = 0;

            while (!pending_job && monotonic_us() < deadline) {
                jobd_io_fence();
                if (sleep_step != 0) {
                    sleep_us(sleep_step);
                }
                pending_job = control_has_pending_job(map_fd, last_seq, last_kernel_seq);
                if (!pending_job && find_pending_arg_slot_job(map_fd, NULL)) {
                    pending_job = true;
                }
                scans++;
            }
            if (pending_job) {
                mirror_progress_status(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7008);
                write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7008,
                                  (uint32_t)(scans & 0xffffffffu));
            }
        }
        if (jobd_wait_for_control_enabled() && jobd_mbox_poll_enabled() && !pending_job) {
            woke_from_poll = wait_for_control(mbox_fd);
            mirror_progress_status(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7003);
            write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7003, 0);
        }

        snapshot_detail = load_dispatch_snapshot(map_fd,
                                                 &control_map,
                                                 g_control_snapshot,
                                                 last_seq,
                                                 last_kernel_seq,
                                                 &dispatch_ctl);
        if (snapshot_detail == 0xffffu) {
            mirror_progress_status(map_fd, 0, heartbeat_tick, 0x6abe);
        }
        if (woke_from_poll && (snapshot_detail & 0x3u) == 0) {
            uint64_t deadline = monotonic_us() + post_irq_scan_us();
            uint64_t sleep_step = post_irq_scan_sleep_us();
            uint64_t scans = 0;

            while ((snapshot_detail & 0x3u) == 0 &&
                   post_irq_scan_us() != 0 &&
                   monotonic_us() < deadline) {
                if (sleep_step != 0) {
                    sleep_us(sleep_step);
                }
                snapshot_detail = load_dispatch_snapshot(map_fd,
                                                         &control_map,
                                                         g_control_snapshot,
                                                         last_seq,
                                                         last_kernel_seq,
                                                         &dispatch_ctl);
                scans++;
            }
            if ((snapshot_detail & 0x3u) == 0 && scans != 0) {
                trace_msg("post-IRQ scan found no new job after %" PRIu64
                          " scans detail=0x%x last_seq=%" PRIu64
                          " last_kernel_seq=%" PRIu64,
                          scans, snapshot_detail, last_seq, last_kernel_seq);
            } else if (scans != 0) {
                trace_msg("post-IRQ scan matched new job after %" PRIu64
                          " scans detail=0x%x",
                          scans, snapshot_detail);
            }
        }
        if ((snapshot_detail & 0x3u) == 0) {
            struct PaccJobDesc mbox_desc;
            if (read_mbox_kernel_desc(mbox_fd, &mbox_desc) &&
                mbox_desc.seq != last_kernel_seq &&
                !kernel_completion_seq_visible(map_fd, mbox_desc.seq)) {
                memset(g_control_snapshot, 0, sizeof(g_control_snapshot));
                memcpy(g_control_snapshot, &mbox_desc, sizeof(mbox_desc));
                dispatch_ctl = (volatile struct Doorbell *)(void *)g_control_snapshot;
                snapshot_detail = control_snapshot_detail(g_control_snapshot,
                                                          last_seq,
                                                          last_kernel_seq,
                                                          NULL);
                write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID,
                                  mbox_desc.seq, 0x7080, snapshot_detail);
            } else {
                write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID,
                                  last_kernel_seq, 0x7081, snapshot_detail);
            }
        }
        if ((snapshot_detail & 0x1f2u) == 0x1f0u) {
            const struct PaccJobDesc *snap_desc =
                (const struct PaccJobDesc *)(const void *)g_control_snapshot;
            if (snap_desc->seq != 0 && snap_desc->seq == last_kernel_seq) {
                last_kernel_seq = 0;
                write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID,
                                  snap_desc->seq, 0x7082, snapshot_detail);
            }
        }
        if (!dispatch_ctl) {
            dispatch_ctl = ctl;
        }
        mirror_progress_status(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                               0x70040000u | snapshot_detail);
        write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq, 0x7004, snapshot_detail);
        if (jobd_boot_marker_enabled() && (snapshot_detail & 0x3u) == 0) {
            uint32_t loop_status = 0x7a000000u | (snapshot_detail & 0xffffu);
            mirror_host_status(map_fd,
                               PACC_KERNEL_JOB_ID,
                               heartbeat_tick + 1,
                               loop_status);
        }
        maybe_heartbeat_control(map_fd, g_control_snapshot, ++heartbeat_tick, last_kernel_seq);
        poll_result = dispatch_any_job(
            map_fd,
            dispatch_ctl,
            &jobs,
            strict,
            &last_seq,
            &last_table_seq,
            &last_kernel_seq);
        write_jobd_beacon(map_fd, PACC_KERNEL_JOB_ID, last_kernel_seq,
                          0x7007, (uint32_t)poll_result);
        if (poll_result == DISPATCH_HANDLED) {
        } else if (!jobd_mbox_poll_enabled() || woke_from_poll) {
            sleep_when_idle();
        }
    }
}
