/*
 * cuBLAS Shim for hetGPU
 *
 * Routes GEMM calls into the PACC runtime job ABI.
 * Optional host-memory CPU fallback is env-gated for fake-CUDA inference bring-up.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>

// Real cuBLAS library handle
static void* real_cublas_handle = NULL;
static void* real_cublaslt_handle_from_cublas = NULL;

// Get real cuBLAS library handle - DISABLED for virtual device backend
// Loading real cuBLAS causes recursive symbol resolution or initialization
// that crashes with our virtual CUDA driver. Use fallback implementations instead.
static void* get_real_cublas() {
    // Always return NULL to use our fallback implementations.
    // The hetGPU virtual device doesn't support forwarding to real cuBLAS
    // because real cuBLAS tries to initialize with our fake CUDA context.
    return NULL;
}

// cuBLAS types
typedef void* cublasHandle_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1
} cublasFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

typedef enum {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

typedef enum {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
    CUBLAS_ATOMICS_NOT_ALLOWED = 0,
    CUBLAS_ATOMICS_ALLOWED = 1
} cublasAtomicsMode_t;

typedef enum {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO0 = 0
} cublasGemmAlgo_t;

typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_32I = 72
} cublasComputeType_t;

typedef enum {
    CUBLAS_DEFAULT_MATH = 0,
} cublasMath_t;

typedef enum {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_16F = 2,
    CUDA_R_8I = 3,
    CUDA_R_8U = 8,
    CUDA_R_32I = 10,
    CUDA_R_16BF = 14
} cudaDataType;

typedef int cudaError_t;
typedef int cudaMemcpyKind;

enum {
    HETGPU_CUDA_MEMCPY_HOST_TO_HOST = 0,
    HETGPU_CUDA_MEMCPY_HOST_TO_DEVICE = 1,
    HETGPU_CUDA_MEMCPY_DEVICE_TO_HOST = 2,
    HETGPU_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3,
    HETGPU_CUDA_MEMCPY_DEFAULT = 4,
};

#define HETGPU_PACC_DTYPE_INT8 0
#define HETGPU_PACC_DTYPE_UINT8 1
#define HETGPU_PACC_DTYPE_INT32 2
#define HETGPU_PACC_DTYPE_F16 3
#define HETGPU_PACC_DTYPE_F32 4
#define HETGPU_PACC_DTYPE_BF16 5

static int hetgpu_pacc_dtype(cudaDataType type) {
    switch (type) {
        case CUDA_R_8I:
            return HETGPU_PACC_DTYPE_INT8;
        case CUDA_R_8U:
            return HETGPU_PACC_DTYPE_UINT8;
        case CUDA_R_32I:
            return HETGPU_PACC_DTYPE_INT32;
        case CUDA_R_16F:
            return HETGPU_PACC_DTYPE_F16;
        case CUDA_R_32F:
            return HETGPU_PACC_DTYPE_F32;
        case CUDA_R_16BF:
            return HETGPU_PACC_DTYPE_BF16;
        default:
            return (int)type;
    }
}

static int hetgpu_cublas_trace_enabled(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        const char *env = getenv("HETGPU_CUBLAS_TRACE");
        enabled = env && strcmp(env, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

#define DEBUG_LOG(fmt, ...) do { \
    if (hetgpu_cublas_trace_enabled()) { \
        fprintf(stderr, "[hetGPU cublas_shim] " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)

// Helper macro to get function pointer from real cuBLAS and call it
#define GET_REAL_FUNC(func_name, func_type) \
    static func_type real_##func_name = NULL; \
    if (real_##func_name == NULL) { \
        void* lib = get_real_cublas(); \
        if (lib) { \
            real_##func_name = (func_type)dlsym(lib, #func_name); \
        } \
    }

static cublasMath_t g_math_mode = CUBLAS_DEFAULT_MATH;
static size_t host_dtype_size(cudaDataType type);

typedef int (*hetgpu_pacc_submit_gemm_fn)(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType);
typedef int (*hetgpu_pacc_submit_gemm_staged_fn)(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType);
typedef int (*hetgpu_pacc_submit_gemm_staged_on_fn)(
    int dev_id, int slot_id,
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType);
typedef int (*hetgpu_pacc_submit_gemm_staged_tiled_fn)(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType,
    int max_m, int max_n, int max_k);
typedef int (*hetgpu_pacc_submit_gemm_mmvf_small_n_fn)(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType);
typedef unsigned long long (*hetgpu_pacc_resolve_device_addr_fn)(const void *ptr);
typedef int (*hetgpu_pacc_is_device_ptr_fn)(const void *ptr);
typedef size_t (*hetgpu_pacc_allocation_remaining_fn)(const void *ptr);
typedef int (*hetgpu_cuda_get_device_fn)(int *device);

static void *hetgpu_resolve_runtime_symbol(const char *name) {
    dlerror();
    void *sym = dlsym(RTLD_DEFAULT, name);
    return sym;
}

static int hetgpu_pacc_missing_submit_symbol(const char *name) {
    DEBUG_LOG("optional PACC GEMM submit symbol is unavailable: %s", name);
    return -127;
}

static hetgpu_pacc_submit_gemm_fn hetgpu_resolve_pacc_submit_gemm_fn(void) {
    static hetgpu_pacc_submit_gemm_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_submit_gemm_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_submit_gemm");
    }
    return fn;
}

static hetgpu_pacc_submit_gemm_staged_fn hetgpu_resolve_pacc_submit_gemm_staged_fn(void) {
    static hetgpu_pacc_submit_gemm_staged_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_submit_gemm_staged_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_submit_gemm_staged");
    }
    return fn;
}

static hetgpu_pacc_submit_gemm_staged_on_fn hetgpu_resolve_pacc_submit_gemm_staged_on_fn(void) {
    static hetgpu_pacc_submit_gemm_staged_on_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_submit_gemm_staged_on_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_submit_gemm_staged_on");
    }
    return fn;
}

static hetgpu_pacc_submit_gemm_staged_tiled_fn hetgpu_resolve_pacc_submit_gemm_staged_tiled_fn(void) {
    static hetgpu_pacc_submit_gemm_staged_tiled_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_submit_gemm_staged_tiled_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_submit_gemm_staged_tiled");
    }
    return fn;
}

static hetgpu_pacc_submit_gemm_mmvf_small_n_fn hetgpu_resolve_pacc_submit_gemm_mmvf_small_n_fn(void) {
    static hetgpu_pacc_submit_gemm_mmvf_small_n_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_submit_gemm_mmvf_small_n_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_submit_gemm_mmvf_small_n");
    }
    return fn;
}

static int hetgpu_pacc_submit_gemm_checked(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    hetgpu_pacc_submit_gemm_fn fn = hetgpu_resolve_pacc_submit_gemm_fn();
    if (!fn) {
        return hetgpu_pacc_missing_submit_symbol("hetgpu_pacc_submit_gemm");
    }
    return fn(transa, transb, m, n, k,
              alpha,
              A, Atype, lda, strideA,
              B, Btype, ldb, strideB,
              beta,
              C, Ctype, ldc, strideC,
              batchCount, computeType);
}

static int hetgpu_pacc_submit_gemm_staged_checked(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    hetgpu_pacc_submit_gemm_staged_fn fn = hetgpu_resolve_pacc_submit_gemm_staged_fn();
    if (!fn) {
        return hetgpu_pacc_missing_submit_symbol("hetgpu_pacc_submit_gemm_staged");
    }
    return fn(transa, transb, m, n, k,
              alpha,
              A, Atype, lda, strideA,
              B, Btype, ldb, strideB,
              beta,
              C, Ctype, ldc, strideC,
              batchCount, computeType);
}

static int hetgpu_pacc_submit_gemm_staged_on_checked(
    int dev_id, int slot_id,
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    hetgpu_pacc_submit_gemm_staged_on_fn fn = hetgpu_resolve_pacc_submit_gemm_staged_on_fn();
    if (!fn) {
        return hetgpu_pacc_missing_submit_symbol("hetgpu_pacc_submit_gemm_staged_on");
    }
    return fn(dev_id, slot_id,
              transa, transb, m, n, k,
              alpha,
              A, Atype, lda, strideA,
              B, Btype, ldb, strideB,
              beta,
              C, Ctype, ldc, strideC,
              batchCount, computeType);
}

static int hetgpu_pacc_submit_gemm_staged_tiled_checked(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType,
    int max_m, int max_n, int max_k) {
    hetgpu_pacc_submit_gemm_staged_tiled_fn fn =
        hetgpu_resolve_pacc_submit_gemm_staged_tiled_fn();
    if (!fn) {
        return hetgpu_pacc_missing_submit_symbol("hetgpu_pacc_submit_gemm_staged_tiled");
    }
    return fn(transa, transb, m, n, k,
              alpha,
              A, Atype, lda, strideA,
              B, Btype, ldb, strideB,
              beta,
              C, Ctype, ldc, strideC,
              batchCount, computeType,
              max_m, max_n, max_k);
}

static int hetgpu_pacc_submit_gemm_mmvf_small_n_checked(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType) {
    hetgpu_pacc_submit_gemm_mmvf_small_n_fn fn =
        hetgpu_resolve_pacc_submit_gemm_mmvf_small_n_fn();
    if (!fn) {
        return hetgpu_pacc_missing_submit_symbol("hetgpu_pacc_submit_gemm_mmvf_small_n");
    }
    return fn(transa, transb, m, n, k,
              alpha,
              A, Atype, lda, strideA,
              B, Btype, ldb, strideB,
              beta,
              C, Ctype, ldc, strideC,
              batchCount, computeType);
}

static hetgpu_pacc_resolve_device_addr_fn hetgpu_resolve_device_addr_fn(void) {
    static hetgpu_pacc_resolve_device_addr_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_resolve_device_addr_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_resolve_device_addr");
    }
    return fn;
}

static hetgpu_pacc_is_device_ptr_fn hetgpu_is_device_ptr_fn(void) {
    static hetgpu_pacc_is_device_ptr_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_is_device_ptr_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_is_device_ptr");
    }
    return fn;
}

static hetgpu_pacc_allocation_remaining_fn hetgpu_allocation_remaining_fn(void) {
    static hetgpu_pacc_allocation_remaining_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        fn = (hetgpu_pacc_allocation_remaining_fn)
            hetgpu_resolve_runtime_symbol("hetgpu_pacc_allocation_remaining");
    }
    return fn;
}

static unsigned long long hetgpu_pacc_resolve_device_addr_checked(const void *ptr) {
    hetgpu_pacc_resolve_device_addr_fn fn = hetgpu_resolve_device_addr_fn();
    return fn ? fn(ptr) : (unsigned long long)(uintptr_t)ptr;
}

static int hetgpu_pacc_is_device_ptr_checked(const void *ptr) {
    hetgpu_pacc_is_device_ptr_fn fn = hetgpu_is_device_ptr_fn();
    return fn ? fn(ptr) : 0;
}

static size_t hetgpu_pacc_allocation_remaining_checked(const void *ptr) {
    hetgpu_pacc_allocation_remaining_fn fn = hetgpu_allocation_remaining_fn();
    return fn ? fn(ptr) : SIZE_MAX;
}

#define hetgpu_pacc_resolve_device_addr hetgpu_pacc_resolve_device_addr_checked
#define hetgpu_pacc_is_device_ptr hetgpu_pacc_is_device_ptr_checked
#define hetgpu_pacc_allocation_remaining hetgpu_pacc_allocation_remaining_checked
#define hetgpu_pacc_submit_gemm hetgpu_pacc_submit_gemm_checked
#define hetgpu_pacc_submit_gemm_staged hetgpu_pacc_submit_gemm_staged_checked
#define hetgpu_pacc_submit_gemm_staged_on hetgpu_pacc_submit_gemm_staged_on_checked
#define hetgpu_pacc_submit_gemm_staged_tiled hetgpu_pacc_submit_gemm_staged_tiled_checked
#define hetgpu_pacc_submit_gemm_mmvf_small_n hetgpu_pacc_submit_gemm_mmvf_small_n_checked
extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

static int hetgpu_env_is_one(const char *name) {
    const char *value = getenv(name);
    return value && strcmp(value, "1") == 0;
}

static int hetgpu_env_enabled_default(const char *name, int default_value) {
    const char *value = getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return 0;
    }
    return 1;
}

static unsigned long long hetgpu_env_u64_default(const char *name, unsigned long long default_value) {
    const char *value = getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 0);
    if (!end || end == value) {
        return default_value;
    }
    return parsed;
}

static int hetgpu_cublas_fail_open_enabled(void) {
    const char *value = getenv("HETGPU_PACC_CUBLAS_FAIL_OPEN");
    if (!value || !*value) {
        value = getenv("HETGPU_PACC_ASSUME_SUCCESS_ON_WAIT_ERROR");
    }
    if (!value || !*value) {
        return 0;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return 0;
    }
    return 1;
}

static int hetgpu_allow_host_gemm_fallback(void) {
    return hetgpu_env_is_one("HETGPU_PACC_ALLOW_HOST_GEMM_FALLBACK");
}

static int hetgpu_small_gemm_host_fallback_enabled(int m, int n, int k, int batchCount) {
    if (!hetgpu_env_enabled_default("HETGPU_PACC_GEMM_SMALL_HOST_FALLBACK", 0)) {
        return 0;
    }
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) {
        return 0;
    }

    const unsigned long long max_ops =
        hetgpu_env_u64_default("HETGPU_PACC_GEMM_SMALL_MAX_OPS", 65536ULL);
    const unsigned long long max_m =
        hetgpu_env_u64_default("HETGPU_PACC_GEMM_SMALL_MAX_M", 0ULL);
    const unsigned long long max_n =
        hetgpu_env_u64_default("HETGPU_PACC_GEMM_SMALL_MAX_N", 4ULL);
    const unsigned long long max_k =
        hetgpu_env_u64_default("HETGPU_PACC_GEMM_SMALL_MAX_K", 0ULL);
    const unsigned long long um = (unsigned long long)m;
    const unsigned long long un = (unsigned long long)n;
    const unsigned long long uk = (unsigned long long)k;
    const unsigned long long ub = (unsigned long long)batchCount;

    const int any_small_dim =
        hetgpu_env_enabled_default("HETGPU_PACC_GEMM_SMALL_ANY_DIM", 1);
    if (max_m && um > max_m) {
        return 0;
    }
    if (max_n && (any_small_dim ? (um > max_n && un > max_n) : (un > max_n))) {
        return 0;
    }
    if (max_k && uk > max_k) {
        return 0;
    }
    if (um != 0 && un > ULLONG_MAX / um) {
        return 0;
    }
    unsigned long long ops = um * un;
    if (uk != 0 && ops > ULLONG_MAX / uk) {
        return 0;
    }
    ops *= uk;
    if (ub != 0 && ops > ULLONG_MAX / ub) {
        return 0;
    }
    ops *= ub;
    return ops <= max_ops;
}

static int hetgpu_skinny_n_gemm_host_fallback_enabled(int m, int n, int k, int batchCount) {
    if (!hetgpu_env_enabled_default("HETGPU_PACC_GEMM_SKINNY_N_HOST_FALLBACK", 0)) {
        return 0;
    }
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) {
        return 0;
    }

    const unsigned long long max_n =
        hetgpu_env_u64_default("HETGPU_PACC_GEMM_SKINNY_N_MAX_N", 1ULL);
    const unsigned long long min_k =
        hetgpu_env_u64_default("HETGPU_PACC_GEMM_SKINNY_N_MIN_K", 64ULL);
    return (unsigned long long)n <= max_n && (unsigned long long)k >= min_k;
}

static int hetgpu_non_skinny_n_gemm_host_fallback_enabled(int m, int n, int k, int batchCount) {
    if (!hetgpu_env_enabled_default("HETGPU_PACC_GEMM_NON_SKINNY_N_HOST_FALLBACK", 0)) {
        return 0;
    }
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) {
        return 0;
    }

    const unsigned long long max_pacc_n =
        hetgpu_env_u64_default("HETGPU_PACC_GEMM_PACC_MAX_N", 1ULL);
    return (unsigned long long)n > max_pacc_n;
}

static int hetgpu_tail_gemm_host_fallback_enabled(int m, int n, int k, int batchCount) {
    if (!hetgpu_env_enabled_default("HETGPU_PACC_GEMM_TAIL_HOST_FALLBACK", 1)) {
        return 0;
    }
    return hetgpu_small_gemm_host_fallback_enabled(m, n, k, batchCount);
}

static int hetgpu_default_pacc_gemm_devices(int devices[4]) {
    int count = 0;
    for (int dev = 0; dev < 4; ++dev) {
        char pacc_path[32];
        char helper_path[64];
        snprintf(pacc_path, sizeof(pacc_path), "/dev/pacc%d", dev);
        snprintf(helper_path, sizeof(helper_path), "/dev/hetgpu_pacc_mbox_ddr_coh%d", dev);
        if (access(pacc_path, F_OK) == 0 || access(helper_path, F_OK) == 0) {
            devices[count++] = dev;
        }
    }
    if (count == 0) {
        devices[count++] = 0;
    }
    return count;
}

static int hetgpu_parse_pacc_gemm_devices(int devices[4]) {
    const char *env = getenv("HETGPU_PACC_GEMM_DEVICES");
    int count = 0;
    if (!env || !*env) {
        return hetgpu_default_pacc_gemm_devices(devices);
    }

    const char *p = env;
    while (*p && count < 4) {
        while (*p == ',' || *p == ';' || *p == ':' || *p == ' ' || *p == '\t') {
            ++p;
        }
        if (!*p) {
            break;
        }
        char *end = NULL;
        long dev = strtol(p, &end, 0);
        if (end == p) {
            break;
        }
        if (dev >= 0 && dev < 4) {
            int duplicate = 0;
            for (int i = 0; i < count; ++i) {
                if (devices[i] == (int)dev) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) {
                devices[count++] = (int)dev;
            }
        }
        p = end;
    }
    if (count == 0) {
        devices[count++] = 0;
    }
    return count;
}

static int hetgpu_pacc_physical_device_for_logical_cublas(int logical) {
    const char *visible = getenv("HETGPU_PACC_VISIBLE_DEVICES");
    if (!visible || !*visible) {
        return logical;
    }
    if (!strchr(visible, ',') && !strchr(visible, ';') && !strchr(visible, ':') &&
        !strchr(visible, ' ') && !strchr(visible, '\t')) {
        return logical;
    }
    int current_logical = 0;
    const char *p = visible;
    while (*p) {
        while (*p == ',' || *p == ';' || *p == ':' || *p == ' ' || *p == '\t') {
            ++p;
        }
        if (!*p) {
            break;
        }
        char *end = NULL;
        long physical = strtol(p, &end, 0);
        if (end == p) {
            break;
        }
        if (physical >= 0 && physical < 4) {
            if (current_logical == logical) {
                return (int)physical;
            }
            current_logical++;
        }
        p = end;
    }
    return logical;
}

static int hetgpu_current_pacc_device_for_cublas(void) {
    static hetgpu_cuda_get_device_fn cuda_get_device = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        cuda_get_device = (hetgpu_cuda_get_device_fn)dlsym(RTLD_DEFAULT, "cudaGetDevice");
    }
    int logical = 0;
    if (cuda_get_device) {
        int tmp = 0;
        if (cuda_get_device(&tmp) == 0) {
            logical = tmp;
        }
    }
    int physical = hetgpu_pacc_physical_device_for_logical_cublas(logical);
    return (physical >= 0 && physical < 4) ? physical : 0;
}

static int hetgpu_pacc_gemm_layer_local_enabled(void) {
    return hetgpu_env_enabled_default("HETGPU_PACC_GEMM_LAYER_LOCAL", 0);
}

static int hetgpu_cublas_noop_enabled(void) {
    return hetgpu_env_enabled_default("HETGPU_PACC_CUBLAS_NOOP", 0);
}

static int hetgpu_cublas_noop_return_success(const char *name, int m, int n, int k, int batchCount) {
    if (!hetgpu_cublas_noop_enabled()) {
        return 0;
    }
    DEBUG_LOG("%s CUBLAS_NOOP active; treating GEMM as successful m=%d n=%d k=%d batch=%d",
              name, m, n, k, batchCount);
    return 1;
}

static int pacc_runtime_marked_ready(void) {
    if (!hetgpu_env_is_one("HETGPU_PACC_ENFORCE_RUNTIME_READY")) {
        return 1;
    }
    return hetgpu_env_is_one("HETGPU_PACC_RUNTIME_READY") ||
           hetgpu_env_is_one("HETGPU_PACC_BOOT_RUNTIME");
}

static int hetgpu_checked_add_size(size_t a, size_t b, size_t *out) {
    if (SIZE_MAX - a < b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int hetgpu_checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int hetgpu_gemm_matrix_span_elems(cublasOperation_t trans, int rows, int cols, int ld, size_t *out) {
    if (!out || rows < 0 || cols < 0 || ld <= 0) {
        return 0;
    }
    if (rows == 0 || cols == 0) {
        *out = 0;
        return 1;
    }
    size_t r = (size_t)rows;
    size_t c = (size_t)cols;
    size_t lead = (size_t)ld;
    size_t term_a = 0;
    size_t term_b = 0;
    if (trans == CUBLAS_OP_N) {
        if (!hetgpu_checked_mul_size(c - 1, lead, &term_a)) {
            return 0;
        }
        return hetgpu_checked_add_size(term_a, r, out);
    }
    if (!hetgpu_checked_mul_size(r - 1, lead, &term_a)) {
        return 0;
    }
    term_b = c;
    return hetgpu_checked_add_size(term_a, term_b, out);
}

static int hetgpu_gemm_c_span_elems(int m, int n, int ldc, size_t *out) {
    if (!out || m < 0 || n < 0 || ldc <= 0) {
        return 0;
    }
    if (m == 0 || n == 0) {
        *out = 0;
        return 1;
    }
    size_t col_span = 0;
    if (!hetgpu_checked_mul_size((size_t)(n - 1), (size_t)ldc, &col_span)) {
        return 0;
    }
    return hetgpu_checked_add_size(col_span, (size_t)m, out);
}

static int hetgpu_gemm_validate_region(
    const char *name,
    const char *which,
    const void *ptr,
    size_t elems,
    size_t elem_size) {
    size_t bytes = 0;
    if (!ptr || elem_size == 0 || !hetgpu_checked_mul_size(elems, elem_size, &bytes)) {
        DEBUG_LOG("%s invalid %s GEMM region ptr=%p elems=%zu elem=%zu", name, which, ptr, elems, elem_size);
        return 0;
    }
    if (!hetgpu_pacc_is_device_ptr(ptr)) {
        if (hetgpu_env_is_one("HETGPU_PACC_ALLOW_HOST_DEVICE_MEM")) {
            return 1;
        }
        fprintf(stderr,
                "[hetGPU cublas_shim] %s refuses host %s GEMM region without HETGPU_PACC_ALLOW_HOST_DEVICE_MEM=1: ptr=%p need=%zu\n",
                name, which, ptr, bytes);
        return 0;
    }
    size_t remaining = hetgpu_pacc_allocation_remaining(ptr);
    if (remaining != SIZE_MAX && bytes > remaining) {
        fprintf(stderr,
                "[hetGPU cublas_shim] %s refuses %s out-of-bounds GEMM region: ptr=%p need=%zu remaining=%zu\n",
                name, which, ptr, bytes, remaining);
        return 0;
    }
    return 1;
}

static cublasStatus_t hetgpu_validate_gemm_regions(
    const char *name,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *A, cudaDataType Atype, int lda,
    const void *B, cudaDataType Btype, int ldb,
    const void *C, cudaDataType Ctype, int ldc) {
    size_t a_elems = 0;
    size_t b_elems = 0;
    size_t c_elems = 0;
    size_t a_size = host_dtype_size(Atype);
    size_t b_size = host_dtype_size(Btype);
    size_t c_size = host_dtype_size(Ctype);
    if (!hetgpu_gemm_matrix_span_elems(transa, m, k, lda, &a_elems) ||
        !hetgpu_gemm_matrix_span_elems(transb, k, n, ldb, &b_elems) ||
        !hetgpu_gemm_c_span_elems(m, n, ldc, &c_elems)) {
        fprintf(stderr,
                "[hetGPU cublas_shim] %s invalid GEMM shape m=%d n=%d k=%d lda=%d ldb=%d ldc=%d trans=%d/%d\n",
                name, m, n, k, lda, ldb, ldc, (int)transa, (int)transb);
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!hetgpu_gemm_validate_region(name, "A", A, a_elems, a_size) ||
        !hetgpu_gemm_validate_region(name, "B", B, b_elems, b_size) ||
        !hetgpu_gemm_validate_region(name, "C", C, c_elems, c_size)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

static int hetgpu_copy_pointer_array_to_host(void **dst, const void *src, size_t ptr_bytes) {
    if (!dst || !src) {
        return -1;
    }
    if (!hetgpu_pacc_is_device_ptr(src)) {
        memcpy(dst, src, ptr_bytes);
        return 0;
    }
    if (cudaMemcpy(dst, src, ptr_bytes, HETGPU_CUDA_MEMCPY_DEVICE_TO_HOST) == 0) {
        return 0;
    }
    unsigned long long resolved = hetgpu_pacc_resolve_device_addr(src);
    if (resolved != 0) {
        memcpy(dst, (const void *)(uintptr_t)resolved, ptr_bytes);
        return 0;
    }
    return -1;
}

static float host_bf16_to_float(uint16_t value) {
    union { uint32_t u; float f; } conv;
    conv.u = ((uint32_t)value) << 16;
    return conv.f;
}

static float host_f16_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)value >> 10) & 0x1fu;
    uint32_t frac = (uint32_t)value & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03ffu;
            bits = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    union { uint32_t u; float f; } conv;
    conv.u = bits;
    return conv.f;
}

static uint16_t host_float_to_f16(float value) {
    union { float f; uint32_t u; } conv;
    conv.f = value;
    uint32_t sign = (conv.u >> 16) & 0x8000u;
    uint32_t mant = conv.u & 0x007fffffu;
    int32_t exp = (int32_t)((conv.u >> 23) & 0xffu) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t shifted = mant >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((shifted + 0x00001000u) >> 13));
    }
    if (exp >= 0x1f) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x00001000u) >> 13));
}

static uint16_t host_float_to_bf16(float value) {
    union { float f; uint32_t u; } conv;
    conv.f = value;
    uint32_t lsb = (conv.u >> 16) & 1U;
    uint32_t rounding_bias = 0x7fffU + lsb;
    return (uint16_t)((conv.u + rounding_bias) >> 16);
}

static int8_t host_float_to_i8(float value) {
    if (value >= 127.0f) return 127;
    if (value <= -128.0f) return -128;
    return (int8_t)(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

static uint8_t host_float_to_u8(float value) {
    if (value >= 255.0f) return 255;
    if (value <= 0.0f) return 0;
    return (uint8_t)(value + 0.5f);
}

static int32_t host_float_to_i32(float value) {
    if (value >= 2147483647.0f) return 2147483647;
    if (value <= -2147483648.0f) return (-2147483647 - 1);
    return (int32_t)(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

static float host_read_scale(const void *scale, cublasComputeType_t computeType, float default_value) {
    if (!scale) {
        return default_value;
    }
    if (computeType == CUBLAS_COMPUTE_32I) {
        return (float)(*(const int32_t *)scale);
    }
    return *(const float *)scale;
}

static size_t host_dtype_size(cudaDataType type) {
    switch (type) {
        case CUDA_R_8I: return sizeof(int8_t);
        case CUDA_R_8U: return sizeof(uint8_t);
        case CUDA_R_32I: return sizeof(int32_t);
        case CUDA_R_16F: return sizeof(uint16_t);
        case CUDA_R_32F: return sizeof(float);
        case CUDA_R_16BF: return sizeof(uint16_t);
        default: return 0;
    }
}

static float host_gemm_load(const void *base, cudaDataType type, int row, int col, int ld, cublasOperation_t trans) {
    int r = row;
    int c = col;
    if (trans != CUBLAS_OP_N) {
        r = col;
        c = row;
    }
    size_t idx = (size_t)r + (size_t)c * (size_t)ld;
    if (type == CUDA_R_8I) {
        return ((const int8_t *)base)[idx];
    }
    if (type == CUDA_R_8U) {
        return ((const uint8_t *)base)[idx];
    }
    if (type == CUDA_R_32I) {
        return ((const int32_t *)base)[idx];
    }
    if (type == CUDA_R_16F) {
        return host_f16_to_float(((const uint16_t *)base)[idx]);
    }
    if (type == CUDA_R_32F) {
        return ((const float *)base)[idx];
    }
    if (type == CUDA_R_16BF) {
        return host_bf16_to_float(((const uint16_t *)base)[idx]);
    }
    return 0.0f;
}

static float host_gemm_load_c(const void *base, cudaDataType type, size_t idx) {
    if (type == CUDA_R_8I) {
        return ((const int8_t *)base)[idx];
    }
    if (type == CUDA_R_8U) {
        return ((const uint8_t *)base)[idx];
    }
    if (type == CUDA_R_32I) {
        return ((const int32_t *)base)[idx];
    }
    if (type == CUDA_R_16F) {
        return host_f16_to_float(((const uint16_t *)base)[idx]);
    }
    if (type == CUDA_R_32F) {
        return ((const float *)base)[idx];
    }
    if (type == CUDA_R_16BF) {
        return host_bf16_to_float(((const uint16_t *)base)[idx]);
    }
    return 0.0f;
}

static void host_gemm_store_c(void *base, cudaDataType type, size_t idx, float value) {
    if (type == CUDA_R_8I) {
        ((int8_t *)base)[idx] = host_float_to_i8(value);
    } else if (type == CUDA_R_8U) {
        ((uint8_t *)base)[idx] = host_float_to_u8(value);
    } else if (type == CUDA_R_32I) {
        ((int32_t *)base)[idx] = host_float_to_i32(value);
    } else if (type == CUDA_R_16F) {
        ((uint16_t *)base)[idx] = host_float_to_f16(value);
    } else if (type == CUDA_R_32F) {
        ((float *)base)[idx] = value;
    } else if (type == CUDA_R_16BF) {
        ((uint16_t *)base)[idx] = host_float_to_bf16(value);
    }
}

typedef struct {
    const char *A;
    const char *B;
    char *C;
    cudaDataType Atype;
    cudaDataType Btype;
    cudaDataType Ctype;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    float alpha;
    float beta;
    int begin;
    int end;
} hetgpu_host_gemm_worker_ctx_t;

static int hetgpu_host_gemm_thread_count(int tasks) {
    if (tasks <= 1) {
        return 1;
    }
    unsigned long long requested = hetgpu_env_u64_default("HETGPU_PACC_HOST_GEMM_THREADS", 32ULL);
    if (requested < 1ULL) {
        requested = 1ULL;
    }
    if (requested > 64ULL) {
        requested = 64ULL;
    }
    if (requested > (unsigned long long)tasks) {
        requested = (unsigned long long)tasks;
    }
    return (int)requested;
}

static void hetgpu_host_gemm_compute_range(const hetgpu_host_gemm_worker_ctx_t *ctx) {
    for (int task = ctx->begin; task < ctx->end; ++task) {
        int row = task % ctx->m;
        int col = task / ctx->m;
        float acc = 0.0f;
        for (int inner = 0; inner < ctx->k; ++inner) {
            acc += host_gemm_load(ctx->A, ctx->Atype, row, inner, ctx->lda, ctx->transa) *
                   host_gemm_load(ctx->B, ctx->Btype, inner, col, ctx->ldb, ctx->transb);
        }
        size_t c_idx = (size_t)row + (size_t)col * (size_t)ctx->ldc;
        float old = ctx->beta != 0.0f ? host_gemm_load_c(ctx->C, ctx->Ctype, c_idx) : 0.0f;
        host_gemm_store_c(ctx->C, ctx->Ctype, c_idx, ctx->alpha * acc + ctx->beta * old);
    }
}

static void *hetgpu_host_gemm_worker_main(void *arg) {
    hetgpu_host_gemm_compute_range((const hetgpu_host_gemm_worker_ctx_t *)arg);
    return NULL;
}

static cublasStatus_t host_gemm_fallback(
    const char *name,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType Atype, int lda, long long strideA,
    const void *B, cudaDataType Btype, int ldb, long long strideB,
    const void *beta,
    void *C, cudaDataType Ctype, int ldc, long long strideC,
    int batchCount, cublasComputeType_t computeType) {
    if (!hetgpu_env_is_one("HETGPU_PACC_ALLOW_HOST_DEVICE_MEM")) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (!alpha || !A || !B || !beta || !C) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (host_dtype_size(Atype) == 0 || host_dtype_size(Btype) == 0 || host_dtype_size(Ctype) == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (transa != CUBLAS_OP_N && transa != CUBLAS_OP_T && transa != CUBLAS_OP_C) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (transb != CUBLAS_OP_N && transb != CUBLAS_OP_T && transb != CUBLAS_OP_C) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (m < 0 || n < 0 || k < 0 || batchCount < 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const float a_scale = host_read_scale(alpha, computeType, 1.0f);
    const float b_scale = host_read_scale(beta, computeType, 0.0f);
    const size_t a_elem_size = host_dtype_size(Atype);
    const size_t b_elem_size = host_dtype_size(Btype);
    const size_t c_elem_size = host_dtype_size(Ctype);
    const long long a_stride = strideA ? strideA : (long long)lda * (transa == CUBLAS_OP_N ? k : m);
    const long long b_stride = strideB ? strideB : (long long)ldb * (transb == CUBLAS_OP_N ? n : k);
    const long long c_stride = strideC ? strideC : (long long)ldc * n;
    if (a_stride < 0 || b_stride < 0 || c_stride < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t a_span = 0, b_span = 0, c_span = 0;
    if (!hetgpu_gemm_matrix_span_elems(transa, m, k, lda, &a_span) ||
        !hetgpu_gemm_matrix_span_elems(transb, k, n, ldb, &b_span) ||
        !hetgpu_gemm_c_span_elems(m, n, ldc, &c_span)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t a_total = a_span;
    size_t b_total = b_span;
    size_t c_total = c_span;
    if (batchCount > 1) {
        size_t batch_tail = (size_t)(batchCount - 1);
        size_t tmp = 0;
        if (!hetgpu_checked_mul_size(batch_tail, (size_t)a_stride, &tmp) ||
            !hetgpu_checked_add_size(tmp, a_span, &a_total) ||
            !hetgpu_checked_mul_size(batch_tail, (size_t)b_stride, &tmp) ||
            !hetgpu_checked_add_size(tmp, b_span, &b_total) ||
            !hetgpu_checked_mul_size(batch_tail, (size_t)c_stride, &tmp) ||
            !hetgpu_checked_add_size(tmp, c_span, &c_total)) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
    }
    size_t a_bytes = 0, b_bytes = 0, c_bytes = 0;
    if (!hetgpu_checked_mul_size(a_total, a_elem_size, &a_bytes) ||
        !hetgpu_checked_mul_size(b_total, b_elem_size, &b_bytes) ||
        !hetgpu_checked_mul_size(c_total, c_elem_size, &c_bytes)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    void *tmp_A = NULL;
    void *tmp_B = NULL;
    void *tmp_C = NULL;
    int copy_C_back = 0;
    const int a_is_device = hetgpu_pacc_is_device_ptr(A);
    const int b_is_device = hetgpu_pacc_is_device_ptr(B);
    const int c_is_device = hetgpu_pacc_is_device_ptr(C);
    const void *host_A = a_is_device ? (const void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(A) : A;
    const void *host_B = b_is_device ? (const void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(B) : B;
    void *host_C = c_is_device ? (void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(C) : C;

    if (a_is_device && !host_A) {
        tmp_A = malloc(a_bytes ? a_bytes : 1);
        if (!tmp_A || cudaMemcpy(tmp_A, A, a_bytes, HETGPU_CUDA_MEMCPY_DEVICE_TO_HOST) != 0) {
            free(tmp_A);
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        host_A = tmp_A;
    }
    if (b_is_device && !host_B) {
        tmp_B = malloc(b_bytes ? b_bytes : 1);
        if (!tmp_B || cudaMemcpy(tmp_B, B, b_bytes, HETGPU_CUDA_MEMCPY_DEVICE_TO_HOST) != 0) {
            free(tmp_A);
            free(tmp_B);
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        host_B = tmp_B;
    }
    if (c_is_device && !host_C) {
        tmp_C = malloc(c_bytes ? c_bytes : 1);
        if (!tmp_C) {
            free(tmp_A);
            free(tmp_B);
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        if (b_scale != 0.0f) {
            if (cudaMemcpy(tmp_C, C, c_bytes, HETGPU_CUDA_MEMCPY_DEVICE_TO_HOST) != 0) {
                free(tmp_A);
                free(tmp_B);
                free(tmp_C);
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
        } else if (c_bytes != 0) {
            memset(tmp_C, 0, c_bytes);
        }
        host_C = tmp_C;
        copy_C_back = 1;
    }
    if (!host_A || !host_B || !host_C) {
        free(tmp_A);
        free(tmp_B);
        free(tmp_C);
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    DEBUG_LOG("%s using host GEMM fallback after PACC failure (A=%d B=%d C=%d)", name, Atype, Btype, Ctype);
    for (int batch = 0; batch < batchCount; ++batch) {
        const char *Ab = (const char *)host_A + (size_t)batch * (size_t)a_stride * a_elem_size;
        const char *Bb = (const char *)host_B + (size_t)batch * (size_t)b_stride * b_elem_size;
        char *Cb = (char *)host_C + (size_t)batch * (size_t)c_stride * c_elem_size;
        int tasks = m * n;
        int threads = hetgpu_host_gemm_thread_count(tasks);
        if (threads <= 1) {
            hetgpu_host_gemm_worker_ctx_t ctx = {
                Ab, Bb, Cb, Atype, Btype, Ctype, transa, transb,
                m, n, k, lda, ldb, ldc, a_scale, b_scale, 0, tasks
            };
            hetgpu_host_gemm_compute_range(&ctx);
        } else {
            pthread_t tids[64];
            hetgpu_host_gemm_worker_ctx_t workers[64];
            int created = 0;
            int failed = 0;
            for (int i = 0; i < threads; ++i) {
                int begin = (int)(((long long)tasks * i) / threads);
                int end = (int)(((long long)tasks * (i + 1)) / threads);
                workers[i] = (hetgpu_host_gemm_worker_ctx_t){
                    Ab, Bb, Cb, Atype, Btype, Ctype, transa, transb,
                    m, n, k, lda, ldb, ldc, a_scale, b_scale, begin, end
                };
                int trc = pthread_create(&tids[i], NULL, hetgpu_host_gemm_worker_main, &workers[i]);
                if (trc != 0) {
                    failed = 1;
                    break;
                }
                created++;
            }
            for (int i = 0; i < created; ++i) {
                pthread_join(tids[i], NULL);
            }
            if (failed) {
                hetgpu_host_gemm_worker_ctx_t ctx = {
                    Ab, Bb, Cb, Atype, Btype, Ctype, transa, transb,
                    m, n, k, lda, ldb, ldc, a_scale, b_scale, 0, tasks
                };
                hetgpu_host_gemm_compute_range(&ctx);
            }
        }
    }
    if (copy_C_back) {
        if (cudaMemcpy(C, host_C, c_bytes, HETGPU_CUDA_MEMCPY_HOST_TO_DEVICE) != 0) {
            free(tmp_A);
            free(tmp_B);
            free(tmp_C);
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    free(tmp_A);
    free(tmp_B);
    free(tmp_C);
    return CUBLAS_STATUS_SUCCESS;
}

static int g_pacc_gemm_disabled_after_failure = 0;
static int g_pacc_gemm_coarse_stage_disabled_after_failure = 0;
static int g_pacc_gemm_mmvf_route_disabled_after_failure = 0;

static int prefer_pacc_gemm_stage_shared_ddr(void) {
    const char *stage_shared = getenv("HETGPU_PACC_GEMM_STAGE_SHARED_DDR");
    if (!stage_shared) {
        return 1;
    }
    return strcmp(stage_shared, "0") != 0;
}

static int prefer_pacc_gemm_coarse_stage(void) {
    const char *coarse = getenv("HETGPU_PACC_GEMM_COARSE_STAGE");
    return coarse && strcmp(coarse, "force") == 0 && pacc_runtime_marked_ready();
}

static int disable_pacc_gemm_mmvf_after_failure(void) {
    const char *env = getenv("HETGPU_PACC_GEMM_MMVF_DISABLE_AFTER_FAILURE");
    if (env) {
        return strcmp(env, "0") != 0;
    }
    return hetgpu_env_is_one("HETGPU_PACC_GEMM_DISABLE_AFTER_FAILURE");
}


typedef struct {
    const char *name;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    const void *alpha;
    const void *A;
    cudaDataType Atype;
    int lda;
    long long strideA;
    const void *B;
    cudaDataType Btype;
    int ldb;
    long long strideB;
    const void *beta;
    void *C;
    cudaDataType Ctype;
    int ldc;
    long long strideC;
    int batchCount;
    cublasComputeType_t computeType;
    int max_m;
    int max_n;
    int max_k;
    size_t a_size;
    size_t b_size;
    size_t c_size;
    int row_tiles;
    int col_tiles;
    int total_tasks;
    volatile int next_task;
    volatile int rc;
} hetgpu_parallel_gemm_ctx_t;

typedef struct {
    hetgpu_parallel_gemm_ctx_t *ctx;
    int worker_id;
    int dev_id;
    int slot_id;
} hetgpu_parallel_gemm_worker_t;

static void *hetgpu_parallel_gemm_worker(void *arg) {
    hetgpu_parallel_gemm_worker_t *worker = (hetgpu_parallel_gemm_worker_t *)arg;
    hetgpu_parallel_gemm_ctx_t *ctx = worker->ctx;
    float one_beta = 1.0f;
    while (ctx->rc == 0) {
        int task = __sync_fetch_and_add(&ctx->next_task, 1);
        if (task >= ctx->total_tasks) {
            break;
        }
        int batch = task / (ctx->row_tiles * ctx->col_tiles);
        int rem = task % (ctx->row_tiles * ctx->col_tiles);
        int col_tile = rem / ctx->row_tiles;
        int row_tile = rem % ctx->row_tiles;
        int row = row_tile * ctx->max_m;
        int col = col_tile * ctx->max_n;
        int chunk_m = ctx->m - row;
        int chunk_n = ctx->n - col;
        if (chunk_m > ctx->max_m) chunk_m = ctx->max_m;
        if (chunk_n > ctx->max_n) chunk_n = ctx->max_n;

        const char *batch_A = (const char *)ctx->A + (ctx->strideA > 0 ? (size_t)batch * (size_t)ctx->strideA * ctx->a_size : 0);
        const char *batch_B = (const char *)ctx->B + (ctx->strideB > 0 ? (size_t)batch * (size_t)ctx->strideB * ctx->b_size : 0);
        char *batch_C = (char *)ctx->C + (ctx->strideC > 0 ? (size_t)batch * (size_t)ctx->strideC * ctx->c_size : 0);
        char *chunk_C = batch_C + ((size_t)row + (size_t)col * (size_t)ctx->ldc) * ctx->c_size;

        for (int kk = 0; kk < ctx->k && ctx->rc == 0; kk += ctx->max_k) {
            int chunk_k = ctx->k - kk;
            if (chunk_k > ctx->max_k) chunk_k = ctx->max_k;
            const char *chunk_A = batch_A;
            const char *chunk_B = batch_B;
            if (ctx->transa == CUBLAS_OP_N) {
                chunk_A += ((size_t)row + (size_t)kk * (size_t)ctx->lda) * ctx->a_size;
            } else {
                chunk_A += ((size_t)kk + (size_t)row * (size_t)ctx->lda) * ctx->a_size;
            }
            if (ctx->transb == CUBLAS_OP_N) {
                chunk_B += ((size_t)kk + (size_t)col * (size_t)ctx->ldb) * ctx->b_size;
            } else {
                chunk_B += ((size_t)col + (size_t)kk * (size_t)ctx->ldb) * ctx->b_size;
            }
            const void *chunk_beta = (kk == 0) ? ctx->beta : &one_beta;
            if (hetgpu_tail_gemm_host_fallback_enabled(chunk_m, chunk_n, chunk_k, 1)) {
                cublasStatus_t fallback = host_gemm_fallback(
                    ctx->name, ctx->transa, ctx->transb, chunk_m, chunk_n, chunk_k,
                    ctx->alpha,
                    chunk_A, ctx->Atype, ctx->lda, 0,
                    chunk_B, ctx->Btype, ctx->ldb, 0,
                    chunk_beta,
                    chunk_C, ctx->Ctype, ctx->ldc, 0,
                    1, ctx->computeType);
                if (fallback == CUBLAS_STATUS_SUCCESS) {
                    continue;
                }
                if (hetgpu_env_enabled_default("HETGPU_PACC_GEMM_TAIL_HOST_REQUIRED", 1)) {
                    ctx->rc = -2;
                    break;
                }
            }
            int chunk_rc = hetgpu_pacc_submit_gemm_staged_on(
                worker->dev_id, worker->slot_id,
                (int)ctx->transa, (int)ctx->transb, chunk_m, chunk_n, chunk_k,
                ctx->alpha,
                chunk_A, hetgpu_pacc_dtype(ctx->Atype), ctx->lda, 0,
                chunk_B, hetgpu_pacc_dtype(ctx->Btype), ctx->ldb, 0,
                chunk_beta,
                chunk_C, hetgpu_pacc_dtype(ctx->Ctype), ctx->ldc, 0,
                1, (int)ctx->computeType);
            if (chunk_rc != 0) {
                if (hetgpu_allow_host_gemm_fallback()) {
                    cublasStatus_t fallback = host_gemm_fallback(
                        ctx->name, ctx->transa, ctx->transb, chunk_m, chunk_n, chunk_k,
                        ctx->alpha,
                        chunk_A, ctx->Atype, ctx->lda, 0,
                        chunk_B, ctx->Btype, ctx->ldb, 0,
                        chunk_beta,
                        chunk_C, ctx->Ctype, ctx->ldc, 0,
                        1, ctx->computeType);
                    if (fallback == CUBLAS_STATUS_SUCCESS) {
                        continue;
                    }
                }
                ctx->rc = chunk_rc;
                break;
            }
        }
    }
    return NULL;
}

static cublasStatus_t submit_pacc_gemm(
    const char *name,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType Atype, int lda, long long strideA,
    const void *B, cudaDataType Btype, int ldb, long long strideB,
    const void *beta,
    void *C, cudaDataType Ctype, int ldc, long long strideC,
    int batchCount, cublasComputeType_t computeType) {
    if (hetgpu_cublas_noop_return_success(name, m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus_t region_status = hetgpu_validate_gemm_regions(
        name, transa, transb, m, n, k,
        A, Atype, lda,
        B, Btype, ldb,
        C, Ctype, ldc);
    if (region_status != CUBLAS_STATUS_SUCCESS) {
        if (hetgpu_cublas_fail_open_enabled()) {
            DEBUG_LOG("%s region validation returned %d; CUBLAS_FAIL_OPEN treating GEMM as successful",
                      name, (int)region_status);
            return CUBLAS_STATUS_SUCCESS;
        }
        return region_status;
    }
    float alpha_f32 = host_read_scale(alpha, computeType, 1.0f);
    float beta_f32 = host_read_scale(beta, computeType, 0.0f);
    const void *alpha_arg = alpha ? (const void *)&alpha_f32 : NULL;
    const void *beta_arg = beta ? (const void *)&beta_f32 : NULL;

    if (hetgpu_env_is_one("HETGPU_PACC_GEMM_HOST_ONLY")) {
        return host_gemm_fallback(
            name, transa, transb, m, n, k,
            alpha_arg, A, Atype, lda, strideA,
            B, Btype, ldb, strideB,
            beta_arg, C, Ctype, ldc, strideC,
            batchCount, computeType);
    }

    if (hetgpu_non_skinny_n_gemm_host_fallback_enabled(m, n, k, batchCount)) {
        cublasStatus_t fallback = host_gemm_fallback(
            name, transa, transb, m, n, k,
            alpha_arg, A, Atype, lda, strideA,
            B, Btype, ldb, strideB,
            beta_arg, C, Ctype, ldc, strideC,
            batchCount, computeType);
        if (fallback == CUBLAS_STATUS_SUCCESS) {
            DEBUG_LOG("%s using non-skinny-N host GEMM fallback m=%d n=%d k=%d batch=%d",
                      name, m, n, k, batchCount);
            return fallback;
        }
        DEBUG_LOG("%s non-skinny-N host GEMM fallback returned %d; continuing with PACC",
                  name, (int)fallback);
    }

    if (hetgpu_skinny_n_gemm_host_fallback_enabled(m, n, k, batchCount)) {
        cublasStatus_t fallback = host_gemm_fallback(
            name, transa, transb, m, n, k,
            alpha_arg, A, Atype, lda, strideA,
            B, Btype, ldb, strideB,
            beta_arg, C, Ctype, ldc, strideC,
            batchCount, computeType);
        if (fallback == CUBLAS_STATUS_SUCCESS) {
            DEBUG_LOG("%s using skinny-N host GEMM fallback m=%d n=%d k=%d batch=%d",
                      name, m, n, k, batchCount);
            return fallback;
        }
        DEBUG_LOG("%s skinny-N host GEMM fallback returned %d; continuing with PACC",
                  name, (int)fallback);
    }

    if (hetgpu_small_gemm_host_fallback_enabled(m, n, k, batchCount)) {
        cublasStatus_t fallback = host_gemm_fallback(
            name, transa, transb, m, n, k,
            alpha_arg, A, Atype, lda, strideA,
            B, Btype, ldb, strideB,
            beta_arg, C, Ctype, ldc, strideC,
            batchCount, computeType);
        if (fallback == CUBLAS_STATUS_SUCCESS) {
            DEBUG_LOG("%s using small-GEMM host fallback m=%d n=%d k=%d batch=%d",
                      name, m, n, k, batchCount);
            return fallback;
        }
        DEBUG_LOG("%s small-GEMM host fallback returned %d; continuing with PACC",
                  name, (int)fallback);
    }

    if (hetgpu_env_is_one("HETGPU_PACC_GEMM_DISABLE_AFTER_FAILURE") &&
        g_pacc_gemm_disabled_after_failure) {
        if (hetgpu_allow_host_gemm_fallback()) {
            return host_gemm_fallback(
                name, transa, transb, m, n, k,
                alpha_arg, A, Atype, lda, strideA,
                B, Btype, ldb, strideB,
                beta_arg, C, Ctype, ldc, strideC,
                batchCount, computeType);
        }
        DEBUG_LOG("%s PACC GEMM disabled after prior failure; host GEMM fallback is not enabled", name);
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    if (!pacc_runtime_marked_ready()) {
        if (hetgpu_allow_host_gemm_fallback()) {
            DEBUG_LOG("%s skipping PACC GEMM submit because runtime is not marked ready; using host fallback", name);
            cublasStatus_t fallback = host_gemm_fallback(
                name, transa, transb, m, n, k,
                alpha_arg, A, Atype, lda, strideA,
                B, Btype, ldb, strideB,
                beta_arg, C, Ctype, ldc, strideC,
                batchCount, computeType);
            if (fallback == CUBLAS_STATUS_SUCCESS || !hetgpu_cublas_fail_open_enabled()) {
                return fallback;
            }
            DEBUG_LOG("%s runtime not ready and host fallback returned %d; CUBLAS_FAIL_OPEN treating GEMM as successful",
                      name, (int)fallback);
            return CUBLAS_STATUS_SUCCESS;
        }
        if (hetgpu_cublas_fail_open_enabled()) {
            DEBUG_LOG("%s runtime not ready; CUBLAS_FAIL_OPEN treating GEMM as successful", name);
            return CUBLAS_STATUS_SUCCESS;
        }
        DEBUG_LOG("%s requires PACC GEMM runtime readiness; host GEMM fallback is not enabled", name);
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    const void *pacc_A = (const void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(A);
    const void *pacc_B = (const void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(B);
    void *pacc_C = (void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(C);
    const void *pacc_alpha = hetgpu_pacc_is_device_ptr(alpha_arg)
        ? (const void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(alpha_arg)
        : NULL;
    const void *pacc_beta = hetgpu_pacc_is_device_ptr(beta_arg)
        ? (const void *)(uintptr_t)hetgpu_pacc_resolve_device_addr(beta_arg)
        : NULL;

    int rc = -1;
    if (prefer_pacc_gemm_stage_shared_ddr()) {
        const char *max_m_env = getenv("HETGPU_PACC_GEMM_MAX_M");
        const char *max_n_env = getenv("HETGPU_PACC_GEMM_MAX_N");
        const char *max_k_env = getenv("HETGPU_PACC_GEMM_MAX_K");
        const char *large_mv_max_n_env = getenv("HETGPU_PACC_GEMM_LARGE_MV_MAX_N");
        int large_mv_max_n = large_mv_max_n_env ? atoi(large_mv_max_n_env) : 32;
        if (large_mv_max_n < 1) {
            large_mv_max_n = 1;
        }
        int large_mv = (n > 0 && n <= large_mv_max_n);
        int max_m = max_m_env ? atoi(max_m_env) : 64;
        int max_n = max_n_env ? atoi(max_n_env) : (large_mv ? n : 16);
        int max_k = max_k_env ? atoi(max_k_env) : (large_mv ? k : 16);
        if (!max_k_env && large_mv && max_k > 4096) {
            max_k = 4096;
        }
        if (max_m <= 0) {
            max_m = m;
        }
        if (max_n <= 0) {
            max_n = n;
        }
        if (max_k <= 0) {
            max_k = k;
        }
        const char *fw_max_m_env = getenv("HETGPU_PACC_GEMM_FW_MAX_M");
        const char *fw_max_n_env = getenv("HETGPU_PACC_GEMM_FW_MAX_N");
        const char *fw_max_k_env = getenv("HETGPU_PACC_GEMM_FW_MAX_K");
        int fw_max_m = fw_max_m_env ? atoi(fw_max_m_env) : 0;
        int fw_max_n = fw_max_n_env ? atoi(fw_max_n_env) : 0;
        int fw_max_k = fw_max_k_env ? atoi(fw_max_k_env) : 0;
        if (fw_max_m > 0 && max_m > fw_max_m) max_m = fw_max_m;
        if (fw_max_n > 0 && max_n > fw_max_n) max_n = fw_max_n;
        if (fw_max_k > 0 && max_k > fw_max_k) max_k = fw_max_k;
        DEBUG_LOG("%s using PACC shared-DDR staged submit trans=%d/%d m=%d n=%d k=%d tile=%dx%dx%d lda=%d ldb=%d ldc=%d batch=%d stride=%lld/%lld/%lld",
                  name, (int)transa, (int)transb, m, n, k, max_m, max_n, max_k,
                  lda, ldb, ldc, batchCount, strideA, strideB, strideC);
        rc = 0;
        size_t a_size = host_dtype_size(Atype);
        size_t b_size = host_dtype_size(Btype);
        size_t c_size = host_dtype_size(Ctype);
        int parallel_workers = 4;
        int pacc_devices[4] = {0, 1, 2, 3};
        int pacc_device_count = hetgpu_parse_pacc_gemm_devices(pacc_devices);
        const int layer_local = hetgpu_pacc_gemm_layer_local_enabled();
        const int layer_local_dev = hetgpu_current_pacc_device_for_cublas();
        if (layer_local) {
            pacc_devices[0] = layer_local_dev;
            pacc_device_count = 1;
        }
        const char *parallel_env = getenv("HETGPU_PACC_GEMM_PARALLEL");
        if (layer_local || (parallel_env && strcmp(parallel_env, "0") == 0)) {
            parallel_workers = 1;
        } else {
            const char *workers_env = getenv("HETGPU_PACC_GEMM_WORKERS");
            parallel_workers = workers_env ? atoi(workers_env) : pacc_device_count;
            if (parallel_workers < 1) parallel_workers = 1;
            if (parallel_workers > 4) parallel_workers = 4;
        }
        if (parallel_workers > pacc_device_count) {
            parallel_workers = pacc_device_count;
        }
        if (!layer_local && !g_pacc_gemm_mmvf_route_disabled_after_failure) {
            int mmvf_rc = hetgpu_pacc_submit_gemm_mmvf_small_n(
                (int)transa, (int)transb, m, n, k,
                alpha_arg,
                pacc_A, hetgpu_pacc_dtype(Atype), lda, strideA,
                pacc_B, hetgpu_pacc_dtype(Btype), ldb, strideB,
                beta_arg,
                pacc_C, hetgpu_pacc_dtype(Ctype), ldc, strideC,
                batchCount, (int)computeType);
            if (mmvf_rc == 0) {
                return CUBLAS_STATUS_SUCCESS;
            }
            if (mmvf_rc != 0 && disable_pacc_gemm_mmvf_after_failure()) {
                g_pacc_gemm_mmvf_route_disabled_after_failure = 1;
                DEBUG_LOG("%s disabling PACC MMVF small-N route after failure rc=%d", name, mmvf_rc);
            }
            if (mmvf_rc < 0 && hetgpu_env_enabled_default("HETGPU_PACC_GEMM_MMVF_ROUTE_STRICT", 0)) {
                DEBUG_LOG("%s PACC MMVF small-N route failed rc=%d in strict mode", name, mmvf_rc);
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
        }
        if (!layer_local &&
            !g_pacc_gemm_coarse_stage_disabled_after_failure &&
            prefer_pacc_gemm_coarse_stage()) {
            rc = hetgpu_pacc_submit_gemm_staged_tiled(
                (int)transa, (int)transb, m, n, k,
                alpha_arg,
                pacc_A, hetgpu_pacc_dtype(Atype), lda, strideA,
                pacc_B, hetgpu_pacc_dtype(Btype), ldb, strideB,
                beta_arg,
                pacc_C, hetgpu_pacc_dtype(Ctype), ldc, strideC,
                batchCount, (int)computeType,
                max_m, max_n, max_k);
            if (rc == 0) {
                return CUBLAS_STATUS_SUCCESS;
            }
            g_pacc_gemm_coarse_stage_disabled_after_failure = 1;
            DEBUG_LOG("%s disabling coarse staged GEMM after failure rc=%d", name, rc);
        }
        int row_tiles = (m + max_m - 1) / max_m;
        int col_tiles = (n + max_n - 1) / max_n;
        if (parallel_workers > 1 && row_tiles * col_tiles * batchCount > 1) {
            hetgpu_parallel_gemm_ctx_t ctx = {
                name, transa, transb, m, n, k,
                alpha_arg, pacc_A, Atype, lda, strideA,
                pacc_B, Btype, ldb, strideB,
                beta_arg, pacc_C, Ctype, ldc, strideC,
                batchCount, computeType,
                max_m, max_n, max_k,
                a_size, b_size, c_size,
                row_tiles, col_tiles, row_tiles * col_tiles * batchCount,
                0, 0
            };
            pthread_t threads[4];
            hetgpu_parallel_gemm_worker_t workers[4];
            for (int i = 0; i < parallel_workers; ++i) {
                workers[i].ctx = &ctx;
                workers[i].worker_id = i;
                workers[i].dev_id = pacc_devices[i];
                workers[i].slot_id = i;
                int trc = pthread_create(&threads[i], NULL, hetgpu_parallel_gemm_worker, &workers[i]);
                if (trc != 0) {
                    ctx.rc = -1;
                    parallel_workers = i;
                    break;
                }
            }
            for (int i = 0; i < parallel_workers; ++i) {
                pthread_join(threads[i], NULL);
            }
            rc = ctx.rc;
        } else {
            float one_beta = 1.0f;
            for (int batch = 0; batch < batchCount && rc == 0; ++batch) {
                const char *batch_A = (const char *)pacc_A + (strideA > 0 ? (size_t)batch * (size_t)strideA * a_size : 0);
                const char *batch_B = (const char *)pacc_B + (strideB > 0 ? (size_t)batch * (size_t)strideB * b_size : 0);
                char *batch_C = (char *)pacc_C + (strideC > 0 ? (size_t)batch * (size_t)strideC * c_size : 0);
                for (int col = 0; col < n && rc == 0; col += max_n) {
                    int chunk_n = n - col;
                    if (chunk_n > max_n) chunk_n = max_n;
                    for (int row = 0; row < m && rc == 0; row += max_m) {
                        int chunk_m = m - row;
                        if (chunk_m > max_m) chunk_m = max_m;
                        char *chunk_C = batch_C + ((size_t)row + (size_t)col * (size_t)ldc) * c_size;
                        for (int kk = 0; kk < k; kk += max_k) {
                            int chunk_k = k - kk;
                            if (chunk_k > max_k) chunk_k = max_k;
                            const char *chunk_A = batch_A;
                            const char *chunk_B = batch_B;
                            if (transa == CUBLAS_OP_N) {
                                chunk_A += ((size_t)row + (size_t)kk * (size_t)lda) * a_size;
                            } else {
                                chunk_A += ((size_t)kk + (size_t)row * (size_t)lda) * a_size;
                            }
                            if (transb == CUBLAS_OP_N) {
                                chunk_B += ((size_t)kk + (size_t)col * (size_t)ldb) * b_size;
                            } else {
                                chunk_B += ((size_t)col + (size_t)kk * (size_t)ldb) * b_size;
                            }
                            const void *chunk_beta = (kk == 0) ? beta_arg : &one_beta;
                            if (hetgpu_tail_gemm_host_fallback_enabled(chunk_m, chunk_n, chunk_k, 1)) {
                                cublasStatus_t fallback = host_gemm_fallback(
                                    name, transa, transb, chunk_m, chunk_n, chunk_k,
                                    alpha_arg,
                                    chunk_A, Atype, lda, 0,
                                    chunk_B, Btype, ldb, 0,
                                    chunk_beta,
                                    chunk_C, Ctype, ldc, 0,
                                    1, computeType);
                                if (fallback == CUBLAS_STATUS_SUCCESS) {
                                    continue;
                                }
                                if (hetgpu_env_enabled_default("HETGPU_PACC_GEMM_TAIL_HOST_REQUIRED", 1)) {
                                    rc = -2;
                                    break;
                                }
                            }
                            int chunk_rc = layer_local
                                ? hetgpu_pacc_submit_gemm_staged_on(
                                    layer_local_dev, 0,
                                    (int)transa, (int)transb, chunk_m, chunk_n, chunk_k,
                                    alpha_arg,
                                    chunk_A, hetgpu_pacc_dtype(Atype), lda, 0,
                                    chunk_B, hetgpu_pacc_dtype(Btype), ldb, 0,
                                    chunk_beta,
                                    chunk_C, hetgpu_pacc_dtype(Ctype), ldc, 0,
                                    1, (int)computeType)
                                : hetgpu_pacc_submit_gemm_staged(
                                    (int)transa, (int)transb, chunk_m, chunk_n, chunk_k,
                                    alpha_arg,
                                    chunk_A, hetgpu_pacc_dtype(Atype), lda, 0,
                                    chunk_B, hetgpu_pacc_dtype(Btype), ldb, 0,
                                    chunk_beta,
                                    chunk_C, hetgpu_pacc_dtype(Ctype), ldc, 0,
                                    1, (int)computeType);
                            if (chunk_rc != 0) {
                                if (hetgpu_allow_host_gemm_fallback()) {
                                    cublasStatus_t fallback = host_gemm_fallback(
                                        name, transa, transb, chunk_m, chunk_n, chunk_k,
                                        alpha_arg,
                                        chunk_A, Atype, lda, 0,
                                        chunk_B, Btype, ldb, 0,
                                        chunk_beta,
                                        chunk_C, Ctype, ldc, 0,
                                        1, computeType);
                                    if (fallback == CUBLAS_STATUS_SUCCESS) {
                                        continue;
                                    }
                                }
                                rc = chunk_rc;
                                break;
                            }
                        }
                    }
                }
            }
        }
    } else {
        rc = hetgpu_pacc_submit_gemm(
        (int)transa, (int)transb, m, n, k,
        pacc_alpha,
        pacc_A, hetgpu_pacc_dtype(Atype), lda, strideA,
        pacc_B, hetgpu_pacc_dtype(Btype), ldb, strideB,
        pacc_beta,
        pacc_C, hetgpu_pacc_dtype(Ctype), ldc, strideC,
        batchCount, (int)computeType);
    }
    if (rc == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (hetgpu_cublas_fail_open_enabled()) {
        DEBUG_LOG("%s PACC GEMM rc=%d; CUBLAS_FAIL_OPEN treating GEMM as successful without host fallback",
                  name, rc);
        return CUBLAS_STATUS_SUCCESS;
    }
    if (hetgpu_env_is_one("HETGPU_PACC_GEMM_DISABLE_AFTER_FAILURE")) {
        g_pacc_gemm_disabled_after_failure = 1;
    }
    if (hetgpu_allow_host_gemm_fallback()) {
        cublasStatus_t fallback = host_gemm_fallback(
            name, transa, transb, m, n, k,
            alpha_arg, A, Atype, lda, strideA,
            B, Btype, ldb, strideB,
            beta_arg, C, Ctype, ldc, strideC,
            batchCount, computeType);
        if (fallback == CUBLAS_STATUS_SUCCESS) {
            return fallback;
        }
    }
    DEBUG_LOG("%s requires PACC GEMM runtime submit; no CPU fallback active/supported", name);
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// cuBLAS handle management
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    DEBUG_LOG("cublasCreate_v2 called");
    typedef cublasStatus_t (*func_type)(cublasHandle_t*);
    GET_REAL_FUNC(cublasCreate_v2, func_type);
    if (real_cublasCreate_v2) {
        return real_cublasCreate_v2(handle);
    }
    // Fallback
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = (void*)0x1;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    DEBUG_LOG("cublasDestroy_v2 called");
    typedef cublasStatus_t (*func_type)(cublasHandle_t);
    GET_REAL_FUNC(cublasDestroy_v2, func_type);
    if (real_cublasDestroy_v2) {
        return real_cublasDestroy_v2(handle);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void* streamId) {
    DEBUG_LOG("cublasSetStream_v2 called");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void** streamId) {
    DEBUG_LOG("cublasGetStream_v2 called");
    if (streamId) *streamId = NULL;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
    DEBUG_LOG("cublasSetPointerMode_v2 called");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
    DEBUG_LOG("cublasGetPointerMode_v2 called");
    if (mode) *mode = CUBLAS_POINTER_MODE_HOST;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    DEBUG_LOG("cublasSetAtomicsMode called");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
    DEBUG_LOG("cublasGetAtomicsMode called");
    if (mode) *mode = CUBLAS_ATOMICS_NOT_ALLOWED;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    DEBUG_LOG("cublasSetMathMode called: mode=%d", mode);
    // Accept all math modes - PyTorch sets TF32/tensor op modes
    g_math_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    DEBUG_LOG("cublasGetMathMode called");
    if (mode) *mode = g_math_mode;
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 1: Vector operations (stubs - no actual computation)
cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n,
                              const float *x, int incx,
                              const float *y, int incy,
                              float *result) {
    DEBUG_LOG("cublasSdot_v2 called: n=%d", n);
    if (result) *result = 0.0f;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n,
                              const double *x, int incx,
                              const double *y, int incy,
                              double *result) {
    DEBUG_LOG("cublasDdot_v2 called: n=%d", n);
    if (result) *result = 0.0;
    return CUBLAS_STATUS_SUCCESS;
}

static cublasStatus_t submit_pacc_gemv_as_gemm(
    const char *name,
    cublasOperation_t trans,
    int m, int n,
    const void *alpha,
    const void *A, int lda,
    const void *x, int incx,
    const void *beta,
    void *y, int incy,
    cudaDataType dtype,
    cublasComputeType_t computeType) {
    if (!hetgpu_env_enabled_default("HETGPU_PACC_FUSE_GEMV_TO_GEMM", 1)) {
        DEBUG_LOG("%s GEMV->GEMM fuse disabled by HETGPU_PACC_FUSE_GEMV_TO_GEMM=0", name);
        return hetgpu_cublas_fail_open_enabled() ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T && trans != CUBLAS_OP_C) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || lda < (m > 1 ? m : 1) || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (incx != 1 || incy != 1) {
        DEBUG_LOG("%s GEMV->GEMM fuse supports only contiguous vectors for now incx=%d incy=%d",
                  name, incx, incy);
        return hetgpu_cublas_fail_open_enabled() ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_SUPPORTED;
    }

    int gemm_m = (trans == CUBLAS_OP_N) ? m : n;
    int gemm_k = (trans == CUBLAS_OP_N) ? n : m;
    int ldb = gemm_k > 0 ? gemm_k : 1;
    int ldc = gemm_m > 0 ? gemm_m : 1;
    DEBUG_LOG("%s fused through GEMM transa=%d transb=N m=%d n=1 k=%d lda=%d ldb=%d ldc=%d",
              name, (int)trans, gemm_m, gemm_k, lda, ldb, ldc);
    return submit_pacc_gemm(name, trans, CUBLAS_OP_N, gemm_m, 1, gemm_k,
                            alpha, A, dtype, lda, 0,
                            x, dtype, ldb, 0,
                            beta, y, dtype, ldc, 0,
                            1, computeType);
}

// BLAS Level 2: Matrix-vector operations
cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const float *alpha,
                               const float *A, int lda,
                               const float *x, int incx,
                               const float *beta,
                               float *y, int incy) {
    DEBUG_LOG("cublasSgemv_v2 called: m=%d, n=%d trans=%d incx=%d incy=%d",
              m, n, (int)trans, incx, incy);
    return submit_pacc_gemv_as_gemm("cublasSgemv_v2", trans, m, n,
                                    alpha, A, lda, x, incx, beta, y, incy,
                                    CUDA_R_32F, CUBLAS_COMPUTE_32F);
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const double *alpha,
                               const double *A, int lda,
                               const double *x, int incx,
                               const double *beta,
                               double *y, int incy) {
    DEBUG_LOG("cublasDgemv_v2 called: m=%d, n=%d trans=%d incx=%d incy=%d",
              m, n, (int)trans, incx, incy);
    return submit_pacc_gemv_as_gemm("cublasDgemv_v2", trans, m, n,
                                    alpha, A, lda, x, incx, beta, y, incy,
                                    CUDA_R_64F, CUBLAS_COMPUTE_64F);
}

// BLAS Level 3: Matrix-matrix operations (most important for ML)
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const float *alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               const float *beta,
                               float *C, int ldc) {
    DEBUG_LOG("cublasSgemm_v2 called: m=%d, n=%d, k=%d", m, n, k);
    if (hetgpu_cublas_noop_return_success("cublasSgemm_v2", m, n, k, 1)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    typedef cublasStatus_t (*func_type)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                        int, int, int, const float*, const float*, int,
                                        const float*, int, const float*, float*, int);
    GET_REAL_FUNC(cublasSgemm_v2, func_type);
    if (real_cublasSgemm_v2) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    return submit_pacc_gemm("cublasSgemm_v2", transa, transb, m, n, k,
                            alpha, A, CUDA_R_32F, lda, 0,
                            B, CUDA_R_32F, ldb, 0,
                            beta, C, CUDA_R_32F, ldc, 0,
                            1, CUBLAS_COMPUTE_32F);
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const double *alpha,
                               const double *A, int lda,
                               const double *B, int ldb,
                               const double *beta,
                               double *C, int ldc) {
    DEBUG_LOG("cublasDgemm_v2 called: m=%d, n=%d, k=%d", m, n, k);
    if (hetgpu_cublas_noop_return_success("cublasDgemm_v2", m, n, k, 1)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    return submit_pacc_gemm("cublasDgemm_v2", transa, transb, m, n, k,
                            alpha, A, CUDA_R_64F, lda, 0,
                            B, CUDA_R_64F, ldb, 0,
                            beta, C, CUDA_R_64F, ldc, 0,
                            1, CUBLAS_COMPUTE_64F);
}

cublasStatus_t cublasHgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void *alpha,
                            const void *A, int lda,
                            const void *B, int ldb,
                            const void *beta,
                            void *C, int ldc) {
    DEBUG_LOG("cublasHgemm called: m=%d, n=%d, k=%d", m, n, k);
    if (hetgpu_cublas_noop_return_success("cublasHgemm", m, n, k, 1)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    return submit_pacc_gemm("cublasHgemm", transa, transb, m, n, k,
                            alpha, A, CUDA_R_16F, lda, 0,
                            B, CUDA_R_16F, ldb, 0,
                            beta, C, CUDA_R_16F, ldc, 0,
                            1, CUBLAS_COMPUTE_32F);
}

// Batched GEMM
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                          cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k,
                                          const float *alpha,
                                          const float *A, int lda, long long int strideA,
                                          const float *B, int ldb, long long int strideB,
                                          const float *beta,
                                          float *C, int ldc, long long int strideC,
                                          int batchCount) {
    DEBUG_LOG("cublasSgemmStridedBatched called: m=%d, n=%d, k=%d, batchCount=%d", m, n, k, batchCount);
    if (hetgpu_cublas_noop_return_success("cublasSgemmStridedBatched", m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    return submit_pacc_gemm("cublasSgemmStridedBatched", transa, transb, m, n, k,
                            alpha, A, CUDA_R_32F, lda, strideA,
                            B, CUDA_R_32F, ldb, strideB,
                            beta, C, CUDA_R_32F, ldc, strideC,
                            batchCount, CUBLAS_COMPUTE_32F);
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
                                          cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k,
                                          const double *alpha,
                                          const double *A, int lda, long long int strideA,
                                          const double *B, int ldb, long long int strideB,
                                          const double *beta,
                                          double *C, int ldc, long long int strideC,
                                          int batchCount) {
    DEBUG_LOG("cublasDgemmStridedBatched called: m=%d, n=%d, k=%d, batchCount=%d", m, n, k, batchCount);
    if (hetgpu_cublas_noop_return_success("cublasDgemmStridedBatched", m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    return submit_pacc_gemm("cublasDgemmStridedBatched", transa, transb, m, n, k,
                            alpha, A, CUDA_R_64F, lda, strideA,
                            B, CUDA_R_64F, ldb, strideB,
                            beta, C, CUDA_R_64F, ldc, strideC,
                            batchCount, CUBLAS_COMPUTE_64F);
}

// GemmEx - Extended GEMM with mixed precision
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                             cublasOperation_t transa, cublasOperation_t transb,
                             int m, int n, int k,
                             const void *alpha,
                             const void *A, cudaDataType Atype, int lda,
                             const void *B, cudaDataType Btype, int ldb,
                             const void *beta,
                             void *C, cudaDataType Ctype, int ldc,
                             cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    DEBUG_LOG("cublasGemmEx called: m=%d, n=%d, k=%d, Atype=%d, Btype=%d, Ctype=%d, ldc=%d", m, n, k, Atype, Btype, Ctype, ldc);
    (void)algo;
    if (hetgpu_cublas_noop_return_success("cublasGemmEx", m, n, k, 1)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    return submit_pacc_gemm("cublasGemmEx", transa, transb, m, n, k,
                            alpha, A, Atype, lda, 0,
                            B, Btype, ldb, 0,
                            beta, C, Ctype, ldc, 0,
                            1, computeType);
}

// Complex number versions (stubs)
cublasStatus_t cublasCgemm_v2(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void *alpha,
                               const void *A, int lda,
                               const void *B, int ldb,
                               const void *beta,
                               void *C, int ldc) {
    DEBUG_LOG("cublasCgemm_v2 called: m=%d, n=%d, k=%d", m, n, k);
    if (hetgpu_cublas_noop_return_success("cublasCgemm_v2", m, n, k, 1)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    DEBUG_LOG("cublasCgemm_v2 complex GEMM is not implemented on PACC yet; no stub fallback active");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void *alpha,
                               const void *A, int lda,
                               const void *B, int ldb,
                               const void *beta,
                               void *C, int ldc) {
    DEBUG_LOG("cublasZgemm_v2 called: m=%d, n=%d, k=%d", m, n, k);
    if (hetgpu_cublas_noop_return_success("cublasZgemm_v2", m, n, k, 1)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    DEBUG_LOG("cublasZgemm_v2 complex GEMM is not implemented on PACC yet; no stub fallback active");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Batched complex
cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle,
                                          cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k,
                                          const void *alpha,
                                          const void *A, int lda, long long int strideA,
                                          const void *B, int ldb, long long int strideB,
                                          const void *beta,
                                          void *C, int ldc, long long int strideC,
                                          int batchCount) {
    DEBUG_LOG("cublasCgemmStridedBatched called: m=%d, n=%d, k=%d, batchCount=%d", m, n, k, batchCount);
    if (hetgpu_cublas_noop_return_success("cublasCgemmStridedBatched", m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    DEBUG_LOG("cublasCgemmStridedBatched complex GEMM is not implemented on PACC yet; no stub fallback active");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Dot products for complex
cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n,
                               const void *x, int incx,
                               const void *y, int incy,
                               void *result) {
    DEBUG_LOG("cublasCdotc_v2 called: n=%d", n);
    if (result) {
        float *res = (float*)result;
        res[0] = 0.0f;
        res[1] = 0.0f;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n,
                               const void *x, int incx,
                               const void *y, int incy,
                               void *result) {
    DEBUG_LOG("cublasCdotu_v2 called: n=%d", n);
    if (result) {
        float *res = (float*)result;
        res[0] = 0.0f;
        res[1] = 0.0f;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n,
                               const void *x, int incx,
                               const void *y, int incy,
                               void *result) {
    DEBUG_LOG("cublasZdotc_v2 called: n=%d", n);
    if (result) {
        double *res = (double*)result;
        res[0] = 0.0;
        res[1] = 0.0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n,
                               const void *x, int incx,
                               const void *y, int incy,
                               void *result) {
    DEBUG_LOG("cublasZdotu_v2 called: n=%d", n);
    if (result) {
        double *res = (double*)result;
        res[0] = 0.0;
        res[1] = 0.0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Batched solve operations
cublasStatus_t cublasCgelsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int m, int n, int nrhs,
                                   void **Aarray, int lda,
                                   void **Carray, int ldc,
                                   int *info,
                                   int *devInfoArray,
                                   int batchSize) {
    DEBUG_LOG("cublasCgelsBatched called: m=%d, n=%d, nrhs=%d, batchSize=%d", m, n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

// Version query
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    DEBUG_LOG("cublasGetVersion_v2 called");
    if (version) *version = 12000; // Report cuBLAS 12.0
    return CUBLAS_STATUS_SUCCESS;
}

const char* cublasGetStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

const char* cublasGetStatusString(cublasStatus_t status) {
    return cublasGetStatusName(status);
}

// Additional batched operations that PyTorch needs
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
                                   cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k,
                                   const float *alpha,
                                   const float *Aarray[], int lda,
                                   const float *Barray[], int ldb,
                                   const float *beta,
                                   float *Carray[], int ldc,
                                   int batchCount) {
    DEBUG_LOG("cublasSgemmBatched called: m=%d, n=%d, k=%d, batchCount=%d", m, n, k, batchCount);
    if (hetgpu_cublas_noop_return_success("cublasSgemmBatched", m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (hetgpu_cublas_fail_open_enabled()) {
        DEBUG_LOG("cublasSgemmBatched CUBLAS_FAIL_OPEN active; treating pointer-array GEMM as successful");
        return CUBLAS_STATUS_SUCCESS;
    }
    DEBUG_LOG("cublasSgemmBatched pointer-array GEMM is not implemented on PACC yet; no stub fallback active");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
                                   cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k,
                                   const double *alpha,
                                   const double *Aarray[], int lda,
                                   const double *Barray[], int ldb,
                                   const double *beta,
                                   double *Carray[], int ldc,
                                   int batchCount) {
    DEBUG_LOG("cublasDgemmBatched called: m=%d, n=%d, k=%d, batchCount=%d", m, n, k, batchCount);
    if (hetgpu_cublas_noop_return_success("cublasDgemmBatched", m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (hetgpu_cublas_fail_open_enabled()) {
        DEBUG_LOG("cublasDgemmBatched CUBLAS_FAIL_OPEN active; treating pointer-array GEMM as successful");
        return CUBLAS_STATUS_SUCCESS;
    }
    DEBUG_LOG("cublasDgemmBatched pointer-array GEMM is not implemented on PACC yet; no stub fallback active");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// TRSM operations
cublasStatus_t cublasStrsm_v2(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const float *alpha,
                               const float *A, int lda,
                               float *B, int ldb) {
    DEBUG_LOG("cublasStrsm_v2 called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const double *alpha,
                               const double *A, int lda,
                               double *B, int ldb) {
    DEBUG_LOG("cublasDtrsm_v2 called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle,
                                   cublasSideMode_t side, cublasFillMode_t uplo,
                                   cublasOperation_t trans, cublasDiagType_t diag,
                                   int m, int n,
                                   const float *alpha,
                                   const float *A[], int lda,
                                   float *B[], int ldb,
                                   int batchCount) {
    DEBUG_LOG("cublasStrsmBatched called: m=%d, n=%d, batchCount=%d", m, n, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle,
                                   cublasSideMode_t side, cublasFillMode_t uplo,
                                   cublasOperation_t trans, cublasDiagType_t diag,
                                   int m, int n,
                                   const double *alpha,
                                   const double *A[], int lda,
                                   double *B[], int ldb,
                                   int batchCount) {
    DEBUG_LOG("cublasDtrsmBatched called: m=%d, n=%d, batchCount=%d", m, n, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// GEAM operations  
cublasStatus_t cublasSgeam(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n,
                            const float *alpha,
                            const float *A, int lda,
                            const float *beta,
                            const float *B, int ldb,
                            float *C, int ldc) {
    DEBUG_LOG("cublasSgeam called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeam(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n,
                            const double *alpha,
                            const double *A, int lda,
                            const double *beta,
                            const double *B, int ldb,
                            double *C, int ldc) {
    DEBUG_LOG("cublasDgeam called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

// Additional GEMV for complex
cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const void *alpha,
                               const void *A, int lda,
                               const void *x, int incx,
                               const void *beta,
                               void *y, int incy) {
    DEBUG_LOG("cublasCgemv_v2 called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const void *alpha,
                               const void *A, int lda,
                               const void *x, int incx,
                               const void *beta,
                               void *y, int incy) {
    DEBUG_LOG("cublasZgemv_v2 called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

// Workspace
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
    DEBUG_LOG("cublasSetWorkspace_v2 called: size=%zu", workspaceSizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}

// SgemmEx
cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n, int k,
                              const float *alpha,
                              const void *A, cudaDataType Atype, int lda,
                              const void *B, cudaDataType Btype, int ldb,
                              const float *beta,
                              void *C, cudaDataType Ctype, int ldc) {
    DEBUG_LOG("cublasSgemmEx called: m=%d, n=%d, k=%d", m, n, k);
    if (hetgpu_cublas_noop_return_success("cublasSgemmEx", m, n, k, 1)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    return submit_pacc_gemm("cublasSgemmEx", transa, transb, m, n, k,
                            alpha, A, Atype, lda, 0,
                            B, Btype, ldb, 0,
                            beta, C, Ctype, ldc, 0,
                            1, CUBLAS_COMPUTE_32F);
}

// DotEx
cublasStatus_t cublasDotEx(cublasHandle_t handle,
                            int n,
                            const void *x, cudaDataType xType, int incx,
                            const void *y, cudaDataType yType, int incy,
                            void *result, cudaDataType resultType,
                            cudaDataType executionType) {
    DEBUG_LOG("cublasDotEx called: n=%d", n);
    return CUBLAS_STATUS_SUCCESS;
}

// GemmStridedBatchedEx
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                           cublasOperation_t transa, cublasOperation_t transb,
                                           int m, int n, int k,
                                           const void *alpha,
                                           const void *A, cudaDataType Atype, int lda, long long int strideA,
                                           const void *B, cudaDataType Btype, int ldb, long long int strideB,
                                           const void *beta,
                                           void *C, cudaDataType Ctype, int ldc, long long int strideC,
                                           int batchCount,
                                           cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    DEBUG_LOG("cublasGemmStridedBatchedEx called: m=%d, n=%d, k=%d, batchCount=%d, Atype=%d, Btype=%d, Ctype=%d", m, n, k, batchCount, Atype, Btype, Ctype);
    (void)algo;
    if (hetgpu_cublas_noop_return_success("cublasGemmStridedBatchedEx", m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    return submit_pacc_gemm("cublasGemmStridedBatchedEx", transa, transb, m, n, k,
                            alpha, A, Atype, lda, strideA,
                            B, Btype, ldb, strideB,
                            beta, C, Ctype, ldc, strideC,
                            batchCount, computeType);
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                    cublasOperation_t transa, cublasOperation_t transb,
                                    int m, int n, int k,
                                    const void *alpha,
                                    const void *const Aarray[], cudaDataType Atype, int lda,
                                    const void *const Barray[], cudaDataType Btype, int ldb,
                                    const void *beta,
                                    void *const Carray[], cudaDataType Ctype, int ldc,
                                    int batchCount,
                                    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    (void)handle;
    (void)algo;
    DEBUG_LOG("cublasGemmBatchedEx called: m=%d, n=%d, k=%d, batchCount=%d, Atype=%d, Btype=%d, Ctype=%d, lda=%d, ldb=%d, ldc=%d",
              m, n, k, batchCount, Atype, Btype, Ctype, lda, ldb, ldc);
    if (hetgpu_cublas_noop_return_success("cublasGemmBatchedEx", m, n, k, batchCount)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (!Aarray || !Barray || !Carray || batchCount < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batchCount == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (hetgpu_cublas_fail_open_enabled()) {
        DEBUG_LOG("cublasGemmBatchedEx CUBLAS_FAIL_OPEN active; treating pointer-array GEMM as successful without reading pointer tables");
        return CUBLAS_STATUS_SUCCESS;
    }
    const void *const *host_Aarray = Aarray;
    const void *const *host_Barray = Barray;
    void *const *host_Carray = Carray;
    void **tmp_Aarray = NULL;
    void **tmp_Barray = NULL;
    void **tmp_Carray = NULL;
    size_t ptr_bytes = (size_t)batchCount * sizeof(void *);

    if (hetgpu_pacc_is_device_ptr(Aarray)) {
        tmp_Aarray = (void **)malloc(ptr_bytes);
        if (!tmp_Aarray || hetgpu_copy_pointer_array_to_host(tmp_Aarray, Aarray, ptr_bytes) != 0) {
            fprintf(stderr,
                    "[hetGPU cublas_shim] cublasGemmBatchedEx failed to copy A pointer array ptr=%p bytes=%zu\n",
                    (const void *)Aarray, ptr_bytes);
            free(tmp_Aarray);
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        host_Aarray = (const void *const *)tmp_Aarray;
    }
    if (hetgpu_pacc_is_device_ptr(Barray)) {
        tmp_Barray = (void **)malloc(ptr_bytes);
        if (!tmp_Barray || hetgpu_copy_pointer_array_to_host(tmp_Barray, Barray, ptr_bytes) != 0) {
            fprintf(stderr,
                    "[hetGPU cublas_shim] cublasGemmBatchedEx failed to copy B pointer array ptr=%p bytes=%zu\n",
                    (const void *)Barray, ptr_bytes);
            free(tmp_Aarray);
            free(tmp_Barray);
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        host_Barray = (const void *const *)tmp_Barray;
    }
    if (hetgpu_pacc_is_device_ptr(Carray)) {
        tmp_Carray = (void **)malloc(ptr_bytes);
        if (!tmp_Carray || hetgpu_copy_pointer_array_to_host(tmp_Carray, Carray, ptr_bytes) != 0) {
            fprintf(stderr,
                    "[hetGPU cublas_shim] cublasGemmBatchedEx failed to copy C pointer array ptr=%p bytes=%zu\n",
                    (const void *)Carray, ptr_bytes);
            free(tmp_Aarray);
            free(tmp_Barray);
            free(tmp_Carray);
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        host_Carray = (void *const *)tmp_Carray;
    }
    DEBUG_LOG("cublasGemmBatchedEx pointer-array GEMM falls back to per-batch submit");
    for (int i = 0; i < batchCount; ++i) {
        if (!host_Aarray[i] || !host_Barray[i] || !host_Carray[i]) {
            fprintf(stderr,
                    "[hetGPU cublas_shim] cublasGemmBatchedEx invalid pointer table entry batch=%d/%d A=%p B=%p C=%p arrays=%p/%p/%p dims m=%d n=%d k=%d lda=%d ldb=%d ldc=%d types=%d/%d/%d\n",
                    i, batchCount,
                    host_Aarray[i], host_Barray[i], host_Carray[i],
                    (const void *)Aarray, (const void *)Barray, (const void *)Carray,
                    m, n, k, lda, ldb, ldc, Atype, Btype, Ctype);
            free(tmp_Aarray);
            free(tmp_Barray);
            free(tmp_Carray);
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        cublasStatus_t st = submit_pacc_gemm("cublasGemmBatchedEx", transa, transb, m, n, k,
                                             alpha,
                                             host_Aarray[i], Atype, lda, 0,
                                             host_Barray[i], Btype, ldb, 0,
                                             beta,
                                             host_Carray[i], Ctype, ldc, 0,
                                             1, computeType);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[hetGPU cublas_shim] cublasGemmBatchedEx per-batch submit failed batch=%d/%d status=%d A=%p B=%p C=%p dims m=%d n=%d k=%d lda=%d ldb=%d ldc=%d types=%d/%d/%d\n",
                    i, batchCount, (int)st,
                    host_Aarray[i], host_Barray[i], host_Carray[i],
                    m, n, k, lda, ldb, ldc, Atype, Btype, Ctype);
            free(tmp_Aarray);
            free(tmp_Barray);
            free(tmp_Carray);
            return st;
        }
    }
    free(tmp_Aarray);
    free(tmp_Barray);
    free(tmp_Carray);
    return CUBLAS_STATUS_SUCCESS;
}

// QR factorization batched
cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle,
                                    int m, int n,
                                    float *Aarray[], int lda,
                                    float *TauArray[],
                                    int *info, int batchSize) {
    DEBUG_LOG("cublasSgeqrfBatched called: m=%d, n=%d, batchSize=%d", m, n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle,
                                    int m, int n,
                                    double *Aarray[], int lda,
                                    double *TauArray[],
                                    int *info, int batchSize) {
    DEBUG_LOG("cublasDgeqrfBatched called: m=%d, n=%d, batchSize=%d", m, n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle,
                                    int m, int n,
                                    void *Aarray[], int lda,
                                    void *TauArray[],
                                    int *info, int batchSize) {
    DEBUG_LOG("cublasCgeqrfBatched called: m=%d, n=%d, batchSize=%d", m, n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle,
                                    int m, int n,
                                    void *Aarray[], int lda,
                                    void *TauArray[],
                                    int *info, int batchSize) {
    DEBUG_LOG("cublasZgeqrfBatched called: m=%d, n=%d, batchSize=%d", m, n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

// LU factorization batched
cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
                                    int n,
                                    float *Aarray[], int lda,
                                    int *PivotArray,
                                    int *infoArray, int batchSize) {
    DEBUG_LOG("cublasSgetrfBatched called: n=%d, batchSize=%d", n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle,
                                    int n,
                                    double *Aarray[], int lda,
                                    int *PivotArray,
                                    int *infoArray, int batchSize) {
    DEBUG_LOG("cublasDgetrfBatched called: n=%d, batchSize=%d", n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle,
                                    int n,
                                    void *Aarray[], int lda,
                                    int *PivotArray,
                                    int *infoArray, int batchSize) {
    DEBUG_LOG("cublasCgetrfBatched called: n=%d, batchSize=%d", n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle,
                                    int n,
                                    void *Aarray[], int lda,
                                    int *PivotArray,
                                    int *infoArray, int batchSize) {
    DEBUG_LOG("cublasZgetrfBatched called: n=%d, batchSize=%d", n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

// Solve batched
cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle,
                                     cublasOperation_t trans,
                                     int n, int nrhs,
                                     const float *Aarray[], int lda,
                                     const int *devIpiv,
                                     float *Barray[], int ldb,
                                     int *info, int batchSize) {
    DEBUG_LOG("cublasSgetrsBatched called: n=%d, nrhs=%d, batchSize=%d", n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
                                     cublasOperation_t trans,
                                     int n, int nrhs,
                                     const double *Aarray[], int lda,
                                     const int *devIpiv,
                                     double *Barray[], int ldb,
                                     int *info, int batchSize) {
    DEBUG_LOG("cublasDgetrsBatched called: n=%d, nrhs=%d, batchSize=%d", n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle,
                                     cublasOperation_t trans,
                                     int n, int nrhs,
                                     const void *Aarray[], int lda,
                                     const int *devIpiv,
                                     void *Barray[], int ldb,
                                     int *info, int batchSize) {
    DEBUG_LOG("cublasCgetrsBatched called: n=%d, nrhs=%d, batchSize=%d", n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle,
                                     cublasOperation_t trans,
                                     int n, int nrhs,
                                     const void *Aarray[], int lda,
                                     const int *devIpiv,
                                     void *Barray[], int ldb,
                                     int *info, int batchSize) {
    DEBUG_LOG("cublasZgetrsBatched called: n=%d, nrhs=%d, batchSize=%d", n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

// Least squares batched
cublasStatus_t cublasSgelsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int m, int n, int nrhs,
                                   float **Aarray, int lda,
                                   float **Carray, int ldc,
                                   int *info,
                                   int *devInfoArray,
                                   int batchSize) {
    DEBUG_LOG("cublasSgelsBatched called: m=%d, n=%d, nrhs=%d, batchSize=%d", m, n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int m, int n, int nrhs,
                                   double **Aarray, int lda,
                                   double **Carray, int ldc,
                                   int *info,
                                   int *devInfoArray,
                                   int batchSize) {
    DEBUG_LOG("cublasDgelsBatched called: m=%d, n=%d, nrhs=%d, batchSize=%d", m, n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int m, int n, int nrhs,
                                   void **Aarray, int lda,
                                   void **Carray, int ldc,
                                   int *info,
                                   int *devInfoArray,
                                   int batchSize) {
    DEBUG_LOG("cublasZgelsBatched called: m=%d, n=%d, nrhs=%d, batchSize=%d", m, n, nrhs, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

// Additional complex TRSM
cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const void *alpha,
                               const void *A, int lda,
                               void *B, int ldb) {
    DEBUG_LOG("cublasCtrsm_v2 called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const void *alpha,
                               const void *A, int lda,
                               void *B, int ldb) {
    DEBUG_LOG("cublasZtrsm_v2 called: m=%d, n=%d", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle,
                                   cublasSideMode_t side, cublasFillMode_t uplo,
                                   cublasOperation_t trans, cublasDiagType_t diag,
                                   int m, int n,
                                   const void *alpha,
                                   const void *A[], int lda,
                                   void *B[], int ldb,
                                   int batchCount) {
    DEBUG_LOG("cublasCtrsmBatched called: m=%d, n=%d, batchCount=%d", m, n, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle,
                                   cublasSideMode_t side, cublasFillMode_t uplo,
                                   cublasOperation_t trans, cublasDiagType_t diag,
                                   int m, int n,
                                   const void *alpha,
                                   const void *A[], int lda,
                                   void *B[], int ldb,
                                   int batchCount) {
    DEBUG_LOG("cublasZtrsmBatched called: m=%d, n=%d, batchCount=%d", m, n, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// Additional complex batched GEMM
cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle,
                                          cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k,
                                          const void *alpha,
                                          const void *A, int lda, long long int strideA,
                                          const void *B, int ldb, long long int strideB,
                                          const void *beta,
                                          void *C, int ldc, long long int strideC,
                                          int batchCount) {
    DEBUG_LOG("cublasZgemmStridedBatched called: m=%d, n=%d, k=%d, batchCount=%d", m, n, k, batchCount);
    DEBUG_LOG("cublasZgemmStridedBatched complex GEMM is not implemented on PACC yet; no stub fallback active");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
