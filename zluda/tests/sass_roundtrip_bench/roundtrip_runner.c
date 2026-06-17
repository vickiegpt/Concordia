#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define BLOCK_SIZE 128

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxDestroy_v2)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree_v2)(CUdeviceptr);
static CUresult (*p_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleUnload)(CUmodule);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuLaunchKernel)(CUfunction,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, CUstream, void **, void **);

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static int load_symbol(void **slot, const char *name) {
    *slot = dlsym(RTLD_DEFAULT, name);
    if (*slot == NULL) {
        fprintf(stderr, "missing CUDA symbol from LD_PRELOAD namespace: %s\n", name);
        return 0;
    }
    return 1;
}

#define LOAD_CU(name)                                                            \
    do {                                                                         \
        if (!load_symbol((void **)&p_##name, #name)) {                           \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static unsigned char *read_file(const char *path, size_t *size_out, int nul_term) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size <= 0) {
        fprintf(stderr, "invalid file size for %s: %ld\n", path, size);
        fclose(f);
        return NULL;
    }
    rewind(f);

    size_t alloc_size = (size_t)size + (nul_term ? 1u : 0u);
    unsigned char *data = (unsigned char *)malloc(alloc_size);
    if (data == NULL) {
        fprintf(stderr, "malloc failed for %zu bytes\n", alloc_size);
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "short read for %s\n", path);
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (nul_term) {
        data[size] = 0;
    }
    *size_out = (size_t)size;
    return data;
}

static void fill_input(uint32_t *values, int n) {
    for (int i = 0; i < n; ++i) {
        values[i] = (uint32_t)(i * 2654435761u) ^ 0x12345678u;
    }
}

static CUresult launch_repeated(CUfunction function, CUdeviceptr d_out, CUdeviceptr d_in,
                                int n, int iterations, uint64_t *elapsed_us) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64_t start = now_us();
    for (int i = 0; i < iterations; ++i) {
        void *args[] = { &d_out, &d_in, &n };
        CUresult status = p_cuLaunchKernel(function, (unsigned int)grid, 1, 1,
                                           BLOCK_SIZE, 1, 1, 0, NULL, args, NULL);
        if (status != CUDA_SUCCESS) {
            return status;
        }
    }
    CUresult status = p_cuCtxSynchronize();
    *elapsed_us = now_us() - start;
    return status;
}

static int first_mismatch(const uint32_t *a, const uint32_t *b, int n) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return -1;
}

static void emit_result(const char *case_name, const char *status, size_t cubin_bytes,
                        size_t lifted_ptx_bytes, uint64_t load_cubin_us,
                        uint64_t load_ptx_us, uint64_t kernel_cubin_us,
                        uint64_t kernel_ptx_us, uint64_t total_us,
                        const char *message) {
    printf("%s\t%s\t%zu\t%zu\t%llu\t%llu\t%llu\t%llu\t%llu\t%s\n",
           case_name, status, cubin_bytes, lifted_ptx_bytes,
           (unsigned long long)load_cubin_us,
           (unsigned long long)load_ptx_us,
           (unsigned long long)kernel_cubin_us,
           (unsigned long long)kernel_ptx_us,
           (unsigned long long)total_us, message);
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s case cubin lifted.ptx sm n warmups iterations\n", argv[0]);
        return 2;
    }

    const char *case_name = argv[1];
    const char *cubin_path = argv[2];
    const char *lifted_ptx_path = argv[3];
    int n = atoi(argv[5]);
    int warmups = atoi(argv[6]);
    int iterations = atoi(argv[7]);
    if (n <= 0 || warmups < 0 || iterations <= 0) {
        fprintf(stderr, "invalid n/warmups/iterations\n");
        return 2;
    }

    LOAD_CU(cuInit);
    LOAD_CU(cuDeviceGet);
    LOAD_CU(cuCtxCreate_v2);
    LOAD_CU(cuCtxDestroy_v2);
    LOAD_CU(cuCtxSynchronize);
    LOAD_CU(cuMemAlloc_v2);
    LOAD_CU(cuMemFree_v2);
    LOAD_CU(cuMemcpyHtoD_v2);
    LOAD_CU(cuMemcpyDtoH_v2);
    LOAD_CU(cuModuleLoadData);
    LOAD_CU(cuModuleUnload);
    LOAD_CU(cuModuleGetFunction);
    LOAD_CU(cuLaunchKernel);

    int rc = 0;
    size_t cubin_bytes = 0;
    size_t lifted_ptx_bytes = 0;
    uint64_t load_cubin_us = 0;
    uint64_t load_ptx_us = 0;
    uint64_t kernel_cubin_us = 0;
    uint64_t kernel_ptx_us = 0;
    uint64_t total_start = now_us();
    unsigned char *cubin = read_file(cubin_path, &cubin_bytes, 0);
    unsigned char *lifted_ptx = NULL;
    uint32_t *h_in = NULL;
    uint32_t *h_cubin = NULL;
    uint32_t *h_ptx = NULL;
    CUcontext ctx = NULL;
    CUmodule cubin_module = NULL;
    CUmodule ptx_module = NULL;
    CUfunction cubin_function = NULL;
    CUfunction ptx_function = NULL;
    CUdeviceptr d_in = 0;
    CUdeviceptr d_cubin = 0;
    CUdeviceptr d_ptx = 0;
    const size_t bytes = (size_t)n * sizeof(uint32_t);

    if (cubin == NULL) {
        return 1;
    }

    h_in = (uint32_t *)malloc(bytes);
    h_cubin = (uint32_t *)calloc((size_t)n, sizeof(uint32_t));
    h_ptx = (uint32_t *)calloc((size_t)n, sizeof(uint32_t));
    if (h_in == NULL || h_cubin == NULL || h_ptx == NULL) {
        fprintf(stderr, "host allocation failed\n");
        rc = 1;
        goto cleanup;
    }
    fill_input(h_in, n);

    CUdevice device = 0;
    if (p_cuInit(0) != CUDA_SUCCESS ||
        p_cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        p_cuCtxCreate_v2(&ctx, 0, device) != CUDA_SUCCESS ||
        p_cuMemAlloc_v2(&d_in, bytes) != CUDA_SUCCESS ||
        p_cuMemAlloc_v2(&d_cubin, bytes) != CUDA_SUCCESS ||
        p_cuMemAlloc_v2(&d_ptx, bytes) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(d_in, h_in, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA setup failed\n");
        rc = 1;
        goto cleanup;
    }

    uint64_t start = now_us();
    CUresult status = p_cuModuleLoadData(&cubin_module, cubin);
    load_cubin_us = now_us() - start;
    if (status != CUDA_SUCCESS) {
        char message[96];
        snprintf(message, sizeof(message), "cuModuleLoadData_cubin_failed_%d", status);
        emit_result(case_name, "load_cubin_failed", cubin_bytes, 0, load_cubin_us,
                    0, 0, 0, now_us() - total_start, message);
        rc = 0;
        goto cleanup;
    }

    status = p_cuModuleGetFunction(&cubin_function, cubin_module, case_name);
    if (status != CUDA_SUCCESS) {
        char message[96];
        snprintf(message, sizeof(message), "cuModuleGetFunction_cubin_failed_%d", status);
        emit_result(case_name, "get_cubin_function_failed", cubin_bytes, 0,
                    load_cubin_us, 0, 0, 0, now_us() - total_start,
                    message);
        rc = 0;
        goto cleanup;
    }

    if (warmups > 0) {
        status = launch_repeated(cubin_function, d_cubin, d_in, n, warmups, &kernel_cubin_us);
        if (status != CUDA_SUCCESS) {
            char message[96];
            snprintf(message, sizeof(message), "cuLaunchKernel_cubin_warmup_failed_%d", status);
            emit_result(case_name, "launch_cubin_warmup_failed", cubin_bytes, 0,
                        load_cubin_us, 0, 0, 0, now_us() - total_start,
                        message);
            rc = 0;
            goto cleanup;
        }
    }
    status = launch_repeated(cubin_function, d_cubin, d_in, n, iterations, &kernel_cubin_us);
    if (status != CUDA_SUCCESS ||
        p_cuMemcpyDtoH_v2(h_cubin, d_cubin, bytes) != CUDA_SUCCESS) {
        char message[96];
        snprintf(message, sizeof(message), "cubin_execution_failed_%d", status);
        emit_result(case_name, "run_cubin_failed", cubin_bytes, 0, load_cubin_us,
                    0, kernel_cubin_us, 0, now_us() - total_start,
                    message);
        rc = 0;
        goto cleanup;
    }

    lifted_ptx = read_file(lifted_ptx_path, &lifted_ptx_bytes, 1);
    if (lifted_ptx == NULL) {
        emit_result(case_name, "missing_lifted_ptx", cubin_bytes, 0, load_cubin_us,
                    0, kernel_cubin_us, 0, now_us() - total_start,
                    "lifted_ptx_dump_missing");
        rc = 0;
        goto cleanup;
    }

    start = now_us();
    status = p_cuModuleLoadData(&ptx_module, lifted_ptx);
    load_ptx_us = now_us() - start;
    if (status != CUDA_SUCCESS) {
        char message[96];
        snprintf(message, sizeof(message), "cuModuleLoadData_lifted_ptx_failed_%d", status);
        emit_result(case_name, "load_ptx_failed", cubin_bytes, lifted_ptx_bytes,
                    load_cubin_us, load_ptx_us, kernel_cubin_us, 0,
                    now_us() - total_start, message);
        rc = 0;
        goto cleanup;
    }

    status = p_cuModuleGetFunction(&ptx_function, ptx_module, case_name);
    if (status != CUDA_SUCCESS) {
        char message[96];
        snprintf(message, sizeof(message), "cuModuleGetFunction_ptx_failed_%d", status);
        emit_result(case_name, "get_ptx_function_failed", cubin_bytes, lifted_ptx_bytes,
                    load_cubin_us, load_ptx_us, kernel_cubin_us, 0,
                    now_us() - total_start, message);
        rc = 0;
        goto cleanup;
    }

    if (warmups > 0) {
        status = launch_repeated(ptx_function, d_ptx, d_in, n, warmups, &kernel_ptx_us);
        if (status != CUDA_SUCCESS) {
            char message[96];
            snprintf(message, sizeof(message), "cuLaunchKernel_ptx_warmup_failed_%d", status);
            emit_result(case_name, "launch_ptx_warmup_failed", cubin_bytes,
                        lifted_ptx_bytes, load_cubin_us, load_ptx_us,
                        kernel_cubin_us, 0, now_us() - total_start,
                        message);
            rc = 0;
            goto cleanup;
        }
    }
    status = launch_repeated(ptx_function, d_ptx, d_in, n, iterations, &kernel_ptx_us);
    if (status != CUDA_SUCCESS ||
        p_cuMemcpyDtoH_v2(h_ptx, d_ptx, bytes) != CUDA_SUCCESS) {
        char message[96];
        snprintf(message, sizeof(message), "lifted_ptx_execution_failed_%d", status);
        emit_result(case_name, "run_ptx_failed", cubin_bytes, lifted_ptx_bytes,
                    load_cubin_us, load_ptx_us, kernel_cubin_us, kernel_ptx_us,
                    now_us() - total_start, message);
        rc = 0;
        goto cleanup;
    }

    int mismatch = first_mismatch(h_cubin, h_ptx, n);
    if (mismatch >= 0) {
        char message[128];
        snprintf(message, sizeof(message), "mismatch_i%d_cubin%08x_ptx%08x",
                 mismatch, h_cubin[mismatch], h_ptx[mismatch]);
        emit_result(case_name, "mismatch", cubin_bytes, lifted_ptx_bytes,
                    load_cubin_us, load_ptx_us, kernel_cubin_us, kernel_ptx_us,
                    now_us() - total_start, message);
    } else {
        emit_result(case_name, "pass", cubin_bytes, lifted_ptx_bytes,
                    load_cubin_us, load_ptx_us, kernel_cubin_us, kernel_ptx_us,
                    now_us() - total_start, "ok");
    }

cleanup:
    if (ptx_module != NULL && p_cuModuleUnload != NULL) {
        (void)p_cuModuleUnload(ptx_module);
    }
    if (cubin_module != NULL && p_cuModuleUnload != NULL) {
        (void)p_cuModuleUnload(cubin_module);
    }
    if (d_ptx != 0 && p_cuMemFree_v2 != NULL) {
        (void)p_cuMemFree_v2(d_ptx);
    }
    if (d_cubin != 0 && p_cuMemFree_v2 != NULL) {
        (void)p_cuMemFree_v2(d_cubin);
    }
    if (d_in != 0 && p_cuMemFree_v2 != NULL) {
        (void)p_cuMemFree_v2(d_in);
    }
    if (ctx != NULL && p_cuCtxDestroy_v2 != NULL) {
        (void)p_cuCtxDestroy_v2(ctx);
    }
    free(h_ptx);
    free(h_cubin);
    free(h_in);
    free(lifted_ptx);
    free(cubin);
    return rc;
}
