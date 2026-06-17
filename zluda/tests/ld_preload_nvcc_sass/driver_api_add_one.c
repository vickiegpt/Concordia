#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

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

#define CHECK_CU(expr)                                                           \
    do {                                                                         \
        CUresult _status = (expr);                                               \
        if (_status != CUDA_SUCCESS) {                                           \
            fprintf(stderr, "%s failed with CUDA result %d\n", #expr, _status);  \
            goto fail;                                                           \
        }                                                                        \
    } while (0)

static unsigned char *read_file(const char *path, size_t *size_out) {
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
        fprintf(stderr, "invalid CUBIN size: %ld\n", size);
        fclose(f);
        return NULL;
    }
    rewind(f);

    unsigned char *data = (unsigned char *)malloc((size_t)size);
    if (data == NULL) {
        fprintf(stderr, "malloc failed for %ld bytes\n", size);
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
    *size_out = (size_t)size;
    return data;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s add_one.cubin\n", argv[0]);
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

    int rc = 1;
    size_t cubin_size = 0;
    unsigned char *cubin = read_file(argv[1], &cubin_size);
    if (cubin == NULL) {
        return 1;
    }

    CUdevice device = 0;
    CUcontext ctx = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr d_in = 0;
    CUdeviceptr d_out = 0;
    enum { N = 64 };
    const size_t bytes = N * sizeof(float);
    float h_in[N];
    float h_out[N];

    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i * 0.25f;
        h_out[i] = 0.0f;
    }

    fprintf(stderr, "[ld-preload-e2e] loaded CUBIN file bytes=%zu\n", cubin_size);

    CHECK_CU(p_cuInit(0));
    CHECK_CU(p_cuDeviceGet(&device, 0));
    CHECK_CU(p_cuCtxCreate_v2(&ctx, 0, device));
    CHECK_CU(p_cuModuleLoadData(&module, cubin));
    CHECK_CU(p_cuModuleGetFunction(&function, module, "add_one"));
    CHECK_CU(p_cuMemAlloc_v2(&d_in, bytes));
    CHECK_CU(p_cuMemAlloc_v2(&d_out, bytes));
    CHECK_CU(p_cuMemcpyHtoD_v2(d_in, h_in, bytes));

    int n = N;
    void *args[] = { &d_out, &d_in, &n };
    CHECK_CU(p_cuLaunchKernel(function, 1, 1, 1, N, 1, 1, 0, NULL, args, NULL));
    CHECK_CU(p_cuCtxSynchronize());
    CHECK_CU(p_cuMemcpyDtoH_v2(h_out, d_out, bytes));

    for (int i = 0; i < N; ++i) {
        float expected = h_in[i] + 1.0f;
        if (h_out[i] != expected) {
            fprintf(stderr, "verification failed at %d: got %.8g expected %.8g\n",
                    i, h_out[i], expected);
            goto fail;
        }
    }

    puts("PASS nvcc cubin ld_preload sass e2e");
    rc = 0;

fail:
    if (d_out != 0 && p_cuMemFree_v2 != NULL) {
        (void)p_cuMemFree_v2(d_out);
    }
    if (d_in != 0 && p_cuMemFree_v2 != NULL) {
        (void)p_cuMemFree_v2(d_in);
    }
    if (module != NULL && p_cuModuleUnload != NULL) {
        (void)p_cuModuleUnload(module);
    }
    if (ctx != NULL && p_cuCtxDestroy_v2 != NULL) {
        (void)p_cuCtxDestroy_v2(ctx);
    }
    free(cubin);
    return rc;
}
