#define _GNU_SOURCE

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

extern int cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
extern int cudaFreeHost(void *pHost);

int main(void) {
    const char *path = "/tmp/hetgpu-test-dax-preload";
    setenv("HETGPU_CXL_KV_DAX", path, 1);
    setenv("HETGPU_CXL_KV_DAX_BYTES", "0x20000000", 1);
    setenv("HETGPU_CXL_KV_DAX_BASE", "0x2000000", 1);
    setenv("HETGPU_CXL_KV_DAX_MIN_BYTES", "0x400000", 1);

    int fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0600);
    assert(fd >= 0);
    assert(ftruncate(fd, 0x20000000) == 0);
    close(fd);

    void *ptr = NULL;
    assert(cudaHostAlloc(&ptr, 0x800000, 0) == 0);
    assert(ptr != NULL);
    ((volatile uint64_t *)ptr)[0] = UINT64_C(0xfeedfacecafebeef);
    assert(((volatile uint64_t *)ptr)[0] == UINT64_C(0xfeedfacecafebeef));
    assert(cudaFreeHost(ptr) == 0);

    unlink(path);
    return 0;
}
