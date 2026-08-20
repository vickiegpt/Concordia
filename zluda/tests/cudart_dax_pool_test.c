#define _GNU_SOURCE

#include "../src/cudart_dax_pool.h"

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int main(void) {
    setenv("HETGPU_CXL_KV_DAX", "/tmp/hetgpu-test-dax", 1);
    setenv("HETGPU_CXL_KV_DAX_BYTES", "0x20000000", 1);
    setenv("HETGPU_CXL_KV_DAX_BASE", "0x2000000", 1);
    setenv("HETGPU_CXL_KV_DAX_MIN_BYTES", "0x400000", 1);

    int fd = open("/tmp/hetgpu-test-dax", O_CREAT | O_RDWR | O_TRUNC, 0600);
    assert(fd >= 0);
    assert(ftruncate(fd, 0x20000000) == 0);
    close(fd);

    assert(hetgpu_cxl_dax_pool_capacity() == 0x20000000ULL);
    assert(hetgpu_cxl_dax_pool_base() == 0x2000000ULL);
    assert(hetgpu_cxl_dax_pool_min_bytes() == 0x400000ULL);
    assert(hetgpu_cxl_dax_should_redirect(0x400000 - 1) == 0);
    assert(hetgpu_cxl_dax_should_redirect(0x400000) == 1);
    assert(hetgpu_cxl_dax_should_redirect(0x800000) == 1);

    void *ptr = hetgpu_cxl_dax_host_alloc(0x800000);
    assert(ptr != NULL);
    ((volatile uint64_t *)ptr)[0] = UINT64_C(0xfeedfacecafebeef);
    assert(((volatile uint64_t *)ptr)[0] == UINT64_C(0xfeedfacecafebeef));
    assert(hetgpu_cxl_dax_host_free(ptr) == 1);
    assert(hetgpu_cxl_dax_host_free(ptr) == 0);
    unlink("/tmp/hetgpu-test-dax");

    return 0;
}
