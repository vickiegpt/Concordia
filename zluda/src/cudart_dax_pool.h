#ifndef HETGPU_CUDART_DAX_POOL_H
#define HETGPU_CUDART_DAX_POOL_H

#include <stddef.h>
#include <stdint.h>

/*
 * Optional DAX-backed storage for CUDA host buffers.  The pool is disabled
 * unless HETGPU_CXL_KV_DAX names a device and only redirects allocations at
 * or above HETGPU_CXL_KV_DAX_MIN_BYTES.
 */
int hetgpu_cxl_dax_should_redirect(size_t size);
uint64_t hetgpu_cxl_dax_pool_capacity(void);
uint64_t hetgpu_cxl_dax_pool_base(void);
size_t hetgpu_cxl_dax_pool_min_bytes(void);
void *hetgpu_cxl_dax_host_alloc(size_t size);
int hetgpu_cxl_dax_host_free(void *ptr);
/* Copy into a pool allocation through the devdax file descriptor. */
int hetgpu_cxl_dax_host_copy(void *dst, const void *src, size_t size);

#endif
