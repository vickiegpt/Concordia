#define _GNU_SOURCE

#include "cudart_dax_pool.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HETGPU_CXL_DAX_DEFAULT_CAPACITY (32ULL * 1024ULL * 1024ULL * 1024ULL)
#define HETGPU_CXL_DAX_DEFAULT_BASE (0x100000000ULL)
#define HETGPU_CXL_DAX_DEFAULT_MIN_BYTES (4ULL * 1024ULL * 1024ULL)

struct hetgpu_cxl_dax_allocation {
    void *address;
    uint64_t offset;
    size_t requested;
    struct hetgpu_cxl_dax_allocation *next;
};

static pthread_mutex_t hetgpu_cxl_dax_lock = PTHREAD_MUTEX_INITIALIZER;
static int hetgpu_cxl_dax_initialized;
static int hetgpu_cxl_dax_fd = -1;
static uint64_t hetgpu_cxl_dax_capacity;
static uint64_t hetgpu_cxl_dax_base;
static size_t hetgpu_cxl_dax_minimum;
static uint64_t hetgpu_cxl_dax_next;
static size_t hetgpu_cxl_dax_page_size;
static char hetgpu_cxl_dax_path[4096];
static void *hetgpu_cxl_dax_mapping;
static size_t hetgpu_cxl_dax_mapping_length;
static struct hetgpu_cxl_dax_allocation *hetgpu_cxl_dax_allocations;

static uint64_t hetgpu_cxl_dax_parse_u64(const char *name, uint64_t fallback) {
    const char *value = getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 0);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
        return fallback;
    }
    return (uint64_t)parsed;
}

static size_t hetgpu_cxl_dax_align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) / alignment * alignment;
}

static int hetgpu_cxl_dax_configure_locked(void) {
    if (hetgpu_cxl_dax_initialized) {
        return hetgpu_cxl_dax_fd >= 0;
    }
    hetgpu_cxl_dax_initialized = 1;

    const char *path = getenv("HETGPU_CXL_KV_DAX");
    if (!path || !*path) {
        return 0;
    }
    snprintf(hetgpu_cxl_dax_path, sizeof(hetgpu_cxl_dax_path), "%s", path);
    hetgpu_cxl_dax_capacity = hetgpu_cxl_dax_parse_u64(
        "HETGPU_CXL_KV_DAX_BYTES", HETGPU_CXL_DAX_DEFAULT_CAPACITY);
    hetgpu_cxl_dax_base = hetgpu_cxl_dax_parse_u64(
        "HETGPU_CXL_KV_DAX_BASE", HETGPU_CXL_DAX_DEFAULT_BASE);
    hetgpu_cxl_dax_minimum = (size_t)hetgpu_cxl_dax_parse_u64(
        "HETGPU_CXL_KV_DAX_MIN_BYTES", HETGPU_CXL_DAX_DEFAULT_MIN_BYTES);
    hetgpu_cxl_dax_page_size = (size_t)sysconf(_SC_PAGESIZE);
    if (hetgpu_cxl_dax_page_size == 0) {
        hetgpu_cxl_dax_page_size = 4096;
    }

    if (hetgpu_cxl_dax_base >= hetgpu_cxl_dax_capacity ||
        hetgpu_cxl_dax_minimum == 0 ||
        hetgpu_cxl_dax_page_size > (size_t)INT64_MAX ||
        hetgpu_cxl_dax_base % hetgpu_cxl_dax_page_size != 0) {
        fprintf(stderr,
                "[cudart_shim] CXL KV DAX disabled: invalid layout path=%s base=0x%" PRIx64
                " capacity=0x%" PRIx64 " min=%zu page=%zu\n",
                hetgpu_cxl_dax_path, hetgpu_cxl_dax_base,
                hetgpu_cxl_dax_capacity, hetgpu_cxl_dax_minimum,
                hetgpu_cxl_dax_page_size);
        return 0;
    }

    hetgpu_cxl_dax_fd = open(hetgpu_cxl_dax_path, O_RDWR | O_CLOEXEC);
    if (hetgpu_cxl_dax_fd < 0) {
        fprintf(stderr, "[cudart_shim] CXL KV DAX disabled: open %s failed: %s\n",
                hetgpu_cxl_dax_path, strerror(errno));
        return 0;
    }
    if (hetgpu_cxl_dax_capacity > SIZE_MAX) {
        fprintf(stderr, "[cudart_shim] CXL KV DAX disabled: capacity does not fit size_t\n");
        close(hetgpu_cxl_dax_fd);
        hetgpu_cxl_dax_fd = -1;
        return 0;
    }
    hetgpu_cxl_dax_mapping_length = (size_t)hetgpu_cxl_dax_capacity;
    hetgpu_cxl_dax_mapping = mmap(NULL, hetgpu_cxl_dax_mapping_length,
                                   PROT_READ | PROT_WRITE, MAP_SHARED,
                                   hetgpu_cxl_dax_fd, 0);
    if (hetgpu_cxl_dax_mapping == MAP_FAILED) {
        fprintf(stderr, "[cudart_shim] CXL KV DAX disabled: mmap %s length=%zu failed: %s\n",
                hetgpu_cxl_dax_path, hetgpu_cxl_dax_mapping_length, strerror(errno));
        hetgpu_cxl_dax_mapping = NULL;
        hetgpu_cxl_dax_mapping_length = 0;
        close(hetgpu_cxl_dax_fd);
        hetgpu_cxl_dax_fd = -1;
        return 0;
    }
    hetgpu_cxl_dax_next = hetgpu_cxl_dax_base;
    fprintf(stderr,
            "[cudart_shim] CXL KV DAX enabled path=%s base=0x%" PRIx64
            " capacity=%" PRIu64 " bytes min=%zu\n",
            hetgpu_cxl_dax_path, hetgpu_cxl_dax_base,
            hetgpu_cxl_dax_capacity, hetgpu_cxl_dax_minimum);
    return 1;
}

int hetgpu_cxl_dax_should_redirect(size_t size) {
    pthread_mutex_lock(&hetgpu_cxl_dax_lock);
    int enabled = hetgpu_cxl_dax_configure_locked();
    int redirect = enabled && size >= hetgpu_cxl_dax_minimum;
    if (redirect) {
        uint64_t bytes = (uint64_t)size;
        redirect = bytes <= hetgpu_cxl_dax_capacity &&
                   hetgpu_cxl_dax_next <= hetgpu_cxl_dax_capacity - bytes;
    }
    pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
    return redirect;
}

uint64_t hetgpu_cxl_dax_pool_capacity(void) {
    pthread_mutex_lock(&hetgpu_cxl_dax_lock);
    (void)hetgpu_cxl_dax_configure_locked();
    uint64_t value = hetgpu_cxl_dax_capacity;
    pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
    return value;
}

uint64_t hetgpu_cxl_dax_pool_base(void) {
    pthread_mutex_lock(&hetgpu_cxl_dax_lock);
    (void)hetgpu_cxl_dax_configure_locked();
    uint64_t value = hetgpu_cxl_dax_base;
    pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
    return value;
}

size_t hetgpu_cxl_dax_pool_min_bytes(void) {
    pthread_mutex_lock(&hetgpu_cxl_dax_lock);
    (void)hetgpu_cxl_dax_configure_locked();
    size_t value = hetgpu_cxl_dax_minimum;
    pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
    return value;
}

void *hetgpu_cxl_dax_host_alloc(size_t size) {
    pthread_mutex_lock(&hetgpu_cxl_dax_lock);
    if (!hetgpu_cxl_dax_configure_locked() || size < hetgpu_cxl_dax_minimum) {
        pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
        return NULL;
    }

    size_t map_length = hetgpu_cxl_dax_align_up(size, hetgpu_cxl_dax_page_size);
    if (map_length == 0 ||
        (uint64_t)map_length > hetgpu_cxl_dax_capacity ||
        hetgpu_cxl_dax_next > hetgpu_cxl_dax_capacity - (uint64_t)map_length) {
        fprintf(stderr,
                "[cudart_shim] CXL KV DAX exhausted request=%zu next=0x%" PRIx64
                " capacity=0x%" PRIx64 "\n",
                size, hetgpu_cxl_dax_next, hetgpu_cxl_dax_capacity);
        pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
        return NULL;
    }

    uint64_t offset = hetgpu_cxl_dax_next;
    void *address = (unsigned char *)hetgpu_cxl_dax_mapping + offset;

    struct hetgpu_cxl_dax_allocation *allocation = malloc(sizeof(*allocation));
    if (!allocation) {
        pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
        return NULL;
    }
    allocation->address = address;
    allocation->offset = offset;
    allocation->requested = size;
    allocation->next = hetgpu_cxl_dax_allocations;
    hetgpu_cxl_dax_allocations = allocation;
    hetgpu_cxl_dax_next += (uint64_t)map_length;
    fprintf(stderr,
            "[cudart_shim] CXL KV DAX alloc ptr=%p offset=0x%" PRIx64
            " size=%zu mapped_pool=%zu\n",
            address, offset, size, hetgpu_cxl_dax_mapping_length);
    pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
    return address;
}

int hetgpu_cxl_dax_host_free(void *ptr) {
    if (!ptr) {
        return 0;
    }
    pthread_mutex_lock(&hetgpu_cxl_dax_lock);
    struct hetgpu_cxl_dax_allocation **cursor = &hetgpu_cxl_dax_allocations;
    while (*cursor && (*cursor)->address != ptr) {
        cursor = &(*cursor)->next;
    }
    struct hetgpu_cxl_dax_allocation *allocation = *cursor;
    if (!allocation) {
        pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
        return 0;
    }
    *cursor = allocation->next;
    fprintf(stderr,
            "[cudart_shim] CXL KV DAX free ptr=%p offset=0x%" PRIx64
            " size=%zu mapped_pool=%zu\n",
            allocation->address, allocation->offset,
            allocation->requested, hetgpu_cxl_dax_mapping_length);
    free(allocation);
    pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
    return 1;
}

int hetgpu_cxl_dax_host_copy(void *dst, const void *src, size_t size) {
    if (!dst || !src || size == 0) {
        return -1;
    }

    pthread_mutex_lock(&hetgpu_cxl_dax_lock);
    if (!hetgpu_cxl_dax_configure_locked()) {
        pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
        return -1;
    }

    struct hetgpu_cxl_dax_allocation *allocation = hetgpu_cxl_dax_allocations;
    while (allocation && allocation->address != dst) {
        allocation = allocation->next;
    }
    if (!allocation || size > allocation->requested) {
        pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
        return -1;
    }

    ssize_t written = pwrite(hetgpu_cxl_dax_fd, src, size, (off_t) allocation->offset);
    pthread_mutex_unlock(&hetgpu_cxl_dax_lock);
    return written == (ssize_t) size ? 0 : -1;
}
