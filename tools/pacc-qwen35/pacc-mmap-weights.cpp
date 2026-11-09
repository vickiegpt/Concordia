#include "pacc-mmap-weights.h"
#include "ggml-backend-impl.h"
#include "llama-impl.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>

namespace {
struct borrowed_weights { void * data; size_t size; };
bool contains(ggml_backend_buffer_t b, const void * ptr, size_t size) {
    auto * ctx = static_cast<borrowed_weights *>(b->context);
    auto start = reinterpret_cast<uintptr_t>(ctx->data);
    auto address = reinterpret_cast<uintptr_t>(ptr);
    return address >= start && address - start <= ctx->size && size <= ctx->size - (address - start);
}
void release(ggml_backend_buffer_t b) {
    // llama_model retains and releases the underlying read-only file mapping.
    delete static_cast<borrowed_weights *>(b->context);
}
void * base(ggml_backend_buffer_t b) { return static_cast<borrowed_weights *>(b->context)->data; }
ggml_status init(ggml_backend_buffer_t b, ggml_tensor * t) {
    if (!contains(b, t->data, ggml_nbytes(t)) ||
        ggml_backend_buft_get_alloc_size(b->buft, t) != ggml_nbytes(t)) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}
void get(ggml_backend_buffer_t b, const ggml_tensor * t, void * dst, size_t offset, size_t size) {
    GGML_ASSERT(offset <= ggml_nbytes(t) && size <= ggml_nbytes(t) - offset);
    auto * src = static_cast<const char *>(t->data) + offset;
    GGML_ASSERT(contains(b, src, size));
    std::memcpy(dst, src, size);
}
void set(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {
    GGML_ABORT("PACC mapped weights are immutable");
}
void fill(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {
    GGML_ABORT("PACC mapped weights are immutable");
}
void clear(ggml_backend_buffer_t, uint8_t) { GGML_ABORT("PACC mapped weights are immutable"); }
}

bool pacc_mmap_weights_enabled(ggml_backend_buffer_type_t buft, ggml_context * ctx) {
    const char * flag = std::getenv("HETGPU_SIFIVE_MMAP_WEIGHTS");
    if (!flag || std::strcmp(flag, "1") || std::strncmp(ggml_backend_buft_name(buft), "CUDA", 4)) return false;
    auto mode = reinterpret_cast<uint32_t (*)()>(dlsym(RTLD_DEFAULT, "hetgpu_sifive_host_backing_mode_v1"));
    if (!mode || mode() != 1) throw std::runtime_error("mapped weights require the explicit PACC host-backed ABI");
    for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (ggml_backend_buft_get_alloc_size(buft, t) != ggml_nbytes(t)) {
            // Never expose adjacent file bytes where a backend expects zero padding.
            throw std::runtime_error("mapped weights do not support tensors requiring backend padding");
        }
    }
    return true;
}

ggml_backend_buffer_t pacc_mmap_weights_buffer(ggml_backend_buffer_type_t buft, void * data, size_t size) {
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = release; iface.get_base = base; iface.init_tensor = init;
    iface.memset_tensor = fill; iface.set_tensor = set; iface.get_tensor = get; iface.clear = clear;
    LLAMA_LOG_INFO("PACC_MMAP_WEIGHTS bytes=%zu immutable=1 borrowed=1\n", size);
    return ggml_backend_buffer_init(buft, iface, new borrowed_weights{data, size}, size);
}
