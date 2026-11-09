#pragma once
#include "ggml-backend.h"

bool pacc_mmap_weights_enabled(ggml_backend_buffer_type_t buft, ggml_context * ctx);
ggml_backend_buffer_t pacc_mmap_weights_buffer(ggml_backend_buffer_type_t buft, void * data, size_t size);
