#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HETGPU_TQ1_ABI_VERSION 1u
#define HETGPU_TQ1_NOT_HANDLED 0
#define HETGPU_TQ1_HANDLED 1
#define HETGPU_TQ1_ERROR (-1)

enum hetgpu_tq1_role_v1 {
    HETGPU_TQ1_ROLE_GATE_EXPS = 1,
    HETGPU_TQ1_ROLE_UP_EXPS = 2,
    HETGPU_TQ1_ROLE_DOWN_EXPS = 3,
    HETGPU_TQ1_ROLE_GATE_UP_EXPS = 4,
};

struct hetgpu_tq1_tensor_v1 {
    uint32_t abi_version;
    uint32_t ggml_type;
    uint32_t role;
    uint32_t file_index;
    const char * name;
    const char * path;
    uint64_t file_offset;
    uint64_t nbytes;
    int64_t ne[4];
    uint64_t nb[4];
};

struct hetgpu_tq1_mul_mat_id_v1 {
    uint32_t abi_version;
    uint32_t src0_type;
    uint32_t src1_type;
    uint32_t ids_type;
    uint32_t dst_type;
    uint32_t reserved;
    const char * src0_name;
    const void * src1_device;
    const void * ids_device;
    void * dst_device;
    void * cuda_stream;
    int64_t src0_ne[4];
    uint64_t src0_nb[4];
    int64_t src1_ne[4];
    uint64_t src1_nb[4];
    int64_t ids_ne[4];
    uint64_t ids_nb[4];
    int64_t dst_ne[4];
    uint64_t dst_nb[4];
};

typedef int (*hetgpu_tq1_register_tensor_v1_fn)(const struct hetgpu_tq1_tensor_v1 *);
typedef int (*hetgpu_tq1_try_mul_mat_id_v1_fn)(const struct hetgpu_tq1_mul_mat_id_v1 *);

#ifdef __cplusplus
}
#endif
