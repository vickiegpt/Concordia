// Generated from tools/qwen35-iq1s-layer-abi.json; do not edit.
// Canonical schema SHA-256: a627be37a468ad9d85a0fdb2d6a92650ac33e345f1e85d393d85bda222f9577f
#ifndef HETGPU_QWEN35_IQ1S_LAYER_GENERATED_H
#define HETGPU_QWEN35_IQ1S_LAYER_GENERATED_H

#include <stddef.h>
#include <stdint.h>

#define HETGPU_IQ1S_SCHEMA_SHA256 "a627be37a468ad9d85a0fdb2d6a92650ac33e345f1e85d393d85bda222f9577f"
#define HETGPU_IQ1S_ABI_VERSION 2u
#define HETGPU_IQ1S_REGISTER_MAGIC UINT32_C(0x324c5149)
#define HETGPU_IQ1S_COMMAND_MAGIC UINT32_C(0x32435149)
#define HETGPU_IQ1S_COMPLETION_MAGIC UINT32_C(0x32445149)
#define HETGPU_IQ1S_PHASE_A 1u
#define HETGPU_IQ1S_PHASE_B 2u
#define HETGPU_IQ1S_ROLE_GATE 1u
#define HETGPU_IQ1S_ROLE_UP 2u
#define HETGPU_IQ1S_ROLE_DOWN 3u
#define HETGPU_IQ1S_WEIGHT_FORMAT_IQ1_S 19u
#define HETGPU_IQ1S_COMPLETION_STATUS_OK 0u
#define HETGPU_IQ1S_COMPLETION_STATUS_FAULT 1u
#define HETGPU_IQ1S_COMPLETION_STATUS_ABORTED 2u
#define HETGPU_IQ1S_FAULT_CODE_NONE 0u
#define HETGPU_IQ1S_FAULT_CODE_BAD_MAGIC 1u
#define HETGPU_IQ1S_FAULT_CODE_BAD_VERSION 2u
#define HETGPU_IQ1S_FAULT_CODE_BAD_LENGTH 3u
#define HETGPU_IQ1S_FAULT_CODE_BAD_GENERATION 4u
#define HETGPU_IQ1S_FAULT_CODE_BAD_CRC 5u
#define HETGPU_IQ1S_FAULT_CODE_BAD_BOUNDS 6u
#define HETGPU_IQ1S_FAULT_CODE_BAD_RELOCATION 7u
#define HETGPU_IQ1S_FAULT_CODE_AXI 8u
#define HETGPU_IQ1S_FAULT_CODE_TIMEOUT 9u
#define HETGPU_IQ1S_FAULT_CODE_NONFINITE 10u
#define HETGPU_IQ1S_FAULT_CODE_RING_OVERFLOW 11u

#define HETGPU_IQ1S_REG_ABI_MAGIC_OFFSET 0u
#define HETGPU_IQ1S_REG_ABI_VERSION_OFFSET 4u
#define HETGPU_IQ1S_REG_CONTROL_OFFSET 8u
#define HETGPU_IQ1S_REG_STATUS_OFFSET 12u
#define HETGPU_IQ1S_REG_SESSION_GENERATION_LO_OFFSET 16u
#define HETGPU_IQ1S_REG_SESSION_GENERATION_HI_OFFSET 20u
#define HETGPU_IQ1S_REG_COMMAND_BASE_LO_OFFSET 24u
#define HETGPU_IQ1S_REG_COMMAND_BASE_HI_OFFSET 28u
#define HETGPU_IQ1S_REG_COMMAND_CAPACITY_OFFSET 32u
#define HETGPU_IQ1S_REG_COMMAND_PRODUCER_OFFSET 36u
#define HETGPU_IQ1S_REG_COMMAND_CONSUMER_OFFSET 40u
#define HETGPU_IQ1S_REG_DOORBELL_OFFSET 44u
#define HETGPU_IQ1S_REG_COMPLETION_BASE_LO_OFFSET 48u
#define HETGPU_IQ1S_REG_COMPLETION_BASE_HI_OFFSET 52u
#define HETGPU_IQ1S_REG_COMPLETION_CAPACITY_OFFSET 56u
#define HETGPU_IQ1S_REG_COMPLETION_PRODUCER_OFFSET 60u
#define HETGPU_IQ1S_REG_COMPLETION_CONSUMER_OFFSET 64u
#define HETGPU_IQ1S_REG_FAULT_CODE_OFFSET 68u
#define HETGPU_IQ1S_REG_FAULT_DETAIL_LO_OFFSET 72u
#define HETGPU_IQ1S_REG_FAULT_DETAIL_HI_OFFSET 76u
#define HETGPU_IQ1S_REG_QUIESCENT_OFFSET 80u

#define HETGPU_IQ1S_COMMAND_BYTES 128u
#define HETGPU_IQ1S_COMMAND_MAGIC_OFFSET 0u
#define HETGPU_IQ1S_COMMAND_ABI_VERSION_OFFSET 4u
#define HETGPU_IQ1S_COMMAND_DESCRIPTOR_BYTES_OFFSET 6u
#define HETGPU_IQ1S_COMMAND_CRC32_OFFSET 8u
#define HETGPU_IQ1S_COMMAND_FLAGS_OFFSET 12u
#define HETGPU_IQ1S_COMMAND_SESSION_GENERATION_OFFSET 16u
#define HETGPU_IQ1S_COMMAND_TRANSACTION_ID_OFFSET 24u
#define HETGPU_IQ1S_COMMAND_PROGRAM_ID_OFFSET 32u
#define HETGPU_IQ1S_COMMAND_TRACE_ID_OFFSET 40u
#define HETGPU_IQ1S_COMMAND_LAYER_ID_OFFSET 48u
#define HETGPU_IQ1S_COMMAND_PHASE_OFFSET 52u
#define HETGPU_IQ1S_COMMAND_ROLE_OFFSET 54u
#define HETGPU_IQ1S_COMMAND_EXPERT_ID_OFFSET 56u
#define HETGPU_IQ1S_COMMAND_LANE_MASK_OFFSET 58u
#define HETGPU_IQ1S_COMMAND_LANE_COUNT_OFFSET 60u
#define HETGPU_IQ1S_COMMAND_WEIGHT_FORMAT_OFFSET 62u
#define HETGPU_IQ1S_COMMAND_ARENA_OFFSET_OFFSET 64u
#define HETGPU_IQ1S_COMMAND_INPUT_OFFSET_OFFSET 72u
#define HETGPU_IQ1S_COMMAND_OUTPUT_OFFSET_OFFSET 80u
#define HETGPU_IQ1S_COMMAND_ROW_START_OFFSET 88u
#define HETGPU_IQ1S_COMMAND_ROW_COUNT_OFFSET 92u
#define HETGPU_IQ1S_COMMAND_INPUT_BYTES_OFFSET 96u
#define HETGPU_IQ1S_COMMAND_OUTPUT_BYTES_OFFSET 100u
#define HETGPU_IQ1S_COMMAND_TOKEN_MAP_OFFSET_OFFSET 104u
#define HETGPU_IQ1S_COMMAND_DEPENDENCY_FENCE_OFFSET 112u
#define HETGPU_IQ1S_COMMAND_COMPLETION_SLOT_OFFSET 120u
#define HETGPU_IQ1S_COMMAND_RESERVED_OFFSET 124u

#pragma pack(push, 1)
typedef struct hetgpu_iq1s_command_v2 {
    uint32_t magic;
    uint16_t abi_version;
    uint16_t descriptor_bytes;
    uint32_t crc32;
    uint32_t flags;
    uint64_t session_generation;
    uint64_t transaction_id;
    uint64_t program_id;
    uint64_t trace_id;
    uint32_t layer_id;
    uint16_t phase;
    uint16_t role;
    uint16_t expert_id;
    uint16_t lane_mask;
    uint16_t lane_count;
    uint16_t weight_format;
    uint64_t arena_offset;
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t row_start;
    uint32_t row_count;
    uint32_t input_bytes;
    uint32_t output_bytes;
    uint64_t token_map_offset;
    uint64_t dependency_fence;
    uint32_t completion_slot;
    uint32_t reserved;
} hetgpu_iq1s_command_v2;
#pragma pack(pop)

_Static_assert(sizeof(hetgpu_iq1s_command_v2) == HETGPU_IQ1S_COMMAND_BYTES, "command ABI size");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, magic) == HETGPU_IQ1S_COMMAND_MAGIC_OFFSET, "command.magic ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, abi_version) == HETGPU_IQ1S_COMMAND_ABI_VERSION_OFFSET, "command.abi_version ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, descriptor_bytes) == HETGPU_IQ1S_COMMAND_DESCRIPTOR_BYTES_OFFSET, "command.descriptor_bytes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, crc32) == HETGPU_IQ1S_COMMAND_CRC32_OFFSET, "command.crc32 ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, flags) == HETGPU_IQ1S_COMMAND_FLAGS_OFFSET, "command.flags ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, session_generation) == HETGPU_IQ1S_COMMAND_SESSION_GENERATION_OFFSET, "command.session_generation ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, transaction_id) == HETGPU_IQ1S_COMMAND_TRANSACTION_ID_OFFSET, "command.transaction_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, program_id) == HETGPU_IQ1S_COMMAND_PROGRAM_ID_OFFSET, "command.program_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, trace_id) == HETGPU_IQ1S_COMMAND_TRACE_ID_OFFSET, "command.trace_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, layer_id) == HETGPU_IQ1S_COMMAND_LAYER_ID_OFFSET, "command.layer_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, phase) == HETGPU_IQ1S_COMMAND_PHASE_OFFSET, "command.phase ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, role) == HETGPU_IQ1S_COMMAND_ROLE_OFFSET, "command.role ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, expert_id) == HETGPU_IQ1S_COMMAND_EXPERT_ID_OFFSET, "command.expert_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, lane_mask) == HETGPU_IQ1S_COMMAND_LANE_MASK_OFFSET, "command.lane_mask ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, lane_count) == HETGPU_IQ1S_COMMAND_LANE_COUNT_OFFSET, "command.lane_count ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, weight_format) == HETGPU_IQ1S_COMMAND_WEIGHT_FORMAT_OFFSET, "command.weight_format ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, arena_offset) == HETGPU_IQ1S_COMMAND_ARENA_OFFSET_OFFSET, "command.arena_offset ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, input_offset) == HETGPU_IQ1S_COMMAND_INPUT_OFFSET_OFFSET, "command.input_offset ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, output_offset) == HETGPU_IQ1S_COMMAND_OUTPUT_OFFSET_OFFSET, "command.output_offset ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, row_start) == HETGPU_IQ1S_COMMAND_ROW_START_OFFSET, "command.row_start ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, row_count) == HETGPU_IQ1S_COMMAND_ROW_COUNT_OFFSET, "command.row_count ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, input_bytes) == HETGPU_IQ1S_COMMAND_INPUT_BYTES_OFFSET, "command.input_bytes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, output_bytes) == HETGPU_IQ1S_COMMAND_OUTPUT_BYTES_OFFSET, "command.output_bytes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, token_map_offset) == HETGPU_IQ1S_COMMAND_TOKEN_MAP_OFFSET_OFFSET, "command.token_map_offset ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, dependency_fence) == HETGPU_IQ1S_COMMAND_DEPENDENCY_FENCE_OFFSET, "command.dependency_fence ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, completion_slot) == HETGPU_IQ1S_COMMAND_COMPLETION_SLOT_OFFSET, "command.completion_slot ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_command_v2, reserved) == HETGPU_IQ1S_COMMAND_RESERVED_OFFSET, "command.reserved ABI offset");

#define HETGPU_IQ1S_COMPLETION_BYTES 128u
#define HETGPU_IQ1S_COMPLETION_MAGIC_OFFSET 0u
#define HETGPU_IQ1S_COMPLETION_ABI_VERSION_OFFSET 4u
#define HETGPU_IQ1S_COMPLETION_COMPLETION_BYTES_OFFSET 6u
#define HETGPU_IQ1S_COMPLETION_STATUS_OFFSET 8u
#define HETGPU_IQ1S_COMPLETION_FAULT_CODE_OFFSET 12u
#define HETGPU_IQ1S_COMPLETION_SESSION_GENERATION_OFFSET 16u
#define HETGPU_IQ1S_COMPLETION_TRANSACTION_ID_OFFSET 24u
#define HETGPU_IQ1S_COMPLETION_PROGRAM_ID_OFFSET 32u
#define HETGPU_IQ1S_COMPLETION_TRACE_ID_OFFSET 40u
#define HETGPU_IQ1S_COMPLETION_LAYER_ID_OFFSET 48u
#define HETGPU_IQ1S_COMPLETION_PHASE_OFFSET 52u
#define HETGPU_IQ1S_COMPLETION_ROLE_OFFSET 54u
#define HETGPU_IQ1S_COMPLETION_CU_ID_OFFSET 56u
#define HETGPU_IQ1S_COMPLETION_EXPERT_ID_OFFSET 58u
#define HETGPU_IQ1S_COMPLETION_LANE_MASK_OFFSET 60u
#define HETGPU_IQ1S_COMPLETION_ROWS_COMPLETED_OFFSET 62u
#define HETGPU_IQ1S_COMPLETION_DESCRIPTOR_CRC32_OFFSET 64u
#define HETGPU_IQ1S_COMPLETION_COMMAND_INDEX_OFFSET 68u
#define HETGPU_IQ1S_COMPLETION_CYCLES_OFFSET 72u
#define HETGPU_IQ1S_COMPLETION_DDR_READ_BYTES_OFFSET 80u
#define HETGPU_IQ1S_COMPLETION_DDR_WRITE_BYTES_OFFSET 88u
#define HETGPU_IQ1S_COMPLETION_IQ1S_BLOCKS_OFFSET 96u
#define HETGPU_IQ1S_COMPLETION_GRID_PASSES_OFFSET 104u
#define HETGPU_IQ1S_COMPLETION_DELTA_PASSES_OFFSET 108u
#define HETGPU_IQ1S_COMPLETION_RESULT_FENCE_OFFSET 112u
#define HETGPU_IQ1S_COMPLETION_FAULT_DETAIL_OFFSET 120u

#pragma pack(push, 1)
typedef struct hetgpu_iq1s_completion_v2 {
    uint32_t magic;
    uint16_t abi_version;
    uint16_t completion_bytes;
    uint32_t status;
    uint32_t fault_code;
    uint64_t session_generation;
    uint64_t transaction_id;
    uint64_t program_id;
    uint64_t trace_id;
    uint32_t layer_id;
    uint16_t phase;
    uint16_t role;
    uint16_t cu_id;
    uint16_t expert_id;
    uint16_t lane_mask;
    uint16_t rows_completed;
    uint32_t descriptor_crc32;
    uint32_t command_index;
    uint64_t cycles;
    uint64_t ddr_read_bytes;
    uint64_t ddr_write_bytes;
    uint64_t iq1s_blocks;
    uint32_t grid_passes;
    uint32_t delta_passes;
    uint64_t result_fence;
    uint64_t fault_detail;
} hetgpu_iq1s_completion_v2;
#pragma pack(pop)

_Static_assert(sizeof(hetgpu_iq1s_completion_v2) == HETGPU_IQ1S_COMPLETION_BYTES, "completion ABI size");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, magic) == HETGPU_IQ1S_COMPLETION_MAGIC_OFFSET, "completion.magic ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, abi_version) == HETGPU_IQ1S_COMPLETION_ABI_VERSION_OFFSET, "completion.abi_version ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, completion_bytes) == HETGPU_IQ1S_COMPLETION_COMPLETION_BYTES_OFFSET, "completion.completion_bytes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, status) == HETGPU_IQ1S_COMPLETION_STATUS_OFFSET, "completion.status ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, fault_code) == HETGPU_IQ1S_COMPLETION_FAULT_CODE_OFFSET, "completion.fault_code ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, session_generation) == HETGPU_IQ1S_COMPLETION_SESSION_GENERATION_OFFSET, "completion.session_generation ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, transaction_id) == HETGPU_IQ1S_COMPLETION_TRANSACTION_ID_OFFSET, "completion.transaction_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, program_id) == HETGPU_IQ1S_COMPLETION_PROGRAM_ID_OFFSET, "completion.program_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, trace_id) == HETGPU_IQ1S_COMPLETION_TRACE_ID_OFFSET, "completion.trace_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, layer_id) == HETGPU_IQ1S_COMPLETION_LAYER_ID_OFFSET, "completion.layer_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, phase) == HETGPU_IQ1S_COMPLETION_PHASE_OFFSET, "completion.phase ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, role) == HETGPU_IQ1S_COMPLETION_ROLE_OFFSET, "completion.role ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, cu_id) == HETGPU_IQ1S_COMPLETION_CU_ID_OFFSET, "completion.cu_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, expert_id) == HETGPU_IQ1S_COMPLETION_EXPERT_ID_OFFSET, "completion.expert_id ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, lane_mask) == HETGPU_IQ1S_COMPLETION_LANE_MASK_OFFSET, "completion.lane_mask ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, rows_completed) == HETGPU_IQ1S_COMPLETION_ROWS_COMPLETED_OFFSET, "completion.rows_completed ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, descriptor_crc32) == HETGPU_IQ1S_COMPLETION_DESCRIPTOR_CRC32_OFFSET, "completion.descriptor_crc32 ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, command_index) == HETGPU_IQ1S_COMPLETION_COMMAND_INDEX_OFFSET, "completion.command_index ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, cycles) == HETGPU_IQ1S_COMPLETION_CYCLES_OFFSET, "completion.cycles ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, ddr_read_bytes) == HETGPU_IQ1S_COMPLETION_DDR_READ_BYTES_OFFSET, "completion.ddr_read_bytes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, ddr_write_bytes) == HETGPU_IQ1S_COMPLETION_DDR_WRITE_BYTES_OFFSET, "completion.ddr_write_bytes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, iq1s_blocks) == HETGPU_IQ1S_COMPLETION_IQ1S_BLOCKS_OFFSET, "completion.iq1s_blocks ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, grid_passes) == HETGPU_IQ1S_COMPLETION_GRID_PASSES_OFFSET, "completion.grid_passes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, delta_passes) == HETGPU_IQ1S_COMPLETION_DELTA_PASSES_OFFSET, "completion.delta_passes ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, result_fence) == HETGPU_IQ1S_COMPLETION_RESULT_FENCE_OFFSET, "completion.result_fence ABI offset");
_Static_assert(offsetof(hetgpu_iq1s_completion_v2, fault_detail) == HETGPU_IQ1S_COMPLETION_FAULT_DETAIL_OFFSET, "completion.fault_detail ABI offset");

// command CRC32 is IEEE CRC32 with bytes 8..11 zeroed.
static inline uint32_t hetgpu_iq1s_command_crc32(const void *record) {
    const uint8_t *bytes = (const uint8_t *)record;
    uint32_t crc = UINT32_C(0xffffffff);
    for (size_t i = 0; i < HETGPU_IQ1S_COMMAND_BYTES; ++i) {
        uint32_t byte = (i >= 8u && i <= 11u) ? 0u : bytes[i];
        crc ^= byte;
        for (unsigned bit = 0; bit < 8u; ++bit)
            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (0u - (crc & 1u)));
    }
    return ~crc;
}

#endif
