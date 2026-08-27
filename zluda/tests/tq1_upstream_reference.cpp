#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> INPUT_MAGIC{'T', 'Q', '1', 'F', 'X', '1', '\0', '\0'};
constexpr std::array<char, 8> OUTPUT_MAGIC{'T', 'Q', '1', 'R', 'F', '1', '\0', '\0'};
constexpr uint32_t FORMAT_VERSION = 1;

template <typename T>
T read_scalar(std::istream & input, const char * name) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error(std::string("failed to read ") + name);
    }
    return value;
}

template <typename T>
void write_scalar(std::ostream & output, const T & value, const char * name) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error(std::string("failed to write ") + name);
    }
}

size_t checked_mul(size_t left, size_t right, const char * name) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::runtime_error(std::string(name) + " overflow");
    }
    return left * right;
}

void read_exact(std::istream & input, void * destination, size_t bytes, const char * name) {
    input.read(reinterpret_cast<char *>(destination), static_cast<std::streamsize>(bytes));
    if (!input || static_cast<size_t>(input.gcount()) != bytes) {
        throw std::runtime_error(std::string("failed to read ") + name);
    }
}

void write_exact(std::ostream & output, const void * source, size_t bytes, const char * name) {
    output.write(reinterpret_cast<const char *>(source), static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error(std::string("failed to write ") + name);
    }
}

int run(const char * input_path, const char * output_path) {
    static_assert(sizeof(block_tq1_0) == 54);
    static_assert(sizeof(block_q8_K::qs) == QK_K);
    static_assert(sizeof(float) == 4);
    ggml_cpu_init();

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input fixture");
    }
    std::array<char, 8> magic{};
    read_exact(input, magic.data(), magic.size(), "input magic");
    if (magic != INPUT_MAGIC) {
        throw std::runtime_error("invalid TQ1 fixture magic");
    }
    const uint32_t version = read_scalar<uint32_t>(input, "format version");
    const uint32_t k = read_scalar<uint32_t>(input, "K");
    const uint32_t rows = read_scalar<uint32_t>(input, "rows");
    const uint32_t tokens = read_scalar<uint32_t>(input, "tokens");
    const uint32_t experts = read_scalar<uint32_t>(input, "experts");
    const uint64_t blocks_bytes = read_scalar<uint64_t>(input, "block bytes");
    const uint64_t activation_count = read_scalar<uint64_t>(input, "activation count");
    if (version != FORMAT_VERSION || k == 0 || k % QK_K != 0 || rows == 0 || tokens == 0 || experts == 0) {
        throw std::runtime_error("invalid TQ1 fixture dimensions or version");
    }

    const size_t blocks_per_row = k / QK_K;
    const size_t matrix_blocks = checked_mul(checked_mul(experts, rows, "matrix rows"), blocks_per_row, "matrix blocks");
    const size_t expected_block_bytes = checked_mul(matrix_blocks, sizeof(block_tq1_0), "matrix bytes");
    const size_t logical_groups = checked_mul(tokens, experts, "logical groups");
    const size_t expected_activations = checked_mul(logical_groups, k, "activation count");
    if (blocks_bytes != expected_block_bytes || activation_count != expected_activations) {
        throw std::runtime_error("TQ1 fixture extent mismatch");
    }

    std::vector<block_tq1_0> weights(matrix_blocks);
    std::vector<float> activations(expected_activations);
    read_exact(input, weights.data(), expected_block_bytes, "TQ1 blocks");
    read_exact(input, activations.data(), checked_mul(activations.size(), sizeof(float), "activation bytes"), "activations");
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("TQ1 fixture has trailing bytes");
    }

    const size_t q8_count = checked_mul(logical_groups, blocks_per_row, "Q8_K block count");
    std::vector<block_q8_K> q8(q8_count);
    for (size_t logical = 0; logical < logical_groups; ++logical) {
        quantize_row_q8_K_ref(activations.data() + logical * k, q8.data() + logical * blocks_per_row, k);
    }

    const size_t output_count = checked_mul(logical_groups, rows, "output count");
    std::vector<float> outputs(output_count);
    for (size_t logical = 0; logical < logical_groups; ++logical) {
        const size_t expert = logical % experts;
        for (size_t row = 0; row < rows; ++row) {
            const block_tq1_0 * weight = weights.data() + (expert * rows + row) * blocks_per_row;
            const block_q8_K * activation = q8.data() + logical * blocks_per_row;
            ggml_vec_dot_tq1_0_q8_K(k, &outputs[logical * rows + row], 0, weight, 0, activation, 0, 1);
        }
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open reference output");
    }
    write_exact(output, OUTPUT_MAGIC.data(), OUTPUT_MAGIC.size(), "output magic");
    write_scalar(output, FORMAT_VERSION, "format version");
    write_scalar(output, k, "K");
    write_scalar(output, rows, "rows");
    write_scalar(output, tokens, "tokens");
    write_scalar(output, experts, "experts");
    write_scalar(output, static_cast<uint64_t>(output_count), "output count");
    write_scalar(output, static_cast<uint64_t>(q8_count), "Q8_K block count");
    write_exact(output, outputs.data(), checked_mul(outputs.size(), sizeof(float), "output bytes"), "outputs");
    for (const block_q8_K & block : q8) {
        write_scalar(output, block.d, "Q8_K scale");
        write_exact(output, block.qs, sizeof(block.qs), "Q8_K quants");
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to flush reference output");
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <fixture.bin> <reference.bin>\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception & error) {
        std::cerr << "TQ1_UPSTREAM_REFERENCE_ERROR: " << error.what() << '\n';
        return 1;
    }
}
