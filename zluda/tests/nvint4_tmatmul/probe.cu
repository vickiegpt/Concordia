#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kDim = 2048;
constexpr size_t kElements = static_cast<size_t>(kDim) * kDim;
constexpr size_t kSourceBytes = kElements / 2;
constexpr size_t kPackedBytes = kElements / 4;
constexpr size_t kInputBytes = kDim * sizeof(int16_t);
constexpr size_t kOutputBytes = kDim * sizeof(int64_t);
constexpr uint32_t kDefaultSeed = 0x4e564934;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess) {                                            \
            std::ostringstream message__;                                       \
            message__ << #call << " failed: " << cudaGetErrorString(error__);   \
            throw std::runtime_error(message__.str());                           \
        }                                                                       \
    } while (0)

__device__ int8_t device_decode_nibble(uint8_t nibble) {
    nibble &= 0x0f;
    return nibble >= 8 ? static_cast<int8_t>(nibble) - 16
                       : static_cast<int8_t>(nibble);
}

__device__ int8_t device_ternarize(int8_t value, uint32_t delta) {
    return value < -static_cast<int32_t>(delta)
               ? -1
               : value > static_cast<int32_t>(delta) ? 1 : 0;
}

}  // namespace

extern "C" __global__ void tmatmul_nvint4_dense(
    const uint8_t* packed_weights,
    const int16_t* input_q8_8,
    int64_t* output_s64,
    uint32_t dim,
    uint32_t delta) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    for (uint32_t row = 0; row < dim; ++row) {
        int64_t sum = 0;
        for (uint32_t col = 0; col < dim; ++col) {
            size_t element = static_cast<size_t>(row) * dim + col;
            uint8_t byte = packed_weights[element / 2];
            uint8_t nibble = (byte >> (4 * (element & 1))) & 0x0f;
            sum += static_cast<int64_t>(
                       device_ternarize(device_decode_nibble(nibble), delta)) *
                   static_cast<int64_t>(input_q8_8[col]);
        }
        output_s64[row] = sum;
    }
}

namespace {

static int8_t decode_nibble(uint8_t nibble) {
    nibble &= 0x0f;
    return nibble >= 8 ? static_cast<int8_t>(nibble) - 16
                       : static_cast<int8_t>(nibble);
}

static int8_t ternarize(int8_t q, uint32_t delta) {
    return q < -static_cast<int32_t>(delta)
               ? -1
               : q > static_cast<int32_t>(delta) ? 1 : 0;
}

static std::vector<int64_t> reference(
    const std::vector<uint8_t>& packed_int4,
    const std::vector<int16_t>& input,
    uint32_t dim,
    uint32_t delta) {
    std::vector<int64_t> output(dim, 0);
    for (uint32_t row = 0; row < dim; ++row) {
        int64_t sum = 0;
        for (uint32_t col = 0; col < dim; ++col) {
            size_t element = static_cast<size_t>(row) * dim + col;
            uint8_t byte = packed_int4[element / 2];
            uint8_t nibble = (byte >> (4 * (element & 1))) & 0x0f;
            sum += static_cast<int64_t>(ternarize(decode_nibble(nibble), delta)) *
                   static_cast<int64_t>(input[col]);
        }
        output[row] = sum;
    }
    return output;
}

static std::vector<uint8_t> pack_ternary(
    const std::vector<uint8_t>& packed_int4,
    uint32_t delta) {
    std::vector<uint8_t> packed(kPackedBytes, 0);
    for (size_t out = 0; out < packed.size(); ++out) {
        uint8_t value = 0;
        for (size_t lane = 0; lane < 4; ++lane) {
            size_t element = out * 4 + lane;
            uint8_t byte = packed_int4[element / 2];
            uint8_t nibble = (byte >> (4 * (element & 1))) & 0x0f;
            int8_t ternary = ternarize(decode_nibble(nibble), delta);
            uint8_t code = ternary < 0 ? 3 : ternary > 0 ? 1 : 0;
            value |= static_cast<uint8_t>(code << (lane * 2));
        }
        packed[out] = value;
    }
    return packed;
}

static void set_nibble(std::vector<uint8_t>& bytes, size_t element, int value) {
    uint8_t nibble = static_cast<uint8_t>(value) & 0x0f;
    uint8_t& slot = bytes[element / 2];
    uint8_t shift = static_cast<uint8_t>(4 * (element & 1));
    slot = static_cast<uint8_t>((slot & ~(0x0fU << shift)) | (nibble << shift));
}

struct CaseData {
    std::string name;
    std::vector<uint8_t> weights;
    std::vector<int16_t> input;
};

static std::vector<int16_t> make_input(int salt) {
    std::vector<int16_t> input(kDim);
    for (uint32_t index = 0; index < kDim; ++index) {
        input[index] =
            static_cast<int16_t>(((index * 37 + salt * 53) % 1021) - 510);
    }
    return input;
}

static std::vector<CaseData> make_cases(uint32_t delta, uint32_t seed) {
    std::vector<CaseData> cases;
    cases.push_back({"zero", std::vector<uint8_t>(kSourceBytes, 0), make_input(0)});

    CaseData identity{
        "identity", std::vector<uint8_t>(kSourceBytes, 0), make_input(1)};
    int identity_value = delta < 7 ? static_cast<int>(delta) + 1 : -8;
    for (uint32_t index = 0; index < kDim; ++index) {
        set_nibble(identity.weights, static_cast<size_t>(index) * kDim + index,
                   identity_value);
    }
    cases.push_back(std::move(identity));

    cases.push_back(
        {"all_positive", std::vector<uint8_t>(kSourceBytes, 0x77), make_input(2)});
    cases.push_back(
        {"all_negative", std::vector<uint8_t>(kSourceBytes, 0x88), make_input(3)});

    CaseData boundaries{
        "threshold_boundaries", std::vector<uint8_t>(kSourceBytes, 0),
        make_input(4)};
    std::vector<int> boundary_values = {
        -8,
        -static_cast<int>(delta),
        0,
        static_cast<int>(delta),
        std::min(7, static_cast<int>(delta) + 1),
    };
    for (size_t element = 0; element < kElements; ++element) {
        set_nibble(boundaries.weights, element,
                   boundary_values[element % boundary_values.size()]);
    }
    cases.push_back(std::move(boundaries));

    CaseData random_case{
        "seed_0x4e564934", std::vector<uint8_t>(kSourceBytes), make_input(5)};
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 255);
    for (uint8_t& byte : random_case.weights) {
        byte = static_cast<uint8_t>(distribution(generator));
    }
    cases.push_back(std::move(random_case));
    return cases;
}

class Sha256 {
  public:
    Sha256() { reset(); }

    void update(const uint8_t* data, size_t size) {
        for (size_t index = 0; index < size; ++index) {
            block_[block_size_++] = data[index];
            bit_count_ += 8;
            if (block_size_ == 64) {
                transform();
                block_size_ = 0;
            }
        }
    }

    template <typename T>
    void update(const std::vector<T>& data) {
        update(reinterpret_cast<const uint8_t*>(data.data()),
               data.size() * sizeof(T));
    }

    std::string finish() {
        uint64_t original_bits = bit_count_;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            while (block_size_ < 64) block_[block_size_++] = 0;
            transform();
            block_size_ = 0;
        }
        while (block_size_ < 56) block_[block_size_++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8) {
            block_[block_size_++] =
                static_cast<uint8_t>((original_bits >> shift) & 0xff);
        }
        transform();
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t word : state_) output << std::setw(8) << word;
        return output.str();
    }

  private:
    static uint32_t rotate_right(uint32_t value, uint32_t bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void reset() {
        state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        block_.fill(0);
        block_size_ = 0;
        bit_count_ = 0;
    }

    void transform() {
        static constexpr std::array<uint32_t, 64> constants = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
            0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
            0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
            0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
            0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
            0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::array<uint32_t, 64> schedule{};
        for (size_t index = 0; index < 16; ++index) {
            size_t offset = index * 4;
            schedule[index] = (static_cast<uint32_t>(block_[offset]) << 24) |
                              (static_cast<uint32_t>(block_[offset + 1]) << 16) |
                              (static_cast<uint32_t>(block_[offset + 2]) << 8) |
                              static_cast<uint32_t>(block_[offset + 3]);
        }
        for (size_t index = 16; index < 64; ++index) {
            uint32_t s0 = rotate_right(schedule[index - 15], 7) ^
                          rotate_right(schedule[index - 15], 18) ^
                          (schedule[index - 15] >> 3);
            uint32_t s1 = rotate_right(schedule[index - 2], 17) ^
                          rotate_right(schedule[index - 2], 19) ^
                          (schedule[index - 2] >> 10);
            schedule[index] =
                schedule[index - 16] + s0 + schedule[index - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t index = 0; index < 64; ++index) {
            uint32_t s1 =
                rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            uint32_t choose = (e & f) ^ (~e & g);
            uint32_t temp1 = h + s1 + choose + constants[index] + schedule[index];
            uint32_t s0 =
                rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> block_{};
    size_t block_size_ = 0;
    uint64_t bit_count_ = 0;
};

struct Options {
    bool converter_only = false;
    bool hardware = false;
    uint32_t delta = 1;
    uint32_t seed = kDefaultSeed;
    bool nondefault_stream = false;
    std::string artifact;
};

static uint32_t parse_u32(const std::string& text) {
    size_t consumed = 0;
    unsigned long value = std::stoul(text, &consumed, 0);
    if (consumed != text.size() || value > UINT32_MAX) {
        throw std::runtime_error("invalid u32: " + text);
    }
    return static_cast<uint32_t>(value);
}

static Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        auto value = [&](const char* name) -> std::string {
            if (++index >= argc) throw std::runtime_error(std::string("missing ") + name);
            return argv[index];
        };
        if (argument == "--converter-only") {
            options.converter_only = true;
        } else if (argument == "--hardware") {
            options.hardware = true;
        } else if (argument == "--delta") {
            options.delta = parse_u32(value("--delta"));
        } else if (argument == "--seed") {
            options.seed = parse_u32(value("--seed"));
        } else if (argument == "--stream") {
            std::string stream = value("--stream");
            if (stream == "nondefault") {
                options.nondefault_stream = true;
            } else if (stream != "default") {
                throw std::runtime_error("--stream must be default or nondefault");
            }
        } else if (argument == "--artifact") {
            options.artifact = value("--artifact");
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (!options.converter_only && !options.hardware) options.hardware = true;
    if (options.converter_only == options.hardware) {
        throw std::runtime_error("select exactly one of --converter-only or --hardware");
    }
    if (options.delta > 7) throw std::runtime_error("--delta must be in [0,7]");
    return options;
}

using ConverterHook = int (*)(
    size_t, uint32_t, uint32_t, cudaStream_t, size_t*, size_t*);

static std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("open " + path + " failed");
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

static uint64_t extract_json_uint(const std::string& json, const std::string& key) {
    std::string token = "\"" + key + "\":";
    size_t position = json.rfind(token);
    if (position == std::string::npos) {
        throw std::runtime_error("route log missing " + key);
    }
    position += token.size();
    while (position < json.size() && std::isspace(json[position])) ++position;
    size_t end = position;
    while (end < json.size() && std::isdigit(json[end])) ++end;
    if (end == position) throw std::runtime_error("route log invalid " + key);
    return std::stoull(json.substr(position, end - position));
}

static size_t count_text(const std::string& text, const std::string& needle) {
    size_t count = 0;
    for (size_t position = 0;
         (position = text.find(needle, position)) != std::string::npos;
         position += needle.size()) {
        ++count;
    }
    return count;
}

struct DeviceBuffers {
    uint8_t* weights = nullptr;
    int16_t* input = nullptr;
    int64_t* output = nullptr;

    DeviceBuffers() {
        CUDA_CHECK(cudaMalloc(&weights, kSourceBytes));
        CUDA_CHECK(cudaMalloc(&input, kInputBytes));
        CUDA_CHECK(cudaMalloc(&output, kOutputBytes));
    }
    ~DeviceBuffers() {
        if (output) cudaFree(output);
        if (input) cudaFree(input);
        if (weights) cudaFree(weights);
    }
};

static void write_converter_artifact(
    const Options& options,
    const std::string& source_sha,
    const std::string& packed_sha,
    const std::vector<CaseData>& cases) {
    if (options.artifact.empty()) return;
    std::ofstream output(options.artifact);
    if (!output) throw std::runtime_error("open artifact failed: " + options.artifact);
    output << "{\n"
           << "  \"route\": \"nvint4_converter_only\",\n"
           << "  \"final_status\": \"pass\",\n"
           << "  \"dim\": " << kDim << ",\n"
           << "  \"delta\": " << options.delta << ",\n"
           << "  \"source_sha256\": \"" << source_sha << "\",\n"
           << "  \"packed_sha256\": \"" << packed_sha << "\",\n"
           << "  \"cases\": [";
    for (size_t index = 0; index < cases.size(); ++index) {
        if (index) output << ", ";
        output << "\"" << cases[index].name << "\"";
    }
    output << "]\n}\n";
}

static int run_converter_only(
    const Options& options,
    const std::vector<CaseData>& cases,
    DeviceBuffers& device,
    cudaStream_t stream) {
    auto hook = reinterpret_cast<ConverterHook>(
        dlsym(RTLD_DEFAULT, "hetgpu_nvint4_convert_for_test"));
    if (!hook) throw std::runtime_error("hetgpu_nvint4_convert_for_test not found");
    setenv("HETGPU_NVINT4_CONVERTER_TEST", "1", 1);

    size_t cached_pointer = 0;
    Sha256 source_hash;
    Sha256 packed_hash;
    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const CaseData& test = cases[case_index];
        CUDA_CHECK(cudaMemcpyAsync(device.weights, test.weights.data(), kSourceBytes,
                                   cudaMemcpyHostToDevice, stream));
        for (uint32_t threshold : std::array<uint32_t, 2>{
                 options.delta, options.delta < 7 ? options.delta + 1
                                                  : options.delta - 1}) {
            size_t scratch = 0;
            size_t bytes = 0;
            int result = hook(reinterpret_cast<size_t>(device.weights), kDim,
                              threshold, stream, &scratch, &bytes);
            if (result != 0) {
                throw std::runtime_error("converter hook failed: rc=" +
                                         std::to_string(result));
            }
            if (bytes != kPackedBytes) {
                throw std::runtime_error("converter returned wrong byte count");
            }
            if (cached_pointer == 0) cached_pointer = scratch;
            if (scratch != cached_pointer) {
                throw std::runtime_error("converter scratch cache was not reused");
            }
            std::vector<uint8_t> actual(kPackedBytes);
            CUDA_CHECK(cudaMemcpyAsync(actual.data(),
                                       reinterpret_cast<void*>(scratch), bytes,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<uint8_t> expected = pack_ternary(test.weights, threshold);
            if (actual != expected) {
                size_t mismatch =
                    std::mismatch(actual.begin(), actual.end(), expected.begin()).first -
                    actual.begin();
                throw std::runtime_error("converter mismatch at byte " +
                                         std::to_string(mismatch));
            }
            if (threshold == options.delta) {
                source_hash.update(test.weights);
                packed_hash.update(expected);
            }
        }
    }
    std::string source_sha = source_hash.finish();
    std::string packed_sha = packed_hash.finish();
    write_converter_artifact(options, source_sha, packed_sha, cases);
    std::cout << "PASS: converter byte-exact\n";
    if (options.nondefault_stream) {
        std::cout << "PASS: non-default stream ordering\n";
    }
    std::cout << "PASS: scratch cache reused\n";
    return 0;
}

static int run_hardware(
    const Options& options,
    const std::vector<CaseData>& cases,
    DeviceBuffers& device,
    cudaStream_t stream) {
    if (options.artifact.empty()) {
        throw std::runtime_error("--artifact is required in hardware mode");
    }
    const char* route_log_path = std::getenv("HETGPU_NVINT4_ROUTE_LOG");
    if (!route_log_path || !*route_log_path) {
        throw std::runtime_error("HETGPU_NVINT4_ROUTE_LOG is required");
    }

    Sha256 source_hash;
    Sha256 packed_hash;
    size_t mismatch_count = 0;
    std::string first_mismatch;
    uint64_t packing_us = 0;
    uint64_t hardware_us = 0;
    auto total_start = std::chrono::steady_clock::now();

    for (const CaseData& test : cases) {
        auto packing_start = std::chrono::steady_clock::now();
        std::vector<uint8_t> packed = pack_ternary(test.weights, options.delta);
        std::vector<int64_t> expected =
            reference(test.weights, test.input, kDim, options.delta);
        packing_us += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - packing_start)
                .count());
        source_hash.update(test.weights);
        packed_hash.update(packed);

        CUDA_CHECK(cudaMemcpyAsync(device.weights, test.weights.data(), kSourceBytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device.input, test.input.data(), kInputBytes,
                                   cudaMemcpyHostToDevice, stream));
        auto hardware_start = std::chrono::steady_clock::now();
        tmatmul_nvint4_dense<<<1, 1, 0, stream>>>(
            device.weights, device.input, device.output, kDim, options.delta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        hardware_us += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - hardware_start)
                .count());

        std::vector<int64_t> actual(kDim);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), device.output, kOutputBytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (size_t index = 0; index < actual.size(); ++index) {
            if (actual[index] != expected[index]) {
                if (first_mismatch.empty()) {
                    std::ostringstream message;
                    message << test.name << "[" << index << "]: expected "
                            << expected[index] << ", got " << actual[index];
                    first_mismatch = message.str();
                }
                ++mismatch_count;
            }
        }
    }

    std::string route_log = read_text(route_log_path);
    size_t route_passes =
        count_text(route_log, "\"route\":\"ptx_nvint4_to_dax_tmatmul\"") == 0
            ? count_text(route_log, "\"route\": \"ptx_nvint4_to_dax_tmatmul\"")
            : count_text(route_log, "\"route\":\"ptx_nvint4_to_dax_tmatmul\"");
    size_t final_passes = count_text(route_log, "\"final_status\":\"pass\"");
    bool fallback_used =
        route_log.find("\"final_status\":\"gpu_fallback\"") != std::string::npos;
    if (route_passes < cases.size() || final_passes < cases.size()) {
        throw std::runtime_error("runtime route log does not contain one pass per case");
    }

    uint64_t total_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - total_start)
            .count());
    bool passed = mismatch_count == 0 && !fallback_used;
    std::ofstream output(options.artifact);
    if (!output) throw std::runtime_error("open artifact failed: " + options.artifact);
    output << "{\n"
           << "  \"route\": \"ptx_nvint4_to_dax_tmatmul\",\n"
           << "  \"final_status\": \"" << (passed ? "pass" : "fail") << "\",\n"
           << "  \"dim\": " << kDim << ",\n"
           << "  \"delta\": " << options.delta << ",\n"
           << "  \"source_bytes\": " << kSourceBytes << ",\n"
           << "  \"packed_bytes\": " << kPackedBytes << ",\n"
           << "  \"input_bytes\": " << kInputBytes << ",\n"
           << "  \"output_bytes\": " << kOutputBytes << ",\n"
           << "  \"source_sha256\": \"" << source_hash.finish() << "\",\n"
           << "  \"packed_sha256\": \"" << packed_hash.finish() << "\",\n"
           << "  \"mismatch_count\": " << mismatch_count << ",\n"
           << "  \"fallback_used\": " << (fallback_used ? "true" : "false")
           << ",\n"
           << "  \"cases\": [";
    for (size_t index = 0; index < cases.size(); ++index) {
        if (index) output << ", ";
        output << "\"" << cases[index].name << "\"";
    }
    output << "],\n"
           << "  \"timing_us\": {\"packing\": " << packing_us
           << ", \"hardware\": " << hardware_us << ", \"total\": " << total_us
           << "},\n"
           << "  \"hardware\": {\n"
           << "    \"instruction_dma_status\": "
           << extract_json_uint(route_log, "instruction_dma_status") << ",\n"
           << "    \"stall_status\": "
           << extract_json_uint(route_log, "stall_status") << ",\n"
           << "    \"wide_dma_status\": "
           << extract_json_uint(route_log, "wide_dma_status") << ",\n"
           << "    \"wide_dma_bytes\": "
           << extract_json_uint(route_log, "wide_dma_bytes") << ",\n"
           << "    \"exec_status\": "
           << extract_json_uint(route_log, "exec_status") << ",\n"
           << "    \"tmatmul_read_beats\": "
           << extract_json_uint(route_log, "tmatmul_read_beats") << "\n"
           << "  }\n"
           << "}\n";
    output.close();

    if (!first_mismatch.empty()) {
        std::cerr << "first mismatch: " << first_mismatch << "\n";
    }
    if (!passed) return 1;
    std::cout << "PASS: hardware results bit-exact across " << cases.size()
              << " bounded cases\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        CUDA_CHECK(cudaSetDevice(0));
        cudaStream_t stream = nullptr;
        if (options.nondefault_stream) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
        DeviceBuffers device;
        std::vector<CaseData> cases = make_cases(options.delta, options.seed);
        int result = options.converter_only
                         ? run_converter_only(options, cases, device, stream)
                         : run_hardware(options, cases, device, stream);
        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
