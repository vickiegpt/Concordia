// Direct libllama generation, based on llama.cpp/examples/simple/simple.cpp.
// Avoids server startup probes; does not skip any prompt or decode computation.
#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <map>
#include <regex>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

struct tensor_trace {
    std::string directory, reference, stop_name, error, result_path;
    std::regex filter;
    std::map<std::string, int> occurrences;
    json records = json::array();
    double tolerance = 0.03;
    int stop_occurrence = 0;
    bool stopped = false, mismatch = false;

    static bool callback(ggml_tensor *t, bool ask, void *opaque) {
        auto &self = *static_cast<tensor_trace *>(opaque);
        bool selected = t->type == GGML_TYPE_F32 &&
            std::regex_search(t->name, self.filter) && ggml_nbytes(t) <= 64 * 1024 * 1024;
        if (ask) return selected;
        if (!selected) return true;
        try {
            std::string name = t->name;
            int occurrence = self.occurrences[name]++;
            std::string key = std::regex_replace(name, std::regex("[^A-Za-z0-9_.-]"), "_") +
                "." + std::to_string(occurrence);
            std::vector<unsigned char> raw(ggml_nbytes(t));
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
            std::vector<float> values;
            values.reserve(ggml_nelements(t));
            for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
                for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                            float value;
                            memcpy(&value, raw.data() + i0*t->nb[0] + i1*t->nb[1] +
                                   i2*t->nb[2] + i3*t->nb[3], sizeof(value));
                            values.push_back(value);
                        }
            json record = {{"name", name}, {"occurrence", occurrence}, {"key", key},
                {"op", ggml_op_name(t->op)}, {"ne", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}},
                {"nb", {t->nb[0], t->nb[1], t->nb[2], t->nb[3]}}, {"count", values.size()}};
            record["sources"] = json::array();
            for (auto *src : t->src) if (src) record["sources"].push_back(src->name);
            std::ofstream data(self.directory + "/" + key + ".f32", std::ios::binary);
            data.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
            if (!data) throw std::runtime_error("trace data write failed");
            bool finite = std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
            record["finite"] = finite;
            if (!self.reference.empty()) {
                std::ifstream reference(self.reference + "/" + key + ".f32", std::ios::binary | std::ios::ate);
                if (reference.good()) {
                    if (reference.tellg() != std::streamoff(values.size()*sizeof(float)))
                        throw std::runtime_error("trace reference shape mismatch: " + key);
                    reference.seekg(0);
                    std::vector<float> expected(values.size());
                    reference.read(reinterpret_cast<char *>(expected.data()), expected.size()*sizeof(float));
                    double square_error = 0, square_ref = 0, square_got = 0, dot = 0, max_error = 0;
                    for (size_t i = 0; i < values.size(); ++i) {
                        double delta = double(values[i]) - expected[i];
                        square_error += delta*delta;
                        square_ref += double(expected[i])*expected[i];
                        square_got += double(values[i])*values[i];
                        dot += double(values[i])*expected[i];
                        max_error = std::max(max_error, std::abs(delta));
                    }
                    double relative_rms = std::sqrt(square_error / std::max(square_ref, 1e-30));
                    record["relative_rms"] = relative_rms;
                    record["max_abs"] = max_error;
                    record["cosine"] = dot / std::max(std::sqrt(square_ref*square_got), 1e-30);
                    self.mismatch = !finite || (relative_rms > self.tolerance && max_error > 1e-5);
                    fprintf(stderr, "TRACE %s op=%s rel_rms=%.8g max_abs=%.8g%s\n", key.c_str(),
                        ggml_op_name(t->op), relative_rms, max_error, self.mismatch ? " MISMATCH" : "");
                } else {
                    record["reference_missing"] = true;
                    fprintf(stderr, "TRACE %s reference_missing\n", key.c_str());
                }
            } else fprintf(stderr, "TRACE %s op=%s count=%zu\n", key.c_str(), ggml_op_name(t->op), values.size());
            self.records.push_back(record);
            std::ofstream manifest(self.directory + "/manifest.json");
            manifest << self.records.dump(2) << '\n';
            if (!manifest) throw std::runtime_error("trace manifest write failed");
            fflush(stderr);
            self.stopped = self.mismatch || (name == self.stop_name && occurrence == self.stop_occurrence);
            if (self.stopped) {
                // This GGML revision only breaks the current split on callback false.
                // Exit this dedicated diagnostic worker after synchronized readback.
                manifest.close();
                data.close();
                std::ofstream result(self.result_path);
                result << json({{"trace_only", true}, {"all_correct", false},
                    {"aggregate_output_tps", nullptr}, {"trace_mismatch", self.mismatch},
                    {"trace_records", self.records.size()}, {"last_tensor", name},
                    {"stop_reason", self.mismatch ? "numeric divergence" : "trace boundary"}}).dump(2) << '\n';
                result.close();
                std::exit(3);
            }
            return !self.stopped;
        } catch (const std::exception &e) {
            self.error = e.what();
            self.stopped = true;
            return false;
        }
    }
};

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL NGL OUTPUT_TOKENS PROMPT RESULT_JSON REFERENCE_JSON_OR_DASH\n", argv[0]);
        return 2;
    }
    if (std::ifstream(argv[5]).good()) return 2;
    json report;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *sampler = nullptr;
    int result = 1;
    tensor_trace trace;
    try {
        int ngl = std::stoi(argv[2]), count = std::stoi(argv[3]);
        if (ngl < 0 || count < 1 || count > 16) throw std::runtime_error("invalid shape");
        if (std::filesystem::path(argv[1]).filename().string().find("DIAGNOSTIC-") == 0 &&
            !getenv("QWEN_TRACE_DIR")) throw std::runtime_error("diagnostic model requires trace-only mode");
        std::string prompt = argv[4], text;
        report = {{"ngl", ngl}, {"model", argv[1]}, {"prompt", prompt}, {"requested_tokens", count},
                  {"batch", 256}, {"ubatch", 1}, {"all_correct", false},
                  {"aggregate_output_tps", nullptr}};
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = ngl;
        ggml_backend_dev_t no_devices[] = {nullptr};
        if (ngl == 0) mp.devices = no_devices;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        const auto *vocab = llama_model_get_vocab(model);
        int n_prompt = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0 || n_prompt + count > 256) throw std::runtime_error("invalid prompt length");
        std::vector<llama_token> input(n_prompt), generated;
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), input.size(), true, true) != n_prompt)
            throw std::runtime_error("tokenization failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = 256;
        cp.n_ubatch = cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = cp.op_offload = ngl != 0;
        cp.no_perf = false;
        if (const char *dir = getenv("QWEN_TRACE_DIR")) {
            trace.directory = dir;
            trace.result_path = argv[5];
            if (!std::filesystem::create_directory(trace.directory))
                throw std::runtime_error("trace directory already exists");
            trace.filter = std::regex(getenv("QWEN_TRACE_REGEX") ? getenv("QWEN_TRACE_REGEX") : "(-0$|model.input_embed)");
            trace.stop_name = getenv("QWEN_TRACE_STOP") ? getenv("QWEN_TRACE_STOP") : "l_out-0";
            if (getenv("QWEN_TRACE_REFERENCE")) trace.reference = getenv("QWEN_TRACE_REFERENCE");
            if (getenv("QWEN_TRACE_TOLERANCE")) trace.tolerance = std::stod(getenv("QWEN_TRACE_TOLERANCE"));
            if (getenv("QWEN_TRACE_STOP_OCCURRENCE")) trace.stop_occurrence = std::stoi(getenv("QWEN_TRACE_STOP_OCCURRENCE"));
            cp.cb_eval = tensor_trace::callback;
            cp.cb_eval_user_data = &trace;
            report["trace_only"] = true;
        }
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        sampler = llama_sampler_init_greedy();
        llama_batch batch = llama_batch_get_one(input.data(), input.size());
        auto started = ggml_time_us();
        report["started_monotonic_us"] = started;
        double prefill_ms = 0, decode_ms = 0;
        json steps = json::array();
        for (int i = 0; i < count; ++i) {
            auto before = ggml_time_us();
            if (i == 1) report["decode_started_monotonic_us"] = before;
            int decode_status = llama_decode(ctx, batch);
            if (trace.stopped) throw std::runtime_error(trace.error.empty() ? "trace stopped intentionally" : trace.error);
            if (decode_status != 0) throw std::runtime_error("llama_decode failed");
            llama_synchronize(ctx);
            double elapsed = (ggml_time_us() - before) / 1000.;
            if (i == 0) prefill_ms = elapsed; else decode_ms += elapsed;
            float *logits = llama_get_logits_ith(ctx, -1);
            int nv = llama_vocab_n_tokens(vocab);
            if (!logits) throw std::runtime_error("missing logits");
            for (int j = 0; j < nv; ++j)
                if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite logits");
            std::vector<int> order(nv);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + std::min(8, nv), order.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            json top = json::array();
            for (int j = 0; j < std::min(8, nv); ++j) top.push_back({order[j], logits[order[j]]});
            llama_token token = llama_sampler_sample(sampler, ctx, -1);
            steps.push_back({{"token", token}, {"elapsed_ms", elapsed}, {"top_logits", top}});
            if (llama_vocab_is_eog(vocab, token)) break;
            char piece[1024];
            int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            if (n < 0) throw std::runtime_error("token piece too long");
            generated.push_back(token);
            text.append(piece, n);
            fprintf(stderr, "GENERATION_STEP index=%d token=%d elapsed_ms=%.3f\n", i, token, elapsed);
            fflush(stderr);
            batch = llama_batch_get_one(&generated.back(), 1);
        }
        auto finished = ggml_time_us();
        double wall = (finished - started) / 1000000.;
        report["finished_monotonic_us"] = finished;
        report.update({{"input_tokens", input}, {"generated_tokens", generated},
                       {"text", text}, {"steps", steps}, {"wall_s", wall},
                       {"prefill_ms", prefill_ms}, {"decode_ms", decode_ms},
                       {"measurement", "includes prompt prefill; excludes model load; short correctness smoke"}});
        if (std::string(argv[6]) == "-") {
            report["reference_only"] = true;
            result = generated.size() == (size_t)count ? 0 : 1;
        } else {
            json reference;
            std::ifstream(argv[6]) >> reference;
            bool ok = reference.at("ngl") == 0 && reference.at("requested_tokens") == count &&
                      reference.at("input_tokens") == input && reference.at("generated_tokens") == generated &&
                      reference.at("text") == text && generated.size() == (size_t)count;
            report["all_correct"] = ok;
            if (ok && trace.directory.empty()) report["aggregate_output_tps"] = generated.size() / wall;
            result = ok ? 0 : 1;
        }
    } catch (const std::exception &e) { report["error"] = e.what(); }
    if (!trace.directory.empty()) report.update({{"trace_directory", trace.directory},
        {"trace_mismatch", trace.mismatch}, {"trace_records", trace.records.size()}});
    FILE *output = fopen(argv[5], "wx");
    if (!output) return 2;
    auto serialized = report.dump(2) + "\n";
    fwrite(serialized.data(), 1, serialized.size(), output);
    fclose(output);
    if (sampler) llama_sampler_free(sampler);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    return result;
}
