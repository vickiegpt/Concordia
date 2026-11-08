// Direct libllama generation, based on llama.cpp/examples/simple/simple.cpp.
// Avoids server startup probes; does not skip any prompt or decode computation.
#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;
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
    try {
        int ngl = std::stoi(argv[2]), count = std::stoi(argv[3]);
        if (ngl < 0 || count < 1 || count > 16) throw std::runtime_error("invalid shape");
        std::string prompt = argv[4], text;
        report = {{"ngl", ngl}, {"prompt", prompt}, {"requested_tokens", count},
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
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        sampler = llama_sampler_init_greedy();
        llama_batch batch = llama_batch_get_one(input.data(), input.size());
        auto started = ggml_time_us();
        double prefill_ms = 0, decode_ms = 0;
        json steps = json::array();
        for (int i = 0; i < count; ++i) {
            auto before = ggml_time_us();
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
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
        double wall = (ggml_time_us() - started) / 1000000.;
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
            if (ok) report["aggregate_output_tps"] = generated.size() / wall;
            result = ok ? 0 : 1;
        }
    } catch (const std::exception &e) { report["error"] = e.what(); }
    FILE *output = fopen(argv[5], "wx");
    if (!output) return 2;
    auto serialized = report.dump(2) + "\n";
    fwrite(serialized.data(), 1, serialized.size(), output);
    fclose(output);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return result;
}
