// Independent sequence state; actual cross-request llama_decode batches.
#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <unistd.h>
using nlohmann::json;

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: MODEL NGL CASE.json RESULT.json REFERENCE.json|- UBATCH\n");
        return 2;
    }
    if (std::filesystem::exists(argv[4])) return 2;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_batch batch = {};
    bool allocated = false;
    int rc = 1;
    json report = {{"all_correct",false},{"aggregate_output_tps",nullptr},
                   {"aggregate_decode_tps",nullptr}};
    try {
        json config, reference;
        std::ifstream(argv[3]) >> config;
        bool reference_only = std::string(argv[5]) == "-";
        if (!reference_only) std::ifstream(argv[5]) >> reference;
        auto prompts = config.at("prompts").get<std::vector<std::string>>();
        int count = config.at("max_tokens"), min_tokens = config.value("min_tokens",count);
        int sequences = prompts.size(), ngl = std::stoi(argv[2]), ubatch = std::stoi(argv[6]);
        if (sequences < 1 || (size_t)sequences > llama_max_parallel_sequences() || count < 2 || count > 1024 ||
            min_tokens < 1 || min_tokens > count || ubatch < 1 || ubatch > 8 || ngl < 0)
            throw std::runtime_error("invalid benchmark dimensions");
        bool diagnostic=config.value("diagnostic",false);
        if ((std::filesystem::path(argv[1]).filename().string().find("DIAGNOSTIC-") == 0)!=diagnostic)
            throw std::runtime_error("diagnostic model must be explicitly marked");
        report.update({{"model",argv[1]},{"ngl",ngl},{"config",config},
                       {"sequences",sequences},{"ubatch",ubatch},{"reference_only",reference_only},{"diagnostic",diagnostic}});
        if (!reference_only && (reference.at("ngl") != 0 || reference.at("config") != config ||
                               !reference.value("reference_complete",false)))
            throw std::runtime_error("invalid CPU reference");
        ggml_backend_load_all();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
        ggml_backend_dev_t none[] = {nullptr}; if (!ngl) mp.devices = none;
        model = llama_model_load_from_file(argv[1],mp);
        if (!model) throw std::runtime_error("model load failed");
        const auto *vocab = llama_model_get_vocab(model);
        std::vector<std::vector<llama_token>> input(sequences), generated(sequences);
        std::vector<std::string> text(sequences);
        std::vector<bool> done(sequences,false);
        int context = 0;
        for (int s=0;s<sequences;s++) {
            int n = -llama_tokenize(vocab,prompts[s].data(),prompts[s].size(),nullptr,0,true,true);
            if (n <= 0 || n+count > 4096) throw std::runtime_error("prompt exceeds context limit");
            input[s].resize(n);
            if (llama_tokenize(vocab,prompts[s].data(),prompts[s].size(),input[s].data(),n,true,true)!=n)
                throw std::runtime_error("tokenization failed");
            context = std::max(context,n+count+16);
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = ((context+255)/256)*256*sequences;
        cp.n_batch = std::max(256,sequences); cp.n_ubatch = ubatch; cp.n_seq_max = sequences;
        cp.n_threads = cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = cp.op_offload = ngl != 0; cp.no_perf = false;
        ctx = llama_init_from_model(model,cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        batch = llama_batch_init(cp.n_batch,0,1); allocated = true;
        auto append = [&](int s,llama_token token,int pos,bool logits) {
            int i=batch.n_tokens++;
            batch.token[i]=token; batch.pos[i]=pos; batch.n_seq_id[i]=1;
            batch.seq_id[i][0]=s; batch.logits[i]=logits;
        };
        auto sample = [&](int s,int index) {
            float *logits=llama_get_logits_ith(ctx,index);
            if (!logits) throw std::runtime_error("missing logits");
            int best=0, nv=llama_vocab_n_tokens(vocab);
            for (int v=0;v<nv;v++) {
                if (!std::isfinite(logits[v])) throw std::runtime_error("nonfinite logits");
                if (logits[v]>logits[best]) best=v;
            }
            int step=generated[s].size();
            if (llama_vocab_is_eog(vocab,best)) {
                done[s]=true;
                if (step<min_tokens) throw std::runtime_error("EOS before minimum length");
                return;
            }
            if (!reference_only && (step >= (int)reference.at("tokens").at(s).size() ||
                                    reference.at("tokens").at(s).at(step) != best)) {
                report["first_mismatch"]={{"sequence",s},{"step",step},{"actual",best},
                    {"expected",step<(int)reference.at("tokens").at(s).size() ? reference.at("tokens").at(s).at(step) : json(nullptr)}};
                throw std::runtime_error("CPU token mismatch");
            }
            generated[s].push_back(best);
            char piece[1024]; int n=llama_token_to_piece(vocab,best,piece,sizeof(piece),0,true);
            if(n<0) throw std::runtime_error("token piece too long");
            text[s].append(piece,n);
            done[s]=generated[s].size()==(size_t)count;
            fprintf(stderr,"MULTISEQ_TOKEN sequence=%d step=%d token=%d\n",s,step,best);
        };
        if (const char * barrier = std::getenv("QWEN_BARRIER_DIR")) {
            const char * worker = std::getenv("QWEN_WORKER_ID");
            if (!worker || std::string(worker).size()!=1 || worker[0]<'0' || worker[0]>'3')
                throw std::runtime_error("invalid barrier worker");
            auto dir=std::filesystem::path(barrier);
            auto ready=dir/(std::string("worker")+worker+".ready");
            if(std::filesystem::exists(ready))throw std::runtime_error("barrier already used");
            std::ofstream(ready)<<json({{"pid",getpid()},{"ngl",ngl},{"sequences",sequences}}).dump()<<"\n";
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(180);
            while(!std::filesystem::exists(dir/"release")) {
                if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("start barrier timeout");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            report["barrier_synchronized"]=true;
        }
        auto started=ggml_time_us(); report["started_monotonic_us"]=started;
        if(config.value("batched_prefill",false)) {
            size_t longest=0;for(const auto &v:input) longest=std::max(longest,v.size());
            for(size_t pos=0;pos<longest;pos++) {
                batch.n_tokens=0;std::vector<int> active;
                for(int s=0;s<sequences;s++) if(pos<input[s].size()) {
                    append(s,input[s][pos],pos,true);active.push_back(s);
                }
                if(llama_decode(ctx,batch)) throw std::runtime_error("batched prefill failed");
                llama_synchronize(ctx);
                for(int i=0;i<(int)active.size();i++) if(pos+1==input[active[i]].size()) sample(active[i],i);
            }
        } else {
            // Serial prefill is a useful isolation control for recurrent state.
            for(int s=0;s<sequences;s++) {
                for(int pos=0;pos<(int)input[s].size();pos++) {
                    batch.n_tokens=0; append(s,input[s][pos],pos,true);
                    if(llama_decode(ctx,batch)) throw std::runtime_error("prefill failed");
                }
                llama_synchronize(ctx); sample(s,-1);
            }
        }
        auto decode_started=ggml_time_us(); report["decode_started_monotonic_us"]=decode_started;
        json steps=json::array();
        while(!std::all_of(done.begin(),done.end(),[](bool v){return v;})) {
            batch.n_tokens=0; std::vector<int> active;
            for(int s=0;s<sequences;s++) if(!done[s]) {
                append(s,generated[s].back(),input[s].size()+generated[s].size()-1,true);
                active.push_back(s);
            }
            auto before=ggml_time_us();
            if(llama_decode(ctx,batch)) throw std::runtime_error("batched decode failed");
            llama_synchronize(ctx);
            steps.push_back({{"active",active.size()},{"elapsed_us",ggml_time_us()-before}});
            for(int i=0;i<(int)active.size();i++) sample(active[i],i);
            fprintf(stderr,"MULTISEQ_BATCH active=%zu elapsed_us=%lld\n",active.size(),(long long)(ggml_time_us()-before));
            fflush(stderr);
        }
        auto finish=ggml_time_us(); size_t total=0; for(const auto &v:generated) total+=v.size();
        bool correct=reference_only || (reference.at("tokens")==generated && reference.at("input_tokens")==input);
        report.update({{"tokens",generated},{"texts",text},{"input_tokens",input},
            {"finished_monotonic_us",finish},{"output_tokens",total},{"decode_tokens",total-sequences},
            {"steps",steps},{"reference_complete",reference_only && ngl==0},
            {"all_correct",!reference_only && correct},
            {"measurement",diagnostic ? "diagnostic model: no valid application TPS" :
                "real model; independent sequences; excludes model load; output includes prefill"}});
        if(!reference_only && correct && !diagnostic) {
            report["aggregate_output_tps"]=total*1e6/(finish-started);
            report["aggregate_decode_tps"]=(total-sequences)*1e6/(finish-decode_started);
        }
        rc=correct?0:1;
    } catch(const std::exception &e) { report["error"]=e.what(); }
    FILE *out=fopen(argv[4],"wx"); if(!out) return 2;
    auto serialized=report.dump(2)+"\n"; fwrite(serialized.data(),1,serialized.size(),out); fclose(out);
    if(allocated) llama_batch_free(batch);
    if(ctx) llama_free(ctx);
    if(model) llama_model_free(model);
    return rc;
}
