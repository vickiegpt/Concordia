// CPU allocation only: this is not an accelerator correctness or TPS benchmark.
#include "llama.h"
#include "json.hpp"
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <sys/resource.h>
using nlohmann::json;
int main(int argc,char **argv) {
    if(argc!=4 && argc!=5)return 2;
    if(std::ifstream(argv[3]).good())return 2;
    json report={{"stage","CPU context allocation only"},{"aggregate_decode_tps",nullptr}};
    llama_model *model=nullptr;llama_context *ctx=nullptr;int rc=1;
    try {
        unsigned seq=std::stoul(argv[2]);
        if(!seq || seq>llama_max_parallel_sequences())throw std::runtime_error("sequence limit");
        // Keep a failed capacity probe from exhausting the 251 GiB shared host.
        rlimit cap={80ULL<<30,80ULL<<30};
        if(setrlimit(RLIMIT_AS,&cap))throw std::runtime_error("address space limit failed");
        auto mp=llama_model_default_params();ggml_backend_dev_t none[]={nullptr};
        mp.devices=none;mp.n_gpu_layers=0;
        model=llama_model_load_from_file(argv[1],mp);
        if(!model)throw std::runtime_error("model load failed");
        auto cp=llama_context_default_params();
        cp.n_seq_max=seq;cp.n_ctx=256*seq;cp.n_batch=seq;cp.n_ubatch=8;
        cp.offload_kqv=false;cp.op_offload=false;cp.n_threads=cp.n_threads_batch=8;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx=llama_init_from_model(model,cp);
        if(!ctx)throw std::runtime_error("context allocation failed");
        rusage usage{};getrusage(RUSAGE_SELF,&usage);
        report.update({{"allocation_passed",true},{"sequences",seq},{"context_tokens",cp.n_ctx},
                       {"max_rss_kib",usage.ru_maxrss},{"max_sequences",llama_max_parallel_sequences()}});
        if(argc==5) {
            auto start=std::chrono::steady_clock::now();
            std::ofstream(std::string(argv[3])+".ready")<<report.dump(2)<<"\n";
            while(!std::filesystem::exists(std::filesystem::path(argv[4])/"release")) {
                if(std::chrono::steady_clock::now()-start>std::chrono::seconds(90))
                    throw std::runtime_error("capacity barrier timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            report["held_until_release"]=true;
        }
        rc=0;
    } catch(const std::exception &e){report["error"]=e.what();report["allocation_passed"]=false;}
    if(ctx)llama_free(ctx);if(model)llama_model_free(model);
    std::ofstream(argv[3])<<report.dump(2)<<"\n";
    puts(report.dump(2).c_str());return rc;
}
