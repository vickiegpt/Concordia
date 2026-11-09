// Capacity and physical memory accounting only; no generated tokens or TPS.
#include "llama.h"
#include "json.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <sys/resource.h>
using nlohmann::json;
static json memory() {
    std::ifstream file("/proc/self/smaps_rollup");
    std::string line; json result;
    while (std::getline(file,line)) {
        std::istringstream input(line); std::string key,unit; uint64_t value;
        if (input>>key>>value>>unit && unit=="kB") result[key.substr(0,key.size()-1)]=value;
    }
    return result;
}
int main(int argc,char **argv) {
    if(argc!=4 && argc!=5)return 2;
    if(std::filesystem::exists(argv[3]))return 2;
    json report={{"stage","ngl=99 context allocation only"},{"aggregate_decode_tps",nullptr}};
    llama_model *model=nullptr;llama_context *ctx=nullptr;int rc=1;
    try {
        unsigned seq=std::stoul(argv[2]);
        if(!seq || seq>llama_max_parallel_sequences())throw std::runtime_error("sequence limit");
        rlimit cap={100ULL<<30,100ULL<<30};
        if(setrlimit(RLIMIT_AS,&cap))throw std::runtime_error("address space limit failed");
        ggml_backend_load_all();
        auto mp=llama_model_default_params();mp.n_gpu_layers=99;
        model=llama_model_load_from_file(argv[1],mp);
        if(!model)throw std::runtime_error("model load failed");
        report["after_model_kib"]=memory();
        auto cp=llama_context_default_params();
        cp.n_seq_max=seq;cp.n_ctx=256*seq;cp.n_batch=seq;cp.n_ubatch=8;
        cp.offload_kqv=true;cp.op_offload=true;cp.n_threads=cp.n_threads_batch=8;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx=llama_init_from_model(model,cp);
        if(!ctx)throw std::runtime_error("context allocation failed");
        rusage usage{};getrusage(RUSAGE_SELF,&usage);
        report.update({{"allocation_passed",true},{"sequences",seq},{"context_tokens",cp.n_ctx},
                       {"max_rss_kib",usage.ru_maxrss},{"after_context_kib",memory()},
                       {"max_sequences",llama_max_parallel_sequences()}});
        if(argc==5) {
            auto start=std::chrono::steady_clock::now();
            std::ofstream(std::string(argv[3])+".ready")<<report.dump(2)<<"\n";
            while(!std::filesystem::exists(std::filesystem::path(argv[4])/"release")) {
                if(std::chrono::steady_clock::now()-start>std::chrono::seconds(180))
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
