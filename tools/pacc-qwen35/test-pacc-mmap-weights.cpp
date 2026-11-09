#include "pacc-mmap-weights.h"
#include "ggml-backend-impl.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
static bool padding = false;
static const char * name(ggml_backend_buffer_type_t) { return "CUDA_TEST"; }
static size_t alignment(ggml_backend_buffer_type_t) { return 128; }
static size_t allocation(ggml_backend_buffer_type_t, const ggml_tensor * t) { return ggml_nbytes(t) + (padding ? 128 : 0); }
int main(int argc, char **argv) {
    assert(argc == 2);
    rlimit core{0,0}; assert(setrlimit(RLIMIT_CORE,&core)==0);
    assert(dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL));
    setenv("HETGPU_SIFIVE_MMAP_WEIGHTS","1",1);
    setenv("HETGPU_SIFIVE_ALLOW_HOST_DEVICE_MEM","1",1);
    setenv("HETGPU_SIFIVE_REAL_DEVICE_MEM","0",1);
    setenv("HETGPU_SIFIVE_SHARED_DEVICE_MEM","0",1);
    ggml_backend_buffer_type buft{{name,nullptr,alignment,nullptr,allocation,nullptr},nullptr,nullptr};
    auto * ctx = ggml_init({4*ggml_tensor_overhead(),nullptr,true}); assert(ctx);
    auto * t = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,32);
    assert(pacc_mmap_weights_enabled(&buft,ctx));
    padding=true; bool rejected=false;
    try { pacc_mmap_weights_enabled(&buft,ctx); } catch(const std::runtime_error &) { rejected=true; }
    assert(rejected); padding=false;
    setenv("HETGPU_SIFIVE_REAL_DEVICE_MEM","1",1);rejected=false;
    try { pacc_mmap_weights_enabled(&buft,ctx); } catch(const std::runtime_error &) { rejected=true; }
    assert(rejected);setenv("HETGPU_SIFIVE_REAL_DEVICE_MEM","0",1);
    FILE * f=tmpfile();assert(f);float input[1024];for(int i=0;i<1024;i++)input[i]=i*.5f;
    assert(fwrite(input,sizeof(input),1,f)==1 && fflush(f)==0);
    void * mapped=mmap(nullptr,sizeof(input),PROT_READ,MAP_SHARED,fileno(f),0);assert(mapped!=MAP_FAILED);
    auto * b=pacc_mmap_weights_buffer(&buft,mapped,sizeof(input));assert(b);
    assert(ggml_backend_tensor_alloc(b,t,mapped)==GGML_STATUS_SUCCESS);
    float result[32];ggml_backend_tensor_get(t,result,0,sizeof(result));
    for(int i=0;i<32;i++)assert(result[i]==input[i]);
    pid_t pid=fork();assert(pid>=0);
    if(pid==0){freopen("/dev/null","w",stderr);ggml_backend_tensor_set(t,input,0,sizeof(result));_exit(10);}
    int status;assert(waitpid(pid,&status,0)==pid && WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
    ggml_backend_buffer_free(b);assert(static_cast<float *>(mapped)[31]==input[31]);
    assert(munmap(mapped,sizeof(input))==0);fclose(f);ggml_free(ctx);
    puts("PASS borrowed weights: padding and native-device rejection, immutable writes, reads and lifetime");
}
