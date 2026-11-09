#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h"
#include "json.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstring>
using nlohmann::json;
int main(int argc,char **argv) {
    if(argc!=5)return 2;
    auto *g=gguf_init_from_file(argv[1],{true,nullptr});
    if(!g)throw std::runtime_error("GGUF open failed");
    int64_t id=gguf_find_tensor(g,"output.weight");
    if(id<0 || gguf_get_tensor_type(g,id)!=GGML_TYPE_Q6_K)throw std::runtime_error("expected Q6_K output");
    constexpr int k=2048;
    const size_t row_bytes=ggml_row_size(GGML_TYPE_Q6_K,k);
    auto offset=gguf_get_data_offset(g)+gguf_get_tensor_offset(g,id);
    std::ifstream weights(argv[1],std::ios::binary),input(argv[2],std::ios::binary),actual(argv[3],std::ios::binary);
    std::vector<float> x(2*k),logits(2*248320),w(k),xk(k);
    input.read(reinterpret_cast<char*>(x.data()),x.size()*4);
    actual.read(reinterpret_cast<char*>(logits.data()),logits.size()*4);
    if(!input || !actual)throw std::runtime_error("snapshot read failed");
    std::vector<unsigned char> packed(row_bytes);
    std::vector<block_q8_K> qk(k/256);
    std::vector<block_q8_1> q1(k/32);
    json rows=json::array();
    for(int s=0;s<2;++s) {
        quantize_row_q8_K_ref(x.data()+s*k,qk.data(),k);
        dequantize_row_q8_K(qk.data(),xk.data(),k);
        for(int block=0;block<k/32;++block) {
            float amax=0;for(int j=0;j<32;++j)amax=std::max(amax,std::abs(x[s*k+block*32+j]));
            float d=amax/127.0f;
            q1[block].d=ggml_fp32_to_fp16(d);
            for(int j=0;j<32;++j)q1[block].qs[j]=amax==0?0:std::round(x[s*k+block*32+j]/d);
        }
        for(int token:{0,1,13,16,279,2876,4530,5689,14890,8242,15808,3050,54013}) {
            weights.seekg(offset+token*row_bytes);
            weights.read(reinterpret_cast<char*>(packed.data()),row_bytes);
            if(!weights)throw std::runtime_error("weight row read failed");
            dequantize_row_q6_K(reinterpret_cast<const block_q6_K*>(packed.data()),w.data(),k);
            double f=0,a=0,b=0,trunc=0;
            for(int j=0;j<k;++j) {
                f+=double(w[j])*x[s*k+j];
                a+=double(w[j])*ggml_fp16_to_fp32(q1[j/32].d)*q1[j/32].qs[j%32];
                b+=double(w[j])*xk[j];
                int block=j/32;float amax=0;
                for(int l=0;l<32;++l)amax=std::max(amax,std::abs(x[s*k+block*32+l]));
                float d=amax/127.0f;uint32_t bits;std::memcpy(&bits,&d,4);
                int exp=int((bits>>23)&255)-112;
                uint16_t h=exp>0?uint16_t((exp<<10)|((bits&0x7fffff)>>13)):0;
                trunc+=double(w[j])*ggml_fp16_to_fp32(h)*q1[block].qs[j%32];
            }
            rows.push_back({{"sequence",s},{"token",token},{"f32",f},{"q8_1",a},{"q8_1_truncated_scale",trunc},{"q8_K",b},{"observed",logits[s*248320+token]}});
        }
    }
    std::ofstream(argv[4])<<rows.dump(2)<<"\n";
    gguf_free(g);
}
