#pragma once
#include "ggml-quants.h"
#include <thread>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <map>
#include <cstring>

// Independent, CPU-only reference. No PACC calls and no device fallback.
// Decode Q6_K weights with GGML, quantize activations per the CUDA Q8_1 contract,
// and accumulate in FP64 rather than reusing the PACC integer/XM implementation.
struct cpu_q81_head {
    std::vector<float> weights;
    int64_t cached_k=0,cached_m=0;
    bool layer39=false;
    std::map<const void*,std::vector<float>> expert_weights;
    void layer_matrix(ggml_tensor *t) {
        auto *w=t->src[0],*x=t->src[1];
        bool moe=t->op==GGML_OP_MUL_MAT_ID;
        if(!w->buffer || !ggml_backend_buffer_is_host(w->buffer) || x->type!=GGML_TYPE_F32 ||
           t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t) || w->ne[3]!=1 || x->ne[3]!=1 || x->nb[0]!=4)std::abort();
        int64_t k=w->ne[0],m=w->ne[1],columns=t->ne[1]*t->ne[2];
        std::vector<unsigned char> raw(ggml_nbytes(x)),ids;
        ggml_backend_tensor_get(x,raw.data(),0,raw.size());
        if(moe){ids.resize(ggml_nbytes(t->src[2]));ggml_backend_tensor_get(t->src[2],ids.data(),0,ids.size());}
        std::vector<const float*> matrices(columns);
        std::vector<float> q(columns*k),out(columns*m);
        for(int64_t col=0;col<columns;++col) {
            int64_t slot=col%t->ne[1],token=col/t->ne[1];int32_t expert=0;
            if(moe)memcpy(&expert,ids.data()+slot*t->src[2]->nb[0]+token*t->src[2]->nb[1],4);
            if(expert<0 || expert>=w->ne[2])std::abort();
            auto *data=static_cast<const char*>(w->data)+expert*w->nb[2];
            auto &decoded=expert_weights[data];
            if(decoded.empty()) {
                decoded.resize(m*k);
                for(int64_t row=0;row<m;++row) {
                    auto *packed=data+row*w->nb[1];auto *dst=decoded.data()+row*k;
                    switch(w->type) {
                        case GGML_TYPE_Q4_K:dequantize_row_q4_K(reinterpret_cast<const block_q4_K*>(packed),dst,k);break;
                        case GGML_TYPE_Q5_K:dequantize_row_q5_K(reinterpret_cast<const block_q5_K*>(packed),dst,k);break;
                        case GGML_TYPE_Q6_K:dequantize_row_q6_K(reinterpret_cast<const block_q6_K*>(packed),dst,k);break;
                        case GGML_TYPE_Q8_0:dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(packed),dst,k);break;
                        default:std::abort();
                    }
                }
            }
            matrices[col]=decoded.data();
            const float *a=reinterpret_cast<const float*>(raw.data()+(moe?slot%x->ne[1]:slot)*x->nb[1]+token*x->nb[2]);
            for(int64_t block=0;block<k;block+=32) {
                float amax=0;for(int j=0;j<32;++j)amax=std::max(amax,std::abs(a[block+j]));
                float d=amax/127.0f,dh=ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
                for(int j=0;j<32;++j)q[col*k+block+j]=(amax==0?0:std::round(a[block+j]/d))*dh;
            }
        }
        std::vector<std::thread> pool;
        for(int worker=0;worker<8;++worker)pool.emplace_back([&,worker]{
            for(int64_t col=0;col<columns;++col)for(int64_t row=worker;row<m;row+=8) {
                double sum=0;for(int64_t j=0;j<k;++j)sum+=double(matrices[col][row*k+j])*q[col*k+j];
                out[col*m+row]=float(sum);
            }
        });
        for(auto &thread:pool)thread.join();
        ggml_backend_tensor_set(t,out.data(),0,out.size()*4);
    }
    static bool callback(ggml_tensor *t,bool ask,void *opaque) {
        auto &s=*static_cast<cpu_q81_head*>(opaque);
        std::string name=t->name;
        bool head=name=="result_output";
        auto *source=t->src[0];
        bool quantized=source && (source->type==GGML_TYPE_Q4_K || source->type==GGML_TYPE_Q5_K ||
            source->type==GGML_TYPE_Q6_K || source->type==GGML_TYPE_Q8_0);
        bool selected=head || (s.layer39 && name.size()>=3 && name.substr(name.size()-3)=="-39" &&
            (t->op==GGML_OP_MUL_MAT || t->op==GGML_OP_MUL_MAT_ID) && quantized);
        if(ask)return selected;
        if(!selected)return true;
        if(!head){s.layer_matrix(t);return true;}
        auto *w=t->src[0],*x=t->src[1];
        if(t->op!=GGML_OP_MUL_MAT || w->type!=GGML_TYPE_Q6_K || x->type!=GGML_TYPE_F32 ||
           t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
           w->ne[2]!=1 || x->ne[2]!=1 || t->ne[2]!=1 || w->ne[3]!=1 || x->ne[3]!=1 ||
           !w->buffer || !ggml_backend_buffer_is_host(w->buffer))std::abort();
        const int64_t k=w->ne[0],m=w->ne[1],n=x->ne[1];
        if(s.weights.empty()) {
            if(k!=2048 || m!=248320)std::abort();
            s.cached_k=k;s.cached_m=m;s.weights.resize(k*m);
            std::vector<std::thread> pool;
            for(int worker=0;worker<8;++worker)pool.emplace_back([&,worker]{
                for(int64_t row=worker;row<m;row+=8)
                    dequantize_row_q6_K(reinterpret_cast<const block_q6_K*>(static_cast<const char*>(w->data)+row*w->nb[1]),s.weights.data()+row*k,k);
            });
            for(auto &th:pool)th.join();
        }
        if(k!=s.cached_k || m!=s.cached_m)std::abort();
        std::vector<float> a(k*n),q(k*n),out(m*n);
        ggml_backend_tensor_get(x,a.data(),0,a.size()*4);
        for(int64_t col=0;col<n;++col)for(int64_t block=0;block<k;block+=32) {
            float amax=0;
            for(int j=0;j<32;++j)amax=std::max(amax,std::abs(a[col*k+block+j]));
            float d=amax/127.0f;
            float dh=ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
            for(int j=0;j<32;++j)q[col*k+block+j]=(amax==0?0:std::round(a[col*k+block+j]/d))*dh;
        }
        std::vector<std::thread> pool;
        for(int worker=0;worker<8;++worker)pool.emplace_back([&,worker]{
            for(int64_t row=worker;row<m;row+=8)for(int64_t col=0;col<n;++col) {
                double sum=0;
                for(int64_t j=0;j<k;++j)sum+=double(s.weights[row*k+j])*q[col*k+j];
                out[col*m+row]=float(sum);
            }
        });
        for(auto &th:pool)th.join();
        ggml_backend_tensor_set(t,out.data(),0,out.size()*4);
        return true;
    }
};
