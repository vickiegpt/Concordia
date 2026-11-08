/* One packed head per launch. Arithmetic runs on the PACC CPU, not
 * the host or CUDA warp emulation. State layout is [value-column][key-row]. */
typedef unsigned long u64;
typedef unsigned int u32;
extern float exp2f(float);
static float exp_gate(float x) { return exp2f(x*1.4426950408889634f); }
extern u32 f___zluda_ptx_impl_sreg_tid(unsigned char);
extern u32 f___zluda_ptx_impl_sreg_ntid(unsigned char);
struct Cell { u64 lo, hi; };
static u64 cell(u64 p) { return ((const struct Cell *)p)->lo; }
static void copy_f32(volatile float *dst,const volatile float *src,u64 count) {
    while(count) {
        u64 vl;
        __asm__ volatile("vsetvli %0, %3, e32, m4, ta, ma\n"
                         "vle32.v v8, (%1)\n" "vse32.v v8, (%2)\n"
                         :"=&r"(vl):"r"(src),"r"(dst),"r"(count)
                         :"v8","v9","v10","v11","memory");
        src+=vl;dst+=vl;count-=vl;
    }
}

__attribute__((visibility("default")))
void pacc_gdn_head_v1(u64 in_cell, u64 out_cell, u64 s_cell, u64 n_cell,
                      u64 kda_cell, u64 keep_cell, u64 slots_cell, u64 scale_cell) {
    const float *in = (const float *)cell(in_cell);
    float *out = (float *)cell(out_cell);
    u64 s = cell(s_cell), n = cell(n_cell), kda = cell(kda_cell);
    u64 keep = cell(keep_cell), slots = cell(slots_cell);
    union { u32 u; float f; } scale = { .u = (u32)cell(scale_cell) };
    if (!in || !out || !s || s > 128 || !n || n > 256 || !slots || slots > 16) return;
    const float *q = in, *k = q+n*s, *v = k+n*s, *g = v+n*s;
    const float *b = g+n*(kda?s:1), *initial = b+n;
    float *states = out+n*s;
    u64 tid=f___zluda_ptx_impl_sreg_tid(0), step=f___zluda_ptx_impl_sreg_ntid(0);
    if (!step) return;
    for (u64 col=tid; col<s; col+=step) {
        /* Keep state accesses explicit while qualifying the compiler path. */
        volatile float st[128],local_q[128],local_k[128],local_g[128];
        copy_f32(st,initial+col*s,s);
        for (u64 t=0; t<n; ++t) {
            copy_f32(local_q,q+t*s,s);copy_f32(local_k,k+t*s,s);
            if(kda)copy_f32(local_g,g+t*s,s);
            float kv=0, gate= kda ? 1.0f : exp_gate(g[t]);
            if (kda) {
                for (u64 i=0; i<s; ++i) { st[i] *= exp_gate(local_g[i]); kv += st[i]*local_k[i]; }
            } else {
                for (u64 i=0; i<s; ++i) kv += st[i]*local_k[i];
                kv *= gate;
            }
            float delta = (v[t*s+col]-kv)*b[t], attn=0;
            for (u64 i=0; i<s; ++i) {
                st[i] = gate*st[i] + local_k[i]*delta;
                attn += st[i]*local_q[i];
            }
            out[t*s+col] = attn*scale.f;
            u64 slot=n-1-t;
            if (keep && slot<slots) {
                copy_f32(states+slot*s*s+col*s,st,s);
            }
        }
        if (!keep) copy_f32(states+col*s,st,s);
    }
}
