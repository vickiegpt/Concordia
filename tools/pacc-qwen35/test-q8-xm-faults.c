/* Host-only fault injector, never linked into firmware. */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern int hetgpu_q8_0_q8_1_xm(const uint8_t *, const uint8_t *, float *,
                              size_t, size_t, size_t, size_t, size_t, size_t);
static float injected;
static int leaf_error;
int xsfmm_native_bf16(const uint16_t *a, const uint16_t *b, float *c,
                     size_t m, size_t n, size_t k) {
    (void)a; (void)b; (void)k;
    for (size_t i=0; i<m*n; ++i) c[i]=injected;
    return leaf_error;
}

int main(void) {
    uint8_t x[34]={0}, y[36]={0};
    uint16_t one=0x3c00;
    memcpy(x,&one,2); memcpy(y,&one,2);
    float out=123;
    const float invalid[]={NAN, INFINITY, -INFINITY, 0.5f, -0.5f, 524289, -524289};
    for (size_t i=0; i<sizeof(invalid)/sizeof(*invalid); ++i) {
        injected=invalid[i];
        assert(hetgpu_q8_0_q8_1_xm(x,y,&out,1,1,1,34,36,1));
        assert(out==123);
    }
    injected=0; leaf_error=1;
    assert(hetgpu_q8_0_q8_1_xm(x,y,&out,1,1,1,34,36,1));
    leaf_error=0;
    assert(!hetgpu_q8_0_q8_1_xm(x,y,&out,1,1,1,34,36,1) && out==0);
    injected=524288;
    assert(!hetgpu_q8_0_q8_1_xm(x,y,&out,1,1,1,34,36,1) && out==524288);
    uint16_t nan=0x7e00;
    memcpy(x,&nan,2);
    assert(hetgpu_q8_0_q8_1_xm(x,y,&out,1,1,1,34,36,1));
    memcpy(x,&one,2); memcpy(y,&nan,2);
    assert(hetgpu_q8_0_q8_1_xm(x,y,&out,1,1,1,34,36,1));
    assert(hetgpu_q8_0_q8_1_xm(x,y,&out,0,1,1,34,36,1));
    assert(hetgpu_q8_0_q8_1_xm(x,y,&out,1,9,1,34,36,1));
    assert(hetgpu_q8_0_q8_1_xm(x,y,&out,1,1,1,33,36,1));
    puts("PASS Q8 XM invalid dots, leaf error, nonfinite scales, bounds, zero and max dot");
    return 0;
}
