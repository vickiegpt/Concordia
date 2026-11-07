/* Host-only fault injection. Never link this file into firmware. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern int hetgpu_q6k_q8_1_batch_xm(const uint8_t *, const uint8_t *, float *,
                                  size_t, size_t, size_t, size_t, size_t, size_t);
static float injected;
static int leaf_error, calls;
int xsfmm_native_bf16(const uint16_t *a, const uint16_t *b, float *c,
                     size_t m, size_t n, size_t k) {
    (void)a; (void)b; (void)k;
    ++calls;
    for (size_t i = 0; i < m*n; ++i) c[i] = injected;
    return leaf_error;
}
int main(void) {
    uint8_t x[210] = {0}, y[288] = {0};
    const float faults[] = {NAN, INFINITY, -INFINITY, 0.5f, -0.5f, 65537, -65537};
    float dst;
    for (size_t i = 0; i < sizeof(faults)/sizeof(faults[0]); ++i) {
        injected = faults[i]; dst = 123;
        if (!hetgpu_q6k_q8_1_batch_xm(x, y, &dst, 1, 1, 1, 210, 288, 1) || dst != 123)
            return 1;
    }
    injected = 0; leaf_error = -1; dst = 123;
    if (!hetgpu_q6k_q8_1_batch_xm(x, y, &dst, 1, 1, 1, 210, 288, 1) || dst != 123)
        return 2;
    leaf_error = 0; calls = 0;
    if (!hetgpu_q6k_q8_1_batch_xm(x, y, &dst, 1, 9, 1, 210, 288, 1) || calls)
        return 3;
    if (hetgpu_q6k_q8_1_batch_xm(x, y, &dst, 1, 1, 1, 210, 288, 1) || dst != 0)
        return 4;
    puts("PASS XM rejection: seven invalid values, leaf error, invalid shape, valid zero");
    return 0;
}
