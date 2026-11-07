/* Host-only test double. Never link this file into jobd firmware. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
int xsfmm_native_bf16(const uint16_t *a, const uint16_t *b, float *c,
                     size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            float sum = 0;
            for (size_t p = 0; p < k; ++p) {
                uint32_t aa = (uint32_t)a[p*m+i] << 16, bb = (uint32_t)b[p*n+j] << 16;
                float x, y;
                memcpy(&x, &aa, 4); memcpy(&y, &bb, 4);
                sum += x*y;
            }
            c[i*n+j] = sum;
        }
    return 0;
}
