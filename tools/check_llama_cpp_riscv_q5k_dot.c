#include "ggml.h"
#include "quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t rng_state = 1;

static float next_value(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float) ((rng_state >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}

int main(void) {
    const int n = 6144;
    const size_t q5_bytes = ggml_row_size(GGML_TYPE_Q5_K, n);
    const size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_K, n);
    float * x = malloc((size_t) n * sizeof(*x));
    float * y = malloc((size_t) n * sizeof(*y));
    void * q5 = malloc(q5_bytes);
    void * q8 = malloc(q8_bytes);
    if (x == NULL || y == NULL || q5 == NULL || q8 == NULL) {
        return 2;
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int trial = 0; trial < 1000; ++trial) {
        for (int i = 0; i < n; ++i) {
            x[i] = next_value() * (1.0f + (float) (trial % 7));
            y[i] = next_value() * (1.0f + (float) (trial % 5));
        }
        quantize_row_q5_K(x, q5, n);
        quantize_row_q8_K(y, q8, n);

        float optimized = 0.0f;
        float generic = 0.0f;
        ggml_vec_dot_q5_K_q8_K(n, &optimized, 0, q5, 0, q8, 0, 1);
        ggml_vec_dot_q5_K_q8_K_generic(n, &generic, 0, q5, 0, q8, 0, 1);

        const float abs_error = fabsf(optimized - generic);
        const float rel_error = abs_error / fmaxf(fabsf(generic), 1.0e-6f);
        max_abs = fmaxf(max_abs, abs_error);
        max_rel = fmaxf(max_rel, rel_error);
        if (!isfinite(optimized) || abs_error > 1.0e-3f + 1.0e-5f * fabsf(generic)) {
            fprintf(stderr,
                    "trial=%d optimized=%g generic=%g abs=%g rel=%g\n",
                    trial, optimized, generic, abs_error, rel_error);
            return 1;
        }
    }

    printf("q5_K trials=1000 max_abs=%g max_rel=%g\n", max_abs, max_rel);
    free(q8);
    free(q5);
    free(y);
    free(x);
    return 0;
}
