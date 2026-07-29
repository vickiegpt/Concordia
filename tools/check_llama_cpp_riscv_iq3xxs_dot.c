#include "ggml.h"
#include "ggml-quants.h"
#include "quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t rng_state = 1;

static uint32_t next_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

int main(void) {
    const int n = 6144;
    const int nb = n / QK_K;
    block_iq3_xxs * iq3 = malloc(ggml_row_size(GGML_TYPE_IQ3_XXS, n));
    block_q8_K * q8 = malloc(ggml_row_size(GGML_TYPE_Q8_K, n));
    if (iq3 == NULL || q8 == NULL) {
        return 2;
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int trial = 0; trial < 1000; ++trial) {
        for (int block = 0; block < nb; ++block) {
            iq3[block].d = ggml_fp32_to_fp16(0.001f * (float) (1 + trial % 9));
            for (size_t i = 0; i < sizeof(iq3[block].qs); ++i) {
                iq3[block].qs[i] = (uint8_t) next_u32();
            }
            q8[block].d = 0.002f * (float) (1 + trial % 7);
            for (int i = 0; i < QK_K; ++i) {
                q8[block].qs[i] = (int8_t) next_u32();
            }
            for (int group = 0; group < QK_K / 16; ++group) {
                int sum = 0;
                for (int i = 0; i < 16; ++i) {
                    sum += q8[block].qs[group * 16 + i];
                }
                q8[block].bsums[group] = sum;
            }
        }

        float optimized = 0.0f;
        float generic = 0.0f;
        ggml_vec_dot_iq3_xxs_q8_K(n, &optimized, 0, iq3, 0, q8, 0, 1);
        ggml_vec_dot_iq3_xxs_q8_K_generic(n, &generic, 0, iq3, 0, q8, 0, 1);
        const float abs_error = fabsf(optimized - generic);
        const float rel_error = abs_error / fmaxf(fabsf(generic), 1.0e-6f);
        max_abs = fmaxf(max_abs, abs_error);
        max_rel = fmaxf(max_rel, rel_error);
        if (!isfinite(optimized) || abs_error > 1.0e-3f + 1.0e-5f * fabsf(generic)) {
            fprintf(stderr, "trial=%d optimized=%g generic=%g abs=%g rel=%g\n",
                    trial, optimized, generic, abs_error, rel_error);
            return 1;
        }
    }

    printf("iq3_xxs trials=1000 max_abs=%g max_rel=%g\n", max_abs, max_rel);
    free(q8);
    free(iq3);
    return 0;
}
