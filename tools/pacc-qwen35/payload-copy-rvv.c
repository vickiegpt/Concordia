#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* Only for bounded, non-overlapping shared-DDR payload -> ordinary RAM copies.
 * The caller retains ownership/coherence fences. Never use for MMIO/control. */
void pacc_payload_copy_rvv(void *restrict destination, const void *restrict source, size_t length) {
    uint8_t *dst = destination;
    const uint8_t *src = source;
    if ((((uintptr_t)dst | (uintptr_t)src) & 7) == 0) {
        while (length >= 8) {
            size_t vl = __riscv_vsetvl_e64m4(length/8);
            vuint64m4_t data = __riscv_vle64_v_u64m4((const uint64_t *)src, vl);
            __riscv_vse64_v_u64m4((uint64_t *)dst, data, vl);
            dst += vl*8; src += vl*8; length -= vl*8;
        }
    }
    while (length) {
        size_t vl = __riscv_vsetvl_e8m4(length);
        vuint8m4_t data = __riscv_vle8_v_u8m4(src, vl);
        __riscv_vse8_v_u8m4(dst, data, vl);
        dst += vl; src += vl; length -= vl;
    }
}
