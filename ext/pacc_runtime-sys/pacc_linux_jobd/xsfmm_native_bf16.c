#include <stddef.h>
#include <stdint.h>

#ifndef HETGPU_PACC_XSFMM_PROBE_STOP_STAGE
#define HETGPU_PACC_XSFMM_PROBE_STOP_STAGE 0
#endif

#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE < 0 || HETGPU_PACC_XSFMM_PROBE_STOP_STAGE > 9
#error "HETGPU_PACC_XSFMM_PROBE_STOP_STAGE must be in [0, 9]"
#endif

#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 9
#define XSFMM_STOP_ENTRY "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_ENTRY ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 1
#define XSFMM_STOP_1 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_1 ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 2
#define XSFMM_STOP_2 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_2 ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 3
#define XSFMM_STOP_3 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_3 ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 4
#define XSFMM_STOP_4 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_4 ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 5
#define XSFMM_STOP_5 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_5 ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 6
#define XSFMM_STOP_6 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_6 ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 7
#define XSFMM_STOP_7 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_7 ""
#endif
#if HETGPU_PACC_XSFMM_PROBE_STOP_STAGE == 8
#define XSFMM_STOP_8 "  li a0, 0\n  ret\n"
#else
#define XSFMM_STOP_8 ""
#endif

/* Xsfmm v0.6.6, Xsfmm32a16f BF16 operands with FP32 tile accumulation. */
__asm__(
    ".text\n"
    ".align 2\n"
    ".global xsfmm_native_bf16\n"
    ".type xsfmm_native_bf16,@function\n"
    "xsfmm_native_bf16:\n"
    XSFMM_STOP_ENTRY
    "  csrwi vstart, 0\n"
    XSFMM_STOP_1
    "  .word 0x508772d7\n" /* sf.vsettnt t0, a4, e16alt, w2 */
    XSFMM_STOP_2
    "  .word 0x8416f2d7\n" /* sf.vsettm t0, a3 */
    XSFMM_STOP_3
    "  li t0, 2\n"
    "  .word 0x8422f357\n" /* sf.vsettk t1, t0 */
    XSFMM_STOP_4
    "  .word 0x43e06057\n" /* sf.vtzero.t mt0 */
    XSFMM_STOP_5
    "  mv t3, a5\n"
    "1:\n"
    "  beqz t3, 2f\n"
    "  mv t0, a3\n"
    "  .word 0x8402f357\n" /* sf.vsettn t1, t0 */
    "  vle16.v v8, (a0)\n"
    "  slli t2, a3, 1\n"
    "  add t4, a0, t2\n"
    "  vle16.v v12, (t4)\n"
    "  slli t2, a3, 2\n"
    "  add a0, a0, t2\n"
    "  mv t0, a4\n"
    "  .word 0x8402f357\n" /* sf.vsettn t1, t0 */
    "  vle16.v v16, (a1)\n"
    "  slli t2, a4, 1\n"
    "  add t4, a1, t2\n"
    "  vle16.v v20, (t4)\n"
    "  slli t2, a4, 2\n"
    "  add a1, a1, t2\n"
    XSFMM_STOP_6
    "  .word 0xf2881077\n" /* sf.mm.f.f mt0, v8, v16 */
    XSFMM_STOP_7
    "  addi t3, t3, -2\n"
    "  j 1b\n"
    "2:\n"
    "  li t0, 0\n"
    "  mv t2, a3\n"
    "  slli t4, a4, 2\n"
    "3:\n"
    "  beqz t2, 4f\n"
    "  .word 0x52567027\n" /* sf.vste32 t0, (a2) */
    XSFMM_STOP_8
    "  add a2, a2, t4\n"
    "  addi t0, t0, 1\n"
    "  addi t2, t2, -1\n"
    "  j 3b\n"
    "4:\n"
    "  li a0, 0\n"
    "  ret\n"
    ".size xsfmm_native_bf16, .-xsfmm_native_bf16\n");

extern int xsfmm_native_bf16(const uint16_t *a_km,
                             const uint16_t *b_kn,
                             float *c_mn,
                             size_t m,
                             size_t n,
                             size_t k);
