typedef unsigned long u64;
typedef unsigned int u32;

struct SifiveGemmTask {
    u64 magic;
    u64 task_id;
    u64 a_addr;
    u64 b_addr;
    u64 c_addr;
    u64 m;
    u64 n;
    u64 k;
};

__attribute__((section(".text.start")))
void _start(void) {
    volatile struct SifiveGemmTask *task = (volatile struct SifiveGemmTask *)0x80000000UL;
    volatile u64 *status = (volatile u64 *)0x80000040UL;
    volatile u32 *result = (volatile u32 *)0x80000080UL;

    /*
     * Fixed 2x2 GEMM smoke:
     *   [1 2; 3 4] x [5 6; 7 8] = [19 22; 43 50]
     *
     * The host-side launcher currently validates compile/load/launch plumbing.
     * Once the firmware ABI for task pointers is confirmed, these constants can
     * be replaced by task->a_addr/task->b_addr/task->c_addr.
     */
    const u32 a[4] = {1, 2, 3, 4};
    const u32 b[4] = {5, 6, 7, 8};

    for (u64 row = 0; row < 2; ++row) {
        for (u64 col = 0; col < 2; ++col) {
            u32 acc = 0;
            for (u64 kk = 0; kk < 2; ++kk) {
                acc += a[row * 2 + kk] * b[kk * 2 + col];
            }
            result[row * 2 + col] = acc;
        }
    }

    *status = task->magic ^ task->task_id ^ 0x5041434347454d4dUL;

    for (;;) {
        __asm__ volatile("wfi");
    }
}
