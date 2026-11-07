#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

extern int xsfmm_native_bf16(const uint16_t *, const uint16_t *, float *,
                            size_t, size_t, size_t);

struct probe_result {
    volatile uint32_t stage, result, vlenb, instruction, offset;
    volatile uint32_t actual, expected, failed_case, passed, calls, elapsed_us, guard;
};
static struct probe_result *child_result;

static void illegal_instruction(int sig, siginfo_t *info, void *context) {
    (void)info;
    uintptr_t pc = ((ucontext_t *)context)->uc_mcontext.__gregs[REG_PC];
    uintptr_t native = (uintptr_t)xsfmm_native_bf16;
    if (pc >= native && pc < native+512) {
        child_result->offset = pc-native;
        child_result->instruction = *(const uint32_t *)pc;
    }
    child_result->result = 0x40000000u | sig;
    _exit(128+sig);
}

/* User mode only. The supervisor does not force MS bits or mask interrupts. */
int hetgpu_probe_xm_userspace(uint32_t result[12]) {
    struct probe_result *shared = mmap(0, 4096, PROT_READ | PROT_WRITE,
                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) return -errno;
    *shared = (struct probe_result){0};
    pid_t child = fork();
    if (child == 0) {
        struct rlimit no_core = {0, 0};
        setrlimit(RLIMIT_CORE, &no_core);
        child_result = shared;
        struct sigaction action = {0};
        action.sa_sigaction = illegal_instruction;
        action.sa_flags = SA_SIGINFO;
        sigemptyset(&action.sa_mask);
        sigaction(SIGILL, &action, 0);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        int cpu = sched_getcpu();
        if (cpu < 0) _exit(2);
        CPU_SET(cpu, &cpus);
        if (sched_setaffinity(0, sizeof(cpus), &cpus)) _exit(3);
        shared->stage = 1;
        unsigned long vl;
        __asm__ volatile("csrr %0, vlenb" : "=r"(vl));
        shared->vlenb = vl;
        shared->stage = 2;
        // No pointers or memory operation in the first custom instruction.
        __asm__ volatile("li a4, 1\n\t.word 0x508772d7" ::: "a4", "t0", "memory");
        shared->stage = 3;
        const unsigned shapes[][3] = {{1,1,2}, {3,5,6}, {32,8,32}, {32,32,128}};
        uint16_t a[128*32], b[128*32];
        float output[32*32+2], expected[32*32];
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (unsigned test = 0; test < 64; ++test) {
            unsigned m = shapes[test%4][0], n = shapes[test%4][1], k = shapes[test%4][2];
            shared->failed_case = test;
            memset(expected, 0, sizeof(expected));
            for (unsigned p = 0; p < k; ++p) {
                for (unsigned i = 0; i < m; ++i) {
                    float x = (int)((p*3+i*7+test)%9)-4;
                    uint32_t bits;
                    memcpy(&bits, &x, 4);
                    a[p*m+i] = bits >> 16;
                    for (unsigned j = 0; j < n; ++j) {
                        float y = (int)((p*5+j*3+test)%7)-3;
                        expected[i*n+j] += x*y;
                    }
                }
                for (unsigned j = 0; j < n; ++j) {
                    float y = (int)((p*5+j*3+test)%7)-3;
                    uint32_t bits;
                    memcpy(&bits, &y, 4);
                    b[p*n+j] = bits >> 16;
                }
            }
            for (unsigned i = 0; i < m*n+2; ++i) output[i] = -1234.0f;
            int rc = xsfmm_native_bf16(a, b, output+1, m, n, k);
            ++shared->calls;
            if (rc || output[0] != -1234.0f || output[m*n+1] != -1234.0f) {
                shared->guard = 1;
                _exit(5);
            }
            for (unsigned i = 0; i < m*n; ++i) {
                if (output[i+1] != expected[i]) {
                    uint32_t actual_bits, expected_bits;
                    memcpy(&actual_bits, &output[i+1], 4);
                    memcpy(&expected_bits, &expected[i], 4);
                    shared->actual = actual_bits;
                    shared->expected = expected_bits;
                    _exit(4);
                }
            }
            ++shared->passed;
            sched_yield();
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        shared->elapsed_us = (end.tv_sec-begin.tv_sec)*1000000 +
                            (end.tv_nsec-begin.tv_nsec)/1000;
        shared->stage = 4;
        shared->result = 1;
        _exit(0);
    }
    if (child < 0) { int error = errno; munmap(shared, 4096); return -error; }
    int status = 0, done = 0;
    for (int i = 0; i < 1000; ++i) {
        pid_t observed = waitpid(child, &status, WNOHANG);
        if (observed == child) { done = 1; break; }
        if (observed < 0 && errno != EINTR) break;
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, 0);
    }
    if (!done) {
        kill(child, SIGKILL);
        // A wedged accelerator may require PACC-only vendor recovery.
        waitpid(child, &status, WNOHANG);
    }
    result[0] = shared->stage;
    result[1] = shared->vlenb;
    result[2] = !done ? 0x80000000u : WIFSIGNALED(status)
        ? 0x40000000u | WTERMSIG(status) : WIFEXITED(status) ? WEXITSTATUS(status) : 0xffffffffu;
    if (shared->result & 0x40000000u) result[2] = shared->result;
    result[3] = shared->instruction;
    result[4] = shared->offset;
    result[5] = shared->actual;
    result[6] = shared->expected;
    result[7] = shared->failed_case;
    result[8] = shared->passed;
    result[9] = shared->calls;
    result[10] = shared->elapsed_us;
    result[11] = shared->guard;
    munmap(shared, 4096);
    return done && WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
}
