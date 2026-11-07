#define _GNU_SOURCE
#include "xm-config-probe.h"
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/klog.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

/* Each config snapshot occurs inside this leaf, before any C can alter vtype. */
__asm__(
    ".pushsection .text\n.option push\n.option arch,+v\n.option norvc\n"
    ".balign 4\n.global xm_config_leaf\n.type xm_config_leaf,@function\n"
    "xm_config_leaf:\n"
    "li t0, 1\n sw t0, 0(a0)\n li a4, 1\n"
    "vsetvl t0, a4, a1\n csrr t1, vtype\n sd t1, 8(a0)\n"
    "csrr t1, vl\n sw t1, 16(a0)\n sw t0, 20(a0)\n"
    "li t0, 2\n sw t0, 0(a0)\n li a3, 1\n .word 0x8416f2d7\n"
    "csrr t1, vtype\n sd t1, 24(a0)\n sw t0, 32(a0)\n"
    "li t0, 3\n sw t0, 0(a0)\n li t0, 2\n .word 0x8422f357\n"
    "sw t1, 44(a0)\n csrr t1, vtype\n sw t1, 36(a0)\n"
    "srli t2, t1, 32\n sw t2, 40(a0)\n"
    "li t0, 4\n sw t0, 0(a0)\n .word 0x43e06057\n"
    "li t0, 1\n sw t0, 48(a0)\n"
    "li t0, 5\n sw t0, 0(a0)\n .word 0x43c06057\n"
    "li t0, 6\n sw t0, 0(a0)\n ret\n"
    ".global xm_config_leaf_end\nxm_config_leaf_end:\n"
    ".size xm_config_leaf, .-xm_config_leaf\n.option pop\n.popsection\n");
extern void xm_config_leaf(uint32_t *, unsigned long);
extern char xm_config_leaf_end[];
static volatile uint32_t *result;

static void caught(int sig, siginfo_t *info, void *ctx) {
    (void)info;
    uintptr_t pc = ((ucontext_t *)ctx)->uc_mcontext.__gregs[REG_PC];
    result[1] = sig;
    if (pc >= (uintptr_t)xm_config_leaf && pc+4 <= (uintptr_t)xm_config_leaf_end) {
        result[13] = *(uint32_t *)pc;
        result[14] = pc-(uintptr_t)xm_config_leaf;
    }
    _exit(128+sig);
}

void hetgpu_xm_config_probe(struct XmConfigProbe *out) {
    memset(out, 0, sizeof(*out));
    char text[65536];
    int fd = open("/proc/cpuinfo", O_RDONLY | O_CLOEXEC);
    ssize_t n = fd < 0 ? -1 : read(fd, text, sizeof(text)-1);
    if (fd >= 0) close(fd);
    if (n >= 0) {
        text[n] = 0;
        out->guest[0] = (strstr(text, "xlxmmbase") ? 1u : 0u) |
                        (strstr(text, "xsfmmbase") ? 2u : 0u);
    }
    n = klogctl(3, text, sizeof(text)-1);
    out->guest[1] = n < 0 ? errno : 0;
    if (n >= 0) {
        text[n] = 0;
        out->guest[2] = strstr(text, "mammoth is not properly configured") != NULL;
        out->guest[3] = strstr(text, "Fail to alloc tile states") != NULL;
    }
    const unsigned long configs[] = {0x508, 0x408, 0x210, 0x600};
    for (unsigned mode = 0; mode < 4; ++mode) {
        uint32_t *shared = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (shared == MAP_FAILED) { out->formats[mode][15] = errno; continue; }
        pid_t child = fork();
        if (child == 0) {
            result = shared;
            struct rlimit limit = {0, 0};
            setrlimit(RLIMIT_CORE, &limit);
            struct sigaction action = {0};
            action.sa_sigaction = caught;
            action.sa_flags = SA_SIGINFO;
            sigemptyset(&action.sa_mask);
            sigaction(SIGILL, &action, NULL);
            sigaction(SIGSEGV, &action, NULL);
            sigaction(SIGBUS, &action, NULL);
            cpu_set_t set;
            CPU_ZERO(&set);
            int cpu = sched_getcpu();
            if (cpu < 0) _exit(1);
            CPU_SET(cpu, &set);
            if (sched_setaffinity(0, sizeof(set), &set)) _exit(2);
            xm_config_leaf(shared, configs[mode]);
            _exit(0);
        }
        int status, done = 0;
        if (child > 0) {
            for (unsigned i = 0; i < 1000; ++i) {
                if (waitpid(child, &status, WNOHANG) == child) { done = 1; break; }
                struct timespec delay = {0, 1000000};
                nanosleep(&delay, NULL);
            }
            if (!done) { kill(child, SIGKILL); waitpid(child, &status, WNOHANG); }
        }
        memcpy(out->formats[mode], shared, sizeof(out->formats[mode]));
        if (child < 0 || !done) out->formats[mode][15] = child < 0 ? errno : ETIMEDOUT;
        munmap(shared, 4096);
    }
}
