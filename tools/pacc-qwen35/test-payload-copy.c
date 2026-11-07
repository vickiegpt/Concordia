#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
extern void pacc_payload_copy_rvv(void *, const void *, size_t);
int main(void) {
    const size_t offsets[] = {0, 1, 7, 8, 15, 31};
    const size_t lengths[] = {0, 1, 7, 8, 9, 31, 63, 64, 127, 128, 511, 512, 513, 1023, 1024, 4095, 4096, 8192};
    uint8_t *src = malloc(8256), *dst = malloc(8256), *reference = malloc(8256);
    assert(src && dst && reference);
    for (size_t i = 0; i < 8256; ++i) src[i] = (i*17+i/11) & 255;
    unsigned cases = 0;
    for (unsigned i = 0; i < 6; ++i)
        for (unsigned j = 0; j < 6; ++j)
            for (unsigned k = 0; k < sizeof(lengths)/sizeof(*lengths); ++k) {
                memset(dst, 0xa5, 8256); memset(reference, 0xa5, 8256);
                memcpy(reference+offsets[j], src+offsets[i], lengths[k]);
                pacc_payload_copy_rvv(dst+offsets[j], src+offsets[i], lengths[k]);
                assert(!memcmp(dst, reference, 8256));
                ++cases;
            }
    size_t page = sysconf(_SC_PAGESIZE);
    uint8_t *guard = mmap(NULL, page*2, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert(guard != MAP_FAILED && !mprotect(guard+page, page, PROT_NONE));
    memset(guard, 0x39, page);
    for (size_t n = 0; n <= 513; ++n) {
        pacc_payload_copy_rvv(dst, guard+page-n, n);
        for (size_t i = 0; i < n; ++i) assert(dst[i] == 0x39);
    }
    munmap(guard, page*2); free(src); free(dst); free(reference);
    printf("PASS RVV payload copy: %u alignment/length cases, 514 guard-page tails\n", cases);
}
