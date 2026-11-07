#include <stdio.h>
int main(void) {
    unsigned long bytes;
    __asm__ volatile("csrr %0, vlenb" : "=r"(bytes));
    printf("%lu\n", bytes*8);
    return 0;
}
