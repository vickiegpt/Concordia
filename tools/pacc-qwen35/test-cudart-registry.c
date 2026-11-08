/* Build with -DCUDART_SOURCE='"/path/to/cudart_shim.c"' -ffunction-sections
 * -fdata-sections -Wl,--gc-sections -ldl -lm. No device is touched. */
#include CUDART_SOURCE
#include <assert.h>
int main(void) {
    const char *name="<unknown>";
    g_function_count=2;
    memset(g_functions,0,2*sizeof(g_functions[0]));
    g_functions[0].hostFun=(void *)0x10068;
    g_functions[0].cuFunc=(void *)0x30000;
    strcpy(g_functions[0].name,"topk128");
    g_functions[1].hostFun=(void *)0x100e4;
    strcpy(g_functions[1].name,"topk256");
    assert(lookup_registered_function((void *)0x100e0,&name)==NULL);
    assert(!strcmp(name,"topk256"));
    g_functions[1].cuFunc=(void *)0x40000;
    assert(lookup_registered_function((void *)0x100e0,&name)==(void *)0x40000);
    assert(!strcmp(name,"topk256"));
    puts("PASS deferred kernel identity preserved before and after compilation");
    return 0;
}
