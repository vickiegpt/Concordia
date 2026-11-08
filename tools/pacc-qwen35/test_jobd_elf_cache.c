#define main jobd_program_main
#include "hetgpu_pacc_jobd.c"
#undef main
#include <assert.h>
#include <elf.h>

int main(void) {
    uint8_t image[4096] = {0};
    Elf64_Ehdr *eh = (Elf64_Ehdr *)image;
    memcpy(eh->e_ident, ELFMAG, SELFMAG);
    eh->e_ident[EI_CLASS] = ELFCLASS64;
    eh->e_ident[EI_DATA] = ELFDATA2LSB;
    eh->e_ident[EI_VERSION] = EV_CURRENT;
    eh->e_type = ET_DYN;
    eh->e_machine = EM_RISCV;
    eh->e_phoff = sizeof(*eh);
    eh->e_phentsize = sizeof(Elf64_Phdr);
    eh->e_phnum = 1;
    eh->e_shoff = 0x300;
    eh->e_shentsize = sizeof(Elf64_Shdr);
    eh->e_shnum = 3;
    Elf64_Phdr *ph = (Elf64_Phdr *)(image + eh->e_phoff);
    ph->p_type = PT_LOAD;
    ph->p_flags = PF_R | PF_W | PF_X;
    ph->p_vaddr = 0x1000;
    ph->p_filesz = sizeof(image);
    ph->p_memsz = 8192;
    Elf64_Shdr *sh = (Elf64_Shdr *)(image + eh->e_shoff);
    sh[1].sh_type = SHT_SYMTAB;
    sh[1].sh_offset = 0x500;
    sh[1].sh_size = 2 * sizeof(Elf64_Sym);
    sh[1].sh_entsize = sizeof(Elf64_Sym);
    sh[1].sh_link = 2;
    sh[2].sh_type = SHT_STRTAB;
    sh[2].sh_offset = 0x600;
    sh[2].sh_size = 12;
    memcpy(image + 0x600, "\0cache_test\0", 12);
    Elf64_Sym *sym = (Elf64_Sym *)(image + 0x500);
    sym[1].st_name = 1;
    sym[1].st_shndx = 1;
    sym[1].st_value = 0x1700;
    image[0x700] = 42;
    struct LoadedKernelImage first = {0}, next = {0};
    assert(load_kernel_elf_cached(image, sizeof(image), "cache_test", &first) == 0);
    for (unsigned i = 0; i < 10000; ++i) {
        ((uint8_t *)first.mapping)[0x700] = 99;
        ((uint8_t *)first.mapping)[4096] = 99;
        assert(load_kernel_elf_cached(image, sizeof(image), "cache_test", &next) == 0);
        assert(next.mapping == first.mapping && next.fn == first.fn);
        assert(((uint8_t *)next.mapping)[0x700] == 42);
        assert(((uint8_t *)next.mapping)[4096] == 0);
    }
    assert(g_direct_elf_mapping_count == 1 && g_direct_elf_mapping_bytes == 8192);
    assert(g_direct_elf_cache_hits == 10000);
    for (unsigned i = 1; i < 64; ++i) {
        image[0x700] = i;
        image[0x701] = i;
        assert(load_kernel_elf_cached(image, sizeof(image), "cache_test", &next) == 0);
        assert(((uint8_t *)next.mapping)[0x700] == i);
    }
    image[0x701] = 255;
    assert(load_kernel_elf_cached(image, sizeof(image), "cache_test", &next) == -1);
    assert(g_kernel_load_error == 0x134 && g_direct_elf_mapping_count == 64);
    puts("PASS ELF cache: 10000 hits, mutable/BSS restoration, byte identity, bounded exhaustion");
    return 0;
}
