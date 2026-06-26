ENTRY(_start)

SECTIONS
{
  . = 0x30080000;

  .text : ALIGN(4) {
    KEEP(*(.text.start))
    *(.text .text.*)
  }

  .rodata : ALIGN(8) {
    *(.srodata .srodata.*)
    *(.rodata .rodata.*)
  }

  .data : ALIGN(8) {
    *(.sdata .sdata.*)
    *(.data .data.*)
  }

  .bss : ALIGN(8) {
    __bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    __bss_end = .;
  }

  /DISCARD/ : {
    *(.comment)
    *(.note .note.*)
    *(.eh_frame .eh_frame.*)
    *(.riscv.attributes)
  }
}
