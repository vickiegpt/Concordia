#!/usr/bin/env python3
import mmap
import os
import struct

fd = os.open('/dev/hetgpu_sifive_mbox_ddr_coh0', os.O_RDWR | os.O_SYNC)
try:
    for label, offset, length in [('beacon', 0x1f40, 32), ('header', 0x200000, 120), ('pacc-header-diagnostic', 0x1f80, 128), ('descriptor', 0, 32)]:
        page = offset // mmap.PAGESIZE * mmap.PAGESIZE
        delta = offset - page
        with mmap.mmap(fd, delta + length, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=page) as region:
            data = region[delta:delta + length]
        print(label, 'mmap', data.hex())
        if label == 'beacon':
            print('magic=%x version=%d job=%d phase=%x detail=%x seq=%d' %
                  struct.unpack('<QIIIIQ', data))
        if label == 'pacc-header-diagnostic':
            print('words', [hex(x) for x in struct.unpack('<16Q', data)])
        print(label, 'pread', os.pread(fd, length, 0x100000 + offset).hex())
finally:
    os.close(fd)
