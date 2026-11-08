#!/usr/bin/env python3
import errno
from pathlib import Path
import runpy
import struct
from unittest.mock import patch

read = runpy.run_path(str(Path(__file__).with_name("profile-jobd-phases.py")))["read_shared_base"]
with patch("os.pread", side_effect=[OSError(errno.EBUSY, "busy"), struct.pack("<Q", 0x2075d000000)]):
    assert read(7) == 0x2075d000000
for failure, timeout in [(OSError(errno.EBUSY, "busy"), 0),
                         (OSError(errno.EIO, "io"), 5), (b"", 5)]:
    with patch("os.pread", side_effect=[failure]):
        try:
            read(7, timeout)
        except (OSError, ValueError):
            pass
        else:
            raise AssertionError("failed-open base check")
print("base retry: PASS")
