#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 <vendor-lx500_pacc.bin> <hetgpu_pacc_jobd> <out.bin> [jobs.conf]" >&2
    exit 2
fi

src="$1"
jobd="$2"
out="$3"
jobs_conf="${4:-}"

python3 - "$src" "$jobd" "$out" "$jobs_conf" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
jobd = Path(sys.argv[2])
out = Path(sys.argv[3])
jobs_conf = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None

image = bytearray(src.read_bytes())
jobd_bytes = jobd.read_bytes()
conf_bytes = b""
if jobs_conf:
    conf_bytes = jobs_conf.read_bytes()

MAGIC = b"070701"

def align4(x: int) -> int:
    return (x + 3) & ~3

def parse_newc(buf: bytes, base: int = 0, size: int | None = None):
    limit = len(buf) if size is None else base + size
    pos = buf.find(MAGIC, base, limit)
    if pos < 0:
        raise SystemExit("no newc cpio magic found")
    entries = {}
    while pos + 110 <= limit and buf[pos:pos + 6] == MAGIC:
        hdr = buf[pos:pos + 110]
        fields = [int(hdr[i:i + 8], 16) for i in range(6, 110, 8)]
        mode = fields[1]
        size = fields[6]
        namesize = fields[11]
        name_start = pos + 110
        name_end = name_start + namesize
        name = bytes(buf[name_start:name_end - 1]).decode("utf-8", "replace")
        data = align4(name_end)
        end = align4(data + size)
        entries[name] = {
            "hdr": pos,
            "data": data,
            "end": end,
            "size": size,
            "mode": mode,
        }
        pos = end
        if name == "TRAILER!!!":
            break
    return entries

def patch_payload(entries, label: str, name: str, payload: bytes, pad: int = 0, required: bool = True):
    if name not in entries:
        print(f"skip {label}:{name}: missing")
        return False
    entry = entries[name]
    size = entry["size"]
    if len(payload) > size:
        msg = f"{label}:{name}: payload {len(payload)} exceeds existing size {size}"
        if required:
            raise SystemExit(msg)
        print(f"skip {msg}")
        return False
    start = entry["data"]
    image[start:start + size] = payload + bytes([pad]) * (size - len(payload))
    print(f"patched {label}:{name}: {len(payload)} bytes into {size}-byte slot")
    return True

rcs = b"""#!/bin/sh
exec /home/root/pacc_skl_test --strict-job-id-only --config /etc/skel/.bashrc </dev/console >/dev/console 2>&1
for i in /etc/rcS.d/S??* /etc/rc5.d/S??*;do
[ ! -f "$i" ]&&continue
case "$i" in
*.sh)(trap - INT QUIT TSTP;set start;. $i);;
*)$i start;;
esac
done
"""

default_conf = b"""gemm 2 2 2 0x20000800 0x20000840 0x20002800 2 2 2 0 0 1
softmax 0x20000900 0x20002900 1 4 4
rmsnorm 0x20000a00 0x20000a40 0x20002a00 1 4 0.00001
"""

outer = parse_newc(image)
patched = 0
patched += patch_payload(outer, "outer", "bin/busybox.nosuid", jobd_bytes, 0, required=False)
patched += patch_payload(outer, "outer", "home/root/pacc_skl_test", jobd_bytes, 0)
patched += patch_payload(outer, "outer", "etc/init.d/rcS", rcs, ord("\n"))
patched += patch_payload(outer, "outer", "etc/skel/.bashrc", conf_bytes or default_conf, ord("\n"))

inner_name = "core-image-minimal-qemuriscv64.cpio"
if inner_name in outer:
    inner = outer[inner_name]
    inner_entries = parse_newc(image, inner["data"], inner["size"])
    patched += patch_payload(inner_entries, "inner", "bin/busybox.nosuid", jobd_bytes, 0, required=False)
    patched += patch_payload(inner_entries, "inner", "home/root/pacc_skl_test", jobd_bytes, 0)
    patched += patch_payload(inner_entries, "inner", "etc/init.d/rcS", rcs, ord("\n"))
    patched += patch_payload(inner_entries, "inner", "etc/skel/.bashrc", conf_bytes or default_conf, ord("\n"))

if patched == 0:
    raise SystemExit("no payloads patched")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(image)
print(f"wrote {out} ({len(image)} bytes)")
PY
