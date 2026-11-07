#!/usr/bin/env python3
"""One-boot PACC0 experiment through the vendor driver, never raw MMIO."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT = Path("/srv/lanxin/qwen35-a3b")
BOOT = "89a6c374-059a-40da-8c4f-c00179fd48d4"
BASE, SIZE = 0x2075d000000, 0x400000
VENDOR = "19cdacf4ddfa8f6861477daa2be79f3e1e52d1a11780c7a209a9419e4a2a83a8"
IMAGE = "e15317aa0e01587b760c2dfbaff8e0ae224bd313bf7fed2c17ed69fc9212e568"
IMAGES = {
    "v26_dynamic": IMAGE,
    "v28_owned": "ad1f231152ce754ec439e5c572f6429e7d3103dfcfc99a82980b5c292519c28d",
    "v29_q6k": "7eb13c20683468a25b2d501fcb98cce3eb4b9bf68042dc10e4e75ecd9cbe6350",
    "v30_phases": "dc5c0408ed90402b0ba01ebbc4870111e7130e17b1e3742087e0d70aaae68ab7",
    "v31_xm_probe": "5a6a2c8bd1f15468b2a2a770073c1ced50e404ea4b0d553df450c1841803c447",
    "v32_rvv_batch": "94dcaa573735caa9d100bbeb48563be9b9f7b081913b7849bf5532742ca79e19",
    "v34_identified": "2c4d16de0a3d3c220f73e11f60b4c747b8499ba083d4cd39fcdf605f49c5ae6f",
    "v35_xm_config": "3f7b2f5b025b54ba2e574f411ea7db579dc8f33e4b3bf5ec7580d2026c059b46",
    "v36_isa_alias": "d6d94b15f91082dc7d786962405be0e286a41ec370f3961d4485bd38fd129252",
    "v37_xm_batch": "5f85c7abc8f74e0571ff0621d36ecb8e25ca679645f3e66d2bc8d7989eac7f24",
    "v38_q6k_xm": "45b7b88ca8447bebfc54dae84c1aed3475fe8a50acc3dff9af32fa93e4ce77fe",
}


def require(test, message):
    if not test:
        raise SystemExit("REFUSED: " + message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(apply, version):
    require(os.geteuid() == 0, "root required for EFI and client checks")
    subprocess.run(["python3", str(ROOT / "bin/pacc-preflight.py")], check=True)
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == BOOT,
            "different boot; dynamic addresses must not be reused")
    module = Path("/sys/module/hetgpu_sifive_mbox_ddr_coh/parameters")
    debug = Path("/sys/kernel/debug/hetgpu_sifive_mbox_ddr_coh")
    require(int((module / "shared_ddr_base_override").read_text(), 0) == 0,
            "helper must own a dynamic allocation")
    require(int((debug / "shared_ddr_base").read_text(), 0) == BASE and
            int((debug / "shared_ddr_size").read_text(), 0) == SIZE,
            "allocation no longer matches the image")
    image = ROOT / f"firmware/lx500_pacc_qwen35_pacc0_{version}.bin"
    backup = ROOT / "firmware/lx500_pacc.vendor-backup.bin"
    canonical = Path("/lib/firmware/lanxin/lx500_pacc.bin")
    require(digest(image) == IMAGES[version], "test image changed")
    require(digest(backup) == VENDOR and digest(canonical) == VENDOR,
            "vendor image or current canonical image changed")
    efidir = Path("/sys/firmware/efi/efivars")
    suffix = "-8be4df61-93ca-11d2-aa0d-00e098032b8c"
    base_data = (efidir / ("PaccBase" + suffix)).read_bytes()
    size_data = (efidir / ("PaccSize" + suffix)).read_bytes()
    require(len(base_data) == len(size_data) == 12, "unexpected EFI layout")
    fwbase = struct.unpack_from("<Q", base_data, 4)[0]
    fwsize = struct.unpack_from("<Q", size_data, 4)[0]
    require(fwbase == 0x20100000000 and fwsize == 0x20000000,
            "vendor fallback layout changed")
    require(image.stat().st_size <= 0x08000000, "image exceeds PACC0 slot")
    end = fwbase + 0x08000000
    reserved = False
    for line in Path("/proc/iomem").read_text().splitlines():
        m = re.match(r"\s*([0-9a-f]+)-([0-9a-f]+) : (.+)", line)
        if not m:
            continue
        lo, hi, kind = int(m[1], 16), int(m[2], 16) + 1, m[3]
        require(not (kind == "System RAM" and lo < end and hi > fwbase),
                "firmware slot overlaps host System RAM")
        reserved |= kind == "Reserved" and lo <= fwbase and hi >= end
    require(reserved, "firmware slot not fully reserved")
    clients = subprocess.run(["fuser", "/dev/pacc0"])
    require(clients.returncode == 1, "device busy or fuser check failed")
    driver = Path("/sys/bus/platform/drivers/pacc")
    require((driver / "PACC0001:00").is_symlink(), "PACC0 not bound")
    print(f"CHECKED firmware_slot=0x{fwbase:x}+0x8000000 owned_ddr=0x{BASE:x}+0x{SIZE:x}", flush=True)
    if not apply:
        print("DRY RUN: no firmware, driver or device writes")
        return
    # Hold a reference so the backing allocation cannot be unloaded mid-load.
    fd = os.open("/dev/hetgpu_sifive_mbox_ddr_coh0", os.O_RDWR)
    try:
        shutil.copyfile(image, canonical)
        print("unbind PACC0 via vendor SBI reset", flush=True)
        (driver / "unbind").write_text("PACC0001:00")
        print("bind PACC0 via vendor firmware loader", flush=True)
        (driver / "bind").write_text("PACC0001:00")
        print("vendor bind returned; running jobd identity NOT yet verified", flush=True)
    finally:
        shutil.copyfile(backup, canonical)
        os.close(fd)
        print("canonical vendor firmware restored", flush=True)
    require(digest(canonical) == VENDOR, "restore verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--version", choices=IMAGES, default="v26_dynamic")
    args = parser.parse_args()
    main(args.apply, args.version)
