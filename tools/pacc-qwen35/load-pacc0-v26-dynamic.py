#!/usr/bin/env python3
"""One-boot, per-device experiment through the vendor driver, never raw MMIO."""
import argparse
import hashlib
import fcntl
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
    "v39_xm_profile": "3b722717375edb371ff74d8d00a6c1f2563562565da6f3f46e5fea578f7a460b",
    "v40_xm_vpack": "91cf12388b48ea3f580c59cefced73d3cb803ead77b7e989acc6cfa37fda4476",
    "v41_xm_vcopy": "bfc0ffbe068ec50cec6a1b615ce4f5f3add1114c79c84e8bc0feae01d93ce3d9",
    "v42_xm_fpack": "7b1fc1d306cf5e673a4ca925444f94d68d04b9c32e7a4c1f8521ac8cc5235eda",
    "v43_xm_workers": "0fe139a38a24b08228fbb30d119d8ce87899f3377997d84fc00eacf3f1efdec5",
    "v44_xm_hotpoll": "602bb8cb5b4f911803da24a2684b69dc84e20b6999bbd52381d846f0e0146eae",
    "v45_xm_quiet": "2aaaa8cd60b7c424cec005f29f357eb0be09a1753d5aed8b5e3673c645d92bde",
    "v46_xm_reduce": "f502ba83c312cbe121cac3cc4144e63815537bb21592f5e84604fc036507bf59",
    "v47_o3_rvv": "12b7eecdadb9b3420fbe899527e64a039c7db0c337527e2742a53f5878b7b232",
    "v48_x280_tune": "4a3ea984ac76d381290024d28a4b6b4f7c52859f39e4b62694757cb405c8dccb",
    "v49_vlen1024": "5d2b98105f48811be216adfeba6edc5e5ea546e299473b489dded2d03b3fc0a3",
    "v50_q8_tiles": "bfc733006cf66d4d7f01e46d9634186bb4f4ff345f91e5d705382d99ea0a5a01",
    "v51_completion_owner": "676ac581327b0afa0cbfbbdcf51be5ad24fccdd84d59ac97f5ec65d36d32d626",
    "v52_persistent_diag": "125b1632ef94ea919f1b69dc182e76bcf205be0a0516efb3b87564bfc1d3a95f",
    "v53_bounded_elf": "b6e1b133a71c34820c55250a6cab341a0de9ec4c784980791606061e4adf9129",
    "v54_owned_slots": "aeae60a8842ad62c6cee316227b9d0ad840a14c4c2b59b58532e5567aefc4323",
}
FOUR_IMAGES = [
    IMAGES["v54_owned_slots"],
    "2456ed78bc86512ca5f00e8e6b34205fa1e795e1e70037434dd038f42dfe9433",
    "14cf909ff0505615b7b62cdec05cfbb1b2896da94cf6e5e20f057688d7bc8542",
    "0ddcb56451ccd3fe3a237ebc83337ae547c72bcf1d2767a5cd4456b91923d2aa",
]


def require(test, message):
    if not test:
        raise SystemExit("REFUSED: " + message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(apply, version, device=0):
    require(os.geteuid() == 0, "root required for EFI and client checks")
    require(0 <= device < 4 and (device == 0 or version == "v54_owned_slots"),
            "only ownership-qualified images may run on PACC1/2/3")
    if device != 0:
        require(digest(Path("/lib/modules/6.18.3+/extra/pacc.ko")) ==
                "5152d12fb10bcb03484a5b51fea5254f87d1dddc3f5b4c4c154a9b5c9c166cc2" and
                Path("/sys/module/pacc/srcversion").read_text().strip() == "4A5DBFEE5CE6CA5CCA36263",
                "vendor driver differs from per-device reset/load audit")
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
    image = ROOT / f"firmware/lx500_pacc_qwen35_pacc{device}_{version}.bin"
    backup = ROOT / "firmware/lx500_pacc.vendor-backup.bin"
    canonical = Path("/lib/firmware/lanxin/lx500_pacc.bin")
    expected = FOUR_IMAGES[device] if version == "v54_owned_slots" else IMAGES[version]
    require(digest(image) == expected, "test image changed")
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
    require(image.stat().st_size <= 0x08000000, "image exceeds per-PACC slot")
    # Vendor pacc_probe uses PaccBase + (physical_id << 27), length 128 MiB.
    start = fwbase + device * 0x08000000
    end = start + 0x08000000
    require(end <= fwbase + fwsize, "firmware slot exceeds reserved EFI region")
    reserved = False
    for line in Path("/proc/iomem").read_text().splitlines():
        m = re.match(r"\s*([0-9a-f]+)-([0-9a-f]+) : (.+)", line)
        if not m:
            continue
        lo, hi, kind = int(m[1], 16), int(m[2], 16) + 1, m[3]
        require(not (kind == "System RAM" and lo < end and hi > start),
                "firmware slot overlaps host System RAM")
        reserved |= kind == "Reserved" and lo <= start and hi >= end
    require(reserved, "firmware slot not fully reserved")
    clients = subprocess.run(["fuser", f"/dev/pacc{device}"])
    require(clients.returncode == 1, "device busy or fuser check failed")
    driver = Path("/sys/bus/platform/drivers/pacc")
    target = f"PACC0001:0{device}"
    require((driver / target).is_symlink(), "target PACC not bound")
    print(f"CHECKED pacc={device} firmware_slot=0x{start:x}+0x8000000 owned_ddr=0x{BASE:x}+0x{SIZE:x}", flush=True)
    if not apply:
        print("DRY RUN: no firmware, driver or device writes")
        return
    # Hold a reference so the backing allocation cannot be unloaded mid-load.
    fd = os.open(f"/dev/hetgpu_sifive_mbox_ddr_coh{device}", os.O_RDWR)
    try:
        shutil.copyfile(image, canonical)
        print(f"unbind PACC{device} via vendor SBI reset", flush=True)
        (driver / "unbind").write_text(target)
        print(f"bind PACC{device} via vendor firmware loader", flush=True)
        (driver / "bind").write_text(target)
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
    parser.add_argument("--device", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    require(os.geteuid() == 0, "root required for EFI and client checks")
    # Serialize the shared canonical image and exclude a cooperating worker on the target.
    locks = ROOT / "locks"
    locks.mkdir(exist_ok=True)
    owner = ROOT.stat()
    os.chown(locks, owner.st_uid, owner.st_gid)
    with open("/run/lock/pacc-firmware-update.lock", "a") as global_lock, \
         open(locks / f"pacc{args.device}.lock", "a") as device_lock:
        os.fchown(device_lock.fileno(), owner.st_uid, owner.st_gid)
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.apply, args.version, args.device)
