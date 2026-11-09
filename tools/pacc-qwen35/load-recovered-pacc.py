#!/usr/bin/env python3
"""Load one freshly packaged image via the audited vendor driver, restore disk image."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import importlib.util

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).with_name('recover-pacc-ddr.py'))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)

p=argparse.ArgumentParser();p.add_argument('--device',type=int,choices=range(4),required=True);a=p.parse_args()
R=recovery.ROOT;driver=recovery.DRIVER;state=json.loads(recovery.STATE.read_text())
assert os.geteuid()==0 and recovery.boot()==state['boot_id']==recovery.BOOT
assert recovery.live()==(state['base'],state['size'])
assert int(Path('/sys/module/'+recovery.MODULE+'/parameters/shared_ddr_base_override').read_text())==0
assert subprocess.run(['systemctl','is-active','--quiet','pacc-owned-ddr-pin-20261108.service']).returncode==0
recovery.reserved()
assert recovery.sha('/lib/modules/6.18.3+/extra/pacc.ko')=='5152d12fb10bcb03484a5b51fea5254f87d1dddc3f5b4c4c154a9b5c9c166cc2'
assert Path('/sys/module/pacc/srcversion').read_text().strip()=='4A5DBFEE5CE6CA5CCA36263'
manifest=json.loads((R/'firmware/recovery-20261108/manifest.json').read_text())
assert manifest['state']==state
assert manifest['jobd_sha256']=='3a0507b2c7be070d07aebd21ee287543869dd69bdd54c50e9a81f184d70288ce'
item=manifest['images'][a.device];assert item['device']==a.device
image=Path(item['path']).read_bytes();assert len(image)<0x8000000
assert hashlib.sha256(image).hexdigest()==item['sha256']
canonical=Path('/lib/firmware/lanxin/lx500_pacc.bin')
vendor=(R/'firmware/lx500_pacc.vendor-backup.bin').read_bytes()
assert hashlib.sha256(vendor).hexdigest()==recovery.VENDOR
target=f'PACC0001:0{a.device}'
with open('/run/lock/pacc-firmware-update.lock','a') as global_lock,open(R/f'locks/pacc{a.device}.lock','a') as lock:
    fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (driver/target).is_symlink(),'device must already be stopped'
    assert subprocess.run(['fuser',f'/dev/pacc{a.device}'],stderr=subprocess.DEVNULL).returncode==1
    assert recovery.sha(canonical)==recovery.VENDOR
    try:
        canonical.write_bytes(image)
        (driver/'bind').write_text(target)
    finally:
        canonical.write_bytes(vendor)
    assert recovery.sha(canonical)==recovery.VENDOR
    assert (driver/target).is_symlink()
    print(f'PACC{a.device} vendor bind returned; canonical firmware restored; numerical validation still required',flush=True)
