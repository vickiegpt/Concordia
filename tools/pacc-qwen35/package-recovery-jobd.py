#!/usr/bin/env python3
"""Repackage the already qualified v54 binary for a newly owned allocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path('/srv/lanxin/qwen35-a3b')
state=json.loads((ROOT/'logs/recovery-20261108/state.json').read_text())
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==state['boot_id']
out=ROOT/'firmware/recovery-20261108';out.mkdir(exist_ok=False)
binary=ROOT/'firmware/hetgpu_pacc_jobd_v54_owned_slots'
vendor=ROOT/'firmware/lx500_pacc.vendor-backup.bin'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(binary)=='3a0507b2c7be070d07aebd21ee287543869dd69bdd54c50e9a81f184d70288ce'
assert sha(vendor)==state['vendor_sha256']
def package(device,base,stem):
    env={k:v for k,v in os.environ.items() if not k.startswith(('PACC_JOBD_','HETGPU_'))}
    env.update(PACC_JOBD_PACC_ID=str(device),PACC_JOBD_KERNEL_THREADS='1',PACC_JOBD_BEACON='1',
               PACC_JOBD_FORCE_DEVMEM='1',PACC_JOBD_STATUS_MMAP_FALLBACK='0',
               PACC_JOBD_EARLY_DEVMEM_MARKER='1',PACC_JOBD_SHARED_DDR_BASE=hex(base),
               PACC_JOBD_SHARED_DDR_SIZE=hex(state['size']),PACC_JOBD_SHARED_DDR_PACC_BASE='')
    raw=out/(stem+'-raw.bin');image=out/(stem+'.bin')
    subprocess.run(['bash',str(ROOT/'pacc-runtime-head/ext/pacc_runtime-sys/patch_lx500_pacc_inplace_jobd.sh'),
                    str(vendor),str(binary),str(raw)],env=env,check=True)
    subprocess.run(['python3',str(ROOT/'bin/patch-xm-isa-alias.py'),str(raw),str(image),'--sha256',sha(raw)],check=True)
    return dict(device=device,path=str(image),sha256=sha(image))
control=package(0,0x2075d000000,'old-boot-control-do-not-load')
assert control['sha256']=='aeae60a8842ad62c6cee316227b9d0ad840a14c4c2b59b58532e5567aefc4323','old package reproduction differs'
images=[package(i,state['base'],f'pacc{i}-owned') for i in range(4)]
report=dict(state=state,jobd_sha256=sha(binary),control_reproduction=control,images=images)
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
