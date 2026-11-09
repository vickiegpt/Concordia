#!/usr/bin/env python3
"""Explicit one-boot recovery; never writes guessed physical RAM or reboots."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time

ROOT=Path('/srv/lanxin/qwen35-a3b')
BOOT='4ca2fa21-0d48-4090-93a5-51206ac1de29'
MODULE='hetgpu_sifive_mbox_ddr_coh'
DRIVER=Path('/sys/bus/platform/drivers/pacc')
STATE=ROOT/'logs/recovery-20261108/state.json'
VENDOR='19cdacf4ddfa8f6861477daa2be79f3e1e52d1a11780c7a209a9419e4a2a83a8'
HELPER='375525cb53213c11a630efe055f56da44f9fbf284ba3a89dd248ae5bec3f6bca'
def require(value,message):
    if not value:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def live():
    debug=Path('/sys/kernel/debug')/MODULE
    return tuple(int((debug/name).read_text(),0) for name in ('shared_ddr_base','shared_ddr_size'))
def reserved():
    efidir=Path('/sys/firmware/efi/efivars');suffix='-8be4df61-93ca-11d2-aa0d-00e098032b8c'
    values=[]
    for name in ('PaccBase','PaccSize'):
        raw=(efidir/(name+suffix)).read_bytes();require(len(raw)==12,'EFI layout')
        values.append(struct.unpack_from('<Q',raw,4)[0])
    require(values==[0x20100000000,0x20000000],'EFI firmware reservation changed')
    start,end=values[0],sum(values);covered=False
    for line in Path('/proc/iomem').read_text().splitlines():
        m=re.match(r'\s*([0-9a-f]+)-([0-9a-f]+) : (.+)',line)
        if not m:continue
        lo,hi,kind=int(m[1],16),int(m[2],16)+1,m[3]
        require(not(kind=='System RAM' and lo<end and hi>start),'firmware overlaps RAM')
        covered|=kind=='Reserved' and lo<=start and hi>=end
    require(covered,'firmware reservation missing')
def hold():
    state=json.loads(STATE.read_text());require(boot()==state['boot_id'],'boot changed')
    fd=os.open('/dev/'+MODULE+'0',os.O_RDWR)
    require(live()==(state['base'],state['size']),'allocation changed')
    def stop(signum,frame):
        if any((DRIVER/f'PACC0001:0{i}').is_symlink() for i in range(4)):
            print('Refusing to release DDR while a PACC is bound',flush=True)
        else:raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (STATE.parent/'pin-ready').write_text(str(os.getpid()))
    print('Holding owned DDR until all PACC devices are unbound',flush=True)
    try:
        while True:signal.pause()
    finally:os.close(fd)
def prepare():
    require(not STATE.exists(),'state already exists; do not allocate twice')
    require(sha('/lib/firmware/lanxin/lx500_pacc.bin')==VENDOR,'vendor image changed')
    require(sha('/lib/modules/6.18.3+/extra/pacc.ko')=='5152d12fb10bcb03484a5b51fea5254f87d1dddc3f5b4c4c154a9b5c9c166cc2','pacc driver changed')
    require(Path('/sys/module/pacc/srcversion').read_text().strip()=='4A5DBFEE5CE6CA5CCA36263','loaded pacc differs')
    require(sha('/lib/modules/6.18.3+/extra/'+MODULE+'.ko')==HELPER,'helper changed')
    require(Path('/sys/module/'+MODULE+'/srcversion').read_text().strip()=='BFD2511F8E2F9F96B934D04','loaded helper differs')
    reserved()
    nodes=[f'/dev/pacc{i}' for i in range(4)]+[f'/dev/{MODULE}{i}' for i in range(4)]
    require(subprocess.run(['fuser',*nodes]).returncode==1,'device clients or fuser failure')
    require(all((DRIVER/f'PACC0001:0{i}').is_symlink() for i in range(4)),'unexpected device binding')
    previous=live();STATE.parent.mkdir(exist_ok=False)
    for i in range(4):
        print('Stopping PACC',i,flush=True)
        (DRIVER/'unbind').write_text(f'PACC0001:0{i}')
    subprocess.run(['modprobe','-r',MODULE],check=True)
    subprocess.run(['modprobe',MODULE,'shared_ddr_base_override=0','shared_ddr_size=4194304',
                    'shared_ddr_dma_sync=1','shared_ddr_memremap_mode=0','local_doorbell_bit=1'],check=True)
    require(int(Path('/sys/module/'+MODULE+'/parameters/shared_ddr_base_override').read_text())==0,'not dynamically owned')
    base,size=live();require(base!=0 and size==0x400000,'wrong allocation geometry')
    fd=os.open('/dev/'+MODULE+'0',os.O_RDWR)
    try:
        state=dict(boot_id=BOOT,base=base,size=size,previous_base=previous[0],previous_size=previous[1],
                   helper_sha256=HELPER,vendor_sha256=VENDOR,build_id=0x20036,
                   payload_base=0x200000,payload_bytes=0x80000)
        STATE.write_text(json.dumps(state,indent=2)+'\n');STATE.chmod(0o444)
        subprocess.run(['systemd-run','--unit=pacc-owned-ddr-pin-20261108','--property=Restart=always',
                        '--property=TimeoutStopSec=infinity','/usr/bin/python3',str(Path(__file__).resolve()),'--hold'],check=True)
        for _ in range(50):
            if (STATE.parent/'pin-ready').exists():break
            time.sleep(.1)
        require((STATE.parent/'pin-ready').exists(),'pin service did not become ready')
        subprocess.run(['python3',str(ROOT/'bin/test-owned-shared-ddr.py')],check=True)
        print(json.dumps(state,indent=2),flush=True)
    finally:os.close(fd)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hold',action='store_true');a=p.parse_args()
    require(os.geteuid()==0 and boot()==BOOT,'root/current boot required')
    if a.hold:hold()
    else:
        with open('/run/lock/pacc-firmware-update.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            leases=[]
            for i in range(4):
                f=open(ROOT/f'locks/pacc{i}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(f)
            prepare()
