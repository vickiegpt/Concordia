#!/usr/bin/env python3
"""Per-device lease and live allocation checks; preserve all existing strict policies."""
import fcntl
import errno
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path('/srv/lanxin/qwen35-a3b');path=ROOT/'logs/recovery-20261108/state.json'
device=int(sys.argv[1]);command=sys.argv[2:]
if command[:1]==['--']:command=command[1:]
assert device in range(4) and command
assert path.stat().st_uid==0 and path.stat().st_mode&0o022==0
state=json.loads(path.read_text())
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==state['boot_id']
params=Path('/sys/module/hetgpu_sifive_mbox_ddr_coh/parameters')
assert int((params/'shared_ddr_base_override').read_text())==0
assert int((params/'shared_ddr_size').read_text())==state['size']==0x400000
assert subprocess.run(['systemctl','is-active','--quiet','pacc-owned-ddr-pin-20261108.service']).returncode==0
lock=os.open(ROOT/f'locks/pacc{device}.lock',os.O_RDWR);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.set_inheritable(lock,True)
assert subprocess.run(['fuser',f'/dev/pacc{device}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==1
fd=os.open(f'/dev/hetgpu_sifive_mbox_ddr_coh{device}',os.O_RDONLY)
try:
    deadline=time.monotonic()+5
    while True:
        try:
            raw=os.pread(fd,8,0x02004000)
            assert len(raw)==8 and struct.unpack('<Q',raw)[0]==state['base']
            break
        except OSError as error:
            if error.errno not in (errno.EBUSY,errno.EAGAIN) or time.monotonic()>=deadline:raise
            time.sleep(.005)
finally:os.close(fd)
env=os.environ.copy();env.update(HETGPU_SIFIVE_VISIBLE_DEVICES=f'{device},',HETGPU_SIFIVE_FORCE_DEVICE=str(device),
    HETGPU_SIFIVE_GEMM_DEVICE=str(device),HETGPU_SIFIVE_GEMM_DEVICES=str(device),HETGPU_SIFIVE_GEMM_MULTI_N_DEVICES=str(device),
    HETGPU_SIFIVE_KERNEL_SLOT_COUNT='1',HETGPU_SIFIVE_KERNEL_TOTAL_SLOTS='1',HETGPU_SIFIVE_KERNEL_SLOT_BASE='0',
    HETGPU_SIFIVE_KERNEL_SLOT_BYTES='524288',HETGPU_SIFIVE_KERNEL_SLOT_LOW='1',HETGPU_SIFIVE_GEMM_SHARED_SLOTS='1',
    HETGPU_SIFIVE_GEMM_SLOT_BYTES='524288',HETGPU_SIFIVE_SHARED_DDR_BASE=hex(state['base']),
    HETGPU_SIFIVE_SHARED_DDR_PAYLOAD_BASE_OFF=hex(0x200000+device*0x80000),
    HETGPU_SIFIVE_SHARED_DDR_BYTES=hex(0x280000+device*0x80000),HETGPU_SIFIVE_JOBD_BOOTSTRAP='0',
    HETGPU_SIFIVE_JOB_TIMEOUT_MS='10000',HETGPU_SIFIVE_ZLUDA_IRQ_TRACE='0',HETGPU_SIFIVE_LOG_MEMORY='0',
    HETGPU_SIFIVE_MMVF_ASYNC_NO_WAIT='0',HETGPU_SIFIVE_SKIP_JOB_WAIT='0')
assert 'CONCORDIA_RELEASE' in env
print(f'isolated_recovered_worker physical={device} base={state["base"]:#x}',flush=True)
os.chdir(ROOT)
os.execvpe('bash',['bash','-c',
    'source bin/pacc-env.sh; '
    'if [[ -n ${LLAMA_RUNTIME_LIBRARY_DIR:-} ]]; then '
    'export LD_LIBRARY_PATH="${LLAMA_RUNTIME_LIBRARY_DIR}:${LD_LIBRARY_PATH}"; fi; '
    'exec "$@"','pacc-recovered',*command],env)
