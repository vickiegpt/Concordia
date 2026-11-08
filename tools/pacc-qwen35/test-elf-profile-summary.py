#!/usr/bin/env python3
from pathlib import Path
import runpy
import tempfile

summarize=runpy.run_path(str(Path(__file__).with_name('summarize-elf-profile.py')))['summarize']
host='PACC_ELF_PROFILE kernel=pacc_q4k_tile_v1 dev=2 seq=42 setup_us=10 stage_us=20 publish_us=30 wait_us=100 readback_us=5 total_us=165\n'
device='PACC_QK_PROFILE dev=2 qtype=4 m=32 n=8 nb=2 gate=1 device_us=80 activation_us=4 weights_us=30 xm_us=20 reduce_us=10 xm_calls=32 local_copy_us=6\n'
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'sample.log'
    path.write_text(host+device)
    result=summarize(path)
    assert result['host_by_kernel']['pacc_q4k_tile_v1']['total_us']==165
    assert result['device_qk']['device_us']==80
    assert result['paired_qk']['publish_wait_minus_device_us']==50
    for invalid in (device, host+device.replace('dev=2','dev=1')):
        path.write_text(invalid)
        try: summarize(path)
        except ValueError: pass
        else: raise AssertionError('unpaired profile accepted')
print('PASS nested durations, residual definition and mismatched device rejection')
