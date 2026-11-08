#!/usr/bin/env python3
import copy
import runpy
from pathlib import Path

aggregate=runpy.run_path(str(Path(__file__).with_name('run-four-pacc-requests.py')))['aggregate']
config=dict(prompts=['a','b'],max_tokens=128)
tokens=[[1]*128,[2]*128]
ref=dict(config=config,ngl=0,reference_complete=True,tokens=tokens,input_tokens=[[3],[4]],model='full.gguf')
r=dict(all_correct=True,ngl=99,sequences=2,model='full.gguf',config=config,tokens=tokens,
       input_tokens=[[3],[4]],output_tokens=256,decode_tokens=254,started_monotonic_us=1000000,
       decode_started_monotonic_us=2000000,finished_monotonic_us=10000000)
results=[copy.deepcopy(r) for _ in range(4)]
profiles=[dict(device=i,capture_valid=True,build_matches=True) for i in range(4)]
proofs=[dict(device=i,multi_tiles=20,max_n=8) for i in range(4)]
def evaluate():return aggregate(ref,results,profiles,proofs,[0]*4)
out=evaluate()
assert out['all_correct'] and out['aggregate_decode_tps']==127
for field,value in [('tokens',[[1]*127,[2]*128]),('ngl',0),('sequences',1),('model','diagnostic.gguf')]:
    old=results[0][field];results[0][field]=value
    assert not evaluate()['all_correct'];results[0][field]=old
for field,value in [('capture_valid',False),('build_matches',False),('device',3)]:
    old=profiles[0][field];profiles[0][field]=value
    assert not evaluate()['all_correct'];profiles[0][field]=old
proofs[0]['max_n']=1
assert evaluate()['aggregate_decode_tps'] is None
proofs[0]['max_n']=8;proofs[0]['device']=3
assert not evaluate()['all_correct']
print('PASS four-request aggregation: token, build, partition, XM-N and shared-clock gates')
