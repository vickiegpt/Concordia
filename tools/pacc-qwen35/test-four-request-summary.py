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
for r in results:r['ngl']=0
out=aggregate(ref,results,[],[],[0]*4,ngl=0)
assert out['all_correct'] and out['aggregate_decode_tps']==127 and 'CPU-only' in out['scope']
assert not aggregate(ref,results,profiles,proofs,[0]*4,ngl=0)['all_correct']
for r in results:r['ngl']=1
proofs[-1]['device']=3;proofs[0]['device']=0
assert aggregate(ref,results,profiles,proofs,[0]*4,ngl=1)['all_correct']
assert not aggregate(ref,results,[],[],[0]*4,ngl=1)['all_correct']
missing=aggregate(ref,results,[],[],[0]*4,ngl=1)
assert missing['tokens_correct'] and not missing['xm_batch_verified']
assert missing['aggregate_decode_tps'] is None and 'not verified' in missing['scope']
assert missing['token_matched_decode_tps']==127
assert not aggregate(ref,results,profiles,proofs,[0]*4,ngl=2)['all_correct']
old=results[0]['finished_monotonic_us'];results[0]['finished_monotonic_us']=1
try:
    aggregate(ref,results,profiles,proofs,[0]*4,ngl=1)
    raise AssertionError('invalid per-worker interval accepted')
except ValueError:pass
results[0]['finished_monotonic_us']=old
print('PASS CPU baseline separation and requested-ngl proof matching')
parse=runpy.run_path(str(Path(__file__).with_name('run-four-pacc-requests.py')))['parse_batch_proof']
line='PACC Q6_KxQ8_1 dev=2 m=248320 n=2 k=2048 tiles=809 elapsed_ms=12 completion=verified'
assert parse([line],2)['multi_tiles']==809
assert parse([line],1)['calls']==0
assert parse([line.replace('n=2','n=1')],2)['multi_tiles']==0
assert parse([line.replace('verified','failed')],2)['calls']==0
print('PASS Q6_K batch evidence: device, columns and completion gates')
