#!/usr/bin/env python3
"""Bounded multi-request correctness/throughput probe; owns and stops its server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['cpu', 'pacc'], required=True)
p.add_argument('--parallel', type=int, default=4)
p.add_argument('--prefix', required=True)
p.add_argument('--workload', choices=['arithmetic', 'count16'], default='arithmetic')
args = p.parse_args()
if not 1 <= args.parallel <= 8:
    p.error('initial validated batch range is 1..8')
root = Path('/srv/lanxin/qwen35-a3b')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
command = [str(root/'llama.cpp/build-pacc-cuda-gcc14/bin/llama-server'),
           '-m',str(root/'models/Qwen3.5-35B-A3B-Q4_K_M.gguf'),
           '-ngl','1' if args.mode=='pacc' else '0', '-np',str(args.parallel),
           '-c',str(256*args.parallel), '-b','256','-ub',str(args.parallel),
           '-t','8','-tb','8','-fa','off','--no-warmup','--host','127.0.0.1',
           '--port',str(port),'--chat-template-kwargs','{"enable_thinking":false}']
barrier = threading.Barrier(args.parallel)
def request(i):
    a, b = i+2, i+3
    prompt = f'What is {a} + {b}? Answer with just the number.'
    expected, token_limit = str(a+b), 8
    if args.workload == 'count16':
        prompt = 'Write the numbers from one to ten in English, separated by commas. Do not add an explanation.'
        expected, token_limit = 'one, two, three, four, five, six, seven, eight,', 16
    payload = {'messages':[{'role':'user','content':prompt}],
               'max_tokens':token_limit,'temperature':0,'seed':42,'cache_prompt':False,
               'chat_template_kwargs':{'enable_thinking':False}}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
          data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    barrier.wait(timeout=30)
    started = time.perf_counter()
    with urllib.request.urlopen(req,timeout=600) as response:
        data = json.load(response)
    finished = time.perf_counter()
    text = data['choices'][0]['message']['content'].strip()
    return {'index':i,'start':started,'finish':finished,'latency_s':finished-started,
            'text':text,'expected':expected,'correct':text==expected,
            'completion_tokens':data['usage']['completion_tokens'],'response':data}

with open(args.prefix+'.server.log','w') as log:
    server = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+300
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('server readiness')
            time.sleep(0.5)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(request,range(args.parallel)))
        wall = max(r['finish'] for r in results)-min(r['start'] for r in results)
        tokens = sum(r['completion_tokens'] for r in results)
        correct = all(r['correct'] for r in results)
        report = {'mode':args.mode,'parallel':args.parallel,'batch':256,'ubatch':args.parallel,
                  'workload':args.workload,
                  'all_correct':correct,'wall_s':wall,'completion_tokens':tokens,
                  'aggregate_output_tps':tokens/wall if correct else None,
                  'measurement':'one concurrent cohort; includes request prefill, excludes server startup; short correctness smoke, not capacity benchmark',
                  'requests':results}
        Path(args.prefix+'.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='requests'},indent=2),flush=True)
        if not correct:
            raise RuntimeError('Incorrect response: aggregate TPS rejected')
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
