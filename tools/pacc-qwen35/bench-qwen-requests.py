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
p.add_argument('--ngl', type=int, default=1)
p.add_argument('--cpu-kv-ffn', action='store_true', help='explicit CPU KV/FFN placement, not failure fallback')
p.add_argument('--request-timeout', type=float, default=600)
p.add_argument('--startup-timeout', type=float, default=300)
p.add_argument('--prefix', required=True)
p.add_argument('--workload', choices=['arithmetic', 'count16', 'count64', 'rawmath'], default='arithmetic')
args = p.parse_args()
if not 1 <= args.parallel <= 8:
    p.error('initial validated batch range is 1..8')
if args.ngl < 0:
    p.error('ngl must be nonnegative')
if not 0 < args.request_timeout <= 7200 or not 0 < args.startup_timeout <= 1800:
    p.error('request timeout must be in (0, 7200], startup timeout in (0, 1800] seconds')
ngl = args.ngl if args.mode == 'pacc' else 0
if Path(args.prefix+'.json').exists() or Path(args.prefix+'.server.log').exists():
    p.error('refusing to overwrite an existing benchmark')
root = Path('/srv/lanxin/qwen35-a3b')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
command = [str(root/'llama.cpp/build-pacc-cuda-gcc14/bin/llama-server'),
           '-m',str(root/'models/Qwen3.5-35B-A3B-Q4_K_M.gguf'),
           '-ngl',str(ngl), '-np',str(args.parallel),
           '-c',str(256*args.parallel), '-b','256','-ub',str(args.parallel),
           '-t','8','-tb','8','-fa','off','--no-warmup','--host','127.0.0.1',
           '--port',str(port),'--chat-template-kwargs','{"enable_thinking":false}']
if args.mode == 'cpu':
    command += ['--device', 'none']
elif args.cpu_kv_ffn:
    command += ['--no-kv-offload', '--override-tensor', r'blk\.[0-9]+\.ffn.*=CPU']
barrier = threading.Barrier(args.parallel)
def request(i):
    a, b = i+2, i+3
    prompt = f'What is {a} + {b}? Answer with just the number.'
    expected, token_limit = str(a+b), 8
    if args.workload == 'count16':
        prompt = 'Write the numbers from one to ten in English, separated by commas. Do not add an explanation.'
        expected, token_limit = 'one, two, three, four, five, six, seven, eight,', 16
    if args.workload == 'count64':
        prompt = 'Write the integers from 1 to 100 in ascending order, separated by a comma and a space. Do not add an explanation.'
        expected, token_limit = ', '.join(str(number) for number in range(1, 101)), 64
    payload = {'messages':[{'role':'user','content':prompt}],
               'max_tokens':token_limit,'temperature':0,'seed':42,'cache_prompt':False,
               'chat_template_kwargs':{'enable_thinking':False}}
    endpoint='/v1/chat/completions'
    if args.workload=='rawmath':
        expected,token_limit='2',1
        payload={'prompt':'1+1=','n_predict':1,'temperature':0,'seed':42,'cache_prompt':False}
        endpoint='/completion'
    req = urllib.request.Request(f'http://127.0.0.1:{port}{endpoint}',
          data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    barrier.wait(timeout=30)
    started = time.perf_counter()
    with urllib.request.urlopen(req,timeout=args.request_timeout) as response:
        data = json.load(response)
    finished = time.perf_counter()
    text = data['content'].strip() if args.workload=='rawmath' else data['choices'][0]['message']['content'].strip()
    completion_tokens=data['tokens_predicted'] if args.workload=='rawmath' else data['usage']['completion_tokens']
    correct = text == expected
    if args.workload == 'rawmath':
        correct = correct and completion_tokens == token_limit
    if args.workload == 'count64':
        # A token-capped answer can stop partway through the next integer.
        correct = (len(text) >= 32 and expected.startswith(text) and
                   data['usage']['completion_tokens'] == token_limit and
                   data['choices'][0]['finish_reason'] == 'length')
    return {'index':i,'start':started,'finish':finished,'latency_s':finished-started,
            'text':text,'expected':expected,'correct':correct,
            'completion_tokens':completion_tokens,'response':data}

with open(args.prefix+'.server.log','w') as log:
    server = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+args.startup_timeout
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
        report = {'mode':args.mode,'ngl':ngl,'parallel':args.parallel,'batch':256,'ubatch':args.parallel,
                  'command':command,
                  'cpu_kv_ffn':args.cpu_kv_ffn,
                  'workload':args.workload,
                  'all_correct':correct,'wall_s':wall,'completion_tokens':tokens,
                  'aggregate_output_tps':tokens/wall if correct else None,
                  'measurement':'one concurrent cohort; includes request prefill, excludes server startup; short correctness smoke, not capacity benchmark',
                  'requests':results}
        Path(args.prefix+'.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='requests'},indent=2),flush=True)
        if not correct:
            raise RuntimeError('Incorrect response: aggregate TPS rejected')
    except Exception as exc:
        # Preserve incorrect-response reports; record startup/transport failures too.
        if not Path(args.prefix+'.json').exists():
            Path(args.prefix+'.json').write_text(json.dumps({
                'mode':args.mode, 'ngl':ngl, 'parallel':args.parallel,
                'batch':256, 'ubatch':args.parallel, 'command':command,
                'cpu_kv_ffn':args.cpu_kv_ffn,
                'workload':args.workload, 'all_correct':False,
                'aggregate_output_tps':None, 'error':str(exc),
                'server_returncode':server.poll(),
            }, indent=2)+'\n')
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
