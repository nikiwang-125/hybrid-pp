#!/usr/bin/env python3
"""Drive a running local server with explicit warmup/main profiling markers.

Nonstreaming requests: reports E2E and usage, NOT TTFT/TPOT. Synthetic prompt
length is measured from returned usage, not inferred from character count.
Set HYBRID_PP_WINDOW_FILE to --marker-file before starting the server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def mark(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--marker-file', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--requests', type=int, default=8)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--prompt-repeats', type=int, default=128)
    p.add_argument('--output-tokens', type=int, default=32)
    a = p.parse_args()
    if min(a.concurrency, a.requests, a.prompt_repeats, a.output_tokens) <= 0 or a.warmup < 0:
        p.error('counts must be positive (warmup may be zero)')
    if a.out.exists():
        p.error('output already exists')
    prompt = 'The quick brown fox jumps over the lazy dog. ' * a.prompt_repeats
    def complete(i):
        body = json.dumps({'model': a.model, 'prompt': prompt, 'max_tokens': a.output_tokens,
                           'temperature': 0, 'ignore_eos': True, 'stream': False}).encode()
        request = urllib.request.Request(a.url.rstrip('/') + '/v1/completions', data=body,
                                         headers={'Content-Type': 'application/json'})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
            return {'index': i, 'e2e_ms': (time.perf_counter()-t)*1000, 'usage': result.get('usage'), 'status': 'ok'}
        except Exception as exc:
            return {'index': i, 'e2e_ms': (time.perf_counter()-t)*1000, 'status': 'failed', 'error': str(exc)}
    report = {'scope': 'single-host synthetic nonstreaming workload; not directly comparable with archived vllm bench',
              'concurrency_limit': a.concurrency, 'warmup': [], 'requests': []}
    try:
        mark(a.marker_file, 'warmup')
        report['warmup'] = [complete(i) for i in range(a.warmup)]
        if any(x['status'] != 'ok' for x in report['warmup']):
            report['status'] = 'warmup_failed'
        else:
            mark(a.marker_file, 'main')
            report['main_start_epoch_s'] = time.time()
            t = time.perf_counter()
            with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                report['requests'] = list(pool.map(complete, range(a.requests)))
            report['duration_s'] = time.perf_counter()-t
            report['main_end_epoch_s'] = time.time()
            report['status'] = 'completed' if all(x['status'] == 'ok' for x in report['requests']) else 'request_failures'
    finally:
        mark(a.marker_file, 'cooldown')
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open('x') as f:
            json.dump(report, f, indent=2)
    print(a.out)


if __name__ == '__main__':
    main()
