#!/usr/bin/env python3
"""Isolated synthetic GPU allocator baseline, not a real layer migration.

python3 scripts/memory_reconfig_bench.py --out results/reconfig_trial
Each allocator runs in a separate subprocess; no vLLM server is started.
Requires an idle GPU, torch and (for cumem) vLLM 0.10.1.1.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def child(a):
    if importlib.util.find_spec('torch') is None:
        return {'status': 'unavailable', 'reason': 'torch is not installed', 'mode': a.child}
    import gc
    import torch
    if not torch.cuda.is_available():
        return {'status': 'unavailable', 'reason': 'CUDA is not available', 'mode': a.child}
    probe = subprocess.run(['nvidia-smi', '-i', str(a.device), '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip():
        return {'status': 'skipped', 'reason': 'GPU idle state not confirmed or compute processes are present', 'mode': a.child}
    torch.cuda.set_device(a.device)
    free, total = torch.cuda.mem_get_info()
    sizes = [int(x * 2**20) for x in (a.weights_mib, a.kv_mib, a.replacement_mib)]
    if sum(sizes) > free // 2:
        return {'status': 'skipped', 'reason': 'requested buffers exceed half of current free memory', 'mode': a.child}
    allocator = None
    if a.child == 'cumem':
        import vllm
        if vllm.__version__ != '0.10.1.1':
            return {'status': 'unsupported', 'reason': f'expected vllm 0.10.1.1, got {vllm.__version__}', 'mode': a.child}
        from vllm.device_allocator.cumem import CuMemAllocator
        allocator = CuMemAllocator.get_instance()
    def snapshot():
        torch.cuda.synchronize()
        f, t = torch.cuda.mem_get_info()
        return {'free': f, 'total': t, 'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(), 'peak_allocated': torch.cuda.max_memory_allocated()}
    events = []
    def timed(label, fn, iteration, cuda=False):
        torch.cuda.synchronize()
        before = snapshot()
        if cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        t = time.perf_counter()
        value = fn()
        if cuda:
            end.record()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t) * 1000
        events.append({'iteration': iteration, 'label': label, 'wall_ms': elapsed,
                       'device_ms': begin.elapsed_time(end) if cuda else None,
                       'before': before, 'after': snapshot()})
        return value
    def allocate(n, fill):
        t = torch.empty(n, dtype=torch.uint8, device=f'cuda:{a.device}')
        t.fill_(fill)
        return t
    if allocator:
        with allocator.use_memory_pool(tag='weights'):
            weights = allocate(sizes[0], 17)
        with allocator.use_memory_pool(tag='kv_cache'):
            kv = allocate(sizes[1], 29)
    else:
        weights, kv = allocate(sizes[0], 17), allocate(sizes[1], 29)
    pointer = weights.data_ptr()
    cpu_source = torch.full((sizes[2],), 41, dtype=torch.uint8, pin_memory=True)
    for iteration in range(a.rounds):
        if allocator:
            timed('sleep_all_backup_weights_to_cpu', lambda: allocator.sleep(offload_tags=('weights',)), iteration)
        else:
            # Drop the last Python reference before emptying the caching allocator.
            torch.cuda.synchronize()
            before = snapshot()
            t = time.perf_counter()
            kv = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            events.append({'iteration': iteration, 'label': 'release_kv', 'wall_ms': (time.perf_counter()-t)*1000,
                           'before': before, 'after': snapshot()})
        replacement = timed('allocate_replacement_uninitialized', lambda: torch.empty(sizes[2], dtype=torch.uint8, device=f'cuda:{a.device}'), iteration)
        timed('copy_synthetic_weights_from_pinned_cpu', lambda: replacement.copy_(cpu_source, non_blocking=True), iteration, cuda=True)
        del replacement
        gc.collect()
        torch.cuda.empty_cache()
        if allocator:
            timed('wake_weights_from_cpu', lambda: allocator.wake_up(tags=['weights']), iteration)
            timed('wake_kv_uninitialized', lambda: allocator.wake_up(tags=['kv_cache']), iteration)
            timed('initialize_kv', lambda: kv.fill_(29), iteration)
        else:
            kv = timed('reallocate_and_initialize_kv', lambda: allocate(sizes[1], 29), iteration)
        if weights.data_ptr() != pointer or not bool(torch.all(weights == 17).item()):
            raise RuntimeError('weights address/content check failed')
    return {'status': 'measured', 'mode': a.child, 'torch': torch.__version__,
            'device': torch.cuda.get_device_name(a.device), 'sizes_bytes': sizes, 'rounds': a.rounds,
            'events': events, 'weights_check': 'full content and stable address passed',
            'scope': 'uint8 synthetic buffers; cumem includes CPU weight backup/restore, torch retains weights; not equivalent service policies',
            'limitations': ['first round reported separately by iteration=0', 'no active requests, no model, no graph, no PP state migration',
                            'CPU copy uses synthetic pinned data, not checkpoint parsing or real layer loading', 'synchronization overhead is part of this controlled benchmark; memory probes surround timed regions']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--child', choices=['torch', 'cumem'])
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--weights-mib', type=float, default=64)
    p.add_argument('--kv-mib', type=float, default=128)
    p.add_argument('--replacement-mib', type=float, default=64)
    p.add_argument('--rounds', type=int, default=5)
    a = p.parse_args()
    if min(a.weights_mib, a.kv_mib, a.replacement_mib, a.rounds) <= 0:
        p.error('buffer sizes and rounds must be positive')
    if a.child:
        try:
            result = child(a)
        except Exception as exc:
            result = {'status': 'failed', 'mode': a.child, 'reason': f'{type(exc).__name__}: {exc}'}
        with a.out.open('x') as f:
            json.dump(result, f, indent=2)
        return
    a.out.mkdir(parents=True, exist_ok=False)
    statuses = {}
    for mode in ('torch', 'cumem'):
        dest = a.out / f'{mode}.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--child', mode, '--out', str(dest),
                   '--device', str(a.device), '--weights-mib', str(a.weights_mib), '--kv-mib', str(a.kv_mib),
                   '--replacement-mib', str(a.replacement_mib), '--rounds', str(a.rounds)]
        with (a.out / f'{mode}.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=log, timeout=180)
                statuses[mode] = json.loads(dest.read_text())['status'] if dest.exists() else f'child_exit_{result.returncode}'
            except subprocess.TimeoutExpired:
                statuses[mode] = 'timeout'
    (a.out / 'summary.json').write_text(json.dumps(statuses, indent=2))
    print(json.dumps(statuses))


if __name__ == '__main__':
    main()
