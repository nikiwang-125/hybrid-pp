"""Byte accounting for the existing Nemotron V0 layout; no GPU dependencies.

python3 -m hybrid_pp.capacity CONFIG --partition 14,14,14,14 \
    --contexts 16384,16384,16384,16384 --state-slots 16 --out capacity.json
Unspecified runtime/transition memory stays unknown, never silently zero.
"""
import argparse
import json
import math
import re
from pathlib import Path
import struct

from hybrid_pp.model_info import load_config, stage_table


def checkpoint_bytes(directory):
    """Read only safetensors headers, without loading tensor payloads."""
    result = {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 128 * 2**20:
                raise ValueError('unreasonable safetensors header size')
            header = json.loads(f.read(size))
        for name, info in header.items():
            if name == '__metadata__':
                continue
            if name in result:
                raise ValueError(f'duplicate checkpoint tensor {name}')
            a, b = info['data_offsets']
            result[name] = b - a
    return result


def estimate(cfg, partition, contexts, slots, block_tokens=16, pool_blocks=None,
             runtime_bytes=None, budget_bytes=None):
    if min(partition) <= 0 or sum(partition) != len(cfg['hybrid_override_pattern']):
        raise ValueError('partition must contain positive layer counts covering the model')
    if slots < len(contexts) or min(contexts, default=0) < 0 or block_tokens <= 0:
        raise ValueError('state slots must cover retained requests; lengths must be nonnegative')
    if pool_blocks is not None and pool_blocks < 0:
        raise ValueError('pool_blocks must be nonnegative')
    if runtime_bytes is not None and runtime_bytes < 0:
        raise ValueError('runtime bytes must be nonnegative')
    pages = sum(math.ceil(n / block_tokens) for n in contexts)
    rows = stage_table(cfg, partition)
    pp = len(partition)
    for row in rows:
        # V0 creates PP VE caches, each floor(global_blocks / PP) blocks.
        actual_pool_pages = None if pool_blocks is None else (pool_blocks // pp) * pp
        row.update({
            'weights_estimated_bytes': round(row['weights_gib'] * 2**30),
            'retained_requests': len(contexts), 'state_slots': slots,
            'mamba_allocated_bytes': row['mamba_bytes_per_seq'] * slots,
            'kv_effective_paged_bytes': pages * block_tokens * row['kv_bytes_per_token'],
            'kv_pool_allocated_bytes': None if actual_pool_pages is None else actual_pool_pages * block_tokens * row['kv_bytes_per_token'],
            'runtime_peak_extra_bytes': runtime_bytes,
            'budget_bytes': budget_bytes,
            'pool_pages_fit_aggregate': None if actual_pool_pages is None else (pages <= actual_pool_pages if row['attention'] else True),
            'transition_peak_bytes': None,
        })
        kv = row['kv_pool_allocated_bytes']
        row['known_components_bytes'] = row['weights_estimated_bytes'] + row['mamba_allocated_bytes'] + (row['kv_effective_paged_bytes'] if kv is None else kv)
        row['estimated_peak_bytes'] = None if runtime_bytes is None else row['known_components_bytes'] + runtime_bytes
        row['fits_budget'] = None if budget_bytes is None or runtime_bytes is None or kv is None else row['estimated_peak_bytes'] <= budget_bytes
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--partition', required=True)
    p.add_argument('--contexts', required=True, help='retained context lengths, comma separated; empty for no requests')
    p.add_argument('--state-slots', required=True, type=int, help='actual allocation slots, including graph padding')
    p.add_argument('--block-tokens', type=int, default=16)
    p.add_argument('--pool-blocks', type=int, help='engine global block count before division among VEs')
    p.add_argument('--runtime-extra-gib', type=float, help='extra peak excluding weights/KV/Mamba; do not use startup activation blindly')
    p.add_argument('--budget-gib', type=float)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    rows = estimate(cfg, list(map(int, args.partition.split(','))),
                    list(map(int, args.contexts.split(','))) if args.contexts else [], args.state_slots,
                    args.block_tokens, args.pool_blocks,
                    None if args.runtime_extra_gib is None else round(args.runtime_extra_gib * 2**30),
                    None if args.budget_gib is None else round(args.budget_gib * 2**30))
    tensors = checkpoint_bytes(Path(args.config).parent)
    start = 0
    for row, count in zip(rows, map(int, args.partition.split(','))):
        end = start + count
        selected = {name: n for name, n in tensors.items()
                    if '.layers.' not in name or start <= int(re.search(r'\.layers\.(\d+)\.', name).group(1)) < end}
        row['checkpoint_stage_bytes_with_replicated_globals'] = sum(selected.values()) if tensors else None
        start = end
    out = {'scope': 'Nemotron V0 TP=1 bf16 weights/conv, fp32 SSM; current model_info weight estimate includes embed and head on every stage',
           'limitations': ['VE-local page capacity must also be checked; aggregate capacity is insufficient',
                          'effective KV excludes prefix sharing, lookahead, null-block and allocator overhead',
                          'unknown runtime and transition peaks are not a feasibility certificate',
                          'checkpoint tensor bytes are storage counts, not actual GPU allocation'],
           'stages': rows, 'checkpoint_tensor_bytes': tensors}
    with open(args.out, 'x') as f:
        json.dump(out, f, indent=2)
    print(args.out)


if __name__ == '__main__':
    main()
