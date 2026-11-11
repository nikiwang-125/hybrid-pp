#!/usr/bin/env python3
"""Read-only audit of archived benchmarks/traces; needs only Python stdlib.

Writes new audit artifacts to --out, never changes source experiment files.
Run from the repository: python3 scripts/audit_baseline.py --out results/audit_20261110
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def command(root, *args):
    p = subprocess.run(args, cwd=root, capture_output=True, text=True)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def read_json(path):
    try:
        return json.loads(path.read_text()), None
    except (ValueError, OSError) as exc:
        return {}, str(exc)


def nearest(path, name, root):
    while path.is_relative_to(root):
        if (path / name).is_file():
            return path / name
        if path == root:
            break
        path = path.parent
    return None


def audit_trace(path):
    counts = Counter()
    meta = []
    mamba_bytes = []
    seen_steps = set()
    duplicate_steps = 0
    previous_step = None
    for line in path.open():
        try:
            rec = json.loads(line)
        except ValueError:
            counts['invalid_json_lines'] += 1
            continue
        kind = rec.get('kind')
        if kind == 'meta':
            meta.append(rec)
        elif kind == 'mamba_cache':
            mamba_bytes.append(rec.get('bytes'))
        elif kind == 'step':
            counts['steps'] += 1
            step = rec.get('step')
            if step in seen_steps:
                duplicate_steps += 1
            seen_steps.add(step)
            if isinstance(step, int) and previous_step is not None and step <= previous_step:
                counts['step_nonincreasing'] += 1
            previous_step = step if isinstance(step, int) else previous_step
            prefill = rec.get('num_prefills')
            decode = rec.get('num_decode_tokens')
            if prefill is None or decode is None:
                phase = 'unknown'
            elif prefill > 0:
                phase = 'mixed' if decode > 0 else 'prefill'
            else:
                phase = 'decode' if decode > 0 else 'empty'
            mode = 'graph' if rec.get('graph') is True else 'eager' if rec.get('graph') is False else 'unknown'
            counts[f'{phase}_{mode}'] += 1
            for field in ('batch', 'ctx_sum', 'ctx_max', 'cpu', 'dur_ms', 'num_prefill_tokens'):
                if rec.get(field) is None:
                    counts[f'missing_{field}'] += 1
            decoder_ops = sum(1 for op in rec.get('layers', []) if op[0] >= 0)
            if decoder_ops:
                counts['steps_with_decoder_ops'] += 1
            if mode == 'eager' and not decoder_ops:
                counts['eager_without_decoder_ops'] += 1
    return {"path": str(path), "meta": meta, "mamba_cache_bytes": mamba_bytes,
            "counts": dict(counts), "duplicate_step_ids": duplicate_steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    families = ['pp4', 'pp4_graphlong', 'pp4_layers']
    records = []
    for family in families:
        base = root / 'results' / family
        # Include failed runs and traces without benchmark results as well.
        run_dirs = {p.parent for p in base.rglob('bench.json')}
        run_dirs |= {p.parent for p in base.rglob('FAILED')}
        run_dirs |= {p.parent.parent for p in base.rglob('profile/*.jsonl')}
        for run in sorted(run_dirs):
            bench, bench_error = read_json(run / 'bench.json')
            arg_path = nearest(run, 'args.json', base)
            config, config_error = read_json(arg_path) if arg_path else ({}, 'missing args.json')
            log_path = nearest(run, 'server.log', base)
            log = log_path.read_text(errors='replace') if log_path else ''
            traces = [audit_trace(p) for p in sorted((run / 'profile').glob('*.jsonl'))]
            ranks = [m.get('pp_rank') for t in traces for m in t['meta']]
            flags = []
            if bench_error:
                flags.append('missing_or_invalid_benchmark')
            if config_error:
                flags.append('missing_or_invalid_args')
            if (run / 'FAILED').exists():
                flags.append('FAILED_marker_present')
            if bench and bench.get('completed') != bench.get('num_prompts'):
                flags.append('incomplete_benchmark')
            if config.get('profile') and Counter(ranks) != Counter(range(config.get('pp', 4))):
                flags.append('missing_or_duplicate_rank_metadata')
            if any(len(t['meta']) != 1 or t['duplicate_step_ids'] for t in traces):
                flags.append('possible_appended_trace_sessions')
            if any(t['counts'].get('invalid_json_lines') for t in traces):
                flags.append('malformed_trace')
            stage_mem = []
            for line in log.splitlines():
                marker = '[hybrid_pp] stage_mem '
                if marker in line:
                    try:
                        stage_mem.append(json.JSONDecoder().raw_decode(line.split(marker, 1)[1])[0])
                    except ValueError:
                        flags.append('malformed_stage_memory_log')
            records.append({
                'run': str(run.relative_to(root)), 'args_path': str(arg_path) if arg_path else None,
                'args': config, 'server_log': str(log_path) if log_path else None,
                'server_command': log.splitlines()[0] if log else None,
                'stage_memory': stage_mem, 'benchmark': bench,
                'benchmark_error': bench_error, 'flags': sorted(set(flags)), 'traces': traces,
            })

    # Snapshot the actual dirty checkout, not just HEAD. No credentials/env dump.
    source_files = sorted(set(root.glob('hybrid_pp/**/*.py')) | set(root.glob('scripts/*')) |
                          {root / 'docker/Dockerfile', root / 'pyproject.toml'})
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_files if p.is_file()}
    for name in hashes:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    snapshot = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'historical_source_match': 'unknown; current checkout hashes do not prove historical runtime code',
                'git_head': command(root, 'git', 'rev-parse', 'HEAD'),
                'git_status': command(root, 'git', 'status', '--short'), 'sha256': hashes,
                'gpu_probe': command(root, 'nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'),
                'docker_probe': command(root, 'docker', 'ps', '--format', '{{.Names}} {{.Image}} {{.Status}}')}
    (out / 'environment.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    (out / 'working_tree.patch').write_text(command(root, 'git', 'diff', '--binary')['stdout'])
    (out / 'inventory.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    fields = ['run', 'completed', 'requested', 'output_tok_s', 'ttft_ms', 'tpot_ms',
              'trace_files', 'ranks', 'steps', 'graph_steps', 'stage_memory_records', 'flags']
    with (out / 'inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            b = rec['benchmark']
            traces = rec['traces']
            writer.writerow({'run': rec['run'], 'completed': b.get('completed'),
                'requested': b.get('num_prompts'), 'output_tok_s': b.get('output_throughput'),
                'ttft_ms': b.get('mean_ttft_ms'), 'tpot_ms': b.get('mean_tpot_ms'),
                'trace_files': len(traces),
                'ranks': ','.join(str(m.get('pp_rank')) for t in traces for m in t['meta']),
                'steps': sum(t['counts'].get('steps', 0) for t in traces),
                'graph_steps': sum(v for t in traces for k, v in t['counts'].items() if k.endswith('_graph')),
                'stage_memory_records': len(rec['stage_memory']), 'flags': ';'.join(rec['flags'])})
    print(json.dumps({'runs': len(records), 'flagged_runs': sum(bool(r['flags']) for r in records),
                      'out': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
