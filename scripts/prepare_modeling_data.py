#!/usr/bin/env python3
"""Export audited old traces and evaluate an exploratory stage lookup model.

No torch/numpy required. Hold out entire 4096-token runs before aggregation.
Never interprets graph traces as decoder-layer measurements.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def bucket(value):
    return 0 if not value else 2 ** round(math.log2(value))


def read_trace(path):
    with open(path) as stream:
        for line in stream:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--skip-first', type=int, default=2, help='exploratory warmup heuristic, not validated window selection')
    ap.add_argument('--holdout-input', type=int, default=4096)
    a = ap.parse_args()
    if a.skip_first < 0:
        ap.error('--skip-first must be nonnegative')
    a.out.mkdir(parents=True, exist_ok=False)
    runs = json.loads(a.inventory.read_text())
    aggregates = defaultdict(list)
    ops = defaultdict(list)
    counts = defaultdict(int)
    fields = ['run', 'split', 'rank', 'step', 'phase', 'mode', 'batch', 'query_tokens',
              'ctx_sum', 'ctx_max', 'dur_ms', 'total_ms', 'window', 'graph_layer_unavailable']
    with (a.out / 'steps.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            if run['flags']:
                counts['excluded_flagged_runs'] += 1
                continue
            if not run['traces']:
                continue
            input_len = int(Path(run['run']).name.split('-')[0])
            split = 'holdout' if input_len == a.holdout_input else 'train'
            for trace in run['traces']:
                rank = trace['meta'][0]['pp_rank']
                for row in read_trace(trace['path']):
                    if row.get('kind') != 'step':
                        continue
                    if row['step'] < a.skip_first:
                        counts['heuristic_warmup_steps'] += 1
                        continue
                    npre = row.get('num_prefills')
                    ndec = row.get('num_decode_tokens')
                    if npre is None or ndec is None:
                        counts['unknown_phase_steps'] += 1
                        continue
                    phase = 'mixed' if npre and ndec else 'prefill' if npre else 'decode' if ndec else 'empty'
                    mode = 'graph' if row['graph'] else 'eager'
                    writer.writerow(dict(zip(fields, [run['run'], split, rank, row['step'], phase, mode,
                        row.get('batch'), row.get('num_prefill_tokens', 0) + ndec,
                        row.get('ctx_sum'), row.get('ctx_max'), row.get('dur_ms'), row.get('total_ms'),
                        row.get('measurement_window', 'unmarked'), bool(row['graph'])])))
                    counts[f'{split}_{phase}_{mode}'] += 1
                    if phase not in ('prefill', 'decode') or row.get('batch') is None:
                        continue
                    # Workload composition is incompletely known in old traces.
                    key = (split, rank, phase, mode, bucket(row['batch']), bucket(row.get('ctx_max', 0)),
                           bucket(row.get('num_prefill_tokens', 0)), trace['meta'][0]['stage_pattern'],
                           run['args'].get('max_num_seqs'), run['args'].get('concurrency'),
                           trace['meta'][0].get('profile_layers'))
                    aggregates[key].append(row['dur_ms'])
                    if mode == 'eager':
                        by_type = defaultdict(list)
                        for idx, kind, ms in row.get('layers', []):
                            if idx >= 0:
                                by_type[kind].append(ms)
                        for kind, values in by_type.items():
                            ops[key + (kind,)].append(statistics.mean(values))
    table = [{'split': k[0], 'rank': k[1], 'phase': k[2], 'mode': k[3], 'batch_bucket': k[4],
              'context_bucket': k[5], 'prefill_query_bucket': k[6], 'n': len(v),
              'stage_pattern': k[7], 'max_num_seqs': k[8], 'concurrency': k[9], 'profile_layers': k[10],
              'median_ms': statistics.median(v), 'mean_ms': statistics.mean(v)}
             for k, v in sorted(aggregates.items())]
    (a.out / 'stage_lookup.json').write_text(json.dumps([x for x in table if x['split'] == 'train'], indent=2))
    (a.out / 'holdout_observations.json').write_text(json.dumps([x for x in table if x['split'] == 'holdout'], indent=2))
    op_table = [{'key': list(k), 'steps': len(v), 'median_per_layer_ms': statistics.median(v)}
                for k, v in sorted(ops.items()) if k[0] == 'train']
    (a.out / 'layer_type_lookup.json').write_text(json.dumps(op_table, indent=2))
    errors = []
    predictions = []
    for key, values in aggregates.items():
        if key[0] != 'holdout':
            continue
        candidates = [k for k in aggregates if k[0] == 'train' and k[1:4] == key[1:4] and k[7:] == key[7:]]
        if not candidates:
            continue
        distance = lambda k: sum(abs(math.log2(1 + k[i]) - math.log2(1 + key[i])) for i in (4, 5, 6))
        near = min(candidates, key=lambda k: (distance(k), k))
        actual = statistics.median(values)
        pred = statistics.median(aggregates[near])
        error = abs(pred - actual) / actual if actual else None
        if error is not None:
            errors.append(error)
        predictions.append({'holdout_key': list(key), 'train_key': list(near), 'distance': distance(near),
                            'actual_median_ms': actual, 'predicted_ms': pred, 'relative_error': error, 'n': len(values)})
    (a.out / 'holdout_predictions.json').write_text(json.dumps(predictions, indent=2))
    report = {'status': 'exploratory_only_not_validated_predictor', 'counts': dict(counts),
              'holdout_input': a.holdout_input, 'holdout_groups': len(predictions),
              'median_group_absolute_relative_error': statistics.median(errors) if errors else None,
              'mean_group_absolute_relative_error': statistics.mean(errors) if errors else None,
              'max_group_absolute_relative_error': max(errors) if errors else None,
              'train_groups': sum(k[0] == 'train' for k in aggregates),
              'error_by_phase': {phase: {
                  'groups': len([x for x in predictions if x['holdout_key'][2] == phase]),
                  'median': statistics.median([x['relative_error'] for x in predictions if x['holdout_key'][2] == phase])}
                  for phase in ('prefill', 'decode') if any(x['holdout_key'][2] == phase for x in predictions)},
              'limitations': ['whole 4096-token runs held out; not new partition generalization',
                             'each group weighted equally; this is not request latency error',
                             'old traces lack main-window markers and complete per-request context/query features',
                             'first-step exclusion is a declared heuristic; no arbitrary outlier trimming',
                             'pure prefill/decode only for model; mixed steps exported but excluded from fit',
                             'nearest log-bucket lookup pools per-step samples within a matched static cohort; exploratory baseline only',
                             'layout and static concurrency/slot configuration now matched; 4K diagnostics informed this revision, so not an untouched validation set',
                             'GPU event spans may include CPU launch gaps; graph decoder-layer times unavailable']}
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
