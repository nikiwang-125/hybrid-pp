#!/usr/bin/env python3
"""Collect read-only runtime evidence without starting containers or Ray."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys


def run(args):
    if not shutil.which(args[0]):
        return {'status': 'not_installed'}
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    packages = {}
    for name in ('torch', 'vllm', 'transformers', 'ray', 'hybrid-pp'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    sources = list((root / 'hybrid_pp').rglob('*.py'))
    report = {'hostname': socket.gethostname(), 'python': sys.executable, 'platform': platform.platform(),
              'packages': packages, 'plugins': [str(x) for x in metadata.entry_points(group='vllm.general_plugins')],
              'gpu': run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,driver_version', '--format=csv']),
              'topology': run(['nvidia-smi', 'topo', '-m']),
              'gpu_processes': run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv']),
              'containers': run(['docker', 'ps', '--format', '{{.Names}} {{.Image}} {{.Status}}']),
              'ray_status': run(['ray', 'status']),
              'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'scope': 'current process namespace; missing GPU access does not prove physical host GPUs are broken'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
    print(a.out)


if __name__ == '__main__':
    main()
