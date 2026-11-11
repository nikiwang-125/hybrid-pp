import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProfileTests(unittest.TestCase):
    def test_lookup_keeps_holdout_out_of_training_artifacts(self):
        prepare = load_module('prepare_data', ROOT / 'scripts/prepare_modeling_data.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = []
            for length in (1024, 4096):
                trace = root / f'{length}.jsonl'
                rows = [{'kind': 'step', 'step': i, 'num_prefills': 0, 'num_decode_tokens': 1,
                         'num_prefill_tokens': 0, 'batch': 1, 'ctx_max': length, 'ctx_sum': length,
                         'graph': False, 'dur_ms': 10, 'total_ms': 10, 'layers': [[0, 'mamba', 10]]} for i in range(4)]
                trace.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
                inventory.append({'run': f'results/mock/default/{length}-128', 'flags': [],
                                  'args': {'max_num_seqs': 1, 'concurrency': 1},
                                  'traces': [{'path': str(trace), 'meta': [{'pp_rank': 0, 'stage_pattern': 'M', 'profile_layers': True}]}]})
            source = root / 'inventory.json'
            source.write_text(json.dumps(inventory))
            out = root / 'out'
            with patch.object(sys, 'argv', ['prepare', '--inventory', str(source), '--out', str(out)]):
                prepare.main()
            train = json.loads((out / 'stage_lookup.json').read_text())
            held = json.loads((out / 'holdout_observations.json').read_text())
            ops = json.loads((out / 'layer_type_lookup.json').read_text())
            self.assertTrue(train and held and ops)
            self.assertTrue(all(x['split'] == 'train' for x in train))
            self.assertTrue(all(x['split'] == 'holdout' for x in held))
            self.assertTrue(all(x['key'][0] == 'train' for x in ops))

    def test_worker_metadata_and_cpu_cleanup_without_cuda(self):
        with patch.dict(sys.modules, {'torch': ModuleType('torch')}):
            p = load_module('profiler_under_test', ROOT / 'hybrid_pp/profiler.py')
        captured = []
        class Worker:
            def execute_model(self, execute_model_req):
                captured.append(p._recorder.schedule.copy())
                p._recorder.step += 1
                p._recorder.last = {'info': {}}
                return 'ok'
        p._recorder = SimpleNamespace(recording=lambda: True, step=0, schedule={}, call=None, last=None)
        p._wrap_worker_call(Worker)
        request = SimpleNamespace(virtual_engine=3, seq_group_metadata_list=[])
        self.assertEqual(Worker().execute_model(request), 'ok')
        self.assertEqual(captured[0]['virtual_engine'], 3)
        self.assertIsNone(p._recorder.call)
        self.assertEqual(p._recorder.schedule, {})
        self.assertGreaterEqual(p._recorder.last['info']['cpu']['call_ms'], 0)

    def test_window_client_marks_actual_request_phases(self):
        client = load_module('window_client', ROOT / 'scripts/profile_window_client.py')
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'window.txt'
            output = Path(directory) / 'result.json'
            observed = []
            def urlopen(request, timeout):
                observed.append(marker.read_text().strip())
                body = json.loads(request.data)
                self.assertFalse(body['stream'])
                return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 20, 'completion_tokens': 4}}).encode())
            argv = ['client', '--model', 'mock', '--marker-file', str(marker), '--out', str(output),
                    '--requests', '3', '--warmup', '2', '--concurrency', '2']
            with patch.object(sys, 'argv', argv), patch.object(client.urllib.request, 'urlopen', side_effect=urlopen):
                client.main()
            result = json.loads(output.read_text())
            self.assertEqual(observed.count('warmup'), 2)
            self.assertEqual(observed.count('main'), 3)
            self.assertEqual(marker.read_text().strip(), 'cooldown')
            self.assertEqual(result['status'], 'completed')
            self.assertNotIn('ttft_ms', result)


if __name__ == '__main__':
    unittest.main()
