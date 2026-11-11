import unittest
from types import SimpleNamespace

from hybrid_pp.capacity import estimate
from hybrid_pp.profile_features import schedule_features


CFG = {'hidden_size': 16, 'vocab_size': 32, 'num_attention_heads': 2,
       'num_key_value_heads': 1, 'head_dim': 8, 'mamba_num_heads': 2,
       'mamba_head_dim': 8, 'n_groups': 1, 'ssm_state_size': 4,
       'conv_kernel': 4, 'intermediate_size': 32,
       'hybrid_override_pattern': 'M*-M'}


class CapacityTests(unittest.TestCase):
    def test_paging_and_pool_are_different(self):
        rows = estimate(CFG, [2, 2], [17, 1], 4, pool_blocks=11)
        self.assertEqual(rows[0]['kv_effective_paged_bytes'], 3*16*32)
        self.assertEqual(rows[0]['kv_pool_allocated_bytes'], 10*16*32)
        self.assertEqual(rows[1]['kv_pool_allocated_bytes'], 0)
        self.assertIsNone(rows[0]['fits_budget'])
        self.assertIsNone(rows[0]['estimated_peak_bytes'])

    def test_slots_not_execution_batch(self):
        a = estimate(CFG, [2, 2], [17], 1)[0]
        b = estimate(CFG, [2, 2], [17], 8)[0]
        self.assertEqual(b['mamba_allocated_bytes'], 8*a['mamba_allocated_bytes'])
        self.assertEqual(a['kv_effective_paged_bytes'], b['kv_effective_paged_bytes'])

    def test_invalid_capacity_inputs(self):
        for partition, lengths, slots in [([0, 4], [1], 1), ([2, 1], [1], 1), ([2, 2], [1, 2], 1)]:
            with self.assertRaises(ValueError):
                estimate(CFG, partition, lengths, slots)


class ScheduleTests(unittest.TestCase):
    def test_chunked_prefill_and_decode(self):
        data = lambda n, c: SimpleNamespace(get_len=lambda: n, get_num_computed_tokens=lambda: c)
        groups = [SimpleNamespace(is_prompt=True, token_chunk_size=16, seq_data={7: data(100, 64)}),
                  SimpleNamespace(is_prompt=False, token_chunk_size=1, seq_data={8: data(200, 199)})]
        result = schedule_features(SimpleNamespace(virtual_engine=2, seq_group_metadata_list=groups))
        self.assertEqual(result['virtual_engine'], 2)
        self.assertEqual(result['scheduled_sequences'][0]['scheduled_query_tokens'], 16)
        self.assertEqual(result['scheduled_sequences'][0]['scheduled_context_tokens'], 64)
        self.assertEqual(result['scheduled_sequences'][1]['scheduled_query_tokens'], 1)
        self.assertIsNone(result['retained_state_count'])

    def test_empty_and_unknown_prompt_chunk(self):
        self.assertEqual(schedule_features(None), {})
        data = SimpleNamespace(get_len=lambda: 100, get_num_computed_tokens=lambda: 64)
        request = SimpleNamespace(seq_group_metadata_list=[SimpleNamespace(is_prompt=True, seq_data={1: data})])
        self.assertIsNone(schedule_features(request)['scheduled_sequences'][0]['scheduled_query_tokens'])


if __name__ == '__main__':
    unittest.main()
