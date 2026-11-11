"""CPU-only schedule metadata extraction; never reads GPU tensors or prompt text."""


def schedule_features(request):
    if request is None:
        return {}
    rows = []
    for group in getattr(request, 'seq_group_metadata_list', []) or []:
        for seq_id, data in (getattr(group, 'seq_data', {}) or {}).items():
            length = data.get_len()
            computed = data.get_num_computed_tokens()
            is_prompt = bool(group.is_prompt)
            chunk = getattr(group, 'token_chunk_size', None)
            query = min(chunk, max(0, length - computed)) if is_prompt and chunk is not None else (None if is_prompt else 1)
            rows.append({'sequence_id': seq_id, 'is_prompt': is_prompt,
                         'sequence_length': length, 'computed_tokens': computed,
                         'scheduled_query_tokens': query,
                         'scheduled_context_tokens': computed if is_prompt else max(0, length - 1)})
    return {'virtual_engine': getattr(request, 'virtual_engine', None),
            'scheduled_sequences': rows,
            'scheduled_sequence_count': len(rows),
            'retained_state_count': None,
            'schedule_semantics': 'CPU scheduler metadata; graph padding and multistep execution require separate validation'}
