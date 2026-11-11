"""Fix vLLM V0 SPMD worker path for hybrid models under PP (bug 5).

LocalOrDistributedWorkerBase._execute_model_spmd (worker/worker_base.py
~447) calls model_runner.prepare_model_input(seq_group_metadata_list) and
drops two arguments that the driver path (_get_driver_input_and_broadcast,
~336) passes:

  finished_requests_ids -> None reaches MambaCacheManager, which iterates it
      to free Mamba state slots: TypeError: 'NoneType' object is not iterable
      on the first step (observed 2026-11-11, results/spmd_20261111).
  virtual_engine        -> defaults to 0, so every VE replays VE0's CUDA
      graphs, which are bound to VE0's KV cache tensors (silent wrong
      KV reads/writes with PP > 1).

The replacement is the upstream body with both arguments passed through.
"""


def apply() -> None:
    from vllm.worker import worker_base
    from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                         extract_previous_hidden_states)

    orig = LocalOrDistributedWorkerBase._execute_model_spmd
    if getattr(orig, "_hybrid_pp_wrapped", False):
        return

    def _execute_model_spmd(self, execute_model_req, intermediate_tensors=None):
        assert execute_model_req is not None, (
            "_execute_model_spmd() requires each worker to take in an "
            "ExecuteModelRequest")
        worker_input = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input = self.model_runner.prepare_model_input(
            execute_model_req.seq_group_metadata_list,
            execute_model_req.virtual_engine,
            execute_model_req.finished_requests_ids)

        self.execute_worker(worker_input)

        if worker_input.num_seq_groups == 0:
            return []

        kwargs = extract_previous_hidden_states(execute_model_req)

        return self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            **kwargs,
        )

    _execute_model_spmd._hybrid_pp_wrapped = True
    LocalOrDistributedWorkerBase._execute_model_spmd = _execute_model_spmd
    assert worker_base.LocalOrDistributedWorkerBase._execute_model_spmd is _execute_model_spmd
