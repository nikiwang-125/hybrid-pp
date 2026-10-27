"""Start / stop a ``vllm serve`` process and wait until it is healthy."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request


def serve_command(model: str, port: int, pp: int, tp: int, max_model_len: int,
                  max_num_seqs: int, gpu_mem_util: float, eager: bool,
                  extra_args: list[str]) -> list[str]:
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--pipeline-parallel-size", str(pp),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--mamba-ssm-cache-dtype", "float32",
        "--trust-remote-code",
    ]
    if eager:
        cmd.append("--enforce-eager")
    return cmd + list(extra_args)


class Server:

    def __init__(self, cmd: list[str], env: dict, log_path: str, port: int,
                 timeout_s: int = 1200):
        self.cmd, self.env, self.log_path = cmd, env, log_path
        self.port, self.timeout_s = port, timeout_s
        self.proc = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self.log = open(self.log_path, "w")
        self.log.write("# " + " ".join(self.cmd) + "\n")
        self.log.flush()
        # Own process group, so stop() also kills the worker processes.
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.log,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self._wait_healthy()
        return self

    def __exit__(self, *exc):
        self.stop()

    def _wait_healthy(self):
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with {self.proc.returncode}; "
                    f"see {self.log_path}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(5)
        self.stop()
        raise TimeoutError(f"server not healthy after {self.timeout_s}s; "
                           f"see {self.log_path}")

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait()
        self.log.close()
