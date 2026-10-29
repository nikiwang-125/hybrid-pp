# Dev image for the shared lab boxes: vLLM 0.10.1.1 plus our pins.
#   bash scripts/docker_run.sh build
FROM vllm/vllm-openai:v0.10.1.1

# vllm 0.10.1.1 has no upper bound on transformers; 5.x breaks it.
RUN pip install --no-cache-dir "transformers>=4.55,<4.56" pandas matplotlib

# The repo is mounted at /workspace/hybrid-pp and imported via PYTHONPATH.
# vLLM finds our plugin through the vllm.general_plugins entry point, so
# register it with a stub dist-info; `pip install -e` would write egg-info
# into the mounted repo on every container start.
RUN SP=$(python3 -c "import site; print(site.getsitepackages()[0])") \
 && D="$SP/hybrid_pp_plugin-0.1.0.dist-info" && mkdir -p "$D" \
 && printf 'Metadata-Version: 2.1\nName: hybrid-pp-plugin\nVersion: 0.1.0\n' > "$D/METADATA" \
 && printf '[vllm.general_plugins]\nhybrid_pp = hybrid_pp.plugin:register\n' > "$D/entry_points.txt"
ENV PYTHONPATH=/workspace/hybrid-pp \
    HF_HOME=/hf_cache \
    VLLM_USE_V1=0

WORKDIR /workspace/hybrid-pp
ENTRYPOINT []
CMD ["bash"]
