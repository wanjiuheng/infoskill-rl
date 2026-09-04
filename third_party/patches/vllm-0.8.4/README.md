# Patched vLLM 0.8.4 runtime

This bundle pins upstream vLLM `v0.8.4` at commit
`dc1b4a6f1300003ae27f033afbdff5e2683721ce` and adds the minimal V1 request
transport needed by INFO-SKILL Hybrid Prefix Input.

The upstream checkout is treated as immutable. The build script verifies the
commit, clean status, upstream file checksums, and patch checksum; it then
applies the patch in a temporary directory and builds a wheel with local version
`0.8.4+infoskill1`.

The upstream checksums are calculated from the pinned commit's Git blob bytes,
not from a platform-specific checkout. Verification is therefore stable across
LF and CRLF working trees, while the clean-tree check still rejects source
changes.

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
python -m pip install -r requirements-vllm-build.txt
bash scripts/build_patched_vllm.sh \
  /root/autodl-tmp/wjh/alfworld_eval/vllm-0.8.4 \
  ./dist/vllm \
  --install
```

This patch changes Python only. To avoid recompiling CUDA extensions, first put
the official vLLM 0.8.4 wheel on disk and point the build at it:

```bash
mkdir -p /root/autodl-tmp/wjh/wheelhouse
python -m pip download vllm==0.8.4 --no-deps \
  -d /root/autodl-tmp/wjh/wheelhouse
VLLM_PRECOMPILED_WHEEL_LOCATION=/root/autodl-tmp/wjh/wheelhouse/vllm-0.8.4-*.whl \
  bash scripts/build_patched_vllm.sh \
  /root/autodl-tmp/wjh/alfworld_eval/vllm-0.8.4 ./dist/vllm --install
```

Expand the wildcard to the single downloaded filename if the shell does not do
so in an environment assignment. Without this environment variable, the script
performs the slower full CUDA build.

The fast path does not use vLLM's `VLLM_USE_PRECOMPILED` build hook. That hook
can emit a Python-only wheel under `pip wheel`. Instead, INFO-SKILL treats the
official wheel as the binary base, overlays only the eight patched Python files,
rewrites wheel metadata and `RECORD`, and verifies both `vllm/_C*.so` and the
hybrid-prefix marker before installation.

The patched path is deliberately constrained to vLLM V1, CPU transport of the
short prefix tensor, `enforce_eager=True`, disabled prefix caching, and no prompt
adapter. Qwen LoRA remains supported because it is a model adapter rather than a
vLLM prompt adapter.

After installation, run the single-GPU numerical gate before any M1 training:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/hybrid_prefix_parity.py \
  --model /absolute/path/to/Qwen2.5-7B-Instruct \
  --output hybrid-prefix-parity.json
```

The gate loads the 7B model once with Transformers, evaluates all fixed cases,
releases it, and then loads it once with vLLM. Its schema-v2 report separates a
same-vLLM transport gate (plain token IDs versus equivalent token embeddings)
from a multi-prompt, multi-seed Transformers/vLLM numerical gate. The default
criteria are 100% first-token agreement, transport maximum logprob error at most
`1e-4`, cross-backend P95 at most `0.05`, and cross-backend maximum at most
`0.10`.
