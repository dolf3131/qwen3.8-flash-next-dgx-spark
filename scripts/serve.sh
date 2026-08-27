#!/bin/bash
# Serve Qwen3.8-Flash-Next NVFP4 on a single DGX Spark (GB10 / sm_121a, 121 GiB unified),
# with the 51B n-gram (PLE) embedding table offloaded to swap.
#
# Measured 2026-08-27. See ../README.md for why each choice is what it is.
#
# The checkpoint is 170.2 GiB on a 121 GiB box. It fits because 95.37 GiB of it is one
# tensor -- the n-gram embedding table, shipped alone in model-00001-of-00004.safetensors
# -- and that tensor is a pure lookup: each token reads a handful of rows.
# VLLM_PLE_CPU_OFFLOAD=1 hands it to a dedicated CPU process which gathers on CPU and DMAs
# the result to the GPU worker. It is ordinary pageable memory, so the kernel pages the
# cold rows out to swap. Measured cost: ~73 KiB of page-ins per decoded token.
set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen38-flash-next-arm64-cu130}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.8-flash-next-nvfp4}"
NAME="${NAME:-qwen38-flash-next}"
PORT="${PORT:-8888}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"

# THE ONE THAT COSTS YOU A DAY -----------------------------------------------------
# PLE offload requires the multiproc executor, even at TP=1. spawn_ple_offload() and
# wait_ple_offload_ready() are called from vllm/v1/executor/multiproc_executor.py and
# from nowhere else -- uniproc_executor.py has no such call. vLLM picks uniproc by
# default at TP=1, so the offload worker is never spawned, the GPU side waits forever on
# a peer that does not exist, and the boot hangs after "Graph capturing finished" with
# EngineCore spinning at 90% of one core, no disk I/O, and no /tmp socket. Nothing is
# ever logged. Diagnostic: `docker exec <container> ps -eo pid,rss,comm` -- if there is
# no PleOffloadWorker process, it was never spawned.
EXECUTOR="${EXECUTOR:-mp}"

# Context. The card documents YaRN to 1M; factor 4.0 x 262144 = 1048576 exactly.
# 524288 is the default here because it is both roomier and FASTER than 1M on this box:
# 1M costs 31.9 GiB of KV against 14-16, and the RAM that frees becomes page cache for
# the PLE table, so n-gram lookups hit RAM more often. Measured 30.0 tok/s at 512K
# against 27.4 at 1M.
MAXLEN="${MAXLEN:-524288}"
ROPE="${ROPE:-yarn}"
ROPE_ARGS=()
LONG_ENV=()
if [[ "${ROPE}" != "none" && "${MAXLEN}" -gt 262144 ]]; then
  factor=$(python3 -c "print(f'{${MAXLEN}/262144:.1f}')")
  ROPE_ARGS=(--hf-overrides "{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":${factor},\"original_max_position_embeddings\":262144}}}")
  LONG_ENV=(-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
fi

# MTP lives in the checkpoint (nvfp4_experts_mtp.safetensors) and vLLM derives the draft
# config from the target. Worth 1.68x here -- more than the acceptance length (2.2-3.7 of
# 4) implies, because the verify pass covers k+1 tokens in one forward and so divides the
# per-step PLE round trip by k+1.
#   SPEC=none ./serve.sh    # unspeculated baseline, 17.4 tok/s
NSPEC="${NSPEC:-2}"
case "${SPEC:-mtp}" in
  mtp)  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NSPEC}}" ;;
  none) SPEC_CFG='' ;;
  *)    SPEC_CFG="${SPEC}" ;;
esac
SPEC_ARGS=()
[[ -n "${SPEC_CFG}" ]] && SPEC_ARGS=(--speculative-config "${SPEC_CFG}")

# KV dtype is NOT tunable on the stock image: models/qwen3_8_flash_next/nvidia/qsa.py
# declares supported_kv_cache_dtypes = ["auto", "bfloat16"] and raises
# NotImplementedError("Qwen3.8-Flash-Next QSA requires a BF16 main KV cache").
# It is patchable -- see the README's comparison section -- but not from here.
KV_DTYPE="${KV_DTYPE:-}"
KV_ARGS=()
[[ -n "${KV_DTYPE}" ]] && KV_ARGS=(--kv-cache-dtype "${KV_DTYPE}")

# Measured on this box: consumed (weights + non-torch + activation + graphs) settles at
# ~77.5 GiB, so KV = GPU_UTIL x 121.69 - 77.5. KV costs ~28.4 KiB/token with MTP.
# 0.78 -> ~16 GiB of KV, comfortably above the 14.2 GiB one 524288 request needs, and
# leaves ~15 GiB of RAM as PLE page cache. Raising it starves that cache; 0.63 already
# left KV at 0.21 GiB and refused to boot.
GPU_UTIL="${GPU_UTIL:-0.78}"
MAXSEQS="${MAXSEQS:-8}"

# Pin the KV pool: vLLM derives it from a runtime measurement that wobbles on unified
# memory (three boots of one config gave 573,862 / 591,889 / 614,423 tokens). 15.0 GiB
# leaves ~5% over the 14.3 GiB one 524288 request needs. KV_MEM= restores the old behaviour.
KV_MEM="${KV_MEM-16106127360}"
KVMEM_ARGS=()
[[ -n "${KV_MEM}" ]] && KVMEM_ARGS=(--kv-cache-memory "${KV_MEM}")

# Prefix caching needs BOTH flags on this hybrid model: without align the GDN state is not
# cacheable and the hit rate is 0 regardless of traffic. With align the attention block
# size becomes 1600 tokens, so only prompts longer than that can hit. Off by default
# because it only pays on repeated prefixes; see the README.
PREFIX_CACHE="${PREFIX_CACHE:-0}"
PREFIX_ARGS=()
[[ "${PREFIX_CACHE}" == "1" ]] && PREFIX_ARGS=(--enable-prefix-caching --mamba-cache-mode align)

# NEVER enable --async-scheduling with MTP: it makes _prepare_ngram_context read the
# optimistic -1 placeholders speculative decoding writes, so the n-gram context is wrong on
# every decode step. Silent quality loss, no crash. See the README.

# Loading 95.37 GiB into the offload process, most of it straight back out to swap, does
# not finish inside the 600 s default.
PLE_TIMEOUT="${PLE_TIMEOUT:-1800}"

[[ -f "${MODEL_DIR}/model-00001-of-00004.safetensors" ]] || {
  echo "FATAL: weights missing at ${MODEL_DIR} -- run ./download-weights.sh first" >&2; exit 1; }
swapon --show=NAME --noheadings | grep -q . || {
  echo "FATAL: no swap is active. The PLE table has nowhere to page out to and the load" >&2
  echo "  will OOM. Add ~128 GiB, e.g.:" >&2
  echo "    sudo fallocate -l 128G /swap-ple.img && sudo chmod 600 /swap-ple.img" >&2
  echo "    sudo mkswap /swap-ple.img && sudo swapon -p 10 /swap-ple.img" >&2
  exit 1; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  echo "FATAL: image ${IMAGE} not present -- docker pull ${IMAGE}" >&2; exit 1; }

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --user root \
  -p "${PORT}:${PORT}" \
  --restart unless-stopped \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  --ipc host \
  --gpus all \
  --workdir /workspace \
  -e VLLM_TARGET_DEVICE=cuda \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT="${PLE_TIMEOUT}" \
  "${LONG_ENV[@]}" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
  "${IMAGE}" \
  /model \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --distributed-executor-backend "${EXECUTOR}" \
    "${KV_ARGS[@]}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    "${KVMEM_ARGS[@]}" \
    --max-model-len "${MAXLEN}" \
    "${ROPE_ARGS[@]}" \
    --max-num-seqs "${MAXSEQS}" \
    --max-num-batched-tokens "${BATCHED_TOKENS:-8192}" \
    --enable-chunked-prefill \
    "${PREFIX_ARGS[@]}" \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image":4}' \
    "${SPEC_ARGS[@]}"

echo "started ${NAME} (executor=${EXECUTOR}, PLE offload=on, SPEC=${SPEC:-mtp}${SPEC_CFG:+ k=${NSPEC}}, maxlen=${MAXLEN}, util=${GPU_UTIL})"
echo "follow with:  docker logs -f ${NAME}"
echo "watch memory: watch -n5 'free -g; swapon --show'"
echo
echo "Expect ~10 min to load and a further ~3 min of PLE paging before the API answers."
