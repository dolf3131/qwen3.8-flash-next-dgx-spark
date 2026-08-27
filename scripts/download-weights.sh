#!/bin/bash
# Download Inferact/Qwen3.8-Flash-Next-NVFP4 (~170.3 GiB, 21 shards).
#
# Qwen3.8-Flash-Next: Qwen4 architecture preview, arch Qwen4ExpForConditionalGeneration /
# qwen4_exp. 125B total with 6B activated, PLUS a 51B n-gram embedding table and a 4B MTP
# layer -- 180B params, 335 GiB in BF16. 48 layers as 12 x (3 x Gated DeltaNet -> 1 x Qwen
# Sparse Attention), 512 experts top-10 + shared, hidden 2560, GQA 24/2 at head_dim 256,
# interleaved M-RoPE, VLM, native 262144 context (YaRN factor 4 -> 1M per the card).
#
# WHY THIS BUILD, and not the two cheaper-looking ones:
#   - Inferact NVFP4 (this one): experts NVFP4, PLE left BF16. 170.3 GiB on disk, but only
#     ~74.9 GiB is resident -- the 95.4 GiB PLE table goes to the CPU offload process and
#     from there to swap. This is the build the vLLM recipe recommends.
#   - RadixArk NVFP4 (126.0 GiB, PLE in fp8 = 47.7 GiB) would halve both the download and
#     the swap, but does NOT load: vLLM picks the fp8 PLE path only for an Fp8Config
#     checkpoint. Verified inside the image at
#     vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:193 (`if not isinstance(
#     quant_config, Fp8Config): return None`). RadixArk is modelopt/NVFP4, so the PLE would
#     be built unquantized and its fp8 tensors would not load into a bf16 parameter.
#   - Qwen/Qwen3.8-Flash-Next-FP8 (172.8 GiB) has the fp8 PLE that path wants, but its
#     body is ~125 GiB, which does not fit in this box at all.
#
# MEMORY PLAN on the Spark (121 GiB unified, ~7 GiB OS):
#   body 74.9 GiB resident + KV 24 KiB/token (only 12 of 48 layers hold KV: 2 kv heads x
#   (256+256) x 2B x 12) => 6 GiB at 262144, 24 GiB at 1M. PLE 95.4 GiB lives in the
#   offload process and is paged to /swap-ple.img (128 GiB, added 2026-08-26).
#   Needs VLLM_PLE_CPU_OFFLOAD=1 and the image vllm/vllm-openai:qwen38-flash-next-arm64-cu130
#   (vllm 0.1.dev20073+g8e685d198, ships vllm/v1/ple_offload/ and the env var).
#
# Sequential curl -C -. Parallel pulls did not beat ~10 MB/s on this host and the hf
# CLI stalled at 0 B/s on Xet, so this keeps it simple and resumable.
set -uo pipefail

REPO="${REPO:-Inferact/Qwen3.8-Flash-Next-NVFP4}"
DEST="${DEST:-${MODELS_DIR:-$HOME/models}/qwen3.8-flash-next-nvfp4}"
BASE="https://huggingface.co/${REPO}/resolve/main"
mkdir -p "$DEST" || exit 1

manifest=$(curl -sL "https://huggingface.co/api/models/${REPO}?blobs=true" | python3 -c "
import sys, json
for f in json.load(sys.stdin).get('siblings', []):
    n = f['rfilename']
    if n.startswith('.'):
        continue
    print(n, f.get('size') or 0)
")

[[ -z "$manifest" ]] && { echo 'FATAL: could not read file manifest' >&2; exit 1; }

fail=0
while read -r name size; do
    [[ -z "$name" ]] && continue
    out="${DEST}/${name}"
    mkdir -p "$(dirname "$out")"
    if [[ -f "$out" ]] && [[ "$(stat -c %s "$out")" == "$size" ]]; then
        printf '  ok    %s\n' "$name"
        continue
    fi
    printf '  get   %s (%.2f GiB)\n' "$name" "$(echo "$size" | awk '{print $1/1073741824}')"
    curl -fL -C - --retry 5 --retry-delay 5 --retry-all-errors --no-progress-meter \
         -o "$out" "${BASE}/${name}" || { echo "  FAIL  $name" >&2; fail=1; }
done <<< "$manifest"

echo
if [[ "$fail" == 0 ]]; then
    echo "done -> ${DEST}  ($(du -sh "$DEST" | cut -f1))"
else
    echo "FINISHED WITH ERRORS -- rerun to resume (curl -C - continues partial files)" >&2
    exit 1
fi
