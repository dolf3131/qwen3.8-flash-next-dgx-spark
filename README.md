# Qwen3.8-Flash-Next on a single DGX Spark

Serving a **170.2 GiB** checkpoint on a box with **121 GiB** of memory, by paging its
51B-parameter n-gram embedding table to SSD swap.

Measured **2026-08-27**. This is a snapshot against unmerged vLLM PRs and a temporary
image tag — see [Versions](#versions) before assuming any of it still holds.

---

## TL;DR

| | |
|---|---|
| Hardware | 1x DGX Spark (GB10, sm_121a, 121 GiB unified memory, 273 GB/s) |
| Checkpoint | `Inferact/Qwen3.8-Flash-Next-NVFP4`, 170.2 GiB |
| Resident on GPU | 76.3 GiB weights + ~16 GiB KV |
| Paged to swap | 95.37 GiB (the n-gram / PLE table) |
| Context | 524288 (YaRN factor 2 over the native 262144) |
| Speculation | in-checkpoint MTP, k=3 |
| **Decode** | **30.0 tok/s** (single stream, short prompt, reasoning on) |
| TTFT | 366–425 ms warm |

**The single thing most likely to cost you a day:** PLE CPU offload silently hangs at
TP=1. Jump to [The trap](#the-trap-ple-offload-hangs-at-tp1).

---

## The problem

Qwen3.8-Flash-Next is the open-weight preview of the Qwen4 architecture: 125B total with
**6B activated**, 48 layers as 12 × (3 × Gated DeltaNet → 1 × Qwen Sparse Attention), 512
experts with top-10 + shared routing, a vision tower, and 262144 native context.

On top of that it carries two things that do not fit the usual mental model:

- a **51B-parameter n-gram embedding table** (20M n-gram vocabulary, bigrams/trigrams
  injected at layer 2), and
- a **4B MTP layer** for speculative decoding.

180B parameters total, 335 GiB in BF16. The smallest published quantization that vLLM
will actually load is 170.2 GiB. The box has 121 GiB.

### Why it fits anyway

The n-gram table is **95.37 GiB of the 170.2**, it ships alone in
`model-00001-of-00004.safetensors`, and it is a pure lookup — each token reads a handful
of rows. Qwen's own card says as much: embeddings "are more amenable to offloading than
Mixture-of-Experts … for memory-constrained accelerators."

vLLM implements exactly that. `VLLM_PLE_CPU_OFFLOAD=1` hands the table to a dedicated CPU
process (`vllm/v1/ple_offload/worker.py`) that gathers on CPU and DMAs the result into the
GPU worker's output buffer, synchronised with `cuStreamWaitValue32`. The table is ordinary
pageable memory — *not* pinned — so the kernel pages the cold rows out to swap.

On a discrete-GPU box that offload buys VRAM. Here memory is unified, so what it actually
buys is **the right to let the kernel page the table**. That makes the swapfile
non-optional: without it the load OOMs.

```
resident on the GPU side : 170.2 - 95.37 = 74.9 GiB of weights (76.3 with MTP)
  model-0000{2,3,4}       10.05 GiB  embeddings, attention, GDN, hyper-connections,
                                     lm_head, vision tower
  nvfp4_experts-*-of-16   63.40 GiB  routed experts, NVFP4
  nvfp4_experts_mtp        1.49 GiB  MTP layer, NVFP4
in the offload process   : 95.37 GiB, mostly swapped
```

### What the offload actually costs

Almost nothing. Measured during decode:

**~73 KiB of disk reads per decoded token** — about 18 pages, matching the ~16 n-gram rows
a token touches. At NVMe latency that is roughly 2 ms against 57 ms/token, i.e. **3%**.

The bottleneck is elsewhere: the per-step CPU→GPU round trip plus 512-expert MoE and QSA
decode latency. Nothing here is bandwidth-limited the way a dense model is.

---

## The trap: PLE offload hangs at TP=1

This is undocumented and cost two full boot cycles to find.

`spawn_ple_offload()` and `wait_ple_offload_ready()` are called from
`vllm/v1/executor/multiproc_executor.py` **and from nowhere else**. `uniproc_executor.py`
has no such call. vLLM picks the uniproc executor by default at TP=1, so with
`VLLM_PLE_CPU_OFFLOAD=1` the offload worker is never spawned and the GPU side then waits
forever on a peer that does not exist.

The symptom gives you nothing:

- the log stops after `Graph capturing finished` and never reaches `Application startup`
- EngineCore spins at ~90% of one core
- **no disk I/O at all** — which is what rules out "it's just slow paging"
- the IPC socket the connector announced never appears in `/tmp`
- nothing is ever logged, and `VLLM_PLE_OFFLOAD_READY_TIMEOUT` just makes you wait longer
  for the eventual failure

Reproduced twice, including once with 36 GiB of RAM free, which is what ruled out memory
pressure as the cause.

**Fix:** `--distributed-executor-backend mp`, which forces the multiproc executor even at
one GPU.

**Diagnostic that settles it in seconds:**

```bash
docker exec <container> ps -eo pid,rss,comm
# no PleOffloadWorker process => it was never spawned
```

Upstream appears to have only ever run this multi-GPU, where TP≥2 selects the multiproc
executor on its own and the bug is invisible.

---

## Which checkpoint

Only one of the three candidates loads. This is worth knowing before you spend hours on a
download.

| Checkpoint | Size | PLE dtype | Loads? |
|---|---|---|---|
| `Inferact/Qwen3.8-Flash-Next-NVFP4` | 170.3 GiB | BF16 (95.4 GiB) | **yes** |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | 126.0 GiB | FP8 (47.7 GiB) | no |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 172.8 GiB | FP8 | body alone is ~125 GiB |

RadixArk looks strictly better — 44 GiB less to download and half the swap — but vLLM
selects its FP8 PLE path only for an `Fp8Config` checkpoint:

```python
# vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
def _get_ple_embedding_quant_method(quant_config, prefix):
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""
    if not isinstance(quant_config, Fp8Config):
        return None
```

RadixArk is `modelopt`/NVFP4, so the PLE would be built unquantized and its FP8 tensors
would not load into a BF16 parameter. Qwen's own FP8 build has the PLE that path wants,
but a ~125 GiB body that does not fit at all.

So: **Inferact**, which is also what the vLLM recipe recommends.

---

## KV is unusually cheap

Only **12 of 48 layers hold KV** — the other 36 are Gated DeltaNet, whose state is
per-sequence, not per-token. Those 12 are GQA 24/2 at head_dim 256:

```
2 kv heads x (256 + 256) x 2 bytes x 12 layers = 24 KiB/token   (28.4 measured, with MTP)
```

| max_model_len | KV pool | RAM used / available |
|---|---|---|
| 262144 | 9.6 GiB — 331,576 tok | — |
| 524288 | ~16 GiB — 573,862 tok | **104–106 / 15 GiB** |
| 1048576 | 31.9 GiB — 1,205,010 tok | 119–120 / 1–2 GiB |

**FP8 KV is refused on the stock image.** `models/qwen3_8_flash_next/nvidia/qsa.py:70`
declares `supported_kv_cache_dtypes = ["auto", "bfloat16"]` and :107 raises
`NotImplementedError("Qwen3.8-Flash-Next QSA requires a BF16 main KV cache")`. It is patchable in
principle, but not from the shipped image, so on this setup context length is the only
lever on KV size. The weights have nothing left
to give either: the experts are already NVFP4 and only ~11 GiB is BF16.

---

## Measurements

`bench_vllm.py`-style harness: four prompts, streaming, decode isolated from TTFT by
timing between the first and last chunk. **Short prompts (~30–50 tokens, so essentially
zero prefill), `temperature: 0`, `max_tokens: 600`, single stream, reasoning ON**
(the checkpoint's chat template defaults to `reasoning_effort: xhigh`).

| config | mean tok/s | four prompts | RAM used / avail |
|---|---|---|---|
| 262144, no speculation | 17.4 | 17.41 / 17.49 / 17.32 / 17.41 | — |
| 262144, MTP-3 | 29.3 | 25.37 / 35.31 / 26.09 / 30.34 | — |
| 1M, MTP-3 | 27.4 | 25.90 / 29.73 / 25.63 / 28.15 | 119–120 / 1–2 GiB |
| **524288, MTP-3** | **30.0** | 31.21 / 32.71 / 24.77 / 31.47 | **104–106 / 15 GiB** |

TTFT 366–425 ms warm, 1.1–3.7 s cold. Mean acceptance length 2.2–3.7 of a possible 4.

### Two results worth explaining

**MTP is worth 1.68x, more than its acceptance length implies.** With a mean acceptance
of ~2.5 you would expect roughly 2x from compute alone, and the usual story is that you
get somewhat less. Here you get more relative to the unspeculated baseline than the PLE
overhead would allow otherwise, because **the verify pass covers k+1 tokens in a single
forward and therefore divides the per-step PLE round trip by k+1**. On a model whose
bottleneck is a synchronous CPU gather every step, speculation buys more than usual.

**512K is faster than 1M.** Dropping 1M → 512K frees 17.7 GiB of KV. That RAM does not
show up as "free" — it becomes **page cache for the PLE table** (15 GiB of it), and decode
gets faster for it: 30.0 against 27.4. The n-gram lookups hit RAM more often instead of
NVMe. If you are memory-constrained, shortening context is not purely a sacrifice here.

---

## Reproducing

```bash
# 1. weights, 170.2 GiB
./scripts/download-weights.sh

# 2. swap for the PLE table -- NOT optional, the load OOMs without it
sudo fallocate -l 128G /swap-ple.img
sudo chmod 600 /swap-ple.img
sudo mkswap /swap-ple.img
sudo swapon -p 10 /swap-ple.img

# 3. engine
docker pull vllm/vllm-openai:qwen38-flash-next-arm64-cu130

# 4. serve
./scripts/serve.sh
```

Expect roughly 10 minutes of weight loading and a further ~3 minutes of PLE paging before
the API answers. The PLE worker loads and swaps out first; the GPU worker follows.

Useful knobs, all environment variables on `serve.sh`:

```bash
MAXLEN=1048576 GPU_UTIL=0.91 ./scripts/serve.sh   # 1M, leaves ~1-2 GiB free
SPEC=none ./scripts/serve.sh                      # unspeculated baseline
NSPEC=5 ./scripts/serve.sh                        # MTP k=5
```

---

## Known limits

This is not an optimized configuration. Things that are open:

- **Decode is slow for a 6B-active model.** ~30 tok/s against a bandwidth ceiling near 90.
  The per-step CPU→GPU PLE round trip is the leading suspect but has not been isolated
  with a profiler.
- **FP8 KV is not enabled**, and enabling it is a kernel project, not a flag. The stock
  tree refuses it in four places — `supported_kv_cache_dtypes = ["auto", "bfloat16"]` in
  both `nvidia/qsa.py:70` and `common/qsa_cache.py:658`, plus two `NotImplementedError`s
  (`qsa.py:107` "requires a BF16 main KV cache", `qsa.py:289` "requires BF16 cache
  storage"). Opening those gates is not enough: **the QSA Triton kernels contain no FP8
  path at all.** `nvidia/ops/qsa.py` has no `fp8`, `float8`, `e4m3` or dequant anywhere —
  the only `scale` in it is the softmax scale — so
  `_qsa_sparse_paged_gqa_splitk_kernel` and `_qsa_mqa_paged_kernel` load KV and use it as
  BF16 directly. Supporting FP8 means writing the dequantisation into those kernels,
  matching the store format on the write path, and then dealing with whatever shared
  memory overflows on SM121, whose budget per SM is well below a datacenter part's.

  Worth being clear about the payoff before anyone starts: at 28.4 KiB/token measured,
  halving KV against **2x the tokens is an exact wash**, so FP8 KV buys **1M context at
  today's memory footprint and today's speed**, not more speed. Keeping 524288 and
  spending the savings on PLE page cache instead would be worth perhaps 4%, extrapolating
  from the measured RAM-to-speed relationship above. And FP8's effect on quality is
  unverified for this architecture, which matters most with reasoning left at `xhigh`,
  where degradation shows up as non-termination rather than as a wrong answer.
- **Concurrency is untested past `max-num-seqs 8`.**
- **Prefix caching is on but its hit rate was not measured here.**
- **Attention backends and kernels were not swept.** FlashInfer autotune is explicitly
  disabled (`--no-enable-flashinfer-autotune`) to keep cold boot short.
- **Long-context retrieval was not verified.** 524288 is YaRN factor 2 over the native
  262144; the card sanctions up to 1M, but no needle test was run here.
- **`temperature: 0` in the benchmark** does not match the card's recommendation of 1.0
  for thinking mode. It was chosen for measurement stability, not quality.

---

## Versions

Everything below is what was running on 2026-08-27. The image tag is a temporary one that
will move; pin by digest if you care.

| | |
|---|---|
| Image | `vllm/vllm-openai:qwen38-flash-next-arm64-cu130` |
| vLLM reports | `0.1.dev20073+g8e685d198` |
| Module | `vllm.models.qwen3_8_flash_next` (a later branch than either PR below) |
| Model support | vLLM PR #53896 (open at the time) |
| PLE offload | vLLM PR #53899 (open at the time) |
| CUDA | 13.0.1 |
| Host | Ubuntu, kernel 6.17, aarch64 |

The image already carries both PRs, so no source build was needed — a plain `docker pull`
is enough, which is the main reason this configuration is reachable at all.

## License

The model weights are covered by `qwen-community-1.0`; nothing here redistributes them.
The scripts in this repository are yours to use.
