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
| Speculation | in-checkpoint MTP, k=2 |
| Engine | stock image **plus one local patch** that unblocks vLLM's own fast decode GEMM (below) |
| **Decode** | **32.7 tok/s** warm, 30.9 first run after boot |
| **Prefill** | **2,719 tok/s** at 30k tokens |
| Concurrency | ~102 tok/s aggregate at 8 concurrent requests |
| Repeated prefix | TTFT **5.4x** better with prefix caching (see below — needs two flags, not one) |
| TTFT | 280–390 ms warm on short prompts |

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

RadixArk looks strictly better — 44 GiB less to download and half the swap — and its PLE
is a faithful quantization of the official BF16 table (cosine 0.9996, verified by another
operator against the source rows). It is rejected because vLLM selects its FP8 PLE path
only for an `Fp8Config` checkpoint:

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

**Correction, 2026-08-27:** RadixArk *does* load if that gate is widened to accept
modelopt — but it then emits garbage, which is still being bisected
([issue #1](../../issues/1), with ~20 hypotheses eliminated there and here). So the
practical answer is unchanged, but "does not load" was the wrong reason. The one isolated
difference found so far: RadixArk excludes `*.self_attn.*` as a wildcard, leaving the QSA
`q/k/v/o_proj` in BF16, while Inferact quantizes them and excludes only 117 `self_attn`
norm/indexer sub-modules. That accounts for the full 2.5 B-parameter gap between the two
bodies, though it is the wrong direction to explain corruption on its own.

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

Decode: four prompts, streaming, decode isolated from TTFT by timing between the first and
last chunk. **Short prompts (~40–90 tokens, so essentially zero prefill), `temperature: 0`,
`max_tokens: 600`, single stream, reasoning ON** (the checkpoint's chat template defaults to
`reasoning_effort: xhigh`). Prefill: long prompts of real prose/code with `max_tokens: 1`,
measured as `prompt_tokens / TTFT`.

### Decode

| config | tok/s (first run / warm) | steps/s | mean acceptance |
|---|---|---|---|
| 262144, no speculation | 17.4 | — | — |
| 1M, MTP-3 | 27.4 | — | 2.81 |
| 524288, MTP-3 | 26.0 / 27.3 | 12.3 | 2.42 |
| 524288, MTP-2 | 26.9 / 28.2 | 13.19 | 2.14 |
| **524288, MTP-2, skinny-GEMM patch** | **30.9 / 32.7** | **14.04** | 2.33 |

**Compare configurations by steps/s, not tok/s.** Throughput is `acceptance x step rate`,
and acceptance wanders between boots of one identical config (2.14–2.33 observed), moving
tok/s by ~10% with nothing actually different. The three patched boots gave 14.11 / 14.06 /
14.04 steps/s; the tok/s from those same boots ranged 30.4–32.8.

**A number in an earlier version of this file did not survive re-measurement.** The 524288
MTP-3 row was first recorded at 30.0 tok/s. Three further boots of that identical
configuration produced 26.0 / 27.3 with byte-identical streamed-chunk counts — the same
generated sequence every time — while the original boot produced a different sequence with
acceptance 2.81. It has not recurred in three attempts and I cannot explain it. The
reproducible figures are the ones above; treat single-boot numbers on this model with
suspicion, including anyone else's.

Within a boot the benchmark is perfectly deterministic (identical chunk counts across runs);
across boots the generated sequence can differ, and since throughput is
`acceptance x step rate`, a sequence difference moves the headline number by ~10% without
anything being wrong.

### Prefill

| prompt | TTFT | prefill |
|---|---|---|
| 8,068 tok | 3.72 s | 2,171 tok/s |
| 16,539 tok | 7.50 s | 2,206 tok/s |
| 30,054 tok | 11.05 s | **2,719 tok/s** |

Prefill is the axis where this model's sparse attention pays off, and it is the reason to
be on vLLM rather than a GGUF at all — QSA prefill kernels do not exist in llama.cpp.

## Two results worth explaining

**MTP: k=2, not k=3.** vLLM logs per-position acceptance. Over 5,928 drafts at k=3 this
configuration gives **0.682 / 0.445 / 0.299**, decaying by a steady factor of 0.66 per
position. Solving the two measured points (17.4 unspeculated, 27.5 at k=3) for the draft
cost puts one MTP forward at **10.5% of a target forward**, which makes the curve computable:

| k | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| predicted tok/s | 26.5 | **28.6** | 27.5 | 25.7 | 24.0 |

Predicted +3.9% for k=2 over k=3; **measured +3.1% warm and +3.4% cold**.

The draft cost is better measured directly than fitted: the step-time difference between
k=2 and k=3 on otherwise identical builds is **10.1 ms, or 20% of the verify forward** —
twice what the curve fit above implied. With that number, k=3 buys +10.7% acceptance for
+14.2% step time, a net **-3.0%**. This still holds after the GEMM patch below, even though
that patch helps the M=1 drafts more than the M=3 verify and should therefore shift the
optimum upward. k=2 and k=3 are within the boot-to-boot band, so k=2 is kept as the
measured median rather than as a decisive winner.

**512K is faster than 1M.** Dropping 1M → 512K frees 17.7 GiB of KV. That RAM does not show
up as "free" — it becomes page cache for the PLE table — and decode gets faster for it. Note
the effect does not scale down linearly: pinning the KV pool later returned a further
1.6 GiB and produced no measurable gain, so page cache is not a lever you can keep pulling.

## Where the decode time actually goes

A torch profiler trace of ~117 decode steps attributes GPU time as:

| | share |
|---|---|
| dense bf16 GEMM (`cutlass_80_wmma` 55% + batch-1 lm_head gemv 18%) | **73%** |
| MoE grouped GEMM | 18.7% |
| MoE routing, QSA, GDN, hyperconnection, everything else | **< 5% combined** |

Two things follow, and both contradicted what looked obvious beforehand.

**The per-step PLE host↔device sync is not the bottleneck.** It is the natural suspect —
the offload gathers on CPU and the GPU waits on a `cuStreamWaitValue32` every step, and
independent write-ups have called removing it the obvious next optimisation. Forcing the
connector down its existing dummy path (a measurement-only image; the output is wrong by
construction) changed the **step rate by 0.6%**. It is worth 0.4 ms of a 75.8 ms step.
Throughput did rise 8% in that ablation, but only because zeroing the n-gram table makes
the model repeat itself and repetition is *easier to draft* — an acceptance artifact, and a
quality symptom rather than a speed gain.

**vLLM already ships the fast GEMM for this model; it is gated off here twice over.**
`models/qwen3_8_flash_next/nvidia/low_latency_gemm.py` carries a CUTE-DSL skinny GEMM with a
per-shape plan table, but `enable_qwen38next_low_latency_gemm` returns immediately unless
`_is_sm103()`, and every key in `QWEN38NEXT_GEMM_PLANS` is a **TP=4** local shape — its
lm_head entry is `(62080, 2560)`, which at TP=1 is `(248320, 2560)`. So on a single Spark
nothing matches even if the gate passes, and every bf16 linear falls back to cuBLAS's
Ampere-era `cutlass_80_wmma` kernel.

The kernel itself is fine on sm_121: `shape_dynamic_skinny_gemm.is_available()` is True and
its results match `F.linear`. `scripts/patch-skinny-gemm-tp1.py` relaxes the gate and adds
TP=1 plan entries whose configs were swept against cuBLAS on the real weight shapes.
Measured per-shape, at M=1 (drafts) / M=3 (verify):

| shape | per forward | M=1 | M=3 |
|---|---|---|---|
| `(320, 10240)` hyperconnection down | ×97 | **2.20x** | **1.92x** |
| `(10240, 2560)` GDN in_proj | ×36 | 1.43x | 1.05x |
| `(248320, 2560)` lm_head | ×1 | 1.40x | 1.05x |
| `(12288, 2560)` GDN qkvz | ×12 | 1.38x | 1.06x |
| `(6144, 2560)` QSA qkv | ×36 | 1.23x | 1.02x |
| `(640, 2560)` indexer / MoE gate | ×108 | **0.87x** | 1.60x |

`(640, 2560)` at M=1 is slower than cuBLAS and is deliberately left out of the plan. End to
end this is worth **+6.9% on the step rate**, reproduced across three boots. The microbench
predicted +17%; isolated timing loops flatter the kernel, so trust the end-to-end number.

This is a patch, not a fix. It is TP=1-specific and it overrides one existing plan key, so
do not carry it into a multi-GPU deployment. It exists because GB10 is not in vLLM's CI and
nobody upstream appears to have one — the TP=4 plans are there because somebody tuned on a
four-GPU box.

**MoE backends are already at the best supported option.** `auto` resolves to
`FLASHINFER_CUTLASS`; the two higher-priority backends both refuse with *"kernel does not
support current device cuda"* on sm_121. Backend validation runs before weight loading, so a
bad `--moe-backend` fails in about two minutes rather than thirteen.

## Configuration notes that cost real time

**`--async-scheduling`: do not enable it with MTP.** Reported upstream on 2026-08-27
(PR #53896): under async scheduling `_prepare_ngram_context` reads the CPU token mirror
while it still holds the optimistic `-1` placeholders that speculative decoding writes
before acceptance is known, so the n-gram context is wrong on **every** decode step — 154 of
154 measured bad in that report, on ROCm. It is a silent failure, not a crash: no log, no
benchmark signal. I could not measure any output difference from the flag on this box —
outputs did change, but a control boot with the flag *off* reproduced the same change, so
that was boot-to-boot variation and not the flag. Absence of evidence at this measurement
floor, not evidence of absence; the upstream code analysis is specific and there is no
reason to want the flag.

**Prefix caching needs two flags.** `--enable-prefix-caching` alone is inert on this model:
`mamba_cache_mode` defaults to `"none"` and nothing promotes it, so the GDN state is never
cacheable and blocks are never reusable. Measured that way: **0 hits in 813 queries**. With
`--mamba-cache-mode align` the attention block size is forced to **1600 tokens** to match the
mamba page size, so any prompt shorter than 1600 tokens still cannot hit — which is why
short-prompt benchmarks report 0% and conclude the feature is broken. With both flags and an
8,053-token prompt repeated:

| request | TTFT | prefill | hits |
|---|---|---|---|
| 1 | 5.41 s | 1,497 tok/s | 0 |
| 2 | 3.81 s | 2,128 tok/s | 0 |
| 3 | **1.01 s** | **8,015 tok/s** | **6,400 tokens** |

Decode is unaffected (28.19 without, 28.25 with). Another GB10 operator reports the
cached-block path crashing a GDN `in_proj` GEMM with `CUBLAS_STATUS_INTERNAL_ERROR`; that did
not reproduce here with `align` set, and may be the half-enabled configuration above.

**Pin the KV pool.** vLLM sizes KV as `(util x total)` minus a *runtime measurement* of
consumed memory, and on unified memory that measurement wobbles with whatever the page cache
and the offload worker happen to hold. Three boots of one identical config produced
573,862 / 591,889 / 614,423 tokens. `--kv-cache-memory` makes it deterministic; the excess is
dead weight, since one 524288 request needs ~14.3 GiB at the measured 28.6 KiB/token.

## Reproducing

```bash
# 1. weights, 170.2 GiB
./scripts/download-weights.sh

# 2. swap for the PLE table -- NOT optional, the load OOMs without it
sudo fallocate -l 128G /swap-ple.img
sudo chmod 600 /swap-ple.img
sudo mkswap /swap-ple.img
sudo swapon -p 10 /swap-ple.img

# 3. engine, plus the TP=1 skinny-GEMM patch (worth +6.9% on the step rate)
docker pull vllm/vllm-openai:qwen38-flash-next-arm64-cu130
docker build -t vllm-skinny-tp1:v1 -f scripts/Dockerfile.skinny-gemm scripts/

# 4. serve
./scripts/serve.sh
```

`serve.sh` defaults to the patched image. `VLLM_IMAGE=vllm/vllm-openai:qwen38-flash-next-arm64-cu130 ./scripts/serve.sh`
runs the stock one if you would rather not carry the patch.

Expect roughly 10 minutes of weight loading and a further ~3 minutes of PLE paging before
the API answers. The PLE worker loads and swaps out first; the GPU worker follows.

Useful knobs, all environment variables on `serve.sh`:

```bash
MAXLEN=1048576 GPU_UTIL=0.91 KV_MEM= ./scripts/serve.sh  # 1M, leaves ~1-2 GiB free
SPEC=none ./scripts/serve.sh                            # unspeculated baseline
NSPEC=3 ./scripts/serve.sh                              # MTP k=3 (k=2 is the default)
PREFIX_CACHE=1 ./scripts/serve.sh                       # + --mamba-cache-mode align
```

Measuring:

```bash
BENCH_MODEL=qwen3.8-flash-next python3 scripts/bench-prefill.py   # prefill tok/s by ctx
```

---

## Known limits

This is not an optimized configuration. Things that are open:

- **Decode is still slow for a 6B-active model.** ~33 tok/s against a bandwidth ceiling near
  90. It is now profiled rather than guessed at: 73% of decode GPU time is dense bf16 GEMM,
  and the part of that reachable without touching kernels has been taken. What is left needs
  either quantised attention/GDN weights (a checkpoint that does not exist, and a quality
  question), or upstream kernel work.
- **The skinny-GEMM configs were swept coarsely.** A finer grid over
  `block_size x outputs_per_block x k_unroll x vector_width`, and a second look at
  `(640, 2560)` at M=1, could plausibly find more.
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
- **The 5.4x prefix-caching win was measured on one repeated prompt**, not on a realistic
  agent or multi-turn trace, and hits only appeared on the third identical request.
- **Attention backends and kernels were not swept.** FlashInfer autotune is explicitly
  disabled (`--no-enable-flashinfer-autotune`) to keep cold boot short.
- **Long-context retrieval was not verified.** 524288 is YaRN factor 2 over the native
  262144; the card sanctions up to 1M, but no needle test was run here.
- **`temperature: 0` in the benchmark** does not match the card's recommendation of 1.0
  for thinking mode. It was chosen for measurement stability, not quality.
- **Boot-to-boot sequence variation is unexplained.** Identical configurations sometimes
  generate a different continuation, which moves decode throughput by ~10% through
  acceptance length. Within a boot the engine is bit-deterministic at `temperature=0`.
  Ruled out: executor choice, KV pool size, prefix caching, the image, `--async-scheduling`,
  and torch.compile (`inductor_compile_config` is empty, so no max_autotune, and the model's
  own Triton kernels carry zero `@triton.autotune` — compilation is deterministic and
  persisting its cache would not help). Not ruled out: allocation alignment, cuBLAS
  first-call algorithm selection.
- **Nothing here is profiled.** The per-step CPU→GPU PLE round trip is the leading suspect
  for decode being ~30 tok/s against a ~90 tok/s bandwidth ceiling, but that is an
  inference from the arithmetic, not a measurement.

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
