"""Enable vLLM's CUTE-DSL skinny GEMM for Qwen3.8-Flash-Next at TP=1 on GB10 (sm_121).

vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py already carries a fast decode
GEMM path, but it is unreachable here for two independent reasons:

  1. `enable_qwen38next_low_latency_gemm` returns immediately unless `_is_sm103()`, i.e.
     only on sm_103. The kernel itself is fine on sm_121 -- `shape_dynamic_skinny_gemm.
     is_available()` returns True and results match `F.linear` -- so this gate is a
     conservative enablement, not a hardware limit.
  2. Every shape in QWEN38NEXT_GEMM_PLANS is a TP=4 local shape. At TP=1 the local shapes
     are 4x larger (lm_head is (248320, 2560), not (62080, 2560)), so nothing matches and
     every bf16 linear falls through to cuBLAS.

Measured on this box, the fallback is cuBLAS's `cutlass_80_wmma_tensorop_bf16` kernel,
which a torch profiler trace puts at 55% of decode GPU time, with the batch-1 lm_head gemv
(the MTP drafts) at a further 18%.

This patch relaxes the gate to include sm_121 and adds TP=1 plan entries whose configs were
found by sweeping block_size x outputs_per_block x k_unroll x vector_width against
F.linear on the real weight shapes. Only entries that measured FASTER than cuBLAS are
included -- notably (640, 2560) at M=1 measured 0.87x, so that combination is deliberately
left to cuBLAS.

The plan table becomes TP=1-specific: the (640, 2560) key is overridden. Do not use this
patch in a TP>1 deployment.
"""
import glob, sys

cands = glob.glob(
    "/usr/local/lib/python3.*/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py"
)
if not cands:
    print("patch_skinny_tp1: FAILED -- low_latency_gemm.py not found", file=sys.stderr)
    raise SystemExit(1)
TARGET = cands[0]

GATE_ANCHOR = """def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))"""
GATE_REPL = """def _is_sm103() -> bool:
    # local patch: sm_121 (GB10) also runs this kernel correctly and faster than cuBLAS
    return current_platform.is_device_capability(
        (10, 3)
    ) or current_platform.is_device_capability((12, 1))"""

PLANS_ANCHOR = "def _is_sm103() -> bool:"
PLANS_EXTRA = '''# --- local patch: TP=1 shapes for a single GB10, measured against F.linear ---
# speedups at (M=1 / M=3): see the module docstring for how these were found.
QWEN38NEXT_GEMM_PLANS.update(
    {
        # GDN fused in_proj, x36 per forward. 1.43x / 1.05x
        (10240, 2560): {
            1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4),
            3: SkinnyGemmConfig(3, 128, 1, k_unroll=4, vector_width=4),
        },
        # GDN/QSA output projection, x48. 1.05x / 1.07x
        (2560, 6144): {
            1: SkinnyGemmConfig(1, 32, 1, k_unroll=2),
            3: SkinnyGemmConfig(3, 32, 1, k_unroll=2),
        },
        # QSA fused QKV/gate, x36. 1.23x / 1.02x
        (6144, 2560): {
            1: SkinnyGemmConfig(1, 32, 4, k_unroll=2),
            3: SkinnyGemmConfig(3, 32, 2, k_unroll=2, vector_width=4),
        },
        # GDN fused QKVZ, x12. 1.38x / 1.06x
        (12288, 2560): {
            1: SkinnyGemmConfig(1, 32, 4, k_unroll=2, vector_width=4),
            3: SkinnyGemmConfig(3, 32, 4, k_unroll=2, vector_width=4),
        },
        # HyperConnection down/inject, x97 -- the biggest relative win. 2.20x / 1.92x
        (320, 10240): {
            1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
            3: SkinnyGemmConfig(3, 64, 1, k_unroll=4),
        },
        # QSA indexer Q/K and MoE gate, x108. M=1 measured 0.87x -> left to cuBLAS.
        (640, 2560): {
            3: SkinnyGemmConfig(3, 128, 2, k_unroll=4, vector_width=4),
        },
        # LM head, x1 per forward but 1.27 GiB of weights. 1.40x / 1.05x
        (248320, 2560): {
            1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
            3: SkinnyGemmConfig(3, 64, 1, k_unroll=4),
        },
    }
)
# --- end local patch ---


'''

src = open(TARGET, encoding="utf-8").read()
if "local patch: TP=1 shapes" in src:
    print("patch_skinny_tp1: already applied")
    raise SystemExit(0)

for name, anchor in (("gate", GATE_ANCHOR), ("plans", PLANS_ANCHOR)):
    if src.count(anchor) != 1:
        print(f"patch_skinny_tp1: FAILED -- {name} anchor found {src.count(anchor)} times",
              file=sys.stderr)
        raise SystemExit(1)

src = src.replace(PLANS_ANCHOR, PLANS_EXTRA + PLANS_ANCHOR, 1)
src = src.replace(GATE_ANCHOR, GATE_REPL, 1)
open(TARGET, "w", encoding="utf-8").write(src)
print("patch_skinny_tp1: applied")
