#!/usr/bin/env python3
"""Measure prefill throughput: prompt_tokens / TTFT at several context sizes.

bench_vllm.py uses ~40-token prompts, so it measures decode only. This sends long
prompts with max_tokens=1 and streams, so the time to the first chunk is the prefill.
Prompt text is real prose/code (not a repeated string), because a degenerate prompt
concentrates the n-gram lookups and flatters the PLE path.
"""
import json, os, sys, time, urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-flash-next")
TARGETS = [int(x) for x in os.environ.get("TARGETS", "8192,16384,32768").split(",")]


def post(path, payload, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def ntok(text):
    r = json.load(post("/tokenize", {"model": MODEL, "prompt": text}))
    return r.get("count") or len(r.get("tokens", []))


def build_corpus():
    """Concatenate whatever real text is around into one long, non-repetitive string."""
    parts = []
    for p in ("/tmp/q38fn-README.md", "/tmp/blazux-how.md", "/tmp/blazux-readme.md"):
        try:
            parts.append(open(p, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    if not parts:
        sys.exit("no corpus files found")
    text = "\n\n".join(parts)
    # tile with a varying marker so repeats are not byte-identical
    out, i = [], 0
    while sum(len(x) for x in out) < 2_000_000:
        out.append(f"\n\n<!-- section {i} -->\n\n" + text)
        i += 1
    return "".join(out)


def main():
    corpus = build_corpus()
    # calibrate chars-per-token once on a sample
    sample = corpus[:40000]
    cpt = len(sample) / ntok(sample)
    print(f"corpus {len(corpus):,} chars | {cpt:.2f} chars/token\n")
    print(f"{'target':>8} {'prompt tok':>11} {'TTFT s':>8} {'prefill tok/s':>14}")
    for tgt in TARGETS:
        text = corpus[: int(tgt * cpt)]
        prompt = text + "\n\nReply with the single word: ok"
        body = {"model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        t0 = time.perf_counter()
        first = None
        usage = None
        with post("/v1/chat/completions", body, stream=True) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                if d.get("choices") and first is None:
                    first = time.perf_counter()
        if first is None:
            print(f"{tgt:>8} -- no tokens")
            continue
        ttft = first - t0
        ptok = usage["prompt_tokens"] if usage else 0
        print(f"{tgt:>8} {ptok:>11,} {ttft:>8.2f} {ptok/ttft:>14,.0f}")


if __name__ == "__main__":
    main()
