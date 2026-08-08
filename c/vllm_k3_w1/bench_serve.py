#!/usr/bin/env python3
"""Prefill and decode timing for the K3 vLLM server, B=1, across contexts.

Separates the two phases the way they actually cost: TTFT is prefill (plus one
decode step), and the mean inter-token gap after that is decode. Streaming, so
the split is measured rather than inferred from a total.

Dependency-free on purpose -- this runs inside the serving container.

Usage:
    python3 bench_serve.py [--host H] [--port P] [--ctx 128,512,2048,8192]
                           [--gen 32] [--reps 2]
"""

import argparse
import json
import statistics
import time
import urllib.request


def post_stream(host, port, prompt, max_tokens, model="kimi-k3"):
    """Returns (ttft, [inter-token gaps], text, n_tokens)."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, prev, gaps, out = None, None, [], []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            tok = d["choices"][0].get("text", "")
            if tok == "":
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                gaps.append(now - prev)
            prev = now
            out.append(tok)
    return ttft, gaps, "".join(out), len(out)


def make_prompt(tok_target):
    """Roughly `tok_target` tokens of ordinary prose, deterministic."""
    base = ("The history of computing is a history of abstraction. Each layer "
            "hides the one beneath it, and every leak in that abstraction "
            "becomes a bug someone must eventually understand. ")
    # ~40 tokens per repetition; overshoot slightly and let the server truncate
    # nothing -- we report the server's own prompt token count.
    return "Summarize the following.\n\n" + base * max(1, tok_target // 38)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", default="128,512,2048,8192")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--model", default="kimi-k3")
    a = ap.parse_args()

    print(f"{'ctx':>7} {'rep':>4} {'TTFT s':>9} {'prefill tok/s':>14} "
          f"{'decode ms/tok':>14} {'decode tok/s':>13}")
    rows = {}
    for ctx in [int(x) for x in a.ctx.split(",")]:
        p = make_prompt(ctx)
        for rep in range(a.reps):
            ttft, gaps, text, n = post_stream(a.host, a.port, p, a.gen, a.model)
            if ttft is None:
                print(f"{ctx:>7} {rep:>4}   no tokens returned")
                continue
            dec = statistics.median(gaps) if gaps else float("nan")
            print(f"{ctx:>7} {rep:>4} {ttft:>9.3f} {ctx / ttft:>14.1f} "
                  f"{dec * 1e3:>14.1f} {1 / dec if gaps else 0:>13.2f}")
            # First rep warms caches; report the last.
            rows[ctx] = (ttft, dec, text, n)
    print("\nsummary (last rep per context)")
    print(f"{'ctx':>7} {'TTFT s':>9} {'decode tok/s':>13}   first 60 chars")
    for ctx, (ttft, dec, text, n) in sorted(rows.items()):
        print(f"{ctx:>7} {ttft:>9.3f} {1 / dec if dec == dec else 0:>13.2f}   "
              f"{text[:60]!r}")


if __name__ == "__main__":
    main()
