#!/usr/bin/env python3
"""Exact-token prefill and decode timing for the K3 vLLM server."""

import argparse
import csv
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

BASE_TEXT = (
    "The history of computing is a history of abstraction. Each layer hides "
    "the one beneath it, and every leak in that abstraction becomes a bug "
    "someone must eventually understand. Databases index records so queries "
    "remain predictable as the stored collection grows. "
)


def post_json(host, port, path, body, timeout=1800):
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def base_tokens(host, port, model):
    response = post_json(
        host,
        port,
        "/tokenize",
        {
            "model": model,
            "prompt": BASE_TEXT,
            "add_special_tokens": False,
        },
    )
    tokens = response["tokens"]
    if not tokens:
        raise RuntimeError("tokenizer returned no tokens for benchmark text")
    return tokens


def exact_prompt(tokens, target):
    return (tokens * math.ceil(target / len(tokens)))[:target]


def post_stream(host, port, prompt, max_tokens, model="kimi-k3"):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft = None
    previous = None
    gaps = []
    output = []
    usage = {}
    with urllib.request.urlopen(request, timeout=86400 * 30) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text", "")
            if not text:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - started
            elif previous is not None:
                gaps.append(now - previous)
            previous = now
            output.append(text)
    elapsed = time.perf_counter() - started
    return {
        "ttft_s": ttft,
        "elapsed_s": elapsed,
        "gaps_s": gaps,
        "text": "".join(output),
        "usage": usage,
    }


def write_results(path, rows):
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2) + "\n")
    csv_path = target.with_suffix(".csv")
    fields = [
        "requested_tokens",
        "prompt_tokens",
        "completion_tokens",
        "rep",
        "ttft_s",
        "prefill_tok_s",
        "decode_ms_tok",
        "decode_tok_s",
        "elapsed_s",
        "text_prefix",
    ]
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--ctx",
        default="128,512,2048,8192,32768,131072,524288,1048544",
    )
    parser.add_argument("--gen", type=int, default=32)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--long-reps", type=int, default=1)
    parser.add_argument("--long-threshold", type=int, default=524288)
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    seed_tokens = base_tokens(args.host, args.port, args.model)
    rows = []
    print(
        f"{'ctx':>8} {'rep':>4} {'TTFT s':>10} {'prefill tok/s':>14} "
        f"{'decode ms/tok':>14} {'decode tok/s':>13}"
    )
    for context in [int(value) for value in args.ctx.split(",")]:
        repetitions = args.long_reps if context >= args.long_threshold else args.reps
        prompt = exact_prompt(seed_tokens, context)
        for rep in range(repetitions):
            result = post_stream(args.host, args.port, prompt, args.gen, args.model)
            ttft = result["ttft_s"]
            if ttft is None:
                raise RuntimeError(f"no tokens returned for context {context}")
            usage = result["usage"]
            prompt_tokens = int(usage.get("prompt_tokens", context))
            completion_tokens = int(
                usage.get("completion_tokens", len(result["gaps_s"]) + 1)
            )
            if prompt_tokens != context:
                raise RuntimeError(
                    f"requested {context} prompt tokens, server used {prompt_tokens}"
                )
            gaps = result["gaps_s"]
            decode = statistics.median(gaps) if gaps else float("nan")
            row = {
                "requested_tokens": context,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "rep": rep,
                "ttft_s": ttft,
                "prefill_tok_s": prompt_tokens / ttft,
                "decode_ms_tok": decode * 1000,
                "decode_tok_s": 1 / decode if gaps else 0,
                "elapsed_s": result["elapsed_s"],
                "text_prefix": result["text"][:80],
            }
            rows.append(row)
            write_results(args.output, rows)
            print(
                f"{context:>8} {rep:>4} {ttft:>10.3f} "
                f"{row['prefill_tok_s']:>14.2f} "
                f"{row['decode_ms_tok']:>14.2f} "
                f"{row['decode_tok_s']:>13.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
