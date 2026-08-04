#!/usr/bin/env python3
"""Quality gate for Kimi-K3 changes that are NOT bit-exact.

Design notes -- this gate is meant to be hard to fool, including by the person
running it:

  * THRESHOLDS ARE PRE-REGISTERED in this file. Read them before you run, and
    change them only in a separate commit with a stated reason. Tightening or
    loosening a bound after seeing a result is the whole failure mode.
  * NO SUBJECTIVE JUDGEMENT. "the text still looks fine" is not a criterion.
    Every check is mechanical and returns a number.
  * BOTH teacher-forced AND free-running. Teacher-forced metrics systematically
    understate generation drift, because one flipped token changes all
    subsequent context. A change may only pass if it passes both.
  * ROUTING IS SCORED SEPARATELY. This is a top-16 MoE: a last-ulp change in
    router scores reorders borderline experts while logit correlation stays at
    1.0000. Correlation-style metrics cannot see that, so they are reported as
    diagnostics and are NOT pass/fail criteria.
  * THE GATE MUST BE SHOWN ABLE TO REJECT. `--negative-control` perturbs the
    baseline by a known amount and asserts the gate FAILS. A gate that has
    never rejected anything is not evidence.
  * THE GATE MUST BE CALIBRATED. `--calibrate` compares two runs of the SAME
    build and asserts a perfect score. If that does not hold, the harness (or
    the engine) is nondeterministic and no result from it means anything.
  * THE CHANGE MUST HAVE ENGAGED. --assert-differs requires the candidate to
    actually differ somewhere; a candidate that silently fell back to the old
    path would otherwise "pass" trivially.

Usage:
  k3_quality.py --base DIR --cand DIR [--assert-differs]
  k3_quality.py --calibrate DIR_A DIR_B
  k3_quality.py --negative-control DIR
"""
import argparse, json, os, sys, glob
import numpy as np

VOCAB = 163840

# ---- PRE-REGISTERED THRESHOLDS ------------------------------------------
# Rationale for each bound is stated so a later reader can judge whether a
# change to it is honest. These are deliberately strict: the fallback is a
# byte-exact build that already works.
TH = {
    # Teacher-forced: fraction of positions whose greedy token is unchanged.
    # This is the metric that actually predicts whether generation diverges.
    "top1_agree_min":      0.995,
    # How far the ranking moves. Catches changes that keep the argmax by luck.
    "top5_overlap_min":    0.995,
    # Distributional distance at temperature 1.
    "kl_mean_max":         1.0e-4,
    "kl_max_max":          1.0e-2,
    # MoE-specific: fraction of (layer, token) top-16 expert SETS unchanged.
    # A routing change is a different computation even when text looks fine.
    "expert_agree_min":    0.99,
    # Free-running: mean tokens generated before the first divergence from the
    # baseline completion, as a fraction of the completion length. 1.0 means
    # identical output.
    "gen_first_div_min":   0.90,
    # Degeneracy guards -- these catch a "passing" change that has actually
    # collapsed the model, which agreement metrics alone can miss.
    "max_repeat_4gram":    0.15,   # fraction of 4-grams that are repeats
    "min_distinct_tokens": 0.25,   # distinct/total token ratio
}

def load_logits(d):
    p = os.path.join(d, "logits.bin")
    if not os.path.exists(p): return None
    a = np.fromfile(p, dtype=np.float32)
    n = a.size // VOCAB
    if n == 0: return None
    return a[:n*VOCAB].reshape(n, VOCAB)

def load_gens(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "gen-*.txt"))):
        out[os.path.basename(p)[4:-4]] = open(p, "rb").read()
    return out

def load_routes(d):
    p = os.path.join(d, "routes.bin")
    if not os.path.exists(p): return None
    return np.fromfile(p, dtype=np.uint32)

def softmax64(x):
    x = x.astype(np.float64)
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)

def degeneracy(text_bytes):
    """Repetition / diversity guards on a completion, token-free (byte 4-grams
    over whitespace-split words, which is enough to catch collapse)."""
    w = text_bytes.split()
    if len(w) < 8: return 0.0, 1.0
    grams = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    rep = 1.0 - (len(set(grams)) / max(1, len(grams)))
    distinct = len(set(w)) / max(1, len(w))
    return rep, distinct

def score(base, cand, label="candidate"):
    res, checks = {}, []
    lb, lc = load_logits(base), load_logits(cand)
    if lb is None or lc is None:
        print("FATAL: missing logits.bin in one of the directories"); sys.exit(2)
    n = min(len(lb), len(lc)); lb, lc = lb[:n], lc[:n]
    res["positions"] = n
    if n < 32:
        print(f"FATAL: only {n} teacher-forced positions; need >=32 for a "
              f"meaningful rate"); sys.exit(2)

    if not np.isfinite(lc).all():
        print("FATAL: candidate logits contain NaN/Inf"); sys.exit(2)

    tb, tc = lb.argmax(1), lc.argmax(1)
    res["top1_agree"] = float((tb == tc).mean())
    checks.append(("top1_agree", res["top1_agree"], TH["top1_agree_min"], "min"))

    k = 5
    ib = np.argpartition(-lb, k, axis=1)[:, :k]
    ic = np.argpartition(-lc, k, axis=1)[:, :k]
    ov = np.mean([len(set(ib[i]) & set(ic[i])) / k for i in range(n)])
    res["top5_overlap"] = float(ov)
    checks.append(("top5_overlap", ov, TH["top5_overlap_min"], "min"))

    pb, pc = softmax64(lb), softmax64(lc)
    kl = (pb * (np.log(pb + 1e-300) - np.log(pc + 1e-300))).sum(1)
    res["kl_mean"], res["kl_max"] = float(kl.mean()), float(kl.max())
    checks.append(("kl_mean", res["kl_mean"], TH["kl_mean_max"], "max"))
    checks.append(("kl_max",  res["kl_max"],  TH["kl_max_max"],  "max"))

    # diagnostics only -- deliberately NOT pass/fail (see module docstring)
    d = np.abs(lb - lc)
    res["max_abs_dlogit"] = float(d.max())
    fb, fc = lb.ravel(), lc.ravel()
    res["pcc"] = float(np.corrcoef(fb, fc)[0, 1])
    res["bit_identical"] = bool((lb == lc).all())

    rb, rc = load_routes(base), load_routes(cand)
    if rb is not None and rc is not None and rb.size == rc.size and rb.size:
        agree = float((rb == rc).mean())
        res["expert_agree"] = agree
        checks.append(("expert_agree", agree, TH["expert_agree_min"], "min"))
    else:
        res["expert_agree"] = None

    gb, gc = load_gens(base), load_gens(cand)
    common = sorted(set(gb) & set(gc))
    if common:
        fracs, reps, dists = [], [], []
        for kx in common:
            a, b = gb[kx].split(), gc[kx].split()
            m = min(len(a), len(b)); i = 0
            while i < m and a[i] == b[i]: i += 1
            fracs.append(i / max(1, len(a)))
            r, dv = degeneracy(gc[kx]); reps.append(r); dists.append(dv)
        res["gen_prompts"] = len(common)
        res["gen_first_div"] = float(np.mean(fracs))
        res["gen_first_div_worst"] = float(np.min(fracs))
        res["max_repeat_4gram"] = float(np.max(reps))
        res["min_distinct"] = float(np.min(dists))
        checks.append(("gen_first_div", res["gen_first_div"], TH["gen_first_div_min"], "min"))
        checks.append(("max_repeat_4gram", res["max_repeat_4gram"], TH["max_repeat_4gram"], "max"))
        checks.append(("min_distinct", res["min_distinct"], TH["min_distinct_tokens"], "min"))
    else:
        res["gen_prompts"] = 0
        print("WARNING: no free-running completions found; teacher-forced only.")
        print("         A change may NOT be accepted on teacher-forced metrics alone.")

    print(f"=== quality scorecard: {label} ===")
    print(f"  teacher-forced positions : {res['positions']}")
    print(f"  free-running prompts     : {res['gen_prompts']}")
    print(f"  bit-identical            : {res['bit_identical']}")
    print("  --- diagnostics (not pass/fail) ---")
    print(f"  PCC                      : {res['pcc']:.9f}")
    print(f"  max |dlogit|             : {res['max_abs_dlogit']:.6e}")
    print("  --- criteria ---")
    ok = True
    for name, val, bound, sense in checks:
        good = (val >= bound) if sense == "min" else (val <= bound)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<18} {val:.6f}  "
              f"({'>=' if sense=='min' else '<='} {bound})")
    if res.get("expert_agree") is None:
        print("  WARN  expert_agree        not captured (set K3_ROUTE_STATS)")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--cand")
    ap.add_argument("--calibrate", nargs=2)
    ap.add_argument("--negative-control")
    ap.add_argument("--assert-differs", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    if a.calibrate:
        print("CALIBRATION: two runs of the SAME build must score perfectly.")
        ok, res = score(a.calibrate[0], a.calibrate[1], "calibration (same build)")
        if not res["bit_identical"]:
            print("CALIBRATION FAILED: the engine is not reproducible run-to-run,")
            print("so no comparison from this harness can be trusted."); sys.exit(3)
        print("CALIBRATION OK: engine reproduces bit-exactly.")
        sys.exit(0 if ok else 3)

    if a.negative_control:
        d = a.negative_control
        lb = load_logits(d)
        if lb is None: print("FATAL: no logits"); sys.exit(2)
        os.makedirs("/tmp/k3nc", exist_ok=True)
        rng = np.random.default_rng(1234)
        # A perturbation small enough that PCC stays ~1.0 -- exactly the kind of
        # change a correlation-based gate would wave through.
        pert = lb + rng.normal(0, 0.02, lb.shape).astype(np.float32)
        pert.tofile("/tmp/k3nc/logits.bin")
        for f in glob.glob(os.path.join(d, "gen-*.txt")):
            open(os.path.join("/tmp/k3nc", os.path.basename(f)), "wb").write(open(f,"rb").read())
        print("NEGATIVE CONTROL: baseline + N(0, 0.02) logit noise must FAIL.")
        ok, res = score(d, "/tmp/k3nc", "negative control")
        print(f"  (PCC was {res['pcc']:.9f} -- note a correlation gate would have passed this)")
        if ok:
            print("NEGATIVE CONTROL FAILED: the gate accepted a knowingly perturbed")
            print("model. The thresholds are too loose to be evidence of anything.")
            sys.exit(3)
        print("NEGATIVE CONTROL OK: the gate rejects a perturbed model.")
        sys.exit(0)

    if not (a.base and a.cand): ap.error("need --base and --cand")
    ok, res = score(a.base, a.cand)
    if a.assert_differs and res["bit_identical"]:
        print("FAIL: candidate is bit-identical to baseline -- the change did not")
        print("      engage. A silent fallback would otherwise 'pass' trivially.")
        sys.exit(3)
    if a.json: open(a.json, "w").write(json.dumps(res, indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
