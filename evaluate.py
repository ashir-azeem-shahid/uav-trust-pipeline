"""
evaluate.py
===========
The three pipeline-performance tables the thesis needs. These are the
numbers a batch system structurally cannot produce, so they are the
clearest evidence that the architectural change was real.

  Table 2  latency     -- per-stage and end-to-end, p50 / p95 / p99
  Table 3  throughput  -- messages per second at 5 / 10 / 25 / 50 drones
  Table 4  detection delay -- seconds from attack onset to first correct verdict

Modes
-----
  python evaluate.py                 offline: throughput, per-stage latency,
                                     detection delay. No broker needed.
  End-to-end latency through Kafka is measured by
  tools/scale_test.py latency (producer and consumer running concurrently).

WHY PERCENTILES AND NOT AVERAGES
--------------------------------
If 99 messages take 5 ms and one takes 10 s, the mean is 105 ms, which
describes nothing that actually happened. p99 exposes the one that hurt.
That distinction is worth stating in the defence.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import config as C
import replay as R
from detect import Pipeline


def _pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    i = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[i]


# ---------------------------------------------------------------------
# Table 4 -- detection delay
# ---------------------------------------------------------------------
def detection_delay(attack: str) -> dict:
    pipe = Pipeline()
    victim = None
    first = None
    onset = None
    false_positives = []

    for _k, msg, ons in R.replay_with_onset(attack):
        onset = ons
        if msg["_gt"]["attack_target"]:
            victim = msg["_gt"]["attack_target"]
        v = pipe.handle(msg)
        if v and v["verdict"] != "QUARANTINE":
            if v["drone"] == victim and first is None:
                first = v["t_session"]
            elif v["drone"] != victim:
                false_positives.append(v["drone"])

    return {
        "attack": attack,
        "victim": victim,
        "onset_s": onset,
        "detected_at_s": first,
        "detection_delay_s": (first - onset) if first is not None else None,
        "false_positives": sorted(set(false_positives)),
    }


# ---------------------------------------------------------------------
# Table 2 -- per-stage latency (offline, pipeline-internal)
# ---------------------------------------------------------------------
def stage_latency(attack: str) -> dict:
    """
    Times each stage separately so the bottleneck is identified rather than
    assumed. The brief asks for a per-stage breakdown specifically.
    """
    from dq import DQValidator
    from rules import SuppressionDetector

    dq = DQValidator()
    sup = SuppressionDetector()
    t_dq, t_rule, t_total = [], [], []

    for _k, msg, _o in R.replay_with_onset(attack):
        t0 = time.perf_counter()
        if msg["role"] == "target":
            a = time.perf_counter()
            dq.evaluate(msg)
            t_dq.append((time.perf_counter() - a) * 1000)
        b = time.perf_counter()
        sup.handle(msg)
        t_rule.append((time.perf_counter() - b) * 1000)
        t_total.append((time.perf_counter() - t0) * 1000)

    out = {"attack": attack, "messages": len(t_total)}
    for label, vals in (("dq", t_dq), ("rule", t_rule), ("total", t_total)):
        v = sorted(vals)
        out[label] = {"p50": _pct(v, .50), "p95": _pct(v, .95),
                      "p99": _pct(v, .99), "max": v[-1] if v else float("nan")}
    return out


# ---------------------------------------------------------------------
# Table 3 -- throughput vs swarm size
# ---------------------------------------------------------------------
def throughput(attack: str, drone_counts=(5, 10, 25, 50)) -> list[dict]:
    """
    Replay only the messages involving the first N drones, and measure how
    fast the detector consumes them. Scaling the swarm rather than the
    message rate is the honest way to answer "what happens as the swarm
    grows", because message volume grows super-linearly with swarm size --
    every drone can talk to every other one.
    """
    all_msgs = []
    for _k, msg, _o in R.replay_with_onset(attack):
        all_msgs.append(msg)

    drones = sorted({m["about"] for m in all_msgs},
                    key=lambda d: (len(d), d))

    rows = []
    for n in drone_counts:
        keep = set(drones[:n])
        subset = [m for m in all_msgs
                  if m["requester"] in keep and m["contacted"] in keep]
        if not subset:
            continue
        pipe = Pipeline()
        t0 = time.perf_counter()
        for m in subset:
            pipe.handle(m)
        elapsed = time.perf_counter() - t0
        rows.append({
            "drones": n,
            "messages": len(subset),
            "seconds": round(elapsed, 4),
            "msg_per_s": round(len(subset) / elapsed, 1) if elapsed else None,
            "us_per_msg": round(elapsed / len(subset) * 1e6, 2),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kafka", action="store_true",
                    help="(retired) end-to-end latency now lives in "
                         "tools/scale_test.py latency")
    ap.add_argument("--out", default="results_performance.json")
    args = ap.parse_args()

    results: dict = {}

    print("=" * 78)
    print("  TABLE 4 -- DETECTION DELAY, TRACK A  (the headline number)")
    print("=" * 78)
    print("  Track A is the deterministic suppression rule. It targets Critical")
    print("  Node specifically -- absence of response despite proof of life.")
    print("  The other four sessions are SPECIFICITY tests: the rule staying")
    print("  silent on them is the correct result, not a miss. Those attacks are")
    print("  covered by Track B (train.py), whose per-attack F1 is Table 1.\n")
    print(f"  {'attack':<20} {'victim':<9} {'onset':>7} {'detected':>9} "
          f"{'delay':>7}  false positives")
    results["detection_delay"] = []
    for name in C.ATTACK_SESSIONS:
        d = detection_delay(name)
        results["detection_delay"].append(d)
        target_attack = name.startswith("CriticalNode")
        delay = (f"{d['detection_delay_s']}s"
                 if d["detection_delay_s"] is not None
                 else ("-" if target_attack else "n/a"))
        det = (f"{d['detected_at_s']}s" if d["detected_at_s"] is not None
               else ("MISSED" if target_attack else "silent (correct)"))
        print(f"  {name:<20} {str(d['victim']):<9} {str(d['onset_s'])+'s':>7} "
              f"{det:>9} {delay:>7}  {d['false_positives'] or 'none'}")

    print("\n" + "=" * 78)
    print("  TABLE 2 -- PER-STAGE LATENCY (ms per message)")
    print("=" * 78)
    print(f"  {'attack':<20} {'stage':<7} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}")
    results["stage_latency"] = []
    for name in C.ATTACK_SESSIONS:
        s = stage_latency(name)
        results["stage_latency"].append(s)
        for stage in ("dq", "rule", "total"):
            print(f"  {name:<20} {stage:<7} {s[stage]['p50']:>9.4f} "
                  f"{s[stage]['p95']:>9.4f} {s[stage]['p99']:>9.4f} "
                  f"{s[stage]['max']:>9.4f}")

    print("\n" + "=" * 78)
    print("  TABLE 3 -- THROUGHPUT vs SWARM SIZE (pipeline compute only, no broker)")
    print("  For throughput through Kafka see tools/scale_test.py consumers / swarm.")
    print("=" * 78)
    print(f"  {'attack':<20} {'drones':>7} {'messages':>9} {'msg/s':>12} {'us/msg':>9}")
    results["throughput"] = {}
    for name in C.ATTACK_SESSIONS:
        rows = throughput(name)
        results["throughput"][name] = rows
        for r in rows:
            print(f"  {name:<20} {r['drones']:>7} {r['messages']:>9} "
                  f"{r['msg_per_s']:>12,.0f} {r['us_per_msg']:>9.2f}")

    if args.kafka:
        # An earlier version measured end-to-end latency here by producing the
        # whole session first and only then consuming it. Every message then
        # waited for the rest to be produced, so the figure (~0.8 s) timed the
        # producer loop, not the pipeline. It was removed rather than kept
        # misleading. The correct measurement runs producer and consumer at the
        # same time at a paced rate: tools/scale_test.py latency.
        print("\n  End-to-end latency through Kafka: run")
        print("    python tools/scale_test.py latency")
        print("  (producer and consumer concurrent, paced load, per-message timing)")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
