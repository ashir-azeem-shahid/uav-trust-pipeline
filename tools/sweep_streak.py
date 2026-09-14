"""
tools/sweep_streak.py
=====================
Sensitivity sweep for the streak-based suppression rule, run through a
faithful simulation of the partitioned consumer.

Each simulated consumer sees ONLY its own partition and its own local
state -- no global view, no lookahead. If it fires here it fires in
detect.py.

Produces the threshold sensitivity table for the evaluation chapter, and
the max-streak separation table, which is the clearest single piece of
evidence for the headline claim.

Run:  python tools/sweep_streak.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
import replay as R
from rules import SuppressionDetector
from simulate import partition_of

STREAKS = [5, 8, 10, 12, 15, 20, 25, 30]


def run_once(attack: str, streak: int):
    """Replay one attack through partitioned consumers; report verdicts."""
    consumers: dict[int, SuppressionDetector] = defaultdict(
        lambda: SuppressionDetector(streak))

    victim = None
    onset = None
    first_fire: dict[str, int] = {}
    pre_onset: dict[str, int] = {}
    reasons: dict[str, str] = {}

    for key, msg, ons in R.replay_with_onset(attack):
        onset = ons
        if msg["_gt"]["attack_target"]:
            victim = msg["_gt"]["attack_target"]

        fired, reason = consumers[partition_of(key)].handle(msg)
        if fired:
            t = msg["t_session"]
            if t < onset:
                pre_onset.setdefault(key, t)
            elif key not in first_fire:
                first_fire[key] = t
                reasons[key] = reason

    # max streak reached by each drone, across all consumers
    maxes = {}
    for det in consumers.values():
        for d, s in det.snapshot().items():
            maxes[d] = s["max_streak"]

    fp = [d for d in first_fire if d != victim]
    return {
        "attack": attack, "victim": victim, "onset": onset,
        "hit": victim in first_fire,
        "delay": (first_fire[victim] - onset) if victim in first_fire else None,
        "fp": fp, "fp_pre": sorted(pre_onset),
        "reason": reasons.get(victim, ""),
        "max_streaks": maxes,
    }


def sweep():
    print("=" * 78)
    print("  THRESHOLD SENSITIVITY -- streak-based suppression rule")
    print("=" * 78)
    print("  hits   = victims correctly detected, out of 5 sessions")
    print("  delay  = mean seconds from known attack onset to first correct verdict")
    print("  FP     = total distinct non-victim drones flagged, all sessions")
    print("  FP_pre = total flagged during clean pre-attack traffic\n")
    print(f"  {'streak':>7} {'hits':>7} {'mean delay':>12} {'FP':>6} {'FP_pre':>8}")
    print("  " + "-" * 46)

    best = []
    for s in STREAKS:
        res = [run_once(a, s) for a in C.ATTACK_SESSIONS]
        hits = sum(r["hit"] for r in res)
        delays = [r["delay"] for r in res if r["delay"] is not None]
        md = sum(delays) / len(delays) if delays else float("nan")
        fp = sum(len(r["fp"]) for r in res)
        fpp = sum(len(r["fp_pre"]) for r in res)
        print(f"  {s:>7} {hits:>5}/5 {md:>11.1f}s {fp:>6} {fpp:>8}")
        best.append((s, hits, md, fp, fpp, res))
    return best


def detail(streak: int):
    print(f"\n{'=' * 78}")
    print(f"  PER-SESSION DETAIL AT streak={streak}")
    print(f"{'=' * 78}")
    print(f"  {'attack':<20} {'victim':<9} {'hit':<5} {'delay':>7} {'FP':>4}  FP drones")
    for a in C.ATTACK_SESSIONS:
        r = run_once(a, streak)
        d = f"{r['delay']}s" if r["delay"] is not None else "-"
        print(f"  {a:<20} {str(r['victim']):<9} {'YES' if r['hit'] else 'no':<5} "
              f"{d:>7} {len(r['fp']):>4}  {r['fp'] if r['fp'] else ''}")
        if r["fp_pre"]:
            print(f"  {'':<20} pre-onset false positives: {r['fp_pre']}")


def separation():
    """
    The max-streak table. This is the single clearest piece of evidence:
    how long the victim goes unanswered versus the worst healthy drone.
    """
    print(f"\n{'=' * 78}")
    print("  MAX STREAK SEPARATION  (longest run of unanswered requests as target)")
    print("=" * 78)
    print(f"  {'attack':<20} {'victim':<9} {'victim streak':>14} "
          f"{'worst other':>12} {'margin':>8}")
    print("  " + "-" * 68)
    for a in C.ATTACK_SESSIONS:
        r = run_once(a, 10 ** 9)     # never fires; we only want the streaks
        ms = r["max_streaks"]
        v = ms.get(r["victim"], 0)
        others = {d: s for d, s in ms.items() if d != r["victim"]}
        worst_d = max(others, key=others.get) if others else None
        worst = others.get(worst_d, 0)
        print(f"  {a:<20} {str(r['victim']):<9} {v:>14} "
              f"{worst:>7} ({worst_d}) {v - worst:>7}")

    print("\n  Baseline sessions have no victim; their worst streaks are the")
    print("  false-positive risk the threshold has to clear:")
    for name, path in (("baseline.csv", C.BASELINE),
                       ("20260510_Baseline", C.BASELINE_HELD_OUT)):
        det = SuppressionDetector(10 ** 9)
        import pandas as pd
        df = pd.read_csv(path)
        df["_t"] = df[C.C_TIME].map(R._to_seconds)
        df["_t"] -= df["_t"].min()
        for _, row in df.sort_values("_t", kind="stable").iterrows():
            for role, key in (("target", str(row[C.C_RES])),
                              ("initiator", str(row[C.C_REQ]))):
                det.handle({"about": key, "role": role,
                            "t_session": int(row["_t"]),
                            "exchange_type": str(row[C.C_EXCHANGE]),
                            "_gt": {"attack_target": None}})
        snap = det.snapshot()
        worst_d = max(snap, key=lambda d: snap[d]["max_streak"])
        print(f"    {name:<22} worst streak {snap[worst_d]['max_streak']:>3} "
              f"({worst_d})")


if __name__ == "__main__":
    sweep()
    separation()
    for s in (15, 20):
        detail(s)
