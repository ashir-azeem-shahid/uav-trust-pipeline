"""
tools/sweep.py
==============
Threshold sensitivity sweep for the suppression rule.

Why this exists: the session-level result (victim 100% timeout rate as
target, next-highest 75%) is computed over hundreds of rows. Inside a
15-second window a drone may only have a handful of target exchanges, and
at a ~50% baseline timeout rate "3 of 3 timed out" happens by chance
several times a session. So the rule needs a minimum-evidence guard, and
this finds where it belongs.

Two knobs:
  min_target_rows -- how much evidence a window must hold before the rule
                     is allowed to fire at all
  timeout_pct     -- the share of target exchanges that must have timed out

Output is the sensitivity table for the evaluation chapter, and the chosen
operating point for detect.py.

Run:  python tools/sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
from simulate import run

MIN_ROWS = [3, 5, 8, 10, 12, 15, 20]
PCTS = [100.0, 90.0, 80.0]


def sweep():
    print("Sweeping the suppression rule. Columns:")
    print("  hit    = victim detected")
    print("  delay  = seconds from attack onset to first correct verdict")
    print("  fp     = distinct non-victim drones the rule fired on")
    print("  fp_pre = distinct drones it fired on during clean pre-attack traffic\n")

    rows = []
    for pct in PCTS:
        for m in MIN_ROWS:
            line = {"timeout_pct": pct, "min_target_rows": m,
                    "hits": 0, "delays": [], "fp": 0, "fp_pre": 0}
            for name in C.ATTACK_SESSIONS:
                r = run(name, min_target_rows=m, timeout_pct=pct, verbose=False)
                hit = r["victim"] in r["fired_on"]
                line["hits"] += int(hit)
                if r["detection_delay_s"] is not None:
                    line["delays"].append(r["detection_delay_s"])
                line["fp"] += len([d for d in r["fired_on"] if d != r["victim"]])
                line["fp_pre"] += len(r["false_positives_pre_onset"])
            rows.append(line)

    print(f"  {'pct':>5} {'min_rows':>9} {'hits':>6} {'mean delay':>11} "
          f"{'fp total':>9} {'fp_pre total':>13}")
    print("  " + "-" * 60)
    for r in rows:
        md = (sum(r["delays"]) / len(r["delays"])) if r["delays"] else float("nan")
        print(f"  {r['timeout_pct']:>5.0f} {r['min_target_rows']:>9} "
              f"{r['hits']:>4}/5 {md:>10.1f}s {r['fp']:>9} {r['fp_pre']:>13}")

    return rows


def detail(min_rows: int, pct: float = 100.0):
    print(f"\n{'=' * 76}")
    print(f"  DETAIL AT min_target_rows={min_rows}, timeout_pct={pct:.0f}")
    print(f"{'=' * 76}")
    print(f"  {'attack':<20} {'victim':<10} {'hit':<5} {'delay':>7} {'fp':>4}  false-positive drones")
    for name in C.ATTACK_SESSIONS:
        r = run(name, min_target_rows=min_rows, timeout_pct=pct, verbose=False)
        hit = "YES" if r["victim"] in r["fired_on"] else "no"
        d = f"{r['detection_delay_s']}s" if r["detection_delay_s"] is not None else "-"
        fp = [x for x in r["fired_on"] if x != r["victim"]]
        print(f"  {name:<20} {str(r['victim']):<10} {hit:<5} {d:>7} {len(fp):>4}  "
              f"{fp if fp else ''}")
        if r["false_positives_pre_onset"]:
            print(f"  {'':<20} {'':<10} pre-onset FPs: {r['false_positives_pre_onset']}")


if __name__ == "__main__":
    sweep()
    for m in (10, 12, 15):
        detail(m)
