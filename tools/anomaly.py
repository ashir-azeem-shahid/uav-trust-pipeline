"""
tools/anomaly.py
================
Unsupervised anomaly detection: an Isolation Forest trained ONLY on clean
baseline traffic, then asked to flag attack traffic it has never seen.

WHY THIS EXISTS
---------------
The supervised Random Forest (train.py) learns from labelled examples of each
attack. It can only recognise what it was shown. The thesis assignment asks
for "machine learning models for anomaly detection" as well as
classification -- that is, detecting traffic that departs from normal without
being told what an attack looks like. That is the realistic setting for a new
attack nobody has labelled yet.

So this model is fitted on windows from the clean baseline recording only.
No attack window, and no label, is ever shown to it. Every attack it is then
tested on is, by construction, unseen.

WHAT IS MEASURED
----------------
For each attack session, on the windows after the attack begins:

  AUC      threshold-free: how well the anomaly score ranks attacked windows
           above clean ones. 0.5 = no better than chance, 1.0 = perfect.
  F1 etc.  at a threshold fixed from the TRAINING data alone, so that 5% of
           clean training windows score as anomalous (--alpha). No attack
           data touches the threshold.

And on clean traffic:

  FPR held-out   the false-alarm rate on the SECOND baseline recording,
                 which was never used for fitting or for the threshold.
                 This is the honest false-alarm number.

Two feature sets, so the contribution of this thesis's added features can be
read off directly, exactly as in the supervised ablation:

  full         all features (trust statistics + DQ scores + suppression)
  trust_only   the previous thesis's trust statistics only

Window modes: "15s" (the real streaming task) and "session" (one window per
observer per phase, the mode the supervised comparison table uses).

The window straddling the attack onset contains both clean and attack
traffic, so it is excluded from both training and testing.

Repeated over --seeds random seeds; mean and standard deviation reported.

Usage
-----
  python tools/anomaly.py                     # both modes, 5 seeds
  python tools/anomaly.py --modes 15s --seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score  # noqa: E402

import config as C  # noqa: E402
import features as F  # noqa: E402

VARIANTS = {
    "full": dict(include_dq=True, include_suppression=True),
    "trust_only": dict(include_dq=False, include_suppression=False),
}


def _clean_part(rows):
    """Windows that end before the attack starts (training material)."""
    out = []
    for r in rows:
        if r["post_onset"]:
            continue
        if r["window_end"] is not None and r["window_end"] > r["onset"]:
            continue            # straddles the onset: mixed traffic
        out.append(r)
    return out


def _attack_part(rows):
    return [r for r in rows if r["post_onset"]]


def _matrix(rows, names):
    leaked = set(names) & C.FORBIDDEN_FEATURES
    if leaked:
        raise AssertionError(f"ground truth leaked into features: {leaked}")
    return np.array([[r["features"][n] for n in names] for r in rows], dtype=float)


def _mean_std(values):
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0}


def run_mode(mode: str, seeds: int, alpha: float) -> dict:
    t0 = time.monotonic()
    # The clean prefix is the same baseline recording in every replay, so it
    # is taken once, from any session.
    train_rows = _clean_part(F.build_windows("CriticalNode", mode))
    held_rows = _clean_part(F.build_windows("CriticalNode", mode,
                                            baseline=C.BASELINE_HELD_OUT))
    attack_rows = {a: _attack_part(F.build_windows(a, mode))
                   for a in C.ATTACK_SESSIONS}
    print(f"  [{mode}] windows built in {time.monotonic() - t0:.1f}s: "
          f"train {len(train_rows)} clean, held-out {len(held_rows)} clean, "
          + ", ".join(f"{a} {len(r)} ({sum(x['label'] for x in r)} attacked)"
                      for a, r in attack_rows.items()))

    out = {"mode": mode, "alpha": alpha, "seeds": seeds,
           "n_train": len(train_rows), "n_heldout": len(held_rows),
           "variants": {}}

    for vname, opts in VARIANTS.items():
        names = F.feature_names(train_rows, **opts)
        Xtr = _matrix(train_rows, names)
        Xho = _matrix(held_rows, names)
        per = {a: {"auc": [], "precision": [], "recall": [], "f1": [],
                   "flagged_rate": []} for a in attack_rows}
        fpr_train, fpr_held = [], []

        for seed in range(seeds):
            clf = IsolationForest(n_estimators=300, random_state=seed).fit(Xtr)
            s_tr = -clf.score_samples(Xtr)           # higher = more anomalous
            thr = float(np.quantile(s_tr, 1 - alpha))
            fpr_train.append(float(np.mean(s_tr > thr)))
            fpr_held.append(float(np.mean(-clf.score_samples(Xho) > thr)))

            for a, rows in attack_rows.items():
                y = np.array([r["label"] for r in rows])
                s = -clf.score_samples(_matrix(rows, names))
                pred = s > thr
                auc = (float(roc_auc_score(y, s))
                       if 0 < y.sum() < len(y) else float("nan"))
                p, r, f, _ = precision_recall_fscore_support(
                    y, pred, average="binary", zero_division=0)
                per[a]["auc"].append(auc)
                per[a]["precision"].append(float(p))
                per[a]["recall"].append(float(r))
                per[a]["f1"].append(float(f))
                per[a]["flagged_rate"].append(float(pred.mean()))

        out["variants"][vname] = {
            "n_features": len(names),
            "fpr_train": _mean_std(fpr_train),
            "fpr_heldout": _mean_std(fpr_held),
            "per_attack": {a: {
                "n_windows": len(attack_rows[a]),
                "n_attacked": int(sum(x["label"] for x in attack_rows[a])),
                **{k: _mean_std(v) for k, v in m.items()}}
                for a, m in per.items()},
        }
    return out


def _fmt(ms, pct=False):
    if ms["mean"] is None:
        return "     n/a     "
    if pct:
        return f"{ms['mean']:6.1%} ±{ms['std']:5.1%}"
    return f"{ms['mean']:.3f} ±{ms['std']:.3f}"


def report(res: dict) -> None:
    print(f"\n  ===== window mode: {res['mode']} · threshold = top {res['alpha']:.0%} "
          f"of clean training scores · {res['seeds']} seeds =====")
    for vname, v in res["variants"].items():
        print(f"\n  {vname}  ({v['n_features']} features)   false alarms: "
              f"train {_fmt(v['fpr_train'], True)}   held-out baseline "
              f"{_fmt(v['fpr_heldout'], True)}")
        print("  attack               windows  AUC            F1             "
              "recall         precision")
        for a, m in v["per_attack"].items():
            print(f"  {a:<20} {m['n_attacked']:>3}/{m['n_windows']:<4} "
                  f"{_fmt(m['auc'])}  {_fmt(m['f1'])}  {_fmt(m['recall'])}  "
                  f"{_fmt(m['precision'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", nargs="+", default=["15s", "session"],
                    choices=["15s", "session"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="share of clean training windows allowed to score anomalous")
    ap.add_argument("--out", default="results_anomaly.json")
    args = ap.parse_args()

    import sklearn
    print(f"\n  Isolation Forest · trained on clean baseline only · "
          f"scikit-learn {sklearn.__version__}\n")
    results = {"model": "IsolationForest(n_estimators=300)",
               "trained_on": "clean baseline windows only (no attack, no labels)",
               "sklearn": sklearn.__version__, "modes": []}
    for mode in args.modes:
        r = run_mode(mode, args.seeds, args.alpha)
        report(r)
        results["modes"].append(r)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n  saved {args.out}\n")


if __name__ == "__main__":
    main()
