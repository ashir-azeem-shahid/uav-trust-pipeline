"""
tools/train_scale.py
====================
Does TRAINING scale? The ML models retrained on 1x, 3x, 10x and 30x the data.

WHY THIS EXISTS
---------------
tools/scale_test.py shows the streaming pipeline scales (throughput with
consumers, cost per message with swarm size). This answers the other
question an examiner asks: "you trained on ~85,000 messages -- what happens
with three or thirty times as much?"

HOW THE DATA IS GROWN
---------------------
Exactly as in the swarm-size experiment: every recorded session is cloned
under fresh drone names (Drone18 -> Drone18~c3) and the clones fly at the
same time, as one bigger swarm. So 30x means a swarm of ~1,600 drones
producing ~2.5 million messages, all of which go through the real pipeline
(DQ, rule state, ML windows) before training.

WHAT IT SHOWS, AND WHAT IT DOES NOT
-----------------------------------
Shows: the cost of the whole training path -- feature extraction through
the pipeline, fitting both models, memory -- grows in proportion to the data
(linearly), with no blow-up, and accuracy stays the same.
Does NOT show: that more data makes the models better. A clone carries no
new information, so stable accuracy is the expected and correct result.
More diverse recordings would be needed for that; the thesis says so.

Accuracy is always measured on the SAME original-size test data (clone 0
carries the original names), so the numbers are comparable across sizes:
  Isolation Forest   AUC per attack; false-alarm rate on the held-out
                     clean recording
  Random Forest      leave-one-attack-out AUC and F1 (attack type unseen)

Each size runs in a fresh process, so its peak memory is measured cleanly.

Usage
-----
  python tools/train_scale.py                       # 1 3 10 30
  python tools/train_scale.py --factors 1 3 10      # smaller machine
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux: KB


def child(k: int) -> dict:
    import numpy as np
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    import config as C
    import ml as M
    import replay as R
    from scale_test import clone

    def stream(attack, baseline=None):
        for key, msg, onset in R.replay_with_onset(attack, baseline):
            for c in range(k):
                ck, cm = clone(key, msg, c)
                yield ck, cm, onset

    # ---- 1. features, through the live pipeline -----------------------
    # Converted to arrays per session at once, so memory holds numbers,
    # not millions of Python objects.
    names = None
    data = {}
    n_msgs = 0
    t0 = time.monotonic()
    sessions = [(a, None) for a in C.ATTACK_SESSIONS] + [("_heldout", C.BASELINE_HELD_OUT)]
    for name, baseline in sessions:
        attack = "CriticalNode" if name == "_heldout" else name
        counted = []

        def counting(src):
            for item in src:
                counted.append(1)
                yield item
        ws = [(w, l) for w, l in M.session_windows(attack, baseline,
                                                   counting(stream(attack, baseline)))
              if w.features["n_requests"] > 0]
        n_msgs += len(counted)
        names = names or sorted(ws[0][0].features)
        data[name] = {
            "X": np.array([[w.features[n] for n in names] for w, _ in ws], dtype=np.float32),
            "y": np.array([l for _, l in ws], dtype=np.int8),
            "pre": np.array([w.end <= w.onset for w, _ in ws]),
            "post": np.array([w.start >= w.onset for w, _ in ws]),
            "orig": np.array(["~c" not in w.subject for w, _ in ws]),
        }
        del ws
    t_extract = time.monotonic() - t0
    n_windows = sum(len(d["y"]) for d in data.values())
    mem_after_features = _peak_mb()

    attacks = list(C.ATTACK_SESSIONS)
    first = data[attacks[0]]
    Xb = first["X"][first["pre"]]                    # k x clean baseline windows
    held = data["_heldout"]
    Xh = held["X"][held["pre"] & held["orig"]]

    # ---- 2. Isolation Forest -------------------------------------------
    t0 = time.monotonic()
    iforest = IsolationForest(n_estimators=300, random_state=0).fit(Xb)
    t_if = time.monotonic() - t0
    s_tr = -iforest.score_samples(Xb)
    thr = float(np.quantile(s_tr, 0.99))
    if_auc = {}
    for a in attacks:
        d = data[a]
        m = d["post"] & d["orig"]
        if_auc[a] = round(float(roc_auc_score(d["y"][m], -iforest.score_samples(d["X"][m]))), 4)
    if_fpr = float(np.mean(-iforest.score_samples(Xh) >= thr))

    # ---- 3. Random Forest, leave one attack out --------------------------
    rf_auc, rf_f1, rf_fit, n_train = {}, {}, [], []
    for a in attacks:
        Xtr = np.vstack([Xb] + [data[b]["X"][data[b]["post"]] for b in attacks if b != a])
        ytr = np.concatenate([np.zeros(len(Xb), dtype=np.int8)]
                             + [data[b]["y"][data[b]["post"]] for b in attacks if b != a])
        n_train.append(len(ytr))
        t0 = time.monotonic()
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    min_samples_leaf=2, random_state=0,
                                    n_jobs=-1).fit(Xtr, ytr)
        rf_fit.append(time.monotonic() - t0)
        d = data[a]
        m = d["post"] & d["orig"]
        p = rf.predict_proba(d["X"][m])[:, 1]
        rf_auc[a] = round(float(roc_auc_score(d["y"][m], p)), 4)
        rf_f1[a] = round(float(precision_recall_fscore_support(
            d["y"][m], p >= 0.5, average="binary", zero_division=0)[2]), 4)
        del Xtr, ytr, rf

    return {
        "factor": k,
        "messages": n_msgs,
        "windows": n_windows,
        "drones_in_swarm": int(54 * k),
        "feature_extraction_s": round(t_extract, 2),
        "messages_per_s": round(n_msgs / t_extract),
        "iforest_train_windows": int(len(Xb)),
        "iforest_fit_s": round(t_if, 3),
        "rf_train_windows_mean": int(sum(n_train) / len(n_train)),
        "rf_fit_s_mean": round(sum(rf_fit) / len(rf_fit), 3),
        "peak_memory_mb_features": round(mem_after_features),
        "peak_memory_mb": round(_peak_mb()),
        "iforest_auc": if_auc,
        "iforest_heldout_fpr": round(if_fpr, 4),
        "rf_loao_auc": rf_auc,
        "rf_loao_f1": rf_f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--factors", nargs="+", type=int, default=[1, 3, 10, 30])
    ap.add_argument("--out", default="results_train_scale.json")
    ap.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        print(json.dumps(child(args.child)))
        return

    import sklearn
    cpus = os.cpu_count()
    print(f"\n  Training scalability · {platform.processor() or platform.machine()} · "
          f"{cpus} CPUs · scikit-learn {sklearn.__version__}")
    print("  Each size is a fresh process; accuracy is always measured on the same "
          "original-size test data.\n")
    print("  size  messages     windows   extract   msg/s     IF fit   RF fit   "
          "peak mem   IF AUC  RF AUC  RF F1")
    rows = []
    for k in args.factors:
        r = subprocess.run([sys.executable, __file__, "--child", str(k)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {k}x failed:\n{r.stderr[-2000:]}")
            break
        row = json.loads(r.stdout.strip().splitlines()[-1])
        rows.append(row)
        mean = lambda d: sum(d.values()) / len(d)       # noqa: E731
        print(f"  {k:>3}x  {row['messages']:>10,}  {row['windows']:>8,}  "
              f"{row['feature_extraction_s']:>6.1f}s  {row['messages_per_s']:>7,}  "
              f"{row['iforest_fit_s']:>6.2f}s  {row['rf_fit_s_mean']:>6.2f}s  "
              f"{row['peak_memory_mb']:>6,} MB   {mean(row['iforest_auc']):.3f}   "
              f"{mean(row['rf_loao_auc']):.3f}   {mean(row['rf_loao_f1']):.3f}")

    if len(rows) > 1:
        a, b = rows[0], rows[-1]
        ratio = b["messages"] / a["messages"]
        print(f"\n  {ratio:.0f}x the data: feature extraction "
              f"{b['feature_extraction_s'] / a['feature_extraction_s']:.1f}x the time, "
              f"Random Forest fit {b['rf_fit_s_mean'] / a['rf_fit_s_mean']:.1f}x, "
              f"Isolation Forest fit {b['iforest_fit_s'] / a['iforest_fit_s']:.1f}x, "
              f"peak memory {b['peak_memory_mb'] / a['peak_memory_mb']:.1f}x.")
    Path(args.out).write_text(json.dumps({
        "cpus": cpus, "machine": platform.machine(),
        "sklearn": sklearn.__version__, "runs": rows}, indent=2))
    print(f"  saved {args.out}\n")


if __name__ == "__main__":
    main()
