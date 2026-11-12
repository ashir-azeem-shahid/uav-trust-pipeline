"""
train_ml.py
===========
Trains the two live Track B models, evaluates them honestly, and writes the
deployable bundle models/ml.joblib that detect.py --ml loads.

  Isolation Forest  anomaly detection, fitted on clean baseline windows only
  Random Forest     trustworthiness classification, supervised

All windows come from ml.session_windows(), i.e. from the live Pipeline
itself, so what is evaluated here is exactly what runs on the stream.

HOW IT IS EVALUATED (results_ml.json)
-------------------------------------
Two levels, because they answer different questions:

  window level   one prediction per (drone, 15 s window).
                 AUC, precision, recall, F1. Mean +- std over --seeds.
  drone level    what the OPERATOR sees: the alert logic of ml.MLScorer
                 replayed over the session in time order. Was the attacked
                 drone alerted, how long after the attack began, and how
                 many honest drones were alerted too.

Splits, each chosen so nothing tested was seen in training:

  Isolation Forest   fitted on the clean baseline recording; tested on every
                     attack session and on a SECOND clean recording
                     (held-out baseline) for the honest false-alarm rate.
  Random Forest      leave-one-attack-out: trained on the baseline plus the
                     other four attack sessions, tested on the one left out.
                     So every number is for an attack TYPE the model has
                     never seen -- the hard, realistic case.
                     Plus DataManipulation -> DataManipulation2: the same
                     attack type in a different recording (the known-attack
                     case).

The window straddling the attack onset mixes clean and attack traffic and is
left out of training and window-level scoring.

THE DEPLOYED MODEL
------------------
The Random Forest in models/ml.joblib is trained on every session EXCEPT
those in --exclude (default: CriticalNode). The live demo replays Critical
Node, so the demo shows the model working on a recording it never saw.

Usage
-----
  python train_ml.py                       # evaluate + write models/ml.joblib
  python train_ml.py --exclude none        # deploy on all sessions
  python train_ml.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import config as C
import ml as M

IF_TREES = 300
RF_TREES = 300


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------
def _scored(w) -> bool:
    return w.features["n_requests"] > 0           # MLScorer skips the rest


def _pre(ws):
    """Windows that end at or before the onset: clean traffic."""
    return [(w, l) for w, l in ws if _scored(w) and w.end <= w.onset]


def _post(ws):
    return [(w, l) for w, l in ws if _scored(w) and w.start >= w.onset]


def load_all() -> dict:
    t0 = time.monotonic()
    data = {a: M.session_windows(a) for a in C.ATTACK_SESSIONS}
    data["_heldout"] = M.session_windows("CriticalNode", C.BASELINE_HELD_OUT)
    n = sum(len(v) for v in data.values())
    print(f"  {n:,} windows from {len(data)} replays in "
          f"{time.monotonic() - t0:.1f}s (through the live pipeline code)")
    return data


def _X(ws, names):
    return np.array([[w.features[n] for n in names] for w, _ in ws], dtype=float)


def _y(ws):
    return np.array([l for _, l in ws])


def fit_iforest(base, names, alpha, seed):
    t0 = time.monotonic()
    clf = IsolationForest(n_estimators=IF_TREES, random_state=seed).fit(_X(base, names))
    s = -clf.score_samples(_X(base, names))
    thr = float(np.quantile(s, 1 - alpha))
    return clf, thr, time.monotonic() - t0


def fit_rf(rows, names, seed):
    t0 = time.monotonic()
    rf = RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced",
                                min_samples_leaf=2, random_state=seed,
                                n_jobs=-1).fit(_X(rows, names), _y(rows))
    return rf, time.monotonic() - t0


# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------
def window_metrics(y, score, thr) -> dict:
    pred = score >= thr
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                 zero_division=0)
    auc = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    return {"auc": auc, "precision": float(p), "recall": float(r), "f1": float(f),
            "flagged_rate": float(pred.mean())}


def drone_level(ws, bundle, which: str) -> dict:
    """Replay the live alert logic over one session, in time order."""
    sc = M.MLScorer(bundle=bundle)
    alerts = [v for v in sc.verdicts([w for w, _ in ws])
              if v["track"] == which]          # SCORE readings have track "B"
    labels = {}
    for w, l in ws:
        labels[w.subject] = labels.get(w.subject, 0) | l
    onset = next((w.onset for w, _ in ws if w.onset is not None), None)
    attacked = {d for d, l in labels.items() if l}
    hit = [a for a in alerts if a["drone"] in attacked and a["t_session"] > onset]
    false_post = [a for a in alerts if a["drone"] not in attacked
                  and a["window_start"] >= onset]
    false_pre = [a for a in alerts if a["window_start"] < onset]
    return {
        "attacked_drones": len(attacked),
        "attacked_alerted": len({a["drone"] for a in hit}),
        "first_delay_s": min((a["detection_delay_s"] for a in hit), default=None),
        "honest_drones_alerted_during_attack": len(false_post),
        "alerts_before_onset": len(false_pre),
        "drones_total": len(labels),
    }


def _ms(vals):
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0}


def _bundle(names, iforest, if_thr, rf, rf_thr, persist, info=None):
    return {"feature_names": names, "iforest": iforest, "if_threshold": if_thr,
            "rf": rf, "rf_threshold": rf_thr, "if_persist": persist,
            "info": info or {}}


# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------
def evaluate(data, names, args) -> dict:
    attacks = list(C.ATTACK_SESSIONS)
    base = _pre(data[attacks[0]])          # same baseline prefix in every replay
    held = _pre(data["_heldout"])
    out = {"iforest": {}, "rf_leave_one_attack_out": {}, "rf_known_attack": {}}

    # ---- Isolation Forest ------------------------------------------
    per = {a: [] for a in attacks}
    fpr_held = []
    for seed in range(args.seeds):
        clf, thr, _ = fit_iforest(base, names, args.alpha, seed)
        fpr_held.append(float(np.mean(-clf.score_samples(_X(held, names)) >= thr)))
        for a in attacks:
            P = _post(data[a])
            per[a].append(window_metrics(_y(P), -clf.score_samples(_X(P, names)), thr))
    clf, thr, _ = fit_iforest(base, names, args.alpha, 0)
    out["iforest"]["held_out_window_fpr"] = _ms(fpr_held)
    out["iforest"]["held_out_drone_alerts"] = drone_level(
        [(w, 0) for w, _ in data["_heldout"] if _scored(w) and w.end <= w.onset],
        _bundle(names, clf, thr, _dummy_rf(names, base), 2.0, args.persist), "B-IF"
    )["alerts_before_onset"]
    out["iforest"]["per_attack"] = {}
    for a in attacks:
        b = _bundle(names, clf, thr, _dummy_rf(names, base), 2.0, args.persist)
        out["iforest"]["per_attack"][a] = {
            "window": {k: _ms([m[k] for m in per[a]]) for k in per[a][0]},
            "drone": drone_level([x for x in data[a] if _scored(x[0])], b, "B-IF"),
        }

    # ---- Random Forest: leave one attack out -------------------------
    for a in attacks:
        train = base + sum((_post(data[b]) for b in attacks if b != a), [])
        P = _post(data[a])
        mets, held_fp = [], []
        for seed in range(args.seeds):
            rf, _ = fit_rf(train, names, seed)
            mets.append(window_metrics(_y(P), rf.predict_proba(_X(P, names))[:, 1],
                                       args.rf_threshold))
            held_fp.append(float(np.mean(
                rf.predict_proba(_X(held, names))[:, 1] >= args.rf_threshold)))
        rf, _ = fit_rf(train, names, 0)
        b = _bundle(names, clf, float("inf"), rf, args.rf_threshold, args.persist)
        out["rf_leave_one_attack_out"][a] = {
            "window": {k: _ms([m[k] for m in mets]) for k in mets[0]},
            "held_out_window_fpr": _ms(held_fp),
            "drone": drone_level([x for x in data[a] if _scored(x[0])], b, "B-RF"),
        }

    # ---- Random Forest: same attack type, different recording --------
    train = base + _post(data["DataManipulation"])
    P = _post(data["DataManipulation2"])
    mets = []
    for seed in range(args.seeds):
        rf, _ = fit_rf(train, names, seed)
        mets.append(window_metrics(_y(P), rf.predict_proba(_X(P, names))[:, 1],
                                   args.rf_threshold))
    rf, _ = fit_rf(train, names, 0)
    b = _bundle(names, clf, float("inf"), rf, args.rf_threshold, args.persist)
    out["rf_known_attack"]["DataManipulation->DataManipulation2"] = {
        "window": {k: _ms([m[k] for m in mets]) for k in mets[0]},
        "drone": drone_level([x for x in data["DataManipulation2"] if _scored(x[0])],
                             b, "B-RF"),
    }
    return out


def _dummy_rf(names, base):
    """An RF that never fires, for scoring the anomaly track on its own."""
    class _Never:
        def predict_proba(self, X):
            return np.tile([1.0, 0.0], (len(X), 1))
    return _Never()


# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------
def _f(ms):
    return "  n/a " if ms["mean"] is None else f"{ms['mean']:.3f}"


def report(ev, args) -> None:
    print(f"\n  ISOLATION FOREST (anomaly) · fitted on clean baseline only · "
          f"threshold top {args.alpha:.0%} · alert after {args.persist} windows in a row")
    print(f"  false alarms on a second clean recording: "
          f"{ev['iforest']['held_out_window_fpr']['mean']:.1%} of windows, "
          f"{ev['iforest']['held_out_drone_alerts']} drone alerts")
    print("  attack               AUC    F1     | attacked drone alerted  delay  "
          "honest drones alerted")
    for a, r in ev["iforest"]["per_attack"].items():
        d = r["drone"]
        print(f"  {a:<20} {_f(r['window']['auc'])}  {_f(r['window']['f1'])}  | "
              f"{d['attacked_alerted']}/{d['attacked_drones']:<21} "
              f"{str(d['first_delay_s']) + ' s':>6}  {d['honest_drones_alerted_during_attack']}"
              f"/{d['drones_total'] - d['attacked_drones']}")

    print(f"\n  RANDOM FOREST (trustworthiness) · leave-one-attack-out: each row is "
          f"an attack type the model never saw · threshold {args.rf_threshold}")
    print("  attack               AUC    F1     P      R      | attacked alerted  "
          "delay  honest alerted  clean FPR")
    for a, r in ev["rf_leave_one_attack_out"].items():
        w, d = r["window"], r["drone"]
        print(f"  {a:<20} {_f(w['auc'])}  {_f(w['f1'])}  {_f(w['precision'])}  "
              f"{_f(w['recall'])}  | {d['attacked_alerted']}/{d['attacked_drones']:<15} "
              f"{str(d['first_delay_s']) + ' s':>6}  "
              f"{d['honest_drones_alerted_during_attack']:>3}            "
              f"{r['held_out_window_fpr']['mean']:.1%}")
    for k, r in ev["rf_known_attack"].items():
        w, d = r["window"], r["drone"]
        print(f"\n  known attack type, new recording ({k}): AUC {_f(w['auc'])}, "
              f"F1 {_f(w['f1'])}, attacked alerted {d['attacked_alerted']}/"
              f"{d['attacked_drones']} after {d['first_delay_s']} s, honest alerted "
              f"{d['honest_drones_alerted_during_attack']}")


# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exclude", nargs="+", default=["CriticalNode"],
                    help="sessions kept out of the DEPLOYED classifier "
                         "('none' to use all)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="share of clean training windows above the anomaly threshold")
    ap.add_argument("--persist", type=int, default=2,
                    help="consecutive anomalous windows before an anomaly alert")
    ap.add_argument("--rf-threshold", type=float, default=0.5)
    ap.add_argument("--no-eval", action="store_true", help="only write the model")
    ap.add_argument("--out", default="results_ml.json")
    args = ap.parse_args()
    exclude = [] if args.exclude == ["none"] else args.exclude

    import sklearn
    print(f"\n  Track B models · scikit-learn {sklearn.__version__}")
    data = load_all()
    names = sorted(next(iter(data.values()))[0][0].features)
    attacks = list(C.ATTACK_SESSIONS)
    base = _pre(data[attacks[0]])
    print(f"  {len(names)} features per window; clean baseline windows: {len(base)}")

    results = {"sklearn": sklearn.__version__, "n_features": len(names),
               "features": names, "alpha": args.alpha, "persist": args.persist,
               "rf_threshold": args.rf_threshold, "seeds": args.seeds}
    if not args.no_eval:
        ev = evaluate(data, names, args)
        report(ev, args)
        results["evaluation"] = ev

    # ---- deployed bundle --------------------------------------------
    clf, thr, t_if = fit_iforest(base, names, args.alpha, 0)
    train = base + sum((_post(data[a]) for a in attacks if a not in exclude), [])
    rf, t_rf = fit_rf(train, names, 0)
    probe = [w for w, _ in _post(data[attacks[0]])][:54]
    sc = M.MLScorer(bundle=_bundle(names, clf, thr, rf, args.rf_threshold, args.persist))
    t0 = time.perf_counter()
    for _ in range(20):
        sc.score_many(probe)
    batch_ms = (time.perf_counter() - t0) / 20 * 1000

    top = sorted(zip(names, rf.feature_importances_), key=lambda x: -x[1])[:10]
    info = {"trained_on": [a for a in attacks if a not in exclude],
            "excluded": exclude, "sklearn": sklearn.__version__,
            "n_train_windows": len(train), "n_baseline_windows": len(base),
            "train_seconds_iforest": round(t_if, 3),
            "train_seconds_rf": round(t_rf, 3),
            "score_ms_per_batch_of_54_windows": round(batch_ms, 2)}
    results["deployed"] = {**info, "rf_top_features":
                           [{"feature": n, "importance": round(float(v), 4)} for n, v in top]}

    import joblib
    M.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(_bundle(names, clf, thr, rf, args.rf_threshold, args.persist, info),
                M.MODEL_PATH)
    print(f"\n  deployed model -> {M.MODEL_PATH}  (classifier trained on "
          f"{', '.join(info['trained_on'])}; {len(train)} windows; "
          f"scoring 54 windows takes {batch_ms:.1f} ms)")
    print("  top features: " + ", ".join(f"{n} {v:.2f}" for n, v in top[:5]))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"  saved {args.out}\n")


if __name__ == "__main__":
    main()
