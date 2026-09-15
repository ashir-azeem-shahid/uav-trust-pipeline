"""
train.py
========
Track B: Random Forest and SVM over the 15-second windows, plus the data
quality ablation that answers RQ2.

Recipe matches Fecak's so the comparison against his Table 8.2 is
like-for-like:
  * 15-second drone-window aggregation        (features.py)
  * 5-fold StratifiedGroupKFold by Requester  -- one observer never spans
    train and test
  * GridSearchCV over a small parameter grid
  * isotonic probability calibration
  * F1-optimal decision threshold chosen on the training fold, never on test

Trained per attack session, exactly as he does, so each row of the output
lines up with a row of his table.

THE ABLATION
------------
Every model is trained three times over identical windows and splits:
  full      -- all features
  no_dq     -- the five data quality dimensions removed
  no_supp   -- the suppression evidence removed
The deltas are attributable because nothing else changes.

Run:  python train.py              all sessions, full report
      python train.py CriticalNode one session
      python train.py --quick      smaller grid, faster
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as C
import features as F

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

GRIDS = {
    "RF": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "SVM": {
        "clf__C": [1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.1],
    },
}

QUICK_GRIDS = {
    "RF": {"clf__n_estimators": [300], "clf__max_depth": [None, 12]},
    "SVM": {"clf__C": [10], "clf__gamma": ["scale"]},
}


def make_model(kind: str):
    if kind == "RF":
        clf = RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
        return SKPipeline([("clf", clf)])
    clf = SVC(kernel="rbf", probability=True, class_weight="balanced",
              random_state=RANDOM_STATE)
    return SKPipeline([("scale", StandardScaler()), ("clf", clf)])


def best_threshold(y_true, proba) -> float:
    """
    F1-optimal cut, chosen on TRAINING-fold probabilities only.

    Picking it on the test fold would quietly leak the answer and inflate
    every number in the table.
    """
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba >= t).astype(int)
        if pred.sum() == 0:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate(rows, kind: str, include_dq: bool, include_supp: bool,
             quick: bool = False) -> dict:
    X, y, groups, names = F.to_matrix(rows, include_dq, include_supp)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    if y.sum() < N_SPLITS or (len(y) - y.sum()) < N_SPLITS:
        return {"error": f"too few samples of one class ({y.sum()} positive)"}

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)
    grid = (QUICK_GRIDS if quick else GRIDS)[kind]

    y_true_all, y_pred_all = [], []
    importances = np.zeros(len(names))
    n_folds = 0

    for tr, te in cv.split(X, y, groups):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue

        search = GridSearchCV(
            make_model(kind), grid, scoring="f1", n_jobs=-1,
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True,
                                    random_state=RANDOM_STATE),
            refit=True, error_score=0.0)
        search.fit(X[tr], y[tr], groups=groups[tr])

        # isotonic calibration so the decision threshold is meaningful
        cal = CalibratedClassifierCV(search.best_estimator_, method="isotonic",
                                     cv=3)
        cal.fit(X[tr], y[tr])

        thr = best_threshold(y[tr], cal.predict_proba(X[tr])[:, 1])
        pred = (cal.predict_proba(X[te])[:, 1] >= thr).astype(int)

        y_true_all.extend(y[te])
        y_pred_all.extend(pred)
        n_folds += 1

        if kind == "RF":
            rf = search.best_estimator_.named_steps["clf"]
            importances += rf.feature_importances_

    if not n_folds:
        return {"error": "no usable folds"}

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="binary", zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "precision": p, "recall": r, "f1": f1,
        "n_windows": len(y), "n_positive": int(y.sum()),
        "n_features": len(names), "folds": n_folds,
    }
    if kind == "RF":
        imp = importances / n_folds
        order = np.argsort(imp)[::-1]
        out["top_features"] = [(names[i], round(float(imp[i]), 4))
                               for i in order[:8]]
        out["dq_importance_share"] = float(
            sum(imp[i] for i, n in enumerate(names) if n in F.DQ_FEATURES))
        out["supp_importance_share"] = float(
            sum(imp[i] for i, n in enumerate(names)
                if n in F.SUPPRESSION_FEATURES))
    return out


# Fecak's Table 8.2, for the side-by-side column.
FECAK = {
    "MITM": {"RF": 0.7654, "SVM": 0.7805},
    "DataManipulation": {"RF": 0.7037, "SVM": 0.8519},
    "DataManipulation2": {"RF": 0.7037, "SVM": 0.8519},
    "Sybil": {"RF": 0.9057, "SVM": 0.9009},
    "CriticalNode": {"RF": 0.6000, "SVM": 0.7536},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", default=[],
                    help=f"sessions to run (default all): "
                         f"{', '.join(C.ATTACK_SESSIONS)}")
    ap.add_argument("--quick", action="store_true",
                    help="smaller parameter grid, much faster")
    ap.add_argument("--window-mode", default="15s", choices=["15s","session","both"],
                    help="'15s' = the real streaming task; 'session' = reproduces "
                         "Fecak's effective aggregation for a like-for-like table")
    ap.add_argument("--out", default="results_trackB.json")
    args = ap.parse_args()

    sessions = args.sessions or list(C.ATTACK_SESSIONS)
    unknown = [s for s in sessions if s not in C.ATTACK_SESSIONS]
    if unknown:
        sys.exit(f"unknown session(s) {unknown}. "
                 f"Choose from: {', '.join(C.ATTACK_SESSIONS)}")
    results: dict = {}

    for name in sessions:
        print(f"\n{'=' * 78}\n  {name}\n{'=' * 78}", flush=True)
        modes = ["15s","session"] if args.window_mode=="both" else [args.window_mode]
        rows = F.build_windows(name, modes[0])
        pos = sum(r["label"] for r in rows)
        print(f"  {len(rows)} windows, {pos} malicious ({pos/len(rows):.0%}), "
              f"{len({r['group'] for r in rows})} observer groups", flush=True)

        results[name] = {"window_mode": modes[0]}
        for kind in ("RF", "SVM"):
            variants = {
                "full":    dict(include_dq=True,  include_supp=True),
                "no_dq":   dict(include_dq=False, include_supp=True),
                "no_supp": dict(include_dq=True,  include_supp=False),
            }
            results[name][kind] = {}
            for vname, kw in variants.items():
                r = evaluate(rows, kind, quick=args.quick, **kw)
                results[name][kind][vname] = r
                if "error" in r:
                    print(f"  {kind:<4} {vname:<8} {r['error']}", flush=True)
                    continue
                ref = FECAK.get(name, {}).get(kind)
                delta = f"  (Fecak {ref:.4f}, {r['f1']-ref:+.4f})" if ref else ""
                print(f"  {kind:<4} {vname:<8} "
                      f"P={r['precision']:.4f} R={r['recall']:.4f} "
                      f"F1={r['f1']:.4f}{delta}", flush=True)

            full = results[name][kind].get("full", {})
            nodq = results[name][kind].get("no_dq", {})
            if "f1" in full and "f1" in nodq:
                print(f"  {kind:<4} ABLATION  DQ contributes "
                      f"{full['f1'] - nodq['f1']:+.4f} F1", flush=True)
            if kind == "RF" and "top_features" in full:
                print(f"  {kind:<4} DQ share of importance: "
                      f"{full['dq_importance_share']:.1%}, "
                      f"suppression share: {full['supp_importance_share']:.1%}",
                      flush=True)
                print(f"  {kind:<4} top features: "
                      f"{[f for f, _ in full['top_features'][:5]]}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {args.out}")

    # ---- the two summary tables for the thesis ----------------------
    print(f"\n{'=' * 78}\n  TABLE 1 -- detection quality vs Fecak\n{'=' * 78}")
    print(f"  {'session':<20} {'model':<5} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'Fecak F1':>9} {'delta':>8}")
    for name, per_model in results.items():
        for kind, variants in per_model.items():
            if not isinstance(variants, dict) or kind == "window_mode":
                continue
            r = variants.get("full", {})
            if "f1" not in r:
                continue
            ref = FECAK.get(name, {}).get(kind)
            print(f"  {name:<20} {kind:<5} {r['precision']:>7.4f} "
                  f"{r['recall']:>7.4f} {r['f1']:>7.4f} "
                  f"{(f'{ref:.4f}' if ref else '-'):>9} "
                  f"{(f'{r[chr(102)+chr(49)]-ref:+.4f}' if ref else '-'):>8}")

    print(f"\n{'=' * 78}\n  TABLE 2 -- data quality ablation (answers RQ2)\n{'=' * 78}")
    print(f"  {'session':<20} {'model':<5} {'with DQ':>8} {'without':>8} "
          f"{'delta':>8}")
    for name, per_model in results.items():
        for kind, variants in per_model.items():
            if not isinstance(variants, dict) or kind == "window_mode":
                continue
            f = variants.get("full", {})
            n = variants.get("no_dq", {})
            if "f1" in f and "f1" in n:
                print(f"  {name:<20} {kind:<5} {f['f1']:>8.4f} "
                      f"{n['f1']:>8.4f} {f['f1']-n['f1']:>+8.4f}")


if __name__ == "__main__":
    main()
