"""
train_ml.py
===========
Trains the two live Track B models, evaluates them honestly, and writes the
deployable bundle models/ml.joblib that detect.py --ml loads.

  Isolation Forest  anomaly detection, fitted on clean baseline windows only
  Random Forest     trustworthiness classification, supervised, multiclass:
                    normal / CriticalNode / DataManipulation / MITM / Sybil

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

THE CASCADE: RULES vs ML vs RULES-THEN-ML (results_ml.json "cascade")
---------------------------------------------------------------------
Can the pipeline name the attack, and what does each stage contribute?
Three predictors label every (drone, 15 s window) as normal or an attack type:

  rules only     the rule that had flagged the drone by the window's end
  ML only        the multiclass Random Forest
  cascade        the rule's label if a rule fired, else the Random Forest's

Known attack types -- blocked cross-validation over time: every attack
session's post-onset windows are cut into --folds contiguous time blocks;
each block is predicted by a forest trained on the clean baseline plus the
OTHER blocks of every session. Contiguous blocks, not shuffled windows, so
neighbouring (near-identical) windows do not leak between train and test.
The held-out clean recording is predicted by a forest trained on everything.
Accuracy, per-class precision/recall/F1, macro F1, a multiclass confusion
matrix, and a 2x2 confusion matrix per attack session.

Unseen attack types -- leave-one-attack-out again, now for all three
predictors (attack vs normal): a forest cannot name a type it never saw,
but the rules need no training data, so this is where the cascade helps.

Caveats, stated in results_ml.json too: there are very few attacked windows
(Critical Node 6, data manipulation 14, MITM 19, Sybil 120), and the rule
thresholds were chosen by looking at these same recordings (rules.py says
how, per rule). The rules' numbers are therefore optimistic.

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
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

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


def _yc(ws):
    """Multiclass labels: 'normal' or the attack type (ml.window_class)."""
    return np.array([w.meta["cls"] for w, _ in ws])


def _p_attack(rf, X):
    """P(compromised) = 1 - P(normal), whatever attack classes the forest knows."""
    return 1.0 - rf.predict_proba(X)[:, list(rf.classes_).index(M.NORMAL)]


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
                                n_jobs=-1).fit(_X(rows, names), _yc(rows))
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
    # an alert from a window that ENDS by the onset saw only clean traffic:
    # a genuine false alarm. The window straddling the onset is counted apart.
    false_pre = [a for a in alerts if a["t_session"] <= onset]
    straddle = [a for a in alerts if a["window_start"] < onset < a["t_session"]]
    return {
        "attacked_drones": len(attacked),
        "attacked_alerted": len({a["drone"] for a in hit}),
        "first_delay_s": min((a["detection_delay_s"] for a in hit), default=None),
        "honest_drones_alerted_during_attack": len(false_post),
        "alerts_before_onset": len(false_pre),
        "alerts_from_onset_window": len(straddle),
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
            mets.append(window_metrics(_y(P), _p_attack(rf, _X(P, names)),
                                       args.rf_threshold))
            held_fp.append(float(np.mean(
                _p_attack(rf, _X(held, names)) >= args.rf_threshold)))
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
        mets.append(window_metrics(_y(P), _p_attack(rf, _X(P, names)),
                                   args.rf_threshold))
    rf, _ = fit_rf(train, names, 0)
    b = _bundle(names, clf, float("inf"), rf, args.rf_threshold, args.persist)
    out["rf_known_attack"]["DataManipulation->DataManipulation2"] = {
        "window": {k: _ms([m[k] for m in mets]) for k in mets[0]},
        "drone": drone_level([x for x in data["DataManipulation2"] if _scored(x[0])],
                             b, "B-RF"),
    }
    return out


# ---------------------------------------------------------------------
# the cascade: rules only vs ML only vs rules then ML
# ---------------------------------------------------------------------
METHODS = ("rules", "ml", "cascade")


def _ml_labels(rf, X, thr):
    """Forest -> label per window: 'normal' unless P(compromised) >= thr, then
    the most likely attack class (what MLScorer alerts with)."""
    proba = rf.predict_proba(X)
    classes = list(rf.classes_)
    k = classes.index(M.NORMAL)
    out = []
    for row in proba:
        if 1.0 - row[k] < thr:
            out.append(M.NORMAL)
        else:
            out.append(max((p, c) for i, (p, c) in enumerate(zip(row, classes))
                           if i != k)[1])
    return np.array(out)


def _three_way(ws, ml_pred):
    rules = np.array([w.meta["rule"] or M.NORMAL for w, _ in ws])
    cascade = np.where(rules != M.NORMAL, rules, ml_pred)
    return {"rules": rules, "ml": np.asarray(ml_pred), "cascade": cascade}


def multiclass_metrics(y, pred) -> dict:
    labels = M.CLASSES
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=labels,
                                                   zero_division=0)
    yb, pb = y != M.NORMAL, pred != M.NORMAL
    bp, br, bf, _ = precision_recall_fscore_support(yb, pb, average="binary",
                                                    zero_division=0)
    present = [c for c in labels if (y == c).any()]
    _, _, f_macro, _ = precision_recall_fscore_support(
        y, pred, labels=present, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f_macro),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(sup[i])}
                      for i, c in enumerate(labels)},
        "attack_vs_normal": {"precision": float(bp), "recall": float(br),
                             "f1": float(bf),
                             "accuracy": float(np.mean(yb == pb))},
        "confusion_matrix": {"labels": labels,
                             "rows_true_cols_predicted":
                             confusion_matrix(y, pred, labels=labels).tolist()},
    }


def _binary_cm(y, pred) -> dict:
    yb, pb = y != M.NORMAL, pred != M.NORMAL
    return {"tp": int((yb & pb).sum()), "fn": int((yb & ~pb).sum()),
            "fp": int((~yb & pb).sum()), "tn": int((~yb & ~pb).sum()),
            "attack_named_correctly": int(((y == pred) & yb).sum())}


def _drone_summary(ws, session_of, pred) -> dict:
    """Operator's view: per session, was each attacked drone flagged with the
    right type at least once, and how many honest drones were flagged."""
    out = {}
    for sess in dict.fromkeys(session_of):
        truth, said = {}, {}
        for (w, _), s_, p in zip(ws, session_of, pred):
            if s_ != sess:
                continue
            truth.setdefault(w.subject, M.NORMAL)
            if w.meta["cls"] != M.NORMAL:
                truth[w.subject] = w.meta["cls"]
            if p != M.NORMAL:
                said.setdefault(w.subject, set()).add(str(p))
        attacked = [d for d, c in truth.items() if c != M.NORMAL]
        out[sess] = {
            "attacked": len(attacked),
            "attacked_flagged": sum(d in said for d in attacked),
            "attacked_named_correctly": sum(truth[d] in said.get(d, ()) for d in attacked),
            "honest_flagged": sorted(d for d in said if truth[d] == M.NORMAL),
            "honest_total": sum(c == M.NORMAL for c in truth.values()),
        }
    return out


def _blocks(ws, k):
    """Contiguous time block (0..k-1) of each post-onset window."""
    onset = ws[0][0].onset
    end = max(w.end for w, _ in ws)
    span = max(end - onset, 1)
    return np.array([min(int((w.start - onset) * k / span), k - 1) for w, _ in ws])


def evaluate_cascade(data, names, args) -> dict:
    attacks = list(C.ATTACK_SESSIONS)
    base = _pre(data[attacks[0]])
    held = _pre(data["_heldout"])
    post = {a: _post(data[a]) for a in attacks}
    blk = {a: _blocks(post[a], args.folds) for a in attacks}
    test = sum((post[a] for a in attacks), []) + held
    y = _yc(test)
    session_of = np.array([a for a in attacks for _ in post[a]] + ["_heldout"] * len(held))

    # ---- known attack types: blocked cross-validation ------------------
    runs = []
    for seed in range(args.seeds):
        pred = {a: np.empty(len(post[a]), dtype=object) for a in attacks}
        for b in range(args.folds):
            train = base + [x for a in attacks
                            for x, bi in zip(post[a], blk[a]) if bi != b]
            rf, _ = fit_rf(train, names, seed)
            for a in attacks:
                m = blk[a] == b
                if m.any():
                    rows = [x for x, keep in zip(post[a], m) if keep]
                    pred[a][m] = _ml_labels(rf, _X(rows, names), args.rf_threshold)
        rf_all, _ = fit_rf(base + sum(post.values(), []), names, seed)
        ml_pred = np.concatenate([pred[a] for a in attacks]
                                 + [_ml_labels(rf_all, _X(held, names),
                                               args.rf_threshold)]).astype(str)
        runs.append(_three_way(test, ml_pred))

    known = {"n_windows": len(test), "folds": args.folds,
             "class_support": {c: int((y == c).sum()) for c in M.CLASSES}}
    for meth in METHODS:
        per_seed = [multiclass_metrics(y, r[meth]) for r in runs]
        m0 = per_seed[0]
        m0["seed_spread"] = {
            "accuracy": _ms([m["accuracy"] for m in per_seed]),
            "macro_f1": _ms([m["macro_f1"] for m in per_seed]),
            "attack_vs_normal_f1": _ms([m["attack_vs_normal"]["f1"] for m in per_seed])}
        m0["per_session_confusion"] = {
            a: _binary_cm(y[session_of == a], runs[0][meth][session_of == a])
            for a in attacks + ["_heldout"]}
        m0["drones"] = _drone_summary(test, session_of, runs[0][meth])
        known[meth] = m0
    rules_hit = runs[0]["rules"] != M.NORMAL
    known["ml_windows_scored_in_cascade"] = {
        "scored": int((~rules_hit).sum()), "skipped_by_rules": int(rules_hit.sum())}

    # ---- unseen attack types: leave one attack out ---------------------
    unseen = {}
    for a in attacks:
        others = [b for b in attacks
                  if M.session_class(b) != M.session_class(a)]
        rf, _ = fit_rf(base + sum((post[b] for b in others), []), names, 0)
        ws = post[a] + held
        yy = _yc(ws)
        three = _three_way(ws, _ml_labels(rf, _X(ws, names), args.rf_threshold))
        unseen[a] = {}
        for meth in METHODS:
            pr = three[meth]
            yb, pb = yy != M.NORMAL, pr != M.NORMAL
            p_, r_, f_, _ = precision_recall_fscore_support(
                yb, pb, average="binary", zero_division=0)
            unseen[a][meth] = {"precision": float(p_), "recall": float(r_),
                               "f1": float(f_), **_binary_cm(yy, pr)}
    return {"known_attack_types": known, "unseen_attack_type": unseen,
            "caveats": [
                "Few attacked windows: see class_support. One window more or "
                "less moves a per-class F1 by 0.1-0.3.",
                "Rule thresholds were chosen by looking at these same "
                "recordings (rules.py documents the sensitivity of each); the "
                "MITM rule has only one recording and is the most optimistic.",
                "The forests never see a window from the block they predict; "
                "rules have no training step.",
                "Unseen-attack ML excludes the whole attack TYPE "
                "(both data manipulation recordings together)."]}


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


def report_cascade(cv) -> None:
    k = cv["known_attack_types"]
    print(f"\n  CASCADE · known attack types · blocked {k['folds']}-fold CV over time "
          f"· {k['n_windows']} windows · attacked windows per class: "
          + ", ".join(f"{c} {n}" for c, n in k["class_support"].items() if c != M.NORMAL))
    print("  method     accuracy  macro F1  attack-vs-normal P / R / F1   "
          + "  ".join(f"{c[:8]:>8}" for c in M.ATTACK_CLASSES) + "   (per-class F1)")
    for meth in METHODS:
        m = k[meth]
        b = m["attack_vs_normal"]
        print(f"  {meth:<9}  {m['accuracy']:.3f}     {m['macro_f1']:.3f}     "
              f"{b['precision']:.2f} / {b['recall']:.2f} / {b['f1']:.2f}          "
              + "  ".join(f"{m['per_class'][c]['f1']:>8.2f}" for c in M.ATTACK_CLASSES))
    print("\n  per drone (attacked drones named correctly / honest drones flagged):")
    for meth in METHODS:
        d = k[meth]["drones"]
        print(f"  {meth:<9}  " + "  ".join(
            f"{s[:10]} {v['attacked_named_correctly']}/{v['attacked']},"
            f"{len(v['honest_flagged'])}" for s, v in d.items()))
    g = k["ml_windows_scored_in_cascade"]
    print(f"  in the cascade the forest scores {g['scored']} windows; rules settle "
          f"the other {g['skipped_by_rules']}")
    cm = k["cascade"]["confusion_matrix"]
    print("\n  cascade confusion matrix (rows = truth, columns = predicted)")
    print("  " + " " * 17 + "".join(f"{c[:10]:>11}" for c in cm["labels"]))
    for c, row in zip(cm["labels"], cm["rows_true_cols_predicted"]):
        print(f"  {c:<17}" + "".join(f"{v:>11}" for v in row))

    print("\n  CASCADE · unseen attack type (forest never saw it) · attack vs normal F1")
    print("  attack               rules   ml      cascade")
    for a, r in cv["unseen_attack_type"].items():
        print(f"  {a:<20} {r['rules']['f1']:.2f}    {r['ml']['f1']:.2f}    "
              f"{r['cascade']['f1']:.2f}")


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
    ap.add_argument("--folds", type=int, default=5,
                    help="time blocks per session for the cascade evaluation")
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
        cv = evaluate_cascade(data, names, args)
        report_cascade(cv)
        results["cascade"] = cv

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
