"""
tools/inject_test.py
====================
Where does each detector stop working? Attacks injected into the clean
flights at increasing strength (inject.py), scored by

  rules            rules.py, thresholds FROZEN (as committed before this test)
  RF               multiclass Random Forest, trained on the recorded attack
                   sessions plus attacks injected into the OTHER clean flight
                   CAVEAT: its injected training attacks come from the same
                   generator as the test attacks (other flight, other seeds,
                   other targets), so it has learned the injector's exact
                   fingerprint. Its curves are an upper bound.
  rules -> RF      the cascade: a rule's verdict, otherwise the forest's
  RF (recorded)    the forest trained on the recorded sessions only -- the
                   deployed recipe, never shown an injected attack
  Isolation Forest anomaly view, fitted on clean windows of the training flight

TRAIN / TEST SEPARATION
  training flight  baseline.csv (it is also the prefix of every recorded
                   session, so the recorded-session forest has seen it)
  test flight      20260510_123058_Baseline.csv, never used for training,
                   for choosing a threshold, or for anything else before
  The forest's injected training runs use different random seeds, targets
  and onsets from the test runs, on the other flight. Rules see only the
  test flight; they have no training step.

Each test run picks a target at random among drones that are asked for data
at least MIN_REQUESTS times after the onset (quieter drones give nobody
anything to detect), and an onset at random between 20% and 35% of the
flight. Each setting is repeated over --seeds runs.

Per setting and method:
  detected          share of attacked identities flagged after the onset
  named             share flagged with the right attack type
  delay             median seconds from onset to the first flag
  false alarms      honest drones flagged per run (any time, any type)

Also: the data manipulation run-1 -> run-2 check on the recorded sessions
(rules, forest trained without the other recording, cascade).

Usage:  python tools/inject_test.py [--seeds 8] [--train-seeds 3]
Writes results_injection.json and docs/injection/*.png
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C              # noqa: E402
import inject as I              # noqa: E402
import ml as M                  # noqa: E402
import rules                    # noqa: E402
import train_ml as T            # noqa: E402
from detect import Pipeline     # noqa: E402

MIN_REQUESTS = 20
TRAIN_FLIGHT = C.BASELINE
TEST_FLIGHT = C.BASELINE_HELD_OUT

# (attack, label for the strength axis, [(x, params), ...]); the recorded
# attack is at the strength marked in the comment
SWEEPS = [
    ("MITM", "loss, delay 1.5 s", "loss",
     [(x, {"loss": x, "delay": 1.5}) for x in (0.05, 0.1, 0.2, 0.4, 0.6)]),   # 0.4
    ("MITM", "delay s, loss 0.4", "delay",
     [(x, {"loss": 0.4, "delay": x}) for x in (0.2, 0.5, 1.0, 1.5, 2.0)]),    # 1.5
    ("CriticalNode", "share of replies suppressed", "q",
     [(x, {"q": x}) for x in (0.25, 0.5, 0.75, 0.9, 1.0)]),                  # 1.0
    ("DataManipulation", "strength (all fields)", "full",
     [(x, {"s": x, "variant": "full"}) for x in (0.1, 0.25, 0.5, 0.75, 1.0)]),  # 1.0
    ("DataManipulation", "strength (speed only)", "speed",
     [(x, {"s": x, "variant": "speed"}) for x in (0.1, 0.25, 0.5, 0.75, 1.0)]),
    ("DataManipulation", "strength (position only)", "position",
     [(x, {"s": x, "variant": "position"}) for x in (0.1, 0.25, 0.5, 0.75, 1.0)]),
    ("Sybil", "ghosts, never answering", "ghosts",
     [(x, {"ghosts": x, "answer": 0.0}) for x in (1, 2, 5, 10)]),            # 8
    ("Sybil", "share answered, 5 ghosts", "answer",
     [(x, {"ghosts": 5, "answer": x}) for x in (0.0, 0.1, 0.3, 0.6)]),        # 0
    # The Critical Node and MITM rules need the target's OWN requests too
    # (proof of life; failure in both directions). In the clean flights only
    # two drones per flight are that busy, so these repeat the two sweeps
    # with the target drawn from them: the best case for the rules.
    ("CriticalNode", "replies suppressed, busy target", "q_busy",
     [(x, {"q": x}) for x in (0.25, 0.5, 0.75, 0.9, 1.0)]),
    ("MITM", "loss, delay 1.5 s, busy target", "loss_busy",
     [(x, {"loss": x, "delay": 1.5}) for x in (0.05, 0.1, 0.2, 0.4, 0.6)]),
]
MIN_OWN_REQUESTS_BUSY = 5
METHODS = ("rules", "rf", "cascade", "rf_recorded", "cascade_recorded", "iforest")


# ---------------------------------------------------------------------
# one injected run
# ---------------------------------------------------------------------
def run(rows, attack, params, seed, busy=False):
    rng = random.Random(seed)
    end = max(r["_t"] for r in rows)
    onset = int(end * rng.uniform(0.20, 0.35))
    counts, own = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["_t"] >= onset:
            counts[r[C.C_RES]] += 1
            own[r[C.C_REQ]] += 1
    cands = sorted(d for d, n in counts.items() if n >= MIN_REQUESTS
                   and (not busy or own[d] >= MIN_OWN_REQUESTS_BUSY))
    target = rng.choice(cands)
    inj = I.inject(rows, attack, target, onset, rng, **params)

    pipe = Pipeline(collect_windows=True)
    for _k, msg, _o in I.stream(inj, onset):
        pipe.handle(msg)
    pipe.finish()
    victim = target
    atype = I._ATTACK_LABEL[attack]
    ws = []
    for w in pipe.windows:
        lab = M.label(w, victim, atype)
        w.meta["cls"] = M.window_class(w, lab, attack)
        hit = pipe.rule_hits.get(w.subject)
        w.meta["rule"] = hit[1] if hit and hit[2] <= w.end else None
        ws.append((w, lab))
    attacked = ({f"Ghost{i + 1}" for i in range(params["ghosts"])}
                if attack == "Sybil" else {target})
    return {"attack": attack, "target": target, "onset": onset, "windows": ws,
            "rule_hits": dict(pipe.rule_hits), "attacked": attacked,
            "ignore": {target} if attack == "Sybil" else set()}


# ---------------------------------------------------------------------
# drone-level flags per method
# ---------------------------------------------------------------------
def _ml_flags(ws, rf, names, thr=0.5, gate=None):
    """drone -> (time, type) of its first window the forest calls an attack.
    gate: drone -> time after which the drone is no longer scored (cascade)."""
    scored = [w for w, _ in ws if w.features["n_requests"] > 0]
    if not scored:
        return {}
    X = np.array([[w.features[n] for n in names] for w in scored], dtype=float)
    pred = T._ml_labels(rf, X, thr)
    flags = {}
    for w, p in sorted(zip(scored, pred), key=lambda x: x[0].end):
        if p == M.NORMAL or w.subject in flags:
            continue
        if gate and w.subject in gate and w.end > gate[w.subject]:
            continue
        flags[w.subject] = (w.end, str(p))
    return flags


def _if_flags(ws, iso, thr, names, persist=2):
    scored = sorted((w for w, _ in ws if w.features["n_requests"] > 0),
                    key=lambda w: (w.subject, w.start))
    if not scored:
        return {}
    X = np.array([[w.features[n] for n in names] for w in scored], dtype=float)
    s = -iso.score_samples(X)
    flags, run_ = {}, {}
    for w, a in sorted(zip(scored, s), key=lambda x: x[0].end):
        last, n = run_.get(w.subject, (None, 0))
        n = (n + 1 if last is not None and w.start - last == M.WINDOW else 1) if a >= thr else 0
        run_[w.subject] = (w.start, n)
        if n >= persist and w.subject not in flags:
            flags[w.subject] = (w.end, "anomaly")
    return flags


def flags_for(r, models):
    rule = {d: (t, a) for d, (_v, a, t) in r["rule_hits"].items()}
    gate = {d: t for d, (t, _a) in rule.items()}
    out = {"rules": rule}
    for tag, rf in (("", models["rf"]), ("_recorded", models["rf_recorded"])):
        ml = _ml_flags(r["windows"], rf, models["names"])
        out["rf" + tag] = ml
        gated = _ml_flags(r["windows"], rf, models["names"], gate=gate)
        casc = dict(gated)
        for d, v in rule.items():
            if d not in casc or v[0] <= casc[d][0]:
                casc[d] = v
        out["cascade" + tag] = casc
    out["iforest"] = _if_flags(r["windows"], models["iso"], models["if_thr"],
                               models["names"])
    return out


def score(r, flags) -> dict:
    res = {}
    for meth, fl in flags.items():
        det = [d for d in r["attacked"] if d in fl and fl[d][0] > r["onset"] - M.WINDOW]
        named = [d for d in det if fl[d][1] in (r["attack"], "anomaly")]
        fa = [d for d in fl if d not in r["attacked"] and d not in r["ignore"]]
        res[meth] = {"detected": len(det) / len(r["attacked"]),
                     "named": len(named) / len(r["attacked"]),
                     "delays": [max(0, fl[d][0] - r["onset"]) for d in det],
                     "false_alarms": len(fa)}
    return res


# ---------------------------------------------------------------------
# training data and models
# ---------------------------------------------------------------------
def train_models(recorded, train_rows, train_seeds, names):
    attacks = list(C.ATTACK_SESSIONS)
    base = T._pre(recorded[attacks[0]])
    rec = base + sum((T._post(recorded[a]) for a in attacks), [])
    rf_recorded, _ = T.fit_rf(rec, names, 0)

    inj = []
    for attack, _lab, key, pts in SWEEPS:
        if key.endswith("_busy"):
            continue                     # same settings as the main sweeps
        for _x, params in pts:
            for s in train_seeds:
                r = run(train_rows, attack, params, s)
                inj += [x for x in r["windows"]
                        if x[0].features["n_requests"] > 0 and x[0].start >= r["onset"]]
    rf, _ = T.fit_rf(rec + inj, names, 0)
    iso, thr, _ = T.fit_iforest(base, names, 0.01, 0)
    return {"rf": rf, "rf_recorded": rf_recorded, "iso": iso, "if_thr": thr,
            "names": names, "n_train_windows": len(rec) + len(inj),
            "n_injected_train_windows": len(inj)}


def dm_run_check(recorded, names, thr=0.5) -> dict:
    """Data manipulation recording 1 -> recording 2, and back. The forest is
    trained on the clean baseline, the other three attack types and ONE data
    manipulation recording, and tested on the other."""
    attacks = list(C.ATTACK_SESSIONS)
    base = T._pre(recorded[attacks[0]])
    others = sum((T._post(recorded[a]) for a in ("CriticalNode", "MITM", "Sybil")), [])
    out = {}
    for tr, te in (("DataManipulation", "DataManipulation2"),
                   ("DataManipulation2", "DataManipulation")):
        rf, _ = T.fit_rf(base + others + T._post(recorded[tr]), names, 0)
        ws = T._post(recorded[te])
        y = T._yc(ws)
        three = T._three_way(ws, T._ml_labels(rf, T._X(ws, names), thr))
        out[f"{tr}->{te}"] = {}
        for meth in T.METHODS:
            m = T.multiclass_metrics(y, three[meth])
            pc = m["per_class"]["DataManipulation"]
            out[f"{tr}->{te}"][meth] = {
                "accuracy": round(m["accuracy"], 4),
                "dm_precision": round(pc["precision"], 4),
                "dm_recall": round(pc["recall"], 4), "dm_f1": round(pc["f1"], 4),
                "dm_windows": pc["support"],
                "victim_windows_named_dm": int(((y == "DataManipulation")
                                               & (three[meth] == "DataManipulation")).sum()),
                "honest_windows_flagged": int(((y == M.NORMAL)
                                               & (three[meth] != M.NORMAL)).sum())}
    return out


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------
COLORS = {"rules": "#2a6fdb", "rf": "#d9480f", "cascade": "#2b8a3e",
          "rf_recorded": "#f59f00", "iforest": "#868e96"}
LABELS = {"rules": "Rules", "rf": "Random Forest", "cascade": "Rules then RF",
          "rf_recorded": "RF, recorded sessions only", "iforest": "Isolation Forest"}


def plot(results, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    groups: dict[str, list] = {}
    for sw in results["sweeps"]:
        groups.setdefault(sw["attack"], []).append(sw)
    for attack, sweeps in groups.items():
        fig, axes = plt.subplots(2, len(sweeps), figsize=(3.6 * len(sweeps), 5.4),
                                 sharey="row", squeeze=False)
        for j, sw in enumerate(sweeps):
            xs = [p["x"] for p in sw["points"]]
            for meth in ("rules", "rf", "cascade", "rf_recorded", "iforest"):
                det = [p["methods"][meth]["detected"] for p in sw["points"]]
                fa = [p["methods"][meth]["false_alarms_per_run"] for p in sw["points"]]
                main = meth in ("rules", "rf", "cascade")
                # the cascade often equals the forest: draw the forest wider
                # underneath so both stay visible
                kw = dict(color=COLORS[meth],
                          lw=4 if meth == "rf" else 1.8 if main else 1.2,
                          ls="-" if main else "--", marker="o", ms=3.5,
                          label=LABELS[meth])
                axes[0, j].plot(xs, det, **kw)
                axes[1, j].plot(xs, fa, **kw)
            axes[0, j].set_title(sw["axis"], fontsize=9)
            axes[1, j].set_xlabel(sw["axis"].split(",")[0], fontsize=8)
            for a_ in axes[:, j]:
                a_.grid(alpha=0.25, lw=0.6)
                a_.tick_params(labelsize=8)
                for sp in ("top", "right"):
                    a_.spines[sp].set_visible(False)
            axes[0, j].set_ylim(-0.03, 1.03)
        axes[0, 0].set_ylabel("attacked drones detected", fontsize=9)
        axes[1, 0].set_ylabel("honest drones flagged per run", fontsize=9)
        h, l_ = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.suptitle(f"{attack} injected into the held-out clean flight", fontsize=11)
        fig.tight_layout(rect=(0, 0.09, 1, 0.94))
        p = out_dir / f"injection_{attack}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------
def _agg(scores: list[dict]) -> dict:
    out = {}
    for meth in METHODS:
        s = [x[meth] for x in scores]
        delays = [d for x in s for d in x["delays"]]
        out[meth] = {
            "detected": round(statistics.mean(x["detected"] for x in s), 3),
            "named": round(statistics.mean(x["named"] for x in s), 3),
            "median_delay_s": statistics.median(delays) if delays else None,
            "false_alarms_per_run": round(statistics.mean(x["false_alarms"] for x in s), 2),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=8, help="test runs per setting")
    ap.add_argument("--train-seeds", type=int, default=3,
                    help="training runs per setting, on the training flight")
    ap.add_argument("--out", default="results_injection.json")
    args = ap.parse_args()
    t0 = time.monotonic()

    print("\n  loading recorded sessions and training the models ...")
    recorded = {a: M.session_windows(a) for a in C.ATTACK_SESSIONS}
    names = sorted(recorded["CriticalNode"][0][0].features)
    train_rows = I.load_flight(TRAIN_FLIGHT)
    test_rows = I.load_flight(TEST_FLIGHT)
    models = train_models(recorded, train_rows,
                          [1000 + s for s in range(args.train_seeds)], names)
    print(f"  forest trained on {models['n_train_windows']:,} windows "
          f"({models['n_injected_train_windows']:,} from injected runs on "
          f"{TRAIN_FLIGHT.name})")

    frozen = {k: getattr(rules, k) for k in (
        "DEFAULT_STREAK", "SYBIL_MIN_REQUESTS", "ROTOR_MIN_AIRBORNE", "AIRBORNE_Z",
        "SPEED_TOLERANCE", "DM_CONSECUTIVE", "MITM_HISTORY", "MITM_FAIL_RATE",
        "MITM_MIN_RTT_S", "MITM_RTT_SAMPLES")}
    results = {"test_flight": TEST_FLIGHT.name, "train_flight": TRAIN_FLIGHT.name,
               "seeds_per_setting": args.seeds, "train_seeds_per_setting": args.train_seeds,
               "min_requests_to_target_after_onset": MIN_REQUESTS,
               "frozen_rule_thresholds": frozen, "sweeps": []}

    # Null control: the same random target and onset, attack strength 0 (the
    # flight unchanged). "Detected" here is the chance that a method flags a
    # randomly chosen busy drone that is NOT attacked -- the floor every
    # detection rate below should be read against.
    ctrl = []
    for s in range(args.seeds):
        r = run(test_rows, "DataManipulation", {"s": 0.0, "variant": "full"}, s)
        ctrl.append(score(r, flags_for(r, models)))
    results["null_control"] = _agg(ctrl)
    print("\n  no attack (control): " + "  ".join(
        f"{m} {results['null_control'][m]['detected']:.2f} / "
        f"{results['null_control'][m]['false_alarms_per_run']:.1f}"
        for m in ("rules", "rf", "cascade", "rf_recorded", "iforest")))

    for attack, axis, key, pts in SWEEPS:
        sw = {"attack": attack, "axis": axis, "key": key, "points": []}
        print(f"\n  {attack} · {axis}")
        print("  strength   " + "  ".join(f"{LABELS.get(m, m)[:13]:>13}" for m in
                                         ("rules", "rf", "cascade", "rf_recorded", "iforest"))
              + "   (detected / honest flagged per run)")
        for x, params in pts:
            scores = []
            for s in range(args.seeds):
                r = run(test_rows, attack, params, s, busy=key.endswith("_busy"))
                scores.append(score(r, flags_for(r, models)))
            agg = _agg(scores)
            sw["points"].append({"x": x, "params": params, "methods": agg})
            print(f"  {x:<9}  " + "  ".join(
                f"{agg[m]['detected']:>6.2f} / {agg[m]['false_alarms_per_run']:<4.1f}"
                for m in ("rules", "rf", "cascade", "rf_recorded", "iforest")))
        results["sweeps"].append(sw)

    results["dm_run1_run2"] = dm_run_check(recorded, names)
    print("\n  data manipulation, one recording -> the other (window level):")
    for k, v in results["dm_run1_run2"].items():
        print(f"  {k}: " + "  ".join(
            f"{m} F1 {v[m]['dm_f1']:.2f} ({v[m]['victim_windows_named_dm']}/"
            f"{v[m]['dm_windows']}, {v[m]['honest_windows_flagged']} honest)"
            for m in T.METHODS))

    results["plots"] = plot(results, Path("docs/injection"))
    results["runtime_s"] = round(time.monotonic() - t0, 1)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  saved {args.out} and {', '.join(results['plots'])} "
          f"({results['runtime_s']} s)\n")


if __name__ == "__main__":
    main()
