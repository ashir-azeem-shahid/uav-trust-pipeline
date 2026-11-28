"""
tests/test_rules.py
===================
The attack rules and the cascade around them.

  * no rule reads ground truth (poisoned '_gt' -> identical verdicts)
  * each rule catches its own attack, names the right type, and flags no
    other drone, in every recording, including the held-out clean one
  * the ML stage skips drones a rule has already caught
  * the multiclass scorer names an attack type

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C          # noqa: E402
import replay as R          # noqa: E402
from rules import RuleEngine, load_registry, recovered_rtt  # noqa: E402

EXPECTED = {
    "CriticalNode": ("SUPPRESSION", {"Drone18"}),
    "MITM": ("MITM", {"Drone34"}),
    "DataManipulation": ("DATA_MANIPULATION", {"Drone35"}),
    "DataManipulation2": ("DATA_MANIPULATION", {"Drone37"}),
    "Sybil": ("SYBIL", {f"Ghost{i}" for i in range(1, 9)}),
}


def _poison(m: dict) -> dict:
    p = dict(m)
    p["_gt"] = {"attack_target": "POISON", "malicious": not m["_gt"]["malicious"],
                "attack_type": "POISON", "trust_label_csv": "POISON"}
    return p


def _flagged(msgs) -> dict[str, str]:
    eng, out = RuleEngine(), {}
    for m in msgs:
        v, _ = eng.handle(m)
        if v and m["about"] not in out:
            out[m["about"]] = v
    return out


@pytest.mark.parametrize("attack", ["Sybil", "MITM", "DataManipulation"])
def test_rules_never_read_ground_truth(attack):
    msgs = [m for _k, m, _o in R.replay_with_onset(attack)]
    clean, poisoned = RuleEngine(), RuleEngine()
    a = [clean.handle(m) for m in msgs]
    b = [poisoned.handle(_poison(m)) for m in msgs]
    assert a == b, "a rule's output changed when ground truth was altered"


@pytest.mark.parametrize("attack", list(EXPECTED))
def test_each_rule_catches_its_attack_and_nothing_else(attack):
    verdict, victims = EXPECTED[attack]
    flagged = _flagged(m for _k, m, _o in R.replay_with_onset(attack))
    assert flagged == {d: verdict for d in victims}


def test_rules_silent_on_held_out_clean_recording():
    msgs = (m for _k, m, o in R.replay_with_onset("CriticalNode", C.BASELINE_HELD_OUT)
            if m["t_session"] < o)
    assert _flagged(msgs) == {}


def test_registry_comes_from_the_clean_baseline_only():
    reg = load_registry()
    assert len(reg) == 54
    assert not any("ghost" in d.lower() for d in reg)


def test_recovered_rtt_inverts_fecak_formulas():
    # Brno.py with RTT 1.5 s and 40 % loss on an answered exchange
    loss, sm, bfm, cdm = 0.4, 0.9, 1.0, 0.8
    ri = 1 - (1.5 / 5.0 + loss) / 2
    cr = (sm + bfm + cdm + 1 - loss) / 4
    trust = {"Trust_RI": ri, "Trust_CR": cr, "Trust_SM": sm,
             "Trust_BFM": bfm, "Trust_CDM": cdm}
    assert recovered_rtt(trust, answered=True) == pytest.approx(1.5)
    assert recovered_rtt({"Trust_RI": ri, "Trust_CR": (1 - loss) / 4},
                         answered=False) == pytest.approx(1.5)


def test_ml_is_gated_by_the_rules():
    """A drone a rule caught is never scored by the models again."""
    from detect import Pipeline

    class _Spy:
        def __init__(self):
            self.seen = []

        def verdicts(self, windows):
            self.seen.extend((w.subject, w.start) for w in windows)
            return []

    pipe = Pipeline(collect_windows=True)
    pipe.scorer = spy = _Spy()
    for _k, m, _o in R.replay_with_onset("Sybil"):
        pipe.handle(m)
    pipe.finish()
    ghosts = EXPECTED["Sybil"][1]
    assert set(pipe.rule_hits) == ghosts
    caught_at = {d: t for d, (_v, _a, t) in pipe.rule_hits.items()}
    after = [(d, s) for d, s in spy.seen if d in ghosts and s > caught_at[d]]
    assert after == [], "the models scored a drone a rule had already caught"
    assert pipe.ml_gated > 0
    assert any(d not in ghosts for d, _s in spy.seen)   # everyone else is scored


def test_multiclass_scorer_names_the_attack():
    import numpy as np
    from sklearn.ensemble import IsolationForest, RandomForestClassifier

    import ml as M

    names = ["a", "b"]
    X = np.array([[0, 0], [0, 1], [5, 5], [5, 6], [9, 0], [9, 1]], dtype=float)
    y = ["normal", "normal", "MITM", "MITM", "Sybil", "Sybil"]
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    iso = IsolationForest(n_estimators=20, random_state=0).fit(X[:2])
    sc = M.MLScorer(bundle={"feature_names": names, "iforest": iso,
                            "if_threshold": 99.0, "rf": rf, "rf_threshold": 0.5,
                            "if_persist": 2})
    w = M.Window("D1", 0, {"a": 9.0, "b": 0.5, "n_requests": 3.0,
                           "unanswered_rate": 0.0, "n_answered": 3.0,
                           "Trust_Score_mean": 0.5, "max_streak": 0.0,
                           "dq_min_score": 1.0}, onset=0)
    out = sc.verdicts([w])
    alert = [v for v in out if v["verdict"] == "ML_COMPROMISED"]
    assert alert and alert[0]["attack_type"] == "Sybil"
    assert alert[0]["stage"] == "ml"
