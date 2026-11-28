"""
tests/test_inject.py
====================
The attack injector (inject.py) behaves like the attacks it mirrors, and
keeps ground truth where the pipeline cannot read it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C          # noqa: E402
import inject as I          # noqa: E402

ROWS = I.load_flight(C.BASELINE_HELD_OUT)
ONSET = 80
TARGET = "Drone19"


def _post(rows, f):
    return [r for r in rows if r["_t"] >= ONSET and f(r)]


def test_rows_before_onset_untouched():
    for attack, kw in (("MITM", {"loss": 0.6, "delay": 1.5}), ("CriticalNode", {"q": 1.0}),
                       ("DataManipulation", {"s": 1.0}), ("Sybil", {"ghosts": 3})):
        inj = I.inject(ROWS, attack, TARGET, ONSET, random.Random(0), **kw)
        assert [r for r in inj if r["_t"] < ONSET] == [r for r in ROWS if r["_t"] < ONSET]


def test_critical_node_full_suppression_answers_nothing():
    inj = I.inject(ROWS, "CriticalNode", TARGET, ONSET, random.Random(0), q=1.0)
    to_target = _post(inj, lambda r: r[C.C_RES] == TARGET)
    assert to_target and all(r[C.C_EXCHANGE] == C.TIMEOUT for r in to_target)


def test_zero_strength_changes_no_payload():
    inj = I.inject(ROWS, "DataManipulation", TARGET, ONSET, random.Random(0), s=0.0)
    def same(x, y):
        return x == y or (x != x and y != y)          # NaN == NaN here
    for a, b in zip(inj, ROWS):
        assert all(same(a.get(c), b.get(c)) for c in C.DT_COLS)


def test_full_data_manipulation_matches_the_recorded_attack_values():
    inj = I.inject(ROWS, "DataManipulation", TARGET, ONSET, random.Random(0), s=1.0)
    rep = _post(inj, lambda r: r[C.C_RES] == TARGET and r[C.C_EXCHANGE] != C.TIMEOUT)
    assert rep and all(r["DT_Speed_mps"] == 12.0 and r["DT_RotorAvgSpeed"] == 0.02
                       and r["Trust_BFM"] == 0.4 for r in rep)


def test_ground_truth_only_in_the_gt_block():
    inj = I.inject(ROWS, "Sybil", TARGET, ONSET, random.Random(0), ghosts=2)
    for _k, msg, _o in I.stream(inj, ONSET):
        for f in C.GROUND_TRUTH_FIELDS:
            assert f not in msg
    ghosts = {m["contacted"] for _k, m, _o in I.stream(inj, ONSET)
              if m["contacted"].startswith("Ghost")}
    assert ghosts == {"Ghost1", "Ghost2"}


def test_same_seed_same_flight():
    a = I.inject(ROWS, "MITM", TARGET, ONSET, random.Random(7), loss=0.4, delay=1.5)
    b = I.inject(ROWS, "MITM", TARGET, ONSET, random.Random(7), loss=0.4, delay=1.5)
    assert a == b
