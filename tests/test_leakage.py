"""
tests/test_leakage.py
=====================
The most important test in the project.

If a ground-truth field reaches a feature vector, the model learns the
answer instead of the pattern and reports a near-perfect F1 that means
nothing. It is the single most common way a student ML thesis produces
garbage numbers, and it is invisible in the results -- the score just
looks good.

Two fields must never appear:
  Attack_Target  -- the obvious one; names the victim directly
  Trust_Label    -- the subtle one; it is a threshold cut on Trust_Score,
                    so it leaks the answer through the back door

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
import replay as R
from rules import SuppressionDetector


def _first_messages(attack: str = "CriticalNode", n: int = 200):
    out = []
    for _key, msg, _onset in R.replay_with_onset(attack):
        out.append(msg)
        if len(out) >= n:
            break
    return out


def _attack_phase_messages(attack: str = "CriticalNode", n: int = 400):
    """
    Messages from after the onset. Needed because the replay starts with the
    whole baseline session -- roughly 4,800 messages before the attack even
    begins -- so a naive 'first N' sample contains no attack rows at all.
    """
    out = []
    for _key, msg, onset in R.replay_with_onset(attack):
        if msg["t_session"] >= onset:
            out.append(msg)
            if len(out) >= n:
                break
    return out


def test_ground_truth_is_quarantined():
    """Ground truth lives only under '_gt', never at the top level."""
    for msg in _first_messages() + _attack_phase_messages():
        for field in C.GROUND_TRUTH_FIELDS:
            assert field not in msg, (
                f"{field!r} found at message top level -- it must live "
                f"under '_gt' so the feature builder cannot reach it")
        assert "_gt" in msg


def test_label_is_derived_not_copied():
    """
    The label must be ContactedDrone == Attack_Target, NOT the CSV's
    Trust_Label column -- which labels 1,040 of 2,420 baseline rows
    MALICIOUS in a session where no attack ran.
    """
    msgs = _attack_phase_messages("CriticalNode", 4000)
    victim = next(m["_gt"]["attack_target"] for m in msgs
                  if m["_gt"]["attack_target"])

    for m in msgs:
        expected = (m["contacted"] == victim) if m["_gt"]["attack_target"] else False
        assert m["_gt"]["malicious"] == expected

    # and it must disagree with the CSV label, or we copied the wrong thing
    disagreements = sum(
        1 for m in msgs
        if m["_gt"]["malicious"] != (m["_gt"]["trust_label_csv"] == "MALICIOUS"))
    assert disagreements > 0, (
        "derived label agrees perfectly with Trust_Label -- suspicious, "
        "Trust_Label is threshold-derived and should differ")


def test_detector_never_reads_ground_truth():
    """
    Feed the rule messages whose '_gt' block has been replaced with poison.
    The verdicts must be byte-identical, proving the rule never looks.
    """
    msgs = _first_messages("CriticalNode", 3000)

    clean = SuppressionDetector()
    verdicts_clean = [clean.handle(m) for m in msgs]

    poisoned = SuppressionDetector()
    verdicts_poisoned = []
    for m in msgs:
        p = dict(m)
        p["_gt"] = {"attack_target": "POISON", "malicious": not m["_gt"]["malicious"],
                    "attack_type": "POISON", "trust_label_csv": "POISON"}
        verdicts_poisoned.append(poisoned.handle(p))

    assert verdicts_clean == verdicts_poisoned, (
        "the rule's output changed when ground truth was altered -- "
        "it is reading _gt somewhere")


def test_partition_key_equals_subject():
    """
    The key must be the drone the message is about. If this drifts to
    requester or contacted, both roles stop co-locating and the rule
    silently never fires.
    """
    for key, msg, _ in R.replay_with_onset("CriticalNode"):
        assert key == msg["about"]
        if msg["role"] == "target":
            assert key == msg["contacted"]
        else:
            assert key == msg["requester"]
        if msg["t_session"] > 5:
            break


def test_both_roles_land_on_one_partition():
    """
    The property everything depends on: for any drone, its target-role and
    initiator-role messages must hash to the same partition.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from simulate import partition_of

    seen: dict[str, set[int]] = {}
    for key, msg, _ in R.replay_with_onset("CriticalNode"):
        seen.setdefault(msg["about"], set()).add(partition_of(key))
        if msg["t_session"] > 30:
            break

    for drone, parts in seen.items():
        assert len(parts) == 1, f"{drone} spread across partitions {parts}"
