"""
tools/sweep_rules.py
====================
Sensitivity of the attack rules in rules.py to their thresholds.

Replays every recording (five attack sessions, each after the clean
baseline, plus the held-out clean recording) through the real RuleEngine
once per setting and counts, per setting:

  hits             attacked drones caught by the rule for their attack type
  false positives  any other drone that rule flagged, in any recording

A threshold that is clean over a wide range is a design choice; one that is
clean at a single value was fitted to the data. The rule thresholds were
read off these same recordings, so this table is how to judge them.

Usage:  python tools/sweep_rules.py            # writes results_rule_sweep.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C          # noqa: E402
import replay as R          # noqa: E402
import rules                # noqa: E402

_STREAMS = None


def _streams():
    """Each recording's messages once, held in memory, with its attacked drones."""
    global _STREAMS
    if _STREAMS is None:
        _STREAMS = []
        for a in C.ATTACK_SESSIONS:
            msgs = [m for _k, m, _o in R.replay_with_onset(a)]
            _STREAMS.append((a, msgs))
        msgs = [m for _k, m, o in R.replay_with_onset("CriticalNode", C.BASELINE_HELD_OUT)
                if m["t_session"] < o]
        _STREAMS.append(("_heldout", msgs))
    return _STREAMS


def _attacked(msgs) -> set[str]:
    """Ground truth, for scoring only: the drones the attack is about."""
    out = set()
    for m in msgs:
        gt = m["_gt"]
        if (gt["attack_type"] or "").lower() == "sybil":
            if m["role"] == "target" and "ghost" in m["about"].lower():
                out.add(m["about"])
        elif gt["attack_target"]:
            out.add(gt["attack_target"])
    return out


def run(rule: str, **consts) -> dict:
    """Run the engine with some rules.py constants overridden."""
    saved = {k: getattr(rules, k) for k in consts}
    for k, v in consts.items():
        setattr(rules, k, v)
    try:
        hits, fps = [], []
        registry = rules.load_registry()
        for name, msgs in _streams():
            eng = rules.RuleEngine(registry=registry)
            truth = _attacked(msgs)
            flagged: dict[str, tuple[str, int]] = {}
            for m in msgs:
                v, _ = eng.handle(m)
                if v and m["about"] not in flagged:
                    flagged[m["about"]] = (v, m["t_session"] - m.get("onset", 0))
            for d, (v, t) in flagged.items():
                if v != rule:
                    continue
                (hits if d in truth else fps).append(f"{name}:{d}@{t}s")
        return {"settings": consts, "hits": hits, "false_positives": fps}
    finally:
        for k, v in saved.items():
            setattr(rules, k, v)


def main() -> None:
    out = {"MITM": [], "DATA_MANIPULATION": [], "SYBIL": []}
    print("\n  MITM rule  (history N, failure share, min median RTT)")
    for n, f, rtt in itertools.product([60, 100, 150], [0.6, 0.7, 0.75, 0.8, 0.9],
                                       [0.8, 1.0, 1.2]):
        r = run("MITM", MITM_HISTORY=n, MITM_FAIL_RATE=f, MITM_MIN_RTT_S=rtt)
        out["MITM"].append(r)
        print(f"  N={n:<4} fail>={f:<5} rtt>={rtt:<4}  hits {len(r['hits'])} "
              f"{r['hits']}  false positives {len(r['false_positives'])}")

    print("\n  Data manipulation rule  (consecutive implausible reports)")
    for k in [1, 2, 3, 5, 10, 20, 30]:
        r = run("DATA_MANIPULATION", DM_CONSECUTIVE=k)
        out["DATA_MANIPULATION"].append(r)
        print(f"  K={k:<3} hits {len(r['hits'])} {r['hits']}  "
              f"false positives {len(r['false_positives'])}")

    print("\n  Sybil rule  (unanswered requests before firing)")
    for k in [1, 5, 20, 100]:
        r = run("SYBIL", SYBIL_MIN_REQUESTS=k)
        out["SYBIL"].append(r)
        print(f"  K={k:<3} hits {len(r['hits'])}  false positives "
              f"{len(r['false_positives'])}")

    Path("results_rule_sweep.json").write_text(json.dumps(out, indent=2))
    print("\n  saved results_rule_sweep.json\n")


if __name__ == "__main__":
    main()
