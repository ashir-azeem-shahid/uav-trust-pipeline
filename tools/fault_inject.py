"""
tools/fault_inject.py
=====================
Benign-fault injection: does the data quality layer catch broken-but-honest
telemetry, and does it keep that telemetry from being mistaken for an attack?

WHY THIS EXISTS
---------------
The recorded sessions are well-formed: nothing in them is corrupt enough to
show the DQ layer working. This experiment injects realistic *faults* --
not attacks -- into a known fraction of records and measures, per fault type:

  flagged       share of faulty records on which at least one DQ rule fired
  quarantined   share routed out of the trust path entirely
  dimension     which of the five DQ dimensions caught it
  clean flagged share of untouched records that were flagged anyway
                (the layer's background false-flag rate, for comparison)
  verdicts      the malicious verdicts the pipeline emitted

The point the thesis needs is the separation between FAULTY and MALICIOUS:
a broken sensor should be caught by data quality, and should NOT produce a
suppression (malicious) verdict, while the real attack must still be caught
exactly as before. The session used is Critical Node, so the expected verdict
in every run is Drone18 at t=294s and nothing else.

FAULT TYPES (defined in faults.py)
----------------------------------
  sensor_noise    small Gaussian noise on position, speed and heading.
                  Realistic GPS/IMU jitter. Should mostly NOT be flagged --
                  this is the test that the layer does not over-react.
  out_of_range    one physical field set to an impossible value
                  (e.g. rotor speed far outside the fitted envelope).  validity
  trust_corrupt   one trust metric outside [0,1], as a corrupted transmission
                  would produce. Unusable, so hard-quarantined.       validity
  dropout         three required fields missing, as a sensor dropout.
                                                                 completeness
  duplicate       the same record delivered twice in the same second, as a
                  retransmission would.                            uniqueness
  position_jump   a 250 m position glitch, as a GPS fault.        consistency

A fault is applied to the record, so both copies of it in the stream (one per
role, see replay.py) carry the same fault. Faults go only on records that
carry a payload -- a timeout has nothing to corrupt. Selection is random but
seeded, so every run is reproducible.

NOTE ON POSITION JUMPS
----------------------
The consistency rule compares a drone's position with its previous report
and needs at least one second between them, because the source timestamps
have one-second resolution. Most glitches land in the same second as the
previous report and cannot be checked from this data at all. A glitch after a
long silence is not flagged because a large move is then physically possible,
and two consecutive glitches by the same offset look like a consistent track.
The script reports how many glitches were checkable and how many of those
were caught, so the result is stated precisely rather than as one percentage.

NOTE ON QUARANTINE
------------------
dq.py quarantines a record when the mean of its five dimension scores falls
below 0.5, or when any trust metric is out of [0,1]. A single failed dimension
therefore FLAGS a record (its score drops, the rule that fired is recorded)
without quarantining it; only unusable trust values, or faults that break
several dimensions at once, are removed outright. That is the designed
behaviour, and the results show it.

Usage
-----
  python tools/fault_inject.py                 # all fault types, 5% of records
  python tools/fault_inject.py --rate 0.10 --faults out_of_range duplicate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import config as C  # noqa: E402
import faults as F  # noqa: E402
import replay as R  # noqa: E402
from detect import Pipeline  # noqa: E402

DIMENSIONS = ["completeness", "validity", "timeliness", "uniqueness", "consistency"]


def run(fault: str | None, rate: float, seed: int, attack: str) -> dict:
    pipe = Pipeline()

    injected = flagged = quarantined = 0
    clean = clean_flagged = 0
    dims = Counter()
    verdicts = []
    # For position jumps: could the consistency check even run on this record?
    # It needs the drone's previous position from at least 1 s earlier, since
    # source timestamps have 1-second resolution.
    checkable = caught_checkable = 0

    def score(msg, is_fault):
        nonlocal injected, flagged, quarantined, clean, clean_flagged
        nonlocal checkable, caught_checkable
        can_check = False
        if is_fault and msg["role"] == "target":
            h = pipe.dq.hist.get(msg["contacted"])
            can_check = (h is not None and h.last_pos is not None
                         and h.last_t is not None and msg["t_session"] - h.last_t > 0)
        v = pipe.handle(msg)
        if v and v.get("track") == "A":
            verdicts.append({"drone": v["drone"], "t": v["t_session"],
                             "rule": v["verdict"],
                             "delay_s": v.get("detection_delay_s")})
        if msg["role"] != "target":
            return
        res = pipe.last_dq
        hit = bool(res.failed_rules)
        if is_fault:
            injected += 1
            flagged += hit
            quarantined += res.quarantine
            for d in DIMENSIONS:
                if getattr(res, d) < 1.0:
                    dims[d] += 1
            if can_check:
                checkable += 1
                caught_checkable += res.consistency < 1.0
        elif F.eligible(msg):
            clean += 1
            clean_flagged += hit

    for _key, msg, _onset, is_fault in F.inject(
            R.replay_with_onset(attack), fault, rate, seed):
        score(msg, is_fault)

    return {
        "fault": fault or "none",
        "rate": rate if fault else 0.0,
        "seed": seed,
        "injected": injected,
        "flagged": flagged,
        "flagged_pct": round(100 * flagged / injected, 2) if injected else None,
        "quarantined": quarantined,
        "quarantined_pct": round(100 * quarantined / injected, 2) if injected else None,
        "dimension_hits": dict(dims),
        "clean_records": clean,
        "clean_flagged_pct": round(100 * clean_flagged / clean, 2) if clean else None,
        "malicious_verdicts": verdicts,
        **({"consistency_checkable": checkable,
            "consistency_caught_of_checkable": caught_checkable}
           if fault == "position_jump" else {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--faults", nargs="+", default=list(F.FAULTS), choices=list(F.FAULTS))
    ap.add_argument("--rate", type=float, default=0.05,
                    help="share of payload-carrying records to corrupt")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attack", default="CriticalNode", choices=list(C.ATTACK_SESSIONS))
    ap.add_argument("--out", default="results_fault_injection.json")
    args = ap.parse_args()

    print(f"\n  Fault injection · {args.attack} session · {args.rate:.0%} of "
          f"payload-carrying records · seed {args.seed}\n")
    ref = run(None, 0.0, args.seed, args.attack)
    expected = [(v["drone"], v["t"]) for v in ref["malicious_verdicts"]]

    rows = [ref]
    for fault in args.faults:
        rows.append(run(fault, args.rate, args.seed, args.attack))

    print("  fault            injected   flagged   quarantined   caught by          "
          "clean flagged   malicious verdicts")
    for r in rows:
        got = [(v["drone"], v["t"]) for v in r["malicious_verdicts"]]
        verdict_txt = ", ".join(f"{d} @{t}s" for d, t in got) or "none"
        same = "unchanged" if got == expected else "CHANGED"
        main_dim = (max(r["dimension_hits"], key=r["dimension_hits"].get)
                    if r["dimension_hits"] else "-")
        inj = f"{r['injected']:>6,}" if r["fault"] != "none" else "     -"
        fl = f"{r['flagged_pct']:>7.1f}%" if r["flagged_pct"] is not None else "       -"
        qu = f"{r['quarantined_pct']:>9.1f}%" if r["quarantined_pct"] is not None else "         -"
        print(f"  {r['fault']:<15} {inj}   {fl}   {qu}     {main_dim:<16}"
              f"   {r['clean_flagged_pct']:>9.1f}%     {verdict_txt} ({same})")
        r["verdicts_unchanged"] = got == expected

    pj = next((r for r in rows if r["fault"] == "position_jump"), None)
    if pj and pj["injected"]:
        print(f"\n  position_jump: {pj['injected'] - pj['consistency_checkable']} of "
              f"{pj['injected']} glitches could not be checked (no earlier report from "
              f"that drone at least 1 s before -- source timestamps have 1-second "
              f"resolution). Of the {pj['consistency_checkable']} that could, "
              f"{pj['consistency_caught_of_checkable']} were caught; the rest followed "
              f"a long reporting gap (a large move is then physically plausible) or "
              f"another glitched record.")

    out = {"attack": args.attack, "rate": args.rate, "seed": args.seed,
           "expected_verdicts": [{"drone": d, "t": t} for d, t in expected],
           "runs": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n  saved {args.out}\n")


if __name__ == "__main__":
    main()
