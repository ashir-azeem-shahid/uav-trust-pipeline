"""
replay.py
=========
Turns the recorded CSVs into an ordered stream of pipeline messages.

Three jobs, each solving a problem found in the real data:

1. ATTACK ONSET. Every attack session has the attack already running in its
   first second, so there is no onset to detect and detection delay cannot
   be measured. Fix: replay a baseline session first, then the attack
   session, as one continuous stream. The join is a known, exact onset.

2. RE-KEYING. The suppression rule needs, for one drone, both its
   successes as Requester and its timeouts as ContactedDrone. A single
   partition key cannot deliver both to one consumer. Fix: each CSV row
   becomes TWO messages -- one keyed by ContactedDrone (role='target'),
   one keyed by Requester (role='initiator'). Both copies for drone X hash
   to the same partition, so one consumer holds X's full two-role history.

3. SEQUENCE NUMBERS. Timestamps are 1-second resolution with up to 476
   rows in a second, so (Timestamp, Requester, ContactedDrone) is NOT
   unique -- 141 collisions in the baseline alone. The pipeline assigns a
   monotonic ingest sequence, which is what a real pipeline does anyway
   and gives the DQ uniqueness check something real to work with.

Ground truth travels in a nested '_gt' block. Nothing in the feature
builder is allowed to read it -- asserted by tests/test_leakage.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

import config as C


def _to_seconds(hhmmss: str) -> int:
    """'23:26:44' -> seconds since midnight. Handles the 1s resolution."""
    h, m, s = (int(p) for p in str(hhmmss).split(":"))
    return h * 3600 + m * 60 + s


def _session_rows(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Load one session, sorted by time, plus its attack target (or None)."""
    df = pd.read_csv(path)
    df["_t"] = df[C.C_TIME].map(_to_seconds)

    # Sessions can cross midnight in principle; normalise to 0-based seconds.
    df["_t"] = df["_t"] - df["_t"].min()
    df = df.sort_values("_t", kind="stable").reset_index(drop=True)

    targets = df[C.C_ATK_TGT].dropna().unique()
    victim = str(targets[0]) if len(targets) else None
    return df, victim


def _clean(value):
    """NaN -> None so the message is valid JSON."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _messages_for_row(row, seq_base: int, t_offset: int, victim: str | None):
    """
    One CSV row -> two messages, one per role.

    The payload is identical in both; only 'key' and 'role' differ. That is
    deliberate: the consumer does not need to know which copy it has in order
    to update state, only which drone the message is about.
    """
    requester = str(row[C.C_REQ])
    contacted = str(row[C.C_RES])
    exchange = str(row[C.C_EXCHANGE])

    payload = {
        "t_session": int(row["_t"]) + t_offset,   # monotonic across the replay
        "ts_original": str(row[C.C_TIME]),
        "requester": requester,
        "contacted": contacted,
        "swarm": _clean(row.get(C.C_SWARM)),
        "exchange_type": exchange,
        "distance_m": _clean(row.get(C.C_DISTANCE)),
        "dt": {c: _clean(row.get(c)) for c in C.DT_COLS},
        "trust": {c: _clean(row.get(c)) for c in C.TRUST_COLS},
        # Ground truth. Quarantined in its own block, never a feature.
        "_gt": {
            "attack_type": _clean(row.get(C.C_ATK_TYPE)),
            "attack_target": _clean(row.get(C.C_ATK_TGT)),
            # THE label: is the drone being evaluated the attacked one?
            "malicious": bool(victim is not None and contacted == victim),
            "trust_label_csv": _clean(row.get(C.C_LABEL)),  # kept for comparison only
        },
    }

    # role='target'    -> keyed by the drone being evaluated
    # role='initiator' -> keyed by the drone doing the evaluating
    for role, key in (("target", contacted), ("initiator", requester)):
        msg = dict(payload)
        msg["role"] = role
        msg["about"] = key          # the drone this copy is about
        msg["seq"] = seq_base
        yield key, msg
        seq_base += 1


def replay(attack_name: str, baseline: Path | None = None):
    """
    Yield (key, message) pairs: the whole baseline session, then the whole
    attack session, in time order, each row duplicated per role.

    The returned generator also carries `.onset` once exhausted -- use
    replay_with_onset() if you need the onset up front.
    """
    for key, msg, _ in replay_with_onset(attack_name, baseline):
        yield key, msg


def replay_with_onset(attack_name: str, baseline: Path | None = None):
    """
    Same as replay(), but each yielded tuple is (key, message, onset_seconds)
    so a consumer can compute detection delay without a side channel.

    onset_seconds = the t_session value at which the attack session begins.
    Everything before it is genuine no-attack traffic.
    """
    baseline = baseline or C.BASELINE
    attack_path = C.ATTACK_SESSIONS[attack_name]

    base_df, _ = _session_rows(baseline)
    atk_df, victim = _session_rows(attack_path)

    if victim is None:
        raise ValueError(f"{attack_path.name} has no Attack_Target -- not an attack session")

    # The attack session starts one second after the baseline ends.
    onset = int(base_df["_t"].max()) + 1

    seq = 0
    for offset, df, vic in ((0, base_df, None), (onset, atk_df, victim)):
        for _, row in df.iterrows():
            for key, msg in _messages_for_row(row, seq, offset, vic):
                msg["onset"] = onset
                yield key, msg, onset
            seq += 2


def summarise(attack_name: str, baseline: Path | None = None) -> dict:
    """Counts only -- cheap sanity check without holding the stream."""
    baseline = baseline or C.BASELINE
    base_df, _ = _session_rows(baseline)
    atk_df, victim = _session_rows(C.ATTACK_SESSIONS[attack_name])
    onset = int(base_df["_t"].max()) + 1
    return {
        "attack": attack_name,
        "victim": victim,
        "baseline_rows": len(base_df),
        "attack_rows": len(atk_df),
        "csv_rows_total": len(base_df) + len(atk_df),
        "messages_total": (len(base_df) + len(atk_df)) * 2,
        "onset_second": onset,
        "pre_attack_seconds": onset,
        "attack_seconds": int(atk_df["_t"].max()) + 1,
    }


if __name__ == "__main__":
    import json
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "CriticalNode"
    print(json.dumps(summarise(name), indent=2))

    print(f"\nFirst 2 messages of the {name} replay:")
    for i, (key, msg, onset) in enumerate(replay_with_onset(name)):
        print(f"\n  key={key!r}  role={msg['role']}  seq={msg['seq']}  t={msg['t_session']}")
        print("  " + json.dumps(msg, default=str)[:300] + " ...")
        if i >= 1:
            break
