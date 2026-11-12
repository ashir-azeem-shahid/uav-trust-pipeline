"""
faults.py
=========
Benign telemetry faults -- broken-but-honest data, not attacks.

One definition, used in two places:
  * tools/fault_inject.py  measures how the DQ layer handles each fault
  * produce.py --fault     injects the same faults into the live Kafka stream,
                           so the dashboard shows the DQ layer working at
                           runtime while attack detection carries on

FAULT TYPES
-----------
  sensor_noise    small Gaussian noise on position, speed and heading --
                  realistic GPS/IMU jitter that should mostly NOT be flagged
  out_of_range    one physical field set far outside its fitted envelope
  trust_corrupt   one trust metric outside [0,1], as a corrupted
                  transmission would produce
  dropout         three required fields missing, as a sensor dropout
  duplicate       the same record delivered twice in the same second
  position_jump   a 250 m position glitch, as a GPS fault

A fault is applied to the record, so both copies of it in the stream (one per
role, see replay.py) carry the same fault. Only records with a payload are
eligible -- a timeout has nothing to corrupt. Selection is seeded, so a run
is reproducible.
"""

from __future__ import annotations

import random

import config as C
import dq as D


def _sensor_noise(msg, rng):
    dt = dict(msg["dt"])
    for f, sigma in (("DT_Pos_X", 1.0), ("DT_Pos_Y", 1.0), ("DT_Pos_Z", 0.5)):
        if dt.get(f) is not None:
            dt[f] = dt[f] + rng.gauss(0, sigma)
    if dt.get("DT_Speed_mps") is not None:
        dt["DT_Speed_mps"] = max(0.0, dt["DT_Speed_mps"] + rng.gauss(0, 0.3))
    if dt.get("DT_Heading_deg") is not None:
        dt["DT_Heading_deg"] = (dt["DT_Heading_deg"] + rng.gauss(0, 2.0)) % 360
    return dt, msg["trust"]


def _out_of_range(msg, rng):
    present = [f for f in D.ENVELOPE if f != "Distance_m" and msg["dt"].get(f) is not None]
    if not present:
        return None
    f = rng.choice(present)
    lo, hi = D.ENVELOPE[f]
    dt = dict(msg["dt"])
    dt[f] = hi + 3 * (hi - lo)          # far outside the fitted envelope
    return dt, msg["trust"]


def _trust_corrupt(msg, rng):
    present = [k for k, v in msg["trust"].items() if v is not None]
    if not present:
        return None
    trust = dict(msg["trust"])
    trust[rng.choice(present)] = rng.choice([1.7, -0.4])
    return msg["dt"], trust


def _dropout(msg, rng):
    required = D.REQUIRED_BY_EXCHANGE.get(msg["exchange_type"], [])
    present = [f for f in required if msg["dt"].get(f) is not None]
    if len(present) < 3:
        return None
    dt = dict(msg["dt"])
    for f in rng.sample(present, 3):
        dt[f] = None
    return dt, msg["trust"]


def _position_jump(msg, rng):
    if msg["dt"].get("DT_Pos_X") is None:
        return None
    dt = dict(msg["dt"])
    dt["DT_Pos_X"] = dt["DT_Pos_X"] + 250.0
    return dt, msg["trust"]


FAULTS = {
    "sensor_noise": _sensor_noise,
    "out_of_range": _out_of_range,
    "trust_corrupt": _trust_corrupt,
    "dropout": _dropout,
    "duplicate": None,                  # handled in the stream, not per payload
    "position_jump": _position_jump,
}


def eligible(msg) -> bool:
    return msg["exchange_type"] != C.TIMEOUT


def inject(stream, fault: str | None, rate: float, seed: int):
    """
    Wrap a (key, msg, onset) stream; yield (key, msg, onset, is_fault).

    Records are emitted as consecutive pairs (target, initiator) with
    seq = 2*row and 2*row+1, so row = seq // 2 identifies the record and
    both copies receive the same fault. A duplicated record yields the
    original (not a fault) followed by its second delivery (the fault).
    """
    if fault is not None and fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {list(FAULTS)}")
    rng = random.Random(seed)
    chosen: dict[int, object] = {}       # row -> faulty (dt, trust), or "dup"
    decided: set[int] = set()

    for key, msg, onset in stream:
        row = msg["seq"] // 2
        if fault and row not in decided:
            decided.add(row)
            if eligible(msg) and rng.random() < rate:
                if fault == "duplicate":
                    chosen[row] = "dup"
                else:
                    faulty = FAULTS[fault](msg, rng)
                    if faulty is not None:
                        chosen[row] = faulty

        f = chosen.get(row)
        if f is None:
            yield key, msg, onset, False
        elif f == "dup":
            yield key, msg, onset, False          # the genuine delivery
            yield key, dict(msg), onset, True     # the repeat is the fault
        else:
            m = dict(msg)
            m["dt"], m["trust"] = f
            yield key, m, onset, True
