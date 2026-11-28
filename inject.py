"""
inject.py
=========
Adversarial telemetry injection: put an attack, at a chosen strength, into a
recorded CLEAN flight, then replay it like any other session.

Why: the bachelor thesis recorded each attack once, at one strength. That is
enough to show a detector works, not to show where it stops working. AirSim
cannot be re-run here, so the attacks are re-applied to the clean recordings
row by row, following how Fecak's simulator implements each one
(bsc-reference/simulation/attack_injector.py and Brno.py), with the strength
turned down and up.

THE ATTACK MODELS (what the simulator does -> what this module does)
--------------------------------------------------------------------
MITM, attack_injector.py: every frame the victim SENDS is delayed 1.5 s and
dropped with probability 0.40.
    -> after the onset, each reply from the victim and each request by the
       victim is lost with probability `loss`; a delivered exchange has its
       round trip lengthened by `delay` s. A reply later than the requester's
       retry interval (1.2 s, Brno.py) first counts as unanswered, then is
       credited when it arrives, as Brno.py does.

Critical Node, Brno.py: the victim stops sending PROVIDE_DT replies but keeps
beaconing and requesting.
    -> each reply from the victim is suppressed with probability `q`
       (1.0 = the recorded attack).

Data manipulation, Brno.py: the victim's own PROVIDE_DT reports speed 2x the
commanded 6 m/s (12 m/s) along x, position offset by (+8, -6, +5) m, rotor
speed 0.02, neighbour count +4.
    -> each reported field is moved a share `s` of the way from its true value
       to the attack value (s = 1 is the recorded attack). Variants: all
       fields ("full"), speed and velocity only ("speed"), position only
       ("position").

Sybil, attack_injector.py: eight Ghost identities beacon within 8 m of the
target; neighbours request data from them and never get an answer.
    -> `ghosts` fabricated identities appear at the onset; every drone that is
       talking to the target asks each ghost about once every two seconds.
       A ghost answers with probability `answer` (0 = the recorded attack),
       with a payload copied from the target's last report plus 8 m jitter.

What every injected exchange goes through, as in Brno.py
  * a request counts as unanswered in the requester's last-8 outcome window;
    a timeout row is only LOGGED when that window's loss exceeds 50% (or the
    peer is a ghost). Lighter attacks therefore leave fewer timeout rows --
    exactly as the simulator would have logged them.
  * the requester's trust metrics are recomputed with Fecak's formulas:
    RI = 1 - (latency_norm + loss) / 2, BFM from rotor speed, SM from speed,
    altitude and XY agreement, BA = (SM+CDM+BFM)/3, CR = (SM+BFM+CDM+link)/4,
    TS = BA x RI x CR. SM is lowered by each component's new disagreement
    (its old components are not in the CSV); CDM is left unchanged because the
    CSV has no neighbour count. The CHANGE the attack makes is applied to
    the recorded values (RI and packet loss are compared against the same
    link replayed without the attack), so rows keep the recording's own
    quirks and differ only where the attack acts.

Limits, stated so they can be reported: the drones do not REACT to the attack
(no re-planning, no change in who talks to whom), beacon staleness is not
modelled, and ghosts never initiate. The injected flights are therefore a
controlled test of the detectors, not a replacement for simulation.

Ground truth goes where replay.py puts it: Attack_Type / Attack_Target on the
attacked rows, then the nested '_gt' block. Nothing here is read by the rules
or the models.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from pathlib import Path

import config as C
import replay as R
from rules import recovered_rtt

RETRY_S = 1.2          # Brno.py request_retry_interval
LOG_LOSS = 0.5         # Brno.py: a timeout is logged only above this loss
WIN = 8                # Brno.py: outcome and latency windows hold 8 samples

ATTACKS = ("MITM", "CriticalNode", "DataManipulation", "Sybil")
_ATTACK_LABEL = {"MITM": "MITM", "CriticalNode": "Critical Node",
                 "DataManipulation": "Data Manipulation", "Sybil": "Sybil"}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def load_flight(path: Path) -> list[dict]:
    """A clean recording as a list of row dicts, time-sorted, with '_t'."""
    df, victim = R._session_rows(path)
    if victim is not None:
        raise ValueError(f"{path.name} is an attack session, not a clean flight")
    return df.to_dict("records")


def _f(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def candidate_targets(rows: list[dict], min_requests: int = 20) -> list[str]:
    """Drones asked for data at least `min_requests` times: enough traffic
    for any detector to see something. Drones that are barely contacted
    cannot be told apart from a quiet drone by anyone."""
    n = defaultdict(int)
    for r in rows:
        n[r[C.C_RES]] += 1
    return sorted(d for d, k in n.items() if k >= min_requests)


class _Pairs:
    """Each requester's view of each peer: last-8 outcomes and round trips."""

    def __init__(self) -> None:
        self.out = defaultdict(lambda: deque(maxlen=WIN))
        self.lat = defaultdict(lambda: deque(maxlen=WIN))

    def loss(self, k) -> float:
        o = self.out[k]
        return 1.0 - sum(o) / len(o) if o else 1.0

    def ri(self, k) -> float:
        lat = self.lat[k]
        avg = sum(lat) / len(lat) if lat else 0.0
        o = self.out[k]
        loss = 1.0 - sum(o) / len(o) if o else 0.0
        return round(max(0.0, 1.0 - (min(1.0, avg / 5.0) + loss) / 2.0), 4)

    def observe_original(self, r: dict) -> None:
        k = (r[C.C_REQ], r[C.C_RES])
        if r[C.C_EXCHANGE] == C.TIMEOUT:
            self.out[k].append(False)
        else:
            self.out[k].append(True)
            rtt = recovered_rtt(_trust(r), True)
            if rtt is not None:
                self.lat[k].append(rtt)


def _trust(r: dict) -> dict:
    return {c: _f(r.get(c)) for c in C.TRUST_COLS}


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _delta(pairs: _Pairs, shadow: _Pairs, k) -> tuple[float, float]:
    """How much the attack moved this pair's RI and packet loss, comparing the
    attacked link state with the same link replayed without the attack.
    Applying the CHANGE to the recorded values (instead of recomputing them
    from scratch) leaves every untouched quirk of the recording in place, so
    a model cannot learn 'this row was rewritten' instead of the attack."""
    def loss0(p):
        o = p.out[k]
        return 1.0 - sum(o) / len(o) if o else 0.0
    return shadow.ri(k) - pairs.ri(k), loss0(pairs) - loss0(shadow)


def _timeout_row(r: dict, pairs: _Pairs, k, shadow: _Pairs | None = None) -> dict:
    t = dict(r)
    for c in C.DT_COLS:
        t[c] = None
    t[C.C_EXCHANGE] = C.TIMEOUT
    tr = _trust(r)
    if shadow is None or tr["Trust_RI"] is None or tr["Trust_CR"] is None:
        loss = pairs.loss(k)                     # a fabricated row: from scratch
        ri, cr = pairs.ri(k), (1.0 - loss) / 4.0
    else:
        d_ri, d_loss = _delta(pairs, shadow, k)
        ri = _clip(tr["Trust_RI"] - d_ri)
        if r[C.C_EXCHANGE] == C.TIMEOUT:
            cr = _clip(tr["Trust_CR"] - d_loss / 4.0)
        else:
            loss0 = _clip(1 - (4 * tr["Trust_CR"] - (tr["Trust_SM"] or 0)
                               - (tr["Trust_BFM"] or 0) - (tr["Trust_CDM"] or 0)))
            cr = (1.0 - _clip(loss0 + d_loss)) / 4.0
    t.update({"Trust_RI": round(ri, 4), "Trust_BFM": 0.0, "Trust_SM": 0.0,
              "Trust_CDM": 0.0, "Trust_BA": 0.0, "Trust_CR": round(cr, 4),
              "Trust_Score": 0.0})
    return t


def _retrust(r: dict, pairs: _Pairs | None, k, sm=None, bfm=None,
             shadow: _Pairs | None = None) -> dict:
    """Recompute an answered row's trust block (Fecak's formulas).
    pairs None: the link is untouched (data manipulation) -- RI and the link
    term stay as recorded. shadow given: the link changed -- apply the change.
    Neither: a fabricated row -- compute the link from scratch."""
    tr = _trust(r)
    sm0, bfm0, cdm, ri0, cr0 = (tr["Trust_SM"], tr["Trust_BFM"], tr["Trust_CDM"],
                                tr["Trust_RI"], tr["Trust_CR"])
    if None in (sm0, bfm0, cdm, ri0, cr0):
        return r
    sm = sm0 if sm is None else sm
    bfm = bfm0 if bfm is None else bfm
    link0 = _clip(4 * cr0 - sm0 - bfm0 - cdm)
    if pairs is None:
        ri, link = ri0, link0
    elif shadow is not None:
        d_ri, d_loss = _delta(pairs, shadow, k)
        ri, link = _clip(ri0 - d_ri), _clip(link0 - d_loss)
    else:
        ri, link = pairs.ri(k), 1.0 - pairs.loss(k)
    ba = (sm + cdm + bfm) / 3.0
    cr = (sm + bfm + cdm + link) / 4.0
    out = dict(r)
    out.update({"Trust_RI": round(ri, 4), "Trust_SM": round(sm, 4),
                "Trust_BFM": round(bfm, 4), "Trust_BA": round(ba, 4),
                "Trust_CR": round(cr, 4), "Trust_Score": round(ba * ri * cr, 4)})
    return out


def _mark(r: dict, attack: str, target: str) -> dict:
    r = dict(r)
    r[C.C_ATK_TYPE] = _ATTACK_LABEL[attack]
    r[C.C_ATK_TGT] = target
    return r


# ---------------------------------------------------------------------
# the attacks
# ---------------------------------------------------------------------
def _link_attack(rows, target, onset, rng, *, loss_to, loss_from, delay):
    """MITM (loss both ways + delay) and Critical Node (loss_to only)."""
    pairs, shadow, out = _Pairs(), _Pairs(), []
    for r in rows:
        shadow.observe_original(r)            # the same link, never attacked
        if r["_t"] < onset:
            pairs.observe_original(r)
            out.append(r)
            continue
        k = (r[C.C_REQ], r[C.C_RES])
        p = (loss_to if r[C.C_RES] == target
             else loss_from if r[C.C_REQ] == target else 0.0)
        if p == 0.0 and not (delay and target in k):   # not the victim's link
            pairs.observe_original(r)
            out.append(r)
            continue
        if r[C.C_EXCHANGE] == C.TIMEOUT:       # already unanswered: still is
            pairs.out[k].append(False)
            out.append(_timeout_row(r, pairs, k, shadow))
            continue
        if rng.random() < p:                   # lost
            pairs.out[k].append(False)
            if pairs.loss(k) > LOG_LOSS:
                out.append(_timeout_row(r, pairs, k, shadow))
            continue                           # below 50% loss: not logged
        if delay >= RETRY_S:                   # late: unanswered at the retry...
            pairs.out[k].append(False)
            if pairs.loss(k) > LOG_LOSS:
                out.append(_timeout_row(r, pairs, k, shadow))
            pairs.out[k][-1] = True            # ...then credited on arrival
        else:
            pairs.out[k].append(True)
        rtt = recovered_rtt(_trust(r), True)
        pairs.lat[k].append((rtt or 0.0) + delay)
        out.append(_retrust(r, pairs, k, shadow=shadow))
    return out


def _data_manipulation(rows, target, onset, s, variant):
    pairs, out = _Pairs(), []
    do_speed = variant in ("full", "speed")
    do_pos = variant in ("full", "position")
    do_rotor = variant == "full"
    for r in rows:
        pairs.observe_original(r)
        if (r["_t"] < onset or r[C.C_RES] != target
                or r[C.C_EXCHANGE] == C.TIMEOUT):
            out.append(r)
            continue
        m = dict(r)
        pen = 0.0
        v = _f(r.get("DT_Speed_mps"))
        if do_speed and v is not None:
            nv = v + s * (12.0 - v)
            m["DT_Speed_mps"] = round(nv, 3)
            vx, vy, vz = (_f(r.get(c)) or 0.0 for c in ("DT_Vel_VX", "DT_Vel_VY", "DT_Vel_VZ"))
            m["DT_Vel_VX"] = round(vx + s * (nv - vx), 3)
            m["DT_Vel_VY"] = round(vy * (1 - s), 3)
            m["DT_Vel_VZ"] = round(vz * (1 - s), 3)
            pen += abs(nv - v) / (max(abs(v), abs(nv)) + 1e-6)
        z = _f(r.get("DT_Pos_Z"))
        if do_pos and z is not None:
            for c, d in (("DT_Pos_X", 8.0), ("DT_Pos_Y", -6.0), ("DT_Pos_Z", 5.0)):
                if _f(r.get(c)) is not None:
                    m[c] = round(_f(r.get(c)) + s * d, 2)
            nz = z + 5.0 * s
            pen += abs(nz - z) / (max(abs(z), abs(nz)) + 1e-6)
            pen += min(1.0, s * 10.0 / 3.0)          # XY error 10 m at s = 1
        bfm = None
        rot = _f(r.get("DT_RotorAvgSpeed"))
        if do_rotor and rot is not None:
            nr = rot + s * (0.02 - rot)
            m["DT_RotorAvgSpeed"] = round(nr, 4)
            bfm = 1.0 if nr > 0.05 else nr / 0.05
        sm0 = _f(r.get("Trust_SM"))
        sm = None if sm0 is None else max(0.0, min(1.0, sm0 - pen / 3.0))
        out.append(_retrust(m, None, None, sm=sm, bfm=bfm))   # link untouched
    return out


def _sybil(rows, target, onset, rng, n_ghosts, answer):
    end = max(r["_t"] for r in rows)
    # who is talking to the target, second by second (+-15 s)
    talk = defaultdict(set)
    for r in rows:
        if target in (r[C.C_REQ], r[C.C_RES]):
            other = r[C.C_RES] if r[C.C_REQ] == target else r[C.C_REQ]
            for t in range(r["_t"] - 15, r["_t"] + 16):
                talk[t].add(other)
    last = None                      # target's latest report, the ghosts' disguise
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["_t"]].append(r)
    ghosts = [f"Ghost{i + 1}" for i in range(n_ghosts)]
    pairs, out = _Pairs(), []
    for t in range(0, end + 1):
        for r in by_t.get(t, []):
            pairs.observe_original(r)
            if r[C.C_RES] == target and r[C.C_EXCHANGE] != C.TIMEOUT:
                last = r
            out.append(r)
        if t < onset or last is None:
            continue
        for g in ghosts:
            for req in sorted(talk.get(t, ())):
                if req == target or rng.random() >= 0.5:
                    continue
                k = (req, g)
                row = {C.C_TIME: last[C.C_TIME], "_t": t, C.C_REQ: req,
                       C.C_RES: g, C.C_SWARM: "Z",
                       C.C_DISTANCE: round(rng.uniform(2.0, 18.0), 2)}
                if rng.random() < answer:
                    pairs.out[k].append(True)
                    pairs.lat[k].append(0.3)
                    row.update({c: last.get(c) for c in C.DT_COLS})
                    for c in ("DT_Pos_X", "DT_Pos_Y"):
                        if _f(last.get(c)) is not None:
                            row[c] = round(_f(last.get(c)) + rng.uniform(-8, 8), 2)
                    row["DT_Pos_Z"] = -40.0
                    row["DT_Speed_mps"] = round(rng.uniform(4.0, 8.0), 3)
                    row[C.C_EXCHANGE] = "inter"
                    row.update(_trust(last))
                    out.append(_mark(_retrust(row, pairs, k), "Sybil", target))
                else:
                    pairs.out[k].append(False)
                    row[C.C_EXCHANGE] = C.TIMEOUT
                    out.append(_mark(_timeout_row(row, pairs, k), "Sybil", target))
    return out


def inject(rows: list[dict], attack: str, target: str, onset: int,
           rng: random.Random, **strength) -> list[dict]:
    """The flight with `attack` running on `target` from `onset` seconds."""
    if attack == "MITM":
        out = _link_attack(rows, target, onset, rng, loss_to=strength["loss"],
                           loss_from=strength["loss"], delay=strength["delay"])
    elif attack == "CriticalNode":
        out = _link_attack(rows, target, onset, rng, loss_to=strength["q"],
                           loss_from=0.0, delay=0.0)
    elif attack == "DataManipulation":
        out = _data_manipulation(rows, target, onset, strength["s"],
                                 strength.get("variant", "full"))
    elif attack == "Sybil":
        out = _sybil(rows, target, onset, rng, strength["ghosts"],
                     strength.get("answer", 0.0))
    else:
        raise ValueError(attack)
    # every row from the onset on belongs to the attack session (as in the
    # recordings, where Attack_Type/Attack_Target fill every row)
    return [_mark(r, attack, target) if r["_t"] >= onset else r for r in out]


def stream(rows: list[dict], onset: int):
    """(key, message, onset) triples, exactly as replay.replay_with_onset."""
    seq = 0
    for r in rows:
        victim = r.get(C.C_ATK_TGT) if r["_t"] >= onset else None
        victim = victim if isinstance(victim, str) else None
        for key, msg in R._messages_for_row(r, seq, 0, victim):
            msg["onset"] = onset
            yield key, msg, onset
        seq += 2
