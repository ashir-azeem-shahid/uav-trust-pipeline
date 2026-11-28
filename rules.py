"""
rules.py
========
Track A: cheap deterministic rules, one per attack type, as streaming state.

  Critical Node     SuppressionDetector   (this docstring, below)
  Sybil             SybilRule             identity not in the fleet registry
  Data manipulation DataManipulationRule  physically impossible telemetry
  MITM              MITMRule              both directions of one drone's link
                                          failing, with slow answers
  RuleEngine        runs all four and names the attack type

Every rule reads only operational fields (about, role, t_session,
exchange_type, the dt payload and the drone-computed trust block). None
reads msg['_gt'] -- tests/test_leakage.py feeds poisoned ground truth and
asserts identical verdicts. Thresholds and how they were chosen are next to
each constant; tools/sweep_rules.py reproduces the sensitivity tables.

THE CRITICAL NODE RULE
----------------------

HOW THIS ARRIVED AT A STREAK COUNTER
------------------------------------
Session-level, the signal is unmistakable: the Critical Node victim times
out on 100% of 302 exchanges where it is the target, while the worst
healthy drone manages 75%.

The obvious streaming translation -- "what share of target exchanges in the
last 15 seconds timed out?" -- does not work, and it is worth knowing why.
The baseline timeout rate is around 50%, so inside a short window "3 of 3
timed out" occurs by chance. Over hundreds of windows per drone per
session, essentially every drone eventually produces one all-timeout
window. Measured: 46-52 false positives per session at min_rows=3, still
11-19 at min_rows=15.

What separates the victim is not the rate in any one window -- it is
PERSISTENCE. The victim never answers again, ever. A healthy drone with a
bad link answers a moment later.

So the statistic is a streak: consecutive exchanges where this drone was
the target and timed out, reset to zero by any successful response. One
integer of state per drone, and a streak is something only a stateful
stream consumer can hold -- Fecak's per-row batch analysis has nowhere to
put it, which is precisely the architectural argument.

THE RULE
--------
    suppression IF  consecutive_timeouts_as_target >= STREAK
                AND successful_self_initiated_exchanges_in_window > 0

Second clause carries the weight the beacon log was supposed to carry:
a drone that is still initiating exchanges is alive, in radio range, and
its radio works, so "out of range" is ruled out by direct evidence. It
also excludes Sybil ghosts for free -- they never initiate, because they
do not exist.
"""

from __future__ import annotations

from collections import deque

from pathlib import Path

import config as C

# Chosen from tools/sweep_streak.py.
#
# Measured: the rule is clean (victim detected, ZERO false positives across
# all five attack sessions and both baselines) for every threshold from 70
# to 300. That is a 4x plateau, not a knife-edge -- which is the answer to
# "you just fitted a threshold to your data".
#
# The worst honest drone in any session reaches a streak of 69. The Critical
# Node victim reaches 302. 80 sits just above the honest maximum with most
# of the plateau still above it.
DEFAULT_STREAK = 80


class DroneState:
    """
    Everything the rule needs about one drone, as streaming state.

    Deliberately tiny: two integers and a short deque. 53 drones of this
    fits in a few kilobytes, which is why no Redis or state store is needed.
    """

    __slots__ = ("streak", "max_streak", "initiated_ok", "last_t", "n_target")

    def __init__(self) -> None:
        self.streak = 0            # consecutive timeouts as target
        self.max_streak = 0
        self.n_target = 0          # total target exchanges seen
        self.initiated_ok = deque()   # timestamps of successful self-initiated exchanges
        self.last_t = 0

    # --- state updates ------------------------------------------------
    def observe(self, role: str, t: int, is_timeout: bool) -> None:
        self.last_t = t

        if role == "target":
            self.n_target += 1
            if is_timeout:
                self.streak += 1
                self.max_streak = max(self.max_streak, self.streak)
            else:
                self.streak = 0        # it answered -- not suppressed
        else:  # initiator
            if not is_timeout:
                self.initiated_ok.append(t)

        # keep the proof-of-life window bounded
        cutoff = t - C.WINDOW_SECONDS
        while self.initiated_ok and self.initiated_ok[0] < cutoff:
            self.initiated_ok.popleft()

    # --- the verdict --------------------------------------------------
    def verdict(self, streak: int = DEFAULT_STREAK) -> tuple[bool, str]:
        """Returns (suppressed, human-readable reason)."""
        if self.streak < streak:
            return False, ""
        alive = len(self.initiated_ok)
        if alive == 0:
            # No proof of life. Either genuinely gone, or a Sybil ghost.
            return False, ""
        return True, (
            f"suppression: {self.streak} consecutive unanswered requests as target, "
            f"while {alive} self-initiated exchanges succeeded in the last "
            f"{C.WINDOW_SECONDS}s (alive and in range, not answering)"
        )


class SuppressionDetector:
    """
    One of these per consumer process. Holds state for the drones whose keys
    land in this consumer's partitions and nothing else.
    """

    def __init__(self, streak: int = DEFAULT_STREAK) -> None:
        self.streak = streak
        self.drones: dict[str, DroneState] = {}

    def handle(self, msg: dict) -> tuple[bool, str]:
        """
        Feed one telemetry message. Returns (fired, reason).

        Reads only: about, role, t_session, exchange_type.
        Never touches msg['_gt'] -- that is ground truth, not input.
        """
        drone = msg["about"]
        st = self.drones.get(drone)
        if st is None:
            st = self.drones[drone] = DroneState()

        st.observe(msg["role"], msg["t_session"],
                   msg["exchange_type"] == C.TIMEOUT)
        return st.verdict(self.streak)

    def snapshot(self) -> dict[str, dict]:
        """For the dashboard: current streak and max streak per drone."""
        return {
            d: {"streak": s.streak, "max_streak": s.max_streak,
                "n_target": s.n_target, "alive_signals": len(s.initiated_ok)}
            for d, s in self.drones.items()
        }


# =====================================================================
# Fleet registry -- the identities that are supposed to be in the swarm
# =====================================================================
def load_registry(path: Path | None = None) -> frozenset[str]:
    """
    The drone identities registered to the fleet.

    A real operator knows which aircraft it flies (UAS Remote ID, a fleet
    database). The recordings carry no registry, so the stand-in is every
    identity that appears in the clean baseline flight -- the same file the
    DQ thresholds are fitted on. No attack session is read.
    """
    import csv
    path = path or C.BASELINE
    ids: set[str] = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ids.add(row[C.C_REQ])
            ids.add(row[C.C_RES])
    return frozenset(ids)


def _registered(drone: str, registry: frozenset[str]) -> bool:
    # tools/scale_test.py and tools/train_scale.py grow the swarm by cloning
    # Drone18 as Drone18~c3; a clone of a registered drone is registered.
    return drone.split("~c")[0] in registry


# =====================================================================
# Sybil: an identity nobody registered, that nobody has ever heard from
# =====================================================================
# Requests to the identity that must go unanswered before the rule fires.
# The ghosts collect 620-670 each per session, so 5 is reached within
# seconds; it only stops one stray request to a mistyped name from firing.
SYBIL_MIN_REQUESTS = 5


class SybilRule:
    """
    sybil IF  identity not in the fleet registry
          AND requested >= SYBIL_MIN_REQUESTS times
          AND it has never answered and never initiated an exchange

    The last clause is proof of absence: a real but unregistered aircraft
    (a guest drone) would answer or make requests of its own. A fabricated
    identity only exists as a beacon, so neither happens.
    """

    def __init__(self, registry: frozenset[str]) -> None:
        self.registry = registry
        self.requests: dict[str, int] = {}
        self.alive: set[str] = set()

    def handle(self, msg: dict) -> tuple[bool, str]:
        drone = msg["about"]
        if _registered(drone, self.registry):
            return False, ""
        if msg["role"] == "initiator" or msg["exchange_type"] != C.TIMEOUT:
            self.alive.add(drone)         # it did something itself: not a ghost
        if msg["role"] == "target":
            self.requests[drone] = self.requests.get(drone, 0) + 1
        n = self.requests.get(drone, 0)
        if drone in self.alive or n < SYBIL_MIN_REQUESTS:
            return False, ""
        return True, (f"sybil: '{drone}' is not a registered fleet identity, "
                      f"{n} requests to it went unanswered and it has never "
                      f"sent anything itself")


# =====================================================================
# Data manipulation: telemetry that breaks physics, persistently
# =====================================================================
# Lowest rotor speed for an airborne drone -- the DQ validity envelope
# (dq.ENVELOPE), fitted on the clean baseline, whose minimum is 324.
ROTOR_MIN_AIRBORNE = 200.0
# Below this NED altitude a drone is airborne (the swarm flies at -34..-45 m).
AIRBORNE_Z = -5.0
# Declared speed vs speed implied by consecutive positions: the DQ tolerance.
SPEED_TOLERANCE = 5.0
# Consecutive implausible reports from one drone before the rule fires.
# In all seven recordings no honest drone produced more than 1 in a row; the
# two manipulated drones produced 29 and 87. Any value from 2 to 29 gives
# the same result on the recordings, so this is not a fitted knife-edge.
# Persistence is what separates an attack from a benign glitch:
# tools/fault_inject.py corrupts a share of honest records at random, and at
# a 20% corruption rate runs of 3 still happen by chance; 5 does not
# (0.2^5 per report). The price is a few seconds of detection delay.
DM_CONSECUTIVE = 5


class _Kin:
    __slots__ = ("last_t", "last_pos", "run")

    def __init__(self) -> None:
        self.last_t: int | None = None
        self.last_pos: tuple[float, float, float] | None = None
        self.run = 0


class DataManipulationRule:
    """
    data_manipulation IF the drone's last DM_CONSECUTIVE answered reports
    each break a physical invariant:
      * rotors (almost) stopped while airborne, or
      * declared speed disagrees with the distance actually covered between
        its consecutive position reports by more than SPEED_TOLERANCE

    Fecak's injector corrupts speed, position, altitude and rotor speed
    together; checking invariants that tie these fields to each other catches
    that without knowing which values the attacker picked.
    """

    def __init__(self) -> None:
        self.kin: dict[str, _Kin] = {}

    @staticmethod
    def _num(v):
        return float(v) if isinstance(v, (int, float)) else None

    def handle(self, msg: dict) -> tuple[bool, str]:
        if msg["role"] != "target" or msg["exchange_type"] == C.TIMEOUT:
            return False, ""
        dt = msg.get("dt") or {}
        pos = tuple(self._num(dt.get(k)) for k in ("DT_Pos_X", "DT_Pos_Y", "DT_Pos_Z"))
        speed = self._num(dt.get("DT_Speed_mps"))
        rotor = self._num(dt.get("DT_RotorAvgSpeed"))
        if any(p is None for p in pos):
            return False, ""
        drone, t = msg["about"], msg["t_session"]
        k = self.kin.get(drone)
        if k is None:
            k = self.kin[drone] = _Kin()

        why = []
        if rotor is not None and rotor < ROTOR_MIN_AIRBORNE and pos[2] < AIRBORNE_Z:
            why.append(f"rotors at {rotor:g} while {-pos[2]:.0f} m up")
        if speed is not None and k.last_pos is not None and t > k.last_t:
            moved = sum((a - b) ** 2 for a, b in zip(pos, k.last_pos)) ** 0.5
            implied = moved / (t - k.last_t)
            if abs(implied - speed) > SPEED_TOLERANCE:
                why.append(f"declares {speed:g} m/s but moved {implied:.1f} m/s")
        # Every report becomes the next reference, glitched or not: holding
        # on to the "last good" position instead turns one early glitch into
        # an endless run of mismatches. A single position glitch therefore
        # costs two implausible reports (the jump and the jump back).
        if k.last_t is None or t > k.last_t:
            k.last_t, k.last_pos = t, pos

        k.run = k.run + 1 if why else 0
        if k.run < DM_CONSECUTIVE:
            return False, ""
        return True, (f"data manipulation: {k.run} consecutive reports break "
                      f"physics ({'; '.join(why)})")


# =====================================================================
# MITM: one drone's link degraded in both directions, answers slow
# =====================================================================
# Fecak's MITM injects 1.5 s of latency and 40% loss into every frame the
# victim SENDS. So requests to it go unanswered (its replies are lost),
# its own requests go unanswered (its requests are lost), and the answers
# that do arrive are slow.
#
# THESE THRESHOLDS WERE CHOSEN ON THE SAME RECORDINGS THEY ARE SCORED ON,
# and there is only one MITM recording. Attack sessions are noisy: honest
# drones lose 70-98% of exchanges in one direction for minutes at a time.
# tools/sweep_rules.py (results_rule_sweep.json): with a history of 60
# exchanges every setting flags 3-32 honest drones; with 100, failure
# shares 0.75-0.9 and a latency bar of 1.0-1.2 s are clean (0.9 catches the
# victim 75 s later), while a 0.8 s bar lets 1-2 honest drones through;
# with 150, shares 0.7-0.8 are clean but detection takes 80-130 s. So the
# clean region exists but is not wide, and there is a single MITM
# recording to check it on. Treat this rule's zero false positives as
# optimistic -- it is the weakest of the four, and the reason the ML stage
# exists.
MITM_HISTORY = 100        # last N exchanges per direction
MITM_FAIL_RATE = 0.75     # share unanswered, required in BOTH directions
MITM_MIN_RTT_S = 1.0      # median round trip of its last answers
MITM_RTT_SAMPLES = 8      # answers the median is taken over (= the RI window)
L_MAX = 5.0               # Fecak's latency normaliser in RI (Brno.py)


def recovered_rtt(trust: dict, answered: bool) -> float | None:
    """
    The round-trip time the requesting drone measured, recovered from the
    trust metrics it reported -- the CSV holds RI and CR, not the latency.

    Fecak (Brno.py): RI = 1 - (latency_norm + packet_loss) / 2,
                     latency_norm = min(avg_RTT / 5 s, 1)
                     CR = (SM + BFM + CDM + 1 - packet_loss) / 4  (answered)
                     CR = (1 - packet_loss) / 4                   (timeout)
    so packet_loss comes from CR, latency_norm from RI, RTT = 5 s x norm.
    These are inputs the drones computed in flight, not labels.
    """
    try:
        ri, cr = float(trust["Trust_RI"]), float(trust["Trust_CR"])
        if answered:
            loss = 1 - (4 * cr - float(trust["Trust_SM"]) - float(trust["Trust_BFM"])
                        - float(trust["Trust_CDM"]))
        else:
            loss = 1 - 4 * cr
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, L_MAX * (2 * (1 - ri) - loss))


class _Link:
    __slots__ = ("to_it", "from_it", "rtt")

    def __init__(self) -> None:
        self.to_it: deque[bool] = deque(maxlen=MITM_HISTORY)    # True = unanswered
        self.from_it: deque[bool] = deque(maxlen=MITM_HISTORY)
        self.rtt: deque[float] = deque(maxlen=MITM_RTT_SAMPLES)


class MITMRule:
    """
    mitm IF  >= MITM_FAIL_RATE of the last MITM_HISTORY requests TO the drone
             went unanswered
         AND >= MITM_FAIL_RATE of its own last MITM_HISTORY requests did too
         AND the median round trip of its last answers is >= MITM_MIN_RTT_S
         AND it still answers sometimes (else it is Critical Node or gone)

    Requests to unregistered identities do not count against the requester:
    in the Sybil session every drone keeps asking ghosts that never answer.
    """

    def __init__(self, registry: frozenset[str]) -> None:
        self.registry = registry
        self.links: dict[str, _Link] = {}

    def handle(self, msg: dict) -> tuple[bool, str]:
        drone = msg["about"]
        if not _registered(drone, self.registry):
            return False, ""
        timeout = msg["exchange_type"] == C.TIMEOUT
        ln = self.links.get(drone)
        if ln is None:
            ln = self.links[drone] = _Link()
        if msg["role"] == "target":
            ln.to_it.append(timeout)
            if not timeout:
                r = recovered_rtt(msg.get("trust") or {}, True)
                if r is not None:
                    ln.rtt.append(r)
        elif _registered(msg["contacted"], self.registry):
            ln.from_it.append(timeout)

        half = MITM_HISTORY // 2
        if len(ln.to_it) < half or len(ln.from_it) < half or len(ln.rtt) < 3:
            return False, ""
        fail_to = sum(ln.to_it) / len(ln.to_it)
        fail_from = sum(ln.from_it) / len(ln.from_it)
        if fail_to < MITM_FAIL_RATE or fail_from < MITM_FAIL_RATE or fail_to == 1.0:
            return False, ""
        rtt = sorted(ln.rtt)[len(ln.rtt) // 2]
        if rtt < MITM_MIN_RTT_S:
            return False, ""
        return True, (f"mitm: {fail_to:.0%} of requests to it and {fail_from:.0%} "
                      f"of its own requests unanswered, and its answers take "
                      f"{rtt:.1f} s (link degraded in both directions)")


# =====================================================================
# All four together
# =====================================================================
# Order is precedence when two rules fire on the same message. Each drone
# gets at most one rule verdict, so the first rule to fire names the type.
RULE_ATTACK_TYPE = {
    "SYBIL": "Sybil",
    "SUPPRESSION": "CriticalNode",
    "DATA_MANIPULATION": "DataManipulation",
    "MITM": "MITM",
}


class RuleEngine:
    """
    The cheap first stage of the cascade. handle() returns the verdict name
    of the first rule that fires for the message's drone, or None.
    """

    def __init__(self, streak: int = DEFAULT_STREAK,
                 registry: frozenset[str] | None = None) -> None:
        self.registry = registry if registry is not None else load_registry()
        self.suppression = SuppressionDetector(streak)
        self.sybil = SybilRule(self.registry)
        self.data_manipulation = DataManipulationRule()
        self.mitm = MITMRule(self.registry)

    def handle(self, msg: dict) -> tuple[str | None, str]:
        # every rule sees every message, so their state stays current
        results = (
            ("SYBIL", self.sybil.handle(msg)),
            ("SUPPRESSION", self.suppression.handle(msg)),
            ("DATA_MANIPULATION", self.data_manipulation.handle(msg)),
            ("MITM", self.mitm.handle(msg)),
        )
        for name, (fired, reason) in results:
            if fired:
                return name, reason
        return None, ""
