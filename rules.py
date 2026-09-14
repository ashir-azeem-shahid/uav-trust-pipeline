"""
rules.py
========
Track A: the deterministic suppression rule, as streaming state.

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
