"""
tools/simulate.py
=================
Proves the streaming architecture works BEFORE Kafka exists.

It replays the stream, partitions it by key exactly as Kafka would, and
gives each simulated consumer nothing but its own partition and a 15-second
sliding window per drone. No global view, no lookahead, no second pass.

If the suppression rule fires here, it will fire in detect.py, because
detect.py has strictly the same information.

What it measures:
  * whether the rule fires, and on which drones
  * detection delay: seconds from the known attack onset to the first
    correct verdict  <-- this is the headline number
  * false positives during the clean pre-attack period

Run:  python tools/simulate.py               (all attacks)
      python tools/simulate.py CriticalNode  (one)
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
import replay as R


def partition_of(key: str, n: int = C.N_PARTITIONS) -> int:
    """
    Stand-in for Kafka's default partitioner.

    Kafka uses murmur2 on the key bytes; the exact function does not matter
    here. What matters is that it is a pure function of the key, so both
    role copies of a drone's messages always land together -- which is the
    property the rule depends on.
    """
    h = 0
    for b in key.encode():
        h = (h * 31 + b) & 0xFFFFFFFF
    return h % n


class DroneWindow:
    """15-second sliding window of one drone's two-role history."""

    __slots__ = ("as_target", "as_initiator")

    def __init__(self) -> None:
        # each entry: (t_session, was_timeout)
        self.as_target: deque[tuple[int, bool]] = deque()
        self.as_initiator: deque[tuple[int, bool]] = deque()

    def add(self, role: str, t: int, is_timeout: bool) -> None:
        q = self.as_target if role == "target" else self.as_initiator
        q.append((t, is_timeout))

    def evict(self, now: int, span: int = C.WINDOW_SECONDS) -> None:
        for q in (self.as_target, self.as_initiator):
            while q and q[0][0] < now - span:
                q.popleft()

    # --- the rule -----------------------------------------------------
    def verdict(self, min_target_rows: int = 3, timeout_pct: float = 100.0):
        """
        Suppression if, inside the window:
          * the drone successfully initiated at least one exchange
            -> it is alive, in radio range, and its radio works
          * every exchange where it was the target timed out
            -> it is receiving and choosing not to answer

        Returns (fired: bool, reason: str) -- the reason string is what
        Track A gives you and the ML track cannot.
        """
        n_target = len(self.as_target)
        if n_target < min_target_rows:
            return False, ""

        n_to = sum(1 for _, to in self.as_target if to)
        pct = n_to / n_target * 100

        init_ok = sum(1 for _, to in self.as_initiator if not to)
        if init_ok == 0:
            # Never initiates: either quiet, or a Sybil ghost that does not exist.
            return False, ""

        if pct >= timeout_pct:
            return True, (
                f"suppression: {n_to}/{n_target} target exchanges timed out "
                f"({pct:.0f}%) while {init_ok} self-initiated exchanges succeeded "
                f"in the last {C.WINDOW_SECONDS}s"
            )
        return False, ""


def run(attack_name: str, min_target_rows: int = 3, timeout_pct: float = 100.0,
        verbose: bool = True) -> dict:
    # One independent state store per simulated consumer. A consumer can only
    # ever see drones whose key hashes into its partition.
    consumers: dict[int, dict[str, DroneWindow]] = defaultdict(
        lambda: defaultdict(DroneWindow))

    victim = None
    onset = None
    first_fire: dict[str, int] = {}      # drone -> t_session of first verdict
    fires_pre_onset: dict[str, int] = {}
    reasons: dict[str, str] = {}
    n_msgs = 0

    for key, msg, ons in R.replay_with_onset(attack_name):
        n_msgs += 1
        onset = ons
        if msg["_gt"]["attack_target"]:
            victim = msg["_gt"]["attack_target"]

        t = msg["t_session"]
        part = partition_of(key)
        win = consumers[part][key]

        win.add(msg["role"], t, msg["exchange_type"] == C.TIMEOUT)
        win.evict(t)

        fired, reason = win.verdict(min_target_rows, timeout_pct)
        if fired:
            if t < onset:
                fires_pre_onset.setdefault(key, t)
            elif key not in first_fire:
                first_fire[key] = t
                reasons[key] = reason

    delay = (first_fire.get(victim, None) or 0) - onset if victim in first_fire else None

    result = {
        "attack": attack_name,
        "victim": victim,
        "onset_second": onset,
        "messages_processed": n_msgs,
        "partitions_used": len(consumers),
        "fired_on": sorted(first_fire),
        "false_positives_pre_onset": sorted(fires_pre_onset),
        "detection_delay_s": delay,
        "reason": reasons.get(victim, ""),
    }

    if verbose:
        print(f"\n{'=' * 76}\n  {attack_name}   (victim: {victim})\n{'=' * 76}")
        print(f"  messages processed        : {n_msgs:,} across {len(consumers)} partitions")
        print(f"  attack onset at t=        : {onset}s "
              f"({onset}s of clean traffic first)")
        print(f"  rule fired on             : {sorted(first_fire) or '(nobody)'}")
        print(f"  false positives pre-onset : "
              f"{sorted(fires_pre_onset) or '(none)'}")
        if victim in first_fire:
            print(f"\n  *** DETECTED {victim} at t={first_fire[victim]}s"
                  f"  -->  DETECTION DELAY = {delay}s ***")
            print(f"  reason string: {reasons[victim]}")
        else:
            print(f"\n  victim {victim} NOT detected by this rule")
        wrong = [d for d in first_fire if d != victim]
        if wrong:
            print(f"  !! fired on non-victims: {wrong}")

    return result


if __name__ == "__main__":
    names = sys.argv[1:] or list(C.ATTACK_SESSIONS)
    out = [run(n) for n in names]

    print(f"\n{'=' * 76}\n  SUMMARY\n{'=' * 76}")
    print(f"  {'attack':<20} {'victim':<10} {'detected':<9} {'delay':>7}  false pos")
    for r in out:
        det = "YES" if r["victim"] in r["fired_on"] else "no"
        d = f"{r['detection_delay_s']}s" if r["detection_delay_s"] is not None else "-"
        fp = len(r["false_positives_pre_onset"]) + len(
            [x for x in r["fired_on"] if x != r["victim"]])
        print(f"  {r['attack']:<20} {str(r['victim']):<10} {det:<9} {d:>7}  {fp}")
