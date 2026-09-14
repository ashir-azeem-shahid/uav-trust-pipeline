"""
feed.py
=======
The dashboard's data layer, kept separate from the UI so it can be tested
without running Streamlit.

Holds live state built from the two Kafka topics:
  uav.telemetry  -> per-drone trust trajectory, DQ scores, streak
  uav.verdict    -> the alert feed an operator acts on

Operator decisions go back out to uav.operator, which makes the
accept/reject loop measurable rather than cosmetic: every verdict gets a
recorded human decision, and the share of verdicts an operator could act
on is the concrete form of "explainability" in RQ3.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import config as C

TOPIC_OPERATOR = "uav.operator"
TRAJECTORY_POINTS = 400


@dataclass
class DroneView:
    """What the dashboard knows about one drone right now."""
    drone: str = ""
    trust: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_POINTS))
    dq: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_POINTS))
    times: deque = field(default_factory=lambda: deque(maxlen=TRAJECTORY_POINTS))
    streak: int = 0
    max_streak: int = 0
    n_target: int = 0
    n_timeout: int = 0
    last_seen: int = 0
    flagged: bool = False

    @property
    def timeout_rate(self) -> float:
        return self.n_timeout / self.n_target if self.n_target else 0.0

    @property
    def last_trust(self) -> float | None:
        return self.trust[-1] if self.trust else None

    @property
    def band(self) -> str:
        """Trust band for the swarm map colouring."""
        if self.flagged:
            return "FLAGGED"
        t = self.last_trust
        if t is None:
            return "UNKNOWN"
        if t >= 0.6:
            return "TRUSTED"
        if t >= 0.3:
            return "DEGRADED"
        return "LOW"


@dataclass
class Verdict:
    drone: str
    t_session: int
    verdict: str
    track: str
    reason: str
    detection_delay_s: int | None = None
    dq_score: float | None = None
    partition: int | None = None
    decision: str | None = None        # operator: ACCEPT / REJECT / None
    decided_at: float | None = None

    @property
    def vid(self) -> str:
        return f"{self.drone}@{self.t_session}:{self.track}"

    @property
    def actionable(self) -> bool:
        """
        Can a human act on this without further investigation?

        True when the verdict carries a reason string naming the evidence.
        Track A always does; a bare model score does not. This is the
        measurable form of RQ3's explainability question.
        """
        return bool(self.reason and self.reason.strip())


class LiveState:
    """
    Accumulates dashboard state. Fed either from Kafka or, for development
    and for the evaluation runs, straight from the replay.
    """

    def __init__(self) -> None:
        self.drones: dict[str, DroneView] = defaultdict(DroneView)
        self.verdicts: list[Verdict] = []
        self.onset: int | None = None
        self.clock: int = 0
        self.messages: int = 0
        self.quarantined: int = 0
        self.attack_type: str | None = None

    # -- ingestion -----------------------------------------------------
    def apply_telemetry(self, msg: dict, dq_score: float | None = None,
                        streak: int | None = None) -> None:
        self.messages += 1
        self.clock = max(self.clock, msg.get("t_session", 0))
        if msg.get("onset") is not None:
            self.onset = msg["onset"]

        drone = msg.get("about") or msg.get("contacted")
        d = self.drones[drone]
        d.drone = drone
        d.last_seen = msg.get("t_session", 0)

        if msg.get("role") == "target":
            d.n_target += 1
            if msg.get("exchange_type") == C.TIMEOUT:
                d.n_timeout += 1
            ts = (msg.get("trust") or {}).get("Trust_Score")
            if ts is not None:
                d.trust.append(ts)
                d.times.append(msg.get("t_session", 0))
                if dq_score is not None:
                    d.dq.append(dq_score)

        if streak is not None:
            d.streak = streak
            d.max_streak = max(d.max_streak, streak)

    def apply_verdict(self, v: dict) -> Verdict:
        ver = Verdict(
            drone=v["drone"],
            t_session=v.get("t_session", 0),
            verdict=v.get("verdict", ""),
            track=v.get("track", ""),
            reason=v.get("reason", ""),
            detection_delay_s=v.get("detection_delay_s"),
            dq_score=v.get("dq_score"),
            partition=v.get("partition"),
        )
        self.verdicts.append(ver)
        if ver.verdict == "QUARANTINE":
            self.quarantined += 1
        else:
            self.drones[ver.drone].flagged = True
        return ver

    # -- operator loop -------------------------------------------------
    def decide(self, vid: str, decision: str) -> Verdict | None:
        """Record an operator's accept/reject on one verdict."""
        for v in self.verdicts:
            if v.vid == vid:
                v.decision = decision
                v.decided_at = time.time()
                if decision == "REJECT":
                    self.drones[v.drone].flagged = False
                return v
        return None

    # -- summaries the UI renders --------------------------------------
    def alert_summary(self) -> dict:
        alerts = [v for v in self.verdicts if v.verdict != "QUARANTINE"]
        actionable = [v for v in alerts if v.actionable]
        decided = [v for v in alerts if v.decision]
        return {
            "alerts": len(alerts),
            "actionable": len(actionable),
            "actionable_pct": (len(actionable) / len(alerts) * 100) if alerts else 0.0,
            "accepted": sum(1 for v in decided if v.decision == "ACCEPT"),
            "rejected": sum(1 for v in decided if v.decision == "REJECT"),
            "pending": len(alerts) - len(decided),
            "quarantined": self.quarantined,
        }

    def swarm_table(self) -> list[dict]:
        rows = []
        for name, d in self.drones.items():
            if not d.n_target:
                continue
            rows.append({
                "drone": name,
                "band": d.band,
                "trust": round(d.last_trust, 4) if d.last_trust is not None else None,
                "timeout_rate": round(d.timeout_rate, 3),
                "streak": d.streak,
                "max_streak": d.max_streak,
                "exchanges": d.n_target,
            })
        rows.sort(key=lambda r: (-r["max_streak"], -r["timeout_rate"]))
        return rows


# ---------------------------------------------------------------------
# Kafka source
# ---------------------------------------------------------------------
class KafkaFeed:
    """Non-blocking poll over both topics. Safe to call from a UI refresh."""

    def __init__(self, bootstrap: str = C.BOOTSTRAP,
                 group: str = None) -> None:
        from confluent_kafka import Consumer, Producer
        group = group or f"uav-dashboard-{int(time.time())}"
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })
        self.consumer.subscribe([C.TOPIC_TELEMETRY, C.TOPIC_VERDICT])
        self.producer = Producer({"bootstrap.servers": bootstrap})

    def drain(self, state: LiveState, max_messages: int = 4000,
              budget_s: float = 0.8) -> int:
        """Pull what's waiting, bounded so the UI stays responsive."""
        from dq import DQValidator
        from rules import SuppressionDetector

        if not hasattr(self, "_dq"):
            self._dq = DQValidator()
            self._sup = SuppressionDetector()

        deadline = time.time() + budget_s
        n = 0
        while n < max_messages and time.time() < deadline:
            rec = self.consumer.poll(0.01)
            if rec is None:
                break
            if rec.error():
                continue
            payload = json.loads(rec.value())
            if rec.topic() == C.TOPIC_VERDICT:
                state.apply_verdict(payload)
            else:
                dq_score = None
                if payload.get("role") == "target":
                    dq_score = self._dq.evaluate(payload).score
                self._sup.handle(payload)
                st = self._sup.drones.get(payload.get("about"))
                state.apply_telemetry(payload, dq_score,
                                      st.streak if st else None)
            n += 1
        return n

    def publish_decision(self, v: Verdict) -> None:
        self.producer.produce(
            TOPIC_OPERATOR,
            key=v.drone.encode(),
            value=json.dumps({
                "drone": v.drone,
                "verdict_id": v.vid,
                "track": v.track,
                "decision": v.decision,
                "reason_shown": v.reason,
                "actionable": v.actionable,
                "decided_at": v.decided_at,
            }).encode(),
        )
        self.producer.poll(0)


# ---------------------------------------------------------------------
# Offline source -- same state, no broker. Used for development and for
# reproducible evaluation runs.
# ---------------------------------------------------------------------
def build_offline(attack: str, until: int | None = None) -> LiveState:
    import replay as R
    from detect import Pipeline

    state = LiveState()
    pipe = Pipeline()

    for _key, msg, _onset in R.replay_with_onset(attack):
        if until is not None and msg["t_session"] > until:
            break
        v = pipe.handle(msg)
        dq_score = None
        if msg["role"] == "target" and pipe.dq_scores:
            dq_score = pipe.dq_scores[-1]
        st = pipe.suppression.drones.get(msg["about"])
        state.apply_telemetry(msg, dq_score, st.streak if st else None)
        if msg["_gt"].get("attack_type"):
            state.attack_type = msg["_gt"]["attack_type"]
        if v:
            state.apply_verdict(v)
    return state
