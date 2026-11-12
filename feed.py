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
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import config as C

TOPIC_OPERATOR = "uav.operator"
TRAJECTORY_POINTS = 400
DQ_DIMENSIONS = ["completeness", "validity", "timeliness", "uniqueness", "consistency"]
RATE_WINDOW_S = 5.0


def _rule_dimension(rule: str) -> str:
    """'validity: DT_Speed_mps=47 outside [0,12]' -> 'validity'."""
    return rule.split(":", 1)[0].strip()


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
    dq_flagged: int = 0          # records with at least one failed DQ rule
    dq_quarantined: int = 0
    dq_issues: Counter = field(default_factory=Counter)   # dimension -> count

    @property
    def timeout_rate(self) -> float:
        return self.n_timeout / self.n_target if self.n_target else 0.0

    @property
    def last_trust(self) -> float | None:
        return self.trust[-1] if self.trust else None

    @property
    def band(self) -> str:
        """Trust band from the last ANSWERED exchange (timeouts carry no score)."""
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
        # pipeline view
        self.partitions: Counter = Counter()          # live only
        self.rate_samples: deque = deque()            # (wall time, messages)
        self.peak_rate: float = 0.0                   # highest msg/s seen
        # data quality view (from the DQ result of every 'target' record)
        self.dq_checked: int = 0
        self.dq_flagged: int = 0
        self.dq_quarantined: int = 0
        self.dim_sum: dict[str, float] = {d: 0.0 for d in DQ_DIMENSIONS}
        self.dim_fail: Counter = Counter()
        self.recent_dq_issues: deque = deque(maxlen=60)  # newest last

    # -- ingestion -----------------------------------------------------
    def apply_telemetry(self, msg: dict, dq_score=None,
                        streak: int | None = None,
                        partition: int | None = None) -> None:
        """
        dq_score may be a plain float (older callers) or the full DQResult
        from dq.py, in which case the five dimensions are tracked too.
        """
        dq_res = None
        if dq_score is not None and not isinstance(dq_score, (int, float)):
            dq_res = dq_score
            dq_score = dq_res.score
        self.messages += 1
        if partition is not None:
            self.partitions[partition] += 1
        self.clock = max(self.clock, msg.get("t_session", 0))
        if msg.get("onset") is not None:
            self.onset = msg["onset"]

        drone = msg.get("about") or msg.get("contacted")
        d = self.drones[drone]
        d.drone = drone
        d.last_seen = msg.get("t_session", 0)

        if msg.get("role") == "target" and dq_res is not None:
            self._apply_dq(msg, d, dq_res)

        if msg.get("role") == "target":
            d.n_target += 1
            if msg.get("exchange_type") == C.TIMEOUT:
                d.n_timeout += 1
            # The simulator writes Trust_Score = 0 on every timeout: there is no
            # payload to assess, so that 0 is not a judgement of the drone.
            # Timeouts are counted by the streak and timeout rate instead; the
            # trust trajectory and status use answered exchanges only.
            ts = (msg.get("trust") or {}).get("Trust_Score")
            quarantined = dq_res is not None and dq_res.quarantine
            if (ts is not None and msg.get("exchange_type") != C.TIMEOUT
                    and not quarantined):           # quarantined = out of trust path
                d.trust.append(ts)
                d.times.append(msg.get("t_session", 0))
                if dq_score is not None:
                    d.dq.append(dq_score)

        if streak is not None:
            d.streak = streak
            d.max_streak = max(d.max_streak, streak)

    def _apply_dq(self, msg: dict, d: "DroneView", res) -> None:
        self.dq_checked += 1
        for dim in DQ_DIMENSIONS:
            val = getattr(res, dim)
            self.dim_sum[dim] += val
            if val < 1.0:
                self.dim_fail[dim] += 1
                d.dq_issues[dim] += 1
        if res.failed_rules:
            self.dq_flagged += 1
            d.dq_flagged += 1
            self.recent_dq_issues.append({
                "t": msg.get("t_session", 0),
                "drone": msg.get("contacted"),
                "dimension": _rule_dimension(res.failed_rules[0]),
                "rule": res.failed_rules[0],
                "quarantined": res.quarantine,
            })
        if res.quarantine:
            self.dq_quarantined += 1
            d.dq_quarantined += 1

    # -- pipeline / data quality summaries ------------------------------
    def mark_rate(self, now: float | None = None) -> None:
        """Record (time, messages) after a poll, for the live msg/s figure."""
        now = time.time() if now is None else now
        self.rate_samples.append((now, self.messages))
        while self.rate_samples and now - self.rate_samples[0][0] > RATE_WINDOW_S:
            self.rate_samples.popleft()
        r = self.rate()
        if r and r > self.peak_rate:
            self.peak_rate = r

    def rate(self) -> float | None:
        if len(self.rate_samples) < 2:
            return None
        (t0, m0), (t1, m1) = self.rate_samples[0], self.rate_samples[-1]
        return (m1 - m0) / (t1 - t0) if t1 > t0 else None

    def dq_summary(self) -> dict:
        n = self.dq_checked
        return {
            "checked": n,
            "flagged": self.dq_flagged,
            "flagged_pct": (self.dq_flagged / n * 100) if n else 0.0,
            "quarantined": self.dq_quarantined,
            "dimensions": [{
                "dimension": dim,
                "mean_score": (self.dim_sum[dim] / n) if n else None,
                "records_failing": self.dim_fail[dim],
                "failing_pct": (self.dim_fail[dim] / n * 100) if n else 0.0,
            } for dim in DQ_DIMENSIONS],
        }

    def dq_table(self) -> list[dict]:
        rows = []
        for name, d in self.drones.items():
            if not d.dq:
                continue
            rows.append({
                "drone": name,
                "mean_dq": round(sum(d.dq) / len(d.dq), 4),
                "min_dq": round(min(d.dq), 4),
                "flagged": d.dq_flagged,
                "quarantined": d.dq_quarantined,
                "main_issue": (d.dq_issues.most_common(1)[0][0]
                               if d.dq_issues else ""),
                "records": len(d.dq),
            })
        rows.sort(key=lambda r: (r["mean_dq"], -r["flagged"]))
        return rows

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
                 group: str = None, start: str = "earliest") -> None:
        """
        start: "earliest" replays everything already on the topics;
               "latest" shows only what arrives after connecting -- use it
               for a live demo, so results of earlier runs don't reappear.

        The dashboard is its own consumer group, independent of detect.py:
        Kafka fans the same stream out to both. It keeps its own read model
        (DQ scores, streaks) by running the same deterministic per-drone code
        on the same per-partition order, so it matches the detector.
        """
        from confluent_kafka import Consumer, Producer
        if start not in ("earliest", "latest"):
            raise ValueError("start must be 'earliest' or 'latest'")
        group = group or f"uav-dashboard-{int(time.time())}"
        self.bootstrap = bootstrap
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": start,
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
                dq_res = None
                if payload.get("role") == "target":
                    dq_res = self._dq.evaluate(payload)
                self._sup.handle(payload)
                st = self._sup.drones.get(payload.get("about"))
                state.apply_telemetry(payload, dq_res,
                                      st.streak if st else None,
                                      partition=rec.partition())
            n += 1
        state.mark_rate()
        return n

    def publish_decision(self, v: Verdict) -> str | None:
        """
        Publish an operator decision to uav.operator and wait for the broker
        to confirm it. Returns None on success, or the error text -- so the
        UI can say so instead of failing silently.
        """
        errors: list[str] = []

        def on_delivery(err, _msg):
            if err is not None:
                errors.append(str(err))

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
            on_delivery=on_delivery,
        )
        left = self.producer.flush(5)
        if errors:
            return errors[0]
        if left:
            return "broker did not confirm the decision within 5 s"
        return None


# ---------------------------------------------------------------------
# Offline source -- same state, no broker. Used for development and for
# reproducible evaluation runs.
# ---------------------------------------------------------------------
def build_offline(attack: str, until: int | None = None,
                  fault: str | None = None, fault_rate: float = 0.05,
                  fault_seed: int = 7) -> LiveState:
    """
    Same state the live feed builds, straight from the recorded session.
    fault optionally injects one benign fault type (see faults.py) into a
    share of records, exactly as produce.py --fault does on the live stream.
    """
    import faults as F
    import replay as R
    from detect import Pipeline

    state = LiveState()
    pipe = Pipeline()

    stream = F.inject(R.replay_with_onset(attack), fault, fault_rate, fault_seed)
    for _key, msg, _onset, _is_fault in stream:
        if until is not None and msg["t_session"] > until:
            break
        v = pipe.handle(msg)
        dq_res = pipe.last_dq if msg["role"] == "target" else None
        st = pipe.suppression.drones.get(msg["about"])
        state.apply_telemetry(msg, dq_res, st.streak if st else None)
        if msg["_gt"].get("attack_type"):
            state.attack_type = msg["_gt"]["attack_type"]
        if v:
            state.apply_verdict(v)
    return state
