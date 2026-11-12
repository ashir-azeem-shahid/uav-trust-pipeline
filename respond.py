"""
respond.py
==========
The autonomous responder: turns verdicts into actions with no human needed,
while keeping a human able to override. "Human on the loop", not "in the
loop": the pipeline acts by itself; an operator watching the dashboard can
stop or reverse an action, but nothing waits for one.

Reads   uav.verdict    alerts from the detector (rule, classifier, anomaly)
        uav.operator   decisions from the dashboard
Writes  uav.action     what happened to each drone -- the topic a fleet
                       controller or ground station would subscribe to

POLICY (one line per signal, deliberately conservative)
------------------------------------------------------
  rule alert        ISOLATE automatically after a grace period (default
                    10 s). The rule had zero false alarms on every recorded
                    session and on the held-out clean recording, so acting
                    on it alone is defensible.
  classifier alert  AWAITING_OPERATOR: shown, never acted on automatically.
                    It is accurate on attack types it was trained on but can
                    miss or over-call types it has not seen (results_ml.json).
  anomaly           nothing: the dashboard shows it as WATCH. It means "this
                    looks unusual", which honest drones near an attack also do.

During the grace period an operator can isolate at once ("Isolate now") or
cancel ("Hold"); afterwards an isolated drone can be released. Every action
records who took it (auto or operator) and when, so the time from detection
to action is measurable.

"Isolate" here is a published decision: downstream systems stop trusting the
drone's telemetry and route around it. Commanding real aircraft is outside
this thesis; the pipeline's job ends at a correct, timely, explained action.

Usage
-----
  python respond.py                 # 10 s grace period
  python respond.py --grace 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import config as C

TOPIC_OPERATOR = "uav.operator"
TOPIC_ACTION = "uav.action"
AUTO_TRACKS = {"A"}          # signals the responder may act on alone
HUMAN_TRACKS = {"B-RF"}      # signals that wait for an operator


class Responder:
    """The policy, separate from Kafka so it can be tested on its own."""

    def __init__(self, grace_s: float = 10.0) -> None:
        self.grace = grace_s
        self.status: dict[str, str] = {}       # drone -> last action
        self.due: dict[str, float] = {}        # drone -> auto-isolate time
        self.detected_at: dict[str, float] = {}
        self.reason: dict[str, str] = {}

    def _act(self, drone: str, action: str, by: str, now: float, **extra) -> dict:
        self.status[drone] = action
        out = {"drone": drone, "action": action, "by": by, "at": now,
               "reason": self.reason.get(drone, "")}
        if drone in self.detected_at:
            out["seconds_since_alert"] = round(now - self.detected_at[drone], 2)
        out.update(extra)
        return out

    def on_verdict(self, v: dict, now: float) -> list[dict]:
        drone, track = v.get("drone"), v.get("track")
        if v.get("verdict") in (None, "SCORE", "QUARANTINE"):
            return []
        state = self.status.get(drone)
        if track in AUTO_TRACKS and state in (None, "AWAITING_OPERATOR"):
            self.detected_at.setdefault(drone, now)
            self.reason[drone] = v.get("reason", "")
            self.due[drone] = now + self.grace
            return [self._act(drone, "ISOLATE_PENDING", "auto", now,
                              due_at=self.due[drone], grace_s=self.grace,
                              t_session=v.get("t_session"))]
        if track in HUMAN_TRACKS and state is None:
            self.detected_at.setdefault(drone, now)
            self.reason[drone] = v.get("reason", "")
            return [self._act(drone, "AWAITING_OPERATOR", "auto", now,
                              t_session=v.get("t_session"))]
        return []

    def on_operator(self, d: dict, now: float) -> list[dict]:
        drone, decision = d.get("drone"), d.get("decision")
        state = self.status.get(drone)
        if decision == "ACCEPT" and state not in ("ISOLATED",):
            self.due.pop(drone, None)
            return [self._act(drone, "ISOLATED", "operator", now)]
        if decision == "REJECT" and state in ("ISOLATE_PENDING", "AWAITING_OPERATOR"):
            self.due.pop(drone, None)
            return [self._act(drone, "CANCELLED", "operator", now)]
        if decision == "RELEASE" and state == "ISOLATED":
            return [self._act(drone, "RELEASED", "operator", now)]
        return []

    def tick(self, now: float) -> list[dict]:
        out = []
        for drone, due in list(self.due.items()):
            if now >= due:
                del self.due[drone]
                out.append(self._act(drone, "ISOLATED", "auto", now))
        return out


def run(grace: float) -> None:
    try:
        from confluent_kafka import Consumer, Producer
    except ImportError:
        sys.exit("confluent-kafka not installed. pip install -r requirements.txt")

    consumer = Consumer({"bootstrap.servers": C.BOOTSTRAP,
                         "group.id": "uav-respond",
                         "auto.offset.reset": "earliest",
                         "enable.auto.commit": True})
    consumer.subscribe([C.TOPIC_VERDICT, TOPIC_OPERATOR])
    producer = Producer({"bootstrap.servers": C.BOOTSTRAP})
    r = Responder(grace)
    print(f"responding to {C.TOPIC_VERDICT} and {TOPIC_OPERATOR}, writing "
          f"{TOPIC_ACTION}. Rule alerts isolate after {grace:g} s unless an "
          f"operator overrides. Ctrl-C to stop.")

    def emit(actions):
        for a in actions:
            producer.produce(TOPIC_ACTION, key=a["drone"].encode(),
                             value=json.dumps(a).encode())
            extra = (f" in {a['grace_s']:g} s unless overridden"
                     if a["action"] == "ISOLATE_PENDING" else "")
            since = (f", {a['seconds_since_alert']} s after the alert"
                     if a["action"] != "ISOLATE_PENDING"
                     and "seconds_since_alert" in a else "")
            print(f"  {a['action']:<17} {a['drone']}  by {a['by']}{extra}{since}")
        if actions:
            producer.flush(5)

    try:
        while True:
            rec = consumer.poll(0.2)
            now = time.time()
            if rec is not None and not rec.error():
                payload = json.loads(rec.value())
                if rec.topic() == C.TOPIC_VERDICT:
                    emit(r.on_verdict(payload, now))
                else:
                    emit(r.on_operator(payload, now))
            emit(r.tick(time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush(5)
        consumer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grace", type=float, default=10.0,
                    help="seconds an operator has to override an automatic isolation")
    run(ap.parse_args().grace)
