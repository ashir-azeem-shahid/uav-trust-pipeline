"""
detect.py
=========
The consumer. Reads uav.telemetry, applies Track A, writes uav.verdict.

Right now it holds stage 3 only (the suppression rule). Stages 1, 2 and 4
slot in at the marked points -- the structure is deliberately laid out so
adding them does not move anything else.

  stage 1  DQ validation       -> dq.py          (next)
  stage 2  trust ingest         -> trust.py       (next)
  stage 3  suppression rule     -> rules.py       DONE
  stage 4  ML inference         -> model.pkl      (after train.py)

Usage
-----
  python detect.py                      # consume from Kafka
  python detect.py --offline Sybil      # replay straight through, no broker
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import config as C
from rules import SuppressionDetector


class Pipeline:
    """
    One pipeline instance per consumer process.

    State is per-drone and lives only for the drones this consumer is
    assigned -- which is the point of keying by the drone under evaluation.
    """

    def __init__(self, streak: int = None) -> None:
        self.suppression = SuppressionDetector(
            streak if streak is not None else __import__("rules").DEFAULT_STREAK)
        self.verdicts_emitted = 0
        self.messages_seen = 0
        self.already_flagged: set[str] = set()
        self.latencies: list[float] = []

    def handle(self, msg: dict) -> dict | None:
        """
        Process one message. Returns a verdict dict, or None if nothing to say.

        Reads only the operational fields. msg['_gt'] is ground truth and is
        never consulted here -- tests/test_leakage.py asserts that.
        """
        t_in = time.perf_counter()
        self.messages_seen += 1

        # ---- stage 1: DQ validation -------------------------------
        # dq_scores = dq.evaluate(msg)
        # if dq_scores["quarantine"]: return quarantine_verdict(...)

        # ---- stage 2: trust ingest --------------------------------
        # trust = trust.validate(msg["trust"])

        # ---- stage 3: suppression rule (Track A) ------------------
        fired, reason = self.suppression.handle(msg)

        # ---- stage 4: ML inference (Track B) ----------------------
        # ml_verdict = self.model.predict_one(features(msg, dq_scores, fired))

        self.latencies.append((time.perf_counter() - t_in) * 1000)

        drone = msg["about"]
        if fired and drone not in self.already_flagged:
            self.already_flagged.add(drone)
            self.verdicts_emitted += 1
            return {
                "drone": drone,
                "t_session": msg["t_session"],
                "verdict": "SUPPRESSION",
                "track": "A",
                "reason": reason,
                "onset": msg.get("onset"),
                "detection_delay_s": (msg["t_session"] - msg["onset"])
                                     if msg.get("onset") is not None else None,
            }
        return None

    def stats(self) -> dict:
        lat = sorted(self.latencies)
        def pct(p):
            return lat[int(len(lat) * p)] if lat else float("nan")
        return {
            "messages": self.messages_seen,
            "verdicts": self.verdicts_emitted,
            "per_msg_ms_p50": pct(0.50),
            "per_msg_ms_p95": pct(0.95),
            "per_msg_ms_p99": pct(0.99),
        }


# ---------------------------------------------------------------------
# offline mode -- no broker, straight from the replay
# ---------------------------------------------------------------------
def run_offline(attack: str) -> None:
    import replay as R

    pipe = Pipeline()
    victim = None

    for _key, msg, _onset in R.replay_with_onset(attack):
        if msg["_gt"]["attack_target"]:
            victim = msg["_gt"]["attack_target"]
        v = pipe.handle(msg)
        if v:
            mark = "CORRECT" if v["drone"] == victim else "FALSE POSITIVE"
            print(f"  [{mark}] {v['drone']} at t={v['t_session']}s  "
                  f"delay={v['detection_delay_s']}s")
            print(f"           {v['reason']}")

    s = pipe.stats()
    print(f"\n  ground-truth victim : {victim}")
    print(f"  messages processed  : {s['messages']:,}")
    print(f"  verdicts emitted    : {s['verdicts']}")
    print(f"  per-message latency : p50 {s['per_msg_ms_p50']:.3f}ms  "
          f"p95 {s['per_msg_ms_p95']:.3f}ms  p99 {s['per_msg_ms_p99']:.3f}ms")


# ---------------------------------------------------------------------
# Kafka mode
# ---------------------------------------------------------------------
def run_kafka() -> None:
    try:
        from confluent_kafka import Consumer, Producer
    except ImportError:
        sys.exit("confluent-kafka not installed. pip install -r requirements.txt\n"
                 "Or use: python detect.py --offline CriticalNode")

    consumer = Consumer({
        "bootstrap.servers": C.BOOTSTRAP,
        "group.id": "uav-detect",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    producer = Producer({"bootstrap.servers": C.BOOTSTRAP})
    consumer.subscribe([C.TOPIC_TELEMETRY])

    pipe = Pipeline()
    print(f"consuming {C.TOPIC_TELEMETRY}, writing {C.TOPIC_VERDICT}. Ctrl-C to stop.")

    try:
        while True:
            rec = consumer.poll(1.0)
            if rec is None:
                continue
            if rec.error():
                print("  kafka error:", rec.error(), file=sys.stderr)
                continue

            msg = json.loads(rec.value())
            v = pipe.handle(msg)
            if v:
                v["partition"] = rec.partition()
                producer.produce(C.TOPIC_VERDICT,
                                 key=v["drone"].encode(),
                                 value=json.dumps(v).encode())
                producer.poll(0)
                print(f"  VERDICT {v['drone']}  t={v['t_session']}s  "
                      f"delay={v['detection_delay_s']}s  (partition {rec.partition()})")
                print(f"          {v['reason']}")
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush(10)
        consumer.close()
        s = pipe.stats()
        print(f"\n  {s['messages']:,} messages, {s['verdicts']} verdicts, "
              f"p99 per-message {s['per_msg_ms_p99']:.3f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", metavar="ATTACK", choices=list(C.ATTACK_SESSIONS),
                    help="replay this session directly instead of reading Kafka")
    args = ap.parse_args()

    if args.offline:
        run_offline(args.offline)
    else:
        run_kafka()
