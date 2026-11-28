"""
detect.py
=========
The consumer. Reads uav.telemetry, runs every stage per message, writes
uav.verdict.

  stage 1  data quality        -> dq.py      QUARANTINE verdicts
  stage 3  rules (Track A)     -> rules.py   one cheap rule per attack type:
                                             SYBIL, SUPPRESSION (Critical Node),
                                             DATA_MANIPULATION, MITM
  stage 4  machine learning    -> ml.py      ANOMALY and ML_COMPROMISED
                                             verdicts (Track B), once per
                                             drone per 15 s window -- ONLY for
                                             drones no rule has flagged

The cascade: rules are cheap and explainable, so they go first. A drone a
rule has already named is not sent to the models (its windows are counted
in ml_gated instead). The models only see what the rules have not caught.
Every alert carries attack_type and the stage that caught it.

Usage
-----
  python detect.py                      # consume from Kafka, rule + DQ
  python detect.py --ml                 # ... plus both ML models
  python detect.py --offline Sybil --ml # replay straight through, no broker
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import config as C
from dq import DQValidator
from rules import RULE_ATTACK_TYPE, RuleEngine


class Pipeline:
    """
    One pipeline instance per consumer process.

    State is per-drone and lives only for the drones this consumer is
    assigned -- which is the point of keying by the drone under evaluation.
    """

    def __init__(self, streak: int = None, ml=None,
                 collect_windows: bool = False) -> None:
        """
        ml: None/False = rule and DQ only. True = also score with the trained
            models in models/ml.joblib (or pass a path).
        collect_windows: keep every closed ML window (train_ml.py uses this
            to build training data through this exact code path).
        """
        self.rules = RuleEngine(
            streak if streak is not None else __import__("rules").DEFAULT_STREAK)
        # the Critical Node rule's per-drone state: the ML windows and the
        # dashboard read streaks from it
        self.suppression = self.rules.suppression
        self.dq = DQValidator()
        self.verdicts_emitted = 0
        self.messages_seen = 0
        self.quarantined = 0
        self.dq_scores: list[float] = []
        self.last_dq = None          # most recent DQResult, for features.py
        self.already_flagged: set[str] = set()
        # drone -> (verdict, attack_type, t_session) of the rule that caught it
        self.rule_hits: dict[str, tuple[str, str, int]] = {}
        self.ml_gated = 0            # windows not scored: a rule already caught the drone
        self.latencies: list[float] = []
        # ---- Track B state ----
        from ml import MODEL_PATH, MLScorer, SubjectWindower
        self.windower = SubjectWindower() if (ml or collect_windows) else None
        self.scorer = (MLScorer(MODEL_PATH if ml is True else ml) if ml else None)
        self.windows: list | None = [] if collect_windows else None
        self.ml_pending: list[dict] = []

    def handle(self, msg: dict, partition: int | None = None) -> dict | None:
        """
        Process one message. Returns a verdict dict, or None if nothing to say.

        Reads only the operational fields. msg['_gt'] is ground truth and is
        never consulted here -- tests/test_leakage.py asserts that.

        partition: the Kafka partition the message came from, if any. Only
        the ML windows use it (they keep time per partition).
        """
        t_in = time.perf_counter()
        self.messages_seen += 1

        # ---- stage 1: DQ validation -------------------------------
        # Scored once per exchange, on the 'target' copy -- the initiator
        # copy carries the same payload and would double-count.
        dq_res = None
        if msg["role"] == "target":
            dq_res = self.dq.evaluate(msg)
            self.last_dq = dq_res
            self.dq_scores.append(dq_res.score)
            if dq_res.quarantine:
                self.quarantined += 1
                self._ml(msg, dq_res, partition)
                self.latencies.append((time.perf_counter() - t_in) * 1000)
                return {
                    "drone": msg["contacted"],
                    "t_session": msg["t_session"],
                    "verdict": "QUARANTINE",
                    "track": "DQ",
                    "reason": "; ".join(dq_res.failed_rules),
                    "dq_score": round(dq_res.score, 4),
                }

        # ---- stage 2: trust ingest --------------------------------
        # trust = trust.validate(msg["trust"])

        # ---- stage 3: rules (Track A) ------------------------------
        fired, reason = self.rules.handle(msg)

        # ---- stage 4: ML (Track B) -------------------------------
        # Adds this message to its drone's 15 s window. When the stream
        # clock passes a window's end, the window is scored by both models;
        # any resulting verdicts wait in ml_pending for drain_ml().
        self._ml(msg, dq_res, partition)

        self.latencies.append((time.perf_counter() - t_in) * 1000)

        drone = msg["about"]
        if fired and drone not in self.already_flagged:
            self.already_flagged.add(drone)
            self.rule_hits[drone] = (fired, RULE_ATTACK_TYPE[fired], msg["t_session"])
            self.verdicts_emitted += 1
            return {
                "drone": drone,
                "t_session": msg["t_session"],
                "verdict": fired,
                "track": "A",
                "attack_type": RULE_ATTACK_TYPE[fired],
                "stage": "rule",
                "reason": reason,
                "onset": msg.get("onset"),
                "detection_delay_s": (msg["t_session"] - msg["onset"])
                                     if msg.get("onset") is not None else None,
            }
        return None

    def _ml(self, msg: dict, dq_res, partition=None) -> None:
        if self.windower is None:
            return
        closed = self.windower.observe(
            msg, dq_res, self.suppression.drones.get(msg["about"]), partition)
        self._take(closed)

    def _take(self, closed: list) -> None:
        if not closed:
            return
        if self.windows is not None:
            self.windows.extend(closed)
        if self.scorer is not None:
            # the cascade gate: a drone a rule already caught is not scored
            todo = [w for w in closed if w.subject not in self.already_flagged]
            self.ml_gated += len(closed) - len(todo)
            self.ml_pending.extend(self.scorer.verdicts(todo))

    def drain_ml(self) -> list[dict]:
        """ML verdicts produced since the last call (Track B)."""
        out, self.ml_pending = self.ml_pending, []
        return out

    def finish(self) -> None:
        """End of stream: close and score the windows still open."""
        if self.windower is not None:
            self._take(self.windower.flush())

    def stats(self) -> dict:
        lat = sorted(self.latencies)
        def pct(p):
            return lat[int(len(lat) * p)] if lat else float("nan")
        return {
            "messages": self.messages_seen,
            "verdicts": self.verdicts_emitted,
            "quarantined": self.quarantined,
            "ml_windows_gated": self.ml_gated,
            "dq_mean": (sum(self.dq_scores) / len(self.dq_scores))
                       if self.dq_scores else float("nan"),
            "per_msg_ms_p50": pct(0.50),
            "per_msg_ms_p95": pct(0.95),
            "per_msg_ms_p99": pct(0.99),
        }


# ---------------------------------------------------------------------
# offline mode -- no broker, straight from the replay
# ---------------------------------------------------------------------
def run_offline(attack: str, use_ml: bool = False) -> None:
    import replay as R

    pipe = Pipeline(ml=use_ml)
    victim = attack_type = None

    def show(v):
        if v["verdict"] in ("QUARANTINE", "SCORE"):
            return
        # Sybil's malicious identities are the fabricated Ghost drones
        bad = ("ghost" in v["drone"].lower()
               if (attack_type or "").lower() == "sybil" else v["drone"] == victim)
        mark = "CORRECT" if bad else "FALSE POSITIVE"
        print(f"  [{mark}] {v['verdict']:<17} {v['drone']} as "
              f"{v.get('attack_type') or '?'} ({v.get('stage', 'ml')}) at "
              f"t={v['t_session']}s  delay={v['detection_delay_s']}s")
        print(f"           {v['reason']}")

    for _key, msg, _onset in R.replay_with_onset(attack):
        if msg["_gt"]["attack_target"]:
            victim = msg["_gt"]["attack_target"]
            attack_type = msg["_gt"]["attack_type"]
        v = pipe.handle(msg)
        if v:
            show(v)
        for mv in pipe.drain_ml():
            show(mv)
    pipe.finish()
    for mv in pipe.drain_ml():
        show(mv)

    s = pipe.stats()
    print(f"\n  ground-truth victim : {victim}")
    print(f"  messages processed  : {s['messages']:,}")
    print(f"  verdicts emitted    : {s['verdicts']}")
    print(f"  mean dq_score       : {s['dq_mean']:.4f}  "
          f"(quarantined {s['quarantined']})")
    print(f"  per-message latency : p50 {s['per_msg_ms_p50']:.3f}ms  "
          f"p95 {s['per_msg_ms_p95']:.3f}ms  p99 {s['per_msg_ms_p99']:.3f}ms")


# ---------------------------------------------------------------------
# Kafka mode
# ---------------------------------------------------------------------
def run_kafka(use_ml: bool = False) -> None:
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

    pipe = Pipeline(ml=use_ml)
    partition_of: dict[str, int] = {}     # drone -> its partition, for ML verdicts
    print(f"consuming {C.TOPIC_TELEMETRY}, writing {C.TOPIC_VERDICT}"
          f"{' (rule + DQ + ML)' if use_ml else ' (rule + DQ)'}. Ctrl-C to stop.")

    def publish(v: dict) -> None:
        producer.produce(C.TOPIC_VERDICT, key=v["drone"].encode(),
                         value=json.dumps(v).encode())
        producer.poll(0)
        if v["verdict"] == "SCORE":
            return                  # per-window ML reading: published, not printed
        if v["verdict"] == "QUARANTINE":
            # Data quality, not an attack: a record too broken to use.
            # (No detection delay -- there is no attack to time.)
            print(f"  QUARANTINE {v['drone']}  t={v['t_session']}s  "
                  f"dq={v.get('dq_score')}  (partition {v['partition']})")
        else:
            print(f"  {v['verdict']} {v['drone']}  t={v['t_session']}s  "
                  f"delay={v.get('detection_delay_s')}s  "
                  f"(partition {v['partition']})")
        print(f"          {v['reason']}")

    try:
        while True:
            rec = consumer.poll(1.0)
            if rec is None:
                continue
            if rec.error():
                print("  kafka error:", rec.error(), file=sys.stderr)
                continue

            msg = json.loads(rec.value())
            partition_of[msg["about"]] = rec.partition()
            v = pipe.handle(msg, rec.partition())
            if v:
                v["partition"] = rec.partition()
                publish(v)
            for mv in pipe.drain_ml():
                mv["partition"] = partition_of.get(mv["drone"])
                publish(mv)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush(10)
        consumer.close()
        s = pipe.stats()
        ml_txt = (f", {pipe.scorer.windows_scored:,} ML windows scored "
                  f"({pipe.windower.late} late messages)" if pipe.scorer else "")
        print(f"\n  {s['messages']:,} messages, {s['verdicts']} verdicts, "
              f"{s['quarantined']} quarantined{ml_txt}, "
              f"p99 per-message {s['per_msg_ms_p99']:.3f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", metavar="ATTACK", choices=list(C.ATTACK_SESSIONS),
                    help="replay this session directly instead of reading Kafka")
    ap.add_argument("--ml", action="store_true",
                    help="also run the Track B models (train_ml.py writes them)")
    args = ap.parse_args()

    if args.offline:
        run_offline(args.offline, args.ml)
    else:
        run_kafka(args.ml)
