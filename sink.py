"""
sink.py
=======
The storage tier. Reads verdicts and enriched telemetry off Kafka and
writes them to Parquet, locally or straight to S3.

This is the "storage" clause of the thesis topic, and it is deliberately
small -- about forty lines of real work -- because the contribution is the
detection, not the archive.

WHY PARQUET AND NOT CSV
-----------------------
Parquet is columnar: it stores each column together instead of each row.
Reading one column out of thirty touches only that column's bytes. For an
archive that gets queried by attack type or by drone, that is the
difference between scanning gigabytes and scanning megabytes. It also
compresses far better, because a column holds values of one type.

PARTITIONING
------------
Files are written under session=<name>/ so a query for one attack reads
only that directory. This is the same partition-pruning idea as Kafka
partitioning, applied to storage rather than to streams.

Usage
-----
  python sink.py --local out/                 write Parquet locally
  python sink.py --s3 s3://my-bucket/uav/     write to S3
  python sink.py --local out/ --from-replay CriticalNode
                                              no broker; archive a replay
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import config as C

BATCH = 5000


def _flush(records: list[dict], dest: str, session: str, part: int,
           use_s3: bool) -> str:
    import pandas as pd

    df = pd.DataFrame(records)
    name = f"part-{part:05d}.parquet"

    if use_s3:
        # s3fs is picked up by pandas via the s3:// prefix; boto3 credentials
        # come from the environment or the EC2 instance role.
        path = f"{dest.rstrip('/')}/session={session}/{name}"
        df.to_parquet(path, index=False, compression="snappy")
    else:
        folder = Path(dest) / f"session={session}"
        folder.mkdir(parents=True, exist_ok=True)
        path = str(folder / name)
        df.to_parquet(path, index=False, compression="snappy")
    return path


def _flatten(msg: dict, verdict: dict | None, dq_score: float | None) -> dict:
    """One flat row per exchange: telemetry + trust + DQ + verdict."""
    row = {
        "t_session": msg.get("t_session"),
        "ts_original": msg.get("ts_original"),
        "requester": msg.get("requester"),
        "contacted": msg.get("contacted"),
        "swarm": msg.get("swarm"),
        "exchange_type": msg.get("exchange_type"),
        "distance_m": msg.get("distance_m"),
        "dq_score": dq_score,
        "verdict": verdict["verdict"] if verdict else None,
        "verdict_track": verdict["track"] if verdict else None,
        "verdict_reason": verdict["reason"] if verdict else None,
        "detection_delay_s": verdict.get("detection_delay_s") if verdict else None,
    }
    for k, v in (msg.get("trust") or {}).items():
        row[k] = v
    for k, v in (msg.get("dt") or {}).items():
        row[k] = v
    # Ground truth is archived for evaluation, clearly prefixed so it can
    # never be mistaken for a feature.
    gt = msg.get("_gt") or {}
    row["gt_attack_type"] = gt.get("attack_type")
    row["gt_attack_target"] = gt.get("attack_target")
    row["gt_malicious"] = gt.get("malicious")
    return row


def from_replay(attack: str, dest: str, use_s3: bool) -> None:
    import replay as R
    from detect import Pipeline

    pipe = Pipeline()
    buf: list[dict] = []
    part = 0
    written = []

    for _k, msg, _o in R.replay_with_onset(attack):
        v = pipe.handle(msg)
        if msg["role"] != "target":
            continue
        dq = pipe.dq_scores[-1] if pipe.dq_scores else None
        buf.append(_flatten(msg, v, dq))
        if len(buf) >= BATCH:
            written.append(_flush(buf, dest, attack, part, use_s3))
            buf, part = [], part + 1

    if buf:
        written.append(_flush(buf, dest, attack, part, use_s3))

    total = sum(os.path.getsize(p) for p in written if not use_s3
                and os.path.exists(p))
    print(f"  {attack}: wrote {len(written)} file(s)"
          + (f", {total/1024:,.0f} KB" if not use_s3 else ""))
    for p in written:
        print(f"    {p}")


def from_kafka(dest: str, use_s3: bool, seconds: int = 60) -> None:
    from confluent_kafka import Consumer
    from detect import Pipeline

    consumer = Consumer({
        "bootstrap.servers": C.BOOTSTRAP,
        "group.id": "uav-sink",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([C.TOPIC_TELEMETRY])

    pipe = Pipeline()
    buf: list[dict] = []
    part = 0
    deadline = time.time() + seconds
    print(f"  archiving for {seconds}s from {C.TOPIC_TELEMETRY} …")

    try:
        while time.time() < deadline:
            rec = consumer.poll(1.0)
            if rec is None:
                continue
            if rec.error():
                continue
            msg = json.loads(rec.value())
            v = pipe.handle(msg)
            if msg.get("role") != "target":
                continue
            dq = pipe.dq_scores[-1] if pipe.dq_scores else None
            buf.append(_flatten(msg, v, dq))
            if len(buf) >= BATCH:
                print("   ", _flush(buf, dest, "live", part, use_s3))
                buf, part = [], part + 1
    except KeyboardInterrupt:
        pass
    finally:
        if buf:
            print("   ", _flush(buf, dest, "live", part, use_s3))
        consumer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", metavar="DIR", help="write Parquet to this folder")
    g.add_argument("--s3", metavar="S3_URI",
                   help="write Parquet to S3, e.g. s3://bucket/uav/")
    ap.add_argument("--from-replay", metavar="ATTACK",
                    help="archive a replayed session instead of reading Kafka")
    ap.add_argument("--seconds", type=int, default=60,
                    help="how long to consume in Kafka mode")
    args = ap.parse_args()

    dest = args.s3 or args.local
    use_s3 = bool(args.s3)

    if args.from_replay:
        sessions = ([args.from_replay] if args.from_replay != "all"
                    else list(C.ATTACK_SESSIONS))
        for s in sessions:
            from_replay(s, dest, use_s3)
    else:
        from_kafka(dest, use_s3, args.seconds)


if __name__ == "__main__":
    main()
