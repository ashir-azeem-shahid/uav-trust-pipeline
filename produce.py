"""
produce.py
==========
Replay -> Kafka. The ingestion end of the pipeline.

Each CSV row becomes two messages, keyed by the drone each one is about
(see replay.py for why). Key choice is the whole reason the detection
logic works, so it is asserted here rather than assumed.

Usage
-----
  # no broker needed -- prints what it would send, useful first check
  python produce.py CriticalNode --dry-run --limit 6

  # real run, replays as fast as possible
  python produce.py CriticalNode

  # real run at 10x wall-clock speed (1 simulated second = 0.1s)
  python produce.py CriticalNode --speed 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import config as C
import replay as R


def make_producer():
    """Imported lazily so --dry-run works without confluent-kafka installed."""
    try:
        from confluent_kafka import Producer
    except ImportError:
        sys.exit("confluent-kafka not installed.\n"
                 "  pip install -r requirements.txt\n"
                 "Or use --dry-run to test without a broker.")
    return Producer({
        "bootstrap.servers": C.BOOTSTRAP,
        "linger.ms": 5,
        "compression.type": "lz4",
        # keep ordering guarantees per key
        "enable.idempotence": True,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("attack", choices=list(C.ATTACK_SESSIONS),
                    help="which attack session to replay after the baseline")
    ap.add_argument("--dry-run", action="store_true",
                    help="print messages instead of sending; no broker needed")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N messages (0 = all)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="replay speed multiplier; 0 = as fast as possible, "
                         "1 = real time, 10 = ten times real time")
    args = ap.parse_args()

    summary = R.summarise(args.attack)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    print(f"\nproducing to {C.TOPIC_TELEMETRY} at {C.BOOTSTRAP}"
          f"{'  [DRY RUN]' if args.dry_run else ''}\n", file=sys.stderr)

    producer = None if args.dry_run else make_producer()

    sent = 0
    t0 = time.time()
    sim_start = None

    for key, msg, _onset in R.replay_with_onset(args.attack):
        # The partition key must be the drone the message is about. If this
        # ever drifts to requester/contacted, the suppression rule silently
        # stops firing -- so fail loudly instead.
        assert key == msg["about"], "partition key must equal msg['about']"

        if args.speed > 0:
            if sim_start is None:
                sim_start = msg["t_session"]
            target = (msg["t_session"] - sim_start) / args.speed
            behind = target - (time.time() - t0)
            if behind > 0:
                time.sleep(behind)

        if args.dry_run:
            print(f"key={key:<10} role={msg['role']:<9} t={msg['t_session']:>4} "
                  f"{msg['requester']}->{msg['contacted']} "
                  f"[{msg['exchange_type']}]")
        else:
            producer.produce(
                C.TOPIC_TELEMETRY,
                key=key.encode(),
                value=json.dumps(msg).encode(),
            )
            if sent % 1000 == 0:
                producer.poll(0)

        sent += 1
        if args.limit and sent >= args.limit:
            break

    if producer is not None:
        producer.flush(30)

    elapsed = time.time() - t0
    print(f"\nsent {sent:,} messages in {elapsed:.1f}s "
          f"({sent / max(elapsed, 1e-9):,.0f} msg/s)", file=sys.stderr)


if __name__ == "__main__":
    main()
