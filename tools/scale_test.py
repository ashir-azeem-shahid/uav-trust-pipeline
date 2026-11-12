"""
tools/scale_test.py
===================
The scalability and latency experiments. Three sub-commands:

  latency     How long does one message take from being sent to being
              processed, and does that hold up as load rises? Producer and
              consumer run at the same time; the producer sends at a fixed,
              paced rate; each message is timed send -> processed.


  consumers   Does total throughput rise when consumers are added?
              Fill one topic with a large swarm, then drain it with 1, 2, 4 ...
              consumer processes working in parallel. This is the
              horizontal-scaling claim the architecture makes.

  swarm       Does a single consumer slow down as the swarm grows?
              Same pipeline, one consumer, swarm grown 1x, 2x, 4x ...
              If per-message cost stays flat, capacity is a function of how
              many consumers you run, not of how big the swarm is.

Together they answer "is it scalable?" in two parts: the per-message cost
does not grow with the swarm, and the total capacity grows with consumers.

HOW THE SWARM IS GROWN
----------------------
The recorded session is 12,886 messages from 53 drones. One consumer drains
that in under a second -- far too short to time reliably. So the session is
cloned under fresh drone names (Drone18 -> Drone18~c3), and the clones are
interleaved message by message so they fly *at the same time*, the way one
bigger swarm would. Each clone behaves exactly like the original drone, so
the suppression rule must fire exactly once per clone of the victim. That
gives a built-in correctness check: with --copies 32 you must get exactly
32 verdicts. If partitioning ever split a drone's evidence across consumers,
that number would drop -- so every run also re-proves the keying design.

WHY STATIC PARTITION ASSIGNMENT
-------------------------------
Each consumer is handed its partitions explicitly (assign(), not subscribe()).
A consumer group would converge on the same split, but only after a few
seconds of join/rebalance negotiation, and on short runs that noise would
swamp the measurement. State this in the thesis: the measurement isolates
processing capacity from group-coordination overhead.

WHERE THE CLOCK STARTS AND STOPS
--------------------------------
All consumers wait at a barrier, then start the clock together. Each stops
its clock when it processes its last message -- not when its final consume()
call returns. consume() waits up to its timeout to fill a batch, so timing to
the return would add that wait to every run as a fixed cost (an earlier
version of this script did, adding ~1 s per run). Wall time for a run is
latest stop minus earliest start, so it is set by the slowest consumer.

WHY A WARM-UP RUN
-----------------
The first read of a freshly written topic can come from disk; later reads
come from the page cache. One discarded warm-up run makes every measured
repetition start from the same state.

Usage
-----
  # the experiment (needs the broker: docker compose up -d)
  python tools/scale_test.py consumers --copies 32 --consumers 1 2 4 --reps 3
  python tools/scale_test.py swarm --copies 1 2 4 8 16 32 --reps 3

  # same experiments with no broker -- partitions simulated in memory.
  # Measures compute scaling only; useful as a sanity check.
  python tools/scale_test.py consumers --copies 8 --consumers 1 2 --offline
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import math
import statistics
import struct
import sys
import time
import uuid
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config.py paths (bsc-reference/...) are relative to the repo

import config as C  # noqa: E402


# ---------------------------------------------------------------------
# building the swarm
# ---------------------------------------------------------------------
def load_base(attack: str) -> list[tuple[str, dict]]:
    """The recorded baseline + attack session, as (key, message) pairs."""
    import replay as R
    return [(k, m) for k, m, _ in R.replay_with_onset(attack)]


def _rename(name, c: int):
    if c == 0 or not name:
        return name
    return f"{name}~c{c}"


def clone(key: str, msg: dict, c: int) -> tuple[str, dict]:
    """Copy c of one message, with every drone name moved to a new identity."""
    if c == 0:
        return key, msg
    m = dict(msg)
    for f in ("requester", "contacted", "about"):
        m[f] = _rename(msg[f], c)
    gt = dict(msg["_gt"])
    gt["attack_target"] = _rename(gt.get("attack_target"), c)
    m["_gt"] = gt
    return _rename(key, c), m


def swarm_stream(base, copies: int):
    """
    All copies interleaved message by message: copy 0..N-1 of message 1,
    then copy 0..N-1 of message 2, and so on. The clones fly simultaneously,
    and each drone's own messages stay in order.
    """
    for key, msg in base:
        for c in range(copies):
            yield clone(key, msg, c)


def drone_count(base) -> int:
    names = set()
    for _k, m in base:
        names.add(m["requester"])
        names.add(m["contacted"])
    return len(names)


def offline_partition(key: str, partitions: int) -> int:
    """Stand-in for Kafka's partitioner: any pure function of the key works."""
    return zlib.crc32(key.encode()) % partitions


# ---------------------------------------------------------------------
# Kafka: filling a topic
# ---------------------------------------------------------------------
def create_topic(partitions: int) -> str:
    from confluent_kafka.admin import AdminClient, NewTopic

    topic = f"uav.scale.{int(time.time())}.{uuid.uuid4().hex[:4]}"
    admin = AdminClient({"bootstrap.servers": C.BOOTSTRAP})
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=partitions, replication_factor=1)])
    for f in futures.values():
        f.result(30)
    time.sleep(1.0)  # let metadata propagate before producing
    return topic


def delete_topics(topics: list[str]) -> None:
    if not topics:
        return
    try:
        from confluent_kafka.admin import AdminClient
        admin = AdminClient({"bootstrap.servers": C.BOOTSTRAP})
        for f in admin.delete_topics(topics).values():
            try:
                f.result(30)
            except Exception:
                pass
    except Exception as e:  # cleanup must never fail the experiment
        print(f"  (topic cleanup skipped: {e})", file=sys.stderr)


def fill_topic(topic: str, base, copies: int) -> dict:
    from confluent_kafka import Producer

    failures = []

    def on_delivery(err, _msg):
        if err is not None:
            failures.append(str(err))

    p = Producer({
        "bootstrap.servers": C.BOOTSTRAP,
        "linger.ms": 20,
        "batch.num.messages": 10000,
        "compression.type": "lz4",
        "queue.buffering.max.messages": 2_000_000,
        "queue.buffering.max.kbytes": 1_048_576,
    })

    n = 0
    t0 = time.monotonic()
    for key, msg in swarm_stream(base, copies):
        value = json.dumps(msg).encode()
        while True:
            try:
                p.produce(topic, key=key.encode(), value=value,
                          on_delivery=on_delivery)
                break
            except BufferError:      # local queue full: let it drain
                p.poll(0.1)
        n += 1
        if n % 5000 == 0:
            p.poll(0)
    left = p.flush(180)
    secs = time.monotonic() - t0

    if left or failures:
        raise RuntimeError(f"fill incomplete: {left} unsent, "
                           f"{len(failures)} failed (first: {failures[:1]})")
    return {"messages": n, "seconds": round(secs, 3),
            "msg_per_s": round(n / secs, 1)}


# ---------------------------------------------------------------------
# consumer workers -- one process each
# ---------------------------------------------------------------------
def _result(idx, parts, n, verdicts, t0, t1, pipe) -> dict:
    s = pipe.stats()
    return {
        "worker": idx,
        "partitions": parts,
        "messages": n,
        "verdicts": verdicts,
        "t0": t0,
        "t1": t1,
        "seconds": round(t1 - t0, 4),
        "pipeline_ms_p50": round(s["per_msg_ms_p50"], 5),
        "pipeline_ms_p99": round(s["per_msg_ms_p99"], 5),
        "quarantined": s["quarantined"],
    }


def _drain_kafka(topic, parts, barrier, q, idx):
    try:
        from confluent_kafka import Consumer, TopicPartition, OFFSET_BEGINNING
        from detect import Pipeline

        consumer = Consumer({
            "bootstrap.servers": C.BOOTSTRAP,
            "group.id": f"scale-{uuid.uuid4().hex[:8]}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "fetch.max.bytes": 52_428_800,
            "max.partition.fetch.bytes": 10_485_760,
        })

        # How far each partition goes. A fresh topic starts at offset 0.
        high = {}
        for p in parts:
            _lo, hi = consumer.get_watermark_offsets(TopicPartition(topic, p),
                                                     timeout=15)
            high[p] = hi
        consumer.assign([TopicPartition(topic, p, OFFSET_BEGINNING) for p in parts])

        pipe = Pipeline()
        pending = {p for p in parts if high[p] > 0}
        n = verdicts = 0

        barrier.wait(timeout=180)
        t0 = time.monotonic()
        last = t0
        t_done = t0
        while pending:
            # consume() returns when it has num_messages OR the timeout passes.
            # On the final, partial batch it therefore sits out the whole
            # timeout -- so the clock stops at the last message processed
            # (t_done), never at the moment consume() gives up waiting.
            batch = consumer.consume(num_messages=2000, timeout=0.1)
            now = time.monotonic()
            if not batch:
                if now - last > 30:
                    raise RuntimeError(f"stalled with partitions {sorted(pending)} "
                                       f"unfinished after {n} messages")
                continue
            last = now
            for m in batch:
                if m.error():
                    continue
                v = pipe.handle(json.loads(m.value()))
                n += 1
                if v and v.get("verdict") == "SUPPRESSION":
                    verdicts += 1
                if m.offset() + 1 >= high[m.partition()]:
                    pending.discard(m.partition())
            t_done = time.monotonic()
        t1 = t_done
        consumer.close()
        q.put(_result(idx, parts, n, verdicts, t0, t1, pipe))
    except Exception as e:
        q.put({"worker": idx, "error": f"{type(e).__name__}: {e}"})


def _drain_offline(attack, copies, partitions, parts, barrier, q, idx):
    """Same work as _drain_kafka, minus the broker. Messages are pre-encoded
    to JSON so the decode cost a real consumer pays is still paid here."""
    try:
        from detect import Pipeline

        base = load_base(attack)
        wanted = set(parts)
        raw = [json.dumps(m).encode()
               for k, m in swarm_stream(base, copies)
               if offline_partition(k, partitions) in wanted]
        del base

        pipe = Pipeline()
        n = verdicts = 0
        barrier.wait(timeout=600)
        t0 = time.monotonic()
        for value in raw:
            v = pipe.handle(json.loads(value))
            n += 1
            if v and v.get("verdict") == "SUPPRESSION":
                verdicts += 1
        t1 = time.monotonic()
        q.put(_result(idx, parts, n, verdicts, t0, t1, pipe))
    except Exception as e:
        q.put({"worker": idx, "error": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------
# one run = k consumers draining the whole topic in parallel
# ---------------------------------------------------------------------
def run_once(k: int, partitions: int, *, topic=None, offline=False,
             attack="CriticalNode", copies=1) -> dict:
    if k > partitions:
        raise ValueError(f"{k} consumers but only {partitions} partitions -- "
                         f"the extra consumers would sit idle")

    ctx = mp.get_context("spawn")      # never fork a process holding Kafka clients
    barrier = ctx.Barrier(k)
    q = ctx.Queue()
    assignment = [[p for p in range(partitions) if p % k == i] for i in range(k)]

    procs = []
    for i, parts in enumerate(assignment):
        if offline:
            args = (attack, copies, partitions, parts, barrier, q, i)
            target = _drain_offline
        else:
            args = (topic, parts, barrier, q, i)
            target = _drain_kafka
        p = ctx.Process(target=target, args=args, daemon=True)
        p.start()
        procs.append(p)

    results = []
    deadline = time.monotonic() + 1800
    while len(results) < k:
        try:
            results.append(q.get(timeout=5))
        except Exception:
            dead = [p for p in procs if p.exitcode not in (None, 0)]
            if dead or time.monotonic() > deadline:
                for p in procs:
                    p.terminate()
                raise RuntimeError("a consumer process died or timed out")
    for p in procs:
        p.join(timeout=30)

    errors = [r for r in results if "error" in r]
    if errors:
        raise RuntimeError(f"consumer failed: {errors[0]['error']}")

    results.sort(key=lambda r: r["worker"])
    total = sum(r["messages"] for r in results)
    wall = max(r["t1"] for r in results) - min(r["t0"] for r in results)
    for r in results:
        r.pop("t0")
        r.pop("t1")
    return {
        "consumers": k,
        "messages": total,
        "verdicts": sum(r["verdicts"] for r in results),
        "wall_seconds": round(wall, 4),
        "msg_per_s": round(total / wall, 1) if wall > 0 else None,
        "workers": results,
    }


# ---------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------
def _summ(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 1),
        "std": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "n": len(values),
    }


def environment(args) -> dict:
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count()
    return {
        "mode": "offline (no broker)" if args.offline else f"kafka @ {C.BOOTSTRAP}",
        "cpus_usable": usable,
        "machine": read("/sys/devices/virtual/dmi/id/product_name"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "note": ("partitions statically assigned (assign(), not subscribe()) so "
                 "group-rebalance time is excluded; one warm-up run discarded "
                 "per configuration"),
    }


def _check_verdicts(run: dict, copies: int, attack: str) -> str:
    if attack != "CriticalNode":
        return ""
    return "ok" if run["verdicts"] == copies else f"EXPECTED {copies}"


def cmd_consumers(args) -> dict:
    base = None if args.offline else load_base(args.attack)
    drones = drone_count(base if base is not None else load_base(args.attack))
    cpus = environment(args)["cpus_usable"]
    if max(args.consumers) > cpus:
        print(f"  NOTE: up to {max(args.consumers)} consumers on {cpus} usable CPUs. "
              f"Speedup will flatten once consumers outnumber cores -- that is the "
              f"machine's limit, not the pipeline's.\n")

    topics = []
    out = {"experiment": "consumers", "attack": args.attack,
           "copies": args.copies, "drones": drones * args.copies,
           "partitions": args.partitions, "runs": [], "summary": []}
    try:
        topic = None
        if not args.offline:
            topic = create_topic(args.partitions)
            topics.append(topic)
            print(f"  filling {topic} with {args.copies} copies "
                  f"({drones * args.copies:,} drones) ...")
            out["fill"] = fill_topic(topic, base, args.copies)
            print(f"  filled: {out['fill']['messages']:,} messages in "
                  f"{out['fill']['seconds']}s\n")

        for k in args.consumers:
            for w in range(args.warmup):
                run_once(k, args.partitions, topic=topic, offline=args.offline,
                         attack=args.attack, copies=args.copies)
            rates = []
            for rep in range(args.reps):
                r = run_once(k, args.partitions, topic=topic, offline=args.offline,
                             attack=args.attack, copies=args.copies)
                r["rep"] = rep + 1
                r["verdict_check"] = _check_verdicts(r, args.copies, args.attack)
                out["runs"].append(r)
                rates.append(r["msg_per_s"])
                print(f"  consumers={k}  rep {rep + 1}/{args.reps}  "
                      f"{r['messages']:,} msgs in {r['wall_seconds']:.2f}s  "
                      f"= {r['msg_per_s']:>10,.0f} msg/s   "
                      f"verdicts {r['verdicts']} {r['verdict_check']}")
            out["summary"].append({"consumers": k, "msg_per_s": _summ(rates)})
    finally:
        if not args.keep:
            delete_topics(topics)

    one = next((s for s in out["summary"] if s["consumers"] == 1), None)
    print("\n  consumers   msg/s (mean ± std)        speedup   efficiency")
    for s in out["summary"]:
        m = s["msg_per_s"]
        if one:
            s["speedup"] = round(m["mean"] / one["msg_per_s"]["mean"], 2)
            s["efficiency"] = round(s["speedup"] / s["consumers"], 2)
        sp = f"{s.get('speedup', float('nan')):>7.2f}x"
        ef = f"{s.get('efficiency', float('nan')):>9.0%}"
        print(f"  {s['consumers']:>9}   {m['mean']:>10,.0f} ± {m['std']:<9,.0f}  "
              f"{sp}  {ef}")
    return out


def cmd_swarm(args) -> dict:
    base = load_base(args.attack)
    drones = drone_count(base)
    if args.offline:
        base = None

    topics = []
    out = {"experiment": "swarm", "attack": args.attack, "consumers": 1,
           "partitions": args.partitions, "runs": [], "summary": []}
    try:
        for copies in args.copies:
            topic = None
            fill = None
            if not args.offline:
                topic = create_topic(args.partitions)
                topics.append(topic)
                fill = fill_topic(topic, base, copies)
            for _ in range(args.warmup):
                run_once(1, args.partitions, topic=topic, offline=args.offline,
                         attack=args.attack, copies=copies)
            rates, p99s = [], []
            for rep in range(args.reps):
                r = run_once(1, args.partitions, topic=topic, offline=args.offline,
                             attack=args.attack, copies=copies)
                r.update(rep=rep + 1, copies=copies, drones=drones * copies,
                         verdict_check=_check_verdicts(r, copies, args.attack))
                out["runs"].append(r)
                rates.append(r["msg_per_s"])
                p99s.append(r["workers"][0]["pipeline_ms_p99"])
            s = {"copies": copies, "drones": drones * copies,
                 "messages": out["runs"][-1]["messages"],
                 "msg_per_s": _summ(rates),
                 "pipeline_ms_p99_mean": round(statistics.mean(p99s), 5)}
            if fill:
                s["fill"] = fill
            out["summary"].append(s)
            print(f"  drones={drones * copies:>6,}  "
                  f"{s['messages']:>9,} msgs   "
                  f"{s['msg_per_s']['mean']:>10,.0f} ± {s['msg_per_s']['std']:<8,.0f} msg/s   "
                  f"p99 {s['pipeline_ms_p99_mean']:.4f} ms   "
                  f"verdicts {out['runs'][-1]['verdicts']} "
                  f"{out['runs'][-1]['verdict_check']}")
            if not args.keep and topic:
                delete_topics([topic])
                topics.remove(topic)
    finally:
        if not args.keep:
            delete_topics(topics)
    return out


# ---------------------------------------------------------------------
# latency under paced load
# ---------------------------------------------------------------------
# Why this exists: timing messages that were all produced *before* the
# consumer started measures how long the producer took, not latency -- the
# first message waits for the whole batch. Here the consumer is already
# listening, the producer sends at a steady rate, and each message carries
# its send time in a Kafka header, so the gap is transit + queueing +
# processing and nothing else. Both ends run on the same machine, so they
# read the same clock.
#
# Latency-oriented client settings, stated so they can be reported:
#   producer linger.ms = 1   (default batching would add up to 5 ms)
#   consumer poll() per message, fetch.wait.max.ms = 10
# The first second of each run is discarded as warm-up.

def _pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(int(len(sorted_vals) * p), len(sorted_vals) - 1)]


def _latency_consumer(topic, partitions, expected, warmup_n, ready, q):
    try:
        from confluent_kafka import Consumer, TopicPartition, OFFSET_BEGINNING
        from detect import Pipeline

        consumer = Consumer({
            "bootstrap.servers": C.BOOTSTRAP,
            "group.id": f"lat-{uuid.uuid4().hex[:8]}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "fetch.wait.max.ms": 10,
        })
        consumer.assign([TopicPartition(topic, p, OFFSET_BEGINNING)
                         for p in range(partitions)])
        consumer.poll(0.5)            # open fetch sessions while the topic is empty
        pipe = Pipeline()
        ready.set()

        lat = []
        n = 0
        last = time.monotonic()
        while n < expected:
            m = consumer.poll(0.2)
            now = time.monotonic()
            if m is None:
                if now - last > 20:
                    break
                continue
            if m.error():
                continue
            last = now
            pipe.handle(json.loads(m.value()))
            done = time.time()
            n += 1
            if n > warmup_n:
                sent = struct.unpack("d", dict(m.headers() or [])["tp"])[0]
                lat.append((done - sent) * 1000.0)
        consumer.close()

        lat.sort()
        q.put({
            "consumed": n,
            "measured": len(lat),
            "ms": {"p50": round(_pct(lat, .50), 3), "p95": round(_pct(lat, .95), 3),
                   "p99": round(_pct(lat, .99), 3),
                   "max": round(lat[-1], 3) if lat else None},
        })
    except Exception as e:
        q.put({"error": f"{type(e).__name__}: {e}"})


def _offer(topic, values, rate, seconds) -> dict:
    """Send len(values) messages at `rate` msg/s, stamping each send time."""
    from confluent_kafka import Producer

    p = Producer({"bootstrap.servers": C.BOOTSTRAP, "linger.ms": 1,
                  "compression.type": "lz4",
                  "queue.buffering.max.messages": 2_000_000})
    total = len(values)
    t0 = time.monotonic()
    i = 0
    while i < total:
        due = min(total, int((time.monotonic() - t0) * rate) + 1)
        while i < due:
            key, val = values[i]
            while True:
                try:
                    p.produce(topic, key=key, value=val,
                              headers=[("tp", struct.pack("d", time.time()))])
                    break
                except BufferError:
                    p.poll(0.01)
            i += 1
        p.poll(0)
        ahead = i / rate - (time.monotonic() - t0)
        if ahead > 0.0005:
            time.sleep(min(ahead, 0.005))
    send_secs = time.monotonic() - t0
    left = p.flush(60)
    if left:
        raise RuntimeError(f"{left} messages not delivered")
    return {"sent": total, "send_seconds": round(send_secs, 3),
            "achieved_msg_per_s": round(total / send_secs, 1)}


def cmd_latency(args) -> dict:
    if args.offline:
        raise SystemExit("latency needs a real broker -- it measures Kafka transit")
    base = load_base(args.attack)
    drones = drone_count(base)
    out = {"experiment": "latency", "attack": args.attack, "consumers": 1,
           "partitions": args.partitions, "seconds_per_rate": args.seconds,
           "warmup_seconds": 1, "client_settings": {
               "producer_linger_ms": 1, "consumer": "poll() per message",
               "consumer_fetch_wait_max_ms": 10},
           "runs": []}
    topics = []
    print("  offered     achieved     drones    p50 ms    p95 ms    p99 ms    max ms")
    try:
        for rate in args.rates:
            total = int(rate * args.seconds)
            copies = max(1, math.ceil(total / len(base)))
            values = []
            for key, msg in swarm_stream(base, copies):
                values.append((key.encode(), json.dumps(msg).encode()))
                if len(values) >= total:
                    break

            topic = create_topic(args.partitions)
            topics.append(topic)
            ctx = mp.get_context("spawn")
            ready, q = ctx.Event(), ctx.Queue()
            proc = ctx.Process(target=_latency_consumer, daemon=True,
                               args=(topic, args.partitions, total, int(rate), ready, q))
            proc.start()
            if not ready.wait(60):
                proc.terminate()
                raise RuntimeError("latency consumer never became ready")

            sent = _offer(topic, values, rate, args.seconds)
            res = q.get(timeout=args.seconds + 120)
            proc.join(timeout=30)
            if "error" in res:
                raise RuntimeError(f"consumer failed: {res['error']}")

            run = {"offered_msg_per_s": rate, "drones": drones * copies, **sent, **res}
            out["runs"].append(run)
            ms = res["ms"]
            print(f"  {rate:>7,}   {sent['achieved_msg_per_s']:>10,.0f}   {drones * copies:>7,}"
                  f"   {ms['p50']:>7.2f}   {ms['p95']:>7.2f}   {ms['p99']:>7.2f}"
                  f"   {ms['max']:>7.2f}"
                  + ("" if res["consumed"] == total else
                     f"   (consumed {res['consumed']:,}/{total:,})"))
            if not args.keep:
                delete_topics([topic])
                topics.remove(topic)
    finally:
        if not args.keep:
            delete_topics(topics)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--attack", default="CriticalNode", choices=list(C.ATTACK_SESSIONS))
        p.add_argument("--partitions", type=int, default=C.N_PARTITIONS)
        p.add_argument("--reps", type=int, default=3)
        p.add_argument("--warmup", type=int, default=1)
        p.add_argument("--offline", action="store_true",
                       help="no broker: simulate partitions in memory")
        p.add_argument("--keep", action="store_true",
                       help="don't delete the test topics afterwards")
        p.add_argument("--out", default=None, help="results file (JSON)")

    pc = sub.add_parser("consumers", help="throughput vs number of consumers")
    pc.add_argument("--copies", type=int, default=32)
    pc.add_argument("--consumers", type=int, nargs="+", default=[1, 2, 4])
    common(pc)

    ps = sub.add_parser("swarm", help="throughput vs swarm size, one consumer")
    ps.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    common(ps)

    pl = sub.add_parser("latency", help="send-to-processed latency at paced load rates")
    pl.add_argument("--rates", type=int, nargs="+", default=[1000, 5000, 20000, 50000],
                    help="offered load, messages per second")
    pl.add_argument("--seconds", type=int, default=10, help="duration per rate")
    common(pl)

    args = ap.parse_args()
    env = environment(args)
    print(f"\n  {args.cmd} experiment · {env['mode']} · "
          f"{env['cpus_usable']} usable CPUs · {env['machine'] or platform.node()}\n")

    t0 = time.monotonic()
    result = {"consumers": cmd_consumers, "swarm": cmd_swarm,
              "latency": cmd_latency}[args.cmd](args)
    result["environment"] = env
    result["total_seconds"] = round(time.monotonic() - t0, 1)

    out = args.out or f"results_scaling_{args.cmd}{'_offline' if args.offline else ''}.json"
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"\n  saved {out}  ({result['total_seconds']}s total)\n")


if __name__ == "__main__":
    main()
