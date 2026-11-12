# uav-trust-pipeline

Cloud-native streaming pipeline for data quality and trust classification of
autonomous UAV telemetry.

Master's thesis, Masaryk University FI. Builds on Matej Fečák's bachelor
thesis (*A Digital Twin Framework for Trust Evaluation of Autonomous
Drones*), consuming its recorded session logs and replacing post-flight
batch analysis with in-flight stream processing.

MIT licensed.

---

## Quick start

```bash
# 1. reference data (gitignored -- ~5 MB of session CSVs)
git clone https://github.com/MattFechack/Digital-Twin-framework-for-trust-evaluation-of-autonomous-drones bsc-reference

# 2. dependencies
pip install -r requirements.txt

# 3. does the data look right?
python tools/analysis3.py

# 4. does the detection logic work? (no broker needed)
python detect.py --offline CriticalNode

# 5. tests -- the leakage suite is the important one
python -m pytest tests/ -v

# 6. the broker
docker compose up -d
docker compose ps                     # redpanda healthy, init-topics exited 0

# 7. end to end: consumer in one terminal, producer in another
python detect.py
python produce.py CriticalNode
```

Redpanda Console on <http://localhost:8080> shows the messages.

---

## What's here

| File | Purpose |
|---|---|
| `config.py` | Every column name, path and constant. Nothing else hard-codes a column name. |
| `replay.py` | CSVs → ordered message stream. Handles onset construction and role re-keying. |
| `produce.py` | Stream → Kafka, keyed by the drone under evaluation. `--speed` paces it, `--fault` injects benign faults, `--dry-run` needs no broker. |
| `dq.py` | Stage 1: data quality. Five dimensions per record, thresholds fitted on the clean baseline, quarantine for unusable records. |
| `rules.py` | Track A: the suppression rule, as streaming state. |
| `detect.py` | The consumer: data quality → suppression rule → verdicts to `uav.verdict`. |
| `faults.py` | Benign fault definitions (noise, out-of-range, corrupt trust, dropout, duplicate, position jump). |
| `features.py` | 15-second windows and features for the models; reproduces the bachelor thesis's windowing for comparison. |
| `train.py` | Track B: supervised Random Forest / SVM with feature-group ablation. |
| `sink.py` | Storage: Parquet, locally or to S3, partitioned by session. |
| `feed.py`, `app.py` | Operator dashboard (Streamlit): live from Kafka or from a recorded session. |
| `evaluate.py` | Detection delay, per-stage processing latency, compute-only throughput. |
| `tools/scale_test.py` | Scalability and latency through Kafka: consumers, swarm size, paced latency. |
| `tools/anomaly.py` | Unsupervised anomaly detection: Isolation Forest trained on clean traffic only. |
| `tools/fault_inject.py` | How the data quality layer handles each benign fault. |
| `tools/live_demo.sh` | The whole pipeline live: broker, detector, dashboard, producer. |
| `tools/run_experiments.sh` | The broker-free experiments in one go. |
| `tools/analysis3.py` | Data diagnostics — the go/no-go test on the raw CSVs. |
| `tools/simulate.py` | Replays through simulated partitioned consumers. Proves the architecture before Kafka. |
| `tools/sweep_streak.py` | Threshold sensitivity sweep + the max-streak separation table. |
| `tests/test_leakage.py` | Asserts ground truth never reaches the operational path. |

---

## Live demo

```bash
bash tools/live_demo.sh                  # clean Critical Node session at 10x
bash tools/live_demo.sh trust_corrupt    # with a benign fault on 5% of records
```

Starts the broker, the detector and the dashboard (port 8501), resets the
topics, then replays the session. In the dashboard choose *Live stream
(Kafka)* → *New messages only* → *Connect* before starting the replay.

---

## Experiments

```bash
bash tools/run_experiments.sh                               # tests, faults, anomaly
python tools/scale_test.py consumers --copies 64 --consumers 1 2 4 --reps 3
python tools/scale_test.py swarm --copies 1 2 4 8 16 32 64 --reps 3
python tools/scale_test.py latency
python evaluate.py
python train.py --window-mode session
```

The scaling and latency runs need the broker (`docker compose up -d`). On a
machine with more cores, give the broker more room so it is never the
bottleneck: `REDPANDA_SMP=2 REDPANDA_MEMORY=4G docker compose up -d`.
Redpanda refuses to start with *fewer* cores than its data was created with,
so before going back to the defaults run `docker compose down` first (this
deletes the stored messages, which the experiments do not need).

---

## Three design decisions worth knowing

**Each CSV row becomes two messages.** The suppression rule needs, for one
drone, both its timeouts as *target* and its successes as *initiator*. No
single partition key delivers both to one consumer — keying by
`ContactedDrone` scatters the proof-of-life evidence, keying by `Requester`
scatters the timeout evidence. So every row is emitted twice, each copy
keyed by the drone it is *about*. Both copies for a drone then hash to the
same partition. `tests/test_leakage.py` asserts this property holds.

**The replay concatenates a baseline session before the attack session.**
Every recorded attack session has the attack already running in its first
second, so there is no onset and detection delay cannot be measured. The
replay plays the whole no-attack baseline first; the join is a known, exact
onset. Everything before it is genuine clean traffic and doubles as a
false-positive test.

**The rule counts a streak, not a rate.** The obvious formulation — share of
target exchanges in the last 15 seconds that timed out — does not work. The
baseline timeout rate is around 50%, so "3 of 3 timed out" happens by
chance, and over a session every drone eventually produces one all-timeout
window (measured: 46–52 false positives per session). What separates the
victim is persistence: it never answers again. So the statistic is
consecutive unanswered requests, reset by any successful response — one
integer of state per drone, and something a per-row batch analysis has
nowhere to put.

---

## Measured results

### Detection (Track A)

Critical Node session, replayed through simulated partitioned consumers
with only per-drone local state:

| | |
|---|---|
| Victim | `Drone18` (ground truth) |
| Detected | yes, 19 s after onset |
| False positives | 0, across all 5 attack sessions and both baselines |
| Victim's max streak | 302 consecutive unanswered requests |
| Worst honest drone | 69 |
| Clean threshold range | 70 – 300 (a 4× plateau, not a fitted constant) |
| Per-message cost | p99 0.003 ms |

The rule is deliberately Critical-Node-specific; the other four sessions
serve as specificity tests, and it stays silent on all of them. Sybil ghosts
reach streaks of 669 but are excluded by the proof-of-life clause — they
never initiate, because they do not exist.

### Scalability and latency through Kafka

AWS EC2 `c7i.2xlarge` (8 vCPUs, 4 physical cores), Redpanda with 2 cores,
4 partitions. Swarm grown by cloning the recorded session under new drone
names; the rule must fire once per clone of the victim, and did in every run.

| Consumers | Throughput (msg/s, mean of 3) | Speedup |
|---|---|---|
| 1 | 83,467 | 1.00× |
| 2 | 160,244 | 1.92× |
| 4 | 274,324 | 3.29× |

One consumer, swarm grown from 54 to 3,456 drones: per-message processing
p99 stays between 0.015 and 0.017 ms, throughput levels at about 80,000
msg/s once runs are long enough to amortise start-up.

Send-to-verdict latency, producer and consumer running concurrently at a
paced rate, one consumer: median 1.3–1.4 ms and p99 under 3.5 ms up to
20,000 msg/s; at 50,000 msg/s median 1.7 ms, p99 26 ms.
