# uav-trust-pipeline

Cloud-native streaming pipeline for data quality and trust classification of
autonomous UAV telemetry: data quality checks, a deterministic attack rule
and two machine learning models run on every drone as its telemetry
arrives, and a responder acts on alerts by itself while an operator can
override.

```
produce.py ─► Kafka uav.telemetry (4 partitions, keyed by drone)
                 │
                 ▼
           detect.py --ml   data quality ─► suppression rule ─► ML per 15 s window
                 │                                   (Isolation Forest + Random Forest)
                 ▼
           Kafka uav.verdict ─► respond.py ─► Kafka uav.action   (auto-isolation)
                 │                   ▲
                 ├─► app.py ─────────┴── Kafka uav.operator      (human override)
                 └─► sink.py ─► S3 Parquet
```

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
| `detect.py` | The consumer: data quality → suppression rule → ML (`--ml`) → verdicts to `uav.verdict`. |
| `ml.py` | Track B live: per-drone 15 s windows built inside the pipeline, Isolation Forest + Random Forest scoring, explained alerts. One feature code path for training and serving. |
| `train_ml.py` | Trains and evaluates the live models (leave-one-attack-out, held-out clean recording), writes `models/ml.joblib`. |
| `respond.py` | Autonomous responder: isolates a drone on a rule alert after a grace period unless an operator holds it; classifier alerts wait for an operator. Writes `uav.action`. |
| `faults.py` | Benign fault definitions (noise, out-of-range, corrupt trust, dropout, duplicate, position jump). |
| `features.py` | 15-second windows and features for the models; reproduces the bachelor thesis's windowing for comparison. |
| `train.py` | Track B: supervised Random Forest / SVM with feature-group ablation. |
| `sink.py` | Storage: Parquet, locally or to S3, partitioned by session. |
| `feed.py`, `app.py` | Operator dashboard (Streamlit): live from Kafka or from a recorded session. |
| `evaluate.py` | Detection delay, per-stage processing latency, compute-only throughput. |
| `tools/scale_test.py` | Scalability and latency through Kafka: consumers, swarm size, paced latency. |
| `tools/anomaly.py` | Isolation Forest on the bachelor thesis's observer windows (comparison). |
| `tools/train_scale.py` | Training scalability: the ML retrained on 1×–30× the data. |
| `tools/fault_inject.py` | How the data quality layer handles each benign fault. |
| `tools/live_demo.sh` | The whole pipeline live: broker, detector with ML, responder, dashboard, producer. |
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

Starts the broker, the detector (rule + DQ + ML), the responder and the
dashboard (port 8501), trains the models if `models/ml.joblib` is missing,
resets the topics, then replays the session. In the dashboard choose *Live
stream (Kafka)* → *New messages only* → *Connect* before starting the replay.
With nobody touching anything, the attacked drone is isolated about 10 s
after the rule's alert; `GRACE=20` lengthens the override window.

---

## Experiments

```bash
bash tools/run_experiments.sh                               # tests, faults, anomaly, ML
python tools/train_scale.py                                 # training at 1x-30x data
python tools/scale_test.py swarm --copies 1 8 32 --ml       # throughput with ML on
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

### Machine learning in the stream (Track B)

`python train_ml.py`. Per drone, per 15 s window, 47 features, built by the
live pipeline itself. Every number is for data the model was not trained on.

| Attack | Isolation Forest AUC | attacked drone alerted | Random Forest AUC (attack type unseen) | attacked drone alerted |
|---|---|---|---|---|
| Critical Node | 0.94 | yes, 25 s | 1.00 | yes, 190 s |
| MITM | 0.91 | yes, 85 s | 0.78 | no |
| Data manipulation | 0.89 | yes, 25 s | 0.91 | yes, 10 s |
| Data manipulation 2 | 0.97 | yes, 25 s | 1.00 | yes, 10 s |
| Sybil | 1.00 | 8 of 8, 25 s | 1.00 | 8 of 8, 10 s |

False alarms on a second clean recording: 0 windows, 0 drones (both
models). The anomaly detector also puts 0–6 honest drones per attack session
on the watch list — drones disturbed by the attack — which is why it never
triggers an action on its own. The deployed classifier is trained without
Critical Node, so the live demo shows it on a recording it never saw.

### Training scalability

`python tools/train_scale.py` (development machine, 2 CPUs). The swarm is
cloned as in the swarm experiment; accuracy is measured on the same
original-size test data at every size.

| Data | Messages | Windows | Feature extraction | RF fit | Peak memory | RF AUC (unseen attack) |
|---|---|---|---|---|---|---|
| 1× | 83,906 | 3,146 | 5.2 s | 0.5 s | 182 MB | 0.93 |
| 3× | 251,718 | 9,438 | 9.4 s | 0.7 s | 199 MB | 0.93 |
| 10× | 839,060 | 31,460 | 21 s | 1.3 s | 238 MB | 0.90 |
| 30× | 2,517,180 | 94,380 | 54 s | 2.5 s | 397 MB | 0.90 |

Cost grows at most linearly; accuracy is stable. Clones add no new
information, so this shows the training path scales, not that more data
improves the models. The Isolation Forest's fit time is nearly constant by
design: each tree is built from a fixed 256-sample subsample.

### Cost of the ML stage in the stream

`python tools/scale_test.py swarm --offline --copies 1 8 32 [--ml]`, one
consumer, development machine (2 CPUs):

| Swarm | Rule + DQ (msg/s) | + ML (msg/s) |
|---|---|---|
| 54 drones | 58,000 | 8,500 |
| 432 drones | 60,000 | 25,000 |
| 1,728 drones | 61,000 | 32,000 |

The ML stage scores every drone's window in one batch per 15 s boundary, so
its cost per message falls as the swarm grows. The recorded swarm produces
about 25 messages per second in real time, so even the smallest case runs
with over 300× headroom. Suppression verdicts equal the number of victim
clones in every run.
