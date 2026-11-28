# uav-trust-pipeline

Cloud-native streaming pipeline for data quality and trust classification of
autonomous UAV telemetry: data quality checks, then one cheap deterministic
rule per attack type, then two machine learning models for the drones no
rule has caught, run on every drone as its telemetry arrives. Every alert
names the attack type and the stage that caught it, and a responder acts on
alerts by itself while an operator can override.

```
produce.py ─► Kafka uav.telemetry (4 partitions, keyed by drone)
                 │
                 ▼
           detect.py --ml   data quality ─► rules (Sybil, Critical Node, ─► ML per 15 s window,
                 │                          data manipulation, MITM)     only for drones no rule caught
                 │                                                       (Isolation Forest + multiclass RF)
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
| `rules.py` | Track A: one rule per attack type, as streaming state -- Sybil (unregistered identity), Critical Node (suppression streak), data manipulation (physically impossible telemetry), MITM (link failing both ways, slow answers). |
| `detect.py` | The consumer: data quality → rules → ML (`--ml`, only for drones no rule caught) → verdicts to `uav.verdict`, each with `attack_type` and `stage`. |
| `ml.py` | Track B live: per-drone 15 s windows built inside the pipeline, Isolation Forest + multiclass Random Forest (normal or attack type), explained alerts. One feature code path for training and serving. |
| `train_ml.py` | Trains and evaluates the live models (leave-one-attack-out, held-out clean recording) and compares rules only, ML only and the cascade (accuracy, per-class P/R/F1, confusion matrices); writes `models/ml.joblib`. |
| `respond.py` | Autonomous responder: isolates a drone on a rule alert after a grace period unless an operator holds it; classifier alerts wait for an operator. Writes `uav.action`. |
| `inject.py` | Attack injection: MITM, Critical Node, data manipulation and Sybil applied to a clean recording at a chosen strength, mirroring Fečák's simulator. |
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
| `tools/inject_test.py` | Detection and false alarms against attack strength on the held-out flight: rules, forest, cascade, Isolation Forest (`results_injection.json`, `docs/injection/`). |
| `tools/sweep_rules.py` | Threshold sensitivity of the Sybil, data manipulation and MITM rules (`results_rule_sweep.json`). |
| `tests/test_leakage.py` | Asserts ground truth never reaches the operational path. |
| `tests/test_rules.py` | Each rule ignores ground truth, catches its own attack and nothing else; ML is gated by the rules; the classifier names the attack. |

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

### Detection (Track A): the Critical Node rule

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

### Rules per attack type and the cascade

`python train_ml.py` (development machine, pinned versions), `results_ml.json`
→ `cascade`. Every alert now carries the attack type; a drone a rule has
caught is not sent to the models.

Rules on their own, every recording replayed after the clean baseline:

| Attack | Rule | Attacked drone(s) caught | Delay | Honest drones flagged |
|---|---|---|---|---|
| Sybil | identity not in the fleet registry, never answers or sends | 8 of 8 ghosts | 0 s | 0 |
| Critical Node | suppression streak (unchanged) | 1 of 1 | 19 s | 0 |
| Data manipulation | 5 reports in a row that break physics (rotors stopped while airborne, declared speed vs distance moved) | 1 of 1, both recordings | 7 s, 2 s | 0 |
| MITM | ≥75% unanswered in both directions over 100 exchanges, answers ≥1 s slow | 1 of 1 | 33 s | 0 |

No rule fires on the held-out clean recording. Rule cost is negligible
(p99 0.03 ms per message, throughput unchanged).

Comparison, every (drone, 15 s window) labelled normal or attack type.
Known attack types: blocked 5-fold cross-validation over time (each time
block predicted by a forest trained on the other blocks), plus the held-out
clean recording; 1,585 windows, mean of 5 seeds.

| Method | Accuracy | Macro F1 | Attack vs normal F1 | F1 Critical Node | F1 Data manip. | F1 MITM | F1 Sybil |
|---|---|---|---|---|---|---|---|
| Rules only | 0.999 | 0.99 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 |
| Random Forest only | 0.983 | 0.78 | 0.94 | 0.43 | 0.92 | 0.53 | 0.97 |
| Rules, then RF | 0.996 | 0.97 | 0.98 | 1.00 | 1.00 | 0.84 | 1.00 |

Cascade confusion matrix (rows truth, columns predicted):

| | normal | Critical Node | Data manip. | MITM | Sybil |
|---|---|---|---|---|---|
| normal | 1420 | 0 | 0 | 6 | 0 |
| Critical Node | 0 | 6 | 0 | 0 | 0 |
| Data manip. | 0 | 0 | 14 | 0 | 0 |
| MITM | 1 | 0 | 0 | 18 | 0 |
| Sybil | 0 | 0 | 0 | 0 | 120 |

In the cascade the forest scores 1,427 windows; the rules settle 158. The
six false MITM windows come from the forest (two honest drones in the MITM
session), which is why the cascade's MITM F1 is below the rule's.

Attack types the forest never saw (leave one type out, attack vs normal F1):
rules 0.97–1.00 on every type; forest 1.00 on Critical Node and Sybil but
0.00 on MITM and data manipulation; cascade 0.97–1.00. Rules need no
training data, which is where they matter most.

**Read these with care.** There are very few attacked windows (Critical
Node 6, data manipulation 14, MITM 19, Sybil 120), so one window moves a
per-class F1 by 0.1–0.3. The rule thresholds were chosen by looking at these
same recordings; `tools/sweep_rules.py` shows how wide each clean range is.
Data manipulation (any run length from 2 to 20) and Sybil (any request
count) are robust. The MITM rule is the weakest: it is clean only for a
narrow band of settings and there is one MITM recording, so its zero false
positives are optimistic. Benign faults (`tools/fault_inject.py`): at a 5%
corruption rate no rule's output changes; at 20%, 250 m position jumps make
2–3 honest drones look like data manipulation and corrupted trust values
make one look like MITM.

### Injected attacks: where each detector stops working

`python tools/inject_test.py` (≈3 min), `results_injection.json`, plots in
`docs/injection/`. The recorded sessions hold each attack once, at one
strength. `inject.py` re-applies each attack to a clean recording at
strengths from far below to above the recorded one, following how Fečák's
simulator implements it (MITM delay and loss on the victim's frames,
Critical Node reply suppression, data manipulation of speed / position /
rotor fields, Sybil ghosts that may partly answer), including Brno.py's rule
that a timeout is only logged above 50% loss and its trust formulas.

* **Test flight:** the held-out clean recording, never used for training or
  for choosing any threshold. 8 runs per setting, random target (any drone
  asked for data at least 20 times after the onset) and random onset.
* **Rules:** thresholds frozen exactly as committed before this test.
* **Random Forest:** trained on the recorded sessions plus attacks injected
  into the *other* clean flight (different seeds, targets, onsets).
  **Caveat:** training and test attacks come from the same injector, so
  this forest has learned the injector's fingerprint; its curves are an
  upper bound. **RF (recorded)** is the forest trained on the recordings
  only, the honest test of generalisation.

Detected share of attacked drones (honest drones flagged per run in
brackets). With no attack at all, no method flags the randomly chosen
target; the injection-trained forest flags the same 2 honest drones in
every run of this flight.

| Attack, strength | Rules | RF (recorded) | RF | Rules → RF | Isolation Forest |
|---|---|---|---|---|---|
| MITM, loss 0.05 → 0.6 (delay 1.5 s) | 0 → 0 | 0 → 0.12 | 0.50 → 0.88 (2) | same as RF | 0 |
| MITM, delay 0.2 → 2 s (loss 0.4) | 0 | 0 | 0.75 → 0.88 (2) | same as RF | 0 |
| Critical Node, 25% → 100% of replies suppressed | 0 | 0 → 0.88 | 0.50 → 0.88 (2) | same as RF | 0 → 0.88 |
| Data manipulation, all fields, 10% → 100% | 0 → 1.00 (from 75%) | 0 → 1.00 (only at 100%) | 1.00 (2) | 1.00, faster | 0 → 0.25 |
| Data manipulation, speed only | 0 → 0.12 | 0 | 0.50 → 1.00 (2) | same as RF | 0 |
| Data manipulation, position only | 0 | 0 | 0.50 → 0.88 (2) | same as RF | 0 |
| Sybil, 1–10 silent ghosts | 1.00 | 1.00 | 1.00 (2) | 1.00, faster | 1.00 |
| Sybil, 5 ghosts answering 0% → 60% | 1.00 → 0.03 | 1.00 → 0 | 1.00 (2) | 1.00 | 1.00 → 0 |

What this says:

* **The frozen rules do not transfer.** The Critical Node and MITM rules
  never fire on the held-out flight, not even at the recorded strength:
  that flight carries about a third of the traffic of the attack sessions,
  so 80 unanswered requests in a row (Critical Node) or 50+ exchanges in
  each direction (MITM) never accumulate, and most targets send no requests
  of their own (proof of life). Even on the two busiest drones
  (`busy target` sweeps) they stay silent. The data manipulation rule needs
  75% of the recorded strength; the Sybil rule is perfect for silent ghosts
  and collapses as soon as ghosts answer one request in ten.
* **The forest trained on the recordings alone** only recognises attacks
  close to the recorded strength, and names the wrong type for Critical
  Node and Sybil.
* **A forest that has seen varied strengths** (here, injected into another
  flight) detects most attacks at every strength tested, but slowly (median
  about 2 minutes for MITM and Critical Node), names the Critical Node type
  poorly, and costs 2 false alarms per run on this flight. It is an upper
  bound, for the reason above.
* **Rules → RF** equals the forest almost everywhere; the rules only make
  blatant data manipulation and silent Sybil faster.
* So for this thesis: machine learning is the core detector, rules are a
  cheap fast path for blatant, well-defined attacks, and rule thresholds
  tuned on one traffic regime should not be expected to hold in another.

Data manipulation, recording 1 → recording 2 and back (forest trained on
the clean baseline, the other three attack types and one data manipulation
recording; window-level F1 for the data manipulation class): rules 1.00 /
1.00, forest 0.95 / 0.75, rules → forest 0.95 / 1.00.

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

### Scaling after the rules and cascade change

The AWS numbers above were measured before this branch added three rules
and the multiclass forest. To see what the change costs, the old code
(`main` before the change) and the new code were run back to back on the
same machine (a 4-CPU cloud container, Python 3.12, Redpanda in docker),
`docs/scaling_rerun/old_vs_new_same_machine.json`. Only the ratio between
old and new is meaningful here, not the absolute values.

| Measurement | Old | New | Change |
|---|---|---|---|
| Kafka, 1 consumer (msg/s) | 81,270 | 61,632 | −24% |
| Kafka, 4 consumers (msg/s, speedup) | 217,974 (2.68×) | 188,763 (3.06×) | −13% |
| Offline, 1 consumer (msg/s) | 95,082 | 71,719 | −25% |
| Offline, 4 consumers (speedup) | 3.78× | 3.62× | near-linear in both |
| Offline, 1,728 drones (msg/s, p99 per message) | 95,000, 0.016 ms | 72,200, 0.027 ms | −24% |
| Offline + ML, 1,728 drones (msg/s) | 44,734 | 38,311 | −14% |
| Latency at 20,000 msg/s (median, p99) | 1.3 ms, 3.2 ms | 1.4 ms, 12 ms | median unchanged |
| Latency at 50,000 msg/s (median, p99) | 1.7 ms, 97 ms | 110 ms, 509 ms | one consumer saturated |
| Training, feature extraction at 30× | 39.5 s | 49.3 s | +25% |
| Training, RF fit and peak memory at 30× | 1.1 s, 396 MB | 1.0 s, 409 MB | unchanged |

The rules cost about a quarter of single-consumer throughput, because every
message now goes through four rules instead of one. Adding consumers still
scales close to linearly, every verdict check passed, and one consumer
still handles about 60,000 msg/s, over 2,000× what the recorded swarm
produces (about 25 msg/s). The cost shows up only near saturation: at
50,000 msg/s a single consumer is now close to its limit, so queues grow
and latency rises; a second consumer removes that. The steps to repeat the
official AWS runs on this code are in `docs/scaling_rerun/ec2_rerun.md`.

### Machine learning in the stream (Track B)

`python train_ml.py` on the AWS instance (scikit-learn 1.5.2, as pinned).
Per drone, per 15 s window, 47 features, built by the live pipeline itself.
Every number is for data the model was not trained on.

| Attack | Isolation Forest AUC | attacked drone alerted | Random Forest AUC (attack session unseen) | F1 | attacked drone alerted |
|---|---|---|---|---|---|
| Critical Node | 0.94 | yes, 25 s | 1.00 | 1.00 | yes, 25 s |
| MITM | 0.91 | yes, 85 s | 0.79 | 0.00 | no |
| Data manipulation | 0.89 | yes, 25 s | 0.93 | 0.71 | yes, 10 s |
| Data manipulation 2 | 0.96 | yes, 25 s | 1.00 | 0.96 | yes, 10 s |
| Sybil | 1.00 | 8 of 8, 25 s | 1.00 | 1.00 | 8 of 8, 10 s |

The Random Forest is now multiclass (normal, Critical Node, data
manipulation, MITM, Sybil); its P(compromised) is 1 − P(normal). The table
above is the leave-one-session-out view (development machine, pinned
versions; the Isolation Forest numbers are identical to the earlier AWS run).
These columns describe the models on their own; in the pipeline they only
see drones no rule caught (see the cascade section).

False alarms on a second clean recording: 0 windows, 0 drones (both
models), and no alert in any session before its attack began. A 10 s delay
means the alert came from the 15 s window in which the attack started.
The anomaly detector also puts 0–6 honest drones per attack session
on the watch list — drones disturbed by the attack — which is why it never
triggers an action on its own. The deployed classifier is trained without
Critical Node, so the live demo shows it on a recording it never saw.

### Training scalability

`python tools/train_scale.py` on the AWS instance (m7i-flex.large, 2 vCPUs,
pinned versions). The swarm is cloned as in the swarm experiment; accuracy
is measured on the same original-size test data at every size.

| Data | Messages | Windows | Feature extraction | RF fit | Peak memory | RF AUC (unseen attack) |
|---|---|---|---|---|---|---|
| 1× | 83,906 | 3,146 | 4.2 s | 0.4 s | 203 MB | 0.95 |
| 3× | 251,718 | 9,438 | 7.1 s | 0.7 s | 218 MB | 0.93 |
| 10× | 839,060 | 31,460 | 16 s | 1.2 s | 266 MB | 0.92 |
| 30× | 2,517,180 | 94,380 | 42 s | 2.8 s | 425 MB | 0.92 |

30× the data costs 7× the fit time and 2× the memory; AUC stays within 0.92–0.95. Clones add no new
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
