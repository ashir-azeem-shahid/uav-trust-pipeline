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
| `produce.py` | Stream → Kafka, keyed by the drone under evaluation. `--dry-run` works with no broker. |
| `rules.py` | Track A: the suppression rule, as streaming state. |
| `detect.py` | The consumer. Track A wired in; stages 1, 2 and 4 have marked slots. |
| `tools/analysis3.py` | Data diagnostics — the go/no-go test on the raw CSVs. |
| `tools/simulate.py` | Replays through simulated partitioned consumers. Proves the architecture before Kafka. |
| `tools/sweep_streak.py` | Threshold sensitivity sweep + the max-streak separation table. |
| `tests/test_leakage.py` | Asserts ground truth never reaches the operational path. |

Still to come: `dq.py` (stage 1), `trust.py` (stage 2), `train.py` + Track B
(stage 4), `app.py` (dashboard), `sink.py` (Parquet → S3).

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

## Current measured result

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
