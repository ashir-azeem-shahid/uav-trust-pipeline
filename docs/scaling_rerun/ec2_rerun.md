# Re-running the scaling tests on AWS (new code)

The official scaling numbers in the README come from AWS (eu-north-1):
broker runs on a `c7i.2xlarge` (8 vCPUs, 4 physical cores) and training
scalability on an `m7i-flex.large` (2 vCPUs). These steps repeat them on the
code in this branch so the thesis reports the system as submitted. Use the
same instance types so the numbers are comparable.

## 1. Instance and code

```bash
# on a fresh c7i.2xlarge (Amazon Linux 2023 or Ubuntu 24.04), as ec2-user/ubuntu
sudo dnf install -y git docker python3.12 || sudo apt-get install -y git docker.io docker-compose-v2 python3.12-venv
sudo systemctl enable --now docker && sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/ashir-azeem-shahid/uav-trust-pipeline
cd uav-trust-pipeline
git checkout claude/project-thread-4yscgb        # or main, once the PR is merged
git clone https://github.com/MattFechack/Digital-Twin-framework-for-trust-evaluation-of-autonomous-drones bsc-reference

python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                  # pinned; needs Python 3.12, not 3.13
python -m pytest tests/ -q                       # expect all tests to pass
python train_ml.py --no-eval                     # writes models/ml.joblib for the --ml runs
```

## 2. Broker

```bash
REDPANDA_SMP=2 REDPANDA_MEMORY=4G docker compose up -d
docker compose ps                                # redpanda healthy, init-topics exited 0
```

`REDPANDA_SMP=2` gives the broker two cores so it is never the bottleneck,
as in the original runs.

## 3. The runs (about 15-20 minutes)

```bash
python tools/scale_test.py consumers --copies 64 --consumers 1 2 4 --reps 3
python tools/scale_test.py swarm --copies 1 2 4 8 16 32 64 --reps 3
python tools/scale_test.py swarm --copies 1 8 32 --reps 3 --ml --out results_scaling_swarm_ml.json
python tools/scale_test.py latency
```

These overwrite `results_scaling_consumers.json`, `results_scaling_swarm.json`
and `results_scaling_latency.json`, and add `results_scaling_swarm_ml.json`
(throughput with the ML stage switched on through Kafka).

## 4. Training scalability (on an m7i-flex.large, as before)

```bash
python tools/train_scale.py                      # factors 1 3 10 30, writes results_train_scale.json
```

## 5. Bring the results back

```bash
git add results_scaling_*.json results_train_scale.json
git commit -m "AWS scaling results on the new pipeline"
git push
docker compose down                              # then stop the instances
```

Each results file records the machine, CPU count, Python and library
versions it ran with, so the thesis can quote them.
