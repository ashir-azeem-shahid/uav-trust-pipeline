#!/usr/bin/env bash
# The pipeline end to end, live and autonomous:
#   producer -> Kafka -> detector (DQ + rule + ML) -> uav.verdict
#            -> responder (auto-isolation) -> uav.action -> dashboard
#
# Starts the broker if needed, trains the ML models if they are missing, runs
# the detector, the responder and the dashboard in the background (tmux
# session "uav"), waits for you to connect the dashboard, then replays a
# session at 10x speed so it plays out in under a minute.
#
# Usage (from the repo root, venv created):
#   bash tools/live_demo.sh                    # clean Critical Node session
#   bash tools/live_demo.sh trust_corrupt      # same, with a benign fault on 5%
#                                              # of records (see faults.py)
#   SPEED=5 SESSION=Sybil bash tools/live_demo.sh
#   GRACE=20 bash tools/live_demo.sh           # longer override countdown
#
# Afterwards:  tmux attach -t uav   to see the detector's output
#              (Ctrl+B then D to leave it running, Ctrl+B then & to close a window)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

FAULT="${1:-}"
SPEED="${SPEED:-10}"
SESSION="${SESSION:-CriticalNode}"
GRACE="${GRACE:-10}"
TOPICS="uav.telemetry uav.verdict uav.operator uav.action"

if [ ! -f models/ml.joblib ]; then
    echo "Training the ML models (first run only, about a minute)..."
    python train_ml.py --no-eval >/dev/null
fi

command -v tmux >/dev/null || sudo apt-get install -y tmux >/dev/null

echo "Starting the broker..."
docker compose up -d >/dev/null
for _ in $(seq 1 30); do
    if docker compose ps redpanda | grep -q "(healthy)"; then break; fi
    sleep 2
done
docker compose ps redpanda | grep -q "(healthy)" || { echo "Broker not healthy"; exit 1; }

# Fresh topics every run. Otherwise messages and verdicts left over from an
# earlier run would be processed first and appear in the demo as a second,
# stale alert. Deleting a topic also clears the consumer offsets on it.
echo "Resetting topics for a clean run..."
sleep 3   # let the compose init job finish its own topic creation first
# shellcheck disable=SC2086
docker exec redpanda rpk topic delete ${TOPICS} >/dev/null 2>&1 || true
sleep 1
docker exec redpanda rpk topic create uav.telemetry -p 4 -r 1 >/dev/null 2>&1 || true
docker exec redpanda rpk topic create uav.verdict   -p 4 -r 1 >/dev/null 2>&1 || true
docker exec redpanda rpk topic create uav.operator  -p 1 -r 1 >/dev/null 2>&1 || true
docker exec redpanda rpk topic create uav.action    -p 1 -r 1 >/dev/null 2>&1 || true
for t in ${TOPICS}; do
    docker exec redpanda rpk topic list | grep -qE "(^|[[:space:]])${t}([[:space:]]|$)" \
        || { echo "Topic ${t} missing after reset"; exit 1; }
done

# The public address, from the EC2 instance metadata (IMDSv2). Falls back to
# localhost when not on EC2.
TOKEN=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
HOST=$(curl -s -m 2 -H "X-aws-ec2-metadata-token: ${TOKEN}" \
       http://169.254.169.254/latest/meta-data/public-ipv4 || true)
HOST="${HOST:-localhost}"

tmux kill-session -t uav 2>/dev/null || true
tmux new-session -d -s uav -n detector \
    "source .venv/bin/activate && python detect.py --ml; bash"
tmux new-window -t uav -n responder \
    "source .venv/bin/activate && python respond.py --grace ${GRACE}; bash"
tmux new-window -t uav -n dashboard \
    "source .venv/bin/activate && streamlit run app.py --server.address 0.0.0.0 \
     --server.port 8501 --server.headless true; bash"
sleep 3

echo
echo "Detector (rule + DQ + ML), responder (${GRACE} s override window) and dashboard are running."
echo
echo "  1. Open  http://${HOST}:8501"
echo "  2. Sidebar: Live stream (Kafka) -> Start from: New messages only -> Connect"
echo "  3. Wait about 5 seconds after it says it is waiting for messages"
echo "     (the dashboard needs a moment to join the topics)"
echo
read -r -p "  Then press Enter here to start the replay... "

echo
echo "Replaying ${SESSION} at ${SPEED}x${FAULT:+ with fault: ${FAULT}}. Watch the dashboard."
python produce.py "${SESSION}" --speed "${SPEED}" ${FAULT:+--fault "${FAULT}"}

echo
echo "Replay finished. The detector, responder and dashboard keep running."
echo "  Their output:     tmux attach -t uav   (Ctrl+B then N for the next window)"
echo "  Actions taken:    docker exec redpanda rpk topic consume uav.action -o start -n 20"
echo "  Stop everything:  tmux kill-session -t uav"
