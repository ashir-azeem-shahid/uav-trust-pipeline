"""
features.py
===========
Turns the message stream into labelled 15-second windows.

MATCHES FECAK'S RECIPE EXACTLY, then adds this thesis's features on top.
His aggregation is in dashboard/ui_app.py, functions _row_labels and
_drone_windows. Reproducing it is what makes the F1 comparison against his
Table 8.2 like-for-like rather than approximate.

HIS RECIPE (reproduced)
-----------------------
* Window = (Requester, 15-second bucket). A window is what ONE OBSERVER saw
  in 15 seconds -- not what one evaluated drone did.
* Row label = Requester or ContactedDrone equals Attack_Target.
  For Sybil, instead: the row touches a Ghost identity. His own comment
  explains why -- labelling every row of the Sybil session turns it into
  "100%/0% session detection, which is a trivial task, not real attacker
  localisation".
* Window label = max of its row labels. Any malicious row makes the window
  malicious.
* Features: mean / std / min of each trust component, plus timeout_rate and
  zero_trust_rate.
* Grouped by Requester for StratifiedGroupKFold, so one observer's windows
  never span train and test.

THIS THESIS ADDS
----------------
* The five data quality dimension scores  -> the ablation answers RQ2
* Suppression evidence from the streaming state (max streak observed,
  proof-of-life flag)                     -> the Critical Node contribution
* max and slope alongside his mean/std/min, since a streaming consumer can
  see a trend within the window and a batch aggregate cannot

The ablation switches the added groups off and leaves his exactly intact,
so the delta is attributable.

The label never comes from Trust_Label -- that column marks 1,040 of 2,420
rows MALICIOUS in a session where no attack ran.
"""

from __future__ import annotations

import statistics as st

import config as C
import replay as R
from detect import Pipeline

WINDOW = C.WINDOW_SECONDS

TRUST_METRICS = ["Trust_Score", "Trust_RI", "Trust_BFM", "Trust_SM",
                 "Trust_CDM", "Trust_BA", "Trust_CR"]

DQ_FEATURES = ["dq_completeness", "dq_validity", "dq_timeliness",
               "dq_uniqueness", "dq_consistency"]

SUPPRESSION_FEATURES = ["max_streak_seen", "alive_while_silent",
                        "peers_fully_timed_out"]


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2
    my = sum(values) / n
    denom = sum((i - mx) ** 2 for i in range(n))
    return 0.0 if denom == 0 else sum(
        (i - mx) * (v - my) for i, v in enumerate(values)) / denom


def _row_label(msg: dict, victim: str | None, attack_type: str | None) -> int:
    """
    Fecak's _row_labels, message-for-message.

    Sybil: the row must touch a Ghost identity.
    Everything else: requester or contacted equals the attack target.
    """
    if attack_type and attack_type.strip().lower() == "sybil":
        return int("ghost" in msg["requester"].lower()
                   or "ghost" in msg["contacted"].lower())
    if not victim:
        return 0
    return int(msg["requester"] == victim or msg["contacted"] == victim)


class _Window:
    """Accumulator for one (requester, 15s bucket) window."""

    __slots__ = ("trust", "dq", "n_rows", "n_timeout", "n_zero_trust",
                 "label", "streaks", "alive_silent", "fully_timed_out_peers")

    def __init__(self) -> None:
        self.trust: dict[str, list[float]] = {m: [] for m in TRUST_METRICS}
        self.dq: dict[str, list[float]] = {f: [] for f in DQ_FEATURES}
        self.n_rows = 0
        self.n_timeout = 0
        self.n_zero_trust = 0
        self.label = 0
        self.streaks: list[int] = []
        self.alive_silent = 0
        self.fully_timed_out_peers: set[str] = set()

    def to_features(self) -> dict[str, float]:
        f: dict[str, float] = {}

        # --- Fecak's feature set, unchanged -------------------------
        for m in TRUST_METRICS:
            v = self.trust[m]
            f[f"{m}_mean"] = st.mean(v) if v else 0.0
            f[f"{m}_std"] = st.pstdev(v) if len(v) > 1 else 0.0
            f[f"{m}_min"] = min(v) if v else 0.0
        f["timeout_rate"] = self.n_timeout / self.n_rows if self.n_rows else 0.0
        f["zero_trust_rate"] = (self.n_zero_trust / self.n_rows
                                if self.n_rows else 0.0)

        # --- added: within-window trend, visible only to a stream ----
        for m in TRUST_METRICS:
            v = self.trust[m]
            f[f"{m}_max"] = max(v) if v else 0.0
            f[f"{m}_slope"] = _slope(v)

        # --- added: suppression evidence -----------------------------
        f["max_streak_seen"] = float(max(self.streaks)) if self.streaks else 0.0
        f["alive_while_silent"] = float(self.alive_silent)
        f["peers_fully_timed_out"] = float(len(self.fully_timed_out_peers))

        # --- added: data quality -------------------------------------
        for d in DQ_FEATURES:
            vals = self.dq[d]
            f[d] = st.mean(vals) if vals else 1.0
        return f


def build_windows(attack: str, window_mode: str = "15s") -> list[dict]:
    """
    Replay one session; return its labelled windows.

    window_mode:
      "15s"      -- genuine 15-second buckets. The real streaming task.
      "session"  -- one window per (requester, session phase), which is what
                    Fecak's code actually produces. Use this, and only this,
                    when comparing against his Table 8.2.

    WHY "session" MODE EXISTS
    -------------------------
    His _drone_windows() computes the bucket as

        ts = pd.to_numeric(work["Timestamp"], errors="coerce")
        work["_bucket"] = ((ts - t0) // WINDOW_SEC)

    but Timestamp is a string of the form "16:57:48", so pd.to_numeric
    coerces every value to NaN. The bucket column is entirely NA, and
    groupby(..., dropna=False) therefore collapses all of a requester's rows
    into a single window covering the whole session.

    Verified against his own reported counts: 53 requesters in the MITM
    session plus 54 in the baseline gives exactly the 107 windows his
    confusion matrix reports, and 54 + 54 gives the 108 reported for the
    other three.

    So his classifier answers "did this observer encounter the attacked
    drone at any point in the session?", not "is this drone under attack
    right now?". That is a materially easier question, it is why his F1 is
    higher on the attacks where payload evidence accumulates over minutes,
    and it is structurally incapable of producing a detection delay.

    Report both modes. "session" makes the comparison to his table honest;
    "15s" is the task this thesis actually solves.
    """
    if window_mode not in ("15s", "session"):
        raise ValueError(f"window_mode must be '15s' or 'session', got {window_mode!r}")

    pipe = Pipeline()
    wins: dict[tuple[str, int], _Window] = {}
    victim = None
    attack_type = None
    onset = None

    # per-peer tallies inside the current bucket, for the added features
    peer_seen: dict[tuple[str, int], dict[str, list[int]]] = {}

    for _key, msg, ons in R.replay_with_onset(attack):
        onset = ons
        if msg["_gt"]["attack_target"]:
            victim = msg["_gt"]["attack_target"]
        if msg["_gt"]["attack_type"]:
            attack_type = msg["_gt"]["attack_type"]

        # Always feed the pipeline so DQ scores and streaks match inference.
        pipe.handle(msg)

        # Fecak aggregates one row per exchange. Our stream carries each
        # exchange twice (once per role), so score the 'target' copy only.
        if msg["role"] != "target":
            continue

        requester = msg["requester"]
        if window_mode == "15s":
            bucket = msg["t_session"] // WINDOW
        else:
            # One window per requester per session phase -- what his code
            # collapses to. Phase 0 = baseline, phase 1 = attack session.
            bucket = int(msg["t_session"] >= ons)
        key = (requester, bucket)
        w = wins.get(key)
        if w is None:
            w = wins[key] = _Window()
            peer_seen[key] = {}

        w.n_rows += 1
        is_timeout = msg["exchange_type"] == C.TIMEOUT
        if is_timeout:
            w.n_timeout += 1

        for m in TRUST_METRICS:
            v = msg["trust"].get(m)
            if v is not None:
                w.trust[m].append(v)
        if (msg["trust"].get("Trust_Score") or 0) <= 0:
            w.n_zero_trust += 1

        if pipe.last_dq is not None:
            for k, v in pipe.last_dq.as_features().items():
                w.dq[k].append(v)

        # suppression evidence about the peer this observer contacted
        peer = msg["contacted"]
        stt = pipe.suppression.drones.get(peer)
        if stt is not None:
            w.streaks.append(stt.streak)
            if stt.streak > 0 and len(stt.initiated_ok) > 0:
                w.alive_silent = 1
        seen = peer_seen[key].setdefault(peer, [0, 0])
        seen[0] += 1
        seen[1] += int(is_timeout)
        if seen[0] >= 3 and seen[0] == seen[1]:
            w.fully_timed_out_peers.add(peer)

        w.label = max(w.label, _row_label(msg, victim, attack_type))

    out = []
    for (requester, bucket), w in wins.items():
        if w.n_rows == 0:
            continue
        out.append({
            "features": w.to_features(),
            "label": w.label,
            "group": requester,             # Fecak groups by Requester
            "session": attack,
            "window_start": bucket * WINDOW,
            "post_onset": onset is not None and bucket * WINDOW >= onset,
        })
    return out


def build_all(sessions: list[str] | None = None) -> list[dict]:
    rows = []
    for name in (sessions or list(C.ATTACK_SESSIONS)):
        rows.extend(build_windows(name))
    return rows


def feature_names(rows: list[dict], include_dq: bool = True,
                  include_suppression: bool = True) -> list[str]:
    names = sorted(rows[0]["features"])
    if not include_dq:
        names = [n for n in names if n not in DQ_FEATURES]
    if not include_suppression:
        names = [n for n in names if n not in SUPPRESSION_FEATURES]
    return names


def to_matrix(rows: list[dict], include_dq: bool = True,
              include_suppression: bool = True):
    """Returns (X, y, groups, feature_names)."""
    names = feature_names(rows, include_dq, include_suppression)

    leaked = set(names) & C.FORBIDDEN_FEATURES
    if leaked:
        raise AssertionError(f"ground truth leaked into features: {leaked}")

    X = [[r["features"][n] for n in names] for r in rows]
    y = [r["label"] for r in rows]
    groups = [r["group"] for r in rows]
    return X, y, groups, names
