"""
ml.py
=====
Track B: machine learning inside the streaming pipeline.

Two models score every drone once per 15-second window, live:

  Isolation Forest  anomaly detection. Fitted on clean baseline traffic only,
                    never shown an attack or a label. Flags a drone whose
                    window looks unlike anything in normal operation.
  Random Forest     trustworthiness classification. Supervised, trained on
                    labelled windows from the recorded attack sessions.
                    Multiclass: normal or the attack type (Critical Node,
                    data manipulation, MITM, Sybil). P(compromised) is
                    1 - P(normal); the alert names the most likely type.

Both run only on drones that no rule has flagged (detect.py is the gate):
the rules are the cheap first stage, the models the second.

WHY WINDOWS ARE KEYED BY THE EVALUATED DRONE
--------------------------------------------
features.py keys windows by the OBSERVER (requester), to reproduce the
previous thesis's recipe for the comparison table. A live pipeline cannot
use that: Kafka partitions by the drone being evaluated, so one consumer does
not see all of an observer's exchanges. Here a window is (subject, 15 s
bucket), built from exactly the messages that land on the subject's
partition -- requests TO the subject (target copies) and the subject's own
requests (initiator copies). One consumer therefore holds everything a
window needs, and the model scales with the partitions like the rule does.

ONE CODE PATH FOR TRAINING AND SERVING
--------------------------------------
SubjectWindower is the only feature code. train_ml.py replays the recorded
sessions through the same Pipeline that runs live and collects the windows
it closes; detect.py scores the same windows as they close. Training and
live features cannot drift apart ("training/serving skew").

WHAT A WINDOW CONTAINS (all from operational fields; never from _gt)
-------------------------------------------------------------------
  requests to the drone      count, share unanswered
  trust of its answers       mean/min/max/std of the 7 trust metrics, on
                             answered exchanges only (a timeout carries a
                             trust score of 0 by construction, not a judgement)
  data quality               mean of the 5 DQ dimensions, lowest record
                             score, quarantined count
  suppression evidence       longest unanswered streak, streak at window end,
                             alive-while-silent flag
  its own activity           requests it made, share that succeeded
  telemetry payload          mean/std of reported speed and rotor speed

A window closes when the consumer's stream clock passes its end. Silent
drones produce no window, which is correct: no data, nothing to score.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

import config as C

WINDOW = C.WINDOW_SECONDS
# A window closes GRACE seconds after its end, so messages that arrive a
# little out of order -- one consumer reads several partitions, which Kafka
# orders only within each partition -- still land in the right window.
GRACE = 5
MODEL_PATH = Path("models") / "ml.joblib"

TRUST_METRICS = ["Trust_Score", "Trust_RI", "Trust_BFM", "Trust_SM",
                 "Trust_CDM", "Trust_BA", "Trust_CR"]
DQ_DIMS = ["completeness", "validity", "timeliness", "uniqueness", "consistency"]
PAYLOAD = ["DT_Speed_mps", "DT_RotorAvgSpeed"]


class _Acc:
    __slots__ = ("n_req", "n_timeout", "trust", "dq", "dq_min", "n_quar",
                 "n_init", "n_init_ok", "max_streak", "streak_end",
                 "alive_end", "payload", "last_t")

    def __init__(self) -> None:
        self.n_req = 0
        self.n_timeout = 0
        self.trust = {m: [] for m in TRUST_METRICS}
        self.dq = {d: [] for d in DQ_DIMS}
        self.dq_min = 1.0
        self.n_quar = 0
        self.n_init = 0
        self.n_init_ok = 0
        self.max_streak = 0
        self.streak_end = 0
        self.alive_end = 0
        self.payload = {p: [] for p in PAYLOAD}
        self.last_t = 0

    def features(self) -> dict[str, float]:
        f: dict[str, float] = {}
        f["n_requests"] = float(self.n_req)
        f["unanswered_rate"] = self.n_timeout / self.n_req if self.n_req else 0.0
        answered = self.n_req - self.n_timeout - self.n_quar
        f["n_answered"] = float(max(answered, 0))
        for m in TRUST_METRICS:
            v = self.trust[m]
            f[f"{m}_mean"] = st.fmean(v) if v else 0.0
            f[f"{m}_min"] = min(v) if v else 0.0
            f[f"{m}_max"] = max(v) if v else 0.0
            f[f"{m}_std"] = st.pstdev(v) if len(v) > 1 else 0.0
        for d in DQ_DIMS:
            v = self.dq[d]
            f[f"dq_{d}"] = st.fmean(v) if v else 1.0
        f["dq_min_score"] = self.dq_min
        f["dq_quarantined"] = float(self.n_quar)
        f["max_streak"] = float(self.max_streak)
        f["streak_end"] = float(self.streak_end)
        f["alive_while_silent"] = float(self.streak_end > 0 and self.alive_end > 0)
        f["n_initiated"] = float(self.n_init)
        f["initiated_ok_rate"] = self.n_init_ok / self.n_init if self.n_init else 0.0
        for p in PAYLOAD:
            v = self.payload[p]
            f[f"{p}_mean"] = st.fmean(v) if v else 0.0
            f[f"{p}_std"] = st.pstdev(v) if len(v) > 1 else 0.0
        return f


@dataclass
class Window:
    subject: str
    start: int
    features: dict[str, float]
    onset: int | None = None            # carried for evaluation only
    meta: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.start + WINDOW


class SubjectWindower:
    """
    Streaming (subject, 15 s) windows. Feed every message with the DQ result
    and the rule state the pipeline already computed; collect closed windows.

    Time is kept PER PARTITION. Kafka orders messages only within a
    partition: a consumer reading several partitions can receive one
    partition's backlog before another's, so a single clock would close the
    other partitions' windows before their messages arrived. A drone's
    messages all live on one partition, so its windows close on that
    partition's clock alone. Offline (no partition) there is one clock.
    """

    def __init__(self, grace: int = GRACE) -> None:
        self.open: dict[tuple[str, int], tuple[_Acc, object]] = {}
        self.grace = grace
        self.clock: dict[object, int] = {}         # partition -> latest t
        self.closed_below: dict[object, int] = {}  # partition -> first open bucket
        self.late = 0           # messages that arrived after their window closed
        self.onset: int | None = None

    def observe(self, msg: dict, dq_res, drone_state,
                partition=None) -> list[Window]:
        t = msg["t_session"]
        self.onset = msg.get("onset", self.onset)
        closed = []
        if t > self.clock.get(partition, -1):
            self.clock[partition] = t
            # close this partition's buckets that ended `grace` seconds ago
            closed = self._close(partition, (t - self.grace) // WINDOW)

        bucket = t // WINDOW
        if bucket < self.closed_below.get(partition, 0):
            self.late += 1      # too late: its window has already been scored
            return closed
        key = (msg["about"], bucket)
        entry = self.open.get(key)
        if entry is None:
            entry = self.open[key] = (_Acc(), partition)
        a = entry[0]
        a.last_t = t
        is_timeout = msg["exchange_type"] == C.TIMEOUT

        if msg["role"] == "target":
            a.n_req += 1
            if is_timeout:
                a.n_timeout += 1
            elif dq_res is not None and dq_res.quarantine:
                a.n_quar += 1             # out of the trust path
            else:
                for m in TRUST_METRICS:
                    v = (msg.get("trust") or {}).get(m)
                    if v is not None:
                        a.trust[m].append(float(v))
                for p in PAYLOAD:
                    v = (msg.get("dt") or {}).get(p)
                    if isinstance(v, (int, float)):
                        a.payload[p].append(float(v))
            if dq_res is not None:
                for d in DQ_DIMS:
                    a.dq[d].append(getattr(dq_res, d))
                a.dq_min = min(a.dq_min, dq_res.score)
        else:
            a.n_init += 1
            a.n_init_ok += int(not is_timeout)

        if drone_state is not None:
            a.max_streak = max(a.max_streak, drone_state.streak)
            a.streak_end = drone_state.streak
            a.alive_end = len(drone_state.initiated_ok)
        return closed

    def _close(self, partition, bucket: int, everything: bool = False) -> list[Window]:
        if not everything:
            self.closed_below[partition] = max(
                self.closed_below.get(partition, 0), bucket)
        done = [k for k, (_a, part) in self.open.items()
                if everything or (part == partition and k[1] < bucket)]
        out = []
        for k in sorted(done, key=lambda k: (k[1], k[0])):
            a, _part = self.open.pop(k)
            out.append(Window(k[0], k[1] * WINDOW, a.features(), self.onset))
        return out

    def flush(self) -> list[Window]:
        return self._close(None, 0, everything=True)


# ---------------------------------------------------------------------
# labels -- evaluation only, from ground truth, never a feature
# ---------------------------------------------------------------------
NORMAL = "normal"
ATTACK_CLASSES = ["CriticalNode", "DataManipulation", "MITM", "Sybil"]
CLASSES = [NORMAL] + ATTACK_CLASSES


def session_class(attack: str) -> str:
    """Session key -> attack class: 'DataManipulation2' -> 'DataManipulation'."""
    name = attack.rstrip("0123456789")
    if name not in ATTACK_CLASSES:
        raise ValueError(f"unknown attack session {attack!r}")
    return name


def window_class(window: "Window", binary_label: int, attack: str) -> str:
    """The multiclass label: the session's attack type if compromised."""
    return session_class(attack) if binary_label else NORMAL


def label(window: Window, victim: str | None, attack_type: str | None) -> int:
    """
    1 if the telemetry about this drone in this window is compromised:
      Sybil:      the subject is a fabricated (Ghost) identity
      otherwise:  the subject is the attack target, after the attack began
    Mirrors features._row_label, applied to the evaluated drone.
    """
    if window.onset is None or window.start < window.onset:
        return 0
    if attack_type and attack_type.strip().lower() == "sybil":
        return int("ghost" in window.subject.lower())
    # a cloned swarm (tools/train_scale.py) renames Drone18 -> Drone18~c3
    return int(victim is not None
               and window.subject.split("~c")[0] == victim.split("~c")[0])


def straddles_onset(window: Window) -> bool:
    return (window.onset is not None
            and window.start < window.onset < window.end)


def session_windows(attack: str, baseline=None,
                    stream=None) -> list[tuple[Window, int]]:
    """
    Replay one recorded session through the live Pipeline and return every
    window it closes, with its label. Training and evaluation use this, so
    their features come from exactly the code that runs on the stream.

    stream: optional iterable of (key, msg, onset) to use instead of the
    plain replay -- tools/train_scale.py passes a cloned, larger swarm.
    """
    import replay as R
    from detect import Pipeline

    pipe = Pipeline(collect_windows=True)
    victim = attack_type = None
    src = stream if stream is not None else R.replay_with_onset(attack, baseline)
    for _key, msg, _onset in src:
        gt = msg["_gt"]                      # read here for the LABEL only
        victim = gt["attack_target"] or victim
        attack_type = gt["attack_type"] or attack_type
        pipe.handle(msg)
    pipe.finish()
    out = []
    for w in pipe.windows:
        lab = label(w, victim, attack_type)
        w.meta["cls"] = window_class(w, lab, attack)
        # what the rule stage had said about this drone by the window's end:
        # the cascade's first stage, replayed for evaluation
        hit = pipe.rule_hits.get(w.subject)
        w.meta["rule"] = hit[1] if hit and hit[2] <= w.end else None
        out.append((w, lab))
    return out


# ---------------------------------------------------------------------
# live scoring
# ---------------------------------------------------------------------
class MLScorer:
    """
    Loads models/ml.joblib (written by train_ml.py) and scores closed
    windows. Returns verdict dicts in the same shape as the rule's.
    """

    def __init__(self, path: Path | str | None = MODEL_PATH,
                 bundle: dict | None = None) -> None:
        if bundle is None:
            import joblib
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found -- run `python train_ml.py` first")
            bundle = joblib.load(path)
        b = bundle
        self.names: list[str] = b["feature_names"]
        self.iforest = b["iforest"]
        self.if_threshold: float = b["if_threshold"]
        self.rf = b["rf"]
        if hasattr(self.rf, "n_jobs"):
            self.rf.n_jobs = 1        # ~54 rows per batch: threads cost more than they save
        # multiclass: classes_ holds "normal" plus attack names. A binary
        # model from before (classes 0/1) still loads: class 1 = compromised.
        classes = list(getattr(self.rf, "classes_", [0, 1]))
        self.classes = [str(c) for c in classes]
        self._normal = (classes.index(NORMAL) if NORMAL in classes
                        else classes.index(0))
        self.rf_threshold: float = b["rf_threshold"]
        self.info: dict = b.get("info", {})
        self.if_persist: int = b.get("if_persist", 2)
        self.flagged_if: set[str] = set()
        self.flagged_rf: set[str] = set()
        self._if_run: dict[str, tuple[int, int]] = {}   # drone -> (last start, run)
        self.windows_scored = 0

    def score_many(self, windows: list[Window]):
        """One batched call per model: windows close together at each
        15 s boundary, and per-row sklearn calls would cost ~100x more."""
        import numpy as np
        anomaly, proba = self.score_full(windows)
        return anomaly, 1.0 - proba[:, self._normal]

    def score_full(self, windows: list[Window]):
        """Anomaly scores and the classifier's full class-probability matrix."""
        import numpy as np
        X = np.array([[w.features[n] for n in self.names] for w in windows],
                     dtype=float)
        return -self.iforest.score_samples(X), self.rf.predict_proba(X)

    def attack_type(self, proba_row) -> str:
        """The most likely attack class (never 'normal')."""
        best = max((p, c) for i, (p, c) in enumerate(zip(proba_row, self.classes))
                   if i != self._normal)
        return best[1] if best[1] not in ("1", "True") else "unknown"

    def verdicts(self, windows: list[Window]) -> list[dict]:
        windows = [w for w in windows if w.features["n_requests"] > 0]
        if not windows:     # only its own activity: nothing to evaluate
            return []
        self.windows_scored += len(windows)
        anomalies, proba = self.score_full(windows)
        out = []
        for w, anomaly, row in zip(windows, anomalies, proba):
            anomaly, p_bad = float(anomaly), 1.0 - float(row[self._normal])
            kind = self.attack_type(row)
            base = {"drone": w.subject, "t_session": w.end,
                    "window_start": w.start, "onset": w.onset,
                    "detection_delay_s": (w.end - w.onset)
                    if w.onset is not None else None,
                    "anomaly_score": round(anomaly, 4),
                    "anomaly_threshold": round(self.if_threshold, 4),
                    "p_compromised": round(p_bad, 4),
                    "attack_type": kind, "stage": "ml"}
            # every window's scores, so the dashboard can show a live ML
            # reading for every drone -- not only for the alerted ones
            out.append({**base, "verdict": "SCORE", "track": "B",
                        "anomalous": anomaly >= self.if_threshold,
                        "compromised": p_bad >= self.rf_threshold,
                        "reason": ""})
            # one alert per drone per model: the first window that crosses
            if p_bad >= self.rf_threshold and w.subject not in self.flagged_rf:
                self.flagged_rf.add(w.subject)
                out.append({**base, "verdict": "ML_COMPROMISED", "track": "B-RF",
                            "reason": _explain_rf(w, p_bad, kind)})
            # anomaly alert needs `if_persist` consecutive anomalous windows:
            # a single odd window is common in a disturbed swarm (see
            # results_ml.json), a persistent one is worth an operator's look
            last, run = self._if_run.get(w.subject, (None, 0))
            if anomaly >= self.if_threshold:
                run = run + 1 if last is not None and w.start - last == WINDOW else 1
            else:
                run = 0
            self._if_run[w.subject] = (w.start, run)
            if run >= self.if_persist and w.subject not in self.flagged_if:
                self.flagged_if.add(w.subject)
                out.append({**base, "attack_type": None,
                            "verdict": "ANOMALY", "track": "B-IF",
                            "reason": _explain_if(w, anomaly, self.if_threshold)})
        return out


def _evidence(f: dict) -> str:
    bits = [f"{int(f['n_requests'])} requests, {f['unanswered_rate']:.0%} unanswered"]
    if f["n_answered"]:
        bits.append(f"mean trust of answers {f['Trust_Score_mean']:.2f}")
    if f["max_streak"] >= 10:
        bits.append(f"unanswered streak {int(f['max_streak'])}")
    if f["dq_min_score"] < 1.0:
        bits.append(f"lowest DQ score {f['dq_min_score']:.2f}")
    return ", ".join(bits)


def _explain_rf(w: Window, p: float, kind: str = "") -> str:
    kind = f", most likely {kind}" if kind else ""
    return (f"classifier: P(compromised) = {p:.2f}{kind} for window "
            f"{w.start}-{w.end}s ({_evidence(w.features)})")


def _explain_if(w: Window, s: float, thr: float) -> str:
    return (f"anomaly: unlike clean traffic for consecutive windows, latest "
            f"{w.start}-{w.end}s scored {s:.3f} (threshold {thr:.3f}; "
            f"{_evidence(w.features)})")
