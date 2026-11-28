"""
app.py
======
Operator dashboard for the UAV telemetry trust pipeline. Streamlit.

Two sources, one view:
  Live stream      consumes uav.telemetry, uav.verdict and uav.action from
                   Kafka as they arrive -- the pipeline at runtime
  Recorded session rebuilds the same state from the recorded CSVs with no
                   broker, optionally with benign faults injected (faults.py).
                   Alerts are then marked right or wrong against ground truth.

Layout:
  Flow strip     the pipeline as five stages, with what each stage did:
                 ingest -> data quality -> rules -> ML -> response
  Overview       alerts (drone, attack type, which layer caught it, delay,
                 evidence) with the operator loop, and the swarm
  Drone detail   one drone's trust, DQ and classifier score over time
  Data quality   the five DQ dimensions, quarantine and recent issues --
                 separate from trust: a faulty sensor is not an attacker
  Evaluation     this run against ground truth, and the offline results:
                 rules vs ML vs cascade, confusion matrices, injection curves

Run:  streamlit run app.py
      (on a server: --server.address 0.0.0.0 --server.port 8501)
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import config as C
import rules as RL
from feed import ATTACK_LABEL, DQ_DIMENSIONS, LiveState, build_offline
from ml import MODEL_PATH

st.set_page_config(page_title="UAV Trust Pipeline", layout="wide")

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# vocabulary -- every label and explanation in one place
# ---------------------------------------------------------------------
SESSION_LABEL = {
    "CriticalNode": "Critical Node",
    "MITM": "Man-in-the-middle",
    "DataManipulation": "Data manipulation",
    "DataManipulation2": "Data manipulation (2)",
    "Sybil": "Sybil",
}
FAULT_LABEL = {
    None: "None",
    "sensor_noise": "Sensor noise",
    "out_of_range": "Out-of-range value",
    "trust_corrupt": "Corrupt trust value",
    "dropout": "Sensor dropout",
    "duplicate": "Duplicate delivery",
    "position_jump": "GPS position jump",
}
RULE_HELP = {
    "Sybil": "an identity not in the fleet registry that receives requests "
             "but never sends anything itself",
    "Critical Node": f"{RL.DEFAULT_STREAK}+ requests in a row unanswered while "
                     "the drone still completes its own (alive and in range, "
                     "but refusing)",
    "Data manipulation": "5 consecutive physically impossible reports "
                         "(speed, position jumps)",
    "Man-in-the-middle": "most requests lost in both directions and answers "
                         "delayed by a second or more",
}
DIM_HELP = {
    "completeness": "Required fields present for this exchange type. A timeout "
                    "legitimately carries no payload, so it is never incomplete.",
    "validity": "Values inside physical envelopes fitted on the clean baseline; "
                "trust metrics inside [0, 1].",
    "timeliness": "Time since this drone's previous report: degrades after 30 s, "
                  "fails at 120 s.",
    "uniqueness": "The same payload from the same requester in the same second "
                  "-- a duplicate delivery or replay.",
    "consistency": "Speed implied by consecutive positions is physically possible "
                   "(<= 15 m/s) and agrees with the declared speed.",
}
STATUS_HELP = ("ISOLATED: no longer trusted -- isolated automatically or by an "
               "operator. FLAGGED: an attack alert from a rule or the classifier. "
               "WATCH: only the anomaly detector finds its traffic unusual. "
               "TRUSTED / DEGRADED / LOW: trust score of the drone's last answered "
               "exchange (>= 0.6, >= 0.3, below). UNKNOWN: no answered exchange yet.")
STATUS_COLOR = {"ISOLATED": "#7c3aed", "FLAGGED": "#b91c1c", "WATCH": "#0e7490",
                "LOW": "#c2410c", "DEGRADED": "#a16207", "TRUSTED": "#15803d",
                "UNKNOWN": "#64748b"}
SIGNAL_LABEL = {"A": "Rule", "B-RF": "Classifier", "B-IF": "Anomaly"}
SERIES_COLOR = {"Trust score": "#2563eb", "DQ score": "#94a3b8",
                "Classifier P(compromised)": "#dc2626"}
QUARANTINE_RULE = ("A record is quarantined -- removed from the trust path -- when "
                   "the mean of its five dimension scores falls below 0.5, or when "
                   "any trust metric is outside [0, 1]. Otherwise a failed check "
                   "flags it and lowers its score, but it is kept.")
METHOD_LABEL = {"rules": "Rules only", "ml": "ML only (Random Forest)",
                "cascade": "Cascade (rules, then ML)"}

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 2rem;}
.uav-head {display:flex; align-items:center; gap:.7rem; flex-wrap:wrap;}
.uav-title {font-size:1.55rem; font-weight:650; letter-spacing:-.015em; line-height:1.2;}
.uav-pill {font-size:.7rem; font-weight:700; letter-spacing:.08em; padding:.12rem .55rem;
           border-radius:999px; border:1.5px solid currentColor;}
.uav-pill.live {color:#16a34a;} .uav-pill.replay {color:#64748b;}
.uav-sub {opacity:.72; font-size:.9rem; margin:.2rem 0 .8rem 0;}
.uav-flow {display:flex; align-items:stretch; gap:.35rem; margin:.2rem 0 1rem 0;
           flex-wrap:wrap;}
.uav-stage {flex:1 1 150px; border:1px solid rgba(128,128,128,.25); border-radius:8px;
            padding:.5rem .7rem; min-width:0;}
.uav-stage .n {font-size:.72rem; font-weight:700; letter-spacing:.06em; opacity:.6;
               text-transform:uppercase;}
.uav-stage .v {font-size:1.35rem; font-weight:650; line-height:1.3;}
.uav-stage .d {font-size:.8rem; opacity:.75; line-height:1.35;}
.uav-stage.hit {border-color:#b91c1c; background:rgba(185,28,28,.05);}
.uav-stage.hit .v {color:#b91c1c;}
.uav-arrow {align-self:center; opacity:.4; font-size:1.1rem;}
.uav-ev {border-left:3px solid #b91c1c; padding:.35rem .7rem; margin:.3rem 0;
         background:rgba(185,28,28,.05); border-radius:0 6px 6px 0; font-size:.88rem;}
.uav-ev.watch {border-left-color:#0e7490; background:rgba(14,116,144,.06);}
.uav-meta {opacity:.72; font-size:.85rem;}
.uav-sel {font-size:1.05rem; font-weight:650;}
.uav-count {text-align:center; font-size:.8rem; opacity:.85; margin-bottom:.4rem;}
.uav-count b {font-size:1.5rem; color:#b91c1c;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22); border-radius:8px;
                             padding:.5rem .8rem; min-width:0;}
div[data-testid="stMetricLabel"] p {font-size:.82rem;}
div[data-testid="stMetricValue"] {font-size:clamp(1.1rem, 1.6vw, 1.6rem);}
/* live mode refreshes every second; the toolbar's RUNNING badge would
   flash on every refresh, which reads as instability on a monitoring screen */
div[data-testid="stStatusWidget"] {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def _attack(t: str | None) -> str:
    return ATTACK_LABEL.get(t, t) if t else "—"


@st.cache_data
def _load_json(name: str) -> dict | None:
    p = ROOT / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------
# state
# ---------------------------------------------------------------------
def _init() -> None:
    ss = st.session_state
    ss.setdefault("state", LiveState())
    ss.setdefault("feed", None)
    ss.setdefault("source", "Recorded session")
    ss.setdefault("loaded", None)        # (session, fault, rate) for recorded mode
    ss.setdefault("last_error", None)


_init()
ss = st.session_state

# ---------------------------------------------------------------------
# sidebar -- source and controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data source")
    source = st.radio(
        "Source", ["Recorded session", "Live stream (Kafka)"],
        index=0 if ss.source == "Recorded session" else 1,
        label_visibility="collapsed",
        help="Recorded: replay a simulator session through the same pipeline "
             "code, no broker needed. Live: read the Kafka topics while the "
             "producer and detector run.")

    auto = False
    interval = 1.0
    if source == "Recorded session":
        session = st.selectbox("Attack session", list(C.ATTACK_SESSIONS),
                               format_func=lambda s: SESSION_LABEL.get(s, s),
                               help="A clean baseline recording followed by this "
                                    "attack session, as one stream.")
        fault = st.selectbox("Inject benign fault", list(FAULT_LABEL),
                             format_func=lambda f: FAULT_LABEL[f],
                             help="Corrupt a share of records with a realistic "
                                  "fault (not an attack) to see the data quality "
                                  "layer catch it. Same faults as "
                                  "produce.py --fault.")
        rate = 0.05
        if fault:
            rate = st.slider("Share of records", 0.01, 0.20, 0.05, 0.01,
                             format="%.2f",
                             help="Fraction of payload-carrying records that get "
                                  "the fault.")
        have_model = MODEL_PATH.exists()
        use_ml = st.toggle("Run ML stage", value=have_model, disabled=not have_model,
                           help="Score every drone the rules have not caught, per "
                                "15 s window, as detect.py --ml does live. Needs "
                                "models/ml.joblib (python train_ml.py).")
        if st.button("Load session", type="primary", use_container_width=True):
            with st.spinner("Replaying session through the pipeline..."):
                ss.state = build_offline(session, fault=fault, fault_rate=rate,
                                         ml=use_ml)
            ss.feed = None
            ss.source = "Recorded session"
            ss.loaded = (session, fault, rate)
            st.rerun()
    else:
        st.caption(f"Broker `{C.BOOTSTRAP}`")
        start = st.radio("Start from", ["New messages only", "Beginning of topics"],
                         help="New messages only: show what arrives after you "
                              "connect -- use this for a live demo, so results "
                              "of earlier runs don't reappear. Beginning: "
                              "replay everything already stored on the topics.")
        if st.button("Connect", type="primary", use_container_width=True):
            try:
                from feed import KafkaFeed
                ss.feed = KafkaFeed(start="latest" if start.startswith("New")
                                    else "earliest")
                ss.state = LiveState()
                ss.source = "Live stream (Kafka)"
                ss.loaded = None
                ss.last_error = None
            except Exception as e:                       # noqa: BLE001
                ss.feed = None
                st.error(f"Could not connect: {e}")
        auto = st.toggle("Auto-refresh", value=True,
                         help="Pull new messages and redraw every interval.")
        interval = st.select_slider("Refresh every", [0.5, 1.0, 2.0, 5.0],
                                    value=1.0, format_func=lambda v: f"{v:g} s")

    st.divider()
    with st.expander("How to read this"):
        st.markdown(f"""
Every message is checked in order by the stages in the strip at the top.

**Data quality** scores each record on five checks. Broken records are
quarantined. Poor data is not treated as an attack.

**Rules** are cheap, explainable checks, one per attack type. Each one names the
attack it catches:
- **Sybil**: {RULE_HELP['Sybil']}
- **Critical Node**: {RULE_HELP['Critical Node']}
- **Data manipulation**: {RULE_HELP['Data manipulation']}
- **Man-in-the-middle**: {RULE_HELP['Man-in-the-middle']}

**ML** scores only the drones that no rule has caught, once per
{C.WINDOW_SECONDS} s window. A Random Forest gives P(compromised) and names the
most likely attack type. An Isolation Forest, trained on clean traffic only,
flags unusual windows for the optional watch list.

**Response** (live, `respond.py`): rule alerts are isolated automatically after
a short countdown. Classifier alerts wait for the operator.

Hover any **(?)**, column header or chart for details.
""")
    with st.expander("Pipeline configuration"):
        st.markdown(
            f"- Rules: Sybil, Critical Node (streak >= {RL.DEFAULT_STREAK}), "
            f"data manipulation, MITM\n"
            f"- ML window: **{C.WINDOW_SECONDS} s**, only for drones no rule "
            f"has caught\n"
            f"- Kafka partitions: **{C.N_PARTITIONS}**, keyed by drone\n"
            f"- Topics: `{C.TOPIC_TELEMETRY}`, `{C.TOPIC_VERDICT}`, `uav.operator`, "
            f"`uav.action`")


# ---------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------
def _flow_strip(state: LiveState, live: bool) -> None:
    """The pipeline as five boxes, each saying what its stage did."""
    dqs = state.dq_summary()
    incs = list(state.incidents.values())
    rule_incs = [i for i in incs if any(v.track == "A" for v in i.signals)]
    ml_incs = [i for i in incs if i.severity == "alert"
               and not any(v.track == "A" for v in i.signals)]
    rule_types = sorted({_attack(i.attack_type) for i in rule_incs})
    isolated = sum(1 for i in incs if i.action == "ISOLATED")
    decided = sum(1 for i in incs if i.decision)

    def stage(name, value, detail, hit=False):
        return (f'<div class="uav-stage{" hit" if hit else ""}"><div class="n">'
                f'{html.escape(name)}</div><div class="v">{html.escape(value)}</div>'
                f'<div class="d">{html.escape(detail)}</div></div>')

    ingest = (f"{state.rate() or 0:,.0f} msg/s from {len(state.partitions)} partitions"
              if live else "replayed from the recording")
    gated = (f"{state.ml_gated:,} windows skipped (already caught by a rule)"
             if state.ml_gated else "windows of drones no rule caught")
    ml_detail = (f"{state.ml_windows:,} windows scored · {gated}"
                 if state.ml_windows else "off for this run")
    if live:
        resp = f"{isolated} isolated · {decided} decided"
    else:
        resp = "isolation runs in live mode (respond.py)"
    boxes = [
        stage("1 Ingest", f"{state.messages:,}", f"messages · {ingest}"),
        stage("2 Data quality", f"{dqs['flagged_pct']:.1f}% flagged",
              f"{dqs['checked']:,} records checked · {dqs['quarantined']:,} "
              f"quarantined"),
        stage("3 Rules", f"{len(rule_incs)} caught",
              ", ".join(rule_types) if rule_types else "no rule has fired",
              hit=bool(rule_incs)),
        stage("4 ML", f"{len(ml_incs)} caught", ml_detail, hit=bool(ml_incs)),
        stage("5 Response", f"{isolated} isolated" if live else "—", resp),
    ]
    arrow = '<div class="uav-arrow">→</div>'
    st.markdown(f'<div class="uav-flow">{arrow.join(boxes)}</div>',
                unsafe_allow_html=True)


def _status_word(inc) -> str:
    if inc.action == "ISOLATE_PENDING" and inc.due_at:
        return f"Isolating in {max(0.0, inc.due_at - time.time()):.0f} s"
    if inc.action == "ISOLATED":
        return "Isolated (auto)" if inc.action_by == "auto" else "Isolated"
    if inc.action == "RELEASED":
        return "Released"
    if inc.action == "CANCELLED" or inc.decision == "REJECT":
        return "Dismissed"
    if inc.action == "AWAITING_OPERATOR":
        return "Waiting for operator"
    return "Open"


def _verdict_mark(state: LiveState, drone: str) -> str | None:
    truth = state.is_attacker(drone)
    if truth is None:
        return None
    return "✓ attacker" if truth else "✗ false alarm"


def _decide(state: LiveState, inc, decision: str) -> None:
    state.decide(inc.iid, decision)
    if ss.feed is not None:
        err = ss.feed.publish_decision(inc)
        ss.last_error = (f"Decision not delivered to uav.operator: {err}"
                         if err else None)


def _incident_actions(state: LiveState, inc) -> None:
    k = inc.iid
    if inc.action == "ISOLATE_PENDING" and inc.due_at:
        left_s = max(0.0, inc.due_at - time.time())
        st.markdown(f'<div class="uav-count">Isolating in<br><b>{left_s:.0f} s</b>'
                    f'</div>', unsafe_allow_html=True)
        if st.button("Isolate now", key=f"i{k}", type="primary",
                     use_container_width=True):
            _decide(state, inc, "ACCEPT")
            st.rerun()
        if st.button("Hold", key=f"h{k}", use_container_width=True,
                     help="Cancel the automatic isolation; the alert stays."):
            _decide(state, inc, "REJECT")
            st.rerun()
        return
    if inc.action == "ISOLATED":
        if st.button("Release", key=f"l{k}", use_container_width=True,
                     help="Trust the drone again."):
            _decide(state, inc, "RELEASE")
            st.rerun()
        return
    if inc.action in ("CANCELLED", "RELEASED") or inc.decision == "REJECT":
        st.caption("Closed by the operator.")
        return
    if st.button("Isolate", key=f"i{k}", type="primary", use_container_width=True,
                 help="Stop trusting this drone. In live mode the decision goes "
                      "to uav.operator and the action to uav.action."):
        _decide(state, inc, "ACCEPT")
        st.rerun()
    if st.button("Dismiss", key=f"d{k}", use_container_width=True,
                 help="Mark as a false alarm."):
        _decide(state, inc, "REJECT")
        st.rerun()


def _evidence(inc) -> None:
    for v in inc.signals:
        css = "uav-ev" if v.track in ("A", "B-RF") else "uav-ev watch"
        st.markdown(f'<div class="{css}"><b>{html.escape(SIGNAL_LABEL[v.track])}'
                    f'</b> · t = {v.t_session} s<br>{html.escape(v.reason)}</div>',
                    unsafe_allow_html=True)


def _incident_table(state: LiveState, incs: list, key: str) -> None:
    gt = state.has_ground_truth
    rows = []
    for i in incs:
        first = i.signals[0].reason if i.signals else ""
        rows.append({
            "drone": i.drone,
            "attack": _attack(i.attack_type),
            "caught_by": i.caught_by,
            "t": i.alert_t,
            "delay": i.alert_delay_s,
            **({"truth": _verdict_mark(state, i.drone)} if gt else {}),
            "status": _status_word(i),
            "evidence": first.split(":", 1)[-1].strip(),
        })
    cfg = {
        "drone": st.column_config.TextColumn("Drone", width="small"),
        "attack": st.column_config.TextColumn(
            "Attack type", help="Named by the rule that fired, or the classifier's "
                                "most likely type when ML caught it."),
        "caught_by": st.column_config.TextColumn(
            "Caught by", help="Which layer of the cascade raised the alert, in the "
                              "order they fired. Rules go first; ML only scores "
                              "drones no rule has caught."),
        "t": st.column_config.NumberColumn("At t (s)"),
        "delay": st.column_config.NumberColumn(
            "Delay (s)", help="Seconds after the attack onset. The onset is "
                              "experiment metadata; no detector reads it."),
        "truth": st.column_config.TextColumn(
            "Ground truth", help="Recorded mode only: was this drone really the "
                                 "attacker? Never shown to the detectors."),
        "status": st.column_config.TextColumn("Status"),
        "evidence": st.column_config.TextColumn("Evidence", width="large"),
    }
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config=cfg,
                 height=min(38 + 35 * len(rows), 330))


# ---------------------------------------------------------------------
# main view
# ---------------------------------------------------------------------
def render() -> None:
    """Everything below the sidebar. In live mode this reruns on its own every
    `interval` seconds as a fragment, so only the content refreshes."""
    if ss.feed is not None:
        try:
            ss.feed.drain(ss.state)
        except Exception as e:                           # noqa: BLE001
            ss.last_error = f"Kafka read failed: {e}"
    state: LiveState = ss.state
    live = ss.source == "Live stream (Kafka)" and ss.feed is not None

    # header -------------------------------------------------------------
    pill = ('<span class="uav-pill live">LIVE</span>' if live
            else '<span class="uav-pill replay">RECORDED</span>')
    st.markdown(f'<div class="uav-head"><span class="uav-title">UAV Telemetry Trust '
                f'Pipeline</span>{pill}</div>', unsafe_allow_html=True)
    sub = []
    if ss.loaded:
        sess, flt, rt = ss.loaded
        sub.append(f"Session: {SESSION_LABEL.get(sess, sess)}")
        if flt:
            sub.append(f"fault: {FAULT_LABEL[flt].lower()} on {rt:.0%} of records")
    elif live:
        sub.append(f"Kafka {C.BOOTSTRAP}")
    sub.append(f"{sum(1 for d in state.drones.values() if d.n_target)} drones")
    if state.onset is not None:
        sub.append(f"attack onset t = {state.onset} s")
    sub.append(f"clock t = {state.clock} s")
    st.markdown(f'<div class="uav-sub">{html.escape(" · ".join(sub))}</div>',
                unsafe_allow_html=True)

    if ss.last_error:
        st.error(ss.last_error)

    if not state.messages:
        if live:
            st.info("Connected. Waiting for messages -- start the producer "
                    "(`python produce.py CriticalNode --speed 10`) and the detector "
                    "(`python detect.py --ml`).")
        else:
            st.info("Choose an attack session in the sidebar and press "
                    "**Load session**, or switch to the live stream.")
        _tab_evaluation(state, only_offline=True)
        return

    _flow_strip(state, live)

    tab_over, tab_drone, tab_dq, tab_eval = st.tabs(
        ["Overview", "Drone detail", "Data quality", "Evaluation"])
    with tab_over:
        _tab_overview(state, live)
    with tab_drone:
        _tab_drone(state)
    with tab_dq:
        _tab_dq(state)
    with tab_eval:
        _tab_evaluation(state)


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
def _tab_overview(state: LiveState, live: bool) -> None:
    incidents = sorted(state.incidents.values(), key=lambda i: (i.alert_t, i.drone))
    alerts = [i for i in incidents if i.severity == "alert"]
    watches = [i for i in incidents if i.severity == "watch"]

    st.markdown(f"#### Alerts · {len(alerts)}")
    if live:
        st.caption("Rule alerts are isolated automatically after a short grace "
                   "period unless you hold them; classifier alerts wait for you. "
                   "Requires respond.py to be running.")
    if not alerts:
        st.success("No drone flagged yet." if live
                   else "No drone flagged in this session.")
    else:
        _incident_table(state, alerts, key="alerts")
        left, right = st.columns([5, 1.3])
        names = [i.drone for i in alerts]
        prev = ss.get("alert_pick")
        with left:
            pick = st.selectbox("Selected alert", names,
                                index=names.index(prev) if prev in names else 0,
                                format_func=lambda n: f"{n} · "
                                f"{_attack(state.incidents[n].attack_type)} · "
                                f"caught by {state.incidents[n].caught_by}")
            ss.alert_pick = pick
            _evidence(state.incidents[pick])
        with right:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            _incident_actions(state, state.incidents[pick])

    if watches:
        show = st.toggle(f"Show anomaly watch list ({len(watches)})", value=False,
                         help="Drones only the Isolation Forest found unusual for "
                              "consecutive windows. Often a side effect of an "
                              "attack nearby, not proof of compromise, so nothing "
                              "is done automatically.")
        if show:
            _incident_table(state, watches, key="watches")

    st.markdown("#### Swarm")
    rows = state.swarm_table()
    if rows:
        df = pd.DataFrame(rows)
        short = {"Man-in-the-middle": "MITM", "Data manipulation": "Data manip."}
        df["detection"] = [
            (f"{short.get(a, a)} · {c}" if a else (c or ""))
            for a, c in zip(df["attack_type"], df["caught_by"])]
        df = df[["drone", "band", "detection", "trust", "p_compromised", "mean_dq"]]
        styled = df.style.map(
            lambda b: f"color: {STATUS_COLOR.get(b, '#64748b')}; font-weight: 600",
            subset=["band"])
        # not use_container_width: on some sessions the grid's auto column
        # widths and the container width chase each other (React error #185)
        st.dataframe(
            styled, hide_index=True, height=390,
            column_config={
                "drone": st.column_config.TextColumn("Drone"),
                "band": st.column_config.TextColumn("Status", help=STATUS_HELP),
                "detection": st.column_config.TextColumn(
                    "Detected as · by", width="medium",
                    help="Attack type named, and which layer caught it (Rule or ML)."),
                "trust": st.column_config.ProgressColumn(
                    "Trust", min_value=0.0, max_value=1.0, format="%.2f", width="medium",
                    help="Trust score (0 to 1, from the simulator's trust model) of "
                         "the drone's last answered exchange. Timeouts carry no "
                         "score."),
                "p_compromised": st.column_config.ProgressColumn(
                    "Classifier", min_value=0.0, max_value=1.0, format="%.2f", width="medium",
                    help="Random Forest P(compromised) for the drone's latest 15 s "
                         "window. Stops updating once a rule has caught the drone: "
                         "the ML stage only scores drones the rules have not."),
                "mean_dq": st.column_config.NumberColumn(
                    "Data quality", format="%.2f",
                    help="Mean data quality score of this drone's records."),
            })
        st.caption("Alerted drones first, then by trust, lowest first.")

    if live:
        with st.expander("Stream health"):
            p = st.columns(2)
            p[0].metric("Read rate", f"{state.rate() or 0:,.0f} msg/s",
                        help="Messages per second this dashboard is reading. This "
                             "is the replay speed, not the pipeline's capacity.")
            p[1].metric("Peak read rate", f"{state.peak_rate:,.0f} msg/s")
            if state.partitions:
                parts = pd.DataFrame(sorted(state.partitions.items()),
                                     columns=["partition", "messages"])
                parts["partition"] = parts["partition"].astype(str)
                st.altair_chart(alt.Chart(parts).mark_bar(color="#64748b").encode(
                    x=alt.X("partition:N", title="Kafka partition",
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("messages:Q", title="Messages"),
                ).properties(height=180), use_container_width=True)
                st.caption("Keyed by drone: every message about one drone lands on "
                           "the same partition and is processed in order.")


# ---------------------------------------------------------------------
# Drone detail
# ---------------------------------------------------------------------
def _tab_drone(state: LiveState) -> None:
    names = [r["drone"] for r in state.swarm_table()]
    if not names:
        st.caption("No drones yet.")
        return
    prev = ss.get("drone_pick")
    pick = st.selectbox("Drone", names,
                        index=names.index(prev) if prev in names else 0,
                        help="Ordered as in the swarm table: alerted drones first.")
    ss.drone_pick = pick
    d = state.drones[pick]
    inc = state.incidents.get(pick)
    alerted = inc is not None and inc.severity == "alert"

    m = st.columns([1, 1.6, 1.2, 0.9, 0.9])
    m[0].metric("Status", d.band, help=STATUS_HELP)
    m[1].metric("Attack type", _attack(inc.attack_type) if alerted else "—")
    m[2].metric("Caught by", inc.caught_by if inc else "—")
    m[3].metric("Last trust", f"{d.last_trust:.2f}" if d.last_trust is not None
                else "—")
    m[4].metric("Classifier", f"{d.ml_p:.2f}" if d.ml_p is not None else "—",
                help="Latest Random Forest P(compromised). Frozen once a rule "
                     "catches the drone.")

    if inc:
        _evidence(inc)

    if d.trust:
        df = pd.DataFrame({"t": list(d.times), "Trust score": list(d.trust)})
        if d.dq and len(d.dq) == len(d.trust):
            df["DQ score"] = list(d.dq)
        long = df.melt("t", var_name="series", value_name="value")
        if d.ml_hist:
            mlh = pd.DataFrame(list(d.ml_hist), columns=["t", "value"])
            mlh["series"] = "Classifier P(compromised)"
            long = pd.concat([long, mlh], ignore_index=True)
        # reports are sparse samples: the value holds until the next report
        lines = alt.Chart(long).mark_line(
            strokeWidth=1.8, interpolate="step-after",
            point=alt.OverlayMarkDef(filled=True, size=14)).encode(
            x=alt.X("t:Q", title="Session time (s)"),
            y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=list(SERIES_COLOR),
                                            range=list(SERIES_COLOR.values())),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("t:Q", title="t (s)"),
                     alt.Tooltip("series:N", title="Series"),
                     alt.Tooltip("value:Q", title="Value", format=".3f")])
        layers = [lines]
        marks = []
        if state.onset is not None:
            marks.append({"t": state.onset, "label": "attack onset",
                          "color": "#b91c1c"})
        if inc:
            marks.append({"t": inc.alert_t if alerted else inc.first_t,
                          "label": f"caught by {inc.caught_by}", "color": "#7c3aed"})
        if marks:
            mk = pd.DataFrame(marks)
            layers.append(alt.Chart(mk).mark_rule(strokeDash=[5, 4], strokeWidth=1.5)
                          .encode(x="t:Q", color=alt.Color("color:N", scale=None),
                                  tooltip=[alt.Tooltip("label:N", title="Marker"),
                                           alt.Tooltip("t:Q", title="t (s)")]))
        st.altair_chart(alt.layer(*layers).properties(height=320),
                        use_container_width=True)
        st.caption("Each dot is one answered exchange; the value holds until the "
                   "next. Red dashed line: attack onset (metadata, no detector reads "
                   "it). Purple dashed line: when the pipeline caught the drone.")
    else:
        st.caption("No trust samples for this drone yet.")

    with st.expander("Behaviour and data quality details"):
        b = st.columns(4)
        b[0].metric("Requests received", f"{d.n_target:,}")
        b[1].metric("Unanswered", f"{d.timeout_rate:.0%}")
        b[2].metric("Longest unanswered streak", d.max_streak,
                    help=f"The Critical Node rule needs >= {RL.DEFAULT_STREAK}.")
        b[3].metric("DQ flagged records", d.dq_flagged)
        if d.dq_issues:
            st.dataframe(pd.DataFrame(
                [{"dimension": k_, "records": v_, "what it checks": DIM_HELP[k_]}
                 for k_, v_ in d.dq_issues.most_common()]),
                hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------
def _tab_dq(state: LiveState) -> None:
    dqs = state.dq_summary()
    st.caption("Kept separate from trust on purpose: a broken sensor makes data "
               "poor without making the drone malicious. To see this layer work, "
               "load a session with a benign fault injected.")
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("##### Failures by dimension")
        dims = pd.DataFrame(dqs["dimensions"])
        dims["what_it_checks"] = dims["dimension"].map(DIM_HELP)
        dims["mean_score"] = dims["mean_score"].fillna(1.0)
        chart = alt.Chart(dims).mark_bar(color="#64748b", cornerRadiusEnd=3).encode(
            y=alt.Y("dimension:N", sort=DQ_DIMENSIONS, title=None),
            x=alt.X("records_failing:Q", title="Records failing"),
            tooltip=[alt.Tooltip("dimension:N", title="Dimension"),
                     alt.Tooltip("records_failing:Q", title="Records failing",
                                 format=","),
                     alt.Tooltip("failing_pct:Q", title="% of records", format=".2f"),
                     alt.Tooltip("what_it_checks:N", title="What it checks")],
        ).properties(height=220)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"{dqs['checked']:,} records checked, {dqs['flagged']:,} flagged "
                   f"({dqs['flagged_pct']:.1f}%), {dqs['quarantined']:,} "
                   f"quarantined. Thresholds were fitted on the clean baseline.")
    with right:
        st.markdown("##### Lowest data quality drones")
        dqt = [r for r in state.dq_table() if r["flagged"]][:10]
        if dqt:
            st.dataframe(
                pd.DataFrame(dqt)[["drone", "mean_dq", "flagged", "quarantined",
                                   "main_issue"]],
                hide_index=True, use_container_width=True, height=280,
                column_config={
                    "drone": "Drone",
                    "mean_dq": st.column_config.ProgressColumn(
                        "Mean DQ", min_value=0.0, max_value=1.0, format="%.3f"),
                    "flagged": st.column_config.NumberColumn("Flagged"),
                    "quarantined": st.column_config.NumberColumn(
                        "Quarantined", help=QUARANTINE_RULE),
                    "main_issue": st.column_config.TextColumn("Most common issue"),
                })
        else:
            st.caption("No drone has a flagged record.")

    st.markdown("##### Recent issues")
    issues = list(state.recent_dq_issues)[-10:][::-1]
    if issues:
        st.dataframe(
            pd.DataFrame(issues)[["t", "drone", "dimension", "quarantined", "rule"]],
            hide_index=True, use_container_width=True,
            column_config={
                "t": st.column_config.NumberColumn("t (s)", width="small"),
                "drone": st.column_config.TextColumn("Drone", width="small"),
                "dimension": "Dimension",
                "quarantined": st.column_config.CheckboxColumn(
                    "Quarantined", help=QUARANTINE_RULE),
                "rule": st.column_config.TextColumn("Check that failed",
                                                    width="large"),
            })
    else:
        st.caption("No data quality issues so far.")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def _this_run(state: LiveState) -> None:
    sc = state.run_scorecard()
    st.markdown("##### This run")
    if sc is None:
        st.caption("Live mode has no ground truth. Load a recorded session to see "
                   "each alert marked right or wrong.")
        return
    k = st.columns(4)
    n_att = len(sc["attacked"])
    k[0].metric("Attackers caught", f"{len(sc['caught'])} / {n_att}",
                help="Attacked drones (ground truth) with an alert from a rule or "
                     "the classifier.")
    k[1].metric("Attack type named right", f"{len(sc['named'])} / {n_att}")
    k[2].metric("False alerts", len(sc["false_alerts"]),
                help="Honest drones raised as alerts: "
                     + (", ".join(sc["false_alerts"]) or "none"))
    k[3].metric("False watch entries", len(sc["false_watches"]),
                help="Honest drones the anomaly detector put on the watch list: "
                     + (", ".join(sc["false_watches"]) or "none"))
    if sc["false_alerts"]:
        st.caption("False alerts in this run: " + ", ".join(sc["false_alerts"])
                   + ". See the Ground truth column on the Overview.")


def _confusion_chart(cm: dict) -> alt.Chart:
    labels = [_attack(x) if x != "normal" else "Normal" for x in cm["labels"]]
    rows = []
    for i, row in enumerate(cm["rows_true_cols_predicted"]):
        total = sum(row) or 1
        for j, n in enumerate(row):
            rows.append({"true": labels[i], "pred": labels[j], "n": n,
                         "share": n / total})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=labels, title="Predicted",
                axis=alt.Axis(labelAngle=-35, labelOverlap=False, labelLimit=140)),
        y=alt.Y("true:N", sort=labels, title="True",
                axis=alt.Axis(labelLimit=140)))
    heat = base.mark_rect().encode(
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        legend=None),
        tooltip=[alt.Tooltip("true:N", title="True"),
                 alt.Tooltip("pred:N", title="Predicted"),
                 alt.Tooltip("n:Q", title="Windows"),
                 alt.Tooltip("share:Q", title="Share of row", format=".1%")])
    text = base.mark_text(fontSize=12).encode(
        text="n:Q",
        color=alt.condition("datum.share > 0.5", alt.value("white"),
                            alt.value("#334155")))
    return (heat + text).properties(height=260)


def _tab_evaluation(state: LiveState, only_offline: bool = False) -> None:
    if not only_offline:
        _this_run(state)
        st.divider()

    res = _load_json("results_ml.json")
    casc = (res or {}).get("cascade")
    st.markdown("##### Offline results: rules vs ML vs cascade")
    if not casc:
        st.caption("results_ml.json has no cascade section. Run "
                   "`python train_ml.py` to produce it.")
        return
    known = casc["known_attack_types"]
    st.caption(f"{known['n_windows']:,} drone windows of {C.WINDOW_SECONDS} s from "
               f"all recorded sessions, blocked {known['folds']}-fold cross-validation "
               f"over time. The cascade sent "
               f"{known['ml_windows_scored_in_cascade']['skipped_by_rules']:,} windows "
               f"no further because a rule had already caught the drone.")
    table = []
    for m in ("rules", "ml", "cascade"):
        r = known[m]
        avn = r["attack_vs_normal"]
        table.append({"Method": METHOD_LABEL[m], "Accuracy": r["accuracy"],
                      "Macro F1 (attack type)": r["macro_f1"],
                      "Precision (attack)": avn["precision"],
                      "Recall (attack)": avn["recall"], "F1 (attack)": avn["f1"]})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                                for c in table[0] if c != "Method"})

    left, right = st.columns([1, 1])
    with left:
        m = st.radio("Confusion matrix", ["cascade", "rules", "ml"], horizontal=True,
                     format_func=lambda x: METHOD_LABEL[x].split(" (")[0])
        st.altair_chart(_confusion_chart(known[m]["confusion_matrix"]),
                        use_container_width=True)
        st.caption("Rows are the true class, columns what the method said. Counts "
                   "are windows; colour is the share of the row.")
    with right:
        st.markdown("Per attack type, F1")
        per = []
        for m_ in ("rules", "ml", "cascade"):
            for cls, v in known[m_]["per_class"].items():
                if cls == "normal":
                    continue
                per.append({"type": _attack(cls), "method": METHOD_LABEL[m_]
                            .split(" (")[0], "f1": v["f1"],
                            "support": v["support"]})
        per = pd.DataFrame(per)
        st.altair_chart(alt.Chart(per).mark_bar().encode(
            y=alt.Y("type:N", title=None, axis=alt.Axis(labelLimit=160)),
            yOffset=alt.YOffset("method:N",
                                sort=["Rules only", "ML only", "Cascade"]),
            x=alt.X("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("method:N", title=None,
                            scale=alt.Scale(domain=["Rules only", "ML only", "Cascade"],
                                            range=["#94a3b8", "#f59e0b", "#2563eb"]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["type", "method", alt.Tooltip("f1:Q", format=".3f"),
                     alt.Tooltip("support:Q", title="attacked windows")],
        ).properties(height=260), use_container_width=True)

    unseen = casc.get("unseen_attack_type") or {}
    if unseen:
        with st.expander("Unseen attack type: ML trained without the type it is "
                         "tested on"):
            rows = []
            for sess, by in unseen.items():
                rows.append({"Session": SESSION_LABEL.get(sess, sess),
                             **{METHOD_LABEL[m_].split(" (")[0]: by[m_]["f1"]
                                for m_ in ("rules", "ml", "cascade") if m_ in by}})
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(
                             f"{c} F1", format="%.2f") for c in df.columns
                             if c != "Session"})
            st.caption("The rules need no training, so a new attack type costs them "
                       "nothing; the classifier cannot name a type it has never seen.")
    caveats = casc.get("caveats") or []
    if caveats:
        with st.expander("Caveats"):
            st.markdown("\n".join(f"- {c}" for c in caveats))

    inj = _load_json("results_injection.json")
    plots = [ROOT / p for p in (inj or {}).get("plots", []) if (ROOT / p).exists()]
    if plots:
        st.divider()
        st.markdown("##### Injected attacks of increasing strength (held-out flight)")
        names = {p: p.stem.replace("injection_", "") for p in plots}
        pick = st.selectbox("Attack", plots, format_func=lambda p: _attack(names[p]))
        st.image(str(pick), use_column_width=True)
        st.caption(f"Attacks injected into a clean flight the models never trained on "
                   f"({inj.get('test_flight', 'held-out')}). Detection rate per "
                   f"method as the attack gets stronger.")


# ---------------------------------------------------------------------
# refresh: in live mode re-run only render() every `interval` seconds
# ---------------------------------------------------------------------
_live_now = ss.source == "Live stream (Kafka)" and ss.feed is not None
_every = interval if (_live_now and auto) else None
if hasattr(st, "fragment"):
    st.fragment(run_every=_every)(render)()
else:                                   # Streamlit < 1.37: reload the page
    render()
    if _every:
        time.sleep(_every)
        st.rerun()
