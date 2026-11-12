"""
app.py
======
Operator dashboard for the UAV telemetry trust pipeline. Streamlit.

Two sources, one view:
  Live stream     consumes uav.telemetry and uav.verdict from Kafka as they
                  arrive -- the pipeline at runtime
  Recorded session rebuilds the same state from the recorded CSVs with no
                  broker, optionally with benign faults injected (faults.py),
                  for development and reproducible figures

Five tabs:
  Overview       alerts with the accept/reject operator loop, and the swarm
                 ranked by suppression evidence
  Data quality   the five DQ dimensions, quarantine, and recent issues --
                 deliberately separate from trust: a record can be poor
                 quality without the drone being malicious
  Drone detail   one drone's trust and DQ over time, attack onset marked
  Pipeline       how many records reached each stage, live throughput and
                 the spread across Kafka partitions
  Guide          how to read the dashboard

Every number has a hover explanation (the (?) icons, table headers and chart
tooltips), so the screen can be read without the thesis next to it.

Run:  streamlit run app.py
      (on a server: --server.address 0.0.0.0 --server.port 8501)
"""

from __future__ import annotations

import html
import time

import altair as alt
import pandas as pd
import streamlit as st

import config as C
import rules as RL
from feed import DQ_DIMENSIONS, LiveState, build_offline
from ml import MODEL_PATH

st.set_page_config(page_title="UAV Trust Pipeline", layout="wide")

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
STATUS_HELP = ("ISOLATED: no longer trusted -- isolated automatically or by "
               "an operator. FLAGGED: an attack alert from the rule or the "
               "classifier. WATCH: the anomaly detector finds its traffic unusual. "
               "TRUSTED / DEGRADED / LOW: the trust score of the drone's last "
               "answered exchange (>= 0.6, >= 0.3, below). UNKNOWN: no answered "
               "exchange yet.")
TRUST_NOTE = ("Timeouts are not scored here: the recording gives them a trust "
              "score of 0 because there is no data to assess. They count in "
              "the streak and the Unanswered share instead.")
STATUS_COLOR = {"ISOLATED": "#7c3aed", "FLAGGED": "#b91c1c", "WATCH": "#0e7490",
                "LOW": "#c2410c", "DEGRADED": "#a16207", "TRUSTED": "#15803d",
                "UNKNOWN": "#64748b"}
SIGNAL_LABEL = {"A": "Rule", "B-RF": "Classifier", "B-IF": "Anomaly"}
ML_HELP = ("Two models score every drone once per 15 s window, live. "
           "Classifier: a Random Forest's probability that the drone's telemetry "
           "is compromised (alert at 0.5). Anomaly: an Isolation Forest fitted on "
           "clean traffic only says whether the window looks unlike normal "
           "operation.")
SERIES_COLOR = {"Trust score": "#2563eb", "DQ score": "#94a3b8",
                "Classifier P(compromised)": "#dc2626"}
QUARANTINE_RULE = ("A record is quarantined -- removed from the trust path -- when "
                   "the mean of its five dimension scores falls below 0.5, or when "
                   "any trust metric is outside [0, 1]. Otherwise a failed rule "
                   "flags it and lowers its score, but it is kept.")

st.markdown("""
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 2rem;}
.uav-head {display:flex; align-items:center; gap:.7rem; flex-wrap:wrap;}
.uav-title {font-size:1.6rem; font-weight:650; letter-spacing:-.015em; line-height:1.2;}
.uav-pill {font-size:.7rem; font-weight:700; letter-spacing:.08em; padding:.12rem .55rem;
           border-radius:999px; border:1.5px solid currentColor;}
.uav-pill.live {color:#16a34a;} .uav-pill.replay {color:#64748b;}
.uav-sub {opacity:.72; font-size:.92rem; margin:.25rem 0 1rem 0;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22); border-radius:8px;
                             padding:.55rem .8rem; min-width:0;}
div[data-testid="stMetricLabel"] p {font-size:.82rem;}
div[data-testid="stMetricValue"] {font-size:clamp(1.15rem, 1.7vw, 1.75rem);}
.uav-reason {border-left:3px solid #b91c1c; padding:.45rem .75rem; margin:.35rem 0 .2rem 0;
             background:rgba(185,28,28,.06); border-radius:0 6px 6px 0; font-size:.92rem;}
.uav-meta {opacity:.72; font-size:.85rem;}
.uav-alert-title {font-size:1.05rem; font-weight:650;}
.uav-tag {font-size:.72rem; font-weight:600; margin-left:.6rem; padding:.1rem .5rem;
          border-radius:4px; background:rgba(128,128,128,.16); vertical-align:middle;}
.uav-reason.watch {border-left-color:#0e7490; background:rgba(14,116,144,.07);}
.uav-count {text-align:center; font-size:.8rem; opacity:.85; margin-bottom:.4rem;}
.uav-count b {font-size:1.6rem; color:#b91c1c;}
.uav-status {text-align:center; font-weight:650; padding:.5rem 0;}
.uav-status span {font-weight:400; font-size:.8rem; opacity:.75;}
.uav-status.iso {color:#7c3aed;}
/* live mode refreshes every second; the toolbar's RUNNING/Stop badge would
   flash on every refresh, which reads as instability on a monitoring screen */
div[data-testid="stStatusWidget"] {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


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
        session = st.selectbox("Session", list(C.ATTACK_SESSIONS),
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
        use_ml = st.toggle("Run ML models", value=have_model, disabled=not have_model,
                           help="Score every drone's 15 s windows with the "
                                "classifier and the anomaly detector, as "
                                "detect.py --ml does live. Needs models/ml.joblib "
                                "(python train_ml.py).")
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

    with st.expander("Pipeline configuration"):
        st.markdown(
            f"- Suppression rule: streak **>= {RL.DEFAULT_STREAK}** consecutive "
            f"unanswered requests, while the drone is still completing its own\n"
            f"- Window: **{C.WINDOW_SECONDS} s**\n"
            f"- Kafka partitions: **{C.N_PARTITIONS}**, keyed by drone\n"
            f"- ML: classifier + anomaly detector per drone per window "
            f"(`detect.py --ml`)\n"
            f"- Autonomous response: rule alerts isolated after a grace period "
            f"(`respond.py`)\n"
            f"- Topics: `{C.TOPIC_TELEMETRY}`, `{C.TOPIC_VERDICT}`, `uav.operator`, "
            f"`uav.action`")

def render() -> None:
    """Everything below the sidebar. In live mode this reruns on its own every
    `interval` seconds as a fragment, so only the content refreshes -- the page
    does not reload and the sidebar keeps its state."""
    if ss.feed is not None:
        try:
            ss.feed.drain(ss.state)
        except Exception as e:                           # noqa: BLE001
            ss.last_error = f"Kafka read failed: {e}"
    state: LiveState = ss.state
    live = ss.source == "Live stream (Kafka)" and ss.feed is not None
    summary = state.alert_summary()
    dqs = state.dq_summary()

    # ---------------------------------------------------------------------
    # header
    # ---------------------------------------------------------------------
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
    if state.onset is not None:
        sub.append(f"attack onset t = {state.onset} s")
    sub.append(f"clock t = {state.clock} s")
    sub.append(f"{state.messages:,} messages")
    st.markdown(f'<div class="uav-sub">{html.escape(" · ".join(sub))}</div>',
                unsafe_allow_html=True)

    if ss.last_error:
        st.error(ss.last_error)

    if not state.messages:
        if live:
            st.info("Connected. Waiting for messages -- start the producer "
                    "(`python produce.py CriticalNode --speed 10`) and the detector "
                    "(`python detect.py`).")
        else:
            st.info("Choose a session in the sidebar and press **Load session**, or "
                    "switch to the live stream.")
        return

    # ---------------------------------------------------------------------
    # KPI row
    # ---------------------------------------------------------------------
    k = st.columns(6)
    k[0].metric("Messages", f"{state.messages:,}",
                help="Messages processed. Each exchange between two drones arrives "
                     "twice, once keyed by each drone it concerns, so all evidence "
                     "about a drone lands on one Kafka partition.")
    k[1].metric("Drones", sum(1 for d in state.drones.values() if d.n_target),
                help="Drones that have been asked for telemetry at least once.")
    k[2].metric("Alerts", summary["alerts"],
                help="Drones with an attack alert from the rule (stopped answering "
                     "while still active) or the classifier. Every alert states its "
                     "evidence in words.")
    k[3].metric("Watch", summary["watch"],
                help="Drones only the anomaly detector flagged: unusual, not "
                     "necessarily compromised. Nothing is done automatically.")
    k[4].metric("DQ flagged", f"{dqs['flagged_pct']:.1f}%",
                help="Share of records where at least one data quality rule failed. "
                     "Flagged records are kept but scored down.")
    k[5].metric("Quarantined", f"{dqs['quarantined']:,}",
                help=QUARANTINE_RULE)

    tab_over, tab_dq, tab_drone, tab_pipe, tab_guide = st.tabs(
        ["Overview", "Data quality", "Drone detail", "Pipeline", "Guide"])


    # ---------------------------------------------------------------------
    # Overview
    # ---------------------------------------------------------------------
    def _decide(inc, decision: str) -> None:
        state.decide(inc.iid, decision)
        if ss.feed is not None:
            err = ss.feed.publish_decision(inc)
            ss.last_error = (f"Decision not delivered to uav.operator: {err}"
                             if err else None)

    def _incident_card(inc) -> None:
        alert = inc.severity == "alert"
        css = "uav-reason" if alert else "uav-reason watch"
        with st.container(border=True):
            left, right = st.columns([5, 1.25])
            with left:
                kinds = " + ".join(SIGNAL_LABEL[t] for t in dict.fromkeys(inc.tracks))
                head = "attack alert" if alert else "watch"
                st.markdown(
                    f'<div class="uav-alert-title">{html.escape(inc.drone)} — {head}'
                    f'<span class="uav-tag">{html.escape(kinds)}</span></div>',
                    unsafe_allow_html=True)
                meta = [f"first signal at t = {inc.first_t} s"]
                if inc.first_delay_s is not None:
                    meta.append(f"{inc.first_delay_s} s after attack onset")
                if inc.partition is not None:
                    meta.append(f"Kafka partition {inc.partition}")
                st.markdown(f'<div class="uav-meta">{html.escape(" · ".join(meta))}'
                            f'</div>', unsafe_allow_html=True)
                for v in inc.signals:
                    st.markdown(
                        f'<div class="{css}"><b>{html.escape(SIGNAL_LABEL[v.track])}'
                        f'</b> · t = {v.t_session} s<br>{html.escape(v.reason)}</div>',
                        unsafe_allow_html=True)
            with right:
                _incident_controls(inc)

    def _incident_controls(inc) -> None:
        k = inc.iid
        if inc.action == "ISOLATE_PENDING" and inc.due_at:
            left_s = max(0.0, inc.due_at - time.time())
            st.markdown(f'<div class="uav-count">Isolating in<br>'
                        f'<b>{left_s:.0f} s</b></div>', unsafe_allow_html=True)
            if st.button("Isolate now", key=f"i{k}", type="primary",
                         use_container_width=True,
                         help="Act immediately instead of waiting."):
                _decide(inc, "ACCEPT")
                st.rerun()
            if st.button("Hold", key=f"h{k}", use_container_width=True,
                         help="Cancel the automatic isolation: the alert stays, "
                              "nothing is done to the drone."):
                _decide(inc, "REJECT")
                st.rerun()
            return
        if inc.action == "ISOLATED":
            by = "automatically" if inc.action_by == "auto" else "by operator"
            took = (f"<br>{inc.seconds_to_action:.0f} s after the alert"
                    if inc.seconds_to_action is not None else "")
            st.markdown(f'<div class="uav-status iso">ISOLATED<br>'
                        f'<span>{by}{took}</span></div>', unsafe_allow_html=True)
            if st.button("Release", key=f"l{k}", use_container_width=True,
                         help="Trust the drone again."):
                _decide(inc, "RELEASE")
                st.rerun()
            return
        if inc.action in ("CANCELLED", "RELEASED") or inc.decision == "REJECT":
            word = {"RELEASED": "Released"}.get(inc.action, "Dismissed")
            st.markdown(f'<div class="uav-status">{word}<br><span>by operator</span>'
                        f'</div>', unsafe_allow_html=True)
            return
        if inc.action == "AWAITING_OPERATOR":
            st.caption("Waiting for you: classifier alerts are never acted on "
                       "automatically.")
        if st.button("Isolate", key=f"i{k}", type="primary", use_container_width=True,
                     help="Stop trusting this drone's telemetry. In live mode the "
                          "decision goes to uav.operator and the action to uav.action."):
            _decide(inc, "ACCEPT")
            st.rerun()
        if st.button("Dismiss", key=f"d{k}", use_container_width=True,
                     help="Mark as a false alarm."):
            _decide(inc, "REJECT")
            st.rerun()

    with tab_over:
        incidents = state.incident_list()
        alerts_ = [i for i in incidents if i.severity == "alert"]
        watches = [i for i in incidents if i.severity == "watch"]
        st.markdown("#### Alerts")
        if live:
            st.caption("Autonomous mode: rule alerts are isolated automatically "
                       "after a short grace period unless you hold them; classifier "
                       "alerts wait for you. Requires respond.py to be running.")
        if not alerts_ and live:
            st.success("No alerts yet. The rule raises one when a drone leaves "
                       f"{RL.DEFAULT_STREAK} requests in a row unanswered while still "
                       "completing its own; the classifier when a drone's 15 s window "
                       "looks compromised.")
        elif not alerts_:
            st.success("No drone flagged in this session.")
        for inc in alerts_:
            _incident_card(inc)
        if watches:
            with st.expander(f"Watch list · {len(watches)} drone"
                             f"{'s' if len(watches) != 1 else ''} behaving unusually",
                             expanded=False):
                st.caption("Raised by the anomaly detector alone: these drones' "
                           "traffic looked unlike clean operation for consecutive "
                           "windows. That is often a side effect of an attack nearby "
                           "-- not proof the drone is compromised -- so nothing is "
                           "done automatically.")
                for inc in watches:
                    _incident_card(inc)

        st.markdown("#### Swarm")
        st.caption("Ranked by each drone's longest run of unanswered requests -- the "
                   "evidence the suppression rule uses. The Classifier and Anomaly "
                   "columns are the two ML models' latest reading for each drone.")
        rows = state.swarm_table()
        if rows:
            df = pd.DataFrame(rows)
            df["timeout_pct"] = (df["timeout_rate"] * 100).round(1)
            df = df[["drone", "band", "max_streak", "streak", "timeout_pct",
                     "trust", "p_compromised", "anomaly", "exchanges"]]
            styled = df.style.map(
                lambda b: f"color: {STATUS_COLOR.get(b, '#64748b')}; font-weight: 600",
                subset=["band"])
            st.dataframe(
                styled, hide_index=True, use_container_width=True, height=360,
                column_config={
                    "drone": st.column_config.TextColumn("Drone"),
                    "band": st.column_config.TextColumn("Status", help=STATUS_HELP),
                    "max_streak": st.column_config.NumberColumn(
                        "Longest streak",
                        help=f"Longest run of consecutive unanswered requests to this "
                             f"drone. The rule needs >= {RL.DEFAULT_STREAK}, while the "
                             f"drone is still completing its own requests."),
                    "streak": st.column_config.NumberColumn(
                        "Current streak", help="Unanswered requests in a row right now. "
                                               "Any answer resets it to 0."),
                    "timeout_pct": st.column_config.ProgressColumn(
                        "Unanswered", min_value=0, max_value=100, format="%.0f%%",
                        help="Share of all requests to this drone that got no answer. "
                             "Honest drones also miss many, while out of range -- "
                             "which is why the rule uses the unbroken run instead."),
                    "trust": st.column_config.ProgressColumn(
                        "Trust", min_value=0.0, max_value=1.0, format="%.2f",
                        help="Trust score (0 to 1, from the simulator's trust "
                             "model) of the drone's last answered exchange. "
                             + TRUST_NOTE),
                    "p_compromised": st.column_config.ProgressColumn(
                        "Classifier", min_value=0.0, max_value=1.0, format="%.2f",
                        help="Random Forest: probability that this drone's telemetry "
                             "in its latest 15 s window is compromised. Alert at "
                             "0.5. Empty until the first window closes, or when ML "
                             "is off."),
                    "anomaly": st.column_config.TextColumn(
                        "Anomaly", help="Isolation Forest, fitted on clean traffic "
                                        "only: does the latest window look unlike "
                                        "normal operation?"),
                    "exchanges": st.column_config.NumberColumn(
                        "Requests received", help="Times other drones asked this drone "
                                                  "for telemetry."),
                })


    # ---------------------------------------------------------------------
    # Data quality
    # ---------------------------------------------------------------------
    with tab_dq:
        st.caption("Kept separate from trust on purpose: a broken sensor makes data "
                   "poor without making the drone malicious, and the two must not be "
                   "blended into one number.")
        c = st.columns(3)
        c[0].metric("Records checked", f"{dqs['checked']:,}",
                    help="One check per exchange, on the copy about the drone whose "
                         "telemetry is being evaluated.")
        c[1].metric("Flagged", f"{dqs['flagged']:,}  ({dqs['flagged_pct']:.1f}%)",
                    help="At least one rule failed. Kept, but scored down.")
        c[2].metric("Quarantined", f"{dqs['quarantined']:,}", help=QUARANTINE_RULE)

        left, right = st.columns([1.1, 1])
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
                         alt.Tooltip("mean_score:Q", title="Mean score", format=".4f"),
                         alt.Tooltip("what_it_checks:N", title="What it checks")],
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)
            st.caption("Hover a bar for what the dimension checks. Thresholds were "
                       "fitted on the clean baseline, not chosen by hand.")
        with right:
            st.markdown("##### Recent issues")
            issues = list(state.recent_dq_issues)[-12:][::-1]
            if issues:
                st.dataframe(
                    pd.DataFrame(issues)[["t", "drone", "dimension", "quarantined", "rule"]],
                    hide_index=True, use_container_width=True, height=250,
                    column_config={
                        "t": st.column_config.NumberColumn("t (s)"),
                        "drone": "Drone",
                        "dimension": "Dimension",
                        "quarantined": st.column_config.CheckboxColumn(
                            "Quarantined", help=QUARANTINE_RULE),
                        "rule": st.column_config.TextColumn(
                            "Rule that fired", width="large"),
                    })
            else:
                st.caption("No data quality issues so far.")

        st.markdown("##### Drones by data quality")
        dqt = state.dq_table()
        if dqt:
            st.dataframe(
                pd.DataFrame(dqt).head(15), hide_index=True, use_container_width=True,
                column_config={
                    "drone": "Drone",
                    "mean_dq": st.column_config.ProgressColumn(
                        "Mean DQ score", min_value=0.0, max_value=1.0, format="%.3f",
                        help="Average of this drone's record scores; each score is the "
                             "mean of the five dimensions."),
                    "min_dq": st.column_config.NumberColumn("Lowest score", format="%.3f"),
                    "flagged": st.column_config.NumberColumn(
                        "Flagged", help="Records with at least one failed rule."),
                    "quarantined": st.column_config.NumberColumn(
                        "Quarantined", help=QUARANTINE_RULE),
                    "main_issue": st.column_config.TextColumn(
                        "Most common issue", help="The dimension that failed most often "
                                                  "for this drone."),
                    "records": st.column_config.NumberColumn("Records"),
                })


    # ---------------------------------------------------------------------
    # Drone detail
    # ---------------------------------------------------------------------
    with tab_drone:
        names = [r["drone"] for r in state.swarm_table()]
        if not names:
            st.caption("No drones yet.")
        else:
            # Keep the operator's choice across live refreshes: the list grows and
            # re-orders as data arrives, which would otherwise reset the widget.
            prev = ss.get("drone_pick")
            pick = st.selectbox("Drone", names,
                                index=names.index(prev) if prev in names else 0,
                                help="Ordered as in the swarm table: strongest "
                                     "suppression evidence first.")
            ss.drone_pick = pick
            d = state.drones[pick]
            m = st.columns(5)
            m[0].metric("Status", d.band, help=STATUS_HELP)
            m[1].metric("Longest streak", d.max_streak,
                        help=f"Rule threshold is {RL.DEFAULT_STREAK}.")
            m[2].metric("Unanswered", f"{d.timeout_rate:.0%}",
                        help="Share of requests to this drone with no answer.")
            m[3].metric("Classifier",
                        f"{d.ml_p:.2f}" if d.ml_p is not None else "—",
                        help="Latest P(compromised) from the Random Forest. "
                             + ML_HELP)
            m[4].metric("DQ flagged", d.dq_flagged,
                        help="This drone's records with at least one failed DQ rule.")

            if d.trust:
                df = pd.DataFrame({"t": list(d.times), "Trust score": list(d.trust)})
                if d.dq and len(d.dq) == len(d.trust):
                    df["DQ score"] = list(d.dq)
                long = df.melt("t", var_name="series", value_name="value")
                if d.ml_hist:
                    mlh = pd.DataFrame(list(d.ml_hist), columns=["t", "value"])
                    mlh["series"] = "Classifier P(compromised)"
                    long = pd.concat([long, mlh], ignore_index=True)
                # Reports are sparse samples: the value holds until the next report,
                # so draw steps with a dot on each real sample, not straight ramps
                # that would invent a gradual change between two distant reports.
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
                if state.onset is not None:
                    on = pd.DataFrame({"t": [state.onset], "label": ["attack onset"]})
                    rule = alt.Chart(on).mark_rule(color="#b91c1c", strokeDash=[5, 4],
                                                   strokeWidth=1.5).encode(
                        x="t:Q", tooltip=[alt.Tooltip("label:N", title="Marker"),
                                          alt.Tooltip("t:Q", title="t (s)")])
                    layers += [rule]
                    st.markdown(f'<div class="uav-meta"><span style="color:#b91c1c">'
                                f'┊ attack onset</span> at t = {state.onset} s</div>',
                                unsafe_allow_html=True)
                st.altair_chart(alt.layer(*layers).properties(height=320),
                                use_container_width=True)
                st.caption("Trust score from the simulator's trust model; DQ score is "
                           "the mean of the five data quality dimensions. Each dot is "
                           "one answered exchange, and the value holds until the next; "
                           "timeouts carry no score and are not plotted. The red line "
                           "is the classifier's P(compromised), one value per 15 s "
                           "window. Hover a dot for its value. The attack onset is "
                           "experiment metadata shown for reference -- no detector "
                           "reads it.")
            else:
                st.caption("No trust samples for this drone yet.")

            if d.dq_issues:
                st.markdown("##### Data quality issues for this drone")
                st.dataframe(pd.DataFrame(
                    [{"dimension": k_, "records": v_, "what it checks": DIM_HELP[k_]}
                     for k_, v_ in d.dq_issues.most_common()]),
                    hide_index=True, use_container_width=True)


    # ---------------------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------------------
    with tab_pipe:
        st.caption("Producer → Kafka (uav.telemetry, keyed by drone) → detector "
                   "(data quality → suppression rule → ML per 15 s window) → "
                   "uav.verdict → responder (uav.action), this dashboard and the S3 "
                   "archive. Every stage runs as the data arrives.")
        p = st.columns(4)
        r = state.rate() or 0.0
        p[0].metric("Throughput", f"{r:,.0f} msg/s" if live else "—",
                    help="Messages per second this dashboard is reading, over the last "
                         "few seconds, and the highest rate seen since connecting. "
                         "Live mode only. This is the replay speed, not a limit: the "
                         "pipeline's measured capacity is far higher (see the "
                         "scalability experiments).")
        if live and state.peak_rate:
            p[0].caption(f"Peak read rate since connecting: "
                       f"{state.peak_rate:,.0f} msg/s.")
        p[1].metric("Partitions active", len(state.partitions) if live else "—",
                    help="Kafka partitions this dashboard has received telemetry from. "
                         "Live mode only.")
        p[2].metric("ML windows scored", f"{state.ml_windows:,}",
                    help="Drone windows scored by both ML models. " + ML_HELP)
        n_auto = sum(1 for a in state.actions
                     if a.get("action") == "ISOLATED" and a.get("by") == "auto")
        p[3].metric("Decisions", summary["decided"],
                    help="Incidents isolated or dismissed, by the responder or an "
                         "operator. Operator decisions go to uav.operator; actions "
                         "to uav.action.")
        if n_auto:
            p[3].caption(f"{n_auto} taken automatically.")

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("##### Records at each stage")
            stages = pd.DataFrame([
                {"stage": "1 Ingested", "records": state.messages,
                 "meaning": "Every message read from Kafka (two per exchange)."},
                {"stage": "2 DQ checked", "records": state.dq_checked,
                 "meaning": "One data quality check per exchange."},
                {"stage": "3 DQ flagged", "records": state.dq_flagged,
                 "meaning": "At least one rule failed: kept, scored down."},
                {"stage": "4 Quarantined", "records": state.dq_quarantined,
                 "meaning": "Removed from the trust path as unusable."},
                {"stage": "5 Attack alerts", "records": summary["alerts"],
                 "meaning": "Drones flagged by the suppression rule."},
            ])
            base = alt.Chart(stages).encode(
                y=alt.Y("stage:N", sort=None, title=None,
                        axis=alt.Axis(ticks=False, domain=False, labelPadding=6)),
                # headroom so the count label after the longest bar is not cut off
                x=alt.X("records:Q", title=None, axis=None,
                        scale=alt.Scale(type="symlog", nice=False,
                                        domain=[0, max(4, 4 * int(stages.records.max()))])),
                tooltip=[alt.Tooltip("stage:N", title="Stage"),
                         alt.Tooltip("records:Q", title="Records", format=","),
                         alt.Tooltip("meaning:N", title="Meaning")])
            bars = base.mark_bar(color="#2563eb", cornerRadiusEnd=3, height=22)
            labels = base.mark_text(align="left", dx=5, fontSize=12,
                                    color="#94a3b8").encode(
                text=alt.Text("records:Q", format=","))
            st.altair_chart((bars + labels).properties(height=210)
                            .configure_view(stroke=None), use_container_width=True)
            st.caption("Bar length is on a log scale so small stages stay visible; "
                       "the number is the exact count.")
        with right:
            st.markdown("##### Messages per partition")
            if live and state.partitions:
                parts = pd.DataFrame(sorted(state.partitions.items()),
                                     columns=["partition", "messages"])
                parts["partition"] = parts["partition"].astype(str)
                st.altair_chart(alt.Chart(parts).mark_bar(color="#64748b").encode(
                    x=alt.X("partition:N", title="Partition", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("messages:Q", title="Messages"),
                    tooltip=[alt.Tooltip("partition:N", title="Partition"),
                             alt.Tooltip("messages:Q", title="Messages", format=",")],
                ).properties(height=230), use_container_width=True)
                st.caption("Messages are keyed by drone, so every message about one "
                           "drone lands on the same partition and is processed by the "
                           "same consumer, in order.")
            else:
                st.caption("Available in live mode: shows how the stream spreads across "
                           "the topic's partitions.")


    # ---------------------------------------------------------------------
    # Guide
    # ---------------------------------------------------------------------
    with tab_guide:
        st.markdown(f"""
    **What this is.** The operator view of a streaming pipeline that evaluates the
    trustworthiness of drone telemetry as it arrives. Drones ask each other for
    telemetry; every exchange flows through Kafka, is checked for data quality, and
    is evaluated for signs of attack, message by message.

    **Alerts (Overview).** Three independent detectors, grouped into one card
    per drone:
    - **Rule** -- flags a drone that has left **{RL.DEFAULT_STREAK} or more**
      requests in a row unanswered *while still completing requests of its own*.
      A drone out of range can't do the second part, so this separates refusing
      to answer from being too far away.
    - **Classifier** -- a Random Forest scores every drone's 15 s window; an
      alert when P(compromised) reaches 0.5.
    - **Anomaly** -- an Isolation Forest fitted on clean traffic only; a drone
      unlike normal operation for consecutive windows goes on the *watch list*.

    Every alert states its evidence in words.

    **Autonomous response (live).** `respond.py` isolates a drone with a rule
    alert automatically after a short countdown -- the rule had no false alarms
    in any recorded session. During the countdown you can **Isolate now** or
    **Hold**; afterwards **Release**. Classifier alerts wait for you; the watch
    list is never acted on. Every action is published to `uav.action`.

    **Swarm (Overview).** Every drone, ranked by its longest unbroken run of
    unanswered requests. Honest drones miss requests too, but their silence ends;
    the attacked drone's does not.

    **Data quality.** Five checks on every record -- completeness, validity,
    timeliness, uniqueness, consistency -- each scored 0 to 1. A failed check
    *flags* a record; a record too broken to use is *quarantined*. Data quality is
    reported separately from trust because a faulty sensor is not an attacker. To
    see it work, load a recorded session with a fault injected, or run the
    producer with `--fault`.

    **Drone detail.** One drone's trust, data quality and classifier score over
    time, with the attack onset marked.

    **Pipeline.** How many records reached each stage, and in live mode the read
    throughput and the spread across Kafka partitions.

    Hover any **(?)**, table header or chart element for details.
    """)


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
