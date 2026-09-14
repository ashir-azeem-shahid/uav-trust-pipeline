"""
app.py
======
Operator dashboard. Streamlit.

Two modes:
  Live    -- consumes uav.telemetry and uav.verdict from Kafka as they arrive
  Replay  -- rebuilds the same state from the CSVs with no broker, so the
             dashboard works for development and for reproducible figures

Four panels, matching the brief:
  1. Alert feed with ACCEPT / REJECT -- the operator loop
  2. Swarm state, drones ranked by suppression evidence
  3. Per-drone trust and data-quality trajectory, attack onset marked
  4. Data quality panel, kept visually separate from trust

The separation in 4 is deliberate and is part of the contribution: a
message can be poor quality without the drone being malicious, and the UI
should say so rather than blending them into one number.

Run:  streamlit run app.py
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import config as C
from feed import LiveState, build_offline

st.set_page_config(page_title="UAV Trust Pipeline", layout="wide",
                   page_icon="🛰️")

BAND_COLOR = {
    "FLAGGED":  "#dc2626",
    "LOW":      "#ea580c",
    "DEGRADED": "#ca8a04",
    "TRUSTED":  "#16a34a",
    "UNKNOWN":  "#6b7280",
}


# ---------------------------------------------------------------------
# state
# ---------------------------------------------------------------------
def _init():
    if "state" not in st.session_state:
        st.session_state.state = LiveState()
    if "feed" not in st.session_state:
        st.session_state.feed = None
    if "mode" not in st.session_state:
        st.session_state.mode = "Replay"


_init()

with st.sidebar:
    st.title("🛰️ Controls")
    mode = st.radio("Source", ["Replay", "Live (Kafka)"],
                    index=0 if st.session_state.mode == "Replay" else 1)

    if mode == "Replay":
        attack = st.selectbox("Session", list(C.ATTACK_SESSIONS))
        if st.button("Load session", type="primary", use_container_width=True):
            with st.spinner(f"replaying {attack}…"):
                st.session_state.state = build_offline(attack)
            st.session_state.mode = "Replay"
            st.rerun()
    else:
        st.caption(f"broker: `{C.BOOTSTRAP}`")
        if st.button("Connect", type="primary", use_container_width=True):
            try:
                from feed import KafkaFeed
                st.session_state.feed = KafkaFeed()
                st.session_state.state = LiveState()
                st.session_state.mode = "Live (Kafka)"
                st.success("connected")
            except Exception as e:                       # noqa: BLE001
                st.error(f"could not connect: {e}")
        auto = st.toggle("Auto-refresh", value=True)
        if st.session_state.feed is not None:
            got = st.session_state.feed.drain(st.session_state.state)
            st.caption(f"pulled {got} messages this refresh")

    st.divider()
    st.caption("Rule threshold")
    st.code(f"streak >= {__import__('rules').DEFAULT_STREAK}", language=None)

state: LiveState = st.session_state.state
summary = state.alert_summary()

# ---------------------------------------------------------------------
# header
# ---------------------------------------------------------------------
st.title("UAV Trust Pipeline — operator view")
sub = []
if state.attack_type:
    sub.append(f"**{state.attack_type}**")
if state.onset is not None:
    sub.append(f"attack onset at t={state.onset}s")
sub.append(f"clock t={state.clock}s")
sub.append(f"{state.messages:,} messages processed")
st.caption(" · ".join(sub))

k = st.columns(6)
k[0].metric("Alerts", summary["alerts"])
k[1].metric("Actionable", f"{summary['actionable_pct']:.0f}%",
            help="Verdicts carrying a human-readable reason. Track A always "
                 "does; a bare model score does not.")
k[2].metric("Accepted", summary["accepted"])
k[3].metric("Rejected", summary["rejected"])
k[4].metric("Pending", summary["pending"])
k[5].metric("Quarantined", summary["quarantined"],
            help="Messages failing a hard data quality rule.")

if not state.messages:
    st.info("Pick a session in the sidebar and press **Load session**, "
            "or connect to Kafka for live mode.")
    st.stop()

# ---------------------------------------------------------------------
# 1. alert feed + operator loop
# ---------------------------------------------------------------------
st.subheader("Alerts")

alerts = [v for v in state.verdicts if v.verdict != "QUARANTINE"]
if not alerts:
    st.success("No drones flagged. The rule stayed silent for this session.")
else:
    for v in reversed(alerts):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                head = f"### 🚨 {v.drone} — {v.verdict}"
                if v.decision == "ACCEPT":
                    head += "  ✅ accepted"
                elif v.decision == "REJECT":
                    head += "  ↩️ rejected"
                st.markdown(head)
                bits = [f"track **{v.track}**", f"t={v.t_session}s"]
                if v.detection_delay_s is not None:
                    bits.append(f"**{v.detection_delay_s}s after onset**")
                if v.partition is not None:
                    bits.append(f"partition {v.partition}")
                st.caption(" · ".join(bits))
                st.info(v.reason if v.reason
                        else "no reason available (model verdict)")
            with right:
                if v.decision is None:
                    if st.button("Accept", key=f"a{v.vid}",
                                 use_container_width=True, type="primary"):
                        state.decide(v.vid, "ACCEPT")
                        if st.session_state.feed:
                            st.session_state.feed.publish_decision(v)
                        st.rerun()
                    if st.button("Reject", key=f"r{v.vid}",
                                 use_container_width=True):
                        state.decide(v.vid, "REJECT")
                        if st.session_state.feed:
                            st.session_state.feed.publish_decision(v)
                        st.rerun()
                else:
                    st.caption(f"decided: {v.decision}")

# ---------------------------------------------------------------------
# 2 + 3. swarm state and trajectory
# ---------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Swarm state")
    st.caption("Ranked by longest run of unanswered requests — the "
               "suppression evidence.")
    rows = state.swarm_table()
    df = pd.DataFrame(rows)
    if not df.empty:
        def _style(r):
            return [f"background-color: {BAND_COLOR[r['band']]}22"] * len(r)
        st.dataframe(df.style.apply(_style, axis=1),
                     use_container_width=True, height=380, hide_index=True)

with right:
    st.subheader("Per-drone trajectory")
    names = [r["drone"] for r in rows]
    pick = st.selectbox("Drone", names, index=0 if names else None)
    if pick:
        d = state.drones[pick]
        chart = pd.DataFrame({
            "t": list(d.times),
            "trust score": list(d.trust),
        }).set_index("t")
        if d.dq and len(d.dq) == len(d.trust):
            chart["dq score"] = list(d.dq)
        st.line_chart(chart, height=300)
        if state.onset is not None:
            st.caption(f"Attack onset at t={state.onset}s. "
                       f"Everything left of it is clean traffic.")
        c = st.columns(4)
        c[0].metric("Current streak", d.streak)
        c[1].metric("Max streak", d.max_streak)
        c[2].metric("Timeout rate", f"{d.timeout_rate:.0%}")
        c[3].metric("Exchanges", d.n_target)

# ---------------------------------------------------------------------
# 4. data quality panel -- deliberately separate from trust
# ---------------------------------------------------------------------
st.subheader("Data quality")
st.caption("Kept apart from trust on purpose: a message can be poor quality "
           "without the drone being malicious, and the two catch different "
           "attacks.")

dq_rows = []
for name, d in state.drones.items():
    if d.dq:
        dq_rows.append({"drone": name,
                        "mean dq": round(sum(d.dq) / len(d.dq), 4),
                        "min dq": round(min(d.dq), 4),
                        "samples": len(d.dq)})
if dq_rows:
    dq_df = pd.DataFrame(dq_rows).sort_values("mean dq").head(15)
    q = st.columns([2, 1])
    q[0].dataframe(dq_df, use_container_width=True, hide_index=True)
    q[1].metric("Quarantined messages", summary["quarantined"])
    q[1].caption("Hard-rule failures routed out of the trust path.")
else:
    st.caption("No data quality samples yet.")

# ---------------------------------------------------------------------
# live auto-refresh
# ---------------------------------------------------------------------
if st.session_state.mode == "Live (Kafka)" and st.session_state.feed is not None:
    try:
        if auto:
            time.sleep(1.0)
            st.rerun()
    except NameError:
        pass
