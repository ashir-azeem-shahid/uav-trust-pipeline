"""
config.py
=========
Every column name, constant and path in one place.

Why this file exists: the CSV column names were discovered by inspection,
not documented. If a name is wrong it should be wrong in exactly one place.
Nothing in this project hard-codes a column name anywhere else.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Where the data is
# ---------------------------------------------------------------------
SESSIONS_DIR = Path("bsc-reference/simulation/sessions")

# The baseline used to fit Track A thresholds.
#
# IMPORTANT: this is baseline.csv, NOT 20260510_123058_Baseline.csv.
# The latter contains ZERO timeout rows, so thresholds fitted on it would
# treat any timeout at all as an attack. baseline.csv has 581 timeouts and
# a realistic spread (one drone reaches 57% timeout rate as target), which
# is what "normal" actually looks like.
BASELINE = SESSIONS_DIR / "baseline.csv"

# The second baseline, held back as an independent test set.
BASELINE_HELD_OUT = SESSIONS_DIR / "20260510_123058_Baseline.csv"

ATTACK_SESSIONS = {
    "CriticalNode":       SESSIONS_DIR / "CriticalNode_20260513_233052.csv",
    "MITM":               SESSIONS_DIR / "MITM_20260518_170536.csv",
    "DataManipulation":   SESSIONS_DIR / "DataManipulation_20260513_234805.csv",
    "DataManipulation2":  SESSIONS_DIR / "DataManipulation_20260513_235829.csv",
    "Sybil":              SESSIONS_DIR / "Sybil_54.csv",
}

# ---------------------------------------------------------------------
# Column names (verified against all 7 files)
# ---------------------------------------------------------------------
C_TIME      = "Timestamp"
C_REQ       = "Requester"          # the drone doing the evaluating
C_RES       = "ContactedDrone"     # the drone being evaluated
C_SWARM     = "TargetSwarm"
C_EXCHANGE  = "Exchange_Type"      # 'intra' | 'inter' | 'timeout'
C_DISTANCE  = "Distance_m"
C_ATK_TYPE  = "Attack_Type"
C_ATK_TGT   = "Attack_Target"
C_LABEL     = "Trust_Label"        # NOT ground truth -- see labels.py

# The declared Digital Twin payload. All null on timeout rows.
DT_COLS = [
    "DT_Pos_X", "DT_Pos_Y", "DT_Pos_Z",
    "DT_NextWP1", "DT_NextWP2",
    "DT_Speed_mps",
    "DT_Vel_VX", "DT_Vel_VY", "DT_Vel_VZ",
    "DT_Heading_deg", "DT_YawRate_rps",
    "DT_NearestNeighbor_m", "DT_RelSpeedNearest_mps",
    "DT_RotorAvgSpeed",
]

# Drone-computed trust metrics. We ingest and validate these; we do not
# recompute them (the CSV holds outputs, not the raw inputs they need).
TRUST_COLS = [
    "Trust_RI", "Trust_BFM", "Trust_SM", "Trust_CDM",
    "Trust_BA", "Trust_CR", "Trust_Score",
]

# ---------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------
TIMEOUT = "timeout"                # the Exchange_Type value meaning "no answer"

# Two different rules, easy to conflate -- keep them apart.
#
# GROUND_TRUTH_FIELDS never appear at a message's top level. They travel in
# the nested '_gt' block so the operational code physically cannot reach
# them by accident.
#
# Attack_Target is the obvious one. Trust_Label is the subtle one and just
# as dangerous: it is a threshold cut on Trust_Score, so including it leaks
# the answer through the back door and yields a meaningless near-perfect F1.
GROUND_TRUTH_FIELDS = {C_ATK_TGT, C_ATK_TYPE, C_LABEL, "gt_malicious", "malicious"}

# FORBIDDEN_FEATURES never appear in a feature vector. Superset of the
# above, plus routing metadata: 'role' and 'about' are needed by the rule
# but would let a model key off the message plumbing instead of the signal.
FORBIDDEN_FEATURES = GROUND_TRUTH_FIELDS | {"role", "about", "_gt", "seq", "onset"}

# Both asserted by tests/test_leakage.py.

# ---------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------
BOOTSTRAP = "localhost:19092"      # matches docker-compose.yml's external listener
TOPIC_TELEMETRY = "uav.telemetry"
TOPIC_VERDICT   = "uav.verdict"
N_PARTITIONS    = 4

# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------
WINDOW_SECONDS = 15               # matches Fecak's 15s drone-window aggregation
