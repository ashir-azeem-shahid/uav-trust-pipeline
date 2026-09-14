"""
analysis3.py
============
The go/no-go test, done properly.

inspect2.py asked "is the victim contacted less often?" -- wrong question.
The Critical Node victim is BY DEFINITION the hub, so it is contacted MORE
than anyone. What the data actually shows is that Exchange_Type has a
'timeout' value, and every exchange with the victim is a timeout.

So the real cross-role rule is:

    victim succeeds as REQUESTER   (alive, in range, radio works)
    victim only ever TIMES OUT as RESPONDER   (receiving, not answering)

This script measures whether that rule separates the victim from all 52
other drones, and -- critically -- whether it produces false positives on
the no-attack baseline, where timeouts also occur from genuine range loss.
"""

from pathlib import Path
import pandas as pd

SESSIONS = Path("bsc-reference/simulation/sessions")
REQ, RES, EX = "Requester", "ContactedDrone", "Exchange_Type"
ATK_TGT, LABEL, TIME = "Attack_Target", "Trust_Label", "Timestamp"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def rule(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


files = sorted(SESSIONS.glob("*.csv"))
frames = {f.name: pd.read_csv(f) for f in files}


# ------------------------------------------------------------------ 1
rule("1. WHAT 'timeout' ROWS LOOK LIKE")

cna = frames["CriticalNode_20260513_233052.csv"]
to = cna[cna[EX] == "timeout"]
ok = cna[cna[EX] != "timeout"]
print(f"  timeout rows: {len(to):,}   non-timeout rows: {len(ok):,}")
print(f"\n  On timeout rows:")
print(f"    DT_Speed_mps null : {to['DT_Speed_mps'].isna().sum():,} of {len(to):,}")
print(f"    Trust_Score  == 0 : {(to['Trust_Score'] == 0).sum():,} of {len(to):,}")
print(f"    Trust_RI     mean : {to['Trust_RI'].mean():.4f}")
print(f"    Trust_Label       : {to[LABEL].value_counts().to_dict()}")
print(f"\n  On non-timeout rows:")
print(f"    Trust_Score  == 0 : {(ok['Trust_Score'] == 0).sum():,} of {len(ok):,}")
print(f"    Trust_Score  mean : {ok['Trust_Score'].mean():.4f}")
print(f"    Trust_Label       : {ok[LABEL].value_counts().to_dict()}")
print("\n  -> a timeout row carries NO declared payload and zeroed trust.")
print("     It is the 'absence' record. This is the signal.")


# ------------------------------------------------------------------ 2
rule("2. THE CROSS-ROLE TEST, PER SESSION")

def cross_role(df, name):
    victim = df[ATK_TGT].dropna().unique()
    victim = victim[0] if len(victim) else None

    rows = []
    for drone in sorted(set(df[REQ]) | set(df[RES])):
        as_req = df[df[REQ] == drone]
        as_res = df[df[RES] == drone]
        req_ok = (as_req[EX] != "timeout").sum()
        res_n = len(as_res)
        res_to = (as_res[EX] == "timeout").sum()
        rows.append({
            "drone": drone,
            "req_success": req_ok,
            "res_total": res_n,
            "res_timeout": res_to,
            "res_timeout_pct": (res_to / res_n * 100) if res_n else float("nan"),
            "is_victim": drone == victim,
        })
    return pd.DataFrame(rows), victim


for name in files:
    df = frames[name.name]
    tab, victim = cross_role(df, name.name)
    tab = tab.sort_values("res_timeout_pct", ascending=False)

    print(f"\n  {name.name}")
    print(f"    victim: {victim}")

    # the rule: succeeds as requester AND 100% timeout as responder
    fires = tab[(tab.req_success > 0) & (tab.res_timeout_pct == 100)]
    print(f"    drones where rule fires (req_success>0 AND res_timeout==100%): "
          f"{len(fires)}  -> {list(fires.drone)}")

    print(f"    top 5 by responder-timeout %:")
    print(tab.head(5).to_string(index=False, float_format=lambda x: f"{x:6.1f}"))

    if victim:
        v = tab[tab.drone == victim].iloc[0]
        others = tab[~tab.is_victim]["res_timeout_pct"].dropna()
        print(f"    VICTIM {victim}: req_success={v.req_success}, "
              f"res_timeout={v.res_timeout}/{v.res_total} = {v.res_timeout_pct:.1f}%")
        print(f"    all other drones: mean {others.mean():.1f}%, "
              f"median {others.median():.1f}%, max {others.max():.1f}%")


# ------------------------------------------------------------------ 3
rule("3. FALSE-POSITIVE CHECK ON THE BASELINES")

for name in ["20260510_123058_Baseline.csv", "baseline.csv"]:
    df = frames[name]
    tab, _ = cross_role(df, name)
    fires = tab[(tab.req_success > 0) & (tab.res_timeout_pct == 100)]
    print(f"\n  {name}  ({len(df):,} rows, "
          f"{(df[EX]=='timeout').sum():,} timeouts)")
    print(f"    drones where the rule fires: {len(fires)} -> {list(fires.drone)}")
    print(f"    responder-timeout % : mean {tab.res_timeout_pct.mean():.1f}, "
          f"max {tab.res_timeout_pct.max():.1f}")
    print("    -> every drone firing here is a FALSE POSITIVE (no attack ran)")


# ------------------------------------------------------------------ 4
rule("4. IS Trust_Label GROUND TRUTH?")

for name in files:
    df = frames[name.name]
    victim = df[ATK_TGT].dropna().unique()
    victim = victim[0] if len(victim) else None
    lab = df[LABEL].value_counts().to_dict()
    print(f"\n  {name.name}")
    print(f"    Trust_Label counts : {lab}")
    if victim:
        is_v = df[RES] == victim
        mal = df[LABEL] == "MALICIOUS"
        print(f"    rows where ContactedDrone == victim : {is_v.sum():,}")
        print(f"    of those, labelled MALICIOUS        : {(is_v & mal).sum():,}")
        print(f"    rows NOT involving victim as target : {(~is_v).sum():,}")
        print(f"    of those, labelled MALICIOUS        : {(~is_v & mal).sum():,}"
              f"   <-- label disagrees with ground truth")
    else:
        print(f"    NO ATTACK RAN, yet MALICIOUS rows   : {lab.get('MALICIOUS', 0):,}"
              f"   <-- label is threshold-derived, not ground truth")


# ------------------------------------------------------------------ 5
rule("5. UNIQUENESS KEY")

for name in ["20260510_123058_Baseline.csv", "CriticalNode_20260513_233052.csv"]:
    df = frames[name]
    print(f"\n  {name}")
    for key in [[TIME, REQ, RES], [TIME, REQ, RES, EX]]:
        d = df.duplicated(subset=key).sum()
        print(f"    dupes on {key}: {d:,}")
    # what do the duplicates look like?
    dupes = df[df.duplicated(subset=[TIME, REQ, RES], keep=False)]
    if len(dupes):
        g = dupes.groupby([TIME, REQ, RES]).size().sort_values(ascending=False)
        print(f"    worst repeated triple appears {g.iloc[0]} times")
        print(f"    Exchange_Type mix within duplicate groups: "
              f"{dupes[EX].value_counts().to_dict()}")


# ------------------------------------------------------------------ 6
rule("6. ATTACK ONSET (for detection-delay measurement)")

for name in files:
    df = frames[name.name]
    victim = df[ATK_TGT].dropna().unique()
    if not len(victim):
        continue
    victim = victim[0]
    as_res = df[df[RES] == victim].copy()
    to_rows = as_res[as_res[EX] == "timeout"]
    ok_rows = as_res[as_res[EX] != "timeout"]
    print(f"\n  {name.name}  victim {victim}")
    print(f"    session span                     : {df[TIME].min()} -> {df[TIME].max()}")
    print(f"    victim's LAST successful response: "
          f"{ok_rows[TIME].max() if len(ok_rows) else '(none in session)'}")
    print(f"    victim's FIRST timeout as target : "
          f"{to_rows[TIME].min() if len(to_rows) else '(none)'}")
    print(f"    -> onset proxy = first timeout; usable for detection delay")
