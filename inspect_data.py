"""
inspect_data.py
===============
Reads Matej's pre-recorded session CSVs and prints everything we need
to know before writing the pipeline.

Run:  python inspect_data.py

Prints five sections:
  1. Which session files exist (is the NoAttack baseline there?)
  2. The exact column names and types
  3. Sample rows
  4. Whether timeout rows exist and what they look like
  5. THE BIG ONE: does the Critical Node target appear as a
     successful REQUESTER while only timing out as a RESPONDER?

Nothing here is clever. It's a fact-finding script.
"""

from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed. Run:  pip install pandas")


# Where Matej's sessions live after cloning his repo
SESSIONS_DIR = Path("bsc-reference/simulation/sessions")

# Column names we might be looking for. The CSV probably uses different
# ones -- this fuzzy matching just helps us guess so we can confirm.
LIKELY = {
    "requester":  ["request", "source", "src", "from", "evaluator", "observer"],
    "responder":  ["respond", "target", "dest", "dst", "to", "peer", "contact"],
    "timeout":    ["timeout", "timed_out", "time_out", "expired"],
    "attack":     ["attack"],
    "trust":      ["trust", "ts", "_cr", "_ba"],
    "metrics":    ["sm", "cdm", "bfm", "ri"],
    "link":       ["latency", "loss", "rtt", "packet"],
    "time":       ["time", "ts_", "timestamp", "t_"],
}


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def guess_columns(columns: list[str]) -> dict[str, list[str]]:
    """Fuzzy-match real column names against what we expect to find."""
    found: dict[str, list[str]] = {}
    lowered = {c: c.lower() for c in columns}
    for concept, needles in LIKELY.items():
        hits = [c for c, low in lowered.items()
                if any(n in low for n in needles)]
        if hits:
            found[concept] = hits
    return found


# ---------------------------------------------------------------------
# 1. What files do we have?
# ---------------------------------------------------------------------
rule("1. SESSION FILES")

if not SESSIONS_DIR.exists():
    sys.exit(f"Directory not found: {SESSIONS_DIR}\n"
             f"Did you clone the repo into 'bsc-reference'?")

csv_files = sorted(SESSIONS_DIR.glob("*.csv"))
if not csv_files:
    sys.exit(f"No CSV files in {SESSIONS_DIR}")

for f in csv_files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:>10,.0f} KB")

baseline = [f for f in csv_files
            if "noattack" in f.name.lower() or "baseline" in f.name.lower()]
print(f"\n  Total files: {len(csv_files)}")
print(f"  Baseline sessions found: {len(baseline)}")
for b in baseline:
    print(f"    - {b.name}")
if not baseline:
    print("    NONE  <-- PROBLEM")


# ---------------------------------------------------------------------
# 2. Columns
# ---------------------------------------------------------------------
rule("2. COLUMNS")

sample_file = csv_files[0]
df = pd.read_csv(sample_file)
print(f"  Reading: {sample_file.name}")
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

for i, col in enumerate(df.columns, 1):
    dtype = str(df[col].dtype)
    nulls = df[col].isna().sum()
    example = df[col].dropna().iloc[0] if df[col].notna().any() else "(all null)"
    example_str = str(example)[:30]
    print(f"  {i:>3}. {col:<32} {dtype:<10} nulls={nulls:<6} e.g. {example_str}")

print("\n  --- Fuzzy matches against what we're looking for ---")
for concept, hits in guess_columns(list(df.columns)).items():
    print(f"  {concept:<12} -> {hits}")


# ---------------------------------------------------------------------
# 3. Sample rows
# ---------------------------------------------------------------------
rule("3. SAMPLE ROWS")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.head(3).to_string())


# ---------------------------------------------------------------------
# 4. Timeout rows
# ---------------------------------------------------------------------
rule("4. TIMEOUT ROWS")

timeout_cols = guess_columns(list(df.columns)).get("timeout", [])
if timeout_cols:
    for col in timeout_cols:
        print(f"  Column '{col}' value counts:")
        print(df[col].value_counts(dropna=False).to_string())
        print()
else:
    print("  No obvious timeout column.")
    print("  Timeouts may instead appear as null/zero trust rows. Checking...")
    # Look for rows where trust-ish columns are null or zero
    trust_cols = guess_columns(list(df.columns)).get("trust", [])
    for col in trust_cols[:3]:
        n_null = df[col].isna().sum()
        n_zero = (df[col] == 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
        print(f"    {col}: {n_null} null, {n_zero} zero")


# ---------------------------------------------------------------------
# 5. THE CRITICAL NODE HYPOTHESIS
# ---------------------------------------------------------------------
rule("5. CRITICAL NODE HYPOTHESIS TEST")

cna_files = [f for f in csv_files
             if "critical" in f.name.lower() or "cna" in f.name.lower()]

if not cna_files:
    print("  No Critical Node session file found.")
    print(f"  Files available: {[f.name for f in csv_files]}")
else:
    cna = pd.read_csv(cna_files[0])
    print(f"  File: {cna_files[0].name}  ({len(cna):,} rows)\n")

    guesses = guess_columns(list(cna.columns))
    req_cols = guesses.get("requester", [])
    res_cols = guesses.get("responder", [])
    atk_cols = guesses.get("attack", [])

    print(f"  Candidate requester columns: {req_cols}")
    print(f"  Candidate responder columns: {res_cols}")
    print(f"  Candidate attack columns:    {atk_cols}\n")

    # Identify the attacked drone
    victim = None
    for col in atk_cols:
        vals = cna[col].dropna().unique()
        vals = [v for v in vals if str(v).lower() not in ("none", "nan", "", "false")]
        if len(vals) == 1:
            victim = vals[0]
            print(f"  Attack target (from '{col}'): {victim}")
            break
        elif len(vals) > 1:
            print(f"  '{col}' has multiple values: {vals[:10]}")

    if victim is None:
        print("  Could not auto-identify the attack target.")
        print("  Look at the attack column output above and tell me the value.")
    elif not req_cols or not res_cols:
        print("  Could not identify requester/responder columns.")
        print("  Look at the column list in section 2 and tell me which they are.")
    else:
        rq, rs = req_cols[0], res_cols[0]
        print(f"  Using '{rq}' as requester, '{rs}' as responder\n")

        as_requester = cna[cna[rq] == victim]
        as_responder = cna[cna[rs] == victim]

        print(f"  Rows where {victim} is the REQUESTER: {len(as_requester):,}")
        print(f"  Rows where {victim} is the RESPONDER: {len(as_responder):,}\n")

        if len(as_requester) > 0:
            print("  *** GOOD NEWS ***")
            print(f"  {victim} still initiates exchanges while under attack.")
            print("  That proves it is alive and in radio range.")
            print("  The requester/responder fallback rule WILL work.\n")
            print("  Sample rows where it is the requester:")
            print(as_requester.head(2).to_string())
        else:
            print("  *** PROBLEM ***")
            print(f"  {victim} never appears as a requester.")
            print("  The attack silences it completely, so we need plan C")
            print("  (derive presence from active_comms.json snapshots).")

print("\n" + "=" * 72)
print("  DONE -- paste this whole output back into the chat")
print("=" * 72)