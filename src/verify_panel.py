"""Spot-check data/processed/bank_quarter_panel.csv against the raw quarterly files.

For each raw file, draws random (bank, column) cells, looks up the same bank,
quarter and column in the panel, and checks the values agree. Raw values are
read as text so the comparison is against exactly what the FDIC file contains.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PANEL = ROOT / "data" / "processed" / "bank_quarter_panel.csv"

CELLS_PER_FILE = 50
SEED = 2026


# Raw text that pandas reads as missing (e.g. OCCDISTDESC is sometimes the literal "null")
MISSING_TEXT = {"", "null", "NULL", "NA", "N/A", "NaN", "nan", "None"}


def values_match(raw, panel) -> bool:
    raw_missing = pd.isna(raw) or str(raw).strip() in MISSING_TEXT
    if raw_missing or pd.isna(panel):
        return raw_missing and pd.isna(panel)
    try:
        return float(raw) == float(panel)
    except (TypeError, ValueError):
        return str(raw) == str(panel)


def main():
    rng = np.random.default_rng(SEED)

    print(f"Loading {PANEL.relative_to(ROOT)} ...")
    panel = pd.read_csv(PANEL, low_memory=False, float_precision="round_trip").set_index(["CERT", "YEAR", "QUARTER"])

    checks = []
    raw_rows = 0
    for path in sorted(RAW.glob("*.csv")):
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw_rows += len(raw)
        repdte = pd.to_datetime(raw["REPDTE"].iloc[0], format="%Y%m%d")
        cols = [c for c in raw.columns if c != "CERT"]

        rows = rng.integers(0, len(raw), CELLS_PER_FILE)
        picked = rng.choice(cols, CELLS_PER_FILE)
        for r, col in zip(rows, picked):
            cert = int(raw.at[r, "CERT"])
            key = (cert, repdte.year, repdte.quarter)
            raw_val = raw.at[r, col]
            panel_val = panel.at[key, col] if key in panel.index else "<row missing>"
            checks.append({
                "file": path.name, "CERT": cert, "YEAR": repdte.year,
                "QUARTER": repdte.quarter, "column": col,
                "raw_value": raw_val, "panel_value": panel_val,
                "match": values_match(raw_val, panel_val),
            })

    results = pd.DataFrame(checks)
    out = ROOT / "output" / "panel_verification.csv"
    results.to_csv(out, index=False)

    n, ok = len(results), results["match"].sum()
    print(f"\nRow count: raw files {raw_rows:,} vs panel {len(panel):,} "
          f"-> {'MATCH' if raw_rows == len(panel) else 'MISMATCH'}")
    print(f"Cells checked: {n:,} across {results['file'].nunique()} files "
          f"and {results['column'].nunique()} distinct columns")
    print(f"Matches: {ok:,} / {n:,} ({ok / n:.2%})")

    print("\nRandom sample of checked cells:")
    # astype(str) shows floats at full precision instead of pandas' 6-decimal display
    with pd.option_context("display.width", 200, "display.max_colwidth", 30):
        print(results.sample(20, random_state=SEED).astype(str).to_string(index=False))

    bad = results[~results["match"]]
    if len(bad):
        print(f"\n{len(bad)} MISMATCHES (first 50):")
        print(bad.head(50).astype(str).to_string(index=False))
    print(f"\nFull results saved to {out.relative_to(ROOT)}")
    sys.exit(1 if len(bad) or raw_rows != len(panel) else 0)


if __name__ == "__main__":
    main()
