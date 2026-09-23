"""Merge the quarterly FDIC financial files in data/raw into one bank-quarter panel.

Output: data/processed/bank_quarter_panel.csv, sorted by CERT then time, with
columns CERT, YEAR, QUARTER, TIME followed by every column from the raw files.
TIME counts quarters since 1960Q1 (Stata %tq convention: 2008Q1 = 192).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT_FILE = PROCESSED / "bank_quarter_panel.csv"

ID_COLS = ["CERT", "YEAR", "QUARTER", "TIME"]


def load_quarter(path: Path) -> pd.DataFrame:
    # round_trip parsing keeps every digit; the default parser can drift in the last few
    df = pd.read_csv(path, low_memory=False, float_precision="round_trip")
    repdte = pd.to_datetime(df["REPDTE"].astype(str), format="%Y%m%d")
    if repdte.nunique() != 1:
        raise ValueError(f"{path.name}: expected one report date, found {repdte.nunique()}")
    ids = pd.DataFrame({"YEAR": repdte.dt.year, "QUARTER": repdte.dt.quarter})
    ids["TIME"] = (ids["YEAR"] - 1960) * 4 + (ids["QUARTER"] - 1)
    return pd.concat([df, ids], axis=1)


def main():
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {RAW}")

    frames = []
    columns = []  # union of raw columns, in first-seen order
    for path in files:
        df = load_quarter(path)
        columns += [c for c in df.columns if c not in columns and c not in ID_COLS]
        frames.append(df)
        print(f"{path.name}: {len(df):>6} banks")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[ID_COLS + columns]
    panel = panel.sort_values(["CERT", "TIME"], kind="stable").reset_index(drop=True)

    dups = panel.duplicated(["CERT", "TIME"]).sum()
    if dups:
        raise ValueError(f"{dups} duplicate CERT-quarter rows")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_FILE, index=False)
    print(
        f"\nWrote {OUT_FILE.relative_to(ROOT)}: {len(panel):,} rows x {panel.shape[1]} columns, "
        f"{panel['CERT'].nunique():,} banks, "
        f"{panel['YEAR'].min()}Q{panel.loc[panel['TIME'].idxmin(), 'QUARTER']} to "
        f"{panel['YEAR'].max()}Q{panel.loc[panel['TIME'].idxmax(), 'QUARTER']}"
    )


if __name__ == "__main__":
    main()
