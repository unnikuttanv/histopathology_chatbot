"""Loads the take-home dataset once at import time."""
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "dataset.csv"

df = pd.read_csv(CSV_PATH)


def cancers() -> list[str]:
    return sorted(df["cancer_indication"].unique().tolist())
