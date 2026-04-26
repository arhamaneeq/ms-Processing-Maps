import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import UnivariateSpline

DIR = Path("data/")
GROUP_KEYS = ["strain", "temperature"]

def read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[()]", "", regex=True)
    )

    df = df.rename(columns={
        "strain_rate_/s": "strain_rate",
        "temperature_deg_c": "temperature",
        "flow_stress_mpa": "flow_stress"
    })

    return df

def process_alloy(df: pd.DataFrame, strategy) -> pd.DataFrame:
    df = df.copy()

    df["log_strain_rate"] = np.log(df["strain_rate"])
    df["log_flow_stress"] = np.log(df["flow_stress"])

    processed = []

    for (strain, temp), g in df.groupby(GROUP_KEYS):
        g = g.sort_values("strain_rate").reset_index(drop=True)

        if len(g) < 3:
            continue

        result = strategy.compute(g)
        if result is None:
            continue

        g["m"], g["xi"] = result
        processed.append(g)

    return pd.concat(processed, ignore_index=True) if processed else pd.DataFrame()

def parse():
    DF_ALLOYS = {}

    csv_files = [
        csv for csv in DIR.iterdir()
        if csv.suffix == ".csv" and csv.name != "meta.csv"
    ]

    for csv in csv_files:
        alloy_name = csv.stem
        print(f"\nProcessing: {alloy_name}")

        df = read_csv(csv)
        # print(df.head())

        DF_ALLOYS[alloy_name] = process_alloy(df, alloy_name)

    return DF_ALLOYS