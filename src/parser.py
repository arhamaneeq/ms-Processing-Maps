import pandas as pd
import numpy as np
from pathlib import Path

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

def compute_m(df: pd.DataFrame) -> np.ndarray:
    dlog_sigma = np.gradient(df["log_flow_stress"])
    dlog_epsdot = np.gradient(df["log_strain_rate"])
    return dlog_sigma / dlog_epsdot


def compute_xi(df: pd.DataFrame) -> np.ndarray:
    dm = np.gradient(df["m"])
    dlog_epsdot = np.gradient(df["log_strain_rate"])
    return df["m"] + (dm / dlog_epsdot)


def process_alloy(df: pd.DataFrame, alloy_name: str) -> pd.DataFrame:
    df = df.copy()

    df["log_strain_rate"] = np.log(df["strain_rate"])
    df["log_flow_stress"] = np.log(df["flow_stress"])

    processed_groups = []

    for (strain, temp), g in df.groupby(GROUP_KEYS):
        g = g.sort_values("strain_rate").reset_index(drop=True)

        if len(g) < 3:
            continue

        g["m"] = compute_m(g)
        g["xi"] = compute_xi(g)

        processed_groups.append(g)

    if not processed_groups:
        return pd.DataFrame()

    df_all = pd.concat(processed_groups, ignore_index=True)

    return df_all

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