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

def compute_m_xi_spline(g: pd.DataFrame, s_factor: float = None):
    x = g["log_strain_rate"].values
    y = g["log_flow_stress"].values

    # Safety: ensure strictly increasing x (required for spline)
    if not np.all(np.diff(x) > 0):
        return None

    # Default smoothing if not provided
    # s ≈ N * variance is a reasonable heuristic
    if s_factor is None:
        s_factor = len(x) * np.var(y) * 0.01  # tune this

    spline = UnivariateSpline(x, y, s=s_factor)

    d1 = spline.derivative(1)(x)  # m
    d2 = spline.derivative(2)(x)  # d(m)/d(log epsdot)

    m = d1
    xi = m + d2

    return m, xi


def process_alloy(df: pd.DataFrame, alloy_name: str) -> pd.DataFrame:
    df = df.copy()

    df["log_strain_rate"] = np.log(df["strain_rate"])
    df["log_flow_stress"] = np.log(df["flow_stress"])

    processed_groups = []

    for (strain, temp), g in df.groupby(GROUP_KEYS):
        g = g.sort_values("strain_rate").reset_index(drop=True)

        if len(g) < 3:
            continue

        result = compute_m_xi_spline(g)

        if result is None:
            continue

        g["m"], g["xi"] = result

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