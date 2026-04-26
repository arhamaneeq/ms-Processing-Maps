import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from src.math.derivatives import DerivativeStrategy
from src.math.derivatives import (
    FiniteDifferenceStrategy,
    Spline1DStrategy,
    SavitzkyGolayStrategy,
    Spline2DStrategy,
)

DIR = Path("data/")
GROUP_KEYS = ["strain", "temperature"]

STRATEGIES = {
    "finite_diff": lambda: FiniteDifferenceStrategy(),
    "spline_1d_s001": lambda: Spline1DStrategy(s_factor=0.01),
    "spline_1d_s01": lambda: Spline1DStrategy(s_factor=0.1),
    "savgol_w5_p2": lambda: SavitzkyGolayStrategy(window=5, poly=2),
    "spline_2d": lambda: Spline2DStrategy(s=0.1),
}

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

def process_alloy(df: pd.DataFrame, strategy: DerivativeStrategy):
    df = df.copy()

    df["log_strain_rate"] = np.log(df["strain_rate"])
    df["log_flow_stress"] = np.log(df["flow_stress"])

    # allow global fitting if needed
    if hasattr(strategy, "fit"):
        strategy.fit(df)

    processed = []

    for (strain, temp), g in df.groupby(["strain", "temperature"]):
        g = g.sort_values("strain_rate").reset_index(drop=True)

        if len(g) < 3:
            continue

        result = strategy.compute(g)
        if result is None:
            continue

        g["m"], g["xi"], g["eta"] = result
        processed.append(g)

    return pd.concat(processed, ignore_index=True) if processed else pd.DataFrame()

def parse():
    results = {}

    csv_files = [
        csv for csv in DIR.iterdir()
        if csv.suffix == ".csv" and csv.name != "meta.csv"
    ]

    for csv in csv_files:
        alloy_name = csv.stem
        print(f"\nProcessing alloy: {alloy_name}")

        df_raw = read_csv(csv)

        results[alloy_name] = {}

        for strat_name, strategy in STRATEGIES.items():
            strategy = strategy()

            print(f"  → Strategy: {strat_name}")

            try:
                df_processed = process_alloy(df_raw, strategy)

                # Skip empty outputs cleanly
                if df_processed.empty:
                    print("     (no valid groups)")
                    continue

                results[alloy_name][strat_name] = df_processed

            except Exception as e:
                print(f"     (failed: {e})")
                continue

    return results