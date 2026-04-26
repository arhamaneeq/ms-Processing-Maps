from src.parser import parse

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata

# -------------------------------
# Utilities
# -------------------------------

def c_to_f(x):
    return x * 9/5 + 32

def f_to_c(x):
    return (x - 32) * 5/9


def clean_data(df):
    df = df.copy()

    # remove unphysical values
    df.loc[(df["m"] < 0) | (df["m"] > 1), ["eta", "xi"]] = np.nan
    df["eta"] = df["eta"].clip(0, 1)

    return df


def interpolate_grid(df, resolution=100):
    points = df[["temperature", "log_strain_rate"]].values

    T_vals = np.linspace(df["temperature"].min(), df["temperature"].max(), resolution)
    eps_vals = np.linspace(df["log_strain_rate"].min(), df["log_strain_rate"].max(), resolution)

    Tg, Eg = np.meshgrid(T_vals, eps_vals)

    Z_eta = griddata(points, df["eta"], (Tg, Eg), method="cubic")
    Z_xi  = griddata(points, df["xi"],  (Tg, Eg), method="cubic")

    return Tg, Eg, Z_eta, Z_xi


def plot_processing_map(Tg, Eg, Z_eta, Z_xi, title, save_path):
    plt.figure(figsize=(8, 8))

    # -------------------------------
    # η contour lines
    # -------------------------------
    cs = plt.contour(Tg, Eg, Z_eta, levels=15, colors='black')
    plt.clabel(cs, inline=True, fontsize=8)

    # -------------------------------
    # instability region (ξ < 0)
    # -------------------------------
    plt.contourf(
        Tg, Eg,
        Z_xi,
        levels=[-1e9, 0],
        colors='lightgray',
        alpha=0.5
    )

    # -------------------------------
    # labels & formatting
    # -------------------------------
    plt.xlabel("Temperature (°C)")
    plt.ylabel(r"log($\dot{\varepsilon}$)")
    plt.title(title)

    # dual axis (°F)
    secax = plt.gca().secondary_xaxis('top', functions=(c_to_f, f_to_c))
    secax.set_xlabel("Temperature (°F)")

    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------------
# Main pipeline
# -------------------------------

ALLOYS = parse()

for alloy, strategies in ALLOYS.items():
    print(f"\nProcessing alloy: {alloy}")

    for strategy, DF in strategies.items():
        print(f"  → Strategy: {strategy}")

        if DF.empty:
            continue

        for strain in sorted(DF["strain"].unique()):
            df = DF[DF["strain"] == strain]

            if len(df) < 5:
                continue

            df = clean_data(df)

            # drop rows with NaNs after cleaning
            df = df.dropna(subset=["eta", "xi"])

            if len(df) < 5:
                continue

            try:
                Tg, Eg, Z_eta, Z_xi = interpolate_grid(df)

                title = f"{alloy} | {strategy} | strain = {strain:.3f}"
                save_path = Path(f"images/{alloy}/{strategy}/strain_{strain:.3f}.png")

                plot_processing_map(Tg, Eg, Z_eta, Z_xi, title, save_path)

            except Exception as e:
                print(f"     (failed at strain {strain}: {e})")
                continue