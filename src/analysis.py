from src.parser import parse

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata

from src.math.interpolation import (
    LinearGridStrategy,
    SmoothedGridStrategy,
    RBFStrategy,
    GridSplineStrategy
)

INTERPOLATORS = {
    "linear": LinearGridStrategy(),
    "smooth": SmoothedGridStrategy(sigma=1.0),
    "spline_grid": GridSplineStrategy(s=0.001),
    # "rbf": RBFStrategy(smooth=0.5),  # keep off unless needed
}

# -------------------------------
# Utilities
# -------------------------------

def c_to_f(x):
    return x * 9/5 + 32

def f_to_c(x):
    return (x - 32) * 5/9


def clean_data(df):
    df = df.copy()

    # clip eta only (for plotting sanity)
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


def plot_processing_map(Tg, Eg, Z_eta, Z_xi, invalid_mask, title, save_path):
    plt.figure(figsize=(8, 8))

    # -------------------------------
    # η contour lines
    # -------------------------------
    cs = plt.contour(Tg, Eg, Z_eta, levels=10, colors='black')
    plt.clabel(cs, inline=True, fontsize=8)

    # -------------------------------
    # instability region (ξ < 0)
    # -------------------------------
    plt.contourf(
        Tg, Eg,
        Z_xi,
        levels=[-1e9, 0],
        colors='red',
        alpha=0.5
    )

    plt.contourf(
        Tg, Eg,
        invalid_mask,
        levels=[0.5, 1],
        colors='none',
        hatches=['....']
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
    for strategy_name, DF in strategies.items():
        for interp_name, interpolator in INTERPOLATORS.items():

            for strain in sorted(DF["strain"].unique()):
                df = DF[DF["strain"] == strain]

                df = clean_data(df)
                df = df.dropna(subset=["eta", "xi"])

                if len(df) < 5:
                    continue

                Tg, Eg, Z_eta, Z_xi, Z_m = interpolator.interpolate(df)

                Z_eta = 100 * Z_eta

                invalid_mask = (Z_m < 0) | (Z_m > 1)

                title = f"{alloy} | {strategy_name} | {interp_name} | strain={strain:.3f}"

                save_path = Path(
                    f"images/{alloy}/{strategy_name}/{interp_name}/strain_{strain:.3f}.png"
                )

                plot_processing_map(
                    Tg, Eg,
                    Z_eta, Z_xi,
                    invalid_mask,
                    title,
                    save_path
                )