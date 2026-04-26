from src.parser import parse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

ALLOYS = parse()

for ALLOY, DFS in ALLOYS.items():
    for STRATEGY, DF in DFS.items():
        print(f"{ALLOY} {STRATEGY}")

        for strain in DF["strain"].unique():
            df = DF[DF["strain"] == strain]

            pivot_eta = df.pivot_table(
                index="temperature",
                columns="log_strain_rate",
                values="eta"
            )

            pivot_xi = df.pivot_table(
                index="temperature",
                columns="log_strain_rate",
                values="xi"
            )

            # sort axes (important)
            pivot_eta = pivot_eta.sort_index().sort_index(axis=1)
            pivot_xi  = pivot_xi.sort_index().sort_index(axis=1)

            # interpolate missing values (you WILL have NaNs)
            pivot_eta = pivot_eta.interpolate(axis=1).interpolate(axis=0)
            pivot_xi  = pivot_xi.interpolate(axis=1).interpolate(axis=0)

            x_vals = pivot_eta.columns.values
            y_vals = pivot_eta.index.values

            X, Y = np.meshgrid(x_vals, y_vals)

            plt.figure(figsize=(8,8))

            cp = plt.contourf(X, Y, pivot_eta.values, levels=20)
            plt.colorbar(cp, label="η")

            plt.contour(X, Y, pivot_xi.values, levels=[0], colors="red")

            plt.xlabel(r"$\log(\dot{\varepsilon})$")
            plt.ylabel("Temperature (°C)")
            plt.title(f"{ALLOY} | {STRATEGY} | strain={strain}")

            path = Path(f"images/{ALLOY}/{STRATEGY}")
            path.mkdir(parents=True, exist_ok=True)

            plt.savefig(path / f"{strain}.png")
            plt.close()
