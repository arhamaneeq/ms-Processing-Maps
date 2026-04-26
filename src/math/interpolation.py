import numpy as np
from scipy.interpolate import griddata, Rbf, RectBivariateSpline
from scipy.ndimage import gaussian_filter


class InterpolationStrategy:
    def interpolate(self, df, resolution=100):
        raise NotImplementedError


# -------------------------------
# Utility
# -------------------------------

def build_grid(df, resolution):
    T = np.linspace(df["temperature"].min(), df["temperature"].max(), resolution)
    E = np.linspace(df["log_strain_rate"].min(), df["log_strain_rate"].max(), resolution)
    return np.meshgrid(T, E)


def safe_gaussian(Z, sigma):
    """Gaussian filter that ignores NaNs"""
    mask = np.isnan(Z)
    Z_filled = np.nan_to_num(Z, nan=0.0)

    Z_smooth = gaussian_filter(Z_filled, sigma=sigma)

    # reapply mask
    Z_smooth[mask] = np.nan
    return Z_smooth


# -------------------------------
# Linear grid interpolation
# -------------------------------

class LinearGridStrategy(InterpolationStrategy):
    def interpolate(self, df, resolution=100):
        points = df[["temperature", "log_strain_rate"]].values

        Tg, Eg = build_grid(df, resolution)

        Z_eta = griddata(points, df["eta"], (Tg, Eg), method="linear")
        Z_xi  = griddata(points, df["xi"],  (Tg, Eg), method="linear")
        Z_m   = griddata(points, df["m"],   (Tg, Eg), method="linear")

        return Tg, Eg, Z_eta, Z_xi, Z_m


# -------------------------------
# Linear + smoothing
# -------------------------------

class SmoothedGridStrategy(InterpolationStrategy):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def interpolate(self, df, resolution=100):
        points = df[["temperature", "log_strain_rate"]].values

        Tg, Eg = build_grid(df, resolution)

        Z_eta = griddata(points, df["eta"], (Tg, Eg), method="linear")
        Z_xi  = griddata(points, df["xi"],  (Tg, Eg), method="linear")
        Z_m   = griddata(points, df["m"],   (Tg, Eg), method="linear")

        Z_eta = safe_gaussian(Z_eta, self.sigma)
        Z_xi  = safe_gaussian(Z_xi,  self.sigma)
        Z_m   = safe_gaussian(Z_m,   self.sigma)

        return Tg, Eg, Z_eta, Z_xi, Z_m


# -------------------------------
# RBF interpolation (expensive)
# -------------------------------

class RBFStrategy(InterpolationStrategy):
    def __init__(self, smooth=0.5):
        self.smooth = smooth

    def interpolate(self, df, resolution=100):
        T = df["temperature"].values
        E = df["log_strain_rate"].values

        Tg, Eg = build_grid(df, resolution)

        # Build interpolators (expensive)
        rbf_eta = Rbf(T, E, df["eta"], smooth=self.smooth)
        rbf_xi  = Rbf(T, E, df["xi"],  smooth=self.smooth)
        rbf_m   = Rbf(T, E, df["m"],   smooth=self.smooth)

        Z_eta = rbf_eta(Tg, Eg)
        Z_xi  = rbf_xi(Tg, Eg)
        Z_m   = rbf_m(Tg, Eg)

        return Tg, Eg, Z_eta, Z_xi, Z_m
    
class GridSplineStrategy(InterpolationStrategy):
    def __init__(self, kx=3, ky=3, s=0):
        """
        kx, ky: spline order (3 = cubic)
        s: smoothing factor (0 = interpolation, >0 = smoothing)
        """
        self.kx = kx
        self.ky = ky
        self.s = s

    def interpolate(self, df, resolution=100):
        # -------------------------------
        # Build structured grid via pivot
        # -------------------------------
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

        pivot_m = df.pivot_table(
            index="temperature",
            columns="log_strain_rate",
            values="m"
        )


        pivot_eta = pivot_eta.sort_index().sort_index(axis=1)
        pivot_xi  = pivot_xi.sort_index().sort_index(axis=1)
        pivot_m   = pivot_m.sort_index().sort_index(axis=1)

        pivot_eta = pivot_eta.dropna(axis=0).dropna(axis=1)
        pivot_xi  = pivot_xi.loc[pivot_eta.index, pivot_eta.columns]
        pivot_m   = pivot_m.loc[pivot_eta.index, pivot_eta.columns]

        if pivot_eta.shape[0] < 3 or pivot_eta.shape[1] < 3:
            raise ValueError("Not enough grid points for spline interpolation")

        T_vals = pivot_eta.index.values
        E_vals = pivot_eta.columns.values

        spline_eta = RectBivariateSpline(T_vals, E_vals, pivot_eta.values,
                                         kx=self.kx, ky=self.ky, s=self.s)

        spline_xi  = RectBivariateSpline(T_vals, E_vals, pivot_xi.values,
                                         kx=self.kx, ky=self.ky, s=self.s)

        spline_m   = RectBivariateSpline(T_vals, E_vals, pivot_m.values,
                                         kx=self.kx, ky=self.ky, s=self.s)

        # -------------------------------
        # Evaluate on fine grid
        # -------------------------------
        T_fine = np.linspace(T_vals.min(), T_vals.max(), resolution)
        E_fine = np.linspace(E_vals.min(), E_vals.max(), resolution)

        Tg, Eg = np.meshgrid(T_fine, E_fine)

        Z_eta = spline_eta(T_fine, E_fine).T
        Z_xi  = spline_xi(T_fine, E_fine).T
        Z_m   = spline_m(T_fine, E_fine).T

        return Tg, Eg, Z_eta, Z_xi, Z_m