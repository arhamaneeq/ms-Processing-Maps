import numpy as np
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter


class InterpolationStrategy:
    def interpolate(self, df, resolution=100):
        raise NotImplementedError
    
class LinearGridStrategy(InterpolationStrategy):
    def interpolate(self, df, resolution=100):
        points = df[["temperature", "log_strain_rate"]].values

        T = np.linspace(df["temperature"].min(), df["temperature"].max(), resolution)
        E = np.linspace(df["log_strain_rate"].min(), df["log_strain_rate"].max(), resolution)

        Tg, Eg = np.meshgrid(T, E)

        Z_eta = griddata(points, df["eta"], (Tg, Eg), method="linear")
        Z_xi  = griddata(points, df["xi"],  (Tg, Eg), method="linear")

        return Tg, Eg, Z_eta, Z_xi
    
class SmoothedGridStrategy(InterpolationStrategy):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def interpolate(self, df, resolution=100):
        points = df[["temperature", "log_strain_rate"]].values

        T = np.linspace(df["temperature"].min(), df["temperature"].max(), resolution)
        E = np.linspace(df["log_strain_rate"].min(), df["log_strain_rate"].max(), resolution)

        Tg, Eg = np.meshgrid(T, E)

        Z_eta = griddata(points, df["eta"], (Tg, Eg), method="linear")
        Z_xi  = griddata(points, df["xi"],  (Tg, Eg), method="linear")

        Z_eta = gaussian_filter(Z_eta, sigma=self.sigma)
        Z_xi  = gaussian_filter(Z_xi, sigma=self.sigma)

        return Tg, Eg, Z_eta, Z_xi
    
class RBFStrategy(InterpolationStrategy):
    def __init__(self, smooth=0.5):
        self.smooth = smooth

    def interpolate(self, df, resolution=100):
        T = df["temperature"].values
        E = df["log_strain_rate"].values

        rbf_eta = Rbf(T, E, df["eta"], smooth=self.smooth)
        rbf_xi  = Rbf(T, E, df["xi"],  smooth=self.smooth)

        T_grid = np.linspace(T.min(), T.max(), resolution)
        E_grid = np.linspace(E.min(), E.max(), resolution)

        Tg, Eg = np.meshgrid(T_grid, E_grid)

        Z_eta = rbf_eta(Tg, Eg)
        Z_xi  = rbf_xi(Tg, Eg)

        return Tg, Eg, Z_eta, Z_xi