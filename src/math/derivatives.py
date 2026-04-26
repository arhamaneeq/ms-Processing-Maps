import pandas as pd
import numpy as np

class DerivativeStrategy:
    def fit(self, df: pd.DataFrame):
        pass  # optional

    def compute(self, g: pd.DataFrame):
        raise NotImplementedError
    
class FiniteDifferenceStrategy(DerivativeStrategy):
    def compute(self, g: pd.DataFrame):
        x = g["log_strain_rate"].values
        y = g["log_flow_stress"].values

        dlog_sigma = np.gradient(y, x)
        m = dlog_sigma

        dm = np.gradient(m, x)
        xi = m + dm
        eta = 2 * m / (m + 1)

        return m, xi, eta
    
from scipy.interpolate import UnivariateSpline

class Spline1DStrategy(DerivativeStrategy):
    def __init__(self, s_factor=0.01):
        self.s_factor = s_factor

    def compute(self, g: pd.DataFrame):
        x = g["log_strain_rate"].values
        y = g["log_flow_stress"].values

        if len(x) < 4 or not np.all(np.diff(x) > 0):
            return None

        s = len(x) * np.var(y) * self.s_factor
        spline = UnivariateSpline(x, y, s=s)

        d1 = spline.derivative(1)(x)
        d2 = spline.derivative(2)(x)

        m = d1
        xi = m + d2
        eta = 2 * m / (m + 1)

        return m, xi, eta
    
from scipy.signal import savgol_filter

class SavitzkyGolayStrategy(DerivativeStrategy):
    def __init__(self, window=5, poly=2):
        self.window = window
        self.poly = poly

    def compute(self, g: pd.DataFrame):
        x = g["log_strain_rate"].values
        y = g["log_flow_stress"].values

        if len(x) < self.window:
            return None

        # assumes roughly uniform spacing
        dy = savgol_filter(y, self.window, self.poly, deriv=1)
        d2y = savgol_filter(y, self.window, self.poly, deriv=2)

        m = dy
        xi = m + d2y
        eta = 2 * m / (m + 1)

        return m, xi, eta
    
from scipy.interpolate import SmoothBivariateSpline

class Spline2DStrategy(DerivativeStrategy):
    def __init__(self, s=0.1):
        self.s = s
        self.spline = None

    def fit(self, df: pd.DataFrame):
        x = df["log_strain_rate"].values
        t = df["temperature"].values
        y = df["log_flow_stress"].values

        self.spline = SmoothBivariateSpline(x, t, y, s=self.s)

    def compute(self, g: pd.DataFrame):
        if self.spline is None:
            raise RuntimeError("Spline2DStrategy must be fit() before use")

        x = g["log_strain_rate"].values
        t = g["temperature"].values

        d1 = self.spline.ev(x, t, dx=1, dy=0)
        d2 = self.spline.ev(x, t, dx=2, dy=0)

        m = d1
        xi = m + d2
        eta = 2 * m / (m + 1)

        return m, xi, eta