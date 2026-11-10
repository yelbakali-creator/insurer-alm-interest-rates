"""US Treasury par yield curves: loading, bootstrapping a zero curve, Nelson-Siegel-Svensson fit.

Par yields are bond-equivalent (semi-annual) yields of par bonds. On a semi-annual grid,
discount factors are bootstrapped from the par condition
    1 = (c / 2) * sum_{i<n} D_i + (1 + c / 2) * D_n
so that every par bond reprices exactly at 100.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

TENORS = {"BC_1MONTH": 1 / 12, "BC_3MONTH": 0.25, "BC_6MONTH": 0.5, "BC_1YEAR": 1, "BC_2YEAR": 2,
          "BC_3YEAR": 3, "BC_5YEAR": 5, "BC_7YEAR": 7, "BC_10YEAR": 10, "BC_20YEAR": 20, "BC_30YEAR": 30}


def load_curves(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df = df[list(TENORS)].replace(0.0, np.nan) / 100          # zeros in the file mean "not published"
    df.columns = [TENORS[c] for c in df.columns]
    return df


def bootstrap(par_curve, max_maturity=30.0):
    """Zero-coupon discount factors on a semi-annual grid from one day's par yields."""
    obs = par_curve.dropna()
    grid = np.arange(0.5, max_maturity + 1e-9, 0.5)
    par = np.interp(grid, obs.index.values, obs.values)        # linear in par yields between tenors
    D = np.empty(len(grid))
    for n, c in enumerate(par):
        D[n] = (1 - c / 2 * D[:n].sum()) / (1 + c / 2)
    zero = -np.log(D) / grid                                   # continuously compounded
    return pd.DataFrame({"maturity": grid, "par_yield": par, "discount": D, "zero_rate": zero})


def reprice_par_bonds(boot):
    """Price per 100 of each grid par bond with the bootstrapped discount factors (should be 100)."""
    D, c = boot["discount"].values, boot["par_yield"].values
    return np.array([100 * (c[n] / 2 * D[: n + 1].sum() + D[n]) for n in range(len(D))])


def nss_zero(t, p):
    """Nelson-Siegel-Svensson zero rate (continuous compounding)."""
    b0, b1, b2, b3, l1, l2 = p
    t = np.maximum(np.asarray(t, float), 1e-6)
    f1 = (1 - np.exp(-t / l1)) / (t / l1)
    f2 = f1 - np.exp(-t / l1)
    f3 = (1 - np.exp(-t / l2)) / (t / l2) - np.exp(-t / l2)
    return b0 + b1 * f1 + b2 * f2 + b3 * f3


def nss_forward(t, p):
    """Instantaneous forward rate implied by the NSS parameters."""
    b0, b1, b2, b3, l1, l2 = p
    t = np.asarray(t, float)
    return (b0 + b1 * np.exp(-t / l1) + b2 * (t / l1) * np.exp(-t / l1)
            + b3 * (t / l2) * np.exp(-t / l2))


def fit_nss(maturity, zero):
    def resid(p):
        return nss_zero(maturity, p) - zero
    best = None
    for l1 in (0.5, 1.5, 3.0):
        for l2 in (5.0, 10.0, 15.0):
            x0 = [zero[-1], zero[0] - zero[-1], 0.0, 0.0, l1, l2]
            r = least_squares(resid, x0, bounds=([0.0, -0.2, -0.3, -0.3, 0.05, 0.5], [0.2, 0.2, 0.3, 0.3, 10, 30]))
            if best is None or r.cost < best.cost:
                best = r
    return best.x


class ZeroCurve:
    """Smooth zero curve: NSS up to the last liquid point, flat instantaneous forward beyond it."""

    def __init__(self, params, last_liquid=30.0):
        self.p, self.llp = params, last_liquid
        self.f_llp = float(nss_forward(last_liquid, params))
        self.z_llp = float(nss_zero(last_liquid, params))

    def zero(self, t):
        t = np.asarray(t, float)
        inside = nss_zero(np.minimum(t, self.llp), self.p)
        beyond = (self.z_llp * self.llp + self.f_llp * (t - self.llp)) / np.maximum(t, 1e-9)
        return np.where(t <= self.llp, inside, beyond)

    def forward(self, t):
        t = np.asarray(t, float)
        return np.where(t <= self.llp, nss_forward(np.minimum(t, self.llp), self.p), self.f_llp)

    def discount(self, t):
        t = np.asarray(t, float)
        return np.exp(-self.zero(t) * t)
