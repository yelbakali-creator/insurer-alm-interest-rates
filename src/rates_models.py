"""Principal components of yield curve changes and the Hull-White one-factor model."""
import numpy as np
import pandas as pd


# ---------- PCA ----------
def pca_changes(curves, tenors, horizon_months=1):
    """PCA of changes in yields over a horizon, on month-end curves (complete tenors only)."""
    m = curves[tenors].resample("ME").last().dropna()
    chg = (m.shift(-horizon_months) - m).dropna()
    cov = np.cov(chg.values, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    for k in range(vecs.shape[1]):                    # sign convention: level up, slope steepening
        if (k == 0 and vecs[:, k].sum() < 0) or (k == 1 and vecs[-1, k] < vecs[0, k]) or \
           (k == 2 and vecs[len(tenors) // 2, k] < 0):
            vecs[:, k] *= -1
    return {"eigenvalues": vals, "eigenvectors": pd.DataFrame(vecs, index=tenors),
            "explained": vals / vals.sum(), "changes": chg}


# ---------- Hull-White ----------
def calibrate_short_rate(short_rate, dt):
    """Real-world estimate of mean reversion a and volatility sigma from an AR(1) on the short rate:
    r(t+dt) = alpha + phi r(t) + e  =>  a = -ln(phi)/dt,  sigma = sd(e) sqrt(2a / (1 - phi^2))."""
    x, y = short_rate[:-1], short_rate[1:]
    phi, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + phi * x)
    a = -np.log(phi) / dt
    sigma = resid.std(ddof=2) * np.sqrt(2 * a / (1 - phi ** 2))
    return {"a": a, "sigma": sigma, "phi": phi}


class HullWhite:
    """dr = (theta(t) - a r) dt + sigma dW, with theta fitted to the initial zero curve.

    r(t) = x(t) + phi(t), x an Ornstein-Uhlenbeck process with x(0) = 0 and
    phi(t) = f(0,t) + sigma^2 / (2 a^2) (1 - e^{-a t})^2.
    """

    def __init__(self, curve, a, sigma):
        self.c, self.a, self.s = curve, a, sigma

    def B(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a

    def zcb(self, t, T, r_t):
        """Analytic zero-coupon bond price P(t, T) given the short rate at t."""
        a, s = self.a, self.s
        B = self.B(t, T)
        lnA = (np.log(self.c.discount(T) / self.c.discount(t)) + B * self.c.forward(t)
               - s ** 2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B ** 2)
        return np.exp(lnA - B * r_t)

    def simulate(self, times, n_paths, rng):
        """Exact joint simulation of x(t) and its integral; returns short rates and path discount factors."""
        a, s = self.a, self.s
        n = len(times)
        x = np.zeros((n_paths, n))
        integ = np.zeros((n_paths, n))
        for k in range(1, n):
            d = times[k] - times[k - 1]
            e = np.exp(-a * d)
            v1 = s ** 2 / (2 * a) * (1 - e ** 2)
            v2 = s ** 2 / a ** 2 * (d - 2 * (1 - e) / a + (1 - e ** 2) / (2 * a))
            cv = s ** 2 / (2 * a ** 2) * (1 - e) ** 2
            L = np.linalg.cholesky(np.array([[v1, cv], [cv, v2]]))
            z = rng.standard_normal((n_paths, 2)) @ L.T
            x[:, k] = x[:, k - 1] * e + z[:, 0]
            integ[:, k] = integ[:, k - 1] + x[:, k - 1] * (1 - e) / a + z[:, 1]
        T = np.asarray(times)
        var_int = s ** 2 / a ** 2 * (T - 2 * (1 - np.exp(-a * T)) / a + (1 - np.exp(-2 * a * T)) / (2 * a))
        int_phi = -np.log(self.c.discount(T)) + 0.5 * var_int      # integral of phi from 0 to T
        phi = self.c.forward(T) + s ** 2 / (2 * a ** 2) * (1 - np.exp(-a * T)) ** 2
        return {"r": x + phi[None, :], "discount": np.exp(-(integ + int_phi[None, :])), "x": x}
