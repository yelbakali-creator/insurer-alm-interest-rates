"""Asset-liability management for a closed block of life annuities.

All valuations discount cash flows on a zero curve given as a function t -> z(t)
(continuous compounding). Curve shocks are additive functions of maturity.
"""
import numpy as np


def annuity_cash_flows(n_annuitants=10_000, payment=10_000.0, age=65, horizon=50,
                       A=0.0001, B=0.00003, c=1.10):
    """Expected annual payments of a closed annuity block (Gompertz-Makeham survival)."""
    t = np.arange(1, horizon + 1)
    ages = age + np.arange(horizon)
    q = 1 - np.exp(-(A + B * c ** (ages + 0.5)))
    surv = np.cumprod(1 - q)
    return t.astype(float), n_annuitants * payment * surv


def pv(times, cfs, zero_fn, shock=None):
    z = zero_fn(times) + (0.0 if shock is None else shock(times))
    return float(np.sum(cfs * np.exp(-z * times)))


def macaulay_duration(times, cfs, zero_fn):
    df = np.exp(-zero_fn(times) * times)
    return float(np.sum(times * cfs * df) / np.sum(cfs * df))


def effective_duration_convexity(times, cfs, zero_fn, h=1e-4):
    p0 = pv(times, cfs, zero_fn)
    up = pv(times, cfs, zero_fn, lambda t: h + 0 * t)
    dn = pv(times, cfs, zero_fn, lambda t: -h + 0 * t)
    return (dn - up) / (2 * h * p0), (up + dn - 2 * p0) / (h ** 2 * p0)


KEY_RATES = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])


def key_rate_bump(k, key=KEY_RATES):
    """Triangular bump of 1 at key rate k, zero at neighbouring key rates, flat outside the range."""
    def f(t):
        t = np.asarray(t, float)
        xp = key
        fp = np.zeros(len(key)); fp[k] = 1.0
        return np.interp(t, xp, fp, left=fp[0], right=fp[-1])
    return f


def key_rate_dollar_durations(times, cfs, zero_fn, h=1e-4):
    """Dollar sensitivity to each key rate: -dPV/dz_k (per unit rate change)."""
    out = []
    for k in range(len(KEY_RATES)):
        bump = key_rate_bump(k)
        up = pv(times, cfs, zero_fn, lambda t: h * bump(t))
        dn = pv(times, cfs, zero_fn, lambda t: -h * bump(t))
        out.append((dn - up) / (2 * h))
    return np.array(out)


def zcb_dollar_krd(maturity, zero_fn, h=1e-4):
    return key_rate_dollar_durations(np.array([maturity]), np.array([1.0]), zero_fn, h)
