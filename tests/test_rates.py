"""Validation tests: bootstrap repricing, durations, key-rate additivity and Hull-White martingale property."""
from pathlib import Path

import numpy as np

from src.alm import annuity_cash_flows, effective_duration_convexity, key_rate_dollar_durations, macaulay_duration, pv
from src.curve import ZeroCurve, bootstrap, fit_nss, load_curves, reprice_par_bonds
from src.rates_models import HullWhite

CURVES = load_curves(Path(__file__).parents[1] / "data" / "ust_raw.csv")
BOOT = bootstrap(CURVES.loc["2021-12-31"])
CURVE = ZeroCurve(fit_nss(BOOT["maturity"].values, BOOT["zero_rate"].values))


def test_bootstrap_reprices_par_bonds():
    assert np.abs(reprice_par_bonds(BOOT) - 100).max() < 1e-10


def test_durations_and_key_rate_additivity():
    t, c = annuity_cash_flows()
    # Central difference with step h has error of order h^2 * E[t^3] / 6: h = 1e-5 gives about 5e-8 here
    D, _ = effective_duration_convexity(t, c, CURVE.zero, h=1e-5)
    assert abs(D - macaulay_duration(t, c, CURVE.zero)) < 1e-6
    krd = key_rate_dollar_durations(t, c, CURVE.zero)
    assert abs(krd.sum() / (pv(t, c, CURVE.zero) * D) - 1) < 1e-5


def test_hull_white_martingale():
    hw = HullWhite(CURVE, 0.117, 0.0072)
    times = np.arange(0, 31, 1.0)
    sim = hw.simulate(times, 20_000, np.random.default_rng(5))
    mean, se = sim["discount"][:, 1:].mean(axis=0), sim["discount"][:, 1:].std(axis=0) / np.sqrt(20_000)
    assert np.all(np.abs(mean - CURVE.discount(times[1:])) < 4 * se)


def test_hull_white_bond_formula_at_time_zero():
    hw = HullWhite(CURVE, 0.117, 0.0072)
    r0 = float(CURVE.forward(0.0))
    for T in (1.0, 10.0, 30.0):
        assert abs(hw.zcb(0.0, T, r0) / CURVE.discount(T) - 1) < 1e-12
