"""Interest-rate risk and asset-liability management for a life annuity block, US Treasury data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.alm import (KEY_RATES, annuity_cash_flows, effective_duration_convexity, key_rate_dollar_durations,
                     macaulay_duration, pv, zcb_dollar_krd)
from src.curve import ZeroCurve, bootstrap, fit_nss, load_curves, nss_zero, reprice_par_bonds
from src.rates_models import HullWhite, calibrate_short_rate, pca_changes

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD, GREEN = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017", "#2E7D32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

VAL_DATE = "2021-12-31"
PCA_TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]      # published every day since 1990
SURPLUS_RATIO = 1.10
N_PATHS, SEED = 20_000, 2026


def zcb_portfolio(maturities, notionals):
    return np.asarray(maturities, float), np.asarray(notionals, float)


def main():
    rng = np.random.default_rng(SEED)
    curves = load_curves(ROOT / "data" / "ust_raw.csv")
    hist = curves.loc[:VAL_DATE]

    # 1. Curve at the valuation date: bootstrap, then smooth NSS curve with flat-forward extrapolation
    boot = bootstrap(curves.loc[VAL_DATE])
    reprice_err = np.abs(reprice_par_bonds(boot) - 100).max()
    p = fit_nss(boot["maturity"].values, boot["zero_rate"].values)
    curve = ZeroCurve(p)
    nss_err = (nss_zero(boot["maturity"].values, p) - boot["zero_rate"].values) * 1e4
    boot.to_csv(TABLES / "bootstrapped_curve_2021_12_31.csv", index=False)

    # 2. PCA of monthly and annual yield changes, 1990-2021
    pca_m = pca_changes(hist, PCA_TENORS, 1)
    pca_y = pca_changes(hist, PCA_TENORS, 12)
    load = pca_y["eigenvectors"].iloc[:, :3].copy()
    load.columns = ["level", "slope", "curvature"]
    sd_y = np.sqrt(pca_y["eigenvalues"][:3])
    scores_y = pca_y["changes"].values @ pca_y["eigenvectors"].values[:, :3]
    pca_table = pd.DataFrame({"component": ["level", "slope", "curvature"],
                              "explained_monthly": pca_m["explained"][:3], "explained_annual": pca_y["explained"][:3],
                              "annual_sd_of_score": sd_y,
                              "shock_99_5_normal": norm.ppf(0.995) * sd_y,
                              "shock_99_5_empirical_up": np.quantile(scores_y, 0.995, axis=0),
                              "shock_99_5_empirical_down": np.quantile(scores_y, 0.005, axis=0)})
    pca_table.to_csv(TABLES / "pca_summary.csv", index=False)
    load.to_csv(TABLES / "pca_loadings_annual.csv")

    def pca_shock(k, size):
        vec = load.iloc[:, k].values * size
        return lambda t: np.interp(np.asarray(t, float), PCA_TENORS, vec)   # flat beyond 10 years

    # 3. Hull-White: real-world calibration on the 3-month yield, then martingale tests
    sr = hist[0.25].resample("ME").last().dropna().values
    cal = calibrate_short_rate(sr, 1 / 12)
    hw = HullWhite(curve, cal["a"], cal["sigma"])
    times = np.arange(0, 51, 1.0)
    sim = hw.simulate(times, N_PATHS, rng)
    disc = sim["discount"]
    mart = pd.DataFrame({"maturity": times[1:], "market_discount": curve.discount(times[1:]),
                         "mc_mean_discount": disc[:, 1:].mean(axis=0),
                         "mc_se": disc[:, 1:].std(axis=0, ddof=1) / np.sqrt(N_PATHS)})
    mart["z_score"] = (mart["mc_mean_discount"] - mart["market_discount"]) / mart["mc_se"]
    mart.to_csv(TABLES / "hull_white_martingale_test.csv", index=False)
    # Second test: E[D(0,5) P(5,T)] = P(0,T), combining simulation and the analytic bond formula
    k5 = 5
    second = []
    for T in (10.0, 20.0, 30.0):
        v = disc[:, k5] * hw.zcb(5.0, T, sim["r"][:, k5])
        second.append((T, float(v.mean()), float(curve.discount(T)), float(v.std(ddof=1) / np.sqrt(N_PATHS))))

    # 4. Liabilities and four asset strategies
    tL, cL = annuity_cash_flows()
    zero_fn = curve.zero
    PV_L = pv(tL, cL, zero_fn)
    D_L, C_L = effective_duration_convexity(tL, cL, zero_fn)
    dollar_L = PV_L * D_L
    krd_L = key_rate_dollar_durations(tL, cL, zero_fn)
    A = SURPLUS_RATIO * PV_L

    def zcb_price(T):
        return float(curve.discount(T))

    strategies = {}
    strategies["Short: 5-year bonds"] = zcb_portfolio([5.0], [A / zcb_price(5.0)])
    T_match = dollar_L / A                                      # ZCB dollar duration = value x maturity
    strategies["Duration-matched bullet"] = zcb_portfolio([T_match], [A / zcb_price(T_match)])
    # Barbell 2y + 30y: value A and dollar duration equal to the liabilities'
    w30 = (dollar_L / A - 2.0) / (30.0 - 2.0)
    strategies["Barbell 2y + 30y"] = zcb_portfolio([2.0, 30.0], [A * (1 - w30) / zcb_price(2.0), A * w30 / zcb_price(30.0)])
    # Key-rate matched: one zero-coupon bond at each key rate, the rest in cash
    notionals = np.array([krd_L[k] / zcb_dollar_krd(T, zero_fn)[k] for k, T in enumerate(KEY_RATES)])
    cash = A - sum(n * zcb_price(T) for n, T in zip(notionals, KEY_RATES))
    strategies["Key-rate matched"] = zcb_portfolio(list(KEY_RATES) + [1e-6], list(notionals) + [cash])

    bs = []
    for name, (tA, cA) in strategies.items():
        PV_A = pv(tA, cA, zero_fn)
        D_A, C_A = effective_duration_convexity(tA, cA, zero_fn)
        bs.append({"strategy": name, "assets": PV_A, "liabilities": PV_L, "surplus": PV_A - PV_L,
                   "asset_duration": D_A, "liability_duration": D_L,
                   "dollar_duration_gap": PV_A * D_A - dollar_L,
                   "asset_convexity": C_A, "liability_convexity": C_L})
    bs = pd.DataFrame(bs)
    bs.to_csv(TABLES / "balance_sheet_strategies.csv", index=False)

    krd_tab = pd.DataFrame({"key_rate": KEY_RATES, "liabilities": krd_L})
    for name, (tA, cA) in strategies.items():
        krd_tab[name] = key_rate_dollar_durations(tA, cA, zero_fn)
    krd_tab.to_csv(TABLES / "key_rate_dollar_durations.csv", index=False)

    # 5. Instantaneous shocks to the surplus
    z995 = norm.ppf(0.995)
    shocks = {
        "Parallel +100bp": lambda t: 0.01 + 0 * np.asarray(t, float),
        "Parallel -100bp": lambda t: -0.01 + 0 * np.asarray(t, float),
        "Steepener (-50bp at 1y to +50bp at 30y)": lambda t: np.interp(np.asarray(t, float), [1, 30], [-0.005, 0.005]),
        "Flattener (+50bp at 1y to -50bp at 30y)": lambda t: np.interp(np.asarray(t, float), [1, 30], [0.005, -0.005]),
        "PCA level, 1-in-200 down": pca_shock(0, -z995 * sd_y[0]),
        "PCA slope, 1-in-200 flattening": pca_shock(1, -z995 * sd_y[1]),
        "PCA curvature, 1-in-200": pca_shock(2, -z995 * sd_y[2]),
    }
    shock_rows = []
    for sname, f in shocks.items():
        dL = pv(tL, cL, zero_fn, f) - PV_L
        row = {"shock": sname, "change_in_liabilities": dL}
        for name, (tA, cA) in strategies.items():
            row[name] = (pv(tA, cA, zero_fn, f) - pv(tA, cA, zero_fn)) - dL
        shock_rows.append(row)
    shock_tab = pd.DataFrame(shock_rows)
    shock_tab.to_csv(TABLES / "surplus_change_under_shocks.csv", index=False)

    # Parametric 3-factor 1-in-200 surplus loss (internal-model style)
    sc = rng.standard_normal((20_000, 3)) * sd_y
    grid = np.arange(1, 51, 1.0)
    shocks_grid = sc @ np.vstack([np.interp(grid, PCA_TENORS, load.iloc[:, k].values) for k in range(3)])
    var_rows = []
    for name, (tA, cA) in strategies.items():
        base_A, base_L = pv(tA, cA, zero_fn), PV_L
        losses = []
        for s in shocks_grid:
            f = lambda t, s=s: np.interp(np.asarray(t, float), grid, s)
            losses.append(-((pv(tA, cA, zero_fn, f) - base_A) - (pv(tL, cL, zero_fn, f) - base_L)))
        losses = np.array(losses)
        var_rows.append({"strategy": name, "surplus_loss_99_5": np.quantile(losses, 0.995),
                         "as_share_of_surplus": np.quantile(losses, 0.995) / (A - PV_L)})
    pca_var = pd.DataFrame(var_rows)
    pca_var.to_csv(TABLES / "pca_1_in_200_surplus_loss.csv", index=False)

    # Hull-White one-year surplus distribution (revaluation with the analytic bond formula)
    r1 = sim["r"][:, 1]
    hw_rows, hw_dist = [], {}
    growth = 1 / disc[:, 1]                                       # cash rolled up at the simulated short rate

    def value_at_1(tt, cc):
        """Value at t = 1: future flows with the analytic bond price, earlier cash rolled up to t = 1."""
        keep = tt > 1
        future = (cc[keep][None, :] * hw.zcb(1.0, tt[keep][None, :], r1[:, None])).sum(axis=1)
        return future + cc[tt < 1].sum() * growth + cc[tt == 1].sum()
    L1 = value_at_1(tL, cL)
    for name, (tA, cA) in strategies.items():
        S1 = value_at_1(tA, cA) - L1
        hw_dist[name] = S1
        hw_rows.append({"strategy": name, "surplus_mean_t1": S1.mean(), "surplus_p0_5_t1": np.quantile(S1, 0.005),
                        "prob_surplus_negative": (S1 < 0).mean()})
    hw_tab = pd.DataFrame(hw_rows)
    hw_tab.to_csv(TABLES / "hull_white_one_year_surplus.csv", index=False)

    # 6. Validation summary
    D_mac = macaulay_duration(tL, cL, zero_fn)
    validation = pd.DataFrame([
        ["Bootstrap: max repricing error of par bonds (per 100)", reprice_err],
        ["NSS fit: RMSE vs bootstrapped zeros (bp)", np.sqrt(np.mean(nss_err ** 2))],
        ["NSS fit: max absolute error (bp)", np.abs(nss_err).max()],
        ["Liabilities: Macaulay duration (analytic)", D_mac],
        ["Liabilities: effective duration (finite difference)", D_L],
        ["Liabilities: sum of key-rate dollar durations / dollar duration", krd_L.sum() / dollar_L],
        ["Hull-White martingale test: max |z-score| over 50 maturities", mart["z_score"].abs().max()],
        *[[f"Hull-White: E[D(0,5) P(5,{int(T)})] / P(0,{int(T)}) (z-score {(m - mk) / se:.2f})", m / mk]
          for T, m, mk, se in second],
        ["Hull-White calibration: mean reversion a", cal["a"]],
        ["Hull-White calibration: volatility sigma", cal["sigma"]],
    ], columns=["check", "value"])
    validation.to_csv(TABLES / "validation.csv", index=False)

    # 7. Charts
    fig, ax = plt.subplots(figsize=(7.5, 4))
    obs = curves.loc[VAL_DATE].dropna()
    tt = np.linspace(0.25, 50, 300)
    ax.scatter(obs.index, obs.values * 100, color="black", s=18, zorder=3, label="Par yields (observed)")
    ax.plot(boot["maturity"], boot["zero_rate"] * 100, color=GREY, lw=1, label="Bootstrapped zero rates")
    ax.plot(tt, curve.zero(tt) * 100, color=NAVY, lw=1.5, label="NSS zero curve, flat forward beyond 30y")
    ax.plot(tt, curve.forward(tt) * 100, color=RED, lw=1, ls="--", label="Instantaneous forward")
    ax.axvline(30, color=GREY, ls=":", lw=1)
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("%"); ax.set_title("US Treasury curve, 31 December 2021")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "yield_curve.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for col, c_ in zip(load.columns, [NAVY, RED, GOLD]):
        axes[0].plot(PCA_TENORS, load[col], marker="o", color=c_, label=col)
    axes[0].axhline(0, color=GREY, lw=0.6)
    axes[0].set_xlabel("Maturity (years)"); axes[0].set_title("PCA loadings of annual yield changes, 1990-2021")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(["Level", "Slope", "Curvature"], pca_y["explained"][:3] * 100, color=[NAVY, RED, GOLD])
    for i, v in enumerate(pca_y["explained"][:3] * 100):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("% of variance"); axes[1].set_title("Variance explained (annual changes)")
    fig.tight_layout(); fig.savefig(FIGS / "pca.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    q = np.quantile(sim["r"][:, :31], [0.025, 0.5, 0.975], axis=0) * 100
    axes[0].fill_between(times[:31], q[0], q[2], color=NAVY, alpha=0.2, label="95% band")
    axes[0].plot(times[:31], q[1], color=NAVY, label="Median short rate")
    axes[0].plot(times[:31], curve.forward(times[:31]) * 100, color=RED, ls="--", label="Initial forward curve")
    axes[0].set_xlabel("Years"); axes[0].set_ylabel("%"); axes[0].set_title("Hull-White short-rate scenarios")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].errorbar(mart["maturity"], mart["z_score"], fmt="o", ms=3, color=NAVY)
    axes[1].axhline(0, color=GREY, lw=0.6); axes[1].axhspan(-1.96, 1.96, color=GREEN, alpha=0.1, label="+/- 1.96")
    axes[1].set_xlabel("Maturity (years)"); axes[1].set_ylabel("z-score (MC - market) / SE")
    axes[1].set_title("Martingale test: simulated discount factors vs market curve")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "hull_white.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.15
    names = ["liabilities"] + list(strategies)
    for i, (nm, c_) in enumerate(zip(names, ["black", GREY, GOLD, RED, NAVY])):
        ax.bar(np.arange(len(KEY_RATES)) + (i - 2) * width, krd_tab[nm] / 1e9 * 1e-2, width=width, color=c_, label=nm)
    ax.set_xticks(np.arange(len(KEY_RATES)), [f"{int(k)}y" for k in KEY_RATES])
    ax.set_ylabel("Value change for -1% at the key rate (bn)")
    ax.set_title("Key-rate dollar durations: assets vs liabilities")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "key_rate_durations.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sel = shock_tab.set_index("shock")[list(strategies)] / 1e6
    y = np.arange(len(sel))
    for i, (nm, c_) in enumerate(zip(strategies, [GREY, GOLD, RED, NAVY])):
        ax.barh(y + (i - 1.5) * 0.2, sel[nm], height=0.2, color=c_, label=nm)
    ax.set_yticks(y, sel.index, fontsize=8); ax.axvline(0, color="black", lw=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Change in surplus (million)"); ax.set_title("Surplus sensitivity to curve shocks")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout(); fig.savefig(FIGS / "surplus_shocks.png", dpi=150); plt.close(fig)

    # Console summary
    pd.set_option("display.width", 220)
    print(f"Valuation date {VAL_DATE}: NSS params {np.round(p, 4)}")
    print(f"Liabilities PV {PV_L / 1e6:,.1f}m, effective duration {D_L:.2f}, convexity {C_L:.1f}; assets {A / 1e6:,.1f}m")
    print("\nValidation\n" + validation.to_string(index=False))
    print("\nPCA\n" + pca_table.round(5).to_string(index=False))
    print("\nBalance sheet\n" + bs.round(3).to_string(index=False))
    print("\nKey-rate dollar durations (per 1.00 rate change, million)\n" + (krd_tab.set_index("key_rate") / 1e6).round(1).to_string())
    print("\nSurplus change under shocks (million)\n" + (shock_tab.set_index("shock") / 1e6).round(2).to_string())
    print("\nPCA 1-in-200 surplus loss\n" + pca_var.round(4).to_string(index=False))
    print("\nHull-White one-year surplus\n" + hw_tab.round(4).to_string(index=False))
    print(f"\nKey-rate matched notionals: {np.round(notionals / 1e6, 1)} cash {cash / 1e6:.1f}m; bullet maturity {T_match:.2f}y; barbell 30y weight {w30:.3f}")


if __name__ == "__main__":
    main()
