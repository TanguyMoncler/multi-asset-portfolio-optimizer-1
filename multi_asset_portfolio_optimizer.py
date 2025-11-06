"""
Multi-Asset Portfolio Optimizer (Interactive) — SIMPLE RETURNS VERSION
(Prices: Daily ; Rebalance: Weekly) — MIN_VAR / MAX_SHARPE + Soft Constraints
======================================================================

- Univers: ETF US (SPY/AGG/...) ou UCITS Europe (CSPX.L/IEAC.L/...)
- Objectifs (mode 'constrained'): 'min_var' | 'max_sharpe'
- Modes:
    • 'constrained' (min_var / max_sharpe)
    • 'max_perf'    (winner-take-all sur μ)
    • 'ts_mom'      (time-series momentum long/flat)
    • 'static_60_40'
- Backtest Hebdo (rééquilibrage), plots et CSV optionnel

Dépendances:
  pip install pandas numpy yfinance matplotlib cvxpy
"""
from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------
# Dataclasses
# ----------------------------
@dataclass
class Universe:
    tickers: List[str]
    groups: Dict[str, List[str]]

@dataclass
class Constraints:
    long_only: bool = True
    weight_cap: Optional[float] = 0.4
    group_caps: Optional[Dict[str, Tuple[float, float]]] = None
    target_vol: Optional[float] = 0.12
    te_max: Optional[float] = None
    turnover_cap: Optional[float] = 0.5
    esg_min: Optional[float] = None

@dataclass
class BacktestConfig:
    start: str = "2015-01-01"
    end: Optional[str] = None
    rebalance: str = "W-FRI"              # Weekly (Friday)
    lookback_months: int = 24
    halflife_months: int = 6
    risk_free_rate: float = 0.01
    objective: str = "max_sharpe"         # "min_var" | "max_sharpe"
    mu_model: str = "momentum"            # "mean" | "momentum"
    mode: str = "constrained"             # "constrained" | "max_perf" | "ts_mom" | "static_60_40"
    export_csv: Optional[str] = "daily_returns.csv"

@dataclass
class Benchmark:
    weights: Dict[str, float]

# ----------------------------
# Utils
# ----------------------------
def load_prices(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Prix JOURNALIERS ajustés (Close) via yfinance."""
    data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, interval="1d"
    )["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all").ffill().dropna()
    return data

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements SIMPLES quotidiens."""
    px = prices.asfreq("B").ffill()
    return px.pct_change().dropna()

def ewma_cov(returns: pd.DataFrame, halflife_days: int) -> pd.DataFrame:
    """EWMA de covariance (sur rendements centrés)."""
    lam = math.exp(math.log(0.5) / max(1, halflife_days))
    w = np.array([lam ** (len(returns)-1 - i) for i in range(len(returns))], dtype=float)
    w = w / w.sum()
    X = returns.values - returns.values.mean(axis=0, keepdims=True)
    cov = X.T @ (w[:, None] * X)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

def annualize_return(ret_series: pd.Series) -> float:
    return float(ret_series.mean() * 252)

def annualize_vol(ret_series: pd.Series) -> float:
    return float(ret_series.std(ddof=0) * math.sqrt(252))

def max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.cummax()
    dd = cum_returns/peak - 1.0
    return float(dd.min())

# ----------------------------
# Expected return models (SIMPLes)
# ----------------------------
def mu_mean(window: pd.DataFrame) -> pd.Series:
    return window.mean()

def momentum_12m_1m(window: pd.DataFrame) -> pd.Series:
    if len(window) < 252 + 21:
        return window.mean()
    px = (1 + window).cumprod()
    ann_mom = px.iloc[-21] / px.iloc[-252] - 1.0
    daily = (1.0 + ann_mom) ** (1/252) - 1.0
    return daily

def pick_safe_asset(universe: Universe) -> Optional[str]:
    for cand in ["AGG", "BNDX", "IEAC.L", "IGLA.L"]:
        if cand in universe.tickers:
            return cand
    return None

# ----------------------------
# Optimization (MIN_VAR / MAX_SHARPE) with SOFT constraints
# ----------------------------
def optimize_weights_constrained(
    mu: pd.Series,
    cov: pd.DataFrame,
    constraints: Constraints,
    benchmark: Optional[Benchmark],
    prev_weights: Optional[np.ndarray],
    groups: Dict[str, List[str]],
    objective: str,  # "min_var" | "max_sharpe"
) -> np.ndarray:
    assert objective in ("min_var", "max_sharpe"), "Objective must be 'min_var' or 'max_sharpe'."

    tickers = list(mu.index)
    n = len(tickers)

    mu_ann = mu.values * 252.0
    Sigma_ann = cov.values * 252.0
    Sigma_ann = 0.5 * (Sigma_ann + Sigma_ann.T) + 1e-6 * np.eye(n)

    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if constraints.long_only:
        cons += [w >= 0]
    if constraints.weight_cap is not None:
        cons += [w <= constraints.weight_cap]

    # --- slack variables (soft constraints) ---
    big_pen = 1e4  # pénalité forte
    penalties = []

    # Group caps (avec slack)
    if constraints.group_caps:
        for g, (gmin, gmax) in constraints.group_caps.items():
            idx = [tickers.index(t) for t in groups.get(g, []) if t in tickers]
            if not idx:
                continue
            if gmin is not None:
                s_lo = cp.Variable(nonneg=True)
                cons += [cp.sum(w[idx]) + s_lo >= gmin]
                penalties.append(s_lo)
            if gmax is not None:
                s_hi = cp.Variable(nonneg=True)
                cons += [cp.sum(w[idx]) - s_hi <= gmax]
                penalties.append(s_hi)

    # Vol cible (soft)
    if constraints.target_vol is not None:
        s_vol = cp.Variable(nonneg=True)
        cons += [cp.quad_form(w, Sigma_ann) <= (constraints.target_vol**2) + s_vol]
        penalties.append(s_vol)

    # Tracking Error (soft)
    if constraints.te_max is not None and benchmark is not None:
        b = np.array([benchmark.weights.get(t, 0.0) for t in tickers], dtype=float)
        s_te = cp.Variable(nonneg=True)
        cons += [cp.quad_form(w - b, Sigma_ann) <= (constraints.te_max**2) + s_te]
        penalties.append(s_te)

    # Turnover (soft)
    if constraints.turnover_cap is not None and prev_weights is not None:
        s_to = cp.Variable(nonneg=True)
        cons += [cp.norm1(w - prev_weights) <= constraints.turnover_cap + s_to]
        penalties.append(s_to)

    # ESG (dur, mais on peut le rendre soft si besoin)
    esg_scores = getattr(mu, "_esg", None)
    if esg_scores is not None and constraints.esg_min is not None:
        s = np.array([esg_scores.get(t, np.nan) for t in tickers], dtype=float)
        mask = np.isfinite(s)
        if mask.sum() > 0:
            cons += [cp.sum(cp.multiply(w[mask], s[mask])) >= constraints.esg_min * cp.sum(w[mask])]

    l2_reg = 1e-3
    pen_term = big_pen * cp.sum(penalties) if len(penalties) else 0

    if objective == "min_var":
        obj = cp.Minimize(cp.quad_form(w, Sigma_ann) + l2_reg * cp.sum_squares(w) + pen_term)
    else:  # "max_sharpe" surrogate
        obj = cp.Maximize(mu_ann @ w - l2_reg * cp.sum_squares(w) - pen_term)

    prob = cp.Problem(obj, cons)

    # Solveur rapide
    try:
        prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-4, eps_rel=1e-4,
            max_iter=8000,
            polish=False,
            verbose=False,
            warm_start=True,
        )
    except Exception:
        prob.solve(solver=cp.SCS, max_iters=2000, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimization failed even with soft constraints.")

    sol = np.array(w.value).astype(float)
    sol = np.clip(sol, 0, 1)
    ssum = sol.sum()
    if ssum > 0:
        sol = sol / ssum
    return sol

def optimize_weights_maxperf(mu: pd.Series) -> np.ndarray:
    n = len(mu)
    w = np.zeros(n, dtype=float)
    best = int(np.nanargmax(mu.values))
    w[best] = 1.0
    return w

# ----------------------------
# Backtest
# ----------------------------
def backtest(
    universe: Universe,
    constraints: Constraints,
    bench: Optional[Benchmark],
    cfg: BacktestConfig,
    esg_scores: Optional[Dict[str, float]] = None,
    rets_override: Optional[pd.DataFrame] = None,
    silent: bool = False,
    do_plots: bool = True,
    do_export: bool = True,
):
    # Data
    if rets_override is None:
        prices = load_prices(universe.tickers, cfg.start, cfg.end)
        rets = compute_returns(prices)
    else:
        rets = rets_override.copy()

    rebal_dates = rets.resample(cfg.rebalance).last().index

    weights_hist: List[pd.Series] = []
    dates_hist: List[pd.Timestamp] = []
    port_parts: List[pd.Series] = []
    bench_parts: List[pd.Series] = []
    prev_w: Optional[np.ndarray] = None

    for i in range(len(rebal_dates) - 1):
        dt = rebal_dates[i]
        next_dt = rebal_dates[i + 1]

        lb_end = dt
        lb_start = dt - pd.DateOffset(months=cfg.lookback_months)
        window = rets.loc[lb_start:lb_end].dropna()
        if len(window) < 60:
            continue

        # μ
        if cfg.mu_model == "momentum":
            mu = momentum_12m_1m(window)
        else:
            mu = mu_mean(window)

        if esg_scores:
            setattr(mu, "_esg", esg_scores)

        # Σ (EWMA)
        halflife_days = int(max(1, cfg.halflife_months) * 21)
        Sigma = ewma_cov(window, halflife_days=halflife_days)

        # Weights
        if cfg.mode == "max_perf":
            w_opt = optimize_weights_maxperf(mu)
        elif cfg.mode == "ts_mom":
            cols = list(rets.columns)
            risky_list = [t for g in ("Equity","Commodities") for t in universe.groups.get(g, []) if t in cols]
            safe = pick_safe_asset(universe)

            mom = momentum_12m_1m(window)
            pos = mom.reindex(risky_list).clip(lower=0.0).fillna(0.0)
            w = np.zeros(len(cols), dtype=float)
            if pos.sum() > 0:
                w_risky = (pos / pos.sum())
                for t, wt in w_risky.items():
                    w[cols.index(t)] = wt

            # vol target (no leverage)
            target_vol = 0.12
            Sigma_win = ewma_cov(window, halflife_days=halflife_days).reindex(index=cols, columns=cols).values * 252.0
            curr_vol = float(np.sqrt(max(1e-12, w @ Sigma_win @ w)))
            if curr_vol > 1e-12 and target_vol is not None:
                lam = min(1.0, target_vol / curr_vol)
                w = w * lam

            leftover = 1.0 - w.sum()
            if leftover > 1e-12 and safe is not None and safe in cols:
                w[cols.index(safe)] += leftover

            w_opt = w
        elif cfg.mode == "static_60_40":
            cols = list(rets.columns)
            w_opt = np.zeros(len(cols), dtype=float)
            eq = [t for t in universe.groups.get("Equity", []) if t in cols]
            bd = [t for t in universe.groups.get("Bonds", []) if t in cols]
            if len(eq) > 0:
                for t in eq:
                    w_opt[cols.index(t)] = 0.6 / len(eq)
            if len(bd) > 0:
                for t in bd:
                    w_opt[cols.index(t)] += 0.4 / len(bd)
        else:  # constrained
            w_opt = optimize_weights_constrained(
                mu=mu, cov=Sigma, constraints=constraints, benchmark=bench,
                prev_weights=prev_w, groups=universe.groups,
                objective=cfg.objective,
            )

        dates_hist.append(dt)
        weights_hist.append(pd.Series(w_opt, index=rets.columns, name=dt))
        prev_w = w_opt

        period = rets.loc[dt:next_dt].iloc[1:]
        if len(period) == 0:
            continue
        port_parts.append((period * w_opt).sum(axis=1))

        if bench is not None:
            b = np.array([bench.weights.get(t, 0.0) for t in rets.columns], dtype=float)
            bench_parts.append((period * b).sum(axis=1))

    # last leg
    if len(rebal_dates) >= 1 and prev_w is not None:
        dt_last = rebal_dates[-1]
        tail = rets.loc[dt_last:].iloc[1:]
        if len(tail) > 0:
            port_parts.append((tail * prev_w).sum(axis=1))
            if bench is not None:
                b = np.array([bench.weights.get(t, 0.0) for t in rets.columns], dtype=float)
                bench_parts.append((tail * b).sum(axis=1))

    port = pd.concat(port_parts).sort_index() if port_parts else pd.Series(dtype=float)
    bench_rets = pd.concat(bench_parts).sort_index() if bench_parts else None

    # metrics
    cum = (1 + port).cumprod()
    ar = annualize_return(port) if len(port) else float("nan")
    av = annualize_vol(port) if len(port) else float("nan")
    sharpe = (ar - cfg.risk_free_rate) / (av + 1e-12) if len(port) else float("nan")
    mdd = max_drawdown(cum) if len(port) else float("nan")

    if not silent:
        print("Performance (Portfolio)")
        print(f"  Annualized Return:  {ar:.2%}")
        print(f"  Annualized Vol:     {av:.2%}")
        print(f"  Sharpe (rf={cfg.risk_free_rate:.2%}): {sharpe:.2f}")
        print(f"  Max Drawdown:       {mdd:.2%}")

    if bench_rets is not None and len(bench_rets) > 0 and not silent:
        cum_b = (1 + bench_rets).cumprod()
        ar_b = annualize_return(bench_rets)
        av_b = annualize_vol(bench_rets)
        sharpe_b = (ar_b - cfg.risk_free_rate) / (av_b + 1e-12)
        mdd_b = max_drawdown(cum_b)
        print("\nPerformance (Benchmark)")
        print(f"  Annualized Return:  {ar_b:.2%}")
        print(f"  Annualized Vol:     {av_b:.2%}")
        print(f"  Sharpe:             {sharpe_b:.2f}")
        print(f"  Max Drawdown:       {mdd_b:.2%}")

    # plots
    if do_plots and len(port) > 0:
        plt.figure(figsize=(10, 5))
        cum.plot(label="Portfolio")
        if bench_rets is not None and len(bench_rets) > 0:
            (1 + bench_rets).cumprod().plot(label="Benchmark")
        plt.legend()
        plt.title("Cumulative Growth (Simple Returns)")
        plt.tight_layout()
        plt.show()

    weights_df = (
        pd.concat(weights_hist, axis=1).T.set_index(pd.DatetimeIndex(dates_hist))
        if weights_hist else pd.DataFrame()
    )
    if do_plots and not weights_df.empty:
        weights_df.plot.area(figsize=(10, 5))
        plt.title("Weights over time (Weekly Rebalance)")
        plt.tight_layout()
        plt.show()

    # export
    if do_export and cfg.export_csv and len(port) > 0:
        out = pd.DataFrame({"date": port.index, "port_ret": port.values}).set_index("date")
        out.to_csv(cfg.export_csv)
        if not silent:
            print(f"Saved daily returns to {cfg.export_csv}")

    return {
        "returns": port,
        "benchmark_returns": bench_rets,
        "weights": weights_df,
        "metrics": {"ar": ar, "av": av, "sharpe": sharpe, "mdd": mdd}
    }

# ----------------------------
# Interactive helpers
# ----------------------------
def _ask_yes_no(prompt: str, default: str = "y") -> bool:
    d = default.lower()
    while True:
        ans = input(f"{prompt} [{'Y/n' if d=='y' else 'y/N'}]: ").strip().lower()
        if ans == "" and d in ("y", "n"):
            return d == "y"
        if ans in ("y", "yes", "o", "oui"):
            return True
        if ans in ("n", "no", "non"):
            return False
        print("Réponds par y/n.")

def _ask_float(prompt: str, default: Optional[float]) -> Optional[float]:
    txt_default = "" if default is None else f" (default {default})"
    ans = input(f"{prompt}{txt_default} (vide = None): ").strip()
    if ans == "":
        return default
    try:
        val = float(ans)
        return val
    except:
        print("Valeur invalide, garde le défaut.")
        return default

def _ask_tuple_minmax(name: str, default_min: Optional[float], default_max: Optional[float]) -> Optional[Tuple[float,float]]:
    ans_min = input(f"{name} - borne MIN (vide pour None, défaut={default_min}): ").strip()
    ans_max = input(f"{name} - borne MAX (vide pour None, défaut={default_max}): ").strip()
    if ans_min == "" and ans_max == "":
        return (default_min, default_max) if (default_min is not None or default_max is not None) else None
    try:
        vmin = float(ans_min) if ans_min != "" else default_min
        vmax = float(ans_max) if ans_max != "" else default_max
        if vmin is None or vmax is None:
            return None
        return (vmin, vmax)
    except:
        print("Bornes invalides, on ignore ce groupe.")
        return None

# ----------------------------
# Main (interactive)
# ----------------------------
def build_universe_interactive() -> Tuple[Universe, Benchmark]:
    use_eu = _ask_yes_no("Utiliser des ETF UCITS Europe (Sinon: ETF US)?", default="n")

    if use_eu:
        equity = ["CSPX.L", "VEVE.L", "EIMI.L"]
        bonds = ["IEAC.L", "IGLA.L"]
        commodities = ["SGLN.L", "CMOD.L"]
        bench = Benchmark(weights={"CSPX.L": 0.60, "IEAC.L": 0.40})
    else:
        equity = ["SPY", "VEA", "EEM"]
        bonds = ["AGG", "BNDX"]
        commodities = ["GLD", "DBC"]
        bench = Benchmark(weights={"SPY": 0.60, "AGG": 0.40})

    tickers = equity + bonds + commodities
    groups = {"Equity": equity, "Bonds": bonds, "Commodities": commodities}
    universe = Universe(tickers=tickers, groups=groups)
    return universe, bench

def build_constraints_interactive(groups: Dict[str, List[str]]) -> Constraints:
    print("\n--- Paramétrage des contraintes ---")
    long_only = _ask_yes_no("Long-only ?", default="y")
    weight_cap = _ask_float("Poids max par actif (ex: 0.5) ?", default=0.4)

    print("Bornes par groupe (laisser vide pour ignorer un groupe).")
    gcaps: Dict[str, Tuple[float, float]] = {}
    for g in ["Equity", "Bonds", "Commodities"]:
        if g in groups:
            t = _ask_tuple_minmax(f"{g}", default_min=None, default_max=None)
            if t is not None:
                gcaps[g] = t
    group_caps = gcaps if len(gcaps) > 0 else None

    target_vol = _ask_float("Cible de volatilité annuelle (ex: 0.12). Laisser vide pour None", default=0.12)
    te_max = _ask_float("Tracking error max annuel vs benchmark (ex: 0.05). Vide pour None", default=None)
    turnover_cap = _ask_float("Turnover max par rebalance (somme |Δw|). Vide pour None", default=0.5)
    esg_min = _ask_float("ESG moyen minimum [0-100]. Vide pour None", default=None)

    return Constraints(
        long_only=long_only,
        weight_cap=weight_cap,
        group_caps=group_caps,
        target_vol=target_vol,
        te_max=te_max,
        turnover_cap=turnover_cap,
        esg_min=esg_min,
    )

def build_config_interactive() -> BacktestConfig:
    print("\n--- Paramétrage backtest ---")
    start = input("Start date (YYYY-MM-DD) [default 2015-01-01]: ").strip() or "2015-01-01"
    end = input("End date (YYYY-MM-DD) [empty = today]: ").strip() or None

    mu_model = input("Modèle de μ ('mean' ou 'momentum') [momentum]: ").strip().lower() or "momentum"
    if mu_model not in ("mean", "momentum"):
        mu_model = "momentum"

    mode = input("Mode ('constrained' | 'max_perf' | 'ts_mom' | 'static_60_40') [constrained]: ").strip().lower() or "constrained"
    if mode not in ("constrained", "max_perf", "ts_mom", "static_60_40"):
        mode = "constrained"

    objective = input("Objectif ('min_var'|'max_sharpe') [max_sharpe]: ").strip().lower() or "max_sharpe"
    if objective not in ("min_var", "max_sharpe"):
        objective = "max_sharpe"

    export_csv = input("Exporter CSV des retours ? Nom de fichier (vide = pas d'export) [daily_returns.csv]: ").strip()
    export_csv = export_csv if export_csv != "" else "daily_returns.csv"

    return BacktestConfig(
        start=start, end=end, mu_model=mu_model, mode=mode,
        objective=objective, export_csv=export_csv
    )

# ----------------------------
# Runner
# ----------------------------
if __name__ == "__main__":
    print("=== Multi-Asset Portfolio Optimizer (Interactive) — Daily prices, Weekly rebalance ===")
    universe, bench = build_universe_interactive()
    cons = build_constraints_interactive(universe.groups)
    cfg = build_config_interactive()

    # (Optionnel) scores ESG
    esg_scores = None
    # try:
    #     esg_df = pd.read_csv("esg_scores.csv")
    #     esg_scores = dict(zip(esg_df["ticker"], esg_df["esg_score"]))

    # Backtest final (avec plots/exports)
    results = backtest(
        universe=universe,
        constraints=cons,
        bench=bench,
        cfg=cfg,
        esg_scores=esg_scores,
        silent=False,
        do_plots=True,
        do_export=True,
    )

    if not results["weights"].empty:
        print("\nDernier snapshot de poids:\n", results["weights"].iloc[-1].round(4))
