"""
Multi-Asset Portfolio Optimizer 1 (Equities/Bonds/Commodities)
===========================================================

Purpose
-------
A concise, production-style Python script that:
  • Downloads ETF price data
  • Computes returns and a risk model (EWMA covariance)
  • Solves a constrained portfolio optimization with "cvxpy"
  • Supports constraints: long-only, weight caps, target volatility, tracking error vs benchmark,
    group (asset-class) bounds, turnover, and ESG threshold (optional)
  • Backtests with monthly rebalancing
  • Plots performance and exports results

Notes
-----
- Default universe uses US-listed ETFs for simplicity. Replace tickers with UCITS equivalents if needed.
- ESG scores can be provided via an external CSV (ticker,esg_score in [0,100]).
- This file is self-contained; install deps and libraries and run directly.

Dependencies
------------
  pip install pandas numpy yfinance matplotlib cvxpy

Author: Tanguy Moncler
License: MIT
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
# Configuration dataclasses
# ----------------------------
@dataclass
class Universe:
    tickers: List[str]
    groups: Dict[str, List[str]]  # {"Equity": ["SPY",...], "Bonds": [...], "Commodities": [...]} 

@dataclass
class Constraints:
    long_only: bool = True
    weight_cap: float = 0.4                 # per-asset cap
    group_caps: Optional[Dict[str, Tuple[float, float]]] = None  # {group: (min,max)}
    target_vol: Optional[float] = 0.10      # annualized target volatility (e.g., 10%)
    te_max: Optional[float] = None          # annualized tracking error max vs benchmark
    turnover_cap: Optional[float] = 0.5     # max sum(|w_t - w_{t-1}|) at rebalance
    esg_min: Optional[float] = None         # min ESG average (0-100)

@dataclass
class BacktestConfig:
    start: str = "2015-01-01"
    end: Optional[str] = None
    rebalance: str = "M"                 # 'M' for month-end
    lookback_months: int = 24            # estimation window
    halflife_months: int = 6             # EWMA half-life for covariance
    risk_free_rate: float = 0.0          # annualized
    objective: str = "max_sharpe"        # or "min_var" or "mean_var"
    risk_aversion: float = 5.0           # used if objective == 'mean_var'

@dataclass
class Benchmark:
    weights: Dict[str, float]             # e.g., {"SPY":0.6, "AGG":0.4}

# ----------------------------
# Utilities
# ----------------------------
def load_prices(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all").ffill().dropna()
    return data


def compute_returns(prices: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Log returns; reindex to business days for stability."""
    px = prices.asfreq("B").ffill()
    rets = np.log(px/px.shift(1)).dropna()
    return rets


def ewma_cov(returns: pd.DataFrame, halflife_days: int) -> pd.DataFrame:
    """Exponentially-weighted covariance matrix."""
    lam = math.exp(math.log(0.5) / halflife_days)
    w = np.array([lam ** (len(returns)-1 - i) for i in range(len(returns))])
    w = w / w.sum()
    X = returns.values - returns.values.mean(axis=0, keepdims=True)
    cov = X.T @ (w[:, None] * X)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def annualize_return(ret_series: pd.Series, freq: str = "D") -> float:
    if freq == "D":
        periods = 252
    elif freq == "M":
        periods = 12
    else:
        periods = 252
    mu = ret_series.mean() * periods
    return float(mu)


def annualize_vol(ret_series: pd.Series, freq: str = "D") -> float:
    if freq == "D":
        periods = 252
    elif freq == "M":
        periods = 12
    else:
        periods = 252
    vol = ret_series.std(ddof=0) * math.sqrt(periods)
    return float(vol)


def max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.cummax()
    dd = cum_returns/peak - 1.0
    return float(dd.min())


def turnover(w_prev: np.ndarray, w_new: np.ndarray) -> float:
    return float(np.abs(w_new - w_prev).sum())


# ----------------------------
# Optimization core
# ----------------------------

def optimize_weights(
    mu: pd.Series,                   # expected returns (periodic, e.g., daily). We will annualize internally for objective.
    cov: pd.DataFrame,               # covariance on the same periodicity; we'll scale to annual.
    constraints: Constraints,
    benchmark: Optional[Benchmark] = None,
    prev_weights: Optional[np.ndarray] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    freq: str = "D",
    objective: str = "max_sharpe",
    risk_aversion: float = 5.0,
) -> np.ndarray:
    tickers = list(mu.index)
    n = len(tickers)

    # Scale to annual
    if freq == "D":
        scale = 252
    elif freq == "M":
        scale = 12
    else:
        scale = 252

    mu_ann = mu.values * scale
    Sigma_ann = cov.values * scale

    w = cp.Variable(n)

    cons = []
    cons += [cp.sum(w) == 1]
    if constraints.long_only:
        cons += [w >= 0]
    if constraints.weight_cap is not None:
        cons += [w <= constraints.weight_cap]

    # Group bounds
    if groups and constraints.group_caps:
        for g, (gmin, gmax) in constraints.group_caps.items():
            idx = [tickers.index(t) for t in groups.get(g, []) if t in tickers]
            if idx:
                cons += [cp.sum(w[idx]) >= gmin, cp.sum(w[idx]) <= gmax]

    # Target volatility
    if constraints.target_vol is not None:
        cons += [cp.sqrt(cp.quad_form(w, Sigma_ann)) <= constraints.target_vol]

    # Tracking error constraint
    if constraints.te_max is not None and benchmark is not None:
        b = np.array([benchmark.weights.get(t, 0.0) for t in tickers])
        cons += [cp.sqrt(cp.quad_form(w - b, Sigma_ann)) <= constraints.te_max]

    # Turnover constraint (L1)
    if constraints.turnover_cap is not None and prev_weights is not None:
        cons += [cp.norm1(w - prev_weights) <= constraints.turnover_cap]

    # ESG average threshold
    # Expect a series esg_score in [0,100]; we'll pass via mu.name hack or attach externally.
    # Better: set a global or pass as function arg. We'll detect from mu.index name mapping.
    # Here we carry ESG via attribute attached on mu (monkey patch acceptable for script use).
    esg_scores = getattr(mu, "_esg", None)
    if esg_scores is not None and constraints.esg_min is not None:
        s = np.array([esg_scores.get(t, np.nan) for t in tickers])
        mask = np.isfinite(s)
        if mask.sum() > 0:
            # Weighted average ESG >= esg_min
            cons += [cp.sum(cp.multiply(w[mask], s[mask])) >= constraints.esg_min * cp.sum(w[mask])]

    # Objective
    # max Sharpe ~ maximize mu^T w / sqrt(w^T Sigma w) -> non-convex.
    # We approximate: maximize (mu^T w - (gamma/2) w^T Sigma w) with gamma chosen by risk_aversion or
    # use target_vol constraint above which convexifies.
    if objective == "min_var":
        obj = cp.Minimize(cp.quad_form(w, Sigma_ann))
    elif objective == "mean_var":
        obj = cp.Minimize(risk_aversion * cp.quad_form(w, Sigma_ann) - mu_ann @ w)
    else:  # "max_sharpe" surrogate under vol constraint
        # With target_vol constraint, maximizing expected return is convex (linear objective)
        obj = cp.Maximize(mu_ann @ w)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        # Try another solver fallback
        prob.solve(solver=cp.ECOS, verbose=False)
    if w.value is None:
        raise RuntimeError("Optimization failed to find a solution.")

    sol = np.array(w.value).round(10)
    # Normalize in case of tiny numerical drift
    sol = np.clip(sol, 0, 1)
    if sol.sum() > 0:
        sol = sol / sol.sum()
    return sol


# ----------------------------
# Backtest
# ----------------------------

def backtest(
    universe: Universe,
    constraints: Constraints,
    bench: Optional[Benchmark],
    cfg: BacktestConfig,
    esg_scores: Optional[Dict[str, float]] = None,
    export_csv: Optional[str] = None,
):
    prices = load_prices(universe.tickers, cfg.start, cfg.end)
    rets = compute_returns(prices, freq="D")

    # Rebalance dates (month-end business days)
    rebal_dates = rets.resample(cfg.rebalance).last().index

    weights_hist = []
    dates_hist = []
    port_rets = []
    bench_rets = []

    prev_w = None

    for dt in rebal_dates:
        # Estimation window
        lb_end = dt
        lb_start = dt - pd.DateOffset(months=cfg.lookback_months)
        window = rets.loc[lb_start:lb_end].dropna()
        if len(window) < 60:  # skip until we have enough data
            continue

        # Risk model
        halflife_days = int(cfg.halflife_months * 21)
        Sigma = ewma_cov(window, halflife_days=halflife_days)
        mu = window.mean()  # simple historical mean (daily)

        # Attach ESG if provided
        if esg_scores:
            setattr(mu, "_esg", esg_scores)

        # Optimize
        w_opt = optimize_weights(
            mu=mu,
            cov=Sigma,
            constraints=constraints,
            benchmark=bench,
            prev_weights=prev_w,
            groups=universe.groups,
            freq="D",
            objective=cfg.objective,
            risk_aversion=cfg.risk_aversion,
        )

        # Record and simulate next period returns until next rebalance
        dates_hist.append(dt)
        weights_hist.append(pd.Series(w_opt, index=rets.columns))
        prev_w = w_opt

        # Next period window
        next_dt = (rets.index[rets.index.get_loc(dt, method="pad")] + pd.offsets.MonthEnd(1))
        period = rets.loc[dt:next_dt].iloc[1:]  # exclude dt itself
        if len(period) == 0:
            continue
        # Portfolio daily ret
        pr = (period * w_opt).sum(axis=1)
        port_rets.append(pr)

        # Benchmark
        if bench is not None:
            b = np.array([bench.weights.get(t, 0.0) for t in rets.columns])
            br = (period * b).sum(axis=1)
            bench_rets.append(br)

    # Concatenate
    port_rets = pd.concat(port_rets).sort_index() if port_rets else pd.Series(dtype=float)
    bench_rets = pd.concat(bench_rets).sort_index() if bench_rets else None

    weights_df = (
        pd.concat(weights_hist, axis=1).T.set_index(pd.DatetimeIndex(dates_hist))
        if weights_hist else pd.DataFrame()
    )

    # Performance metrics
    cum = (1 + port_rets).cumprod()
    ar = annualize_return(port_rets)
    av = annualize_vol(port_rets)
    sharpe = (ar - cfg.risk_free_rate) / (av + 1e-12)
    mdd = max_drawdown(cum)

    print("Performance (Portfolio)")
    print(f"  Annualized Return:  {ar:.2%}")
    print(f"  Annualized Vol:     {av:.2%}")
    print(f"  Sharpe (rf={cfg.risk_free_rate:.2%}): {sharpe:.2f}")
    print(f"  Max Drawdown:       {mdd:.2%}")

    if bench_rets is not None:
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

    # Plots
    plt.figure(figsize=(10, 5))
    cum.plot(label="Portfolio")
    if bench_rets is not None:
        (1 + bench_rets).cumprod().plot(label="Benchmark")
    plt.legend()
    plt.title("Cumulative Growth")
    plt.tight_layout()
    plt.show()

    if not weights_df.empty:
        (weights_df).plot.area(figsize=(10, 5))
        plt.title("Weights over time")
        plt.tight_layout()
        plt.show()

    if export_csv:
        out = pd.DataFrame({
            "date": port_rets.index,
            "port_ret": port_rets.values,
        }).set_index("date")
        out.to_csv(export_csv)
        print(f"Saved daily returns to {export_csv}")

    return {
        "returns": port_rets,
        "benchmark_returns": bench_rets,
        "weights": weights_df,
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # --- Define universe ---
    # Replace with UCITS/Europe-friendly tickers if desired (e.g., CSPX, IEAC, SGLN, etc.)
    equity = ["SPY", "VEA", "EEM"]          # US, Dev ex-US, EM
    bonds = ["AGG", "BNDX"]                  # US Aggregate, Global ex-US bonds
    commodities = ["GLD", "DBC"]             # Gold, Broad commodities

    tickers = equity + bonds + commodities
    groups = {
        "Equity": equity,
        "Bonds": bonds,
        "Commodities": commodities,
    }

    universe = Universe(tickers=tickers, groups=groups)

    # --- Constraints ---
    cons = Constraints(
        long_only=True,
        weight_cap=0.5,
        group_caps={
            "Equity": (0.30, 0.70),
            "Bonds": (0.20, 0.70),
            "Commodities": (0.00, 0.20),
        },
        target_vol=0.10,     # 10% annual vol target
        te_max=0.05,         # 5% tracking error vs benchmark (optional)
        turnover_cap=0.6,    # 60% max turnover per rebalance
        esg_min=None,        # set e.g. 60 to enforce avg ESG >= 60
    )

    # --- Benchmark 60/40 ---
    bench_w = {"SPY": 0.60, "AGG": 0.40}
    bench = Benchmark(weights=bench_w)

    # --- Backtest config ---
    cfg = BacktestConfig(
        start="2015-01-01",
        end=None,
        rebalance="M",
        lookback_months=24,
        halflife_months=6,
        risk_free_rate=0.01,
        objective="max_sharpe",  # or 'min_var' / 'mean_var'
        risk_aversion=5.0,
    )

    # --- Optional: load ESG scores from CSV ---
    # Expect a CSV with columns: ticker,esg_score (0-100)
    esg_scores = None
    # try:
    #     esg_df = pd.read_csv("esg_scores.csv")
    #     esg_scores = dict(zip(esg_df["ticker"], esg_df["esg_score"]))
    # except Exception:
    #     esg_scores = None

    # --- Run backtest ---
    results = backtest(
        universe=universe,
        constraints=cons,
        bench=bench,
        cfg=cfg,
        esg_scores=esg_scores,
        export_csv="daily_returns.csv",
    )

    # Print last weights snapshot
    if not results["weights"].empty:
        print("\nLast Weights:\n", results["weights"].iloc[-1].round(4))
