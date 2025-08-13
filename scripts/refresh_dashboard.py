from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.reportিং.dashboard_writer import build_dashboard
from tools.audit_workbook import audit_workbook
from tools.curate_charts import curate_plan
from src.models.portfolio import PortfolioConstructor, OptimizationMethod


def _compute_regime_info(regime_df: pd.DataFrame) -> Dict:
    info: Dict = {"label": None, "months": 0, "proba": {}}
    if regime_df is None or regime_df.empty:
        return info
    if "Regime_Ensemble" in regime_df.columns:
        info["label"] = str(regime_df["Regime_Ensemble"].iloc[-1])
        rev = regime_df["Regime_Ensemble"][::-1]
        m = 0
        for v in rev:
            if str(v) == info["label"]:
                m += 1
            else:
                break
        info["months"] = m
    prob_cols = [c for c in regime_df.columns if isinstance(c, str) and c.startswith("Ensemble_Prob_")]
    if prob_cols:
        probs = regime_df[prob_cols].iloc[-1].astype(float)
        info["proba"] = {k: float(v) for k, v in probs.items()}
    return info


def _top_features(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly is None or monthly.empty:
        return pd.DataFrame()
    z = monthly.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    last = z.iloc[-1].abs().sort_values(ascending=False).head(5)
    return pd.DataFrame({"z": last})


def _optimal_allocation(returns: pd.DataFrame, regime_df: pd.DataFrame) -> pd.Series:
    if returns is None or returns.empty or regime_df is None or regime_df.empty:
        return pd.Series(dtype=float)
    if "Regime_Ensemble" not in regime_df.columns:
        return pd.Series(dtype=float)
    cur = str(regime_df["Regime_Ensemble"].iloc[-1])
    merged = returns.join(regime_df[["Regime_Ensemble"]], how="inner")
    sub = merged[merged["Regime_Ensemble"].astype(str) == cur].drop(columns=["Regime_Ensemble"]).dropna(how="all")
    if len(sub) < 12 or sub.shape[1] < 3:
        return pd.Series(dtype=float)
    mean_ret = pd.to_numeric(sub.mean(), errors="coerce")
    cov = sub.cov()
    stats = {cur: {"mean_returns": mean_ret, "covariance": cov, "periods_per_year": 12}}
    pc = PortfolioConstructor()
    res = pc.optimize_portfolio(stats, cur, method=OptimizationMethod.SHARPE)
    return res.weights.sort_values(ascending=False)


def _factor_heatmap(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly is None or monthly.empty:
        return pd.DataFrame()
    cols = [c for c in monthly.columns if str(c).startswith("F_")]
    if not cols:
        # fallback on key themes if present
        cols = [c for c in monthly.columns if c in ("F_Growth", "F_Inflation", "F_Risk", "F_Housing", "F_FX")]
    return monthly[cols].tail(24)


def _ytd_pnl(returns: pd.DataFrame, alloc: pd.Series) -> pd.DataFrame:
    if returns is None or returns.empty or alloc is None or alloc.empty:
        return pd.DataFrame()
    year = returns.index.max().year
    ytd = returns[returns.index.year == year].fillna(0.0)
    w = alloc.reindex(ytd.columns).fillna(0.0)
    rp = (ytd.dot(w)).add(1).cumprod() - 1
    eq = (ytd.mean(axis=1)).add(1).cumprod() - 1
    return pd.DataFrame({"Regime Portfolio": rp, "Equal-Weight": eq})


def main():
    p = argparse.ArgumentParser(description="Refresh curated macro dashboard")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--monthly", dest="monthly", required=True)
    p.add_argument("--regimes", dest="regimes", required=True)
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    tmp_audit = Path("Output/audit")
    tmp_audit.mkdir(parents=True, exist_ok=True)

    # 1) Audit
    audit_workbook(inp, tmp_audit, Path(args.monthly))

    # 2) Curate
    plan_path = tmp_audit / "plan.json"
    plan = curate_plan(tmp_audit / "charts.csv", Path(args.monthly))
    plan_path.write_text(json.dumps(plan, indent=2))

    # 3) Copy workbook keeping VBA
    wb = load_workbook(filename=str(inp), keep_vba=True)

    # Optionally remove charts based on plan (skip for safety; left as future work)

    # 4) Compute content
    monthly_df = pd.read_csv(args.monthly, index_col=0, parse_dates=[0])
    regime_df = pd.read_csv(args.regimes, index_col=0, parse_dates=[0])
    regime_info = _compute_regime_info(regime_df)
    top = _top_features(monthly_df)
    alloc = _optimal_allocation(monthly_df.select_dtypes("number"), regime_df)
    heat = _factor_heatmap(monthly_df)
    pnl = _ytd_pnl(monthly_df.select_dtypes("number"), alloc)

    # 5) Build new dashboard
    build_dashboard(wb, regime_info, top, alloc, heat, pnl)

    wb.save(str(out))
    print(f"Saved cleaned workbook to {out}")


if __name__ == "__main__":
    main()


