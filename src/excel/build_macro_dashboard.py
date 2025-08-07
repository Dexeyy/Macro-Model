"""
Builds visual, insightful overview pages for the Excel macro-regime dashboard.

Pip dependencies (add to requirements.txt if missing):
  - pandas
  - numpy
  - matplotlib
  - openpyxl
  - pillow

This script:
  - Loads the existing workbook
  - For each theme sheet, moves the overview table to J3 and inserts PNG charts
  - Adds a correlation heatmap on Dashboard if possible
  - Adds a simple history chart on Regime Labels if possible

It is non-destructive to raw data (writes to new ranges only) and skips optional
charts gracefully when inputs are unavailable.
"""
from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.worksheet.worksheet import Worksheet
from .excel_live import group_features  # reuse grouping rules
from openpyxl.styles import Font


THEME_SHEETS: List[str] = [
    "Growth & Labour",
    "Inflation & Liquidity",
    "Credit & Risk",
    "Housing",
    "FX & Commodities",
]


# Theme identity: primary colors
THEME_COLOR: Dict[str, str] = {
    "Growth & Labour": "#1f77b4",       # blue
    "Inflation & Liquidity": "#d62728",  # red
    "Credit & Risk": "#ff7f0e",         # orange
    "Housing": "#2ca02c",               # green
    "FX & Commodities": "#9467bd",      # purple
}


# Optional component lists and weights per theme (edit as needed)
COMPONENT_SERIES: Dict[str, List[str]] = {
    "Growth & Labour": ["GDP_YoY", "IP_YoY", "UNRATE", "PAYEMS_YoY", "PMI"]
}
WEIGHTS: Dict[str, Dict[str, float]] = {
    "Growth & Labour": {"GDP_YoY": 0.3, "IP_YoY": 0.2, "UNRATE": 0.2, "PAYEMS_YoY": 0.2, "PMI": 0.1}
}


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("&", "and").replace("/", "_")


def load_workbook_safe(path: str | os.PathLike):
    path_str = str(path)
    keep_vba = path_str.lower().endswith(".xlsm")
    return load_workbook(filename=path_str, keep_vba=keep_vba)


def _read_table_from_sheet(ws: Worksheet) -> Optional[pd.DataFrame]:
    """Attempt to read an existing small table that contains a ThemeComposite.

    Strategy: scan first 200 rows and 40 columns; find header row containing
    ThemeComposite; build a DataFrame until first all-empty row below.
    """
    max_rows, max_cols = 200, 40
    # Build a 2D list of strings
    values = [[ws.cell(r, c).value for c in range(1, max_cols + 1)] for r in range(1, max_rows + 1)]
    header_row = None
    for r in range(len(values)):
        row = values[r]
        if row and any((isinstance(v, str) and "themecomposite" in v.lower()) for v in row if v):
            header_row = r
            break
    if header_row is None:
        return None
    # Determine last column via last non-empty header cell
    headers = values[header_row]
    last_col = max(i for i, v in enumerate(headers, start=1) if v is not None)
    data_rows: List[List[object]] = []
    for r in range(header_row + 1, len(values)):
        row = values[r][: last_col]
        if all(v is None for v in row):
            break
        data_rows.append(row)
    if not data_rows:
        return None
    # Fallback date header name
    headers = [h if h is not None else "Date" for h in headers[: last_col]]
    df = pd.DataFrame(data_rows, columns=headers)
    # Try parse first column as dates
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    except Exception:
        pass
    return df


def load_theme_df(wb, sheetname: str) -> Optional[pd.DataFrame]:
    ws: Worksheet = wb[sheetname]
    df = _read_table_from_sheet(ws)
    if df is None:
        return None
    # Ensure required columns exist
    needed = {"ThemeComposite", "ThemeComposite_3M_MA", "ThemeComposite_YoY"}
    if not needed.intersection(set(df.columns)):
        return None
    return df


def _write_df(ws: Worksheet, start_cell: str, df: pd.DataFrame) -> None:
    # Write headers and values starting at start_cell
    # Parse start_cell like 'J3'
    col_letters = ''.join([ch for ch in start_cell if ch.isalpha()])
    row_num = int(''.join([ch for ch in start_cell if ch.isdigit()]))
    # Convert column letters to index
    def col_to_idx(col: str) -> int:
        idx = 0
        for ch in col:
            idx = idx * 26 + (ord(ch.upper()) - 64)
        return idx
    c0 = col_to_idx(col_letters)
    # Header
    ws.cell(row=row_num, column=c0, value=df.index.name or "Date")
    for j, h in enumerate(df.columns, start=1):
        ws.cell(row=row_num, column=c0 + j, value=h)
    # Body
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        ws.cell(row=row_num + i, column=c0, value=idx)
        for j, v in enumerate(row, start=1):
            ws.cell(row=row_num + i, column=c0 + j, value=None if pd.isna(v) else float(v))


def _load_monthly_df_from_disk() -> Optional[pd.DataFrame]:
    candidates = [
        Path("Data/processed/merged.parquet"),
        Path("Data/processed/macro_data_featured.csv"),
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, index_col=0, parse_dates=[0])
            return df
    return None


def _compute_theme_overview_from_dump(theme: str, monthly_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        groups = group_features(monthly_df)
        cols = groups.get(theme, [])
        if not cols:
            return None
        sub = monthly_df[cols].dropna(how="all")
        if sub.empty:
            return None
        z = sub.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
        comp = z.mean(axis=1)
        out = pd.DataFrame({
            "ThemeComposite": comp,
        })
        out["ThemeComposite_3M_MA"] = out["ThemeComposite"].rolling(3, min_periods=1).mean()
        out["ThemeComposite_YoY"] = out["ThemeComposite"].pct_change(12)
        out.index.name = monthly_df.index.name or "Date"
        return out
    except Exception:
        return None


@dataclass
class ThemeMetrics:
    latest: float
    chg_3m: float
    yoy: float
    percentile: pd.Series
    zscore: pd.Series
    roll_vol_12m: pd.Series
    slope_6m: float


def compute_metrics(df: pd.DataFrame) -> ThemeMetrics:
    comp = df["ThemeComposite"].astype(float)
    # 3m change computed on composite
    chg_3m = comp.diff(3).iloc[-1] if len(comp) >= 4 else np.nan
    latest = comp.iloc[-1]
    yoy = df["ThemeComposite_YoY"].astype(float).iloc[-1] if "ThemeComposite_YoY" in df.columns else np.nan
    # Percentile of entire history, each point = rank percentile 0..100
    ranks = comp.rank(method="average")
    percentile = 100 * ranks / ranks.max()
    # Rolling z-score vs trailing 36 months (fallback 24)
    window = 36 if len(comp) >= 36 else 24 if len(comp) >= 24 else max(6, len(comp))
    roll_mean = comp.rolling(window, min_periods=max(3, window // 3)).mean()
    roll_std = comp.rolling(window, min_periods=max(3, window // 3)).std(ddof=0)
    zscore = (comp - roll_mean) / (roll_std.replace(0, np.nan))
    # Rolling vol 12m
    roll_vol_12m = comp.rolling(12, min_periods=6).std(ddof=0)
    # 6m slope (simple linear trend)
    slope_6m = np.nan
    if len(comp) >= 6:
        y = comp.iloc[-6:]
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y.values, 1)
        slope_6m = float(coeffs[0])
    return ThemeMetrics(
        latest=float(latest),
        chg_3m=float(chg_3m) if pd.notna(chg_3m) else np.nan,
        yoy=float(yoy) if pd.notna(yoy) else np.nan,
        percentile=percentile,
        zscore=zscore,
        roll_vol_12m=roll_vol_12m,
        slope_6m=slope_6m,
    )


def _save_plot(fig, outdir: Path, filename: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / filename
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_plot_trend(theme: str, df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    try:
        color = THEME_COLOR.get(theme, "#1f77b4")
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        df["ThemeComposite"].plot(ax=ax, color=color, linewidth=1.8, label="Composite")
        if "ThemeComposite_3M_MA" in df.columns:
            df["ThemeComposite_3M_MA"].plot(ax=ax, color="#999999", linewidth=1.5, linestyle="--", label="3M MA")
        ax.set_title(f"{theme} Composite — Trend")
        ax.set_ylabel("Index")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        return _save_plot(fig, outdir, f"{_slug(theme)}_trend.png")
    except Exception:
        return None


def render_plot_momentum(theme: str, df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    try:
        metrics = compute_metrics(df)
        fig, ax1 = plt.subplots(figsize=(7.4, 3.8))
        # Left: YoY
        if "ThemeComposite_YoY" in df.columns:
            df["ThemeComposite_YoY"].plot(ax=ax1, color="#1f77b4", label="YoY")
            ax1.set_ylabel("YoY")
        ax2 = ax1.twinx()
        metrics.percentile.plot(ax=ax2, color="#ff7f0e", label="Percentile (0-100)")
        ax2.set_ylabel("Percentile")
        # Legend
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax1.set_title(f"{theme} Momentum & Percentile")
        ax1.grid(True, alpha=0.25)
        return _save_plot(fig, outdir, f"{_slug(theme)}_momentum.png")
    except Exception:
        return None


def render_plot_deviation(theme: str, df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    try:
        metrics = compute_metrics(df)
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        metrics.zscore.plot(ax=ax, color="#2ca02c", label="Z-score (36m)")
        metrics.roll_vol_12m.plot(ax=ax, color="#d62728", linewidth=1.0, label="12m Vol")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{theme} Deviation & Volatility")
        ax.grid(True, alpha=0.25)
        ax.legend()
        return _save_plot(fig, outdir, f"{_slug(theme)}_deviation.png")
    except Exception:
        return None


def render_plot_contributions(theme: str, theme_df: pd.DataFrame, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    series_list = COMPONENT_SERIES.get(theme)
    if not series_list or dump_df is None:
        return None
    try:
        # Align component series on the theme_df index
        comp = {}
        for s in series_list:
            if s in dump_df.columns:
                comp[s] = pd.to_numeric(dump_df[s], errors="coerce").reindex(theme_df.index)
        if not comp:
            return None
        comp_df = pd.DataFrame(comp)
        z = comp_df.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
        weights = WEIGHTS.get(theme, {c: 1.0 / len(z.columns) for c in z.columns})
        w = pd.Series({c: weights.get(c, 0.0) for c in z.columns})
        contrib = z.mul(w, axis=1)
        contrib_sum = contrib.sum(axis=1)

        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        contrib.tail(120).plot(kind="bar", stacked=True, ax=ax, width=1.0, alpha=0.85)
        theme_df["ThemeComposite"].tail(120).plot(ax=ax, color="black", linewidth=1.5, label="Composite")
        ax.set_title(f"Contributions to {theme} Composite")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        return _save_plot(fig, outdir, f"{_slug(theme)}_contrib.png")
    except Exception:
        return None


def render_plot_diffusion(theme: str, dump_df: Optional[pd.DataFrame], outdir: Path, index: pd.Index) -> Optional[Path]:
    series_list = COMPONENT_SERIES.get(theme)
    if not series_list or dump_df is None:
        return None
    try:
        comp = {}
        for s in series_list:
            if s in dump_df.columns:
                comp[s] = pd.to_numeric(dump_df[s], errors="coerce")
        if not comp:
            return None
        df = pd.DataFrame(comp).reindex(index)
        mom = df.diff(1)
        diffusion = (mom.gt(0).sum(axis=1) / mom.shape[1]) * 100.0
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        diffusion.plot(ax=ax, color="#9467bd")
        ax.axhline(50, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 100)
        ax.set_title(f"{theme} Diffusion (m/m improving %)")
        ax.grid(True, alpha=0.25)
        return _save_plot(fig, outdir, f"{_slug(theme)}_diffusion.png")
    except Exception:
        return None


def insert_image(ws: Worksheet, image_path: Path, anchor_cell: str) -> None:
    try:
        img = XLImage(str(image_path))
        ws.add_image(img, anchor_cell)
    except Exception:
        pass


def _save_text_card(title: str, outdir: Path, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.axis('off')
    ax.text(0.02, 0.6, title, fontsize=12, fontweight='bold')
    ax.text(0.02, 0.3, "Data not available", fontsize=11)
    return _save_plot(fig, outdir, filename)


def _get_series(dump_df: Optional[pd.DataFrame], index: pd.Index, patterns: List[str]) -> Optional[pd.Series]:
    if dump_df is None or dump_df.empty:
        return None
    cols = [c for c in dump_df.columns if isinstance(c, str)]
    lowered = {c.lower(): c for c in cols}
    for pat in patterns:
        pat_l = pat.lower()
        # exact match first
        if pat_l in lowered:
            s = pd.to_numeric(dump_df[lowered[pat_l]], errors='coerce').reindex(index)
            if s.notna().any():
                return s
        # substring match
        for lc, orig in lowered.items():
            if pat_l in lc:
                s = pd.to_numeric(dump_df[orig], errors='coerce').reindex(index)
                if s.notna().any():
                    return s
    return None


def _lines_plot(x: pd.Index, lines: Dict[str, pd.Series], title: str, colors: Optional[Dict[str, str]], outdir: Path, fname: str, ylabel: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for name, s in lines.items():
        if s is None or s.dropna().empty:
            continue
        color = colors.get(name) if colors else None
        ax.plot(x, s.values, label=name, color=color, linewidth=1.6)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    return _save_plot(fig, outdir, fname)


# --------------------- Category-specific renderers ---------------------------

def render_growth_business_cycle_clock(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Business Cycle Clock", outdir, "growth_clock_na.png")
    gdp = _get_series(dump_df, index, ["gdp_yoy", "real_gdp_yoy", "gdpyoy", "% gdp"])
    unemp = _get_series(dump_df, index, ["unrate", "unemployment", "unemployment_rate"])
    if gdp is None or unemp is None:
        return _save_text_card("Business Cycle Clock", outdir, "growth_clock_na.png")
    try:
        # color by time
        t = np.linspace(0, 1, len(index))
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        sc = ax.scatter(gdp.values, unemp.values, c=t, cmap='viridis', s=18)
        ax.set_xlabel('GDP YoY (%)')
        ax.set_ylabel('Unemployment (%)')
        ax.set_title('Business Cycle Clock')
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Time')
        return _save_plot(fig, outdir, "growth_clock.png")
    except Exception:
        return _save_text_card("Business Cycle Clock", outdir, "growth_clock_na.png")


def render_growth_sector_contrib(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("GDP Sector Contributions", outdir, "growth_sector_na.png")
    manu = _get_series(dump_df, index, ["manufacturing", "ip_manuf", "manu_yoy"]) 
    serv = _get_series(dump_df, index, ["services", "services_yoy", "pmi_services"]) 
    cons = _get_series(dump_df, index, ["construction", "construction_yoy"]) 
    have = [("Manufacturing", manu), ("Services", serv), ("Construction", cons)]
    have = [(n, s) for n, s in have if s is not None and s.dropna().any()]
    if not have:
        return _save_text_card("GDP Sector Contributions", outdir, "growth_sector_na.png")
    try:
        df = pd.DataFrame({n: s for n, s in have}).tail(120)
        # normalize to z and treat as contributions proxy
        z = df.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        z.plot(kind='bar', stacked=True, width=1.0, ax=ax, alpha=0.9, colormap='tab20')
        ax.set_title('GDP Growth — Sector Contributions (proxy)')
        ax.grid(True, axis='y', alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        return _save_plot(fig, outdir, "growth_sector_contrib.png")
    except Exception:
        return _save_text_card("GDP Sector Contributions", outdir, "growth_sector_na.png")


def render_inflation_decomposition(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Inflation Decomposition", outdir, "infl_decomp_na.png")
    food = _get_series(dump_df, index, ["cpi_food", "food_inflation"]) 
    energy = _get_series(dump_df, index, ["cpi_energy", "energy_inflation"]) 
    core_goods = _get_series(dump_df, index, ["core_goods", "cpi_core_goods"]) 
    services = _get_series(dump_df, index, ["services_inflation", "cpi_services"]) 
    have = [("Food", food), ("Energy", energy), ("Core goods", core_goods), ("Services", services)]
    have = [(n, s) for n, s in have if s is not None and s.dropna().any()]
    if not have:
        return _save_text_card("Inflation Decomposition", outdir, "infl_decomp_na.png")
    try:
        df = pd.DataFrame({n: s for n, s in have}).tail(60)
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        df.plot(kind='bar', stacked=True, width=1.0, ax=ax, alpha=0.9, colormap='Set2')
        ax.set_title('Inflation decomposition — CPI contributions (proxy)')
        ax.grid(True, axis='y', alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        return _save_plot(fig, outdir, "infl_decomp.png")
    except Exception:
        return _save_text_card("Inflation Decomposition", outdir, "infl_decomp_na.png")


def render_inflation_real_policy_rate(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Real policy rate", outdir, "infl_realrate_na.png")
    policy = _get_series(dump_df, index, ["fed funds", "policy rate", "ffr", "policy_rate"]) 
    core_cpi = _get_series(dump_df, index, ["core_cpi", "cpi_core", "core inflation"]) 
    if policy is None or core_cpi is None:
        return _save_text_card("Real policy rate", outdir, "infl_realrate_na.png")
    try:
        real = policy - core_cpi
        return _lines_plot(index, {"Real policy rate": real}, "Real policy rate (policy - core CPI)", None, outdir, "infl_realrate.png")
    except Exception:
        return _save_text_card("Real policy rate", outdir, "infl_realrate_na.png")


def render_credit_spreads(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Credit Spreads", outdir, "credit_spreads_na.png")
    ig = _get_series(dump_df, index, ["ig_oas", "ig spread", "baa_aaa_spread", "ig_spread"]) 
    hy = _get_series(dump_df, index, ["hy_oas", "hy spread", "high_yield_spread", "hy_spread"]) 
    have = {"IG": ig, "HY": hy}
    if all(s is None or s.dropna().empty for s in have.values()):
        return _save_text_card("Credit Spreads", outdir, "credit_spreads_na.png")
    colors = {"IG": "#1f77b4", "HY": "#d62728"}
    return _lines_plot(index, have, "Credit Spreads (bps)", colors, outdir, "credit_spreads.png", ylabel="bps")


def render_financial_stress(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Financial Stress Index", outdir, "fsi_na.png")
    fsi = _get_series(dump_df, index, ["financial_stress", "fsi", "stlfsi"]) 
    if fsi is None or fsi.dropna().empty:
        return _save_text_card("Financial Stress Index", outdir, "fsi_na.png")
    return _lines_plot(index, {"FSI": fsi}, "Financial Stress Index", None, outdir, "fsi.png")


def render_housing_affordability(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Housing Affordability", outdir, "housing_aff_na.png")
    pti = _get_series(dump_df, index, ["price_to_income", "home_price_income", "affordability"]) 
    mort = _get_series(dump_df, index, ["mortgage_rate", "30y mortgage", "30y_rate"]) 
    if pti is None and mort is None:
        return _save_text_card("Housing Affordability", outdir, "housing_aff_na.png")
    # twin axis plot
    fig, ax1 = plt.subplots(figsize=(7.4, 3.8))
    if pti is not None:
        ax1.plot(index, pti.values, color="#2ca02c", label="Price-to-income")
        ax1.set_ylabel("Ratio", color="#2ca02c")
    ax2 = ax1.twinx()
    if mort is not None:
        ax2.plot(index, mort.values, color="#ff7f0e", label="Mortgage rate")
        ax2.set_ylabel("%", color="#ff7f0e")
    ax1.set_title("Affordability: price-to-income vs mortgage rate")
    ax1.grid(True, alpha=0.25)
    # legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best', fontsize=8)
    return _save_plot(fig, outdir, "housing_affordability.png")


def render_housing_supply(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Housing Supply", outdir, "housing_supply_na.png")
    permits = _get_series(dump_df, index, ["permits", "building_permits"]) 
    starts = _get_series(dump_df, index, ["housing_starts", "starts"]) 
    completes = _get_series(dump_df, index, ["completions", "housing_completions"]) 
    have = {"Permits": permits, "Starts": starts, "Completions": completes}
    if all(s is None or s.dropna().empty for s in have.values()):
        return _save_text_card("Housing Supply", outdir, "housing_supply_na.png")
    colors = {"Permits": "#1f77b4", "Starts": "#ff7f0e", "Completions": "#2ca02c"}
    return _lines_plot(index, have, "Housing supply pipeline", colors, outdir, "housing_supply.png")


def render_commodities_breakdown(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("Commodity Breakdown", outdir, "fxcmd_commodities_na.png")
    energy = _get_series(dump_df, index, ["energy_index", "energy", "crb_energy"]) 
    metals = _get_series(dump_df, index, ["metals_index", "metals", "crb_metals"]) 
    agri = _get_series(dump_df, index, ["agriculture_index", "agri", "crb_agri"]) 
    have = {"Energy": energy, "Metals": metals, "Agriculture": agri}
    if all(s is None or s.dropna().empty for s in have.values()):
        return _save_text_card("Commodity Breakdown", outdir, "fxcmd_commodities_na.png")
    colors = {"Energy": "#d62728", "Metals": "#9467bd", "Agriculture": "#2ca02c"}
    return _lines_plot(index, have, "Commodities breakdown", colors, outdir, "fxcmd_commodities.png")


def render_reer_trend(index: pd.Index, dump_df: Optional[pd.DataFrame], outdir: Path) -> Optional[Path]:
    if dump_df is None:
        return _save_text_card("REER", outdir, "fxcmd_reer_na.png")
    reer = _get_series(dump_df, index, ["reer", "real effective exchange", "broad_reer"]) 
    if reer is None or reer.dropna().empty:
        return _save_text_card("REER", outdir, "fxcmd_reer_na.png")
    return _lines_plot(index, {"REER": reer}, "Real Effective Exchange Rate (REER)", None, outdir, "fxcmd_reer.png")


def _write_text_summary(ws: Worksheet, cell: str, theme: str, df: pd.DataFrame):
    try:
        metrics = compute_metrics(df)
        direction = "firming" if metrics.slope_6m and metrics.slope_6m > 0 else "softening" if metrics.slope_6m and metrics.slope_6m < 0 else "flat"
        txt = (
            f"Composite at {metrics.latest:.2f}; 3m chg {metrics.chg_3m:+.2f}; "
            f"YoY {metrics.yoy:+.2%}. Momentum over 6m is {direction}."
        )
        ws[cell] = txt
        try:
            ws[cell].font = Font(bold=True, size=14)
        except Exception:
            pass
    except Exception:
        pass


def _data_dump_df(wb) -> Optional[pd.DataFrame]:
    if "Data Dump" not in wb.sheetnames:
        return None
    ws = wb["Data Dump"]
    # Read all used range
    data = []
    for row in ws.iter_rows(values_only=True):
        data.append(list(row))
    # determine header row: look for a row with many non-nulls
    header_idx = None
    for i, row in enumerate(data[:200]):
        if row and sum(x is not None for x in row) >= 3:
            header_idx = i
            break
    if header_idx is None:
        return None
    headers = [h if h is not None else f"col{j}" for j, h in enumerate(data[header_idx])]
    df = pd.DataFrame(data[header_idx + 1 :], columns=headers)
    # try parse first col as date index
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    except Exception:
        pass
    return df


def move_table_right(ws: Worksheet, start_cell: str = "J3") -> Optional[Tuple[int, int]]:
    """Copy the detected existing overview table to start at start_cell and clear old area.

    Returns the size (rows, cols) of the moved table if successful.
    """
    df = _read_table_from_sheet(ws)
    if df is None or df.empty:
        # Try to regenerate overview directly from processed data on disk
        monthly_df = _load_monthly_df_from_disk()
        if monthly_df is not None:
            theme = ws.title
            df = _compute_theme_overview_from_dump(theme, monthly_df)
        if df is None or df.empty:
            return None
    row0 = int(''.join([ch for ch in start_cell if ch.isdigit()]))
    col0 = 10  # J
    # write values with header
    _write_df(ws, start_cell, df)
    # clear old area (left side) under charts to reduce clutter
    for r in range(10, 300):
        for c in range(1, 9):  # columns A..H
            ws.cell(row=r, column=c, value=None)
    return (len(df) + 1, len(df.columns) + 1)


def build_theme_sheet(wb, sheetname: str, outdir: Path) -> None:
    if sheetname not in wb.sheetnames:
        return
    ws: Worksheet = wb[sheetname]
    df = load_theme_df(wb, sheetname)
    if df is None or df.empty:
        # Try rebuild overview from on-disk processed monthly data
        monthly_df = _load_monthly_df_from_disk()
        if monthly_df is not None:
            df = _compute_theme_overview_from_dump(sheetname, monthly_df)
    if df is None or df.empty:
        return

    # Base charts
    p1 = render_plot_trend(sheetname, df, outdir)
    p2 = render_plot_momentum(sheetname, df, outdir)
    p3 = render_plot_deviation(sheetname, df, outdir)

    # Category-specific charts using Data Dump
    dump_df = _data_dump_df(wb)
    if dump_df is None or dump_df.empty:
        dump_df = _load_monthly_df_from_disk()

    extra1 = None
    extra2 = None

    if sheetname == "Growth & Labour":
        extra1 = render_growth_business_cycle_clock(df.index, dump_df, outdir)
        extra2 = render_growth_sector_contrib(df.index, dump_df, outdir)
    elif sheetname == "Inflation & Liquidity":
        extra1 = render_inflation_decomposition(df.index, dump_df, outdir)
        extra2 = render_inflation_real_policy_rate(df.index, dump_df, outdir)
    elif sheetname == "Credit & Risk":
        extra1 = render_credit_spreads(df.index, dump_df, outdir)
        extra2 = render_financial_stress(df.index, dump_df, outdir)
    elif sheetname == "Housing":
        extra1 = render_housing_affordability(df.index, dump_df, outdir)
        extra2 = render_housing_supply(df.index, dump_df, outdir)
    elif sheetname == "FX & Commodities":
        extra1 = render_commodities_breakdown(df.index, dump_df, outdir)
        extra2 = render_reer_trend(df.index, dump_df, outdir)

    # Insert in columns A–H, stacked ~18–20 rows apart
    # Tighter, consistent layout to avoid overlaps; tune as needed
    anchors = ["A2", "A21", "A40", "A59", "A78"]
    imgs = [p1, p2, p3, extra1, extra2]
    for anchor, img in zip(anchors, imgs):
        if img:
            insert_image(ws, img, anchor)

    # Move the table to J3
    move_table_right(ws, start_cell="J3")
    _write_text_summary(ws, "H2", sheetname, df)


def build_dashboard(wb, outdir: Path) -> None:
    if "Dashboard" not in wb.sheetnames:
        return
    # Build composites DF from theme sheets
    series = {}
    for theme in THEME_SHEETS:
        df = load_theme_df(wb, theme)
        if df is not None and not df.empty and "ThemeComposite" in df.columns:
            series[theme] = pd.to_numeric(df["ThemeComposite"], errors="coerce")
    if len(series) < 3:
        return
    dfc = pd.DataFrame(series).dropna(how="all")
    # last 5 years if possible
    if isinstance(dfc.index, pd.DatetimeIndex) and len(dfc) > 0:
        cutoff = dfc.index.max() - pd.DateOffset(years=5)
        dfc = dfc[dfc.index >= cutoff]
    corr = dfc.corr()
    ws: Worksheet = wb["Dashboard"]
    # Clear a known image area to avoid overlay clutter (optional)
    try:
        for img in list(ws._images):
            ws._images.remove(img)
    except Exception:
        pass

    # Correlation heatmap
    try:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        ax.set_title("Theme Composites — Correlation (5y)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path_corr = _save_plot(fig, outdir, "dashboard_correlation.png")
        insert_image(ws, path_corr, "A2")
    except Exception:
        pass

    # Macro Momentum Index: average z-score of all theme composites
    try:
        z = dfc.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
        mmi = z.mean(axis=1)
        fig, ax = plt.subplots(figsize=(7.4, 2.8))
        mmi.plot(ax=ax, color="#333333")
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_title("Macro Momentum Index (avg z-score)")
        ax.grid(True, alpha=0.25)
        path_mmi = _save_plot(fig, outdir, "dashboard_mmi.png")
        insert_image(ws, path_mmi, "A28")
    except Exception:
        pass

    # Macro Risk Gauge: average 12m rolling volatility of theme composites
    try:
        vol = dfc.rolling(12, min_periods=6).std(ddof=0)
        mrg = vol.mean(axis=1)
        fig, ax = plt.subplots(figsize=(7.4, 2.8))
        mrg.plot(ax=ax, color="#aa0000")
        ax.set_title("Macro Risk Gauge (avg 12m volatility)")
        ax.grid(True, alpha=0.25)
        path_mrg = _save_plot(fig, outdir, "dashboard_risk.png")
        insert_image(ws, path_mrg, "A46")
    except Exception:
        pass


def build_regime_labels(wb, outdir: Path) -> None:
    if "Regime Labels" not in wb.sheetnames:
        return
    ws: Worksheet = wb["Regime Labels"]
    # Try to read a table like in load_theme_df
    # We assume first column is date and there's a column named 'Regime'
    # Build values
    data = []
    for row in ws.iter_rows(values_only=True):
        data.append(list(row))
    if not data:
        return
    header = data[0]
    if "Regime" not in header:
        # try to find a column closely named
        try:
            idx = [str(h).lower() for h in header].index("regime")
        except Exception:
            return
    df = pd.DataFrame(data[1:], columns=header)
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    except Exception:
        pass
    if "Regime" not in df.columns:
        return
    try:
        # Encode regimes to integers for a simple step-like plot
        codes = pd.Categorical(df["Regime"]).codes
        fig, ax = plt.subplots(figsize=(7.4, 2.6))
        ax.plot(df.index, codes, drawstyle="steps-post")
        ax.set_title("Global Macro Regime (history)")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.25)
        path = _save_plot(fig, outdir, "regime_history.png")
    except Exception:
        return
    insert_image(ws, path, "A2")


def build_inplace(workbook_path: str, out: Optional[str] = None) -> None:
    """Programmatic entry point (usable from VBA RunPython):
    builds charts and moves tables, saving the workbook in place unless 'out' is provided.
    """
    wb = load_workbook_safe(workbook_path)
    outdir = Path("Output/excel_charts")
    for theme in THEME_SHEETS:
        try:
            build_theme_sheet(wb, theme, outdir)
        except Exception:
            continue
    build_dashboard(wb, outdir)
    build_regime_labels(wb, outdir)
    # Default: overwrite the same workbook (ensure it's closed in Excel)
    out_path = out or workbook_path
    wb.save(out_path)
    print(f"Workbook saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build charts into Macro Reg Template workbook")
    parser.add_argument("--workbook", default="tests/Macro Reg Template.xlsm", help="Path to workbook (.xlsx or .xlsm)")
    parser.add_argument("--out", default=None, help="Optional output path (overwrite in place if omitted)")
    args = parser.parse_args()

    build_inplace(args.workbook, out=args.out)


if __name__ == "__main__":
    main()
