"""
Headless Excel dashboard builder using openpyxl and matplotlib.
Rebuilds visuals in-place and supports fallbacks to processed CSVs if the
template lacks pre-populated tables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.styles import Font


THEME_SHEETS: List[str] = [
    "Growth & Labour",
    "Inflation & Liquidity",
    "Credit & Risk",
    "Housing",
    "FX & Commodities",
]


THEME_COLOR: Dict[str, str] = {
    "Growth & Labour": "#1f77b4",
    "Inflation & Liquidity": "#d62728",
    "Credit & Risk": "#ff7f0e",
    "Housing": "#2ca02c",
    "FX & Commodities": "#9467bd",
}


def load_workbook_safe(path: str | Path):
    path_str = str(path)
    keep_vba = path_str.lower().endswith(".xlsm")
    return load_workbook(filename=path_str, keep_vba=keep_vba)


def _save_plot(fig, outdir: Path, filename: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / filename
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def insert_image(ws: Worksheet, image_path: Path, anchor_cell: str) -> None:
    try:
        img = XLImage(str(image_path))
        ws.add_image(img, anchor_cell)
    except Exception:
        pass


def _read_table_from_sheet(ws: Worksheet) -> Optional[pd.DataFrame]:
    max_rows, max_cols = 200, 40
    values = [[ws.cell(r, c).value for c in range(1, max_cols + 1)] for r in range(1, max_rows + 1)]
    header_row = None
    for r in range(len(values)):
        row = values[r]
        if row and any((isinstance(v, str) and "themecomposite" in v.lower()) for v in row if v):
            header_row = r
            break
    if header_row is None:
        return None
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
    headers = [h if h is not None else "Date" for h in headers[: last_col]]
    df = pd.DataFrame(data_rows, columns=headers)
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    except Exception:
        pass
    return df


def load_theme_df(wb, sheetname: str) -> Optional[pd.DataFrame]:
    if sheetname not in wb.sheetnames:
        return None
    ws: Worksheet = wb[sheetname]
    df = _read_table_from_sheet(ws)
    if df is None:
        return None
    needed = {"ThemeComposite", "ThemeComposite_3M_MA", "ThemeComposite_YoY"}
    if not needed.intersection(set(df.columns)):
        return None
    return df


def compute_metrics(df: pd.DataFrame):
    comp = df["ThemeComposite"].astype(float)
    latest = comp.iloc[-1]
    chg_3m = comp.diff(3).iloc[-1] if len(comp) >= 4 else np.nan
    yoy = df["ThemeComposite_YoY"].astype(float).iloc[-1] if "ThemeComposite_YoY" in df.columns else np.nan
    ranks = comp.rank(method="average")
    percentile = 100 * ranks / ranks.max()
    window = 36 if len(comp) >= 36 else 24 if len(comp) >= 24 else max(6, len(comp))
    roll_mean = comp.rolling(window, min_periods=max(3, window // 3)).mean()
    roll_std = comp.rolling(window, min_periods=max(3, window // 3)).std(ddof=0)
    zscore = (comp - roll_mean) / (roll_std.replace(0, np.nan))
    roll_vol_12m = comp.rolling(12, min_periods=6).std(ddof=0)
    slope_6m = np.nan
    if len(comp) >= 6:
        y = comp.iloc[-6:]
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y.values, 1)
        slope_6m = float(coeffs[0])
    return latest, chg_3m, yoy, percentile, zscore, roll_vol_12m, slope_6m


def load_data_dump_df(wb) -> Optional[pd.DataFrame]:
    """Read the 'Data Dump' sheet as a DataFrame.

    Assumes headers on first row, index in first column (dates).
    """
    if "Data Dump" not in wb.sheetnames:
        return None
    ws: Worksheet = wb["Data Dump"]
    values = list(ws.values)
    if not values:
        return None
    headers = [h if h is not None else f"col_{i}" for i, h in enumerate(values[0])]
    rows = values[1:]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=headers)
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    except Exception:
        pass
    return df


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
        return _save_plot(fig, outdir, f"{theme.replace(' & ', '_').replace(' ', '_').lower()}_trend.png")
    except Exception:
        return None


def render_theme_signature(theme: str, monthly: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """Render a theme-specific signature chart. Skip gracefully if data missing."""
    try:
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        t = theme
        m = monthly
        if t == "Growth & Labour":
            # Business cycle clock: GDP_YoY vs UNRATE, colored by date order
            if all(c in m.columns for c in ("GDP_YoY", "UNRATE")):
                xy = m[["GDP_YoY", "UNRATE"]].dropna()
                if xy.empty:
                    raise ValueError
                colors = np.linspace(0, 1, len(xy))
                sc = ax.scatter(xy["GDP_YoY"], xy["UNRATE"], c=colors, cmap="viridis", s=12)
                ax.set_xlabel("GDP YoY (%)")
                ax.set_ylabel("Unemployment Rate (%)")
                ax.set_title("Business Cycle Clock")
            else:
                ax.text(0.1, 0.5, "Data not available", transform=ax.transAxes)
                ax.set_axis_off()

        elif t == "Inflation & Liquidity":
            # CPI contributions (if component series exist) and real policy rate line
            components = [c for c in ("CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC") if c in m.columns]
            plotted = False
            if components:
                comp_yoy = pd.DataFrame({c: pd.to_numeric(m[c], errors="coerce").pct_change(12) * 100 for c in components}).dropna()
                if not comp_yoy.empty:
                    comp_yoy.tail(180).plot.area(ax=ax, stacked=True, linewidth=0)
                    plotted = True
            if "FEDFUNDS" in m.columns and "CPI_YoY" in m.columns:
                rr = (pd.to_numeric(m["FEDFUNDS"], errors="coerce") - pd.to_numeric(m["CPI_YoY"], errors="coerce")).dropna()
                if not plotted:
                    rr.plot(ax=ax, color="#444444", linewidth=1.5, label="Real Policy Rate")
                else:
                    ax2 = ax.twinx()
                    rr.tail(180).plot(ax=ax2, color="#444444", linewidth=1.5, label="Real Policy Rate")
                    ax2.set_ylim(rr.tail(180).min()*1.1, rr.tail(180).max()*1.1)
                plotted = True
            if not plotted:
                ax.text(0.1, 0.5, "Data not available", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                ax.set_title("CPI Contributions & Real Policy Rate")

        elif t == "Credit & Risk":
            # IG spread (BAA-AAA) and NFCI stress if available
            plotted = False
            if all(c in m.columns for c in ("BAA", "AAA")):
                ig = (pd.to_numeric(m["BAA"], errors="coerce") - pd.to_numeric(m["AAA"], errors="coerce")).dropna()
                ig.tail(180).plot(ax=ax, color="#cc5500", linewidth=1.5, label="IG Spread (BAA-AAA)")
                plotted = True
            if "NFCI" in m.columns:
                ax2 = ax.twinx() if plotted else ax
                pd.to_numeric(m["NFCI"], errors="coerce").tail(180).plot(ax=ax2, color="#5555aa", linewidth=1.2, label="NFCI")
                plotted = True
            if not plotted:
                ax.text(0.1, 0.5, "Data not available", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                ax.set_title("Credit Spreads & Financial Stress")
                ax.grid(True, axis="y", alpha=0.2)

        elif t == "Housing":
            # Permits vs Starts; mortgage rate if available
            plotted = False
            cols = [c for c in ("PERMIT", "HOUST") if c in m.columns]
            if cols:
                pd.DataFrame({c: pd.to_numeric(m[c], errors="coerce") for c in cols}).tail(240).plot(ax=ax, linewidth=1.5)
                plotted = True
            if "MORTGAGE30US" in m.columns:
                ax2 = ax.twinx()
                pd.to_numeric(m["MORTGAGE30US"], errors="coerce").tail(240).plot(ax=ax2, color="#777777", linewidth=1.2, label="Mortgage 30Y")
                plotted = True
            if not plotted:
                ax.text(0.1, 0.5, "Data not available", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                ax.set_title("Housing Pipeline & Mortgage Rate")

        elif t == "FX & Commodities":
            # Energy proxy (WTI), Dollar index (TWEX), optional gold if present
            plotted = False
            if "DCOILWTICO" in m.columns:
                pd.to_numeric(m["DCOILWTICO"], errors="coerce").tail(240).plot(ax=ax, color="#8c564b", linewidth=1.5, label="WTI Oil")
                plotted = True
            if "TWEXAFEGSMTH" in m.columns:
                ax2 = ax.twinx() if plotted else ax
                pd.to_numeric(m["TWEXAFEGSMTH"], errors="coerce").tail(240).plot(ax=ax2, color="#1f77b4", linewidth=1.2, label="Dollar Index")
                plotted = True
            if not plotted:
                ax.text(0.1, 0.5, "Data not available", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                ax.set_title("Energy & Dollar Index")

        else:
            ax.text(0.1, 0.5, "Data not available", transform=ax.transAxes)
            ax.set_axis_off()

        return _save_plot(fig, outdir, f"{theme.replace(' & ', '_').replace(' ', '_').lower()}_signature.png")
    except Exception:
        return None


def build_theme_sheet(wb, sheetname: str, outdir: Path) -> None:
    if sheetname not in wb.sheetnames:
        return
    ws: Worksheet = wb[sheetname]
    df = load_theme_df(wb, sheetname)
    if df is None or df.empty:
        return
    p1 = render_plot_trend(sheetname, df, outdir)
    if p1:
        insert_image(ws, p1, "A2")
    # Signature chart using Data Dump
    monthly = load_data_dump_df(wb)
    if monthly is not None and not monthly.empty:
        p2 = render_theme_signature(sheetname, monthly, outdir)
        if p2:
            insert_image(ws, p2, "A22")


def build_dashboard(wb, outdir: Path) -> None:
    if "Dashboard" not in wb.sheetnames:
        return
    series = {}
    for theme in THEME_SHEETS:
        df = load_theme_df(wb, theme)
        if df is not None and not df.empty and "ThemeComposite" in df.columns:
            series[theme] = pd.to_numeric(df["ThemeComposite"], errors="coerce")
    if len(series) < 3:
        return
    dfc = pd.DataFrame(series).dropna(how="all")
    if isinstance(dfc.index, pd.DatetimeIndex) and len(dfc) > 0:
        cutoff = dfc.index.max() - pd.DateOffset(years=5)
        dfc = dfc[dfc.index >= cutoff]
    corr = dfc.corr()
    ws: Worksheet = wb["Dashboard"]
    try:
        for img in list(ws._images):
            ws._images.remove(img)
    except Exception:
        pass
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


def build_regime_labels(wb, outdir: Path) -> None:
    def _render(df: pd.DataFrame) -> Optional[Path]:
        try:
            regime_col = "Regime_Ensemble" if "Regime_Ensemble" in df.columns else "Regime"
            if regime_col not in df.columns:
                return None
            codes = pd.Categorical(df[regime_col]).codes
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
            ax1.plot(df.index, codes, drawstyle="steps-post")
            ax1.set_title("Global Macro Regime (history)")
            ax1.set_yticks([])
            ax1.grid(True, axis="x", alpha=0.25)
            prob_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Ensemble_Prob_")]
            if prob_cols:
                probs = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in prob_cols}).fillna(0.0)
                rs = probs.sum(axis=1).replace(0.0, np.nan)
                probs = probs.div(rs, axis=0).fillna(0.0).tail(180)
                probs.plot.area(ax=ax2, stacked=True, linewidth=0)
                ax2.set_ylim(0, 1)
                ax2.set_yticks([0, 0.5, 1.0])
                ax2.set_title("Ensemble Probabilities (last 180 months)")
                ax2.grid(True, axis="y", alpha=0.2)

                # Low-confidence shading (confidence < 0.6)
                try:
                    if "Ensemble_Confidence" in df.columns:
                        conf = pd.to_numeric(df["Ensemble_Confidence"], errors="coerce").fillna(0.0)
                    else:
                        conf = probs.max(axis=1)
                    conf = conf.reindex(probs.index)
                    low = conf < 0.6
                    # find contiguous low segments
                    if low.any():
                        # convert to segments of start/end indices
                        in_seg = False
                        start = None
                        dates = probs.index
                        for i, flag in enumerate(low.values):
                            if flag and not in_seg:
                                in_seg = True
                                start = dates[i]
                            if in_seg and (not flag or i == len(low.values) - 1):
                                end = dates[i] if not flag else dates[i]
                                # draw shaded region on both axes
                                ax1.axvspan(start, end, color="#000000", alpha=0.08)
                                ax2.axvspan(start, end, color="#000000", alpha=0.08)
                                in_seg = False
                except Exception:
                    pass
            else:
                ax2.text(0.01, 0.5, "No ensemble probabilities available", transform=ax2.transAxes)
                ax2.set_axis_off()
            return _save_plot(fig, outdir, "regime_history.png")
        except Exception:
            return None

    ws: Worksheet
    if "Regime Labels" in wb.sheetnames:
        ws = wb["Regime Labels"]
        data = []
        for row in ws.iter_rows(values_only=True):
            data.append(list(row))
        if data:
            header = data[0]
            df = pd.DataFrame(data[1:], columns=header)
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
            except Exception:
                pass
            path = _render(df)
            if path:
                insert_image(ws, path, "A2")
                return

    try:
        csv_path = Path("Data/processed/macro_data_with_regimes.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
            path = _render(df)
            if path is None:
                return
            if "Regime Labels" not in wb.sheetnames:
                wb.create_sheet("Regime Labels")
            ws = wb["Regime Labels"]
            insert_image(ws, path, "A2")
    except Exception:
        return


def build_inplace(workbook_path: str, out: Optional[str] = None) -> None:
    wb = load_workbook_safe(workbook_path)
    outdir = Path("Output/excel_charts")
    for theme in THEME_SHEETS:
        try:
            build_theme_sheet(wb, theme, outdir)
        except Exception:
            continue
    build_dashboard(wb, outdir)
    build_regime_labels(wb, outdir)
    try:
        from glob import glob
        shots = sorted(glob("Output/portfolio_comparison_*.png"))
        if shots:
            sheetname = "Portfolios"
            if sheetname not in wb.sheetnames:
                wb.create_sheet(sheetname)
            ws: Worksheet = wb[sheetname]
            try:
                for img in list(ws._images):
                    ws._images.remove(img)
            except Exception:
                pass
            anchors = ["A2", "A30", "A58", "A86", "A114", "A142"]
            for anchor, shot in zip(anchors, shots):
                try:
                    insert_image(ws, Path(shot), anchor)
                except Exception:
                    continue
    except Exception:
        pass
    out_path = out or "Output/Macro_Reg_Report_with_category_dashboards.xlsm"
    wb.save(out_path)
    print(f"Workbook saved: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build charts into Macro Reg Template workbook")
    parser.add_argument("--workbook", default="tests/Macro Reg Template.xlsm", help="Path to workbook (.xlsx or .xlsm)")
    parser.add_argument("--out", default=None, help="Optional output path (overwrite in place if omitted)")
    args = parser.parse_args()
    build_inplace(args.workbook, out=args.out)


if __name__ == "__main__":
    main()


