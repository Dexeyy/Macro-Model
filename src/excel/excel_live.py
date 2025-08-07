"""Utilities to push DataFrames into the Excel dashboard template via xlwings.

Assumes a template workbook that already contains the following sheets
(don't worry about extra whitespace ‑ we look them up case-insensitively)

    Dashboard | Growth & Labour | Inflation & Liquidity | Credit & Risk
    Housing   | FX & Commodities | Regime Labels | Data Dump

Typical usage (from main.py or CLI):

    from src.excel.excel_live import write_excel_report
    write_excel_report(monthly_df, regime_df,
                       template_path="tests/Macro Reg Template.xlsx",
                       output_path="Output/Macro_Reg_Report.xlsx")
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xlwings as xw
import numpy as np

logger = logging.getLogger(__name__)

def _normalise(name: str) -> str:
    """Helper to make sheet / name comparisons case-insensitive & trimmed."""
    return name.strip().lower().replace(" ", "")

# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------

def group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Assign each feature column to a theme sheet.

    This is a very lightweight heuristic: edit the rules as needed.
    Columns that don't fit any rule go to *Growth & Labour* by default.
    """
    groups: Dict[str, List[str]] = {
        "Growth & Labour": [],
        "Inflation & Liquidity": [],
        "Credit & Risk": [],
        "Housing": [],
        "FX & Commodities": [],
    }

    infl_keywords = ("cpi", "infl", "ppi", "m2", "liqu")
    credit_keywords = ("spread", "risk", "nfc", "baml", "move")
    housing_keywords = ("houst", "housing", "case", "mort")
    fx_comm_keywords = ("fx", "usd", "eur", "oil", "gold", "btc", "commod", "dcoil")

    for col in df.columns:
        lname = col.lower()
        if lname.startswith(infl_keywords) or any(k in lname for k in infl_keywords):
            groups["Inflation & Liquidity"].append(col)
        elif any(k in lname for k in credit_keywords):
            groups["Credit & Risk"].append(col)
        elif any(k in lname for k in housing_keywords):
            groups["Housing"].append(col)
        elif any(k in lname for k in fx_comm_keywords):
            groups["FX & Commodities"].append(col)
        else:
            groups["Growth & Labour"].append(col)

    return groups

# ---------------------------------------------------------------------------
# Excel push logic
# ---------------------------------------------------------------------------

def _locate_sheet(wb: xw.Book, target_name: str) -> xw.Sheet:
    norm = _normalise(target_name)
    for sht in wb.sheets:
        if _normalise(sht.name) == norm:
            return sht
    # fallback – create if missing
    logger.warning("Sheet '%s' not found; creating.", target_name)
    return wb.sheets.add(target_name)

# ---------------------------------------------------------------------------
# Helper functions for tables, named ranges & KPI cards
# ---------------------------------------------------------------------------

def _convert_range_to_table(sheet: xw.Sheet, start_cell: str = "A1", table_name: str | None = None):
    """Convert the contiguous range starting at *start_cell* into an Excel Table.

    Parameters
    ----------
    sheet : xw.Sheet
        Target worksheet.
    start_cell : str
        Anchor cell where the DataFrame was written (defaults to A1).
    table_name : str, optional
        Name to assign to the created table. Will be converted to a valid
        Excel identifier by stripping spaces etc. If a table with the same
        name already exists it will be replaced.
    """
    try:
        rng = sheet.range(start_cell).expand()
        # Delete existing table with same name to avoid COM errors
        if table_name:
            for lo in sheet.api.ListObjects:
                if lo.Name.lower() == table_name.lower():
                    lo.Delete()
                    break
        lo = sheet.api.ListObjects.Add(1, rng.api, None, 1)
        if table_name:
            lo.Name = table_name.replace(" ", "_")
    except Exception as exc:
        logger.warning("Could not convert range to table on sheet %s: %s", sheet.name, exc)


def _set_named_value(wb: xw.Book, name: str, value):
    """Update an existing named range *name* with *value* if it exists."""
    nm = next((n for n in wb.names if _normalise(n.name) == _normalise(name)), None)
    if nm:
        try:
            nm.refers_to_range.value = value
        except Exception as exc:
            logger.debug("Failed assigning value to named range '%s': %s", name, exc)


def _update_dashboard_kpis(wb: xw.Book, monthly_df: pd.DataFrame):
    """Push latest KPI values + 3-month deltas into pre-defined named ranges."""
    kpi_map = {
        "Growth": "GDP_YoY",
        "Inflation": "CPI_YoY",
        "YieldCurveSlope": "YieldCurveSlope",
        "FinConditionsComposite": "FinConditionsComposite",
    }
    for kpi, col in kpi_map.items():
        if col not in monthly_df.columns:
            continue
        series = monthly_df[col].dropna()
        if series.empty:
            continue
        latest = series.iloc[-1]
        delta = series.iloc[-1] - series.iloc[-4] if len(series) >= 4 else None
        _set_named_value(wb, f"{kpi}Latest", latest)
        if delta is not None:
            _set_named_value(wb, f"{kpi}Delta", delta)



def _write_frame(sheet: xw.Sheet, df: pd.DataFrame, start_cell: str = "A1", index: bool = True):
    sheet.clear()  # remove old content
    sheet.range(start_cell).options(index=index, header=True).value = df
    sheet.autofit()


def _slug(name: str) -> str:
    """Create a compact identifier for names used in named ranges/tables."""
    return _normalise(name).replace("&", "and").replace("/", "_")


def _theme_overview(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Build a compact overview dataset for a theme.

    Returns a DataFrame with:
    - ThemeComposite: row-wise mean of z-scored inputs
    - ThemeComposite_3M_MA: 3‑month moving average
    - ThemeComposite_YoY: 12‑month pct change of the composite
    """
    if not cols:
        return pd.DataFrame()

    sub = df[cols].dropna(how="all")
    if sub.empty:
        return pd.DataFrame()

    # z-score each column to comparable scale, then average row-wise
    z = sub.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    composite = z.mean(axis=1)
    out = pd.DataFrame({
        "ThemeComposite": composite,
    })
    out["ThemeComposite_3M_MA"] = out["ThemeComposite"].rolling(3, min_periods=1).mean()
    out["ThemeComposite_YoY"] = out["ThemeComposite"].pct_change(12)
    return out


def _update_theme_named_ranges(wb: xw.Book, theme: str, overview_df: pd.DataFrame):
    """Optionally update per-theme KPI named ranges if they exist.

    Looks for names like 'ThemeLatest_<slug>' and 'ThemeDelta_<slug>'.
    """
    if overview_df.empty:
        return
    latest = overview_df["ThemeComposite_3M_MA"].dropna()
    if latest.empty:
        latest = overview_df["ThemeComposite"].dropna()
    if latest.empty:
        return
    last_val = latest.iloc[-1]
    delta = None
    if len(latest) >= 4:
        delta = latest.iloc[-1] - latest.iloc[-4]

    slug = _slug(theme)
    _set_named_value(wb, f"ThemeLatest_{slug}", last_val)
    if delta is not None:
        _set_named_value(wb, f"ThemeDelta_{slug}", delta)


def _theme_summary_stats(overview_df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary stats used in the header KPI card for a theme."""
    if overview_df.empty:
        return {}
    comp = overview_df["ThemeComposite"].dropna()
    comp_ma = overview_df["ThemeComposite_3M_MA"].dropna()
    last = float(comp.iloc[-1]) if not comp.empty else np.nan
    last_ma = float(comp_ma.iloc[-1]) if not comp_ma.empty else np.nan
    change_3m = float(comp.iloc[-1] - comp.iloc[-4]) if len(comp) >= 4 else np.nan
    yoy = float(overview_df["ThemeComposite_YoY"].dropna().iloc[-1]) if not overview_df["ThemeComposite_YoY"].dropna().empty else np.nan
    z = (last - float(comp.mean())) / (float(comp.std(ddof=0)) or 1.0) if not np.isnan(last) else np.nan
    # simple slope over last 6 months
    slope_6m = np.nan
    if len(comp) >= 6:
        y = comp.iloc[-6:]
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y.values, 1)
        slope_6m = float(coeffs[0])
    return {
        "Latest": last,
        "3M_MA": last_ma,
        "Change_3M": change_3m,
        "YoY": yoy,
        "ZScore": z,
        "Slope_6M": slope_6m,
    }


def _write_theme_sheet(wb: xw.Book, sheet: xw.Sheet, theme: str, overview_df: pd.DataFrame):
    """Render a visually clear overview with KPIs, commentary, sparkline, and data table."""
    # Clear and layout regions
    sheet.clear()

    # Header title
    sheet.range("A1").value = f"{theme} Overview"
    sheet.range("A1").api.Font.Bold = True
    sheet.range("A1").api.Font.Size = 14

    # KPI grid
    stats = _theme_summary_stats(overview_df)
    if stats:
        kpi_df = pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])
        sheet.range("A3").options(index=True, header=True).value = kpi_df
        sheet.range("A3:B3").api.Font.Bold = True
        # basic number format
        sheet.range("B4:B10").number_format = "0.00"

    # Commentary
    try:
        latest = stats.get("Latest", np.nan)
        yoy = stats.get("YoY", np.nan)
        ch3 = stats.get("Change_3M", np.nan)
        slope = stats.get("Slope_6M", np.nan)
        direction = "improving" if slope and slope > 0 else "softening" if slope and slope < 0 else "flat"
        comment = (
            f"Composite at {latest:.2f}; 3m change {ch3:+.2f}, YoY {yoy:+.2%}. "
            f"Momentum over 6m is {direction}."
        )
        sheet.range("D3").value = comment
        sheet.range("D3").api.WrapText = True
        sheet.range("D3").column_width = 60
    except Exception:
        pass

    # Sparkline based on overview series (3M MA preferred)
    try:
        n = len(overview_df)
        if n > 1:
            src = sheet.range(f"B10:B{9 + n}")
            source_rng = src.get_address(row_absolute=False, column_absolute=False, external=False)
            # place sparkline at H3
            sheet.range("H3").value = None
            sheet.range("H3").api.SparklineGroups.Add(1, source_rng)
    except Exception:
        pass

    # Data table starts at A10
    if not overview_df.empty:
        sheet.range("A10").options(index=True, header=True).value = overview_df
        sheet.autofit()
        # Conditional formatting on composite
        try:
            n = len(overview_df)
            rng = sheet.range(f"B10:B{9+n}")
            # Clear previous conditions
            rng.api.FormatConditions.Delete()
            # 3-color scale
            rng.api.FormatConditions.AddColorScale(3)
        except Exception:
            pass

        _convert_range_to_table(sheet, start_cell="A10", table_name=f"tbl_{_slug(theme)}_overview")
        _update_theme_named_ranges(wb, theme, overview_df)

        # Skip native Excel chart creation by default to let the builder own visuals.
        # Set EXCEL_LIVE_ADD_CHARTS=1 to re-enable.
        try:
            import os as _os
            if _os.getenv("EXCEL_LIVE_ADD_CHARTS", "0") == "1":
                # Remove any previous charts for idempotency
                for ch in list(sheet.charts):
                    if ch.name.lower().startswith(f"chart_{_slug(theme)}"):
                        ch.delete()
                rng = sheet.range("A10").expand()
                chart = sheet.charts.add(left=sheet.range("D8").left, top=sheet.range("D8").top, width=520, height=300)
                chart.name = f"Chart_{_slug(theme)}_trend"
                chart.set_source_data(rng)
                chart.chart_type = 'line'
                chart.api["HasTitle"] = True
                chart.api["ChartTitle"]["Text"] = f"{theme}: Composite & 3M MA"
                try:
                    chart.api.FullSeriesCollection(1).Format.Line.Weight = 1.75
                    chart.api.FullSeriesCollection(2).Format.Line.Weight = 1.25
                except Exception:
                    pass
        except Exception:
            pass


def write_excel_report(
    monthly_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    template_path: str | os.PathLike = "tests/Macro Reg Template.xlsx",
    output_path: str | os.PathLike = "Output/Macro_Reg_Report.xlsx",
    theme_test: Optional[str] = None,
) -> None:
    """Push data into Excel template and save.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Full processed monthly data (features + raw + engineered).
    regime_df : pd.DataFrame
        DataFrame returned by `fit_regimes`.
    template_path : str | Path
        Path to the xlsx template.
    output_path : str | Path
        Where to save the populated workbook.
    theme_test : str, optional
        If provided, only this theme sheet is refreshed (for quick dev cycles).
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    if not template_path.exists():
        raise FileNotFoundError(f"Template workbook not found: {template_path}")

    logger.info("Opening Excel template (%s) headless…", template_path)
    app = xw.App(visible=False, add_book=False)
    try:
        wb = app.books.open(str(template_path))

        # --- DataDump -------------------------------------------------------
        dump_sh = _locate_sheet(wb, "Data Dump")
        _write_frame(dump_sh, monthly_df, index=True)
        # Keep the heavy raw data sheet hidden from users
        try:
            dump_sh.visible = False
        except Exception:
            pass
        _convert_range_to_table(dump_sh, table_name="tbl_DataDump")

        # --- Theme sheets (overview only; full data lives in Data Dump) -----
        groups = group_features(monthly_df)
        for theme, cols in groups.items():
            if theme_test and _normalise(theme) != _normalise(theme_test):
                continue  # skip others in test mode
            sheet = _locate_sheet(wb, theme)
            if not cols:
                # No columns matched this theme; clear to avoid stale content
                sheet.clear()
                continue
            overview_df = _theme_overview(monthly_df, cols)
            _write_theme_sheet(wb, sheet, theme, overview_df)

        # --- Regime labels --------------------------------------------------
        reg_sh = _locate_sheet(wb, "Regime Labels")
        _write_frame(reg_sh, regime_df, index=True)
        _convert_range_to_table(reg_sh, table_name="tbl_RegimeLabels")

        # Update KPI named ranges on Dashboard
        _update_dashboard_kpis(wb, monthly_df)

        # --- Dashboard named ranges ----------------------------------------
        try:
            dash = _locate_sheet(wb, "Dashboard")
            # update timestamp if a named range exists
            ts_name = next((n for n in wb.names if _normalise(n.name) == "lastupdate"), None)
            if ts_name:
                ts_name.refers_to_range.value = pd.Timestamp.now()
            # current ensemble regime cell named "CurrentRegime" (?)
            cur_reg = next((n for n in wb.names if _normalise(n.name) == "currentregime"), None)
            if cur_reg and "Regime_Ensemble" in regime_df.columns:
                cur_reg.refers_to_range.value = regime_df["Regime_Ensemble"].iloc[-1]
        except Exception as exc:
            logger.warning("Dashboard named-range update failed: %s", exc)

        # Refresh pivots / charts if supported on this platform
        try:
            wb.api.RefreshAll()
        except Exception:
            pass

        # Save to new file (don’t overwrite template)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(output_path))
        logger.info("Workbook saved to %s", output_path)
    finally:
        wb.close()
        app.quit()

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _cli():
    parser = argparse.ArgumentParser(description="Push macro-regime data into Excel template.")
    parser.add_argument("--template", default="tests/Macro Reg Template.xlsx", help="Path to template workbook")
    parser.add_argument("--output", default="Output/Macro_Reg_Report.xlsx", help="Path for populated workbook")
    parser.add_argument("--theme-test", help="Only refresh this theme sheet (speeds up dev)")
    args = parser.parse_args()

    # For CLI we load the parquet/CSV files saved by the pipeline
    # Load processed data automatically detecting file type
    # Prefer merged.parquet; fallback to macro_data_featured.csv
    merged_parquet = Path("Data/processed/merged.parquet")
    featured_csv = Path("Data/processed/macro_data_featured.csv")
    if merged_parquet.exists():
        monthly_df = pd.read_parquet(merged_parquet)
    elif featured_csv.exists():
        monthly_df = pd.read_csv(featured_csv, index_col=0, parse_dates=[0])
    else:
        raise FileNotFoundError("Neither 'Data/processed/merged.parquet' nor 'Data/processed/macro_data_featured.csv' found.")

    # Load regime labels, keeping all method and confidence columns if present
    regime_path = Path("Data/processed/macro_data_with_regimes.csv")
    if not regime_path.exists():
        raise FileNotFoundError("'Data/processed/macro_data_with_regimes.csv' not found.")
    regime_df_full = pd.read_csv(regime_path, index_col=0, parse_dates=[0])
    keep_cols = [
        c for c in regime_df_full.columns
        if c in ("Rule", "KMeans", "HMM", "Regime_Ensemble") or c.endswith("_conf")
    ]
    regime_df = regime_df_full[keep_cols]

    write_excel_report(monthly_df, regime_df, template_path=args.template, output_path=args.output, theme_test=args.theme_test)

if __name__ == "__main__":
    _cli()
