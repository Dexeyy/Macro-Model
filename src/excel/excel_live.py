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

import numpy as np
import pandas as pd
import xlwings as xw

logger = logging.getLogger(__name__)


def _normalise(name: str) -> str:
    """Helper to make sheet / name comparisons case-insensitive & trimmed."""
    return name.strip().lower().replace(" ", "")


# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------


def group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Assign each feature column to an Excel theme sheet.

    Preference order:
    1) Use explicit groupings from config.FEATURE_GROUPS and map them to
       Excel sheet headers.
    2) Fall back to keyword heuristics for any remaining columns.
    """
    groups: Dict[str, List[str]] = {
        "Growth & Labour": [],
        "Inflation & Liquidity": [],
        "Credit & Risk": [],
        "Housing": [],
        "FX & Commodities": [],
    }

    # Map your config feature groups -> Excel sheet names
    sheet_map: Dict[str, str] = {
        "Growth": "Growth & Labour",
        "Inflation": "Inflation & Liquidity",
        "Liquidity": "Inflation & Liquidity",
        "Risk": "Credit & Risk",
        "YieldCurve": "Credit & Risk",
        # Extend if you later add explicit groups for Housing / FX
    }

    # Try to use config.FEATURE_GROUPS if available
    remaining_cols: List[str]
    try:
        from config import config as cfg  # type: ignore

        feature_groups: Dict[str, List[str]] = getattr(cfg, "FEATURE_GROUPS", {})  # type: ignore
        used_cols: set[str] = set()
        for grp_name, cols in (feature_groups or {}).items():
            target_sheet = sheet_map.get(grp_name)
            if not target_sheet:
                continue
            for col in cols:
                if col in df.columns and col not in used_cols:
                    groups[target_sheet].append(col)
                    used_cols.add(col)
        remaining_cols = [c for c in df.columns if c not in used_cols]
    except Exception:
        remaining_cols = list(df.columns)

    # Heuristic fallback for remaining columns
    infl_keywords = ("cpi", "infl", "ppi", "m2", "liqu")
    credit_keywords = ("spread", "risk", "nfc", "baml", "move")
    housing_keywords = ("houst", "housing", "case", "mort")
    fx_comm_keywords = ("fx", "usd", "eur", "oil", "gold", "btc", "commod", "dcoil")

    for col in remaining_cols:
        lname = str(col).lower()
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
    """Convert the contiguous range starting at start_cell into an Excel Table."""
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
    """Update an existing named range name with value if it exists."""
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
        # Ensure scalar numerics for Excel
        try:
            series = pd.to_numeric(series, errors="coerce").dropna()
        except Exception:
            continue
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        delta = float(series.iloc[-1] - series.iloc[-4]) if len(series) >= 4 else None
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
    """Build a compact overview dataset for a theme."""
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
    """Optionally update per-theme KPI named ranges if they exist."""
    if overview_df.empty:
        return
    comp = overview_df["ThemeComposite"].dropna()
    comp_ma = overview_df["ThemeComposite_3M_MA"].dropna()
    if comp.empty and comp_ma.empty:
        return
    last = float((comp_ma.iloc[-1] if not comp_ma.empty else comp.iloc[-1]))
    delta = None
    base = comp_ma if len(comp_ma) >= 4 else comp
    if len(base) >= 4:
        delta = float(base.iloc[-1] - base.iloc[-4])
    slug = _slug(theme)
    _set_named_value(wb, f"ThemeLatest_{slug}", last)
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


def _write_theme_sheet(
    wb: xw.Book,
    sheet: xw.Sheet,
    theme: str,
    overview_df: pd.DataFrame,
    *,
    ensemble_conf: pd.Series | None = None,
):
    """Render a visually clear overview with KPIs, commentary, sparkline, and data table."""
    sheet.clear()
    sheet.range("A1").value = f"{theme} Overview"
    try:
        sheet.range("A1").api.Font.Bold = True
        sheet.range("A1").api.Font.Size = 14
    except Exception:
        pass

    stats = _theme_summary_stats(overview_df)
    if ensemble_conf is not None and not ensemble_conf.dropna().empty:
        try:
            stats = dict(stats)
            stats["Confidence"] = float(ensemble_conf.dropna().iloc[-1])
        except Exception:
            pass
    if stats:
        display_stats = stats.copy()
        if "Confidence" in display_stats and display_stats["Confidence"] is not None:
            try:
                display_stats["Confidence"] = f"{display_stats['Confidence']:.0%}"
            except Exception:
                pass
        kpi_df = pd.DataFrame.from_dict(display_stats, orient="index", columns=["Value"])
        sheet.range("A3").options(index=True, header=True).value = kpi_df
        try:
            sheet.range("A3:B3").api.Font.Bold = True
            sheet.range("B4:B10").number_format = "0.00"
        except Exception:
            pass

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

    if not overview_df.empty:
        sheet.range("A10").options(index=True, header=True).value = overview_df
        sheet.autofit()
        try:
            n = len(overview_df)
            rng = sheet.range(f"B10:B{9+n}")
            rng.api.FormatConditions.Delete()
            rng.api.FormatConditions.AddColorScale(3)
        except Exception:
            pass
        _convert_range_to_table(sheet, start_cell="A10", table_name=f"tbl_{_slug(theme)}_overview")
        _update_theme_named_ranges(wb, theme, overview_df)


def write_excel_report(
    monthly_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    template_path: str | os.PathLike = "tests/Macro Reg Template.xlsx",
    output_path: str | os.PathLike = "Output/Macro_Reg_Report.xlsx",
    theme_test: Optional[str] = None,
) -> None:
    """Push data into Excel template and save."""
    template_path = Path(template_path)
    output_path = Path(output_path)

    # Sanitize dataframes for Excel: drop non-scalar/dict/list columns and coerce numerics
    def _sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        # Best-effort parse of index to datetime
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass
        cols_to_drop: List[str] = []
        for col in list(out.columns):
            s = out[col]
            if s.dtype == object:
                sample = s.dropna().head(10)
                if any(isinstance(v, (dict, list, tuple)) for v in sample):
                    cols_to_drop.append(col)
                    continue
                s_num = pd.to_numeric(s, errors="coerce")
                if s_num.notna().any():
                    out[col] = s_num
                else:
                    try:
                        out[col] = s.astype(str)
                    except Exception:
                        cols_to_drop.append(col)
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop, errors="ignore")
        return out

    monthly_df = _sanitize_for_excel(monthly_df)
    regime_df = _sanitize_for_excel(regime_df)

    if not template_path.exists():
        raise FileNotFoundError(f"Template workbook not found: {template_path}")

    logger.info("Opening Excel template (%s) headless…", template_path)
    app = xw.App(visible=False, add_book=False)
    try:
        wb = app.books.open(str(template_path))

        # Data Dump
        dump_sh = _locate_sheet(wb, "Data Dump")
        _write_frame(dump_sh, monthly_df, index=True)
        try:
            dump_sh.visible = False
        except Exception:
            pass
        _convert_range_to_table(dump_sh, table_name="tbl_DataDump")

        # Theme sheets
        groups = group_features(monthly_df)
        ens_conf_series = None
        try:
            ens_prob_cols = [c for c in regime_df.columns if isinstance(c, str) and c.startswith("Ensemble_Prob_")]
            if ens_prob_cols:
                ens_conf_series = regime_df[ens_prob_cols].max(axis=1)
        except Exception:
            ens_conf_series = None
        for theme, cols in groups.items():
            if theme_test and _normalise(theme) != _normalise(theme_test):
                continue
            sheet = _locate_sheet(wb, theme)
            if not cols:
                sheet.clear()
                continue
            overview_df = _theme_overview(monthly_df, cols)
            _write_theme_sheet(wb, sheet, theme, overview_df, ensemble_conf=ens_conf_series)

        # Regime labels
        reg_sh = _locate_sheet(wb, "Regime Labels")
        try:
            ens_prob_cols = [c for c in regime_df.columns if isinstance(c, str) and c.startswith("Ensemble_Prob_")]
            if ens_prob_cols and "Ensemble_Confidence" not in regime_df.columns:
                regime_df = regime_df.copy()
                regime_df["Ensemble_Confidence"] = regime_df[ens_prob_cols].max(axis=1)
        except Exception:
            pass
        _write_frame(reg_sh, regime_df, index=True)
        _convert_range_to_table(reg_sh, table_name="tbl_RegimeLabels")

        _update_dashboard_kpis(wb, monthly_df)

        try:
            dash = _locate_sheet(wb, "Dashboard")
            ts_name = next((n for n in wb.names if _normalise(n.name) == "lastupdate"), None)
            if ts_name:
                ts_name.refers_to_range.value = pd.Timestamp.now()
            cur_reg = next((n for n in wb.names if _normalise(n.name) == "currentregime"), None)
            if cur_reg and "Regime_Ensemble" in regime_df.columns:
                cur_reg.refers_to_range.value = regime_df["Regime_Ensemble"].iloc[-1]
        except Exception as exc:
            logger.warning("Dashboard named-range update failed: %s", exc)

        try:
            wb.api.RefreshAll()
        except Exception:
            pass

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(output_path))
        logger.info("Workbook saved to %s", output_path)
    finally:
        try:
            wb.close()
        except Exception:
            pass
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

    merged_parquet = Path("Data/processed/merged.parquet")
    featured_csv = Path("Data/processed/macro_data_featured.csv")
    if merged_parquet.exists():
        monthly_df = pd.read_parquet(merged_parquet)
    elif featured_csv.exists():
        monthly_df = pd.read_csv(featured_csv, index_col=0, parse_dates=[0])
    else:
        raise FileNotFoundError("Neither 'Data/processed/merged.parquet' nor 'Data/processed/macro_data_featured.csv' found.")

    regime_path = Path("Data/processed/macro_data_with_regimes.csv")
    if not regime_path.exists():
        raise FileNotFoundError("'Data/processed/macro_data_with_regimes.csv' not found.")
    regime_df_full = pd.read_csv(regime_path, index_col=0, parse_dates=[0])
    keep_cols = [
        c for c in regime_df_full.columns
        if c in ("Rule", "KMeans", "HMM", "Regime_Ensemble")
        or (isinstance(c, str) and (c.endswith("_conf") or "_Prob_" in c))
    ]
    regime_df = regime_df_full[keep_cols].copy()
    try:
        ens_prob_cols = [c for c in regime_df.columns if isinstance(c, str) and c.startswith("Ensemble_Prob_")]
        if ens_prob_cols:
            regime_df["Ensemble_Confidence"] = regime_df[ens_prob_cols].max(axis=1)
    except Exception:
        pass

    write_excel_report(monthly_df, regime_df, template_path=args.template, output_path=args.output, theme_test=args.theme_test)


if __name__ == "__main__":
    _cli()


