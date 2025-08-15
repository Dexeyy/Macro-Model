from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl.worksheet.worksheet import Worksheet

from .helpers import (
    breadth_z,
    credit_spread,
    insert_image,
    make_move,
    theme_composite,
    theme_composite_from_cfg,
    rolling_percentile_cone,
    save_fig,
    safe_cols,
    to_periodic_rf,
    yoy,
    zscore,
)


OUTDIR = Path("Output/excel_charts")


def _resolve_sheet(book, sheet_name: str) -> Worksheet:
    try:
        ws = book[sheet_name]
    except Exception:
        ws = book.create_sheet(sheet_name)
    return ws


def _note(ws: Worksheet, cell: str, msg: str) -> None:
    try:
        ws[cell].value = msg
    except Exception:
        pass


def build_growth_sheet(df: pd.DataFrame, book, cfg, sheet_name: str = "Growth") -> None:
    if df is None or df.empty:
        return
    ws = _resolve_sheet(book, sheet_name)

    # G1: Composite + 5y cone
    comp_col = next((c for c in ["F_Growth", "Growth_Composite"] if c in df.columns), None)
    s = None
    if comp_col is not None:
        s = pd.to_numeric(df[comp_col], errors="coerce").dropna()
    else:
        s = theme_composite_from_cfg(df, cfg, "growth")
    if s is not None and len(s) > 0:
        if len(s) > 0:
            # last 5 years when long
            s_plot = s.tail(60) if len(s) > 60 else s
            cone = rolling_percentile_cone(s, window=60).reindex(s_plot.index)
            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            try:
                ax.fill_between(cone.index, cone["p10"], cone["p90"], alpha=0.2, label="5y cone")
                ax.plot(cone.index, cone["p50"], linewidth=1.2, alpha=0.8, label="median")
            except Exception:
                pass
            ax.plot(s_plot.index, s_plot.values, linewidth=1.8, label=comp_col)
            ax.set_title("Growth Composite: 5y cone")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            p = save_fig(fig, OUTDIR, "growth_cone.png")
            insert_image(ws, p, "T2", max_width=440, max_height=260)
            _note(ws, "T1", "Growth: cone chart inserted")
    else:
        _note(ws, "J2", "Growth composite not available — cone chart skipped")

    # G2: Breadth of z-scores (components)
    comp_list = getattr(cfg, "GROUPS", {}).get("growth", []) or getattr(cfg, "THEME_GROUPS", {}).get("growth", [])
    z_cols = [c for c in comp_list if f"{c}_ZScore" in df.columns]
    if not z_cols:
        # fallback: any *_ZScore present with growth-like naming
        z_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_ZScore") and "EMP" not in c.upper()]
    if z_cols:
        zdf = df[z_cols].apply(pd.to_numeric, errors="coerce")
        br = breadth_z(zdf).tail(180)
        if not br.empty:
            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            br["pct_pos"].plot(ax=ax, label=">0")
            br["pct_pos1"].plot(ax=ax, label=">+1")
            ax.set_ylim(0, 100)
            ax.set_title("Growth breadth (% of components)")
            ax.set_ylabel("Percent")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            p = save_fig(fig, OUTDIR, "growth_breadth.png")
            insert_image(ws, p, "T22", max_width=440, max_height=260)
            _note(ws, "T21", "Growth: breadth chart inserted")
    else:
        _note(ws, "J22", "Growth component z-scores not available — breadth chart skipped")

    # G3: INDPRO YoY vs PAYEMS YoY (with diff)
    a = df.get("INDPRO_YoY")
    b = df.get("PAYEMS_YoY")
    if a is None and "INDPRO" in df.columns:
        a = yoy(df["INDPRO"]) * 100
    if b is None and "PAYEMS" in df.columns:
        b = yoy(df["PAYEMS"]) * 100
    if a is not None and b is not None:
        idx = a.dropna().index.intersection(b.dropna().index)
        a2 = pd.to_numeric(a, errors="coerce").reindex(idx).dropna()
        b2 = pd.to_numeric(b, errors="coerce").reindex(idx).dropna()
        if not a2.empty and not b2.empty:
            diff = a2 - b2
            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            a2.tail(240).plot(ax=ax, label="INDPRO YoY")
            b2.tail(240).plot(ax=ax, label="PAYEMS YoY")
            ax2 = ax.twinx()
            diff.tail(240).plot(ax=ax2, color="#777777", linewidth=1.2, label="Diff (L2R)")
            ax.set_title("Production vs Employment — YoY")
            ax.set_ylabel("% YoY")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.25)
            try:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="best")
            except Exception:
                ax.legend(loc="best")
            p = save_fig(fig, OUTDIR, "growth_divergence.png")
            insert_image(ws, p, "T42", max_width=440, max_height=260)
            _note(ws, "T41", "Growth: divergence chart inserted")
    else:
        _note(ws, "J42", "INDPRO/PAYEMS YoY not available — divergence chart skipped")


def build_housing_sheet(df: pd.DataFrame, book, cfg, sheet_name: str = "Housing") -> None:
    if df is None or df.empty:
        return
    ws = _resolve_sheet(book, sheet_name)

    # H1: Affordability ratio
    price_col = next((c for c in ["CSUSHPINSA", "CSUSHPISA", "CSUSHPINSA_Index"] if c in df.columns), None)
    rate_col = next((c for c in ["MORTGAGE30US", "MORTGAGE30Y", "MORTGAGE_30Y"] if c in df.columns), None)
    inc_col = next((c for c in ["MEDINC", "INCOME", "W875RX1"] if c in df.columns), None)
    if price_col and rate_col:
        rate = pd.to_numeric(df[rate_col], errors="coerce") / 100.0
        price = pd.to_numeric(df[price_col], errors="coerce")
        price = price / price.iloc[0]
        if inc_col:
            inc = pd.to_numeric(df[inc_col], errors="coerce")
            inc = inc / inc.iloc[0]
            affordability = (rate * price) / inc
            title_suffix = ""
        else:
            affordability = (rate * price)
            title_suffix = " (no income proxy)"
        aff = affordability.dropna().tail(180)
        cone = rolling_percentile_cone(affordability.dropna(), window=60).reindex(aff.index)
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        try:
            ax.fill_between(cone.index, cone["p10"], cone["p90"], alpha=0.2, label="5y cone")
        except Exception:
            pass
        ax.plot(aff.index, aff.values, linewidth=1.8, label="Affordability")
        ax.set_title("Housing Affordability" + title_suffix)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        p = save_fig(fig, OUTDIR, "housing_afford.png")
        insert_image(ws, p, "T42", max_width=440, max_height=260)
        _note(ws, "T41", "Housing: affordability chart inserted")
    else:
        _note(ws, "A42", "Mortgage rate and/or house price index missing — affordability chart skipped")

    # H2: Permits vs Starts
    if "PERMIT" in df.columns or "HOUST" in df.columns:
        cols = [c for c in ["PERMIT", "HOUST"] if c in df.columns]
        sub = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols}).tail(240)
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        sub.plot(ax=ax, linewidth=1.5)
        if "PERMIT" in sub.columns:
            try:
                lead3 = sub["PERMIT"].shift(3)
                lead3.tail(240).plot(ax=ax, linewidth=1.0, alpha=0.5, linestyle="--", label="Permits +3m")
            except Exception:
                pass
        ax.set_title("Permits vs Starts")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        p = save_fig(fig, OUTDIR, "housing_permits_starts.png")
        insert_image(ws, p, "T22", max_width=440, max_height=260)
        _note(ws, "T21", "Housing: permits vs starts chart inserted")
    else:
        _note(ws, "A22", "PERMIT/HOUST not available — permits vs starts chart skipped")

    # H3: Builder Sentiment vs Permits
    nahb_col = next((c for c in ["NAHB", "NAHB_INDEX"] if c in df.columns), None)
    proxy = df.get("F_Housing") if nahb_col is None else pd.to_numeric(df[nahb_col], errors="coerce")
    if proxy is not None and ("PERMIT" in df.columns):
        a = proxy.tail(240)
        b = pd.to_numeric(df["PERMIT"], errors="coerce").tail(240)
        if len(a) > 0 and len(b) > 0:
            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            a.plot(ax=ax, label=nahb_col or "F_Housing")
            ax2 = ax.twinx()
            b.plot(ax=ax2, color="#777777", linewidth=1.2, label="Permits")
            ax.set_title("Builder Sentiment vs Permits")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.25)
            try:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="best")
            except Exception:
                ax.legend(loc="best")
            p = save_fig(fig, OUTDIR, "housing_sentiment.png")
            insert_image(ws, p, "T2", max_width=440, max_height=260)
            _note(ws, "T1", "Housing: sentiment chart inserted")
    else:
        _note(ws, "A2", "NAHB (or F_Housing) and/or PERMIT missing — sentiment chart skipped")


def build_inflation_sheet(df: pd.DataFrame, book, cfg, sheet_name: str = "Inflation") -> None:
    if df is None or df.empty:
        return
    ws = _resolve_sheet(book, sheet_name)

    # I1: Composite + 5y cone
    comp_col = next((c for c in ["F_Inflation", "Inflation_Composite"] if c in df.columns), None)
    s = None
    if comp_col is not None:
        s = pd.to_numeric(df[comp_col], errors="coerce").dropna()
    else:
        s = theme_composite_from_cfg(df, cfg, "inflation")
    if s is not None and len(s) > 0:
        s_plot = s.tail(60) if len(s) > 60 else s
        cone = rolling_percentile_cone(s, window=60).reindex(s_plot.index)
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        try:
            ax.fill_between(cone.index, cone["p10"], cone["p90"], alpha=0.2, label="5y cone")
            ax.plot(cone.index, cone["p50"], linewidth=1.2, alpha=0.8, label="median")
        except Exception:
            pass
        ax.plot(s_plot.index, s_plot.values, linewidth=1.8, label=comp_col)
        ax.set_title("Inflation Composite: 5y cone")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        p = save_fig(fig, OUTDIR, "inflation_cone.png")
        insert_image(ws, p, "T2", max_width=440, max_height=260)
        _note(ws, "T1", "Inflation: cone chart inserted")
    else:
        _note(ws, "J2", "Inflation composite not available — cone chart skipped")

    # I2: Core - Headline spread
    core = df.get("CoreCPI_YoY")
    head = df.get("CPI_YoY")
    if core is None and "CPILFESL" in df.columns:
        core = yoy(df["CPILFESL"]) * 100
    if head is None and "CPIAUCSL" in df.columns:
        head = yoy(df["CPIAUCSL"]) * 100
    if core is not None and head is not None:
        idx = core.dropna().index.intersection(pd.to_numeric(head, errors="coerce").dropna().index)
        s = (pd.to_numeric(core, errors="coerce").reindex(idx) - pd.to_numeric(head, errors="coerce").reindex(idx)).dropna().tail(180)
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        s.plot(ax=ax, linewidth=1.8, label="Core - Headline")
        ax.axhline(0.0, color="#888888", linewidth=1.0)
        ax.set_title("Core - Headline Inflation Spread")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        p = save_fig(fig, OUTDIR, "inflation_core_spread.png")
        insert_image(ws, p, "T22", max_width=440, max_height=260)
        _note(ws, "T21", "Inflation: core-headline chart inserted")
    else:
        _note(ws, "J22", "Core or headline inflation missing — spread chart skipped")

    # I3: Liquidity pulse (M2 YoY & real policy rate)
    m2 = df.get("M2_YoY")
    if m2 is None and "M2SL" in df.columns:
        m2 = yoy(df["M2SL"]) * 100
    real_rate = None
    if "FEDFUNDS" in df.columns:
        if "CPI_YoY" in df.columns:
            real_rate = pd.to_numeric(df["FEDFUNDS"], errors="coerce") - pd.to_numeric(df["CPI_YoY"], errors="coerce")
    curve = df.get("T10Y3M")
    if m2 is not None and (real_rate is not None or curve is not None):
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        pd.to_numeric(m2, errors="coerce").tail(180).plot(ax=ax, label="M2 YoY")
        ax2 = ax.twinx()
        if curve is not None:
            pd.to_numeric(curve, errors="coerce").tail(180).plot(ax=ax2, color="#777777", linewidth=1.2, label="Curve (10y-3m)")
        else:
            pd.to_numeric(real_rate, errors="coerce").tail(180).plot(ax=ax2, color="#777777", linewidth=1.2, label="Real Policy Rate")
        ax.set_title("Liquidity Pulse")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        try:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best")
        except Exception:
            ax.legend(loc="best")
        p = save_fig(fig, OUTDIR, "inflation_liquidity.png")
        insert_image(ws, p, "T42", max_width=440, max_height=260)
        _note(ws, "T41", "Inflation: liquidity chart inserted")
    else:
        _note(ws, "J42", "M2 YoY and/or real rate/curve missing — liquidity chart skipped")


def build_credit_sheet(df: pd.DataFrame, book, cfg, sheet_name: str = "Credit") -> None:
    if df is None or df.empty:
        return
    ws = _resolve_sheet(book, sheet_name)

    vix = pd.to_numeric(df.get("VIXCLS", df.get("VIX")), errors="coerce") if ("VIXCLS" in df.columns or "VIX" in df.columns) else None
    move = make_move(df)
    spr = credit_spread(df)
    parts = [x for x in [vix, move, spr] if x is not None]
    if not parts:
        _note(ws, "A2", "Credit stress inputs missing — stress chart skipped")
        return
    # align
    idx = parts[0].index
    for p in parts[1:]:
        idx = idx.intersection(p.index)
    parts = [p.reindex(idx) for p in parts]
    zs = [zscore(p) for p in parts]
    stress_z = pd.DataFrame(zs).T.mean(axis=1)

    # C1: mini line (last 60m)
    s60 = stress_z.dropna().tail(60)
    if not s60.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        s60.plot(ax=ax, label="Stress z")
        ax.axhline(0.0, color="#888888", linewidth=1.0)
        ax.set_title("Credit/Risk Stress (z)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        p = save_fig(fig, OUTDIR, "credit_stress_line.png")
        insert_image(ws, p, "T2", max_width=440, max_height=260)
        _note(ws, "T1", "Credit: stress line inserted")

    # Histogram bars: latest vs p10/p50/p90
    s_all = stress_z.dropna()
    if not s_all.empty:
        latest = float(s_all.iloc[-1])
        p10, p50, p90 = np.nanpercentile(s_all.values, [10, 50, 90])
        data = pd.Series({"p10": p10, "p50": p50, "p90": p90, "latest": latest})
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        data[["p10", "p50", "p90"]].plot(kind="bar", ax=ax)
        ax.axhline(latest, color="#cc3333", linewidth=2.0, label="latest")
        ax.set_title("Stress distribution: p10/p50/p90 vs latest")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
        p = save_fig(fig, OUTDIR, "credit_stress_bar.png")
        insert_image(ws, p, "T22", max_width=440, max_height=260)
        _note(ws, "T21", "Credit: stress p10/p50/p90 inserted")


