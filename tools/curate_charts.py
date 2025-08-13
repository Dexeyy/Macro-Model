from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ChartRow:
    sheet: str
    title: str
    series_titles: List[str]
    series_formulas: List[str]


def _load_charts(csv_path: Path) -> List[ChartRow]:
    df = pd.read_csv(csv_path)
    rows: List[ChartRow] = []
    for _, r in df.iterrows():
        rows.append(
            ChartRow(
                sheet=str(r["sheet"]),
                title=str(r["title"]),
                series_titles=[s for s in str(r["series_titles"]).split("|") if s and s != "nan"],
                series_formulas=[s for s in str(r["series_formulas"]).split("|") if s and s != "nan"],
            )
        )
    return rows


def _series_from_titles(monthly: pd.DataFrame, titles: List[str]) -> List[str]:
    out: List[str] = []
    cols = set(monthly.columns)
    for t in titles:
        key = t.strip().split()[-1].upper()
        if key in cols:
            out.append(key)
    return out


def curate_plan(charts_csv: Path, monthly_csv: Path, corr_thresh: float = 0.95) -> Dict[str, Dict[str, List[str]]]:
    rows = _load_charts(charts_csv)
    monthly = pd.read_csv(monthly_csv, index_col=0, parse_dates=[0])
    monthly = monthly.apply(pd.to_numeric, errors="coerce")

    # Build candidates per sheet
    per_sheet: Dict[str, List[Tuple[str, str]]] = {}
    for r in rows:
        cols = _series_from_titles(monthly, r.series_titles)
        if not cols:
            continue
        # choose primary column
        c0 = cols[0]
        per_sheet.setdefault(r.sheet, []).append((r.title, c0))

    plan: Dict[str, Dict[str, List[str]]] = {"keep": {}, "drop": {}}
    for sh, pairs in per_sheet.items():
        if not pairs:
            continue
        # Rank by volatility×coverage (5y)
        def score(col: str) -> float:
            s = monthly[col].dropna()
            last = s.iloc[-60:]
            cov = last.notna().mean()
            vol = last.std(ddof=0) or 0.0
            return float(cov * vol)

        pairs_sorted = sorted(pairs, key=lambda kv: score(kv[1]), reverse=True)
        kept: List[str] = []
        kept_cols: List[str] = []
        for title, col in pairs_sorted:
            drop = False
            for kc in kept_cols:
                c = monthly[[col, kc]].dropna().corr().iloc[0, 1]
                if abs(c) >= corr_thresh:
                    drop = True
                    break
            if not drop:
                kept.append(title)
                kept_cols.append(col)
        drop_titles = [t for t, _ in pairs if t not in kept]
        plan["keep"][sh] = kept
        plan["drop"][sh] = drop_titles
    return plan


def main():
    p = argparse.ArgumentParser(description="Curate charts based on correlations and volatility")
    p.add_argument("--charts", required=True)
    p.add_argument("--monthly", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--corr", type=float, default=0.95)
    args = p.parse_args()
    plan = curate_plan(Path(args.charts), Path(args.monthly), corr_thresh=args.corr)
    Path(args.out).write_text(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()


