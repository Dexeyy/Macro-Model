from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

import typer

from src.utils.helpers import load_yaml_config

app = typer.Typer(name="macro", add_completion=False)


# ---------------------------- helpers ---------------------------------

def _save_yaml_config(cfg: dict, path: str = "config/regimes.yaml") -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        typer.secho("PyYAML not installed; cannot persist config overrides.", fg=typer.colors.YELLOW)
        return

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as exc:
        typer.secho(f"Failed to write YAML config: {exc}", fg=typer.colors.RED)


def _load_processed_monthly(mode: Optional[str] = None) -> Optional[object]:
    import pandas

    # Prefer mode-specific parquet if requested and available
    candidates = []
    if mode:
        m = (mode or "").lower()
        if m in ("rt", "real-time", "realtime"):
            candidates.append(Path("Data/processed/macro_features_rt.parquet"))
        elif m in ("retro", "revised", "full"):
            candidates.append(Path("Data/processed/macro_features_retro.parquet"))

    # Fallbacks
    candidates.extend([
        Path("Data/processed/macro_features.parquet"),
        Path("Data/processed/merged.parquet"),
        Path("Data/processed/macro_data_featured.csv"),
    ])
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".parquet":
                return pandas.read_parquet(p)
            return pandas.read_csv(p, index_col=0, parse_dates=[0])
    return None


def _open_file(path: str) -> None:
    """Open a file with the default OS handler (best-effort, non-blocking)."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        try:
            import typer  # ensure typer exists in this context
            typer.secho(f"Cannot open: file not found at {abs_path}", fg=typer.colors.RED)
        except Exception:
            pass
        return
    try:
        if os.name == "nt":
            try:
                os.startfile(abs_path)  # type: ignore[attr-defined]
            except Exception:
                subprocess.Popen(["cmd", "/c", "start", "", abs_path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
    except Exception as exc:
        try:
            import typer
            typer.secho(f"Failed to open {abs_path}: {exc}", fg=typer.colors.YELLOW)
        except Exception:
            pass


# ---------------------------- run group --------------------------------

run_app = typer.Typer()
app.add_typer(run_app, name="run")


@run_app.command("full")
def run_full(
    mode: Optional[str] = typer.Option(None, "--mode", help="Feature mode: 'rt' (real-time) or 'retro' (revised)")
) -> None:
    """Run the full pipeline (equivalent to main.py).

    If --mode is provided, it updates YAML run.mode for this project so downstream uses RT/RETRO consistently.
    """
    import pandas as pd
    from main import (
        fetch_and_process_data,
        classify_regimes,
        analyze_regime_performance,
        create_visualizations,
        create_portfolios,
    )
    from src.excel.build_macro_dashboard import build_inplace
    from src.excel.excel_live import write_excel_report

    # Respect requested mode by persisting to YAML (used by fetch_and_process_data)
    if mode:
        m = (mode or "").lower()
        if m in ("rt", "real-time", "realtime", "retro", "revised", "full"):
            cfg = load_yaml_config() or {}
            run_cfg = cfg.get("run") or {}
            run_cfg["mode"] = "rt" if m in ("rt", "real-time", "realtime") else "retro"
            cfg["run"] = run_cfg
            _save_yaml_config(cfg)

    macro_data, asset_returns = fetch_and_process_data()
    macro_data_with_regimes = classify_regimes(macro_data)
    # Ensure downstream receives a DataFrame, not a Series
    if isinstance(macro_data_with_regimes, pd.Series):
        name = macro_data_with_regimes.name or "Regime"
        df_with_regime = macro_data.copy()
        df_with_regime[name] = macro_data_with_regimes
        macro_data_with_regimes = df_with_regime

    analysis_results = analyze_regime_performance(macro_data_with_regimes, asset_returns)
    create_visualizations(macro_data_with_regimes, analysis_results)
    create_portfolios(analysis_results)

    # Build Excel workbook and open it
    out_wb = "Output/Macro_Reg_Report_with_category_dashboards.xlsm"
    template = "src/excel/Macro Reg Template.xlsm"
    try:
        # First push data + probabilities (Confidence in KPI) into the template
        write_excel_report(
            monthly_df=macro_data,
            regime_df=macro_data_with_regimes,
            template_path=template,
            output_path=out_wb,
        )
        # Then layer charts/clean layout in-place
        build_inplace(out_wb, out=out_wb)
        abs_out = os.path.abspath(out_wb)
        typer.secho(f"Workbook saved: {abs_out}", fg=typer.colors.GREEN)
        _open_file(abs_out)
    except Exception as exc:
        typer.secho(f"Excel build step failed: {exc}", fg=typer.colors.YELLOW)


# --------------------------- regimes group ------------------------------

regimes_app = typer.Typer()
app.add_typer(regimes_app, name="regimes")


@regimes_app.command("fit")
def regimes_fit(
    models: List[str] = typer.Option(None, "--models", help="Models to run (e.g. hmm gmm rule)"),
    use_pca: bool = typer.Option(False, "--use-pca", help="Use PC_* features if available"),
    n_regimes: int = typer.Option(4, "--n-regimes", help="Target number of regimes for clustering models"),
    mode: Optional[str] = typer.Option(None, "--mode", help="Feature mode: 'rt' (real-time) or 'retro' (revised)"),
    bundle: Optional[str] = typer.Option("coincident", "--bundle", help="Feature bundle: 'coincident' or 'coincident_plus_leading'"),
) -> None:
    """Fit selected regime models and save results alongside processed data."""
    import pandas as pd
    from src.models.regime_classifier import fit_regimes
    from config import config as cfg_paths  # output dir
    from src.utils.helpers import save_data

    df = _load_processed_monthly(mode)
    if df is None or df.empty:
        raise typer.Exit(code=1)

    # Optionally drop PC_* if use_pca is False
    if not use_pca:
        df = df.drop(columns=[c for c in df.columns if isinstance(c, str) and c.startswith("PC_")], errors="ignore")

    # Optionally override YAML models list
    if models:
        current = load_yaml_config() or {}
        current["models"] = list(models)
        _save_yaml_config(current)

    out = fit_regimes(df, features=None, n_regimes=int(n_regimes), mode=mode, bundle=bundle)
    # Persist merged file as in main.py
    merged = df.join(out, how="left")
    out_path = Path(cfg_paths.PROCESSED_DATA_DIR) / "macro_data_with_regimes.csv"
    save_data(merged, str(out_path))
    typer.secho(f"Saved regimes to {out_path}", fg=typer.colors.GREEN)


# ---------------------------- excel group -------------------------------

excel_app = typer.Typer()
app.add_typer(excel_app, name="excel")


@excel_app.command("build")
def excel_build(
    template: str = typer.Option("src/excel/Macro Reg Template.xlsm", "--template", help="Path to Excel template"),
    out: str = typer.Option(
        "Output/Macro_Reg_Report_with_category_dashboards.xlsm", "--out", help="Output workbook path"
    ),
    open_after: bool = typer.Option(False, "--open", help="Open the workbook after saving"),
    also_xlsx: bool = typer.Option(False, "--also-xlsx", help="Additionally save an .xlsx copy for viewers that block macros"),
) -> None:
    """Build the Excel report with category dashboards and probabilities."""
    from src.excel.build_macro_dashboard import build_inplace
    from openpyxl import load_workbook

    build_inplace(template, out=out)
    abs_out = os.path.abspath(out)
    typer.secho(f"Workbook saved: {abs_out}", fg=typer.colors.GREEN)
    xlsx_path = None
    if also_xlsx:
        try:
            wb = load_workbook(abs_out, keep_vba=True)
            xlsx_path = os.path.splitext(abs_out)[0] + ".xlsx"
            wb.save(xlsx_path)
            typer.secho(f"Also saved non-macro copy: {xlsx_path}", fg=typer.colors.GREEN)
        except Exception as exc:
            typer.secho(f"Failed to create .xlsx copy: {exc}", fg=typer.colors.YELLOW)
    if open_after:
        _open_file(xlsx_path or abs_out)


if __name__ == "__main__":
    app()




