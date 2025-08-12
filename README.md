## Macro Regime Analysis Model (v2)

A production‑oriented framework for macroeconomic regime detection, probabilities, stability post‑processing, explainability, and Excel reporting.

### Highlights

- Theme factor composites `F_*` (Growth, Inflation, Liquidity, CreditRisk, Housing, External); optional PCA `PC_*`
- Pluggable models: Rule, KMeans, GMM, HMM, HSMM (fallback), MSDyn (Markov‑Switching on MacroMomentum)
- Per‑model probabilities (`*_Prob_*`) + probability‑based ensemble with confirmation and min‑duration stabilizers
- Validators and diagnostics: duration sanity, local mean/variance tests (Chow‑like), change‑point hints
- Real‑time (“rt”; publication‑lag‑disciplined) and Retro (“retro”) feature modes
- Vintage‑aware ALFRED fetching (no look‑ahead): each date uses observations available as‑of that date; publication lags applied before transforms
- Excel dashboard: regime timeline, stacked probability strip, low‑confidence shading, per‑theme signature charts, KPI confidence
- Performance analytics with regime separation scorecard (ANOVA) and portfolio helpers

---

## Architecture (high‑level)

```
FRED / Yahoo  →  Processing →  F_* / (optional) PC_*
                                  │
                                  ▼
                   Models (Rule / KMeans / GMM / HMM / HSMM / MSDyn)
                 per‑model labels + probabilities ( *_Prob_* )
                                  │
                                  ▼
                Ensemble (average probs → confirm → min‑duration)
                                  │
                                  ▼
        Validators (duration, Chow/Wald‑like, change‑points) → Diagnostics
                                  │
                                  ▼
   Performance (regime scorecard) + Excel (timeline + prob strip + KPIs)
```

---

## Installation

```
pip install -r requirements.txt
```

Optional/extra packages used if available: `ruptures`, `hsmmlearn`, `pomegranate`.

Set your FRED key (optional) in `.env`:

```
FRED_API_KEY=your_key
```

---

## Configuration (config/regimes.yaml)

Sane defaults are applied automatically if the file is missing or partial.

```yaml
models: [rule, gmm, hmm]        # optionally include: hsmm, msdyn, kmeans

hmm:
  n_states_range: [2, 6]
  covariance_type: full

postprocess:
  min_duration: 3               # minimum run length after confirmation
  prob_threshold: 0.7           # probability required to start confirming a switch
  consecutive: 2                # consecutive periods above threshold to confirm switch

ensemble:
  confirm_consecutive: 2        # used when building ensemble labels

run:
  mode: retro                   # rt | retro (feature alignment)
  bundle: coincident            # coincident | coincident_plus_leading

publication_lags:              # months; applied prior to transforms/z-scores
  CPIAUCSL: 1
  UNRATE: 0

factors:                        # config-driven factor builder
  F_Growth:
    bases: [INDPRO, PAYEMS, GDPC1, CUMFNS]
    min_k: 2
  F_Inflation:
    bases: [CPIAUCSL, PCEPI, PPICMM]

series_types:                   # z-score window presets per series
  VIX: fast                     # fast: (36,18), typical: (60,24), slow: (120,36)
  NFCI: slow

outliers:
  method: hampel                # or winsorize
  zmax: 6

pca:
  enabled: false

themes:                         # optional explicit theme column lists
  growth: []
  inflation: []
  liquidity: []
  credit_risk: []
  housing: []
  external: []
```

Key behaviors toggled here:
- Change models list (e.g., add `hsmm`, `msdyn`)
- Adjust stabilizers (`min_duration`, `prob_threshold`, `consecutive`)
- Switch between real‑time (`rt`) and `retro` features
- Select feature bundle (`coincident` vs `coincident_plus_leading`)

---

## Real‑time vs Retro features

- Retro: fully revised history
- Real‑time: publication‑lag alignment via `LagAligner`; you may set per‑series lags under `publication_lags:` in `config/regimes.yaml` (optional). Both are persisted:
  - `Data/processed/macro_features_retro.parquet`
  - `Data/processed/macro_features_rt.parquet`

Policy for rolling z-scores (monthly):
- fast/volatile (VIX, MOVE, oil): window=36, min_periods=18
- typical macro (CPI YoY, INDPRO YoY, credit spreads): window=60, min_periods=24
- slow/structural (NFCI): window=120, min_periods=36
- unknown types: min_periods ≈ 0.4 × window (cap 36). Auto-tuner evaluates (36,18),(60,24),(120,36) by coverage/smoothness/low-flips.

Real-time discipline: (1) apply lags, (2) recompute transforms on lagged panel, (3) compute robust z and build factors. All rolling use past windows only (no look-ahead).

Financial Conditions synonyms: `VIXCLS→VIX`, `^MOVE→MOVE`, and `CorporateBondSpread` from `credit_spread` or `BAA−AAA`.

Diagnostics: per-factor coverage count/ratio and `low_coverage` flags; warnings for <50% coverage in first 24 months. Snapshot saved to `Output/diagnostics/coverage_snapshot.csv`.

---

## Pluggable Models

- Rule: simple heuristics, one‑hot probabilities
- KMeans: clustering with soft probabilities (distance‑based)
- GMM: Gaussian mixture with full covariance + soft probabilities
- HMM: hmmlearn with BIC model selection; smoothed probabilities; confirmation + min‑duration applied
- HSMM: tries native HSMM libs; falls back to HMM + explicit duration enforcement
- MSDyn: Markov‑Switching Dynamic Regression on MacroMomentum (mean of `F_Growth`,`F_Inflation`,`F_Liquidity`, or `PC_*`)

Outputs include both legacy `<Model>` and `<Model>_Regime` label columns, and probability columns `<Model>_Prob_*` where supported.

---

## Ensemble & Validators

- Ensemble probabilities: average aligned matrices; row‑wise renormalize
- Ensemble labels: argmax → probability confirmation (`prob_threshold`, `consecutive`) → min‑duration smoothing
- Diagnostics recorded in `Validation_Flags` (per‑timestamp dict):
  - `duration` stats (mean/median/min/too_short_share)
  - Local mean/variance tests (Chow/Wald‑like) around each proposed switch
  - (Optional) change‑point hints on `F_Growth` / `F_Inflation`

---

## Excel Dashboard

Generated by `write_excel_report` and `build_macro_dashboard`:
- Regime timeline chart with stacked `Ensemble_Prob_*` probability strip (last 180 months)
- Low‑confidence shading when `Ensemble_Confidence < 0.6`
- Theme KPI cards include “Confidence: …%” (latest row‑wise max(Ensemble_Prob_*))
- Per‑theme signature charts:
  - Growth: Business cycle clock (GDP YoY vs UNRATE)
  - Inflation: CPI contributions stacked + real policy rate
  - Credit: IG vs HY proxy (BAA‑AAA), NFCI if present
  - Housing: Permits vs Starts + mortgage rate
  - FX/Commodities: WTI and Dollar Index (TWEX)

Outputs:
- `Output/Macro_Reg_Report_with_category_dashboards.xlsm`
- `Output/excel_charts/*.png`

---

## Performance Analytics

Use `PerformanceAnalytics` to produce:
- Regime separation scorecard (`Output/diagnostics/regime_scorecard.csv`): per‑regime mean/vol/Sharpe, max drawdown, ANOVA p‑value across regimes (monthly returns)
- Portfolio helpers and visualization utilities

---

## CLI Quickstart

Fit regimes (respects YAML `run.bundle` and `run.mode` unless overridden):

``` 
python -m cli regimes fit --mode rt --bundle coincident --models hmm gmm rule --n-regimes 4
```

Build Excel dashboard:

```
python -m cli excel build \
  --template src/excel/Macro\ Reg\ Template.xlsm \
  --out Output/Macro_Reg_Report_with_category_dashboards.xlsm
```

End‑to‑end (equivalent of `main.py`):

```
python -m cli run full
```

Run end‑to‑end with custom dates and outlier/coverage options via `main.py`:

```
python main.py --start 1990-01-01 --end 2025-06-01 --outlier hampel --coverage_k 2
```

---

## Outputs (key files)

- Processed features
  - `Data/processed/macro_features_retro.parquet`
  - `Data/processed/macro_features_rt.parquet`
- Regimes (labels + probabilities + validators)
  - `Data/processed/macro_data_with_regimes.csv`
- Diagnostics & Explainability
  - `Output/diagnostics/regime_scorecard.csv`
  - `Output/diagnostics/regime_profiles.json`
- Excel & charts
  - `Output/Macro_Reg_Report_with_category_dashboards.xlsm`
  - `Output/excel_charts/*.png`

---

## Testing & CI

Run the synthetic smoke tests and postprocess unit tests:

```
pytest -q
```

CI (GitHub Actions) runs `pytest -q` and `mypy` on push/PR to `main`.

---

## Contributing

Issues and PRs welcome. Please run tests locally before opening a PR.

---

## License

MIT

