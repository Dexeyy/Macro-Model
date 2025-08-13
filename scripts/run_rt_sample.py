import os
import sys
import pandas as pd

from src.data.processors import create_advanced_features


def main():
    retro_path = os.path.join('Data', 'processed', 'macro_features_retro.parquet')
    rt_path = os.path.join('Data', 'processed', 'macro_features_rt.parquet')
    path = retro_path if os.path.exists(retro_path) else (rt_path if os.path.exists(rt_path) else None)
    print('USING:', path)
    if not path:
        print('No processed parquet found; skipping sample build.')
        return

    df = pd.read_parquet(path)
    rt = create_advanced_features(df, mode='rt')

    # Coverage snapshot
    out_dir = os.path.join('Output', 'diagnostics')
    cov_path = os.path.join(out_dir, 'coverage_snapshot.csv')
    if os.path.exists(cov_path):
        print('COVERAGE (last 12m):')
        cov = pd.read_csv(cov_path)
        # show last 12 months per factor
        print(cov.tail(12 * max(1, cov.get('factor', pd.Series()).nunique() or 1)).to_string(index=False))
    else:
        # Fallback quick coverage based on factor NaNs
        fac_cols = [c for c in rt.columns if c.startswith('F_')]
        if 'FinConditions_Composite' in rt.columns:
            fac_cols.append('FinConditions_Composite')
        rows = []
        for fc in fac_cols:
            s = rt[fc]
            rows.append({
                'factor': fc,
                'valid_count_last12': int(s.tail(12).notna().sum()),
                'total_last12': int(min(12, len(s)))
            })
        print('COVERAGE (valid counts, last 12m):')
        print(pd.DataFrame(rows).to_string(index=False))

    # FinConditions detected inputs at latest date
    latest = rt.index.max()
    inputs = [c for c in ['NFCI', 'VIX', 'MOVE', 'CorporateBondSpread'] if c in rt.columns]
    avail = {c: (pd.notna(rt.loc[latest, c]) if latest in rt.index else False) for c in inputs}
    print('FINCONDITIONS inputs at latest date:', avail)


if __name__ == '__main__':
    sys.exit(main() or 0)


