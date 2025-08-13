import os
import json
import pandas as pd

from src.excel.excel_live import group_features


def main():
    # Prefer RT features
    p_rt = os.path.join('Data', 'processed', 'macro_features_rt.parquet')
    p_retro = os.path.join('Data', 'processed', 'macro_features_retro.parquet')
    path = p_rt if os.path.exists(p_rt) else p_retro
    if not os.path.exists(path):
        raise FileNotFoundError('No processed features parquet found.')

    df = pd.read_parquet(path)

    # Build canonical mapping like processors.py
    mapping = group_features(df)
    key_map = {
        'Growth & Labour': 'growth',
        'Inflation & Liquidity': 'inflation',
        'Credit & Risk': 'credit_risk',
        'Housing': 'housing',
        'FX & Commodities': 'external',
    }
    canon = {}
    for k, cols in mapping.items():
        canon.setdefault(key_map.get(k, k), []).extend(cols)

    themes = ['growth', 'inflation', 'credit_risk', 'housing', 'external']
    out = {}
    for t in themes:
        cols = [c for c in canon.get(t, []) if c in df.columns]
        nn = (
            df[cols].notna().sum().sort_values().to_dict() if cols else {}
        )
        out[t] = {
            'num_inputs': len(cols),
            'inputs': nn,
        }

    # Print concise view and save full JSON
    print('Using features from:', path)
    for t in themes:
        info = out.get(t, {})
        cols = info.get('inputs', {})
        print(f"\n[{t}] inputs: {info.get('num_inputs', 0)}")
        # Show up to 15 with least coverage
        items = list(cols.items())
        items.sort(key=lambda kv: kv[1])
        for name, cnt in items[:15]:
            print(f"  {name}: {cnt}")
        if items:
            print(f"  ... min={items[0][1]}, max={items[-1][1]}")

    os.makedirs(os.path.join('Output', 'diagnostics'), exist_ok=True)
    out_path = os.path.join('Output', 'diagnostics', 'composite_inputs_rt.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('\nSaved full mapping to', out_path)


if __name__ == '__main__':
    main()


