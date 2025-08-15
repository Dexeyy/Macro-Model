from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

def debug_chart_pipeline():
    # Print quick diagnostics so the user can see what the builder sees
    feat = Path('Data/processed/macro_data_featured.csv')
    print('feature_csv_exists:', feat.exists())
    if feat.exists():
        try:
            df = pd.read_csv(feat, index_col=0, parse_dates=[0])
            print('feature_csv_rows,cols:', df.shape)
            print('sample_cols:', list(df.columns[:10]))
        except Exception as e:
            print('feature_csv_read_error:', e)
    outdir = Path('Output/excel_charts')
    print('outdir_exists:', outdir.exists())
    if outdir.exists():
        print('outdir_files:', [p.name for p in outdir.glob('*')])

if __name__ == '__main__':
    debug_chart_pipeline()


