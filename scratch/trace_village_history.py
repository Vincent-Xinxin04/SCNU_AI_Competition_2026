import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

rows_to_check = [526, 1146, 1172, 1349, 2604, 3933]

for fn in ['result/submission_0.709.csv', 'result/submission_0.747.csv', 'result/submission_0.7791.csv', 'result/submission_0.782.csv']:
    try:
        df = pd.read_csv(fn, encoding='utf-8-sig')
        lbl_col = [c for c in df.columns if c.lower() == 'label'][0]
        results = [f"Row {r}: {df.loc[r, lbl_col]}" for r in rows_to_check]
        print(f"{fn}: {', '.join(results)}")
    except Exception as e:
        print(f"Error reading {fn}: {e}")
