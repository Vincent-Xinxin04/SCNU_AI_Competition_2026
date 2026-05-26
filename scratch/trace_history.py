import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

for fn in ['result/submission_0.709.csv', 'result/submission_0.747.csv', 'result/submission_0.7791.csv', 'result/submission_0.782.csv']:
    try:
        df = pd.read_csv(fn, encoding='utf-8-sig')
        # find the column that matches 'label' case-insensitively
        lbl_col = [c for c in df.columns if c.lower() == 'label'][0]
        print(f"{fn}: Row 1157 = {df.loc[1157, lbl_col]}, Row 1979 = {df.loc[1979, lbl_col]}, Row 3457 = {df.loc[3457, lbl_col]}")
    except Exception as e:
        print(f"Error reading {fn}: {e}")
