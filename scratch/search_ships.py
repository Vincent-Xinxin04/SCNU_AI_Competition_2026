import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

ship_prefixes = ('SS ', 'MS ', 'MV ', 'RV ')
ship_mask = test_df['Subject'].astype(str).str.startswith(ship_prefixes)

for idx in test_df[ship_mask].index:
    sub = test_df.loc[idx, 'Subject']
    obj = test_df.loc[idx, 'Object']
    pred = sub_df.loc[idx, 'Label']
    print(f"Row {idx:4d} | {sub} -> {obj} | Pred: {pred}")
