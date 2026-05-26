import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

keywords = ('route', 'bridge', 'highway', 'road', 'pass', 'tunnel')

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    s_lower = s.lower()
    if any(k in s_lower for k in keywords):
        print(f"Row {idx:4d} | Subject: '{s}' | Object: '{o}' | Pred: '{pred}'")
