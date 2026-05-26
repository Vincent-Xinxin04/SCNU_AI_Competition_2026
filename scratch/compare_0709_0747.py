import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df1 = pd.read_csv('result/submission_0.709.csv', encoding='utf-8-sig')
df2 = pd.read_csv('result/submission_0.747.csv', encoding='utf-8-sig')

col1 = [c for c in df1.columns if c.lower() == 'label'][0]
col2 = [c for c in df2.columns if c.lower() == 'label'][0]

diff = df1[col1] != df2[col2]
print('Total differences:', diff.sum())
for idx in diff[diff].index:
    print(f"Row {idx:4d}: {df1.loc[idx, 'Subject']} -> {df1.loc[idx, 'Object']} | 0.709: '{df1.loc[idx, col1]}' | 0.747: '{df2.loc[idx, col2]}'")
