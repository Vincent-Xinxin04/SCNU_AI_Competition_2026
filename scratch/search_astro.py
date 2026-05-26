import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

# Find astronomical subjects
astro_prefixes = ('HD', 'Gliese', '2MASS', 'TYC', 'V*', 'Cl*', 'SN', 'Tupi', 'Makemake', '50000')
astro_mask = test_df['Subject'].astype(str).str.startswith(astro_prefixes)

for idx in test_df[astro_mask].index:
    sub = test_df.loc[idx, 'Subject']
    obj = test_df.loc[idx, 'Object']
    pred = sub_df.loc[idx, 'Label']
    print(f"Row {idx:4d} | {sub} -> {obj} | Pred: {pred}")
