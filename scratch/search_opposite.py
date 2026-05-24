import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

opposites = [('class', 'individual'), ('falsity', 'truth'), ('difference', 'identity'),
             ('individual', 'class'), ('truth', 'falsity'), ('identity', 'difference')]

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip().lower()
    o = str(row['Object']).strip().lower()
    for osub, oobj in opposites:
        if s == osub and o == oobj:
            print(f"Row {idx}: {row['Subject']} -> {row['Object']} | Pred: {sub_df.loc[idx, 'Label']}")
