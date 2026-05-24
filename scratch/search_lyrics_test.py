import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    if 'Paul Anka' in s or 'Paul Anka' in o or 'Tony Hatch' in s or 'Tony Hatch' in o or 'Mike Stoller' in s or 'Mike Stoller' in o:
        print(f"Row {idx}: {s} -> {o} | Pred: {sub_df.loc[idx, 'Label']}")
