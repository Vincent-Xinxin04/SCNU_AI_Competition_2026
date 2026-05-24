import pandas as pd
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

matches = []
for idx, row in test_df.iterrows():
    o = str(row['Object']).strip()
    if re.match(r'^t\d+$', o):
        matches.append((idx, row['Subject'], o, sub_df.loc[idx, 'Label']))

print(f"Total objects matching t\\d+: {len(matches)}")
for m in matches:
    print(f"Row {m[0]}: {m[1]} -> {m[2]} | Pred: {m[3]}")
