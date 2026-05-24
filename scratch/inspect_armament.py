import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

matches = []
for idx, row in test_df.iterrows():
    pred = sub_df.loc[idx, 'Label']
    if pred == 'armament':
        matches.append((idx, row['Subject'], row['Object']))

print(f"Total test rows predicted as 'armament': {len(matches)}")
for m in matches:
    print(f"Row {m[0]}: {m[1]} -> {m[2]}")
