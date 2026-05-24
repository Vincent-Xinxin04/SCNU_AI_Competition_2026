import pandas as pd
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

matches = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    if 'station' in s.lower() or 'station' in o.lower() or 'bahnhof' in s.lower() or 'bahnhof' in o.lower():
        try:
            val = float(o)
            # if it is a small integer, let's print it
            if val < 50:
                matches.append((idx, s, o, pred))
        except ValueError:
            pass

print(f"Total small number matches for stations in test: {len(matches)}")
for m in matches:
    print(f"Row {m[0]}: {m[1]} -> {m[2]} | Pred: {m[3]}")
