import pandas as pd
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

matches = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    # Subject is integer, Object is integer
    if re.match(r'^\d+$', s) and re.match(r'^\d+$', o):
        # check if object is length of subject
        if len(s) == int(o):
            matches.append((idx, s, o, pred))

print(f"Total decimal digits candidates: {len(matches)}")
for m in matches:
    print(f"Row {m[0]}: {m[1]} -> {m[2]} | Pred: {m[3]}")
