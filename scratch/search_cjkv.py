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
    
    # Check if both contain Chinese characters or are CJKV characters
    # Unicode range for CJK Unified Ideographs: 4E00 - 9FFF
    # Also check if object has U+ or similar
    if re.search(r'[\u4e00-\u9fff]', s) or re.search(r'[\u4e00-\u9fff]', o):
        matches.append((idx, s, o, pred))

print(f"Total CJK matches in test: {len(matches)}")
for m in matches:
    print(f"Row {m[0]}: {m[1]} -> {m[2]} | Pred: {m[3]}")
