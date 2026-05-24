import os
import glob
import pandas as pd
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

# Collect all country names from Train Set
train_files = glob.glob("dataset/Train_Set/*.csv")
countries = set()
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    if lbl in ['country', 'country of origin', 'country of citizenship', 'basin country']:
        df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]
        if 'Subject' in df.columns and 'Object' in df.columns:
            for s in df['Subject'].dropna():
                countries.add(str(s).strip())
            for o in df['Object'].dropna():
                countries.add(str(o).strip())

# Clean countries list to keep only actual country names
actual_countries = {c for c in countries if len(c) < 30 and not re.search(r'\d', c)}
print(f"Total countries collected: {len(actual_countries)}")

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

matches = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    if s in actual_countries:
        try:
            val = float(o)
            matches.append((idx, s, o, pred))
        except ValueError:
            pass

print(f"Total country-numeric matches in test: {len(matches)}")
for m in matches:
    print(f"Row {m[0]}: {m[1]} -> {m[2]} | Pred: {m[3]}")
