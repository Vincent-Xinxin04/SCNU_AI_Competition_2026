import os
import glob
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Find all station names from Train
train_files = glob.glob("dataset/Train_Set/*.csv")
stations = set()
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    if lbl in ['adjacent station', 'Deutsche Bahn station category']:
        df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]
        if 'Subject' in df.columns and 'Object' in df.columns:
            for s in df['Subject'].dropna():
                stations.add(str(s).strip())
            for o in df['Object'].dropna():
                stations.add(str(o).strip())

print(f"Total unique stations collected from train: {len(stations)}")

# Check test set
test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

matches = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    if s in stations:
        try:
            val = float(o)
            if val < 50:
                matches.append((idx, s, o, pred))
        except ValueError:
            pass

print(f"Total test matches: {len(matches)}")
for m in matches:
    print(f"Row {m[0]}: {m[1]} -> {m[2]} | Pred: {m[3]}")
