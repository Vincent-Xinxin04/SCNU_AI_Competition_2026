import os
import glob
import pandas as pd
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

train_files = glob.glob("dataset/Train_Set/*.csv")
train_records = []
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        for idx, row in df.iterrows():
            s = str(row['Subject']).strip()
            o = str(row['Object']).strip()
            # If subject looks like "Month Year" and object is "Year"
            if re.match(r'^[A-Za-z]+\s+\d{4}$', s) and re.match(r'^\d{4}$', o):
                train_records.append({'Subject': s, 'Object': o, 'Label': lbl})

print(f"Total matches in train: {len(train_records)}")
for r in train_records[:30]:
    print(f"Sub: {r['Subject']} -> Obj: {r['Object']} | Label: {r['Label']}")
