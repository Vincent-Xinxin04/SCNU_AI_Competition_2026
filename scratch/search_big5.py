import os
import glob
import pandas as pd
import sys

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
            if 'big5' in o.lower() or 'big5' in s.lower() or 'hanja' in o.lower() or 'hanja' in s.lower():
                train_records.append({'Subject': s, 'Object': o, 'Label': lbl})

print(f"Total Big5/Hanja matches in train: {len(train_records)}")
for r in train_records:
    print(f"Sub: {r['Subject']} -> Obj: {r['Object']} | Label: {r['Label']}")
