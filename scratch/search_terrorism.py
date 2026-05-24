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
            if 'terrorism' in s or 'terrorism' in o:
                train_records.append({'Subject': s, 'Object': o, 'Label': lbl})

print(f"Total matches in train: {len(train_records)}")
for r in train_records[:30]:
    print(f"Sub: {r['Subject']} -> Obj: {r['Object']} | Label: {r['Label']}")
