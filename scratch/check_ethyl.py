import os
import glob
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

train_files = glob.glob("dataset/Train_Set/*.csv")
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        matches = df[df['Subject'].astype(str) == 'ethyl acrylate']
        if len(matches) > 0:
            print(f"Label: {lbl}")
            print(matches)
