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
        matches = df[(df['Subject'].astype(str) == '2018 Clásico RCN') & (df['Object'].astype(str) == 'Óscar Sevilla')]
        if len(matches) > 0:
            print(f"2018 Clásico RCN, Óscar Sevilla found in: {lbl}")
            
        matches_gomez = df[(df['Subject'].astype(str) == '2017 Clásico RCN') & (df['Object'].astype(str) == 'Javier Gómez')]
        if len(matches_gomez) > 0:
            print(f"2017 Clásico RCN, Javier Gómez found in: {lbl}")
