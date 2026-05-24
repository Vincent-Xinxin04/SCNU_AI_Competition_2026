import os
import glob
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load train data
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
            train_records.append({'Subject': s, 'Object': o, 'Label': lbl})
train_df = pd.DataFrame(train_records)

# Search for "dub" or "buk" or "javor" or "lípa" in Train Subject
tree_matches = train_df[train_df['Subject'].str.contains(r'\b(dub|buk|javor|lípa|klen)\b', case=False, na=False)]
print("--- Tree matches in Train ---")
print(tree_matches.head(30))

# Search for "solitary" in Train
solitary_matches = train_df[train_df['Subject'].str.contains('solitary', case=False, na=False) | train_df['Object'].str.contains('solitary', case=False, na=False)]
print("\n--- Solitary matches in Train ---")
print(solitary_matches)
