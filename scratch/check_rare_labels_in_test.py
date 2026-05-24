import os
import glob
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Find all rare labels (train count <= 5)
train_files = glob.glob("dataset/Train_Set/*.csv")
rare_labels = {}
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        if len(df) <= 5:
            rare_labels[lbl] = df

print(f"Number of rare labels (count <= 5): {len(rare_labels)}")

# Load test and sub
test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

# 2. Search test set for any exact/close matches of Subject/Object from rare label train sets
matches = []
for lbl, train_df_rare in rare_labels.items():
    # Collect all unique subjects and objects for this rare label
    rare_subjs = set(train_df_rare['Subject'].astype(str).str.lower())
    rare_objs = set(train_df_rare['Object'].astype(str).str.lower())
    
    # We want to filter out very common objects like "1", "0", etc.
    rare_objs = {o for o in rare_objs if len(o) > 2 and o not in ['yes', 'no']}
    
    for idx, row in test_df.iterrows():
        s = str(row['Subject']).strip().lower()
        o = str(row['Object']).strip().lower()
        pred = sub_df.loc[idx, 'Label']
        
        # Check matching
        match_type = None
        if s in rare_subjs:
            match_type = "Subject match"
        elif o in rare_objs:
            match_type = "Object match"
            
        if match_type:
            matches.append({
                'row': idx,
                'Subject': row['Subject'],
                'Object': row['Object'],
                'Pred': pred,
                'Rare_Label': lbl,
                'Match_Type': match_type
            })

print(f"Total potential rare label matches in test: {len(matches)}")
df_matches = pd.DataFrame(matches)
if len(df_matches) > 0:
    for lbl in rare_labels.keys():
        subset = df_matches[df_matches['Rare_Label'] == lbl]
        if len(subset) > 0:
            print(f"\n--- Matches for Rare Label '{lbl}' ---")
            for idx, r in subset.iterrows():
                print(f"Row {r['row']}: (Sub: {r['Subject']}, Obj: {r['Object']}) | Pred: {r['Pred']} | Match: {r['Match_Type']}")
