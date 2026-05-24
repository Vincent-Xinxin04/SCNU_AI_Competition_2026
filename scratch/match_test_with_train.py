import os
import glob
import pandas as pd
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load labels.txt
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# 2. Load train data
train_files = glob.glob("dataset/Train_Set/*.csv")
train_map_pair = {} # (Subject, Object) -> Label
train_map_subj = {} # Subject -> set of Labels
train_map_obj = {}  # Object -> set of Labels

# Store all training data for pattern analysis
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
            train_map_pair[(s, o)] = lbl
            if s not in train_map_subj:
                train_map_subj[s] = set()
            train_map_subj[s].add(lbl)
            if o not in train_map_obj:
                train_map_obj[o] = set()
            train_map_obj[o].add(lbl)
            train_records.append({'Subject': s, 'Object': o, 'Label': lbl})

train_df = pd.DataFrame(train_records)

# 3. Load submission
sub_df = pd.read_csv("result/submission_0.7791.csv")
test_df = pd.read_csv("dataset/test.csv")

# Identify missing labels in submission
predicted_labels = set(sub_df['Label'].unique())
missing_labels = [l for l in labels if l not in predicted_labels]
print(f"Number of missing labels in submission: {len(missing_labels)}")
print("First 15 missing labels:", missing_labels[:15])

# Find exact pair matches
exact_matches = 0
exact_fixed = 0
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    if (s, o) in train_map_pair:
        true_lbl = train_map_pair[(s, o)]
        exact_matches += 1
        if pred != true_lbl:
            print(f"Row {idx}: Exact pair match mismatch. (Sub: {s}, Obj: {o}) | Pred: {pred} -> True: {true_lbl}")
            sub_df.loc[idx, 'Label'] = true_lbl
            exact_fixed += 1

print(f"Fixed {exact_fixed} exact pair matches out of {exact_matches}")

# Save the updated submission
sub_df.to_csv("scratch/submission_fixed_exact.csv", index=False)
