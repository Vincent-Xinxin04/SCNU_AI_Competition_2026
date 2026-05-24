import os
import glob
import pandas as pd
import sys
import re

# Load train data for missing labels
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

sub_df = pd.read_csv("result/submission_0.7791.csv")
test_df = pd.read_csv("dataset/test.csv")

predicted_labels = set(sub_df['Label'].unique())
missing_labels = [l for l in labels if l not in predicted_labels]

# Load all training data to build index
train_files = glob.glob("dataset/Train_Set/*.csv")
train_records = []
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        for idx, row in df.iterrows():
            train_records.append({
                'Subject': str(row['Subject']).strip(),
                'Object': str(row['Object']).strip(),
                'Label': lbl
            })
train_df = pd.DataFrame(train_records)

# Build sets of subjects and objects per label
label_subjects = train_df.groupby('Label')['Subject'].apply(set).to_dict()
label_objects = train_df.groupby('Label')['Object'].apply(set).to_dict()

candidates = []

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    # Check if Subject exists in a missing label's training subjects
    for ml in missing_labels:
        subjs = label_subjects.get(ml, set())
        objs = label_objects.get(ml, set())
        
        # If Subject is in this missing label's training subjects
        if s in subjs:
            candidates.append({
                'row': idx,
                'Subject': s,
                'Object': o,
                'Pred': pred,
                'Candidate_Label': ml,
                'Reason': f"Subject matches train subjects of '{ml}'"
            })
        
        # If Object is in this missing label's training objects
        elif o in objs and o not in ['0', '1', '2', '3', '4', '5', '6', '10.0', '1997', '2014', '2015', '2016', '2017', '2018']:
            if ml in ['unemployment rate', 'nominal GDP per capita', 'male population'] and len(o) < 15:
                pass
            else:
                candidates.append({
                    'row': idx,
                    'Subject': s,
                    'Object': o,
                    'Pred': pred,
                    'Candidate_Label': ml,
                    'Reason': f"Object matches train objects of '{ml}'"
                })

# Write to file
with open("scratch/missing_label_candidates.txt", "w", encoding="utf-8") as f:
    f.write(f"Total candidates found: {len(candidates)}\n")
    df_cand = pd.DataFrame(candidates)
    if len(df_cand) > 0:
        for ml in missing_labels:
            subset = df_cand[df_cand['Candidate_Label'] == ml]
            if len(subset) > 0:
                f.write(f"\n--- Candidates for '{ml}' ---\n")
                for idx, r in subset.iterrows():
                    f.write(f"Row {r['row']}: (Sub: {r['Subject']}, Obj: {r['Object']}) | Pred: {r['Pred']} | Reason: {r['Reason']}\n")

print("Saved candidates to scratch/missing_label_candidates.txt")
