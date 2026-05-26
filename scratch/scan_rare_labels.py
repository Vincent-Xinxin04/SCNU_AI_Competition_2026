import os
import glob
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load all labels and count their training samples
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

train_dir = "dataset/Train_Set"
train_counts = {}
label_train_data = {}

for l in labels:
    filename = l.replace('/', '_').replace(':', '_') + '.csv'
    filepath = os.path.join(train_dir, filename)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df.columns = [str(c).strip() for c in df.columns]
            if 'Subject' in df.columns and 'Object' in df.columns:
                df = df[['Subject', 'Object']].dropna()
                train_counts[l] = len(df)
                label_train_data[l] = df
            else:
                train_counts[l] = 0
        except Exception:
            train_counts[l] = 0
    else:
        train_counts[l] = 0

# Rare labels have <= 15 training samples
rare_labels = [l for l, count in train_counts.items() if 1 <= count <= 15]
print(f"Number of rare labels (1 <= count <= 15): {len(rare_labels)}")

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_final.csv")

matches = []

for rl in rare_labels:
    df_rare = label_train_data[rl]
    train_pairs = set(zip(df_rare['Subject'].astype(str).str.strip(), df_rare['Object'].astype(str).str.strip()))
    train_subjs = set(df_rare['Subject'].astype(str).str.strip())
    train_objs = set(df_rare['Object'].astype(str).str.strip())
    
    # Remove generic/common objects from obj matching to prevent false positives
    train_objs_filtered = {o for o in train_objs if len(o) > 2 and o not in [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10.0', '1997', '2014', '2015', '2016', '2017', '2018', '2019', '2020',
        'USA', 'United States of America', 'UK', 'United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Japan',
        'yes', 'no', 'male', 'female'
    ]}

    for idx, row in test_df.iterrows():
        s = str(row['Subject']).strip()
        o = str(row['Object']).strip()
        pred = sub_df.loc[idx, 'Label']
        
        # 1. Exact pair match
        if (s, o) in train_pairs:
            matches.append({
                'row': idx,
                'Subject': row['Subject'],
                'Object': row['Object'],
                'Current_Pred': pred,
                'Rare_Label': rl,
                'Match_Type': 'Exact Pair Match',
                'Train_Count': train_counts[rl]
            })
        # 2. Cross match (both subject and object exist in train but not paired)
        elif s in train_subjs and o in train_objs_filtered:
            matches.append({
                'row': idx,
                'Subject': row['Subject'],
                'Object': row['Object'],
                'Current_Pred': pred,
                'Rare_Label': rl,
                'Match_Type': 'Cross-row Match',
                'Train_Count': train_counts[rl]
            })
        # 3. Subject matches a rare label's subject in train
        elif s in train_subjs:
            # We only keep this if the current prediction is generic/weak
            if pred in ['instance of', 'subclass of', 'part of', 'located in the administrative territorial entity', 'country', 'location', 'use']:
                matches.append({
                    'row': idx,
                    'Subject': row['Subject'],
                    'Object': row['Object'],
                    'Current_Pred': pred,
                    'Rare_Label': rl,
                    'Match_Type': 'Subject-only Match',
                    'Train_Count': train_counts[rl]
                })

df_matches = pd.DataFrame(matches)
if len(df_matches) > 0:
    print(f"Total rare label candidate matches found: {len(df_matches)}")
    df_matches.to_csv("scratch/rare_label_candidates.csv", index=False)
    # Print them by Rare_Label
    for rl in sorted(df_matches['Rare_Label'].unique()):
        subset = df_matches[df_matches['Rare_Label'] == rl]
        print(f"\n--- Rare Label '{rl}' (Train Count: {train_counts[rl]}, Matches: {len(subset)}) ---")
        for _, r in subset.iterrows():
            print(f"Row {r['row']:4d} | ({r['Subject']} -> {r['Object']}) | Current: {r['Current_Pred']} | Match: {r['Match_Type']}")
else:
    print("No rare label matches found.")
