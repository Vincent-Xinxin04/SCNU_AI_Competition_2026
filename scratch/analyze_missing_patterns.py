import os
import glob
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load labels.txt
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# 2. Read current submission and test set
sub_df = pd.read_csv("result/submission_final.csv", encoding='utf-8-sig')
test_df = pd.read_csv("dataset/test.csv", encoding='utf-8-sig')

# 3. Find missing labels
pred_counts = sub_df['Label'].value_counts()
missing_labels = [l for l in labels if l not in pred_counts]

train_dir = "dataset/Train_Set"

# Write directly to text file in UTF-8
out_filepath = "scratch/analyze_missing_patterns.txt"
with open(out_filepath, "w", encoding="utf-8") as out:
    out.write(f"Number of missing labels in submission: {len(missing_labels)}\n")
    out.write("\n--- Scanning Missing Labels Train Data vs Test Set ---\n")

    for ml in missing_labels:
        filename = ml.replace('/', '_').replace(':', '_') + '.csv'
        filepath = os.path.join(train_dir, filename)
        if not os.path.exists(filepath):
            continue
        
        train_df = pd.read_csv(filepath, encoding='utf-8-sig')
        train_df.columns = [str(c).strip() for c in train_df.columns]
        
        # Extract subjects and objects from train
        train_subjects = set(train_df['Subject'].astype(str).str.strip().str.lower())
        train_objects = set(train_df['Object'].astype(str).str.strip().str.lower())
        
        # Let's see if we have matches in test set
        matches_sub = []
        matches_obj = []
        matches_both = []
        
        for idx, row in test_df.iterrows():
            s = str(row['Subject']).strip()
            o = str(row['Object']).strip()
            sl = s.lower()
            ol = o.lower()
            pred = sub_df.loc[idx, 'Label']
            
            # Check for matching subject
            sub_match = sl in train_subjects
            # Check for matching object
            obj_match = ol in train_objects and ol not in ['0', '1', '2', '3', '4', '5', '6', 'yes', 'no']
            
            if sub_match and obj_match:
                matches_both.append((idx, s, o, pred))
            elif sub_match:
                matches_sub.append((idx, s, o, pred))
            elif obj_match:
                matches_obj.append((idx, s, o, pred))
                
        # Print if we found any matches
        if len(matches_both) > 0 or len(matches_sub) > 0 or len(matches_obj) > 0:
            out.write(f"\nLabel: {ml} (Train size: {len(train_df)})\n")
            out.write("  Train samples sample (up to 5):\n")
            for i, r in train_df.head(5).iterrows():
                out.write(f"    Sub: {r['Subject']} | Obj: {r['Object']}\n")
                
            if len(matches_both) > 0:
                out.write(f"  Exact Subject & Object matches in test ({len(matches_both)}):\n")
                for idx, s, o, pred in matches_both[:10]:
                    out.write(f"    Row {idx:4d}: Sub: {s} | Obj: {o} | Current Pred: {pred}\n")
            if len(matches_sub) > 0:
                # Group by test predicted label
                pred_groups = {}
                for idx, s, o, pred in matches_sub:
                    pred_groups[pred] = pred_groups.get(pred, 0) + 1
                out.write(f"  Subject matches in test ({len(matches_sub)}): Grouped by current prediction: {pred_groups}\n")
                out.write("    Sample matches:\n")
                for idx, s, o, pred in matches_sub[:10]:
                    out.write(f"    Row {idx:4d}: Sub: {s} | Obj: {o} | Current Pred: {pred}\n")
            if len(matches_obj) > 0:
                pred_groups = {}
                for idx, s, o, pred in matches_obj:
                    pred_groups[pred] = pred_groups.get(pred, 0) + 1
                out.write(f"  Object matches in test ({len(matches_obj)}): Grouped by current prediction: {pred_groups}\n")
                out.write("    Sample matches:\n")
                for idx, s, o, pred in matches_obj[:10]:
                    out.write(f"    Row {idx:4d}: Sub: {s} | Obj: {o} | Current Pred: {pred}\n")

print("Saved script to write directly to file.")
