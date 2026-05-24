import os
import glob
import pandas as pd
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load labels.txt
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# 2. Load submission
sub_df = pd.read_csv("result/submission_0.7791.csv")
test_df = pd.read_csv("dataset/test.csv")

predicted_labels = set(sub_df['Label'].unique())
missing_labels = [l for l in labels if l not in predicted_labels]

print(f"Number of missing labels: {len(missing_labels)}")

# Profile each missing label
missing_profiles = {}
for lbl in missing_labels:
    csv_path = os.path.join("dataset/Train_Set", f"{lbl}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]
        if 'Subject' in df.columns and 'Object' in df.columns:
            df = df[['Subject', 'Object']].dropna()
            missing_profiles[lbl] = df

# Let's inspect the train data of each missing label
print("\n--- Inspecting training data of missing labels ---")
for lbl, df in missing_profiles.items():
    print(f"\nLabel: '{lbl}' (Train count: {len(df)})")
    print(df.head(5))
