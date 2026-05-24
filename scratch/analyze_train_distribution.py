import os
import glob
import pandas as pd
import json

train_files = glob.glob("dataset/Train_Set/*.csv")
label_counts = {}
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        # count unique pairs
        df_unique = df.drop_duplicates(subset=['Subject', 'Object'])
        label_counts[lbl] = len(df_unique)

# Sort by count ascending
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])

output_data = {
    "total_labels": len(sorted_labels),
    "distribution": sorted_labels
}

with open("scratch/train_distribution.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("Saved training distribution to scratch/train_distribution.json")
