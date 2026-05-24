import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load labels
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# Load submission
sub = pd.read_csv("result/submission_0.7791.csv")
pred_labels = set(sub['Label'].unique())

missing = [l for l in labels if l not in pred_labels]
print(f"Total missing: {len(missing)}")
for idx, m in enumerate(sorted(missing)):
    print(f"{idx+1}: {m}")
