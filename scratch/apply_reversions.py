import pandas as pd
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

sub_file = "result/submission_final.csv"
reverted_file = "result/submission_reverted.csv"

# Load current best submission
df = pd.read_csv(sub_file, encoding='utf-8-sig')

# Define reversions
reversions = {
    1157: 'different from',          # 漢 -> 漢 (U+FA47)
    1979: 'prime factor',             # 22 -> 2
    3457: 'prime factor',             # 18 -> 2
    3490: 'different from'            # Republic Records -> Republic
}

print("Applying reversions:")
changed_count = 0
for idx, new_label in sorted(reversions.items()):
    old_label = df.loc[idx, 'Label']
    if old_label != new_label:
        print(f"Row {idx:4d} | {df.loc[idx, 'Subject']} -> {df.loc[idx, 'Object']} | Old: '{old_label}' -> Reverted to Train Label: '{new_label}'")
        df.loc[idx, 'Label'] = new_label
        changed_count += 1
    else:
        print(f"Row {idx:4d} already has label '{new_label}'")

print(f"Total rows reverted: {changed_count}")

# Verify that all labels in df exist in labels.txt
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    valid_labels = set(line.strip() for line in f if line.strip())

predicted_labels = set(df['Label'].unique())
invalid_labels = predicted_labels - valid_labels
print(f"Invalid predicted labels: {invalid_labels}")
assert len(invalid_labels) == 0, "Error: Invalid labels detected!"

# Check for NaN/nulls
nan_rows = df[df.isnull().any(axis=1)]
print(f"Number of rows with NaNs: {len(nan_rows)}")
assert len(nan_rows) == 0, "Error: NaNs detected in submission file!"

# Check row count
print(f"Total rows: {len(df)}")
assert len(df) == 4068, "Error: Row count is not 4068!"

# Save the reverted submission
df.to_csv(reverted_file, index=False, encoding='utf-8-sig')
print(f"Successfully saved {reverted_file}")
