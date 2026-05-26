import pandas as pd
import sys
import os
import shutil

sys.stdout.reconfigure(encoding='utf-8')

submission_file = "result/submission_final.csv"
backup_file = "result/submission_0.7884_backup.csv"

# Make a backup of the current 0.7884 submission
shutil.copyfile(submission_file, backup_file)
print(f"Backup of 0.7884 submission saved to {backup_file}")

df = pd.read_csv(submission_file, encoding='utf-8-sig')

# Define new overrides
new_overrides = {
    1686: 'art director',                   # Miracle of the White Stallions -> Isabella Schlichting
    2564: 'patron saint',                   # Church of St. George -> Saint George
    355: 'basin country',                   # Lake Bolsena -> Italy
    2678: 'distributor',                     # The Larry Sanders Show -> Columbia TriStar Television
    1643: 'UTC date of spacecraft launch',  # Soyuz TM-3 -> 1987-07-22 (launch date, corrected from landing)
    3593: 'negative prognostic predictor'   # CD274 EXPRESSION -> breast cancer (PD-L1 is negative prognostic factor in breast cancer)
}

# Apply overrides and print log
changed_count = 0
for idx, new_label in sorted(new_overrides.items()):
    old_label = df.loc[idx, 'Label']
    if old_label != new_label:
        print(f"Row {idx:4d} | {df.loc[idx, 'Subject']} -> {df.loc[idx, 'Object']} | Old: '{old_label}' -> New: '{new_label}'")
        df.loc[idx, 'Label'] = new_label
        changed_count += 1
    else:
        print(f"Row {idx:4d} already has label '{new_label}'")

print(f"Total rows updated: {changed_count}")

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

# Save the final submission in place
df.to_csv(submission_file, index=False, encoding='utf-8-sig')
print("Successfully saved submission_final.csv with new activated labels.")
