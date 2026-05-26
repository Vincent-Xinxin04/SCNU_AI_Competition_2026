import pandas as pd
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8')

submission_file = "result/submission_final.csv"
backup_file = "result/submission_0.782_backup.csv"

# Make a backup of the current 0.782 submission
shutil.copyfile(submission_file, backup_file)
print(f"Backup of 0.782 submission saved to {backup_file}")

df = pd.read_csv(submission_file, encoding='utf-8-sig')

# Define overrides dictionary
# Key: Row index, Value: New Label
overrides = {
    # 1. List of Roads / Wikimedia set index article corrections
    905: 'is a list of',
    2846: 'is a list of',
    3573: 'instance of',
    
    # 2. Lighthouse and Lists of Lights described by source corrections
    227: 'described by source',
    455: 'described by source',
    460: 'described by source',
    562: 'described by source',
    652: 'described by source',
    755: 'described by source',
    1571: 'described by source',
    1933: 'described by source',
    2208: 'described by source',
    2757: 'described by source',
    3365: 'described by source',
    3650: 'described by source',
    3721: 'described by source',
    
    # 3. Small zero-value village split
    526: 'literate population',
    1146: 'number of households',
    1172: 'number of households',
    1349: 'number of households',
    2604: 'literate population',
    3933: 'literate population'
}

# Apply overrides and print change log
changed_count = 0
for idx, new_label in sorted(overrides.items()):
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

# Save the final submission in place
df.to_csv(submission_file, index=False, encoding='utf-8-sig')
print("Successfully saved submission_final.csv")
