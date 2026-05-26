import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load labels.txt
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# 2. Load the current submission file
sub_path = "result/submission_final.csv"
sub_df = pd.read_csv(sub_path)

# New corrections dictionary
corrections_v2 = {
    72: 'general classification of race participants',
    546: 'participating team',
    645: 'general classification of race participants',
    795: 'general classification of race participants',
    859: 'general classification of race participants',
    885: 'general classification of race participants',
    941: 'general classification of race participants',
    963: 'points classification',
    1943: 'general classification of race participants',
    2170: 'general classification of race participants',
    2412: 'competition class',
    2878: 'general classification of race participants',
    3059: 'winner',
    3158: 'general classification of race participants',
    3215: 'general classification of race participants'
}

# Apply corrections and print changes
changed_count = 0
for r, new_lbl in sorted(corrections_v2.items()):
    old_lbl = sub_df.loc[r, 'Label']
    if old_lbl != new_lbl:
        print(f"Row {r} ({sub_df.loc[r, 'Subject']} -> {sub_df.loc[r, 'Object']}): '{old_lbl}' -> '{new_lbl}'")
        sub_df.loc[r, 'Label'] = new_lbl
        changed_count += 1
    else:
        print(f"Row {r} already set to '{new_lbl}'")

print(f"Total rows updated in v2: {changed_count}")

# Verify all predicted labels are valid
predicted_labels = set(sub_df['Label'].unique())
invalid_preds = [l for l in predicted_labels if l not in labels]
print(f"Invalid predicted labels: {invalid_preds}")
assert len(invalid_preds) == 0, "Error: Invalid labels detected in predictions!"

# Save back to result/submission_final.csv
sub_df.to_csv("result/submission_final.csv", index=False)
print("Saved updated result/submission_final.csv successfully!")
