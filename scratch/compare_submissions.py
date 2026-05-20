import csv
from collections import Counter

file_before = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv'
file_after = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

counts_before = Counter()
with open(file_before, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        counts_before[row['Label']] += 1

counts_after = Counter()
with open(file_after, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        counts_after[row['Label']] += 1

print("Label Changes (Before -> After):")
all_labels = sorted(list(set(counts_before.keys()) | set(counts_after.keys())))
for lbl in all_labels:
    cb = counts_before.get(lbl, 0)
    ca = counts_after.get(lbl, 0)
    if cb != ca:
        print(f"  {lbl:45s}: {cb:3d} -> {ca:3d} (Diff: {ca-cb:+d})")

print(f"\nTotal active labels before: {len([k for k, v in counts_before.items() if v > 0])}")
print(f"Total active labels after:  {len([k for k, v in counts_after.items() if v > 0])}")
