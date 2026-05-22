import os
import csv
import sys
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Train counts
train_counts = {}
for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        label = filename[:-4]
        with open(os.path.join(train_dir, filename), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            count = sum(1 for row in reader if len(row) >= 2)
            train_counts[label] = count

# Sub counts
sub_counts = Counter()
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sub_counts[row['Label'].strip()] += 1

print(f"{'Label':<60} | {'Train':<6} | {'Submission':<10}")
print("-" * 82)
# Sort by train count ascending
for label in sorted(train_counts.keys(), key=lambda x: train_counts[x]):
    print(f"{label:<60} | {train_counts[label]:<6} | {sub_counts.get(label, 0):<10}")
