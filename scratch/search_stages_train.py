import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

print("Searching train set for 'Stage' in Subject:")
found = 0
for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        label = filename[:-4]
        filepath = os.path.join(train_dir, filename)
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    sub = row[0].strip()
                    obj = row[1].strip()
                    if 'stage' in sub.lower():
                        found += 1
                        print(f"[{label}] {sub} -> {obj}")

print(f"Total 'Stage' matches in Train: {found}")
