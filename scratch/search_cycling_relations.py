import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
labels = [
    'winner', 'general classification of race participants', 
    'mountains classification', 'young rider classification', 'points classification'
]

for label in labels:
    filepath = os.path.join(train_dir, f"{label}.csv")
    if os.path.exists(filepath):
        print(f"\n--- {label}.csv (first 10 rows) ---")
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for i, row in enumerate(reader):
                if i >= 10:
                    break
                print(row)
