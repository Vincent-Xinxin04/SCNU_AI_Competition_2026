import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

results = []
for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(train_dir, filename)
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if any('municipal unit of albania' in cell.lower() or 'municipality of albania' in cell.lower() for cell in row):
                    results.append((filename[:-4], row))

print(f"Found {len(results)} matches for Albania classes in train:")
for r in results[:30]:
    print(r)
