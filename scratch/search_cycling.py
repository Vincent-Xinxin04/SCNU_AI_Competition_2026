import os
import csv
import sys

# Reconfigure stdout to utf-8 if possible
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
                if any('portugal' in cell.lower() or 'gonçalves' in cell.lower() or 'goncalves' in cell.lower() for cell in row):
                    results.append((filename, row))

print(f"Found {len(results)} matches:")
for r in results:
    try:
        print(f"{r[0]}: {r[1]}")
    except Exception as e:
        print(f"Error printing: {r[0]}: {repr(r[1])}")
