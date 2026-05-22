import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'

# Find in train
train_matches = []
for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        label = filename[:-4]
        with open(os.path.join(train_dir, filename), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    s, o = row[0].strip(), row[1].strip()
                    if 'clásico' in s.lower() or 'clasico' in s.lower():
                        train_matches.append((s, o, label))

print("Train matches:")
for s, o, l in train_matches:
    print(f"  Sub: {repr(s)} | Obj: {repr(o)} | Label: {l}")

# Find in test
test_matches = []
with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s, o = row['Subject'].strip(), row['Object'].strip()
        if 'clásico' in s.lower() or 'clasico' in s.lower():
            test_matches.append((s, o))

print("\nTest matches:")
for s, o in test_matches:
    print(f"  Sub: {repr(s)} | Obj: {repr(o)}")
