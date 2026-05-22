import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

for f in os.listdir(train_dir):
    if f.endswith('.csv'):
        label = f[:-4]
        filepath = os.path.join(train_dir, f)
        with open(filepath, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    s, o = row[0].strip(), row[1].strip()
                    if 'Volta a Portugal' in s or 'Volta a Portugal' in o:
                        print(f"Train Match [{label}]: {s} -> {o}")
