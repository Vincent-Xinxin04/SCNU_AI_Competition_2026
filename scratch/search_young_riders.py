import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(train_dir, filename)
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if 'valverde' in row[1].lower() or 'boonen' in row[1].lower():
                    print(f"{filename[:-4]}: {row}")
