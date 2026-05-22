import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

filepath = os.path.join(train_dir, 'interested in.csv')
if os.path.exists(filepath):
    print("--- interested in.csv ---")
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, row in enumerate(reader):
            print(row)
else:
    print("interested in.csv does not exist")
