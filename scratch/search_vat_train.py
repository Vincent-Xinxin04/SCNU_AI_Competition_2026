import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
filepath = os.path.join(train_dir, 'VAT-rate.csv')
if os.path.exists(filepath):
    print("--- VAT-rate.csv ---")
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            print(row)
else:
    print("VAT-rate.csv does not exist")
