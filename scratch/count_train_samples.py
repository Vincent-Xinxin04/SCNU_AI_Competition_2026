import os
import csv

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
for label in ['positive prognostic predictor', 'negative prognostic predictor']:
    filepath = os.path.join(train_dir, f"{label}.csv")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            count = sum(1 for _ in reader)
            print(f"{label}: count={count}")
    else:
        print(f"{label} does not exist")
