import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load predictions
preds = {}
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        preds[(row['Subject'].strip(), row['Object'].strip())] = row['Label'].strip()

print("All stations in Test set:")
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        s, o = row['Subject'].strip(), row['Object'].strip()
        if 'station' in s.lower() or 'bahnhof' in s.lower():
            pred = preds.get((s, o), 'NONE')
            print(f"  {s} -> {o} | Pred: {pred}")
