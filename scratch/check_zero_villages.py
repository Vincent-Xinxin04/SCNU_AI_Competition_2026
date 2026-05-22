import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load predictions
preds = []
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        preds.append(row)

villages = ['Kunduakori', 'Bandagal', 'Faridpur', 'Mautara', 'Arazi Brahmandihi', 'Chak Modpur']

for v in villages:
    print(f"\nChecking village: {v}")
    # In test
    with open(test_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            s, o = row['Subject'].strip(), row['Object'].strip()
            if s == v:
                # Find its prediction
                pred_row = preds[idx]
                print(f"  Test Row {idx}: {s} -> {o} | Pred: {pred_row['Label']}")
