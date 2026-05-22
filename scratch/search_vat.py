import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Let's load the list of countries from training country.csv
import os
train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
countries = set()
with open(os.path.join(train_dir, 'country.csv'), 'r', encoding='utf-8-sig') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) >= 2:
            countries.add(row[1].strip().lower())
            countries.add(row[0].strip().lower())

print(f"Loaded {len(countries)} country names")

with open(test_file, 'r', encoding='utf-8-sig') as f_test, open(sub_file, 'r', encoding='utf-8-sig') as f_sub:
    r_test = csv.DictReader(f_test)
    r_sub = csv.DictReader(f_sub)
    for row_t, row_s in zip(r_test, r_sub):
        sub = row_t['Subject'].strip()
        obj = row_t['Object'].strip()
        l_sub = row_s['Label'].strip()
        if sub.lower() in countries:
            try:
                val = float(obj)
                if 5.0 <= val <= 30.0:
                    print(f"Test: ({sub} -> {obj}) | Current Pred: {l_sub}")
            except ValueError:
                pass
