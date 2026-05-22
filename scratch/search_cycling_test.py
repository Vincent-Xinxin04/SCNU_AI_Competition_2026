import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

cycling_keywords = ["tour", "giro", "vuelta", "volta", "grand prix", "classic", "prix", "championships"]

results = []
with open(test_file, 'r', encoding='utf-8-sig') as f_test, open(sub_file, 'r', encoding='utf-8-sig') as f_sub:
    r_test = csv.DictReader(f_test)
    r_sub = csv.DictReader(f_sub)
    for row_t, row_s in zip(r_test, r_sub):
        sub = row_t['Subject'].strip()
        obj = row_t['Object'].strip()
        l_sub = row_s['Label'].strip()
        if any(kw in sub.lower() for kw in cycling_keywords):
            results.append((sub, obj, l_sub))

print(f"Found {len(results)} cycling-related rows in test:")
for sub, obj, l_sub in results:
    print(f"  {sub} -> {obj} | Pred: {l_sub}")
