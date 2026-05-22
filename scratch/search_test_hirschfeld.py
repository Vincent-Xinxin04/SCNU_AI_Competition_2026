import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

with open(test_file, 'r', encoding='utf-8-sig') as f_test, open(sub_file, 'r', encoding='utf-8-sig') as f_sub:
    reader_test = csv.DictReader(f_test)
    reader_sub = csv.DictReader(f_sub)
    for idx, (row_t, row_s) in enumerate(zip(reader_test, reader_sub)):
        s_t = row_t['Subject'].strip()
        o_t = row_t['Object'].strip()
        l_s = row_s['Label'].strip()
        if 'hirschfeld' in o_t.lower() or 'hirschfeld' in s_t.lower():
            print(f"Row {idx}: {s_t} -> {o_t} | Current Pred: {l_s}")
