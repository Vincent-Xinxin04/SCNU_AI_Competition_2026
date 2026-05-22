import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

subjects = [
    'Verrazzano-Narrows Bridge', 'Japan National Route 499', 'Throgs Neck Bridge',
    'Oregon City Bridge', 'Yaquina Bay Bridge', 'Big Creek Bridge',
    'Japan National Route 436', 'Japan National Route 57', 'Japan National Route 197',
    'Hardwick Hall', 'Kingston Lacy', 'Castle Drogo', 'Japan National Route 317',
    'Golden Gate Bridge', 'George Washington Bridge', 'Japan National Route 350', 'Cotter Bridge'
]

sub_set = {s.lower() for s in subjects}

with open(test_file, 'r', encoding='utf-8-sig') as f_test, open(sub_file, 'r', encoding='utf-8-sig') as f_sub:
    r_test = csv.DictReader(f_test)
    r_sub = csv.DictReader(f_sub)
    for row_t, row_s in zip(r_test, r_sub):
        sub = row_t['Subject'].strip()
        obj = row_t['Object'].strip()
        l_sub = row_s['Label'].strip()
        if sub.lower() in sub_set or 'national route' in sub.lower() or 'bridge' in sub.lower():
            print(f"Test: ({sub} -> {obj}) | Current Pred: {l_sub}")
