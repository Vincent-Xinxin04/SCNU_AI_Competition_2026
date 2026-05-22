import csv
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load train pairs for winner and gc
winner_train = set()
with open(os.path.join(train_dir, 'winner.csv'), 'r', encoding='utf-8-sig') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) >= 2:
            winner_train.add((row[0].strip(), row[1].strip()))

gc_train = set()
with open(os.path.join(train_dir, 'general classification of race participants.csv'), 'r', encoding='utf-8-sig') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) >= 2:
            gc_train.add((row[0].strip(), row[1].strip()))

print(f"Winner train triplets: {len(winner_train)}")
print(f"GC train triplets: {len(gc_train)}")

# Let's inspect all test rows containing "Volta a Portugal", "Giro del Piemonte", "Grand Prix d'Isbergues", "Volta ao Alentejo"
target_races = ["volta a portugal", "giro del piemonte", "grand prix d'isbergues", "volta ao alentejo"]
with open(test_file, 'r', encoding='utf-8-sig') as f_test, open(sub_file, 'r', encoding='utf-8-sig') as f_sub:
    r_test = csv.DictReader(f_test)
    r_sub = csv.DictReader(f_sub)
    for row_t, row_s in zip(r_test, r_sub):
        sub = row_t['Subject'].strip()
        obj = row_t['Object'].strip()
        l_sub = row_s['Label'].strip()
        if any(tr in sub.lower() for tr in target_races):
            print(f"Test: ({sub} -> {obj}) | Current Pred: {l_sub}")
