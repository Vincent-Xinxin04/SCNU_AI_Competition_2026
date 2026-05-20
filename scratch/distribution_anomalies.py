import os, csv
from collections import Counter

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.742.csv'

train_counts = {}
total_train = 0
for file in os.listdir(train_dir):
    if file.endswith('.csv'):
        label = file[:-4]
        with open(os.path.join(train_dir, file), 'r', encoding='utf-8-sig') as f:
            count = sum(1 for _ in f) - 1
            train_counts[label] = count
            total_train += count

test_counts = Counter()
with open(test_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_counts[row['Label']] += 1
total_test = sum(test_counts.values())

print('Labels massively OVERPREDICTED in test set (Test% / Train% > 3.0 or Train is 0):')
for label, t_count in test_counts.items():
    t_pct = t_count / total_test * 100
    tr_count = train_counts.get(label, 0)
    tr_pct = tr_count / total_train * 100 if total_train else 0
    
    ratio = t_pct / tr_pct if tr_pct else float('inf')
    if ratio > 3.0:
        if t_count > 2: # only care if we predicted it multiple times
            print(f"{label:<35} | Test: {t_count} ({t_pct:.3f}%) | Train: {tr_count} ({tr_pct:.3f}%) | Ratio: {ratio:.1f}")

print('\nLabels massively UNDERPREDICTED in test set (Train% / Test% > 3.0):')
for label, tr_count in train_counts.items():
    tr_pct = tr_count / total_train * 100
    if tr_pct < 0.1: continue # ignore rare labels in train set
    t_count = test_counts.get(label, 0)
    t_pct = t_count / total_test * 100
    
    ratio = tr_pct / t_pct if t_pct else float('inf')
    if ratio > 3.0:
        print(f"{label:<35} | Train: {tr_count} ({tr_pct:.3f}%) | Test: {t_count} ({t_pct:.3f}%) | Ratio: {ratio:.1f}")
