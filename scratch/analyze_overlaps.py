import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv'

# Read all train data for missing labels
missing_train_data = defaultdict(list)
train_labels = set(f[:-4] for f in os.listdir(train_dir) if f.endswith('.csv'))

test_preds = set()
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        test_preds.add(row['Label'])

missing_labels = sorted(list(train_labels - test_preds))

for l in missing_labels:
    file_path = os.path.join(train_dir, f"{l}.csv")
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            missing_train_data[l].append((row[0], row[1]))

# Load test data
test_data = []
with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_data.append((row['Subject'], row['Object'], row['Label']))

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\overlap_results.txt', 'w', encoding='utf-8') as out_f:
    # Find exact overlaps where Subject and Object in test match a missing label in train
    out_f.write("Exact subject-object matches:\n")
    exact_count = 0
    for l, pairs in missing_train_data.items():
        for ts, to, tl in test_data:
            for trs, tro in pairs:
                if ts == trs and to == tro:
                    out_f.write(f"Test: ({ts}, {to}, {tl}) -> Train: ({trs}, {tro}, '{l}')\n")
                    exact_count += 1

    out_f.write(f"\nExact matches found: {exact_count}\n")

    # Find cases where Subject matches a missing label's subject in train
    out_f.write("\nSubject only matches:\n")
    subj_count = 0
    for l, pairs in missing_train_data.items():
        train_subjs = set(p[0] for p in pairs)
        for ts, to, tl in test_data:
            if ts in train_subjs:
                # check if we can print
                # Only print if the prediction looks generic
                if tl in ['numeric value', 'part of', 'located in the administrative territorial entity', 'instance of', 'has part', 'country', 'location', 'use', 'subclass of']:
                    out_f.write(f"Test Subj: {ts} -> {to} (Pred: {tl}) | Train has label '{l}' for this subject\n")
                    subj_count += 1
    out_f.write(f"Subject matches found: {subj_count}\n")
