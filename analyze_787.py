import os
import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.787.csv'

train_pairs = {}
for f_name in os.listdir(train_dir):
    if f_name.endswith('.csv'):
        label = f_name[:-4]
        with open(os.path.join(train_dir, f_name), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    subj, obj = row[0].strip(), row[1].strip()
                    if (subj, obj) not in train_pairs:
                        train_pairs[(subj, obj)] = set()
                    train_pairs[(subj, obj)].add(label)

print(f"Total unique (Subject, Object) pairs in training: {len(train_pairs)}")

discrepancies = []
exact_matches = []
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        subj, obj = row['Subject'].strip(), row['Object'].strip()
        pred_label = row['Label'].strip()
        if (subj, obj) in train_pairs:
            train_labels = train_pairs[(subj, obj)]
            if pred_label not in train_labels:
                discrepancies.append((idx, subj, obj, pred_label, list(train_labels)))
            else:
                exact_matches.append((idx, subj, obj, pred_label))

print(f"\nExact matches (prediction matches training): {len(exact_matches)}")
print(f"Discrepancies (prediction does NOT match training): {len(discrepancies)}")

print("\n=== DISCREPANCIES (These need to be fixed) ===")
for idx, subj, obj, pred, train_lbls in discrepancies[:50]:
    print(f"Row {idx:4d} | ({subj[:40]} -> {obj[:40]}) | Pred: '{pred}' | Train: {train_lbls}")

if len(discrepancies) > 50:
    print(f"\n... and {len(discrepancies) - 50} more discrepancies")
