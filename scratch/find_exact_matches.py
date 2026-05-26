import os
import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

train_dir = 'dataset/Train_Set'
sub_file = 'result/submission_final.csv'

# Load train pairs
train_pairs = {}
for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        label = filename[:-4]
        with open(os.path.join(train_dir, filename), 'r', encoding='utf-8-sig') as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2:
                    s, o = row[0].strip(), row[1].strip()
                    if (s, o) not in train_pairs:
                        train_pairs[(s, o)] = set()
                    train_pairs[(s, o)].add(label)

# Compare with submission
matches = []
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        s, o = row['Subject'].strip(), row['Object'].strip()
        pred = row['Label'].strip()
        if (s, o) in train_pairs:
            train_labels = train_pairs[(s, o)]
            matches.append((idx, s, o, pred, list(train_labels)))

print(f"Total exact matches found: {len(matches)}")
for idx, s, o, pred, train_lbls in matches:
    status = "MATCH" if pred in train_lbls else "DISCREPANCY"
    print(f"Row {idx:4d} | {s} -> {o} | Pred: \"{pred}\" | Train: {train_lbls} | {status}")
