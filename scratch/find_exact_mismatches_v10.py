import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# 1. Load all training set triplets
# Mapping: (norm(Subject), norm(Object)) -> set of labels
train_triplets = {}

def norm(s):
    return ''.join(c for c in s.lower() if c.isalnum())

for f in os.listdir(train_dir):
    if f.endswith('.csv'):
        label = f[:-4]
        filepath = os.path.join(train_dir, f)
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as file:
                reader = csv.reader(file)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        s = row[0].strip()
                        o = row[1].strip()
                        key = (norm(s), norm(o))
                        if key not in train_triplets:
                            train_triplets[key] = set()
                        train_triplets[key].add(label)
        except Exception as e:
            print(f"Error reading {f}: {e}")

# 2. Check submission_final.csv
mismatches = []
exact_matches = 0

with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        s = row['Subject'].strip()
        o = row['Object'].strip()
        l = row['Label'].strip()
        key = (norm(s), norm(o))
        
        if key in train_triplets:
            exact_matches += 1
            labels_in_train = train_triplets[key]
            if l not in labels_in_train:
                mismatches.append({
                    'row_idx': idx + 2, # 1-indexed, +1 for header
                    'subject': s,
                    'object': o,
                    'pred_label': l,
                    'train_labels': list(labels_in_train)
                })

print(f"Total exact triplet matches between train and test: {exact_matches}")
print(f"Found {len(mismatches)} mismatches where prediction doesn't match training label:")
for m in mismatches:
    print(f"Row {m['row_idx']}: ({m['subject']} -> {m['object']}) | Pred: '{m['pred_label']}' | Train: {m['train_labels']}")
