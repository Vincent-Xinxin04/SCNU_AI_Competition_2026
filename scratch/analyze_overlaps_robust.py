import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load test labels to find missing ones
train_labels = set(f[:-4] for f in os.listdir(train_dir) if f.endswith('.csv'))
test_preds = set()
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        test_preds.add(row['Label'].strip())
missing_labels = sorted(list(train_labels - test_preds))

print(f"Total missing labels: {len(missing_labels)}")

def normalize(val):
    val = val.strip().lower()
    # Try parsing as float
    try:
        fval = float(val)
        return f"{fval:.4f}" # Normalize float representation
    except ValueError:
        pass
    return val

# Load all train data for missing labels
missing_train_data = defaultdict(list)
for l in missing_labels:
    file_path = os.path.join(train_dir, f"{l}.csv")
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                s, o = row[0], row[1]
                missing_train_data[l].append((s, o, normalize(s), normalize(o)))

# Load test data
test_data = []
with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_data.append((row['Subject'], row['Object'], row['Label'], normalize(row['Subject']), normalize(row['Object'])))

print("Analyzing overlaps...")
overlaps_found = 0
for l, samples in missing_train_data.items():
    for ts, to, tl, norm_ts, norm_to in test_data:
        for trs, tro, norm_trs, norm_tro in samples:
            if norm_ts == norm_trs and norm_to == norm_tro:
                print(f"Match found for label '{l}':")
                print(f"  Test:  {ts} -> {to} (Pred: {tl})")
                print(f"  Train: {trs} -> {tro}")
                overlaps_found += 1

print(f"Total robust overlaps found: {overlaps_found}")
