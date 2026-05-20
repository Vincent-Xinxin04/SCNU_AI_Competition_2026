import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv'

# Load all train pairs with their labels
train_pairs = {}
for f_name in os.listdir(train_dir):
    if f_name.endswith('.csv'):
        label = f_name[:-4]
        with open(os.path.join(train_dir, f_name), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    subj, obj = row[0].strip(), row[1].strip()
                    # If already exists, we might have multiple labels for same pair in train
                    if (subj, obj) not in train_pairs:
                        train_pairs[(subj, obj)] = set()
                    train_pairs[(subj, obj)].add(label)

# Load test data and find discrepancies
discrepancies = []
with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        subj, obj = row['Subject'].strip(), row['Object'].strip()
        pred_label = row['Label'].strip()
        if (subj, obj) in train_pairs:
            train_labels = train_pairs[(subj, obj)]
            if pred_label not in train_labels:
                discrepancies.append((subj, obj, pred_label, list(train_labels)))

print(f"Total discrepancies found: {len(discrepancies)}")
# Write them to a file
with open(r'f:\github\SCNU_AI_Competition_2026\scratch\discrepancies.txt', 'w', encoding='utf-8') as out:
    for subj, obj, pred, train_lbls in discrepancies:
        out.write(f"Pair: ({subj} -> {obj}) | Pred: {pred} | Train: {train_lbls}\n")
