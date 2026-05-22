import csv
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'
train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

# Load predictions
preds = []
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        preds.append(row['Label'].strip())

# Load train set to see what labels are associated with object '0' or '0.0'
zero_train = {}
for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        label = filename[:-4]
        with open(os.path.join(train_dir, filename), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    s, o = row[0].strip(), row[1].strip()
                    if o in ['0', '0.0']:
                        if s not in zero_train:
                            zero_train[s] = set()
                        zero_train[s].add(label)

print("Rows in Test set where Object is '0' or '0.0':")
with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        s, o = row['Subject'].strip(), row['Object'].strip()
        if o in ['0', '0.0']:
            pred = preds[idx]
            train_labels = zero_train.get(s, set())
            print(f"  Row {idx}: {s} -> {o} | Pred: {pred} | Train Labels: {list(train_labels)}")
