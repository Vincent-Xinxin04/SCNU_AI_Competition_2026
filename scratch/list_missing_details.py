import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv'

train_labels = set(f[:-4] for f in os.listdir(train_dir) if f.endswith('.csv'))

test_preds = set()
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        test_preds.add(row['Label'])

missing_labels = sorted(list(train_labels - test_preds))

print(f"Total missing labels: {len(missing_labels)}")
for l in missing_labels:
    file_path = os.path.join(train_dir, f"{l}.csv")
    samples = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i < 3:
                samples.append(f"{row[0]} -> {row[1]}")
    print(f"Label: '{l}' (samples: {len(samples)}): {', '.join(samples)}")
