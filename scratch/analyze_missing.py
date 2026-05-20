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

print(f"There are {len(missing_labels)} missing labels.")

# Gather subject/object characteristics for missing labels
label_profiles = {}
for l in missing_labels:
    file_path = os.path.join(train_dir, f"{l}.csv")
    s_types = defaultdict(int)
    o_types = defaultdict(int)
    total = 0
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            s, o = row['Subject'], row['Object']
            
            # Simple heuristic type
            if o.replace('.','',1).isdigit(): o_types['NUMBER'] += 1
            elif len(o) > 0 and o[0].isupper(): o_types['PROPER_NOUN'] += 1
            
            if s.replace('.','',1).isdigit(): s_types['NUMBER'] += 1
            elif len(s) > 0 and s[0].isupper(): s_types['PROPER_NOUN'] += 1
            
    label_profiles[l] = {'total': total, 'o_num_ratio': o_types['NUMBER'] / total if total > 0 else 0}

# Print the most interesting missing labels (where Object is frequently a number, easy to spot)
print("Numeric missing labels (where Object is usually a number):")
for l in missing_labels:
    if label_profiles[l]['o_num_ratio'] > 0.5:
        print(f"- {l} (Train samples: {label_profiles[l]['total']})")
