import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

obj_total_counts = defaultdict(int)
obj_label_counts = defaultdict(lambda: defaultdict(int))
few_shot_labels = set()

for file in os.listdir(train_dir):
    if file.endswith('.csv'):
        label = file[:-4]
        with open(os.path.join(train_dir, file), 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            count = len(lines) - 1
            if count > 0 and count <= 25:
                few_shot_labels.add(label)
                
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    o = parts[1]
                    obj_total_counts[o] += 1
                    obj_label_counts[o][label] += 1

few_shot_objects = {}
for o, counts in obj_label_counts.items():
    if not o.isdigit() and len(o) > 2:
        for label, l_count in counts.items():
            if label in few_shot_labels:
                if l_count == obj_total_counts[o]:
                    few_shot_objects[o] = label

potential_fixes = []
with open(test_path, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        s, o, l = row['Subject'], row['Object'], row['Label']
        if o in few_shot_objects:
            fs_label = few_shot_objects[o]
            if l != fs_label:
                potential_fixes.append(f"Subject: '{s}' -> Object: '{o}' | Current: '{l}' -> Few-Shot: '{fs_label}'")

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\few_shot_object_fixes.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total few-shot labels: {len(few_shot_labels)}\n")
    f.write(f"Found {len(few_shot_objects)} Objects that uniquely identify a few-shot label.\n")
    f.write(f"Found {len(potential_fixes)} potential few-shot corrections in Test Set.\n\n")
    for fix in potential_fixes:
        f.write(fix + '\n')
