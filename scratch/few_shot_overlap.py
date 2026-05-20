import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Find few-shot labels (<= 20) and their subjects/objects
few_shot_subjects = defaultdict(list)
few_shot_objects = defaultdict(list)

for file in os.listdir(train_dir):
    if file.endswith('.csv'):
        label = file[:-4]
        with open(os.path.join(train_dir, file), 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            count = len(lines) - 1
            if 0 < count <= 20: # Few-shot
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        s, o = parts[0], parts[1]
                        few_shot_subjects[s].append((label, o))
                        few_shot_objects[o].append((label, s))

# Scan test set for these entities
hits = []
with open(test_path, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        s, o, l = row['Subject'], row['Object'], row['Label']
        
        # If the subject was in a few-shot category in train
        if s in few_shot_subjects:
            for fs_label, fs_o in few_shot_subjects[s]:
                if fs_label != l:
                    hits.append(f"Subject '{s}' had few-shot label '{fs_label}' in Train. In Test it has '{l}' (Object: {o})")
                    
        # If the object was in a few-shot category in train
        if o in few_shot_objects:
            for fs_label, fs_s in few_shot_objects[o]:
                if fs_label != l:
                    hits.append(f"Object '{o}' had few-shot label '{fs_label}' in Train. In Test it has '{l}' (Subject: {s})")

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\few_shot_hits.txt', 'w', encoding='utf-8') as f:
    for hit in set(hits):
        f.write(hit + "\n")
