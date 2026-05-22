import csv, os, glob

# Paths
train_dir = r'f:\\github\\SCNU_AI_Competition_2026\\dataset\\Train_Set'
submission_file = r'f:\\github\\SCNU_AI_Competition_2026\\result\\submission_final.csv'
output_candidates = r'f:\\github\\SCNU_AI_Competition_2026\\scratch\\candidate_overrides_v2.txt'

# Load current submission mapping (Subject, Object) -> label
submission_map = {}
with open(submission_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['Subject'].strip(), row['Object'].strip())
        submission_map[key] = row['Label'].strip()

candidates = []
# Each training CSV file name is the label name
for csv_path in glob.glob(os.path.join(train_dir, '*.csv')):
    label = os.path.splitext(os.path.basename(csv_path))[0]
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Subject'].strip(), row['Object'].strip())
            if key in submission_map:
                current_label = submission_map[key]
                if current_label != label:
                    candidates.append((key[0], key[1], label))

# Write candidates
with open(output_candidates, 'w', encoding='utf-8-sig') as f:
    for subj, obj, new_label in candidates:
        f.write(f"('{subj}', '{obj}') -> {new_label}\n")

print(f"Found {len(candidates)} candidate overrides.")
print(f"Wrote to {output_candidates}")
