import csv, os, glob

# Paths
train_dir = r'f:\\github\\SCNU_AI_Competition_2026\\dataset\\Train_Set'
submission_file = r'f:\\github\\SCNU_AI_Competition_2026\\result\\submission_final.csv'
output_file = r'f:\\github\\SCNU_AI_Competition_2026\\scratch\\candidate_overrides.txt'

# Load current submission mapping (Subject, Object) -> label
submission_map = {}
with open(submission_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['Subject'].strip(), row['Object'].strip())
        submission_map[key] = row['Label'].strip()

# Scan training set and collect possible overrides
candidate_overrides = []
for csv_path in glob.glob(os.path.join(train_dir, '*.csv')):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Subject'].strip(), row['Object'].strip())
            train_label = row['Label'].strip()
            if key in submission_map:
                sub_label = submission_map[key]
                if train_label != sub_label:
                    candidate_overrides.append((key[0], key[1], train_label))

# Write candidates to file
with open(output_file, 'w', encoding='utf-8-sig') as f:
    for subj, obj, new_label in candidate_overrides:
        f.write(f"('{subj}', '{obj}') -> {new_label}\n")

print(f"Found {len(candidate_overrides)} candidate overrides.")
print(f"Wrote to {output_file}")
