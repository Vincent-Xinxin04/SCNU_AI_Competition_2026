import os, csv

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load test labels
train_labels = set(f[:-4] for f in os.listdir(train_dir) if f.endswith('.csv'))
test_preds = set()
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        test_preds.add(row['Label'].strip())
remaining_missing = sorted(list(train_labels - test_preds))

print(f"Remaining missing labels count: {len(remaining_missing)}")

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\missing_samples.txt', 'w', encoding='utf-8') as out:
    for ml in remaining_missing:
        file_path = os.path.join(train_dir, f"{ml}.csv")
        samples = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i < 3:
                    samples.append(f"({row[0]}, {row[1]})")
                else:
                    break
        out.write(f"{ml:50s} : {', '.join(samples)}\n")

print("Saved samples to missing_samples.txt")
