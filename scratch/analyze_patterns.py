import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv'

# Load test labels to find missing ones
train_labels = set(f[:-4] for f in os.listdir(train_dir) if f.endswith('.csv'))
test_preds = set()
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        test_preds.add(row['Label'].strip())
missing_labels = sorted(list(train_labels - test_preds))

print(f"Total missing labels: {len(missing_labels)}")

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\missing_patterns.txt', 'w', encoding='utf-8') as out:
    for ml in missing_labels:
        file_path = os.path.join(train_dir, f"{ml}.csv")
        samples = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    samples.append((row[0].strip(), row[1].strip()))
        
        # Analyze subjects and objects
        subjs = [s[0] for s in samples]
        objs = [s[1] for s in samples]
        
        out.write(f"Label: '{ml}' (samples: {len(samples)})\n")
        out.write(f"  Sample Subjects: {subjs[:5]}\n")
        out.write(f"  Sample Objects:  {objs[:5]}\n")
        
        # Check if objects are numeric
        is_numeric = True
        for o in objs:
            try:
                float(o)
            except ValueError:
                is_numeric = False
                break
        
        # Check if objects are dates (YYYY-MM-DD)
        is_date = True
        for o in objs:
            if not (len(o) == 10 and o[4] == '-' and o[7] == '-'):
                is_date = False
                break
                
        out.write(f"  Is Numeric: {is_numeric} | Is Date: {is_date}\n\n")
