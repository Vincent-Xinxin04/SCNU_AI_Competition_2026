import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

labels_file = r'f:\github\SCNU_AI_Competition_2026\dataset\labels.txt'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load all labels
all_labels = set()
with open(labels_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            all_labels.add(line)

# Load submission labels
pred_labels = set()
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pred_labels.add(row['Label'].strip())

unpredicted = all_labels - pred_labels
print(f"Total labels: {len(all_labels)}")
print(f"Predicted labels: {len(pred_labels)}")
print(f"Unpredicted labels ({len(unpredicted)}):")
for label in sorted(unpredicted):
    print(f"  - {label}")
