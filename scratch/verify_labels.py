import csv
from collections import Counter
counts = Counter()
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.75x.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        counts[row['Label']] += 1

target_labels = [
    'compulsory education (maximum age)',
    'number of spoilt votes',
    'points goal scored by',
    'coextensive with',
    'instance of'
]

for label in target_labels:
    print(f"{label}: {counts[label]}")
