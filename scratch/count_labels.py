import csv
from collections import Counter
counts = Counter()
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        counts[row['Label']] += 1

print("Target labels and their predicted counts:")
for label in ['winner', 'general classification of race participants', 'mountains classification', 'young rider classification', 'points classification']:
    print(f"{label}: {counts[label]}")
