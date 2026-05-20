import csv
from collections import Counter
counts = Counter()
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        counts[row['Label']] += 1
for label, count in counts.most_common():
    print(f"{label}: {count}")
