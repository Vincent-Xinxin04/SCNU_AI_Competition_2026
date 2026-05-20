import csv
from collections import defaultdict

test_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.742.csv'
overpredicted = [
    'has facility', 'platform', 'highway system', 'use', 'has grammatical gender',
    'parity quantum number', 'operating system', 'sports season of league or competition',
    'discovery method', 'linguistic typology', 'instrumentation', 'religious order',
    'position held', 'type of orbit', 'armament'
]

examples = defaultdict(list)
with open(test_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Label'] in overpredicted:
            examples[row['Label']].append(f"{row['Subject']} -> {row['Object']}")

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\overpredicted_examples.txt', 'w', encoding='utf-8') as f:
    for label in overpredicted:
        f.write(f"\n--- {label} ---\n")
        for ex in examples[label][:15]: # print up to 15 examples
            f.write(ex + "\n")
