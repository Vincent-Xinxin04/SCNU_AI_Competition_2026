import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        if 'fonds' in o.lower() or 'fonds' in s.lower():
            line = f"Test candidate: {s} -> {o}"
            print(line.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
