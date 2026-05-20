import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row['Label'] == 'publisher':
            s = row['Subject'].lower()
            if 'journal' in s or 'magazine' in s:
                print(f"{row['Subject']} -> {row['Object']}")
