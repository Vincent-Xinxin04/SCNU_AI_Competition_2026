import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if 'fonds' in row['Object'].lower() or 'fonds' in row['Subject'].lower():
            print(f"{row['Subject']} -> {row['Object']} ({row['Label']})")
