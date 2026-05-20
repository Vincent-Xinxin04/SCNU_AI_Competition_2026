import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row['Label'] == 'age of majority':
            print(f"{row['Subject']} -> {row['Object']} ({row['Label']})")
