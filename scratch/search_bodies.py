import csv
bodies = ['Eris', 'Makemake', 'Haumea', 'Quaoar']
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        for b in bodies:
            if b in row['Subject'] or b in row['Object']:
                print(f"{row['Subject']} -> {row['Object']} ({row['Label']})")
