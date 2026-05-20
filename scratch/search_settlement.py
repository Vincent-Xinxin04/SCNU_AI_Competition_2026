import csv
targets = ['Raspenava', 'Heerhugowaard']
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        for t in targets:
            if t in row['Subject'] or t in row['Object']:
                print(f"{row['Subject']} -> {row['Object']} ({row['Label']})")
