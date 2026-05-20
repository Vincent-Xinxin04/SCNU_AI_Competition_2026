import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if 'station' in row['Subject'].lower() or 'station' in row['Object'].lower():
            try:
                float(row['Object'])
                print(f"{row['Subject']} -> {row['Object']} ({row['Label']})")
            except ValueError:
                pass
