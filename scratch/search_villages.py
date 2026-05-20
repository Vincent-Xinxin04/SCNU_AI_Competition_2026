import csv
targets = ['Bandagal', 'Mautara', 'Faridpur', 'Chak Modpur', 'Kunduakori', 'Arazi Brahmandihi']
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        for t in targets:
            if t in row['Subject']:
                print(f"{row['Subject']} -> {row['Object']} ({row['Label']})")
