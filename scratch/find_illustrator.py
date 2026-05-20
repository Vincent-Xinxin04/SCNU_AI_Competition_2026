import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row['Label'] in ['composer', 'author', 'creator']:
            print(f"{row['Subject']} -> {row['Object']} (Pred: {row['Label']})")
