import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.745.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row['Subject'] == 'Gran Premio Femenino de Plouay 2018':
            print(f"{row['Subject']} -> {row['Object']} (Pred: {row['Label']})")
