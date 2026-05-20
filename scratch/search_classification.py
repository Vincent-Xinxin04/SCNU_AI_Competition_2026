import csv

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\classification_results.txt', 'w', encoding='utf-8') as out_f:
    with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.745.csv', 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if 'classification' in row['Label'] or 'classification' in row['Object'].lower() or 'classification' in row['Subject'].lower():
                out_f.write(f"{row['Subject']} -> {row['Object']} (Pred: {row['Label']})\n")
