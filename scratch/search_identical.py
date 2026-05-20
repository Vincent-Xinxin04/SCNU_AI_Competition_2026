import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    with open(r'f:\github\SCNU_AI_Competition_2026\scratch\identical_matches.txt', 'w', encoding='utf-8') as out:
        for row in csv.DictReader(f):
            if row['Subject'].strip() == row['Object'].strip():
                out.write(f"{row['Subject']} -> {row['Object']} ({row['Label']})\n")
