import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv', 'r', encoding='utf-8-sig') as f:
    with open(r'f:\github\SCNU_AI_Competition_2026\scratch\similar_pairs.txt', 'w', encoding='utf-8') as out:
        for row in csv.DictReader(f):
            s, o = row['Subject'], row['Object']
            if s != o and (o in s or s in o):
                if len(s) > 3 and len(o) > 3:
                    out.write(f"{s} -> {o} ({row['Label']})\n")
