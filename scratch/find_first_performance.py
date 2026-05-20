import csv
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row['Label'] in ['location', 'located in the administrative territorial entity', 'country']:
            s = row['Subject'].lower()
            if 'symphony' in s or 'concerto' in s or 'opera' in s or 'sonata' in s or 'festival' in s:
                print(f"{row['Subject']} -> {row['Object']} (Pred: {row['Label']})")
