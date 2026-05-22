import csv

sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

candidates = [
    ('Draw the Line', 'Al Hirschfeld'),
    ('Tango Rey', '8600'),
    ('Tango Rey', 'Avatiu'),
    ('CCGS Des Groseilliers', '7.44'),
    ('2018 Giro del Piemonte', 'Sonny Colbrelli'),
    ('2018 Vuelta a Andalucía', 'Silvan Dillier'),
    ('2015 Presidential Tour of Turkey', 'Songezo Jim'),
    ('Lake Bolsena', 'Italy'),
    ('Lake Bracciano', 'Italy'),
    ('Lake Bolsena', '305.0'),
    ("2011 Grand Prix d'Isbergues", "Stuart O'Grady"),
    ('Appliqué inv. 848', 'appliqué'),
    ('Appliqué inv. 734', 'appliqué'),
    ('Appliqué inv. 733', 'appliqué'),
    ('Appliqué inv. 9619', 'appliqué'),
    ('Appliqué inv. 739', 'appliqué'),
    ('Appliqué inv. 9472', 'appliqué')
]

cand_set = {(s.strip().lower(), o.strip().lower()): (s, o) for s, o in candidates}

found = {}
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        l = row['Label'].strip()
        key = (s.lower(), o.lower())
        if key in cand_set:
            found[cand_set[key]] = l

print("Current predictions for candidates:")
for c in candidates:
    print(f"{c[0]} -> {c[1]}: {found.get(c, 'NOT FOUND')}")
