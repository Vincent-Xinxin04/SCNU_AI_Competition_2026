import csv

sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'
v9_candidates = [
    ('Lake Bolsena', 'Italy'),
    ('Lake Bracciano', 'Italy'),
    ('Alphonse Allard Fonds', 'fonds'),
    ('2016 Volta a Portugal', 'José Gonçalves'),
    ('Draw the Line', 'Al Hirschfeld'),
    ('Milano Porta Garibaldi', '1'),
    ("2017 Giro d'Italia, Stage 4", 'Dimension Data 2017'),
    ("2017 Giro d'Italia, Stage 9", 'Dimension Data 2017'),
    ('Appliqué inv. 848', 'appliqué'),
    ('Nenque', 'Phoenix')
]

v9_candidates_set = { (s.lower(), o.lower()) for s, o in v9_candidates }

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s, o, l = row['Subject'], row['Object'], row['Label']
        if (s.lower(), o.lower()) in v9_candidates_set:
            print(f"Found: ({s} -> {o}) | Predicted: {l}")
