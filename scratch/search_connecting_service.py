import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'

stations = {'Méndez Álvaro', '30th Street Station', 'Wheaton', 'Forest Glen station', 'Glenmont station'}

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        if s in stations:
            line = f"Test candidate: {s} -> {o}"
            print(line.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
