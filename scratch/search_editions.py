import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'

subjects = {'Too Much Love Will Kill You', 'Freden', 'The Boys of Fall', 'Madame Bovary', 'Thunder', 'El la skizlibro', 'Dead Horse'}

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        if s in subjects:
            line = f"Test candidate: {s} -> {o}"
            print(line.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
