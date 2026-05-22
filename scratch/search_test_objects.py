import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'

settlements = {
    'Peklo', 'Butterhuizen', 'Stănești', 'Keřkov', 'Xibër-Hane', 'Rinas', 'Kacni', 
    'Peshtan i Vogël', 'Bezděkov', 'Cruquius-Oost', 'Nieuwe Meer', 'Godolesh', 
    'Sinești', 'Hidrovori', 'Schineni', 'Gjallicë', 'Lalm-Lukaj', 'Slobozia'
}

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        if o in settlements or any(sett.lower() in o.lower() for sett in settlements):
            line = f"Match in test: {s} -> {o}"
            print(line.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
