import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'

distributors = {
    'Warner Bros. Television', 'Sony Pictures Releasing', 'NBCUniversal Television Distribution', 
    'Walt Disney Studios Motion Pictures', '20th Television', 'Warner Bros.'
}

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        if o in distributors or any(d.lower() in o.lower() for d in distributors):
            line = f"Test candidate: {s} -> {o}"
            print(line.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
