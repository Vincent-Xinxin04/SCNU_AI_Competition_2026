import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    print("Header:", header)
    
    rows = list(reader)
    print("Total rows:", len(rows))
    print("Sample rows:")
    for i in range(min(30, len(rows))):
        s, o = rows[i]
        line = f"Row {i}: {s} -> {o}"
        print(line.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
