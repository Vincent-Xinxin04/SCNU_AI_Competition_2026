import csv

input_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'
output_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final_temp.csv'

fixes = {
    ('droperidol', '2.5', 'lower flammable limit'): 'defined daily dose'
}

data = []
c = 0
with open(input_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s, o, l = row['Subject'], row['Object'], row['Label']
        if (s, o, l) in fixes:
            row['Label'] = fixes[(s, o, l)]
            c += 1
        data.append(row)

with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Subject', 'Object', 'Label'])
    writer.writeheader()
    writer.writerows(data)

print(f"Applied {c} dose fixes.")
