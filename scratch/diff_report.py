import csv

old_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.742.csv'
new_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

old_data = {}
with open(old_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        old_data[(row['Subject'], row['Object'])] = row['Label']

changes = []
with open(new_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        s, o, l_new = row['Subject'], row['Object'], row['Label']
        l_old = old_data.get((s, o))
        if l_old != l_new:
            changes.append(f"Subject: '{s}' | Object: '{o}'\n  - 原预测 (0.742): {l_old}\n  + 新修正 (Final): {l_new}\n")

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\diff_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total changes: {len(changes)}\n\n")
    for c in changes:
        f.write(c)
