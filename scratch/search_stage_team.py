import os
import csv

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

results = []
for f in os.listdir(train_dir):
    if f.endswith('.csv'):
        label = f[:-4]
        filepath = os.path.join(train_dir, f)
        with open(filepath, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    s = row[0].strip()
                    o = row[1].strip()
                    if 'Stage' in s and ('201' in o or '202' in o or '200' in o or 'team' in o.lower() or 'floors' in o.lower() or 'jumbo' in o.lower()):
                        results.append(f"Train match in [{label}]: {s} -> {o}")

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\stage_team_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print(f"Done. Saved {len(results)} results.")
