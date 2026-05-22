import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

def get_triplets(filename):
    triplets = set()
    with open(os.path.join(train_dir, filename), 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                triplets.add((row[0].strip(), row[1].strip()))
    return triplets

w_triplets = get_triplets('winner.csv')
gc_triplets = get_triplets('general classification of race participants.csv')

overlap = w_triplets.intersection(gc_triplets)
print(f"Winner count: {len(w_triplets)}")
print(f"GC count: {len(gc_triplets)}")
print(f"Overlap count: {len(overlap)}")
print("First 10 overlaps:")
for item in list(overlap)[:10]:
    print(item)
