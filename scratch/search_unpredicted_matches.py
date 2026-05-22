import os
import csv
import sys
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# 53 unpredicted labels
unpredicted = [
    "CJKV variant character", "VAT-rate", "absolute magnitude", "after a work by",
    "art director", "commissioned by", "connecting service", "contains settlement",
    "contributor to the creative work or subject", "cover art by", "distributor",
    "edition or translation of", "editor-in-chief", "facet of", "gross tonnage",
    "has decorative pattern", "home world", "inflows", "interested in", "languages spoken",
    "level of description", "located in the ecclesiastical territorial entity",
    "longitude of ascending node", "lower flammable limit", "lyrics by", "maintained by",
    "male population", "mean anomaly", "name", "narrator", "negative prognostic predictor",
    "nominal GDP per capita", "notable work", "number of decimal digits",
    "number of platform faces", "official symbol", "opposite of",
    "organization directed from the office or person", "patron saint", "payload mass",
    "place of death", "powered by", "presenter", "recording or performance of",
    "service retirement", "short name", "stage classification", "statistical leader",
    "target", "territory overlaps", "time of spacecraft orbit decay",
    "unemployment rate", "young rider classification"
]

# Load Train Set for these labels
train_subjects = defaultdict(set)
train_objects = defaultdict(set)
train_triplets = defaultdict(set)

for label in unpredicted:
    filepath = os.path.join(train_dir, f"{label}.csv")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    s, o = row[0].strip(), row[1].strip()
                    train_subjects[label].add(s)
                    train_objects[label].add(o)
                    train_triplets[label].add((s, o))

# Load test set rows and their current predictions
test_rows = []
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        test_rows.append({
            'idx': idx,
            's': row['Subject'].strip(),
            'o': row['Object'].strip(),
            'l': row['Label'].strip()
        })

print("Checking for candidate matches for unpredicted labels:")
for label in unpredicted:
    print(f"\n--- Label: {label} ---")
    triplets = train_triplets[label]
    subjs = train_subjects[label]
    objs = train_objects[label]
    
    # 1. Exact triplet match
    exact_hits = []
    for row in test_rows:
        if (row['s'], row['o']) in triplets:
            exact_hits.append(row)
    if exact_hits:
        print("  Exact triplet matches in Test:")
        for h in exact_hits:
            print(f"    Row {h['idx']}: {h['s']} -> {h['o']} | Current Pred: {h['l']}")
            
    # 2. Subject exact match (to identify if the object could be of the same relation)
    subj_hits = []
    for row in test_rows:
        if row['s'] in subjs and (row['s'], row['o']) not in triplets:
            subj_hits.append(row)
    if subj_hits:
        print("  Subject-only matches in Test (different objects):")
        for h in subj_hits[:15]:  # limit to 15
            print(f"    Row {h['idx']}: {h['s']} -> {h['o']} | Current Pred: {h['l']}")
        if len(subj_hits) > 15:
            print(f"    ... and {len(subj_hits) - 15} more")

    # 3. Object exact match (useful for numeric values or specific entities)
    obj_hits = []
    for row in test_rows:
        if row['o'] in objs and (row['s'], row['o']) not in triplets:
            # For common strings or numbers, avoid printing too many hits
            if len(row['o']) > 2 and row['o'] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                obj_hits.append(row)
    if obj_hits:
        print("  Object-only matches in Test (different subjects):")
        for h in obj_hits[:15]:
            print(f"    Row {h['idx']}: {h['s']} -> {h['o']} | Current Pred: {h['l']}")
        if len(obj_hits) > 15:
            print(f"    ... and {len(obj_hits) - 15} more")
