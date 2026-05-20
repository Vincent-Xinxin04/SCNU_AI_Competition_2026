import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load test labels
train_labels = set(f[:-4] for f in os.listdir(train_dir) if f.endswith('.csv'))
test_preds = set()
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        test_preds.add(row['Label'].strip())
remaining_missing = sorted(list(train_labels - test_preds))

print(f"Remaining missing labels: {len(remaining_missing)}")

# Load test rows
test_rows = []
with open(test_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        test_rows.append(row)

# Load all training samples for remaining missing labels
train_samples = {}
for ml in remaining_missing:
    file_path = os.path.join(train_dir, f"{ml}.csv")
    samples = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                samples.append((row[0].strip(), row[1].strip()))
    train_samples[ml] = samples

# We will look for:
# 1. Subject overlaps: Test row has the exact same Subject as a Train row for a missing label.
# 2. Object overlaps: Test row has the exact same Object as a Train row for a missing label.
# 3. Fuzzy matches: Subject contains/is contained, or Object contains/is contained.

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\remaining_candidates.txt', 'w', encoding='utf-8') as out:
    out.write(f"Remaining missing labels count: {len(remaining_missing)}\n\n")
    
    for ml in remaining_missing:
        samples = train_samples[ml]
        if not samples:
            continue
            
        train_subjs = set(s[0] for s in samples)
        train_objs = set(s[1] for s in samples)
        
        candidates = []
        
        for r in test_rows:
            ts, to, tl = r['Subject'], r['Object'], r['Label']
            
            # Match 1: Exact Subject match AND similar Object type/value
            if ts in train_subjs:
                candidates.append((r, "Exact Subject Match", f"Train Subj: '{ts}'"))
                
            # Match 2: Exact Object match
            elif to in train_objs:
                candidates.append((r, "Exact Object Match", f"Train Obj: '{to}'"))
                
            # Match 3: Fictional/special patterns (e.g. Star Trek, songs, etc.)
            else:
                # If Subject is similar to any train Subject
                for trs, tro in samples:
                    if len(trs) > 4 and len(ts) > 4:
                        if trs.lower() in ts.lower() or ts.lower() in trs.lower():
                            if tro.lower() in to.lower() or to.lower() in tro.lower():
                                candidates.append((r, "Fuzzy Double Match", f"Train: '{trs}' -> '{tro}'"))
                                break
                                
        if candidates:
            out.write(f"=== Label: '{ml}' (Train samples: {len(samples)}) ===\n")
            # De-duplicate candidates
            seen = set()
            for r, mtype, reason in candidates:
                key = (r['Subject'], r['Object'], r['Label'])
                if key not in seen:
                    seen.add(key)
                    out.write(f"  Candidate: {r['Subject']} -> {r['Object']} (Current Pred: {r['Label']})\n")
                    out.write(f"    Match Type: {mtype} | Reason: {reason}\n")
            out.write("\n")
            
print("Done. Saved to remaining_candidates.txt")
