import csv, os, re, shutil, sys

# Ensure console can handle Unicode
sys.stdout.reconfigure(encoding='utf-8')

# Paths
submission_file = r'f:\\github\\SCNU_AI_Competition_2026\\result\\submission_final.csv'
backup_file = r'f:\\github\\SCNU_AI_Competition_2026\\result\\submission_0.7791_backup.csv'
candidate_file = r'f:\\github\\SCNU_AI_Competition_2026\\scratch\\candidate_overrides_v2.txt'
output_file = submission_file  # overwrite in place after backup

# Backup current submission (one‑time backup)
shutil.copyfile(submission_file, backup_file)
print(f"Backup saved to {backup_file}")

# Load current submission into dict
submission_map = {}
rows = []
with open(submission_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        key = (row['Subject'].strip(), row['Object'].strip())
        submission_map[key] = row
        rows.append(row)

# Parse candidate overrides (format: ('Subject', 'Object') -> NewLabel)
pattern = re.compile(r"\('([^']+)',\s*'([^']+)'\)\s*->\s*(.+)")
overrides = {}
with open(candidate_file, 'r', encoding='utf-8-sig') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            subj, obj, new_label = m.groups()
            new_label = new_label.strip()
            overrides[(subj, obj)] = new_label
        else:
            print(f"Unparsed line: {line}")

modified = 0
for key, row in submission_map.items():
    if key in overrides:
        old_label = row['Label']
        new_label = overrides[key]
        if old_label != new_label:
            row['Label'] = new_label
            modified += 1
            # safe print with try/except to avoid encoding issues
            try:
                print(f"Override: ({key[0]} -> {key[1]}) | Old: {old_label} | New: {new_label}")
            except Exception:
                pass

# Write updated submission
with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Total rows: {len(rows)}")
print(f"Modified rows: {modified}")
print("Saved submission_final.csv successfully with utf-8-sig encoding.")
