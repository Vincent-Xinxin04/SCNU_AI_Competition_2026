import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

# Load predictions
preds = {}
with open(sub_file, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        preds[(row['Subject'].strip(), row['Object'].strip())] = row['Label'].strip()

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        if 'stage' in s.lower():
            # Check if object looks like a team or rider
            # usually teams end with a year or contain team, season, floors, jumbo, etc. or riders' names
            is_team_or_rider = any(kw in o.lower() for kw in ['team', 'season', 'floors', 'jumbo', 'alpecin', 'sky', 'sunweb', 'canyon', 'orica', 'dimension', 'movistar', 'trek', 'quick-step', 'bora', 'bmc', 'astana', 'mitchelton', 'lotto', 'bahrain', 'katusha', 'ag2r', 'cofidis', 'direct énerge', 'direct energie', 'wanty', 'fortuneo', 'vital concept']) or o.endswith('2017') or o.endswith('2018') or o.endswith('2019') or o.endswith('2016') or o.endswith('2015')
            
            # Or if the prediction is currently something like participating team, or combination classification, etc.
            curr = preds.get((s, o), 'NONE')
            if is_team_or_rider or curr in ['participating team', 'combination classification', 'teams classification by time', 'teams classification by points']:
                line = f"Candidate: {s} -> {o} | Pred: {curr}"
                print(line.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
