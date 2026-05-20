import csv

test_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

candidates = {
    'number of episodes': [],
    'location of first performance': [],
    'illustrator': [],
    'editor-in-chief': [],
    'enemy of': [],
    'teams classification by points': [],
    'wavelength': [],
    'defined daily dose': []
}

with open(test_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s, o, l = row['Subject'], row['Object'], row['Label']
        
        # Look for numbers that might be episodes
        if o.isdigit() and int(o) < 1000 and l in ['numeric value', 'number of parts of this work']:
            candidates['number of episodes'].append(row)
            
        # Look for locations that might be first performance
        if l in ['location', 'located in the administrative territorial entity'] and 'premiere' in s.lower() or 'symphony' in s.lower() or 'concerto' in s.lower() or 'opera' in s.lower():
            candidates['location of first performance'].append(row)
            
        # Look for illustrators
        if l in ['author', 'creator', 'contributor to the creative work or subject', 'publisher'] and ('book' in s.lower() or 'comic' in s.lower() or 'novel' in s.lower() or 'manga' in s.lower()):
            candidates['illustrator'].append(row)

        # Look for editor in chief
        if l in ['editor', 'director', 'author', 'chairperson'] and ('magazine' in s.lower() or 'journal' in s.lower() or 'newspaper' in s.lower()):
            candidates['editor-in-chief'].append(row)
            
        # Look for wavelength
        if l in ['numeric value', 'length'] and ('nm' in o.lower() or 'laser' in s.lower() or 'light' in s.lower()):
            candidates['wavelength'].append(row)
            
        # Teams classification by points
        if l in ['winner', 'part of', 'participant'] and ('classification' in s.lower() or 'tour de' in s.lower()):
            candidates['teams classification by points'].append(row)

for k, v in candidates.items():
    if v:
        print(f"--- Potential {k} ---")
        for row in v[:10]:
            print(f"{row['Subject']} -> {row['Object']} (Currently: {row['Label']})")
