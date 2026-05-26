import os
import csv
import sys
import pandas as pd
import re

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_final.csv")

# Load training labels list to match against labels.txt
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

pred_counts = sub_df['Label'].value_counts()
missing_labels = [l for l in labels if pred_counts.get(l, 0) == 0]

print(f"Searching candidates for {len(missing_labels)} missing labels...\n")

def check_person(s):
    # Quick heuristic if a string looks like a person's name (2 or 3 capitalized words, no common company/place nouns)
    s = str(s).strip()
    words = s.split()
    if len(words) >= 2 and len(words) <= 4:
        if all(w[0].isupper() if w[0].isalpha() else True for w in words):
            # Exclude common place/org keywords
            excl = ['association', 'station', 'stadium', 'university', 'school', 'council', 'museum', 'library', 'cathedral', 'church', 'basilica', 'palace', 'bridge', 'route', 'highway', 'lake', 'river', 'park', 'national', 'state', 'county', 'district', 'town', 'municipality', 'group', 'company', 'records', 'center', 'centre', 'club', 'team', 'cup', 'trophy']
            if not any(k in s.lower() for k in excl):
                return True
    return False

# Rule dictionary
# Key: missing label, Value: list of tuples (test_row_index, confidence, reason)
candidates = {}

for ml in missing_labels:
    candidates[ml] = []

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    current_pred = sub_df.loc[idx, 'Label']
    
    # 1. negative prognostic predictor
    if 'CDKN2A p16 EXPRESSION' in s or 'CD274 EXPRESSION' in s:
        # Check if the object is a cancer (excluding generic 'cancer' which is positive in train)
        if 'cancer' in o.lower() or 'carcinoma' in o.lower() or 'tumor' in o.lower() or 'tumour' in o.lower():
            if o.lower() not in ['cancer', 'breast cancer']: # these are positive in train
                candidates['negative prognostic predictor'].append((idx, 0.9, f"Subject: '{s}', Object: '{o}' (cancer), Current: '{current_pred}'"))

    # 2. art director
    # Train: Greyfriars Bobby -> Michael Stringer
    # Test candidates: subject is a movie/book/play, object is a person.
    # Check if current prediction is composer, screenplay, director, film editor, performer
    if current_pred in ['composer', 'screenwriter', 'director', 'film editor', 'costume designer'] and check_person(o):
        # Let's check if the subject is a creative work (often has title-like casing or words like movie, film)
        candidates['art director'].append((idx, 0.5, f"Subject: '{s}', Object: '{o}' (person), Current: '{current_pred}'"))

    # 3. patron saint
    # Subject: Basilica/Church, Object: Virgin Mary or Saint
    if 'basilica' in s.lower() or 'church' in s.lower() or 'cathedral' in s.lower():
        if 'virgin mary' in o.lower() or 'saint' in o.lower() or 'st.' in o.lower() or o in ['Mary', 'Joseph', 'Peter', 'Paul', 'Nicholas']:
            candidates['patron saint'].append((idx, 0.8, f"Subject: '{s}', Object: '{o}' (saint), Current: '{current_pred}'"))

    # 4. target
    # Subject: attack/shooting/bombing, Object: building/group
    if 'attack' in s.lower() or 'shooting' in s.lower() or 'bombing' in s.lower() or 'massacre' in s.lower():
        # exclude numbers
        if not o.isdigit() and len(o) > 3:
            candidates['target'].append((idx, 0.6, f"Subject: '{s}', Object: '{o}', Current: '{current_pred}'"))

    # 5. languages spoken
    # Subject: Character/Person, Object: Language
    if 'language' in o.lower() or o in ['Galactic Basic', 'Huttese', 'English', 'Spanish', 'French', 'German']:
        candidates['languages spoken'].append((idx, 0.7, f"Subject: '{s}', Object: '{o}', Current: '{current_pred}'"))

    # 6. payload mass
    # Subject: spacecraft/satellite/rocket, Object: number + kg or just number
    if any(k in s.lower() for k in ['satellite', 'spacecraft', 'rocket', 'soyuz', 'kosmos', 'hiten', 'dawn']):
        # If object is a number representing mass
        try:
            val = float(o)
            if 10.0 <= val <= 20000.0:
                candidates['payload mass'].append((idx, 0.5, f"Subject: '{s}', Object: '{o}' (number), Current: '{current_pred}'"))
        except ValueError:
            pass

    # 7. service retirement
    # Subject: train/locomotive/ship, Object: date
    # Let's check if subject contains no. or class or locomotive or ship
    if 'class' in s.lower() or 'no.' in s.lower() or 'locomotive' in s.lower() or 'submarine' in s.lower() or 'ss' in s.lower() or 'hms' in s.lower():
        # Object is a date/year
        if re.match(r'^\d{4}-\d{2}-\d{2}$', o) or (o.isdigit() and len(o) == 4):
            candidates['service retirement'].append((idx, 0.5, f"Subject: '{s}', Object: '{o}' (date/year), Current: '{current_pred}'"))

    # 8. notable work
    # Subject: software company / developer / author, Object: product / game / book
    if current_pred in ['developer', 'publisher', 'author', 'creator']:
        candidates['notable work'].append((idx, 0.4, f"Subject: '{s}', Object: '{o}', Current: '{current_pred}'"))

    # 9. absolute magnitude
    # Subject: asteroid/comet/star/J2000, Object: small float or integer (typically -5.0 to 15.0)
    # Check if subject is star/comet (e.g. HD, Gliese, 2MASS, UCAC, SDSS, TYC, Cl*)
    if any(k in s for k in ['HD', 'Gliese', '2MASS', 'UCAC', 'SDSS', 'TYC', 'Cl*', 'NGC', 'BD', 'V*']):
        try:
            val = float(o)
            if -10.0 <= val <= 25.0:
                # exclude common physical parameters like parallax (usually small positive), temperature (large), etc.
                if current_pred not in ['effective temperature', 'distance from Earth', 'radial velocity', 'parallax']:
                    candidates['absolute magnitude'].append((idx, 0.6, f"Subject: '{s}', Object: '{o}' (float), Current: '{current_pred}'"))
        except ValueError:
            pass

    # 10. presenter
    # Subject: TV show/award/pageant, Object: person
    if 'miss' in s.lower() or 'award' in s.lower() or 'show' in s.lower() or 'festival' in s.lower():
        if check_person(o) and current_pred in ['follows', 'followed by', 'creator', 'director']:
            candidates['presenter'].append((idx, 0.5, f"Subject: '{s}', Object: '{o}' (person), Current: '{current_pred}'"))

    # 11. distributor
    # Subject: movie/film/show, Object: company (contains Corp, Inc, Group, Pictures, Releasing, Distribution, Television, Entertainment)
    if any(k in o.lower() for k in ['corp', 'inc', 'group', 'pictures', 'releasing', 'distribution', 'television', 'entertainment', 'studios', 'records', 'media', 'films']):
        if current_pred in ['publisher', 'production company', 'country', 'headquarters location']:
            candidates['distributor'].append((idx, 0.5, f"Subject: '{s}', Object: '{o}' (company), Current: '{current_pred}'"))

    # 12. stage classification
    # Subject: stage of a race (Stage 1, Stage 2, stage 3a, stage), Object: cycling team / rider / location
    if 'stage' in s.lower() and ('tour' in s.lower() or 'giro' in s.lower() or 'vuelta' in s.lower() or 'class' in s.lower()):
        candidates['stage classification'].append((idx, 0.5, f"Subject: '{s}', Object: '{o}', Current: '{current_pred}'"))

    # 13. after a work by
    # Subject: painting/sculpture/derivative work, Object: author/creator
    # If current_pred is author or creator or director, check if subject is adaptation/artwork
    if current_pred in ['creator', 'author', 'director'] and check_person(o):
        candidates['after a work by'].append((idx, 0.4, f"Subject: '{s}', Object: '{o}' (person), Current: '{current_pred}'"))

    # 14. powered by
    # Subject: car/locomotive/vehicle, Object: engine / fuel type / power plant
    if any(k in o.lower() for k in ['engine', 'locomotive', 'reactor', 'motor', 'turbine', 'propeller', 'fuel']):
        candidates['powered by'].append((idx, 0.6, f"Subject: '{s}', Object: '{o}', Current: '{current_pred}'"))

    # 15. basin country
    # Subject: lake/river/sea, Object: country
    if 'lake' in s.lower() or 'river' in s.lower() or 'sea' in s.lower() or 'reservoir' in s.lower():
        if current_pred in ['country', 'country of origin', 'located in the administrative territorial entity']:
            candidates['basin country'].append((idx, 0.6, f"Subject: '{s}', Object: '{o}' (country), Current: '{current_pred}'"))

    # 16. gross tonnage
    # Subject: ship/vessel/boat/SS/HMS/RV, Object: large integer (typically 1000 to 300000)
    if 'ss' in s.lower() or 'hms' in s.lower() or 'rv' in s.lower() or 'cruise' in s.lower() or 'vessel' in s.lower() or 'ship' in s.lower():
        try:
            val = int(o)
            if 500 <= val <= 250000:
                candidates['gross tonnage'].append((idx, 0.6, f"Subject: '{s}', Object: '{o}' (int), Current: '{current_pred}'"))
        except ValueError:
            pass

    # 17. maintained by
    # Subject: route/highway/bridge/road, Object: authority/government/department
    if 'route' in s.lower() or 'highway' in s.lower() or 'bridge' in s.lower() or 'road' in s.lower():
        if any(k in o.lower() for k in ['department', 'prefecture', 'ministry', 'authority', 'corporation', 'agency', 'district', 'trust', 'national']):
            candidates['maintained by'].append((idx, 0.6, f"Subject: '{s}', Object: '{o}' (authority), Current: '{current_pred}'"))

    # 18. contains settlement
    # Subject: municipality / district / county, Object: village / town
    if current_pred == 'located in the administrative territorial entity':
        candidates['contains settlement'].append((idx, 0.3, f"Subject: '{s}', Object: '{o}', Current: '{current_pred}'"))

for label, cand_list in sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True):
    if len(cand_list) > 0:
        print(f"\nLabel: {label} (Found {len(cand_list)} candidates):")
        for idx, conf, reason in cand_list[:15]:
            print(f"  Row {idx:4d} [Conf {conf:.1f}]: {reason}")
