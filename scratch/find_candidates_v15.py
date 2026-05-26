"""
Comprehensive v15 override candidate scanner.
Focuses on finding truly high-confidence candidates for missing labels.
"""
import pandas as pd
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

# Load training data for all key labels
train = {}
import os
for fname in os.listdir('dataset/Train_Set'):
    label = fname.replace('.csv', '')
    try:
        train[label] = pd.read_csv(f'dataset/Train_Set/{fname}')
    except:
        pass

with open('dataset/labels.txt', 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f if line.strip()]

pred_counts = sub_df['Label'].value_counts()
missing_labels = [l for l in labels if pred_counts.get(l, 0) == 0]
print(f"Currently missing {len(missing_labels)} labels from submission.")
print()

# ===== SECTION 1: inflows =====
# Pattern: Lake/Sea -> River (river flows into the lake)
# Training: Lake Maggiore -> San Bernardino, Maggia, Ticino
# Training: Lake Neuchâtel -> Seyon, Thielle

print("=== inflows candidates ===")
inflow_subjects = {'Lake Maggiore', 'Lake Neuchâtel', 'Lake Constance'}  # from training + new
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    if 'lake' in s.lower() or 'sea' in s.lower():
        # Object should be a river name
        if not o.isdigit():
            try:
                float(o)
            except:
                if pred not in ['inflows']:
                    if o in ['Rhine', 'Rhone', 'Seyon', 'Thielle', 'Ticino', 'Maggia', 'San Bernardino']:
                        print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}' [HIGH CONFIDENCE]")

print()

# ===== SECTION 2: service retirement =====
# Pattern: Military equipment -> 1945-01-01 (end of WWII)
# Training: LSWR 177-class -> 1895-xx-01 as service retirement

print("=== service retirement candidates ===")
# Get service entry dates from training
service_entry_items = set(train.get('service entry', pd.DataFrame()).get('Subject', pd.Series()).unique())
service_retire_items = set(train.get('service retirement', pd.DataFrame()).get('Subject', pd.Series()).unique())

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    # Check if it's a known service entry item with a retirement date
    if s in service_entry_items and pred == 'service entry':
        # Check if the object is LATER than the known service entry date
        entry_rows = train['service entry'][train['service entry']['Subject'] == s]
        if not entry_rows.empty:
            known_entry_date = str(entry_rows['Object'].iloc[0])
            if o > known_entry_date:
                print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}' | Known entry: {known_entry_date} [Could be retirement]")

print()

# ===== SECTION 3: Connecting service =====
# Pattern: Station -> line/service (different from connecting LINE)
# Training: Forest Glen station -> Red Line, Wheaton -> Red Line
print("=== connecting service candidates ===")
conn_service_objects = set(train.get('connecting service', pd.DataFrame()).get('Object', pd.Series()).unique())
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    if o in conn_service_objects and pred != 'connecting service':
        print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}' [EXACT MATCH]")
        
# Also check training subjects
conn_service_subjects = set(train.get('connecting service', pd.DataFrame()).get('Subject', pd.Series()).unique())
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    if s in conn_service_subjects and pred != 'connecting service':
        train_obj = train['connecting service'][train['connecting service']['Subject'] == s]['Object'].values
        print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}' | Training obj: {train_obj} [SUBJECT MATCH]")

print()

# ===== SECTION 4: stage classification =====
# Cycling stages paired with team names
# Training: 2018 Tour de France stage 3 -> Katusha-Alpecin 2018 = stage classification
print("=== stage classification candidates ===")
stage_teams = set(train.get('stage classification', pd.DataFrame()).get('Object', pd.Series()).unique())
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    if 'stage' in s.lower() and ('tour' in s.lower() or 'giro' in s.lower() or 'vuelta' in s.lower()):
        if o in stage_teams or any(t in o for t in ['2017', '2018', '2019', '2013']):
            if pred in ['points classification', 'teams classification by points', 'general classification of race participants']:
                print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}'")

print()

# ===== SECTION 5: target (attacks) =====
# Pattern: attack/shooting -> location/target entity
# Training: 2019 Pulwama attack -> Central Reserve Police Force = target
# Training: Christchurch mosque shootings -> Al Noor Mosque = target
# Training: Santa Fe High School shooting -> high school student = target
print("=== target candidates ===")
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    if ('attack' in s.lower() or 'shooting' in s.lower() or 'bombing' in s.lower() 
        or 'massacre' in s.lower() or 'arson' in s.lower()):
        # Object should be a specific target entity (not generic type, not a number, not a date)
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', o) and not o.isdigit():
            if pred == 'location' and o not in ['terrorism', 'suicide attack', 'hostage crisis', 'Boston']:
                print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}' [location -> target?]")
            if pred == 'participant' and 'shooting' in s.lower():
                # participant who was targeted vs perpetrator
                print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}' [participant -> target?]")

print()

# ===== SECTION 6: Exact matches against ALL training labels =====
print("=== Exact subject+object matches against training (current pred != training label) ===")
all_train_pairs = {}
for label, df in train.items():
    if 'Subject' in df.columns and 'Object' in df.columns:
        for _, trow in df.iterrows():
            key = (str(trow['Subject']).strip(), str(trow['Object']).strip())
            if key not in all_train_pairs:
                all_train_pairs[key] = []
            all_train_pairs[key].append(label)

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    key = (s, o)
    if key in all_train_pairs:
        true_labels = all_train_pairs[key]
        if pred not in true_labels:
            print(f"Row {idx:4d} | '{s}' -> '{o}' | Pred: '{pred}' | Training labels: {true_labels}")
