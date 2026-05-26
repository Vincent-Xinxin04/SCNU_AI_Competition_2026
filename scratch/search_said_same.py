import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv", encoding='utf-8-sig')
sub_df = pd.read_csv("result/submission_final.csv", encoding='utf-8-sig')

god_pairs = {
    'ares': 'mars', 'mars': 'ares',
    'enyo': 'bellona', 'bellona': 'enyo',
    'zeus': 'jupiter', 'jupiter': 'zeus',
    'hera': 'juno', 'juno': 'hera',
    'poseidon': 'neptune', 'neptune': 'poseidon',
    'athena': 'minerva', 'minerva': 'athena',
    'artemis': 'diana', 'diana': 'artemis',
    'aphrodite': 'venus', 'venus': 'aphrodite',
    'hermes': 'mercury', 'mercury': 'hermes',
    'hephaestus': 'vulcan', 'vulcan': 'hephaestus',
    'dionysus': 'bacchus', 'bacchus': 'dionysus',
    'hestia': 'vesta', 'vesta': 'hestia'
}

keys_pairs = {
    'c-sharp major': 'd-flat major', 'd-flat major': 'c-sharp major',
    'g-flat major': 'f-sharp major', 'f-sharp major': 'g-flat major',
    'g-sharp major': 'a-flat major', 'a-flat major': 'g-sharp major',
    'd-sharp major': 'e-flat major', 'e-flat major': 'd-sharp major',
    'a-sharp major': 'b-flat major', 'b-flat major': 'a-sharp major',
    'e-sharp major': 'f major', 'f major': 'e-sharp major',
    'b-sharp major': 'c major', 'c major': 'b-sharp major',
}

def is_variant(s, o):
    s = s.lower().strip()
    o = o.lower().strip()
    
    if s == o:
        return False
        
    if s in god_pairs and god_pairs[s] == o:
        return "God Equivalent"
        
    if s in keys_pairs and keys_pairs[s] == o:
        return "Enharmonic Key Equivalent"
        
    if o == s + " smart interchange" or s == o + " smart interchange":
        return "Smart Interchange"
        
    def normalize(name):
        name = name.replace('í', 'i').replace('é', 'e').replace('á', 'a').replace('ó', 'o').replace('ú', 'u')
        name = name.replace('ñ', 'n')
        name = name.replace('tx', 'ch').replace('x', 'ch').replace('tz', 'z').replace('k', 'c')
        name = name.replace('v', 'b')
        name = name.replace('g', 'j')
        name = name.replace('z', 's')
        return name
        
    if normalize(s) == normalize(o):
        return "Spelling/Orthographic Variant"
        
    if (len(s) > len(o) and s.startswith(o)) or (len(o) > len(s) and o.startswith(s)):
        if len(s) - len(o) <= 15:
            return "Prefix/Shortened name"
            
    return None

with open("scratch/said_same_candidates.txt", "w", encoding="utf-8") as f:
    f.write("Scanning test set for 'said to be the same as' candidates...\n")
    count = 0
    for idx, row in test_df.iterrows():
        s = str(row['Subject'])
        o = str(row['Object'])
        pred = sub_df.loc[idx, 'Label']
        
        var_type = is_variant(s, o)
        if var_type:
            count += 1
            f.write(f"Row {idx:4d} | Sub: {s:35s} | Obj: {o:35s} | Current Pred: {pred:35s} | Type: {var_type}\n")
    f.write(f"Total candidates found: {count}\n")
print(f"Done. Saved to scratch/said_same_candidates.txt.")
