import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv", encoding='utf-8-sig')
sub_df = pd.read_csv("result/submission_final.csv", encoding='utf-8-sig')

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

with open("scratch/edit_distance_candidates.txt", "w", encoding="utf-8") as f:
    f.write("Scanning for small edit distance pairs in test set...\n")
    count = 0
    for idx, row in test_df.iterrows():
        s = str(row['Subject'])
        o = str(row['Object'])
        pred = sub_df.loc[idx, 'Label']
        
        if s == o:
            continue
            
        dist = edit_distance(s.lower(), o.lower())
        if dist <= 2:
            count += 1
            f.write(f"Row {idx:4d} | Sub: {s:35s} | Obj: {o:35s} | Dist: {dist} | Current Pred: {pred}\n")
    f.write(f"Total small edit distance pairs: {count}\n")
print("Done. Saved to scratch/edit_distance_candidates.txt.")
