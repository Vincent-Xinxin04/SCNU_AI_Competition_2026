import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

characters = ['Amelia', 'Hawkins', 'Scroop', 'Treasure Planet', 'Ranieri', 'Pablos', 'Hofstedt', 'Husband', 'Keane', 'Duncan', 'Ripa']

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    match = False
    for c in characters:
        if c in s or c in o:
            match = True
            break
            
    if match:
        print(f"Row {idx}: {s} -> {o} | Pred: {pred}")
