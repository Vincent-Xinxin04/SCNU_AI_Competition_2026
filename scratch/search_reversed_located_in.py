import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

keywords = ('county', 'district', 'municipality', 'voivodeship', 'commune', 'canton', 'department', 'province', 'region')

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    # Check if subject contains a larger entity keyword
    s_lower = s.lower()
    o_lower = o.lower()
    
    if any(k in s_lower for k in keywords) and not any(k in o_lower for k in keywords):
        if pred == 'located in the administrative territorial entity':
            print(f"Row {idx:4d} | Subject: '{s}' | Object: '{o}' | Pred: '{pred}'")
