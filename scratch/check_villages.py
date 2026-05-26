import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv", encoding='utf-8-sig')
sub_df = pd.read_csv("result/submission_final.csv", encoding='utf-8-sig')

# Typical village names or shapes
# Let's check all rows in the test set where the Object is '0' or '0.0' or similar small integers
zero_rows = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    # Check if object is zero or numeric
    if o in ['0', '0.0', '1', '1.0', '2', '2.0', '3', '3.0', '4', '4.0', '5', '5.0']:
        # Let's check if the prediction is one of the population / household / literate population labels
        if pred in ['number of households', 'literate population', 'population', 'male population']:
            zero_rows.append((idx, s, o, pred))

print(f"Total zero/small value rows with population/household labels: {len(zero_rows)}")
for idx, s, o, pred in zero_rows:
    print(f"Row {idx:4d} | Sub: {s:35s} | Obj: {o:5s} | Current Pred: {pred}")
