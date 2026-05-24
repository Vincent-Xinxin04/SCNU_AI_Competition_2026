import os
import pandas as pd
import glob
import re

# Load training data
train_files = glob.glob("dataset/Train_Set/*.csv")
train_data = []
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        df['Label'] = lbl
        train_data.append(df)
train_df = pd.concat(train_data, ignore_index=True)

# Build a mapping of (Subject, Object) -> Label from train
train_map = {}
for idx, row in train_df.iterrows():
    s, o, l = str(row['Subject']).strip(), str(row['Object']).strip(), row['Label']
    train_map[(s, o)] = l

# Load test and submission
test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

# 1. Check exact matches
exact_matches = 0
exact_mismatch = 0
mismatch_details = []
for idx, row in test_df.iterrows():
    s, o = str(row['Subject']).strip(), str(row['Object']).strip()
    pred_l = sub_df.loc[idx, 'Label']
    if (s, o) in train_map:
        true_l = train_map[(s, o)]
        exact_matches += 1
        if pred_l != true_l:
            exact_mismatch += 1
            mismatch_details.append((idx, s, o, pred_l, true_l))

print(f"Total test samples: {len(test_df)}")
print(f"Test samples found exactly in Train: {exact_matches}")
print(f"Mismatches on exact train samples: {exact_mismatch}")
if exact_mismatch > 0:
    print("Example exact mismatches:")
    for detail in mismatch_details[:10]:
        print(f"Row {detail[0]}: ({detail[1]}, {detail[2]}) -> Pred: {detail[3]}, True: {detail[4]}")
