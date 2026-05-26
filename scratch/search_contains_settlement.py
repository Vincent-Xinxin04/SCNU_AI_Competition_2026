import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

# Load located in the administrative territorial entity.csv from training
df_loc = pd.read_csv('dataset/Train_Set/located in the administrative territorial entity.csv')

# Create a set of (Object, Subject) pairs from training
train_pairs = set(zip(df_loc['Subject'], df_loc['Object']))

# Search test set for reversed pairs
for idx, row in test_df.iterrows():
    s = row['Subject']
    o = row['Object']
    pred = sub_df.loc[idx, 'Label']
    
    # If the reversed pair (o, s) exists in training located-in relation
    if (o, s) in train_pairs:
        print(f"Row {idx:4d} | Subject: '{s}' (municipality/larger) | Object: '{o}' (village/smaller) | Pred: '{pred}'")
