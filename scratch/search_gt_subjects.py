import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

df_gt = pd.read_csv('dataset/Train_Set/gross tonnage.csv')
gt_subjects = set(df_gt['Subject'].unique())

for idx, row in test_df.iterrows():
    if row['Subject'] in gt_subjects:
        print(f"Row {idx:4d} | Subject: '{row['Subject']}' | Object: '{row['Object']}' | Pred: '{sub_df.loc[idx, 'Label']}'")
