import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

# Let's collect all subjects from service entry and service retirement
df_entry = pd.read_csv('dataset/Train_Set/service entry.csv')
df_ret = pd.read_csv('dataset/Train_Set/service retirement.csv')

subjects = set(df_entry['Subject'].unique()) | set(df_ret['Subject'].unique())

for idx, row in test_df.iterrows():
    if row['Subject'] in subjects:
        print(f"Row {idx:4d} | Subject: '{row['Subject']}' | Object: '{row['Object']}' | Pred: '{sub_df.loc[idx, 'Label']}'")
