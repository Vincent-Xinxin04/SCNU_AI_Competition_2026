import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

def find_test_matches(pattern):
    matches = test_df[test_df['Subject'].str.contains(pattern, case=False, na=False) | test_df['Object'].str.contains(pattern, case=False, na=False)]
    print(f"\n--- Test Matches for '{pattern}' ---")
    for idx, row in matches.iterrows():
         print(f"Row {idx}: {row['Subject']}, {row['Object']} | Pred: {sub_df.loc[idx, 'Label']}")

find_test_matches("former entity")
find_test_matches("solitary")
find_test_matches("lighter")
find_test_matches("t1154233968")
