import os
import glob
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load train data
train_files = glob.glob("dataset/Train_Set/*.csv")
train_records = []
for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        for idx, row in df.iterrows():
            s = str(row['Subject']).strip()
            o = str(row['Object']).strip()
            train_records.append({'Subject': s, 'Object': o, 'Label': lbl})
train_df = pd.DataFrame(train_records)

# 1. Search for objects
def search_object(obj_name):
    matches = train_df[train_df['Object'] == obj_name]
    print(f"\n--- Search for Object: {obj_name} ---")
    if len(matches) > 0:
        print(matches.head(10))
    else:
        # substring search
        sub_matches = train_df[train_df['Object'].str.contains(obj_name, case=False, na=False)]
        print(f"No exact matches. Substring matches ({len(sub_matches)}):")
        print(sub_matches.head(10))

search_object("solitary")
search_object("t1154233968")
search_object("lighter")
search_object("1")

# 2. Search for subjects/objects of 'fare zone' in train
print("\n--- Train samples for 'fare zone' ---")
print(train_df[train_df['Label'] == 'fare zone'])

# 3. Check Row 300, 56, etc.
test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

print("\n--- Test samples with Object = solitary ---")
for idx, row in test_df.iterrows():
    if str(row['Object']).strip() == 'solitary':
        print(f"Row {idx}: {row['Subject']}, {row['Object']} | Pred: {sub_df.loc[idx, 'Label']}")

# 4. Check Row 2189 in test
print("\n--- Row 2189 in test ---")
print(test_df.iloc[2189])
print("Prediction:", sub_df.iloc[2189])

# 5. Check Row 3608 in test
print("\n--- Row 3608 in test ---")
print(test_df.iloc[3608])
print("Prediction:", sub_df.iloc[3608])
