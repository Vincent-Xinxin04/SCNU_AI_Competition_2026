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

# Build Object -> Labels mapping
obj_labels = train_df.groupby('Object')['Label'].unique()
single_label_objs = obj_labels[obj_labels.apply(len) == 1].apply(lambda x: x[0]).to_dict()

print(f"Total unique objects in train: {len(obj_labels)}")
print(f"Objects with exactly one label in train: {len(single_label_objs)}")

# Load test and current predictions
test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_0.7791.csv")

mismatches = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    if o in single_label_objs:
        expected = single_label_objs[o]
        if pred != expected:
            mismatches.append({
                'row': idx,
                'Subject': s,
                'Object': o,
                'Pred': pred,
                'Expected': expected
            })

print(f"Number of test rows where Object exists in train with a single label, but prediction is different: {len(mismatches)}")
for m in mismatches[:20]:
    print(f"Row {m['row']}: (Sub: {m['Subject']}, Obj: {m['Object']}) | Pred: {m['Pred']} -> Expected: {m['Expected']}")
