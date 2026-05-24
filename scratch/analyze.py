import os
import pandas as pd
import glob

# Load labels
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

print(f"Total labels in labels.txt: {len(labels)}")

# Load submission
sub = pd.read_csv("result/submission_0.7791.csv")
print(f"Submission shape: {sub.shape}")
print(f"Unique labels predicted in submission: {sub['Label'].nunique()}")

# Count labels predicted
sub_counts = sub['Label'].value_counts()
print("Top 10 predicted labels:")
print(sub_counts.head(10))
print("Bottom 10 predicted labels:")
print(sub_counts.tail(10))

# Check if all predicted labels are in labels.txt
invalid_preds = sub[~sub['Label'].isin(labels)]
print(f"Number of invalid predictions: {len(invalid_preds)}")
if len(invalid_preds) > 0:
    print(invalid_preds.head())

# Load train set
train_files = glob.glob("dataset/Train_Set/*.csv")
print(f"Number of CSV files in Train_Set: {len(train_files)}")

# Check if there are training files whose name is not in labels.txt
train_relations = [os.path.basename(f)[:-4] for f in train_files]
not_in_labels = [r for r in train_relations if r not in labels]
print(f"Train relations not in labels.txt: {len(not_in_labels)}")
if len(not_in_labels) > 0:
    print(not_in_labels[:5])
