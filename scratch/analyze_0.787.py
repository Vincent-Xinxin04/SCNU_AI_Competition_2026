import os
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

sub_file = "result/submission_0.787.csv"
if not os.path.exists(sub_file):
    print(f"Error: {sub_file} does not exist!")
    sys.exit(1)

sub = pd.read_csv(sub_file, encoding='utf-8-sig')
print("Total rows in submission:", len(sub))

# Get predicted labels count
pred_counts = sub['Label'].value_counts()
print("Unique labels predicted:", len(pred_counts))

# Load labels.txt
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]
print("Total labels in labels.txt:", len(labels))

missing_labels = [l for l in labels if l not in pred_counts]
print("Missing labels count:", len(missing_labels))

# Load all training data to see which of the missing labels have training data
train_dir = "dataset/Train_Set"
missing_with_train = []
for ml in missing_labels:
    filename = ml.replace('/', '_').replace(':', '_') + '.csv'
    filepath = os.path.join(train_dir, filename)
    train_count = 0
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            train_count = len(df)
        except Exception:
            pass
    missing_with_train.append((ml, train_count))

missing_with_train = sorted(missing_with_train, key=lambda x: x[1], reverse=True)
print("\nMissing labels and their train counts:")
for idx, (ml, tc) in enumerate(missing_with_train):
    print(f"{idx+1:2d}. {ml:50s} : train_count = {tc}")
