import os, csv
from collections import defaultdict

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
test_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

zero_preds = ['illustrator', 'contributor to the creative work or subject', 'defined daily dose',
              'editor-in-chief', 'enemy of', 'located in the ecclesiastical territorial entity',
              'location of first performance', 'number of episodes', 'teams classification by points',
              'wavelength', 'World Health Organisation International Nonproprietary Name']

train_subjects = defaultdict(list)
for file in os.listdir(train_dir):
    label = file[:-4]
    if label in zero_preds:
        with open(os.path.join(train_dir, file), 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    train_subjects[label].append(parts[0])

print('Checking test set for subjects that had 0-prediction few-shot labels in train:')
with open(test_path, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        s, o, l = row['Subject'], row['Object'], row['Label']
        for label, subjects in train_subjects.items():
            if s in subjects:
                print(f"{label}: Subject '{s}' -> Object '{o}' (Pred: {l})")
