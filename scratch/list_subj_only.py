import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('scratch/rare_label_candidates.csv')
subj_only = df[(df['Current_Pred'] != df['Rare_Label']) & (df['Match_Type'] == 'Subject-only Match')]
print('Number of Subject-only rare label matches:', len(subj_only))
for idx, r in subj_only.iterrows():
    print(f"Row {r['row']:4d} | ({r['Subject']} -> {r['Object']}) | Current: '{r['Current_Pred']}' | Rare: '{r['Rare_Label']}' | TrainCount: {r['Train_Count']}")
