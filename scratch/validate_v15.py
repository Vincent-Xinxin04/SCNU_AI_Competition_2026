import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('result/submission_final.csv')
print('Total rows:', len(df))
print('NaN rows:', df.isnull().sum().sum())
valid = set(open('dataset/labels.txt','r',encoding='utf-8').read().splitlines())
valid = {l for l in valid if l}
inv = set(df.Label.unique()) - valid
print('Invalid labels:', inv)
missing = [l for l in valid if df.Label.value_counts().get(l,0)==0]
print(f'Still missing: {len(missing)} labels')
print(missing[:15])

print('\nNew labels activated:')
for lbl in ['inflows', 'service retirement']:
    print(f'  {lbl}: {df.Label.value_counts().get(lbl, 0)} predictions')
