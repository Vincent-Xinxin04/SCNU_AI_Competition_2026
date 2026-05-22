import pandas as pd
import sys

# Ensure UTF-8 printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('result/submission_final.csv', encoding='utf-8-sig')
pairs = [
    ('Lake Bolsena', 'Italy'),
    ('Lake Bracciano', 'Italy'),
    ('Alphonse Allard Fonds', 'fonds'),
    ('2016 Volta a Portugal', 'José Gonçalves'),
    ('Draw the Line', 'Al Hirschfeld'),
    ('Milano Porta Garibaldi', '1'),
    ("2017 Giro d'Italia, Stage 4", 'Dimension Data 2017'),
    ("2017 Giro d'Italia, Stage 9", 'Dimension Data 2017'),
    ('Appliqué inv. 848', 'appliqué')
]
for s, o in pairs:
    match = df[(df['Subject'] == s) & (df['Object'] == o)]
    if len(match) > 0:
        print(f'{s} -> {o}: {match.iloc[0]["Label"]}')
    else:
        print(f'{s} -> {o}: NOT FOUND')
