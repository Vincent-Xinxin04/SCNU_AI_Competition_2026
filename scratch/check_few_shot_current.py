import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('result/submission_final.csv', encoding='utf-8-sig')

fixes = [
    ('Kurdistan Workers\' Party', 'Turkey'),
    ('Parramatta Eels', 'National Rugby League'),
    ('Neubrügg', '92.2'),
    ('Wests Tigers', 'National Rugby League'),
    ('Penrith Panthers', 'National Rugby League'),
    ('Derive', 'MS-DOS'),
    ('Bahçeli Kayaları', 'Turkey'),
    ('2011 Australian Grand Prix', 'Felipe Massa'),
    ('RV MTA Turkuaz', 'Turkey'),
    ('Fenerbahçe Women\'s Volleyball', 'Turkey'),
    ('Manly-Warringah Sea Eagles', 'National Rugby League'),
    ('İller Bankası Women\'s Volleyball', 'Turkey'),
    ('2009 Copa Sony Ericsson Colsanitas', '2009-02-16'),
    ('South Sydney Rabbitohs', 'National Rugby League'),
    ('Bornova Ice Sports Hall', 'Turkey'),
    ('RV Derinsu', 'Turkey'),
    ('Beşiktaş Women\'s Volleyball Team', 'Turkey'),
    ('anthropology', 'anthropologist'),
    ('Sarıyer Belediyesi SK', 'Turkey'),
    ('Vaillant Arena', 'Graubünden'),
    ('North Queensland Cowboys', 'National Rugby League'),
    ('New Zealand Warriors', 'National Rugby League'),
    ('Bosphorus Istanbul Cup 2018 - senior ice dance rhythm dance', 'Turkey'),
    ('St. George Illawarra Dragons', 'National Rugby League'),
    ('Newcastle Knights', 'National Rugby League')
]

for s, o in fixes:
    matches = df[(df['Subject'] == s) & (df['Object'] == o)]
    for idx, row in matches.iterrows():
        print(f"Row {idx:4d} | {s} -> {o} | Current: {row['Label']}")
