import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\dataset\test.csv'
sub_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'

subjects = {
    'Kilgarren castle, Cardiganshire',
    'The vale of Llangollen',
    'Pylle Priory, Pembrokeshire',
    'The Menai Bridge',
    'Isle of Anglesea near the Paris mine',
    'Cardiganshire: near Aberystwith',
    'S.W. view Milford town, Pembrokeshire',
    'View near the Usk at Crickhowell',
    'Crickhowel Castle',
    'Gateway Of Carnarvon Castle',
    "The Emperor's New Groove",
    'Peter Pan',
    'Beauty and the Beast',
    'Treasure Planet'
}

with open(test_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row['Subject'].strip()
        o = row['Object'].strip()
        if s in subjects:
            print(f"Candidate: {s} -> {o}")
