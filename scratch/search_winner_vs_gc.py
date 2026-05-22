import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'

def search_race(race_name):
    print(f"\nSearching for '{race_name}':")
    for filename in os.listdir(train_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(train_dir, filename)
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2 and race_name.lower() in row[0].lower():
                        print(f"  {filename[:-4]}: {row}")

search_race("Tour de France")
search_race("Giro d'Italia")
search_race("Vuelta a España")
search_race("Ster ZLM Toer")
search_race("Volta a Portugal")
