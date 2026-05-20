import os, csv
train_dir = r'f:\github\SCNU_AI_Competition_2026\dataset\Train_Set'
for f_name in os.listdir(train_dir):
    if f_name.endswith('.csv'):
        with open(os.path.join(train_dir, f_name), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[0] in ['Zambia', 'United States of America', 'India']:
                    try:
                        float(row[1])
                        print(f"{f_name[:-4]}: {row[0]} -> {row[1]}")
                    except ValueError:
                        pass
