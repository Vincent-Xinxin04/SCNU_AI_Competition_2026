import csv

test_file = r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv'

generic_num_labels = ['numeric value', 'mass', 'area', 'length', 'width', 'height', 'duration', 'population', 'quantity']

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\numeric_predictions.txt', 'w', encoding='utf-8') as out_f:
    with open(test_file, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['Label'] in generic_num_labels:
                o = row['Object']
                if o.replace('.', '', 1).isdigit() or o.replace('-', '', 1).replace('.', '', 1).isdigit():
                    out_f.write(f"{row['Subject']} -> {row['Object']} (Pred: {row['Label']})\n")
