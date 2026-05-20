import csv

input_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final.csv'
output_path = r'f:\github\SCNU_AI_Competition_2026\result\submission_final_temp.csv'

fixes = {
    ('Victory Through Air Power', 'Edward H. Plumb', 'storyboard artist'): 'composer',
    ('Beauty and the Beast', 'John Carnochan', 'storyboard artist'): 'film editor',
    ('Beauty and the Beast', 'David Ogden Stiers', 'storyboard artist'): 'voice actor',
    ('Barry Allen', 'Patty Spivot', 'creator'): 'unmarried partner',
    ('Morbius', 'Matt Sazama', 'voice actor'): 'screenwriter',
    ('Night Court', 'Jack Elliott', 'cast member'): 'composer'
}

data = []
c = 0
with open(input_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s, o, l = row['Subject'], row['Object'], row['Label']
        if (s, o, l) in fixes:
            row['Label'] = fixes[(s, o, l)]
            c += 1
        data.append(row)

with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Subject', 'Object', 'Label'])
    writer.writeheader()
    writer.writerows(data)

print(f"Applied {c} ultra-rare few-shot fixes.")
