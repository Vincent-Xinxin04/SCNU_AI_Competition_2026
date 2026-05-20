import csv
ships = ['Cosco Africa', 'Cosco America', 'Silja Serenade', 'Gerner Maersk', 'Ciudad de Palma', 'USNS Kane', 'Maersk Nimes', 'HMS Beagle', 'RV Derinsu', 'Stickers Gat', 'Santa Isabel', 'Cruise Olympia', 'RV MTA Turkuaz', 'Cap Harvey', 'Stena Nautica', 'MS Norman Atlantic', 'Huckleberry Finn', 'SAS Protea', 'Sagitta']
with open(r'f:\github\SCNU_AI_Competition_2026\result\submission_0.747.csv', 'r', encoding='utf-8-sig') as f:
    for row in f:
        for s in ships:
            if s in row:
                print(row.strip())
