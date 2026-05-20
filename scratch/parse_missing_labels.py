import os

missing_labels = [
    'CJKV variant character', 'VAT-rate', 'World Health Organisation International Nonproprietary Name',
    'absolute magnitude', 'after a work by', 'anatomical location', 'art director', 'catalog',
    'coextensive with', 'commissioned by', 'compulsory education (maximum age)', 'connecting service',
    'contains settlement', 'continent', 'contributor to the creative work or subject', 'conversion to SI unit',
    'cover art by', 'designated as terrorist by', 'distributor', 'edition or translation of',
    'editor-in-chief', 'enemy of', 'facet of', 'family name identical to this given name', 'film editor',
    'genetic association', 'gross tonnage', 'has cause', 'has decorative pattern', 'head of state',
    'home port', 'home world', 'illiterate population', 'inflows', 'instance of', 'interested in',
    'languages spoken', 'located in the ecclesiastical territorial entity', 'location of first performance',
    'longitude of ascending node', 'lower flammable limit', 'lyrics by', 'maintained by', 'male population',
    'mean anomaly', 'name', 'narrator', 'next higher rank', 'nominal GDP per capita', 'notable work',
    'number of decimal digits', 'number of episodes', 'number of platform faces', 'number of spoilt votes',
    'official symbol', 'officially opened by', 'opposite of', 'orbital period',
    'organization directed from the office or person', 'parent astronomical body', 'patron saint',
    'payload mass', 'place of death', 'points goal scored by', 'powered by', 'presenter', 'producer',
    'recorded at', 'recording or performance of', 'referee', 'relative', 'rotation period',
    'service retirement', 'short name', 'sibling', 'spouse', 'statistical leader', 'subtitle', 'target',
    'teams classification by points', 'territory overlaps', 'time of spacecraft orbit decay', 'translator',
    'tributary', 'tussenvoegsel', 'unemployment rate', 'unmarried partner', 'urban population', 'uses',
    'victory', 'wavelength', 'young rider classification'
]

hits = []
with open(r'f:\github\SCNU_AI_Competition_2026\scratch\few_shot_hits.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        # Format: Subject 'X' had few-shot label 'Y' in Train. In Test it has 'Z' (Object: W)
        for label in missing_labels:
            if f"few-shot label '{label}'" in line:
                hits.append(line)

with open(r'f:\github\SCNU_AI_Competition_2026\scratch\missing_label_hits.txt', 'w', encoding='utf-8') as f:
    for h in hits:
        f.write(h + '\n')
