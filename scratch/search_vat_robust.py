import pandas as pd

test_df = pd.read_csv("dataset/test.csv", encoding='utf-8-sig')
sub_df = pd.read_csv("result/submission_final.csv", encoding='utf-8-sig')

# A comprehensive list of countries and their standard VAT rates (as of 2026/recent years)
# If a country is in test set and its object matches the VAT rate, we print it.
country_vat = {
    'afghanistan': 0, 'albania': 20, 'algeria': 19, 'andorra': 4.5, 'angola': 14, 'argentina': 21, 'armenia': 20,
    'australia': 10, 'austria': 20, 'azerbaijan': 18, 'bahamas': 12, 'bahrain': 10, 'bangladesh': 15, 'barbados': 17,
    'belarus': 20, 'belgium': 21, 'belize': 12.5, 'benin': 18, 'bhutan': 5, 'bolivia': 13, 'bosnia and herzegovina': 17,
    'botswana': 14, 'brazil': 17, 'brunei': 0, 'bulgaria': 20, 'burkina faso': 18, 'burundi': 18, 'cambodia': 10,
    'cameroon': 19.25, 'canada': 5, 'cape verde': 15, 'central african republic': 19, 'chad': 18, 'chile': 19,
    'china': 13, 'colombia': 19, 'comoros': 10, 'congo': 18.9, 'costa rica': 13, 'croatia': 25, 'cuba': 10,
    'cyprus': 19, 'czech republic': 21, 'denmark': 25, 'djibouti': 33, 'dominica': 15, 'dominican republic': 18,
    'ecuador': 12, 'egypt': 14, 'el salvador': 13, 'equatorial guinea': 15, 'eritrea': 4, 'estonia': 22,
    'eswatini': 15, 'ethiopia': 15, 'fiji': 9, 'finland': 24, 'france': 20, 'gabon': 18, 'gambia': 15,
    'georgia': 18, 'germany': 19, 'ghana': 15, 'greece': 24, 'grenada': 15, 'guatemala': 12, 'guinea': 18,
    'guyana': 14, 'haiti': 10, 'honduras': 15, 'hungary': 27, 'iceland': 24, 'india': 18, 'indonesia': 11,
    'iran': 9, 'iraq': 0, 'ireland': 23, 'israel': 17, 'italy': 22, 'jamaica': 15, 'japan': 10, 'jordan': 16,
    'kazakhstan': 12, 'kenya': 16, 'kiribati': 0, 'kosovo': 18, 'kuwait': 0, 'kyrgyzstan': 12, 'laos': 10,
    'latvia': 21, 'lebanon': 11, 'lesotho': 15, 'liberia': 7, 'libya': 0, 'liechtenstein': 7.7, 'lithuania': 21,
    'luxembourg': 17, 'madagascar': 20, 'malawi': 16.5, 'malaysia': 6, 'maldives': 8, 'mali': 18, 'malta': 18,
    'mauritania': 16, 'mauritius': 15, 'mexico': 16, 'micronesia': 0, 'moldova': 20, 'monaco': 20, 'mongolia': 10,
    'montenegro': 21, 'morocco': 20, 'mozambique': 16, 'myanmar': 5, 'namibia': 15, 'nauru': 0, 'nepal': 13,
    'netherlands': 21, 'new zealand': 15, 'nicaragua': 15, 'niger': 19, 'nigeria': 7.5, 'north korea': 0,
    'north macedonia': 18, 'norway': 25, 'oman': 5, 'pakistan': 18, 'palau': 0, 'panama': 7, 'papua new guinea': 10,
    'paraguay': 10, 'peru': 18, 'philippines': 12, 'poland': 23, 'portugal': 23, 'qatar': 0, 'romania': 19,
    'russia': 20, 'rwanda': 18, 'saint kitts and nevis': 17, 'saint lucia': 12.5, 'saint vincent and the grenadines': 16,
    'samoa': 15, 'san marino': 0, 'sao tome and principe': 7.5, 'saudi arabia': 15, 'senegal': 18, 'serbia': 20,
    'seychelles': 15, 'sierra leone': 15, 'singapore': 8, 'slovakia': 20, 'slovenia': 22, 'solomon islands': 10,
    'somalia': 0, 'south africa': 15, 'south korea': 10, 'south sudan': 18, 'spain': 21, 'sri lanka': 15,
    'sudan': 17, 'suriname': 10, 'sweden': 25, 'switzerland': 7.7, 'syria': 0, 'taiwan': 5, 'tajikistan': 15,
    'tanzania': 18, 'thailand': 7, 'timor-leste': 2.5, 'togo': 18, 'tonga': 15, 'trinidad and tobago': 12.5,
    'tunisia': 19, 'turkey': 20, 'turkmenistan': 15, 'tuvalu': 0, 'uganda': 18, 'ukraine': 20, 'united arab emirates': 5,
    'united kingdom': 20, 'united states': 0, 'uruguay': 22, 'uzbekistan': 12, 'vanuatu': 12.5, 'venezuela': 16,
    'vietnam': 10, 'yemen': 5, 'zambia': 16, 'zimbabwe': 15
}

matches = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip().lower()
    o = str(row['Object']).strip()
    
    if s in country_vat:
        try:
            val = float(o)
            # Standard rate
            expected_rate = country_vat[s]
            if abs(val - expected_rate) < 1e-5:
                matches.append((idx, row['Subject'], o, sub_df.loc[idx, 'Label'], f"Standard VAT for {row['Subject']} is {expected_rate}"))
        except ValueError:
            pass

print(f"Total VAT-rate candidates matching standard rate: {len(matches)}")
for idx, s, o, pred, reason in matches:
    print(f"Row {idx:4d} | Sub: {s} | Obj: {o} | Current Pred: {pred} | Reason: {reason}")
