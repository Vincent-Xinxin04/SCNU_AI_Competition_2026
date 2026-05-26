import pandas as pd

test_df = pd.read_csv("dataset/test.csv", encoding='utf-8-sig')
sub_df = pd.read_csv("result/submission_final.csv", encoding='utf-8-sig')

# Standard VAT rates for some countries
vat_map = {
    'Italy': [22, 22.0, '22', '22.0'],
    'Japan': [10, 10.0, '10', '10.0', 8, 8.0, '8', '8.0'],
    'Austria': [20, 20.0, '20', '20.0'],
    'Slovakia': [20, 20.0, '20', '20.0'],
    'Zambia': [16, 16.0, '16', '16.0'],
    'Saudi Arabia': [5, 5.0, '5', '5.0', 15, 15.0, '15', '15.0'],
    'Chad': [18, 18.0, '18', '18.0'],
    'Kazakhstan': [12, 12.0, '12', '12.0'],
    'Central African Republic': [19, 19.0, '19', '19.0'],
    'Uzbekistan': [20, 20.0, '20', '20.0'],
    'Morocco': [20, 20.0, '20', '20.0'],
    'Ethiopia': [15, 15.0, '15', '15.0'],
    'Bolivia': [13, 13.0, '13', '13.0'],
    'Mongolia': [10, 10.0, '10', '10.0'],
    'Paraguay': [10, 10.0, '10', '10.0'],
    'Czech Republic': [21, 21.0, '21', '21.0'],
    'Argentina': [21, 21.0, '21', '21.0'],
}

print("Searching test set for VAT-rate candidates...")
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    if s in vat_map:
        # Check if Object is in the list of VAT rates
        # Try to parse object as float/int
        val = None
        try:
            val = float(o)
        except ValueError:
            pass
        
        expected_vals = vat_map[s]
        is_match = False
        if o in expected_vals or (val is not None and any(abs(val - ev) < 1e-5 for ev in [5.0, 8.0, 10.0, 12.0, 13.0, 15.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0] if isinstance(ev, (int, float)))):
            # More specific check: does it match the country's specific VAT rate?
            if val is not None and any(abs(val - float(x)) < 1e-5 for x in expected_vals if str(x).replace('.','').isdigit()):
                is_match = True
        
        if is_match:
            print(f"Row {idx:4d} | Sub: {s} | Obj: {o} | Current Pred: {pred} | MATCH!")
        elif val is not None and val in [5.0, 8.0, 10.0, 12.0, 13.0, 15.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0]:
            print(f"Row {idx:4d} | Sub: {s} | Obj: {o} | Current Pred: {pred} | Country match, numeric Obj = {val}")
