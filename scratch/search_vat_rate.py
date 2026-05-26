import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_df = pd.read_csv('dataset/test.csv')
sub_df = pd.read_csv('result/submission_final.csv')

# Load countries from country.csv in training
df_countries = pd.read_csv('dataset/Train_Set/country.csv')
countries = set(df_countries['Object'].unique()) | set(df_countries['Subject'].unique())

# Also load some common country names to be safe
common_countries = {'Saudi Arabia', 'Chad', 'Zambia', 'Kazakhstan', 'Central African Republic', 
                    'Uzbekistan', 'Slovakia', 'Austria', 'Morocco', 'Italy', 'Ethiopia', 
                    'Bolivia', 'Mongolia', 'Paraguay', 'Japan', 'Czech Republic', 'Argentina', 
                    'France', 'Germany', 'Spain', 'United Kingdom', 'Netherlands', 'Belgium', 
                    'Sweden', 'Norway', 'Finland', 'Denmark', 'Switzerland', 'Portugal', 'Greece', 
                    'Turkey', 'Canada', 'United States of America', 'Brazil', 'China', 'India'}

countries.update(common_countries)

for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    if s in countries:
        # Check if object is a number representing a VAT rate
        try:
            val = float(o)
            if 5.0 <= val <= 27.0:
                print(f"Row {idx:4d} | Subject: '{s}' | Object: '{o}' | Pred: '{pred}'")
        except ValueError:
            pass
