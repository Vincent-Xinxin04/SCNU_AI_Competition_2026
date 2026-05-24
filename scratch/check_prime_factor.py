import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    df = pd.read_csv("dataset/Train_Set/prime factor.csv", low_memory=False, encoding='utf-8-sig')
    print(df[df['Subject'].astype(str) == '1201'])
except Exception as e:
    print(e)
