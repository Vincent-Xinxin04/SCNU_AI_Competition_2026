import os
import glob
import pandas as pd
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Helper to check types
def detect_type(val_str):
    val_str = str(val_str).strip()
    if not val_str:
        return 'empty'
    
    # Check date YYYY-MM-DD or YYYY-MM or YYYY (with optional timezone or BC)
    if re.match(r'^-?\d{1,4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?$', val_str):
        return 'date'
    if re.match(r'^-?\d{1,4}$', val_str) and len(val_str) == 4:
        return 'year_or_int'
    
    # Check float/int
    try:
        float(val_str)
        return 'number'
    except ValueError:
        pass
    
    return 'entity'

# Load train data and profile relations
train_files = glob.glob("dataset/Train_Set/*.csv")
relation_profiles = {}

for f in train_files:
    lbl = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Subject' in df.columns and 'Object' in df.columns:
        df = df[['Subject', 'Object']].dropna()
        types = [detect_type(o) for o in df['Object']]
        type_counts = pd.Series(types).value_counts()
        most_common_type = type_counts.index[0] if len(type_counts) > 0 else 'entity'
        relation_profiles[lbl] = {
            'common_type': most_common_type,
            'all_types': set(type_counts.index),
            'sample_count': len(df)
        }

# Load test and submission_reverted
test_df = pd.read_csv("dataset/test.csv")
sub_df = pd.read_csv("result/submission_reverted.csv")

violations = []
for idx, row in test_df.iterrows():
    s = str(row['Subject']).strip()
    o = str(row['Object']).strip()
    pred = sub_df.loc[idx, 'Label']
    
    o_type = detect_type(o)
    
    if pred in relation_profiles:
        profile = relation_profiles[pred]
        expected_type = profile['common_type']
        
        # Define compatibility
        compatible = False
        if expected_type == 'date':
            compatible = (o_type in ['date', 'year_or_int'])
        elif expected_type in ['number', 'year_or_int']:
            compatible = (o_type in ['number', 'year_or_int'])
        else: # entity
            compatible = (o_type == 'entity')
            
        if not compatible:
            violations.append({
                'row': idx,
                'Subject': s,
                'Object': o,
                'Object_Type': o_type,
                'Pred': pred,
                'Expected_Type': expected_type,
                'Sample_Count': profile['sample_count']
            })

print(f"Total violations found in reverted submission: {len(violations)}")
for v in violations:
    print(f"Row {v['row']:4d}: (Sub: {v['Subject']}, Obj: {v['Object']}) [{v['Object_Type']}] | Pred: {v['Pred']} (expects {v['Expected_Type']}, train count: {v['Sample_Count']})")
