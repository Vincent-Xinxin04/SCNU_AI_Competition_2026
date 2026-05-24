import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load labels
with open("dataset/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

corrections = {
    # 1. Exact matches
    344: 'location',
    1146: 'number of households',
    1172: 'number of households',
    1349: 'number of households',
    
    # 2. Tree solitary -> instance of
    54: 'instance of',
    300: 'instance of',
    373: 'instance of',
    987: 'instance of',
    1045: 'instance of',
    1054: 'instance of',
    1125: 'instance of',
    1740: 'instance of',
    2040: 'instance of',
    2519: 'instance of',
    2756: 'instance of',
    3401: 'instance of',
    3654: 'instance of',
    3670: 'instance of',
    
    # 3. Treasure Planet animators
    366: 'contributor to the creative work or subject',
    2675: 'contributor to the creative work or subject',
    
    # 4. Chinese characters
    784: 'CJKV variant character',
    1157: 'CJKV variant character',
    3136: 'instance of',
    
    # 5. Station number of platform faces
    3464: 'number of platform faces',
    
    # 6. Arson weapons & events
    3608: 'armament',
    223: 'instance of',
    359: 'instance of',
    374: 'instance of',
    1399: 'instance of',
    
    # 7. Number of decimal digits
    1836: 'number of decimal digits',
    1929: 'number of decimal digits',
    1937: 'number of decimal digits',
    1979: 'number of decimal digits',
    3457: 'number of decimal digits',
    4060: 'number of decimal digits'
}

# Verify all target labels are valid
invalid_labels = [l for l in corrections.values() if l not in labels]
print(f"Invalid labels in corrections: {invalid_labels}")

# Print the corrections
print(f"Total corrections planned: {len(corrections)}")
for r, l in sorted(corrections.items()):
    print(f"Row {r}: {l}")
