import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_wikidata_entity(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data['entities'][qid]
    except Exception as e:
        return None

# Let's check a few trees:
# Chodovský buk (Q11739268)
# Tomeškův dub (Q12059397)
# Kaasův buk (Q11732912)
qids = ["Q11739268", "Q12059397", "Q11732912"]

for qid in qids:
    entity = get_wikidata_entity(qid)
    if entity:
        print(f"\nEntity {qid} ({entity.get('labels', {}).get('en', {}).get('value', 'No EN Label')}):")
        claims = entity.get('claims', {})
        for prop, val_list in claims.items():
            for val in val_list:
                mainsnak = val.get('mainsnak', {})
                datavalue = mainsnak.get('datavalue', {})
                value = datavalue.get('value', {})
                if isinstance(value, dict) and 'id' in value:
                    target_id = value['id']
                    # We can fetch the label of target_id or print it
                    print(f"  {prop} -> {target_id}")
