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
        print(f"Error fetching {qid}: {e}")
        return None

# Fetch Ctiborův dub
entity = get_wikidata_entity("Q11773736")
if entity:
    print("Ctiborův dub claims:")
    claims = entity.get('claims', {})
    for prop, val_list in claims.items():
        for val in val_list:
            mainsnak = val.get('mainsnak', {})
            datavalue = mainsnak.get('datavalue', {})
            value = datavalue.get('value', {})
            if isinstance(value, dict) and 'id' in value:
                target_id = value['id']
                if target_id == 'Q15893266':
                    print(f"Property for 'former entity': {prop}")
                elif target_id == 'Q5':
                    print(f"Property for human (Q5): {prop}")
                # We can also check if the property label can be fetched
else:
    print("Could not fetch Ctiborův dub")
