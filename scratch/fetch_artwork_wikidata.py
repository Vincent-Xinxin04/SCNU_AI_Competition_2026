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
        print("error:", e)
        return None

entity = get_wikidata_entity("Q25917153")
if entity:
    claims = entity.get('claims', {})
    for prop, val_list in claims.items():
        for val in val_list:
            mainsnak = val.get('mainsnak', {})
            datavalue = mainsnak.get('datavalue', {})
            value = datavalue.get('value', {})
            print(f"Prop: {prop}, Value: {value}")
else:
    print("Could not fetch Q25917153")
