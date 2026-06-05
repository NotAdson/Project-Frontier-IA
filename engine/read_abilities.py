import json

with open("gen3_abilities.json", "r") as f:
    abilities = json.load(f)

print(f"Total abilities found: {len(abilities)}")
for a in sorted(abilities, key=lambda x: x['id']):
    print(f"- {a['name']} ({a['id']}): {a['desc']}")
