import json

with open("gen3_moves.json", "r") as f:
    moves = json.load(f)

print(f"Total moves found: {len(moves)}")
categories = {}
for m in moves:
    cat = m['category']
    categories[cat] = categories.get(cat, 0) + 1
print("Categories:", categories)
