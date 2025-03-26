import json
import difflib
from collections import defaultdict

with open("./dataset/plantnet_300K/plantnet300K_species_names.json", "r", encoding="utf-8") as f:
    class_map = json.load(f)

entries = [(cid, name.strip()) for cid, name in class_map.items()]

name_to_ids = defaultdict(list)
for cid, name in entries:
    name_to_ids[name].append(cid)

merge_map = {}

for name, ids in name_to_ids.items():
    if len(ids) > 1:
        primary_id = ids[0]
        for duplicate_id in ids[1:]:
            merge_map[duplicate_id] = primary_id


use_fuzzy_matching = True
fuzzy_threshold = 0.97

if use_fuzzy_matching:
    checked = set()
    for i, (cid1, name1) in enumerate(entries):
        for cid2, name2 in entries[i+1:]:
            if cid1 == cid2 or cid1 in merge_map or cid2 in merge_map:
                continue
            sim = difflib.SequenceMatcher(None, name1, name2).ratio()
            if sim > fuzzy_threshold:
                merge_map[cid2] = cid1
                checked.add((cid1, cid2))

with open("./src/ai/evaluation/merge_map.json", "w", encoding="utf-8") as f:
    json.dump(merge_map, f, indent=4, ensure_ascii=False)

print(f"Merge Map mit {len(merge_map)} Einträgen gespeichert als merge_map.json")
