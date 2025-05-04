# scripts/generate_relevance.py
import os, json

RESULT_DIR = "./result"
DATA_JSON  = "/Users/vedanshsurjan/Downloads/data2.json"

# Load your docs to get URLs
with open(DATA_JSON) as f:
    docs = json.load(f)["response"]["docs"]
urls = [d["url"] for d in docs]

# Give every URL a uniform score of 1.0
scores = {u: 1.0 for u in urls}

# Write both page_rank and hits
for name in ("page_rank", "hits"):
    path = os.path.join(RESULT_DIR, f"{name}_scores.json")
    with open(path, "w") as f:
        json.dump(scores, f)

print("Wrote page_rank_scores.json and hits_scores.json")
