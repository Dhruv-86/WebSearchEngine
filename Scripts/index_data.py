# scripts/index_data.py
import json
import pysolr

# point to your local core
solr = pysolr.Solr('http://localhost:8983/solr/nutch', always_commit=True)

# load your data.json
with open('/Users/dhruv/Downloads/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = []
for d in data.get('response', {}).get('docs', []):
    docs.append({
        'id':      d['url'],              # unique key
        'url':     d['url'],              # field our code uses for clustering & display
        'title':   d.get('title', ''),    # optional, if you ever want to show titles
        'content': d.get('content', '')   # field our Flask search uses when querying
    })

# wipe out any previous docs so we don’t get duplicates
solr.delete(q='*:*')

# push all at once
solr.add(docs)
print(f"Indexed {len(docs)} documents.")
