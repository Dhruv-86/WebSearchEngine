# scripts/index_data.py
import json
import pysolr
import nltk

nltk.download('stopwords', quiet=True)
# point to your local core
solr = pysolr.Solr('http://localhost:8983/solr/nutch', always_commit=True)

# load your output.json
with open('/Users/vedanshsurjan/Downloads/output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = []
# The data is a list of objects directly, not nested under 'response'/'docs'
for d in data:
    # Extract title from content if possible (first few words)
    title = ''
    if d.get('content'):
        # Use first 50 characters as a basic title if no explicit title
        title = d.get('content', '')[:50].strip()
        if title:
            title += '...'

    docs.append({
        'id':      d['url'],              # unique key
        'url':     d['url'],              # field our code uses for clustering & display
        'title':   title,                 # generate a basic title from content
        'content': d.get('content', ''),  # field our Flask search uses when querying
        'links':   d.get('links', []),    # store links for potential use
        'depth':   d.get('depth', 0)      # store depth information
    })

# wipe out any previous docs so we don't get duplicates
solr.delete(q='*:*')

# push all at once
solr.add(docs)
print(f"Indexed {len(docs)} documents.")
