# scripts/generate_relevance.py

import os, json, re
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

# df = pd.read_parquet("/Users/vedanshsurjan/Downloads/gymnastics_output.parquet")
# docs = []
# # print(df.values)
# print(df.columns)
# print(df.shape)
# for _, row in df.iterrows():
#     docs.append({
#         "url": row["URL"],
#         "content": row["content"]
#     })
#
# with open("/Users/vedanshsurjan/Downloads/data.json", "w") as f:
#     json.dump({"response": {"docs": docs}}, f)

RESULT_DIR = "./result"
DATA_JSON  = "/Users/vedanshsurjan/Downloads/data2.json"

def extract_links(content):
    # Simple regex to extract URLs (you may want to improve this)
    return re.findall(r'https?://[^\s"\'>]+', content)

# Load docs
with open(DATA_JSON) as f:
    docs = json.load(f)["response"]["docs"]
    print(len(docs), "docs loaded")

url_to_doc = {d["url"]: d for d in docs}

# Build directed graph
G = nx.DiGraph()
for d in docs:
    src = d["url"]
    G.add_node(src)
    links = extract_links(d.get("content", ""))
    # print(links, " links for", src)
    for tgt in links:
        if tgt in url_to_doc:  # Only link to known URLs
            G.add_edge(src, tgt)

# Compute PageRank and HITS
page_rank = nx.pagerank(G)
hits_h, hits_a = nx.hits(G, max_iter=1000)

# Save scores
for name, scores in [("page_rank", page_rank), ("hits", hits_h)]:
    path = os.path.join(RESULT_DIR, f"{name}_scores.json")
    with open(path, "w") as f:
        json.dump(scores, f)

# Generate Vector Space Model
# Create TF-IDF vectors for all documents
print("Generating vector space model...")
corpus = [d.get("content", "") for d in docs]
urls = [d["url"] for d in docs]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(corpus)

# Create a dictionary to store vector space relevance scores
vector_space_scores = {}

# For each document, compute its similarity to all other documents
for i, url in enumerate(urls):
    # Get the document vector
    doc_vector = tfidf_matrix[i]

    # Compute cosine similarity with all other documents
    similarities = cosine_similarity(doc_vector, tfidf_matrix).flatten()

    # Create a dictionary of {url: similarity_score} for the top N most similar docs
    # Skip the document itself (which would have similarity 1.0)
    top_indices = np.argsort(similarities)[::-1][1:11]  # Top 10 most similar docs
    similar_docs = {urls[j]: float(similarities[j]) for j in top_indices if similarities[j] > 0}

    vector_space_scores[url] = similar_docs

# Save vector space scores
path = os.path.join(RESULT_DIR, "vector_space_scores.json")
with open(path, "w") as f:
    json.dump(vector_space_scores, f)

print("Wrote real page_rank_scores.json, hits_scores.json, and vector_space_scores.json")