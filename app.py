import os
import json
import random
from urllib.parse import urlparse

from flask import Flask, request, jsonify
from flask_cors import CORS
import pysolr

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

from ClusterManager import ClusterManager
import qe

app = Flask(__name__)
CORS(app)

# NLTK setup
nltk.download('stopwords', quiet=True)
STOP      = set(stopwords.words("english"))
TOKENIZER = TreebankWordTokenizer()

def clean_query(q: str) -> str:
    tokens = TOKENIZER.tokenize(q.lower())
    return " ".join(w for w in tokens if w.isalnum() and w not in STOP)

# Solr client
SOLR_URL = os.getenv("SOLR_URL", "http://localhost:8983/solr/nutch")
solr     = pysolr.Solr(SOLR_URL, always_commit=True)

# Clustering helper
RESULTS_DIR = os.getenv("RESULTS_DIR", "./result")
cluster_mgr = ClusterManager(results_dir=RESULTS_DIR)

def limit_domain(docs: list, max_per: int = 4) -> list:
    seen, out = {}, []
    for d in docs:
        host = urlparse(d["url"]).netloc
        if seen.get(host, 0) < max_per:
            out.append(d)
            seen[host] = seen.get(host, 0) + 1
    return out

def fetch_solr(q: str, target: int = 50) -> list:
    rows = 100
    while True:
        resp = solr.search(q, **{"defType": "edismax", "qf": "title^3 content", "wt": "json", "rows": rows})
        docs = []
        for r in resp:
            d = dict(r)

            raw_url = r.get("url") or r.get("url_s") or r.get("id") or ""
            url = raw_url[0] if isinstance(raw_url, list) else raw_url
            d["url"] = url

            raw_content = r.get("content") or r.get("content_t", "")
            content = raw_content[0] if isinstance(raw_content, list) else raw_content
            d["content"] = content

            # Provide 'digest' for QE routines
            d["digest"] = url

            docs.append(d)

        docs = limit_domain(docs)
        if len(docs) >= target or rows > 10000:
            head, tail = docs[:2], docs[2:]
            random.shuffle(tail)
            return head + tail
        rows *= 2

@app.route("/ping")
def ping():
    return "pong"

@app.route("/api/v1/search", methods=["GET"])
def search_endpoint():
    print(">>> /api/v1/search called with args:", request.args)

    raw  = request.args.get("query", "")
    mode = request.args.get("type", "")
    if not raw or not mode:
        return jsonify({"error": "provide both query and type"}), 400

    # 1) Preprocess & initial fetch
    cq     = clean_query(raw)
    solr_q = f'content:"{cq}"'
    results = fetch_solr(solr_q)

    # 2) Relevance re-ranking
    if mode in ("page_rank", "hits"):
        scores_path = os.path.join(RESULTS_DIR, f"{mode}_scores.json")
        with open(scores_path) as f:
            scores = json.load(f)
        results = sorted(
            results,
            key=lambda d: scores.get(d["url"], 0),
            reverse=True
        )

    # 3) Clustering
    if mode == "flat_clustering":
        results = cluster_mgr.cluster_flat(raw, results)
    elif mode == "single_hac":
        results = cluster_mgr.cluster_single(raw, results)
    elif mode == "average_hac":
        results = cluster_mgr.cluster_average(raw, results)

    # 4) Query Expansion with robust fallback
    resp_q = raw

    if mode == "rocchio":
        ext = qe.rocchio_expand(raw, results)
        newr = fetch_solr(f'content:"{ext}"')
        results = newr or results
        resp_q  = ext or raw

    elif mode == "association_qe":
        try:
            ext = qe.association_main(raw, results)
        except Exception as e:
            print("association_qe error:", e)
            ext = ""
        if ext:
            newr = fetch_solr(f'content:"{ext}"')
            results = newr or results
            resp_q = ext
        # if ext empty, leave results & resp_q==raw

    elif mode == "metric_qe":
        ext = qe.metric_cluster_main(raw, results)
        newr = fetch_solr(f'content:"{ext}"')
        results = newr or results
        resp_q  = ext or raw

    elif mode == "scalar_qe":
        ext = qe.scalar_main(raw, results)
        newr = fetch_solr(f'content:"{ext}"')
        results = newr or results
        resp_q  = ext or raw

    return jsonify({
        "query": resp_q,
        "query_results": results
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
