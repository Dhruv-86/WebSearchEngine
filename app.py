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
            # print(r.get("content"))
            raw_url = r.get("url") or r.get("url_s") or r.get("id") or ""
            url = raw_url[0] if isinstance(raw_url, list) else raw_url
            d["url"] = url

            raw_content = r.get("content") or r.get("content_t", "")
            content = raw_content[0] if isinstance(raw_content, list) else raw_content
            d["content"] = content

            # Provide 'digest' for QE routines
            d["digest"] = url

            docs.append(d)
        # print(f"Searching for query {q} returned {len(docs)} results, trying again with rows={rows}...")
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

    # Required parameters
    raw = request.args.get("query", "")
    if not raw:
        return jsonify({"error": "query parameter is required"}), 400

    # Optional parameters with defaults
    relevance_model = request.args.get("relevance", "default")  # default, page_rank, hits
    clustering_method = request.args.get("clustering", "none")  # none, flat_clustering, single_hac, average_hac
    expansion_method = request.args.get("expansion", "none")    # none, rocchio, association_qe, metric_qe, scalar_qe
    use_hybrid = request.args.get("hybrid", "true").lower() == "true"  # Whether to use hybrid scoring with vector space

    # 1) Preprocess & initial fetch
    cq = clean_query(raw)
    solr_q = f'content:"{cq}"'
    results = fetch_solr(solr_q)

    # For debugging
    print(f"Initial fetch returned {len(results)} results")

    # 2) Apply relevance re-ranking if specified
    if relevance_model in ("page_rank", "hits"):
        try:
            # Load the primary relevance scores (PageRank or HITS)
            scores_path = os.path.join(RESULTS_DIR, f"{relevance_model}_scores.json")
            with open(scores_path) as f:
                primary_scores = json.load(f)

            # If hybrid scoring is enabled, also load vector space scores
            if use_hybrid:
                try:
                    vs_scores_path = os.path.join(RESULTS_DIR, "vector_space_scores.json")
                    with open(vs_scores_path) as f:
                        vector_space_scores = json.load(f)

                    # Get URLs of top documents from initial results
                    top_docs = [doc["url"] for doc in results[:10] if "url" in doc]  # Adjust number as needed

                    # Create a scoring dictionary
                    doc_scores = {}

                    # Initialize scores with a small value
                    for doc in results:
                        if "url" in doc:
                            doc_scores[doc["url"]] = 0.0001

                    # Apply scoring based on the relevance model
                    if relevance_model == "page_rank":
                        print("Page Rank with Vector Space")

                        # For each document in our results
                        for doc in results:
                            if "url" in doc:
                                doc_id = doc["url"]
                                # Start with 70% of the current score
                                doc_scores[doc_id] = 0.7 * doc_scores.get(doc_id, 0)
                                # Add 40% of the PageRank score
                                doc_scores[doc_id] += 0.4 * primary_scores.get(doc_id, 0)

                    elif relevance_model == "hits":
                        print("Hit Algorithm with Vector Space")

                        # For each document in our results
                        for doc in results:
                            if "url" in doc:
                                doc_id = doc["url"]
                                # Start with 70% of the current score
                                doc_scores[doc_id] = 0.7 * doc_scores.get(doc_id, 0)
                                # Add 20% of the authority score
                                doc_scores[doc_id] += 0.2 * primary_scores.get(doc_id, 0)

                    # Now add vector space component for all documents
                    for doc_id in doc_scores:
                        # For each top document, check if this doc is similar to it
                        for top_doc in top_docs:
                            if top_doc in vector_space_scores and doc_id in vector_space_scores[top_doc]:
                                # Add the similarity score
                                doc_scores[doc_id] += 0.3 * vector_space_scores[top_doc][doc_id]

                    # Sort results by the combined score
                    results = sorted(
                        results,
                        key=lambda d: doc_scores.get(d.get("url", ""), 0),
                        reverse=True
                    )
                    print(f"Applied hybrid scoring ({relevance_model} + vector space)")
                except Exception as e:
                    print(f"Error applying hybrid scoring: {e}")
                    # Fallback to just primary scores
                    results = sorted(
                        results,
                        key=lambda d: primary_scores.get(d.get("url", ""), 0),
                        reverse=True
                    )
                    print(f"Applied {relevance_model} relevance model (fallback)")
            else:
                # Just use primary scores if hybrid is disabled
                results = sorted(
                    results,
                    key=lambda d: primary_scores.get(d.get("url", ""), 0),
                    reverse=True
                )
                print(f"Applied {relevance_model} relevance model")
        except Exception as e:
            print(f"Error applying {relevance_model}: {e}")

    # 3) Apply clustering if specified
    if clustering_method == "flat_clustering":
        try:
            results = cluster_mgr.cluster_flat(raw, results)
            print("Applied flat clustering")
        except Exception as e:
            print(f"Error applying flat clustering: {e}")
    elif clustering_method == "single_hac":
        try:
            results = cluster_mgr.cluster_single(raw, results)
            print("Applied single-link HAC")
        except Exception as e:
            print(f"Error applying single-link HAC: {e}")
    elif clustering_method == "average_hac":
        try:
            results = cluster_mgr.cluster_average(raw, results)
            print("Applied average-link HAC")
        except Exception as e:
            print(f"Error applying average-link HAC: {e}")

    # 4) Apply query expansion if specified
    resp_q = raw
    ext = ""
    if expansion_method != "none":
        try:
            if expansion_method == "rocchio":
                ext = qe.rocchio_expand(raw, results)
                print(f"Rocchio expansion: '{ext}'")
            elif expansion_method == "association_qe":
                ext = qe.association_main(raw, results)
                print(f"Association QE: '{ext}'")
            elif expansion_method == "metric_qe":
                ext = qe.metric_cluster_main(raw, results)
                print(f"Metric QE: '{ext}'")
            elif expansion_method == "scalar_qe":
                ext = qe.scalar_main(raw, results)
                print(f"Scalar QE: '{ext}'")

            if ext:
                # Combine original query with expansion terms
                combined_query = f"{raw} {ext}"

                # Build a better Solr query that searches for each term individually
                all_terms = combined_query.split()
                expanded_solr_query = " ".join([f'content:"{term}"' for term in all_terms])
                print(f"Constructed Solr query: {expanded_solr_query}")

                # Store both the original and expanded queries
                original_query = raw
                expanded_query = combined_query

                newr = fetch_solr(expanded_solr_query)
                print(f"Expansion query returned {len(newr)} results")

                if newr:
                    # If we're using hybrid scoring, reapply it to the new results
                    if relevance_model in ("page_rank", "hits") and use_hybrid:
                        try:
                            # Load scores again (or reuse from above)
                            scores_path = os.path.join(RESULTS_DIR, f"{relevance_model}_scores.json")
                            with open(scores_path) as f:
                                primary_scores = json.load(f)

                            vs_scores_path = os.path.join(RESULTS_DIR, "vector_space_scores.json")
                            with open(vs_scores_path) as f:
                                vector_space_scores = json.load(f)

                            # Get URLs of top documents from new results
                            top_docs = [doc["url"] for doc in newr[:10] if "url" in doc]

                            # Create a scoring dictionary
                            doc_scores = {}

                            # Initialize scores with a small value
                            for doc in newr:
                                if "url" in doc:
                                    doc_scores[doc["url"]] = 0.0001

                            # Apply scoring based on the relevance model
                            if relevance_model == "page_rank":
                                # For each document in our results
                                for doc in newr:
                                    if "url" in doc:
                                        doc_id = doc["url"]
                                        # Start with 70% of the current score
                                        doc_scores[doc_id] = 0.7 * doc_scores.get(doc_id, 0)
                                        # Add 40% of the PageRank score
                                        doc_scores[doc_id] += 0.4 * primary_scores.get(doc_id, 0)

                            elif relevance_model == "hits":
                                # For each document in our results
                                for doc in newr:
                                    if "url" in doc:
                                        doc_id = doc["url"]
                                        # Start with 70% of the current score
                                        doc_scores[doc_id] = 0.7 * doc_scores.get(doc_id, 0)
                                        # Add 20% of the authority score
                                        doc_scores[doc_id] += 0.2 * primary_scores.get(doc_id, 0)

                            # Now add vector space component for all documents
                            for doc_id in doc_scores:
                                # For each top document, check if this doc is similar to it
                                for top_doc in top_docs:
                                    if top_doc in vector_space_scores and doc_id in vector_space_scores[top_doc]:
                                        # Add the similarity score
                                        doc_scores[doc_id] += 0.3 * vector_space_scores[top_doc][doc_id]

                            # Sort results by the combined score
                            newr = sorted(
                                newr,
                                key=lambda d: doc_scores.get(d.get("url", ""), 0),
                                reverse=True
                            )
                            print(f"Reapplied hybrid scoring to expanded results")
                        except Exception as e:
                            print(f"Error reapplying hybrid scoring: {e}")

                    results = newr
                    resp_q = expanded_query  # Use the combined query
                else:
                    print("No results from expansion query, using original results")
            else:
                print("Expansion returned empty string, using original results")
        except Exception as e:
            print(f"Error in {expansion_method}: {e}")

    # Enhance the response with both original and expanded queries
    response_data = {
        "query": resp_q,
        "query_results": results,
        "applied_methods": {
            "relevance": relevance_model,
            "clustering": clustering_method,
            "expansion": expansion_method,
            "hybrid": str(use_hybrid).lower()
        }
    }

    # Add original_query field if query expansion was applied
    if expansion_method != "none" and ext:
        response_data["original_query"] = raw
        response_data["expanded_terms"] = ext

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
