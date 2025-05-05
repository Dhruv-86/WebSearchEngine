import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# load the same vectorizer you dumped for clustering
VEC_PATH = "./result/tfidfVec.pkl"
try:
    vectorizer = pickle.load(open(VEC_PATH, "rb"))
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    # Fallback to create a new vectorizer if loading fails
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# basic Rocchio
def rocchio_expand(query, docs, top_n=5,
                   alpha=1.0, beta=0.75, gamma=0.15):
    try:
        # Ensure we have content to work with
        texts = [d.get("content", "") for d in docs if d.get("content")]
        if not texts:
            print("No content found in documents for Rocchio expansion")
            return ""

        # Transform texts to vectors
        try:
            X = vectorizer.transform(texts).toarray()
            qv = vectorizer.transform([query]).toarray()[0]
        except Exception as e:
            print(f"Error in vectorizer transform: {e}")
            return ""

        n_rel = min(20, len(X))
        rel = X[:n_rel].mean(axis=0)
        nonrel = X[n_rel:].mean(axis=0) if len(X) > n_rel else np.zeros_like(qv)

        newv = alpha*qv + beta*rel - gamma*nonrel

        # Get feature names safely
        try:
            terms = vectorizer.get_feature_names_out()
        except AttributeError:
            # For older scikit-learn versions
            try:
                terms = vectorizer.get_feature_names()
            except Exception as e:
                print(f"Error getting feature names: {e}")
                return ""

        idxs = np.argsort(newv)[::-1]
        out = []
        query_terms = query.lower().split()

        for i in idxs:
            if len(out) >= top_n:
                break
            t = terms[i]
            if t.lower() not in query_terms and len(t) > 2:  # Avoid short terms
                out.append(t)

        return " ".join(out)
    except Exception as e:
        print(f"Rocchio expansion error: {e}")
        return ""

# Association QE (co-occurrence)
def association_main(query, solr_results, start=0, end=30):
    try:
        if not solr_results:
            return ""

        stop_words = set(stopwords.words("english"))
        tokens = {}

        # Limit to available results
        end = min(end, len(solr_results))

        for r in solr_results[:end]:
            content = r.get("content", "")
            if not content:
                continue

            words = [w.lower() for w in word_tokenize(content)
                     if w.isalnum() and w.lower() not in stop_words]
            tokens[r['url']] = words

        if not tokens:
            return ""

        vocab = set(w for lst in tokens.values() for w in lst)
        scores = []
        query_terms = query.split()

        for v in vocab:
            if v in query_terms or len(v) < 3:  # Skip query terms and short words
                continue

            c2 = c3 = c4 = 0
            for w in query_terms:
                for doc, ws in tokens.items():
                    c0 = ws.count(v)
                    c1 = ws.count(w)
                    c2 += c0 * c1
                    c3 += c0 * c0
                    c4 += c1 * c1

            denom = (c2 + c3 + c4) or 1
            sim = c2 / denom
            if sim > 0:
                scores.append((v, sim))

        scores.sort(key=lambda x: -x[1])
        added = []

        for v, _ in scores:
            if len(added) >= 3:
                break
            if v not in query_terms:
                added.append(v)

        return " ".join(added)
    except Exception as e:
        print(f"Association expansion error: {e}")
        return ""

# Metric QE: improved version
def metric_cluster_main(query, solr_results):
    try:
        if not solr_results:
            return ""

        # Get all content
        all_content = " ".join([d.get("content", "") for d in solr_results[:5]])

        # Tokenize and filter
        stop_words = set(stopwords.words("english"))
        words = [w.lower() for w in word_tokenize(all_content)
                 if w.isalnum() and w.lower() not in stop_words and len(w) > 3]

        # Count word frequencies
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

        # Get top words not in query
        query_terms = query.lower().split()
        expansion_terms = []

        for word, _ in sorted_words:
            if word not in query_terms and len(expansion_terms) < 3:
                expansion_terms.append(word)

        return " ".join(expansion_terms)
    except Exception as e:
        print(f"Metric expansion error: {e}")
        return ""

# Scalar QE: improved version
def scalar_main(query, solr_results):
    try:
        if not solr_results:
            return ""

        # Combine content from multiple docs
        combined_content = ""
        for doc in solr_results[:3]:
            content = doc.get("content", "")
            if content:
                combined_content += " " + content

        if not combined_content:
            return ""

        # Extract meaningful terms
        stop_words = set(stopwords.words("english"))
        words = [w.lower() for w in word_tokenize(combined_content)
                 if w.isalnum() and w.lower() not in stop_words and len(w) > 3]

        # Get unique words
        unique_words = list(set(words))

        # Filter out query terms
        query_terms = query.lower().split()
        expansion_terms = [w for w in unique_words if w not in query_terms]

        # Return random selection of terms
        import random
        random.shuffle(expansion_terms)
        return " ".join(expansion_terms[:3])
    except Exception as e:
        print(f"Scalar expansion error: {e}")
        return ""