import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# load the same vectorizer you dumped for clustering
VEC_PATH = "./result/tfidfVec.pkl"
vectorizer = pickle.load(open(VEC_PATH, "rb"))

# basic Rocchio
def rocchio_expand(query, docs, top_n=5,
                   alpha=1.0, beta=0.75, gamma=0.15):
    texts = [d.get("content","") for d in docs]
    X = vectorizer.transform(texts).toarray()
    qv = vectorizer.transform([query]).toarray()[0]
    n_rel = min(20, len(X))
    rel    = X[:n_rel].mean(axis=0)
    nonrel = X[n_rel:].mean(axis=0) if len(X)>n_rel else 0
    newv = alpha*qv + beta*rel - gamma*nonrel
    terms = vectorizer.get_feature_names_out()
    idxs = np.argsort(newv)[::-1]
    out=[]
    for i in idxs:
        if len(out)>=top_n: break
        t=terms[i]
        if t.lower() not in query.lower().split(): out.append(t)
    return " ".join(out)

# Association QE (co-occurrence)
def association_main(query, solr_results, start=0, end=30):
    stop_words = set(stopwords.words("english"))
    id_map, tokens = {}, {}
    for r in solr_results[:end]:
        content = r.get("content","")
        words = [w.lower() for w in word_tokenize(content)
                 if w.isalnum() and w.lower() not in stop_words]
        tokens[r['url']] = words

    vocab = set(w for lst in tokens.values() for w in lst)
    scores=[]
    for v in vocab:
        c2=c3=c4=0
        for w in query.split():
            for doc,ws in tokens.items():
                c0=ws.count(v); c1=ws.count(w)
                c2+=c0*c1; c3+=c0*c0; c4+=c1*c1
        denom = (c2+c3+c4) or 1
        sim = c2/denom
        if sim>0: scores.append((v,sim))
    scores.sort(key=lambda x:-x[1])
    added=[]
    for v,_ in scores:
        if len(added)>=3: break
        if v not in query.split(): added.append(v)
    return " ".join(added)

# Metric QE placeholder: you can refine
def metric_cluster_main(query, solr_results):
    # minimal stub: fallback to first 3 words of first doc
    if solr_results:
        words = solr_results[0].get("content","").split()
        return " ".join(words[:3])
    return ""

# Scalar QE placeholder: likewise
def scalar_main(query, solr_results):
    if solr_results:
        words = solr_results[0].get("content","").split()
        return " ".join(words[-3:])
    return ""
