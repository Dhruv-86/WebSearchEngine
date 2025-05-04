# scripts/generate_results.py

import os
import json
import pickle
import argparse

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

def main(data_json, result_dir, k):
    os.makedirs(result_dir, exist_ok=True)

    # 1) Load your data.json
    with open(data_json, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    docs = payload.get("response", {}).get("docs", [])

    # 2) Build URL & text lists
    urls  = [d["url"] for d in docs]
    texts = [ (d.get("title","") + " " + d.get("content","")).strip() for d in docs ]

    # 3) TF–IDF vectorization
    print("Vectorizing texts with TF–IDF...")
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(texts)
    pickle.dump(vectorizer, open(os.path.join(result_dir, "tfidfVec.pkl"), "wb"))

    # 4) Flat clustering via KMeans
    print(f"Running KMeans (k={k}) for flat clustering...")
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = km.labels_
    centers = km.cluster_centers_

    # write url→clusterNum_flat.txt
    with open(os.path.join(result_dir, "url_clusterNum_flat.txt"), "w") as f:
        for u, lbl in zip(urls, labels):
            f.write(f"{u} {lbl}\n")

    # write cluster_center_flat.txt
    with open(os.path.join(result_dir, "cluster_center_flat.txt"), "w") as f:
        for idx, ctr in enumerate(centers):
            coords = ",".join(map(str, ctr.tolist()))
            f.write(f"{idx} [{coords}]\n")

    # 5) Agglomerative single & average
    for mode, linkage in [("avg","average"), ("single","single")]:
        print(f"Running AgglomerativeClustering (k={k}, linkage={linkage})...")
        # Need dense array for scipy linkage
        arr = X.toarray()
        cl = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(arr)
        lbls = cl.labels_

        # url→clusterNum_{mode}.txt
        fn_map = f"url_clusterNum_{mode}.txt"
        with open(os.path.join(result_dir, fn_map), "w") as f:
            for u, l in zip(urls, lbls):
                f.write(f"{u} {l}\n")

        # compute centers as mean of member vectors
        centers = []
        for i in range(k):
            members = arr[lbls == i]
            center = members.mean(axis=0)
            centers.append(center)

        # cluster_center_{mode}.txt
        fn_cent = f"cluster_center_{mode}.txt"
        with open(os.path.join(result_dir, fn_cent), "w") as f:
            for idx, ctr in enumerate(centers):
                coords = ",".join(map(str, ctr.tolist()))
                f.write(f"{idx} [{coords}]\n")

    print("All result files generated under:", result_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate TF–IDF and clustering result files for ClusterManager"
    )
    p.add_argument(
        "--data-json", "-d",
        default="/Users/vedanshsurjan/Downloads/data2.json",
        help="Path to your crawled data.json"
    )
    p.add_argument(
        "--out", "-o",
        default="./result",
        help="Output directory for tfidfVec.pkl and cluster files"
    )
    p.add_argument(
        "--k", "-k",
        type=int,
        default=10,
        help="Number of clusters"
    )
    args = p.parse_args()
    main(args.data_json, args.out, args.k)
