import pickle
from collections import defaultdict, deque
import os

class ClusterManager:
    def __init__(self, results_dir="./result"):
        self.modes = ["flat", "average", "single"]
        self.results_dir = results_dir.rstrip(os.sep)
        # url→cluster and centers
        self.url_map = {m: {} for m in self.modes}
        self.centers = {m: {} for m in self.modes}

        # TF–IDF model
        tfidf_path = os.path.join(self.results_dir, "tfidfVec.pkl")
        self.tfidf = pickle.load(open(tfidf_path, "rb"))

        self._load_all_clusters()

        # simple LRU cache for hierarchical
        self.cache  = {}
        self.recent = deque(maxlen=5)

    def _load_all_clusters(self):
        for mode in self.modes:
            suffix = "avg" if mode=="average" else mode
            # url→cluster
            fn1 = f"url_clusterNum_{suffix}.txt"
            with open(os.path.join(self.results_dir, fn1)) as f:
                for line in f:
                    url, cid = line.strip().split()
                    self.url_map[mode][url] = int(cid)
            # centers
            fn2 = f"cluster_center_{suffix}.txt"
            with open(os.path.join(self.results_dir, fn2)) as f:
                for line in f:
                    idx, coords = line.strip().split(" ",1)
                    arr = coords.strip()[1:-1].split(",")
                    self.centers[mode][int(idx)] = [float(x) for x in arr]

    @staticmethod
    def _euclid(a, b):
        return sum((x-y)**2 for x,y in zip(a,b))**0.5

    def _vectorize(self, text):
        return self.tfidf.transform([text]).toarray()[0].tolist()

    def _rank(self, qvec, mode):
        items = list(self.centers[mode].items())
        items.sort(key=lambda kv: (self._euclid(kv[1], qvec), kv[0]))
        return [cid for cid,_ in items]

    def _apply(self, mode, query, docs):
        key = (mode, query)
        if key in self.cache:
            return self.cache[key]

        qvec  = self._vectorize(query)
        order = self._rank(qvec, mode)

        buckets, unassigned = defaultdict(list), []
        for d in docs:
            cid = self.url_map[mode].get(d['url'], -1)
            e = dict(d); e['cluster_num']=cid
            if cid>=0: buckets[cid].append(e)
            else:      unassigned.append(e)

        out=[]
        for cid in order:
            out.extend(buckets.get(cid, []))
        out.extend(unassigned)

        self.cache[key]=out
        self.recent.append(key)
        if len(self.recent)>self.recent.maxlen:
            old=self.recent.popleft()
            self.cache.pop(old,None)
        return out

    def cluster_flat(self, query, docs):      return self._apply("flat",    query, docs)
    def cluster_average(self, query, docs):   return self._apply("average", query, docs)
    def cluster_single(self, query, docs):    return self._apply("single",  query, docs)
