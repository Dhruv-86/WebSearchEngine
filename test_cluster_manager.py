import json
import os
import pytest
from ClusterManager import ClusterManager

# Adjust these paths to match your layout
BASE_DIR    = "/Users/dhruv/Downloads"
DATA_PATH   = os.path.join(BASE_DIR, "data.json")
RESULTS_DIR = "/Users/dhruv/Desktop/Documents/project/Proooo/Web-SearchEngine/result"

# Load JSON docs
with open(DATA_PATH, "r") as f:
    data = json.load(f)
docs = data["response"]["docs"]

@pytest.fixture(scope="module")
def cm():
    return ClusterManager(results_dir=RESULTS_DIR)

@pytest.mark.parametrize("mode, method", [
    ("flat",    lambda cm, q, d: cm.cluster_flat(q, d)),
    ("average", lambda cm, q, d: cm.cluster_average(q, d)),
    ("single",  lambda cm, q, d: cm.cluster_single(q, d)),
])
def test_contiguous_clusters(mode, method, cm):
    """
    1) Every returned item has 'url' and integer 'cluster_num'.
    2) Each cluster_num only appears in one contiguous block.
    """
    clustered = method(cm, "test query", docs)

    # Basic format checks
    assert isinstance(clustered, list), f"{mode} did not return a list"
    for doc in clustered:
        assert isinstance(doc, dict), f"{mode} returned non-dict"
        assert "url" in doc,        f"{mode} missing 'url'"
        assert "cluster_num" in doc, f"{mode} missing 'cluster_num'"
        assert isinstance(doc["cluster_num"], int), f"{mode} cluster_num not int"

    # Check contiguous blocks
    seen = set()
    last = None
    for doc in clustered:
        cid = doc["cluster_num"]
        if cid != last:
            # switching to a new cluster → it must not have been seen before
            assert cid not in seen, (
                f"{mode}: cluster {cid} reappears after another cluster"
            )
            seen.add(cid)
        last = cid

    # Optionally: ensure that at least 2 distinct clusters occurred
    assert len(seen) >= 2, f"{mode}: expected multiple clusters, got {seen}"
