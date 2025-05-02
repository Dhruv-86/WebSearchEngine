# test_integration.py

import pytest
import requests

BASE = "http://127.0.0.1:5000"

# Ensure the Flask server is running before any tests
@pytest.fixture(scope="session", autouse=True)
def ensure_server_running():
    try:
        r = requests.get(f"{BASE}/ping", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip("Flask server not running on port 5000")
    assert r.status_code == 200 and r.text == "pong"

# Parametrize over all modes your API supports
@pytest.mark.parametrize("mode", [
    "flat_clustering",
    "single_hac",
    "average_hac",
    "page_rank",
    "hits",
    "rocchio",
    "association_qe",
    "metric_qe",
    "scalar_qe",
])
def test_search_gymnastics_returns_relevant_docs(mode):
    """
    For each ?type=mode, /api/v1/search?query=gymnastics must return:
     - HTTP 200
     - a non-empty 'query_results'
     - every doc must mention 'gymnastics' in url, title, or content
    """
    resp = requests.get(f"{BASE}/api/v1/search", params={
        "query": "gymnastics",
        "type":  mode
    }, timeout=5)
    assert resp.status_code == 200, f"{mode} gave {resp.status_code}"
    data = resp.json()

    # Must echo back a query (original or expanded)
    assert "query" in data and isinstance(data["query"], str)

    results = data.get("query_results")
    assert isinstance(results, list) and len(results) > 0, f"{mode} returned no results"

    for doc in results:
        # collect the fields to search
        combined = " ".join(
            str(doc.get(k, "") or "") for k in ("url", "title", "content")
        ).lower()
        assert "gymnastics" in combined, (
            f"{mode}: doc does not mention 'gymnastics': "
            f"{combined[:80]}..."
        )
