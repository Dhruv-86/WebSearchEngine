# Gymnastics Search Engine (Backend)

A Flask + Solr–based search engine backend for gymnastics content.  
Supports:
- Vector‐space + PageRank/HITS relevance models  
- Flat, single‐link & average‐link hierarchical clustering  
- Four query‐expansion methods (Rocchio, association, metric, scalar)  

---

## Prerequisites

- **Python 3.7+**  
- **Java 11+** (required by Solr)  
- **Git** (to clone this repo)  

You’ll also need **Solr 9.x**—we show how to install it below.

---
Place your `data.json` (sample of crawled gymnastics pages) anywhere on disk and update paths accordingly in the scripts.

## macOS Setup

1. **Install Java & Solr**  
   brew install openjdk@11
   brew install solr
2. **Start Solr & create core**
    brew services start solr
    solr create -c nutch -n data_driven_schema_configs
3. **Clone & enter repo**
    git clone <your-repo-url>
    cd <repo-dir>
4. **Create & activate Python virtualenv**
    python3 -m venv env
    source env/bin/activate

5. **Install Python dependencies**
    pip install -r requirements.txt
6. **Download NLTK data**
    python -c "import nltk; nltk.download('stopwords')"
7. **Index data.json into Solr**
    python scripts/index_data.py (in index_data.py change the location of data.json file)
    *verify*
    curl 'http://localhost:8983/solr/nutch/select?q=*:*&wt=json&rows=3' | jq .

8. **Generate TF–IDF & clustering files**
    python scripts/generate_results.py \
    --data-json ~/Downloads/data2.json \
    --out ./result \
    --k 10

9. **Generate relevance‐score JSONs**
    python scripts/generate_relevance.py
10. **Run the Flask server**
    export FLASK_APP=app.py
    export FLASK_ENV=development
    flask run

11. **Smoke-test endpoints**
    curl http://127.0.0.1:5000/ping 
        Reply---> PONG
    curl 'http://127.0.0.1:5000/api/v1/search?query=gymnastics&type=flat_clustering' | jq .
12. **Run integration tests & generate report**
    pytest test_integration.py -q \
        && python report.py > report.md



## Windows Setup

1. **Install Java & Solr**  
   Download & install from https://adoptium.net or Oracle JDK.
2. **Download and Start Solr & create core**
    cd C:\tools
    curl -O https://dlcdn.apache.org/lucene/solr/9.7.0/solr-9.7.0.zip
    tar -xf solr-9.7.0.zip
    cd solr-9.7.0
    .\bin\solr start
    .\bin\solr create -c nutch -n data_driven_schema_configs

3. **Clone & enter repo**
    git clone <your-repo-url>
    cd <repo-dir>

4. **Create & activate Python virtualenv**
    python -m venv env
    .\env\Scripts\activate

5. **Install Python dependencies**
    pip install -r requirements.txt

6. **Download NLTK data**
    python -c "import nltk; nltk.download('stopwords')"

7. **Index data.json into Solr**
    python .\scripts\index_data.py (would require to change the location of data.json file)

8. **Generate TF–IDF & clustering files**
    python .\scripts\generate_results.py `
  --data-json C:\Users\you\Downloads\data2.json `
  --out .\result --k 10

9. **Generate relevance‐score JSONs**
    python .\scripts\generate_relevance.py

10. **Run the Flask server**
    set FLASK_APP=app.py
    set FLASK_ENV=development
    flask run
11. **Smoke-test endpoints**
    curl http://127.0.0.1:5000/ping
    curl "http://127.0.0.1:5000/api/v1/search?query=gymnastics&type=flat_clustering" | jq .
12. **Run integration tests & generate report**
    pytest test_integration.py -q ; python report.py > report.md
