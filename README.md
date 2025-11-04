# gptgrep

Super-fast local search over your ChatGPT export (`chat.html`, `conversations.json`, `message_feedback.json`, `shared_conversations.json`, `user.json`) using **SQLite FTS5**. Optional semantic search via local embeddings or Google Vertex AI.

## Features
- Ingests your export JSONs (robust to shape differences)
- Optional: stores raw `chat.html` as a searchable blob
- Full-text search via SQLite **FTS5** with highlighted snippets
- Web UI with mode toggle (FTS / Vector / Hybrid), date filters, and relevance score
- Optional semantic search using local embeddings (fully offline) or Google Vertex AI
- CLI search: `python search.py "robotics AND supplier"`
- Enhanced CLI: `python search_local.py` and `python search_vertex.py`
- Tiny web app: `python serve.py` → open http://127.0.0.1:5000
- Zero heavy deps for basic text search; semantic search deps are optional

## Quickstart (Base)

1) Put your export files in a folder, e.g. `./my_export/`
   - `chat.html`
   - `conversations.json`
   - `shared_conversations.json`
   - `message_feedback.json`
   - `user.json`

2) Create a virtualenv (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3) Install base deps (for the web UI):
```bash
pip install -r requirements.txt
```

4) Build the index (produces `index/chatgpt.db`):
```bash
python index.py --export ./my_export --out ./index
```

5) CLI search:
```bash
python search.py "e-coat OR electrocoating" --limit 10
```

6) Tiny web UI:
```bash
python serve.py --db ./index/chatgpt.db
# open http://127.0.0.1:5000
```

## Optional: Local Semantic Search (offline)

Install local extras and generate embeddings once. This enables Vector and Hybrid modes in the UI.

```bash
pip install -r requirements-local.txt

# Generate embeddings (one-time, can take a while)
python local_embeddings.py --db ./index/chatgpt.db --type sentence-transformers --batch-size 64

# Try CLI hybrid search
python search_local.py "deep learning" --mode hybrid --alpha 0.7
```

## Optional: Semantic Search with Google Vertex AI

For more powerful semantic search capabilities:

### Setup Vertex AI
```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com
```

### Generate Embeddings
```bash
# Generate embeddings for your database
python vertex_embeddings.py --db ./index/chatgpt.db --project YOUR_PROJECT_ID
```

### Search with Semantic Understanding
```bash
# Hybrid search (best of both worlds)
python search_vertex.py "machine learning concepts" --mode hybrid

# Pure semantic search
python search_vertex.py "explain neural networks" --mode vector

# Adjust the balance (0.7 = 70% semantic, 30% text matching)
python search_vertex.py "deep learning" --mode hybrid --alpha 0.7
```

See [VERTEX_AI_SETUP.md](VERTEX_AI_SETUP.md) for detailed setup instructions.

## Data model

We normalize all messages/documents into a single `docs` table and an FTS5 virtual table:

- `docs(id TEXT PRIMARY KEY, conv_id TEXT, title TEXT, role TEXT, ts REAL, source TEXT, extra TEXT, content TEXT)`
- `docs_fts(content, title, role, source, conv_id, ts, doc_id UNINDEXED)`

`extra` is JSON for any metadata we don't explicitly model (e.g., feedback labels).

## UI Tips

- Use Mode to switch between FTS (keyword), Vector (semantic), or Hybrid.
- Alpha balances Hybrid scoring: higher = more semantic weighting.
- Date filters constrain results when the index has date metadata (use `index_enhanced.py`).
- Relevance score appears for Vector/Hybrid results.

## Extending

- Add embeddings/RAG: create another table with vector columns (e.g., pgvector, sqlite-vss) and store a vector per doc. Use your favorite LLM to retrieve + generate answers.
- Import other data: write a small adapter and call `insert_doc(...)`.

## Safety & privacy
- Everything stays local.
- Review your export for secrets before indexing if you're worried.
- DB is a single file; delete it to wipe the index.

## License
MIT
