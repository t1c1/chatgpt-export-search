# Google Vertex AI Setup for gptgrep

This guide helps you set up Google Vertex AI for semantic search in gptgrep.

## Prerequisites

1. A Google Cloud Platform (GCP) account
2. A GCP project with billing enabled
3. gcloud CLI installed on your machine

## Setup Steps

### 1. Install gcloud CLI (if not already installed)

```bash
# macOS
brew install google-cloud-sdk

# Linux/WSL
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth application-default login
```

### 3. Set your project

```bash
# List your projects
gcloud projects list

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 4. Enable required APIs

```bash
gcloud services enable aiplatform.googleapis.com
```

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Generate Embeddings for Your Database

First, you need to generate embeddings for all documents in your database:

```bash
# Generate embeddings for all documents
python vertex_embeddings.py --db ./index/chatgpt.db --project YOUR_PROJECT_ID

# Or test with a limited number of documents
python vertex_embeddings.py --db ./index/chatgpt.db --project YOUR_PROJECT_ID --limit 100
```

### Search with Different Modes

```bash
# Hybrid search (combines text and semantic search)
python search_vertex.py "machine learning concepts" --mode hybrid

# Pure semantic/vector search
python search_vertex.py "explain neural networks" --mode vector

# Traditional text search (FTS5 only)
python search_vertex.py "python AND tensorflow" --mode fts

# Adjust the balance between vector and text search (alpha=0.7 means 70% vector, 30% text)
python search_vertex.py "deep learning" --mode hybrid --alpha 0.7
```

## Configuration Options

### Environment Variables

You can set these environment variables instead of passing command-line arguments:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json  # Optional
```

### Using a Service Account (Optional)

For production use or CI/CD:

1. Create a service account:
```bash
gcloud iam service-accounts create gptgrep-sa \
    --display-name="gptgrep Service Account"
```

2. Grant necessary permissions:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:gptgrep-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

3. Create and download a key:
```bash
gcloud iam service-accounts keys create ~/gptgrep-key.json \
    --iam-account=gptgrep-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

4. Set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/gptgrep-key.json
```

## Cost Considerations

- Vertex AI Text Embeddings API charges per 1,000 characters
- Current pricing (check GCP for latest): ~$0.025 per 1,000 requests
- For a typical ChatGPT export with 1,000 conversations, expect ~$1-5 in embedding costs
- Subsequent searches are much cheaper (only the query is embedded)

## Troubleshooting

### "Permission denied" errors
- Ensure you've run `gcloud auth application-default login`
- Check that the Vertex AI API is enabled in your project

### "Quota exceeded" errors
- The free tier has limits on API calls per minute
- Use `--batch-size 1` to slow down embedding generation
- Consider upgrading your GCP account if needed

### "Module not found" errors
- Ensure you've activated your virtual environment
- Run `pip install -r requirements.txt` again

## Advanced Configuration

### Using Different Embedding Models

Edit `vertex_embeddings.py` to use different models:

```python
# In VertexEmbeddingsManager.__init__
model_name = "textembedding-gecko-multilingual@001"  # For multilingual support
model_name = "textembedding-gecko@003"  # Default, English-optimized
```

### Adjusting Vector Similarity Threshold

When searching, you can set a minimum similarity threshold:

```bash
python search_vertex.py "your query" --mode vector --threshold 0.5
```

This only returns results with cosine similarity >= 0.5 (range: 0-1)