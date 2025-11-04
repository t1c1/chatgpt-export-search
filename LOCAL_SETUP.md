# Local Semantic Search Setup for gptgrep

Run semantic search 100% locally without any cloud services. Choose between Ollama or sentence-transformers.

## Quick Start

### Option 1: Sentence-Transformers (Easiest)

This runs completely offline after the initial model download:

```bash
# 1. Install dependencies
pip install sentence-transformers torch numpy

# 2. Index your ChatGPT export with enhanced metadata
python index_enhanced.py --export ./my_export --out ./index --stats

# 3. Generate embeddings (one-time, takes ~5-30 minutes)
python local_embeddings.py --db ./index/chatgpt.db --type sentence-transformers

# 4. Search with semantic understanding
python search_local.py "machine learning concepts" --mode vector

# Search with date filters
python search_local.py "python code" --date-from 2024-01-01 --date-to 2024-12-31

# Hybrid search (combines text + semantic)
python search_local.py "explain neural networks" --mode hybrid --alpha 0.7
```

### Option 2: Ollama (More Powerful Models)

Use larger, more capable models through Ollama:

```bash
# 1. Install Ollama
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama server (keep running in background)
ollama serve

# 3. Pull an embedding model (in another terminal)
ollama pull nomic-embed-text  # Fast, good quality
# or
ollama pull mxbai-embed-large  # Larger, better quality

# 4. Generate embeddings
python local_embeddings.py --db ./index/chatgpt.db --type ollama --model nomic-embed-text

# 5. Search
python search_local.py "deep learning" --mode vector --model-type ollama
```

## Search Examples

### Basic Semantic Search
```bash
# Find conceptually similar conversations
python search_local.py "machine learning algorithms"
```

### Date-Filtered Search
```bash
# Search within a date range
python search_local.py "python" --date-from 2024-06-01 --date-to 2024-12-31

# Search this year's conversations
python search_local.py "project ideas" --date-from 2024-01-01
```

### Browse Conversations by Date
```bash
# List all conversations in a time period
python search_local.py --mode conversations --date-from 2024-10-01 --limit 20

# Search conversation titles
python search_local.py "code review" --mode conversations
```

### Fine-tune Search Balance
```bash
# More semantic (0.8 = 80% semantic, 20% keyword)
python search_local.py "neural networks" --mode hybrid --alpha 0.8

# More keyword-based (0.2 = 20% semantic, 80% keyword)
python search_local.py "def calculate_loss" --mode hybrid --alpha 0.2
```

### View Statistics
```bash
# See what's in your database
python search_local.py --stats
```

## Model Comparison

### Sentence-Transformers Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 80MB | Very Fast | Good | Default, balanced |
| all-mpnet-base-v2 | 420MB | Fast | Better | Higher quality |
| all-MiniLM-L12-v2 | 120MB | Fast | Good | Slightly better than L6 |

Change model:
```bash
python local_embeddings.py --db ./index/chatgpt.db --model all-mpnet-base-v2
```

### Ollama Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| nomic-embed-text | 274MB | Fast | Very Good | Recommended |
| mxbai-embed-large | 669MB | Medium | Excellent | Best quality |
| all-minilm | 45MB | Very Fast | Good | Lightweight |

## Performance Tips

1. **First-time setup**: Generating embeddings takes time (5-30 minutes for typical exports)
2. **GPU acceleration**: If you have a GPU, torch will use it automatically
3. **Apple Silicon**: M1/M2/M3 Macs get automatic acceleration
4. **Batch size**: Increase for faster processing if you have enough RAM:
   ```bash
   python local_embeddings.py --db ./index/chatgpt.db --batch-size 64
   ```

## Storage Requirements

- Database with embeddings: ~500MB-2GB depending on export size
- Model downloads: 80MB-700MB depending on model choice
- Original export files: Keep for re-indexing if needed

## Troubleshooting

### "No embeddings found"
Generate embeddings first:
```bash
python local_embeddings.py --db ./index/chatgpt.db
```

### "Ollama connection refused"
Start Ollama server:
```bash
ollama serve
```

### Out of memory
Reduce batch size:
```bash
python local_embeddings.py --db ./index/chatgpt.db --batch-size 8
```

### Slow search
- Use a smaller model
- Reduce the number of results: `--limit 5`
- Use FTS mode for keyword search: `--mode fts`

## Advanced Usage

### Re-generate with Different Model
```bash
# Clear existing embeddings (optional)
sqlite3 ./index/chatgpt.db "DELETE FROM doc_embeddings"

# Generate with new model
python local_embeddings.py --db ./index/chatgpt.db --model all-mpnet-base-v2
```

### Export Results
```bash
# Save search results to file
python search_local.py "machine learning" --limit 50 > results.txt
```

### Custom Similarity Threshold
```bash
# Only show highly similar results (0.7+ similarity)
python search_local.py "specific concept" --mode vector --threshold 0.7
```

## Privacy & Security

- ✅ Everything runs locally - no data leaves your machine
- ✅ Models are downloaded once and cached locally
- ✅ No API keys or authentication required
- ✅ Database is a single file - easy to backup or delete
- ✅ Works offline after initial setup