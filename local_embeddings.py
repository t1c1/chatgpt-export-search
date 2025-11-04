import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
from pathlib import Path
import pickle
import requests
from sentence_transformers import SentenceTransformer
import torch

class LocalEmbeddingsManager:
    """Manage embeddings using local models - either Ollama or sentence-transformers."""
    
    def __init__(
        self,
        model_type: str = "sentence-transformers",
        model_name: Optional[str] = None,
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize local embedding manager.
        
        Args:
            model_type: "ollama" or "sentence-transformers"
            model_name: Model to use (defaults based on type)
            ollama_host: Ollama API endpoint if using Ollama
        """
        self.model_type = model_type
        
        if model_type == "ollama":
            self.ollama_host = ollama_host
            # Default Ollama models with embeddings support
            self.model_name = model_name or "nomic-embed-text"
            self.embedding_dimension = self._get_ollama_dimensions()
            print(f"Using Ollama model: {self.model_name}")
            print(f"Make sure Ollama is running: ollama serve")
            print(f"Pull the model if needed: ollama pull {self.model_name}")
        else:
            # Use sentence-transformers for fully offline operation
            self.model_name = model_name or "all-MiniLM-L6-v2"
            print(f"Loading sentence-transformers model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            # Enable GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print("Using GPU for embeddings")
            elif torch.backends.mps.is_available():
                self.model = self.model.to('mps')
                print("Using Apple Silicon GPU for embeddings")
    
    def _get_ollama_dimensions(self) -> int:
        """Get embedding dimensions for Ollama model."""
        # Common Ollama embedding models and their dimensions
        dimensions = {
            "nomic-embed-text": 768,
            "llama2": 4096,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
        }
        return dimensions.get(self.model_name, 768)
    
    def _get_ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting Ollama embedding: {e}")
            print(f"Make sure Ollama is running and {self.model_name} is pulled")
            raise
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once (for sentence-transformers)
            
        Returns:
            List of numpy arrays containing embeddings
        """
        embeddings = []
        
        if self.model_type == "ollama":
            # Ollama processes one at a time
            for i, text in enumerate(texts):
                if i % 10 == 0 and i > 0:
                    print(f"  Processed {i}/{len(texts)} texts...")
                
                # Truncate very long texts
                text = text[:8000] if text else ""
                embedding = self._get_ollama_embedding(text)
                embeddings.append(embedding)
        else:
            # Sentence-transformers can batch process
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Truncate very long texts
                batch = [text[:8000] if text else "" for text in batch]
                
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                for embedding in batch_embeddings:
                    embeddings.append(embedding.astype(np.float32))
                
                if i % 100 == 0 and i > 0:
                    print(f"  Processed {i}/{len(texts)} texts...")
        
        return embeddings
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a single query string."""
        if self.model_type == "ollama":
            return self._get_ollama_embedding(query)
        else:
            return self.model.encode(query, convert_to_numpy=True).astype(np.float32)


class VectorDatabase:
    """SQLite-based vector storage with efficient similarity search."""
    
    def __init__(self, db_path: Path):
        """
        Initialize vector database.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._init_vector_tables()
    
    def _init_vector_tables(self):
        """Create tables for storing vectors."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB,
                embedding_model TEXT,
                embedding_dimension INTEGER,
                created_at REAL
            )
        """)
        
        # Create index for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc_id 
            ON doc_embeddings(doc_id)
        """)
        
        # Store embedding metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        self.conn.commit()
    
    def store_embedding(self, doc_id: str, embedding: np.ndarray, model_name: str):
        """
        Store a document embedding.
        
        Args:
            doc_id: Document ID from the docs table
            embedding: Numpy array containing the embedding vector
            model_name: Name of the model used to generate embedding
        """
        import time
        
        # Serialize the numpy array
        embedding_blob = pickle.dumps(embedding)
        
        self.conn.execute("""
            INSERT OR REPLACE INTO doc_embeddings (
                doc_id, embedding, embedding_model, embedding_dimension, created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, embedding_blob, model_name, len(embedding), time.time()))
        
        self.conn.commit()
    
    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a document embedding.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Numpy array containing the embedding, or None if not found
        """
        cursor = self.conn.execute("""
            SELECT embedding FROM doc_embeddings WHERE doc_id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        return None
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Retrieve all embeddings from the database.
        
        Returns:
            Dictionary mapping doc_id to embedding vector
        """
        cursor = self.conn.execute("""
            SELECT doc_id, embedding FROM doc_embeddings
        """)
        
        embeddings = {}
        for doc_id, embedding_blob in cursor:
            embeddings[doc_id] = pickle.loads(embedding_blob)
        
        return embeddings
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
        date_filter: Optional[Tuple[str, str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Find most similar documents using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            date_filter: Optional (start_date, end_date) tuple in YYYY-MM-DD format
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        # Build query with optional date filter
        if date_filter:
            start_date, end_date = date_filter
            cursor = self.conn.execute("""
                SELECT e.doc_id, e.embedding
                FROM doc_embeddings e
                JOIN docs d ON e.doc_id = d.id
                WHERE d.date >= ? AND d.date <= ?
            """, (start_date, end_date))
            embeddings_list = cursor.fetchall()
        else:
            cursor = self.conn.execute("SELECT doc_id, embedding FROM doc_embeddings")
            embeddings_list = cursor.fetchall()
        
        if not embeddings_list:
            return []
        
        # Compute cosine similarities
        similarities = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for doc_id, embedding_blob in embeddings_list:
            embedding = pickle.loads(embedding_blob)
            doc_norm = embedding / np.linalg.norm(embedding)
            similarity = np.dot(query_norm, doc_norm)
            
            if similarity >= threshold:
                similarities.append((doc_id, float(similarity)))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        fts_results: List[str],
        alpha: float = 0.5,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Combine vector similarity with FTS5 results.
        
        Args:
            query_embedding: Query vector
            fts_results: Document IDs from FTS5 search
            alpha: Weight for vector similarity (0-1)
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, combined_score) tuples
        """
        # Get vector similarities
        vector_results = self.search_similar(query_embedding, top_k=top_k*2)
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        
        # Normalize FTS results (higher rank = higher score)
        fts_scores = {}
        for i, doc_id in enumerate(fts_results[:top_k*2]):
            fts_scores[doc_id] = 1.0 - (i / len(fts_results))
        
        # Combine scores
        all_doc_ids = set(vector_scores.keys()) | set(fts_scores.keys())
        combined_scores = []
        
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            f_score = fts_scores.get(doc_id, 0.0)
            combined = alpha * v_score + (1 - alpha) * f_score
            combined_scores.append((doc_id, combined))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]


def create_embeddings_for_db(
    db_path: Path,
    model_type: str = "sentence-transformers",
    model_name: Optional[str] = None,
    batch_size: int = 32,
    limit: Optional[int] = None
):
    """
    Generate and store embeddings for all documents in the database.
    
    Args:
        db_path: Path to the SQLite database
        model_type: "ollama" or "sentence-transformers"
        model_name: Optional model name
        batch_size: Number of documents to process at once
        limit: Maximum number of documents to process (for testing)
    """
    print(f"Initializing {model_type} embedding model...")
    embeddings_mgr = LocalEmbeddingsManager(
        model_type=model_type,
        model_name=model_name
    )
    vector_db = VectorDatabase(db_path)
    
    # Get all documents from the database
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT id, content, title, date 
        FROM docs 
        WHERE content IS NOT NULL AND content != ''
    """
    if limit:
        query += f" LIMIT {limit}"
    
    cursor = conn.execute(query)
    docs = cursor.fetchall()
    
    print(f"Processing {len(docs)} documents...")
    
    # Process in batches
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        
        # Prepare texts for embedding (combine title and content)
        texts = []
        doc_ids = []
        for doc_id, content, title, date in batch:
            # Combine title and content for richer embeddings
            text = f"{title}\n{content}" if title else content
            # Add date context if available
            if date:
                text = f"Date: {date}\n{text}"
            texts.append(text)
            doc_ids.append(doc_id)
        
        # Generate embeddings
        try:
            embeddings = embeddings_mgr.get_embeddings(texts, batch_size=batch_size)
            
            # Store embeddings
            for doc_id, embedding in zip(doc_ids, embeddings):
                vector_db.store_embedding(
                    doc_id, 
                    embedding, 
                    f"{model_type}:{embeddings_mgr.model_name}"
                )
            
            print(f"Processed {min(i+batch_size, len(docs))}/{len(docs)} documents")
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {e}")
            continue
    
    print("Embedding generation complete!")
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate local embeddings for gptgrep")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--type", choices=["ollama", "sentence-transformers"], 
                      default="sentence-transformers",
                      help="Embedding model type")
    parser.add_argument("--model", help="Model name (optional)")
    parser.add_argument("--batch-size", type=int, default=32, 
                      help="Batch size for embedding generation")
    parser.add_argument("--limit", type=int, help="Limit number of documents (for testing)")
    
    args = parser.parse_args()
    
    create_embeddings_for_db(
        Path(args.db),
        model_type=args.type,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit
    )