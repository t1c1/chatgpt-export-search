import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
from pathlib import Path
import pickle

# Google Cloud Vertex AI imports
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import google.auth

class VertexEmbeddingsManager:
    """Manage embeddings using Google Vertex AI's text embedding models."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model_name: str = "textembedding-gecko@003"
    ):
        """
        Initialize Vertex AI embedding manager.
        
        Args:
            project_id: GCP project ID. If None, will use default credentials
            location: GCP region for Vertex AI
            model_name: Vertex AI embedding model to use
        """
        if project_id:
            aiplatform.init(project=project_id, location=location)
        else:
            # Use default credentials
            credentials, project = google.auth.default()
            aiplatform.init(project=project, location=location, credentials=credentials)
        
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.embedding_dimension = 768  # gecko models output 768-dimensional vectors
        
    def get_embeddings(self, texts: List[str], batch_size: int = 5) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts using Vertex AI.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once (Vertex AI has limits)
            
        Returns:
            List of numpy arrays containing embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.get_embeddings(batch)
            
            for embedding in batch_embeddings:
                embeddings.append(np.array(embedding.values, dtype=np.float32))
                
        return embeddings
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a single query string."""
        embeddings = self.model.get_embeddings([query])
        return np.array(embeddings[0].values, dtype=np.float32)


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
    
    def store_embedding(self, doc_id: str, embedding: np.ndarray, model_name: str = "textembedding-gecko@003"):
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
            INSERT OR REPLACE INTO doc_embeddings (doc_id, embedding, embedding_model, created_at)
            VALUES (?, ?, ?, ?)
        """, (doc_id, embedding_blob, model_name, time.time()))
        
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
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Find most similar documents using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        # Get all embeddings
        all_embeddings = self.get_all_embeddings()
        
        if not all_embeddings:
            return []
        
        # Compute cosine similarities
        similarities = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for doc_id, embedding in all_embeddings.items():
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
            alpha: Weight for vector similarity (0-1), remainder for FTS
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
    project_id: Optional[str] = None,
    batch_size: int = 5,
    limit: Optional[int] = None
):
    """
    Generate and store embeddings for all documents in the database.
    
    Args:
        db_path: Path to the SQLite database
        project_id: GCP project ID (optional, will use default if not provided)
        batch_size: Number of documents to process at once
        limit: Maximum number of documents to process (for testing)
    """
    print(f"Initializing Vertex AI embedding model...")
    embeddings_mgr = VertexEmbeddingsManager(project_id=project_id)
    vector_db = VectorDatabase(db_path)
    
    # Get all documents from the database
    conn = sqlite3.connect(str(db_path))
    query = "SELECT id, content, title FROM docs"
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
        for doc_id, content, title in batch:
            # Combine title and content for richer embeddings
            text = f"{title}\n{content}" if title else content
            # Truncate to avoid token limits (Vertex AI has ~3000 token limit)
            text = text[:8000] if text else ""
            texts.append(text)
            doc_ids.append(doc_id)
        
        # Generate embeddings
        try:
            embeddings = embeddings_mgr.get_embeddings(texts)
            
            # Store embeddings
            for doc_id, embedding in zip(doc_ids, embeddings):
                vector_db.store_embedding(doc_id, embedding)
            
            print(f"Processed {i+len(batch)}/{len(docs)} documents")
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {e}")
            continue
    
    print("Embedding generation complete!")
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for gptgrep database")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--project", help="GCP project ID (uses default if not provided)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for embedding generation")
    parser.add_argument("--limit", type=int, help="Limit number of documents (for testing)")
    
    args = parser.parse_args()
    
    create_embeddings_for_db(
        Path(args.db),
        project_id=args.project,
        batch_size=args.batch_size,
        limit=args.limit
    )