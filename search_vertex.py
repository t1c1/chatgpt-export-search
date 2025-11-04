import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
import json

from vertex_embeddings import VertexEmbeddingsManager, VectorDatabase

def search_fts(conn: sqlite3.Connection, query: str, limit: int = 10) -> List[Tuple[str, str, str]]:
    """
    Perform FTS5 full-text search.
    
    Returns:
        List of (doc_id, snippet, title) tuples
    """
    cursor = conn.execute("""
        SELECT 
            doc_id,
            snippet(docs_fts, 0, '[', ']', '...', 30) as snippet,
            title
        FROM docs_fts
        WHERE docs_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (query, limit))
    
    return cursor.fetchall()

def search_vector(
    vector_db: VectorDatabase,
    embeddings_mgr: VertexEmbeddingsManager,
    query: str,
    limit: int = 10,
    threshold: float = 0.0
) -> List[Tuple[str, float]]:
    """
    Perform vector similarity search using Vertex AI embeddings.
    
    Returns:
        List of (doc_id, similarity_score) tuples
    """
    # Generate query embedding
    query_embedding = embeddings_mgr.get_query_embedding(query)
    
    # Search for similar documents
    results = vector_db.search_similar(query_embedding, top_k=limit, threshold=threshold)
    
    return results

def search_hybrid(
    conn: sqlite3.Connection,
    vector_db: VectorDatabase,
    embeddings_mgr: VertexEmbeddingsManager,
    query: str,
    limit: int = 10,
    alpha: float = 0.5
) -> List[Tuple[str, float, str, str]]:
    """
    Perform hybrid search combining FTS5 and vector similarity.
    
    Args:
        conn: SQLite connection
        vector_db: Vector database instance
        embeddings_mgr: Vertex AI embeddings manager
        query: Search query
        limit: Number of results
        alpha: Weight for vector search (0-1)
    
    Returns:
        List of (doc_id, score, snippet, title) tuples
    """
    # Get FTS results
    fts_results = search_fts(conn, query, limit=limit*2)
    fts_doc_ids = [doc_id for doc_id, _, _ in fts_results]
    
    # Generate query embedding
    query_embedding = embeddings_mgr.get_query_embedding(query)
    
    # Perform hybrid search
    hybrid_results = vector_db.hybrid_search(
        query_embedding,
        fts_doc_ids,
        alpha=alpha,
        top_k=limit
    )
    
    # Fetch document details
    results = []
    for doc_id, score in hybrid_results:
        cursor = conn.execute("""
            SELECT title, content FROM docs WHERE id = ?
        """, (doc_id,))
        row = cursor.fetchone()
        
        if row:
            title, content = row
            # Create snippet from content
            snippet = content[:200] + "..." if len(content) > 200 else content
            results.append((doc_id, score, snippet, title))
    
    return results

def format_results(results: List[Tuple], search_type: str = "hybrid"):
    """Format and print search results."""
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(results)} results ({search_type} search):")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        if search_type == "fts":
            doc_id, snippet, title = result
            print(f"{i}. [{doc_id[:20]}...] {title or 'No title'}")
            print(f"   {snippet}\n")
        elif search_type == "vector":
            doc_id, score = result
            print(f"{i}. [{doc_id[:20]}...] Score: {score:.4f}")
        else:  # hybrid
            doc_id, score, snippet, title = result
            print(f"{i}. [{doc_id[:20]}...] {title or 'No title'} (Score: {score:.4f})")
            print(f"   {snippet}\n")

def main():
    parser = argparse.ArgumentParser(description="Search gptgrep with Vertex AI")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--db", default="./index/chatgpt.db", help="Path to database")
    parser.add_argument("--limit", type=int, default=10, help="Number of results")
    parser.add_argument("--mode", choices=["fts", "vector", "hybrid"], default="hybrid",
                      help="Search mode: fts (text only), vector (semantic), or hybrid")
    parser.add_argument("--alpha", type=float, default=0.5,
                      help="Weight for vector search in hybrid mode (0-1)")
    parser.add_argument("--project", help="GCP project ID (uses default if not provided)")
    parser.add_argument("--threshold", type=float, default=0.0,
                      help="Minimum similarity threshold for vector search (0-1)")
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run 'python index.py' first to create the index.")
        return
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    if args.mode == "fts":
        # FTS-only search
        results = search_fts(conn, args.query, args.limit)
        format_results(results, "fts")
    
    elif args.mode == "vector":
        # Vector-only search
        print("Initializing Vertex AI...")
        embeddings_mgr = VertexEmbeddingsManager(project_id=args.project)
        vector_db = VectorDatabase(db_path)
        
        results = search_vector(
            vector_db, embeddings_mgr, args.query, 
            args.limit, args.threshold
        )
        
        # Fetch titles for vector results
        enhanced_results = []
        for doc_id, score in results:
            cursor = conn.execute("SELECT title FROM docs WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            title = row[0] if row else "No title"
            print(f"[{doc_id[:20]}...] {title} (Score: {score:.4f})")
    
    else:  # hybrid
        # Hybrid search
        print("Initializing Vertex AI for hybrid search...")
        embeddings_mgr = VertexEmbeddingsManager(project_id=args.project)
        vector_db = VectorDatabase(db_path)
        
        results = search_hybrid(
            conn, vector_db, embeddings_mgr,
            args.query, args.limit, args.alpha
        )
        format_results(results, "hybrid")
    
    conn.close()

if __name__ == "__main__":
    main()