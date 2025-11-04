import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import json

from local_embeddings import LocalEmbeddingsManager, VectorDatabase

def search_fts_with_date(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> List[Tuple[str, str, str, str, str]]:
    """
    Perform FTS5 full-text search with optional date filtering.
    
    Returns:
        List of (doc_id, snippet, title, date, role) tuples
    """
    # Build the query
    base_query = """
        SELECT 
            f.doc_id,
            snippet(docs_fts, 0, '[', ']', '...', 30) as snippet,
            f.title,
            d.date,
            d.role,
            d.word_count
        FROM docs_fts f
        JOIN docs d ON f.doc_id = d.id
        WHERE docs_fts MATCH ?
    """
    
    params = [query]
    
    # Add date filters if provided
    if date_from:
        base_query += " AND d.date >= ?"
        params.append(date_from)
    if date_to:
        base_query += " AND d.date <= ?"
        params.append(date_to)
    
    base_query += " ORDER BY rank LIMIT ?"
    params.append(limit)
    
    cursor = conn.execute(base_query, params)
    return cursor.fetchall()

def search_vector_with_metadata(
    conn: sqlite3.Connection,
    vector_db: VectorDatabase,
    embeddings_mgr: LocalEmbeddingsManager,
    query: str,
    limit: int = 10,
    threshold: float = 0.0,
    date_filter: Optional[Tuple[str, str]] = None
) -> List[Tuple[str, float, str, str, str, str]]:
    """
    Perform vector similarity search with metadata.
    
    Returns:
        List of (doc_id, similarity_score, content_preview, title, date, role) tuples
    """
    # Generate query embedding
    query_embedding = embeddings_mgr.get_query_embedding(query)
    
    # Search for similar documents
    results = vector_db.search_similar(
        query_embedding,
        top_k=limit,
        threshold=threshold,
        date_filter=date_filter
    )
    
    # Fetch metadata for results
    enhanced_results = []
    for doc_id, score in results:
        cursor = conn.execute("""
            SELECT content, title, date, role, word_count
            FROM docs 
            WHERE id = ?
        """, (doc_id,))
        row = cursor.fetchone()
        
        if row:
            content, title, date, role, word_count = row
            # Create snippet from content
            snippet = content[:200] + "..." if len(content) > 200 else content
            enhanced_results.append((doc_id, score, snippet, title, date, role))
    
    return enhanced_results

def search_conversations(
    conn: sqlite3.Connection,
    query: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 10
) -> List[Tuple]:
    """
    Search for conversations with optional filters.
    """
    base_query = """
        SELECT 
            conv_id,
            title,
            first_message_date,
            last_message_date,
            message_count,
            total_word_count
        FROM conversations
        WHERE 1=1
    """
    
    params = []
    
    if query:
        base_query += " AND title LIKE ?"
        params.append(f"%{query}%")
    
    if date_from:
        base_query += " AND last_message_date >= ?"
        params.append(date_from)
    
    if date_to:
        base_query += " AND first_message_date <= ?"
        params.append(date_to)
    
    base_query += " ORDER BY last_message_date DESC LIMIT ?"
    params.append(limit)
    
    cursor = conn.execute(base_query, params)
    return cursor.fetchall()

def format_results(results: List[Tuple], search_type: str = "hybrid", show_metadata: bool = True):
    """Format and print search results with metadata."""
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(results)} results ({search_type} search):")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        if search_type == "fts":
            doc_id, snippet, title, date, role, word_count = result
            print(f"{i}. {title or 'Untitled Conversation'}")
            if show_metadata:
                print(f"   Date: {date or 'Unknown'} | Role: {role} | Words: {word_count}")
            print(f"   ID: {doc_id[:30]}...")
            print(f"   {snippet}\n")
        
        elif search_type == "vector":
            doc_id, score, snippet, title, date, role = result
            print(f"{i}. {title or 'Untitled Conversation'} (Score: {score:.4f})")
            if show_metadata:
                print(f"   Date: {date or 'Unknown'} | Role: {role}")
            print(f"   ID: {doc_id[:30]}...")
            print(f"   {snippet}\n")
        
        elif search_type == "conversations":
            conv_id, title, first_date, last_date, msg_count, word_count = result
            print(f"{i}. {title or 'Untitled Conversation'}")
            if show_metadata:
                print(f"   Period: {first_date} to {last_date}")
                print(f"   Messages: {msg_count} | Total words: {word_count:,}")
            print(f"   ID: {conv_id[:30]}...\n")

def main():
    parser = argparse.ArgumentParser(description="Search gptgrep with local embeddings")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--db", default="./index/chatgpt.db", help="Path to database")
    parser.add_argument("--limit", type=int, default=10, help="Number of results")
    parser.add_argument("--mode", choices=["fts", "vector", "hybrid", "conversations"],
                      default="hybrid", help="Search mode")
    parser.add_argument("--alpha", type=float, default=0.5,
                      help="Weight for vector search in hybrid mode (0-1)")
    parser.add_argument("--model-type", choices=["ollama", "sentence-transformers"],
                      default="sentence-transformers", help="Embedding model type")
    parser.add_argument("--model", help="Specific model name")
    parser.add_argument("--threshold", type=float, default=0.0,
                      help="Minimum similarity threshold for vector search (0-1)")
    parser.add_argument("--date-from", help="Filter results from date (YYYY-MM-DD)")
    parser.add_argument("--date-to", help="Filter results to date (YYYY-MM-DD)")
    parser.add_argument("--no-metadata", action="store_true",
                      help="Hide metadata in results")
    parser.add_argument("--stats", action="store_true",
                      help="Show database statistics")
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run 'python index_enhanced.py' first to create the index.")
        return
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    # Show statistics if requested
    if args.stats:
        from index_enhanced import print_statistics
        print_statistics(conn)
        return
    
    # Search conversations mode
    if args.mode == "conversations":
        results = search_conversations(
            conn,
            query=args.query,
            date_from=args.date_from,
            date_to=args.date_to,
            limit=args.limit
        )
        format_results(results, "conversations", not args.no_metadata)
        conn.close()
        return
    
    # Regular search modes require a query
    if not args.query:
        print("Error: Query required for this search mode")
        parser.print_help()
        return
    
    # Date filter tuple for vector search
    date_filter = None
    if args.date_from or args.date_to:
        date_filter = (
            args.date_from or "1900-01-01",
            args.date_to or "2100-12-31"
        )
    
    if args.mode == "fts":
        # FTS-only search with date filtering
        results = search_fts_with_date(
            conn, args.query, args.limit,
            args.date_from, args.date_to
        )
        format_results(results, "fts", not args.no_metadata)
    
    elif args.mode == "vector":
        # Vector-only search
        print(f"Initializing {args.model_type} embeddings...")
        embeddings_mgr = LocalEmbeddingsManager(
            model_type=args.model_type,
            model_name=args.model
        )
        vector_db = VectorDatabase(db_path)
        
        results = search_vector_with_metadata(
            conn, vector_db, embeddings_mgr,
            args.query, args.limit, args.threshold,
            date_filter
        )
        format_results(results, "vector", not args.no_metadata)
    
    else:  # hybrid
        # Hybrid search combining FTS and vector
        print(f"Initializing {args.model_type} embeddings for hybrid search...")
        embeddings_mgr = LocalEmbeddingsManager(
            model_type=args.model_type,
            model_name=args.model
        )
        vector_db = VectorDatabase(db_path)
        
        # Get FTS results
        fts_results = search_fts_with_date(
            conn, args.query, args.limit*2,
            args.date_from, args.date_to
        )
        fts_doc_ids = [doc_id for doc_id, _, _, _, _, _ in fts_results]
        
        # Generate query embedding
        query_embedding = embeddings_mgr.get_query_embedding(args.query)
        
        # Perform hybrid search
        hybrid_results = vector_db.hybrid_search(
            query_embedding,
            fts_doc_ids,
            alpha=args.alpha,
            top_k=args.limit
        )
        
        # Fetch document details for hybrid results
        enhanced_results = []
        for doc_id, score in hybrid_results:
            cursor = conn.execute("""
                SELECT content, title, date, role, word_count
                FROM docs WHERE id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            
            if row:
                content, title, date, role, word_count = row
                snippet = content[:200] + "..." if len(content) > 200 else content
                enhanced_results.append((doc_id, score, snippet, title, date, role))
        
        format_results(enhanced_results, "vector", not args.no_metadata)
    
    conn.close()

if __name__ == "__main__":
    main()