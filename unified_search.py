"""
Unified Search Service

This module provides a consistent search interface across all data sources
and search implementations in the codebase.
"""

import sqlite3
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
import html


class UnifiedSearchService:
    """Unified search service that works with SQLite FTS5 databases."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Ensure database and tables exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # Create main docs table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS docs(
                id TEXT PRIMARY KEY,
                conv_id TEXT,
                title TEXT,
                role TEXT,
                ts REAL,
                date TEXT,
                source TEXT,
                content TEXT,
                account TEXT,
                extra TEXT
            )
        """)

        # Create FTS5 virtual table if it doesn't exist
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                content, title, role, source, conv_id, ts, date, account, doc_id UNINDEXED, tokenize='porter'
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_date ON docs(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_account ON docs(account)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_ts ON docs(ts)")

        # Populate FTS table if empty
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0]
            if fts_count == 0:
                conn.execute("""
                    INSERT INTO docs_fts(rowid, content, title, role, source, conv_id, ts, date, account, doc_id)
                    SELECT rowid, content, title, role, source, conv_id, CAST(ts AS TEXT), date, COALESCE(account, 'default'), id
                    FROM docs
                """)
        except Exception:
            pass

        conn.commit()
        conn.close()

    def _normalize_text(self, text: Any) -> str:
        """Normalize text for consistent searching."""
        if text is None:
            return ""
        text = html.unescape(str(text))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _expand_query(self, query: str) -> str:
        """Expand query with wildcards for better matching."""
        parts = [p for p in query.replace('\u2013', ' ').replace('\u2014', ' ').split() if p]
        expanded = []
        for part in parts:
            if any(op in part for op in ['"', "'", "AND", "OR", "NOT", "NEAR", ":"]):
                expanded.append(part)
            elif len(part) > 2:
                expanded.append(part + '*')
            else:
                expanded.append(part)
        return " ".join(expanded)

    def search(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
        role: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort_by: str = "rank",
        account: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Unified search across all entities.

        Returns:
            Tuple of (results, total_count)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Build base query
        base_sql = """
            SELECT d.id, d.conv_id, d.title, d.role, d.date, d.source, d.ts, d.account,
                   snippet(docs_fts, 0, '<mark>', '</mark>', ' … ', 12) as snip
            FROM docs_fts
            JOIN docs d ON d.rowid = docs_fts.rowid
            WHERE docs_fts MATCH ?
        """

        params = []
        search_query = self._expand_query(query) if query else ""
        params.append(search_query)

        # Add filters
        if provider:
            if provider == "claude":
                base_sql += " AND d.source LIKE '%anthropic%'"
            elif provider == "chatgpt":
                base_sql += " AND d.source LIKE '%chatgpt%'"

        if role:
            if role == "assistant":
                base_sql += " AND (d.role = 'assistant' OR d.role = 'system')"
            else:
                base_sql += " AND d.role = ?"
                params.append(role)

        if date_from:
            base_sql += " AND (d.date IS NOT NULL AND d.date >= ?)"
            params.append(date_from)

        if date_to:
            base_sql += " AND (d.date IS NOT NULL AND d.date <= ?)"
            params.append(date_to)

        if account:
            base_sql += " AND d.account = ?"
            params.append(account)

        # Get total count
        count_sql = base_sql.replace(
            "SELECT d.id, d.conv_id, d.title, d.role, d.date, d.source, d.ts, d.account, snippet(docs_fts, 0, '<mark>', '</mark>', ' … ', 12) as snip",
            "SELECT COUNT(*)"
        )

        total_count = conn.execute(count_sql, tuple(params)).fetchone()[0]

        # Add sorting and pagination
        if sort_by == "newest":
            base_sql += " ORDER BY (d.date IS NULL), d.date DESC, rank"
        elif sort_by == "oldest":
            base_sql += " ORDER BY (d.date IS NULL), d.date ASC, rank"
        else:
            base_sql += " ORDER BY rank"

        base_sql += f" LIMIT {limit} OFFSET {offset}"

        rows = conn.execute(base_sql, tuple(params)).fetchall()
        conn.close()

        # Format results
        results = []
        for row in rows:
            result = {
                "id": row["id"],
                "conv_id": row["conv_id"],
                "title": row["title"] or "Untitled",
                "role": row["role"],
                "date": row["date"],
                "source": row["source"],
                "timestamp": row["ts"],
                "account": row["account"],
                "snippet": row["snip"],
                "relevance_score": 1.0  # Could be enhanced with actual scoring
            }
            results.append(result)

        return results, total_count

    def get_conversation_context(self, conv_id: str) -> List[Dict[str, Any]]:
        """Get full conversation context for a given conversation ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            "SELECT title, role, date, ts, content, source FROM docs WHERE conv_id=? ORDER BY ts, rowid",
            (conv_id,)
        ).fetchall()

        conn.close()

        messages = []
        for row in rows:
            messages.append({
                "role": row["role"],
                "date": row["date"],
                "content": row["content"],
                "timestamp": row["ts"],
                "source": row["source"]
            })

        return messages

    def get_conversation_title(self, conv_id: str) -> str:
        """Get conversation title."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            "SELECT title FROM docs WHERE conv_id=? LIMIT 1",
            (conv_id,)
        ).fetchone()

        conn.close()

        return row["title"] if row else "Untitled Conversation"

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            total_docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            total_conversations = conn.execute("SELECT COUNT(DISTINCT conv_id) FROM docs").fetchone()[0]
            sources = conn.execute("SELECT source, COUNT(*) as count FROM docs GROUP BY source ORDER BY count DESC").fetchall()
            accounts = conn.execute("SELECT account, COUNT(*) as count FROM docs GROUP BY account ORDER BY count DESC").fetchall()

            stats = {
                "total_documents": total_docs,
                "total_conversations": total_conversations,
                "sources": {row["source"]: row["count"] for row in sources},
                "accounts": {row["account"]: row["count"] for row in accounts}
            }
        except Exception as e:
            stats = {"error": str(e)}

        conn.close()
        return stats

    def add_document(self, doc_data: Dict[str, Any]) -> bool:
        """Add a single document to the search index."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Extract and normalize data
            doc_id = doc_data.get("id")
            if not doc_id:
                return False

            conv_id = doc_data.get("conv_id", "")
            title = doc_data.get("title", "")
            role = doc_data.get("role", "")
            ts = doc_data.get("ts", 0.0)
            source = doc_data.get("source", "")
            content = self._normalize_text(doc_data.get("content", ""))
            account = doc_data.get("account", "default")
            extra = json.dumps(doc_data.get("extra", {})) if doc_data.get("extra") else None

            # Convert timestamp to date
            date_str = None
            if ts:
                try:
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    date_str = dt.strftime('%Y-%m-%d')
                except Exception:
                    date_str = None

            # Insert into main table
            conn.execute("""
                INSERT OR REPLACE INTO docs(id, conv_id, title, role, ts, date, source, content, account, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (doc_id, conv_id, title, role, ts, date_str, source, content, account, extra))

            # Insert into FTS table
            conn.execute("""
                INSERT INTO docs_fts(rowid, content, title, role, source, conv_id, ts, date, account, doc_id)
                VALUES ((SELECT rowid FROM docs WHERE id=?),?,?,?,?,?,?,?,?,?)
            """, (doc_id, content, title, role, source, conv_id, str(ts), date_str, account, doc_id))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error adding document: {e}")
            return False

    def bulk_add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add multiple documents to the search index. Returns number added."""
        added = 0
        conn = sqlite3.connect(self.db_path)

        try:
            for doc_data in documents:
                if self.add_document(doc_data):
                    added += 1

            conn.close()
            return added

        except Exception as e:
            print(f"Error in bulk add: {e}")
            conn.close()
            return added




