import argparse, json, os, re, sqlite3, time, html
from pathlib import Path
from datetime import datetime, timezone
try:
    # Optional AnthropIc data ingestion
    from anthropic_ingest import (
        parse_anthropic_conversations,
        parse_anthropic_projects,
        parse_anthropic_users,
    )
except Exception:
    parse_anthropic_conversations = None
    parse_anthropic_projects = None
    parse_anthropic_users = None

def ensure_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Enhanced schema with better date handling and metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS docs(
            id TEXT PRIMARY KEY,
            conv_id TEXT,
            title TEXT,
            role TEXT,
            ts REAL,
            date TEXT,  -- ISO date string for easier querying
            year INTEGER,
            month INTEGER,
            day INTEGER,
            source TEXT,
            extra TEXT,
            content TEXT,
            msg_index INTEGER,
            word_count INTEGER
        )
    """)
    
    # FTS5 with additional metadata
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts 
        USING fts5(
            content, 
            title, 
            role, 
            source, 
            conv_id, 
            ts, 
            date,
            doc_id UNINDEXED, 
            tokenize='porter'
        )
    """)
    
    # Create indexes for efficient date queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_ts ON docs(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_date ON docs(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_year_month ON docs(year, month)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_conv_id ON docs(conv_id)")
    
    # Create conversation summary table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations(
            conv_id TEXT PRIMARY KEY,
            title TEXT,
            first_message_date TEXT,
            last_message_date TEXT,
            message_count INTEGER,
            total_word_count INTEGER,
            participants TEXT,  -- JSON array of unique roles
            source TEXT
        )
    """)
    
    return conn

def ts_to_date_parts(ts):
    """Convert timestamp to date components."""
    if ts is None or ts == 0:
        return None, None, None, None
    
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        date_str = dt.strftime('%Y-%m-%d')
        return date_str, dt.year, dt.month, dt.day
    except:
        return None, None, None, None

def count_words(text):
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())

def insert_doc(conn, *, id, conv_id, title, role, ts, source, extra, content, msg_index=0):
    # Extract date components
    date_str, year, month, day = ts_to_date_parts(ts)
    word_count = count_words(content)
    
    conn.execute(
        """INSERT OR REPLACE INTO docs(
            id, conv_id, title, role, ts, date, year, month, day, 
            source, extra, content, msg_index, word_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (id, conv_id, title, role, ts, date_str, year, month, day, 
         source, json.dumps(extra) if extra else None, content, msg_index, word_count)
    )
    
    conn.execute(
        """INSERT INTO docs_fts(
            rowid, content, title, role, source, conv_id, ts, date, doc_id
        ) VALUES (
            (SELECT rowid FROM docs WHERE id=?),?,?,?,?,?,?,?,?
        )""",
        (id, content or "", title or "", role or "", source or "", 
         conv_id or "", str(ts) if ts is not None else "", date_str or "", id)
    )

def update_conversation_summary(conn):
    """Create conversation summaries with metadata."""
    print("Creating conversation summaries...")
    
    conn.execute("DELETE FROM conversations")
    
    conn.execute("""
        INSERT INTO conversations (
            conv_id, 
            title,
            first_message_date,
            last_message_date,
            message_count,
            total_word_count,
            participants,
            source
        )
        SELECT 
            conv_id,
            MAX(title) as title,
            MIN(date) as first_message_date,
            MAX(date) as last_message_date,
            COUNT(*) as message_count,
            SUM(word_count) as total_word_count,
            json_group_array(DISTINCT role) as participants,
            MAX(source) as source
        FROM docs
        WHERE conv_id IS NOT NULL
        GROUP BY conv_id
    """)

def norm_text(x):
    if x is None:
        return None
    x = html.unescape(str(x))
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def parse_conversations_json(p: Path):
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(raw, list):
        convs = raw
    elif isinstance(raw, dict):
        convs = raw.get("conversations", [])
    else:
        convs = []
    
    for c in convs:
        if not isinstance(c, dict):
            continue
        
        conv_id = c.get("id") or c.get("conversation_id") or c.get("uuid")
        title = c.get("title") or ""
        
        # Track conversation date range
        conv_create_time = c.get("create_time") or c.get("created_at")
        conv_update_time = c.get("update_time") or c.get("updated_at")
        
        # handle both flat 'messages' or 'mapping'-style payloads
        msgs = c.get("messages")
        if msgs and isinstance(msgs, list):
            for i, m in enumerate(msgs):
                role = m.get("role") or m.get("author", {}).get("role")
                content = m.get("content") or m.get("text") or m.get("parts") or ""
                if isinstance(content, dict):
                    content = content.get("content") or content.get("text") or ""
                if isinstance(content, list):
                    content = " ".join([str(x) for x in content])
                
                ts = m.get("create_time") or m.get("timestamp") or conv_create_time
                
                # Extract additional metadata
                extra = {
                    "raw_id": m.get("id"),
                    "status": m.get("status"),
                    "model_slug": m.get("metadata", {}).get("model_slug"),
                    "finish_details": m.get("metadata", {}).get("finish_details"),
                }
                
                doc_id = f"{conv_id}:{i}:{role}:{abs(hash(str(m)))%10**9}"
                yield {
                    "id": doc_id,
                    "conv_id": conv_id,
                    "title": title,
                    "role": role,
                    "ts": ts if ts is not None else 0.0,
                    "source": "conversations.json",
                    "extra": extra,
                    "content": norm_text(content),
                    "msg_index": i
                }
        elif c.get("mapping"):
            # older exports used 'mapping' dict with message nodes
            mapping = c["mapping"]
            i = 0
            for k, node in mapping.items():
                m = node.get("message") or {}
                role = (m.get("author") or {}).get("role")
                # content can be {content_type:..., parts:[...]} or array of blocks
                content = ""
                if "content" in m:
                    mc = m["content"]
                    if isinstance(mc, dict) and "parts" in mc:
                        content = " ".join([str(x) for x in mc["parts"]])
                    elif isinstance(mc, list):
                        content = " ".join([str(x) for x in mc])
                    else:
                        content = str(mc)
                
                ts = m.get("create_time") or m.get("timestamp") or conv_create_time
                
                # Extract additional metadata
                extra = {
                    "node": k,
                    "status": m.get("status"),
                    "model_slug": m.get("metadata", {}).get("model_slug"),
                }
                
                doc_id = f"{conv_id}:{i}:{role}:{abs(hash(str(m)))%10**9}"
                i += 1
                yield {
                    "id": doc_id,
                    "conv_id": conv_id,
                    "title": title,
                    "role": role,
                    "ts": ts if ts is not None else 0.0,
                    "source": "conversations.json",
                    "extra": extra,
                    "content": norm_text(content),
                    "msg_index": i
                }

def parse_shared_conversations_json(p: Path):
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("shared_conversations", [])
    else:
        items = []
    
    for c in items:
        if not isinstance(c, dict):
            continue
        
        conv_id = c.get("id") or c.get("uuid") or f"shared:{abs(hash(json.dumps(c)))%10**9}"
        title = c.get("title") or ""
        msgs = c.get("messages") or []
        
        for i, m in enumerate(msgs):
            role = m.get("role") or m.get("author", {}).get("role")
            content = m.get("content") or m.get("text") or ""
            ts = m.get("create_time") or m.get("timestamp")
            doc_id = f"{conv_id}:{i}:{role}:{abs(hash(str(m)))%10**9}"
            yield {
                "id": doc_id,
                "conv_id": conv_id,
                "title": title,
                "role": role,
                "ts": ts if ts is not None else 0.0,
                "source": "shared_conversations.json",
                "extra": None,
                "content": norm_text(content),
                "msg_index": i
            }

def parse_message_feedback_json(p: Path):
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("message_feedback", [])
    else:
        items = []
    
    for i, fb in enumerate(items):
        conv_id = fb.get("conversation_id") or fb.get("conv_id") or f"feedback:{i}"
        label = fb.get("label") or fb.get("rating") or "feedback"
        comment = fb.get("text") or fb.get("comment") or ""
        ts = fb.get("create_time") or fb.get("timestamp") or 0.0
        doc_id = f"{conv_id}:feedback:{i}:{abs(hash(str(fb)))%10**9}"
        content = f"[{label}] {comment}"
        yield {
            "id": doc_id,
            "conv_id": conv_id,
            "title": f"Feedback: {label}",
            "role": "user",
            "ts": ts,
            "source": "message_feedback.json",
            "extra": fb,
            "content": norm_text(content),
            "msg_index": i
        }

def parse_user_json(p: Path):
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    doc_id = f"user:{abs(hash(json.dumps(raw)))%10**9}"
    yield {
        "id": doc_id,
        "conv_id": "user_profile",
        "title": "User Profile",
        "role": "system",
        "ts": time.time(),
        "source": "user.json",
        "extra": None,
        "content": norm_text(json.dumps(raw, ensure_ascii=False)),
        "msg_index": 0
    }

def parse_chat_html_as_blob(p: Path):
    # We store the entire HTML as a single doc for search.
    raw = p.read_text(encoding="utf-8", errors="ignore")
    doc_id = f"chat_html:{abs(hash(raw))%10**9}"
    yield {
        "id": doc_id,
        "conv_id": "chat_html",
        "title": "chat.html",
        "role": "system",
        "ts": time.time(),
        "source": "chat.html",
        "extra": None,
        "content": raw,  # searchable blob; FTS will tokenize
        "msg_index": 0
    }

def print_statistics(conn):
    """Print indexing statistics."""
    print("\n" + "="*60)
    print("INDEXING STATISTICS")
    print("="*60)
    
    # Total documents
    cursor = conn.execute("SELECT COUNT(*) FROM docs")
    total_docs = cursor.fetchone()[0]
    print(f"Total messages indexed: {total_docs:,}")
    
    # Conversations
    cursor = conn.execute("SELECT COUNT(*) FROM conversations")
    total_convs = cursor.fetchone()[0]
    print(f"Total conversations: {total_convs:,}")
    
    # Date range
    cursor = conn.execute("SELECT MIN(date), MAX(date) FROM docs WHERE date IS NOT NULL")
    min_date, max_date = cursor.fetchone()
    if min_date and max_date:
        print(f"Date range: {min_date} to {max_date}")
    
    # Messages by year
    cursor = conn.execute("""
        SELECT year, COUNT(*) as cnt 
        FROM docs 
        WHERE year IS NOT NULL 
        GROUP BY year 
        ORDER BY year DESC 
        LIMIT 10
    """)
    year_stats = cursor.fetchall()
    if year_stats:
        print("\nMessages by year:")
        for year, count in year_stats:
            print(f"  {year}: {count:,} messages")
    
    # Top conversations by message count
    cursor = conn.execute("""
        SELECT title, message_count, first_message_date
        FROM conversations
        ORDER BY message_count DESC
        LIMIT 5
    """)
    top_convs = cursor.fetchall()
    if top_convs:
        print("\nTop conversations by message count:")
        for title, count, date in top_convs:
            title_display = title[:50] + "..." if len(title or "") > 50 else (title or "Untitled")
            print(f"  {title_display}: {count} messages ({date})")
    
    # Roles distribution
    cursor = conn.execute("""
        SELECT role, COUNT(*) as cnt 
        FROM docs 
        GROUP BY role 
        ORDER BY cnt DESC
    """)
    role_stats = cursor.fetchall()
    if role_stats:
        print("\nMessages by role:")
        for role, count in role_stats:
            print(f"  {role}: {count:,} messages")
    
    print("="*60 + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Path to folder with ChatGPT export files")
    ap.add_argument("--out", required=True, help="Output folder for index (creates chatgpt.db)")
    ap.add_argument("--stats", action="store_true", help="Show statistics after indexing")
    args = ap.parse_args()
    
    export_dir = Path(args.export)
    out_dir = Path(args.out)
    db_path = out_dir / "chatgpt.db"
    conn = ensure_db(db_path)
    
    parsers = []

    # Detect AnthropIc dataset shape and prefer those parsers when present
    anthropic_mode = False
    if (export_dir / "projects.json").exists() or (export_dir / "users.json").exists():
        anthropic_mode = True

    # Support layout: data/{openai,anthropic}
    if export_dir.is_dir() and (export_dir / "openai").exists() or (export_dir / "anthropic").exists():
        # OpenAI
        openai_dir = export_dir / "openai"
        if openai_dir.exists():
            p = openai_dir / "conversations.json"
            if p.exists():
                parsers.append(("conversations.json", parse_conversations_json(p)))
            p = openai_dir / "shared_conversations.json"
            if p.exists():
                parsers.append(("shared_conversations.json", parse_shared_conversations_json(p)))
            p = openai_dir / "message_feedback.json"
            if p.exists():
                parsers.append(("message_feedback.json", parse_message_feedback_json(p)))
            p = openai_dir / "user.json"
            if p.exists():
                parsers.append(("user.json", parse_user_json(p)))
            p = openai_dir / "chat.html"
            if p.exists():
                parsers.append(("chat.html", parse_chat_html_as_blob(p)))
        # Anthropic
        anth_dir = export_dir / "anthropic"
        if anth_dir.exists() and parse_anthropic_conversations:
            p = anth_dir / "conversations.json"
            if p.exists():
                parsers.append(("anthropic.conversations.json", parse_anthropic_conversations(p)))
            p = anth_dir / "projects.json"
            if p.exists() and parse_anthropic_projects:
                parsers.append(("anthropic.projects.json", parse_anthropic_projects(p)))
            p = anth_dir / "users.json"
            if p.exists() and parse_anthropic_users:
                parsers.append(("anthropic.users.json", parse_anthropic_users(p)))
    elif anthropic_mode and parse_anthropic_conversations and parse_anthropic_projects and parse_anthropic_users:
        p = export_dir / "conversations.json"
        if p.exists():
            parsers.append(("anthropic.conversations.json", parse_anthropic_conversations(p)))
        p = export_dir / "projects.json"
        if p.exists():
            parsers.append(("anthropic.projects.json", parse_anthropic_projects(p)))
        p = export_dir / "users.json"
        if p.exists():
            parsers.append(("anthropic.users.json", parse_anthropic_users(p)))
    else:
        # Default ChatGPT export parsers
        p = export_dir / "conversations.json"
        if p.exists():
            parsers.append(("conversations.json", parse_conversations_json(p)))
        p = export_dir / "shared_conversations.json"
        if p.exists():
            parsers.append(("shared_conversations.json", parse_shared_conversations_json(p)))
        p = export_dir / "message_feedback.json"
        if p.exists():
            parsers.append(("message_feedback.json", parse_message_feedback_json(p)))
        p = export_dir / "user.json"
        if p.exists():
            parsers.append(("user.json", parse_user_json(p)))
        p = export_dir / "chat.html"
        if p.exists():
            parsers.append(("chat.html", parse_chat_html_as_blob(p)))
    
    total = 0
    with conn:
        for name, gen in parsers:
            print(f"Processing {name}...")
            for doc in gen:
                insert_doc(conn, **doc)
                total += 1
                if total % 1000 == 0:
                    print(f"  Processed {total:,} messages...")
    
    # Create conversation summaries
    update_conversation_summary(conn)
    
    print(f"\nIndexed {total:,} docs into {db_path}")
    
    if args.stats:
        print_statistics(conn)
    
    conn.close()

if __name__ == "__main__":
    main()