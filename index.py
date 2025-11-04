\
import argparse, json, os, re, sqlite3, time, html
from pathlib import Path

def ensure_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, conv_id TEXT, title TEXT, role TEXT, ts REAL, source TEXT, extra TEXT, content TEXT)")
    # FTS5 with trigram/porter options vary by build; keep minimal
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(content, title, role, source, conv_id, ts, doc_id UNINDEXED, tokenize='porter')")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_ts ON docs(ts)")
    return conn

def insert_doc(conn, *, id, conv_id, title, role, ts, source, extra, content):
    conn.execute(
        "INSERT OR REPLACE INTO docs(id, conv_id, title, role, ts, source, extra, content) VALUES (?,?,?,?,?,?,?,?)",
        (id, conv_id, title, role, ts, source, json.dumps(extra) if extra else None, content)
    )
    conn.execute(
        "INSERT INTO docs_fts(rowid, content, title, role, source, conv_id, ts, doc_id) VALUES ((SELECT rowid FROM docs WHERE id=?),?,?,?,?,?,?,?)",
        (id, content or "", title or "", role or "", source or "", conv_id or "", str(ts) if ts is not None else "", id)
    )

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
                ts = m.get("create_time") or m.get("timestamp")
                doc_id = f"{conv_id}:{i}:{role}:{abs(hash(str(m)))%10**9}"
                yield {
                    "id": doc_id,
                    "conv_id": conv_id,
                    "title": title,
                    "role": role,
                    "ts": ts if ts is not None else 0.0,
                    "source": "conversations.json",
                    "extra": {"raw_id": m.get("id")},
                    "content": norm_text(content)
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
                ts = m.get("create_time") or m.get("timestamp")
                doc_id = f"{conv_id}:{i}:{role}:{abs(hash(str(m)))%10**9}"
                i += 1
                yield {
                    "id": doc_id,
                    "conv_id": conv_id,
                    "title": title,
                    "role": role,
                    "ts": ts if ts is not None else 0.0,
                    "source": "conversations.json",
                    "extra": {"node": k},
                    "content": norm_text(content)
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
                "content": norm_text(content)
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
            "content": norm_text(content)
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
        "content": norm_text(json.dumps(raw, ensure_ascii=False))
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
        "content": raw  # searchable blob; FTS will tokenize
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Path to folder with ChatGPT export files")
    ap.add_argument("--out", required=True, help="Output folder for index (creates chatgpt.db)")
    args = ap.parse_args()

    export_dir = Path(args.export)
    out_dir = Path(args.out)
    db_path = out_dir / "chatgpt.db"
    conn = ensure_db(db_path)

    parsers = []
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
            for doc in gen:
                insert_doc(conn, **doc)
                total += 1
    print(f"Indexed {total} docs into {db_path}")

if __name__ == "__main__":
    main()
