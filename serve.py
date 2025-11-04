\
import argparse, sqlite3, sys, subprocess, os, time
from pathlib import Path
from flask import Flask, request, render_template_string, redirect, url_for, flash

TEMPLATE = """
<!doctype html>
<title>AI Conversation Search</title>
<style>
:root{ --bg:#ffffff; --text:#111; --muted:#666; --card:#fafafa; --border:#e3e3e3; --accent:#0b74ff; --mark:#fffd7a; --success:#10a37f; --warning:#ff8c00; --error:#ef4444; }
.dark{ --bg:#0f1115; --text:#eef1f6; --muted:#9aa4b2; --card:#171a21; --border:#262b36; --accent:#6aa8ff; --mark:#3f3a00; --success:#34d399; --warning:#fbbf24; --error:#f87171; }
* { box-sizing:border-box; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:0; background:var(--bg); color:var(--text); line-height:1.5; }
.container{ max-width: 1200px; margin: 0 auto; padding: 20px; }
.search-container { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:24px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
input[type=text]{ width: 100%; padding: 12px 16px; font-size: 16px; background:var(--card); color:var(--text); border:2px solid var(--border); border-radius:8px; transition:border-color 0.2s; }
input[type=text]:focus{ outline:none; border-color:var(--accent); box-shadow:0 0 0 3px rgba(11,116,255,0.1); }
input[type=date], input[type=number], select{ padding: 10px 12px; font-size: 14px; background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:6px; }
button{ padding: 12px 20px; font-size: 14px; background:var(--accent); color:white; border:none; border-radius:8px; cursor:pointer; font-weight:500; transition:background-color 0.2s; }
button:hover{ background:#0958d9; }
button.ghost{ background:transparent; color:var(--text); border:1px solid var(--border); }
button.ghost:hover{ background:rgba(11,116,255,0.05); }
button.secondary{ background:var(--muted); color:var(--bg); }
button.secondary:hover{ background:#555; }
.result{ border: 1px solid var(--border); padding: 20px; margin: 16px 0; border-radius: 12px; background: var(--card); transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s; }
.result:hover{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.1); border-color:var(--accent); }
.meta{ color: var(--muted); font-size: 0.85em; display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
mark{ background: var(--mark); padding:2px 4px; border-radius:3px; }
.row{ display:flex; align-items:center; gap:12px; }
.muted{ color:var(--muted); }
.actions a{ margin-right:12px; color:var(--accent); text-decoration:none; font-weight:500; }
.actions a:hover{ text-decoration:underline; }
.topbar{ position:sticky; top:0; z-index:10; background:var(--bg); border-bottom:1px solid var(--border); box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.topwrap{ display:flex; align-items:center; gap:16px; justify-content:space-between; padding:16px 20px; }
.title{ font-size:24px; font-weight:700; color:var(--text); }
.reindex{ display:flex; gap:8px; align-items:center; }
.search-form { display:flex; gap:12px; align-items:center; margin-bottom:16px; }
.search-input { flex:1; }
.advanced-controls{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:12px; align-items:center; padding:16px; background:rgba(11,116,255,0.02); border-radius:8px; margin-top:16px; }
.note{ margin-top:16px; padding:12px; background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); border-radius:8px; color:#92400e; }
.loading{ text-align:center; padding:40px; color:var(--muted); }
.loading-spinner{ display:inline-block; width:20px; height:20px; border:2px solid var(--border); border-top:2px solid var(--accent); border-radius:50%; animation:spin 1s linear infinite; margin-right:8px; }
@keyframes spin{ 0%{ transform:rotate(0deg); } 100%{ transform:rotate(360deg); } }
.stats{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }
.stats-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:16px; }
.stat-item{ text-align:center; }
.stat-number{ font-size:24px; font-weight:700; color:var(--accent); }
.stat-label{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }
.badge{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }
.badge.anthropic{ background:rgba(255,140,0,0.15); color:#ea580c; border:1px solid rgba(255,140,0,0.3); }
.badge.chatgpt{ background:rgba(16,163,127,0.15); color:#059669; border:1px solid rgba(16,163,127,0.3); }
.badge.user{ background:rgba(59,130,246,0.15); color:#2563eb; border:1px solid rgba(59,130,246,0.3); }
.badge.assistant{ background:rgba(168,85,247,0.15); color:#7c3aed; border:1px solid rgba(168,85,247,0.3); }
.badge.system{ background:rgba(156,163,175,0.15); color:#6b7280; border:1px solid rgba(156,163,175,0.3); }
.pill{ display:inline-block; padding:2px 8px; border-radius:999px; background:var(--accent); color:white; font-weight:600; font-size:11px; }
.filters{ display:grid; grid-template-columns: 1fr auto auto auto auto; gap:12px; align-items:center; margin-top:12px; }
.grid{ display:grid; grid-template-columns: 1fr; gap:0; }
.empty-state{ text-align:center; padding:60px 20px; color:var(--muted); }
.empty-icon{ font-size:48px; margin-bottom:16px; opacity:0.5; }
@media(max-width: 768px){ .topwrap{ flex-direction:column; gap:12px; } .search-form{ flex-direction:column; } .advanced-controls{ grid-template-columns: 1fr; } .filters{ grid-template-columns: 1fr 1fr; } }
</style>
<div class="topbar">
  <div class="container topwrap">
    <div class="row">
      <div class="title">🔍 AI Conversation Search</div>
      <button id="theme" class="ghost" type="button" title="Toggle theme">🌙</button>
    </div>
    <form class="reindex" method="POST" action="{{ url_for('reindex') }}">
      <input type="text" name="export" value="{{export_default|e}}" placeholder="Path to export folder"/>
      <button type="submit">Reindex</button>
    </form>
  </div>
</div>
<div class="container">
  <div class="search-container">
    <form method="GET" class="search-form">
      <div class="search-input">
        <input type="text" name="q" value="{{q|e}}" placeholder="Search ChatGPT & Claude conversations..." autofocus/>
      </div>
      <button type="submit">Search</button>
      <button type="button" class="ghost" onclick="toggleAdvanced()">⚙️</button>
    </form>

    <div id="advanced-controls" class="advanced-controls" style="display: none;">
      <label class="row">
        <input type="checkbox" name="wild" value="1" {% if wild %}checked{% endif %}> Wildcard search
      </label>
      <label>
        Mode
        <select name="mode">
          <option value="fts" {% if mode=='fts' %}selected{% endif %}>Full-Text</option>
          <option value="vector" {% if mode=='vector' %}selected{% endif %}>Semantic</option>
          <option value="hybrid" {% if mode=='hybrid' %}selected{% endif %}>Hybrid</option>
        </select>
      </label>
      <label>
        Balance
        <input type="number" name="alpha" min="0" max="1" step="0.1" value="{{alpha}}" style="width:60px"/>
      </label>
      <label>
        From
        <input type="date" name="date_from" value="{{date_from}}"/>
      </label>
      <label>
        To
        <input type="date" name="date_to" value="{{date_to}}"/>
      </label>
    </div>
  </div>

  {% if rows is not none %}
    {% if rows|length > 0 %}
      <div class="stats">
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-number">{{rows|length}}</div>
            <div class="stat-label">Results</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">{% if mode=='vector' %}🔮{% elif mode=='hybrid' %}⚖️{% else %}📝{% endif %}</div>
            <div class="stat-label">{{mode|title}} Mode</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">{{alpha}}</div>
            <div class="stat-label">Balance</div>
          </div>
        </div>
      </div>

      {% for r in rows %}
        <div class="result">
          <div class="meta">
            <div><strong><a href="{{ url_for('conversation', conv_id=r['conv_id'], q=q) }}">{{r['title']}}</a></strong></div>
            <span class="badge {% if 'anthropic' in r['source'] %}anthropic{% elif 'chatgpt' in r['source'] or 'conversations' in r['source'] %}chatgpt{% endif %}">{% if 'anthropic' in r['source'] %}Claude{% else %}ChatGPT{% endif %}</span>
            <span class="badge {{r['role']}}">{{r['role']|title}}</span>
            <div>{{r['pretty_ts']}}</div>
            {% if r['score'] is defined %}
              <span class="pill">{{ '%.3f'|format(r['score']) }}</span>
            {% endif %}
          </div>
          <div style="margin: 12px 0; line-height: 1.6; color: var(--text);">{{r['snip']|safe}}</div>
          <div class="meta">
            <div class="muted" style="font-size: 0.8em;">{{r['conv_id'][:20]}}...</div>
            {% if r['chat_url'] %}
              <div class="actions">
                {% if 'anthropic' in r['source'] %}
                  <a href="{{r['chat_url']}}" target="_blank" rel="noopener noreferrer">📱 Open in Claude</a>
                {% else %}
                  <a href="{{r['chat_url']}}" target="_blank" rel="noopener noreferrer">💬 Open in ChatGPT</a>
                {% endif %}
              </div>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    {% else %}
      <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <h3>No results found</h3>
        <p>Try adjusting your search terms or filters to find what you're looking for.</p>
      </div>
    {% endif %}
  {% endif %}
{% if notice %}
  <div class="note">{{ notice }}</div>
{% endif %}
</div>
<script>
  function toggleAdvanced() {
    const controls = document.getElementById('advanced-controls');
    if (controls) {
      controls.style.display = controls.style.display === 'none' ? 'grid' : 'none';
    }
  }

  (function(){
    try{
      const pref = localStorage.getItem('theme') || 'light';
      if(pref==='dark') document.body.classList.add('dark');
    }catch(e){}
    const btn = document.getElementById('theme');
    if(btn){
      btn.addEventListener('click', function(){
        document.body.classList.toggle('dark');
        try{ localStorage.setItem('theme', document.body.classList.contains('dark') ? 'dark' : 'light'); }catch(e){}
      });
    }

    // Auto-hide advanced controls on mobile
    if (window.innerWidth <= 768) {
      const controls = document.getElementById('advanced-controls');
      if (controls) {
        controls.style.display = 'none';
      }
    }
  })();
</script>
"""

CONV_TEMPLATE = """
<!doctype html>
<title>Conversation - AI Conversation Search</title>
<style>
  :root{ --bg:#ffffff; --text:#111; --muted:#666; --card:#fafafa; --border:#e3e3e3; --accent:#0b74ff; --mark:#fffd7a; }
  body{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); }
  .container{ max-width: 900px; margin: 0 auto; padding: 20px; }
  a{ color:var(--accent); text-decoration:none; }
  .header{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; }
  .mmeta{ display:flex; gap:10px; color:var(--muted); font-size:12px; }
  .msg{ border:1px solid var(--border); background:var(--card); border-radius:10px; padding:12px; margin:10px 0; }
  mark{ background: var(--mark); }
</style>
<div class="container">
  <div class="header">
    <div>
      <div style="font-weight:700; font-size:18px;">{{ title or 'Conversation' }}</div>
      <div class="mmeta">conv_id={{ conv_id }}</div>
    </div>
    <div><a href="{{ url_for('home', q=q) }}">← Back</a></div>
  </div>
  {% for m in messages %}
    <div class="msg">
      <div class="mmeta"><span>{{ m['role'] or 'unknown' }}</span><span>{{ m['pretty_ts'] }}</span><span>{{ m['source'] }}</span></div>
      <div>{{ m['content']|safe }}</div>
    </div>
  {% endfor %}
</div>
"""

SQL = """
SELECT d.id, d.conv_id, d.title, d.role, d.ts, d.source,
       snippet(docs_fts, 0, '<mark>', '</mark>', ' … ', 12) as snip
FROM docs_fts
JOIN docs d ON d.rowid = docs_fts.rowid
WHERE docs_fts MATCH ?
ORDER BY rank
LIMIT 50
"""

def _format_timestamp(ts_value):
    try:
        ts_float = float(ts_value)
        if ts_float <= 0:
            return ""
        return time.strftime('%Y-%m-%d %H:%M', time.localtime(ts_float))
    except Exception:
        return ""

def _guess_chat_url(conv_id: str, source: str):
    if not conv_id:
        return None
    if source in ("conversations.json", "shared_conversations.json"):
        return f"https://chatgpt.com/c/{conv_id}"
    if source == "anthropic.conversations.json":
        # Anthropic uses UUIDs for conversation IDs
        return f"https://claude.ai/chat/{conv_id}"
    return None

def make_app(db_path: str):
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET", "dev")
    db_holder = {"conn": sqlite3.connect(db_path, check_same_thread=False), "db_path": db_path}
    db_holder["conn"].row_factory = sqlite3.Row
    embed_holder = {"mgr": None, "vec": None}

    @app.route("/", methods=["GET"])
    def home():
        q = request.args.get("q", "").strip()
        wild = request.args.get("wild") == "1"
        mode = request.args.get("mode", "fts").strip()
        try:
            alpha = float(request.args.get("alpha", "0.5"))
        except Exception:
            alpha = 0.5
        date_from = request.args.get("date_from") or None
        date_to = request.args.get("date_to") or None

        rows = None
        enriched = None
        notice = None

        def expand_tokens(txt):
            parts = [p for p in txt.replace('\u2013', ' ').replace('\u2014', ' ').split() if p]
            expanded = []
            for p in parts:
                if any(op in p for op in ['"', "'", "AND", "OR", "NOT", "NEAR", ":"]):
                    expanded.append(p)
                elif len(p) > 2:
                    expanded.append(p + '*')
                else:
                    expanded.append(p)
            return " ".join(expanded)

        if q:
            if mode == "fts":
                # Build SQL with optional date filters when available
                base_sql = (
                    "SELECT d.id, d.conv_id, d.title, d.role, d.ts, d.source, "
                    "snippet(docs_fts, 0, '<mark>', '</mark>', ' … ', 12) as snip "
                    "FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid "
                    "WHERE docs_fts MATCH ?"
                )
                params = []
                q_try = expand_tokens(q) if wild else q
                params.append(q_try)
                if date_from:
                    base_sql += " AND (d.date IS NOT NULL AND d.date >= ?)"
                    params.append(date_from)
                if date_to:
                    base_sql += " AND (d.date IS NOT NULL AND d.date <= ?)"
                    params.append(date_to)
                base_sql += " ORDER BY rank LIMIT 50"
                try:
                    rows = db_holder["conn"].execute(base_sql, tuple(params)).fetchall()
                except Exception:
                    # Fallback without date filters if schema lacks date
                    rows = db_holder["conn"].execute(
                        "SELECT d.id, d.conv_id, d.title, d.role, d.ts, d.source, "
                        "snippet(docs_fts, 0, '<mark>', '</mark>', ' … ', 12) as snip "
                        "FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid "
                        "WHERE docs_fts MATCH ? ORDER BY rank LIMIT 50",
                        (q_try,)
                    ).fetchall()
                if not rows and not wild:
                    q_try = expand_tokens(q)
                    rows = db_holder["conn"].execute(base_sql, tuple([q_try] + params[1:])).fetchall()

                enriched = [
                    {
                        "id": r["id"],
                        "conv_id": r["conv_id"],
                        "title": r["title"],
                        "role": r["role"],
                        "ts": r["ts"],
                        "source": r["source"],
                        "snip": r["snip"],
                        "pretty_ts": _format_timestamp(r["ts"]),
                        "chat_url": _guess_chat_url(r["conv_id"], r["source"]) 
                    }
                    for r in rows
                ]
            else:
                # Vector or hybrid search
                try:
                    # Lazy import to keep base install light
                    from local_embeddings import LocalEmbeddingsManager, VectorDatabase  # type: ignore
                except Exception:
                    notice = "Vector search requires local extras. Run: pip install -r chatgpt-export-search/requirements-local.txt"
                    enriched = []
                    export_default = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    return render_template_string(
                        TEMPLATE,
                        q=q,
                        rows=enriched,
                        export_default=export_default,
                        wild=wild,
                        mode=mode,
                        alpha=alpha,
                        date_from=date_from or "",
                        date_to=date_to or "",
                        notice=notice
                    )

                if embed_holder["mgr"] is None:
                    embed_holder["mgr"] = LocalEmbeddingsManager(model_type=os.environ.get("EMBED_MODEL_TYPE", "sentence-transformers"))
                if embed_holder["vec"] is None:
                    embed_holder["vec"] = VectorDatabase(Path(db_holder["db_path"]))

                mgr = embed_holder["mgr"]
                vec = embed_holder["vec"]

                # Optional date filter tuple
                date_filter = None
                if date_from or date_to:
                    date_filter = (
                        date_from or "1900-01-01",
                        date_to or "2100-12-31",
                    )

                scores_map = {}
                if mode == "vector":
                    query_embedding = mgr.get_query_embedding(q)
                    results = vec.search_similar(query_embedding, top_k=50, threshold=0.0, date_filter=date_filter)
                    doc_ids = [doc_id for doc_id, score in results]
                    scores_map = {doc_id: float(score) for doc_id, score in results}
                else:
                    # hybrid: get FTS candidates first
                    fts_sql = (
                        "SELECT d.id "
                        "FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid "
                        "WHERE docs_fts MATCH ? ORDER BY rank LIMIT 100"
                    )
                    q_try = expand_tokens(q) if wild else q
                    fts_rows = db_holder["conn"].execute(fts_sql, (q_try,)).fetchall()
                    fts_doc_ids = [r["id"] for r in fts_rows]
                    query_embedding = mgr.get_query_embedding(q)
                    results = vec.hybrid_search(query_embedding, fts_doc_ids, alpha=alpha, top_k=50)
                    doc_ids = [doc_id for doc_id, score in results]
                    scores_map = {doc_id: float(score) for doc_id, score in results}

                # Fetch details for doc_ids preserving order
                enriched = []
                if doc_ids:
                    placeholders = ",".join(["?"] * len(doc_ids))
                    cur = db_holder["conn"].execute(
                        f"SELECT id, conv_id, title, role, ts, source, content FROM docs WHERE id IN ({placeholders})",
                        tuple(doc_ids)
                    )
                    rows_map = {r["id"]: r for r in cur.fetchall()}
                    for doc_id in doc_ids:
                        r = rows_map.get(doc_id)
                        if not r:
                            continue
                        content = r["content"] or ""
                        snip = content[:240] + ("…" if len(content) > 240 else "")
                        enriched.append({
                            "id": r["id"],
                            "conv_id": r["conv_id"],
                            "title": r["title"],
                            "role": r["role"],
                            "ts": r["ts"],
                            "source": r["source"],
                            "snip": snip,
                            "score": scores_map.get(doc_id),
                            "pretty_ts": _format_timestamp(r["ts"]),
                            "chat_url": _guess_chat_url(r["conv_id"], r["source"]) 
                        })

        export_default = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        return render_template_string(
            TEMPLATE,
            q=q,
            rows=enriched,
            export_default=export_default,
            wild=wild,
            mode=mode,
            alpha=alpha,
            date_from=date_from or "",
            date_to=date_to or "",
            notice=notice
        )

    @app.route("/reindex", methods=["POST"])
    def reindex():
        export = request.form.get("export", "").strip()
        if not export or not os.path.isdir(export):
            flash("Provide a valid export directory containing your ChatGPT export files.")
            return redirect(url_for('home'))
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index'))
        # Use enhanced indexer for better metadata + date columns
        index_script = os.path.abspath(os.path.join(os.path.dirname(__file__), 'index_enhanced.py'))
        try:
            # Recreate DB from scratch
            db_path = os.path.join(out_dir, 'chatgpt.db')
            if os.path.exists(db_path):
                os.remove(db_path)
            subprocess.run([sys.executable, index_script, "--export", export, "--out", out_dir], check=True)
        except subprocess.CalledProcessError:
            flash("Reindex failed. Check server logs for details.")
            return redirect(url_for('home'))
        # reopen connection to see latest index
        try:
            db_path = os.path.join(out_dir, 'chatgpt.db')
            db_holder["conn"].close()
            db_holder["conn"] = sqlite3.connect(db_path, check_same_thread=False)
            db_holder["conn"].row_factory = sqlite3.Row
            db_holder["db_path"] = db_path
            # reset embeddings manager/vector DB to force reload on next vector search
            embed_holder["mgr"] = None
            embed_holder["vec"] = None
            flash("Reindex complete.")
        except Exception:
            flash("Reindex complete, but failed to reload database connection.")
        return redirect(url_for('home'))

    @app.route('/favicon.ico')
    def favicon():
        return ("", 204)

    @app.route("/conversation", methods=["GET"])
    def conversation():
        conv_id = request.args.get("conv_id", "").strip()
        q = request.args.get("q", "").strip()
        if not conv_id:
            return redirect(url_for('home'))

        # Fetch messages in conversation
        try:
            rows = db_holder["conn"].execute(
                "SELECT role, ts, title, content, source FROM docs WHERE conv_id = ? ORDER BY COALESCE(msg_index, ts, rowid)",
                (conv_id,)
            ).fetchall()
        except Exception:
            rows = db_holder["conn"].execute(
                "SELECT role, ts, title, content, source FROM docs WHERE conv_id = ? ORDER BY ts",
                (conv_id,)
            ).fetchall()

        def highlight(text: str, query: str) -> str:
            if not text:
                return ""
            if not query:
                return text
            try:
                parts = [t for t in query.replace('\u2013',' ').replace('\u2014',' ').split() if len(t) > 1]
                out = text
                for t in parts:
                    out = out.replace(t, f"<mark>{t}</mark>")
                    out = out.replace(t.capitalize(), f"<mark>{t.capitalize()}</mark>")
                return out
            except Exception:
                return text

        messages = [{
            "role": r["role"],
            "pretty_ts": _format_timestamp(r["ts"]),
            "source": r["source"],
            "content": highlight(r["content"] or "", q)
        } for r in rows]

        title = None
        for r in rows:
            if r["title"]:
                title = r["title"]
                break
        return render_template_string(CONV_TEMPLATE, conv_id=conv_id, messages=messages, title=title, q=q)

    return app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./index/chatgpt.db")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app = make_app(args.db)
    app.run(host=args.host, port=args.port, debug=False)
