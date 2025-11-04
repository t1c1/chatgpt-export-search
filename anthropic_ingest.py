import json, time, html, re
from pathlib import Path
from typing import Iterator, Dict, Any

def _norm_text(x):
    if x is None:
        return None
    x = html.unescape(str(x))
    x = re.sub(r"\s+", " ", x).strip()
    return x

def parse_anthropic_conversations(p: Path) -> Iterator[Dict[str, Any]]:
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    convs = raw.get("conversations") if isinstance(raw, dict) else raw
    if not isinstance(convs, list):
        return iter(())

    for c in convs:
        if not isinstance(c, dict):
            continue
        conv_id = c.get("uuid") or c.get("id") or c.get("conversation_id")
        title = c.get("title") or ""
        msgs = c.get("messages") or []
        for i, m in enumerate(msgs):
            role = (m.get("role") or m.get("author", {}).get("role") or "").lower()
            # AnthropIc exports often store text under 'text' or 'content'
            content = m.get("text") or m.get("content") or ""
            if isinstance(content, dict):
                content = content.get("text") or content.get("content") or ""
            if isinstance(content, list):
                content = " ".join([str(x) for x in content])
            ts = m.get("create_time") or m.get("timestamp") or c.get("create_time")
            doc_id = f"{conv_id}:{i}:{role}:{abs(hash(str(m)))%10**9}"
            yield {
                "id": doc_id,
                "conv_id": conv_id,
                "title": title,
                "role": role,
                "ts": ts if ts is not None else 0.0,
                "source": "anthropic.conversations.json",
                "extra": None,
                "content": _norm_text(content),
                "msg_index": i,
            }

def parse_anthropic_projects(p: Path) -> Iterator[Dict[str, Any]]:
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    doc_id = f"anthropic.projects:{abs(hash(p.as_posix()))%10**9}"
    yield {
        "id": doc_id,
        "conv_id": "anthropic_projects",
        "title": "Anthropic Projects",
        "role": "system",
        "ts": time.time(),
        "source": "projects.json",
        "extra": None,
        "content": _norm_text(json.dumps(raw, ensure_ascii=False)),
        "msg_index": 0,
    }

def parse_anthropic_users(p: Path) -> Iterator[Dict[str, Any]]:
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    doc_id = f"anthropic.users:{abs(hash(p.as_posix()))%10**9}"
    yield {
        "id": doc_id,
        "conv_id": "anthropic_users",
        "title": "Anthropic Users",
        "role": "system",
        "ts": time.time(),
        "source": "users.json",
        "extra": None,
        "content": _norm_text(json.dumps(raw, ensure_ascii=False)),
        "msg_index": 0,
    }
