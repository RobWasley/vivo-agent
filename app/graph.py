from __future__ import annotations

import datetime
import logging
import re
from collections import Counter

log = logging.getLogger("vivo.graph")

# Words too generic to carry meaning in a "shares a theme" edge.
STOPWORDS = {
    "about", "above", "after", "again", "all", "also", "and", "any", "are",
    "because", "been", "before", "being", "below", "between", "both", "but",
    "can", "could", "did", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "he", "her", "here",
    "hers", "him", "his", "how", "into", "its", "itself", "just", "keep",
    "kept", "like", "made", "make", "many", "may", "me", "might", "more",
    "most", "much", "must", "my", "no", "not", "now", "of", "off", "on",
    "once", "only", "other", "our", "out", "over", "own", "said", "same",
    "she", "should", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "your", "yours",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 4
_MIN_SHARED = 2
_MAX_THEME_EDGES_PER_NODE = 6
_SESSION_TEXT_CAP = 20000


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) >= _MIN_TOKEN_LEN and t not in STOPWORDS
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _date_of(ts: float | int | None) -> str:
    if not ts:
        return ""
    return datetime.datetime.fromtimestamp(
        float(ts), tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d")


def build_graph(memory, dreams, sessions) -> dict:
    """Assemble the vault as an Obsidian-style graph: one node per document,
    edges for explicit relations (core anchor, dream -> facts it produced,
    session -> facts discussed in it) plus weak theme edges between nodes
    that share salient words."""
    nodes: list[dict] = []
    links: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    def add_link(source: str, target: str, kind: str, weight: float = 1, label: str = "") -> bool:
        key = (source, target) if source < target else (target, source)
        if key in seen_pairs or source == target:
            return False
        seen_pairs.add(key)
        link = {"source": source, "target": target, "kind": kind, "weight": weight}
        if label:
            link["label"] = label
        links.append(link)
        return True

    nodes.append({
        "id": "core",
        "type": "core",
        "label": "MEMORY",
        "sub": "core memory index",
    })

    facts = memory.list()
    fact_ids = set()
    fact_tokens: dict[str, set[str]] = {}
    for fact in facts:
        fact_ids.add(fact["id"])
        fact_tokens[fact["id"]] = _tokens(fact.get("text", ""))
        nodes.append({
            "id": fact["id"],
            "type": "fact",
            "label": fact.get("text", ""),
            "sub": fact.get("date") or "",
            "core": bool(fact.get("core")),
        })
        if fact.get("core"):
            add_link("core", fact["id"], "core")

    dream_nodes: list[tuple[str, set[str]]] = []
    for dream in dreams.list():
        date = _date_of(dream.get("ts"))
        summary = dream.get("summary") or ""
        nodes.append({
            "id": dream["id"],
            "type": "dream",
            "label": summary or f"Dream {date}",
            "sub": date,
            "llm": bool(dream.get("llm")),
        })
        dream_nodes.append((dream["id"], _tokens(summary)))
        for produced in dream.get("new_facts") or []:
            produced_norm = _norm(produced)
            if not produced_norm:
                continue
            for fact in facts:
                if produced_norm in _norm(fact.get("text", "")) or _norm(fact.get("text", "")) in produced_norm:
                    add_link(dream["id"], fact["id"], "fact")

    session_texts: dict[str, str] = {}
    for row in sessions.list_sessions():
        sid = row["id"]
        label = row.get("name") or sid
        nodes.append({
            "id": sid,
            "type": "session",
            "label": label,
            "sub": f"{row.get('turns', 0)} turns",
        })
        try:
            conv = sessions.conversation_for(sid)
        except KeyError:
            conv = None
        text = " ".join(
            [conv.summary or ""]
            + [u for u, a in (conv.turns if conv else [])]
            + [a for u, a in (conv.turns if conv else [])]
        )[:_SESSION_TEXT_CAP]
        session_texts[sid] = text
        for fact in facts:
            fact_norm = _norm(fact.get("text", ""))
            if len(fact_norm) >= 8 and fact_norm in _norm(text):
                add_link(sid, fact["id"], "talk")

    theme_nodes = [
        (n["id"], fact_tokens.get(n["id"], _tokens(n["label"])))
        for n in nodes
        if n["type"] == "fact"
    ]
    theme_nodes += dream_nodes
    for sid, text in session_texts.items():
        theme_nodes.append((sid, _tokens(text)))

    candidates: list[tuple[str, str, int, list[str]]] = []
    for i in range(len(theme_nodes)):
        for j in range(i + 1, len(theme_nodes)):
            a_id, a_tokens = theme_nodes[i]
            b_id, b_tokens = theme_nodes[j]
            shared = a_tokens & b_tokens
            if len(shared) >= _MIN_SHARED:
                candidates.append((a_id, b_id, len(shared), sorted(shared)))
    candidates.sort(key=lambda c: c[2], reverse=True)
    per_node: Counter = Counter()
    for a_id, b_id, count, shared in candidates:
        if per_node[a_id] >= _MAX_THEME_EDGES_PER_NODE or per_node[b_id] >= _MAX_THEME_EDGES_PER_NODE:
            continue
        if add_link(a_id, b_id, "theme", weight=count, label=", ".join(shared[:3])):
            per_node[a_id] += 1
            per_node[b_id] += 1

    degree: Counter = Counter()
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1
    for node in nodes:
        node["degree"] = degree.get(node["id"], 0)

    return {"nodes": nodes, "links": links}
