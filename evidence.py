"""Pure evidence preparation and reference validation."""

import json
import re
from dataclasses import dataclass, field

from . import prompts

SOURCE_FIELDS = (
    "kb_id",
    "doc_id",
    "chunk_id",
    "chunk_index",
    "doc_name",
    "kb_name",
    "content",
)


def text_only(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def history_text(history, count=6):
    history = [m if isinstance(m, dict) else m.model_dump() for m in history]
    history = [m for m in history if m.get("role") in ("user", "assistant")]
    return [
        {"role": m["role"], "text": text_only(m.get("content", ""))[-3000:]}
        for m in history[-count:]
    ]


def json_object(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


def strings(value, limit):
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(v.strip() for v in value if isinstance(v, str) and v.strip())
    )[:limit]


@dataclass
class Plan:
    route: str
    question: str
    queries: list[str] = field(default_factory=list)
    exclude_memory: bool = False
    source: str = "web"
    kb_names: list[str] = field(default_factory=list)
    reference: str = ""
    item_ids: list[int] = field(default_factory=list)
    detailed: bool = False
    local_only: bool = False

    @classmethod
    def parse(cls, raw, original):
        data = json_object(raw)
        route = data.get("route")
        if route not in ("chat", "research", "citation", "memory_control", "delegate"):
            raise ValueError("Invalid task route")
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            question = original
        queries = strings(data.get("queries"), 4)
        if route == "research" and not queries:
            queries = [question]
        source = data.get("source", "web")
        if source not in ("web", "kb", "both"):
            source = "web"
        return cls(
            route,
            question.strip(),
            queries,
            data.get("exclude_memory") is True,
            source,
            strings(data.get("kb_names"), 20),
            str(data.get("reference") or ""),
            [i for i in data.get("item_ids", []) if type(i) is int and i >= 0]
            if isinstance(data.get("item_ids"), list)
            else [],
            data.get("detailed") is True,
            data.get("local_only") is True,
        )


@dataclass
class Packet:
    question: str
    original_question: str = ""
    sources: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)

    def add(self, query, result, *, record_query=True):
        if record_query:
            self.queries.append(query)
        if not result:
            return
        rows = list(result.get("results", [])) + list(
            result.get("equipment_context", [])
        )
        keys = ("kb_id", "doc_id", "chunk_id", "url", "content")
        seen = {tuple(s.get(k, "") for k in keys) for s in self.sources}
        for row in rows:
            if not isinstance(row.get("content"), str):
                self.errors.append("invalid_source_content")
                continue
            key = tuple(row.get(k, "") for k in keys)
            if key in seen or not row.get("content"):
                continue
            seen.add(key)
            self.sources.append(
                {
                    **dict.fromkeys(SOURCE_FIELDS, ""),
                    **row,
                    "id": f"S{len(self.sources) + 1}",
                }
            )

    def payload(self):
        return {
            "question": self.question,
            "original_question": self.original_question,
            "sources": self.sources,
            "queries": self.queries,
            "operations": self.operations,
            "retrieval_errors": self.errors,
        }


def validate_items(data, packet):
    if not isinstance(packet.sources, list) or any(
        not isinstance(s, dict)
        or not isinstance(s.get("id"), str)
        or not isinstance(s.get("content"), str)
        for s in packet.sources
    ):
        raise TypeError("Invalid evidence sources")
    sources = {s["id"]: s for s in packet.sources}
    items = data.get("items")
    if not isinstance(items, list) or len(items) > 12:
        raise ValueError("Invalid evidence items")
    clean = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise TypeError("Invalid claim")
        refs = item.get("evidence")
        if (
            not item["text"].strip()
            or not isinstance(refs, list)
            or not 1 <= len(refs) <= 3
        ):
            raise ValueError("Claim needs evidence")
        for ref in refs:
            if not isinstance(ref, dict):
                raise TypeError("Invalid reference")
            source = sources.get(ref.get("source"))
            quote = ref.get("quote")
            if source is None or not isinstance(quote, str) or not quote.strip():
                raise ValueError("Unknown source or empty quote")
            if quote not in source["content"]:
                raise ValueError("Quote is not verbatim")
        clean.append(item)
    return clean


def source_label(source, packet):
    if source.get("url"):
        kind = "网页正文" if source.get("content_kind") == "page_text" else "搜索摘要"
        return f"{source.get('title') or source['url']} — {source['url']}（{kind}，抓取时间 {source.get('retrieved_at', '')}）"
    label = source["doc_name"] or source["doc_id"]
    if source["kb_name"]:
        label = source["kb_name"] + " / " + label
    identities = {
        (s["kb_id"], s["doc_id"])
        for s in packet.sources
        if s["doc_name"] == source["doc_name"] and s["kb_name"] == source["kb_name"]
    }
    if len(identities) > 1:
        label += f"（库ID {source['kb_id']}；文档ID {source['doc_id']}）"
    label += f"，片段 {source['chunk_index']}"
    versions = {
        s["content"]
        for s in packet.sources
        if all(s[k] == source[k] for k in ("kb_id", "doc_id", "chunk_id"))
    }
    if len(versions) > 1:
        label += f"（本次返回版本 {source['id']}）"
    return label


def render(items, missing, packet, *, citations=False):
    sources = {s["id"]: s for s in packet.sources}
    lines = []
    for item in items:
        lines.append(item["text"].strip())
        for ref in item["evidence"] if citations else []:
            s = sources[ref["source"]]
            lines.append(f"出处：{source_label(s, packet)}。")
            lines.append("\n".join("> " + line for line in ref["quote"].splitlines()))
    return "\n\n".join(line for line in lines if line).strip()


async def review(ask, retrieve, packet, draft):
    """Validate the native structured answer; repair only on a failed audit."""
    last_error = ""
    approved = []
    extra_queries = 0
    for attempt in range(2):
        try:
            data = json_object(draft)
            items = validate_items(data, packet)
            missing = strings(data.get("missing"), 12)
            queries = strings(data.get("queries"), 2)
            if items:
                audit = json_object(
                    await ask(
                        prompts.AUDIT,
                        {**packet.payload(), "items": items, "missing": missing},
                    )
                )
                if all(
                    audit.get(k) is True
                    for k in ("supported", "complete", "process_ok")
                ):
                    return items, "model_audited"
                if audit.get("process_ok") is True:
                    indices = audit.get("accepted", [])
                    if isinstance(indices, list):
                        approved = [
                            items[i]
                            for i in indices
                            if type(i) is int and 0 <= i < len(items)
                        ]
                last_error = (
                    "; ".join(strings(audit.get("issues"), 12))
                    or "Evidence audit failed."
                )
                queries = strings(audit.get("queries"), 2) or queries
            else:
                last_error = (
                    "No supported items. Do not infer a whole book lacks information."
                )
        except (ValueError, TypeError) as error:
            last_error = str(error)
            queries = []
        if attempt == 0:
            try:
                for query in queries:
                    if query not in packet.queries and extra_queries < 2:
                        await retrieve(query, packet)
                        extra_queries += 1
                draft = await ask(
                    prompts.REPAIR,
                    {
                        **packet.payload(),
                        "draft": draft,
                        "issues": last_error,
                    },
                )
            except Exception as error:  # noqa: BLE001 - provider boundary
                packet.errors.append(type(error).__name__)
                break
    return approved, "partial_audited" if approved else "silent_unverified"
