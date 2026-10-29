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

    @classmethod
    def parse(cls, raw, original):
        data = json_object(raw)
        route = data.get("route")
        if route not in ("chat", "research", "memory_control"):
            raise ValueError("Invalid task route")
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            question = original
        queries = strings(data.get("queries"), 4)
        if route == "research" and not queries:
            queries = [question]
        return cls(route, question.strip(), queries, data.get("exclude_memory") is True)


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
        keys = ("kb_id", "doc_id", "chunk_id", "content")
        seen = {tuple(s[k] for k in keys) for s in self.sources}
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


def render(items, missing, packet):
    sources = {s["id"]: s for s in packet.sources}
    lines = []
    for item in items:
        lines.append(item["text"].strip())
        for ref in item["evidence"]:
            s = sources[ref["source"]]
            lines.append(f"出处：{source_label(s, packet)}。")
            lines.append("\n".join("> " + line for line in ref["quote"].splitlines()))
    if missing:
        lines.append("本次返回片段尚未覆盖：" + "；".join(missing) + "。")
    return "\n\n".join(line for line in lines if line).strip()


def excerpt_fallback(packet):
    if not packet.sources:
        return "本次知识库检索没有取得可引用的片段，尚不能回答这项资料问题。"
    lines = ["本次未能完成可靠的答案整理，先列出实际返回的原文："]
    for source in packet.sources[:3]:
        lines.append(f"{source_label(source, packet)}：")
        lines.append("\n".join("> " + line for line in source["content"].splitlines()))
    return "\n\n".join(lines)


async def review(ask, retrieve, packet, draft):
    """Up to two review/audit passes, with at most two extra native searches."""
    last_error = ""
    for attempt in range(2):
        payload = {**packet.payload(), "draft": draft, "format_error": last_error}
        try:
            data = json_object(await ask(prompts.REVIEW, payload))
            items = validate_items(data, packet)
            missing = strings(data.get("missing"), 12)
            queries = [
                q for q in strings(data.get("queries"), 2) if q not in packet.queries
            ]
            if attempt == 0 and missing and queries:
                for query in queries:
                    await retrieve(query, packet)
                continue
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
                    return render(items, missing, packet), "model_audited"
                last_error = (
                    "; ".join(strings(audit.get("issues"), 12))
                    or "Evidence audit failed."
                )
                if attempt == 0:
                    for query in strings(audit.get("queries"), 2):
                        if query not in packet.queries:
                            await retrieve(query, packet)
                continue
            last_error = "No supported items; extract the relevant original sentences."
        except (ValueError, TypeError) as error:
            last_error = str(error)
    return excerpt_fallback(packet), "excerpts"
