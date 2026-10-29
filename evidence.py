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
    if not isinstance(raw, str):
        raise TypeError("Expected JSON text")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # Some providers wrap a single JSON object in a short explanation.
        start = raw.find("{")
        if start < 0:
            raise
        value, end = json.JSONDecoder().raw_decode(raw[start:])
        if "{" in raw[start + end :]:
            raise ValueError("Multiple JSON objects") from None
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


def diagnostic(value, limit=16000):
    """Bound diagnostic text and remove common credential representations."""
    text = str(value)
    text = re.sub(r"(?i)bearer\s+[^\s\"',}]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|access_token|authorization|token)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,}]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", text)
    return text[:limit]


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
    diagnostics: list[dict] = field(default_factory=list)

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


def validate_items(data, packet, *, strict=True, issues=None):
    if not isinstance(packet.sources, list) or any(
        not isinstance(s, dict)
        or not isinstance(s.get("id"), str)
        or not isinstance(s.get("content"), str)
        for s in packet.sources
    ):
        raise TypeError("Invalid evidence sources")
    sources = {s["id"]: s for s in packet.sources}
    items = data.get("items")
    if not isinstance(items, list):
        raise TypeError("Invalid evidence items")
    clean = []
    issues = issues if issues is not None else []
    if len(items) > 12:
        if strict:
            raise ValueError("Too many evidence items")
        issues.append(
            "Only the first 12 items are processed; combine overlapping points."
        )
    for index, item in enumerate(items[:12]):
        try:
            clean.append(
                validate_item(item, sources, strict=strict, issues=issues, index=index)
            )
        except (ValueError, TypeError) as error:
            if strict:
                raise
            issues.append(f"item[{index}]: {error}")
    return clean


def validate_item(item, sources, *, strict, issues, index):
    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
        raise TypeError("Invalid claim")
    refs = item.get("evidence")
    if (
        not item["text"].strip()
        or not isinstance(refs, list)
        or not 1 <= len(refs) <= 3
    ):
        raise ValueError("Claim needs evidence")
    clean_refs = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise TypeError("Invalid reference")
        source = sources.get(ref.get("source"))
        quote = ref.get("quote")
        if source is None:
            raise ValueError("Unknown source")
        citation = {"source": ref["source"]}
        if isinstance(quote, str) and quote.strip():
            if quote not in source["content"]:
                # Restore original whitespace, without changing words, digits or units.
                pattern = r"\s+".join(re.escape(part) for part in quote.split())
                match = re.search(pattern, source["content"])
                quote = match.group() if match else None
            if quote:
                citation["quote"] = quote
        if "quote" not in citation:
            if source.get("channel") != "web":
                raise ValueError("Local quote is empty or not verbatim")
            if ref.get("quote"):
                if strict:
                    raise ValueError("Quote is not verbatim")
                issues.append(
                    f"item[{index}]: unverified web quotation removed; audit source support"
                )
            # Web claims cite a retrieved source, just as a search tool response does.
            # Semantic support is still checked by AUDIT; never render an invented quote.
        clean_refs.append(citation)
    return {"text": item["text"].strip(), "evidence": clean_refs}


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
            if ref.get("quote"):
                lines.append(
                    "\n".join("> " + line for line in ref["quote"].splitlines())
                )
    return "\n\n".join(line for line in lines if line).strip()


async def review(ask, retrieve, packet, draft):
    """Keep supported items and allow bounded search/read follow-ups or one repair."""
    approved = []
    extra_queries = 0
    repaired = False
    for attempt in range(3):
        step = {
            "attempt": attempt + 1,
            "output": diagnostic(draft),
            "output_chars": len(str(draft)),
            "output_truncated": len(str(draft)) > 16000,
            "issues": [],
        }
        packet.diagnostics.append(step)
        queries = []
        try:
            data = json_object(draft)
            missing = strings(data.get("missing"), 12)
            queries = strings(data.get("queries"), 2)
            items = validate_items(data, packet, strict=False, issues=step["issues"])
            step["valid_items"] = len(items)
            if items:
                audit_text = await ask(
                    prompts.AUDIT,
                    {**packet.payload(), "items": items, "missing": missing},
                )
                step["audit_output"] = diagnostic(audit_text)
                audit = json_object(audit_text)
                step["audit"] = audit
                step.pop("audit_output")
                if all(
                    audit.get(k) is True
                    for k in ("supported", "complete", "process_ok")
                ):
                    if not step["issues"]:
                        return items, "model_audited"
                    # Other malformed items do not invalidate these individually audited items.
                    approved = items
                    if not missing and not queries:
                        return approved, "partial_audited"
                elif audit.get("process_ok") is True:
                    indices = audit.get("accepted", [])
                    if isinstance(indices, list):
                        approved = [
                            items[i]
                            for i in indices
                            if type(i) is int and 0 <= i < len(items)
                        ]
                else:
                    approved = []
                step["issues"].extend(strings(audit.get("issues"), 12))
                queries = strings(audit.get("queries"), 2) or queries
            else:
                step["issues"].append(
                    "No supported items; use the available sources or search/read further."
                )
        except (ValueError, TypeError) as error:
            step["issues"].append(diagnostic(error, 600))
        except Exception as error:  # noqa: BLE001 - audit provider boundary
            step["error"] = diagnostic(error, 600)
            packet.errors.append(type(error).__name__)
            break
        if attempt == 2:
            break
        followups = [q for q in queries if q not in packet.queries][: 2 - extra_queries]
        if not followups and repaired:
            break
        if not followups:
            repaired = True
        try:
            for query in followups:
                await retrieve(query, packet)
                extra_queries += 1
            system = prompts.FOLLOWUP if followups else prompts.REPAIR
            draft = await ask(
                system,
                {
                    **packet.payload(),
                    "draft": draft,
                    "accepted_items": approved,
                    "issues": step["issues"],
                    "remaining_followups": 2 - extra_queries,
                },
            )
        except Exception as error:  # noqa: BLE001 - provider boundary
            step["error"] = diagnostic(error, 600)
            packet.errors.append(type(error).__name__)
            break
    return approved, "partial_audited" if approved else "silent_unverified"
