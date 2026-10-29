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
    technical: bool = False

    @classmethod
    def parse(cls, raw, original):
        data = json_object(raw)
        route = data.get("route")
        if route not in (
            "chat",
            "research",
            "catalogue",
            "citation",
            "memory_control",
            "delegate",
        ):
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
            data.get("technical") is True,
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


def validate_items(data, packet, *, strict=True, issues=None, require_local_quote=True):
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
                validate_item(
                    item,
                    sources,
                    strict=strict,
                    issues=issues,
                    index=index,
                    require_local_quote=require_local_quote,
                )
            )
        except (ValueError, TypeError) as error:
            if strict:
                raise
            issues.append(f"item[{index}]: {error}")
    return clean


def validate_item(item, sources, *, strict, issues, index, require_local_quote=True):
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
            if source.get("channel") != "web" and require_local_quote:
                raise ValueError("Local quote is empty or not verbatim")
            if ref.get("quote"):
                if strict:
                    raise ValueError("Quote is not verbatim")
                issues.append(
                    f"item[{index}]: unverified web quotation removed; audit source support"
                )
            # Direct answers may cite a source without an exact quotation.
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
            elif s.get("channel") != "web" and s.get("content"):
                lines.append("当时检索到的原文片段（未逐句核验支持关系）：")
                lines.append(
                    "\n".join("> " + line for line in s["content"].splitlines())
                )
    return "\n\n".join(line for line in lines if line).strip()


def direct_answer(text, packet):
    """Remove internal source tags without another model call or rewriting the answer."""
    sources = {s["id"]: s for s in packet.sources}
    items = []
    marker = re.compile(r"\[\[(S\d+)\]\]")
    for paragraph in re.split(r"\n\s*\n", text):
        refs = list(dict.fromkeys(marker.findall(paragraph)))
        clean = marker.sub("", paragraph).strip()
        refs = [identity for identity in refs if identity in sources][:3]
        if clean and refs and len(items) < 12:
            items.append({"text": clean, "evidence": [{"source": i} for i in refs]})
    return marker.sub("", text).strip(), items


def focused_sources(candidates, packet):
    """Include cited sources and available adjacent local context, without truncating it."""
    ids = {
        r.get("source")
        for item in candidates
        for r in item.get("evidence", [])
        if isinstance(r, dict) and isinstance(r.get("source"), str)
    }
    cited = [s for s in packet.sources if s["id"] in ids]
    if not cited:
        return [s for s in packet.sources if s.get("channel") != "web"]
    nearby = {
        (s.get("kb_id"), s.get("doc_id"), s.get("chunk_index"))
        for s in cited
        if s.get("channel") != "web" and type(s.get("chunk_index")) is int
    }
    return [
        s
        for s in packet.sources
        if s["id"] in ids
        or (
            s.get("channel") != "web"
            and type(s.get("chunk_index")) is int
            and any(
                (s.get("kb_id"), s.get("doc_id"), s["chunk_index"] + delta) in nearby
                for delta in (-1, 0, 1)
            )
        )
    ]


async def focused_review(ask, packet, draft):
    """One local-technical audit, with inline corrections; no repair or re-audit loop."""
    step = {"stage": "local_audit", "output": diagnostic(draft), "issues": []}
    packet.diagnostics.append(step)
    try:
        raw_items = json_object(draft).get("items")
        if not isinstance(raw_items, list):
            raise TypeError("Missing items")
    except (TypeError, ValueError) as error:
        # The single audit can recover an ordinary prose draft; no separate formatter.
        raw_items = [{"text": draft, "evidence": []}]
        step["issues"].append(diagnostic(error, 600))
    if not raw_items:
        return [], "local_no_items"
    sources = {s["id"]: s for s in packet.sources}
    valid, direct, candidates = {}, {}, []
    for index, raw in enumerate(raw_items[:12]):
        try:
            item = validate_item(
                raw, sources, strict=False, issues=step["issues"], index=index
            )
            web_only = all(
                sources[r["source"]].get("channel") == "web" for r in item["evidence"]
            )
            if web_only:
                direct[index] = item
                continue
            valid[index] = item
        except (TypeError, ValueError) as error:
            step["issues"].append(f"item[{index}]: {error}")
        if isinstance(raw, dict):
            candidates.append(
                {
                    "index": index,
                    "text": str(raw.get("text", "")),
                    "evidence": raw.get("evidence")
                    if isinstance(raw.get("evidence"), list)
                    else [],
                }
            )
    if not candidates:
        return list(direct.values()), "direct_web"
    selected = focused_sources(candidates, packet)
    focus = Packet(packet.question, sources=selected)
    checked = dict(direct)
    try:
        result = await ask(
            prompts.AUDIT,
            {
                "question": packet.question,
                "original_question": packet.original_question,
                "items": candidates,
                "sources": selected,
                "program_issues": step["issues"],
            },
        )
        step["audit_output"] = diagnostic(result)
        audit = json_object(result)
        step["audit"] = audit
        step.pop("audit_output")
        for index in audit.get("accepted", []):
            if type(index) is int and index in valid:
                checked[index] = valid[index]
        for correction in audit.get("corrections", []):
            if not isinstance(correction, dict):
                continue
            index = correction.get("index")
            if type(index) is not int or index not in {i["index"] for i in candidates}:
                continue
            checked.pop(index, None)
            try:
                checked[index] = validate_items({"items": [correction]}, focus)[0]
            except (TypeError, ValueError) as error:
                step["issues"].append(f"correction[{index}]: {error}")
        step["issues"].extend(strings(audit.get("issues"), 12))
    except Exception as error:  # noqa: BLE001 - single audit provider boundary
        step["error"] = diagnostic(error, 600)
        packet.errors.append(type(error).__name__)
        return list(direct.values()), "local_audit_failed"
    return [checked[i] for i in sorted(checked)], "local_audited"
