"""Manual scope refresh. Startup and questions only read the saved cache."""

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from astrbot.api import logger

from .evidence import Packet, json_object, validate_items


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def document_samples(path, kb_id):
    """A manual read-only scan, bounded memory; hashes detect even same-size edits."""
    documents = {}
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        rows = db.execute(
            "SELECT text, metadata FROM documents "
            "WHERE json_extract(metadata, '$.kb_id')=? "
            "ORDER BY json_extract(metadata, '$.kb_doc_id'), "
            "CAST(json_extract(metadata, '$.chunk_index') AS INTEGER), doc_id",
            (kb_id,),
        )
        for text, raw_metadata in rows:
            metadata = json.loads(raw_metadata)
            doc_id = metadata.get("kb_doc_id")
            if not doc_id or not text:
                continue
            doc = documents.setdefault(
                doc_id,
                {
                    "hash": hashlib.sha256(),
                    "headings": [],
                    "samples": [],
                    "count": 0,
                },
            )
            doc["hash"].update(text.encode("utf-8") + b"\x00")
            doc["count"] += 1
            headings = [
                line.strip() for line in text.splitlines() if line.startswith("#")
            ]
            # Keep samples spread across the whole document without holding its body.
            sample = (hashlib.sha256(text.encode("utf-8")).hexdigest(), text[:650])
            doc["samples"].append(sample)
            doc["samples"].sort()
            del doc["samples"][12:]
            doc["headings"].extend(headings)
            del doc["headings"][80:]
    return {
        key: {
            "digest": value["hash"].hexdigest(),
            "count": value["count"],
            "sample": "\n".join(value["headings"])[:4000]
            + "\n"
            + "\n\n".join(text for _, text in value["samples"]),
        }
        for key, value in documents.items()
    }


class Catalogue:
    def __init__(self, path):
        self.path = path
        self.documents = {}
        self.lock = asyncio.Lock()

    def load(self):
        documents = read_json(self.path, {})
        if not isinstance(documents, dict) or any(
            not isinstance(d, dict)
            or not all(
                isinstance(d.get(k), str) for k in ("kb_name", "doc_name", "summary")
            )
            for d in documents.values()
        ):
            raise ValueError("Invalid catalogue cache")
        self.documents = documents

    def scope(self, names):
        return [
            {k: d[k] for k in ("kb_name", "doc_name", "summary")}
            for d in self.documents.values()
            if d["kb_name"] in names
        ]

    async def refresh(self, manager, summarize):
        async with self.lock:
            updated, changed = {}, 0
            for kb_id, helper in manager.kb_insts.items():
                samples = await asyncio.to_thread(
                    document_samples, helper.vec_db.document_storage.db_path, kb_id
                )
                offset = 0
                while True:
                    docs = await manager.kb_db.list_documents_by_kb(
                        kb_id, offset=offset, limit=100
                    )
                    for doc in docs:
                        sample = samples.get(doc.doc_id)
                        if not sample:
                            continue
                        key = f"{kb_id}/{doc.doc_id}"
                        fingerprint = [
                            sample["digest"],
                            doc.doc_name,
                            helper.kb.kb_name,
                            helper.kb.description or "",
                        ]
                        old = self.documents.get(key)
                        if old and old.get("fingerprint") == fingerprint:
                            updated[key] = old
                            continue
                        data = json_object(
                            await summarize(
                                {
                                    "doc_name": doc.doc_name,
                                    "description": helper.kb.description,
                                    "samples": sample["sample"],
                                }
                            )
                        )
                        summary = data.get("summary")
                        if not isinstance(summary, str) or not summary.strip():
                            raise ValueError("Invalid catalogue summary")
                        updated[key] = {
                            "kb_id": kb_id,
                            "doc_id": doc.doc_id,
                            "kb_name": helper.kb.kb_name,
                            "doc_name": doc.doc_name,
                            "summary": summary.strip()[:300],
                            "fingerprint": fingerprint,
                        }
                        changed += 1
                    offset += len(docs)
                    if len(docs) < 100:
                        break
            await asyncio.to_thread(write_json, self.path, updated)
            self.documents = updated
            return changed, len(updated)


class EvidenceStore:
    """Small evidence snapshots, scoped to an actual platform conversation."""

    def __init__(self, path):
        self.path = path
        self.records = []

    def load(self):
        records = read_json(self.path, [])
        if not isinstance(records, list):
            raise TypeError("Invalid evidence cache")
        valid = []
        for record in records:
            try:
                match record:
                    case {
                        "id": str(),
                        "question": str() as question,
                        "scope": [str(), str()],
                        "sources": sources,
                        "items": items,
                    }:
                        record["items"] = validate_items(
                            {"items": items}, Packet(question, sources=sources)
                        )
                    case _:
                        raise ValueError("Invalid evidence record")
                valid.append(record)
            except (KeyError, TypeError, ValueError):
                logger.warning("[证据问答] 忽略一条损坏的来源记录")
        self.records = valid[-20:]

    def recent(self, scope):
        return [r for r in self.records if r["scope"] == scope]

    def save(self, record):
        self.records = [r for r in self.records if r["id"] != record["id"]]
        self.records.append(record)
        self.records = self.records[-20:]
        write_json(self.path, self.records)

    def find(self, scope, identity):
        return next((r for r in self.recent(scope) if r["id"] == identity), None)
