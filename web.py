"""Execute AstrBot's configured search tool; no separate keys or HTTP client."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import urlsplit

from .evidence import json_object

PROVIDERS = {
    "deepseek": "DeepSeekWebSearchTool",
    "tavily": "TavilyWebSearchTool",
    "bocha": "BochaWebSearchTool",
    "brave": "BraveWebSearchTool",
    "firecrawl": "FirecrawlWebSearchTool",
    "baidu_ai_search": "BaiduWebSearchTool",
    "exa": "ExaWebSearchTool",
    "anysearch": "AnySearchWebSearchTool",
    "serpapi": "SerpApiWebSearchTool",
}
READERS = {
    "deepseek": "DeepSeekFetchWebPageTool",
    "tavily": "TavilyExtractWebPageTool",
    "firecrawl": "FirecrawlExtractWebPageTool",
    "exa": "ExaGetContentsTool",
}


def is_url(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
        )
    except ValueError:
        return False


async def search(context, event, query, page_timeout=15):
    from astrbot.core.tools import web_search_tools

    config = context.get_config(umo=event.unified_msg_origin).get(
        "provider_settings", {}
    )
    if not config.get("web_search", False):
        raise ValueError("native_web_search_disabled")
    provider = config.get("websearch_provider", "tavily")
    tool_class = getattr(web_search_tools, PROVIDERS[provider])
    tool = context.get_llm_tool_manager().get_builtin_tool(tool_class)
    wrapper = SimpleNamespace(context=SimpleNamespace(context=context, event=event))

    def source(url, title="", content="", kind="unread"):
        return {
            "url": url,
            "title": title or url,
            "content": content[:18000],
            "channel": "web",
            "content_kind": kind,
            "search_provider": provider,
            "retrieved_at": datetime.now().astimezone().isoformat(),
        }

    async def read(row):
        if provider not in READERS:
            return row
        reader = context.get_llm_tool_manager().get_builtin_tool(
            getattr(web_search_tools, READERS[provider])
        )
        args = {"url": row["url"]}
        if provider == "exa":
            args["max_characters"] = 18000
        started = asyncio.get_running_loop().time()
        try:
            # A slow full-page request must not discard already returned search snippets.
            async with asyncio.timeout(page_timeout):
                raw = await reader.call(wrapper, **args)
            marker = "\nContent: "
            if not isinstance(raw, str) or marker not in raw:
                raise ValueError("page_content_missing")
            content = raw.split(marker, 1)[1].strip()
            if not content or content in ("No content", "None"):
                raise ValueError("page_content_empty")
            row.update(
                content=content[:18000],
                content_truncated=len(content) > 18000,
                content_kind="page_text",
                fetch_status="completed",
            )
        except Exception as error:  # noqa: BLE001 - native tool boundary
            row.update(fetch_status="failed", fetch_error=type(error).__name__)
        row["fetch_seconds"] = round(asyncio.get_running_loop().time() - started, 3)
        return row

    if is_url(query):
        return {"results": [await read(source(query))]}
    arguments = {"query": query}
    if provider == "deepseek":
        arguments.update(count=5, need_content=False)
        if len(query) > 128:
            raise ValueError("web_query_exceeds_128_characters")
    elif provider == "tavily":
        arguments.update(max_results=5, search_depth="advanced")
    raw = await tool.call(wrapper, **arguments)
    data = json_object(raw)
    rows = []
    by_url = {}
    for result in data.get("results", []):
        if not isinstance(result, dict):
            continue
        content = result.get("snippet") or result.get("content")
        if (
            not isinstance(content, str)
            or not content.strip()
            or not is_url(result.get("url", ""))
        ):
            continue
        if result["url"] in by_url:
            previous = by_url[result["url"]]
            if content not in previous["content"]:
                previous["content"] += "\n\n" + content
            continue
        rows.append(
            source(result["url"], result.get("title"), content, "search_snippet")
        )
        by_url[result["url"]] = rows[-1]
    # Independent full-page reads run concurrently, without a second model call.
    rows[:2] = await asyncio.gather(*(read(row) for row in rows[:2]))
    return {"results": rows}
