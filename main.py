"""Native search routing, verified answers and short everyday conversation."""

import asyncio
import json
import math
import re
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

from astrbot.api import AstrBotConfig, logger, sp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart

from . import prompts, web
from .catalogue import Catalogue, EvidenceStore
from .evidence import (
    Packet,
    Plan,
    diagnostic,
    history_text,
    json_object,
    render,
    review,
    strings,
)
from .integration import SelectionBridge
from .memory import MEMORY_STEPS, SKIP_MEMORY, MemoryBridge

STATE = "evidence_qa.state"


def short_chat(text):
    return (
        bool(text.strip())
        and len(text.strip()) <= 80
        and len([s for s in re.split(r"[。！？!?\n]+", text) if s.strip()]) <= 3
    )


class BufferedPrivateChat(filter.CustomFilter):
    plugin = None

    def filter(self, event, cfg):
        if self.plugin and self.plugin.enabled(event):
            event.set_extra("enable_streaming", False)
        return False


class EvidenceQAPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.limits = {}
        for name, default in (
            ("model_timeout_seconds", 90),
            ("retrieval_timeout_seconds", 45),
            ("page_timeout_seconds", 15),
            ("turn_timeout_seconds", 240),
        ):
            value = float(config.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为大于零的秒数")
            self.limits[name] = value
        folder = StarTools.get_data_dir("astrbot_plugin_evidence_qa")
        self.catalogue = Catalogue(folder / "catalogue.json")
        self.evidence_store = EvidenceStore(folder / "evidence.json")
        self.memory = MemoryBridge()
        self.selection = SelectionBridge(self)
        BufferedPrivateChat.plugin = self

    async def initialize(self):
        from astrbot.core import astr_main_agent
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

        for cache in (self.catalogue, self.evidence_store):
            try:
                await asyncio.to_thread(cache.load)
            except (OSError, ValueError, TypeError):
                logger.warning("[证据问答] 缓存读取失败：%s", cache.path.name)
        self.selection.install(astr_main_agent, internal)
        logger.info(
            "[证据问答] AstrBot 4.29.3：接管3个入口；目录仅加载缓存，更新请用 /资料目录更新。"
        )

    def enabled(self, event):
        selected = getattr(event, "plugins_name", None)
        return (
            self.config.get("enabled", True)
            and event.get_message_type() == MessageType.FRIEND_MESSAGE
            and event.get_extra("action_type") != "live"
            and (selected is None or "astrbot_plugin_evidence_qa" in selected)
        )

    @filter.custom_filter(BufferedPrivateChat, False)
    async def buffer_reply(self, event: AstrMessageEvent):
        pass

    @filter.command("资料目录更新")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def refresh_catalogue(self, event: AstrMessageEvent):
        provider = await self.provider_id(event)
        try:
            changed, total = await self.catalogue.refresh(
                self.context.kb_manager,
                lambda payload: self.ask(provider, prompts.CATALOGUE, payload),
            )
            yield event.plain_result(
                f"目录已更新：{total}篇文档，其中{changed}篇重新生成概览。"
            )
        except Exception as error:  # noqa: BLE001 - native/provider boundary
            logger.warning("[证据问答] 手动目录刷新失败：%s", type(error).__name__)
            yield event.plain_result("目录更新未完成，仍使用上次缓存；详细原因见日志。")

    async def provider_id(self, event):
        return self.config.get(
            "worker_provider_id", ""
        ).strip() or await self.selected_provider_id(event)

    async def selected_provider_id(self, event):
        selected = event.get_extra("selected_provider")
        if selected:
            return selected
        provider = await self.context.get_using_provider_async(event.unified_msg_origin)
        if provider is None:
            raise ValueError("没有可用的文本模型")
        return provider.meta().id

    def window(self, event, limit):
        state = event.get_extra(STATE) if event else None
        if state and "deadline" in state:
            remaining = state["deadline"] - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("turn_budget_exhausted")
            limit = min(limit, remaining)
        return asyncio.timeout(limit)

    def memory_status(self, event, state):
        state["memory_adapter_available"] = self.memory.attach(self.context)
        state["memory_skip_requested"] = bool(event.get_extra(SKIP_MEMORY, False))
        steps = event.get_extra(MEMORY_STEPS, {})
        state["memory_recall_skipped"] = steps.get("recall")
        state["memory_record_skipped"] = steps.get("reflection")
        state["memory_isolated"] = all(
            steps.get(k) is True for k in ("recall", "reflection")
        )

    async def ask(self, provider, system, payload, event=None):
        start = asyncio.get_running_loop().time()
        try:
            async with self.window(event, self.limits["model_timeout_seconds"]):
                response = await self.context.llm_generate(
                    chat_provider_id=provider,
                    system_prompt=system,
                    prompt=json.dumps(payload, ensure_ascii=False, default=str),
                    contexts=[],
                    tools=None,
                )
            return response.completion_text
        finally:
            state = event.get_extra(STATE) if event else None
            if state:
                phase = next(
                    (
                        name.lower()
                        for name in ("PLAN", "AUDIT", "REPAIR", "FOLLOWUP", "SHORTEN")
                        if system == getattr(prompts, name)
                    ),
                    "model",
                )
                state.setdefault("timings", []).append(
                    {
                        "phase": phase,
                        "provider": provider,
                        "seconds": round(asyncio.get_running_loop().time() - start, 3),
                    }
                )

    async def selected_kbs(self, event):
        config = self.context.get_config(umo=event.unified_msg_origin)
        session = await sp.session_get(
            event.unified_msg_origin, "kb_config", default={}
        )
        names, top_k = config.get("kb_names", []), config.get("kb_final_top_k", 5)
        if session and "kb_ids" in session:
            names = []
            for kb_id in session["kb_ids"]:
                helper = await self.context.kb_manager.get_kb(kb_id)
                if helper:
                    names.append(helper.kb.kb_name)
            top_k = session.get("top_k", 5)
        return names, top_k

    async def plan_turn(self, event):
        if not self.enabled(event) or event.get_extra(STATE) is not None:
            return
        question = event.get_message_str()
        if not question.strip():
            return
        started = asyncio.get_running_loop().time()
        state = {
            "trace_id": uuid4().hex,
            "question": question,
            "status": "planning",
            "started": started,
            "deadline": started + self.limits["turn_timeout_seconds"],
        }
        event.set_extra(STATE, state)
        try:
            async with self.window(event, self.limits["model_timeout_seconds"]):
                manager = self.context.conversation_manager
                cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
                conv = (
                    await manager.get_conversation(event.unified_msg_origin, cid)
                    if cid
                    else None
                )
                names, top_k = await self.selected_kbs(event)
            # No CID means no durable cross-turn lookup; do not combine unrelated sessions.
            scope = [event.unified_msg_origin, cid or state["trace_id"]]
            state.update(scope=scope, kb_names=names, top_k=top_k)
            history = json.loads(conv.history) if conv and conv.history else []
            provider = await self.provider_id(event)
            recent = [
                {
                    "id": r["id"],
                    "question": r["question"],
                    "items": [i["text"] for i in r["items"]],
                }
                for r in self.evidence_store.recent(scope)
            ]
            for record in recent[:-4]:
                record.pop("items")
            catalogue = self.catalogue.scope(names)
            plan = Plan.parse(
                await self.ask(
                    provider,
                    prompts.PLAN,
                    {
                        "message": question,
                        "history": history_text(history),
                        "local_catalogue": catalogue,
                        "recent_answers": recent,
                        "now": datetime.now().astimezone().isoformat(),
                    },
                    event,
                ),
                question,
            )
            allowed = {d["kb_name"] for d in catalogue}
            plan.kb_names = [n for n in plan.kb_names if n in allowed]
            if plan.local_only:
                plan.source = "kb"
                plan.kb_names = plan.kb_names or names
            elif plan.source in ("kb", "both") and not plan.kb_names:
                plan.source = "web"
            state.update(plan=plan, provider=provider, status="planned")
            if plan.route == "research":
                target = self.config.get("research_provider_id", "").strip()
                if target and not event.get_extra("selected_provider"):
                    event.set_extra("selected_provider", target)
            state["writer_provider"] = await self.selected_provider_id(event)
            event.set_extra(
                SKIP_MEMORY,
                plan.exclude_memory
                or plan.route in ("memory_control", "citation")
                or (
                    plan.route == "research"
                    and self.config.get("isolate_research_memory", True)
                ),
            )
            self.memory_status(event, state)
        except asyncio.CancelledError:
            event.set_extra(STATE, None)
            event.set_extra(SKIP_MEMORY, True)
            raise
        except Exception as error:  # noqa: BLE001 - native/provider boundary
            state.update(
                plan=Plan("unavailable", question),
                status="planning_failed",
                error=type(error).__name__,
            )
            event.set_extra(SKIP_MEMORY, True)
            self.memory.attach(self.context)
            logger.warning(
                "[证据问答 %s] 规划失败：%s", state["trace_id"], type(error).__name__
            )

    async def retrieve(self, event, query, packet, source_override=None):
        state = event.get_extra(STATE)
        plan = state["plan"]
        channels = (
            (source_override,)
            if source_override
            else (("kb", "web") if plan.source == "both" else (plan.source,))
        )
        if web.is_url(query) and not plan.local_only:
            channels = ("web",)
        for channel in channels:
            if len(packet.operations) >= 6:
                return
            start = asyncio.get_running_loop().time()
            operation = {"query": query, "channel": channel}
            packet.operations.append(operation)
            packet.queries.append(query)
            try:
                async with self.window(event, self.limits["retrieval_timeout_seconds"]):
                    if channel == "web":
                        result = await web.search(
                            self.context,
                            event,
                            query,
                            page_timeout=self.limits["page_timeout_seconds"],
                        )
                    else:
                        names = [n for n in plan.kb_names if n in state["kb_names"]]
                        if not names:
                            operation.update(status="skipped", reason="no_selected_kb")
                            continue
                        config = self.context.get_config(umo=event.unified_msg_origin)
                        result = await self.context.kb_manager.retrieve(
                            query=query,
                            kb_names=names,
                            top_k_fusion=config.get("kb_fusion_top_k", 20),
                            top_m_final=state["top_k"],
                        )
                    packet.add(query, result, record_query=False)
                    operation["status"] = "completed"
            except Exception as error:  # noqa: BLE001 - native/provider boundary
                packet.errors.append(type(error).__name__)
                operation.update(
                    status="failed",
                    error=type(error).__name__,
                    detail=diagnostic(error, 600),
                )
            finally:
                operation["seconds"] = round(
                    asyncio.get_running_loop().time() - start, 3
                )

    @filter.on_llm_request(priority=100)
    async def prepare_evidence(self, event: AstrMessageEvent, req: ProviderRequest):
        state = event.get_extra(STATE)
        if not self.enabled(event) or not state:
            return
        self.memory_status(event, state)
        plan = state["plan"]
        if req.conversation is not None:
            state["scope"] = [event.unified_msg_origin, req.conversation.cid]
        if plan.route == "delegate":
            return
        req.func_tool = None
        if plan.route == "chat":
            if not plan.detailed:
                req.system_prompt = (req.system_prompt or "") + "\n\n" + prompts.CHAT
        elif plan.route == "research" and "packet" not in state:
            packet = Packet(plan.question, state["question"])
            state["packet"] = packet
            initial = plan.queries[: 2 if plan.source == "both" else 4]
            await asyncio.gather(*(self.retrieve(event, q, packet) for q in initial))
            if not packet.sources and plan.source == "kb" and not plan.local_only:
                await self.retrieve(
                    event,
                    initial[0] if initial else plan.question,
                    packet,
                    source_override="web",
                )
            req.system_prompt = (
                (req.system_prompt or "") + "\n\n" + prompts.ANSWER_RULES
            )
            req.extra_user_content_parts.append(
                TextPart(
                    text=json.dumps(packet.payload(), ensure_ascii=False, default=str)
                ).mark_as_temp()
            )
        state["generation_started"] = asyncio.get_running_loop().time()

    def citation_answer(self, state):
        plan = state["plan"]
        record = self.evidence_store.find(state["scope"], plan.reference)
        if not record:
            return ""
        items = record["items"]
        if plan.item_ids:
            items = [items[i] for i in plan.item_ids if i < len(items)]
        packet = Packet(record["question"], sources=record["sources"])
        state["citation_record"] = record["id"]
        return render(items, [], packet, citations=True)

    @filter.on_llm_response(priority=100)
    async def verify_answer(self, event: AstrMessageEvent, resp: LLMResponse):
        state = event.get_extra(STATE)
        if (
            not self.enabled(event)
            or not state
            or "plan" not in state
            or resp.role != "assistant"
            or resp.tools_call_name
            or state.get("status") == "finished"
        ):
            return
        plan = state["plan"]
        if plan.route == "delegate":
            return
        self.memory_status(event, state)
        if "generation_started" in state:
            state.setdefault("timings", []).append(
                {
                    "phase": "generation",
                    "seconds": round(
                        asyncio.get_running_loop().time() - state["generation_started"],
                        3,
                    ),
                }
            )
        answer = ""
        try:
            if plan.route == "chat":
                answer = resp.completion_text
                if not plan.detailed and not short_chat(answer):
                    answer = await self.ask(
                        state["writer_provider"],
                        prompts.SHORTEN,
                        {"message": state["question"], "reply": answer},
                        event,
                    )
                    if not short_chat(answer):
                        answer = ""
            elif plan.route == "citation":
                answer = self.citation_answer(state)
            elif plan.route == "memory_control":
                answer = "这轮已请求跳过记忆记录，已有记忆没有改动。"
            elif plan.route == "research":
                packet = state.get("packet", Packet(plan.question))
                try:
                    followups = strings(
                        json_object(resp.completion_text).get("queries"), 2
                    )
                except (ValueError, TypeError):
                    followups = []
                if packet.sources or followups:
                    items, status = await review(
                        lambda system, payload: self.ask(
                            state["provider"]
                            if system == prompts.AUDIT
                            else state["writer_provider"],
                            system,
                            payload,
                            event,
                        ),
                        lambda query, p: self.retrieve(event, query, p),
                        packet,
                        resp.completion_text,
                    )
                    state.update(items=items, verification=status)
                    answer = render(items, [], packet)
                    if not answer:
                        logger.warning(
                            "[证据问答 %s] 无可发送答案：%s；各轮校验=%s",
                            state["trace_id"],
                            status,
                            diagnostic(
                                [step.get("issues", []) for step in packet.diagnostics],
                                1000,
                            ),
                        )
                else:
                    state["verification"] = "silent_no_sources"
        except Exception as error:  # noqa: BLE001 - native/provider boundary
            state.update(verification="silent_error", error=type(error).__name__)
            logger.warning(
                "[证据问答 %s] 核验失败：%s", state["trace_id"], type(error).__name__
            )
        resp.result_chain = None
        resp.completion_text = answer
        resp.reasoning_content = None
        resp.reasoning_signature = None
        event.set_extra("_llm_reasoning_content", None)
        state.update(status="finished", response_replaced=True)

    @filter.on_agent_done(priority=100)
    async def sync_final_history(self, event, run_context, resp):
        state = event.get_extra(STATE)
        if not state or state.get("status") != "finished":
            return
        self.memory_status(event, state)
        state["elapsed_seconds"] = round(
            asyncio.get_running_loop().time() - state["started"], 3
        )
        if state["plan"].route == "memory_control":
            resp.completion_text = (
                "这轮已跳过记忆召回和记录，已有记忆没改动。"
                if state["memory_isolated"]
                else "这轮已请求不保存，但尚未确认记忆插件执行了跳过。"
            )
        messages = run_context.messages
        state["history_synced"] = False
        if messages and messages[-1].role == "assistant":
            messages[-1].content = [TextPart(text=resp.completion_text)]
            state["history_synced"] = True
        logger.info(
            "[证据问答 %s] route=%s source=%s streaming=%s replaced=%s elapsed=%.3fs memory=%s",
            state["trace_id"],
            state["plan"].route,
            state["plan"].source,
            state.get("streaming"),
            state.get("response_replaced"),
            state["elapsed_seconds"],
            state["memory_isolated"],
        )
        if state["plan"].route == "research":
            packet = state.get("packet", Packet(state["question"]))
            try:
                items = state.get("items", []) if resp.completion_text else []
                used = {ref["source"] for item in items for ref in item["evidence"]}
                # Include an empty evidence record for failed turns so 'last turn'
                # cannot silently become the preceding successful research answer.
                self.evidence_store.save(
                    {
                        "id": state["trace_id"],
                        "scope": state["scope"],
                        "question": state["plan"].question,
                        "answer": resp.completion_text,
                        "items": items,
                        "sources": [s for s in packet.sources if s["id"] in used],
                    }
                )
                if self.config.get("save_traces", True):
                    await asyncio.to_thread(
                        self.save_trace, state, packet, resp.completion_text
                    )
            except (OSError, ValueError, TypeError) as error:
                logger.warning(
                    "[证据问答 %s] 记录保存失败：%s",
                    state["trace_id"],
                    type(error).__name__,
                )

    async def finish_interrupted(self, event, runner, reason):
        state = event.get_extra(STATE)
        if not state:
            started = asyncio.get_running_loop().time()
            state = {
                "trace_id": uuid4().hex,
                "question": event.get_message_str(),
                "started": started,
                "plan": Plan("unavailable", event.get_message_str()),
            }
            event.set_extra(STATE, state)
        state.update(status="finished", verification=reason, response_replaced=True)
        event.set_extra(SKIP_MEMORY, True)
        response = LLMResponse(role="assistant", completion_text="")
        logger.warning("[证据问答 %s] 本轮静默结束：%s", state["trace_id"], reason)
        if runner.done():
            runner.final_llm_resp = response
            await self.sync_final_history(event, runner.run_context, response)
        else:
            await runner._complete_with_assistant_response(response)
        return response

    def save_trace(self, state, packet, answer):
        folder = StarTools.get_data_dir("astrbot_plugin_evidence_qa") / "traces"
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            **{
                k: v
                for k, v in state.items()
                if k not in ("packet", "plan", "deadline")
            },
            "plan": asdict(state["plan"]),
            "timeouts": self.limits,
            **packet.payload(),
            "validation": packet.diagnostics,
            "answer": answer,
        }
        (folder / (state["trace_id"] + ".json")).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        for old in sorted(
            folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[20:]:
            old.unlink()

    async def terminate(self):
        self.selection.close()
        self.memory.close()
        if BufferedPrivateChat.plugin is self:
            BufferedPrivateChat.plugin = None
