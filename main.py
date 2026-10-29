"""Native search routing, verified answers and short everyday conversation."""

import asyncio
import json
import math
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

from astrbot.api import AstrBotConfig, logger, sp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.tool import FunctionTool, ToolSet

from . import media, prompts, web
from .catalogue import Catalogue, EvidenceStore
from .evidence import (
    Packet,
    Plan,
    diagnostic,
    direct_answer,
    focused_review,
    history_text,
    json_object,
    render,
    source_label,
)
from .integration import SelectionBridge
from .memory import MEMORY_STEPS, SKIP_MEMORY, MemoryBridge
from .routing import ModelRouting

STATE = "evidence_qa.state"


class BufferedChat(filter.CustomFilter):
    plugin = None

    def filter(self, event, cfg):
        if self.plugin and self.plugin.routing.blocked(event):
            event.stop_event()
            return False
        if self.plugin and self.plugin.enabled(event):
            self.plugin.routing.attach_gcp()
            media.collect(event)
            event.set_extra("enable_streaming", False)
            if self.plugin.routing.direct_group(event):
                if event.get_sender_id() == event.get_self_id():
                    event.stop_event()
                    return False
                raw = event.message_obj.raw_message
                if (
                    isinstance(raw, dict)
                    and raw.get("post_type", "message") != "message"
                ):
                    return False
                if event.get_message_str().strip() or event.get_messages():
                    return True
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
        self.routing = ModelRouting(self, folder / "routing.json")
        self.memory = MemoryBridge()
        self.selection = SelectionBridge(self)
        BufferedChat.plugin = self

    async def initialize(self):
        from astrbot.core import astr_main_agent
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

        for cache in (self.catalogue, self.evidence_store, self.routing):
            try:
                await asyncio.to_thread(cache.load)
            except (OSError, ValueError, TypeError):
                logger.warning("[证据问答] 缓存读取失败：%s", cache.path.name)
        self.selection.install(astr_main_agent, internal)
        self.routing.attach_gcp()
        logger.info(
            "[证据问答] AstrBot 4.29.3：接管%d个入口；目录仅加载缓存，更新请用 /资料目录更新。",
            len(self.selection.patches),
        )

    def enabled(self, event):
        selected = getattr(event, "plugins_name", None)
        return (
            self.config.get("enabled", True)
            and not self.routing.blocked(event)
            and (
                event.get_message_type() == MessageType.FRIEND_MESSAGE
                or (
                    event.get_message_type() == MessageType.GROUP_MESSAGE
                    and self.config.get("enable_group", True)
                )
            )
            and event.get_extra("action_type") != "live"
            and (selected is None or "astrbot_plugin_evidence_qa" in selected)
        )

    @filter.custom_filter(BufferedChat, False, priority=10000)
    async def buffer_reply(self, event: AstrMessageEvent):
        # Run after native access and session-plugin checks, before GCP's handler.
        if self.routing.direct_group(event):
            event.is_at_or_wake_command = True
            event.set_extra("evidence_qa.direct_group", True)

    @filter.command("回答模式")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def answer_mode(self, event: AstrMessageEvent, mode: str = "状态"):
        if not self.routing.enabled:
            yield event.plain_result("请先启用模型分工。")
            return
        choices = {"gpt": "gpt", "旧分工": "legacy", "legacy": "legacy"}
        selected = choices.get(mode.lower())
        if selected:
            self.routing.switch(selected, "manual")
        elif mode != "状态":
            yield event.plain_result("用法：/回答模式 gpt、旧分工 或 状态")
            return
        yield event.plain_result(
            f"回答模式：{self.routing.mode}；原因：{self.routing.reason}。辅助模型始终使用 DeepSeek。"
        )

    @filter.command("模型检查")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def check_models(self, event: AstrMessageEvent):
        from .routing import GPT_ROLES

        lines = [f"回答模式：{self.routing.mode}（{self.routing.reason}）"]
        for label, key, effort in [
            ("辅助", "worker_provider_id", None),
            *(
                (role, row[0], self.config.get(row[1], row[2]))
                for role, row in GPT_ROLES.items()
            ),
            ("备用聊天", "legacy_chat_provider_id", None),
            ("备用分析", "research_provider_id", None),
            ("备用看图", "vision_provider_id", None),
        ]:
            identity = self.config.get(key, "").strip()
            provider = self.context.get_provider_by_id(identity) if identity else None
            detail = (
                f"{identity} / {provider.get_model()}"
                if provider
                else "未配置或ID不存在"
            )
            lines.append(f"{label}：{detail}" + (f" / {effort}" if effort else ""))
            if (
                effort
                and provider
                and provider.provider_config.get("type") != "openai_responses"
            ):
                lines.append("  请将此模型的接口类型设为 OpenAI Responses。")
        lines.append(f"GCP接管入口：{len(self.routing.gcp_patches)}；本检查不调用API。")
        yield event.plain_result("\n".join(lines))

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
        if self.routing.enabled:
            return self.routing.worker()
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

    async def vision_provider_id(self, event):
        explicit = self.config.get("vision_provider_id", "").strip()
        if explicit:
            return explicit
        settings = self.context.get_config(umo=event.unified_msg_origin).get(
            "provider_settings", {}
        )
        caption = settings.get("default_image_caption_provider_id", "")
        if caption:
            return caption
        return await self.selected_provider_id(event)

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
        state["memory_isolated"] = not state.get("group", False) and all(
            steps.get(k) is True for k in ("recall", "reflection")
        )

    async def ask(self, provider, system, payload, event=None, image_urls=None):
        start = asyncio.get_running_loop().time()
        try:
            async with self.window(event, self.limits["model_timeout_seconds"]):
                response = await self.context.llm_generate(
                    chat_provider_id=provider,
                    system_prompt=system,
                    prompt=json.dumps(payload, ensure_ascii=False, default=str),
                    contexts=[],
                    tools=None,
                    image_urls=image_urls or [],
                )
            return response.completion_text
        finally:
            state = event.get_extra(STATE) if event else None
            if state:
                phase = next(
                    (
                        name.lower()
                        for name in ("PLAN", "AUDIT")
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
        attachments = media.collect(event)
        chat_context = media.chat_context(event)
        if (
            not question.strip()
            and not chat_context.get("group_chat_plus")
            and not any(attachments[k] for k in ("refs", "quotes", "faces"))
        ):
            return
        question = question.strip() or "请结合当前对话回应这条消息及其附件。"
        started = asyncio.get_running_loop().time()
        state = {
            "trace_id": uuid4().hex,
            "question": question,
            "status": "planning",
            "started": started,
            "deadline": started + self.limits["turn_timeout_seconds"],
            "media": media.describe(attachments),
            "group": event.get_message_type() == MessageType.GROUP_MESSAGE,
            "direct_group": bool(event.get_extra("evidence_qa.direct_group")),
            "sender_id": chat_context["sender_id"],
            "sender_name": chat_context["sender_name"],
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
            # GCP supplies its own history and may never create a native conversation.
            session = cid or (
                "group_chat_plus"
                if chat_context.get("group_chat_plus")
                else state["trace_id"]
            )
            scope = [event.unified_msg_origin, session]
            state.update(scope=scope, kb_names=names, top_k=top_k)
            history = json.loads(conv.history) if conv and conv.history else []
            provider = await self.provider_id(event)
            recent = [
                {
                    "id": r["id"],
                    "question": r["question"],
                    "items": [i["text"] for i in r["items"]],
                    "has_answer": bool(r.get("answer")),
                    "sender_id": r.get("sender_id", ""),
                }
                for r in self.evidence_store.recent(scope)
            ]
            for record in recent[:-4]:
                record.pop("items")
            catalogue = self.catalogue.scope(names)
            image_urls = []
            planner = provider
            config = self.context.get_config(umo=event.unified_msg_origin)
            async with self.window(event, self.limits["retrieval_timeout_seconds"]):
                await media.resolve_quotes(event, config)
                attachments = media.collect(event)
                if attachments["refs"]:
                    image_urls = await media.prepare(event, config)
                    if not self.routing.enabled:
                        planner = await self.vision_provider_id(event)
                        state["vision_provider"] = planner
            state["media"] = media.describe(attachments)
            if self.routing.enabled:
                state["original_images"] = attachments["refs"]
                state["prepared_images"] = image_urls
            state["planner_provider"] = planner
            result = await self.ask(
                planner,
                prompts.PLAN,
                {
                    "message": question,
                    "history": history_text(history),
                    "local_catalogue": catalogue,
                    "recent_answers": recent,
                    "now": datetime.now().astimezone().isoformat(),
                    "media": state["media"],
                    "chat_context": chat_context,
                },
                event,
                image_urls=[] if self.routing.enabled else image_urls,
            )
            plan = Plan.parse(result, question)
            observation = json_object(result).get("visual_context", "")
            if image_urls and isinstance(observation, str) and not self.routing.enabled:
                state["visual_context"] = observation.strip()
            allowed = {d["kb_name"] for d in catalogue}
            plan.kb_names = [n for n in plan.kb_names if n in allowed]
            if plan.local_only:
                plan.source = "kb"
                plan.kb_names = plan.kb_names or names
            elif plan.source in ("kb", "both") and not plan.kb_names:
                plan.source = "web"
            state.update(plan=plan, provider=provider, status="planned")
            if plan.route == "catalogue":
                state["catalogue"] = catalogue
            if self.routing.enabled:
                self.routing.select(state, event)
            elif plan.route == "research":
                target = self.config.get("research_provider_id", "").strip()
                if target and not event.get_extra("selected_provider"):
                    event.set_extra("selected_provider", target)
            elif (
                plan.route == "chat"
                and image_urls
                and not event.get_extra("selected_provider")
            ):
                event.set_extra("selected_provider", planner)
            state["writer_provider"] = await self.selected_provider_id(event)
            state["describe_images_only"] = bool(
                not self.routing.enabled
                and plan.route != "delegate"
                and state.get("visual_context")
                and state["writer_provider"] != state.get("vision_provider")
            )
            event.set_extra(
                SKIP_MEMORY,
                plan.exclude_memory
                or plan.route in ("memory_control", "citation", "catalogue")
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
            # An image or emoji must not become silent merely because planning failed.
            if any(attachments[k] for k in ("refs", "quotes", "faces")):
                state.update(
                    plan=Plan("chat" if self.routing.enabled else "delegate", question),
                    status="planned",
                    error=type(error).__name__,
                    visual_fallback=True,
                )
                if self.routing.enabled:
                    try:
                        self.routing.select(state, event)
                    except ValueError:
                        state.update(
                            plan=Plan("unavailable", question), status="planning_failed"
                        )
                if state.get("vision_provider") and not event.get_extra(
                    "selected_provider"
                ):
                    event.set_extra("selected_provider", state["vision_provider"])
                event.set_extra(SKIP_MEMORY, False)
                logger.warning(
                    "[证据问答 %s] 图片规划失败，保留附件交给回答模型：%s",
                    state["trace_id"],
                    type(error).__name__,
                )
                return
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

    @filter.on_llm_request(priority=-100)
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
        media.apply_to_request(state, req)
        if state.get("group"):
            req.system_prompt = (req.system_prompt or "") + "\n\n" + prompts.GROUP
            req.extra_user_content_parts.append(
                TextPart(
                    text="当前发言者："
                    + json.dumps(
                        {"id": state["sender_id"], "name": state["sender_name"]},
                        ensure_ascii=False,
                    )
                )
            )
        if state.get("media", {}).get("image_count") or state.get("media", {}).get(
            "face_ids"
        ):
            req.system_prompt = (req.system_prompt or "") + "\n\n" + prompts.VISUAL
        if plan.route != "research":
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
            state["answer_mode"] = (
                "local_technical"
                if self.needs_audit(plan, packet) and not state.get("gpt_answer")
                else "direct"
            )
            rules = (
                prompts.ANSWER_RULES
                if state["answer_mode"] == "local_technical"
                else prompts.DIRECT
            )
            req.system_prompt = (req.system_prompt or "") + "\n\n" + rules
            req.func_tool = self.lookup_tools(event, packet)
            req.extra_user_content_parts.append(
                TextPart(
                    text=json.dumps(packet.payload(), ensure_ascii=False, default=str)
                ).mark_as_temp()
            )
        state["generation_started"] = asyncio.get_running_loop().time()

    @staticmethod
    def needs_audit(plan, packet):
        return (
            plan.technical
            and plan.source in ("kb", "both")
            and any(s.get("channel") != "web" for s in packet.sources)
        )

    def lookup_tools(self, event, packet):
        """Request-scoped native tool: further searches happen before the final answer."""
        state = event.get_extra(STATE)

        async def lookup(_event, query):
            if state.get("lookup_calls", 0) >= 2:
                return "本轮补查预算已用完，请根据现有材料直接完成回答。"
            state["lookup_calls"] = state.get("lookup_calls", 0) + 1
            if isinstance(query, str) and query.strip():
                await self.retrieve(event, query.strip(), packet)
            if state["lookup_calls"] >= 2:
                toolset.tools.clear()
            return json.dumps(
                {**packet.payload(), "remaining_lookups": 2 - state["lookup_calls"]},
                ensure_ascii=False,
                default=str,
            )

        toolset = ToolSet(
            [
                FunctionTool(
                    name="evidence_lookup",
                    description="材料不足时换词补查或读取具体HTTP(S)网址，沿用本轮允许的资料范围。最多调用两次；材料足够就直接回答。",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    handler=lookup,
                )
            ]
        )
        return toolset

    def catalogue_answer(self, state):
        documents = state.get("catalogue", [])
        if not documents:
            return ""
        return "已缓存的资料目录与内容概览：\n\n" + "\n\n".join(
            f"- **{d['doc_name']}**（{d['kb_name']}）\n  {d['summary']}"
            for d in documents
        )

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
        if record.get("catalogue"):
            return (
                "来源：手动更新时保存的原生知识库文档目录与概览。\n\n"
                + self.catalogue_answer(record)
            )
        if not record["items"] and record.get("answer") and packet.sources:
            return "本轮检索到的来源（未建立逐句对应）：\n\n" + "\n".join(
                "- " + source_label(s, packet) for s in packet.sources
            )
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
                state["verification"] = "direct_chat"
            elif plan.route == "catalogue":
                answer = self.catalogue_answer(state)
                state["verification"] = "cached_catalogue"
            elif plan.route == "citation":
                answer = self.citation_answer(state)
            elif plan.route == "memory_control":
                answer = "这轮已请求跳过记忆记录，已有记忆没有改动。"
            elif plan.route == "research":
                packet = state.get("packet", Packet(plan.question))
                if packet.sources:
                    if self.needs_audit(plan, packet) and not state.get("gpt_answer"):
                        state["answer_mode"] = "local_technical"
                        items, status = await focused_review(
                            lambda system, payload: self.ask(
                                state["provider"], system, payload, event
                            ),
                            packet,
                            resp.completion_text,
                        )
                        answer = render(items, [], packet)
                    else:
                        state["answer_mode"] = "direct"
                        answer, items = direct_answer(resp.completion_text, packet)
                        status = "direct_research"
                    state.update(items=items, verification=status)
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
                "这轮已请求跳过回复相关记忆；群消息自动采集不在此控制范围内。"
                if state.get("group")
                else "这轮已跳过记忆召回和记录，已有记忆没改动。"
                if state["memory_isolated"]
                else "这轮已请求不保存，但尚未确认记忆插件执行了跳过。"
            )
        messages = run_context.messages
        # Keep final dialogue across models, without carrying search tool payloads forever.
        lookup_ids = {
            call.id
            for message in messages
            for call in message.tool_calls or []
            if call.function.name == "evidence_lookup"
        }
        if lookup_ids:
            retained = []
            for message in messages:
                if message.tool_call_id in lookup_ids:
                    continue
                if message.tool_calls:
                    calls = [c for c in message.tool_calls if c.id not in lookup_ids]
                    if not calls:
                        continue
                    message.tool_calls = calls
                retained.append(message)
            messages[:] = retained
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
        if state["plan"].route in ("research", "catalogue"):
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
                        "sender_id": state.get("sender_id", ""),
                        "answer": resp.completion_text,
                        "items": items,
                        "sources": [s for s in packet.sources if s["id"] in used]
                        if items
                        else (packet.sources if resp.completion_text else []),
                        "mode": state.get("answer_mode", "direct"),
                        "catalogue": state.get("catalogue", []),
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
        self.routing.close()
        self.memory.close()
        if BufferedChat.plugin is self:
            BufferedChat.plugin = None
