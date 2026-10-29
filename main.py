"""Native KB evidence workflow, per-turn routing and optional memory isolation."""

import asyncio
import json
from dataclasses import asdict
from uuid import uuid4

from astrbot.api import AstrBotConfig, logger, sp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart

from . import prompts
from .evidence import Packet, Plan, excerpt_fallback, history_text, review
from .integration import SelectionBridge
from .memory import SKIP_MEMORY, MemoryBridge

STATE = "evidence_qa.state"


class BufferedPrivateChat(filter.CustomFilter):
    plugin = None

    def filter(self, event, cfg):
        plugin = self.plugin
        if plugin and plugin.enabled(event):
            event.set_extra("enable_streaming", False)
        return False  # Never wake a bot or consume a message.


class EvidenceQAPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.memory = MemoryBridge()
        self.selection = SelectionBridge(self)
        BufferedPrivateChat.plugin = self

    async def initialize(self):
        from astrbot.core import astr_main_agent

        self.selection.install(astr_main_agent)

    def enabled(self, event):
        selected = getattr(event, "plugins_name", None)
        return (
            self.config.get("enabled", True)
            and event.get_message_type() == MessageType.FRIEND_MESSAGE
            and (selected is None or "astrbot_plugin_evidence_qa" in selected)
        )

    @filter.custom_filter(BufferedPrivateChat, False)
    async def buffer_reply(self, event: AstrMessageEvent):
        pass

    async def provider_id(self, event):
        configured = self.config.get("worker_provider_id", "").strip()
        return configured or await self.selected_provider_id(event)

    async def selected_provider_id(self, event):
        selected = event.get_extra("selected_provider")
        if selected:
            return selected
        provider = await self.context.get_using_provider_async(event.unified_msg_origin)
        if provider is None:
            raise ValueError("没有可用的文本模型")
        return provider.meta().id

    async def ask(self, provider, system, payload):
        async with asyncio.timeout(90):
            response = await self.context.llm_generate(
                chat_provider_id=provider,
                system_prompt=system,
                prompt=json.dumps(payload, ensure_ascii=False, default=str),
                contexts=[],
                tools=None,
            )
        return response.completion_text

    async def plan_turn(self, event: AstrMessageEvent):
        if not self.enabled(event) or event.get_extra(STATE) is not None:
            return
        question = event.get_message_str()
        if not question.strip():
            return
        state = {"trace_id": uuid4().hex, "question": question, "status": "planning"}
        event.set_extra(STATE, state)
        try:
            manager = self.context.conversation_manager
            cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
            conv = (
                await manager.get_conversation(event.unified_msg_origin, cid)
                if cid
                else None
            )
            history = json.loads(conv.history) if conv and conv.history else []
            provider = await self.provider_id(event)
            for attempt in range(2):
                try:
                    plan = Plan.parse(
                        await self.ask(
                            provider,
                            prompts.PLAN,
                            {"message": question, "history": history_text(history)},
                        ),
                        question,
                    )
                    break
                except (TimeoutError, ValueError, TypeError):
                    if attempt:
                        raise
            state.update(plan=plan, provider=provider, status="planned")
            if plan.route == "research":
                target = self.config.get("research_provider_id", "").strip()
                if target and not event.get_extra("selected_provider"):
                    event.set_extra("selected_provider", target)
                state["writer_provider"] = await self.selected_provider_id(event)
            event.set_extra(
                SKIP_MEMORY,
                plan.exclude_memory
                or plan.route == "memory_control"
                or (
                    plan.route == "research"
                    and self.config.get("isolate_research_memory", True)
                ),
            )
            state["memory_isolated"] = self.memory.attach(self.context)
        except asyncio.CancelledError:
            event.set_extra(STATE, None)
            event.set_extra(SKIP_MEMORY, True)
            raise
        except Exception as error:  # noqa: BLE001 - provider/extension boundary
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

    async def retrieve(self, event, query, packet):
        stage = "selection"
        try:
            config = self.context.get_config(umo=event.unified_msg_origin)
            session = await sp.session_get(
                event.unified_msg_origin, "kb_config", default={}
            )
            names = config.get("kb_names", [])
            top_k = config.get("kb_final_top_k", 5)
            if session and "kb_ids" in session:
                names = []
                for kb_id in session["kb_ids"]:
                    helper = await self.context.kb_manager.get_kb(kb_id)
                    if helper:
                        names.append(helper.kb.kb_name)
                top_k = session.get("top_k", 5)
            if not names:
                packet.operations.append(
                    {"query": query, "status": "skipped", "reason": "no_selected_kb"}
                )
                return
            stage = "retrieval"
            packet.queries.append(query)
            result = await self.context.kb_manager.retrieve(
                query=query,
                kb_names=names,
                top_k_fusion=config.get("kb_fusion_top_k", 20),
                top_m_final=top_k,
            )
            packet.add(query, result, record_query=False)
            packet.operations.append({"query": query, "status": "completed"})
        except Exception as error:  # noqa: BLE001 - native retrieval/provider boundary
            packet.errors.append(type(error).__name__)
            packet.operations.append(
                {
                    "query": query,
                    "status": "failed",
                    "stage": stage,
                    "error": type(error).__name__,
                }
            )

    @filter.on_llm_request(priority=100)
    async def prepare_evidence(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.enabled(event):
            return
        state = event.get_extra(STATE)
        if not state:
            return
        state["memory_isolated"] = self.memory.attach(self.context)
        state["memory_request_isolated"] = state["memory_isolated"] and event.get_extra(
            SKIP_MEMORY, False
        )
        plan = state["plan"]
        if plan.route in ("unavailable", "memory_control"):
            req.func_tool = None
            return
        if plan.route != "research" or "packet" in state:
            return
        req.func_tool = None
        packet = Packet(plan.question, state["question"])
        for query in plan.queries:
            await self.retrieve(event, query, packet)
        state["packet"] = packet
        req.system_prompt = (req.system_prompt or "") + "\n\n" + prompts.ANSWER_RULES
        req.extra_user_content_parts.append(
            TextPart(
                text=json.dumps(packet.payload(), ensure_ascii=False, default=str)
            ).mark_as_temp()
        )
        logger.info(
            "[证据问答 %s] %s次检索，%s个原文片段，记忆隔离=%s",
            state["trace_id"],
            len(packet.queries),
            len(packet.sources),
            state["memory_isolated"],
        )

    @filter.on_llm_response(priority=100)
    async def verify_answer(self, event: AstrMessageEvent, resp: LLMResponse):
        state = event.get_extra(STATE)
        if (
            not self.enabled(event)
            or not state
            or "plan" not in state
            or resp.role != "assistant"
        ):
            return
        if resp.tools_call_name or state.get("status") == "finished":
            return
        state["memory_isolated"] = self.memory.attach(self.context) and state.get(
            "memory_request_isolated", False
        )
        plan = state["plan"]
        if plan.route == "unavailable":
            resp.completion_text = (
                "任务规划暂时不可用，本轮未完成知识库核验，请稍后重试。"
            )
        elif plan.route == "memory_control":
            resp.completion_text = (
                "本轮已跳过 LivingMemory 的记忆召回和记录；插件接管的技术问答也会按此处理。已有记忆和之前的总结输入未修改。"
                if state["memory_isolated"] and event.get_extra(SKIP_MEMORY, False)
                else "已收到不将本轮结论作为长期事实保存的要求，但当前没有建立可验证的 LivingMemory 写入控制；已有记忆未修改。"
            )
        elif plan.route == "research":
            packet = state.get("packet", Packet(plan.question))
            try:
                answer, status = await review(
                    lambda system, payload: self.ask(
                        state["provider"]
                        if system == prompts.AUDIT
                        else state.get("writer_provider", state["provider"]),
                        system,
                        payload,
                    ),
                    lambda query, p: self.retrieve(event, query, p),
                    packet,
                    resp.completion_text,
                )
            except Exception as error:  # noqa: BLE001 - provider boundary
                packet.errors.append(type(error).__name__)
                answer, status = excerpt_fallback(packet), "excerpts"
            resp.completion_text = answer
            state["verification"] = status
        if plan.route != "chat":
            resp.reasoning_content = None
            resp.reasoning_signature = None
            event.set_extra("_llm_reasoning_content", None)
        state["status"] = "finished"
        if plan.route == "research" and self.config.get("save_traces", True):
            try:
                await asyncio.to_thread(self.save_trace, state, packet, answer)
            except (OSError, ValueError, TypeError) as error:
                logger.warning(
                    "[证据问答 %s] 无法保存检索记录：%s",
                    state["trace_id"],
                    type(error).__name__,
                )

    @filter.on_agent_done(priority=100)
    async def sync_final_history(self, event, run_context, resp):
        state = event.get_extra(STATE)
        if (
            not state
            or state.get("status") != "finished"
            or state["plan"].route == "chat"
        ):
            return
        messages = run_context.messages
        if messages and messages[-1].role == "assistant":
            # AstrBot appends its draft before response hooks. Replace only this turn.
            messages[-1].content = [TextPart(text=resp.completion_text)]

    def save_trace(self, state, packet, answer):
        folder = StarTools.get_data_dir("astrbot_plugin_evidence_qa") / "traces"
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "trace_id": state["trace_id"],
            "plan": asdict(state["plan"]),
            "worker_provider": state["provider"],
            "writer_provider": state.get("writer_provider", state["provider"]),
            "verification": state["verification"],
            "memory_isolated": state["memory_isolated"],
            **packet.payload(),
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
