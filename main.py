"""Native KB evidence workflow, per-turn routing and optional memory isolation."""

import asyncio
import json
import math
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
from .memory import MEMORY_STEPS, SKIP_MEMORY, MemoryBridge

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
        self.limits = {}
        for name, default in (
            ("model_timeout_seconds", 90),
            ("retrieval_timeout_seconds", 45),
            ("turn_timeout_seconds", 240),
        ):
            value = float(config.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为大于零的秒数")
            self.limits[name] = value
        self.memory = MemoryBridge()
        self.selection = SelectionBridge(self)
        BufferedPrivateChat.plugin = self

    async def initialize(self):
        from astrbot.core import astr_main_agent
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

        self.selection.install(astr_main_agent, internal)
        logger.info(
            "[证据问答] AstrBot 4.29.3：已接管模型选择、原生知识库阶段和普通 Agent 输出；Live 不接管。"
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
        async with self.window(event, self.limits["model_timeout_seconds"]):
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
            async with self.window(event, self.limits["turn_timeout_seconds"]):
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
                            event,
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
            self.memory_status(event, state)
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
        async def selected_search():
            nonlocal stage
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

        stage = "selection"
        try:
            async with self.window(event, self.limits["retrieval_timeout_seconds"]):
                await selected_search()
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
        self.memory_status(event, state)
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
            if asyncio.get_running_loop().time() >= state["deadline"]:
                break
        state["packet"] = packet
        req.system_prompt = (req.system_prompt or "") + "\n\n" + prompts.ANSWER_RULES
        req.extra_user_content_parts.append(
            TextPart(
                text=json.dumps(packet.payload(), ensure_ascii=False, default=str)
            ).mark_as_temp()
        )
        logger.info(
            "[证据问答 %s] %s次检索，%s个原文片段，记忆适配器=%s，要求跳过记录=%s",
            state["trace_id"],
            len(packet.queries),
            len(packet.sources),
            state["memory_adapter_available"],
            state["memory_skip_requested"],
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
        self.memory_status(event, state)
        plan = state["plan"]
        if plan.route == "unavailable":
            resp.completion_text = (
                "任务规划暂时不可用，本轮未完成知识库核验，请稍后重试。"
            )
        elif plan.route == "memory_control":
            resp.completion_text = (
                "本轮已请求跳过 LivingMemory 的记忆召回和记录；已有记忆和之前的总结输入未修改。"
                if state["memory_adapter_available"]
                and event.get_extra(SKIP_MEMORY, False)
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
                        event,
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
            final_text = resp.completion_text
            resp.result_chain = None
            resp.completion_text = final_text
            resp.reasoning_content = None
            resp.reasoning_signature = None
            event.set_extra("_llm_reasoning_content", None)
            state["response_replaced"] = True
        state["status"] = "finished"

    @filter.on_agent_done(priority=100)
    async def sync_final_history(self, event, run_context, resp):
        state = event.get_extra(STATE)
        if (
            not state
            or state.get("status") != "finished"
            or state["plan"].route == "chat"
        ):
            return
        self.memory_status(event, state)
        if "started" in state:
            state["elapsed_seconds"] = round(
                asyncio.get_running_loop().time() - state["started"], 3
            )
        if state["plan"].route == "memory_control":
            resp.completion_text = (
                "本轮已跳过 LivingMemory 的记忆召回和记录；已有记忆和之前的总结输入未修改。"
                if state["memory_isolated"]
                else "本轮已请求不保存，但未观察到 LivingMemory 召回与记录两个入口都被跳过；已有记忆未修改。"
            )
        messages = run_context.messages
        state["history_synced"] = False
        if messages and messages[-1].role == "assistant":
            # AstrBot appends its draft before response hooks. Replace only this turn.
            messages[-1].content = [TextPart(text=resp.completion_text)]
            state["history_synced"] = True
        logger.info(
            "[证据问答 %s] streaming=%s，最终替换=%s，历史同步=%s，召回跳过=%s，记录跳过=%s",
            state["trace_id"],
            state.get("streaming"),
            state.get("response_replaced", False),
            state["history_synced"],
            state["memory_recall_skipped"],
            state["memory_record_skipped"],
        )
        if state["plan"].route == "research" and self.config.get("save_traces", True):
            try:
                await asyncio.to_thread(
                    self.save_trace,
                    state,
                    state.get("packet", Packet(state["question"])),
                    resp.completion_text,
                )
            except (OSError, ValueError, TypeError) as error:
                logger.warning(
                    "[证据问答 %s] 无法保存检索记录：%s",
                    state["trace_id"],
                    type(error).__name__,
                )

    async def finish_interrupted(self, event, runner, reason):
        state = event.get_extra(STATE)
        if not state:
            state = {
                "trace_id": uuid4().hex,
                "question": event.get_message_str(),
                "plan": Plan("unavailable", event.get_message_str()),
                "status": "unavailable",
            }
            event.set_extra(STATE, state)
        packet = state.get("packet", Packet(state["question"]))
        packet.errors.append(reason)
        answer = (
            excerpt_fallback(packet)
            if state["plan"].route == "research"
            else "本轮未完成处理，请稍后重试。"
        )
        state.update(status="finished", verification=reason, response_replaced=True)
        event.set_extra(SKIP_MEMORY, True)
        response = LLMResponse(role="assistant", completion_text=answer)
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
            "trace_id": state["trace_id"],
            "plan": asdict(state["plan"]),
            "worker_provider": state["provider"],
            "writer_provider": state.get("writer_provider", state["provider"]),
            "verification": state["verification"],
            "memory_isolated": state["memory_isolated"],
            **{
                key: state.get(key)
                for key in (
                    "memory_adapter_available",
                    "memory_skip_requested",
                    "memory_recall_skipped",
                    "memory_record_skipped",
                    "streaming",
                    "response_replaced",
                    "history_synced",
                    "native_kb_stage",
                )
            },
            "elapsed_seconds": state.get("elapsed_seconds"),
            "timeouts": self.limits,
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
