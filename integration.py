"""Reversible adapters for AstrBot 4.29.3; managed turns emit final text only."""

import asyncio
from functools import wraps

from astrbot.api import logger
from astrbot.api.provider import LLMResponse
from astrbot.core.message.message_event_result import (
    MessageChain,
    MessageEventResult,
    ResultContentType,
)

from .media import apply_to_request, recover_onebot_image

STATE = "evidence_qa.state"


async def emit(event, response):
    if not response or not response.completion_text:
        event.clear_result()
        return
    chain = MessageChain().message(response.completion_text)
    event.set_result(
        MessageEventResult(
            chain=chain.chain, result_content_type=ResultContentType.LLM_RESULT
        )
    )
    try:
        yield chain
    finally:
        event.clear_result()


class SelectionBridge:
    def __init__(self, plugin):
        self.plugin = plugin
        self.patches = []
        self.active = False

    def managed(self, event):
        return self.active and self.plugin.enabled(event)

    def install(self, owner, run_module=None):
        if self.active:
            return
        original = owner._select_provider

        @wraps(original)
        async def select(event, plugin_context):
            if self.plugin.routing.blocked(event):
                event.stop_event()
                return None
            if self.managed(event) and plugin_context is self.plugin.context:
                await self.plugin.plan_turn(event)
            return await original(event, plugin_context)

        self._patch(owner, "_select_provider", select)
        self.active = True
        if run_module is not None:
            try:
                if hasattr(owner, "collect_initial_request"):
                    self._install_media(owner)
                self._install_kb(owner)
                self._install_helpers(owner)
                self._install_runner(run_module)
            except Exception:
                self.close()
                raise

    def _patch(self, owner, name, replacement):
        self.patches.append((owner, name, getattr(owner, name), replacement))
        setattr(owner, name, replacement)

    def install_image_recovery(self, stage):
        original = stage._normalize_image_component

        @wraps(original)
        async def normalize(stage_self, event, component):
            try:
                return await original(stage_self, event, component)
            except Exception:
                if not (
                    self.managed(event)
                    and event.get_platform_name() == "aiocqhttp"
                    and component.file
                ):
                    raise
                before = component.file, component.path, component.url
                try:
                    async with asyncio.timeout(
                        self.plugin.limits["page_timeout_seconds"]
                    ):
                        ref = await recover_onebot_image(event, component)
                        component.file = component.url = ref
                        await original(stage_self, event, component)
                except (Exception, asyncio.CancelledError):
                    component.file, component.path, component.url = before
                    raise
                logger.info("[证据问答] QQ图片已通过 OneBot get_image 恢复")

        self._patch(stage, "_normalize_image_component", normalize)

    def _install_media(self, owner):
        original = owner.collect_initial_request

        @wraps(original)
        async def collect(event, plugin_context, *args, **kwargs):
            req, quote_ref = await original(event, plugin_context, *args, **kwargs)
            state = event.get_extra(STATE)
            if (
                req is not None
                and state
                and self.managed(event)
                and plugin_context is self.plugin.context
            ):
                apply_to_request(state, req)
                if state.get("visual_context") and state["plan"].route != "delegate":
                    # Already seen by the planner; avoid a second quote-caption model call.
                    quote_ref = None
            return req, quote_ref

        self._patch(owner, "collect_initial_request", collect)

    def _install_kb(self, owner):
        original = owner._apply_kb

        @wraps(original)
        async def apply(event, req, context, config):
            state = event.get_extra(STATE)
            if (
                self.managed(event)
                and context is self.plugin.context
                and state
                and state["plan"].route != "delegate"
            ):
                state["native_kb_stage"] = {
                    "status": "skipped",
                    "reason": "managed_routing",
                }
                return
            return await original(event, req, context, config)

        self._patch(owner, "_apply_kb", apply)

    def _install_helpers(self, owner):
        if hasattr(owner, "_get_compress_provider"):
            original_compress = owner._get_compress_provider

            @wraps(original_compress)
            async def compress(config, context, event=None):
                if (
                    self.plugin.routing.enabled
                    and context is self.plugin.context
                    and (event is None or self.managed(event))
                    and config.context_limit_reached_strategy == "llm_compress"
                ):
                    return self.plugin.routing.get_provider(
                        self.plugin.routing.worker()
                    )
                return await original_compress(config, context, event)

            self._patch(owner, "_get_compress_provider", compress)
        if hasattr(owner, "_handle_webchat"):
            original_title = owner._handle_webchat

            @wraps(original_title)
            async def title(event, request, provider):
                if self.plugin.routing.enabled and self.managed(event):
                    provider = self.plugin.routing.get_provider(
                        self.plugin.routing.worker()
                    )
                return await original_title(event, request, provider)

            self._patch(owner, "_handle_webchat", title)

    def _install_runner(self, owner):
        original = owner.run_agent

        @wraps(original)
        async def run(runner, *args, **kwargs):
            event = runner.run_context.context.event
            state = event.get_extra(STATE)
            if self.plugin.routing.blocked(event):
                event.stop_event()
                return
            if (
                self.managed(event)
                and state
                and self.plugin.routing.enabled
                and state["plan"].route == "delegate"
            ):
                runner.provider = self.plugin.routing.get_provider(
                    self.plugin.routing.worker()
                )
                runner.fallback_providers = []
                runner.req.model = None
            if (
                not self.managed(event)
                or not state
                or state["plan"].route == "delegate"
            ):
                async for part in original(runner, *args, **kwargs):
                    yield part
                return
            runner.streaming = False
            state["streaming"] = False
            if state["plan"].route in (
                "citation",
                "catalogue",
                "memory_control",
                "unavailable",
            ):
                await runner._complete_with_assistant_response(
                    LLMResponse(role="assistant", completion_text="")
                )
                if state.get("status") == "finished":
                    async for chain in emit(event, runner.get_final_llm_resp()):
                        yield chain
                return

            if self.plugin.routing.enabled:
                runner.provider = self.plugin.routing.answer_provider(state, event)
                runner.fallback_providers = []
                runner.req.model = None

            # Native lookup steps may yield intermediate text; only the final answer is sent.
            options = list(args)
            for index, key, value in (
                (1, "show_tool_use", False),
                (2, "show_tool_call_result", False),
            ):
                if len(options) > index:
                    options[index] = value
                else:
                    kwargs[key] = value
            if state["plan"].route == "research":
                if options:
                    options[0] = min(options[0], 3)
                else:
                    kwargs["max_step"] = min(kwargs.get("max_step", 128), 3)
            iterator = original(runner, *options, **kwargs)
            reason = None
            try:
                try:
                    async with self.plugin.window(
                        event, self.plugin.limits["turn_timeout_seconds"]
                    ):
                        async for _ in iterator:
                            if state.get("status") == "finished":
                                break
                    if state.get("status") != "finished":
                        if (
                            event.is_stopped()
                            or event.get_extra("agent_user_aborted")
                            or event.get_extra("agent_stop_requested")
                        ):
                            return
                        reason = "runner_incomplete"
                except TimeoutError:
                    reason = "turn_timeout"
                except Exception as error:  # noqa: BLE001 - native runner boundary
                    reason = "runner_error:" + type(error).__name__
                if reason:
                    response = await self.plugin.finish_interrupted(
                        event, runner, reason
                    )
                else:
                    response = runner.get_final_llm_resp()
                async for chain in emit(event, response):
                    yield chain
                # Cleanup may log an error, but must not undo or append to an emitted answer.
                try:
                    async with asyncio.timeout(2):
                        async for _ in iterator:
                            pass
                except Exception as error:  # noqa: BLE001 - native generator cleanup
                    logger.warning(
                        "[证据问答] 输出后的清理异常：%s", type(error).__name__
                    )
            finally:
                try:
                    await iterator.aclose()
                except Exception as error:  # noqa: BLE001 - native generator cleanup
                    logger.warning(
                        "[证据问答] 关闭生成器异常：%s", type(error).__name__
                    )

        self._patch(owner, "run_agent", run)

    def close(self):
        self.active = False
        for owner, name, original, wrapper in reversed(self.patches):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
        self.patches.clear()
