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
            if self.managed(event) and plugin_context is self.plugin.context:
                await self.plugin.plan_turn(event)
            return await original(event, plugin_context)

        self._patch(owner, "_select_provider", select)
        self.active = True
        if run_module is not None:
            try:
                self._install_kb(owner)
                self._install_runner(run_module)
            except Exception:
                self.close()
                raise

    def _patch(self, owner, name, replacement):
        self.patches.append((owner, name, getattr(owner, name), replacement))
        setattr(owner, name, replacement)

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

    def _install_runner(self, owner):
        original = owner.run_agent

        @wraps(original)
        async def run(runner, *args, **kwargs):
            event = runner.run_context.context.event
            state = event.get_extra(STATE)
            if not self.managed(event) or (state and state["plan"].route == "delegate"):
                async for part in original(runner, *args, **kwargs):
                    yield part
                return
            runner.streaming = False
            if state:
                state["streaming"] = False
            if state and state["plan"].route in (
                "citation",
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

            iterator = original(runner, *args, **kwargs)
            reason = None
            try:
                try:
                    if state is None:
                        reason = "planning_not_observed"
                    else:
                        async with self.plugin.window(
                            event, self.plugin.limits["turn_timeout_seconds"]
                        ):
                            await anext(iterator)
                        if state.get("status") != "finished":
                            reason = "unverified_output"
                except StopAsyncIteration:
                    if (
                        event.is_stopped()
                        or event.get_extra("agent_user_aborted")
                        or event.get_extra("agent_stop_requested")
                    ):
                        return
                    if state and state.get("status") == "finished":
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
