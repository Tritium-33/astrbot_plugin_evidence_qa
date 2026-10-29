"""Reversible AstrBot 4.29.3 selection, KB and non-Live output adapters."""

import asyncio
from functools import wraps

from astrbot.core.message.message_event_result import (
    MessageChain,
    MessageEventResult,
    ResultContentType,
)

STATE = "evidence_qa.state"


class SelectionBridge:
    def __init__(self, plugin):
        self.plugin = plugin
        self.patches = []
        self.active = False

    def install(self, owner, run_module=None):
        if self.active:
            return
        original = owner._select_provider

        @wraps(original)
        async def select(event, plugin_context):
            if (
                self.active
                and plugin_context is self.plugin.context
                and self.plugin.enabled(event)
            ):
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
                not (
                    self.active
                    and context is self.plugin.context
                    and self.plugin.enabled(event)
                    and state
                )
                or config.kb_agentic_mode
            ):
                return await original(event, req, context, config)
            stage = {"status": "running"}
            state["native_kb_stage"] = stage
            try:
                async with self.plugin.window(
                    event, self.plugin.limits["retrieval_timeout_seconds"]
                ):
                    await original(event, req, context, config)
                stage["status"] = "completed"
            except TimeoutError:
                stage.update(status="timeout", error="TimeoutError")

        self._patch(owner, "_apply_kb", apply)

    def _install_runner(self, owner):
        original = owner.run_agent

        @wraps(original)
        async def run(runner, *args, **kwargs):
            event = runner.run_context.context.event
            if not self.active or not self.plugin.enabled(event):
                async for part in original(runner, *args, **kwargs):
                    yield part
                return
            state = event.get_extra(STATE)
            runner.streaming = (
                False  # Actual provider execution, before its first step.
            )
            if state:
                state["streaming"] = runner.streaming
            iterator = original(runner, *args, **kwargs)
            try:
                while True:
                    reason = None
                    try:
                        if state is None:
                            reason = "planning_not_observed"
                        else:
                            finished = state.get("status") == "finished"
                            async with (
                                asyncio.timeout(None)
                                if finished
                                else self.plugin.window(
                                    event, self.plugin.limits["turn_timeout_seconds"]
                                )
                            ):
                                part = await anext(iterator)
                    except StopAsyncIteration:
                        if (
                            event.is_stopped()
                            or event.get_extra("agent_user_aborted")
                            or event.get_extra("agent_stop_requested")
                        ):
                            return
                        if (
                            state
                            and state["plan"].route != "chat"
                            and state.get("status") != "finished"
                        ):
                            reason = "runner_incomplete"
                        else:
                            return
                    except TimeoutError:
                        reason = "turn_timeout"

                    guarded = state is None or state["plan"].route != "chat"
                    if guarded and not reason and state.get("status") != "finished":
                        reason = "unverified_output"
                    if reason:
                        await iterator.aclose()
                        response = await self.plugin.finish_interrupted(
                            event, runner, reason
                        )
                    elif guarded:
                        response = runner.get_final_llm_resp()
                    else:
                        yield part
                        continue

                    # Only the final, reconciled text is passed to the send pipeline.
                    chain = MessageChain().message(response.completion_text)
                    event.set_result(
                        MessageEventResult(
                            chain=chain.chain,
                            result_content_type=ResultContentType.LLM_RESULT,
                        )
                    )
                    yield chain
                    event.clear_result()
                    async for _ in iterator:
                        pass  # Let the native generator finish its stop-watcher cleanup.
                    return
            finally:
                await iterator.aclose()

        self._patch(owner, "run_agent", run)

    def close(self):
        self.active = False
        for owner, name, original, wrapper in reversed(self.patches):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
        self.patches.clear()
