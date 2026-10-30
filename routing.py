"""Answer-only model profiles; helpers retain their own DeepSeek provider."""

import copy
import json
import sys
from datetime import datetime
from functools import wraps
from types import MethodType

from astrbot.api import logger
from astrbot.api.platform import MessageType

from .group_policy import GroupPolicy

GPT_ROLES = {
    "private": ("gpt_private_provider_id", "private_effort", "low"),
    "chat": ("gpt_chat_provider_id", "chat_effort", "low"),
    "analysis": ("gpt_analysis_provider_id", "analysis_effort", "medium"),
}
ANSWER_MODES = {"GPT优先": "gpt", "Gemini/DeepSeek": "legacy"}
QUOTA_CODES = {
    "credit_balance_exhausted",
    "insufficient_quota",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    "billing_hard_limit_reached",
}


def quota_code(error):
    """Only structured billing/quota codes trigger the persistent fallback."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    code = body.get("code") if isinstance(body, dict) else None
    code = code or getattr(error, "code", None)
    return code if isinstance(code, str) and code in QUOTA_CODES else None


class ModelRouting:
    def __init__(self, plugin, path):
        self.plugin = plugin
        self.config = plugin.config
        self.path = path
        self._mode = "gpt"
        self.reason = "default"
        self.gcp_patches = []
        self.active = True
        self.groups = GroupPolicy(plugin)

    @property
    def enabled(self):
        return self.active and bool(self.config.get("model_routing_enabled", False))

    @property
    def mode(self):
        selected = ANSWER_MODES.get(self.config.get("answer_mode"), self._mode)
        if selected != self._mode:
            self._mode, self.reason = selected, "settings"
        return self._mode

    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("mode") not in ("gpt", "legacy"):
                raise ValueError("invalid answer mode")
            # A saved UI choice wins over the previous version's runtime cache.
            selected = ANSWER_MODES.get(self.config.get("answer_mode"), data["mode"])
            self._mode = selected
            self.reason = (
                data.get("reason", "saved") if selected == data["mode"] else "settings"
            )

    def switch(self, mode, reason):
        self.config["answer_mode"] = next(
            label for label, value in ANSWER_MODES.items() if value == mode
        )
        self._mode, self.reason = mode, reason
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "reason": reason,
                    "updated": datetime.now().astimezone().isoformat(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def owner(self, sender, platform=""):
        entries = self.config.get("owner_ids", [])
        return str(sender) in entries or f"{platform}:{sender}" in entries

    def blocked(self, event):
        if not self.enabled or not self.config.get("enabled", True):
            return False
        if event.get_message_type() != MessageType.FRIEND_MESSAGE:
            return False
        platform = event.get_platform_name()
        return platform != "webchat" and not self.owner(event.get_sender_id(), platform)

    def worker(self):
        identity = self.config.get("worker_provider_id", "").strip()
        if not identity:
            raise ValueError("请填写辅助模型 Provider ID")
        return identity

    def role(self, state):
        if not state.get("group"):
            return "private"
        plan = state.get("plan")
        return (
            "analysis"
            if plan and (plan.route == "research" or plan.technical)
            else "chat"
        )

    def target(self, state, *, legacy=False):
        plan = state.get("plan")
        if plan and plan.route == "delegate":
            return self.worker(), None, False
        use_gpt = self.mode == "gpt" and not legacy
        if use_gpt:
            key, effort_key, default = GPT_ROLES[self.role(state)]
            identity = self.config.get(key, "").strip()
            effort = self.config.get(effort_key, default)
            allowed = {"low", "medium", "high", "xhigh", "max"}
            if self.role(state) == "chat":
                allowed.add("none")
            if effort not in allowed:
                raise ValueError(f"无效思考档位：{effort_key}")
            if not identity:
                raise ValueError(f"请填写 {key}")
            return identity, effort, True
        if state.get("media", {}).get("image_count"):
            identity = self.config.get("vision_provider_id", "").strip()
        elif plan and plan.route == "research":
            identity = (
                self.config.get("research_provider_id", "").strip() or self.worker()
            )
        else:
            identity = self.config.get("legacy_chat_provider_id", "").strip()
        if not identity:
            raise ValueError("请填写备用聊天/看图 Provider ID")
        return identity, None, False

    def select(self, state, event=None):
        identity, effort, gpt = self.target(state)
        state.update(
            writer_provider=identity,
            reasoning_effort=effort,
            gpt_answer=gpt,
            answer_profile="gpt" if gpt else "legacy",
        )
        if event:
            event.set_extra("selected_provider", identity)
        return identity

    def get_provider(self, identity):
        provider = self.plugin.context.get_provider_by_id(identity)
        if provider is None:
            raise ValueError(f"找不到 Provider：{identity}")
        return provider

    def answer_provider(self, state, event=None):
        return AnswerProvider(self, state, event)

    def attach_gcp(self):
        """GCP's independent proactive generation does not enter the native runner."""
        if not self.active:
            return
        for name, module in list(sys.modules.items()):
            if "group_chat_plus" in name:
                self.groups.attach(module, name)
            if "group_chat_plus" in name and name.endswith(".utils.decision_ai"):
                self._attach_decision(module)
            if "group_chat_plus" not in name or not name.endswith(
                (
                    ".utils.proactive_chat_manager",
                    ".private_chat_utils.private_chat_proactive_chat_manager",
                )
            ):
                continue
            owner = getattr(module, "ProactiveChatManager", None)
            if owner is None or any(p[0] is owner for p in self.gcp_patches):
                continue
            descriptor = owner.__dict__.get("_process_proactive_chat_simplified")
            if not isinstance(descriptor, classmethod):
                continue
            original = descriptor.__func__

            def wrap(fn):
                @wraps(fn)
                async def process(
                    cls,
                    context,
                    config,
                    plugin_instance,
                    platform_id,
                    is_private,
                    chat_id,
                    chat_key,
                ):
                    if self.enabled and self.config.get("enabled", True):
                        if is_private and not self.owner(chat_id, platform_id):
                            return
                        state = {"group": not is_private, "media": {}}
                        context = ProactiveContext(context, self, state)
                    return await fn(
                        cls,
                        context,
                        config,
                        plugin_instance,
                        platform_id,
                        is_private,
                        chat_id,
                        chat_key,
                    )

                return classmethod(process)

            replacement = wrap(original)
            owner._process_proactive_chat_simplified = replacement
            self.gcp_patches.append(
                (owner, "_process_proactive_chat_simplified", descriptor, replacement)
            )

    def _attach_decision(self, module):
        owner = getattr(module, "DecisionAI", None)
        if owner is None or any(p[0] is owner for p in self.gcp_patches):
            return
        original = owner.__dict__.get("should_reply")
        if not isinstance(original, staticmethod):
            return

        @wraps(original.__func__)
        async def judge(context, event, *args, **kwargs):
            if self.blocked(event):
                return False
            if self.enabled and self.plugin.enabled(event):
                context = ProactiveContext(context, self, {})
            return await original.__func__(context, event, *args, **kwargs)

        replacement = staticmethod(judge)
        owner.should_reply = replacement
        self.gcp_patches.append((owner, "should_reply", original, replacement))

    def close(self):
        self.active = False
        self.groups.close()
        for owner, name, original, replacement in reversed(self.gcp_patches):
            if owner.__dict__.get(name) is replacement:
                setattr(owner, name, original)
        self.gcp_patches.clear()


class AnswerProvider:
    """Per-turn facade: no shared provider configuration or context is mutated."""

    def __init__(self, routing, state, event=None):
        self.routing, self.state, self.event = routing, state, event
        self.provider = routing.get_provider(routing.select(state, event))

    def __getattr__(self, name):
        return getattr(self.provider, name)

    async def text_chat(self, **kwargs):
        for attempt in range(2):
            identity = self.routing.select(self.state, self.event)
            base = self.routing.get_provider(identity)
            self.provider = base
            call = dict(kwargs)
            call.pop("model", None)  # Never carry the previous provider's model slug.
            if self.state["gpt_answer"]:
                base = copy.copy(base)
                base.provider_config = copy.deepcopy(base.provider_config)
                extra = base.provider_config.setdefault("custom_extra_body", {})
                for key in (
                    "temperature",
                    "top_p",
                    "top_logprobs",
                    "logprobs",
                    "reasoning_effort",
                ):
                    extra.pop(key, None)
                    call.pop(key, None)
                extra["reasoning"] = {"effort": self.state["reasoning_effort"]}
                call["request_max_retries"] = 1
                if hasattr(base, "client") and hasattr(base.client, "with_options"):
                    base.client = base.client.with_options(max_retries=0)
                # AstrBot's key-rotation loop otherwise treats billing 429 as throttling.
                if hasattr(base, "_handle_api_error"):
                    previous = base._handle_api_error

                    async def handle(
                        _self, error, *args, _previous=previous, **options
                    ):
                        if quota_code(error):
                            raise error
                        return await _previous(error, *args, **options)

                    base._handle_api_error = MethodType(handle, base)
            try:
                response = await base.text_chat(**call)
                logger.info(
                    "[证据问答] answer=%s effort=%s profile=%s images=%d",
                    identity,
                    self.state["reasoning_effort"],
                    self.state["answer_profile"],
                    len(call.get("image_urls") or []),
                )
                return response
            except Exception as error:
                code = quota_code(error)
                if attempt or not self.state["gpt_answer"] or not code:
                    raise
                self.routing.switch("legacy", code)
                self.state["fallback_reason"] = code
                logger.warning("[证据问答] GPT额度不可用，切回原分工：%s", code)


class ProactiveContext:
    def __init__(self, context, routing, state):
        self.context, self.routing, self.state = context, routing, state

    def __getattr__(self, name):
        return getattr(self.context, name)

    def get_using_provider(self, *args, **kwargs):
        return ProactiveProvider(self.routing, self.state)

    def get_provider_by_id(self, identity):
        provider = self.context.get_provider_by_id(identity)
        if provider is not None and hasattr(provider, "text_chat"):
            return ProactiveProvider(self.routing, self.state)
        return provider


class ProactiveProvider:
    def __init__(self, routing, state):
        self.routing, self.state = routing, state
        self.worker = routing.get_provider(routing.worker())

    def __getattr__(self, name):
        return getattr(self.worker, name)

    async def text_chat(self, **kwargs):
        # GCP V1.2.3.hotfix.2 passes this argument only on its final generation call.
        if "func_tool_manager" in kwargs:
            kwargs.pop("func_tool_manager")
            from .prompts import CHAT

            kwargs["system_prompt"] = (kwargs.get("system_prompt") or "") + "\n" + CHAT
            return await self.routing.answer_provider(self.state).text_chat(**kwargs)
        if kwargs.get("image_urls"):
            kwargs["prompt"] = (kwargs.get("prompt") or "") + (
                f"\n当前消息附有{len(kwargs['image_urls'])}张图片；本次接话判断不读取画面。"
                "请依据文字、对话走向及邀请关系判断，不猜测图片内容；回应时会由回答模型读取原图。"
            )
            kwargs["image_urls"] = []
        return await self.worker.text_chat(**kwargs)
