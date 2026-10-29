"""Per-group GCP cooldown exemptions; probability and decision AI stay native."""

import inspect
from functools import wraps

from astrbot.api.platform import MessageType


def no_fatigue():
    return {
        "enabled": False,
        "consecutive_replies": 0,
        "fatigue_level": "none",
        "probability_decrease": 0.0,
    }


POLICIES = {
    "reply_density_manager": (
        "ReplyDensityManager",
        {
            "get_probability_factor": lambda: 1.0,
            "should_block": lambda: False,
            "get_ai_hint_text": lambda: "",
        },
    ),
    "attention_manager": (
        "AttentionManager",
        {
            "get_conversation_fatigue_info": no_fatigue,
            "_is_fatigue_attention_blocked": lambda: False,
            "_add_fatigue_attention_block": lambda: False,
        },
    ),
    "cooldown_manager": (
        "CooldownManager",
        {
            "get_pending_info": lambda: None,
            "get_cooldown_info": lambda: None,
            "is_in_pending_cooldown": lambda: False,
            "is_in_cooldown": lambda: False,
            "add_pending_cooldown": lambda: False,
            "add_to_cooldown": lambda: False,
        },
    ),
}


class GroupPolicy:
    def __init__(self, plugin):
        self.plugin = plugin
        self.patches = []
        self.managers = {}

    def exempt(self, group_id):
        cfg = self.plugin.config
        return (
            cfg.get("enabled", True)
            and cfg.get("enable_group", True)
            and str(group_id) in cfg.get("no_cooldown_groups", [])
        )

    def exempt_call(self, values):
        if "chat_key" in values:
            _, separator, group_id = str(values["chat_key"]).rpartition("_group_")
            return bool(separator) and self.exempt(group_id)
        return not values["is_private"] and self.exempt(values["chat_id"])

    def clear_cooldown(self, event):
        if (
            not self.plugin.config.get("no_cooldown_groups")
            or event.get_message_type() != MessageType.GROUP_MESSAGE
            or not self.plugin.enabled(event)
            or not self.exempt(event.get_group_id())
        ):
            return
        key = f"{event.get_platform_name()}_group_{event.get_group_id()}"
        # GCP also reads these maps directly when increasing attention. Remove
        # only this group's old cooldown entries; keep its history and attention.
        for manager, names in (
            ("CooldownManager", ("_cooldown_map", "_pending_cooldown_map")),
            ("AttentionManager", ("_fatigue_attention_block",)),
        ):
            owner = self.managers.get(manager)
            if owner is not None:
                for name in names:
                    getattr(owner, name).pop(key, None)

    def attach(self, module, module_name):
        spec = POLICIES.get(module_name.rsplit(".", 1)[-1])
        if spec is None:
            return
        class_name, overrides = spec
        owner = getattr(module, class_name, None)
        if owner is None:
            return
        self.managers[class_name] = owner
        for name, value in overrides.items():
            self._patch(owner, name, value)

    def _patch(self, owner, name, value):
        if any(p[0] is owner and p[1] == name for p in self.patches):
            return
        original = owner.__dict__.get(name)
        if not isinstance(original, (classmethod, staticmethod)):
            return
        fn = original.__func__
        signature = inspect.signature(fn)

        def exempt(args, kwargs):
            return self.exempt_call(signature.bind(*args, **kwargs).arguments)

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def wrapper(*args, **kwargs):
                if exempt(args, kwargs):
                    return value()
                return await fn(*args, **kwargs)
        else:

            @wraps(fn)
            def wrapper(*args, **kwargs):
                if exempt(args, kwargs):
                    return value()
                return fn(*args, **kwargs)

        replacement = type(original)(wrapper)
        setattr(owner, name, replacement)
        self.patches.append((owner, name, original, replacement))

    def close(self):
        for owner, name, original, replacement in reversed(self.patches):
            if owner.__dict__.get(name) is replacement:
                setattr(owner, name, original)
        self.patches.clear()
        self.managers.clear()
