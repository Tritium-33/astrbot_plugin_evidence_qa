"""Owner-only group commands and a persistent, per-group reply pause."""

import json

from astrbot.api.platform import MessageType
from astrbot.core.message.components import At
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter

EPOCH = "evidence_qa.group_epoch"
DENIED = "evidence_qa.group_denied"


class GroupControl:
    def __init__(self, plugin, path):
        self.plugin, self.path = plugin, path
        self.groups = {}

    def load(self):
        if self.path.exists():
            self.groups = json.loads(self.path.read_text(encoding="utf-8"))

    def active(self, event):
        return (
            self.plugin.config.get("group_owner_control", False)
            and event.get_message_type() == MessageType.GROUP_MESSAGE
            and event.get_platform_name() == "aiocqhttp"
            and self.plugin.enabled(event)
        )

    def owner(self, event):
        return self.plugin.routing.owner(
            event.get_sender_id(), event.get_platform_name()
        )

    def restricted(self, event):
        return self.active(event) and not self.owner(event)

    @staticmethod
    def key(event):
        return json.dumps(
            [event.get_platform_id(), event.get_self_id(), event.get_group_id()]
        )

    def state(self, event):
        return self.groups.get(self.key(event), {"paused": False, "epoch": 0})

    def set_paused(self, event, paused):
        state = self.state(event)
        if state["paused"] == paused:
            return
        updated = {
            **self.groups,
            self.key(event): {
                "paused": paused,
                "epoch": state["epoch"] + int(paused),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(updated, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)
        self.groups = updated

    def blocked(self, event):
        if not self.restricted(event):
            return False
        state = self.state(event)
        return (
            event.get_extra(DENIED, False)
            or state["paused"]
            or event.get_extra(EPOCH, state["epoch"]) != state["epoch"]
        )

    @staticmethod
    def mentioned(event):
        return any(
            isinstance(part, At) and str(part.qq) == str(event.get_self_id())
            for part in event.get_messages()
        )

    def planner_context(self, event):
        if not self.active(event):
            return {"allowed": False}
        return {
            "allowed": self.owner(event),
            "paused": self.state(event)["paused"],
            "mentioned": self.mentioned(event),
        }

    def resolve(self, event, action):
        if not self.active(event) or not self.owner(event):
            return None
        # A pause request containing @ must win over automatic reactivation.
        if action == "pause":
            self.set_paused(event, True)
            return "好，先只回复你。下次你 @我就恢复本群接话。"
        if action == "resume" or (action == "none" and self.mentioned(event)):
            self.set_paused(event, False)
            if action == "resume":
                return "已恢复本群接话。"
        return None

    def admit(self, event):
        """Called after native wake/access checks, before any plugin callbacks."""
        if not self.active(event):
            return None
        text = event.get_message_str().strip().lstrip("/").strip().rstrip("。！!")
        if self.owner(event):
            if text in ("群聊暂停", "群聊恢复"):
                return self.resolve(event, "pause" if text == "群聊暂停" else "resume")
            if not text and self.mentioned(event):
                self.set_paused(event, False)
            # Nonempty owner messages use the existing routing call to distinguish
            # a new pause instruction from a normal @ that resumes participation.
        else:
            command = any(
                isinstance(rule, (CommandFilter, CommandGroupFilter))
                for handler in event.get_extra("activated_handlers", [])
                for rule in handler.event_filters
            )
            if command or text in ("群聊暂停", "群聊恢复") or self.blocked(event):
                event.set_extra(DENIED, True)
                event.stop_event()
                return None
        event.set_extra(EPOCH, self.state(event)["epoch"])
        return None
