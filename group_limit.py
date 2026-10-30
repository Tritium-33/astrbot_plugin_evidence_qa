"""One shared sliding-window budget for all replies in selected QQ groups."""

from collections import deque
from dataclasses import dataclass, field
from time import monotonic

from astrbot.api import logger
from astrbot.api.platform import MessageType

TICKET = "evidence_qa.reply_ticket"
LIMITED = "evidence_qa.reply_limited"
MANAGEMENT = "evidence_qa.group_management"


@dataclass
class Budget:
    sent: deque = field(default_factory=deque)
    pending: int = 0


@dataclass
class Ticket:
    budget: Budget
    pending: bool = True
    sent: bool = False


class GroupLimit:
    def __init__(self, plugin):
        self.plugin = plugin
        self.budgets = {}

    def enabled(self, event):
        cfg = self.plugin.config
        return (
            event.get_message_type() == MessageType.GROUP_MESSAGE
            and event.get_platform_name() == "aiocqhttp"
            and str(event.get_group_id()) in cfg.get("reply_limit_groups", [])
            and str(event.get_group_id()) not in cfg.get("no_cooldown_groups", [])
            and self.plugin.enabled(event)
        )

    def management(self, event):
        control = self.plugin.group_control
        return control.owner(event) and (
            event.get_extra(MANAGEMENT, False) or control.is_command(event)
        )

    def enter(self, event):
        # Let the owner's existing planner identify natural-language controls.
        # Their ordinary chat must still acquire a slot before answer generation.
        return self.plugin.group_control.owner(event) or self.acquire(event)

    def acquire(self, event):
        if not self.enabled(event) or self.management(event):
            return True
        if event.get_extra(LIMITED, False):
            return False
        ticket = event.get_extra(TICKET)
        if ticket and (ticket.pending or ticket.sent):
            return True
        budget = self.budgets.setdefault(self.plugin.group_control.key(event), Budget())
        now = monotonic()
        window = max(1.0, float(self.plugin.config.get("reply_limit_seconds", 300)))
        maximum = max(1, int(self.plugin.config.get("reply_limit_count", 10)))
        while budget.sent and now - budget.sent[0] >= window:
            budget.sent.popleft()
        # No await between checking and reserving: concurrent turns share slots.
        if len(budget.sent) + budget.pending >= maximum:
            event.set_extra(LIMITED, True)
            event.stop_event()
            logger.info(
                "[证据问答] 群%s 达到统一回复上限：%s秒/%d次，丢弃本轮。",
                event.get_group_id(),
                window,
                maximum,
            )
            return False
        budget.pending += 1
        event.set_extra(TICKET, Ticket(budget))
        return True

    def sent(self, event):
        ticket = event.get_extra(TICKET)
        if ticket and ticket.pending:
            ticket.budget.sent.append(monotonic())
            ticket.sent = True
            self.release(event)

    @staticmethod
    def release(event):
        ticket = event.get_extra(TICKET)
        if ticket and ticket.pending:
            ticket.budget.pending -= 1
            ticket.pending = False
