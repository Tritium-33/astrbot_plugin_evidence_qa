"""Reversible adapter for LivingMemory 2.8.1 private-chat event entry points."""

from functools import wraps

SKIP_MEMORY = "evidence_qa.exclude_memory"


class MemoryBridge:
    def __init__(self):
        self.patches = []
        self.active = True

    def attach(self, context):
        supported = False
        for meta in context.get_all_stars():
            if (
                meta.name != "astrbot_plugin_livingmemory"
                or not meta.activated
                or str(meta.version).lstrip("v") != "2.8.1"
            ):
                continue
            owner = getattr(meta.star_cls, "event_handler", None)
            if owner is None:
                continue
            for name in ("handle_memory_recall", "handle_memory_reflection"):
                if any(o is owner and n == name for o, n, _, _ in self.patches):
                    continue
                original = getattr(owner, name)
                wrapper = self._wrap(original)
                setattr(owner, name, wrapper)
                self.patches.append((owner, name, original, wrapper))
            supported = True
        return supported

    def _wrap(self, original):
        @wraps(original)
        async def wrapped(event, *args, **kwargs):
            if self.active and event.get_extra(SKIP_MEMORY, False):
                return
            return await original(event, *args, **kwargs)

        return wrapped

    def close(self):
        self.active = False
        for owner, name, original, wrapper in self.patches:
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
        self.patches.clear()
