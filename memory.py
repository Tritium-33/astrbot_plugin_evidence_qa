"""Reversible adapter for LivingMemory 2.8.1/2.8.2 private-chat entry points."""

from functools import wraps

SKIP_MEMORY = "evidence_qa.exclude_memory"
MEMORY_STEPS = "evidence_qa.memory_steps"


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
                or str(meta.version).lstrip("v") not in ("2.8.1", "2.8.2")
            ):
                continue
            owner = getattr(meta.star_cls, "event_handler", None)
            if owner is None:
                continue
            for name in ("handle_memory_recall", "handle_memory_reflection"):
                if any(o is owner and n == name for o, n, _, _ in self.patches):
                    continue
                original = getattr(owner, name)
                wrapper = self._wrap(original, name.removeprefix("handle_memory_"))
                setattr(owner, name, wrapper)
                self.patches.append((owner, name, original, wrapper))
            supported = True
        return supported

    def _wrap(self, original, name):
        @wraps(original)
        async def wrapped(event, *args, **kwargs):
            if self.active:
                skip = bool(event.get_extra(SKIP_MEMORY, False))
                steps = event.get_extra(MEMORY_STEPS, {})
                steps[name] = steps.get(name, True) and skip
                event.set_extra(MEMORY_STEPS, steps)
                if skip:
                    return
            return await original(event, *args, **kwargs)

        return wrapped

    def close(self):
        self.active = False
        for owner, name, original, wrapper in self.patches:
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
        self.patches.clear()
