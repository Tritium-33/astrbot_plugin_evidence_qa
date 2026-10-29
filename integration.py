"""AstrBot 4.29.3 provider selection runs inside its native session lock."""

from functools import wraps


class SelectionBridge:
    def __init__(self, plugin):
        self.plugin = plugin
        self.owner = None
        self.active = False

    def install(self, owner):
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

        self.owner, self.original, self.wrapper = owner, original, select
        owner._select_provider = select
        self.active = True

    def close(self):
        self.active = False
        if self.owner and self.owner._select_provider is self.wrapper:
            self.owner._select_provider = self.original
