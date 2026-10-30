"""Authenticated AstrBot View APIs for the group speaking switches."""

import asyncio
import json
import re

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request
from astrbot.core.star.session_llm_manager import SessionServiceManager

PLUGIN = "astrbot_plugin_evidence_qa"


class GroupPanel:
    def __init__(self, plugin):
        self.plugin = plugin
        self.gate = plugin.group_gate
        self.catalogue = {}
        self.refreshed = False
        self.warnings = []
        self.refresh_lock = asyncio.Lock()
        self.routes = []

    def register(self):
        for endpoint, handler, method in (
            ("groups", self.list_groups, "GET"),
            ("groups/refresh", self.refresh_groups, "POST"),
            ("groups/switch", self.switch_groups, "POST"),
        ):
            route = f"/{PLUGIN}/{endpoint}"
            self.plugin.context.register_web_api(route, handler, [method], "群聊控制")
            self.routes.append(handler)

    def close(self):
        if not self.routes:
            return
        apis = self.plugin.context.registered_web_apis
        apis[:] = [row for row in apis if row[1] not in self.routes]
        self.routes.clear()

    async def refresh(self):
        async with self.refresh_lock:
            self.warnings = []

            async def read(platform):
                identity = platform.meta().id
                try:
                    async with asyncio.timeout(8):
                        groups = await platform.get_client().get_group_list()
                    rows = {
                        self.gate.key(identity, str(group["group_id"])): {
                            "platform_id": identity,
                            "group_id": str(group["group_id"]),
                            "name": str(group.get("group_name", "")),
                        }
                        for group in groups
                    }
                except Exception as error:  # noqa: BLE001 - OneBot boundary
                    self.warnings.append(f"{identity} 群列表暂不可用，仍可按群号控制。")
                    logger.warning(
                        "[群聊控制] %s 群列表读取失败：%s",
                        identity,
                        type(error).__name__,
                    )
                    return
                self.catalogue = {
                    key: row
                    for key, row in self.catalogue.items()
                    if row["platform_id"] != identity
                }
                self.catalogue.update(rows)

            await asyncio.gather(*(read(p) for p in self.gate.platforms()))
            self.refreshed = True

    async def snapshot(self):
        platforms = [{"id": p.meta().id} for p in self.gate.platforms()]
        available = {p["id"] for p in platforms}
        rows = {}
        for key, state in self.gate.states.items():
            platform, group = json.loads(key)
            rows[key] = {
                "platform_id": platform,
                "group_id": group,
                "name": state.get("name", ""),
            }
        # Configured groups are visible even before the bot has joined them.
        for platform in available:
            for field in ("reply_limit_groups", "no_cooldown_groups"):
                for group in self.plugin.config.get(field, []):
                    rows.setdefault(
                        self.gate.key(platform, group),
                        {"platform_id": platform, "group_id": str(group), "name": ""},
                    )
        rows.update(self.catalogue)
        paused = {
            self.gate.key(parts[0], parts[2])
            for key, state in self.plugin.group_control.groups.items()
            if state["paused"] and len(parts := json.loads(key)) == 3
        }
        output = []
        for key, row in sorted(rows.items()):
            umo = f"{row['platform_id']}:GroupMessage:{row['group_id']}"
            output.append(
                {
                    **row,
                    "enabled": self.gate.state(key)["enabled"],
                    "available": row["platform_id"] in available,
                    "native_enabled": await SessionServiceManager.is_session_enabled(
                        umo
                    ),
                    "owner_paused": key in paused
                    and self.plugin.config.get("group_owner_control", False),
                }
            )
        return {"groups": output, "platforms": platforms, "warnings": self.warnings}

    async def list_groups(self):
        if not self.refreshed:
            await self.refresh()
        return json_response(
            await self.snapshot(), headers={"Cache-Control": "no-store"}
        )

    async def refresh_groups(self):
        await self.refresh()
        return json_response(await self.snapshot())

    async def switch_groups(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict) or type(payload.get("enabled")) is not bool:
            return error_response("请选择开启或关闭")
        groups = payload.get("groups")
        if not isinstance(groups, list) or not 1 <= len(groups) <= 500:
            return error_response("请选择 1–500 个群")
        available = {p.meta().id for p in self.gate.platforms()}
        validated = []
        for row in groups:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("platform_id"), str)
                or row["platform_id"] not in available
                or not isinstance(row.get("group_id"), str)
                or not re.fullmatch(r"[1-9][0-9]{4,15}", row["group_id"])
            ):
                return error_response("请选择有效的 QQ 平台，并填写正确群号")
            key = self.gate.key(row["platform_id"], row["group_id"])
            validated.append(
                {
                    "platform_id": row["platform_id"],
                    "group_id": row["group_id"],
                    "name": self.catalogue.get(key, {}).get(
                        "name", self.gate.state(key).get("name", "")
                    ),
                }
            )
        try:
            self.gate.set_enabled(validated, payload["enabled"])
        except OSError:
            logger.exception("[群聊控制] 保存群发言开关失败")
            return error_response("保存失败，原开关状态未改变", status_code=500)
        return json_response(await self.snapshot())
