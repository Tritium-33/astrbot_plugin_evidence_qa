"""Describe effective session tools before the chat route hides their schemas."""

import json


def describe(plugin, event, request):
    tools = getattr(request.func_tool, "tools", [])
    available = [tool for tool in tools if getattr(tool, "active", True)]
    entries = [
        {"name": tool.name, "purpose": " ".join(tool.description.split())[:100]}
        for tool in available
    ]
    blocked = plugin.group_control.restricted(event)
    image_status = None
    if any(tool.name == "gemini_image_generation" for tool in available):
        image_status = {"supports_reference_images": True}
        context = getattr(plugin, "context", None)
        stars = (
            context.get_all_stars()
            if context and hasattr(context, "get_all_stars")
            else []
        )
        for star in stars:
            if star.name == "astrbot_plugin_group_control" and star.activated:
                adapter = getattr(star.star_cls, "image_adapter", None)
                if adapter:
                    image_status.update(adapter.capability(event))
    return (
        "\n\n[当前会话工具能力]\n"
        + json.dumps(
            {
                "tools": entries,
                "execution_restricted_for_sender": blocked,
                "image_generation": image_status,
            },
            ensure_ascii=False,
        )
        + "\n以上为本轮路由处理前实际启用的工具。普通聊天轮只回答问题，可能不附工具调用定义，"
        "这不表示系统没有这些功能。回答能力问题时依据这份清单；区分功能存在、当前群允许、"
        "当前用户允许和实际调用成功。不得因为本轮未执行而声称没有生图接口，也不得虚构已调用。"
    )
