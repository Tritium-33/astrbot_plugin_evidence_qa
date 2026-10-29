"""Read message media without changing wake-up or group participation decisions."""

import os

from astrbot.core.message.components import Face, Image, Reply
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.utils.image_input import prepare_request_images
from astrbot.core.utils.media_utils import (
    file_uri_to_path,
    normalize_model_image_max_size,
)

MEDIA = "evidence_qa.media"


def unique_refs(refs):
    found = {}
    for ref in refs:
        if not isinstance(ref, str) or not ref:
            continue
        key = file_uri_to_path(ref) if ref.startswith("file://") else ref
        if os.path.isabs(key):
            key = os.path.normcase(os.path.realpath(key))
        found.setdefault(key, ref)
    return list(found.values())


def collect(event):
    cached = event.get_extra(MEDIA)
    if cached is not None:
        request = event.get_extra("provider_request")
        if isinstance(request, ProviderRequest):
            cached["refs"] = unique_refs(cached["refs"] + (request.image_urls or []))
        return cached
    obj = getattr(event, "message_obj", None)
    chain = getattr(obj, "message", [])
    raw = getattr(obj, "raw_message", None)
    raw_parts = raw.get("message", []) if isinstance(raw, dict) else []
    # OneBot's mface segment is skipped by AstrBot 4.29.3's adapter. Restore its
    # actual image URL, without trying to infer the picture from an emoji ID.
    existing = {c.url or c.file for c in chain if isinstance(c, Image)}
    stickers = 0
    for part in raw_parts if isinstance(raw_parts, list) else []:
        if not isinstance(part, dict) or part.get("type") != "mface":
            continue
        data = part.get("data", {})
        url = data.get("url") if isinstance(data, dict) else None
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            stickers += 1
            if url not in existing:
                chain.append(Image(file=url, url=url))
                existing.add(url)
    refs, quotes, faces = [], [], []
    for part in chain:
        if isinstance(part, Image):
            refs.append(part.url or part.file or part.path)
        elif isinstance(part, Reply):
            quoted_images = [c for c in part.chain or [] if isinstance(c, Image)]
            refs.extend(c.url or c.file or c.path for c in quoted_images)
            quotes.append(
                {
                    "sender_id": str(part.sender_id),
                    "text": part.message_str or "",
                    "images": len(quoted_images),
                }
            )
        elif isinstance(part, Face):
            faces.append(str(part.id))
    request = event.get_extra("provider_request")
    if isinstance(request, ProviderRequest):
        refs.extend(request.image_urls or [])
    refs = unique_refs(refs)
    result = {"refs": refs, "quotes": quotes, "faces": faces, "stickers": stickers}
    event.set_extra(MEDIA, result)
    return result


async def prepare(event, config):
    media = collect(event)
    if "prepared" not in media:
        req = ProviderRequest(image_urls=media["refs"])
        prepared = {}
        options = config.get("provider_settings", {}).get("image_compress_options", {})
        await prepare_request_images(
            req,
            event,
            max_size=normalize_model_image_max_size(options.get("max_size")),
            prepared=prepared,
        )
        media["prepared"] = req.image_urls
        media["animation"] = any(row["montage"] for row in prepared.values())
    return media["prepared"]


def describe(media):
    return {
        "image_count": len(media["refs"]),
        "sticker_count": media["stickers"],
        "face_ids": media["faces"],
        "quoted_messages": media["quotes"],
        "animation_frames": media.get("animation", False),
    }


def chat_context(event):
    """GCP's prompt carries speaker-labelled group history absent from native history."""
    obj = getattr(event, "message_obj", None)
    sender = getattr(obj, "sender", None)
    result = {
        "sender_id": str(getattr(sender, "user_id", "")),
        "sender_name": getattr(sender, "nickname", ""),
    }
    if event.get_extra("_group_chat_plus_request"):
        result["group_chat_plus"] = True
        result["current_message"] = event.get_extra(
            "_group_chat_plus_current_message", ""
        )
        result["recent_context"] = str(event.get_extra("_group_chat_plus_prompt", ""))[
            -6000:
        ]
    return result


def apply_to_request(state, req):
    """Carry the visual observation to text models; keep pixels for the vision writer."""
    from astrbot.core.agent.message import TextPart

    observation = state.get("visual_context")
    if observation:
        text = "<visual_observation>\n" + observation + "\n</visual_observation>"
        if not any(
            isinstance(p, TextPart) and p.text == text
            for p in req.extra_user_content_parts
        ):
            req.extra_user_content_parts.append(TextPart(text=text))
    if state.get("describe_images_only"):
        req.image_urls = []
        req.extra_user_content_parts = [
            p
            for p in req.extra_user_content_parts
            if getattr(p, "type", None) != "image_url"
            and not (isinstance(p, dict) and p.get("type") == "image_url")
        ]
    if not req.prompt and (observation or state.get("media")):
        req.prompt = state["question"]
    faces = state.get("media", {}).get("face_ids", [])
    if faces:
        text = (
            "收到QQ内置表情ID："
            + ", ".join(faces)
            + "。未提供表情像素，不能仅凭ID断定表情内容。"
        )
        if not any(
            isinstance(p, TextPart) and p.text == text
            for p in req.extra_user_content_parts
        ):
            req.extra_user_content_parts.append(TextPart(text=text))
