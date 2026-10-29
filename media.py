"""Read message media without changing wake-up or group participation decisions."""

import os

from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.message.components import Face, Image, Plain, Reply
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.utils.image_input import prepare_request_images
from astrbot.core.utils.media_utils import (
    file_uri_to_path,
    normalize_model_image_max_size,
)
from astrbot.core.utils.quoted_message.chain_parser import ReplyChainParser
from astrbot.core.utils.quoted_message.settings import SETTINGS as QUOTE_SETTINGS
from astrbot.core.utils.quoted_message_parser import (
    extract_quoted_message_images,
    extract_quoted_message_text,
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


def request_images(request):
    refs = list(request.image_urls or [])
    for part in request.extra_user_content_parts:
        if isinstance(part, ImageURLPart):
            refs.append(part.image_url.url)
        elif isinstance(part, dict) and part.get("type") == "image_url":
            refs.append(part["image_url"]["url"])
    return refs


def collect(event):
    # WakingCheck precedes native media localization. Re-read the live chain;
    # only reuse prepared pixels while the effective references remain identical.
    cached = event.get_extra(MEDIA)
    obj = getattr(event, "message_obj", None)
    chain = getattr(obj, "message", [])
    raw = getattr(obj, "raw_message", None)
    raw_parts = raw.get("message", []) if isinstance(raw, dict) else []
    # OneBot's mface segment is skipped by AstrBot 4.29.3's adapter. Restore its
    # actual image URL, without trying to infer the picture from an emoji ID.
    existing = {c.url or c.file for c in chain if isinstance(c, Image)}
    stickers = cached.get("stickers", 0) if cached else 0
    parts_to_restore = (
        raw_parts if cached is None and isinstance(raw_parts, list) else []
    )
    for part in parts_to_restore:
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
        selected = request_images(request)
        # GCP has already applied its scope, count limit or image-to-text policy.
        refs = (
            selected if event.get_extra("_group_chat_plus_request") else refs + selected
        )
    refs = unique_refs(refs)
    result = {"refs": refs, "quotes": quotes, "faces": faces, "stickers": stickers}
    if cached is None:
        cached = result
        event.set_extra(MEDIA, cached)
    else:
        if cached["refs"] != refs:
            cached.pop("prepared", None)
            cached.pop("animation", None)
        cached.update(result)
    return cached


async def resolve_quotes(event, config):
    """Use the native quote resolver before selecting a vision/text provider."""
    if isinstance(event.get_extra("provider_request"), ProviderRequest):
        return  # The request owner has already selected the effective attachments.
    settings = config.get("provider_settings", {})
    parser = QUOTE_SETTINGS.with_overrides(settings.get("quoted_message_parser"))
    chain_parser = ReplyChainParser(parser)
    remaining = max(0, int(settings.get("max_quoted_fallback_images", 20)))
    obj = getattr(event, "message_obj", None)
    for part in getattr(obj, "message", []):
        if not isinstance(part, Reply) or any(
            isinstance(c, Image) for c in part.chain or []
        ):
            continue
        if remaining == 0:
            break
        refs = unique_refs(
            await extract_quoted_message_images(event, part, settings=parser)
        )[:remaining]
        if not refs:
            continue
        # Adding an Image makes native quote parsing prefer chain over message_str.
        # Carry the original/fetched quote text forward before inserting images.
        chain = list(part.chain or [])
        if not chain or chain_parser.is_forward_placeholder_only_text(
            chain_parser.extract_text_from_reply_component(part)
        ):
            text = await extract_quoted_message_text(event, part, settings=parser)
            if text:
                chain.insert(0, Plain(text=text))
        part.chain = chain + [Image(file=ref) for ref in refs]
        remaining -= len(refs)


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
    observation = state.get("visual_context")
    if state.get("prepared_images"):
        originals = state.get("original_images", [])
        req.image_urls = unique_refs(
            [
                *(ref for ref in request_images(req) if ref not in originals),
                *state["prepared_images"],
            ]
        )
        req.extra_user_content_parts = [
            p
            for p in req.extra_user_content_parts
            if not isinstance(p, ImageURLPart)
            and not (isinstance(p, dict) and p.get("type") == "image_url")
        ]
    if observation:
        text = "<visual_observation>\n" + observation + "\n</visual_observation>"
        if not any(
            isinstance(p, TextPart) and p.text == text
            for p in req.extra_user_content_parts
        ):
            req.extra_user_content_parts.append(TextPart(text=text))
    if (
        state.get("describe_images_only")
        and getattr(state.get("plan"), "route", None) != "delegate"
    ):
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
