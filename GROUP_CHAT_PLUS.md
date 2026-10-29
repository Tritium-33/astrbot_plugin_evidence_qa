# Group Chat Plus 与看图配置

[返回使用说明](README.md)

适用：本插件 v0.4.0、AstrBot 4.29.3；Group Chat Plus（下称 GCP）按 **V1.2.3.hotfix.2** 的源码核对。设置名称可能随版本调整，下面同时列出配置键，方便在后台搜索。

## 1. 分清两插件的职责

```text
QQ 消息 → 原生唤醒 / GCP 决定是否接话
                     ↓ 决定回应后
          证据问答：图片理解与任务路由
                     ↓
      自然交流       本地技术资料       联网查询
      直接回复       回答＋一次核验      直接回复
```

GCP 管理群聊参与、上下文和频率。本插件不会监听每张图片就调用模型，也不新增随机插话。GCP 判断“不接话”的消息不会进入本插件的规划和回答调用。

直接 @ 机器人表示明确找它；一般聊天可由 GCP 的概率筛选、读空气判断及密度限制决定。机器人名称可作为关键词，但它是大小写敏感的子串匹配，不要填“我”“你”等常见词。`keyword_smart_mode=false` 时关键词可直接触发，开启后仍需读空气判断；希望稳妥提问时直接 @。

## 2. 先设置本插件与模型

| 位置 / 设置 | 建议 |
|---|---|
| AstrBot 默认聊天模型 | 你已验证可用、支持图片输入的 Gemini Provider |
| 本插件 `worker_provider_id` | DeepSeek：文本路由、本地技术核验和目录概括 |
| 本插件 `research_provider_id` | DeepSeek：本地与联网资料回答 |
| 本插件 `vision_provider_id` | 上述 Gemini 的 **Provider ID**，不是模型名称或 Key |
| 本插件 `enable_group` | 开启 |
| 原生搜索、知识库、Rerank | 沿用已验证的设置 |

视觉请求使用 Gemini 做一次“画面理解＋路由”；日常看图仍由 Gemini 自然回应。资料问题会把画面说明传给 DeepSeek，继续检索。模型切换沿用本轮请求的历史和人设。GCP 场景的群历史由 GCP 提供，不会为每个模型另建群会话。

如果 Gemini 的反代只支持文字，即使后台勾选了 `image` 也不能看图；先用私聊图片测试其真实能力。QQ 商城表情有图片 URL 时可读取；只有内置表情编号时没有可供视觉模型识别的画面。GIF 使用原生抽帧，不等于逐帧视频分析。

## 3. 群聊基础设置

在 AstrBot 对应 QQ 配置中启用群聊上下文相关设置，GCP 所需的历史记录项按其后台说明开启。初始保留约 **30–60 条**上下文即可。原生随机主动回复与其他自动接话插件先关闭，让 GCP 负责这件事。

| GCP 设置 | 初始建议 | 作用 |
|---|---:|---|
| `enable_group_chat` | `true` | 开启群聊增强 |
| `enabled_groups` | 只填一个测试群号 | 空列表意味着所有群，不是禁用 |
| `trigger_keywords` | `Tia` 等独特称呼 | 不需斜杠；直接需求也可 @ |
| `enable_ignore_at_others` | `true` | 降低抢答别人对话的概率 |
| `ignore_at_others_mode` | `allow_with_bot` | 同时 @ 机器人时允许处理 |
| `initial_probability` | `0.02` | 普通消息进入进一步判断的初始概率，不是最终回复率 |
| `after_reply_probability` | `0.20` | 回复后短时间较容易继续接话 |
| `probability_duration` | `60` 秒 | 上述持续窗口 |
| `enable_reply_density_limit` | `true` | 限制自主接话密度 |
| `reply_density_window_seconds` | `300` | 五分钟统计窗口 |
| `reply_density_max_replies` | `3` | 自主参与先从每窗口三次开始 |
| `enable_proactive_chat` | `false` | 暂不开冷场后主动另起话题 |
| `enable_typing_simulator` | `false` | 不额外模拟打字等待 |
| `enable_humanize_mode` | `false` | 先验证基础概率与读空气流程 |

这些是起步建议，不是插件默认值。密度限制主要作用于自主接话，不应当作“包括所有 @ 在内的统一硬限”。群里很活跃时先降低 `initial_probability`，总是连续插话时优先降低 `after_reply_probability` 或缩短持续窗口。

GCP 的命令过滤通常包含 `/`。**测试自然接话与看图时不要在普通话前加 `/`**；使用“@Tia 这张图好好笑”或直接发图。保留 `/资料目录更新` 等真正的管理命令，避免被聊天吞掉。

## 4. 图片、梗图与读空气

想让机器人对未 @ 的表情包偶尔自然接话，不能只启用 `mention_only`。

| GCP 设置 | 建议 |
|---|---|
| `enable_image_processing` | `true` |
| `image_to_text_scope` | `all` |
| `max_images_per_message` | `3`，需要多图比较再增加 |
| `enable_emoji_filter` | `false`，否则可能把你希望它评论的表情包筛掉 |

`all` 表示适用于普通消息，不代表每张图都必定获得回复；仍受前面的触发与概率控制。

**优先采用直接多模态路径：** GCP 的 `decision_ai_provider_id` 填支持图片的 Gemini，`image_to_text_provider_id` 留空。这样读空气能看到原图，无需先单独做图转文。`enable_decision_ai_reasoning=false`，避免为是否接话开启长推理。正常回应后，本插件的视觉规划与回答各自是一次调用；普通群聊还会有 GCP 的判断开销。

**若希望 DeepSeek 负责读空气：** `decision_ai_provider_id` 填 DeepSeek，`image_to_text_provider_id` 填 Gemini，并开启 `enable_image_description_cache`。这会先把图片转成文字再交给文本模型，多一次首次图片描述调用；相同 URL 的缓存命中可省去它。缓存按 URL，不保证重新上传的同一张梗图也命中。图转文提示可用：

> 简洁说明可见主体、表情动作和图中文字。若是梗图，说明画面与文字的反差及可能传达的情绪；把猜测与可见内容分开。不要替用户发言。

不要给 DeepSeek 直接传图后指望“已开启图片处理”替它识图。本插件位于 GCP 决定接话之后，无法补救更早的读空气模型看不到图的问题。

## 5. 上下文、记忆与等待窗口

- 私聊由本插件与原生流程处理即可，不必为了群聊而同时开启 GCP 的私聊接管。
- GCP 若使用 LivingMemory，`memory_plugin_mode=livingmemory`，`memory_insertion_timing=post_decision` 可避免在最终不回复的消息上先做记忆召回。先确认基础回复正常，再开启所需记忆注入。
- 本插件已有的 LivingMemory 隔离适配针对私聊。LivingMemory 群消息独立采集、GCP 自行注入等路径不在完整隔离保证内；不要把日志中某个入口已跳过等同于所有群记忆都没写入。需要严格隔离技术资料时，先关闭相应群聊自动记忆写入。
- 初始关闭 `enable_group_wait_window`，便于测量原始耗时。以后需要合并连续短消息时再开启，窗口可从 `2000–3000 ms` 试起。启用 GCP 等待窗口前关闭原生 `empty_mention_waiting`，避免两种等待机制干扰发送者归属。
- GCP 有自己的群历史。其重置命令不等于更换 AstrBot conversation ID；本插件来源快照按原生会话隔离，不能把 GCP 清空历史当作已经删除全部证据或长期记忆。

## 6. 按这个顺序联调

1. **私聊纯图**：发一张有明显文字反差的梗图，不加文字。应有简短回应，不应因文字为空而消失。
2. **图文与引用**：发图并问“这张图在吐槽什么”；再引用一张旧图提问，检查回答指向引用对象。
3. **群内明确需求**：@ 机器人发图；应只回复一次。再让另一个人引用你的消息提问，检查发言者归属。
4. **自然接话**：在测试群临时把初始概率设为 `1.0`，保留读空气判断，发梗图和普通闲聊。验证后恢复 `0.02`。即使概率为 1，读空气仍可决定不回复。
5. **三路切换**：同一群先聊图，再问本地 906B 参数，接着问近期 Gemini 新闻。检查日志分别进入 `chat`、`research/kb`、`research/web`；追问来源时显示对应那轮资料。
6. **观察实际节奏**：确认未重复回复、未对 @ 别人的话题频繁插嘴，再逐步调整概率。耗时测试时先保持打字模拟、等待窗口关闭。

开发验证包含 AstrBot 原生图片预处理和发送链、GCP 实际请求钩子回放；模型与搜索使用固定夹具。未在你的运行实例安装或发送 QQ 消息，因此真实图片识别和接话质量由你更新后完成这组验收。

## 官方资料

- [Group Chat Plus 仓库](https://github.com/Him666233/astrbot_plugin_group_chat_plus)
- [完整配置参考](https://github.com/Him666233/astrbot_plugin_group_chat_plus/blob/main/docs/CONFIG_REFERENCE.md)
- [实际配置字段](https://github.com/Him666233/astrbot_plugin_group_chat_plus/blob/main/_conf_schema.json)
- [Desktop 说明](https://github.com/Him666233/astrbot_plugin_group_chat_plus/blob/main/docs/DESKTOP_COMPATIBILITY.md)
