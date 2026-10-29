# 模型分工与配置

v0.5.2 将面向用户的回答与辅助处理分开。只填写 AstrBot 中已存在的 **Provider ID**；插件不保存 API Key，也不更改全局提供商配置。配置为「直聊群」的群仍使用群聊 RP / 分析模型，省去 GCP 读空气调用，任务规划继续使用辅助模型。

## 填写插件配置

| 界面名称 | 填写内容 | 用途 |
|---|---|---|
| 模型分工 | 开启 | 启用本页规则 |
| 回答模式 | GPT优先 / Gemini/DeepSeek | 在插件界面直接选择并保存；辅助分工不变 |
| 私聊白名单 | 你的用户 ID，一行一个 | 其他人的私聊在路由前停止；空列表不开放 QQ 私聊 |
| 辅助模型 | DeepSeek Provider ID | 路由、指代补全、目录摘要、压缩、WebChat 标题、旧分工技术核验 |
| 私聊GPT | GPT-6 Astra Provider ID | 白名单私聊及 WebChat 的回答 |
| 私聊思考 | `low`，可修改 | 原样传给 `reasoning.effort` |
| 群聊GPT | GPT-6.1 Terra Provider ID | 群聊 RP、自然接话、梗图评论与主动发言 |
| 群聊思考 | `low`，可修改 | 原样传给 `reasoning.effort` |
| 分析GPT | GPT-6.1 Sol Provider ID | 群聊资料问答、技术图解、分析与计算 |
| 分析思考 | `medium`，可修改 | 原样传给 `reasoning.effort` |
| 备用聊天 | Gemini Provider ID | 原分工的日常回答 |
| 备用分析 | DeepSeek Provider ID | 原分工的资料回答；留空沿用辅助模型 |
| 备用看图 | 支持图片的 Gemini Provider ID | 原分工的图片回答，包括带图的资料问题 |

填写的是提供商实例 ID，不是 `gpt-6-astra` 等模型名称。聊天白名单不限制群成员在群里提问；同一个主人在群里发言也采用群聊分工。WebChat 是已登录管理面板中的测试入口，单独允许。需要限定平台时，白名单也支持 `平台名:用户ID`。

三个 GPT 提供商均选择 **OpenAI Responses** 接口，并启用 `image` 与 `tool_use` 能力。这里使用 OpenAI 官方 API 的 `reasoning.effort`，不是自定义思考等级：

- Astra、Sol：`low / medium / high / xhigh / max`。
- Terra：`none / low / medium / high / xhigh / max`。

界面是单选菜单，因为同一次请求只能有一个思考程度。插件在每次回答的请求副本中写入所选值，移除不兼容的采样参数，不修改提供商的共享配置。无需手写请求 JSON。工具调用遵循 [OpenAI GPT-6 指南](https://developers.openai.com/api/docs/guides/latest-model)，通过 Responses 发送。各模型的官方取值见 [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)、[Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra) 和 [Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)。

图片直接进入回答模型；辅助路由只读取文字、引用和附件元信息，不另调 GPT 做视觉规划。技术图分析的 Sol 也会收到原图。纯图且没有其他意图线索时按自然聊天处理；若希望分析结构或参数，可在图片旁说明要求。

## 还需在 AstrBot 中配置

1. **全局默认聊天模型设为 DeepSeek**。受管主对话由本插件选择 GPT；插件外的辅助与委托调用不应通过默认模型意外消耗 GPT。不要将 GPT 填进全局回退列表。
2. **管理员列表**添加你自己的用户 ID，方便使用下方管理命令。插件私聊白名单仍需单独填写。
3. **LivingMemory → LLM 模型 ID**填 DeepSeek。Embedding 与可选 Rerank 继续使用原有专用提供商。
4. **Group Chat Plus → 读空气 AI 提供商**填 DeepSeek。其他独立辅助 LLM 设置也使用 DeepSeek；不要填写 GPT。群聊与私聊的图转文 Provider 均留空，原图保留给回答模型，避免预先改写或重复识图；若启用独立图转文服务，需要其实际支持图片，不能靠勾选能力伪装支持。
5. **原生联网搜索**启用，选择 DeepSeek 并配置搜索密钥。搜索／读取服务独立于聊天模型，本插件调用它们取得材料后由 GPT 回答。
6. **原生知识库、Embedding、Rerank**沿用现有配置。不要因此重新嵌入语料。上传更新文献后按需执行 `/资料目录更新`。
7. **可选入库文本修复、其他插件委托、原生群旁听图片描述**若启用，分别检查其独立 Provider。辅助模型统一 DeepSeek；已清洗的 Markdown 不需要再启用入库改写。

本插件强制将受管对话的原生上下文压缩与 WebChat 标题交给辅助模型。LivingMemory 等第三方组件仍通过其自己的设置指定模型，本插件不擅自写入它们的配置。

GCP 普通回复进入原生路由；接话判断适配器向 DeepSeek 传文字与附件数量，不传原图。纯梗图的视觉趣味不能作为该阶段的判断依据，GPT 在决定回应后才看图。它的独立主动发言路径另有运行时适配：判断使用 DeepSeek，最终发言使用私聊／群聊的回答模型；私聊主动发言同样受白名单约束。兼容契约为 GCP **V1.2.3.hotfix.2**。建议更新后重启 AstrBot，让所有插件完成加载，再运行 `/模型检查` 查看 GCP 接管入口数量；未启用主动发言无需为了测试而开启。

## 切换与额度

**插件管理 → 证据问答与会话路由 → 配置 → 回答模式**：

- **GPT优先**：采用私聊 Astra、群聊 Terra／Sol 分工。
- **Gemini/DeepSeek**：切回备用聊天／分析／看图模型。

选择后点保存，AstrBot 4.29.3 会重载此插件；无需卸载、重新填写 API 或在 QQ 中发命令。模型分工总开关保持开启。新版以配置界面的选择为准，不让旧 `routing.json` 覆盖手动选择。

明确额度错误也会把“回答模式”保存为 Gemini/DeepSeek；重新打开配置页可见。恢复额度后在界面选回 GPT优先 并保存。已有管理员命令保留为可选快捷方式，并同步同一界面字段：

```text
/模型检查
/回答模式 状态
/回答模式 旧分工
/回答模式 gpt
```

`/模型检查` 只核对 ID、模型和档位，不发送付费测试请求。选择保存到插件配置，切换原因另记在 `routing.json`，重启继续沿用。没有定时探测、按自然月一号重置或自动充值。

GPT 返回结构化的额度耗尽／用量上限错误时，本轮改用备用模型，并让后续回答沿用原分工。普通 429 限流、网络错误、模型不存在不会被误记为额度耗尽。当前请求的历史、人设、材料与工具结果继续传递，不新建会话。辅助处理始终使用 DeepSeek。

这是**API 错误触发的额度回退**，不是账户余额查询或美元记账器。OpenAI 赠送额度耗尽而仍有付费余额时，API 会继续扣费，不会触发这个回退；其他应用共用账户的消费也不在插件掌握范围。仅使用赠送额度时，应结合自己的账户余额及付费设置，必要时手动切换。参见 [官方额度说明](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)。

## 核验与来源

**GPT 生成的回答全部直接输出，包括本地技术查询。**无后置 DeepSeek 审查、GPT 重写或自动二次生成。自然语言答案中的内部来源标记在发送前移除，来源保存供追问使用；资料真实性由回答模型负责。

原分工仍只对本地技术查询保留一次 DeepSeek 聚焦核验；联网、闲聊和本地概述直接输出。若在本轮回答前因额度耗尽切回备用模型，按实际回答模型所属分工执行。目录、来源原文和管理命令直接读取缓存或程序状态，不为固定结果额外调用 GPT。

## 手动验收

1. `/模型检查`：全部 ID 可解析，三个 GPT 为 Responses，档位符合设置。
2. 你的私聊发一句闲聊；日志应为辅助 DeepSeek，回答 Astra，思考 `low`。
3. 换干净会话问本地参数；trace 应为 `gpt_answer=true`、`answer_mode=direct`，没有 `local_audit`。
4. 群里分别发闲聊、技术问题和梗图；查看 Terra／Sol 分配及图片回复。GCP 是否接话仍由其原有频率设置决定。
5. 查询近期新闻；trace 的检索来源应是 `web`，搜索供应商 DeepSeek，回答按私聊／群聊分配。
6. 界面选 Gemini/DeepSeek 并保存，重复闲聊与本地问题，确认备用模型及本地单次核验；再选回 GPT优先。不要人为消耗额度制造测试。
7. 从另一账号私聊机器人，应没有自动回复；群里该账号正常参与。

日志记录实际回答 Provider、思考程度和分工；资料 trace 另外记录 `gpt_answer`、`reasoning_effort`、`writer_provider` 和 `fallback_reason`。离线回归不能代替你所配置的实际服务权限、模型质量与群聊效果验收。
