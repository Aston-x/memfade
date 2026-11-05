# memfade · NoneBot2 插件

给 NoneBot2 用的长期记忆：**会记得，也会忘得恰到好处。**

NoneBot2 没有统一的「往 LLM prompt 里插东西」的钩子——多数 LLM 插件是自己
组装 prompt 再调接口。所以本插件的定位是**记忆服务 + 助手 API**：

* 自动旁听消息并存进记忆库（不阻塞任何其它插件）；
* 提供指令与公开函数；
* 另外把记忆文本挂到 `event.set_extra("memfade_context", ...)`，
  想自动接管的插件直接读这个 extra 即可。

## 装法

```bash
pip install nonebot-plugin-memfade
```

或者把 `adapters/nonebot2/src/` 整个目录塞进你的插件目录。

## 指令

| 指令 | 作用 |
|---|---|
| `记忆` | 这个会话记住了多少东西 + 最近记下的内容 |
| `记忆 搜 <关键词>` | 检索（必须真命中） |
| `记忆 整理` | 立刻跑一轮整理 |
| `记忆 置顶 <内容>` | 让它永不淡化 |
| `记住 <内容>` | 显式记住 |
| `忘掉 <内容>` | 忘掉一条 |

## 给别的插件用的 API

```python
from nonebot_plugin_memfade import build_context, remember, recall_matching, forget, set_pinned

text = await build_context(event)          # 可直接注入 prompt 的记忆文本
await remember(event, "他叫BotName")        # 显式写入一条
hits = await recall_matching(event, "辣")  # 检索（必须真命中）
await set_pinned(event, "辣的食物")        # 置顶
await forget(event, "辣的食物")            # 忘掉
```

## 配置

| 键 | 默认 | 说明 |
|---|---|---|
| `memfade_db_path` | `memfade.db` | SQLite 记忆库路径 |
| `memfade_top_k` | 6 | 每轮最多召回多少条 |
| `memfade_max_inject_tokens` | 600 | 注入 token 预算 |
| `memfade_echo_max` | 2 | 已淡出的旧事最多注入几条 |
| `memfade_extract_enabled` | true | 是否抽取记忆 |
| `memfade_extract_every_n` | 3 | 每隔多少条消息抽取一次 |
| `memfade_min_importance` | 0.35 | 低于此不入库 |
| `memfade_llm_base_url` / `_model` / `_key` | 空 | 抽取用的 OpenAI 兼容接口（留空则用规则抽取） |
| `memfade_embedding_base_url` / `_model` / `_key` | 空 | 嵌入接口（留空则只用关键词检索） |

## 注意

**作用域**用 `event.get_session_id()`——群聊一个群一份、私聊一个人一份，
所有读取路径共用同一条过滤，**私聊的记忆不会漏进群里**。

抽取**不阻塞**：它挂在 `on_message(priority=99, block=False)` 上，
真正的抽取丢后台任务，LLM 慢也不影响别的插件。
